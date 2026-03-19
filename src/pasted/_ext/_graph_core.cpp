/**
 * pasted._ext._graph_core  (v0.1.12)
 * ====================================
 * C++17 implementations of the three bonded-pair graph metrics that were
 * the dominant bottlenecks in compute_all_metrics (N=1000 profile):
 *
 *   ring_fraction      — O(N·α(N))  Union-Find ring detection
 *   charge_frustration — O(N·k)     variance of |Δχ| over bonded pairs
 *   graph_metrics      — O(N·k)     LCC fraction + mean clustering coefficient
 *
 * All three share a single O(N) bonded-pair enumeration via FlatCellList
 * (same design as _relax_core / _steinhardt_core).  The Python fallbacks
 * used O(N²) full-matrix walks; the C++ paths are O(N·k) where k is the
 * mean bonded-pair count per atom (~constant for typical structures).
 *
 * bond_strain_rms (previously in _metrics.py) is intentionally omitted:
 * after relax_positions converges, d_ij >= cov_scale*(r_i+r_j) for every
 * pair by construction, so the metric is structurally zero and carries no
 * information.
 *
 * Dependencies: C++17 stdlib + pybind11 only.  No Eigen, no OpenMP.
 *
 * Python API
 * ----------
 *   graph_metrics_cpp(pts, radii, cov_scale, en_vals, cutoff)
 *       -> dict{"graph_lcc": float, "graph_cc": float,
 *               "ring_fraction": float, "charge_frustration": float}
 *
 * All four values are returned in a single call so the FlatCellList and
 * bonded-pair list are built only once.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

namespace py = pybind11;
using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using I32Array = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

static constexpr int CELL_LIST_THRESHOLD = 64;

// ===========================================================================
// FlatCellList  (identical pattern to _relax_core / _steinhardt_core)
// ===========================================================================

struct FlatCellList {
    double inv_cell;
    int    nx, ny, nz;
    double ox, oy, oz;
    std::vector<int> cell_head;
    std::vector<int> next;

    void build(const double* pts, int n, double cell_size) {
        inv_cell = 1.0 / cell_size;
        double xmn=pts[0],xmx=pts[0],ymn=pts[1],ymx=pts[1],zmn=pts[2],zmx=pts[2];
        for (int i = 1; i < n; ++i) {
            xmn=std::min(xmn,pts[i*3  ]); xmx=std::max(xmx,pts[i*3  ]);
            ymn=std::min(ymn,pts[i*3+1]); ymx=std::max(ymx,pts[i*3+1]);
            zmn=std::min(zmn,pts[i*3+2]); zmx=std::max(zmx,pts[i*3+2]);
        }
        ox=xmn-cell_size; oy=ymn-cell_size; oz=zmn-cell_size;
        nx=static_cast<int>((xmx-ox)*inv_cell)+2;
        ny=static_cast<int>((ymx-oy)*inv_cell)+2;
        nz=static_cast<int>((zmx-oz)*inv_cell)+2;
        cell_head.assign(nx*ny*nz, -1);
        next.resize(n);
        for (int i = 0; i < n; ++i) {
            int cx=static_cast<int>((pts[i*3  ]-ox)*inv_cell);
            int cy=static_cast<int>((pts[i*3+1]-oy)*inv_cell);
            int cz=static_cast<int>((pts[i*3+2]-oz)*inv_cell);
            int cid=cx+nx*(cy+ny*cz);
            next[i]=cell_head[cid]; cell_head[cid]=i;
        }
    }

    template<typename F>
    void for_each_pair(int /*n*/, F process) const {
        for (int cz=0;cz<nz;++cz)
        for (int cy=0;cy<ny;++cy)
        for (int cx=0;cx<nx;++cx) {
            int cid=cx+nx*(cy+ny*cz);
            for (int i=cell_head[cid];i>=0;i=next[i]) {
                for (int j=next[i];j>=0;j=next[j]) process(i,j);
                for (int dz=-1;dz<=1;++dz)
                for (int dy=-1;dy<=1;++dy)
                for (int dx=-1;dx<=1;++dx) {
                    if (!dx&&!dy&&!dz) continue;
                    int ncx=cx+dx,ncy=cy+dy,ncz=cz+dz;
                    if (ncx<0||ncy<0||ncz<0||ncx>=nx||ncy>=ny||ncz>=nz) continue;
                    int nid=ncx+nx*(ncy+ny*ncz);
                    if (nid<=cid) continue;
                    for (int k=cell_head[nid];k>=0;k=next[k]) process(i,k);
                }
            }
        }
    }
};

// ===========================================================================
// Union-Find (path compression + union by rank)
// ===========================================================================

struct UnionFind {
    std::vector<int> parent, rank_;

    explicit UnionFind(int n) : parent(n), rank_(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }

    // Returns false when a and b are already in the same component (back-edge).
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
        return true;
    }
};

// ===========================================================================
// Main computation
// ===========================================================================
// Collects all bonded pairs once, then computes all four metrics in O(N·k).

py::dict graph_metrics_cpp(
    F64Array pts_in,
    F64Array radii_in,
    double   cov_scale,
    F64Array en_in,       // per-atom Pauling electronegativity
    double   cutoff)      // adjacency cutoff for graph_lcc / graph_cc
{
    auto pts_buf = pts_in.request();
    auto rad_buf = radii_in.request();
    auto en_buf  = en_in.request();
    const int     n      = static_cast<int>(pts_buf.shape[0]);
    const double* pts    = static_cast<const double*>(pts_buf.ptr);
    const double* radii  = static_cast<const double*>(rad_buf.ptr);
    const double* en     = static_cast<const double*>(en_buf.ptr);

    // ── Trivial cases ──────────────────────────────────────────────────────
    if (n < 2) {
        py::dict result;
        result["graph_lcc"]          = (n == 1) ? 1.0 : 0.0;
        result["graph_cc"]           = 0.0;
        result["ring_fraction"]      = 0.0;
        result["charge_frustration"] = 0.0;
        return result;
    }

    // ── Build bonded-pair list via FlatCellList (O(N)) or O(N²) ──────────
    // Cell size = max possible bonded-pair threshold for bond detection,
    // extended to also cover the graph_metrics cutoff.
    double max_r = *std::max_element(radii, radii + n);
    double bond_cell = std::max(1e-6, cov_scale * 2.0 * max_r);
    double cell_size = std::max(bond_cell, cutoff);

    // Adjacency lists: bonded (for ring/charge metrics) and cutoff (for graph)
    std::vector<std::vector<int>> bond_adj(n), graph_adj(n);

    auto accumulate = [&](int i, int j) {
        const double dx = pts[3*i  ]-pts[3*j  ];
        const double dy = pts[3*i+1]-pts[3*j+1];
        const double dz = pts[3*i+2]-pts[3*j+2];
        const double d  = std::sqrt(dx*dx+dy*dy+dz*dz);
        const double thr = cov_scale*(radii[i]+radii[j]);
        if (d < thr) {
            bond_adj[i].push_back(j);
            bond_adj[j].push_back(i);
        }
        if (d <= cutoff && d > 0.0) {
            graph_adj[i].push_back(j);
            graph_adj[j].push_back(i);
        }
    };

    if (n >= CELL_LIST_THRESHOLD) {
        FlatCellList cl;
        cl.build(pts, n, cell_size);
        cl.for_each_pair(n, accumulate);
    } else {
        for (int i = 0; i < n-1; ++i)
            for (int j = i+1; j < n; ++j)
                accumulate(i, j);
    }

    // ── ring_fraction: Union-Find on bond graph ────────────────────────────
    double ring_fraction = 0.0;
    if (n >= 3) {
        UnionFind uf(n);
        std::vector<bool> in_ring(n, false);
        for (int i = 0; i < n; ++i) {
            for (int j : bond_adj[i]) {
                if (j <= i) continue;          // process each edge once
                if (!uf.unite(i, j)) {
                    in_ring[i] = true;
                    in_ring[j] = true;
                }
            }
        }
        int ring_count = 0;
        for (int i = 0; i < n; ++i) if (in_ring[i]) ++ring_count;
        ring_fraction = static_cast<double>(ring_count) / n;
    }

    // ── charge_frustration: variance of |Δχ| over bonded pairs ────────────
    double charge_frustration = 0.0;
    {
        std::vector<double> diffs;
        diffs.reserve(n * 4);
        for (int i = 0; i < n; ++i)
            for (int j : bond_adj[i])
                if (j > i)
                    diffs.push_back(std::fabs(en[i] - en[j]));

        if (diffs.size() >= 2) {
            const int m = static_cast<int>(diffs.size());
            double mean = 0.0;
            for (double v : diffs) mean += v;
            mean /= m;
            double var = 0.0;
            for (double v : diffs) { double d = v-mean; var += d*d; }
            charge_frustration = var / m;   // population variance (matches Python)
        }
    }

    // ── graph_metrics: LCC + mean clustering coefficient ──────────────────
    // LCC via Union-Find on graph adjacency
    double graph_lcc = 0.0;
    double graph_cc  = 0.0;
    {
        UnionFind uf(n);
        for (int i = 0; i < n; ++i)
            for (int j : graph_adj[i])
                if (j > i) uf.unite(i, j);

        std::vector<int> comp_size(n, 0);
        for (int i = 0; i < n; ++i) ++comp_size[uf.find(i)];
        graph_lcc = static_cast<double>(*std::max_element(comp_size.begin(), comp_size.end())) / n;

        // Clustering coefficient: for each node count triangles / possible
        double cc_sum = 0.0;
        int    cc_cnt = 0;
        for (int i = 0; i < n; ++i) {
            const auto& nb = graph_adj[i];
            const int   k  = static_cast<int>(nb.size());
            if (k < 2) continue;
            // Count triangles using neighbour set lookup (O(k²) per node, k small)
            int tri = 0;
            for (int a = 0; a < k; ++a)
                for (int b = a+1; b < k; ++b) {
                    // Check if nb[a] and nb[b] are connected
                    const auto& nb_a = graph_adj[nb[a]];
                    if (std::find(nb_a.begin(), nb_a.end(), nb[b]) != nb_a.end())
                        ++tri;
                }
            cc_sum += static_cast<double>(tri) / (k * (k-1) / 2);
            ++cc_cnt;
        }
        if (cc_cnt > 0) graph_cc = cc_sum / cc_cnt;
    }

    // ── moran_I_chi: spatial autocorrelation for electronegativity ──────────
    // Reuses the same pts/cutoff/en arrays; no extra FlatCellList build needed.
    double moran_I = 0.0;
    {
        double chi_bar = 0.0;
        for (int i = 0; i < n; ++i) chi_bar += en[i];
        chi_bar /= n;

        std::vector<double> dev_v(n);
        double denom_m = 0.0;
        for (int i = 0; i < n; ++i) {
            dev_v[i] = en[i] - chi_bar;
            denom_m += dev_v[i] * dev_v[i];
        }
        if (denom_m > 1e-30) {
            // Reuse graph_adj (cutoff-based adjacency) for Moran weights
            double numer_m = 0.0, W_sum_m = 0.0;
            for (int i = 0; i < n; ++i) {
                for (int j : graph_adj[i]) {
                    if (j > i) {
                        numer_m += 2.0 * dev_v[i] * dev_v[j];
                        W_sum_m += 2.0;
                    }
                }
            }
            if (W_sum_m > 1e-30)
                moran_I = (static_cast<double>(n) / W_sum_m) * (numer_m / denom_m);
        }
    }

    py::dict result;
    result["graph_lcc"]          = graph_lcc;
    result["graph_cc"]           = graph_cc;
    result["ring_fraction"]      = ring_fraction;
    result["charge_frustration"] = charge_frustration;
    result["moran_I_chi"]        = moran_I;
    return result;
}

// ===========================================================================

// Helper: compute cell size from pts bounding box and cutoff
static double cell_size_from(const double* pts, int n, double cutoff) {
    (void)pts; (void)n;
    return std::max(1e-6, cutoff);
}


// ===========================================================================
// moran_I_chi_cpp
// ===========================================================================
// Moran's I spatial autocorrelation for Pauling electronegativity.
//
//   I = (N / W) * (Σ_{i≠j} w_ij (χ_i−χ̄)(χ_j−χ̄)) / (Σ_i (χ_i−χ̄)²)
//
// Spatial weight w_ij = 1 when r_ij <= cutoff (step function), 0 otherwise.
// FlatCellList for N >= CELL_LIST_THRESHOLD (O(N·k)), full-pair otherwise.
//
// Returns: float in (-1, 1].
//   I ≈  0 : random (desired for PASTED structures)
//   I >  0 : same-electronegativity atoms cluster spatially
//   I <  0 : alternating high/low electronegativity (NaCl-like order)
//
// Returns 0.0 when all atoms share the same electronegativity (denominator=0)
// or when no pair falls within cutoff (W_sum=0).

double moran_I_chi_cpp(
    F64Array pts_in,
    F64Array en_in,   // per-atom Pauling electronegativity
    double   cutoff)
{
    auto pts_buf = pts_in.request();
    auto en_buf  = en_in.request();
    const int     n   = static_cast<int>(pts_buf.shape[0]);
    const double* pts = static_cast<const double*>(pts_buf.ptr);
    const double* en  = static_cast<const double*>(en_buf.ptr);

    if (n < 2) return 0.0;

    // Mean electronegativity
    double chi_bar = 0.0;
    for (int i = 0; i < n; ++i) chi_bar += en[i];
    chi_bar /= n;

    // Denominator: Σ (χ_i - χ̄)²
    double denom = 0.0;
    std::vector<double> dev(n);
    for (int i = 0; i < n; ++i) {
        dev[i] = en[i] - chi_bar;
        denom += dev[i] * dev[i];
    }
    if (denom < 1e-30) return 0.0;  // all same electronegativity

    // Accumulate W_sum and cross-term over bonded pairs within cutoff
    double numer  = 0.0;
    double W_sum  = 0.0;

    auto accumulate = [&](int i, int j) {
        const double dx = pts[3*i  ] - pts[3*j  ];
        const double dy = pts[3*i+1] - pts[3*j+1];
        const double dz = pts[3*i+2] - pts[3*j+2];
        const double d  = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (d > 1e-6 && d <= cutoff) {
            // Symmetric: pair (i,j) contributes twice to the full-sum
            const double contrib = dev[i] * dev[j];
            numer += 2.0 * contrib;
            W_sum += 2.0;
        }
    };

    if (n >= CELL_LIST_THRESHOLD) {
        FlatCellList cl;
        cl.build(pts, n, cell_size_from(pts, n, cutoff));
        cl.for_each_pair(n, accumulate);
    } else {
        for (int i = 0; i < n-1; ++i)
            for (int j = i+1; j < n; ++j)
                accumulate(i, j);
    }

    if (W_sum < 1e-30) return 0.0;

    return (static_cast<double>(n) / W_sum) * (numer / denom);
}


PYBIND11_MODULE(_graph_core, m) {
    m.doc() =
        "pasted._ext._graph_core (v0.1.12)\n"
        "O(N·k) graph metrics: graph_lcc, graph_cc, ring_fraction, charge_frustration.\n"
        "FlatCellList spatial index for N >= 64; O(N^2) full-pair for N < 64.\n"
        "bond_strain_rms is intentionally excluded (always 0 after relax).";

    m.def(
        "moran_I_chi_cpp", &moran_I_chi_cpp,
        py::arg("pts"), py::arg("en_vals"), py::arg("cutoff"),
        R"(
Moran's I spatial autocorrelation for Pauling electronegativity.

  I = (N/W) * sum_{i!=j} w_ij*(chi_i-chi_bar)*(chi_j-chi_bar) / sum_i (chi_i-chi_bar)^2

w_ij = 1 when d_ij <= cutoff (step function), 0 otherwise.
Returns float in (-1, 1]:  0=random, +1=clustered, -1=alternating.
Returns 0.0 when all atoms share the same electronegativity or W=0.

Parameters
----------
pts     : (n, 3) float64  -- atom positions in Angstrom
en_vals : (n,)   float64  -- per-atom Pauling electronegativity
cutoff  : float           -- distance cutoff (Ang)
        )"
    );

    m.def(
        "graph_metrics_cpp", &graph_metrics_cpp,
        py::arg("pts"), py::arg("radii"), py::arg("cov_scale"),
        py::arg("en_vals"), py::arg("cutoff"),
        R"(
Compute graph_lcc, graph_cc, ring_fraction, and charge_frustration in O(N·k).

Parameters
----------
pts       : (n, 3) float64  -- atom positions in Angstrom
radii     : (n,)   float64  -- covalent radii in Angstrom
cov_scale : float           -- bond detection threshold scale
en_vals   : (n,)   float64  -- per-atom Pauling electronegativity
cutoff    : float           -- distance cutoff for graph_lcc / graph_cc (Ang)

Returns
-------
dict with keys: graph_lcc, graph_cc, ring_fraction, charge_frustration
        )"
    );
}
