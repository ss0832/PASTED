/**
 * pasted._ext._graph_core  (v0.2.9)
 * ====================================
 * C++17 O(N·k) implementations of five graph-based disorder metrics and two
 * distance-histogram metrics, all sharing a FlatCellList spatial index.
 *
 * Graph / bond metrics (graph_metrics_cpp, single FlatCellList pass):
 *   ring_fraction      — O(N·alpha(N))  Union-Find ring detection
 *   charge_frustration — O(N·k)         variance of |delta-chi| over pairs
 *   graph_lcc          — O(N·k)         largest-connected-component fraction
 *   graph_cc           — O(N·k^2)       mean clustering coefficient (k small)
 *   moran_I_chi        — O(N·k)         Moran's I spatial autocorrelation
 *
 * Distance-histogram metrics (rdf_h_cpp, one FlatCellList pass):
 *   h_spatial          — O(N·k)  Shannon entropy of distance histogram
 *   rdf_dev            — O(N·k)  RMS deviation of g(r) from ideal gas
 *
 * All functions use FlatCellList for N >= CELL_LIST_THRESHOLD (64); an O(N^2)
 * full-pair loop is used for smaller structures.
 *
 * bond_strain_rms is intentionally omitted: after relax_positions converges,
 * d_ij >= cov_scale*(r_i+r_j) for every pair by construction, so the metric
 * is structurally zero and carries no information.
 *
 * Dependencies: C++17 stdlib + pybind11 only.  No Eigen, no OpenMP.
 * All computation is single-threaded.  The two-pass pair-collection +
 * OpenMP scaffolding introduced in v0.2.3–v0.2.8 added heap allocation
 * overhead without any parallelism benefit (OpenMP was not linked) and
 * was reverted in v0.2.9 to the original single-pass lambda pattern.
 *
 * Python API
 * ----------
 *   graph_metrics_cpp(pts, radii, cov_scale, en_vals, cutoff)
 *       -> dict{graph_lcc, graph_cc, ring_fraction, charge_frustration,
 *               moran_I_chi}
 *
 *   rdf_h_cpp(pts, cutoff, n_bins)
 *       -> dict{h_spatial, rdf_dev}
 *
 *   moran_I_chi_cpp(pts, en_vals, cutoff)  [retained for backward compat]
 *       -> float
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

static constexpr int    CELL_LIST_THRESHOLD = 64;
static constexpr double PI                  = 3.14159265358979323846;

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

    // Enumerate unique unordered pairs (i, j) with i < j within the cell radius.
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
// graph_metrics_cpp
// ===========================================================================
// Collects all bonded pairs once via FlatCellList, then computes all five
// metrics in O(N·k).

py::dict graph_metrics_cpp(
    F64Array pts_in,
    F64Array radii_in,
    double   cov_scale,
    F64Array en_in,   // per-atom Pauling electronegativity
    double   cutoff)  // adjacency cutoff for all five metrics
{
    auto pts_buf = pts_in.request();
    auto rad_buf = radii_in.request();
    auto en_buf  = en_in.request();
    const int     n      = static_cast<int>(pts_buf.shape[0]);
    const double* pts    = static_cast<const double*>(pts_buf.ptr);
    const double* radii  = static_cast<const double*>(rad_buf.ptr);
    const double* en     = static_cast<const double*>(en_buf.ptr);

    // Trivial cases
    if (n < 2) {
        py::dict result;
        result["graph_lcc"]          = (n == 1) ? 1.0 : 0.0;
        result["graph_cc"]           = 0.0;
        result["ring_fraction"]      = 0.0;
        result["charge_frustration"] = 0.0;
        result["moran_I_chi"]        = 0.0;
        return result;
    }

    // Cell size covers both the cov_scale bond threshold and the cutoff.
    double max_r    = *std::max_element(radii, radii + n);
    double bond_cell = std::max(1e-6, cov_scale * 2.0 * max_r);
    double cell_size = std::max(bond_cell, cutoff);

    // Unified adjacency list: d_ij <= cutoff.
    // Using cov_scale*(r_i+r_j) for bond detection is structurally zero after
    // relax_positions convergence, because relax guarantees
    // d_ij >= cov_scale*(r_i+r_j) for every pair.  The cutoff (~1.5x median
    // covalent diameter) captures genuine nearest-neighbor contacts in the
    // relaxed structure and produces informative non-zero values for
    // ring_fraction and charge_frustration.
    std::vector<std::vector<int>> bond_adj(n), graph_adj(n);

    auto accumulate = [&](int i, int j) {
        const double dx = pts[3*i  ]-pts[3*j  ];
        const double dy = pts[3*i+1]-pts[3*j+1];
        const double dz = pts[3*i+2]-pts[3*j+2];
        const double d  = std::sqrt(dx*dx+dy*dy+dz*dz);
        if (d <= cutoff && d > 1e-6) {
            bond_adj[i].push_back(j);
            bond_adj[j].push_back(i);
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

    // ring_fraction: Union-Find on bond graph
    double ring_fraction = 0.0;
    if (n >= 3) {
        UnionFind uf(n);
        std::vector<bool> in_ring(n, false);
        for (int i = 0; i < n; ++i) {
            for (int j : bond_adj[i]) {
                if (j <= i) continue;
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

    // charge_frustration: variance of |delta-chi| over bonded pairs
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
            for (double v : diffs) { double d = v - mean; var += d * d; }
            charge_frustration = var / m;  // population variance (matches Python)
        }
    }

    // graph_lcc + graph_cc: LCC via Union-Find; CC via triangle counting
    double graph_lcc = 0.0;
    double graph_cc  = 0.0;
    {
        UnionFind uf(n);
        for (int i = 0; i < n; ++i)
            for (int j : graph_adj[i])
                if (j > i) uf.unite(i, j);

        std::vector<int> comp_size(n, 0);
        for (int i = 0; i < n; ++i) ++comp_size[uf.find(i)];
        graph_lcc = static_cast<double>(
            *std::max_element(comp_size.begin(), comp_size.end())) / n;

        double cc_sum = 0.0;
        int    cc_cnt = 0;
        for (int i = 0; i < n; ++i) {
            const auto& nb = graph_adj[i];
            const int   k  = static_cast<int>(nb.size());
            if (k < 2) continue;
            int tri = 0;
            for (int a = 0; a < k; ++a)
                for (int b = a + 1; b < k; ++b) {
                    const auto& nb_a = graph_adj[nb[a]];
                    if (std::find(nb_a.begin(), nb_a.end(), nb[b]) != nb_a.end())
                        ++tri;
                }
            cc_sum += static_cast<double>(tri) / (k * (k - 1) / 2);
            ++cc_cnt;
        }
        if (cc_cnt > 0) graph_cc = cc_sum / cc_cnt;
    }

    // moran_I_chi: spatial autocorrelation for electronegativity
    // Reuses graph_adj (same cutoff adjacency) — no extra FlatCellList build.
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
// rdf_h_cpp
// ===========================================================================
// Enumerates all pairs within cutoff via FlatCellList and computes:
//   h_spatial  — Shannon entropy of the pair-distance histogram over [0, cutoff]
//   rdf_dev    — RMS deviation of the empirical g(r) from an ideal-gas baseline
//
// Both metrics previously used scipy pdist (O(N^2) condensed array) and
// np.histogram over all N*(N-1)/2 distances.  This function achieves O(N*k)
// complexity (k = mean neighbors per atom within cutoff, roughly constant)
// and avoids allocating the O(N^2) distance array entirely.
//
// RDF normalization:
//   rho     = N / (4/3 * pi * r_bound^3)   where r_bound = max |r_i - centroid|
//   ideal_b = rho * 4*pi * r_b^2 * bw * N/2   (expected pairs in bin b)
//   rdf_dev = sqrt( mean_b [ (count_b / ideal_b - 1)^2 ] )  over bins with ideal>0

py::dict rdf_h_cpp(F64Array pts_in, double cutoff, int n_bins) {
    auto buf = pts_in.request();
    const int     n   = static_cast<int>(buf.shape[0]);
    const double* pts = static_cast<const double*>(buf.ptr);

    py::dict result;
    result["h_spatial"] = 0.0;
    result["rdf_dev"]   = 0.0;

    if (n < 2 || cutoff <= 0.0 || n_bins < 1) return result;

    // Enumerate pairs and compute distances in a single FlatCellList pass.
    const double cutoff2 = cutoff * cutoff;
    std::vector<double> pair_dists;

    auto collect = [&](int i, int j) {
        const double dx = pts[3*i  ] - pts[3*j  ];
        const double dy = pts[3*i+1] - pts[3*j+1];
        const double dz = pts[3*i+2] - pts[3*j+2];
        const double d2 = dx*dx + dy*dy + dz*dz;
        if (d2 <= cutoff2 && d2 > 1e-20)
            pair_dists.push_back(std::sqrt(d2));
    };

    if (n >= CELL_LIST_THRESHOLD) {
        pair_dists.reserve(static_cast<std::size_t>(n) * 10);
        FlatCellList cl;
        cl.build(pts, n, cutoff);
        cl.for_each_pair(n, collect);
    } else {
        for (int i = 0; i < n - 1; ++i)
            for (int j = i + 1; j < n; ++j)
                collect(i, j);
    }

    if (pair_dists.empty()) return result;

    // Build histogram over [0, cutoff]
    const double bin_width = cutoff / n_bins;
    std::vector<int> hist(static_cast<std::size_t>(n_bins), 0);
    for (double d : pair_dists) {
        int b = static_cast<int>(d / bin_width);
        if (b >= n_bins) b = n_bins - 1;
        ++hist[static_cast<std::size_t>(b)];
    }

    // h_spatial: Shannon entropy of the histogram
    const double total = static_cast<double>(pair_dists.size());
    double h_spatial = 0.0;
    for (int c : hist) {
        if (c > 0) {
            const double p = c / total;
            h_spatial -= p * std::log(p);
        }
    }

    // rdf_dev: RMS deviation from ideal gas
    // r_bound = max distance from the centroid
    double cx = 0.0, cy = 0.0, cz = 0.0;
    for (int i = 0; i < n; ++i) { cx += pts[3*i]; cy += pts[3*i+1]; cz += pts[3*i+2]; }
    cx /= n; cy /= n; cz /= n;
    double r_bound = 0.0;
    for (int i = 0; i < n; ++i) {
        const double dx = pts[3*i] - cx;
        const double dy = pts[3*i+1] - cy;
        const double dz = pts[3*i+2] - cz;
        r_bound = std::max(r_bound, std::sqrt(dx*dx + dy*dy + dz*dz));
    }

    double rdf_dev = 0.0;
    if (r_bound > 0.0) {
        const double rho = n / (4.0 / 3.0 * PI * r_bound * r_bound * r_bound);
        double sum_sq = 0.0;
        int    valid  = 0;
        for (int b = 0; b < n_bins; ++b) {
            const double center = (b + 0.5) * bin_width;
            const double ideal  = rho * 4.0 * PI * center * center * bin_width
                                  * static_cast<double>(n) / 2.0;
            if (ideal > 0.0) {
                const double ratio = hist[static_cast<std::size_t>(b)] / ideal - 1.0;
                sum_sq += ratio * ratio;
                ++valid;
            }
        }
        if (valid > 0) rdf_dev = std::sqrt(sum_sq / valid);
    }

    result["h_spatial"] = h_spatial;
    result["rdf_dev"]   = rdf_dev;
    return result;
}

// ===========================================================================
// moran_I_chi_cpp  (retained for backward compatibility)
// ===========================================================================

static double _cell_size(double cutoff) { return std::max(1e-6, cutoff); }

double moran_I_chi_cpp(F64Array pts_in, F64Array en_in, double cutoff) {
    auto pts_buf = pts_in.request();
    auto en_buf  = en_in.request();
    const int     n   = static_cast<int>(pts_buf.shape[0]);
    const double* pts = static_cast<const double*>(pts_buf.ptr);
    const double* en  = static_cast<const double*>(en_buf.ptr);

    if (n < 2) return 0.0;

    double chi_bar = 0.0;
    for (int i = 0; i < n; ++i) chi_bar += en[i];
    chi_bar /= n;

    double denom = 0.0;
    std::vector<double> dev(n);
    for (int i = 0; i < n; ++i) {
        dev[i] = en[i] - chi_bar;
        denom += dev[i] * dev[i];
    }
    if (denom < 1e-30) return 0.0;

    double numer = 0.0, W_sum = 0.0;

    auto accumulate = [&](int i, int j) {
        const double dx = pts[3*i  ] - pts[3*j  ];
        const double dy = pts[3*i+1] - pts[3*j+1];
        const double dz = pts[3*i+2] - pts[3*j+2];
        const double d  = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (d > 1e-6 && d <= cutoff) {
            numer += 2.0 * dev[i] * dev[j];
            W_sum += 2.0;
        }
    };

    if (n >= CELL_LIST_THRESHOLD) {
        FlatCellList cl;
        cl.build(pts, n, _cell_size(cutoff));
        cl.for_each_pair(n, accumulate);
    } else {
        for (int i = 0; i < n - 1; ++i)
            for (int j = i + 1; j < n; ++j)
                accumulate(i, j);
    }

    if (W_sum < 1e-30) return 0.0;
    return (static_cast<double>(n) / W_sum) * (numer / denom);
}

// ===========================================================================
// Module bindings
// ===========================================================================

PYBIND11_MODULE(_graph_core, m) {
    m.doc() =
        "pasted._ext._graph_core (v0.2.9)\n"
        "O(N*k) graph and distance-histogram metrics via FlatCellList.\n"
        "FlatCellList spatial index for N >= 64; O(N^2) full-pair for N < 64.\n"
        "All computation is single-threaded (no OpenMP).\n"
        "bond_strain_rms is intentionally excluded (always 0 after relax).";

    m.def(
        "graph_metrics_cpp", &graph_metrics_cpp,
        py::arg("pts"), py::arg("radii"), py::arg("cov_scale"),
        py::arg("en_vals"), py::arg("cutoff"),
        R"(
Compute graph_lcc, graph_cc, ring_fraction, charge_frustration, and
moran_I_chi in O(N*k) using a single FlatCellList pass.

Parameters
----------
pts       : (n, 3) float64  -- atom positions in Angstrom
radii     : (n,)   float64  -- covalent radii in Angstrom
cov_scale : float           -- bond detection threshold scale (retained for
                               API compatibility; adjacency now uses cutoff)
en_vals   : (n,)   float64  -- per-atom Pauling electronegativity
cutoff    : float           -- distance cutoff for all five metrics (Ang)

Returns
-------
dict with keys: graph_lcc, graph_cc, ring_fraction, charge_frustration,
                moran_I_chi
        )"
    );

    m.def(
        "rdf_h_cpp", &rdf_h_cpp,
        py::arg("pts"), py::arg("cutoff"), py::arg("n_bins"),
        R"(
Compute h_spatial and rdf_dev using O(N*k) FlatCellList pair enumeration.

Replaces the O(N^2) scipy pdist + np.histogram path for both metrics.
Only pairs within *cutoff* are included in the distance histogram, which
matches the locality assumption used by all other metrics.

Parameters
----------
pts    : (n, 3) float64  -- atom positions in Angstrom
cutoff : float           -- neighbor distance cutoff (Ang)
n_bins : int             -- number of histogram bins

Returns
-------
dict with keys:
  h_spatial : float  -- Shannon entropy of the pair-distance histogram
  rdf_dev   : float  -- RMS deviation of g(r) from an ideal-gas baseline
        )"
    );

    m.def(
        "moran_I_chi_cpp", &moran_I_chi_cpp,
        py::arg("pts"), py::arg("en_vals"), py::arg("cutoff"),
        R"(
Moran's I spatial autocorrelation for Pauling electronegativity.

Retained for backward compatibility.  graph_metrics_cpp computes the same
value as part of a single FlatCellList pass and is preferred.

Parameters
----------
pts     : (n, 3) float64  -- atom positions in Angstrom
en_vals : (n,)   float64  -- per-atom Pauling electronegativity
cutoff  : float           -- distance cutoff (Ang)

Returns
-------
float  -- Moran's I in (-1, 1]:  0=random, +1=clustered, -1=alternating.
        )"
    );
}
