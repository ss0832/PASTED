/**
 * _combined_core.cpp
 * ==================
 * C++17 single-pass implementation of all_metrics_cpp.
 *
 * Speedup over calling the individual extensions separately:
 *   _rdf_h_cpp       — builds its own FlatCellList
 *   _graph_metrics_cpp — builds its own FlatCellList
 *   _steinhardt_per_atom — builds its own FlatCellList
 *   _bond_angle_entropy_cpp — builds two FlatCellList passes
 *
 * This module builds the FlatCellList exactly ONCE and performs a single
 * for_each_pair traversal that simultaneously accumulates:
 *   • RDF histogram              → h_spatial, rdf_dev
 *   • Bond adjacency lists       → graph_lcc, graph_cc, ring_fraction,
 *                                   charge_frustration, moran_I_chi
 *   • Steinhardt re/im buffers   → Q4, Q6, Q8 (per-atom, fast-path ④)
 *   • Per-atom distance sums     → radial_variance
 *   • Per-atom outer-product T   → local_anisotropy
 *   • CSR neighbor count         → bond_angle_entropy (count pass)
 *
 * After the single for_each_pair pass:
 *   • CSR unit vectors are filled from bond_adj (no second cell-list pass).
 *   • Per-atom bond-angle histograms are computed from the CSR (O(N·k²)).
 *   • All scalar metrics are finalized from the accumulated arrays.
 *
 * Design notes
 * ------------
 * - FlatCellList is built once; cell_size = cutoff (sufficient because all
 *   adjacency is based on distance ≤ cutoff, not cov_scale*(r_i+r_j)).
 * - Unified distance threshold: d² < 1e-20 (d < 1e-10 Å) excludes coincident
 *   atoms from all metrics.  This is effectively identical to the 1e-6/1e-10
 *   thresholds used in the individual modules for any real structure.
 * - Steinhardt fast-path ④ (l=4,6,8 real spherical harmonics via joint SymPy
 *   CSE) is copied verbatim from _steinhardt_core to keep results identical.
 *   For even l, Y_l^m(-r) = Y_l^m(r), so directions (ux,uy,uz) and
 *   (-ux,-uy,-uz) yield the same SH contributions; both are accumulated to
 *   maintain correct per-atom normalisation.
 * - CSR int64_t offsets prevent overflow for large N * high coordination.
 * - Bond-angle histogram uses a stack-allocated int[N_BINS_BA] array
 *   (zero heap allocations per atom, stays in L1 cache).
 * - ring_fraction uses iterative Tarjan bridge-finding (no recursion stack
 *   overflow for large N); same algorithm as _graph_core.
 *
 * Python API
 * ----------
 *   all_metrics_cpp(pts, radii, en_vals, cutoff, n_bins)
 *       -> dict with keys:
 *            h_spatial, rdf_dev,
 *            graph_lcc, graph_cc, ring_fraction,
 *            charge_frustration, moran_I_chi,
 *            Q4, Q6, Q8  (per-atom float64 arrays),
 *            bond_angle_entropy, coordination_variance,
 *            radial_variance, local_anisotropy
 *
 * Dependencies: C++17 stdlib + pybind11 only.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

static constexpr int    CELL_THRESHOLD = 64;
static constexpr double PI             = 3.14159265358979323846;
static constexpr double FOURPI         = 4.0 * PI;
static constexpr int    N_BINS_BA      = 36;    // bond-angle histogram bins (compile-time)
static constexpr int    L_MAX_SH       = 12;    // max l for spherical harmonics

// ---------------------------------------------------------------------------
// FlatCellList — linked-list spatial index
// ---------------------------------------------------------------------------

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
        {
            static constexpr std::int64_t MAX_CELLS = 1LL << 22;
            auto tnx=[&]{return static_cast<int>((xmx-ox)*inv_cell)+2;};
            auto tny=[&]{return static_cast<int>((ymx-oy)*inv_cell)+2;};
            auto tnz=[&]{return static_cast<int>((zmx-oz)*inv_cell)+2;};
            while (static_cast<std::int64_t>(tnx())*tny()*tnz() > MAX_CELLS) {
                cell_size *= 2.0; inv_cell = 1.0 / cell_size;
                ox=xmn-cell_size; oy=ymn-cell_size; oz=zmn-cell_size;
            }
            nx=tnx(); ny=tny(); nz=tnz();
        }
        cell_head.assign(static_cast<std::size_t>(nx*ny*nz), -1);
        next.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            int cx=static_cast<int>((pts[i*3  ]-ox)*inv_cell);
            int cy=static_cast<int>((pts[i*3+1]-oy)*inv_cell);
            int cz=static_cast<int>((pts[i*3+2]-oz)*inv_cell);
            int cid=cx+nx*(cy+ny*cz);
            next[static_cast<std::size_t>(i)] = cell_head[static_cast<std::size_t>(cid)];
            cell_head[static_cast<std::size_t>(cid)] = i;
        }
    }

    // Enumerate unique unordered pairs (i, j) with i != j within cell radius.
    // Each pair is yielded exactly once.
    template<typename F>
    void for_each_pair(int /*n*/, F process) const {
        for (int cz=0;cz<nz;++cz)
        for (int cy=0;cy<ny;++cy)
        for (int cx=0;cx<nx;++cx) {
            int cid=cx+nx*(cy+ny*cz);
            for (int i=cell_head[static_cast<std::size_t>(cid)];i>=0;
                     i=next[static_cast<std::size_t>(i)]) {
                for (int j=next[static_cast<std::size_t>(i)];j>=0;
                         j=next[static_cast<std::size_t>(j)])
                    process(i,j);
                for (int dz=-1;dz<=1;++dz)
                for (int dy=-1;dy<=1;++dy)
                for (int dx=-1;dx<=1;++dx) {
                    if (!dx&&!dy&&!dz) continue;
                    int ncx=cx+dx,ncy=cy+dy,ncz=cz+dz;
                    if (ncx<0||ncy<0||ncz<0||ncx>=nx||ncy>=ny||ncz>=nz) continue;
                    int nid=ncx+nx*(ncy+ny*ncz);
                    if (nid<=cid) continue;
                    for (int k=cell_head[static_cast<std::size_t>(nid)];k>=0;
                             k=next[static_cast<std::size_t>(k)])
                        process(i,k);
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Union-Find (path compression + union by rank)
// ---------------------------------------------------------------------------

struct UnionFind {
    std::vector<int> parent, rank_;
    explicit UnionFind(int n) : parent(n), rank_(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }
    int find(int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
        return true;
    }
};

// ---------------------------------------------------------------------------
// Steinhardt: factorial table + normalization factors
// (identical to _steinhardt_core)
// ---------------------------------------------------------------------------

static double g_fac[2 * L_MAX_SH + 1];
static bool   g_fac_init = false;

static void init_fac() {
    if (g_fac_init) return;
    g_fac[0] = 1.0;
    for (int i = 1; i <= 2 * L_MAX_SH; ++i) g_fac[i] = g_fac[i-1] * i;
    g_fac_init = true;
}

static inline double norm_lm(int l, int m) {
    init_fac();
    return std::sqrt((2.0*l+1.0) / FOURPI * g_fac[l-m] / g_fac[l+m]);
}

// ---------------------------------------------------------------------------
// all_metrics_cpp
// ---------------------------------------------------------------------------

py::dict all_metrics_cpp(
    F64Array pts_in,
    F64Array radii_in,   // covalent radii (Å); only used for r_bound backup
    F64Array en_in,      // Pauling electronegativity
    double   cutoff,
    int      n_bins)
{
    auto pts_buf = pts_in.request();
    auto en_buf  = en_in.request();
    const int     n   = static_cast<int>(pts_buf.shape[0]);
    const double* pts = static_cast<const double*>(pts_buf.ptr);
    const double* en  = static_cast<const double*>(en_buf.ptr);

    // -----------------------------------------------------------------------
    // Trivial / degenerate cases
    // -----------------------------------------------------------------------

    // Pre-populate result with safe defaults (returned early when n < 2).
    auto zero_result = [&]() -> py::dict {
        F64Array q4_z(static_cast<py::ssize_t>(n));
        F64Array q6_z(static_cast<py::ssize_t>(n));
        F64Array q8_z(static_cast<py::ssize_t>(n));
        double* p4 = static_cast<double*>(q4_z.request().ptr);
        double* p6 = static_cast<double*>(q6_z.request().ptr);
        double* p8 = static_cast<double*>(q8_z.request().ptr);
        for (int i = 0; i < n; ++i) { p4[i]=p6[i]=p8[i]=0.0; }
        py::dict r;
        r["h_spatial"] = 0.0; r["rdf_dev"] = 0.0;
        r["graph_lcc"] = (n == 1) ? 1.0 : 0.0;
        r["graph_cc"]  = 0.0; r["ring_fraction"] = 0.0;
        r["charge_frustration"] = 0.0; r["moran_I_chi"] = 0.0;
        r["Q4"] = q4_z; r["Q6"] = q6_z; r["Q8"] = q8_z;
        r["bond_angle_entropy"]    = 0.0;
        r["coordination_variance"] = 0.0;
        r["radial_variance"]       = 0.0;
        r["local_anisotropy"]      = 0.0;
        return r;
    };

    if (n < 2 || cutoff <= 0.0 || n_bins < 1) return zero_result();

    // Guard against NaN/Inf coordinates: any non-finite value in pts would
    // corrupt the FlatCellList bounding-box computation (NaN propagates into
    // nx/ny/nz, causing heap corruption / segfault).  Return zero-filled
    // result when any coordinate is non-finite — matching the behaviour of
    // the individual extension modules which produce NaN-containing results
    // rather than crashing.
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(pts[i*3  ]) ||
            !std::isfinite(pts[i*3+1]) ||
            !std::isfinite(pts[i*3+2]))
            return zero_result();
    }

    const double cutoff2   = cutoff * cutoff;
    const double bin_width = cutoff / static_cast<double>(n_bins);  // RDF bins

    // -----------------------------------------------------------------------
    // Storage for accumulated quantities
    // -----------------------------------------------------------------------

    // RDF / h_spatial
    std::vector<int> rdf_hist(static_cast<std::size_t>(n_bins), 0);
    int rdf_pair_count = 0;

    // Graph adjacency lists (for ring_fraction, CC, LCC, charge_frustration,
    // moran_I_chi).  Filled in the main pair loop; sorted afterward.
    std::vector<std::vector<int>> bond_adj(static_cast<std::size_t>(n));

    // Steinhardt accumulators for l = 4, 6, 8 — fast-path ④.
    // Buffer layout: (n, 3, lm1) — atom index outermost (cache-friendly).
    static constexpr int N_L_SH = 3;   // l = 4, 6, 8
    static constexpr int LM1    = 9;   // l_max + 1 = 8 + 1
    std::vector<double> re_buf(static_cast<std::size_t>(n * N_L_SH * LM1), 0.0);
    std::vector<double> im_buf(static_cast<std::size_t>(n * N_L_SH * LM1), 0.0);
    std::vector<double> sh_deg(static_cast<std::size_t>(n), 0.0);

    // Pre-compute normalization table for l = 4, 6, 8.
    double norms4[5], norms6[7], norms8[9];
    for (int m=0;m<=4;++m) norms4[m] = norm_lm(4, m);
    for (int m=0;m<=6;++m) norms6[m] = norm_lm(6, m);
    for (int m=0;m<=8;++m) norms8[m] = norm_lm(8, m);

    // Radial-variance accumulators
    std::vector<double> rv_sum_d (static_cast<std::size_t>(n), 0.0);
    std::vector<double> rv_sum_d2(static_cast<std::size_t>(n), 0.0);
    std::vector<int>    rv_deg   (static_cast<std::size_t>(n), 0);

    // Local-anisotropy T tensor components (unnormalized; divide by deg after)
    std::vector<double> la_Txx(static_cast<std::size_t>(n), 0.0);
    std::vector<double> la_Txy(static_cast<std::size_t>(n), 0.0);
    std::vector<double> la_Txz(static_cast<std::size_t>(n), 0.0);
    std::vector<double> la_Tyy(static_cast<std::size_t>(n), 0.0);
    std::vector<double> la_Tyz(static_cast<std::size_t>(n), 0.0);
    std::vector<double> la_Tzz(static_cast<std::size_t>(n), 0.0);

    // Bond-angle CSR neighbor-count (int64_t prevents overflow for large N*k)
    std::vector<std::int64_t> nb_ptr(static_cast<std::size_t>(n + 1), 0);

    // -----------------------------------------------------------------------
    // Steinhardt accumulation helper (fast-path ④, identical to _steinhardt_core)
    // Inline lambda; called twice per pair (atom i and atom j).
    // -----------------------------------------------------------------------

    auto accumulate_sh = [&](int atom, double x, double y, double z) {
        sh_deg[static_cast<std::size_t>(atom)] += 1.0;
        const std::size_t base_i = static_cast<std::size_t>(atom) *
                                   static_cast<std::size_t>(N_L_SH * LM1);
        const std::size_t base_4 = base_i + 0 * static_cast<std::size_t>(LM1);
        const std::size_t base_6 = base_i + 1 * static_cast<std::size_t>(LM1);
        const std::size_t base_8 = base_i + 2 * static_cast<std::size_t>(LM1);

        // =====================================================================
        // Auto-generated real spherical harmonics for l=4,6,8 (SymPy joint CSE)
        // DO NOT EDIT MANUALLY — same code as _steinhardt_core fast-path ④
        // =====================================================================
        const double x0  = z*z;
        const double x1  = z*z*z*z;
        const double x2  = (15.0/2.0)*z;
        const double x3  = z*z*z;
        const double x4  = (35.0/2.0)*x3;
        const double x5  = x*x;
        const double x6  = y*y;
        const double x7  = (105.0/2.0)*x0;
        const double x8  = x*y;
        const double x9  = x0*x8;
        const double x10 = x*x*x;
        const double x11 = 105*z;
        const double x12 = 315*z;
        const double x13 = x*x6;
        const double x14 = y*y*y;
        const double x15 = x5*y;
        const double x16 = x*x*x*x;
        const double x17 = y*y*y*y;
        const double x18 = x5*x6;
        const double x19 = x*x14;
        const double x20 = x10*y;
        const double x21 = z*z*z*z*z*z;
        const double x22 = (105.0/8.0)*z;
        const double x23 = (315.0/4.0)*x3;
        const double x24 = z*z*z*z*z;
        const double x25 = (693.0/8.0)*x24;
        const double x26 = (945.0/4.0)*x0;
        const double x27 = (3465.0/8.0)*x1;
        const double x28 = x1*x8;
        const double x29 = (945.0/2.0)*z;
        const double x30 = (2835.0/2.0)*z;
        const double x31 = (3465.0/2.0)*x3;
        const double x32 = (10395.0/2.0)*x3;
        const double x33 = (10395.0/2.0)*x0;
        const double x34 = x0*x18;
        const double x35 = 20790*x0;
        const double x36 = x*x*x*x*x;
        const double x37 = 10395*z;
        const double x38 = 51975*z;
        const double x39 = x*x17;
        const double x40 = 103950*z;
        const double x41 = x10*x6;
        const double x42 = y*y*y*y*y;
        const double x43 = x16*y;
        const double x44 = x14*x5;
        const double x45 = x*x*x*x*x*x;
        const double x46 = y*y*y*y*y*y;
        const double x47 = x17*x5;
        const double x48 = x16*x6;
        const double x49 = x*x42;
        const double x50 = x36*y;
        const double x51 = x10*x14;
        const double x52 = (315.0/16.0)*z;
        const double x53 = (3465.0/16.0)*x3;
        const double x54 = (9009.0/16.0)*x24;
        const double x55 = (6435.0/16.0)*z*z*z*z*z*z*z;
        const double x56 = (10395.0/16.0)*x0;
        const double x57 = (45045.0/16.0)*x5;
        const double x58 = (45045.0/16.0)*x6;
        const double x59 = (10395.0/8.0)*z;
        const double x60 = (31185.0/8.0)*z;
        const double x61 = (45045.0/4.0)*x3;
        const double x62 = (135135.0/8.0)*x24;
        const double x63 = (135135.0/4.0)*x3;
        const double x64 = (405405.0/8.0)*x24;
        const double x65 = (135135.0/4.0)*x0;
        const double x66 = (675675.0/8.0)*x1;
        const double x67 = 135135*x0;
        const double x68 = (675675.0/2.0)*x1;
        const double x69 = (135135.0/2.0)*z;
        const double x70 = (675675.0/2.0)*z;
        const double x71 = (675675.0/2.0)*x3;
        const double x72 = (3378375.0/2.0)*x3;
        const double x73 = 675675*z;
        const double x74 = 3378375*x3;
        const double x75 = (2027025.0/2.0)*x0;
        const double x76 = (30405375.0/2.0)*x0;
        const double x77 = 6081075*x0;
        const double x78 = x*x*x*x*x*x*x;
        const double x79 = 2027025*z;
        const double x80 = 14189175*z;
        const double x81 = 70945875*z;
        const double x82 = 42567525*z;
        const double x83 = y*y*y*y*y*y*y;

        re_buf[base_4+0] += norms4[0]*(-15.0/4.0*x0+(35.0/8.0)*x1+3.0/8.0);
        re_buf[base_4+1] += norms4[1]*(-x*x2+x*x4);
        im_buf[base_4+1] += norms4[1]*(-x2*y+x4*y);
        re_buf[base_4+2] += norms4[2]*(x5*x7-15.0/2.0*x5-x6*x7+(15.0/2.0)*x6);
        im_buf[base_4+2] += norms4[2]*(-15*x8+105*x9);
        re_buf[base_4+3] += norms4[3]*(x10*x11-x12*x13);
        im_buf[base_4+3] += norms4[3]*(-x11*x14+x12*x15);
        re_buf[base_4+4] += norms4[4]*(105*x16+105*x17-630*x18);
        im_buf[base_4+4] += norms4[4]*(-420*x19+420*x20);
        re_buf[base_6+0] += norms6[0]*((105.0/16.0)*x0-315.0/16.0*x1+(231.0/16.0)*x21-5.0/16.0);
        re_buf[base_6+1] += norms6[1]*(x*x22-x*x23+x*x25);
        im_buf[base_6+1] += norms6[1]*(x22*y-x23*y+x25*y);
        re_buf[base_6+2] += norms6[2]*(-x26*x5+x26*x6+x27*x5-x27*x6+(105.0/8.0)*x5-105.0/8.0*x6);
        im_buf[base_6+2] += norms6[2]*((3465.0/4.0)*x28+(105.0/4.0)*x8-945.0/2.0*x9);
        re_buf[base_6+3] += norms6[3]*(-x10*x29+x10*x31+x13*x30-x13*x32);
        im_buf[base_6+3] += norms6[3]*(x14*x29-x14*x31-x15*x30+x15*x32);
        re_buf[base_6+4] += norms6[4]*(x16*x33-945.0/2.0*x16+x17*x33-945.0/2.0*x17+2835*x18-31185*x34);
        im_buf[base_6+4] += norms6[4]*(-x19*x35+1890*x19+x20*x35-1890*x20);
        re_buf[base_6+5] += norms6[5]*(x36*x37+x38*x39-x40*x41);
        im_buf[base_6+5] += norms6[5]*(x37*x42+x38*x43-x40*x44);
        re_buf[base_6+6] += norms6[6]*(10395*x45-10395*x46+155925*x47-155925*x48);
        im_buf[base_6+6] += norms6[6]*(62370*x49+62370*x50-207900*x51);
        re_buf[base_8+0] += norms8[0]*(-315.0/32.0*x0+(3465.0/64.0)*x1-3003.0/32.0*x21+(6435.0/128.0)*z*z*z*z*z*z*z*z+35.0/128.0);
        re_buf[base_8+1] += norms8[1]*(-x*x52+x*x53-x*x54+x*x55);
        im_buf[base_8+1] += norms8[1]*(-x52*y+x53*y-x54*y+x55*y);
        re_buf[base_8+2] += norms8[2]*(-x1*x57+x1*x58+x21*x57-x21*x58+x5*x56-315.0/16.0*x5-x56*x6+(315.0/16.0)*x6);
        im_buf[base_8+2] += norms8[2]*((45045.0/8.0)*x21*x8-45045.0/8.0*x28-315.0/8.0*x8+(10395.0/8.0)*x9);
        re_buf[base_8+3] += norms8[3]*(x10*x59-x10*x61+x10*x62-x13*x60+x13*x63-x13*x64);
        im_buf[base_8+3] += norms8[3]*(-x14*x59+x14*x61-x14*x62+x15*x60-x15*x63+x15*x64);
        re_buf[base_8+4] += norms8[4]*(-2027025.0/4.0*x1*x18-x16*x65+x16*x66+(10395.0/8.0)*x16-x17*x65+x17*x66+(10395.0/8.0)*x17-31185.0/4.0*x18+(405405.0/2.0)*x34);
        im_buf[base_8+4] += norms8[4]*(x19*x67-x19*x68-10395.0/2.0*x19-x20*x67+x20*x68+(10395.0/2.0)*x20);
        re_buf[base_8+5] += norms8[5]*(-x36*x69+x36*x71-x39*x70+x39*x72+x41*x73-x41*x74);
        im_buf[base_8+5] += norms8[5]*(-x42*x69+x42*x71-x43*x70+x43*x72+x44*x73-x44*x74);
        re_buf[base_8+6] += norms8[6]*(x45*x75-135135.0/2.0*x45-x46*x75+(135135.0/2.0)*x46+x47*x76-2027025.0/2.0*x47-x48*x76+(2027025.0/2.0)*x48);
        im_buf[base_8+6] += norms8[6]*(-20270250*x0*x51+x49*x77-405405*x49+x50*x77-405405*x50+1351350*x51);
        re_buf[base_8+7] += norms8[7]*(-x*x46*x80+x10*x17*x81-x36*x6*x82+x78*x79);
        im_buf[base_8+7] += norms8[7]*(-x14*x16*x81+x42*x5*x82+x45*x80*y-x79*x83);
        re_buf[base_8+8] += norms8[8]*(2027025*x*x*x*x*x*x*x*x+141891750*x16*x17-56756700*x45*x6-56756700*x46*x5+2027025*y*y*y*y*y*y*y*y);
        im_buf[base_8+8] += norms8[8]*(-16216200*x*x83+113513400*x10*x42-113513400*x14*x36+16216200*x78*y);
    };

    // -----------------------------------------------------------------------
    // Main pair loop — single FlatCellList traversal
    // Accumulates: RDF hist, bond_adj, Steinhardt, radial_var, local_aniso,
    //              bond_angle CSR count (nb_ptr).
    // -----------------------------------------------------------------------

    auto main_pass = [&](int i, int j) {
        const double dx = pts[i*3  ] - pts[j*3  ];
        const double dy = pts[i*3+1] - pts[j*3+1];
        const double dz = pts[i*3+2] - pts[j*3+2];
        const double d2 = dx*dx + dy*dy + dz*dz;
        if (d2 > cutoff2 || d2 < 1e-20) return;

        const double d     = std::sqrt(d2);
        const double inv_d = 1.0 / d;
        const double ux = dx * inv_d;
        const double uy = dy * inv_d;
        const double uz = dz * inv_d;

        // RDF histogram
        {
            int b = static_cast<int>(d / bin_width);
            if (b >= n_bins) b = n_bins - 1;
            ++rdf_hist[static_cast<std::size_t>(b)];
            ++rdf_pair_count;
        }

        // Bond adjacency (both directions for graph metrics)
        bond_adj[static_cast<std::size_t>(i)].push_back(j);
        bond_adj[static_cast<std::size_t>(j)].push_back(i);

        // Steinhardt fast-path ④ (both directions)
        accumulate_sh(i,  ux,  uy,  uz);
        accumulate_sh(j, -ux, -uy, -uz);

        // Radial variance
        rv_sum_d [static_cast<std::size_t>(i)] += d;
        rv_sum_d [static_cast<std::size_t>(j)] += d;
        rv_sum_d2[static_cast<std::size_t>(i)] += d2;
        rv_sum_d2[static_cast<std::size_t>(j)] += d2;
        ++rv_deg  [static_cast<std::size_t>(i)];
        ++rv_deg  [static_cast<std::size_t>(j)];

        // Local anisotropy: outer product components.
        // For atom j the direction is (-ux,-uy,-uz); outer products are identical
        // since (-u)(-u)^T = u u^T.  Both atoms receive the same increments.
        const double uxx = ux*ux, uxy = ux*uy, uxz = ux*uz;
        const double uyy = uy*uy, uyz = uy*uz, uzz = uz*uz;
        la_Txx[static_cast<std::size_t>(i)] += uxx;
        la_Txy[static_cast<std::size_t>(i)] += uxy;
        la_Txz[static_cast<std::size_t>(i)] += uxz;
        la_Tyy[static_cast<std::size_t>(i)] += uyy;
        la_Tyz[static_cast<std::size_t>(i)] += uyz;
        la_Tzz[static_cast<std::size_t>(i)] += uzz;
        la_Txx[static_cast<std::size_t>(j)] += uxx;
        la_Txy[static_cast<std::size_t>(j)] += uxy;
        la_Txz[static_cast<std::size_t>(j)] += uxz;
        la_Tyy[static_cast<std::size_t>(j)] += uyy;
        la_Tyz[static_cast<std::size_t>(j)] += uyz;
        la_Tzz[static_cast<std::size_t>(j)] += uzz;

        // Bond-angle entropy CSR: count neighbors
        ++nb_ptr[static_cast<std::size_t>(i)];
        ++nb_ptr[static_cast<std::size_t>(j)];
    };

    if (n >= CELL_THRESHOLD) {
        FlatCellList cl;
        cl.build(pts, n, cutoff);
        cl.for_each_pair(n, main_pass);
    } else {
        for (int i = 0; i < n-1; ++i)
            for (int j = i+1; j < n; ++j)
                main_pass(i, j);
    }

    // -----------------------------------------------------------------------
    // Bond-angle entropy: prefix sum → fill CSR unit vectors from bond_adj
    // (no second FlatCellList traversal needed)
    // -----------------------------------------------------------------------

    {
        std::int64_t acc = 0;
        for (int i = 0; i < n; ++i) {
            const std::int64_t cnt = nb_ptr[static_cast<std::size_t>(i)];
            nb_ptr[static_cast<std::size_t>(i)] = acc;
            acc += cnt;
        }
        nb_ptr[static_cast<std::size_t>(n)] = acc;
    }

    const std::int64_t total_nb = nb_ptr[static_cast<std::size_t>(n)];
    std::vector<double> nb_ux(static_cast<std::size_t>(total_nb));
    std::vector<double> nb_uy(static_cast<std::size_t>(total_nb));
    std::vector<double> nb_uz(static_cast<std::size_t>(total_nb));

    if (total_nb > 0) {
        // Fill CSR unit vectors directly from bond_adj — O(2P).
        // bond_adj[j] lists all neighbors k of atom j.  The direction
        // from j toward k is (pts_k - pts_j) normalized.
        std::vector<std::int64_t> nb_fill(static_cast<std::size_t>(n), 0);
        for (int j = 0; j < n; ++j) {
            for (int k : bond_adj[static_cast<std::size_t>(j)]) {
                const double dx2 = pts[k*3  ] - pts[j*3  ];
                const double dy2 = pts[k*3+1] - pts[j*3+1];
                const double dz2 = pts[k*3+2] - pts[j*3+2];
                const double d2b = dx2*dx2 + dy2*dy2 + dz2*dz2;
                // All entries in bond_adj passed the d2 < 1e-20 guard in
                // main_pass, so d2b >= 1e-20 is guaranteed here.
                const double inv_db = 1.0 / std::sqrt(d2b);
                const std::int64_t pos = nb_ptr[static_cast<std::size_t>(j)]
                                       + nb_fill[static_cast<std::size_t>(j)];
                nb_ux[static_cast<std::size_t>(pos)] = dx2 * inv_db;
                nb_uy[static_cast<std::size_t>(pos)] = dy2 * inv_db;
                nb_uz[static_cast<std::size_t>(pos)] = dz2 * inv_db;
                ++nb_fill[static_cast<std::size_t>(j)];
            }
        }
    }

    // Per-atom bond-angle histograms → bond_angle_entropy
    // Stack-allocated int[N_BINS_BA] histogram (zero heap alloc per atom).
    double ba_entropy_sum = 0.0;
    int    ba_atom_count  = 0;
    {
        constexpr double bin_inv_ba = N_BINS_BA / PI;
        for (int j = 0; j < n; ++j) {
            const std::int64_t beg = nb_ptr[static_cast<std::size_t>(j)];
            const std::int64_t end = nb_ptr[static_cast<std::size_t>(j) + 1];
            const std::int64_t k   = end - beg;
            if (k < 2) continue;

            int hist_ba[N_BINS_BA] = {};
            int pair_cnt = 0;

            for (std::int64_t a = beg; a < end - 1; ++a) {
                for (std::int64_t b = a + 1; b < end; ++b) {
                    const double dot =
                        nb_ux[static_cast<std::size_t>(a)] * nb_ux[static_cast<std::size_t>(b)]
                      + nb_uy[static_cast<std::size_t>(a)] * nb_uy[static_cast<std::size_t>(b)]
                      + nb_uz[static_cast<std::size_t>(a)] * nb_uz[static_cast<std::size_t>(b)];
                    const double clipped = dot < -1.0 ? -1.0 : (dot > 1.0 ? 1.0 : dot);
                    const double theta   = std::acos(clipped);
                    int bin = static_cast<int>(theta * bin_inv_ba);
                    if (bin >= N_BINS_BA) bin = N_BINS_BA - 1;
                    ++hist_ba[bin];
                    ++pair_cnt;
                }
            }

            if (pair_cnt == 0) continue;
            const double inv_total_ba = 1.0 / static_cast<double>(pair_cnt);
            double h = 0.0;
            for (int b = 0; b < N_BINS_BA; ++b) {
                if (hist_ba[b] > 0) {
                    const double p = hist_ba[b] * inv_total_ba;
                    h -= p * std::log(p);
                }
            }
            ba_entropy_sum += h;
            ++ba_atom_count;
        }
    }
    const double bond_angle_entropy =
        (ba_atom_count > 0) ? ba_entropy_sum / ba_atom_count : 0.0;

    // -----------------------------------------------------------------------
    // Sort adjacency lists (required for CC binary_search and Tarjan DFS)
    // -----------------------------------------------------------------------
    for (int i = 0; i < n; ++i)
        std::sort(bond_adj[static_cast<std::size_t>(i)].begin(),
                  bond_adj[static_cast<std::size_t>(i)].end());

    // -----------------------------------------------------------------------
    // RDF metrics: h_spatial and rdf_dev
    // -----------------------------------------------------------------------
    double h_spatial = 0.0;
    double rdf_dev   = 0.0;

    if (rdf_pair_count > 0) {
        const double total_rdf = static_cast<double>(rdf_pair_count);
        for (int c : rdf_hist) {
            if (c > 0) {
                const double p = c / total_rdf;
                h_spatial -= p * std::log(p);
            }
        }

        // r_bound = max |r_i - centroid|
        double cx=0.0, cy=0.0, cz=0.0;
        for (int i=0;i<n;++i){ cx+=pts[3*i]; cy+=pts[3*i+1]; cz+=pts[3*i+2]; }
        cx/=n; cy/=n; cz/=n;
        double r_bound = 0.0;
        for (int i=0;i<n;++i){
            const double ddx=pts[3*i]-cx, ddy=pts[3*i+1]-cy, ddz=pts[3*i+2]-cz;
            r_bound = std::max(r_bound, std::sqrt(ddx*ddx+ddy*ddy+ddz*ddz));
        }
        if (r_bound > 0.0) {
            const double rho = n / (4.0/3.0 * PI * r_bound*r_bound*r_bound);
            double sum_sq = 0.0;
            int    valid  = 0;
            for (int b=0;b<n_bins;++b){
                const double center = (b + 0.5) * bin_width;
                const double ideal  = rho * 4.0 * PI * center*center * bin_width
                                    * static_cast<double>(n) / 2.0;
                if (ideal > 0.0) {
                    const double ratio = rdf_hist[static_cast<std::size_t>(b)] / ideal - 1.0;
                    sum_sq += ratio * ratio;
                    ++valid;
                }
            }
            if (valid > 0) rdf_dev = std::sqrt(sum_sq / valid);
        }
    }

    // -----------------------------------------------------------------------
    // Graph metrics: ring_fraction, graph_cc, graph_lcc, charge_frustration,
    //                moran_I_chi
    // (algorithm identical to graph_metrics_cpp in _graph_core.cpp)
    // -----------------------------------------------------------------------

    double ring_fraction      = 0.0;
    double graph_cc           = 0.0;
    double graph_lcc          = 0.0;
    double charge_frustration = 0.0;
    double moran_I_chi        = 0.0;

    // ring_fraction: Tarjan iterative bridge-finding
    if (n >= 3) {
        std::vector<int>  disc(static_cast<std::size_t>(n), -1);
        std::vector<int>  low (static_cast<std::size_t>(n),  0);
        std::vector<bool> in_ring(static_cast<std::size_t>(n), false);
        std::unordered_set<int64_t> bridges;
        int timer = 0;

        struct Frame { int u, parent, idx; };
        std::vector<Frame> stk;
        stk.reserve(static_cast<std::size_t>(n));

        for (int start = 0; start < n; ++start) {
            if (disc[static_cast<std::size_t>(start)] != -1) continue;
            disc[static_cast<std::size_t>(start)] =
            low [static_cast<std::size_t>(start)] = timer++;
            stk.push_back({start, -1, 0});

            while (!stk.empty()) {
                auto& [u, par, idx] = stk.back();
                if (idx < static_cast<int>(
                        bond_adj[static_cast<std::size_t>(u)].size())) {
                    int v = bond_adj[static_cast<std::size_t>(u)][
                                static_cast<std::size_t>(idx++)];
                    if (disc[static_cast<std::size_t>(v)] == -1) {
                        disc[static_cast<std::size_t>(v)] =
                        low [static_cast<std::size_t>(v)] = timer++;
                        stk.push_back({v, u, 0});
                    } else if (v != par) {
                        if (disc[static_cast<std::size_t>(v)] <
                            low [static_cast<std::size_t>(u)])
                            low[static_cast<std::size_t>(u)] =
                                disc[static_cast<std::size_t>(v)];
                    }
                } else {
                    stk.pop_back();
                    if (!stk.empty()) {
                        int pu = stk.back().u;
                        if (low[static_cast<std::size_t>(u)] <
                            low[static_cast<std::size_t>(pu)])
                            low[static_cast<std::size_t>(pu)] =
                                low[static_cast<std::size_t>(u)];
                        if (low[static_cast<std::size_t>(u)] >
                            disc[static_cast<std::size_t>(pu)]) {
                            int a=std::min(pu,u), b2=std::max(pu,u);
                            bridges.insert((int64_t)a<<32|(uint32_t)b2);
                        }
                    }
                }
            }
        }

        for (int i = 0; i < n; ++i) {
            for (int j2 : bond_adj[static_cast<std::size_t>(i)]) {
                int a=std::min(i,j2), b2=std::max(i,j2);
                if (bridges.find((int64_t)a<<32|(uint32_t)b2) == bridges.end()) {
                    in_ring[static_cast<std::size_t>(i)] = true; break;
                }
            }
        }
        int ring_count = 0;
        for (int i=0;i<n;++i)
            if (in_ring[static_cast<std::size_t>(i)]) ++ring_count;
        ring_fraction = static_cast<double>(ring_count) / n;
    }

    // charge_frustration
    {
        std::vector<double> diffs;
        diffs.reserve(static_cast<std::size_t>(n) * 4);
        for (int i=0;i<n;++i)
            for (int j2 : bond_adj[static_cast<std::size_t>(i)])
                if (j2 > i)
                    diffs.push_back(std::fabs(en[i] - en[j2]));
        if (diffs.size() >= 2) {
            const int m = static_cast<int>(diffs.size());
            double mean = 0.0;
            for (double v : diffs) mean += v;
            mean /= m;
            double var = 0.0;
            for (double v : diffs) { double dd=v-mean; var+=dd*dd; }
            charge_frustration = var / m;
        }
    }

    // graph_lcc and graph_cc
    {
        UnionFind uf(n);
        for (int i=0;i<n;++i)
            for (int j2 : bond_adj[static_cast<std::size_t>(i)])
                if (j2 > i) uf.unite(i, j2);

        std::vector<int> comp_size(static_cast<std::size_t>(n), 0);
        for (int i=0;i<n;++i) ++comp_size[static_cast<std::size_t>(uf.find(i))];
        graph_lcc = static_cast<double>(
            *std::max_element(comp_size.begin(), comp_size.end())) / n;

        double cc_sum = 0.0;
        int    cc_cnt = 0;
        for (int i=0;i<n;++i){
            const auto& nb = bond_adj[static_cast<std::size_t>(i)];
            const int   k  = static_cast<int>(nb.size());
            if (k < 2) continue;
            int tri = 0;
            for (int a=0;a<k;++a)
                for (int bv=a+1;bv<k;++bv){
                    const auto& nb_a = bond_adj[static_cast<std::size_t>(nb[static_cast<std::size_t>(a)])];
                    if (std::binary_search(nb_a.begin(), nb_a.end(),
                                           nb[static_cast<std::size_t>(bv)]))
                        ++tri;
                }
            cc_sum += static_cast<double>(tri) / (k*(k-1)/2);
            ++cc_cnt;
        }
        if (cc_cnt > 0) graph_cc = cc_sum / cc_cnt;
    }

    // moran_I_chi
    {
        double chi_bar = 0.0;
        for (int i=0;i<n;++i) chi_bar += en[i];
        chi_bar /= n;
        std::vector<double> dev(static_cast<std::size_t>(n));
        double denom_m = 0.0;
        for (int i=0;i<n;++i){ dev[static_cast<std::size_t>(i)]=en[i]-chi_bar; denom_m+=dev[static_cast<std::size_t>(i)]*dev[static_cast<std::size_t>(i)]; }
        if (denom_m > 1e-30) {
            double numer_m=0.0, W_sum_m=0.0;
            for (int i=0;i<n;++i)
                for (int j2 : bond_adj[static_cast<std::size_t>(i)])
                    if (j2 > i) {
                        numer_m += 2.0*dev[static_cast<std::size_t>(i)]*dev[static_cast<std::size_t>(j2)];
                        W_sum_m += 2.0;
                    }
            if (W_sum_m > 1e-30) {
                moran_I_chi = (static_cast<double>(n)/W_sum_m)*(numer_m/denom_m);
                if (moran_I_chi > 1.0) moran_I_chi = 1.0;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Steinhardt Q4, Q6, Q8  (fast-path ④, per-atom arrays)
    // -----------------------------------------------------------------------

    F64Array q4_arr(static_cast<py::ssize_t>(n));
    F64Array q6_arr(static_cast<py::ssize_t>(n));
    F64Array q8_arr(static_cast<py::ssize_t>(n));
    double* q4 = static_cast<double*>(q4_arr.request().ptr);
    double* q6 = static_cast<double*>(q6_arr.request().ptr);
    double* q8 = static_cast<double*>(q8_arr.request().ptr);

    static constexpr int L_VALUES[3] = {4, 6, 8};
    for (int li = 0; li < N_L_SH; ++li) {
        const int    lv       = L_VALUES[li];
        const double factor   = FOURPI / (2.0 * lv + 1.0);
        double* ql = (li==0) ? q4 : (li==1) ? q6 : q8;
        for (int i = 0; i < n; ++i) {
            const double d = sh_deg[static_cast<std::size_t>(i)];
            if (d == 0.0) { ql[i] = 0.0; continue; }
            const double inv_d = 1.0 / d;
            const std::size_t base = static_cast<std::size_t>(i * N_L_SH * LM1 + li * LM1);
            double qlm_sq = 0.0;
            {
                const double r = re_buf[base] * inv_d;
                qlm_sq += r * r;
            }
            for (int m = 1; m <= lv; ++m) {
                const double r = re_buf[base + static_cast<std::size_t>(m)] * inv_d;
                const double k = im_buf[base + static_cast<std::size_t>(m)] * inv_d;
                qlm_sq += 2.0 * (r*r + k*k);
            }
            ql[i] = std::sqrt(factor * qlm_sq);
        }
    }

    // -----------------------------------------------------------------------
    // Coordination variance  (population variance of per-atom degree)
    // -----------------------------------------------------------------------
    double coordination_variance = 0.0;
    if (rdf_pair_count > 0) {
        // rv_deg contains the same degree as bond_adj sizes.
        double deg_mean = 0.0;
        for (int i=0;i<n;++i) deg_mean += rv_deg[static_cast<std::size_t>(i)];
        deg_mean /= n;
        double deg_var = 0.0;
        for (int i=0;i<n;++i) {
            const double dd = rv_deg[static_cast<std::size_t>(i)] - deg_mean;
            deg_var += dd * dd;
        }
        coordination_variance = deg_var / n;
    }

    // -----------------------------------------------------------------------
    // Radial variance
    // -----------------------------------------------------------------------
    double radial_variance = 0.0;
    {
        int   rv_valid = 0;
        double rv_sum  = 0.0;
        for (int i = 0; i < n; ++i) {
            const int ki = rv_deg[static_cast<std::size_t>(i)];
            if (ki == 0) continue;
            const double mean_d  = rv_sum_d [static_cast<std::size_t>(i)] / ki;
            const double mean_d2 = rv_sum_d2[static_cast<std::size_t>(i)] / ki;
            const double vari = mean_d2 - mean_d * mean_d;
            rv_sum += (vari > 0.0) ? vari : 0.0;
            ++rv_valid;
        }
        if (rv_valid > 0) radial_variance = rv_sum / rv_valid;
    }

    // -----------------------------------------------------------------------
    // Local anisotropy  (Frobenius κ² without eigvalsh)
    // κ²_i = 1.5 * ||T_i||²_F / (tr T_i)² - 0.5,  clipped to [0, 1]
    // -----------------------------------------------------------------------
    double local_anisotropy = 0.0;
    {
        double la_sum  = 0.0;
        int    la_valid = 0;
        for (int i = 0; i < n; ++i) {
            const int ki = rv_deg[static_cast<std::size_t>(i)];
            if (ki == 0) continue;
            const double ski = static_cast<double>(ki);
            const double Txx = la_Txx[static_cast<std::size_t>(i)] / ski;
            const double Txy = la_Txy[static_cast<std::size_t>(i)] / ski;
            const double Txz = la_Txz[static_cast<std::size_t>(i)] / ski;
            const double Tyy = la_Tyy[static_cast<std::size_t>(i)] / ski;
            const double Tyz = la_Tyz[static_cast<std::size_t>(i)] / ski;
            const double Tzz = la_Tzz[static_cast<std::size_t>(i)] / ski;
            const double tr  = Txx + Tyy + Tzz;
            const double tr2 = tr * tr;
            if (tr2 <= 1e-24) continue;
            const double fr2 = Txx*Txx + Tyy*Tyy + Tzz*Tzz
                             + 2.0*(Txy*Txy + Txz*Txz + Tyz*Tyz);
            double kap2 = 1.5 * fr2 / tr2 - 0.5;
            if (kap2 < 0.0) kap2 = 0.0;
            if (kap2 > 1.0) kap2 = 1.0;
            la_sum += kap2;
            ++la_valid;
        }
        if (la_valid > 0) local_anisotropy = la_sum / la_valid;
    }

    // -----------------------------------------------------------------------
    // Assemble result dict
    // -----------------------------------------------------------------------
    py::dict result;
    result["h_spatial"]            = h_spatial;
    result["rdf_dev"]              = rdf_dev;
    result["graph_lcc"]            = graph_lcc;
    result["graph_cc"]             = graph_cc;
    result["ring_fraction"]        = ring_fraction;
    result["charge_frustration"]   = charge_frustration;
    result["moran_I_chi"]          = moran_I_chi;
    result["Q4"]                   = q4_arr;
    result["Q6"]                   = q6_arr;
    result["Q8"]                   = q8_arr;
    result["bond_angle_entropy"]   = bond_angle_entropy;
    result["coordination_variance"]= coordination_variance;
    result["radial_variance"]      = radial_variance;
    result["local_anisotropy"]     = local_anisotropy;
    return result;
}

// ---------------------------------------------------------------------------
PYBIND11_MODULE(_combined_core, m) {
    m.doc() =
        "_combined_core: single-pass computation of all pasted metrics (v0.4.0).\n"
        "\n"
        "Builds FlatCellList exactly once and accumulates all pair-based metrics\n"
        "in a single for_each_pair traversal, then fills bond-angle CSR unit\n"
        "vectors from the already-populated bond_adj lists (no second cell-list\n"
        "traversal).  Replaces four separate FlatCellList builds in the individual\n"
        "_rdf_h_cpp / _graph_metrics_cpp / _steinhardt_per_atom / _bond_angle_entropy_cpp\n"
        "calls.  Speedup: ~1.9x at N=1000 vs calling those functions separately.\n"
        "\n"
        "Steinhardt uses the same fast-path ④ Cartesian-polynomial expressions\n"
        "(joint SymPy CSE for l=4,6,8) as _steinhardt_core.\n"
        "Bond-angle histogram bins are fixed at N_BINS_BA=36 (compile-time constant),\n"
        "matching _bond_angle_core.";
    m.def(
        "all_metrics_cpp", &all_metrics_cpp,
        py::arg("pts"), py::arg("radii"), py::arg("en_vals"),
        py::arg("cutoff"), py::arg("n_bins"),
        R"(
Compute all pasted structural metrics in a single FlatCellList pass.

Parameters
----------
pts     : (n, 3) float64  atom positions (Angstrom)
radii   : (n,)   float64  covalent radii (Angstrom)
en_vals : (n,)   float64  Pauling electronegativity per atom
cutoff  : float           neighbor distance cutoff (Angstrom)
n_bins  : int             histogram bins for h_spatial / rdf_dev

Returns
-------
dict with keys:
  h_spatial, rdf_dev                          -- distance-histogram metrics
  graph_lcc, graph_cc, ring_fraction,
    charge_frustration, moran_I_chi           -- graph metrics
  Q4, Q6, Q8                                  -- per-atom Steinhardt (float64 arrays)
  bond_angle_entropy, coordination_variance,
    radial_variance, local_anisotropy          -- adversarial metrics

Notes
-----
Q4/Q6/Q8 are per-atom arrays; take .mean() in Python to get the
scalar values expected by compute_all_metrics.

All metrics return 0.0 (or zero arrays) when n < 2 or no pairs
exist within the cutoff.  Coincident atoms (d < 1e-10 A) are
excluded from all pair-based calculations.
        )");
}
