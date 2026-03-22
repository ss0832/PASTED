/**
 * pasted._ext._steinhardt_core
 * ============================
 * Sparse Steinhardt Q_l computation (C++17).
 *
 * Exported function
 * -----------------
 *   steinhardt_per_atom(pts, cutoff, l_values)
 *       -> dict {"Q{l}": ndarray(n,), ...}
 *
 * Algorithm
 * ---------
 * For each atom i, iterate only over neighbors j (d_ij <= cutoff) instead
 * of the full N^2 pair matrix used by the dense Python/scipy path.
 *
 * Spherical harmonics: Condon-Shortley convention
 *   Y_l^m(theta, phi) = N_lm * P_l^m(cos theta) * exp(i*m*phi)
 *
 * Exploiting |Y_l^{-m}|^2 = |Y_l^m|^2 we accumulate only m = 0..l (not
 * negative m) and multiply the m > 0 contributions by 2.
 *
 * Associated Legendre polynomials P_l^m(x) are computed via the standard
 * three-term recurrence; normalization factors N_lm are pre-computed from
 * factorial tables.
 *
 * Neighbour finding
 * -----------------
 * N < CELL_THRESHOLD : O(N^2) full-pair scan.
 * N >= CELL_THRESHOLD: FlatCellList O(N) amortized (same structure as
 *                       _relax_core; cell size = cutoff).
 *
 * Complexity per structure
 * ------------------------
 *   O(n_bonds * l_max^2) where n_bonds = N * mean_neighbors.
 *   For N=2000, cutoff=4 A, ~30 neighbors/atom:
 *     n_bonds ~ 60 000  vs  N^2 = 4 000 000 in the dense Python path.
 *
 * Threading
 * ---------
 * All computation is single-threaded.  A two-pass neighbor-list build +
 * OpenMP parallel loop was introduced in v0.2.3 but OpenMP was never linked,
 * making the intermediate nb_list allocation pure overhead.  The original
 * single-pass lambda accumulation was restored in v0.2.9.
 *
 * Hot-path optimisations (v0.3.7)
 * --------------------------------
 * ① + ② — atan2 elimination + Chebyshev recurrence for cos(mφ)/sin(mφ).
 *   The former code called std::atan2 once per bond and then issued
 *   l_max separate std::cos / std::sin pairs (18 libm calls at l_max=8,
 *   each ≈ 20–50 CPU cycles).  The new code computes
 *     cos_phi = dx/r_xy,  sin_phi = dy/r_xy          (1 sqrt + 2 divs)
 *   and derives all higher orders via the Chebyshev recurrence
 *     cos(m·φ) = 2·cos_phi·cos((m-1)·φ) − cos((m-2)·φ)   (2 mults + 1 sub)
 *     sin(m·φ) = 2·cos_phi·sin((m-1)·φ) − sin((m-2)·φ)
 *   Total: 1 sqrt + (l_max−1)×4 arithmetic ops vs. 1 atan2 + 18 libm calls.
 *   Measured speedup on the phi-trig component: ~4× (vectorised benchmark).
 *
 * ③ — Stack-allocated P_lm table.
 *   compute_plm now writes into a caller-supplied double[L_MAX+1][L_MAX+1]
 *   on the stack instead of a heap-allocated vector<vector<double>>.
 *   Eliminates one heap alloc + assign() per bond and keeps the 936-byte
 *   table in L1 cache for the full duration of the bond loop.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace py = pybind11;
using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

static constexpr int    CELL_THRESHOLD = 64;
static constexpr double PI             = 3.14159265358979323846;
static constexpr double FOURPI         = 4.0 * PI;
static constexpr int    L_MAX          = 12;   // maximum l we ever need

// ---------------------------------------------------------------------------
// Factorial table  (0! .. 2*L_MAX!)
// ---------------------------------------------------------------------------

static double g_fac[2 * L_MAX + 1];
static bool   g_fac_init = false;

static void init_fac() {
    if (g_fac_init) return;
    g_fac[0] = 1.0;
    for (int i = 1; i <= 2 * L_MAX; ++i) g_fac[i] = g_fac[i - 1] * i;
    g_fac_init = true;
}

// N_lm = sqrt( (2l+1)/4pi * (l-m)!/(l+m)! )
static inline double norm_lm(int l, int m) {
    init_fac();
    return std::sqrt((2.0 * l + 1.0) / FOURPI * g_fac[l - m] / g_fac[l + m]);
}

// ---------------------------------------------------------------------------
// Associated Legendre polynomial P_l^m(x),  0 <= m <= l,  x = cos(theta)
// Standard (non-Condon-Shortley) recurrence; sign convention does not
// affect Q_l values since we compute absolute squares.
//
// Recurrences:
//   P_0^0          = 1
//   P_{m+1}^{m+1}  = -(2m+1) * sin(theta) * P_m^m
//   P_{m+1}^m      = (2m+1)  * x           * P_m^m
//   P_l^m           = ((2l-1)*x*P_{l-1}^m - (l+m-1)*P_{l-2}^m) / (l-m)
//
// Returns plm[l][m] for all 0<=m<=l in 0..lmax.
// ---------------------------------------------------------------------------

// ③ Stack-allocated P_lm: caller supplies double plm[L_MAX+1][L_MAX+1].
// No heap allocation per bond; the 936-byte table lives in L1 cache.
static void compute_plm(double x, double s /* sin(theta) */, int lmax,
                         double plm[][L_MAX + 1]) {
    // Zero only the triangle we will fill.
    for (int l = 0; l <= lmax; ++l)
        for (int m = 0; m <= l; ++m)
            plm[l][m] = 0.0;

    plm[0][0] = 1.0;
    if (lmax == 0) return;

    for (int m = 0; m < lmax; ++m) {
        const double pmm = plm[m][m];
        plm[m + 1][m]     = (2 * m + 1) * x * pmm;   // superdiagonal
        plm[m + 1][m + 1] = -(2 * m + 1) * s * pmm;  // diagonal
    }

    // Three-term recurrence for l >= m+2
    for (int m = 0; m <= lmax; ++m)
        for (int l = m + 2; l <= lmax; ++l)
            plm[l][m] = ((2*l - 1)*x*plm[l-1][m] - (l+m-1)*plm[l-2][m])
                        / static_cast<double>(l - m);
}

// ---------------------------------------------------------------------------
// FlatCellList (read-only version; adapted from _relax_core)
// ---------------------------------------------------------------------------

struct FlatCellList {
    double inv_cell;
    int    nx, ny, nz;
    double ox, oy, oz;
    std::vector<int> cell_head;
    std::vector<int> next;

    void build(const double* pts, int n, double cell_size) {
        inv_cell = 1.0 / cell_size;
        double xmin = pts[0], xmax = pts[0];
        double ymin = pts[1], ymax = pts[1];
        double zmin = pts[2], zmax = pts[2];
        for (int i = 1; i < n; ++i) {
            xmin = std::min(xmin, pts[i * 3 + 0]); xmax = std::max(xmax, pts[i * 3 + 0]);
            ymin = std::min(ymin, pts[i * 3 + 1]); ymax = std::max(ymax, pts[i * 3 + 1]);
            zmin = std::min(zmin, pts[i * 3 + 2]); zmax = std::max(zmax, pts[i * 3 + 2]);
        }
        ox = xmin - cell_size; oy = ymin - cell_size; oz = zmin - cell_size;
        nx = static_cast<int>((xmax - ox) * inv_cell) + 2;
        ny = static_cast<int>((ymax - oy) * inv_cell) + 2;
        nz = static_cast<int>((zmax - oz) * inv_cell) + 2;
        cell_head.assign(static_cast<std::size_t>(nx * ny * nz), -1);
        next.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const int cx = static_cast<int>((pts[i * 3 + 0] - ox) * inv_cell);
            const int cy = static_cast<int>((pts[i * 3 + 1] - oy) * inv_cell);
            const int cz = static_cast<int>((pts[i * 3 + 2] - oz) * inv_cell);
            const int cid = cx + nx * (cy + ny * cz);
            next[static_cast<std::size_t>(i)] = cell_head[static_cast<std::size_t>(cid)];
            cell_head[static_cast<std::size_t>(cid)] = i;
        }
    }

    // Call process(i, j) for all unique unordered pairs within cutoff.
    // Also calls process(j, i) so that both directions are accumulated.
    template <typename F>
    void for_each_neighbor(const double* pts, int /*n*/, double cutoff2, F process) const {
        const double co2 = cutoff2;
        for (int cz = 0; cz < nz; ++cz)
        for (int cy = 0; cy < ny; ++cy)
        for (int cx = 0; cx < nx; ++cx) {
            const int cid = cx + nx * (cy + ny * cz);
            int i = cell_head[static_cast<std::size_t>(cid)];
            while (i >= 0) {
                // Same-cell pairs (upper triangle, then both directions)
                int j = next[static_cast<std::size_t>(i)];
                while (j >= 0) {
                    const double dx = pts[i*3]-pts[j*3];
                    const double dy = pts[i*3+1]-pts[j*3+1];
                    const double dz = pts[i*3+2]-pts[j*3+2];
                    if (dx*dx + dy*dy + dz*dz <= co2) { process(i, j); process(j, i); }
                    j = next[static_cast<std::size_t>(j)];
                }
                // Cross-cell (only higher-index cells to avoid duplicates)
                for (int ddz = -1; ddz <= 1; ++ddz)
                for (int ddy = -1; ddy <= 1; ++ddy)
                for (int ddx = -1; ddx <= 1; ++ddx) {
                    if (ddx == 0 && ddy == 0 && ddz == 0) continue;
                    const int ncx=cx+ddx, ncy=cy+ddy, ncz=cz+ddz;
                    if (ncx<0||ncy<0||ncz<0||ncx>=nx||ncy>=ny||ncz>=nz) continue;
                    const int nid = ncx + nx*(ncy + ny*ncz);
                    if (nid <= cid) continue;
                    int k = cell_head[static_cast<std::size_t>(nid)];
                    while (k >= 0) {
                        const double dx = pts[i*3]-pts[k*3];
                        const double dy = pts[i*3+1]-pts[k*3+1];
                        const double dz = pts[i*3+2]-pts[k*3+2];
                        if (dx*dx + dy*dy + dz*dz <= co2) { process(i, k); process(k, i); }
                        k = next[static_cast<std::size_t>(k)];
                    }
                }
                i = next[static_cast<std::size_t>(i)];
            }
        }
    }
};

// O(N^2) full-pair version for small N
template <typename F>
static void for_each_neighbor_full(const double* pts, int n, double cutoff2, F process) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            const double dx = pts[i*3]-pts[j*3];
            const double dy = pts[i*3+1]-pts[j*3+1];
            const double dz = pts[i*3+2]-pts[j*3+2];
            if (dx*dx + dy*dy + dz*dz <= cutoff2) process(i, j);
        }
}

// ---------------------------------------------------------------------------
// Main computation
// ---------------------------------------------------------------------------

py::dict steinhardt_per_atom_cpp(F64Array pts_in, double cutoff,
                                  std::vector<int> l_values) {
    const auto buf = pts_in.request();
    const int  n   = static_cast<int>(buf.shape[0]);
    const double* pts = static_cast<const double*>(buf.ptr);

    if (l_values.empty() || n == 0) return py::dict{};
    for (int lv : l_values)
        if (lv < 0 || lv > L_MAX)
            throw std::runtime_error("l value out of range [0, " +
                                     std::to_string(L_MAX) + "]");

    const int n_l   = static_cast<int>(l_values.size());
    const int l_max = *std::max_element(l_values.begin(), l_values.end());

    // Precompute normalization factors N_lm
    std::vector<std::vector<double>> norms(static_cast<std::size_t>(l_max + 1));
    for (int l = 0; l <= l_max; ++l) {
        norms[static_cast<std::size_t>(l)].resize(static_cast<std::size_t>(l + 1));
        for (int m = 0; m <= l; ++m)
            norms[static_cast<std::size_t>(l)][static_cast<std::size_t>(m)] = norm_lm(l, m);
    }

    // Per-atom accumulators — layout (n, n_l, lm1), atom index OUTERMOST.
    // Index formula:  i * n_l * lm1  +  li * lm1  +  m
    //
    // The former layout was (n_l, lm1, n) with index li*lm1*n + m*n + i.
    // That placed consecutive m-writes N*8 bytes apart, causing L2→L3 cache
    // spill for N ≥ ~1000 and superlinear wall-time growth.  With the new
    // layout all (li, m) writes for a given atom i are contiguous (stride 8 B),
    // so every bond's accumulation touches a single cache line regardless of N.
    const int lm1 = l_max + 1;
    std::vector<double> re_buf(static_cast<std::size_t>(n * n_l * lm1), 0.0);
    std::vector<double> im_buf(static_cast<std::size_t>(n * n_l * lm1), 0.0);
    std::vector<double> deg(static_cast<std::size_t>(n), 0.0);

    auto accumulate = [&](int i, int j) {
        const double dxr = pts[i*3+0] - pts[j*3+0];
        const double dyr = pts[i*3+1] - pts[j*3+1];
        const double dzr = pts[i*3+2] - pts[j*3+2];
        const double d   = std::sqrt(dxr*dxr + dyr*dyr + dzr*dzr);
        if (d < 1e-10) return;  // coincident atoms
        const double inv_d = 1.0 / d;
        const double cos_t = dzr * inv_d;             // cos(theta) = z/r
        const double sin_t = std::sqrt(std::max(0.0, 1.0 - cos_t*cos_t));

        // ① + ② — atan2 elimination + Chebyshev recurrence.
        // cos_phi = x/r_xy and sin_phi = y/r_xy are computed from a single
        // sqrt instead of calling atan2.  Higher orders follow from the
        // two-term Chebyshev recurrence (2 mults + 1 sub each) instead of
        // l_max independent std::cos/std::sin calls.
        const double r_xy  = std::sqrt(dxr*dxr + dyr*dyr);
        const double cp    = (r_xy > 1e-10) ? dxr / r_xy : 1.0; // cos(phi)
        const double sp    = (r_xy > 1e-10) ? dyr / r_xy : 0.0; // sin(phi)
        double cos_m[L_MAX + 1], sin_m[L_MAX + 1];
        cos_m[0] = 1.0;  sin_m[0] = 0.0;
        if (l_max >= 1) { cos_m[1] = cp; sin_m[1] = sp; }
        for (int m = 2; m <= l_max; ++m) {
            cos_m[m] = 2.0 * cp * cos_m[m-1] - cos_m[m-2];
            sin_m[m] = 2.0 * cp * sin_m[m-1] - sin_m[m-2];
        }

        // ③ — Stack-allocated P_lm (no heap alloc per bond).
        double plm[L_MAX + 1][L_MAX + 1];
        deg[static_cast<std::size_t>(i)] += 1.0;
        compute_plm(cos_t, sin_t, l_max, plm);

        // Base offset for atom i — all writes for this bond stay within
        // n_l * lm1 doubles from here (one cache line for typical n_l=3, lm1=9).
        const std::size_t base_i = static_cast<std::size_t>(i) *
                                   static_cast<std::size_t>(n_l * lm1);
        for (int li = 0; li < n_l; ++li) {
            const int l = l_values[static_cast<std::size_t>(li)];
            const std::size_t base_li = base_i +
                                        static_cast<std::size_t>(li * lm1);
            for (int m = 0; m <= l; ++m) {
                const double Nlm_Plm =
                    norms[static_cast<std::size_t>(l)][static_cast<std::size_t>(m)] *
                    plm[l][m];
                const std::size_t idx = base_li + static_cast<std::size_t>(m);
                re_buf[idx] += Nlm_Plm * cos_m[m];
                im_buf[idx] += Nlm_Plm * sin_m[m];
            }
        }
    };

    if (n < CELL_THRESHOLD) {
        for_each_neighbor_full(pts, n, cutoff * cutoff, accumulate);
    } else {
        FlatCellList cl;
        cl.build(pts, n, cutoff);
        cl.for_each_neighbor(pts, n, cutoff * cutoff, accumulate);
    }

    // Build result dict
    py::dict result;
    for (int li = 0; li < n_l; ++li) {
        const int l = l_values[static_cast<std::size_t>(li)];
        F64Array ql_arr(static_cast<py::ssize_t>(n));
        double* ql = static_cast<double*>(ql_arr.request().ptr);
        const double factor = FOURPI / (2.0 * l + 1.0);

        for (int i = 0; i < n; ++i) {
            const double d = deg[static_cast<std::size_t>(i)];
            if (d == 0.0) { ql[i] = 0.0; continue; }
            const double inv_d = 1.0 / d;

            const std::size_t base = static_cast<std::size_t>(i * n_l * lm1 + li * lm1);
            double qlm_sq = 0.0;
            // m = 0: only real part (sin(0)=0)
            {
                const double r = re_buf[base] * inv_d;
                qlm_sq += r * r;
            }
            // m = 1..l: both real and imaginary, weight 2
            for (int m = 1; m <= l; ++m) {
                const double r = re_buf[base + static_cast<std::size_t>(m)] * inv_d;
                const double k = im_buf[base + static_cast<std::size_t>(m)] * inv_d;
                qlm_sq += 2.0 * (r * r + k * k);
            }
            ql[i] = std::sqrt(factor * qlm_sq);
        }
        result[py::str("Q" + std::to_string(l))] = ql_arr;
    }
    return result;
}

// ---------------------------------------------------------------------------

PYBIND11_MODULE(_steinhardt_core, m) {
    m.doc() =
        "pasted._ext._steinhardt_core: sparse Steinhardt Q_l (C++17).\n"
        "Uses an explicit neighbor list (FlatCellList for N>=64) to avoid\n"
        "the O(N^2) dense matrix built by the Python/scipy path.\n"
        "\n"
        "Accumulator layout (v0.3.6+): (N, n_l, lm1) — atom index outermost.\n"
        "All (l_idx, m) writes for a given atom are contiguous (stride 8 B),\n"
        "eliminating the L2-to-L3 cache-thrash of the former (n_l, lm1, N) layout.\n"
        "\n"
        "v0.3.7: atan2 replaced by sqrt+div (cos_phi/sin_phi); higher-order\n"
        "cos(m*phi)/sin(m*phi) via Chebyshev recurrence (2 mults+sub each).\n"
        "P_lm table now stack-allocated (double[13][13]), no heap alloc per bond.";
    m.def(
        "steinhardt_per_atom", &steinhardt_per_atom_cpp,
        py::arg("pts"), py::arg("cutoff"), py::arg("l_values"),
        R"(
Compute per-atom Steinhardt Q_l for each l in l_values.

Parameters
----------
pts      : (n, 3) float64  – Cartesian coordinates (Angstrom)
cutoff   : float           – neighbor distance cutoff (Angstrom)
l_values : list[int]       – l indices (e.g. [4, 6, 8])

Returns
-------
dict  mapping "Q{l}" -> (n,) float64 array of per-atom Q_l values.
Atoms with no neighbors within cutoff are assigned Q_l = 0.

Notes
-----
Accumulator buffer layout is (N, n_l, lm1) with atom index outermost
(v0.3.6+).  Each bond's writes are contiguous regardless of N, avoiding
the superlinear cache-pressure effect of the former (n_l, lm1, N) layout.
        )"
    );
}
