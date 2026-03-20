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
 * For each atom i, iterate only over neighbours j (d_ij <= cutoff) instead
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
 * N >= CELL_THRESHOLD: FlatCellList O(N) amortised (same structure as
 *                       _relax_core; cell size = cutoff).
 *
 * Complexity per structure
 * ------------------------
 *   O(n_bonds * l_max^2) where n_bonds = N * mean_neighbours.
 *   For N=2000, cutoff=4 A, ~30 neighbours/atom:
 *     n_bonds ~ 60 000  vs  N^2 = 4 000 000 in the dense Python path.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

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

static void compute_plm(double x, double s /* sin(theta) */, int lmax,
                         std::vector<std::vector<double>>& plm) {
    plm.resize(static_cast<std::size_t>(lmax + 1));
    for (int l = 0; l <= lmax; ++l)
        plm[static_cast<std::size_t>(l)].assign(static_cast<std::size_t>(l + 1), 0.0);

    plm[0][0] = 1.0;
    if (lmax == 0) return;

    for (int m = 0; m < lmax; ++m) {
        const double pmm = plm[static_cast<std::size_t>(m)][static_cast<std::size_t>(m)];
        // Superdiagonal: P_{m+1}^m = (2m+1)*x * P_m^m
        plm[static_cast<std::size_t>(m + 1)][static_cast<std::size_t>(m)] = (2 * m + 1) * x * pmm;
        // Diagonal:      P_{m+1}^{m+1} = -(2m+1)*s * P_m^m
        plm[static_cast<std::size_t>(m + 1)][static_cast<std::size_t>(m + 1)] = -(2 * m + 1) * s * pmm;
    }

    // Three-term recurrence for l >= m+2
    for (int m = 0; m <= lmax; ++m) {
        for (int l = m + 2; l <= lmax; ++l) {
            plm[static_cast<std::size_t>(l)][static_cast<std::size_t>(m)] =
                ((2 * l - 1) * x * plm[static_cast<std::size_t>(l - 1)][static_cast<std::size_t>(m)]
                 - (l + m - 1) * plm[static_cast<std::size_t>(l - 2)][static_cast<std::size_t>(m)])
                / static_cast<double>(l - m);
        }
    }
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

    // Per-atom accumulators: re[l_idx][m][i], im[l_idx][m][i]
    // Flattened for cache efficiency: re_buf[l_idx * (l_max+1) * n + m * n + i]
    // Use (l_idx, m, i) layout with m in 0..l_values[l_idx]
    // For simplicity, allocate (n_l, l_max+1, n) buffers.
    const int lm1 = l_max + 1;
    std::vector<double> re_buf(static_cast<std::size_t>(n_l * lm1 * n), 0.0);
    std::vector<double> im_buf(static_cast<std::size_t>(n_l * lm1 * n), 0.0);
    std::vector<double> deg(static_cast<std::size_t>(n), 0.0);

    // Build neighbour list once (thread-safe read-only after construction)
    std::vector<std::vector<int>> nb_list(static_cast<std::size_t>(n));
    if (n < CELL_THRESHOLD) {
        const double cut2 = cutoff * cutoff;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                const double dx = pts[i*3]-pts[j*3];
                const double dy = pts[i*3+1]-pts[j*3+1];
                const double dz = pts[i*3+2]-pts[j*3+2];
                if (dx*dx+dy*dy+dz*dz <= cut2)
                    nb_list[static_cast<std::size_t>(i)].push_back(j);
            }
    } else {
        FlatCellList cl;
        cl.build(pts, n, cutoff);
        const double cut2 = cutoff * cutoff;
        cl.for_each_neighbor(pts, n, cut2, [&](int i, int j){
            nb_list[static_cast<std::size_t>(i)].push_back(j);
            nb_list[static_cast<std::size_t>(j)].push_back(i);
        });
    }

#ifdef _OPENMP
    // A guard: only parallelize when > 2 threads available (omp `if` clause)
#pragma omp parallel for schedule(dynamic,64) if(omp_get_max_threads() > 2)
#endif
    for (int i = 0; i < n; ++i) {
        const auto& nbi = nb_list[static_cast<std::size_t>(i)];
        // thread-local P_lm table
        std::vector<std::vector<double>> plm_local;
        double deg_i = static_cast<double>(nbi.size());
        deg[static_cast<std::size_t>(i)] = deg_i;

        for (int j : nbi) {
            const double dxr = pts[i*3+0] - pts[j*3+0];
            const double dyr = pts[i*3+1] - pts[j*3+1];
            const double dzr = pts[i*3+2] - pts[j*3+2];
            const double d   = std::sqrt(dxr*dxr + dyr*dyr + dzr*dzr);
            if (d < 1e-10) continue;
            const double inv_d = 1.0 / d;
            const double cos_t = dzr * inv_d;
            const double sin_t = std::sqrt(std::max(0.0, 1.0 - cos_t*cos_t));
            const double phi   = std::atan2(dyr, dxr);

            compute_plm(cos_t, sin_t, l_max, plm_local);

            for (int li = 0; li < n_l; ++li) {
                const int l = l_values[static_cast<std::size_t>(li)];
                for (int m = 0; m <= l; ++m) {
                    const double Nlm_Plm =
                        norms[static_cast<std::size_t>(l)][static_cast<std::size_t>(m)] *
                        plm_local[static_cast<std::size_t>(l)][static_cast<std::size_t>(m)];
                    const double cos_mp = std::cos(m * phi);
                    const double sin_mp = std::sin(m * phi);
                    const std::size_t idx = static_cast<std::size_t>(li * lm1 * n + m * n + i);
                    re_buf[idx] += Nlm_Plm * cos_mp;
                    im_buf[idx] += Nlm_Plm * sin_mp;
                }
            }
        }
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

            double qlm_sq = 0.0;
            // m = 0: only real part (sin(0)=0)
            {
                const std::size_t idx = static_cast<std::size_t>(li * lm1 * n + 0 * n + i);
                const double r = re_buf[idx] * inv_d;
                qlm_sq += r * r;
            }
            // m = 1..l: both real and imaginary, weight 2
            for (int m = 1; m <= l; ++m) {
                const std::size_t idx = static_cast<std::size_t>(li * lm1 * n + m * n + i);
                const double r = re_buf[idx] * inv_d;
                const double k = im_buf[idx] * inv_d;
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
        "Uses an explicit neighbour list (FlatCellList for N>=64) to avoid\n"
        "the O(N^2) dense matrix built by the Python/scipy path.";
    m.def(
        "steinhardt_per_atom", &steinhardt_per_atom_cpp,
        py::arg("pts"), py::arg("cutoff"), py::arg("l_values"),
        R"(
Compute per-atom Steinhardt Q_l for each l in l_values.

Parameters
----------
pts      : (n, 3) float64  – Cartesian coordinates (Angstrom)
cutoff   : float           – neighbour distance cutoff (Angstrom)
l_values : list[int]       – l indices (e.g. [4, 6, 8])

Returns
-------
dict  mapping "Q{l}" -> (n,) float64 array of per-atom Q_l values.
Atoms with no neighbours within cutoff are assigned Q_l = 0.
        )"
    );
}
