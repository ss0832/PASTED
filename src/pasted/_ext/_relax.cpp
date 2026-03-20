/**
 * pasted._ext._relax_core  (v0.2.1)
 * ===================================
 * L-BFGS minimisation of the harmonic steric-clash penalty energy:
 *
 *   E = sum_{i<j}  0.5 * max(0,  d_thr_ij - d_ij)^2
 *   d_thr_ij = cov_scale * (r_i + r_j)
 *
 * Converged = true when E < 1e-6 (all overlaps resolved within tolerance).
 *
 * Dependencies
 * ------------
 * C++17 standard library only.  OpenMP optional (Linux, -fopenmp).
 *
 * Architecture
 * ------------
 *   Vec                   — thin RAII wrapper over std::vector<double>
 *   FlatCellList          — O(N·k) pair enumeration via cell list
 *   PenaltyEvaluator      — computes E and analytical gradient in O(N·k)
 *   lbfgs_minimize()      — L-BFGS, history m=7, Armijo backtracking
 *
 * Memory management (v0.2.1)
 * --------------------------
 * PenaltyEvaluator holds two persistent scratch members:
 *
 *   pairs_  (vector<pair<int,int>>)       — pair list; cleared (capacity kept)
 *                                           and rebuilt each evaluate() call.
 *   tgrad_  (vector<vector<double>>)      — per-thread gradient buffers
 *                                           (nthreads × 3N × 8 B); zeroed with
 *                                           std::fill on each call, never freed.
 *
 * This eliminates the multi-GB malloc/free churn that occurred at large N
 * (e.g. ~8 GB / structure at n=150 000, 8 threads, 300 L-BFGS iterations).
 *
 * OpenMP / A+B guard (v0.2.0+)
 * -----------------------------
 * The parallel pair loop fires only when BOTH:
 *   A) omp_get_max_threads() > 2   (avoid overhead on ≤2-core machines), AND
 *   B) pairs_.size() >= PAIR_PARALLEL_THRESHOLD (50 000)  (enough work to
 *      amortise thread-spawn and tgrad-merge cost).
 * On ≤2 threads the serial path writes directly into grad[], bypassing tgrad_
 * allocation overhead entirely.
 *
 * Python API
 * ----------
 *   relax_positions(pts, radii, cov_scale, max_cycles, seed=-1)
 *       -> (pts_out : ndarray(n,3), converged : bool)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace py = pybind11;
using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------
static constexpr int    CELL_LIST_THRESHOLD     = 64;
static constexpr int    PAIR_PARALLEL_THRESHOLD = 50000; // A+B guard: min pairs to use OpenMP
static constexpr int    LBFGS_M                 = 7;
static constexpr double ENERGY_TOL          = 1e-12; // per-pair residual < sqrt(2e-12) ~ 1.4e-6 Ang
static constexpr double ARMIJO_C1           = 1e-4;
static constexpr int    MAX_LS_STEPS        = 50;

// ===========================================================================
// Vec — minimal dense vector backed by std::vector<double>
// ===========================================================================
// Provides only the arithmetic operations used by L-BFGS.
// All member functions are inlined; -O3 auto-vectorizes the loops.

struct Vec {
    std::vector<double> d;

    Vec() = default;
    explicit Vec(int n, double v = 0.0)
        : d(static_cast<std::size_t>(n), v) {}

    int    size()  const noexcept { return static_cast<int>(d.size()); }
    double&       operator[](int i)       noexcept { return d[i]; }
    double        operator[](int i) const noexcept { return d[i]; }
    double*       data()       noexcept { return d.data(); }
    const double* data() const noexcept { return d.data(); }

    void zero() noexcept { std::fill(d.begin(), d.end(), 0.0); }

    double dot(const Vec& o) const noexcept {
        double s = 0.0;
        const int n = size();
        for (int i = 0; i < n; ++i) s += d[i] * o.d[i];
        return s;
    }
    double norm2() const noexcept { return dot(*this); }
    double norm()  const noexcept { return std::sqrt(norm2()); }

    // *this += alpha * x
    void add_scaled(double alpha, const Vec& x) noexcept {
        const int n = size();
        for (int i = 0; i < n; ++i) d[i] += alpha * x.d[i];
    }

    // *this = a + alpha * b
    void assign_sum(const Vec& a, double alpha, const Vec& b) noexcept {
        const int n = size();
        for (int i = 0; i < n; ++i) d[i] = a.d[i] + alpha * b.d[i];
    }

    // *this = s * x
    void assign_scaled(double s, const Vec& x) noexcept {
        const int n = size();
        for (int i = 0; i < n; ++i) d[i] = s * x.d[i];
    }

    // *this = x - y
    void assign_diff(const Vec& x, const Vec& y) noexcept {
        const int n = size();
        for (int i = 0; i < n; ++i) d[i] = x.d[i] - y.d[i];
    }

    void copy_from(const Vec& o)           { d = o.d; }
    void copy_from(const double* src, int n) {
        d.assign(src, src + static_cast<std::size_t>(n));
    }
    void copy_to(double* dst) const noexcept {
        std::copy(d.begin(), d.end(), dst);
    }
};

// ===========================================================================
// FlatCellList  (identical to v0.1.10)
// ===========================================================================

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
            xmin = std::min(xmin, pts[i*3  ]); xmax = std::max(xmax, pts[i*3  ]);
            ymin = std::min(ymin, pts[i*3+1]); ymax = std::max(ymax, pts[i*3+1]);
            zmin = std::min(zmin, pts[i*3+2]); zmax = std::max(zmax, pts[i*3+2]);
        }
        ox = xmin - cell_size;
        oy = ymin - cell_size;
        oz = zmin - cell_size;
        nx = static_cast<int>((xmax - ox) * inv_cell) + 2;
        ny = static_cast<int>((ymax - oy) * inv_cell) + 2;
        nz = static_cast<int>((zmax - oz) * inv_cell) + 2;
        const int total = nx * ny * nz;
        cell_head.assign(static_cast<std::size_t>(total), -1);
        next.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const int cx  = static_cast<int>((pts[i*3  ] - ox) * inv_cell);
            const int cy  = static_cast<int>((pts[i*3+1] - oy) * inv_cell);
            const int cz  = static_cast<int>((pts[i*3+2] - oz) * inv_cell);
            const int cid = cx + nx * (cy + ny * cz);
            next[i]        = cell_head[cid];
            cell_head[cid] = i;
        }
    }

    template<typename F>
    void for_each_pair(const double* /*pts*/, int /*n*/, F process) const {
        for (int cz = 0; cz < nz; ++cz)
        for (int cy = 0; cy < ny; ++cy)
        for (int cx = 0; cx < nx; ++cx) {
            const int cid = cx + nx * (cy + ny * cz);
            for (int i = cell_head[cid]; i >= 0; i = next[i]) {
                for (int j = next[i]; j >= 0; j = next[j])
                    process(i, j);
                for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    const int ncx = cx+dx, ncy = cy+dy, ncz = cz+dz;
                    if (ncx<0||ncy<0||ncz<0||ncx>=nx||ncy>=ny||ncz>=nz) continue;
                    const int nid = ncx + nx*(ncy + ny*ncz);
                    if (nid <= cid) continue;
                    for (int k = cell_head[nid]; k >= 0; k = next[k])
                        process(i, k);
                }
            }
        }
    }
};

// ===========================================================================
// PenaltyEvaluator
// ===========================================================================
// E      = sum_{i<j}  0.5 * max(0, thr_ij - d_ij)^2
// dE/dr_i = sum_{j}  -(thr_ij - d_ij)/d_ij * (r_i - r_j)   [when d_ij < thr]

class PenaltyEvaluator {
    const double* radii_;
    double        cov_scale_;
    int           n_;
    double        cell_size_;
    FlatCellList  cl_;
    // Persistent scratch buffers — allocated once, reused every evaluate() call.
    // Eliminates ~27 MB malloc/free churn per call at n=150000 with 8 threads.
    std::vector<std::pair<int,int>>  pairs_;
    std::vector<std::vector<double>> tgrad_;

public:
    PenaltyEvaluator(const double* radii, double cov_scale, int n)
        : radii_(radii), cov_scale_(cov_scale), n_(n)
    {
        double max_r = 0.0;
        for (int i = 0; i < n; ++i) max_r = std::max(max_r, radii[i]);
        cell_size_ = std::max(1e-6, cov_scale_ * 2.0 * max_r);
#ifdef _OPENMP
        const int nthreads = omp_get_max_threads();
#else
        const int nthreads = 1;
#endif
        tgrad_.assign(static_cast<std::size_t>(nthreads),
                      std::vector<double>(static_cast<std::size_t>(n_ * 3), 0.0));
        if (n_ >= CELL_LIST_THRESHOLD)
            pairs_.reserve(static_cast<std::size_t>(n_) * 6);
    }

    double evaluate(const Vec& x, Vec& grad) {
        grad.zero();
        const double* xd = x.data();
        double*       gd = grad.data();

        // ── Build pair list — reuse pairs_ capacity, only clear contents ──────
        if (n_ >= CELL_LIST_THRESHOLD) {
            pairs_.clear();
            cl_.build(xd, n_, cell_size_);
            cl_.for_each_pair(xd, n_, [&](int i, int j){ pairs_.emplace_back(i, j); });
        }

        // ── Parallel reduction over pairs ──────────────────────────────────────
        // tgrad_ is pre-allocated — just zero it.
#ifdef _OPENMP
        const int nthreads = omp_get_max_threads();
#else
        const int nthreads = 1;
#endif
        const int dim3 = n_ * 3;
        // Adapt if set_num_threads() changed the thread count since construction.
        if (static_cast<int>(tgrad_.size()) != nthreads)
            tgrad_.assign(static_cast<std::size_t>(nthreads),
                          std::vector<double>(static_cast<std::size_t>(dim3), 0.0));
        for (auto& tg : tgrad_) std::fill(tg.begin(), tg.end(), 0.0);
        auto& tgrad = tgrad_;
        double energy = 0.0;

        if (n_ < CELL_LIST_THRESHOLD) {
            // Small N: O(N²) serial loop — overhead of thread spawning > gain.
            for (int i = 0; i < n_-1; ++i) {
                for (int j = i+1; j < n_; ++j) {
                    const double dx  = xd[3*i  ] - xd[3*j  ];
                    const double dy  = xd[3*i+1] - xd[3*j+1];
                    const double dz  = xd[3*i+2] - xd[3*j+2];
                    const double d2  = dx*dx + dy*dy + dz*dz;
                    const double thr = cov_scale_ * (radii_[i] + radii_[j]);
                    if (d2 >= thr * thr) continue;
                    const double d       = std::sqrt(d2);
                    const double overlap = thr - d;
                    energy += 0.5 * overlap * overlap;
                    if (d > 1e-10) {
                        const double gf = -overlap / d;
                        gd[3*i  ] += gf*dx;  gd[3*i+1] += gf*dy;  gd[3*i+2] += gf*dz;
                        gd[3*j  ] -= gf*dx;  gd[3*j+1] -= gf*dy;  gd[3*j+2] -= gf*dz;
                    }
                }
            }
        } else {
            const int npairs = static_cast<int>(pairs_.size());
            // A+B guard: only use OpenMP when enough threads AND enough pairs
            // exist to amortise thread-spawn / merge overhead.
#ifdef _OPENMP
            if (nthreads > 2 && npairs >= PAIR_PARALLEL_THRESHOLD) {
#pragma omp parallel for schedule(static) reduction(+:energy)
                for (int p = 0; p < npairs; ++p) {
                    const int tid = omp_get_thread_num();
                    const int i = pairs_[static_cast<std::size_t>(p)].first;
                    const int j = pairs_[static_cast<std::size_t>(p)].second;
                    const double dx  = xd[3*i  ] - xd[3*j  ];
                    const double dy  = xd[3*i+1] - xd[3*j+1];
                    const double dz  = xd[3*i+2] - xd[3*j+2];
                    const double d2  = dx*dx + dy*dy + dz*dz;
                    const double thr = cov_scale_ * (radii_[i] + radii_[j]);
                    if (d2 >= thr * thr) continue;
                    const double d       = std::sqrt(d2);
                    const double overlap = thr - d;
                    energy += 0.5 * overlap * overlap;
                    if (d > 1e-10) {
                        const double gf = -overlap / d;
                        const double gx = gf * dx, gy = gf * dy, gz = gf * dz;
                        auto& tg = tgrad[static_cast<std::size_t>(tid)];
                        tg[3*i  ] += gx;  tg[3*i+1] += gy;  tg[3*i+2] += gz;
                        tg[3*j  ] -= gx;  tg[3*j+1] -= gy;  tg[3*j+2] -= gz;
                    }
                }
                // Merge thread-local gradients into gd
                for (int t = 0; t < nthreads; ++t)
                    for (int k = 0; k < dim3; ++k)
                        gd[k] += tgrad[static_cast<std::size_t>(t)][k];
            } else {
                // Serial fallback: write directly to gd — no tgrad overhead.
                for (int p = 0; p < npairs; ++p) {
                    const int i = pairs_[static_cast<std::size_t>(p)].first;
                    const int j = pairs_[static_cast<std::size_t>(p)].second;
                    const double dx  = xd[3*i  ] - xd[3*j  ];
                    const double dy  = xd[3*i+1] - xd[3*j+1];
                    const double dz  = xd[3*i+2] - xd[3*j+2];
                    const double d2  = dx*dx + dy*dy + dz*dz;
                    const double thr = cov_scale_ * (radii_[i] + radii_[j]);
                    if (d2 >= thr * thr) continue;
                    const double d       = std::sqrt(d2);
                    const double overlap = thr - d;
                    energy += 0.5 * overlap * overlap;
                    if (d > 1e-10) {
                        const double gf = -overlap / d;
                        const double gx = gf * dx, gy = gf * dy, gz = gf * dz;
                        gd[3*i  ] += gx;  gd[3*i+1] += gy;  gd[3*i+2] += gz;
                        gd[3*j  ] -= gx;  gd[3*j+1] -= gy;  gd[3*j+2] -= gz;
                    }
                }
            }
#else
            for (int p = 0; p < npairs; ++p) {
                const int tid = 0;
                const int i = pairs_[static_cast<std::size_t>(p)].first;
                const int j = pairs_[static_cast<std::size_t>(p)].second;
                const double dx  = xd[3*i  ] - xd[3*j  ];
                const double dy  = xd[3*i+1] - xd[3*j+1];
                const double dz  = xd[3*i+2] - xd[3*j+2];
                const double d2  = dx*dx + dy*dy + dz*dz;
                const double thr = cov_scale_ * (radii_[i] + radii_[j]);
                if (d2 >= thr * thr) continue;
                const double d       = std::sqrt(d2);
                const double overlap = thr - d;
                energy += 0.5 * overlap * overlap;
                if (d > 1e-10) {
                    const double gf = -overlap / d;
                    const double gx = gf * dx, gy = gf * dy, gz = gf * dz;
                    auto& tg = tgrad[static_cast<std::size_t>(tid)];
                    tg[3*i  ] += gx;  tg[3*i+1] += gy;  tg[3*i+2] += gz;
                    tg[3*j  ] -= gx;  tg[3*j+1] -= gy;  tg[3*j+2] -= gz;
                }
            }
            // Merge thread-local gradients into gd (nthreads=1)
            for (int t = 0; t < nthreads; ++t)
                for (int k = 0; k < dim3; ++k)
                    gd[k] += tgrad[static_cast<std::size_t>(t)][k];
#endif
        }
        return energy;
    }
};

// ===========================================================================
// lbfgs_minimize
// ===========================================================================
// Minimises f(x) using L-BFGS with Armijo backtracking line search.
//
// Algorithm details
// -----------------
//   History depth : m = LBFGS_M (circular buffer; buf_ptr = next write slot).
//   H0 scaling    : Barzilai-Borwein  gamma = s^T y / y^T y
//   First iter    : H0 = (1 / ||g||) * I  (unit-gradient step)
//   Descent check : if d^T g >= -tol, reset to -g and flush history
//   Line search   : Armijo sufficient decrease (c1 = ARMIJO_C1), step halved
//   History update: skipped when s^T y <= 1e-10 * ||s||^2 (curvature check)

static std::pair<double, bool> lbfgs_minimize(
    Vec& x,
    std::function<double(const Vec&, Vec&)> eval,
    int max_iter)
{
    const int dim = x.size();
    const int m   = LBFGS_M;

    std::vector<Vec>    s_buf(m, Vec(dim));
    std::vector<Vec>    y_buf(m, Vec(dim));
    std::vector<double> rho_buf(m, 0.0);
    int buf_ptr   = 0;
    int buf_count = 0;

    std::vector<double> alpha_arr(m, 0.0);
    Vec g(dim), g_new(dim), q(dim), r(dim), d(dim), x_trial(dim);

    double E = eval(x, g);
    if (E <= ENERGY_TOL) return {E, true};

    // slot(i): index of the i-th newest history entry
    auto slot = [&](int i) -> int {
        return (buf_ptr + m - 1 - i) % m;
    };

    for (int iter = 0; iter < max_iter; ++iter) {

        // ── Two-loop L-BFGS direction ─────────────────────────────────────
        q.copy_from(g);

        for (int i = 0; i < buf_count; ++i) {
            const int k  = slot(i);
            alpha_arr[i] = rho_buf[k] * s_buf[k].dot(q);
            q.add_scaled(-alpha_arr[i], y_buf[k]);
        }

        if (buf_count > 0) {
            const int    k     = slot(0);
            const double sy    = s_buf[k].dot(y_buf[k]);
            const double yy    = y_buf[k].norm2();
            const double gamma = (yy > 1e-20) ? sy / yy : 1.0;
            r.assign_scaled(gamma, q);
        } else {
            const double gnorm = g.norm();
            if (gnorm > 1e-20) r.assign_scaled(1.0 / gnorm, q);
            else               r.copy_from(q);
        }

        for (int i = buf_count - 1; i >= 0; --i) {
            const int    k    = slot(i);
            const double beta = rho_buf[k] * y_buf[k].dot(r);
            r.add_scaled(alpha_arr[i] - beta, s_buf[k]);
        }

        d.assign_scaled(-1.0, r);

        // Descent direction safeguard
        const double dg0_raw  = d.dot(g);
        const double desc_tol = 1e-14 * d.norm() * g.norm();
        double dg0;
        if (dg0_raw >= -desc_tol) {
            d.assign_scaled(-1.0, g);
            dg0       = -g.norm2();
            buf_count = 0;
        } else {
            dg0 = dg0_raw;
        }

        // ── Armijo backtracking line search ───────────────────────────────
        double alpha = 1.0;
        double E_new = E;
        bool   ls_ok = false;

        for (int ls = 0; ls < MAX_LS_STEPS; ++ls) {
            x_trial.assign_sum(x, alpha, d);
            E_new = eval(x_trial, g_new);
            if (E_new <= E + ARMIJO_C1 * alpha * dg0) { ls_ok = true; break; }
            alpha *= 0.5;
            if (alpha < 1e-15) break;
        }

        if (!ls_ok) {
            // Emergency fallback: minimal steepest-descent step
            alpha = 1e-8 / std::max(1.0, g.norm());
            x_trial.assign_sum(x, -alpha, g);
            E_new = eval(x_trial, g_new);
            buf_count = 0;
        }

        // ── History update ────────────────────────────────────────────────
        {
            Vec s_new(dim), y_new(dim);
            s_new.assign_diff(x_trial, x);
            y_new.assign_diff(g_new, g);
            const double sy = s_new.dot(y_new);
            const double ss = s_new.norm2();
            if (sy > 1e-10 * ss) {
                s_buf[buf_ptr].copy_from(s_new);
                y_buf[buf_ptr].copy_from(y_new);
                rho_buf[buf_ptr] = 1.0 / sy;
                buf_ptr   = (buf_ptr + 1) % m;
                buf_count = std::min(buf_count + 1, m);
            }
        }

        x.copy_from(x_trial);
        g.copy_from(g_new);
        E = E_new;

        if (E <= ENERGY_TOL) return {E, true};
    }

    return {E, E <= ENERGY_TOL};
}

// ===========================================================================
// Public entry point  (Pybind11 signature identical to v0.1.10)
// ===========================================================================

std::tuple<F64Array, bool> relax_positions_cpp(
    F64Array pts_in, F64Array radii_in,
    double cov_scale, int max_cycles, long long seed)
{
    auto pts_buf = pts_in.request();
    auto rad_buf = radii_in.request();
    const int n   = static_cast<int>(pts_buf.shape[0]);

    F64Array pts_out({static_cast<py::ssize_t>(n),
                      static_cast<py::ssize_t>(3)});
    double*       out_ptr = static_cast<double*>(pts_out.request().ptr);
    const double* src_ptr = static_cast<const double*>(pts_buf.ptr);
    std::copy(src_ptr, src_ptr + n * 3, out_ptr);
    const double* radii   = static_cast<const double*>(rad_buf.ptr);

    if (n < 2) return {pts_out, true};

    // RNG: one-time jitter only
    std::mt19937_64 rng;
    if (seed < 0) { std::random_device rd; rng.seed(rd()); }
    else          { rng.seed(static_cast<std::uint64_t>(seed)); }
    std::normal_distribution<double> ndist(0.0, 1.0);

    // Pack positions into Vec x (3N DOF)
    Vec x(3 * n);
    std::copy(out_ptr, out_ptr + 3 * n, x.data());

    // Build evaluator (computes cell_size from radii; reused throughout)
    PenaltyEvaluator evaluator(radii, cov_scale, n);

    // Fast early-exit: if no overlaps exist, return without touching positions.
    // This also avoids applying jitter to already-valid structures.
    {
        Vec grad_tmp(3 * n);
        if (evaluator.evaluate(x, grad_tmp) <= ENERGY_TOL) {
            return {pts_out, true};
        }
    }

    // Coincident-atom jitter: only perturb atoms in pairs with d < 1e-10.
    // This mirrors v0.1.10 GS behaviour: RNG consumed only when coincident
    // atoms exist, so seed=None yields deterministic results for normal
    // structures.  sigma ~ 1e-6 * max_r (~3e-8 Ang for H).
    {
        const double max_r      = *std::max_element(radii, radii + n);
        const double jitter_sig = 1e-6 * std::max(max_r, 1e-3);
        const double* xd = x.data();
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const double ddx = xd[3*i  ] - xd[3*j  ];
                const double ddy = xd[3*i+1] - xd[3*j+1];
                const double ddz = xd[3*i+2] - xd[3*j+2];
                if (ddx*ddx + ddy*ddy + ddz*ddz < 1e-20) {
                    x[3*i  ] += jitter_sig * ndist(rng);
                    x[3*i+1] += jitter_sig * ndist(rng);
                    x[3*i+2] += jitter_sig * ndist(rng);
                    x[3*j  ] += jitter_sig * ndist(rng);
                    x[3*j+1] += jitter_sig * ndist(rng);
                    x[3*j+2] += jitter_sig * ndist(rng);
                }
            }
        }
    }
    const auto [final_energy, converged] = lbfgs_minimize(
        x,
        [&](const Vec& pos, Vec& grad) {
            return evaluator.evaluate(pos, grad);
        },
        max_cycles);

    x.copy_to(out_ptr);
    return {pts_out, converged};
}

// ===========================================================================

PYBIND11_MODULE(_relax_core, m) {
    m.doc() =
        "pasted._ext._relax_core (v0.1.11 candidate)\n"
        "L-BFGS minimisation of harmonic steric-clash penalty.\n"
        "E = sum_{i<j} 0.5 * max(0, cov_scale*(r_i+r_j) - d_ij)^2\n"
        "No external dependencies beyond C++17 stdlib + pybind11.\n"
        "setup.py requires no changes from v0.1.10.\n"
        "O(N) FlatCellList for N >= 64; O(N^2) for N < 64.";

    m.def(
        "relax_positions", &relax_positions_cpp,
        py::arg("pts"), py::arg("radii"), py::arg("cov_scale"),
        py::arg("max_cycles"), py::arg("seed") = -1LL,
        R"(
Resolve steric clashes via L-BFGS minimisation of the harmonic penalty:
  E = sum_{i<j} 0.5 * max(0, cov_scale*(r_i+r_j) - d_ij)^2

converged = True when E < 1e-6.
relax_cycles=1500 (Python default) is unchanged and backward-compatible;
L-BFGS exits early as soon as E drops below the tolerance.

Parameters
----------
pts        : (n, 3) float64  -- atom positions in Angstrom (C-contiguous, copied)
radii      : (n,)   float64  -- covalent radii in Angstrom
cov_scale  : float           -- minimum-distance scale factor
max_cycles : int             -- maximum L-BFGS outer iterations
seed       : int, optional   -- RNG seed for coincident-atom jitter; -1 = random

Returns
-------
(pts_out, converged) : ((n, 3) ndarray, bool)
        )"
    );
}
