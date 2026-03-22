/**
 * pasted._ext._relax_core  (v0.2.11)
 * ====================================
 * L-BFGS minimisation of the harmonic steric-clash penalty energy:
 *
 *   E = sum_{i<j}  0.5 * max(0,  d_thr_ij - d_ij)^2
 *   d_thr_ij = cov_scale * (r_i + r_j)
 *
 * Converged = true when E < 1e-6 (all overlaps resolved within tolerance).
 *
 * Dependencies
 * ------------
 * C++17 standard library only.  No OpenMP.
 *
 * Architecture
 * ------------
 *   Vec                   — thin RAII wrapper over std::vector<double>
 *   FlatCellList          — O(N·k) pair enumeration via cell list
 *   PenaltyEvaluator      — computes E and analytical gradient in O(N·k)
 *   lbfgs_minimize()      — L-BFGS, history m=7, Armijo backtracking
 *
 * Memory management
 * -----------------
 * PenaltyEvaluator holds one persistent scratch member:
 *
 *   pairs_  (vector<pair<int,int>>)  — pair list; cleared (capacity kept)
 *                                      and rebuilt each evaluate() call.
 *
 * The per-thread tgrad_ scratch used in earlier versions has been removed:
 * libgomp was never linked in the distributed wheels, so the #else serial
 * branch was always the active path.  The pair loop now writes directly
 * into grad[], eliminating the per-evaluate() zero-fill and merge pass.
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
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------
static constexpr int    CELL_LIST_THRESHOLD     = 64;
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
        // Guard: coarsen grid until nx*ny*nz ≤ 1<<22 (4M cells).
        {
            static constexpr std::int64_t MAX_CELLS = 1LL << 22;
            auto tnx=[&]{return static_cast<int>((xmax-ox)*inv_cell)+2;};
            auto tny=[&]{return static_cast<int>((ymax-oy)*inv_cell)+2;};
            auto tnz=[&]{return static_cast<int>((zmax-oz)*inv_cell)+2;};
            while (static_cast<std::int64_t>(tnx())*tny()*tnz() > MAX_CELLS) {
                cell_size *= 2.0; inv_cell = 1.0 / cell_size;
                ox = xmin - cell_size; oy = ymin - cell_size; oz = zmin - cell_size;
            }
            nx = tnx(); ny = tny(); nz = tnz();
        }
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
//
// Verlet-list optimisation (v0.2.2):
// pairs_ is rebuilt only when any atom has moved more than skin/2 since the
// last rebuild.  Between rebuilds the same extended pair list is reused,
// eliminating the dominant serial FlatCellList traversal cost.
// Skin = 0.8 Å  =>  rebuild trigger at 0.4 Å displacement (< trust_radius).

// Verlet rebuild interval: rebuild pair list every N_VERLET_REBUILD evaluate() calls.
// Using a fixed interval avoids the O(N) displacement-check loop and is safer
// when trust_radius is large relative to the skin.  With N_VERLET_REBUILD=4 and
// trust_radius=0.5 Å the implicit skin is ~1 Å, safely above zero.
static constexpr int    N_VERLET_REBUILD = 4;  // rebuild every 4 evaluate() calls
// Adaptive skin: min(0.8 Å, cell_size × 0.3).
// Caps extended pair list at ≤ (1.3)^3 ≈ 2.2× the original count,
// preventing the 3-4× overhead that occurs for small-radius elements (C, O, H).
static constexpr double VERLET_SKIN_MAX  = 0.8;  // Å — absolute upper bound
static constexpr double VERLET_SKIN_FRAC = 0.3;  // fraction of cell_size

class PenaltyEvaluator {
    const double* radii_;
    double        cov_scale_;
    int           n_;
    double        cell_size_;       // cell list cell width (2 × max_r)
    double        cell_size_ext_;   // extended cell for Verlet list (+ skin)
    FlatCellList  cl_;
    // Persistent scratch — allocated once, reused every evaluate() call.
    std::vector<std::pair<int,int>>  pairs_;
    // Verlet tracking
    int  eval_count_    = 0;   // number of evaluate() calls since last rebuild
    bool needs_rebuild_ = true;

    void _rebuild(const double* xd) {
        pairs_.clear();
        cl_.build(xd, n_, cell_size_ext_);
        cl_.for_each_pair(xd, n_, [&](int i, int j){ pairs_.emplace_back(i, j); });
        eval_count_  = 0;
        needs_rebuild_ = false;
    }

public:
    PenaltyEvaluator(const double* radii, double cov_scale, int n)
        : radii_(radii), cov_scale_(cov_scale), n_(n)
    {
        double max_r = 0.0;
        for (int i = 0; i < n; ++i) max_r = std::max(max_r, radii[i]);
        cell_size_     = std::max(1e-6, cov_scale_ * 2.0 * max_r);
        const double skin = std::min(VERLET_SKIN_MAX, cell_size_ * VERLET_SKIN_FRAC);
        cell_size_ext_ = cell_size_ + skin;
        if (n_ >= CELL_LIST_THRESHOLD)
            pairs_.reserve(static_cast<std::size_t>(n_) * 8);
    }

    double evaluate(const Vec& x, Vec& grad) {
        grad.zero();
        const double* xd = x.data();
        double*       gd = grad.data();

        // ── Verlet rebuild check (counter-based) ──────────────────────────────
        if (n_ >= CELL_LIST_THRESHOLD) {
            if (needs_rebuild_ || eval_count_ >= N_VERLET_REBUILD) {
                _rebuild(xd);
            } else {
                ++eval_count_;
            }
        }

        double energy = 0.0;

        if (n_ < CELL_LIST_THRESHOLD) {
            // Small N: O(N²) serial loop.
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
            // Large N: iterate pre-built pair list, write directly to gd.
            const int npairs = static_cast<int>(pairs_.size());
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
    // This mirrors v0.1.10 GS behavior: RNG consumed only when coincident
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
// Bridson Poisson-disk sampling (v0.2.2)
// ===========================================================================
// Flat-array grid replaces unordered_map: O(1) cell lookup, cache-friendly.
// Grid stores first index of the occupant point (-1 = empty).
// Collision chains are stored in a separate next_[] array (linked list).

static void _bridson(
    int n, double min_dist,
    bool is_sphere, double R,
    double hx, double hy, double hz,
    int k, std::mt19937_64& rng,
    std::vector<std::array<double,3>>& pts_out)
{
    const double cell  = min_dist / std::sqrt(3.0);
    const double inv_c = 1.0 / cell;
    const double md2   = min_dist * min_dist;
    const double R2    = R * R;

    // Bounding box origin (all coords mapped to >= 0)
    const double ox = is_sphere ? -(R + cell) : -(hx + cell);
    const double oy = is_sphere ? -(R + cell) : -(hy + cell);
    const double oz = is_sphere ? -(R + cell) : -(hz + cell);
    const double wx = is_sphere ? 2*(R+cell) : 2*(hx+cell);
    const double wy = is_sphere ? 2*(R+cell) : 2*(hy+cell);
    const double wz = is_sphere ? 2*(R+cell) : 2*(hz+cell);

    const int gx = static_cast<int>(wx * inv_c) + 2;
    const int gy = static_cast<int>(wy * inv_c) + 2;
    const int gz = static_cast<int>(wz * inv_c) + 2;

    // Flat grid: head_[cell_idx] = first point index in that cell, -1 if empty
    // next_[pt_idx] = next point in same cell (linked list), -1 if none
    std::vector<int> head_(static_cast<std::size_t>(gx*gy*gz), -1);
    std::vector<int> next_;
    next_.reserve(static_cast<std::size_t>(n));

    pts_out.clear();
    pts_out.reserve(static_cast<std::size_t>(n));
    std::vector<int> active;
    active.reserve(static_cast<std::size_t>(n));

    auto ci = [&](double x, double y, double z) -> int {
        int ix = static_cast<int>((x - ox) * inv_c);
        int iy = static_cast<int>((y - oy) * inv_c);
        int iz = static_cast<int>((z - oz) * inv_c);
        return ix + gx*(iy + gy*iz);
    };

    auto in_region = [&](double x, double y, double z) -> bool {
        return is_sphere ? (x*x+y*y+z*z <= R2)
                         : (std::fabs(x)<=hx && std::fabs(y)<=hy && std::fabs(z)<=hz);
    };

    std::uniform_real_distribution<double> u(-1.0, 1.0);

    auto rand_in_region = [&]() -> std::array<double,3> {
        for (;;) {
            double px = u(rng)*(is_sphere?R:hx);
            double py = u(rng)*(is_sphere?R:hy);
            double pz = u(rng)*(is_sphere?R:hz);
            if (in_region(px,py,pz)) return {px,py,pz};
        }
    };

    auto try_add = [&](double px, double py, double pz) -> bool {
        int ix = static_cast<int>((px-ox)*inv_c);
        int iy = static_cast<int>((py-oy)*inv_c);
        int iz = static_cast<int>((pz-oz)*inv_c);
        // Check 5×5×5 neighborhood
        for (int dz=-2;dz<=2;++dz) for (int dy=-2;dy<=2;++dy) for (int dx=-2;dx<=2;++dx) {
            int nx2=ix+dx, ny2=iy+dy, nz2=iz+dz;
            if (nx2<0||ny2<0||nz2<0||nx2>=gx||ny2>=gy||nz2>=gz) continue;
            int cell_id = nx2 + gx*(ny2 + gy*nz2);
            for (int j = head_[static_cast<std::size_t>(cell_id)]; j>=0; j=next_[static_cast<std::size_t>(j)]) {
                const auto& q = pts_out[static_cast<std::size_t>(j)];
                double ddx=px-q[0],ddy=py-q[1],ddz=pz-q[2];
                if (ddx*ddx+ddy*ddy+ddz*ddz < md2) return false;
            }
        }
        int idx = static_cast<int>(pts_out.size());
        int cell_id = ci(px,py,pz);
        next_.push_back(head_[static_cast<std::size_t>(cell_id)]);
        head_[static_cast<std::size_t>(cell_id)] = idx;
        pts_out.push_back({px,py,pz});
        active.push_back(idx);
        return true;
    };

    auto seed = rand_in_region();
    try_add(seed[0], seed[1], seed[2]);

    while (!active.empty() && static_cast<int>(pts_out.size()) < n) {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(active.size())-1);
        int ai = pick(rng);
        int base_idx = active[static_cast<std::size_t>(ai)];
        const auto& base = pts_out[static_cast<std::size_t>(base_idx)];
        bool placed = false;
        for (int attempt = 0; attempt < k; ++attempt) {
            double dx,dy,dz,d2;
            do {
                dx=u(rng)*2*min_dist; dy=u(rng)*2*min_dist; dz=u(rng)*2*min_dist;
                d2=dx*dx+dy*dy+dz*dz;
            } while (d2 < md2 || d2 > 4.0*md2);
            double cx=base[0]+dx, cy=base[1]+dy, cz=base[2]+dz;
            if (!in_region(cx,cy,cz)) continue;
            if (try_add(cx,cy,cz)) { placed=true; break; }
        }
        if (!placed) active.erase(active.begin()+ai);
    }
}

// Public: Poisson-disk sampling for a sphere.
// Returns (n, 3) float64 array.  Falls back to uniform random for any
// points that could not be placed with the minimum-distance guarantee.
static F64Array poisson_disk_sphere_cpp(
    int n, double radius, double min_dist, long long seed, int k)
{
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed < 0 ? 42 : seed));

    std::vector<std::array<double,3>> pts;
    _bridson(n, min_dist, true, radius, 0,0,0, k, rng, pts);
    int placed = static_cast<int>(pts.size());

    // Uniform random fallback for remaining slots
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const double R2 = radius * radius;
    while (placed < n) {
        double px, py, pz;
        do { px=u(rng)*radius; py=u(rng)*radius; pz=u(rng)*radius; }
        while (px*px+py*py+pz*pz > R2);
        pts.push_back({px, py, pz});
        ++placed;
    }

    F64Array out({static_cast<py::ssize_t>(n),
                  static_cast<py::ssize_t>(3)});
    double* op = static_cast<double*>(out.request().ptr);
    for (int i = 0; i < n; ++i) {
        op[3*i  ] = pts[static_cast<std::size_t>(i)][0];
        op[3*i+1] = pts[static_cast<std::size_t>(i)][1];
        op[3*i+2] = pts[static_cast<std::size_t>(i)][2];
    }
    return out;
}

// Public: Poisson-disk sampling for a box.
static F64Array poisson_disk_box_cpp(
    int n, double lx, double ly, double lz,
    double min_dist, long long seed, int k)
{
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed < 0 ? 42 : seed));

    std::vector<std::array<double,3>> pts;
    double hx=lx/2, hy=ly/2, hz=lz/2;
    _bridson(n, min_dist, false, 0, hx, hy, hz, k, rng, pts);
    int placed = static_cast<int>(pts.size());

    std::uniform_real_distribution<double> ux(-hx, hx), uy(-hy, hy), uz(-hz, hz);
    while (placed < n) {
        pts.push_back({ux(rng), uy(rng), uz(rng)});
        ++placed;
    }

    F64Array out({static_cast<py::ssize_t>(n),
                  static_cast<py::ssize_t>(3)});
    double* op = static_cast<double*>(out.request().ptr);
    for (int i = 0; i < n; ++i) {
        op[3*i  ] = pts[static_cast<std::size_t>(i)][0];
        op[3*i+1] = pts[static_cast<std::size_t>(i)][1];
        op[3*i+2] = pts[static_cast<std::size_t>(i)][2];
    }
    return out;
}


// ===========================================================================

PYBIND11_MODULE(_relax_core, m) {
    m.doc() =
        "pasted._ext._relax_core (v0.2.2)\n"
        "L-BFGS steric-clash relaxation + Bridson Poisson-disk placement.\n"
        "Verlet list reuse (skin=0.8 A) reduces pair-list rebuild cost.";

    m.def(
        "poisson_disk_sphere",
        &poisson_disk_sphere_cpp,
        py::arg("n"), py::arg("radius"), py::arg("min_dist"),
        py::arg("seed") = -1LL, py::arg("k") = 30,
        "Bridson Poisson-disk sampling inside a sphere.\n"
        "Returns (n, 3) float64 array with min_dist separation guarantee.\n"
        "Falls back to uniform random for slots that cannot be placed.\n"
        "seed: RNG seed (-1 = use 42); k: candidates per active point (30)."
    );

    m.def(
        "poisson_disk_box",
        &poisson_disk_box_cpp,
        py::arg("n"), py::arg("lx"), py::arg("ly"), py::arg("lz"),
        py::arg("min_dist"), py::arg("seed") = -1LL, py::arg("k") = 30,
        "Bridson Poisson-disk sampling inside an axis-aligned box.\n"
        "Returns (n, 3) float64 array with min_dist separation guarantee."
    );

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
