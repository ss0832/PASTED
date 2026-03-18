/**
 * pasted._ext._relax_core
 * =======================
 * C++17 implementation of the repulsion-relaxation inner loop.
 *
 * Exported function
 * -----------------
 *   relax_positions(pts, radii, cov_scale, max_cycles, seed=-1)
 *       -> (pts_out: ndarray(n,3), converged: bool)
 *
 * Spatial partitioning (Flat Cell List)
 * --------------------------------------
 * For N >= CELL_LIST_THRESHOLD (64) a flat 3-D grid is built each cycle.
 * Cell width = cov_scale x 2 x max(radii), the maximum possible threshold.
 *
 * Implementation uses a flat vector<int> (cell_heads + next-atom linked list)
 * rather than unordered_map to avoid per-cycle hash-table allocation overhead.
 *
 * Complexity
 * ----------
 *   N <  64 : O(N^2)  full-pair loop
 *   N >= 64 : O(N)    amortised per cycle (uniform density)
 *
 * Seed convention: seed >= 0 deterministic, -1 std::random_device.
 * Update strategy: Gauss-Seidel (positions updated within each cycle).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <tuple>
#include <vector>

namespace py = pybind11;
using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

static constexpr int CELL_LIST_THRESHOLD = 64;

// ---------------------------------------------------------------------------
// Push one violating pair apart (distance comparison, not d^2)
// ---------------------------------------------------------------------------

static inline bool check_and_push(
    double* pts, int i, int j, double thr,
    std::mt19937_64& rng,
    std::normal_distribution<double>& ndist)
{
    const double dx = pts[i*3+0] - pts[j*3+0];
    const double dy = pts[i*3+1] - pts[j*3+1];
    const double dz = pts[i*3+2] - pts[j*3+2];
    const double d  = std::sqrt(dx*dx + dy*dy + dz*dz);

    if (d >= thr) return false;

    double vx, vy, vz;
    if (d < 1e-10) {
        do {
            vx = ndist(rng); vy = ndist(rng); vz = ndist(rng);
        } while (vx*vx + vy*vy + vz*vz < 1e-20);
        const double inv = 1.0 / std::sqrt(vx*vx + vy*vy + vz*vz);
        vx *= inv; vy *= inv; vz *= inv;
    } else {
        const double inv = 1.0 / d;
        vx = dx*inv; vy = dy*inv; vz = dz*inv;
    }

    const double push = (thr - d) * 0.5;
    pts[i*3+0] += push*vx; pts[i*3+1] += push*vy; pts[i*3+2] += push*vz;
    pts[j*3+0] -= push*vx; pts[j*3+1] -= push*vy; pts[j*3+2] -= push*vz;
    return true;
}

// ---------------------------------------------------------------------------
// O(N^2) full-pair loop
// ---------------------------------------------------------------------------

static bool relax_full_pair(
    double* pts, const double* radii, double cov_scale,
    int n, int max_cycles,
    std::mt19937_64& rng, std::normal_distribution<double>& ndist)
{
    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        bool any = false;
        for (int i = 0; i < n-1; ++i)
            for (int j = i+1; j < n; ++j)
                if (check_and_push(pts, i, j,
                        cov_scale * (radii[i] + radii[j]), rng, ndist))
                    any = true;
        if (!any) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Flat Cell List (vector-based, no heap allocation per cycle beyond resize)
// ---------------------------------------------------------------------------

struct FlatCellList {
    double   inv_cell;
    int      nx, ny, nz;       // grid dimensions
    double   ox, oy, oz;       // origin offset
    // Linked-list style: cell_head[cell_id] = first atom, next[i] = next atom
    // -1 = end of list.
    std::vector<int> cell_head;
    std::vector<int> next;

    void build(const double* pts, int n, double cell_size) {
        inv_cell = 1.0 / cell_size;

        // Bounding box
        double xmin=pts[0], xmax=pts[0];
        double ymin=pts[1], ymax=pts[1];
        double zmin=pts[2], zmax=pts[2];
        for (int i = 1; i < n; ++i) {
            xmin=std::min(xmin,pts[i*3+0]); xmax=std::max(xmax,pts[i*3+0]);
            ymin=std::min(ymin,pts[i*3+1]); ymax=std::max(ymax,pts[i*3+1]);
            zmin=std::min(zmin,pts[i*3+2]); zmax=std::max(zmax,pts[i*3+2]);
        }
        // Pad by 1 cell to avoid edge effects
        ox = xmin - cell_size; oy = ymin - cell_size; oz = zmin - cell_size;
        nx = static_cast<int>((xmax - ox) * inv_cell) + 2;
        ny = static_cast<int>((ymax - oy) * inv_cell) + 2;
        nz = static_cast<int>((zmax - oz) * inv_cell) + 2;

        const int total = nx * ny * nz;
        cell_head.assign(static_cast<std::size_t>(total), -1);
        next.resize(static_cast<std::size_t>(n));

        for (int i = 0; i < n; ++i) {
            const int cx = static_cast<int>((pts[i*3+0]-ox)*inv_cell);
            const int cy = static_cast<int>((pts[i*3+1]-oy)*inv_cell);
            const int cz = static_cast<int>((pts[i*3+2]-oz)*inv_cell);
            const int cid = cx + nx*(cy + ny*cz);
            next[static_cast<std::size_t>(i)] = cell_head[static_cast<std::size_t>(cid)];
            cell_head[static_cast<std::size_t>(cid)] = i;
        }
    }

    // Iterate over all unique (i<j) pairs within the 27-cell neighbourhood
    // and call process(i,j) for each.
    template<typename F>
    void for_each_pair(const double* pts, int n, F process) const {
        (void)n;
        for (int cz = 0; cz < nz; ++cz)
        for (int cy = 0; cy < ny; ++cy)
        for (int cx = 0; cx < nx; ++cx) {
            const int cid = cx + nx*(cy + ny*cz);
            int i = cell_head[static_cast<std::size_t>(cid)];
            while (i >= 0) {
                // Same-cell pairs (upper triangle only)
                int j = next[static_cast<std::size_t>(i)];
                while (j >= 0) {
                    process(i, j);
                    j = next[static_cast<std::size_t>(j)];
                }
                // Cross-cell neighbours with index > current cell
                for (int dz=-1; dz<=1; ++dz)
                for (int dy=-1; dy<=1; ++dy)
                for (int dx=-1; dx<=1; ++dx) {
                    if (dx==0 && dy==0 && dz==0) continue;
                    const int ncx=cx+dx, ncy=cy+dy, ncz=cz+dz;
                    if (ncx<0||ncy<0||ncz<0||ncx>=nx||ncy>=ny||ncz>=nz) continue;
                    const int nid = ncx + nx*(ncy + ny*ncz);
                    // Only process if neighbour cell index > current cell
                    if (nid <= cid) continue;
                    int k = cell_head[static_cast<std::size_t>(nid)];
                    while (k >= 0) {
                        process(i, k);
                        k = next[static_cast<std::size_t>(k)];
                    }
                }
                i = next[static_cast<std::size_t>(i)];
            }
        }
    }
};

static bool relax_cell_list(
    double* pts, const double* radii, double cov_scale,
    int n, int max_cycles, double cell_size,
    std::mt19937_64& rng, std::normal_distribution<double>& ndist)
{
    FlatCellList cl;

    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        cl.build(pts, n, cell_size);

        bool any = false;
        cl.for_each_pair(pts, n, [&](int i, int j) {
            if (check_and_push(pts, i, j,
                    cov_scale * (radii[i] + radii[j]), rng, ndist))
                any = true;
        });

        if (!any) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

std::tuple<F64Array, bool> relax_positions_cpp(
    F64Array pts_in, F64Array radii_in,
    double cov_scale, int max_cycles, long long seed)
{
    auto pts_buf = pts_in.request();
    auto rad_buf = radii_in.request();
    const int n  = static_cast<int>(pts_buf.shape[0]);

    F64Array pts_out({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(3)});
    double*       pts = static_cast<double*>(pts_out.request().ptr);
    const double* src = static_cast<const double*>(pts_buf.ptr);
    std::copy(src, src + n*3, pts);
    const double* radii = static_cast<const double*>(rad_buf.ptr);

    std::mt19937_64 rng;
    if (seed < 0) { std::random_device rd; rng.seed(rd()); }
    else          { rng.seed(static_cast<std::uint64_t>(seed)); }
    std::normal_distribution<double> ndist(0.0, 1.0);

    bool converged = false;
    if (n < CELL_LIST_THRESHOLD) {
        converged = relax_full_pair(
            pts, radii, cov_scale, n, max_cycles, rng, ndist);
    } else {
        double max_r = *std::max_element(radii, radii + n);
        double cell_size = cov_scale * 2.0 * max_r;
        if (cell_size <= 0.0) cell_size = 1.0;
        converged = relax_cell_list(
            pts, radii, cov_scale, n, max_cycles, cell_size, rng, ndist);
    }

    return {pts_out, converged};
}

// ---------------------------------------------------------------------------

PYBIND11_MODULE(_relax_core, m) {
    m.doc() = "pasted._ext._relax_core: repulsion-relaxation (C++17).\n"
              "O(N^2) full-pair for N<64, flat Cell List O(N) for N>=64.";
    m.def(
        "relax_positions", &relax_positions_cpp,
        py::arg("pts"), py::arg("radii"), py::arg("cov_scale"),
        py::arg("max_cycles"), py::arg("seed") = -1LL,
        R"(
Resolve interatomic distance violations by iterative pair repulsion.

Uses a flat Cell List spatial index for N >= 64 (O(N) per cycle) and
falls back to a full O(N^2) pair loop for smaller structures.
Cell size = cov_scale * 2 * max(radii), computed automatically.

Parameters
----------
pts        : (n, 3) float64  – positions (C-contiguous, copied internally)
radii      : (n,)   float64  – Pyykkoe covalent radii (Ang)
cov_scale  : float           – minimum-distance scale factor
max_cycles : int             – iteration limit
seed       : int, optional   – RNG seed; -1 -> std::random_device

Returns
-------
(pts_out, converged) : ((n, 3) ndarray, bool)
        )"
    );
}
