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
 * Used by every placement mode (gas, chain, shell, maxent) through the
 * Python dispatcher in _placement.py.
 *
 * Seed convention
 * ---------------
 *   seed >= 0  : deterministic (std::mt19937_64 seeded with that value)
 *   seed == -1 : non-deterministic (seeded from std::random_device)
 *
 * Update strategy
 * ---------------
 * Gauss-Seidel: positions are updated immediately within each cycle so that
 * each subsequent pair sees the already-corrected coordinates.  This differs
 * from the NumPy fallback in _placement.py which uses a Jacobi-style update
 * (distance matrix frozen for the whole cycle).  Both strategies converge to
 * a valid structure; they produce different coordinate trajectories.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <tuple>
#include <vector>

namespace py = pybind11;

using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// ---------------------------------------------------------------------------

std::tuple<F64Array, bool> relax_positions_cpp(
    F64Array  pts_in,
    F64Array  radii_in,
    double    cov_scale,
    int       max_cycles,
    long long seed
) {
    auto pts_buf = pts_in.request();
    auto rad_buf = radii_in.request();
    const int n  = static_cast<int>(pts_buf.shape[0]);

    // Output: mutable copy of input positions
    F64Array pts_out({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(3)});
    double*       pts   = static_cast<double*>(pts_out.request().ptr);
    const double* src   = static_cast<const double*>(pts_buf.ptr);
    std::copy(src, src + n * 3, pts);

    const double* radii = static_cast<const double*>(rad_buf.ptr);

    // Precompute pairwise minimum-distance thresholds
    std::vector<double> thresh(static_cast<std::size_t>(n * n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            thresh[static_cast<std::size_t>(i * n + j)] =
                cov_scale * (radii[i] + radii[j]);

    // RNG used only when two atoms are exactly coincident (d < 1e-10)
    std::mt19937_64 rng;
    if (seed < 0) {
        std::random_device rd;
        rng.seed(rd());
    } else {
        rng.seed(static_cast<std::uint64_t>(seed));
    }
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    bool converged = false;

    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        bool any_violation = false;

        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const double dx  = pts[i * 3 + 0] - pts[j * 3 + 0];
                const double dy  = pts[i * 3 + 1] - pts[j * 3 + 1];
                const double dz  = pts[i * 3 + 2] - pts[j * 3 + 2];
                const double d   = std::sqrt(dx * dx + dy * dy + dz * dz);
                const double thr = thresh[static_cast<std::size_t>(i * n + j)];

                if (d >= thr) continue;
                any_violation = true;

                double vx, vy, vz;
                if (d < 1e-10) {
                    // Coincident atoms: push in a random unit direction
                    do {
                        vx = normal_dist(rng);
                        vy = normal_dist(rng);
                        vz = normal_dist(rng);
                    } while (vx * vx + vy * vy + vz * vz < 1e-20);
                    const double inv_len =
                        1.0 / std::sqrt(vx * vx + vy * vy + vz * vz);
                    vx *= inv_len; vy *= inv_len; vz *= inv_len;
                } else {
                    const double inv_d = 1.0 / d;
                    vx = dx * inv_d; vy = dy * inv_d; vz = dz * inv_d;
                }

                const double push = (thr - d) * 0.5;
                pts[i * 3 + 0] += push * vx;
                pts[i * 3 + 1] += push * vy;
                pts[i * 3 + 2] += push * vz;
                pts[j * 3 + 0] -= push * vx;
                pts[j * 3 + 1] -= push * vy;
                pts[j * 3 + 2] -= push * vz;
            }
        }

        if (!any_violation) {
            converged = true;
            break;
        }
    }

    return {pts_out, converged};
}

// ---------------------------------------------------------------------------

PYBIND11_MODULE(_relax_core, m) {
    m.doc() = "pasted._ext._relax_core: repulsion-relaxation inner loop (C++17)";

    m.def(
        "relax_positions",
        &relax_positions_cpp,
        py::arg("pts"),
        py::arg("radii"),
        py::arg("cov_scale"),
        py::arg("max_cycles"),
        py::arg("seed") = -1LL,
        R"(
Resolve interatomic distance violations by iterative pair repulsion.

Parameters
----------
pts        : (n, 3) float64  – positions (C-contiguous, copied internally)
radii      : (n,)   float64  – Pyykkö covalent radii (Å)
cov_scale  : float           – minimum-distance scale factor
max_cycles : int             – iteration limit
seed       : int, optional   – RNG seed for coincident-atom case;
                               -1 → std::random_device (default)

Returns
-------
(pts_out, converged) : ((n, 3) ndarray, bool)
        )"
    );
}
