/**
 * pasted._ext._maxent_core
 * ========================
 * C++17 implementation of the angular repulsion gradient used exclusively
 * by the ``maxent`` placement mode.
 *
 * Exported function
 * -----------------
 *   angular_repulsion_gradient(pts, cutoff)
 *       -> grad: ndarray(n, 3)
 *
 * Background
 * ----------
 * For each atom i and every pair (j, k) of its neighbours within *cutoff*,
 * the potential
 *
 *     U_jk = 1 / (1 - cos θ_jk + ε)
 *
 * penalises neighbour directions that are close on the unit sphere.
 * ε = 1e-6 guards against division by zero when two directions coincide.
 *
 * This function returns ∂U/∂r_i of shape (n, 3).  The Python fallback in
 * _placement.py uses an identical formula; see that module for the reference
 * implementation and algorithmic notes.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <limits>
#include <vector>

namespace py = pybind11;

using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// ---------------------------------------------------------------------------

F64Array angular_repulsion_gradient_cpp(F64Array pts_in, double cutoff) {
    auto buf = pts_in.request();
    const int     n   = static_cast<int>(buf.shape[0]);
    const double* pts = static_cast<const double*>(buf.ptr);

    F64Array grad_out({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(3)});
    double* grad = static_cast<double*>(grad_out.request().ptr);
    std::fill(grad, grad + n * 3, 0.0);

    constexpr double eps = 1e-6;
    const double     inf = std::numeric_limits<double>::infinity();

    // Precompute pairwise distances and unit vectors (j→i direction)
    const std::size_t nn = static_cast<std::size_t>(n * n);
    std::vector<double> dist(nn, inf);
    std::vector<double> uhat(nn * 3, 0.0);  // uhat[(i*n+j)*3 + {0,1,2}]

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            const double dx = pts[i * 3 + 0] - pts[j * 3 + 0];
            const double dy = pts[i * 3 + 1] - pts[j * 3 + 1];
            const double dz = pts[i * 3 + 2] - pts[j * 3 + 2];
            const double d  = std::sqrt(dx * dx + dy * dy + dz * dz);
            dist[static_cast<std::size_t>(i * n + j)] = d;
            if (d > 0.0) {
                const double       inv_d = 1.0 / d;
                const std::size_t  base  = static_cast<std::size_t>((i * n + j) * 3);
                uhat[base + 0] = dx * inv_d;
                uhat[base + 1] = dy * inv_d;
                uhat[base + 2] = dz * inv_d;
            }
        }
    }

    // Accumulate gradient for each atom i
    for (int i = 0; i < n; ++i) {
        // Collect neighbour indices within cutoff
        std::vector<int> nb;
        nb.reserve(16);
        for (int j = 0; j < n; ++j)
            if (j != i && dist[static_cast<std::size_t>(i * n + j)] <= cutoff)
                nb.push_back(j);

        if (nb.size() < 2) continue;

        for (const int j : nb) {
            const double      d_ij    = dist[static_cast<std::size_t>(i * n + j)];
            if (d_ij <= 0.0) continue;
            const double      inv_dij = 1.0 / d_ij;
            const std::size_t bj      = static_cast<std::size_t>((i * n + j) * 3);
            const double      ux_j    = uhat[bj + 0];
            const double      uy_j    = uhat[bj + 1];
            const double      uz_j    = uhat[bj + 2];

            // k == j contributes perp == 0; included for parity with Python
            for (const int k : nb) {
                const std::size_t bk  = static_cast<std::size_t>((i * n + k) * 3);
                const double ux_k     = uhat[bk + 0];
                const double uy_k     = uhat[bk + 1];
                const double uz_k     = uhat[bk + 2];

                const double cos_val  = ux_j * ux_k + uy_j * uy_k + uz_j * uz_k;
                const double denom    = 1.0 - cos_val + eps;
                const double weight   = 1.0 / (denom * denom);

                // perp = uhat_k - cos_val * uhat_j
                grad[i * 3 + 0] += weight * (ux_k - cos_val * ux_j) * inv_dij;
                grad[i * 3 + 1] += weight * (uy_k - cos_val * uy_j) * inv_dij;
                grad[i * 3 + 2] += weight * (uz_k - cos_val * uz_j) * inv_dij;
            }
        }
    }

    return grad_out;
}

// ---------------------------------------------------------------------------

PYBIND11_MODULE(_maxent_core, m) {
    m.doc() = "pasted._ext._maxent_core: angular repulsion gradient for "
              "maxent placement (C++17)";

    m.def(
        "angular_repulsion_gradient",
        &angular_repulsion_gradient_cpp,
        py::arg("pts"),
        py::arg("cutoff"),
        R"(
Gradient of the angular repulsion potential U = Σ 1/(1 - cos θ_jk + ε).

Parameters
----------
pts    : (n, 3) float64 – atom positions (C-contiguous)
cutoff : float          – neighbour distance cutoff (Å)

Returns
-------
grad : (n, 3) float64   – ∂U/∂r_i for each atom i
        )"
    );
}
