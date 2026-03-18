/**
 * pasted._ext._maxent_core
 * ========================
 * C++17 implementation of the angular repulsion gradient used exclusively
 * by the maxent placement mode.
 *
 * Exported function
 * -----------------
 *   angular_repulsion_gradient(pts, cutoff)
 *       -> grad: ndarray(n, 3)
 *
 * Background
 * ----------
 * For each atom i and every pair (j, k) of its neighbours within cutoff,
 * the potential
 *
 *     U_jk = 1 / (1 - cos theta_jk + eps)
 *
 * penalises neighbour directions that are close on the unit sphere.
 * eps = 1e-6 guards against division by zero.
 *
 * This function returns dU/dr_i of shape (n, 3).
 *
 * Spatial partitioning (Cell List)
 * ---------------------------------
 * For N >= CELL_LIST_THRESHOLD (32) the neighbour list for each atom is built
 * using a Cell List with cell_size = cutoff.  Only atoms in the 27-cell
 * neighbourhood are considered, reducing the neighbour-search from O(N^2)
 * to O(N) and the overall gradient from O(N^3) to O(N^2).
 *
 * Complexity
 * ----------
 *   N <  32 : O(N^3)  (full pairwise neighbour search)
 *   N >= 32 : O(N^2)  (Cell List neighbour search + O(k^2) per atom,
 *                       k = average neighbour count)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

static constexpr int    CELL_LIST_THRESHOLD = 32;
static constexpr double EPS                 = 1e-6;

// ---------------------------------------------------------------------------
// Cell List helpers (shared with _relax.cpp logic, duplicated for independence)
// ---------------------------------------------------------------------------

using CellKey = std::array<int,3>;

struct CellKeyHash {
    std::size_t operator()(const CellKey& k) const noexcept {
        std::size_t h = static_cast<std::size_t>(k[0]);
        h ^= static_cast<std::size_t>(k[1]) * 2654435761ULL
             + 0x9e3779b9ULL + (h<<6) + (h>>2);
        h ^= static_cast<std::size_t>(k[2]) * 2246822519ULL
             + 0x9e3779b9ULL + (h<<6) + (h>>2);
        return h;
    }
};

using CellMap = std::unordered_map<CellKey, std::vector<int>, CellKeyHash>;

// Build neighbour list for every atom using a Cell List.
// nb[i] contains indices of atoms within cutoff of atom i (i excluded).
static std::vector<std::vector<int>> build_neighbour_list_cell(
    const double* pts, int n, double cutoff)
{
    const double inv_cell = 1.0 / cutoff;

    CellMap cells;
    cells.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        CellKey key = {
            static_cast<int>(std::floor(pts[i*3+0] * inv_cell)),
            static_cast<int>(std::floor(pts[i*3+1] * inv_cell)),
            static_cast<int>(std::floor(pts[i*3+2] * inv_cell))
        };
        cells[key].push_back(i);
    }

    const double cutoff2 = cutoff * cutoff;
    std::vector<std::vector<int>> nb(static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i) {
        CellKey ck = {
            static_cast<int>(std::floor(pts[i*3+0] * inv_cell)),
            static_cast<int>(std::floor(pts[i*3+1] * inv_cell)),
            static_cast<int>(std::floor(pts[i*3+2] * inv_cell))
        };
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            CellKey nk = {ck[0]+dx, ck[1]+dy, ck[2]+dz};
            auto it = cells.find(nk);
            if (it == cells.end()) continue;
            for (const int j : it->second) {
                if (j == i) continue;
                const double ddx = pts[i*3+0]-pts[j*3+0];
                const double ddy = pts[i*3+1]-pts[j*3+1];
                const double ddz = pts[i*3+2]-pts[j*3+2];
                if (ddx*ddx + ddy*ddy + ddz*ddz <= cutoff2)
                    nb[static_cast<std::size_t>(i)].push_back(j);
            }
        }
    }
    return nb;
}

// Build neighbour list with full O(N^2) search (used when N < threshold).
static std::vector<std::vector<int>> build_neighbour_list_full(
    const double* pts, int n, double cutoff)
{
    const double cutoff2 = cutoff * cutoff;
    std::vector<std::vector<int>> nb(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            const double dx = pts[i*3+0]-pts[j*3+0];
            const double dy = pts[i*3+1]-pts[j*3+1];
            const double dz = pts[i*3+2]-pts[j*3+2];
            if (dx*dx + dy*dy + dz*dz <= cutoff2)
                nb[static_cast<std::size_t>(i)].push_back(j);
        }
    }
    return nb;
}

// ---------------------------------------------------------------------------
// Gradient accumulation (common to both paths)
// ---------------------------------------------------------------------------

static void accumulate_gradient(
    const double* pts, int n,
    const std::vector<std::vector<int>>& nb,
    double* grad)
{
    for (int i = 0; i < n; ++i) {
        const auto& nbi = nb[static_cast<std::size_t>(i)];
        if (nbi.size() < 2) continue;

        // Precompute unit vectors from i to each neighbour j
        // and store distances.
        const std::size_t nb_count = nbi.size();
        std::vector<double> ux(nb_count), uy(nb_count), uz(nb_count), inv_d(nb_count);
        for (std::size_t idx = 0; idx < nb_count; ++idx) {
            const int j = nbi[idx];
            const double dx = pts[i*3+0] - pts[j*3+0];
            const double dy = pts[i*3+1] - pts[j*3+1];
            const double dz = pts[i*3+2] - pts[j*3+2];
            const double d  = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (d > 0.0) {
                const double id = 1.0 / d;
                ux[idx] = dx*id; uy[idx] = dy*id; uz[idx] = dz*id;
                inv_d[idx] = id;
            } else {
                ux[idx] = uy[idx] = uz[idx] = inv_d[idx] = 0.0;
            }
        }

        for (std::size_t ji = 0; ji < nb_count; ++ji) {
            if (inv_d[ji] <= 0.0) continue;
            const double id_j = inv_d[ji];
            const double ujx = ux[ji], ujy = uy[ji], ujz = uz[ji];

            for (std::size_t ki = 0; ki < nb_count; ++ki) {
                const double cos_val = ujx*ux[ki] + ujy*uy[ki] + ujz*uz[ki];
                const double denom   = 1.0 - cos_val + EPS;
                const double weight  = 1.0 / (denom * denom);

                grad[i*3+0] += weight * (ux[ki] - cos_val*ujx) * id_j;
                grad[i*3+1] += weight * (uy[ki] - cos_val*ujy) * id_j;
                grad[i*3+2] += weight * (uz[ki] - cos_val*ujz) * id_j;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

F64Array angular_repulsion_gradient_cpp(F64Array pts_in, double cutoff) {
    auto buf = pts_in.request();
    const int     n   = static_cast<int>(buf.shape[0]);
    const double* pts = static_cast<const double*>(buf.ptr);

    F64Array grad_out({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(3)});
    double* grad = static_cast<double*>(grad_out.request().ptr);
    std::fill(grad, grad + n*3, 0.0);

    auto nb = (n < CELL_LIST_THRESHOLD)
        ? build_neighbour_list_full(pts, n, cutoff)
        : build_neighbour_list_cell(pts, n, cutoff);

    accumulate_gradient(pts, n, nb, grad);
    return grad_out;
}

// ---------------------------------------------------------------------------

PYBIND11_MODULE(_maxent_core, m) {
    m.doc() = "pasted._ext._maxent_core: angular repulsion gradient (C++17).\n"
              "O(N^3) for N<32, O(N^2) with Cell List for N>=32.";
    m.def(
        "angular_repulsion_gradient", &angular_repulsion_gradient_cpp,
        py::arg("pts"), py::arg("cutoff"),
        R"(
Gradient of the angular repulsion potential U = sum 1/(1 - cos theta_jk + eps).

Uses Cell List spatial index for N >= 32, full O(N^2) neighbour search for
N < 32.

Parameters
----------
pts    : (n, 3) float64 – atom positions (C-contiguous)
cutoff : float          – neighbour distance cutoff (Ang)

Returns
-------
grad : (n, 3) float64   – dU/dr_i for each atom i
        )"
    );
}
