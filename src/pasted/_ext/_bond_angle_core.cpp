/**
 * _bond_angle_core.cpp
 * ====================
 * C++17 implementation of bond_angle_entropy.
 *
 * Algorithm:
 *   1. Enumerate all pairs within the cutoff using FlatCellList and collect
 *      the per-atom neighbor unit vectors in CSR layout.
 *   2. For each atom j, compute the angle
 *        theta = arccos(clip(u_ja . u_jb, -1, +1))
 *      for every upper-triangular neighbor pair (a, b) and bin it into a
 *      fixed-size histogram.
 *   3. Compute Shannon entropy H = -sum p ln p for each atom's histogram
 *      and return the mean over all atoms that have at least one angle pair.
 *
 * Design notes:
 *   - The outer atom loop runs in C++, eliminating Python interpreter overhead.
 *   - Neighbor unit vectors are stored in three flat CSR arrays (nb_ux/uy/uz).
 *     The exact total entry count is known after count_pass, so reserve(acc)
 *     is called before resize to avoid any reallocation.
 *   - The angle histogram is a stack-allocated int[N_BINS_DEFAULT] array,
 *     keeping it in L1 cache and producing zero heap allocations per atom.
 *   - The FlatCellList is built once and reused for both count_pass and
 *     fill_pass, halving the cell-list construction cost.
 *   - CSR offsets and fill pointers use int64_t to prevent overflow for
 *     large structures with high coordination numbers.
 *   - The histogram bin count is fixed at N_BINS_DEFAULT (36) as a
 *     compile-time constant. If a different bin count is needed, wrap this
 *     function in a Python lambda on the calling side.
 *
 * Dependencies: C++17 stdlib + pybind11 only.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace py = pybind11;
using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

static constexpr int    CELL_THRESHOLD = 64;
static constexpr int    N_BINS_DEFAULT = 36;
static constexpr double PI             = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// FlatCellList — linked-list spatial index (identical to _graph_core /
// _steinhardt_core). Doubles cell_size until the total cell count fits
// within MAX_CELLS (1<<22) to bound memory use on pathological inputs.
// ---------------------------------------------------------------------------
struct FlatCellList {
    double inv_cell;
    int nx, ny, nz;
    double ox, oy, oz;
    std::vector<int> cell_head, next;

    void build(const double* pts, int n, double cell_size) {
        inv_cell = 1.0 / cell_size;
        double xmn=pts[0],xmx=pts[0],ymn=pts[1],ymx=pts[1],zmn=pts[2],zmx=pts[2];
        for (int i=1;i<n;++i){
            xmn=std::min(xmn,pts[i*3  ]); xmx=std::max(xmx,pts[i*3  ]);
            ymn=std::min(ymn,pts[i*3+1]); ymx=std::max(ymx,pts[i*3+1]);
            zmn=std::min(zmn,pts[i*3+2]); zmx=std::max(zmx,pts[i*3+2]);
        }
        ox=xmn-cell_size; oy=ymn-cell_size; oz=zmn-cell_size;
        {
            static constexpr std::int64_t MAX_CELLS = 1LL<<22;
            auto tnx=[&]{return static_cast<int>((xmx-ox)*inv_cell)+2;};
            auto tny=[&]{return static_cast<int>((ymx-oy)*inv_cell)+2;};
            auto tnz=[&]{return static_cast<int>((zmx-oz)*inv_cell)+2;};
            while(static_cast<std::int64_t>(tnx())*tny()*tnz()>MAX_CELLS){
                cell_size*=2.; inv_cell=1./cell_size;
                ox=xmn-cell_size; oy=ymn-cell_size; oz=zmn-cell_size;
            }
            nx=tnx(); ny=tny(); nz=tnz();
        }
        cell_head.assign(static_cast<std::size_t>(nx*ny*nz),-1);
        next.resize(static_cast<std::size_t>(n));
        for(int i=0;i<n;++i){
            int cx=static_cast<int>((pts[i*3  ]-ox)*inv_cell);
            int cy=static_cast<int>((pts[i*3+1]-oy)*inv_cell);
            int cz=static_cast<int>((pts[i*3+2]-oz)*inv_cell);
            int cid=cx+nx*(cy+ny*cz);
            next[i]=cell_head[cid]; cell_head[cid]=i;
        }
    }

    // Enumerate every unique unordered pair (i, j) with i != j within one
    // cell-size radius. Each pair is yielded exactly once (i < j by cell
    // ordering or the nid > cid guard for cross-cell pairs).
    template<typename F>
    void for_each_pair(int, F process) const {
        for(int cz=0;cz<nz;++cz)
        for(int cy=0;cy<ny;++cy)
        for(int cx=0;cx<nx;++cx){
            int cid=cx+nx*(cy+ny*cz);
            for(int i=cell_head[cid];i>=0;i=next[i]){
                // Pairs within the same cell (i already visited j via next[i])
                for(int j=next[i];j>=0;j=next[j]) process(i,j);
                // Pairs across neighboring cells (only cells with nid > cid
                // to avoid double-counting)
                for(int dz=-1;dz<=1;++dz)
                for(int dy=-1;dy<=1;++dy)
                for(int dx2=-1;dx2<=1;++dx2){
                    if(!dx2&&!dy&&!dz) continue;
                    int ncx=cx+dx2,ncy=cy+dy,ncz=cz+dz;
                    if(ncx<0||ncy<0||ncz<0||ncx>=nx||ncy>=ny||ncz>=nz) continue;
                    int nid=ncx+nx*(ncy+ny*ncz);
                    if(nid<=cid) continue;
                    for(int k=cell_head[nid];k>=0;k=next[k]) process(i,k);
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// bond_angle_entropy_cpp
// ---------------------------------------------------------------------------
// Edge-case handling:
//   n < 2 or cutoff <= 0     -> return 0.0
//   no pairs within cutoff   -> return 0.0 (fully isolated atoms)
//   atom with k < 2 neighbors -> skipped (no angle pairs possible)
//   d_ij < 1e-10             -> pair excluded (direction vector undefined
//                               for coincident coordinates)
//   dot outside [-1, 1]      -> clipped before arccos to suppress domain errors
//   no atom has angle pairs  -> return 0.0
double bond_angle_entropy_cpp(F64Array pts_in, double cutoff) {
    auto buf = pts_in.request();
    const int     n   = static_cast<int>(buf.shape[0]);
    const double* pts = static_cast<const double*>(buf.ptr);

    if (n < 2 || cutoff <= 0.0) return 0.0;

    const double     cutoff2 = cutoff * cutoff;
    constexpr int    n_bins  = N_BINS_DEFAULT;
    constexpr double bin_inv = n_bins / PI;  // maps theta in [0, pi] -> bin index

    // CSR storage for per-atom neighbor unit vectors.
    // nb_ptr[i] .. nb_ptr[i+1]-1 is the range of entries for atom i.
    // int64_t prevents overflow when N * avg_k exceeds INT_MAX.
    std::vector<std::int64_t> nb_ptr(static_cast<std::size_t>(n+1), 0);
    std::vector<double> nb_ux, nb_uy, nb_uz;

    // --- Pass 1: count neighbors per atom ---
    auto count_pass = [&](int i, int j) {
        const double dx=pts[i*3]-pts[j*3], dy=pts[i*3+1]-pts[j*3+1], dz=pts[i*3+2]-pts[j*3+2];
        const double d2=dx*dx+dy*dy+dz*dz;
        if (d2 >= cutoff2 || d2 < 1e-20) return;  // exclude out-of-cutoff and coincident pairs
        ++nb_ptr[static_cast<std::size_t>(i)];
        ++nb_ptr[static_cast<std::size_t>(j)];
    };

    // The FlatCellList is built once here and reused for both passes,
    // avoiding a second O(N) cell-list construction.
    if (n >= CELL_THRESHOLD) {
        FlatCellList cl; cl.build(pts, n, cutoff);
        cl.for_each_pair(n, count_pass);

        // Convert per-atom counts to CSR offsets via prefix sum.
        // reserve(acc) before resize guarantees a single allocation with no
        // reallocation, because the exact total entry count is now known.
        {
            std::int64_t acc = 0;
            for (int i = 0; i < n; ++i) {
                std::int64_t cnt = nb_ptr[static_cast<std::size_t>(i)];
                nb_ptr[static_cast<std::size_t>(i)] = acc;
                acc += cnt;
            }
            nb_ptr[static_cast<std::size_t>(n)] = acc;
            nb_ux.reserve(static_cast<std::size_t>(acc));
            nb_uy.reserve(static_cast<std::size_t>(acc));
            nb_uz.reserve(static_cast<std::size_t>(acc));
            nb_ux.resize(static_cast<std::size_t>(acc), 0.0);
            nb_uy.resize(static_cast<std::size_t>(acc), 0.0);
            nb_uz.resize(static_cast<std::size_t>(acc), 0.0);
        }

        // --- Pass 2: store unit vectors (reuse same FlatCellList) ---
        std::vector<std::int64_t> nb_fill(static_cast<std::size_t>(n), 0);

        auto fill_pass = [&](int i, int j) {
            const double dx=pts[i*3]-pts[j*3], dy=pts[i*3+1]-pts[j*3+1], dz=pts[i*3+2]-pts[j*3+2];
            const double d2=dx*dx+dy*dy+dz*dz;
            if (d2 >= cutoff2 || d2 < 1e-20) return;
            const double inv_d = 1.0 / std::sqrt(d2);

            auto write = [&](int atom, double ux, double uy, double uz) {
                const std::int64_t base = nb_ptr[static_cast<std::size_t>(atom)];
                std::int64_t& f         = nb_fill[static_cast<std::size_t>(atom)];
                nb_ux[static_cast<std::size_t>(base + f)] = ux;
                nb_uy[static_cast<std::size_t>(base + f)] = uy;
                nb_uz[static_cast<std::size_t>(base + f)] = uz;
                ++f;
            };
            write(i,  dx*inv_d,  dy*inv_d,  dz*inv_d);
            write(j, -dx*inv_d, -dy*inv_d, -dz*inv_d);
        };

        cl.for_each_pair(n, fill_pass);

    } else {
        // N < CELL_THRESHOLD: O(N^2) brute-force pair scan
        for(int i=0;i<n-1;++i)
            for(int j=i+1;j<n;++j)
                count_pass(i,j);

        // Same prefix-sum / reserve / resize pattern as above.
        {
            std::int64_t acc = 0;
            for (int i = 0; i < n; ++i) {
                std::int64_t cnt = nb_ptr[static_cast<std::size_t>(i)];
                nb_ptr[static_cast<std::size_t>(i)] = acc;
                acc += cnt;
            }
            nb_ptr[static_cast<std::size_t>(n)] = acc;
            nb_ux.reserve(static_cast<std::size_t>(acc));
            nb_uy.reserve(static_cast<std::size_t>(acc));
            nb_uz.reserve(static_cast<std::size_t>(acc));
            nb_ux.resize(static_cast<std::size_t>(acc), 0.0);
            nb_uy.resize(static_cast<std::size_t>(acc), 0.0);
            nb_uz.resize(static_cast<std::size_t>(acc), 0.0);
        }

        std::vector<std::int64_t> nb_fill(static_cast<std::size_t>(n), 0);

        auto fill_pass = [&](int i, int j) {
            const double dx=pts[i*3]-pts[j*3], dy=pts[i*3+1]-pts[j*3+1], dz=pts[i*3+2]-pts[j*3+2];
            const double d2=dx*dx+dy*dy+dz*dz;
            if (d2 >= cutoff2 || d2 < 1e-20) return;
            const double inv_d = 1.0 / std::sqrt(d2);

            auto write = [&](int atom, double ux, double uy, double uz) {
                const std::int64_t base = nb_ptr[static_cast<std::size_t>(atom)];
                std::int64_t& f         = nb_fill[static_cast<std::size_t>(atom)];
                nb_ux[static_cast<std::size_t>(base + f)] = ux;
                nb_uy[static_cast<std::size_t>(base + f)] = uy;
                nb_uz[static_cast<std::size_t>(base + f)] = uz;
                ++f;
            };
            write(i,  dx*inv_d,  dy*inv_d,  dz*inv_d);
            write(j, -dx*inv_d, -dy*inv_d, -dz*inv_d);
        };

        for(int i=0;i<n-1;++i)
            for(int j=i+1;j<n;++j)
                fill_pass(i,j);
    }

    // --- Pass 3: per-atom angle histogram and Shannon entropy ---
    // The histogram is a stack-allocated int[N_BINS_DEFAULT] array so it
    // stays in L1 cache and causes zero heap allocations inside the loop.
    double entropy_sum = 0.0;
    int    atom_count  = 0;  // number of atoms that contributed at least one angle pair

    for (int j = 0; j < n; ++j) {
        const std::int64_t beg = nb_ptr[static_cast<std::size_t>(j)];
        const std::int64_t end = nb_ptr[static_cast<std::size_t>(j) + 1];
        const std::int64_t k   = end - beg;
        if (k < 2) continue;  // need at least 2 neighbors to form an angle pair

        // Zero-initialize the stack histogram for this atom.
        int hist[N_BINS_DEFAULT] = {};
        int pair_cnt = 0;

        for (std::int64_t a = beg; a < end - 1; ++a) {
            for (std::int64_t b = a + 1; b < end; ++b) {
                const double dot = nb_ux[static_cast<std::size_t>(a)] * nb_ux[static_cast<std::size_t>(b)]
                                 + nb_uy[static_cast<std::size_t>(a)] * nb_uy[static_cast<std::size_t>(b)]
                                 + nb_uz[static_cast<std::size_t>(a)] * nb_uz[static_cast<std::size_t>(b)];
                // Clip to [-1, 1] before arccos to guard against floating-point
                // rounding that pushes the dot product slightly outside the domain.
                const double clipped = dot < -1.0 ? -1.0 : (dot > 1.0 ? 1.0 : dot);
                const double theta   = std::acos(clipped);
                int bin = static_cast<int>(theta * bin_inv);
                if (bin >= n_bins) bin = n_bins - 1;  // clamp theta == pi edge case
                ++hist[bin];
                ++pair_cnt;
            }
        }

        if (pair_cnt == 0) continue;  // unreachable in practice; safety guard

        // Shannon entropy: H = -sum_{b} p_b * ln(p_b)
        const double inv_total = 1.0 / static_cast<double>(pair_cnt);
        double h = 0.0;
        for (int b = 0; b < n_bins; ++b) {
            if (hist[b] > 0) {
                const double p = hist[b] * inv_total;
                h -= p * std::log(p);
            }
        }
        entropy_sum += h;
        ++atom_count;
    }

    return (atom_count > 0) ? entropy_sum / atom_count : 0.0;
}

// ---------------------------------------------------------------------------
PYBIND11_MODULE(_bond_angle_core, m) {
    m.doc() =
        "_bond_angle_core: O(N*k^2) bond-angle entropy in C++17.\n"
        "Uses FlatCellList for N>=64, shared across both CSR passes.\n"
        "Histogram bins are fixed at N_BINS_DEFAULT=36 (compile-time constant).\n"
        "CSR offsets use int64_t to support large N*k without overflow.";
    m.def(
        "bond_angle_entropy_cpp", &bond_angle_entropy_cpp,
        py::arg("pts"), py::arg("cutoff"),
        R"(
Compute bond-angle-distribution Shannon entropy, averaged over all atoms.

For each atom j with >= 2 neighbors within cutoff, all pairwise angles
theta_{ajb} = arccos(u_ja . u_jb) are histogrammed into 36 bins
over [0, pi]. Shannon entropy of the per-atom histogram is averaged.

The number of histogram bins is fixed at 36 (compile-time constant).
If a different bin count is required, wrap this function in Python.

Edge cases:
  n < 2 or no pairs within cutoff  -> 0.0
  atom with k < 2 neighbors        -> skipped (no angle pairs)
  d_ij < 1e-10 (coincident atoms)  -> pair excluded (direction undefined)
  dot product clipped to [-1, 1]   -> safe arccos

Parameters
----------
pts     : (n, 3) float64  atom positions (Å)
cutoff  : float           neighbor distance cutoff (Å)

Returns
-------
float  Mean per-atom bond-angle Shannon entropy in [0, ln(36)].
        )");
}