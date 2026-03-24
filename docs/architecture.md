# Architecture

## Design philosophy

PASTED has one job: produce atomic structures that are strange enough to
stress-test quantum-chemistry (QC) and machine-learning potential (MLP)
codes, but not so malformed that they cannot be parsed at all.

Three principles guide every design decision:

**The tool does not judge.**
Physical validity is not checked beyond the minimum-distance constraint.
Charge/multiplicity parity is validated; everything else is the engine's
problem.  The point is to find the engine's breaking points, not to
generate textbook molecules.

**Fail gracefully, not loudly.**
The C++ acceleration layer is optional.  If the extensions are not compiled,
the package falls back to pure Python/NumPy with no change in behavior.
If `relax_positions` does not converge within `max_cycles`, the best
available structure is returned with `converged=False` — the run continues.

**Small surface, stable interface.**
New parameters are added only when they cannot be reproduced by composing
existing ones.  `chain_bias=0.0` is the default precisely because it
changes nothing for existing users.  Internals can be restructured freely
as long as the public API stays the same.

---

## Source layout

```
src/pasted/
├── __init__.py          Public API exports and package docstring
├── _atoms.py            Element data, covalent radii, pool/filter parsing
├── _config.py           GeneratorConfig frozen dataclass (all generator params)
├── _generator.py        StructureGenerator class and generate() function
├── _io.py               XYZ serialization (format_xyz, parse_xyz)
├── _metrics.py          All 17 disorder metrics: entropy, RDF, Steinhardt,
│                        graph, and four adversarial metrics (v0.4.0)
├── _optimizer.py        StructureOptimizer — basin-hopping on existing structures
├── _placement.py        Placement algorithms + relax_positions + _affine_move
├── cli.py               argparse CLI entry point
├── neighbor_list.py     NeighborList — lazy-cached neighbor list for Python
│                        fallback metric computations (added v0.4.0)
└── _ext/
    ├── __init__.py      HAS_RELAX / HAS_MAXENT / HAS_MAXENT_LOOP /
    │                    HAS_STEINHARDT / HAS_GRAPH / HAS_BA_CPP /
    │                    HAS_COMBINED flags; None fallbacks
    ├── _relax.cpp       C++17: relax_positions with Verlet-list reuse (N≥64)
    ├── _maxent.cpp      C++17: angular_repulsion_gradient with Cell List (N≥64)
    ├── _steinhardt.cpp  C++17: Steinhardt Q_l with sparse neighbor list (N≥64)
    ├── _graph_core.cpp  C++17: graph_lcc/cc, ring_fraction, charge_frustration,
    │                            moran_I_chi, h_spatial, rdf_dev — all O(N·k)
    ├── _bond_angle_core.cpp  C++17: bond_angle_entropy_cpp — mean per-atom
    │                                 bond-angle Shannon entropy (added v0.4.0)
    └── _combined_core.cpp    C++17: all_metrics_cpp — single-pass FlatCellList
                                      kernel computing all 17 metrics in one
                                      traversal (added v0.4.0)
```

---

## Placement modes

### gas

Atoms are placed uniformly at random inside a sphere or axis-aligned box.
No clash checking at placement time — `relax_positions` is called afterwards
to resolve any distance violations.

### chain

A random-walk chain grows from the origin.  Each atom is attached to a
randomly chosen existing tip.  Two parameters control shape:

- `chain_persist` — local directional memory (cone constraint on turn angle)
- `chain_bias` — global axis drift; the first bond direction becomes a bias
  axis and all subsequent steps are blended toward it

`chain_bias=0.0` (the default) reproduces the behavior of versions prior
to 0.1.6 exactly.

### shell

A central atom is placed at the origin, surrounded by `coord_num` shell
atoms at a random radial distance.  Remaining atoms are attached as tails
via a short random walk from shell atoms.

### maxent

Atoms start from a random gas placement and are iteratively repositioned by
gradient descent on an angular repulsion potential:

$$U = \sum_i \sum_{j,k \in N(i)} \frac{1}{1 - \cos\theta_{jk} + \varepsilon}$$

The result is the constrained maximum-entropy solution: neighbor directions
spread as uniformly over the sphere as the distance constraints allow.

#### Performance — v0.2.6: O(N) neighbor-cutoff computation

Before v0.2.6, `place_maxent` computed the angular-repulsion neighbor cutoff
(`ang_cutoff`) by enumerating all N*(N+1)/2 pairwise covalent-radius sums,
sorting them, and taking the median — an O(N² log N) operation executed once
per structure before the L-BFGS loop.  For N=2,000 this single step consumed
~88% of total wall time when `maxent_steps` was small.

**Fix (v0.2.6):** the identity `median(rᵢ + rⱼ) = 2 · median(rᵢ)` holds
whenever the radius distribution is unimodal and approximately symmetric
about its median.  This is true for typical element pools (C/N/O, 1–30, etc.)
where errors are below 3 %.  For strongly **bimodal** pools such as H + heavy
metals, the approximation can overestimate `ang_cutoff` by up to ~50 % — the
angular repulsion is then applied over a wider neighbourhood, resulting in a
slightly weaker uniformity guarantee rather than a hard failure.  Pass an
explicit `cutoff=` value if strict `ang_cutoff` control is required.
The replacement:

```python
median_sum = float(np.median(radii)) * 2.0
```

is O(N), allocates no extra memory, and yields a cutoff within 3 % of the
exact value for all standard element pools.  Measured speedups vs. v0.2.5:

| n_atoms | v0.2.5    | v0.2.6   | speedup |
|--------:|----------:|---------:|--------:|
|     100 |    35 ms  |   22 ms  |   1.6 × |
|   1,000 |   364 ms  |  189 ms  |   1.9 × |
|   5,000 | 4,821 ms  | 1,274 ms |   3.8 × |
|  10,000 | 18,084 ms | 4,028 ms |   4.5 × |

`gas`, `chain`, and `shell` modes are unaffected by this change.

---

## Disorder metrics

All metrics are computed once per structure in `compute_all_metrics` and
stored in `Structure.metrics`.  Since v0.1.14, every metric uses cutoff-based
local pair enumeration (O(N·k)) — the O(N²) `scipy.spatial.distance.pdist`
path has been removed entirely.

Since v0.4.0 the full set comprises **17 metrics** (13 original + 4 adversarial
metrics introduced to stress-test GNN and MLP feature-engineering assumptions).

| Metric | Range | Description |
|---|---|---|
| `H_atom` | ≥ 0 | Shannon entropy of element composition |
| `H_spatial` | ≥ 0 | Shannon entropy of pair-distance histogram within *cutoff* |
| `H_total` | ≥ 0 | `w_atom × H_atom + w_spatial × H_spatial` |
| `RDF_dev` | ≥ 0 | RMS deviation of empirical g(r) from ideal-gas baseline within *cutoff* |
| `shape_aniso` | [0, 1] | Relative shape anisotropy from gyration tensor (0=sphere, 1=rod) |
| `Q4`, `Q6`, `Q8` | [0, 1] | Steinhardt bond-order parameters |
| `graph_lcc` | [0, 1] | Largest connected-component fraction |
| `graph_cc` | [0, 1] | Mean clustering coefficient |
| `ring_fraction` | [0, 1] | Fraction of atoms in at least one cycle in the cutoff-adjacency graph (Tarjan bridge-finding) |
| `charge_frustration` | ≥ 0 | Variance of \|Δχ\| across cutoff-adjacent pairs |
| `moran_I_chi` | (−∞, 1] | Moran's I spatial autocorrelation for Pauling electronegativity; 0 = random. Clamped to 1.0: sparse graphs (W < N) can inflate the raw n/W prefactor above 1 (see v0.3.8 fix). |
| `bond_angle_entropy` | [0, ln 36] | Mean per-atom Shannon entropy of bond-angle histogram (36 bins over [0, π]). Targets GNN/spherical-harmonic bias. *(added v0.4.0)* |
| `coordination_variance` | ≥ 0 | Population variance of per-atom coordination number. Targets GNN aggregation bias (octet-rule uniformity). *(added v0.4.0)* |
| `radial_variance` | ≥ 0 Å² | Mean per-atom variance of neighbor distances within cutoff. Captures local shell disorder independent of global g(r). *(added v0.4.0)* |
| `local_anisotropy` | [0, 1] | Mean per-atom relative shape anisotropy of local coordination tensor (0=isotropic, 1=rod-like). Targets bias toward symmetric local environments. *(added v0.4.0)* |

### Adversarial metrics (v0.4.0)

The four new metrics are computed by `_compute_adversarial()` in `_metrics.py`,
which shares a single `NeighborList` instance across all four to avoid
redundant tree queries.  When `HAS_COMBINED = True`, the combined C++ kernel
(`_combined_core`) computes them in the same single `FlatCellList` pass as the
original 13 metrics.

**Design motivation.**  Standard disorder metrics (entropy, Steinhardt,
graph topology) do not cover all the inductive biases exploited by GNN and
MLP codes.  The four adversarial metrics target specific assumptions that such
codes rely on:

- `bond_angle_entropy` — breaks the "bond angles cluster at 109.5°/120°"
  assumption common to MPNN message-passing.
- `coordination_variance` — breaks the "coordination number is nearly
  constant" assumption common to GNN aggregation.
- `radial_variance` — breaks the "first and second coordination shells are
  well-separated" assumption encoded by RBF descriptor grids.
- `local_anisotropy` — breaks the "local environment is isotropic or follows
  a known symmetry group" assumption in equivariant architectures.

### NeighborList (v0.4.0)

`pasted.neighbor_list.NeighborList` is a pure-Python helper class that
wraps `scipy.spatial.cKDTree` and lazily caches derived arrays:

| Property | Description |
|---|---|
| `deg` | Per-atom coordination number (first access triggers `np.bincount`) |
| `unit_diff` | Directed unit-vector differences `(2P, 3)`, coincident pairs masked (d < 1e-10) |
| `dists_sq` | Squared directed distances `(2P,)` |

All four adversarial metrics accept a pre-built `NeighborList` so they share
the pair enumeration done at construction time.  The coincident-pair guard
(`d < 1e-10`) in `unit_diff` mirrors the C++ threshold in
`_bond_angle_core.cpp`, ensuring the Python fallback and the C++ path produce
numerically identical results.

### compute_all_metrics dispatch chain

`compute_all_metrics` selects the fastest available path at runtime:

1. **`HAS_COMBINED = True`** (v0.4.0+): calls `all_metrics_cpp` — one
   `FlatCellList` traversal for all 17 metrics (~1.9× speedup vs. legacy C++
   path at N=1000).
2. **`HAS_GRAPH = True`, `HAS_COMBINED = False`**: calls `rdf_h_cpp`,
   `graph_metrics_cpp`, `steinhardt_per_atom`, and `bond_angle_entropy_cpp`
   (if available) independently — four separate `FlatCellList` passes.
3. **Pure Python** (`HAS_GRAPH = False`): uses `scipy.spatial.cKDTree` for
   pair enumeration; graph metrics fall back to a full O(N²)
   `pdist`/`squareform` pass.  Significantly slower for N ≳ 500.

### Unified adjacency definition

All nine cutoff-based metrics (`H_spatial`, `RDF_dev`, `Q4/Q6/Q8`,
`graph_lcc`, `graph_cc`, `ring_fraction`, `charge_frustration`,
`moran_I_chi`) treat a pair (i, j) as connected when `d_ij ≤ cutoff`.
Prior to v0.1.13, `ring_fraction` and `charge_frustration` used a
`cov_scale × (r_i + r_j)` bond-detection threshold that was structurally
never satisfied in relaxed structures (since `relax_positions` guarantees
`d_ij ≥ cov_scale × (r_i + r_j)` on convergence), causing both metrics to
return 0.0 for every PASTED output.

Prior to the current release, `ring_fraction` used a Union-Find (DSU)
algorithm that only marked the two direct endpoints of each detected
back-edge, causing systematic undercounting: an N-atom ring was reported
as 2/N instead of N/N (e.g. a benzene-like 6-cycle returned 0.33 instead of
1.0).  Both the Python fallback (`compute_ring_fraction` in `_metrics.py`)
and the C++ path (`graph_metrics_cpp` in `_graph_core.cpp`) had this bug.
The fix replaces Union-Find with **Tarjan's iterative bridge-finding
algorithm** (O(N + E)): a bond is a *bridge* if its removal disconnects the
graph; every atom whose incident edges include at least one non-bridge is
counted as a ring member.

---

## C++ acceleration layer

The `_ext` sub-package contains six independently compiled pybind11
modules.  Each can be absent without affecting the others.

| Flag | Module | Purpose |
|---|---|---|
| `HAS_RELAX` | `_relax_core` | `relax_positions` inner loop |
| `HAS_MAXENT` | `_maxent_core` | `angular_repulsion_gradient` |
| `HAS_MAXENT_LOOP` | `_maxent_core` | Full C++ L-BFGS loop for `place_maxent` |
| `HAS_STEINHARDT` | `_steinhardt_core` | `compute_steinhardt_per_atom` |
| `HAS_GRAPH` | `_graph_core` | Graph + RDF metrics (legacy multi-pass) |
| `HAS_BA_CPP` | `_bond_angle_core` | `bond_angle_entropy_cpp` (v0.4.0) |
| `HAS_COMBINED` | `_combined_core` | `all_metrics_cpp` single-pass kernel (v0.4.0) |

### `_relax_core` — `relax_positions`

Resolves distance violations by L-BFGS minimization of the harmonic
steric-clash penalty energy:

$$E = \sum_{i<j} \tfrac{1}{2} \max\!\bigl(0,\; \text{cov\_scale}\cdot(r_i+r_j) - d_{ij}\bigr)^2$$

Gradients are computed analytically; pair enumeration uses a **Verlet list**
for N ≥ 64: the list is rebuilt every `N_VERLET_REBUILD = 4` `evaluate()`
calls using an extended cutoff `cell_size + skin`, where the skin is
**adaptive** — `min(0.8 Å, cell_size × 0.3)` — capping the extended pair
list at ≤ (1.3)³ ≈ 2.2× the base count regardless of element radii.  For N
< 64 an O(N²) full-pair loop is used.  The L-BFGS solver (history depth m =
7, Armijo backtracking) is implemented in C++17 standard library with no
external dependencies.

Convergence criterion: E < 1 × 10⁻⁶.  Typical iteration count: 50–300
for dense 5000-atom structures.


### `_graph_core` — graph / ring / charge / Moran / RDF metrics

Exposes two functions, each performing a single O(N·k) `FlatCellList` pass
(N ≥ 64) or O(N²) full-pair scan (N < 64):

**`graph_metrics_cpp(pts, radii, cov_scale, en_vals, cutoff)`**
returns `{graph_lcc, graph_cc, ring_fraction, charge_frustration,
moran_I_chi}` — all five cutoff-based graph metrics in one call.

Internal design (v0.2.10):

- A **single shared adjacency list** (`bond_adj`) is built from the
  `FlatCellList` pass and reused for all five metrics.  The previous
  implementation maintained two identical lists (`bond_adj` and `graph_adj`)
  — a redundant O(N·k) allocation that has been removed.
- `graph_cc` triangle counting now uses **sorted adjacency lists +
  `std::binary_search`** (O(N·k²·log k)) rather than `std::find`
  (O(N·k³)).  For realistic k ≈ 2–16 this is 2–4× faster.
- `ring_fraction` uses **Tarjan's iterative bridge-finding** (O(N+E)) as of
  the `ring_fraction` bug-fix release.  See the Unified adjacency section.

**`rdf_h_cpp(pts, cutoff, n_bins)`** (added in v0.1.14)
returns `{h_spatial, rdf_dev}` — Shannon entropy and RDF deviation of the
pair-distance histogram within *cutoff*, replacing the former O(N²)
`scipy.spatial.distance.pdist` path.

Internal design (v0.2.10): distances are **streamed directly into the
histogram** inside the `FlatCellList` `collect` lambda.  The previous
implementation first collected all pair distances into a `std::vector<double>`
before binning — an unnecessary O(N·k) heap allocation that has been removed.

#### Per-bond arithmetic optimizations (v0.3.7)

**① + ② — atan2 elimination + Chebyshev recurrence.**  The former code
called `std::atan2` once per bond and then issued `l_max` independent
`std::cos` / `std::sin` pairs (18 libm calls at l_max=8, each ≈ 20–50 CPU
cycles).  The replacement:

1. Computes `cos_phi = dx/r_xy` and `sin_phi = dy/r_xy` from a single extra
   `sqrt` — no `atan2`.
2. Derives `cos(m·phi)` and `sin(m·phi)` for m = 2 … l_max via the
   Chebyshev two-term recurrence (2 multiplies + 1 subtract each) instead of
   independent libm calls.

Total cost per bond: 2 sqrts + (l_max−1)×4 arithmetic ops vs. 1 atan2 +
18 libm calls.  The ratio favours the new approach by roughly 4–5× on the
phi-trig component alone.

**③ — Stack-allocated P_lm.**  `compute_plm` previously received a
`std::vector<std::vector<double>>&` whose inner vectors were re-assigned
(zeroed and resized) on every bond call.  The function now takes a
caller-supplied `double[L_MAX+1][L_MAX+1]` allocated on the stack by the
lambda, eliminating the per-bond heap traffic and keeping the 936-byte
table in L1 cache for the entire pair-enumeration loop.

Combined speedup on `compute_steinhardt` (PASTED gas structures, k ≈ 0.7):

| N | v0.3.5 (baseline) | v0.3.6 (buf-transpose) | v0.3.7 (arithmetic) | total vs v0.3.5 |
|--:|:--:|:--:|:--:|:--:|
| 100 | 0.132 ms | 0.076 ms | **0.065 ms** | **2.0×** |
| 500 | 0.330 ms | 0.306 ms | **0.156 ms** | **2.1×** |
| 1 000 | 0.716 ms | 0.654 ms | **0.311 ms** | **2.3×** |
| 2 000 | 2.438 ms | 2.307 ms | **1.687 ms** | **1.4×** |
| 5 000 | 6.383 ms | 5.605 ms | **4.484 ms** | **1.4×** |

`compute_all_metrics` improves by **1.1–1.3×** at typical PASTED sizes
(N = 20–5 000), with the largest gain at N = 500–1 000 where trig was the
dominant cost.

#### Real spherical harmonics fast-path for l=4,6,8 (v0.3.8, optimization ④)

When `l_values = [4, 6, 8]` (the default), the `accumulate` lambda takes a
dedicated code path that replaces both the P_lm recurrence and the Chebyshev
trig recurrence with a single block of straight-line Cartesian polynomial
arithmetic.

**Why it works.**  On the unit sphere, `S_lm(x,y,z)` is a pure
integer-coefficient polynomial in `x,y,z`.  The `(1−z²)^(m/2)` factor in
`P_l^m(z)` is cancelled exactly by the `r_xy^m` from expanding
`cos(mφ)·r_xy^m` and `sin(mφ)·r_xy^m` — so no `sqrt`, no trig, no division
remains.

**Code generation.**  A SymPy script applied joint CSE across all three `l`
values simultaneously, maximising sharing of sub-expressions (`z²`, `z⁴`,
`x·y`, `x²−y²`, …).  Output: 84 CSE intermediates + 39 accumulation lines.
`std::pow(var, N)` for integer N was replaced by explicit multiplications
(`z*z*z*z` etc.) to avoid libm call overhead.  The generated code is
embedded verbatim in `_steinhardt.cpp`; the generator script is not shipped.

**Dispatch.**  The fast-path fires only when
`n_l == 3 && l_values == [4, 6, 8]` (runtime check, zero overhead for other
`l` combinations, which continue to use the ①②③ generic path).

**Measured speedup** (gas structures k≈0.7, `-O3 -std=c++17`, no `-march=native`):

| N | generic ①②③ | fast-path ④ | speedup |
|--:|---:|---:|---:|
| 20 | 0.015 ms | 0.013 ms | 1.2× |
| 100 | 0.040 ms | 0.028 ms | **1.4×** |
| 500 | 0.182 ms | 0.115 ms | **1.6×** |
| 1 000 | 0.374 ms | 0.239 ms | **1.6×** |
| 2 000 | 2.115 ms | 1.667 ms | 1.3× |
| 5 000 | 4.329 ms | 3.966 ms | 1.1× |

The 1.4–1.6× peak at N = 100–1 000 reflects the removal of the ~148-op
sequential-dependency chain in `compute_plm`, which had low IPC even under
`-O3` because each recurrence step reads the result of the previous one.
The straight-line polynomial code has no such dependency and allows the CPU
to schedule independent multiplications in parallel.

All metrics share the unified adjacency `d_ij ≤ cutoff`.
All computation is **single-threaded** (no OpenMP dependency).

> **v0.2.9 performance fix:** v0.2.3 introduced a two-pass design —
> `FlatCellList` candidates were buffered in an intermediate
> `std::vector<std::pair<int,int>>` and then redistributed through
> per-thread `local_pairs` buckets for a planned OpenMP parallel distance
> test.  Because `setup.py` never passes `-fopenmp`, `_OPENMP` is never
> defined at compile time, so `nthreads` was always `1` and the extra
> allocations were pure overhead (~35% slower at N = 10,000).  v0.2.9
> reverts to the original single-pass capturing-lambda pattern that writes
> directly into the adjacency lists as pairs are yielded by `for_each_pair`.

Performance at N = 10 000 (v0.1.13 → v0.1.14 → v0.2.9):

| Path | v0.1.13 | v0.1.14 | v0.2.3–v0.2.8 | v0.2.9 |
|---|:---:|:---:|:---:|:---:|
| `pdist` + `squareform` (removed) | ~2 880 ms | — | — | — |
| `graph_metrics_cpp` | ~4 ms | ~4 ms | ~5–6 ms | ~4 ms |
| `rdf_h_cpp` | — | ~2 ms | ~3 ms | ~2 ms |
| **`compute_all_metrics` total** | **~2 880 ms** | **~194 ms** | **~260 ms** | **~200 ms** |

### `_bond_angle_core` — `bond_angle_entropy_cpp` (v0.4.0)

Computes mean per-atom bond-angle Shannon entropy in a single O(N·k)
`FlatCellList` pass.

For each atom *j* with ≥ 2 neighbors within `cutoff`, all pairwise angles
θ_{ajb} = arccos(u_ja · u_jb) are histogrammed into **36 equal bins** over
[0, π].  The Shannon entropy of the per-atom histogram is averaged over all
atoms that have at least two neighbors.

- Bonds with d < 1e-10 Å are skipped (coincident-coordinate guard, mirrored
  in the Python `NeighborList.unit_diff` fallback).
- `HAS_BA_CPP` in `pasted._ext` is `True` when this module is compiled.
- When `HAS_COMBINED = True`, `_combined_core` subsumes this module — no
  separate call to `bond_angle_entropy_cpp` is issued by `compute_all_metrics`.

---

### `_combined_core` — `all_metrics_cpp` (v0.4.0)

Single-pass C++17 kernel that replaces the four independent `FlatCellList`
builds previously issued by `rdf_h_cpp`, `graph_metrics_cpp`,
`steinhardt_per_atom`, and `bond_angle_entropy_cpp`.

**One shared `FlatCellList` traversal** accumulates:

- RDF histogram → `h_spatial`, `rdf_dev`
- Bond adjacency lists → `graph_lcc`, `graph_cc`, `ring_fraction`,
  `charge_frustration`, `moran_I_chi`
- Steinhardt re/im buffers → `Q4`, `Q6`, `Q8` (fast-path ④ Cartesian
  polynomial expressions, same as `_steinhardt_core`)
- Per-atom distance sums → `radial_variance`
- Per-atom outer-product tensor components → `local_anisotropy`
- CSR neighbor count + unit-vector accumulation → `bond_angle_entropy`
  (no second cell-list pass: unit vectors are filled from the
  already-populated adjacency lists)
- Per-atom degree counts → `coordination_variance`

**NaN/Inf guard:** non-finite coordinates in `pts` are detected at entry and
return a zero-filled result dict matching the behavior of the individual
extension modules, rather than crashing in `FlatCellList::build`.

**Dispatch:** `compute_all_metrics` checks `HAS_COMBINED` first (highest
priority in the dispatch chain).  When `True`, `all_metrics_cpp` is the sole
C++ call; all other extension modules are unused for that call.

**Measured speedup** (C++ level, 4× `FlatCellList` → 1×):

| N    | Old multi-pass (ms) | `all_metrics_cpp` (ms) | Speedup |
|-----:|--------------------:|-----------------------:|--------:|
|   50 |               0.110 |                  0.057 |   1.93× |
|  500 |               0.615 |                  0.328 |   1.87× |
| 1000 |               1.131 |                  0.591 |   1.91× |

`HAS_COMBINED` in `pasted._ext` is `True` when this module is compiled.

---



**`angular_repulsion_gradient(pts, cutoff)`** — computes ∂U/∂rᵢ for the
angular repulsion potential used by `place_maxent`.

- **N < 32**: O(N³) full neighbor search
- **N ≥ 32**: O(N²) Cell List neighbor search (cell width = `cutoff`)

**`place_maxent_cpp(pts, radii, cov_scale, region_radius, ang_cutoff,
maxent_steps, trust_radius=0.5, seed=-1)`** (added in v0.1.15) — runs the
entire maxent gradient-descent loop in C++, replacing the Python
steepest-descent loop in `place_maxent()`.

> **`region_radius` vs `region` (Python API):** the C++ function takes a
> plain float `region_radius` (the numeric radius in Å extracted from the
> region string).  The public Python API uses the *string* form
> `region="sphere:R"` or `region="box:L"`.  `_generator.py` parses the string
> and passes the extracted float to the C++ layer — you never call
> `place_maxent_cpp` directly.

1. L-BFGS (m=7, Armijo backtracking) on the angular repulsion potential
2. Per-atom trust-radius cap: step uniformly rescaled so no atom moves more
   than `trust_radius` Å — replaces the fixed `maxent_lr × unit-norm clip`
3. Soft restoring force and center-of-mass pinning in C++
4. Embedded steric-clash relaxation (same L-BFGS penalty as `_relax_core`)

`HAS_MAXENT_LOOP` in `pasted._ext` is `True` when `place_maxent_cpp` is
available.  `place_maxent()` dispatches to it automatically.

Measured speedup (n_samples=20, repeats=10):

| Scenario | v0.1.13 | v0.1.14 | v0.1.15 | vs v0.1.14 |
|---|---:|---:|---:|---:|
| maxent small (n=8)   | ~157 ms | ~156 ms | **~7 ms** | **~22×** |
| maxent medium (n=15) | ~310 ms | ~300 ms | **~29 ms** | **~10×** |
| maxent large (n=20)  | ~320 ms | ~310 ms | **~30 ms** | **~10×** |

### `_steinhardt_core` — `compute_steinhardt_per_atom`

Computes per-atom Steinhardt Q_l bond-order parameters using a sparse
neighbor list, replacing the dense O(N²) Python/scipy path.

- **N < 64**: O(N²) full-pair loop
- **N ≥ 64**: O(N·k) flat Cell List (k = mean neighbor count)

Spherical harmonics are evaluated via the associated Legendre polynomial
three-term recurrence with no scipy call in the hot loop.  The symmetry
`|Y_l^{-m}|² = |Y_l^m|²` halves harmonic evaluations by computing only
m = 0..l.  When absent, `_steinhardt_per_atom_sparse` provides the same
O(N·k) complexity using `scipy.spatial.cKDTree` for neighbor enumeration
and `np.bincount` for accumulation.

All computation is **single-threaded** (no OpenMP dependency).  v0.2.3
introduced a pre-built `std::vector<std::vector<int>> nb_list` so that the
atom loop could carry a `#pragma omp parallel for` annotation — but without
`-fopenmp` the pragma is a no-op, leaving only the extra neighbor-list
allocation as overhead.  v0.2.9 restores the original single-pass lambda
that writes `re_buf`/`im_buf` directly during neighbor traversal.

| Path | N=2000 | Speed-up vs dense Python |
|---|:---:|:---:|
| Dense Python (original) | ~35 s | 1× |
| Sparse Python fallback | ~0.21 s | ~164× |
| C++ (`_steinhardt_core`) | ~17 ms | **~2 000×** |

#### Accumulator buffer layout — (N, n_l, l_max+1) since v0.3.6

**Background — former superlinear scaling.**  Prior to v0.3.6, the C++
accumulator used layout `(n_l, l_max+1, N)` with atom index innermost:

```cpp
// OLD: re_buf[li * lm1 * n  +  m * n  +  i]   stride = N × 8 B per m-step
std::vector<double> re_buf(n_l * (l_max + 1) * n, 0.0);
```

When the inner `m`-loop (0 … l_max = 8) wrote to `re_buf`, consecutive
iterations accessed addresses `N × 8 bytes` apart.  At N ≤ 500 those
strides fit in L2 cache; at N ≥ 1 000 the two buffers crossed into L3
(422 KB → 2.1 MB total), inflating each write latency ~5–10× and causing
wall time to grow superlinearly even though the algorithm is O(N·k·l²).

**Fix applied (v0.3.6).**  The buffer is now laid out as `(N, n_l, l_max+1)` —
atom index **outermost**:

```cpp
// NEW: re_buf[i * n_l * lm1  +  li * lm1  +  m]   stride = 8 B per m-step
std::vector<double> re_buf(n * n_l * (l_max + 1), 0.0);
```

Every bond's `(l_idx, m)` writes for atom `i` now fall in a contiguous block
of `n_l × (l_max+1) × 8 = 216` bytes — fitting in a single cache line
regardless of N.  Measured speedup (PASTED default structures, k ≈ 0.7):

| N | v0.3.5 | v0.3.6 (buf-transpose) | v0.3.7 (+Chebyshev+stack) | vs v0.3.5 |
|--:|:--:|:--:|:--:|:--:|
| 20 | – | 0.052 ms | **0.015 ms** | – |
| 100 | 0.132 ms | 0.076 ms | **0.065 ms** | **2.0×** |
| 500 | 0.330 ms | 0.306 ms | **0.156 ms** | **2.1×** |
| 1 000 | 0.716 ms | 0.654 ms | **0.311 ms** | **2.3×** |
| 2 000 | 2.438 ms | 2.307 ms | **1.687 ms** | **1.4×** |
| 5 000 | 6.383 ms | 5.605 ms | **4.484 ms** | **1.4×** |

The buffer-transpose (v0.3.6) mainly helped at N ≥ 2 000 (cache boundary).
The arithmetic optimizations (v0.3.7) are most visible at N = 500–1 000
because that is where the per-bond trig was the dominant cost.  Together
the two changes deliver **2.1–2.3×** on `compute_steinhardt` and
**1.3–1.4×** on `compute_all_metrics` across typical PASTED structures.

---

## StructureGenerator internals

`StructureGenerator.stream()` is the single implementation of the generation
loop.  It yields each passing structure immediately, enabling incremental
file output and early stopping via `n_success`.

`generate()` delegates to `stream()`, collects results into a
`GenerationResult`, and returns it.  `__iter__` also delegates to `stream()`,
so all three call sites share the same loop logic.

### GenerationResult

`generate()` returns a `GenerationResult` rather than a plain
`list[Structure]`.  `GenerationResult` is fully list-compatible — indexing,
iteration, `len()`, and boolean coercion all work as expected — while also
exposing per-run rejection metadata:

| Attribute | Type | Description |
|---|---|---|
| `structures` | `list[Structure]` | Structures that passed all filters |
| `n_attempted` | `int` | Total placement attempts |
| `n_passed` | `int` | Structures that passed (equals `len(result)`) |
| `n_rejected_parity` | `int` | Attempts rejected by charge/multiplicity parity |
| `n_rejected_filter` | `int` | Attempts rejected by metric filters |
| `n_success_target` | `int \| None` | The `n_success` value in effect |

Calling `result.summary()` returns a one-line diagnostic string, e.g.:

```
passed=5  attempted=50  rejected_parity=12  rejected_filter=33
```

> **Label vs attribute name:** the labels in the `summary()` string
> (`passed`, `attempted`, `rejected_parity`, `rejected_filter`) are
> shortened display names.  The corresponding Python attributes use an
> `n_` prefix — `result.n_passed`, `result.n_attempted`,
> `result.n_rejected_parity`, `result.n_rejected_filter`.
> Accessing `result.passed` or `result.attempted` directly raises
> `AttributeError`.

### warnings.warn behavior

`stream()` emits a `UserWarning` (via `warnings.warn`) in the following situations:

| Situation | Warning message |
|---|---|
| All attempts rejected by parity (no filter failures) | *"All N attempt(s) were rejected by the charge/multiplicity parity check …"* |
| Parity failures **and** filter failures both present, no structures passed | *"M of N attempt(s) were rejected by the parity check, and the remaining K that passed parity were rejected by metric filters …"* |
| No parity failures but all structures rejected by filters | *"No structures passed the metric filters after N attempt(s) …"* |
| Budget exhausted before `n_success` | *"Attempt budget exhausted (N attempts) before reaching n_success=K …"* |

> **Note — partial parity rejection with passing structures:** when some
> attempts are rejected by parity but at least one structure still passes,
> **no warning is emitted**.  Mixed-element pools (e.g. `elements="6,7,8"`)
> routinely produce ~50 % parity failures by chance; this is expected
> behavior and is reported only in the verbose summary line
> (`rejected_parity=N`), not as a `UserWarning`.

These warnings fire regardless of the `verbose` flag so that downstream
consumers (ASE, high-throughput pipelines) receive a machine-visible signal
even when PASTED is not in verbose mode.  Previously, all such messages were
printed to stderr only when `verbose=True`, making silent empty-list returns
indistinguishable from successful runs in automated pipelines.

### n_success and n_samples termination semantics

- `n_samples > 0`, `n_success = None` — attempt exactly `n_samples` times
  (original behavior).
- `n_samples > 0`, `n_success = N` — stop at N successes or `n_samples`
  attempts, whichever comes first.
- `n_samples = 0`, `n_success = N` — unlimited attempts; stop only when N
  structures have passed.

---

## Reproducibility

All random decisions pass through a single `random.Random(seed)` instance
created at the start of each `stream()` call.  The C++ extensions accept
an integer `seed` for the coincident-atom RNG edge case.  With the same
`seed`, the same `n_atoms`, and the same parameters, `stream()` (and
therefore `generate()`) returns bit-for-bit identical output.

---

## GeneratorConfig (v0.2.2)

`GeneratorConfig` is a `frozen=True` dataclass defined in `_config.py` that
captures every parameter accepted by `StructureGenerator`.

**Design goals:**

- **Type safety** — every field carries a type annotation; mypy and IDEs can
  check all call sites.
- **Immutability** — `frozen=True` prevents accidental mutation after
  construction and makes configs hashable.
- **Convenient overrides** — `dataclasses.replace(cfg, seed=99)` creates a
  new config with one field changed, leaving the original intact.
- **Backward compatibility** — `StructureGenerator` and `generate()` retain
  their original keyword-argument signatures; a `GeneratorConfig` is built
  internally when raw kwargs are passed.

**Calling conventions:**

```python
# Config-based (type-checked end-to-end)
cfg = GeneratorConfig(n_atoms=20, charge=0, mult=1, mode="gas",
                      region="sphere:10", elements="6,7,8", seed=42)
gen = StructureGenerator(cfg)   # or generate(cfg)

# Keyword-based (backward-compatible, unchanged from pre-v0.2.2)
gen = StructureGenerator(n_atoms=20, charge=0, mult=1, mode="gas",
                          region="sphere:10", elements="6,7,8", seed=42)
```

`StructureGenerator.__getattr__` proxies attribute access to `_cfg`, so
`gen.n_atoms`, `gen.seed`, etc. continue to work in existing code.

## Affine transform in StructureGenerator (v0.2.2 / v0.2.10)

When `affine_strength > 0.0`, each structure goes through an extra step
between placement and relax:

```
place_gas / place_chain / place_shell / place_maxent
          ↓
_affine_move(positions, move_step=0.0, affine_strength, rng,
             affine_stretch=..., affine_shear=..., affine_jitter=...)
          ↓
relax_positions
```

`_affine_move` is defined in `_placement.py` and shared with
`StructureOptimizer` (which uses it per MC step when
`allow_affine_moves=True`, passing `move_step=self.move_step`).  The
Generator always passes `move_step=0.0` (pure geometric transform, no
per-atom jitter) because a jitter before relax would simply be undone.

### Per-operation strength (v0.2.10)

Three optional parameters let callers adjust each affine operation
independently without changing the global `affine_strength` baseline:

| Parameter | Operation | Default |
|---|---|---|
| `affine_stretch` | Scale along one random axis by `Uniform(1−s, 1+s)` | `affine_strength` |
| `affine_shear` | Off-diagonal shear by `Uniform(−s/2, s/2)` | `affine_strength` |
| `affine_jitter` | Per-atom translation noise proportional to `move_step` | `affine_strength` |

When a parameter is `None` (the default), the value of `affine_strength` is
used — preserving full backward compatibility.  Set to `0.0` to disable a
specific operation while keeping the others active:

```python
# Shear only — no stretch, no jitter
gen = StructureGenerator(
    n_atoms=20, charge=0, mult=1, mode="gas", region="sphere:8",
    affine_strength=0.2, affine_stretch=0.0, affine_jitter=0.0,
)

# Stretch only — no shear, no jitter
gen = StructureGenerator(
    n_atoms=20, charge=0, mult=1, mode="gas", region="sphere:8",
    affine_strength=0.2, affine_shear=0.0, affine_jitter=0.0,
)
```

The same parameters are accepted by `StructureOptimizer` when
`allow_affine_moves=True`.

---

## StructureOptimizer internals

`StructureOptimizer.run()` returns an `OptimizationResult` containing all
per-restart structures sorted by objective value (highest first).

### Move types (all methods)

| Move | Probability | Description |
|---|---|---|
| Fragment coordinate | 50 % (or 25 % when affine moves enabled) | Atoms with local Q6 > `frag_threshold` are displaced by up to `move_step` Å |
| Affine coordinate | 0 % (25 % when `allow_affine_moves=True`) | Random stretch/compress along one axis + shear of one axis pair, applied to all atoms; per-atom jitter of `move_step × 0.25` added; center of mass pinned. Controlled by `affine_strength` (default 0.1 = ±10 % stretch); each operation can be overridden individually via `affine_stretch`, `affine_shear`, `affine_jitter`. |
| Composition | 50 % | Parity-preserving pool replacement (see below) |

**Parity-preserving composition move** — two paths:

1. **Pool replacement** (primary, up to 20 tries): pick a random atom and
   replace it with a *different* element drawn from `element_pool` whose
   atomic number has the same Z%2 parity as the original atom.  Because
   ΔZ = Z(new) − Z(old) is even, total electron count parity is preserved.
2. **Two-atom replacement** (fallback when Path 1 finds no same-parity
   candidate): replace two atoms simultaneously so ΔZ_total is even.

If the initial structure supplied to `run()` contains atoms outside the
pool, each foreign atom is replaced by a parity-compatible pool element
via `_sanitize_atoms_to_pool` before the MC loop begins.  This applies to
all three methods (`"annealing"`, `"basin_hopping"`, `"parallel_tempering"`).

### Optimization methods

#### `"annealing"` (Simulated Annealing)

Exponential temperature cooling from `T_start` to `T_end` over `max_steps`.
Each step proposes a move and accepts via the Metropolis criterion:
`accept if ΔE ≥ 0 or random < exp(ΔE / T)`.  `n_restarts` independent runs
are launched; `OptimizationResult` contains all of them sorted best-first.

#### `"basin_hopping"` (Basin-Hopping)

Constant temperature `T_start` with 3× relax cycles per step for stronger
local minimization.  Otherwise identical to SA.

#### `"parallel_tempering"` (Replica-Exchange MC)

`n_replicas` independent Markov chains run at temperatures spaced
geometrically from `T_end` (coldest) to `T_start` (hottest).  Every
`pt_swap_interval` steps, adjacent replica pairs attempt a state exchange:

```
ΔE = (β_k − β_{k+1}) × (f_{k+1} − f_k)   # β = 1/T, f = objective
accept with probability min(1, exp(ΔE))
```

Hot replicas cross energy barriers; swaps tunnel high-quality states to the
cold replica.  `OptimizationResult` includes the global best (tracked across
all replicas and all steps) plus each replica's final state.

Measured quality vs SA at equal wall time (n=10, C/N/O/P/S, H_total − Q6):

| Method | wall time | H_total (↑) | Q6 (↓) |
|---|---:|---:|---:|
| SA restarts=4, steps=500 | 460 ms | 1.591 | 0.401 |
| PT replicas=4, restarts=1, steps=200 | **102 ms** | **1.685** | 0.403 |
| PT replicas=4, restarts=2, steps=500 | 579 ms | **1.713** | **0.293** |

## Memory management in C++ extensions (v0.2.1 / v0.2.2)

### `_relax_core` — `PenaltyEvaluator`

`tgrad_` (gradient buffers, `3N × 8 bytes`) and
`pairs_` (pair-list vector) are **persistent members** allocated once at
construction.  `evaluate()` zeroes `tgrad_` with `std::fill` and calls
`pairs_.clear()` (preserving capacity) instead of creating new vectors on
every L-BFGS iteration.

This eliminates several gigabytes of heap churn per structure at large N
(e.g. ~8 GB / structure at n=150 000, 300 iterations).

**Verlet-list reuse (v0.2.2):** `pairs_` is rebuilt only every
`N_VERLET_REBUILD = 4` evaluate() calls via an extended `FlatCellList`
pass with cutoff `cell_size + skin` (adaptive skin =
`min(0.8 Å, cell_size × 0.3)`).  Between rebuilds the same extended pair
list is reused, eliminating the dominant serial traversal cost per L-BFGS
iteration.  The counter-based trigger avoids the O(N) displacement-check
loop and is safe when L-BFGS trust radius is large relative to the skin.

### `_maxent_core` — `place_maxent_cpp_impl`

Two persistent scratch objects are declared before the step loop and passed
into the hot-path functions:

| Object | Type | Purpose |
|---|---|---|
| `tgrad_scratch` | `vector<vector<double>>` | `eval_angular` gradient buffers |
| `ux_s / uy_s / uz_s / id_s` | `vector<double>` | per-atom unit-vector scratch (serial path) |
| `nb_scratch` | `vector<vector<int>>` | neighbor list reused via `build_nb_inplace` |

`build_nb_inplace()` clears inner vectors without deallocating, then refills
them — avoiding `~0.9 GB` of `nb` churn per structure at n=50 000.
`eval_angular()` accepts the scratch buffers as reference parameters; it
resizes them lazily (only grows, never shrinks).

