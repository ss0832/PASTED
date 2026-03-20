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
├── _io.py               XYZ serialization (format_xyz)
├── _metrics.py          All disorder metrics: entropy, RDF, Steinhardt, graph
├── _optimizer.py        StructureOptimizer — basin-hopping on existing structures
├── _placement.py        Placement algorithms + relax_positions + _affine_move
├── cli.py               argparse CLI entry point
└── _ext/
    ├── __init__.py      HAS_RELAX / HAS_POISSON / HAS_MAXENT / HAS_STEINHARDT / HAS_GRAPH
    │                    flags; None fallbacks
    ├── _relax.cpp       C++17: relax_positions with Verlet-list reuse (N≥64);
    │                    Bridson Poisson-disk sampling (poisson_disk_sphere / poisson_disk_box)
    ├── _maxent.cpp      C++17: angular_repulsion_gradient with Cell List (N≥64)
    ├── _steinhardt.cpp  C++17: Steinhardt Q_l with sparse neighbor list (N≥64)
    └── _graph_core.cpp  C++17: graph_lcc/cc, ring_fraction, charge_frustration,
                                         moran_I_chi, h_spatial, rdf_dev — all O(N·k)
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
whenever the radius distribution is unimodal and symmetric about its median,
which is true for all built-in element pools.  The replacement:

```python
median_sum = float(np.median(radii)) * 2.0
```

is O(N), allocates no extra memory, and yields a numerically identical
`ang_cutoff` for all tested element pools.  Measured speedups vs. v0.2.5:

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
| `ring_fraction` | [0, 1] | Fraction of atoms in at least one cycle in the cutoff-adjacency graph |
| `charge_frustration` | ≥ 0 | Variance of \|Δχ\| across cutoff-adjacent pairs |
| `moran_I_chi` | (−∞, 1] | Moran's I spatial autocorrelation for Pauling electronegativity; 0 = random |

### Unified adjacency definition

All nine cutoff-based metrics (`H_spatial`, `RDF_dev`, `Q4/Q6/Q8`,
`graph_lcc`, `graph_cc`, `ring_fraction`, `charge_frustration`,
`moran_I_chi`) treat a pair (i, j) as connected when `d_ij ≤ cutoff`.
Prior to v0.1.13, `ring_fraction` and `charge_frustration` used a
`cov_scale × (r_i + r_j)` bond-detection threshold that was structurally
never satisfied in relaxed structures (since `relax_positions` guarantees
`d_ij ≥ cov_scale × (r_i + r_j)` on convergence), causing both metrics to
return 0.0 for every PASTED output.

---

## C++ acceleration layer

The `_ext` sub-package contains four independently compiled pybind11
modules.  Each can be absent without affecting the others.

### `_relax_core` — `relax_positions` and Poisson-disk sampling

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

**Bridson Poisson-disk sampling** (added in v0.2.2):

Two additional functions are exported from the same module and accessible via
`pasted._ext` when `HAS_POISSON` is `True`:

- **`_poisson_disk_sphere_cpp(n, radius, min_dist, seed=-1, k=30)`** — places
  *n* atoms inside a sphere of *radius* Å with a guaranteed `min_dist`
  separation between any two atoms (Bridson algorithm, flat-array grid).
  Falls back to uniform random for any slots that could not be placed.
- **`_poisson_disk_box_cpp(n, lx, ly, lz, min_dist, seed=-1, k=30)`** — same
  for an axis-aligned box.

`place_gas()` uses uniform random placement for performance predictability
across all density regimes.  Call these functions directly when a
minimum-separation guarantee is needed:

```python
from pasted._ext import HAS_POISSON, _poisson_disk_sphere_cpp

if HAS_POISSON:
    pts = _poisson_disk_sphere_cpp(n=100, radius=10.0, min_dist=1.5, seed=42)
    # pts is an (n, 3) float64 ndarray
```

### `_graph_core` — graph / ring / charge / Moran / RDF metrics

Exposes two functions, each performing a single O(N·k) `FlatCellList` pass
(N ≥ 64) or O(N²) full-pair scan (N < 64):

**`graph_metrics_cpp(pts, radii, cov_scale, en_vals, cutoff)`**
returns `{graph_lcc, graph_cc, ring_fraction, charge_frustration,
moran_I_chi}` — all five cutoff-based graph metrics in one call.

**`rdf_h_cpp(pts, cutoff, n_bins)`** (added in v0.1.14)
returns `{h_spatial, rdf_dev}` — Shannon entropy and RDF deviation of the
pair-distance histogram within *cutoff*, replacing the former O(N²)
`scipy.spatial.distance.pdist` path.

All metrics share the unified adjacency `d_ij ≤ cutoff`.

Performance at N = 10 000 (v0.1.13 → v0.1.14):

| Path | v0.1.13 | v0.1.14 | Speed-up |
|---|:---:|:---:|:---:|
| `pdist` + `squareform` (removed) | ~2 880 ms | — | — |
| `graph_metrics_cpp` | ~4 ms | ~4 ms | — |
| `rdf_h_cpp` (new) | — | ~2 ms | — |
| **`compute_all_metrics` total** | **~2 880 ms** | **~194 ms** | **~15×** |

### `_maxent_core` — `angular_repulsion_gradient` and `place_maxent_cpp`

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

| Path | N=2000 | Speed-up vs dense Python |
|---|:---:|:---:|
| Dense Python (original) | ~35 s | 1× |
| Sparse Python fallback | ~0.21 s | ~164× |
| C++ (`_steinhardt_core`) | ~17 ms | **~2 000×** |


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

### warnings.warn behavior

`stream()` emits a `UserWarning` (via `warnings.warn`) in three situations:

| Situation | Warning message |
|---|---|
| All attempts rejected by parity | *"All N attempt(s) were rejected by the charge/multiplicity parity check …"* |
| Some attempts rejected by parity | *"M of N attempt(s) were rejected by the charge/multiplicity parity check …"* |
| No structures pass filters | *"No structures passed the metric filters after N attempt(s) …"* |
| Budget exhausted before `n_success` | *"Attempt budget exhausted (N attempts) before reaching n_success=K …"* |

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

## Affine transform in StructureGenerator (v0.2.2)

When `affine_strength > 0.0`, each structure goes through an extra step
between placement and relax:

```
place_gas / place_chain / place_shell / place_maxent
          ↓
_affine_move(positions, move_step=0.0, affine_strength, rng)
          ↓
relax_positions
```

`_affine_move` is defined in `_placement.py` and shared with
`StructureOptimizer` (which uses it per MC step when
`allow_affine_moves=True`, passing `move_step=self.move_step`).  The
Generator always passes `move_step=0.0` (pure geometric transform, no
per-atom jitter) because a jitter before relax would simply be undone.



`StructureOptimizer.run()` returns an `OptimizationResult` containing all
per-restart structures sorted by objective value (highest first).

### Move types (all methods)

| Move | Probability | Description |
|---|---|---|
| Fragment coordinate | 50 % (or 25 % when affine moves enabled) | Atoms with local Q6 > `frag_threshold` are displaced by up to `move_step` Å |
| Affine coordinate | 0 % (25 % when `allow_affine_moves=True`) | Random stretch/compress along one axis + shear of one axis pair, applied to all atoms; per-atom jitter of `move_step × 0.25` added; center of mass pinned. Controlled by `affine_strength` (default 0.1 = ±10 % stretch). |
| Composition | 50 % | Parity-preserving element swap or replacement (see below) |

**Parity-preserving composition move** — two paths:

1. **Swap** (primary): exchange the element types of two atoms with different
   symbols.  Total electron count is unchanged → always parity-valid.
2. **Same-Z-parity replace** (fallback when all atoms are the same element):
   replace atom `i` with an element whose atomic number has the same Z%2 as
   `atoms[i]`.  Net ΔZ is even → parity preserved.  If only odd-Z pool
   elements differ, two atoms are replaced simultaneously so ΔZ_total is even.

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

