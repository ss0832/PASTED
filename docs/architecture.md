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
├── _generator.py        StructureGenerator class and generate() function
├── _io.py               XYZ serialization (format_xyz)
├── _metrics.py          All disorder metrics: entropy, RDF, Steinhardt, graph
├── _optimizer.py        StructureOptimizer — basin-hopping on existing structures
├── _placement.py        Placement algorithms + relax_positions dispatcher
├── cli.py               argparse CLI entry point
└── _ext/
    ├── __init__.py      HAS_RELAX / HAS_MAXENT / HAS_STEINHARDT / HAS_GRAPH /
    │                    HAS_OPENMP flags; set_num_threads(); None fallbacks
    ├── _relax.cpp       C++17 + OpenMP: relax_positions with flat Cell List (N≥64)
    ├── _maxent.cpp      C++17 + OpenMP: angular_repulsion_gradient with Cell List (N≥64)
    ├── _steinhardt.cpp  C++17 + OpenMP: Steinhardt Q_l with sparse neighbor list (N≥64)
    └── _graph_core.cpp  C++17 + OpenMP: graph_lcc/cc, ring_fraction, charge_frustration,
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

### `_relax_core` — `relax_positions`

Resolves distance violations by L-BFGS minimization of the harmonic
steric-clash penalty energy:

$$E = \sum_{i<j} \tfrac{1}{2} \max\!\bigl(0,\; \text{cov\_scale}\cdot(r_i+r_j) - d_{ij}\bigr)^2$$

Gradients are computed analytically; pair enumeration uses `FlatCellList`
for N ≥ 64 (O(N) per energy evaluation) and an O(N²) full-pair loop for
N < 64.  The L-BFGS solver (history depth m = 7, Armijo backtracking) is
implemented in C++17 standard library with no external dependencies.

Convergence criterion: E < 1 × 10⁻⁶.  Typical iteration count: 50–300
for dense 5000-atom structures.

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

---

## OpenMP parallelization (v0.2.0+)

All four C++ extension modules are parallelized with OpenMP on Linux when
built with `-fopenmp`.  The build system detects availability automatically;
`PASTED_DISABLE_OPENMP=1` opts out.

**Platform support**

| Platform | C++ extensions | OpenMP |
|---|:---:|:---:|
| Linux (GCC ≥ 7 or Clang + libomp) | ✅ | ✅ automatic |
| macOS | best-effort | ❌ not attempted |
| Windows | best-effort | ❌ not attempted |

**Parallelization strategy**

The `FlatCellList` linked-list traversal is inherently serial, so
parallelization follows a *pair-list-then-parallel* pattern:

1. Build the pair list (or neighbor list) serially in a single
   `FlatCellList` pass — O(N·k) work, not the bottleneck.
2. Distribute the resulting flat `vector<pair<int,int>>` over threads.
3. Each thread owns a private gradient buffer (or distance / adjacency
   buffer); results are merged into the shared output after the parallel
   loop, avoiding false sharing and atomic instructions.

This pattern delivers linear speedup with thread count for large N where
the pair-list step is negligible compared to the per-pair work.

**Runtime thread control**

```python
import pasted

print(pasted.HAS_OPENMP)        # True on Linux with -fopenmp build
pasted.set_num_threads(4)       # use 4 threads for all subsequent calls
```

CLI equivalent::

    pasted --n-atoms 50000 --mode gas --region sphere:250 \
        --charge 0 --mult 1 --n-threads 4 -o out.xyz

`OMP_NUM_THREADS` is also respected as the standard OpenMP environment
variable.  `set_num_threads()` takes precedence over `OMP_NUM_THREADS`
when called after import.

---


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

## Optimizer

`StructureOptimizer.run()` returns an `OptimizationResult` containing all
per-restart structures sorted by objective value (highest first).

### Move types (all methods)

| Move | Probability | Description |
|---|---|---|
| Fragment coordinate | 50 % | Atoms with local Q6 > `frag_threshold` are displaced by up to `move_step` Å |
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
