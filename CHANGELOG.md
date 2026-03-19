# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.17] - 2026-03-19

### Added

- **`allow_displacements` parameter for `StructureOptimizer`.**

  Controls whether atomic-position moves (fragment moves) are performed
  during optimisation.

  | Value | Behaviour |
  |---|---|
  | `True` (default) | Fragment moves (atomic displacements) are included in the MC step pool — unchanged from v0.1.16 |
  | `False` | Only composition moves (element-type swaps) are executed; atomic coordinates are held fixed for the entire run |

  Use `allow_displacements=False` when exploring compositional disorder on
  a pre-relaxed geometry (e.g. a fixed lattice).  Passing both
  `allow_displacements=False` and `allow_composition_moves=False`
  simultaneously raises a `ValueError` because no move type would remain
  enabled.

  Applies to all three optimisation methods: `"annealing"`,
  `"basin_hopping"`, and `"parallel_tempering"`.

  CLI: `--no-displacements` flag added to the `--optimize` mode.

  ```python
  opt = StructureOptimizer(
      n_atoms=50, charge=0, mult=1,
      objective={"H_atom": 1.0, "Q6": -2.0},
      elements=["Cr", "Mn", "Fe", "Co", "Ni"],
      allow_displacements=False,   # composition-only optimisation
      max_steps=5000, seed=42,
  )
  result = opt.run(initial=fixed_geometry)
  ```

### Fixed

- **`OptimizationResult.method` docstring** now lists all three valid
  methods (`"annealing"`, `"basin_hopping"`, `"parallel_tempering"`);
  previously `"parallel_tempering"` was omitted.



### Added

- **`allow_composition_moves` parameter for `StructureOptimizer`.**

  Controls whether element-type swaps are performed during optimisation.

  | Value | Behaviour |
  |---|---|
  | `True` (default) | Each MC step randomly chooses between a fragment move (position change) and a composition move (element-type swap) with equal probability — unchanged from v0.1.15 |
  | `False` | Only fragment moves are executed; element types are held fixed for the entire run |

  Use `allow_composition_moves=False` when the composition is predetermined
  and should not be modified during optimisation (e.g. optimising the
  geometry of a fixed stoichiometry).

  Applies to all three optimisation methods: `"annealing"`,
  `"basin_hopping"`, and `"parallel_tempering"`.

  CLI: `--no-composition-moves` flag added to the `--optimize` mode.

  ```python
  opt = StructureOptimizer(
      n_atoms=12, charge=0, mult=1,
      objective={"H_total": 1.0, "Q6": -2.0},
      elements="24,25,26,27,28",
      allow_composition_moves=False,   # position-only optimisation
      max_steps=5000, seed=42,
  )
  result = opt.run(initial=my_structure)
  ```

- **`element_fractions` parameter for `StructureGenerator` / `generate()`.**

  Specifies relative sampling weights per element as a `{symbol: weight}`
  dict.  Weights are normalised internally; elements absent from the dict
  receive weight `1.0`.  Default (`None`) keeps the original uniform
  sampling.

  ```python
  gen = StructureGenerator(
      n_atoms=20, charge=0, mult=1,
      mode="gas", region="sphere:10",
      elements="6,7,8",
      element_fractions={"C": 0.6, "N": 0.3, "O": 0.1},
      n_samples=50, seed=0,
  )
  ```

  CLI: `--element-fractions SYM:WEIGHT` (repeatable).

  ```
  pasted --n-atoms 20 --elements 6,7,8 --charge 0 --mult 1 \
      --mode gas --region sphere:10 --n-samples 50 \
      --element-fractions C:0.6 --element-fractions N:0.3 --element-fractions O:0.1
  ```

- **`element_min_counts` and `element_max_counts` parameters for
  `StructureGenerator` / `generate()`.**

  Hard per-element atom count bounds enforced at sampling time.

  | Parameter | Type | Effect |
  |---|---|---|
  | `element_min_counts` | `dict[str, int] \| None` | Guaranteed lower bound; atoms are placed first, remaining slots filled by weighted sampling |
  | `element_max_counts` | `dict[str, int] \| None` | Upper bound; elements that have reached their cap are excluded from further sampling |

  Both default to `None` (no bounds).  The generator raises `ValueError`
  at construction time when constraints are inconsistent (sum of mins
  exceeds `n_atoms`, or any min > its paired max).  A `RuntimeError` is
  raised during sampling if all elements are simultaneously capped before
  `n_atoms` is reached.

  ```python
  gen = StructureGenerator(
      n_atoms=15, charge=0, mult=1,
      mode="gas", region="sphere:10",
      elements="6,7,8,15,16",
      element_min_counts={"C": 4},        # at least 4 carbon atoms
      element_max_counts={"N": 3, "O": 3}, # at most 3 N and 3 O
      n_samples=100, seed=42,
  )
  ```

  CLI: `--element-min-counts SYM:N` and `--element-max-counts SYM:N`
  (both repeatable).

  ```
  pasted --n-atoms 15 --elements 6,7,8,15,16 --charge 0 --mult 1 \
      --mode gas --region sphere:10 --n-samples 100 \
      --element-min-counts C:4 \
      --element-max-counts N:3 --element-max-counts O:3
  ```

- **20 new tests** across `test_generator.py` and `test_optimizer.py`:
  - `TestElementFractions` (6 tests) — bias validation, unknown/negative/zero
    weight errors, uniform-weight seed parity, functional-API forwarding.
  - `TestElementMinMaxCounts` (8 tests) — min/max enforcement, combined
    constraints, sum-exceeds-n_atoms error, min > max error, unknown element
    errors, impossible-cap RuntimeError.
  - `TestAllowCompositionMoves` (6 tests) — default True, composition
    preservation when disabled (SA and PT), still optimises, `repr`
    behaviour.

## [0.1.15] - 2026-03-19

### Changed

- **`place_maxent` now uses L-BFGS with a per-atom trust radius instead of
  steepest descent.**

  When `HAS_MAXENT_LOOP` is `True` (i.e. `_maxent_core.place_maxent_cpp` is
  available), the entire gradient-descent loop runs in C++:

  | Step | v0.1.14 (Python SD) | v0.1.15 (C++ L-BFGS) |
  |---|---|---|
  | Gradient computation | C++ `angular_repulsion_gradient` | C++ (inlined, same Cell List) |
  | Optimiser | steepest descent, fixed `maxent_lr` | L-BFGS m=7, Armijo backtracking |
  | Step limit | unit-norm clip × `maxent_lr` | per-atom trust radius (default 0.5 Å) |
  | Restoring force | Python NumPy | C++ |
  | CoM pinning | Python NumPy | C++ |
  | Steric relaxation | Python wrapper → C++ | C++ direct (embedded PenaltyEvaluator) |
  | list ↔ ndarray conversion | every step | none |

  Measured wall-time improvement (n_atoms=8–20, n_samples=20, repeats=10):

  | Scenario | v0.1.13 | v0.1.14 | v0.1.15 | speedup vs 0.1.14 |
  |---|---:|---:|---:|---:|
  | maxent small (n=8)   | ~157 ms | ~156 ms |  **~7 ms** | **~22×** |
  | maxent medium (n=15) | ~310 ms | ~300 ms | **~29 ms** | **~10×** |
  | maxent large (n=20)  | ~320 ms | ~310 ms | **~30 ms** | **~10×** |

  Output quality (H_total, 30 structures, 3 seeds): C++ L-BFGS mean ≈ 1.09
  vs Python SD mean ≈ 1.04.  L-BFGS converges to comparable or better local
  optima with far fewer wall-clock seconds.

  The L-BFGS curvature information reduces the number of steps needed for
  convergence; the trust-radius cap (uniform step rescaling so no atom moves
  more than `trust_radius` Å) replaces the fixed `maxent_lr` unit-norm clip
  and provides better convergence on anisotropic landscapes.

- **New C++ function `place_maxent_cpp`** added to `_maxent.cpp` and exported
  from `pasted._ext`.  Signature:

  ```
  place_maxent_cpp(pts, radii, cov_scale, region_radius, ang_cutoff,
                   maxent_steps, trust_radius=0.5, seed=-1) -> ndarray(n,3)
  ```

- **New flag `HAS_MAXENT_LOOP`** in `pasted._ext` (`bool`).
  `True` when `place_maxent_cpp` is available.
  `HAS_MAXENT` remains `True` whenever `angular_repulsion_gradient` is
  available (unchanged semantics).

- **`place_maxent` gains a `trust_radius` parameter** (float, default `0.5` Å).
  Ignored by the steepest-descent fallback (which continues to use
  `maxent_lr` and unit-norm clipping).  The `maxent_lr` parameter is
  retained for backward compatibility.

- **`_optimizer._run_one` patched** (Metropolis loop):
  - `cov_radius_ang` results are pre-computed once per restart into a `radii`
    array and reused every step, eliminating per-step dict lookups.
  - `relax_positions` Python wrapper is bypassed; `_relax_core.relax_positions`
    is called directly when `HAS_RELAX` is `True`, eliminating per-step
    `list → ndarray → list` conversions.

- **`_maxent.cpp` refactored**: `angular_repulsion_gradient` and
  `place_maxent_cpp` now share a single `build_nb` / `eval_angular` pair
  instead of duplicating neighbour-list and gradient logic.



### Added

- **`OptimizationResult`** — new return type for `StructureOptimizer.run()`.

  `run()` previously returned a single `Structure` (the best across all
  restarts).  It now returns an `OptimizationResult` that collects **all
  per-restart structures** sorted by objective value, highest first.

  `OptimizationResult` is list-compatible — indexing, iteration, `len()`,
  and `bool()` all work — while also exposing dedicated metadata:

  | Attribute | Description |
  |---|---|
  | `best` | Highest-scoring `Structure` — equivalent to `result[0]` |
  | `all_structures` | All per-restart structures, best-first |
  | `objective_scores` | Scalar objective values, same order |
  | `n_restarts_attempted` | Restarts that produced a valid initial structure |
  | `method` | `"annealing"` or `"basin_hopping"` |

  `result.summary()` returns a one-line diagnostic, e.g.:
  ```
  restarts=5  best_f=1.2294  worst_f=0.8123  method='annealing'
  ```

  **Migration from v0.1.14**: code that uses `opt.run()` as a `Structure`
  must add `.best`: `opt.run().best`.  The CLI is already updated.
  Code that only iterates or indexes the result works without changes.

  A `UserWarning` is emitted when restarts are skipped due to failed
  initial-structure generation.

- **`OptimizationResult` added to `pasted.__all__`** and exported from the
  top-level `pasted` namespace.

- **`cli.py`** updated: `opt.run()` → `opt.run().best` in the
  `--optimize` code path.

- **Objective alignment verified** — SA and BH reliably improve the
  user-supplied objective over random gas-mode baselines:

  | Scenario | Baseline mean | Optimized |
  |---|---:|---:|
  | maximize `H_total` (n=8, C+O) | 0.495 | **1.229** (+148%) |
  | maximize `H_spatial − 2×Q6` (n=12, C/N/O/H) | −0.108 | **+0.892** |

  Temperature schedule confirmed: T decays exponentially from `T_start`
  to `T_end` over `max_steps`.  `n_restarts` returns the global best
  (not just the last restart's result).

- **16 new tests** in `tests/test_optimizer.py`:
  - `TestOptimizationResult` — list interface, `best`, `summary`, sort
    order, `repr`, `n_restarts` count.
  - `TestObjectiveAlignment` — SA and BH both beat random baseline on
    `H_total`; negative weight reduces penalized metric; callable
    objective works; `n_restarts=4` best ≥ any single-restart result.


  `StructureGenerator.generate()`.

  `GenerationResult` is a `dataclass` that is fully list-compatible
  (supports indexing, iteration, `len()`, and boolean coercion) while also
  exposing per-run rejection metadata:

  | Attribute | Description |
  |---|---|
  | `structures` | `list[Structure]` — structures that passed all filters |
  | `n_attempted` | Total placement attempts |
  | `n_passed` | Structures that passed (equals `len(result)`) |
  | `n_rejected_parity` | Attempts rejected by charge/multiplicity parity check |
  | `n_rejected_filter` | Attempts rejected by metric filters |
  | `n_success_target` | The `n_success` value in effect (`None` if not set) |

  `result.summary()` returns a one-line diagnostic string.

  **Backward compatibility:** all code that treats the return value of
  `generate()` as a list — iteration, indexing, `len()`, `bool()` — works
  without modification.  The only breaking change is `isinstance(result,
  list)` returning `False`; use `isinstance(result, GenerationResult)` or
  `hasattr(result, "structures")` instead.

- **`warnings.warn` on silent-failure paths** — `stream()` now emits a
  `UserWarning` via Python's standard `warnings` module whenever:

  - any attempts are rejected by the charge/multiplicity parity check,
  - no structures pass the metric filters after all attempts are exhausted, or
  - the attempt budget is exhausted before `n_success` is reached.

  These warnings fire regardless of the `verbose` flag.  Previously, all
  such messages were printed to `stderr` only when `verbose=True`,
  making a silent empty-list return indistinguishable from a successful run
  in automated pipelines (ASE, high-throughput workflows).  A downstream
  `IndexError` or `AttributeError` on an empty list would then point to
  the wrong place in user code.

  Callers that want to suppress the warnings can use
  `warnings.filterwarnings("ignore", category=UserWarning, module="pasted")`.

- `GenerationResult` added to `pasted.__all__` and exported from the
  top-level `pasted` namespace.

- **11 new tests** in `tests/test_generator.py` covering
  `TestGenerationResult`: list-compatibility, indexing, `bool`, `summary`,
  `repr`, metadata count correctness, `UserWarning` on filter rejection,
  `UserWarning` on parity rejection, and no spurious warnings on clean runs.

- **`method="parallel_tempering"`** added to `StructureOptimizer`.

  Parallel Tempering (replica-exchange Monte Carlo) runs `n_replicas`
  independent Markov chains at a geometric temperature ladder from
  `T_end` (coldest, most selective) to `T_start` (hottest, most
  exploratory).  Every `pt_swap_interval` steps, adjacent replica pairs
  attempt a state exchange using the Metropolis criterion:

  ```
  ΔE = (β_k − β_{k+1}) × (f_{k+1} − f_k)
  accept with probability min(1, exp(ΔE))
  ```

  where β = 1/T and f is the objective value (higher is better).  Hot
  replicas cross energy barriers that trap SA; accepted swaps tunnel good
  structures from hot replicas down to the cold replica, improving both
  exploration and exploitation simultaneously.

  New parameters:

  | Parameter | Default | Description |
  |---|---|---|
  | `n_replicas` | `4` | Number of temperature replicas |
  | `pt_swap_interval` | `10` | Attempt replica exchange every N steps |

  `run()` returns an `OptimizationResult` containing the global best
  (tracked across all replicas and all steps) plus each replica's final
  state, sorted by objective value.  `n_restarts` launches independent PT
  runs and aggregates all results.

  Measured quality improvement over SA (n=10, C/N/O/P/S, `H_total − Q6`
  objective, 6-seed mean):

  | Method | wall time | H_total (↑) | Q6 (↓) |
  |---|---:|---:|---:|
  | SA steps=500, restarts=4 | 460 ms | 1.591 | 0.401 |
  | BH steps=200, restarts=4 | 187 ms | 1.597 | 0.420 |
  | **PT steps=200, rep=4, restarts=1** | **102 ms** | **1.685** | **0.403** |
  | **PT steps=500, rep=4, restarts=2** | 579 ms | **1.713** | **0.293** |

  PT at 102 ms matches SA-4restart quality at 460 ms.  PT's Q6 suppression
  (0.293) is markedly better than either SA or BH at equivalent wall time.

- **Parity-preserving composition move** (`_composition_move` Path 2).

  The replace fallback — triggered when all atoms are the same element and
  no atom-pair swap is possible — previously drew a replacement element
  uniformly from the full pool, which violated the charge/multiplicity
  parity constraint in up to **64 %** of calls when the pool contained a
  mix of odd-Z and even-Z elements.

  The new implementation uses the user's insight that swapping elements
  within the same Z-parity class preserves the electron-count parity:

  - **Same-Z-parity replace** (primary): replace atom `i` with an element
    whose atomic number has the same parity as `Z(atoms[i])`.  Net ΔZ is
    even → parity invariant.
  - **Dual opposite-parity replace** (fallback when only odd-Z elements
    differ): replace *two* atoms simultaneously with elements from the
    odd-Z pool so that each ΔZ is odd but the total ΔZ is even.

  Parity failure rate in the worst case (all-same composition, wide pool):
  **64 % → 0 %**.  Normal usage (mixed composition, typical element pools)
  was already near zero via the primary swap path and is unchanged.

- **Objective alignment verified** for `Q6`, `H_total`, `moran_I_chi`,
  and `charge_frustration`:

  | Objective | Baseline | Optimized (SA, n=10) |
  |---|---:|---:|
  | maximize `Q6` | mean 0.081, max 0.274 | **0.801** (+192 %) |
  | maximize `H_total` | mean 0.495 | **1.229** (+148 %) |
  | minimize `moran_I_chi` | mean +0.06 | **−2.27** |
  | maximize `charge_frustration` | mean 0.010 | **0.372** |

  Temperature schedule (SA) confirmed as exponential decay T_start → T_end.
  `n_restarts` correctly returns the global best across all independent runs.

- **10 new tests** in `tests/test_optimizer.py`:
  - `TestParallelTempering` — return type, best-first sort, mode label,
    geometric temperature ladder, `n_replicas` parameter, `repr`,
    bad-method error, PT improves over baseline, multi-restart accumulation.

## [0.1.14] - 2026-03-19

### Changed

- **`pdist` / `squareform` removed from `compute_all_metrics`.**

  The O(N^2) `scipy.spatial.distance.pdist` + `squareform` call that
  dominated `compute_all_metrics` at large N has been replaced throughout
  by O(N*k) local pair enumeration (k = mean neighbors within *cutoff*,
  roughly constant):

  | Path | N=5 000 | N=10 000 | Scaling |
  |---|---:|---:|---:|
  | v0.1.13 (`pdist` + `squareform`) | ~730 ms | ~2 880 ms | O(N^2) |
  | v0.1.14 (FlatCellList / cKDTree) | ~30 ms  | ~60 ms   | O(N*k) |

  All seven affected metrics (`H_spatial`, `RDF_dev`, `Q4/Q6/Q8`,
  `graph_lcc`, `graph_cc`, `ring_fraction`, `charge_frustration`,
  `moran_I_chi`) now operate on pairs within *cutoff* only, consistent
  with the locality assumption shared by the graph and Steinhardt metrics.

- **New C++ function `rdf_h_cpp(pts, cutoff, n_bins)`** added to
  `_graph_core.cpp` and exported from `pasted._ext`.

  Enumerates pairs within *cutoff* via `FlatCellList` in a single O(N*k)
  pass and returns `{"h_spatial": float, "rdf_dev": float}`.  Called by
  `compute_all_metrics` when `HAS_GRAPH` is `True`.

- **`compute_h_spatial` signature changed** from
  `(dists: ndarray, n_bins: int)` to `(pts: ndarray, cutoff: float, n_bins: int)`.

  The condensed `dists` array (O(N^2) elements) is no longer accepted.
  The Python fallback uses `scipy.spatial.cKDTree.query_pairs` for O(N*k)
  pair enumeration within *cutoff*.

- **`compute_rdf_deviation` signature changed** from
  `(pts: ndarray, dists: ndarray, n_bins: int)` to
  `(pts: ndarray, cutoff: float, n_bins: int)`.

  The histogram range is now `[0, cutoff]` instead of `[0, r_max]` where
  `r_max` was the maximum pairwise distance.  Values will differ from
  v0.1.13 for the same structure, but are now consistent with the local
  pair assumption used by all other metrics.

- **`compute_steinhardt_per_atom` and `compute_steinhardt` signatures
  changed**: the `dmat` parameter has been removed.

  The C++ path (`HAS_STEINHARDT`) never used `dmat`.  The Python fallback
  (`_steinhardt_per_atom_sparse`) now uses `scipy.spatial.cKDTree` for
  neighbor enumeration instead of indexing into a pre-built distance matrix.
  Both paths accept `(pts, l_values, cutoff)`.

- **`compute_angular_entropy`** (diagnostic, not in `ALL_METRICS`) now uses
  `scipy.spatial.cKDTree` instead of a full O(N^2) distance matrix.

- **Inconsistent docstrings fixed** throughout `_metrics.py`:

  - `compute_ring_fraction` and `compute_charge_frustration`: removed
    stale references to `cov_scale * (r_i + r_j)` bond detection; updated
    to describe the cutoff-based adjacency introduced in v0.1.13.
  - `compute_moran_I_chi`: corrected return-value description.
  - `_steinhardt_per_atom_sparse`: rewritten to reflect removal of `dmat`.
  - `compute_all_metrics`: documents removal of `pdist` / `squareform`.

- `pasted._ext.__init__`: `rdf_h_cpp` added to `__all__` and to the
  `_graph_core` import block.  `HAS_GRAPH = True` now implies both
  `graph_metrics_cpp` and `rdf_h_cpp` are available.

- `_graph_core.cpp` module docstring updated to v0.1.14; `rdf_h_cpp`
  binding and inline documentation added.

- `pyproject.toml`: version bumped to `0.1.14`.

---

## [0.1.13] - 2026-03-19

### Changed

- **`ring_fraction` and `charge_frustration` now use `cutoff` for adjacency
  instead of `cov_scale × (r_i + r_j)`.**

  Previously these metrics defined a bond as any pair satisfying
  `d_ij < cov_scale × (r_i + r_j)`.  Because `relax_positions` guarantees
  `d_ij >= cov_scale × (r_i + r_j)` for every pair on convergence, this
  criterion was *structurally never satisfied* in relaxed structures — both
  metrics returned 0.0 for every output of PASTED, carrying no information.

  **New definition:** a pair (i, j) is adjacent when `d_ij <= cutoff`,
  the same cutoff used by `graph_lcc`, `graph_cc`, and `moran_I_chi`.
  All five cutoff-based metrics now share a single unified adjacency.

  **Physical interpretation of the updated metrics:**

  - `ring_fraction` — fraction of atoms that belong to at least one cycle
    in the cutoff-adjacency graph.  A high value indicates that atoms are
    densely connected enough to form closed loops at the chosen interaction
    radius, reflecting structural compactness or clustering.
  - `charge_frustration` — variance of |Δχ| (absolute Pauling
    electronegativity difference) across all cutoff-adjacent pairs.
    High values indicate that each atom is surrounded by a mix of
    electronegative and electropositive neighbours — i.e. the local
    electrostatic environment is inconsistent, analogous to geometric
    frustration in spin systems.  Low values indicate compositionally
    homogeneous neighbourhoods.

  Both metrics now produce informative non-zero values for typical PASTED
  structures (N = 100, mixed elements, auto cutoff ~2.13 Å).

  **API change:** the `cov_scale` parameter of `compute_ring_fraction` and
  `compute_charge_frustration` has been renamed to `cutoff`.  The parameter
  was previously forwarded from `compute_all_metrics`; callers who pass
  keyword arguments need to update to `cutoff=...`.

- `compute_all_metrics` no longer forwards `cov_scale` to ring/charge
  functions; it now forwards `cutoff` instead.  `cov_scale` is retained in
  the `compute_all_metrics` signature for backward compatibility.

- `_graph_core.cpp` updated to build a single unified adjacency list
  (`d_ij <= cutoff`) shared by all five metrics, removing the separate
  `cov_scale`-based bond list.

- **README.md** fully rewritten to reflect the current feature set
  (v0.1.12+), including the `maxent` mode, `StructureOptimizer`, `n_success`,
  `moran_I_chi`, unified cutoff, noble gas EN values, and C++ acceleration
  flags (`HAS_GRAPH`).

- `pyproject.toml`: version bumped to `0.1.13`.

---

## [0.1.12] - 2026-03-19

### Added

- **`pasted._ext._graph_core`** — new C++17 extension that replaces four
  Python O(N²) metrics bottlenecks with a single O(N·k) FlatCellList pass
  (k = mean bonded-pair count per atom, approximately constant):

  | Metric | v0.1.11 Python (N=1000) | v0.1.12 C++ (N=1000) | Speedup |
  |---|---:|---:|---:|
  | `ring_fraction` | ~90 ms | — | — |
  | `charge_frustration` | ~88 ms | — | — |
  | `graph_lcc` / `graph_cc` | ~35 ms | — | — |
  | `moran_I_chi` (new) | n/a | — | — |
  | **metrics TOTAL** | **~419 ms** | **~17 ms** | **~25×** |

  All five metrics (`graph_lcc`, `graph_cc`, `ring_fraction`,
  `charge_frustration`, `moran_I_chi`) are computed in a single C++ call;
  the `FlatCellList`, bonded-pair adjacency list, and cutoff adjacency list
  are built only once per `compute_all_metrics` invocation.  A `HAS_GRAPH`
  flag in `pasted._ext` controls transparent fallback to the Python path
  when the extension is absent.

- **`moran_I_chi`** — new metric: Moran's I spatial autocorrelation for
  Pauling electronegativity, added to `ALL_METRICS` and exported from
  `pasted`:

  ```
  I = (N / W) * Σ_{i≠j} w_ij (χ_i − χ̄)(χ_j − χ̄) / Σ_i (χ_i − χ̄)²
  ```

  where w_ij = 1 when d_ij ≤ cutoff (step-function weight; uses the
  existing `cutoff` parameter — no new API parameter).

  Interpretation:
  - I ≈ 0 : random spatial arrangement of electronegativity (desired for
    disordered structures)
  - I > 0 : same-electronegativity atoms cluster spatially
  - I < 0 : alternating high/low electronegativity (NaCl-like order)

  Note: Moran's I is not bounded to [-1, 1] for sparse weight matrices.

- `HAS_GRAPH` flag added to `pasted._ext`.

- CLI `--filter` help text and `docs/cli.md` updated with a `moran_I_chi`
  example: `--filter "moran_I_chi:-0.1:0.1"` selects structures with
  spatially random electronegativity arrangement.

### Changed

- **Noble gas Pauling electronegativity values updated** in `_PAULING_EN`:

  | Element | Before | After | Rationale |
  |---|---|---|---|
  | He, Ne, Ar, Rn | 1.0 | 4.0 | changed — no stable compounds known |
  | **Kr** | 1.0 | **3.0** | KrF₂ known; Allen/Allred-Rochow scale estimate |
  | **Xe** | 1.0 | **2.6** | XeF₂/XeO₃ well characterised; literature estimate |

- `graph_metrics_cpp` now computes all five metrics in a single FlatCellList
  pass; the former separate `moran_I_chi_cpp` call has been inlined,
  halving the number of spatial-index builds per `compute_all_metrics` call.

- `compute_all_metrics` docstring no longer hardcodes the metric count;
  it now refers to `len(ALL_METRICS)` to avoid future update churn.

- `setup.py`: fourth `Pybind11Extension` entry added for `_graph_core`.

- `pyproject.toml`: version bumped to `0.1.12`.

### Removed

- **`bond_strain_rms`** removed from `ALL_METRICS`, `compute_all_metrics`,
  the public API (`pasted.__init__`), and `_metrics.py` entirely.

  Rationale: `relax_positions` guarantees `d_ij >= cov_scale * (r_i + r_j)`
  for every pair on convergence, so `bond_strain_rms` is structurally zero
  under normal usage (`cov_scale = 1.0`) and carries no information about
  the generated structures.

  Migration: remove `"bond_strain_rms"` from any `--filter` or `objective`
  dict.  `ALL_METRICS` now has **13 keys** (12 after removing
  `bond_strain_rms`, then back to 13 after adding `moran_I_chi`).

---

## [0.1.11] - 2026-03-19

### Changed

- **`_relax_core` solver replaced: Gauss-Seidel → L-BFGS.**
  The per-cycle `check_and_push` Gauss-Seidel loop in `_relax.cpp` has been
  replaced by a global L-BFGS minimization of the harmonic steric-clash
  penalty energy:

  ```
  E = Σ_{i<j}  ½ · max(0,  cov_scale·(rᵢ + rⱼ) − dᵢⱼ)²
  ```

  The gradient is computed analytically; pair enumeration still uses
  `FlatCellList` for N ≥ 64 (O(N) per evaluation) and an O(N²) full-pair
  loop for N < 64 — identical to v0.1.10.

  **Additional fixes (applied during test validation):**
  - `ENERGY_TOL` tightened from `1e-6` to `1e-12`.  The convergence
    criterion is on the *total* penalty energy, so `1e-6` permitted
    per-pair residual overlaps up to √(2×10⁻⁶) ≈ 1.4×10⁻³ Å — too
    coarse for the existing test suite (tolerance 1e-5 Å).  `1e-12`
    bounds per-pair residuals to ≤ 1.4×10⁻⁶ Å.
  - Jitter scope narrowed from *all* coordinates to *coincident-pair*
    atoms only (d < 1e-10 Å), matching the v0.1.10 GS behaviour.
    The unconditional jitter made `relax_positions(seed=None)` non-
    deterministic for normal structures, breaking the optimizer
    reproducibility test.

  **Key behavioral differences vs v0.1.10:**

  | | v0.1.10 (Gauss-Seidel) | v0.1.11 (L-BFGS) |
  |---|---|---|
  | Convergence on dense random structures | 0 % (1500 cycles) | 100 % |
  | N = 5000, normal density | 2.28 s | **0.044 s** (~52×) |
  | N = 5000, highly dense packing | 3.04 s | **0.084 s** (~36×) |
  | External dependencies | none | none |
  | `setup.py` changes required | — | **none** |

  The L-BFGS implementation (history depth m = 7, Armijo backtracking line
  search) is written entirely in C++17 standard library — no Eigen, no
  OpenMP, no new build-time dependencies.  A thin `Vec` struct backed by
  `std::vector<double>` provides the required linear algebra; `-O3` produces
  code equivalent to an Eigen-based implementation.

  `converged = True` when E < 1 × 10⁻⁶ (all overlaps resolved).

  A one-time pre-perturbation jitter (σ ≈ 1 × 10⁻⁶ × max_r, seeded by
  the `seed` parameter) prevents zero-gradient singularities at exactly
  coincident atom positions.  The perturbation is negligible on the final
  geometry (~3 × 10⁻⁸ Å for hydrogen).

- `max_cycles` semantics for `relax_positions` (C++ path only):
  Previously counted Gauss-Seidel sweeps; now counts L-BFGS outer
  iterations.  The Python-side default `relax_cycles = 1500` is unchanged
  and backward-compatible — L-BFGS exits early when E < 1 × 10⁻⁶, so
  the limit is rarely reached.

- `seed` semantics for `relax_positions` (C++ path only):
  Previously seeded the per-push random direction for coincident atoms.
  Now seeds the one-time pre-perturbation jitter.  Downstream callers are
  unaffected.

- `pyproject.toml`: version bumped to `0.1.11`.

---

## [0.1.10] - 2026-03-18

### Added

- **`pasted._ext._steinhardt_core`** — new C++17 extension for Steinhardt
  Q_l computation, replacing the dense O(N²) Python/scipy path with a
  sparse O(N·k) algorithm (k = mean neighbor count).

  | Path | N=2000 | Speed-up vs dense Python |
  |---|:---:|:---:|
  | Dense Python (original) | ~35 s | 1× |
  | Sparse Python fallback | ~0.21 s | ~164× |
  | C++ (`_steinhardt_core`) | ~17 ms | **~2 000×** |

  The extension uses a `FlatCellList` spatial index (same pattern as
  `_relax_core`) for neighbor finding, and evaluates spherical harmonics
  via the standard associated Legendre polynomial three-term recurrence —
  no scipy call inside the hot loop.  The symmetry `|Y_l^{-m}|² = |Y_l^m|²`
  halves the number of harmonic evaluations by computing only m = 0..l.

  When the extension is absent, a sparse Python/NumPy fallback
  (`_steinhardt_per_atom_sparse`) provides the same O(N·k) complexity
  using `np.bincount` for accumulation.  The public
  `compute_steinhardt_per_atom` function dispatches transparently.

  `HAS_STEINHARDT` flag added to `pasted._ext`.

### Changed

- `setup.py`: third `Pybind11Extension` entry added for `_steinhardt_core`.
- `pyproject.toml`: version bumped to `0.1.10`.

---

## [0.1.9] - 2026-03-18

### Added

- **Three MM-level structural descriptors** added to `ALL_METRICS` (and
  therefore available as `--filter` targets on the CLI):

  | Metric | Description |
  |---|---|
  | `bond_strain_rms` | RMS relative deviation of bonded-pair distances from their Pyykkö ideal lengths |
  | `ring_fraction` | Fraction of atoms that belong to at least one ring, detected via Union-Find spanning-tree construction |
  | `charge_frustration` | Variance of Pauling electronegativity differences across bonded pairs |

  Bond detection uses the same `cov_scale × (r_i + r_j)` threshold as
  `relax_positions`, keeping the definition of a "bond" consistent across
  placement, relaxation, and metric computation.

  Example usage::

      from pasted import generate

      structs = generate(
          n_atoms=14, charge=0, mult=1,
          mode="chain", elements="6,7,8,1",
          n_samples=50, seed=0,
          filters=["bond_strain_rms:-:0.15", "ring_fraction:-:0.3"],
      )

- **Pauling electronegativity table** (`pasted._atoms._PAULING_EN`) covering
  Z = 1–106 (Pauling 1960 / IUPAC 2016).  Noble gases and elements without
  a literature value return the module-level constant `PAULING_EN_FALLBACK`
  (1.0).  The public accessor `pauling_electronegativity(sym)` is exported
  from the top-level `pasted` namespace.

- `compute_all_metrics()` now accepts an optional `cov_scale: float = 1.0`
  keyword argument, forwarded to the three new MM-level descriptors.
  Existing call sites without `cov_scale` are unaffected (default preserved).

- `pasted.PAULING_EN_FALLBACK`, `pasted.pauling_electronegativity`,
  `pasted.compute_bond_strain_rms`, `pasted.compute_ring_fraction`, and
  `pasted.compute_charge_frustration` added to the public API and `__all__`.

- **21 new tests** in `tests/test_metrics.py` covering
  `TestComputeBondStrainRms` (5 tests), `TestComputeRingFraction` (5 tests),
  `TestComputeChargeFrustration` (5 tests), and updated integration tests
  for `compute_all_metrics` (4 tests) and `passes_filters` (1 test).

### Changed

- `ALL_METRICS` expanded from 10 to 13 keys.
- `compute_all_metrics` docstring updated: "ten" → "thirteen".
- `pyproject.toml`: version bumped to `0.1.9`.

---

## [0.1.8] - 2026-03-18

### Added

- **Python 3.13 officially supported.**  `cp313` wheels are now built and
  tested in CI.  The C++ extensions (`_relax_core`, `_maxent_core`) compile
  and load correctly under Python 3.13.

### Fixed

- `__version__` is now derived dynamically from package metadata via
  `importlib.metadata.version("pasted")` instead of being hardcoded in
  `__init__.py`.  This eliminates the version skew that caused
  `pasted.__version__` to report `"0.1.4"` even after upgrading to a
  newer release.  Falls back to `"unknown"` when the package is not
  installed (e.g., running directly from the source tree without `pip install`).

### Changed

- `pyproject.toml`: version bumped to `0.1.8`.

---

## [0.1.7] - 2026-03-18

### Added

- **`n_success` parameter** for `StructureGenerator` / `generate()`
  (`n_success: int | None = None`, CLI: `--n-success N`).

  Generation stops as soon as *N* structures have passed all filters,
  without exhausting the full `n_samples` attempt budget.

  | `n_samples` | `n_success` | Behavior |
  |:---:|:---:|---|
  | > 0 | `None` | Original behavior: attempt exactly `n_samples` times |
  | > 0 | N | Stop at N successes **or** `n_samples` attempts, whichever comes first |
  | 0 | N | Unlimited attempts; stop only when N structures have passed |

  `n_samples=0` without `n_success` raises `ValueError` to prevent
  accidental infinite loops.  If `n_samples` is exhausted before
  `n_success` is reached, the structures collected so far are returned
  with a warning — never an empty result or an exception.

- **`StructureGenerator.stream()`** — yields each passing structure
  immediately rather than collecting all results into a list first.

  - **Incremental file output**: each structure is written to disk the
    moment it passes, so a `Ctrl-C` mid-run still produces valid XYZ
    output up to that point.
  - **Early termination**: combined with `n_success`, the caller receives
    results without waiting for the full attempt budget to be exhausted.

  `generate()` now delegates to `stream()` internally — behavior and
  return type are unchanged for existing callers.  The CLI uses `stream()`
  and appends each structure to the output file immediately on PASS.

### Changed

- `StructureGenerator.__repr__` now includes `n_success`.
- CLI `--n-samples` help text updated to document the `0 = unlimited`
  semantics.
- `pyproject.toml`: version bumped to `0.1.7`.

---

## [0.1.6] - 2026-03-18

### Added

- **Cell List spatial partitioning in `_relax.cpp` and `_maxent.cpp`.**

  Both C++ extension modules now use a flat 3-D Cell List to restrict
  neighbor searches to a 27-cell neighborhood instead of scanning all
  atom pairs.  A linked-list style flat `vector<int>` grid (`FlatCellList`)
  is rebuilt once per relaxation cycle, avoiding the per-cycle heap
  allocation overhead of an `unordered_map`-based grid.

  Strategy is selected automatically based on atom count:

  | N | `_relax_core` | `_maxent_core` |
  |:---:|:---:|:---:|
  | < 64 | O(N²) full-pair loop | O(N³) full-pair |
  | ≥ 64 | O(N) Cell List | O(N²) Cell List |

  Cell size is computed automatically — `cov_scale × 2 × max(radii)` for
  `_relax_core`; `cutoff` for `_maxent_core` — with no new API parameters.

  Measured speed-ups vs pure-Python/NumPy fallback:

  `relax_positions`:

  | N | Python (ms) | C++ (ms) | Speed-up |
  |:---:|:---:|:---:|:---:|
  | 20 | 2.1 | 0.09 | 22× |
  | 200 | 87 | 13 | 6.6× |
  | 500 | 544 | 43 | 12.7× |
  | 1000 | 2237 | 113 | **19.8×** |

  `angular_repulsion_gradient`:

  | N | Python (ms) | C++ (ms) | Speed-up |
  |:---:|:---:|:---:|:---:|
  | 30 | 10.8 | 0.07 | 153× |
  | 100 | 154 | 3.0 | 52× |
  | 200 | 881 | 19 | **46×** |

- **`chain_bias` parameter** for `place_chain` / `StructureGenerator` /
  `generate()` (`chain_bias: float = 0.0`, CLI: `--chain-bias`).

  The direction of the first bond placed becomes a global *bias axis*.
  Every subsequent step direction is blended toward that axis before
  normalization::

      d_biased = d + axis * chain_bias
      d_final  = d_biased / ||d_biased||

  Effect on `shape_aniso ≥ 0.5` rate (n = 20, branch_prob = 0.0):

  | `chain_bias` | mean shape_aniso | ≥ 0.5 rate |
  |:---:|:---:|:---:|
  | 0.0 (default) | 0.40 | 33% |
  | 0.3 | 0.55 | 63% |
  | 0.6 | 0.74 | 92% |
  | 1.0 | 0.89 | 100% |

  Default is `0.0` — fully backward-compatible.

### Fixed

- Distance violation check in `_relax.cpp` changed from `d² >= thr²` to
  `d >= thr` throughout.  The squared comparison caused atoms sitting
  exactly at the threshold distance to be re-flagged as violating in the
  following cycle due to floating-point rounding, preventing convergence.

### Changed

- `_relax.cpp`: Cell List threshold raised from 32 to 64 after benchmarking
  showed that `unordered_map`-based grid reconstruction at N ≈ 32–63 is
  slower than the full-pair loop.
- `pyproject.toml`: version bumped to `0.1.6`.

---

## [0.1.5] - 2026-03-18

### Added

- **`src/pasted/_ext/` sub-package** — C++ extensions reorganized into
  separate source files by function:
  - `_relax.cpp` → `_ext._relax_core`: distance constraint relaxation loop
    (used by all placement modes).
  - `_maxent.cpp` → `_ext._maxent_core`: angular repulsion gradient
    (`maxent` mode only).
  - `_ext/__init__.py`: exposes independent `HAS_RELAX` / `HAS_MAXENT`
    flags so that a build failure in one module does not disable the other.

- **Optional C++ extension** (`pasted._ext`, built via `pybind11`).
  When a C++17 toolchain is present at install time, two inner-loop
  hotspots are compiled to native code.  When absent, pure-Python/NumPy
  fallbacks are used transparently — no user-facing API change.
  - `relax_positions`: per-cycle pair-repulsion loop rewritten as a tight
    C++ double loop, eliminating the `(n, n, 3)` NumPy broadcast diff
    array allocated on every iteration.  Typical speed-up: 5–20× for
    10–100 atoms.
  - `_angular_repulsion_gradient`: O(N³) Python double `for` loop replaced
    by a cache-friendly C++ loop.  Speed-up: 20–50× for `maxent` mode.

- **`seed` parameter for `relax_positions`** (`seed: int | None = None`).
  The RNG used for the coincident-atom edge case (distance < 1e-10 Å) is
  now seeded deterministically when a value is provided.
  `StructureGenerator` automatically forwards its master seed.

- **`seed` parameter for `place_maxent`** — threaded through to the two
  internal `relax_positions` calls.

### Fixed

- Pure-Python fallback for `relax_positions` now creates a single
  `np.random.default_rng(seed)` instance before the loop instead of calling
  `np.random.default_rng()` (unseeded) on every coincident-atom hit,
  fixing a latent non-reproducibility bug.

### Changed

- `pyproject.toml`: `pybind11>=2.12` added to `[build-system].requires`
  and `[project.optional-dependencies].dev`.
- `setup.py` (new file): declares two `Pybind11Extension` entries, one per
  C++ source file.
- `pyproject.toml`: version bumped to `0.1.5`.

---

## [0.1.4] - 2026-03-17

### Added

- Documentation of `maxent` mode algorithmic behavior, numerical stability
  measures, and parameter tuning guidelines.
- Clarification that `compute_angular_entropy` is a placement-quality
  diagnostic excluded from `ALL_METRICS` and XYZ comment lines.

---

## [0.1.3] - 2026-03-17

### Added

- **`--mode maxent`** — maximum-entropy placement mode.  Atoms are
  initialized at random, then iteratively repositioned by gradient descent
  on an angular repulsion potential so that each atom's neighbor directions
  become as uniformly distributed over the sphere as the Pyykkö distance
  constraints allow.  Implements the constrained-maximum-entropy solution to
  `max S = −∫ p(Ω) ln p(Ω) dΩ` subject to `d_ij ≥ cov_scale·(r_i + r_j)`.
- `place_maxent(atoms, region, cov_scale, rng, ...)` — low-level placement
  function, exported from the public API.
- `compute_angular_entropy(positions, cutoff)` — diagnostic metric: mean
  per-atom Shannon entropy of neighbor direction distributions.  Not
  included in `ALL_METRICS` or XYZ comment lines.
- `_angular_repulsion_gradient(pts, cutoff)` — internal NumPy gradient of
  the angular repulsion potential `U = Σ 1/(1 − cos θ + ε)`.
- Three new CLI flags: `--maxent-steps` (default 300), `--maxent-lr`
  (default 0.05), `--maxent-cutoff-scale` (default 2.5).
- `tests/test_maxent.py` — 15 tests covering gradient, placement, entropy
  metric, and generator integration.

### Changed

- `StructureGenerator` now accepts `mode="maxent"`.
- `__init__.py` exports: `place_maxent`, `compute_angular_entropy`.

---

## [0.1.2] - 2026-03-17

### Added

- **`StructureOptimizer`** — objective-based structure optimization that
  maximizes a user-defined disorder metric instead of sampling randomly.
  - Two methods: `"annealing"` (Simulated Annealing with exponential
    cooling) and `"basin_hopping"` (Metropolis with per-step relaxation).
  - Fragment coordinate move: atoms with local Q6 above `frag_threshold`
    are preferentially displaced to break accidentally ordered regions.
  - Composition move: element types of two atoms are swapped to explore
    composition space alongside geometry.
- `parse_objective_spec(["METRIC:WEIGHT", ...])` — utility for converting
  CLI strings to a weight dict.
- `compute_steinhardt_per_atom` — public function returning per-atom Q_l
  arrays of shape `(n,)`, used internally by the optimizer.
- `--optimize` CLI flag enabling optimization mode while preserving full
  backward compatibility with sampling mode.
- `tests/test_optimizer.py` — 25 tests covering helpers, construction, and
  all `run()` paths.

### Changed

- `compute_steinhardt` refactored to delegate to
  `compute_steinhardt_per_atom`, eliminating code duplication.
- `__init__.py` exports: `StructureOptimizer`, `parse_objective_spec`,
  `compute_steinhardt_per_atom`.

---

## [0.1.1] - 2026-03-17

### Added

- `pasted.py` — direct-run entry point at the project root, enabling
  `python pasted.py` without a prior `pip install`.
- `conftest.py` at the project root to prevent `pasted.py` from shadowing
  the `src/pasted/` package during `pytest` collection.
- CI badges (status, PyPI version, Python versions, License) in `README.md`.
- `.github/workflows/ci.yml` — GitHub Actions workflow: `lint` (ruff),
  `test` (Python 3.10/3.11/3.12 matrix), `typecheck` (mypy), `build`
  (sdist + wheel artifact).

### Fixed

- `ruff format --check` removed from CI lint job to prevent version-skew
  failures between environments.

### Changed

- CI actions updated to Node.js 24 compatible versions:
  `actions/checkout@v4` → `@v5`, `actions/setup-python@v5` → `@v6`.
- `_metrics.py`: `_sph_harm` wrappers now return `np.ndarray` via
  `np.asarray()`, eliminating `no-any-return` mypy errors.
- `pyproject.toml`: added `[tool.mypy.overrides]` for `pasted._metrics` to
  suppress `warn_unused_ignores` on the `scipy < 1.15` compatibility branch.
- `pyproject.toml`: pytest runs with `--import-mode=importlib`.
- `license` field updated to SPDX string format (`"MIT"`) per PEP 639.

---

## [0.1.0] - 2026-03-17

### Added

- Initial release.
- `StructureGenerator` class API and `generate()` functional wrapper.
- `Structure` dataclass with `.to_xyz()` and `.write_xyz()` helpers.
- Three placement modes: `gas`, `chain`, `shell`.
- Ten disorder metrics: `H_atom`, `H_spatial`, `H_total`, `RDF_dev`,
  `shape_aniso`, `Q4`, `Q6`, `Q8`, `graph_lcc`, `graph_cc`.
- `pasted` CLI entry-point.
- `src/` layout with separated concerns:
  `_atoms`, `_placement`, `_metrics`, `_io`, `_generator`, `cli`.
- `pyproject.toml` with `ruff`, `mypy`, and `pytest` dev extras.
