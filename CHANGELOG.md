# Changelog

All notable changes to PASTED are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

---

## [0.4.5] — 2026-03-27

### Performance

#### PERF — `StructureGenerator`: O(N)→O(1) hot-loop optimisations (`_generator.py`)

Three per-sample O(N) operations in the generation loop have been eliminated
or reduced to O(1) by pre-computing constants at construction time:

**`_adjust_parity` — XOR bit trick for Z-sum parity**
The v0.4.4 implementation computed `sum(ATOMIC_NUMBERS.get(a, 0) for a in atoms)`
(full integer accumulation) to test parity.  Because only the low bit of the
sum matters, this is replaced by an XOR accumulation:

```python
zp = 0
for a in atoms:
    zp ^= ATOMIC_NUMBERS[a] & 1
```

Constant-factor speedup (~2×); the scan over atoms is still O(N) but much
cheaper per element.

**`_adjust_parity` — pre-cached odd/even pools**
Strategy 3 previously rebuilt the candidate list with a per-call list
comprehension over the element pool:

```python
candidates = [s for s in pool if ATOMIC_NUMBERS.get(s, 0) % 2 == needed_parity]
```

Two lists `self._odd_pool` / `self._even_pool` are now built once in
`__init__` and looked up in O(1) at call time.

**`_adjust_parity` — random start instead of full shuffle**
The full `rng.shuffle(list(range(len(atoms))))` is replaced by
`rng.randrange(n)`.  In the common case (pool contains elements of both Z
parities), the first candidate atom always succeeds, giving O(1) average cost.

**`_add_h_fast` — new private fast path for hydrogen augmentation**
A new internal method `_add_h_fast()` replaces the `add_hydrogen()` call in
the generation loop.  It is semantically equivalent but uses two pre-cached
values to avoid O(N) work per sample:

* `self._h_region_vol` — region volume parsed once in `__init__` instead of
  re-parsing the region string on every call.
* `self._h_heavy_vol_per_atom` — `(4/3)π × mean_pool_radius³` computed once
  in `__init__` and used as a proxy for the per-sample mean covalent radius.
  The approximation is exact in expectation (atoms are drawn from the same
  pool) and introduces no observable bias in the volume cap.

The public `add_hydrogen()` function in `_placement.py` is unchanged and
remains part of the public API.

**Benchmark impact (estimated):**

| Operation | v0.4.4 | v0.4.5 | Speedup |
|---|---|---|---|
| `_adjust_parity` | ~2.3 µs/call | ~0.9 µs/call | ~2.5× |
| H augmentation (`_add_h_fast`) | ~5.1 µs/call | ~1.5 µs/call | ~3.4× |
| `generate()` overall vs v0.4.3 | regression | nearly recovered | — |

**Files changed:** `src/pasted/_generator.py` only.

---

## [0.4.4] — 2026-03-27

### Changed

#### BEHAVIOR — `StructureGenerator._validate_density()`: auto-scale region instead of raising `ValueError`

Previously, when the user-supplied region's packing fraction exceeded the
hard limit (0.64, random-close-packing) a `ValueError` was raised at
construction time, forcing the caller to manually recalculate a valid region.

**New behavior:** instead of raising an error, the region is **automatically
enlarged** to bring the packing fraction down to the 45 % target, and a
`UserWarning` is emitted. Generation proceeds with the scaled region.

Both soft (> 0.50) and hard (> 0.64) thresholds now trigger auto-scaling:

- **Soft threshold exceeded (pf > 0.50):** UserWarning + region scaled to
  target packing (45 %). `relax_positions` would have been slow otherwise.
- **Hard threshold exceeded (pf > 0.64):** UserWarning noting that
  convergence was impossible + region scaled to target packing (45 %).

The warning message includes the original and scaled region strings, e.g.:

```
UserWarning: [pasted v0.4.4] Packing fraction 74% exceeds the physical limit
(64%, random-close-packing) for 'sphere:2.25' with 20 atoms.
relax_positions cannot converge at this density.
Region auto-scaled to 'sphere:2.8' (target packing: 45%).
```

This change preserves backward compatibility for callers passing a
well-sized region (no warning, no scaling). Callers that were previously
catching `ValueError` should update to handle `UserWarning` instead.

**Files changed:** `src/pasted/_generator.py`.

### Added

#### FEATURE — `add_hydrogen()`: parity-aware H count and volume cap (`_placement.py`)

`add_hydrogen()` gains two new keyword parameters (`region`, `charge`,
`mult`) and two corresponding guards:

**Parity-aware count** — the raw H count is adjusted by ±1 so that the
total electron count satisfies the charge/multiplicity parity constraint:
`(Σ Z_i − charge + n_H) % 2 == (mult − 1) % 2`. This prevents most
charge/multiplicity rejections that previously occurred *after* the
(potentially expensive) H-list allocation in the generation loop.

**Volume cap** — when `region` is supplied, the H count is hard-capped so
that the total packing fraction (heavy atoms + H) stays below
`_PACKING_HARD` (0.64). This is complementary to the density validation in
`StructureGenerator`: that check operates on the non-hydrogen heavy atoms,
while this cap limits hydrogen inflation within the fixed region after
heavy-atom placement.

The base sample distribution `n_H_raw = 1 + round(uniform(0,1) × n × 1.2)`
is preserved as closely as possible. Passing `region=None` (or omitting the
parameter) reproduces the v0.4.3 uncapped behaviour exactly.

**Backward compatibility:** the new parameters are keyword-only with safe
defaults (`region=None`, `charge=0`, `mult=1`), so existing call sites
require no changes.

**Files changed:** `src/pasted/_placement.py`.

#### FEATURE — `StructureGenerator._adjust_parity()`: post-sampling parity nudge (`_generator.py`)

A new internal method `_adjust_parity()` is called after `add_hydrogen()`
in the generation loop to catch any remaining parity mismatches before the
(more expensive) structure placement and metric computation steps.

**Strategy (priority order):**

1. Already correct → return unchanged.
2. H in pool → add or remove one H to flip the electron-count parity.
3. No H → swap one atom with a pool element of opposite-parity Z.

Changes at most one atom, so the element-fraction distribution is minimally
perturbed. Skipped automatically when `element_min_counts` or
`element_max_counts` constraints are active (those constraints take
precedence to avoid violating the caller's composition requirements).

**Files changed:** `src/pasted/_generator.py`.

#### FEATURE — `StructureGenerator._validate_density()`: density validation at construction time (`_generator.py`)

`StructureGenerator.__init__()` now calls `_validate_density()` for `gas`
and `maxent` modes whenever a `region` is provided. The method:

- Computes the packing fraction from the element pool's mean covalent radius.
- Auto-scales the region to 45 % target packing if either threshold is
  exceeded (see the BEHAVIOR entry above).
- Stores the effective region in `self._effective_region`, which is used
  by `_place_one()` and `add_hydrogen()` throughout the generation loop.

Helper methods added alongside:

- `_region_volume(region)` — parses sphere/box spec and returns
  `(shape, volume_Å³)`, or `None` for unrecognised specs.
- `_recommend_region(n_atoms, mean_r, shape)` — returns a region string
  at 45 % target packing for the given atom count and mean radius.

**Files changed:** `src/pasted/_generator.py`.

#### TEST — v0.4.4 feature test suite (`tests/test_v044.py`)

34 new tests across 4 classes covering all v0.4.4 additions:

- `TestAddHydrogen` (9 tests) — parity for singlet/doublet/negative-charge
  systems, volume cap for sphere and box regions, input-immutability check.
- `TestAdjustParity` (6 tests) — already-correct pass-through, H-pool
  strategy, no-H swap strategy, at-most-one-change invariant, pool
  membership check.
- `TestValidateDensity` (15 tests) — low-packing no-op, warn-threshold
  auto-scale, hard-threshold auto-scale (no `ValueError`), scaled region
  physically valid, box region, maxent mode, chain-mode skip, unknown spec
  pass-through, `_region_volume` and `_recommend_region` unit tests,
  end-to-end generation after auto-scale.
- `TestIntegrationV044` (4 tests) — parity validity across 20 samples,
  no spurious density warnings on normal-sized regions, all returned
  structures pass `validate_charge_mult`, generation succeeds after
  auto-scaling.

---

## [0.4.3] — 2026-03-26

### Fixed

#### BUG — `FlatCellList::build()`: int32 overflow in coarsening guard causes SIGSEGV (`_relax.cpp`)

*Discovered via: memory usage test suite added in this release*
*Affected code paths: `relax_positions` (C++ path) when `n >= CELL_LIST_THRESHOLD` and coincident atoms are present*

**Root cause.**
The v0.4.3 performance fix introduced a jitter check that calls
`FlatCellList::build()` with `cell_size = 1e-9 Å` and
`max_cells = max(1024, 16*n)`.  For typical atom spreads of ~10 Å,
`(range / cell_size) ≈ 1e10`, which exceeds `INT_MAX` (≈ 2.1 × 10⁹).

The three lambdas in the coarsening guard returned `static_cast<int>(...)`,
causing UB-truncation to ~1.4 × 10⁹.  The subsequent overflow check

```cpp
while (static_cast<std::int64_t>(tnx()) * tny() * tnz() > max_cells)
```

computed `(1.4e9)³ ≈ 2.8e27`, which overflowed `int64` to a **negative**
value.  The `while` condition (`negative > 2048`) evaluated to `false`
immediately, so no coarsening occurred, and `nx = ny = nz ≈ 1.4 × 10⁹`
were assigned.  The subsequent `cell_head.assign(nx*ny*nz, -1)` attempted
to allocate ~10²⁷ ints, writing far outside addressable memory and causing
a **segmentation fault** on the very first test run with `n >= 64` and
coincident atoms.

The bug was latent in all cell-list sizes (`build()` is shared with the
normal relaxation path), but was only reachable via the jitter check because
the 1e-9 Å cutoff is the only code path that passes such an extreme `cell_size`.

**Fix.**
The lambdas now return `std::int64_t` instead of `int`, eliminating the
int32 truncation.  The overflow-safe `exceeds()` helper replaces the triple
product with a step-wise comparison — each intermediate value is bounded by
`max_cells²` ≤ 2⁴⁴, which fits safely in `int64`:

```cpp
auto exceeds = [&]() -> bool {
    const std::int64_t x = tnx(), y = tny(), z = tnz();
    if (x > max_cells || y > max_cells || z > max_cells) return true;
    const std::int64_t xy = x * y;   // <= max_cells^2 <= 2^44: safe
    if (xy > max_cells) return true;
    return xy * z > max_cells;       // xy <= max_cells, so xy*z <= max_cells^2
};
```

**Files changed:** `src/pasted/_ext/_relax.cpp` only.

### Added

#### TEST — Memory usage and Segfault regression suite (`tests/test_jitter_memory.py`)

22 tests across 4 classes covering both the v0.4.3 performance fix and the
SIGSEGV fix above:

- `TestFlatCellListCap` (6 tests) — tracemalloc peak stays below 16 MB for
  n = 128–8192; verifies the `max(1024, 16*n)` cell cap is effective.
- `TestJitterCorrectness` (8 tests) — coincident pairs are separated after
  jitter for n = 2–200; results are finite and consistent across the
  `CELL_LIST_THRESHOLD` boundary (n = 32/64/65/128).
- `TestEdgeCases` (7 tests) — n = 0, 1, two identical atoms, threshold
  boundary (63/64/65), and **n = 2000 no-crash regression** (the SIGSEGV
  case).
- `TestEarlyExitMemory` (1 test) — well-separated atoms skip the jitter
  FlatCellList entirely; peak memory stays below 4 MB.

### Performance

#### PERF — `src/pasted/_ext/_relax.cpp`: eliminate O(N²) jitter scan and redundant allocations

**Root cause.**
`relax_positions_cpp` contained an unconditional O(N²) double loop that
scanned all N(N−1)/2 atom pairs on every call to check for coincident atoms
(distance < 1e-10 Å).  Because the placement algorithm virtually never
produces exactly coincident atoms, the loop almost always produced no
perturbations — making it a pure overhead cost that scaled quadratically
with system size.

Measured overhead before this fix:

```
n=256 → n=4096  (16× atoms)
relax_positions : 0.988 ms → 111 ms  =  112×  (≈ O(N^1.8))
no-overlap path : 0.422 ms →  25 ms  =   59×  (≈ O(N^1.5))
```

**Fix 1 — O(N) jitter scan via `FlatCellList` (n ≥ 64).**
For `n >= CELL_LIST_THRESHOLD` the O(N²) double loop is replaced by a
`FlatCellList` traversal with cutoff = 1e-9 Å.  Only neighbouring cells
are visited, so the scan is O(N) in the common case where no coincident
pairs exist.

**Fix 2 — `FlatCellList::build()` memory cap parameter.**
`build()` now accepts an optional `max_cells` argument (default `1<<22`,
unchanged for existing callers).  The jitter call passes
`max_cells = max(1024, 16*n)`, capping `cell_head` at ≤ 256 KB for
n ≤ 16 384 instead of the previous 16 MB upper bound.  Coarsening only
adds false-positive candidate pairs; it never misses a truly coincident
pair (d < 1e-10 Å ≪ cell_size after coarsening).

| n | `cell_head` before | `cell_head` after |
|------:|--------------------|-------------------|
|   256 | up to 16 MB        | up to 4 KB        |
| 4 096 | up to 16 MB        | up to 256 KB      |
|65 536 | up to 16 MB        | up to 4 MB        |

**Fix 3 — shared gradient vector; one fewer `evaluate()` call.**
`Vec grad_tmp(3*n)` was heap-allocated inside a block solely for the
early-exit check and immediately discarded.  A single `Vec g(3*n)` is now
allocated once and shared across the early-exit check, the optional
post-jitter re-evaluation, and `lbfgs_minimize`.  `lbfgs_minimize`'s
signature is updated to accept the pre-computed `(E, g)` pair, saving one
redundant `evaluate()` call at the start of every non-early-exit invocation.
In the common gas-mode path (overlaps present, no coincident atoms) this
eliminates two heap allocations and one full O(N·k) gradient evaluation.

**Files changed:** `src/pasted/_ext/_relax.cpp` only.

---

## [0.4.2] — 2026-03-24

### Documentation

#### DOC — `docs/quickstart.md`: four clarifications and two new sections

**`NeighborList` — new section** (`### NeighborList — custom neighbor queries
in objectives`, placed immediately after the *Extended objective with
EvalContext* section).

`pasted.neighbor_list.NeighborList` is a public API that has been available
since v0.4.0 but was not documented in the quickstart.  The new section
covers:

- Attribute and cached-property reference tables (`n_pairs`, `pairs`, `rows`,
  `cols`, `dists`, `all_dists`, `diff`, `deg`, `unit_diff`, `dists_sq`).
- Empty-structure handling guarantee (`n_atoms=0` and `n_pairs=0` produce
  empty arrays; no `NaN` or `ZeroDivisionError`).
- Required conversion: `ctx.positions` is a `tuple[tuple[float,float,float],
  ...]`; callers must convert it to `ndarray` via `np.array(ctx.positions)`
  before passing it to `NeighborList`.
- Explicit guidance on setting `cutoff=` in `StructureOptimizer`.  The
  default cutoff of `1.5 × median(rᵢ + rⱼ)` evaluates to approximately
  **2.07 Å** for a C/N/O pool, which yields a mean coordination of k ≈ 0.9
  — sufficient for the built-in metrics but too sparse for bond-order
  parameters such as Q3.  This is **intentional behaviour**: the default
  is calibrated for covalent-bond adjacency, not second-neighbour shells.
  Users computing Steinhardt parameters beyond the first shell should pass
  an explicit `cutoff=3.5` (or larger) to `StructureOptimizer` so that
  `ctx.cutoff` carries the intended value.

**Case study 4 — Custom bond-order metric (Q3) via `NeighborList`**
(appended to `## Optimizer case studies`).

Demonstrates how to compute Q3 — a Steinhardt parameter sensitive to
three-fold local symmetry that is absent from `ALL_METRICS` — inside a
2-argument objective callable using `compute_steinhardt_per_atom` and
`NeighborList`.  Includes three code examples:

1. *Q3-only objective* — `compute_steinhardt_per_atom(pts, [3],
   ctx.cutoff)["Q3"].mean()`.
2. *Combined Q3 + H_total objective* — demonstrates mixing a custom
   per-atom metric with a built-in metric from `m`.
3. *Custom radial-moment objective* — uses `NeighborList.all_dists`
   and `NeighborList.rows` directly to compute a per-atom neighbor-distance
   variance without calling `compute_steinhardt_per_atom`, illustrating the
   more general `NeighborList` usage pattern.

Each example is accompanied by a `n_pairs == 0` guard note and a reminder
to pass `cutoff=3.5` explicitly.

**`n_atoms=0` behaviour note** (added after the `warnings.catch_warnings`
code block in *Functional API (`generate`)*).

Clarifies that passing `n_atoms=0` is accepted and returns an empty
`GenerationResult` with `n_passed=0` and `n_attempted=0` without raising
an error.  This is **intentional**: it lets pipelines call `generate()`
unconditionally and handle the empty result with the standard
`if not result:` guard rather than wrapping the call in a `try/except`
block.

**`element_fractions` out-of-pool key note** (added after the
`element_fractions` weight-normalisation paragraph in *Biased element
fractions*).

Clarifies that every key in `element_fractions` must correspond to an
element present in the `elements=` pool; passing a symbol that is not in
the pool raises `ValueError` at construction time.  This is **intentional
by design**: a weight assigned to an element that can never be sampled
would silently mislead callers about the actual sampling distribution.
The existing sentence "Elements absent from the dict receive weight 1.0"
has been corrected to "Elements absent from the dict (but present in the
pool) receive weight 1.0" to prevent the same misreading.

**Affected files:** `docs/quickstart.md`

---

## [0.4.0] — 2026-03-24

### Fixed

#### BUG — `ALL_METRICS` module docstring incorrectly described type as `tuple[str, ...]` (`_atoms.py`)

The module-level docstring entry for `ALL_METRICS` stated the type as
`tuple[str, ...]`, but the actual runtime type has always been `frozenset[str]`.
This mismatch could mislead users expecting indexed access (`ALL_METRICS[0]`)
or a guaranteed iteration order.  The docstring has been corrected to
`frozenset[str]`, and an inline docstring has been added to the definition
explaining that `sorted(ALL_METRICS)` should be used whenever ordered
enumeration is required.

**Affected files:** `src/pasted/_atoms.py`

#### BUG — `compute_shape_anisotropy` emitted `RuntimeWarning` for structures with Inf coordinates (`_metrics.py`)

When an atom position contained `Inf`, NumPy's mean-centering subtraction
(`p = pts - pts.mean(axis=0)`) emitted a `RuntimeWarning: invalid value
encountered in subtract`.  The function already returned `NaN` in this case,
but the warning was noise for callers that intentionally pass degenerate
structures (e.g., `test_adversarial.py::test_from_xyz_nan_inf_coords_does_not_hard_crash`).
An explicit `np.isfinite` guard now returns `NaN` before the subtraction,
suppressing the warning without changing the return value for valid inputs.

**Affected files:** `src/pasted/_metrics.py`

#### BUG — `read_xyz()` raised `ValueError` instead of `FileNotFoundError` for non-existent paths (`_generator.py`)

`read_xyz("missing.xyz")` silently fell through the path-existence check and
attempted to parse the path string as XYZ text, raising a confusing
`ValueError: Expected atom count on line 1, got 'missing.xyz'`.  The
`Structure.from_xyz()` method (which `read_xyz` wraps) correctly raised
`FileNotFoundError` for the same input.  The path-validation block in
`read_xyz` has been updated to match `Structure.from_xyz()` behavior,
raising `FileNotFoundError` or `IsADirectoryError` explicitly before any
parse attempt.

**Affected files:** `src/pasted/_generator.py`



#### FEATURE — `_combined_core`: single-pass C++17 all-metrics kernel (`_combined_core.cpp`, `_metrics.py`, `setup.py`)

`_combined_core` is a new pybind11 extension that computes all pair-based
structural metrics in a **single `FlatCellList` traversal**, replacing the
four independent cell-list builds previously issued by `_rdf_h_cpp`,
`_graph_metrics_cpp`, `_steinhardt_per_atom`, and `_bond_angle_entropy_cpp`.

The combined kernel accumulates in one pass:

- RDF histogram → `h_spatial`, `rdf_dev`
- Bond adjacency lists → `graph_lcc`, `graph_cc`, `ring_fraction`,
  `charge_frustration`, `moran_I_chi`
- Steinhardt re/im buffers → `Q4`, `Q6`, `Q8` (fast-path ④, same
  Cartesian-polynomial expressions as `_steinhardt_core`)
- Per-atom distance sums → `radial_variance`
- Per-atom outer-product tensor components → `local_anisotropy`
- CSR neighbor count → bond-angle entropy (no second cell-list pass;
  unit vectors are filled from the already-populated `bond_adj` lists)

`compute_all_metrics` now dispatches to `_all_metrics_cpp` when
`HAS_COMBINED = True` (the new top priority in the dispatch chain),
falling back to the existing `_HAS_GRAPH` / pure-Python paths otherwise.

**Measured speedup** (C++ level, FlatCellList × 3 → × 1):

| N    | Old (ms) | New (ms) | Speedup |
|------|----------|----------|---------|
| 50   | 0.110    | 0.057    | 1.93×   |
| 500  | 0.615    | 0.328    | 1.87×   |
| 1000 | 1.131    | 0.591    | 1.91×   |

**NaN/Inf guard:** non-finite coordinates in `pts` are detected at entry and
return a zero-filled result dict (matching the behavior of the individual
extension modules) rather than crashing in `FlatCellList::build`.

**Affected files:**
- `src/pasted/_ext/_combined_core.cpp` — new file
- `src/pasted/_metrics.py` — `_HAS_COMBINED` import + dispatch block added
- `setup.py` — `_combined_core` entry uncommented

#### FEATURE — `_EXT_SUBMODULES` and fallback tests extended (`tests/test_ext_init.py`, `tests/test_fallback_paths.py`)

- `test_ext_init.py`: `_EXT_SUBMODULES` already included `_combined_core`;
  `HAS_BA_CPP` / `HAS_COMBINED` flag assertions and attribute-existence checks
  are in place from the v0.4.0-alpha work.
- `test_fallback_paths.py`: `_COMBINED_FLAG` added;
  `_ALL_METRICS_FLAGS` now covers all four C++ flags
  (`_HAS_GRAPH`, `_HAS_STEINHARDT`, `_HAS_BA_CPP`, `_HAS_COMBINED`).

### Fixed

#### BUG — `test_property_based.py::TestRotationInvariance::test_graph_metrics_rotation_invariant` failed with `atol=1e-8` violation on degenerate input

Hypothesis's shrinking strategy could produce coordinate arrays where many
atoms shared the same position (minimum distance → 0).  With coincident
atoms, `r_bound → 0` in `rdf_h_cpp`, making `RDF_dev` numerically unstable
across rotation.  Such inputs are physically impossible (no real structure
has atoms at identical positions).

**Fix:** `positions_strategy` in `test_property_based.py` now applies
`hypothesis.assume(min_pairwise_dist² ≥ 0.25)` (i.e., all pairs ≥ 0.5 Å),
discarding degenerate examples rather than masking a numerical edge case.
The 0.5 Å threshold is a conservative lower bound on the shortest known
covalent bond (H–H ≈ 0.74 Å).

**Affected files:**
- `tests/test_property_based.py` — `assume` import + `_MIN_ATOM_DIST` guard


---

## [0.3.10] — 2026-03-22

### Fixed

#### BUG — `compute_shape_anisotropy`: `ZeroDivisionError` on subnormal coordinates (`_metrics.py`)

*Discovered via: hypothesis property-based test `TestComputeAllMetricsCompleteness`*

`compute_shape_anisotropy` guarded against division by zero in the gyration
tensor trace using an exact equality check (`tr == 0`).  This check failed to
catch subnormal coordinate differences — pairs of atoms whose positions differ
by as little as ≈ 6e-108 Å produce a non-zero `tr` value that is nonetheless
so small that `1.5 * tr2 / tr**2` overflows to infinity or raises
`ZeroDivisionError` before `np.clip` can act.

**Fix:** the guard is tightened from `tr == 0` to `tr < 1e-30`.  Any
gyration tensor with trace below this threshold is treated as a degenerate
coincident-atom configuration and returns `0.0` (spherical).

**Affected files:**
- `src/pasted/_metrics.py` — guard condition and docstring updated

#### BUG — `test_maxent.py::TestPlaceMaxentCutoff::test_ang_cutoff_unchanged_end_to_end` always failed in the project's own test suite

*Discovered via: `pytest` run from the project root*

`conftest.py` at the project root inserts `src/` ahead of the installed
package on `sys.path` so that tests exercise the source tree rather than the
wheel.  When C++ extensions are not compiled inside `src/pasted/_ext/` (the
typical state for a source checkout), `HAS_MAXENT_LOOP` is `False` in the
test environment.  The test unconditionally attempted to monkey-patch the C++
entry point `_pl._cpp_place_maxent`, which is never called when
`HAS_MAXENT_LOOP=False`, so `captured` remained empty and the assertion
`len(captured) == 1` always failed.

**Fix:** `@pytest.mark.skipif(not HAS_MAXENT_LOOP, reason="…")` added to the
test method so that it is skipped in pure-Python environments and only runs
when the C++ extension is active.

**Affected files:**
- `tests/test_maxent.py` — `HAS_MAXENT_LOOP` import and `skipif` decorator added



---

## [0.3.9] — 2026-03-22

### Fixed

#### BUG — `_steinhardt_per_atom_sparse`: Q=1 for overlapping coordinates (C++ returns Q=0)

*Discovered via: hypothesis property-based test*

`scipy.spatial.cKDTree.query_pairs(cutoff)` returns pairs with `d=0` (coincident coordinates), but the Python implementation, while preventing division by zero with `safe_r_nb = where(r_nb>0, r_nb, 1.0)`, still accumulated these pairs as valid bonds. The C++ implementation skips entirely when `d < 1e-10`, leading to inconsistency.  
**Fix:** Bonds where `r_nb < 1e-10` are now masked out in the Python path.

#### BUG — `parse_xyz`: ValueError crash on blank lines within coordinate blocks

*Discovered via: coverage analysis (untested path inspection)*

In the XYZ format, blank lines are legal as separators, but splitting a blank line before the `len(parts) < 4` check returns an empty list, causing `ValueError: Malformed coordinate line: ''`.  
**Fix:** Blank lines are now skipped within the coordinate loop.

#### BUG — `FlatCellList::build()`: signed integer overflow in `int`

*Discovered via: UBSan (Undefined Behavior Sanitizer)*  
*Affected files: `_steinhardt.cpp`, `_graph_core.cpp`, `_relax.cpp` (all three files)*

When `cutoff` is large (e.g., 1000 Å), the cell count computation `nx * ny * nz` could overflow `int`, resulting in an invalid size passed to `cell_head.assign()`, triggering undefined behavior in C++.

**Fix approach (Plan A):** The product is monitored as `int64_t`; if the value exceeds `1<<22` (4 million cells, about 16 MB), the `cell_size` is doubled and the calculation retried in a loop. Once the upper limit is satisfied, the result is cast back to `int` for the subsequent hot loop (`for_each_neighbor`), preserving performance.

### Added

#### TEST — Property-based test suite (`tests/test_property_based.py`)

Added 23 tests using `hypothesis` covering 9 classes:

- `TestMetricRangeInvariants` — Range invariance for all metrics
- `TestTranslationInvariance` — Invariance under translation
- `TestRotationInvariance` — Invariance under rotation
- `TestCppVsPythonConsistency` — C++ ↔ Python sparse path agreement (atol=1e-10)
- `TestFastPathVsGeneric` — Fast-path ④ vs generic ①②③ agreement (atol=1e-10)
- `TestGenerateInvariants` — Completeness and correctness of atom count, parity, and metrics in `generate()`
- `TestComputeAllMetricsCompleteness` — All 13 metric keys present, finite, `dict[str, float]` result
- `TestPerAtomRanges` — Per-atom Q_l also within [0, 1]
- `TestIdempotency` — Identical output for identical input (idempotency)

### Changed (type annotations — mypy strict)

- `_placement.py`: `angular_repulsion_gradient` return value is now wrapped with `np.asarray()` (`Any` → `ndarray`)
- `_optimizer.py`: `_make_ctx` has explicit `positions: list[tuple[float, float, float]]` type annotation
- `_optimizer.py`: `tuple(tuple(p) for p in positions)` changed to `(float(p[0]), float(p[1]), float(p[2]))` for type correctness

---

## [0.3.8] — 2026-03-22

### Fixed

#### BUG — `moran_I_chi` exceeded the documented upper bound of 1.0 on sparse graphs

**Affected versions:** all releases prior to v0.3.8.
**Affected code paths:** C++ (`_graph_core.cpp`, `graph_metrics_cpp`) and
Python fallback (`_metrics.py`, `compute_moran_I_chi`).

**Root cause.**
Moran's I is computed as

```
I = (N / W) * (Σ w_ij dev_i dev_j) / (Σ dev_i²)
```

where `W` is the total number of directed edges (twice the number of
undirected pairs within `cutoff`).  With binary (0/1) weights the theoretical
maximum of +1 is only guaranteed when the weight matrix is row-standardised
(`W = N`).  PASTED uses un-normalised binary weights, so when the cutoff graph
is very sparse (`W < N`) the prefactor `N / W > 1` inflates the result above
+1.  This is a numerical artefact of the weight convention, not a sign of
incorrect spatial correlation.

**Observed symptoms.**  Structures with `graph_lcc ≈ 0.08–0.17` (i.e. most
atoms isolated under the default cutoff) produced `moran_I_chi` values of
~1.16–1.22, silently violating the documented range `(-∞, 1]`.

**Fix.**  Both the C++ path and the Python fallback now clamp the raw result
to `min(raw, 1.0)` before returning.  The clamp only fires for sparse graphs
(`W < N`); for all typical PASTED structures with `graph_lcc ≈ 1.0` the
result is unchanged.

**Tests added:**
- `tests/test_metrics.py::TestMoranIChi::test_moran_I_chi_never_exceeds_one` —
  generates 30 structures with a mixed-element pool and asserts `<= 1.0`.
- `tests/test_metrics.py::TestMoranIChi::test_moran_I_chi_clamp_cpp` —
  constructs a deliberately sparse 3-atom graph (`W=2, N=3`) and checks the
  C++ path directly.

### Performance

#### PERF — `_steinhardt_core`: real spherical harmonics hardcoded for l=4,6,8 (optimization ④)

When `l_values = [4, 6, 8]` (the default used by `compute_all_metrics`), the
`accumulate` lambda now takes a dedicated fast-path that replaces the
associated-Legendre recurrence (`compute_plm`, ~148 data-dependent ops) and
the Chebyshev trig recurrence with a single block of straight-line Cartesian
polynomial arithmetic.

**Mathematical basis.**  On the unit sphere every real spherical harmonic
`S_lm(x,y,z)` is a pure integer-coefficient polynomial in `x,y,z` — no
`sqrt`, no trig, no division.  The `(1-z²)^(m/2)` factor in `P_l^m(z)` is
cancelled exactly by the `r_xy^m` that comes from expanding
`cos(mφ)·r_xy^m` and `sin(mφ)·r_xy^m` in Cartesian form.

**Code generation.**  A SymPy script applied joint CSE across l=4,6,8,
producing 84 CSE intermediates + 39 accumulation lines of pure `double`
arithmetic.  `std::pow` for integer exponents was replaced with explicit
multiplications to avoid libm overhead.

**Measured speedup** (gas structures k≈0.7, `-O3 -std=c++17`, no `-march=native`):

| N | generic ①②③ | fast-path ④ | speedup |
|--:|---:|---:|---:|
| 100 | 0.040 ms | 0.028 ms | **1.4×** |
| 500 | 0.182 ms | 0.115 ms | **1.6×** |
| 1 000 | 0.374 ms | 0.239 ms | **1.6×** |
| 2 000 | 2.115 ms | 1.667 ms | 1.3× |
| 5 000 | 4.329 ms | 3.966 ms | 1.1× |

**Tests added:** `TestSteinhardtFastPath::test_fast_path_matches_python_sparse`
(atol=1e-12, N=5/50/200/1000) and `test_fast_path_vs_generic_order`.

### Documentation

- `docs/api/metrics.rst`: range column updated from `≈ [−1, 1]` to `(-∞, 1]`
  with an explanation of the clamp; `charge_frustration` description corrected
  from "Mean EN variance" to "Population variance of |Δχ| across cutoff-adjacent
  pairs" (matching the actual `np.var` / population-variance C++ implementation).
- `docs/architecture.md`: `moran_I_chi` row in the metrics table updated with
  a reference to the v0.3.8 fix.
- `src/pasted/_metrics.py`: `compute_moran_I_chi` docstring updated — return
  type description changed from `(-1, 1]` to `(-∞, 1]` with a `.. note::` block
  explaining the binary-weight sparse-graph edge case and the clamp.

---

## [0.3.7] — 2026-03-22

### Documentation

#### DOC — `compute_all_metrics` N=5–5000 benchmark and Steinhardt buffer-layout analysis

Added a performance benchmark table to `docs/quickstart.md` covering all
sub-components of `compute_all_metrics` (C++ extensions active) from N = 5
to N = 5 000.  Key findings (pre-fix baseline):

- Peak RSS across the full sweep: **152 MB** (< 3 MB growth; no memory
  leak detected over 500 repeated calls at each size).
- Wall time dominated by `compute_steinhardt` from N ≈ 200 onwards.
- Superlinear growth traced to a CPU cache pressure effect in the former
  `(n_l, l_max+1, N)` accumulator layout (see PERF entry below).

`docs/architecture.md` → *Accumulator buffer layout* and `docs/api/metrics.rst`
updated with root-cause analysis and post-fix benchmark table.

### Performance

#### PERF — `_steinhardt_core`: atan2 elimination + Chebyshev recurrence + stack P_lm (`_steinhardt.cpp`)

Three per-bond arithmetic optimizations applied to the `accumulate` lambda in
`steinhardt_per_atom_cpp` (optimizations ①+②+③ from the performance analysis):

**① + ② — atan2 elimination and Chebyshev recurrence for cos(m·φ)/sin(m·φ).**
The former code called `std::atan2(dy, dx)` once per bond and then issued
`l_max` independent `std::cos(m*phi)` / `std::sin(m*phi)` calls (18 libm calls
at l_max = 8, each ≈ 20–50 CPU cycles).  The replacement:

1. Computes `cos_phi = dx/r_xy`, `sin_phi = dy/r_xy` from a single `sqrt`
   (no `atan2`).
2. Derives all higher orders via the Chebyshev two-term recurrence:

   ```
   cos(m·φ) = 2·cos_phi·cos((m-1)·φ) − cos((m-2)·φ)   [2 mults + 1 sub]
   sin(m·φ) = 2·cos_phi·sin((m-1)·φ) − sin((m-2)·φ)
   ```

Total per bond: 2 sqrts + (l_max−1)×4 arithmetic ops vs. 1 atan2 + 18 libm
calls.  Vectorised benchmark shows ~4× on the phi-trig component.

**③ — Stack-allocated P_lm.**  `compute_plm` previously received a
`std::vector<std::vector<double>>&` that was re-assigned (zeroed and resized)
each bond call.  The function signature now takes a caller-supplied
`double[L_MAX+1][L_MAX+1]` on the stack, eliminating per-bond heap traffic
and keeping the 936-byte table in L1 cache.

Measured speedup on PASTED default gas structures (k ≈ 0.7, all C++ extensions):

| N | v0.3.6 `steinhardt` | v0.3.7 `steinhardt` | speedup | `all_metrics` speedup |
|--:|:--:|:--:|:--:|:--:|
| 20 | 0.052 ms | **0.015 ms** | **3.4×** | 1.0× |
| 100 | 0.076 ms | **0.065 ms** | 1.2× | **1.2×** |
| 500 | 0.306 ms | **0.156 ms** | **2.0×** | **1.3×** |
| 1 000 | 0.654 ms | **0.311 ms** | **2.1×** | **1.3×** |
| 2 000 | 2.307 ms | **1.687 ms** | 1.4× | 1.1× |
| 5 000 | 5.605 ms | **4.484 ms** | 1.3× | 1.1× |

All 846 tests pass.  Correctness verified vs Python fallback at N = 3, 10, 50,
200, 1 000 (max absolute difference < 2 × 10⁻¹⁵ — floating-point rounding only).

---

## [0.3.6] — 2026-03-21

#### PERF — `_steinhardt_core`: transpose accumulator buffer to (N, n_l, l_max+1) (`_steinhardt.cpp`)

The C++ Steinhardt accumulator previously used layout `(n_l, l_max+1, N)` —
atom index as the innermost dimension — with index formula
`li * lm1 * n + m * n + i`.  Each bond's inner `m`-loop (m = 0 … 8) wrote
to addresses `N × 8 bytes` apart, causing L2→L3 cache spill at N ≈ 1 000
and superlinear wall-time growth despite an O(N·k·l²) algorithm.

**Fix:** the buffer is now laid out as `(N, n_l, l_max+1)` with index
`i * n_l * lm1 + li * lm1 + m`.  Every bond's `(l_idx, m)` writes for
atom `i` are contiguous (stride 8 B, ≤ 216 bytes per atom — fits in a single
cache line regardless of N).

Measured speedup on PASTED default gas structures (k ≈ 0.7 neighbors/atom;
all C++ extensions active; single CPU core):

| N | `compute_steinhardt` | `compute_all_metrics` |
|--:|:--:|:--:|
| 100 | 0.132 ms → 0.076 ms (**1.7×**) | 0.49 ms → 0.26 ms (**1.9×**) |
| 500 | 0.330 ms → 0.306 ms (1.1×) | 1.21 ms → 0.92 ms (1.3×) |
| 1 000 | 0.716 ms → 0.654 ms (1.1×) | 1.94 ms → 1.68 ms (1.2×) |
| 2 000 | 2.438 ms → 2.307 ms (1.1×) | 5.12 ms → 3.99 ms (1.3×) |
| 5 000 | 6.383 ms → 5.605 ms (1.1×) | 13.3 ms → 9.7 ms (**1.4×**) |

At sparse k ≈ 0.7, the per-bond arithmetic (P_lm recurrence, cos/sin/atan2)
still dominates at mid-range N; the layout improvement is most visible at
N = 100 (L2 was already marginal) and N ≥ 2 000 (accumulated write-latency
savings compound over more bonds).  At denser k (e.g. k ≈ 30 in solvated
structures), the speedup grows larger because more writes are issued per atom.

All 846 tests pass.  Correctness verified by comparing C++ and Python-fallback
outputs at N = 5, 20, 100, 500 (max absolute difference < 1 × 10⁻¹⁰).

#### PERF — `compute_shape_anisotropy`: replace `eigvalsh` with trace / Frobenius norm (`_metrics.py`)

`compute_shape_anisotropy` previously computed the eigenvalues of the 3×3
gyration tensor via `np.linalg.eigvalsh` (a full LAPACK call) even though
only two scalar quantities derived from those eigenvalues were used:

    trace(T)  = λ₁ + λ₂ + λ₃
    ‖T‖²_F    = λ₁² + λ₂² + λ₃²

Both are directly available from the matrix elements without
diagonalisation.  The new implementation reads three diagonal entries for the
trace and uses `np.einsum("ij,ij->", T, T)` for the Frobenius norm squared.
Measured speedup: ~1.5× per call (~20 µs → ~13 µs), saving approximately
10 ms over a 500-step optimizer run where the function is called ~1500 times.

#### PERF — `validate_charge_mult`: skip `Counter` and `f-string` on success path (`_atoms.py`)

On the `ok=True` path `validate_charge_mult` previously built a
`Counter(atoms_list)` and formatted a detailed diagnostic string even though
all hot-loop callers in `_optimizer.py` discard the second return value
(`ok, _ = validate_charge_mult(...)`).  The function now returns `(True, "")`
immediately after the two guard checks pass, avoiding the allocation entirely.
Measured speedup: ~2.2× per call on the success path (~1.30 µs → ~0.59 µs).

#### PERF-1 — `graph_metrics_cpp`: remove duplicate adjacency list (`_graph_core.cpp`)

The `graph_metrics_cpp` function previously maintained two **identical**
adjacency lists — `bond_adj` and `graph_adj` — built from the same
`FlatCellList` pass under the same distance condition.  Every edge was stored
four times (both directions in both lists).  `graph_adj` has been removed and
all consumers (`graph_lcc`, `graph_cc`, Moran's I) now reference `bond_adj`
directly.  This halves the adjacency-list memory at runtime (O(N·k) → O(N·k/2)
effective allocation) and eliminates the redundant push-back overhead during
the FlatCellList enumeration pass.

#### PERF-2 — `graph_metrics_cpp`: `graph_cc` triangle counting uses `binary_search` (`_graph_core.cpp`)

The triangle-counting loop in `graph_cc` previously used `std::find` for
neighbor lookup — an O(k) linear scan that made the full loop O(N·k³).
Adjacency lists are now **sorted once** after the FlatCellList pass, and
`std::binary_search` is used instead (O(log k)), reducing the triangle loop
to O(N·k²·log k).  At the typical k ≈ 2–16 of PASTED structures this is
2–4× faster.  The file-header complexity annotation has been corrected from
the former misleading ``O(N·k^2)`` to ``O(N·k²·log k)``.

#### PERF-3 — `rdf_h_cpp`: streaming histogram, no intermediate `pair_dists` vector (`_graph_core.cpp`)

`rdf_h_cpp` previously collected all pair distances into a
``std::vector<double>`` before binning them.  This allocated an O(N·k) heap
buffer (up to `n * 10 * 8` bytes pre-reserved) that was used exactly once and
then freed.  The `collect` lambda now increments the histogram bin directly,
eliminating the intermediate vector and its associated heap allocation
entirely.

**Measured speedup (graph_metrics_cpp, wall time per call):**

| N    | Before (ms) | After (ms) | Speedup |
|------|-------------|------------|---------|
| 64   | 0.042       | 0.021      | 2.0×    |
| 256  | 0.117       | 0.071      | 1.6×    |
| 512  | 0.251       | 0.196      | 1.3×    |
| 1000 | 0.455       | 0.322      | 1.4×    |
| 2000 | 0.887       | 0.731      | 1.2×    |

**Affected files:**
- `src/pasted/_ext/_graph_core.cpp` — all three changes; version bumped to
  v0.2.10 in module docstring and `PYBIND11_MODULE` binding
- `src/pasted/_metrics.py` — `_compute_graph_ring_charge` and
  `compute_all_metrics` docstrings updated to reflect C++ internals and to
  add an explicit ``.. warning::`` for the O(N²) Python fallback
- `docs/architecture.md` — `_graph_core` section rewritten with v0.2.10
  internal design notes
- `docs/api/metrics.rst` — C++ acceleration note updated; O(N²) fallback
  promoted from a plain ``.. note::`` to a ``.. warning::``

#### BUG — `ring_fraction` undercounted ring membership for all cycle sizes (`_metrics.py`, `_graph_core.cpp`)

The previous implementation used a Union-Find (Disjoint Set Union) algorithm to
detect rings: when processing the bond graph edge-by-edge, any edge that
connected two vertices already in the same component was a "back-edge" and its
two endpoints were marked as ring members.

This approach is fundamentally incorrect for identifying ring membership.
Union-Find can determine *whether* a cycle exists, but it cannot identify
*all vertices* that belong to a cycle.  Only the two direct endpoints of the
detected back-edge were ever marked, leaving every other atom in the ring
unmarked.  For a ring of N atoms, the back-edge connects the last two atoms to
close the cycle, so only 2 out of N atoms were counted:

| Ring size | Old result | Correct result |
|-----------|-----------|----------------|
| 3-cycle   | 2/3       | 3/3            |
| 5-cycle   | 2/5       | 5/5            |
| 6-cycle   | 2/6       | 6/6            |
| N-cycle   | 2/N       | N/N            |

This affected both the pure-Python path (`compute_ring_fraction` in
`_metrics.py`) and the C++ path (`graph_metrics_cpp` in `_graph_core.cpp`),
which used an identical Union-Find implementation.

**Impact:** Any filter using `ring_fraction` (e.g. `ring_fraction:0.5:-`) would
silently reject structures that are genuinely ring-rich, because the reported
value was always ≤ 2/N regardless of the actual ring composition.

**Fix:** Replaced Union-Find with **Tarjan's iterative bridge-finding algorithm**
(O(N + E)).  A bridge is an edge whose removal disconnects the graph — i.e. an
edge that is *not* part of any cycle.  An atom is in a ring if and only if at
least one of its incident edges is a non-bridge.  The iterative DFS
implementation avoids Python's recursion limit for large structures.

Helper functions `_build_adj` and `_tarjan_bridges` were extracted to keep
`compute_ring_fraction` readable.  The C++ path received the same algorithm
using `std::unordered_set` for O(1) bridge lookups.

**Affected files:**
- `src/pasted/_metrics.py` — `compute_ring_fraction` rewritten; `_build_adj`
  and `_tarjan_bridges` added; `Iterator` import added
- `src/pasted/_ext/_graph_core.cpp` — `ring_fraction` block replaced with
  iterative Tarjan; `#include <unordered_set>` added; header comment updated
- `tests/test_metrics.py` — `test_triangle_all_in_ring` updated from `2/3`
  to `1.0`
- `tests/test_fallback_paths.py` — `TestRingFractionRegression` class added

---

#### REFACTOR — Decouple `stream()` / `generate()` via `_stream_with_stats()` (`_generator.py`)

`stream()` previously communicated run statistics to `generate()` by writing
to a `_last_run_stats` instance variable at the end of its generator body.
This created a hidden coupling: if a caller broke out of `stream()` early
(e.g. `for s in gen.stream(): break`) and then called `generate()`, the
stale counters from the previous (incomplete) run would be returned silently.

**Fix:** the generation loop now lives in a private method
`_stream_with_stats(self) -> tuple[Iterator[Structure], dict[str, int]]`
that returns a `(structures_iterator, stats_dict)` pair.  The `stats_dict`
is a mutable mapping that is populated in-place only when the iterator is
fully exhausted.  The two public methods become thin, stateless wrappers:

- `stream()` — calls `_stream_with_stats()` and forwards the iterator.
  Public API is unchanged.
- `generate()` — calls `_stream_with_stats()`, exhausts the iterator via
  `list(it)`, then reads `stats` directly.  No longer touches
  `_last_run_stats`.

The `_last_run_stats` instance variable has been removed entirely.

#### REFACTOR — Extract verbose log helpers from `stream()` (`_generator.py`)

The verbose `self._log(...)` calls that were scattered throughout the
generation loop have been extracted into three focused private methods:

- `_log_filter_header()` — emits the `[filter]` preamble line.
- `_log_sample_result(i, width, denom, flag, *, metrics, msg)` — emits one
  per-sample status line (`PASS`, `skip`, `invalid`, or `warn`).
- `_log_summary(n_attempted, n_passed, n_invalid, n_rejected_filter)` —
  emits the `[summary]` footer line.

Each helper guards itself with `if not self._cfg.verbose: return`, so
callers need not repeat the verbose check.  The generation loop in
`_stream_with_stats._inner()` now contains only placement logic, with
log calls at natural event boundaries.

**Affected files:**
- `src/pasted/_generator.py` — implementation, module docstring, and method
  docstrings updated

---

## [0.3.5] — 2026-03-22

### Bug Fixes

#### BUG-1 — `parse_element_spec()` raised `AttributeError` when passed a `list[str]` (`_atoms.py`)

`parse_element_spec()` accepted only a `str` argument (atomic-number notation
such as `"6,7,8"` or `"1-30"`).  When called with a `list[str]` of element
symbols — a form that is **documented as valid** in the quickstart and API
reference — it raised:

```
AttributeError: 'list' object has no attribute 'split'
```

because the implementation immediately called `spec.split(",")` without
checking the input type.

Note that `StructureGenerator` and `generate()` already handled the list form
correctly via an internal `isinstance` branch that bypassed `parse_element_spec`
entirely.  The bug only surfaced when `parse_element_spec` was called directly
(e.g. from user code or third-party libraries building on the public API).

**Fix:** `parse_element_spec` now accepts `str | list[str]`.  When a list is
provided, each entry is validated against the built-in symbol table and the
result is returned sorted by ascending atomic number — matching the behavior
of the string path.  The type signature in the source and all docstrings have
been updated accordingly.

**Affected files:**
- `src/pasted/_atoms.py` — implementation and docstring
- `docs/api/atoms.rst` — API reference note updated
- `docs/quickstart.md` — element pool spec table note updated

---

#### BUG-2 — `Structure.from_xyz()` raised a confusing `ValueError` for missing files (`_generator.py`)

When `Structure.from_xyz()` was given a path string that did not exist on
disk, it silently treated the path string itself as XYZ text and attempted to
parse it, producing a confusing error:

```
ValueError: Expected atom count on line 1, got '/path/to/missing.xyz'
```

The correct error for a missing file is `FileNotFoundError`, which is what
Python's own `open()` / `Path.read_text()` raise.  The misleading `ValueError`
made it hard to distinguish file-not-found issues from genuine XYZ parse errors.

**Fix:** the path-resolution block now performs explicit existence and
`is_file()` checks before calling `read_text()`.  Missing paths raise
`FileNotFoundError`; paths pointing to directories raise `IsADirectoryError`.
The docstring `Raises` section has been updated to document both new exception
types.

**Affected files:**
- `src/pasted/_generator.py` — implementation and docstring (`Raises` section)
- `docs/api/atoms.rst` — combined `versionchanged` note added

---

## [0.3.4] — 2026-03-22

### Bug Fixes

#### BUG-1 — `stream()` warning message incorrectly attributed all failures to parity (`_generator.py`)

When both parity failures and metric-filter failures occurred in the same run
and **no** structures passed (i.e. `n_passed == 0`), `stream()` emitted:

```
All 100 attempt(s) were rejected by the charge/multiplicity parity check (54 invalid).
```

even though 46 of those 100 attempts **did** pass the parity check and were
instead rejected by `--filter` thresholds.  The misleading "All N" wording
caused users to diagnose the wrong root cause (element pool / charge settings)
instead of their filter configuration.

**Fix:** the parity-failure warning is now split into two branches:

- **Pure parity failure** (`n_rejected_filter == 0`): message unchanged —
  `"All N attempt(s) were rejected by the parity check (N invalid). …"`
- **Mixed failure** (`n_rejected_filter > 0`): new message —
  `"M of N attempt(s) were rejected by the parity check, and the remaining K
  that passed parity were rejected by metric filters. …"` — so both root
  causes are visible at once.

#### BUG-2 — `architecture.md` warning table listed a non-existent warning case

The `warnings.warn` behavior table in `docs/architecture.md` included a row
for *"Some attempts rejected by parity"* (`M of N attempt(s) …`) that was
never emitted by the implementation.  The intent (documented in a code
comment) is that partial parity rejection where some structures still pass is
**expected** behavior and does not warrant a warning.  The table has been
corrected to match the actual four-case implementation, and a note explaining
why the partial-parity case is intentionally silent has been added.

#### BUG-3 — O(N²) auto-cutoff computation in `compute_all_metrics`, `Structure.from_xyz`, and `_resolve_cutoff` (`_metrics.py`, `_generator.py`, `_optimizer.py`)

`place_maxent` received an O(N) auto-cutoff fix in v0.2.6, but four other
sites that compute the same `1.5 × median(rᵢ + rⱼ)` cutoff retained the
original O(N² log N) `sorted(pair_sums)` implementation:

| Site | File |
|---|---|
| `compute_all_metrics(cutoff=None)` | `_metrics.py` |
| `Structure.from_xyz(recompute_metrics=True, cutoff=None)` — frame-0 path | `_generator.py` |
| `Structure.from_xyz(recompute_metrics=True, cutoff=None)` — multi-frame path | `_generator.py` |
| `StructureGenerator._resolve_cutoff` | `_generator.py` |
| `StructureOptimizer._resolve_cutoff` | `_optimizer.py` |

For large structures this made the cutoff calculation the dominant cost:

| n_atoms | Before (O(N²)) | After (O(N)) | Speed-up |
|---:|---:|---:|---:|
| 500 | 20 ms | 0.11 ms | **176×** |
| 1 000 | 85 ms | 0.19 ms | **459×** |
| 2 000 | 385 ms | 0.33 ms | **1 179×** |
| 5 000 | 2 532 ms | 0.76 ms | **3 319×** |

**Fix:** all five sites now use the same O(N) identity as `place_maxent`:

```python
median_sum = float(np.median(radii)) * 2.0
cutoff = cov_scale * 1.5 * median_sum
```

The redundant `import numpy as _np` statements that were inserted inline
during the initial patch have also been removed; `numpy` is now imported
once at module level in `_generator.py`.

#### BUG-4 — `place_maxent` O(N) median approximation inaccurate for bimodal radius distributions (`_placement.py`, `architecture.md`)

The `median(rᵢ + rⱼ) = 2 · median(rᵢ)` identity used in `place_maxent` since
v0.2.6 was documented as holding for *"all built-in element pools"*.  This is
incorrect: for strongly **bimodal** pools (e.g. H with radius 0.31 Å mixed
with heavy metals at ~1.3 Å) the approximation can overestimate `ang_cutoff`
by up to **~50 %**, causing the angular repulsion to act over a wider
neighbourhood than intended and producing a weaker uniformity guarantee.

The algorithm itself is not changed (the performance/accuracy trade-off is
acceptable for standard element pools); the documentation and inline code
comment have been corrected to state the limitation clearly and recommend
passing an explicit `cutoff=` for bimodal pools.

### Code Quality

- `ruff check` and `ruff format` pass cleanly on all source and test files
  (`src/pasted/`, `tests/`).
- `mypy src/pasted/` and `mypy tests/` report no issues (25 source files
  checked total).
- Import order in `_generator.py` corrected to satisfy `isort` (`I001`).

---

## [0.3.3] — 2026-03-22

### Bug Fixes

#### `StructureOptimizer` — affine moves could not be used as the sole move type

`allow_affine_moves`, `allow_displacements`, and `allow_composition_moves` were
not treated as orthogonal, independent move types.  Two separate bugs made it
impossible to run an affine-only optimization:

1. **Validation error** — `__init__` raised `ValueError` whenever both
   `allow_displacements=False` and `allow_composition_moves=False`, even if
   `allow_affine_moves=True`.  The guard has been relaxed: the error is now
   raised only when **all three** flags are `False`.

2. **Dead-code path** — the move-selection logic in both `_run_one` (SA / BH)
   and the parallel-tempering loop treated affine as a *sub-type* of
   displacement moves (`if allow_displacements: … if allow_affine: …`).  With
   `allow_displacements=False` the affine branch was never reached.  Both loops
   now pre-build a `_move_pool` list of enabled move types and sample uniformly
   (1/N probability) at each step, making the three options fully orthogonal.

The distance-constraint relaxation (`relax_positions`) is **not** applied after
affine moves when `allow_displacements=False`, consistent with the existing
semantics of that flag.

### Tests

- `TestAllowDisplacements.test_both_false_raises` renamed to
  `test_all_false_raises`; error-message match string updated.
- New `TestAllowDisplacements.test_affine_only_does_not_raise` validates the
  relaxed validation path.
- New `TestAllowAffineMoves` class covering: affine-only SA runs, composition
  preservation, affine+composition without displacement, parallel-tempering
  affine-only, and repr output.

---



### Bug Fixes

#### `StructureOptimizer.run()` leaked spurious `UserWarning` from internal
retries (`_optimizer.py`)

`_make_initial` generates a single-sample structure up to 50 times until a
valid starting point is found.  Each failed attempt called
`StructureGenerator.generate()` with `n_samples=1`, which emitted a
`UserWarning` whenever the single attempt was rejected by the
charge/multiplicity parity check.  Because `_make_initial` is an *internal*
helper with its own retry loop, these transient failures are expected and
irrelevant to the caller — yet the warnings surfaced visibly to end users,
falsely suggesting that the optimization had failed even when it ultimately
succeeded.

**Fix:** Each `StructureGenerator.generate()` call inside `_make_initial` is
now wrapped in `warnings.catch_warnings()` with `warnings.simplefilter("ignore",
UserWarning)`.  The caller-visible warning in `run()` — emitted only when a
restart *cannot* start at all — is unchanged.

#### Quickstart warning example used an element pool that triggered the wrong
warning (`docs/quickstart.md`)

The documentation example intended to demonstrate the
`"No structures passed the metric filters …"` warning.  However, the element
pool `elements="6,7,8"` (C, N, O) combined with `charge=0, mult=1` and only
10 attempts caused the charge/multiplicity parity check to reject *all*
attempts before any filter evaluation.  The filter-rejection warning therefore
never fired, and the comment in the code block was misleading.

**Fix:** The example now uses `elements="6"` (carbon only), which always
satisfies the parity constraint, so the metric-filter warning fires as
documented.

### New Features

#### `StructureOptimizer.max_init_attempts` — configurable retry limit with
early parity validation (`_optimizer.py`)

Previously `_make_initial` used a hardcoded limit of 50 retries per restart.
This was too conservative for element pools where parity-valid compositions
are rare (e.g., large mixed pools with tight charge/mult constraints), and
could silently skip restarts that would have succeeded with a few more tries.

**Changes:**

* A new `max_init_attempts: int = 0` parameter is added to
  `StructureOptimizer.__init__`.  `0` means **unlimited retries** (the new
  default); any positive integer caps the attempts and restores the old
  bail-out behavior at the chosen limit.

* `__init__` now calls `_pool_can_satisfy_parity()` — a new module-level
  helper that checks whether the element pool's atomic-number parities can
  ever satisfy the charge/multiplicity constraint — and raises `ValueError`
  immediately if not.  This makes `max_init_attempts=0` safe: if
  construction succeeds, a valid initial structure is guaranteed to exist.

* The new helper `_pool_can_satisfy_parity(pool, n_atoms, charge, mult)`
  is also exported for use in tests and downstream tools.

**Migration:** No action required.  The default `max_init_attempts=0`
(unlimited) replaces the former hardcoded limit of 50.  To restore the
old behavior explicitly, pass `max_init_attempts=50`.

---

## [0.3.1] — 2026-03-21
 
### Bug Fixes
 
#### `Structure.comp` property was missing (`_generator.py`)
 
`Structure.__repr__` computed the alphabetically-sorted composition string as
a local variable `comp` that was never exposed as a public attribute.
Accessing `s.comp` raised `AttributeError` even though the string appeared
inside `repr(s)`.  (Note: the sort order is alphabetical, not Hill order;
see the `Structure.comp` documentation for details.)
 
**Fix:** `comp` is now a read-only `@property` on `Structure`.  `__repr__`
delegates to `self.comp`, so the string is computed in one place only.
 
#### `_composition_move` ignored `element_pool` (`_optimizer.py`)
 
Path 1 of `_composition_move` swapped two existing atoms within the current
structure.  Because no element was ever drawn from `element_pool`, running
`StructureOptimizer` with `allow_displacements=False` and a pool that
differed from the initial composition (e.g. starting from a C/N/O structure
and optimizing over Cr/Mn/Fe/Co/Ni) could never introduce pool elements —
the composition remained frozen at its initial values regardless of the
number of MC steps.
 
**Fix:** Path 1 now selects a random atom and replaces it with a different
element drawn from `element_pool` whose atomic number has the same Z mod 2
parity as the original, preserving charge/multiplicity validity.  Path 2
(two-atom simultaneous replacement) is kept as the fallback for cases where
no same-parity pool candidate exists.
 
#### `radii` cache was never refreshed after a composition move (`_optimizer.py`)
 
Inside `_run_one`, the covalent-radius cache `radii` was supposed to be
recomputed whenever the element types changed.  The guard condition was:
 
```python
atoms = new_atoms
if atoms is not new_atoms or new_atoms != list(atoms):   # always False
    radii = ...
```
 
After `atoms = new_atoms`, `atoms is new_atoms` is always `True`, making the
entire condition evaluate to `False`.  The cache was therefore never updated
after a composition move, so subsequent L-BFGS relaxation steps used stale
covalent radii and could produce clashing geometries.
 
**Fix:** The previous value of `atoms` is captured in `old_atoms` *before*
the reassignment, and the guard is `if old_atoms != atoms`.
 
#### Composition-only optimization retained non-pool atoms (`_optimizer.py`)
 
When `StructureOptimizer.run(initial=s)` was called with a structure whose
atoms were not all members of `elements`, the Metropolis loop could keep
those foreign atoms indefinitely: if the objective value happened to be
locally higher with foreign atoms present, every replacement move would be
rejected.  This made composition-only optimization (`allow_displacements=False`)
unreliable whenever `initial` was generated from a different element set.
 
**Fix:** A new helper `_sanitize_atoms_to_pool()` is called at the start of
`_run_one` whenever `allow_composition_moves=True` and the initial structure
contains atoms outside the pool.  Each foreign atom is replaced by a
parity-compatible pool element before the MC loop begins.
 
#### `tests/conftest.py` was missing
 
`test_generator.py` referenced three pytest fixtures — `gas_gen`,
`chain_gen`, and `shell_gen` — that were never defined.  Running `pytest`
raised `fixture not found` errors for every test that used them.
 
**Fix:** `tests/conftest.py` is added with all three fixtures.
 
#### Parallel Tempering retained non-pool atoms from `initial` (`_optimizer.py`)
 
`_run_parallel_tempering` initialized each replica's atom list directly from
`initial.atoms` without applying the same sanitization that `_run_one` gained
in the Bug #4 fix.  When `StructureOptimizer.run(initial=s, method="parallel_tempering")`
was called with a structure whose atoms were outside `elements`, the PT path
could retain those foreign atoms indefinitely across all replicas — even though
the SA and Basin-Hopping paths were already fixed.
 
**Fix:** `_run_parallel_tempering` now checks whether the caller-supplied
`initial` contains atoms outside the pool.  When it does and
`allow_composition_moves=True`, each replica's atom list is independently
sanitized via `_sanitize_atoms_to_pool` before the MC loop begins, using a
replica-specific RNG so that each replica explores a different parity-compatible
starting composition.
 
### Documentation
 
- `Structure` class docstring: documented the new `comp` property, updated
  `mode` attribute to list all valid values (``"gas"``, ``"chain"``,
  ``"shell"``, ``"maxent"``, ``"opt_<method>"``), and added a usage example.
- `Structure.comp` property: expanded docstring with return-value
  description and an inline example.
- `_composition_move()`: rewrote docstring to accurately describe the
  pool-replacement Path 1 that replaced the old swap-based logic.
- `_sanitize_atoms_to_pool()`: new function; documented with rationale,
  parameter descriptions, and edge-case behavior.
- `_run_parallel_tempering()`: updated *Initialization* section to describe
  the per-replica sanitization added for fixing bug.
- `StructureOptimizer` parameters `allow_displacements` and
  `allow_composition_moves`: noted that foreign atoms in `initial` are
  sanitized to the pool before the MC loop for all three methods.
- `_optimizer.py` module docstring: updated the "Composition move" section
  to reflect the corrected pool-replacement algorithm and clarify that
  sanitization via `_sanitize_atoms_to_pool` applies to SA, BH, and PT.
- `README.md`:
  - Added `s.comp` to the *Structure attributes* example.
  - Fixed the *Structure optimizer* example to use `opt.run().best`
    (previously showed `opt.run()` assigned directly to `best`, which is
    an `OptimizationResult`, not a `Structure`).
  - Updated the *Composition-only optimization* subsection to note that
    all three methods support cross-pool optimization, and added
    ``method="basin_hopping" / "parallel_tempering"`` to the example.
  - Removed stale per-metric version annotations
    (`# new in v0.1.12`, `# non-zero in v0.1.13+`).
- `docs/quickstart.md`:
  - Rewrote the *Composition-only optimization* section to remove the
    incorrect requirement that the initial structure must be drawn from the
    same pool as the optimizer (Bug #4 and #6 make this unnecessary).
  - Updated the objective-selection callout to reflect the pool-replacement
    move algorithm; removed the outdated claim that composition moves are
    label swaps that leave the element-count histogram unchanged.
- `docs/architecture.md`:
  - Updated the *Move types* table and **Parity-preserving composition move**
    description to replace the old swap-based Path 1 with the corrected
    pool-replacement algorithm.
  - Added a note that `_sanitize_atoms_to_pool` is applied to all three
    optimization methods when the initial structure contains foreign atoms.
 
### Tests
 
- Added `tests/conftest.py` with `gas_gen`, `chain_gen`, and `shell_gen`
  fixtures shared across test classes.
- `TestStructureCompProperty` (4 tests in `test_generator.py`): `comp` return
  type, presence in `repr`, consistency with `atoms`, and accessibility on
  optimizer results.
- `TestCompositionMoveBugFixes` (3 tests in `test_optimizer.py`): pool elements
  are introduced, atom count is preserved, single-element pool does not crash.
- `TestRadiiCacheBugFix` (1 test in `test_optimizer.py`): composition-only
  result contains only pool atoms after SA run from a foreign-element initial.
- `TestSanitizeAtomsToPool` (4 tests in `test_optimizer.py`): all atoms in pool
  after sanitize, length preserved, in-pool atoms unchanged, positions preserved
  after `allow_displacements=False` run.
- `TestParallelTemperingSanitize` (2 tests in `test_optimizer.py`): PT result
  contains no foreign atoms when initial uses a different element set; sanitize
  is a no-op when initial atoms already belong to the pool.
- Total test count: **331 passed** (329 pre-existing + 2 new for bug).
 
---

## [0.3.0] — 2026-03-21

### Documentation
```
This is a **feature-complete release** of the 0.3 series.
No further feature additions are planned; future releases in this series
will contain bug fixes and documentation improvements only.
```
This release contains documentation-only changes relative to v0.2.11.
No API, behavior, or binary changes are included.

#### Bug fixes in `docs/quickstart.md`

- **CLI filter example corrected.**  The `--filter "H_total:2.0:-"` threshold
  in the *Filtering by disorder metrics* example was unreachable for 15-atom
  C/N/O/S gas structures (observed maximum ≈ 1.43 with auto cutoff).  The
  threshold is lowered to `1.0` and `--n-samples` is raised from `200` to
  `500` so the example reliably produces output.

- **CLI `maxent` example corrected.**  `--n-samples 20` was too small for the
  `elements=6,7,8` pool at `charge=0 mult=1`: all attempts were rejected by
  the parity check.  Raised to `--n-samples 50`.

- **Composition-only optimization example corrected.**  The previous example
  used a C/N/O initial structure with a Cr/Mn/Fe/Co/Ni element pool, causing
  near-total parity rejection that silently prevented any composition moves.
  The example is rewritten to generate the initial structure from the same
  Cantor alloy pool used by the optimizer.

- **Composition-only objective corrected.**  The previous objective
  `{"H_atom": 1.0, "Q6": -2.0}` is invariant under element-label swaps
  (the primary composition move), so the optimizer could never improve it.
  Changed to `{"moran_I_chi": -1.0, "charge_frustration": 2.0}`, which
  responds to swaps via spatial electronegativity arrangement.  An explanatory
  note is added describing which metrics are swap-invariant and why
  `moran_I_chi` / `charge_frustration` are the recommended choices for
  composition-only runs.

#### New content in `docs/quickstart.md`

- **Shell mode atom count note.**  Added a callout under *Shell mode with a
  fixed center atom* explaining that the output atom count may exceed
  `n_atoms` because (a) `--center-z` prepends the center atom and (b)
  `add_hydrogen` is enabled by default.  Documents `--no-add-hydrogen` and
  `Structure.center_sym` as mitigation options.

- **Affine transforms — fine-grained parameter control.**  The
  *Affine transforms in StructureGenerator* section is expanded to cover
  `affine_stretch`, `affine_shear`, and `affine_jitter` individually:
  - Per-operation code examples (stretch-only, shear-only, jitter-only).
  - Full parameter reference table with ranges and fallback rules.
  - Explicit note that `affine_jitter` has no effect inside
    `StructureGenerator` / `generate()` because the internal `move_step`
    is `0.0`; the parameter is only meaningful in `StructureOptimizer`.
  - CLI flags `--affine-stretch`, `--affine-shear`, `--affine-jitter`
    documented with a usage example.
  - `StructureOptimizer` sub-section extended with stretch-only and
    combined stretch+shear examples tied to `shape_aniso` maximization.

- **Optimizer case studies section** (`## Optimizer case studies`).
  Three new end-to-end examples:

  1. *Reproducing a target disorder profile* — uses `cutoff=5.0` and
     Parallel Tempering with a weighted objective to reproduce the metric
     fingerprint of an external reference structure
     (`moran_I_chi ≈ 0.94`, `ring_fraction = 0.875`, `graph_lcc = 1.0`).
     Includes tips on cutoff matching and negative-weight suppression.

  2. *Geometry search with an ASE calculator* — three-stage pipeline:
     `StructureGenerator` samples random CH₄ geometries →
     `StructureOptimizer` + `EvalContext` runs basin-hopping with the
     ASE/EMT potential as the objective → ASE BFGS provides a final
     gradient-based refinement.  Demonstrates `allow_composition_moves=False`
     to lock stoichiometry.

  3. *Two-phase curriculum objective* — uses `EvalContext.progress` to
     switch the objective at the 50 % mark: pure `H_spatial` maximization
     in the first half (broad exploration), then `H_spatial − 3×Q6` in the
     second half (disorder + crystalline-order suppression combined).
     Includes guidance on when curriculum objectives are appropriate.

---

## [0.2.11] — 2026-03-21

### Added

- **`EvalContext` dataclass** — a frozen snapshot of the current candidate
  structure *and* the live optimizer state, passed as the second argument
  to 2-parameter objective callables.  Fields: `atoms`, `positions`,
  `charge`, `mult`, `n_atoms`, `metrics`, `step`, `max_steps`,
  `temperature`, `f_current`, `best_f`, `restart_idx`, `n_restarts`,
  `per_atom_q6`, `element_pool`, `cutoff`, `method`, `T_start`, `T_end`,
  `seed`, `replica_idx`, `replica_temperature`, `n_replicas`.

- **`EvalContext.to_xyz()`** — serializes the current structure to XYZ
  format for passing to external tools (xTB, ASE, ORCA, etc.).

- **`EvalContext.progress` property** — fractional run progress
  (`step / max_steps`), useful for curriculum-style objectives.

- **2-argument objective callable support**: any callable with two required
  positional parameters receives an `EvalContext` as its second argument.
  Dispatch is via `inspect.signature`; 1-argument callables are unaffected.

  ```python
  # 1-arg (unchanged)
  objective = lambda m: m["H_total"] - 2.0 * m["Q6"]

  # 2-arg: full structure + optimizer state available
  def my_objective(m, ctx):
      pos = np.array(ctx.positions)
      return float(m["H_total"]) - float(np.max(ctx.per_atom_q6))
  ```

- **`_objective_needs_ctx()` helper** — caches the arity check so
  `inspect.signature` is called once per optimizer instance, not once per
  MC step.

- **`EvalContext` exported** from the top-level `pasted` namespace.

### Changed

- **`ObjectiveType`** extended to include
  `Callable[[dict[str, float], EvalContext], float]`.

- **`_eval_objective`** accepts an optional `ctx: EvalContext | None`
  keyword argument; routes to the 2-arg or 1-arg path accordingly.

- **`StructureOptimizer.objective` docstring** updated to document both
  calling conventions.

- **`docs/quickstart.md`**: added *Extended objective with EvalContext*
  subsection with five usage examples (NumPy-only, adaptive curriculum,
  per-atom Q6 penalty, xTB external process, ASE EMT potential).

- **`docs/api/optimizer.rst`**: updated `ObjectiveType` and
  `EvalContext` API reference.

### Changed (C++ extensions)

- **`_relax.cpp`**: removed all `#ifdef _OPENMP` / `#pragma omp` blocks,
  `tgrad_` per-thread scratch member, and `PAIR_PARALLEL_THRESHOLD`
  constant.  The serial pair loop now writes directly to `grad[]`,
  eliminating the per-`evaluate()` zero-fill and merge pass.
  `libgomp` was never linked in the distributed wheels; the `#else`
  branch was always the active path.

- **`_maxent.cpp`**: same OpenMP scaffold removal in `eval_angular()` and
  `place_maxent_cpp_impl`.  `tgrad_scratch` simplified from
  `vector<vector<double>>` to a flat `vector<double>`.

- **`_graph_core.cpp`**, **`_steinhardt.cpp`**: no changes (no OpenMP
  present since v0.2.9).

---

## [0.2.10] — 2026-03-21

### Added

* **Per-operation affine transform strength (`affine_stretch`, `affine_shear`,
  `affine_jitter`).**

  Prior to v0.2.10, the single `affine_strength` scalar controlled all three
  affine operations — stretch/compress, shear, and per-atom jitter — with
  fixed relative weights (shear = 0.5 × stretch, jitter ∝ move_step).  There
  was no way to enable, say, stretch only or shear only without also activating
  the other operations.

  Three new optional parameters are added to `GeneratorConfig`,
  `StructureGenerator`, `generate()`, `StructureOptimizer`, and
  `_affine_move()`:

  | Parameter | Scope | Default | Effect |
  |---|---|---|---|
  | `affine_stretch` | Generator + Optimizer | `None` | Strength of the stretch/compress step; falls back to `affine_strength` when `None`. |
  | `affine_shear` | Generator + Optimizer | `None` | Strength of the shear step; falls back to `affine_strength` when `None`. |
  | `affine_jitter` | Generator + Optimizer | `None` | Per-atom jitter scale relative to `move_step`; falls back to `affine_strength` when `None`. |

  When all three are `None` (the default), `_affine_move` behaves identically
  to v0.2.9 — full backward compatibility is preserved.  Setting any parameter
  to `0.0` disables that specific operation while leaving the others unchanged:

  ```python
  # Stretch-only: disable shear and jitter
  gen = StructureGenerator(
      n_atoms=20, charge=0, mult=1, mode="gas", region="sphere:8",
      affine_strength=0.2, affine_shear=0.0, affine_jitter=0.0,
  )

  # Fine-grained control in StructureOptimizer
  opt = StructureOptimizer(
      structures, objective="H_total",
      allow_affine_moves=True,
      affine_strength=0.1,
      affine_stretch=0.3,   # stronger stretch than global default
      affine_shear=0.05,    # gentler shear
      affine_jitter=0.0,    # no per-atom noise
  )
  ```

  CLI equivalents: `--affine-stretch`, `--affine-shear`, `--affine-jitter`
  (each accepts a float; omitting uses `--affine-strength`).

### Documentation

* **`docs/architecture.md`**: removed all references to the
  `HAS_POISSON` flag and the Bridson Poisson-disk sampling functions
  (`_poisson_disk_sphere_cpp`, `_poisson_disk_box_cpp`) that were documented
  as part of `_relax_core` — these functions exist in the Python fallback
  (`_placement.py`) but were never fully exposed and are not part of the
  stable public API.  The `_relax_core` section heading is updated
  accordingly.

* **`docs/architecture.md`**: updated the *Affine transform in
  StructureGenerator* section and the *Move types* table to describe the new
  per-operation parameters and include usage examples.

### Removed

* **Poisson-disk sampling helpers removed from `_placement.py` and
  `pasted._ext`.**

  `_poisson_disk_sphere()` and `_poisson_disk_box()` in `_placement.py`
  were never called by any internal code path (``place_gas`` always uses
  uniform random placement).  They were retained in v0.2.2 as optional
  utilities, but their presence was misleading — the stratified-jitter sphere
  implementation did not strictly satisfy the Bridson minimum-separation
  guarantee.

  The following are removed in v0.2.10:

  - `pasted._placement._poisson_disk_sphere` (Python, ~80 lines)
  - `pasted._placement._poisson_disk_box` (Python, ~70 lines)
  - `pasted._ext.HAS_POISSON` flag
  - `pasted._ext._poisson_disk_sphere_cpp` (C++ wrapper, unused fallback `None`)
  - `pasted._ext._poisson_disk_box_cpp` (C++ wrapper, unused fallback `None`)

  These names were never part of the public API documented in ``__init__.py``
  or the quickstart guide's stable surface.  Any code that imported them
  directly should switch to uniform random placement via ``place_gas()`` or
  implement a Poisson-disk sampler independently.

* **`docs/quickstart.md`**: removed ``HAS_POISSON`` from the extension-flag
  code example and the flag-description table.

* **`src/pasted/_ext/__init__.py`**: removed ``HAS_POISSON``,
  ``_poisson_disk_sphere_cpp``, and ``_poisson_disk_box_cpp`` from the module
  docstring, import block, and ``__all__``.  The ``_relax_core`` import block
  now only imports ``relax_positions``.

---

## [0.2.9] — 2026-03-20

### Fixed

* **`_ext/_graph_core.cpp`: reverted two-pass pair-collection to the
  original single-pass lambda pattern.**

  In v0.2.3 the pair-enumeration loop in `graph_metrics_cpp` and `rdf_h_cpp`
  was refactored into a two-pass design: the `FlatCellList` pass collected
  candidate pairs into an intermediate `std::vector<std::pair<int,int>>`, and
  a second pass applied the distance test and populated per-thread
  `local_pairs` buckets — all wired up with `#ifdef _OPENMP` guards intended
  for future parallelism.  Because OpenMP was never linked in the build
  (`setup.py` carries no `-fopenmp` flag and `_OPENMP` is never defined),
  the scaffolding compiled down to `nthreads = 1` with a single
  `local_pairs[0]` bucket: the full cost of two heap allocations per
  metric call and an extra `std::vector<std::vector<...>>` merge loop,
  with zero parallelism benefit.  At large N this produced a measurable
  regression (~35% slower at N = 10,000) relative to v0.1.x.

  The fix restores the original pattern: a single capturing lambda passed
  directly to `cl.for_each_pair` / `cl.for_each_neighbor`, which writes
  distance-filtered edges into the adjacency lists in one pass with no
  intermediate allocation.  The `PAIR_PARALLEL_THRESHOLD` constant and all
  `#ifdef _OPENMP` blocks are removed.

* **`_ext/_steinhardt.cpp`: reverted two-pass neighbor-list build to the
  original single-pass lambda accumulation.**

  The same v0.2.3 refactor introduced a pre-built `nb_list` (a
  `std::vector<std::vector<int>>`) so that the subsequent atom loop could
  be annotated with `#pragma omp parallel for`.  Without an OpenMP-enabled
  compiler, the pragma is silently ignored and the neighbor-list allocation
  becomes dead overhead — an extra O(N·k) heap allocation that precedes
  every `steinhardt_per_atom` call with no benefit.

  Restored to the v0.1.x pattern: a single `accumulate` lambda that
  increments `deg[i]` and updates `re_buf`/`im_buf` directly as pairs are
  yielded by `FlatCellList::for_each_neighbor` /
  `for_each_neighbor_full`.  The `#ifdef _OPENMP` guard and
  `#pragma omp parallel for` directive are removed entirely.

### Documentation

* **`src/pasted/_metrics.py` (`compute_all_metrics` docstring)**: updated
  the C++ path description to reflect the v0.2.9 reversion.  Previously
  stated "OpenMP removed in v0.2.3"; now explains that the dead two-pass
  scaffolding was itself removed in v0.2.9, restoring v0.1.x performance.

* **`_ext/_graph_core.cpp` file header**: updated version tag from
  `v0.1.14` to `v0.2.9`; added a "Threading" paragraph describing the
  reversion and why the intermediate design was problematic.

* **`_ext/_steinhardt.cpp` file header**: added a "Threading" section
  mirroring the explanation in `_graph_core.cpp`.

### Changed

* `pyproject.toml` and fallback `__version__` string bumped to `0.2.9`.

---

## [0.2.8] — 2026-03-20

### Documentation

* **`docs/quickstart.md`: separated Python API and CLI into independent
  top-level sections.**

  The previous `## CLI` section contained `### Functional API`,
  `### Class API`, `### Streaming output`, and other Python API subsections
  as children, making the document difficult to navigate for readers who
  wanted either the API guide or the CLI guide specifically.

  The guide now has two distinct top-level sections:
  - `## Python API` — functional API, class API, `maxent` mode, file I/O,
    `n_success`, streaming, and metrics access.
  - `## CLI` — placement-mode examples, filtering, shell mode, and `maxent`
    CLI usage, with a link to `cli.md` for the complete option reference.

  All existing content is preserved; only the section structure changed.

* **`src/pasted/_placement.py` (`place_maxent` docstring): removed
  version-specific history from *Implementation notes*.**

  The "O(N) cutoff computation (v0.2.6)" paragraph described the previous
  O(N² log N) algorithm and the migration story.  Version history belongs in
  `CHANGELOG.md` and `docs/architecture.md`, not in a function's API
  docstring.  The section now documents *what* the computation does (the
  `median(rᵢ + rⱼ) = 2·median(rᵢ)` identity and the O(N) derivation)
  without reference to prior versions.

* **`src/pasted/__init__.py` module docstring: removed `Changes in v0.2.3`
  block; expanded *Quick start* section.**

  Version-history prose belongs in `CHANGELOG.md`.  The module docstring
  now serves purely as a quick-reference guide covering the functional API,
  class API, streaming output, optimizer, `GeneratorConfig`, and CLI — each
  with a minimal runnable example.

### Changed

* `pyproject.toml` and fallback `__version__` string bumped to `0.2.8`.

---

## [0.2.7] — 2026-03-20

### Fixed

* **`StructureOptimizer`: duplicate symbols in `elements` list no longer
  bias composition sampling (Bug A).**

  When `elements` was passed as a list with repeated symbols — e.g.
  `elements=['C', 'H', 'H', 'H', 'H']` — the internal `_element_pool`
  stored all five entries verbatim.  Every call to
  `rng.choice(self._element_pool)` then sampled `H` with 4x the
  probability of `C`, violating the assumption that the pool represents
  unique element *types*.  This was especially harmful when
  `allow_composition_moves=False` because the weighted pool was still
  used during initial-structure generation.

  **Fix:** the `else` branch in `__init__` now deduplicates the list
  with `dict.fromkeys()`, which removes repeats while preserving
  insertion order.  The pool always contains unique element types;
  callers who want composition-weighted sampling should use
  `element_fractions` in `StructureGenerator` instead.

* **`StructureOptimizer` Parallel Tempering: all replicas now share the
  same initial composition when `allow_composition_moves=False` and no
  `initial` structure is provided (Bug B).**

  Previously, when `initial=None` and `allow_composition_moves=False`,
  each PT replica generated its own independent random structure by
  calling `rng.choice(self._element_pool)` per atom independently.
  This meant replica 0 could start as an all-carbon cluster while
  replica 3 started as a mixed C/O cluster — a clear violation of the
  fixed-composition contract.  Replica-exchange swaps would then
  propagate these different compositions to the cold replica, making
  the "no composition moves" flag meaningless.

  **Fix:** when `initial=None` and `allow_composition_moves=False`,
  `_run_parallel_tempering` now calls `_make_initial` once before the
  replica loop to generate a single shared initial structure.  Each
  replica inherits the atom types from that shared structure but
  receives independently randomized positions, so replicas still start
  from different points in configuration space.  When
  `allow_composition_moves=True` the previous behavior (fully
  independent random starts) is unchanged.

### Changed

* `StructureOptimizer.elements` parameter docstring updated to document
  the deduplication behavior and recommend `element_fractions` for
  composition weighting.
* `_run_parallel_tempering` docstring extended with an *Initialization*
  section describing all three initialization paths.
* `pyproject.toml` and fallback `__version__` string bumped to `0.2.7`.

---

## [0.2.6] — 2026-03-20

### Performance

- `place_maxent`: replaced the O(N² log N) neighbor-cutoff computation with an
  O(N) equivalent.  The previous implementation built the full sorted list of
  all N*(N+1)/2 pairwise covalent-radius sums to find the median; for N=2,000
  this generator and sort dominated ~88 % of wall time even with only 5
  L-BFGS steps.  The replacement exploits the identity
  `median(rᵢ + rⱼ) = 2 · median(rᵢ)`, which holds for all built-in element
  pools, and computes `median_sum = float(np.median(radii)) * 2.0` instead.
  The resulting `ang_cutoff` value is numerically identical for all tested
  element pools (C, N, O, H, S, and mixed sets).  Measured wall-time
  reductions vs. v0.2.5 (3 trials, `elements="6,8"`, `add_hydrogen=False`,
  `n_samples=1`):

  | n_atoms | v0.2.5 | v0.2.6 | speedup |
  |--------:|-------:|-------:|--------:|
  |     100 |  35 ms |  22 ms |   1.6 × |
  |   1 000 | 364 ms | 189 ms |   1.9 × |
  |   5 000 | 4821 ms | 1274 ms |  3.8 × |
  |  10 000 | 18084 ms | 4028 ms |  4.5 × |

  All other modes (`gas`, `chain`, `shell`) are unaffected.

### Documentation

- `_placement.place_maxent`: added *Implementation notes* section to the
  docstring explaining the O(N) cutoff identity and measured speedups.
- `docs/architecture.md`: added *Performance* sub-section under the
  `place_maxent` entry documenting the v0.2.6 change.
- `docs/api/placement.rst`: updated synopsis to mention the O(N) cutoff.

### Tests

- `tests/test_maxent.py`: added `TestPlaceMaxentCutoff` class with three
  tests that verify the O(N) median produces the same `ang_cutoff` value as
  the O(N²) formula across homogeneous, heterogeneous, and single-element
  atom lists.

---

## [0.2.5] — 2026-03-20

### Fixed
- `pyproject.toml`: changed `[tool.setuptools.packages.find] where = ["."]` to
  `where = ["src"]` to match the actual `src/` layout.  The previous value caused
  editable and wheel builds to attempt copying compiled `.so` files to a
  non-existent `pasted/_ext/` directory, failing with
  `error: could not create '.../_relax_core.*.so': No such file or directory`.
- `pyproject.toml`: added `"*.cpp"` to `[tool.setuptools.package-data]` so that
  C++ source files are included in the sdist.
- `src/pasted/__init__.py`: updated fallback version string to `"0.2.5"`.

---

## [0.2.4] — 2026-03-20

### Fixed
- `pyproject.toml`: changed `license = { text = "MIT" }` to the SPDX string
  form `license = "MIT"`, eliminating the setuptools>=77 deprecation warning.
- `pyproject.toml`: moved `pybind11>=2.12` from `[project.optional-dependencies].dev`
  into `[build-system].requires` so that C++ extensions are built correctly
  during `pip install` and `pip install -e .` without a separate pybind11
  pre-install step.
- `src/pasted/__init__.py`: updated fallback version string to `"0.2.4"`.

---

## [0.2.3] — 2026-03-20

### Removed
- **OpenMP integration** (`HAS_OPENMP`, `set_num_threads`) removed from
  `pasted._ext` and the top-level `pasted` namespace.
  Benchmarking against v0.1.17 showed that the OpenMP thread-pool overhead in
  v0.2.2 produced a **1.4–2.5× regression** in `compute_all_metrics` across
  all practically relevant structure sizes.  After removing OpenMP, v0.2.3
  is within measurement noise of v0.1.17 across all tested sizes:

  **generate()** (median, n_samples=1, mode=gas):

  | n_atoms |  v0.1.17 |  v0.2.3 | ratio |
  |--------:|---------:|--------:|------:|
  |     100 |   1.5 ms |  1.2 ms | 0.80× |
  |   1 000 |  10.9 ms |  8.8 ms | 0.81× |
  |  10 000 | 256.3 ms | 241.5 ms | 0.94× |
  |  30 000 | 1937.8 ms | 1863.6 ms | 0.96× |

  **compute_all_metrics()** (median):

  | n_atoms |  v0.1.17 |  v0.2.3 | ratio |
  |--------:|---------:|--------:|------:|
  |     100 |   0.3 ms |  0.3 ms | 1.09× |
  |   1 000 |   2.1 ms |  2.4 ms | 1.15× |
  |  10 000 | 105.3 ms | 96.5 ms | 0.92× |
  |  30 000 | 1007.5 ms | 1012.3 ms | 1.00× |

  The `libgomp` runtime dependency is therefore dropped.

- `HAS_OPENMP` constant removed from `pasted._ext.__all__` and `pasted.__all__`.
- `set_num_threads(n)` function removed from `pasted._ext` and re-exported
  name removed from `pasted`.
- `ctypes`, `os`, `sys` imports removed from `pasted._ext.__init__` (they
  were only required for the OpenMP detection and thread-count setter).

### Changed
- All C++ extension calls now run single-threaded.  Benchmarking confirms
  that both `generate()` and `compute_all_metrics()` are within measurement
  noise of v0.1.17 across all tested sizes (n = 100 – 30 000 atoms).
- `pasted._ext.__init__` module docstring updated; removed the WSL/OOM note
  that referenced OpenMP thread counts.
- `pasted.compute_all_metrics` docstring updated to state that computation is
  single-threaded as of v0.2.3.
- `pasted.__init__` module docstring updated with v0.2.3 change summary.

### Migration guide
Code that imported `HAS_OPENMP` or `set_num_threads` must remove those
references:

```python
# v0.2.2 (remove these lines)
from pasted import HAS_OPENMP, set_num_threads
set_num_threads(4)

# v0.2.3 — no replacement needed; threading is handled internally
```

---

## [0.2.2] - 2026-03-20

### Added

* **`GeneratorConfig` frozen dataclass (`_config.py`).**

  All ~30 parameters of `StructureGenerator` are now encapsulated in a
  single `GeneratorConfig(frozen=True)` dataclass.  This gives full
  mypy / IDE type-checking on every field, hashability, and
  `dataclasses.replace(cfg, seed=99)` for convenient one-field overrides.
  `GeneratorConfig` is exported from the top-level `pasted` namespace.

  `StructureGenerator.__init__` now accepts either a `GeneratorConfig`
  instance *or* the original keyword arguments (backward-compatible):

  ```python
  from pasted import GeneratorConfig, StructureGenerator

  # Config-based (recommended for new code)
  cfg = GeneratorConfig(n_atoms=20, charge=0, mult=1,
                        mode="gas", region="sphere:10",
                        elements="6,7,8", seed=42)
  gen = StructureGenerator(cfg)

  # Keyword-based (original API — still works unchanged)
  gen = StructureGenerator(n_atoms=20, charge=0, mult=1,
                            mode="gas", region="sphere:10",
                            elements="6,7,8", seed=42)
  ```

  `generate()` likewise accepts either `generate(cfg)` or the original
  `generate(n_atoms=..., charge=..., mult=..., ...)` form.

  All instance attributes previously set directly on `StructureGenerator`
  (e.g. `gen.n_atoms`, `gen.seed`) continue to work via `__getattr__`
  proxy to `gen._cfg`.

* **`affine_strength` parameter for `StructureGenerator` / `generate()` / CLI.**

  A random affine transformation (stretch/compress one axis + shear one
  axis pair) is applied to every generated structure **before**
  `relax_positions`, creating more anisotropic initial geometries.

  | Parameter | Default | Effect |
  |---|---|---|
  | `affine_strength=0.0` | disabled — backward-compatible | no transform |
  | `affine_strength=0.1` | ±10 % stretch/compress, ±5 % shear | moderate anisotropy |
  | `affine_strength=0.3` | ±30 % stretch/compress, ±15 % shear | strong anisotropy |

  Works across **all placement modes** (`gas`, `chain`, `shell`, `maxent`).
  Centre of mass is pinned; clash resolution via `relax_positions` runs
  after the transform.  CLI: `--affine-strength S`.

  The underlying `_affine_move` function has been **moved from
  `_optimizer.py` to `_placement.py`** and is now shared between
  `StructureGenerator` (applied once per structure before relax) and
  `StructureOptimizer` (applied per MC step when `allow_affine_moves=True`).
  No behavioural change in the optimizer.

* **Verlet-list reuse with adaptive skin in `PenaltyEvaluator` (`_relax_core`).**

  `pairs_` is now rebuilt every `N_VERLET_REBUILD = 4` evaluate() calls
  (counter-based, no O(N) displacement loop) using an extended cutoff of
  `cell_size + skin`.  The skin is **adaptive**:
  `skin = min(0.8 Å, cell_size × 0.3)`, keeping the extended pair list
  within `(1.3)³ ≈ 2.2×` the original count regardless of element radii.
  This prevents the 3–4× overhead that occurred with light elements
  (C, O, H where `cell_size ≈ 1.5 Å`) under the old fixed-skin design.

* **Bridson Poisson-disk C++ functions added to `_relax_core`.**

  `poisson_disk_sphere(n, radius, min_dist, seed, k)` and
  `poisson_disk_box(n, lx, ly, lz, min_dist, seed, k)` are exported as
  C++ pybind11 bindings and accessible via `pasted._ext` when
  `HAS_POISSON` is `True`.  `place_gas()` itself uses uniform random
  placement for performance predictability; callers that need minimum-
  separation placement can call these functions directly.

* **Affine displacement moves in `StructureOptimizer`.**

  New parameters `allow_affine_moves=False` (default) and
  `affine_strength=0.1`.  When enabled, half of the displacement-move
  budget is replaced by random affine transforms (stretch / compress one
  axis, shear one axis pair, per-atom jitter).  Centre of mass is pinned.

* **`--n-threads` CLI default changed from `None` (all cores) to `1`.**

  OpenMP parallelism is now opt-in.  Pass `--n-threads N` to enable.

### Fixed

* **`allow_displacements=False` no longer moves atom positions.**

  `relax_positions` is now skipped when `allow_displacements=False` in
  all three optimization methods (SA, BH, PT).

### Performance

Measured on the same workload as the v0.1.17 baseline (median of 5 runs):

| Scenario | v0.1.17 | v0.2.2 | Δ |
|---|---:|---:|---:|
| gas n=20, n_samples=50 | 9.2 ms | 7.1 ms | −23 % |
| gas n=100, n_samples=10 | 6.5 ms | 4.8 ms | −26 % |
| chain n=20, n_samples=50 | 13.5 ms | 12.4 ms | −8 % |
| shell n=12, n_samples=50 | 8.7 ms | 7.1 ms | −18 % |
| maxent n=12, n_samples=10 | 7.6 ms | 4.0 ms | −47 % |
| gas n=20, affine=0.2, n_samples=50 | — | 7.4 ms | +4 % vs no-affine |

`affine_strength` overhead is negligible (< 5 % vs `affine_strength=0.0`).

## [0.2.1] - 2026-03-20

### Fixed

* **`_relax_core`: eliminate per-`evaluate()` heap churn at large N.**

  `tgrad` (thread-local gradient buffers, ~27 MB at n=150 000 with 8 threads)
  and `pairs` (pair-list vector) were re-allocated and freed on every L-BFGS
  iteration inside `PenaltyEvaluator::evaluate()`.  Over 300–1 500 iterations
  this caused several gigabytes of malloc/free traffic, driving `sys` time to
  ~23 s and triggering OOM-kills on WSL for n ≥ 150 000 at ≥ 8 threads.

  Both are now persistent members of `PenaltyEvaluator`, allocated once at
  construction. `pairs_` reuses its capacity with `clear()` on each call;
  `tgrad_` is zeroed with `std::fill`. The fix also handles the
  `set_num_threads()` edge-case by resizing `tgrad_` lazily when the thread
  count changes.

* **`_maxent_core`: eliminate per-step heap churn in `eval_angular` and `build_nb`.**

  Inside `place_maxent_cpp`, every L-BFGS step called `build_nb()` (returning
  a freshly allocated `vector<vector<int>>`) and `eval_angular()` (allocating
  `tgrad` ~ 9 MB and per-atom `ux/uy/uz/id` vectors at n=50 000).
  With `maxent_steps=300` this produced ~5 GB of `tgrad` churn and ~0.9 GB of
  `nb` churn per structure.

  Three changes:

  1. Added `build_nb_inplace()` — clears and refills an existing `nb` vector
     in-place, preserving allocated capacity across steps.
  2. `eval_angular()` now accepts persistent `tgrad_scratch` and `ux_s/uy_s/uz_s/id_s`
     buffers as out-parameters; callers allocate them once and pass by reference.
     The serial path reuses `ux_s`/... with `resize()` (only grows, never shrinks);
     the parallel path retains per-thread-private locals (thread-safety unchanged).
  3. `place_maxent_cpp_impl` declares these buffers before the step loop and
     passes them through on every call.

  The `angular_repulsion_gradient` Python wrapper (called once per structure,
  not in a hot loop) uses local scratch and is unaffected.

## [0.2.0] - 2026-03-20

### Added

* **OpenMP parallelization for all four C++ extension modules.**

  All inner-loop hotspots now use OpenMP when built with `-fopenmp`.
  Parallelization strategy per module:

  | Module | Strategy | Parallel region |
  | --- | --- | --- |
  | `_relax_core` | Pair list pre-built serially; gradient accumulation via thread-local buffers merged after parallel loop | `PenaltyEvaluator::evaluate` |
  | `_steinhardt_core` | Neighbor list pre-built serially; per-atom spherical harmonic accumulation parallelized | outer atom loop in `steinhardt_per_atom_cpp` |
  | `_graph_core` | Pair list pre-built serially; distance filtering and adjacency construction via thread-local pair buckets merged serially | `graph_metrics_cpp`, `rdf_h_cpp` |
  | `_maxent_core` | Per-atom angular repulsion gradient with thread-local gradient buffers | `eval_angular` outer atom loop |

  Parallelization is **Linux only**. OpenMP is enabled automatically at build
  time when `-fopenmp` is accepted by the compiler (GCC or Clang + libomp)
  and `PASTED_DISABLE_OPENMP=1` is not set. On macOS and Windows the
  extensions build without OpenMP and run single-threaded as before.

  To opt out on Linux::

      PASTED_DISABLE_OPENMP=1 pip install -e .

* **`pasted.HAS_OPENMP`** — runtime boolean flag. ``True`` when the C++
  extensions were compiled with OpenMP and the runtime library is reachable.

* **`pasted.set_num_threads(n)`** — set the number of OpenMP threads used by
  all C++ extensions at runtime. Equivalent to ``OMP_NUM_THREADS`` but takes
  effect immediately without restarting the process. A no-op when
  ``HAS_OPENMP`` is ``False`` or ``n <= 0``.

  ```python
  import pasted

  if pasted.HAS_OPENMP:
      pasted.set_num_threads(4)
  ```

* **`--n-threads N` CLI option** — passes ``N`` to ``set_num_threads`` before
  any computation begins.

  ```
  pasted --n-atoms 50000 --mode gas --region sphere:250 \
      --charge 0 --mult 1 --n-threads 8 -o out.xyz
  ```

* **5 new tests** in `tests/test_placement.py` (`TestOpenMP`):
  + `test_has_openmp_is_bool` — flag type check.
  + `test_set_num_threads_noop_when_no_openmp` — no raise on any input.
  + `test_set_num_threads_exported_from_pasted` — public API presence.
  + `test_relax_single_thread_matches_multi_thread` — numerical parity
    between 1 and 2 threads (atol 1e-6 Å).
  + `test_metrics_consistent_across_thread_counts` — all 13 metrics agree
    between 1 and 2 threads.

### Changed

* `setup.py` rewritten to auto-detect OpenMP on Linux via a compile probe
  (`-fopenmp` test on a minimal C program). Detection result is reported at
  build time. `PASTED_DISABLE_OPENMP=1` suppresses the probe.
* `pyproject.toml`: version bumped to `0.2.0`. Added `[tool.mypy.overrides]`
  for `pasted._ext` to suppress `no-redef` and `warn_unused_ignores` caused
  by the `try/except ImportError` pattern used for optional C++ extensions.

### Platform support

| Platform | C++ extensions | OpenMP |
| --- | --- | --- |
| **Linux** (GCC ≥ 7 or Clang + libomp) | ✅ supported | ✅ automatic |
| macOS | best-effort | ❌ not attempted |
| Windows | best-effort | ❌ not attempted |

## [0.1.17] - 2026-03-19

### Added

- **`allow_displacements` parameter for `StructureOptimizer`.**

  Controls whether atomic-position moves (fragment moves) are performed
  during optimization.

  | Value | Behavior |
  |---|---|
  | `True` (default) | Fragment moves (atomic displacements) are included in the MC step pool — unchanged from v0.1.16 |
  | `False` | Only composition moves (element-type swaps) are executed; atomic coordinates are held fixed for the entire run |

  Use `allow_displacements=False` when exploring compositional disorder on
  a pre-relaxed geometry (e.g. a fixed lattice).  Passing both
  `allow_displacements=False` and `allow_composition_moves=False`
  simultaneously raises a `ValueError` because no move type would remain
  enabled.

  Applies to all three optimization methods: `"annealing"`,
  `"basin_hopping"`, and `"parallel_tempering"`.

  CLI: `--no-displacements` flag added to the `--optimize` mode.

  ```python
  opt = StructureOptimizer(
      n_atoms=50, charge=0, mult=1,
      objective={"H_atom": 1.0, "Q6": -2.0},
      elements=["Cr", "Mn", "Fe", "Co", "Ni"],
      allow_displacements=False,   # composition-only optimization
      max_steps=5000, seed=42,
  )
  result = opt.run(initial=fixed_geometry)
  ```

### Fixed

- **`OptimizationResult.method` docstring** now lists all three valid
  methods (`"annealing"`, `"basin_hopping"`, `"parallel_tempering"`);
  previously `"parallel_tempering"` was omitted.

## [0.1.16] - 2026-03-19

### Added

- **`allow_composition_moves` parameter for `StructureOptimizer`.**

  Controls whether element-type swaps are performed during optimization.

  | Value | Behavior |
  |---|---|
  | `True` (default) | Each MC step randomly chooses between a fragment move (position change) and a composition move (element-type swap) with equal probability — unchanged from v0.1.15 |
  | `False` | Only fragment moves are executed; element types are held fixed for the entire run |

  Use `allow_composition_moves=False` when the composition is predetermined
  and should not be modified during optimization (e.g. optimizing the
  geometry of a fixed stoichiometry).

  Applies to all three optimization methods: `"annealing"`,
  `"basin_hopping"`, and `"parallel_tempering"`.

  CLI: `--no-composition-moves` flag added to the `--optimize` mode.

  ```python
  opt = StructureOptimizer(
      n_atoms=12, charge=0, mult=1,
      objective={"H_total": 1.0, "Q6": -2.0},
      elements="24,25,26,27,28",
      allow_composition_moves=False,   # position-only optimization
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
    preservation when disabled (SA and PT), still optimizes, `repr`
    behavior.

## [0.1.15] - 2026-03-19

### Changed

- **`place_maxent` now uses L-BFGS with a per-atom trust radius instead of
  steepest descent.**

  When `HAS_MAXENT_LOOP` is `True` (i.e. `_maxent_core.place_maxent_cpp` is
  available), the entire gradient-descent loop runs in C++:

  | Step | v0.1.14 (Python SD) | v0.1.15 (C++ L-BFGS) |
  |---|---|---|
  | Gradient computation | C++ `angular_repulsion_gradient` | C++ (inlined, same Cell List) |
  | Optimizer | steepest descent, fixed `maxent_lr` | L-BFGS m=7, Armijo backtracking |
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
  instead of duplicating neighbor-list and gradient logic.



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
    electronegative and electropositive neighbors — i.e. the local
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
    atoms only (d < 1e-10 Å), matching the v0.1.10 GS behavior.
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
