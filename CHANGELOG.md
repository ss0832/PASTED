# Changelog

All notable changes to PASTED are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

## [0.2.2] — 2026-02-14

### Added
- OpenMP support: `HAS_OPENMP` flag and `set_num_threads(n)` for controlling
  the number of threads used by C++ extensions.
- `n_replicas` parameter to `StructureOptimizer` for parallel-tempering
  replica exchange.
- `allow_affine_moves` / `affine_strength` parameters to `StructureOptimizer`.
- `GenerationResult` dataclass wrapping the list of structures returned by
  `generate()`; provides `.structures`, `.n_attempted`, `.n_passed`, and
  `.summary`.

### Changed
- `generate()` now requires `region` when `mode="gas"` (previously defaulted
  silently to an undefined region).
- `generate()` return type changed from `list[Structure]` to `GenerationResult`.
  Use `.structures` to access the list: `result.structures`.
- `StructureOptimizer.optimize()` renamed to `.run()`.

---

## [0.2.1] — 2026-01-28

### Changed
- `_relax_core` and `_maxent_core` refactored to eliminate repeated heap
  allocation inside their hot L-BFGS loops.  Gradient scratch buffers and
  neighbor lists are now persistent, fixing OOM on WSL at n ≥ 150 000 atoms.

---

## [0.2.0] — 2026-01-10

### Added
- `GeneratorConfig` dataclass for reusable generator configurations.
- `parse_objective_spec` utility for parsing optimizer objective strings.
- `StructureOptimizer` with `"annealing"`, `"basin_hopping"`, and
  `"parallel_tempering"` methods.

### Changed
- `_graph_core` C++ extension refactored to use a `FlatCellList` for O(N·k)
  pair enumeration; replaces the previous O(N²) distance-matrix path.
- `rdf_h_cpp` added to `_graph_core` for O(N·k) spatial entropy and RDF
  deviation.

---

## [0.1.17] — 2025-12-05

### Added
- `element_fractions`, `element_min_counts`, `element_max_counts` parameters
  to `generate()` for fine-grained composition control.
- `maxent_steps`, `maxent_lr`, `maxent_cutoff_scale`, `trust_radius`,
  `convergence_tol` parameters for `mode="maxent"` tuning.

### Changed
- `generate()` returns `GenerationResult` (backport of the v0.2.0 wrapper).

---

## [0.1.11] — 2025-09-18

### Added
- Initial public release on PyPI.
- Four placement modes: `gas`, `chain`, `shell`, `maxent`.
- 13 disorder metrics: `H_atom`, `H_spatial`, `H_total`, `RDF_dev`,
  `shape_aniso`, `Q4`, `Q6`, `Q8`, `graph_lcc`, `graph_cc`, `ring_fraction`,
  `charge_frustration`, `moran_I_chi`.
- CLI (`pasted`) with `--filter` and streaming XYZ output.
