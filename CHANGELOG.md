# Changelog

All notable changes to PASTED are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.2.4] — 2026-03-20

### Fixed
- `pyproject.toml`: `license = { text = "MIT" }` を SPDX 文字列形式 `license = "MIT"` に変更
  （setuptools>=77 の非推奨警告を解消）。
- `pyproject.toml` `[build-system].requires` に `pybind11>=2.12` を追加。
  これまで `[project.optional-dependencies].dev` にしか存在せず、
  sdist ビルド時に C++ 拡張がビルドされない問題を修正。
- `src/pasted/__init__.py` のフォールバックバージョン文字列を `0.2.4` に更新。

---

## [0.2.3] — 2026-03-20

### Removed
- **OpenMP integration** (`HAS_OPENMP`, `set_num_threads`) removed from
  `pasted._ext` and the top-level `pasted` namespace.
  Benchmarking against v0.1.17 revealed that the thread-pool overhead in
  `compute_all_metrics` produced a **1.4–2.5× performance regression** across
  all practically relevant structure sizes (n = 30 – 30 000 atoms):

  | n_atoms | v0.1.17 | v0.2.2 | ratio |
  |--------:|--------:|-------:|------:|
  |     100 |  2.0 ms |  5.0 ms | 2.5× slower |
  |   1 000 | 21.4 ms | 36.0 ms | 1.7× slower |
  |  10 000 |  254 ms |  377 ms | 1.5× slower |
  |  30 000 |  909 ms | 1301 ms | 1.4× slower |

  Batch processing (up to 1 000 structures in a loop) did not recover the
  regression: thread-pool startup and data-serialisation costs dominate for
  all structure sizes tested.  The `libgomp` runtime dependency is therefore
  dropped.

- `HAS_OPENMP` constant removed from `pasted._ext.__all__` and `pasted.__all__`.
- `set_num_threads(n)` function removed from `pasted._ext` and re-exported
  name removed from `pasted`.
- `ctypes`, `os`, `sys` imports removed from `pasted._ext.__init__` (they
  were only required for the OpenMP detection and thread-count setter).

### Changed
- All C++ extension calls now run single-threaded.  Performance of
  `compute_all_metrics` is restored to v0.1.17 levels.
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
  neighbour lists are now persistent, fixing OOM on WSL at n ≥ 150 000 atoms.

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
