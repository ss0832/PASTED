# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.6] - 2026-03-18

### Added
- **Cell List spatial partitioning in `_relax.cpp` and `_maxent.cpp`**

  Both C++ extension modules now use a flat 3-D Cell List to limit
  neighbour searches to a 27-cell neighbourhood instead of scanning all
  atoms.  A linked-list style flat `vector<int>` grid is used (no
  `unordered_map` per cycle) to keep per-cycle allocation cost minimal.

  Automatic strategy selection per call:

  | N | `_relax_core` | `_maxent_core` |
  |:---:|:---:|:---:|
  | < 64 | O(N²) full-pair loop | O(N³) full-pair |
  | ≥ 64 | O(N) Cell List | O(N²) Cell List |

  Cell size is computed automatically: `cov_scale × 2 × max(radii)` for
  `_relax_core`; `cutoff` for `_maxent_core`.  No new API parameters.

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

### Changed
- `pyproject.toml`: version bumped to `0.1.6`.
- `_relax.cpp`: Cell List threshold changed from 32 to 64 after benchmarking
  showed that `unordered_map`-based grid reconstruction at N ≈ 32–63 is
  slower than the full-pair loop; the flat `FlatCellList` struct eliminates
  heap allocation per cycle and moves the crossover to N ≈ 64.
- Distance violation check uses `d >= thr` (not `d² >= thr²`) throughout to
  avoid a floating-point precision bug where atoms exactly at threshold
  distance are incorrectly flagged as violating in the next cycle.

### Added
- **`chain_bias` parameter for `place_chain` / `StructureGenerator` / `generate`**
  (`chain_bias: float = 0.0`, CLI: `--chain-bias`).

  The direction of the **first bond** placed becomes a global *bias axis*.
  Every subsequent step direction is blended toward that axis before
  normalisation:

  ```
  d_biased = d + axis × chain_bias
  d_final  = d_biased / |d_biased|
  ```

  Effect on `shape_aniso ≥ 0.5` rate (n = 20, branch_prob = 0.0):

  | `chain_bias` | mean shape_aniso | ≥ 0.5 rate |
  |:---:|:---:|:---:|
  | 0.0 (default) | 0.40 | 33 % |
  | 0.3 | 0.55 | 63 % |
  | 0.6 | 0.74 | 92 % |
  | 1.0 | 0.89 | 100 % |

  Default is `0.0` — **fully backwards-compatible**; existing scripts and
  seeds produce identical output unless `chain_bias` is explicitly set.

## [0.1.5] - 2026-03-18

### Added
- `src/pasted/_ext/` Subpackage — Organizes C++ extensions into separate files based on functional units:
   - `_relax.cpp → _ext._relax_core`  : Distance constraint relaxation loop (applicable across all modes).
   - `_maxent.cpp → _ext._maxent_core`: Angular repulsion gradient (specific to the maxent mode).
   - `_ext/__init__.py`: Independently manages the HAS_RELAX and HAS_MAXENT flags for each module. This configuration enables a partial fallback mechanism; a compilation failure in one module allows the other to remain operational.

- **Optional C++ extension** (`pasted._ext`, built via `pybind11`):
  two inner-loop hotspots are now compiled to native code when a C++17
  toolchain is present at install time.  When the extension is absent the
  pure-Python / NumPy paths are used transparently — no user-facing API
  change.
  - `relax_positions` — the per-cycle pair-repulsion loop is now a tight
    C++ double loop over `(i, j)` pairs, eliminating NumPy
    broadcast allocation (`(n, n, 3)` diff array) on every iteration.
    Typical speed-up: 5–20× for 10–100 atoms.
  - `_angular_repulsion_gradient` — the O(N³) Python double `for` loop
    over neighbour pairs in the `maxent` gradient descent is now a
    cache-friendly C++ loop.  For `maxent` mode with 300 steps this is
    the dominant cost; speed-up: 20–50×.
- **`seed` parameter for `relax_positions`** (`seed: int | None = None`):
  the RNG used for the coincident-atom edge case (distance < 1e-10 Å) is
  now seeded when a value is provided, enabling full end-to-end
  reproducibility.  Callers that do not pass `seed` behave exactly as
  before.  `StructureGenerator` automatically forwards its master seed.
- **`seed` parameter for `place_maxent`** — threads through to the two
  internal `relax_positions` calls.

### Changed
- `pyproject.toml`: `pybind11>=2.12` added to `[build-system].requires`
  and `[project.optional-dependencies].dev`.
- `setup.py` (new file): declares two `Pybind11Extension` entries,
  one per C++ source file.
- Pure-Python fallback for `relax_positions` now pre-creates
  `np.random.default_rng(seed)` once before the relaxation loop instead of
  calling `np.random.default_rng()` (unseeded) inside the loop on every
  coincident-atom hit — fixing a latent non-reproducibility bug.

## [0.1.4] - 2026-03-17

### Added (Documentation & Behavior Details)
- **Algorithmic Behavior**: The `maxent` mode functions as a constrained maximum-entropy sampling method. It optimizes the objective function $S = -\int p(\Omega) \ln p(\Omega) d\Omega$ under the strict condition $d_{ij} \ge \text{cov\_scale} \times (r_i + r_j)$.
- **Optimization Mechanics & Numerical Stability**: The gradient descent utilizes an angular repulsion potential defined as $U = \sum_i \sum_{j,k \in N(i), j \neq k} \frac{1}{1 - \cos \theta_{jk} + \varepsilon}$. A numerically small constant ($\varepsilon = 10^{-6}$) is explicitly integrated into the denominator. This implementation prevents division by zero and mitigates the risk of gradient explosion in scenarios where two neighbor directions strictly coincide. Distance constraint re-enforcement is systematically executed via `relax_positions` after each gradient step to prevent physical violations.
- **Parameter Tuning Guidelines**:
  - `--maxent-lr`: Values between 0.02 and 0.05 are considered practical for dense structures. The application of larger learning rates exhibits a mathematical tendency to cause spatial drift outside the designated initial region.
  - `--maxent-steps`: Increased iterations correspond structurally to higher uniformity in neighbor directions, accompanied by potential spatial expansion.
- **Statistical Tendencies**: Compared to the `gas` mode, `maxent` systematically yields configurations with computationally lower $Q_6$ values (typically $< 0.2$), reducing the quantitative reliance on post-placement filtering processes.
- **API Clarifications**: `compute_angular_entropy` is defined strictly as a placement-quality diagnostic variable, validating its exclusion from general structural disorder metrics such as `ALL_METRICS` and standard XYZ comment lines.


## [0.1.3] - 2026-03-17

### Added
- `--mode maxent` — Maximum-entropy placement mode.  Atoms are initialised
  at random, then iteratively repositioned by gradient descent on an angular
  repulsion potential so that each atom's neighbour directions become as
  uniformly distributed over the sphere as the Pyykkö distance constraints
  allow.  The result is the constrained-maximum-entropy solution to
  `max S = −∫ p(Ω) ln p(Ω) dΩ` subject to `d_ij ≥ cov_scale·(r_i + r_j)`.
- `place_maxent(atoms, region, cov_scale, rng, ...)` — low-level placement
  function, exported from the public API.
- `compute_angular_entropy(positions, cutoff)` — diagnostic metric: mean
  per-atom Shannon entropy of neighbour direction distributions.  Not
  included in `ALL_METRICS` or XYZ comment lines; intended for comparing
  placement quality.
- `_angular_repulsion_gradient(pts, cutoff)` — internal numpy gradient of
  the angular repulsion potential `U = Σ 1/(1 − cos θ + ε)`.
- Three new CLI flags for `--mode maxent`:
  `--maxent-steps` (default 300), `--maxent-lr` (default 0.05),
  `--maxent-cutoff-scale` (default 2.5).
- `tests/test_maxent.py` — 15 tests covering gradient, placement, entropy,
  and generator integration.

### Changed
- `StructureGenerator` now accepts `mode="maxent"` alongside the existing
  `gas`, `chain`, `shell` modes.
- `_generator.py` `_place_one`: added `maxent` branch.
- `__init__.py` exports: `place_maxent`, `compute_angular_entropy`.

## [0.1.2] - 2026-03-17

### Added
- `StructureOptimizer` class — objective-based structure optimisation that
  **maximises** a user-defined disorder metric instead of sampling randomly.
- Two optimisation methods: `"annealing"` (Simulated Annealing with
  exponential cooling) and `"basin_hopping"` (Metropolis with more thorough
  per-step relaxation).
- Fragment coordinate move: atoms whose local Q6 exceeds `frag_threshold`
  are preferentially displaced, targeting accidentally ordered regions.
- Composition move: element types of two atoms are swapped (or one atom is
  replaced) to explore composition space alongside geometry.
- `parse_objective_spec(["METRIC:WEIGHT", ...])` utility function for
  converting CLI strings to a weight dict.
- `compute_steinhardt_per_atom` public function returning per-atom Q_l
  arrays of shape `(n,)`, used internally by the optimizer for fragment
  selection.
- `--optimize` CLI flag (and associated options) enabling optimization mode
  while preserving full backward compatibility with sampling mode.
- `tests/test_optimizer.py` — 25 tests covering helpers, construction, and
  all run() paths (SA, BH, callable objective, restarts, provided initial).

### Changed
- `compute_steinhardt` refactored to delegate to `compute_steinhardt_per_atom`,
  eliminating code duplication.
- `__init__.py` exports: `StructureOptimizer`, `parse_objective_spec`,
  `compute_steinhardt_per_atom`.

## [0.1.1] - 2026-03-17

### Added
- `pasted.py` — direct-run entry point at the project root, enabling
  `python pasted.py` without a prior `pip install` as documented in README.
- `conftest.py` at the project root to prevent `pasted.py` from shadowing
  the `src/pasted/` package during `pytest` collection.
- CI badges (CI status, PyPI version, PyPI Python versions, License) added
  to `README.md`.
- `.github/workflows/ci.yml` — GitHub Actions workflow with four jobs:
  `lint` (ruff), `test` (Python 3.10/3.11/3.12 matrix), `typecheck` (mypy),
  `build` (sdist + wheel artifact).

### Changed
- CI actions updated to Node.js 24 compatible versions:
  `actions/checkout@v4` → `@v5`, `actions/setup-python@v5` → `@v6`.
- `_metrics.py`: `_sph_harm` wrappers now return `np.ndarray` via
  `np.asarray()` instead of `complex | np.ndarray`, eliminating
  `no-any-return` mypy errors.
- `pyproject.toml`: added `[tool.mypy.overrides]` for `pasted._metrics` to
  suppress `warn_unused_ignores` on the `scipy < 1.15` fallback branch.
- `pyproject.toml`: pytest runs with `--import-mode=importlib` to avoid
  `pasted.py` shadowing the installed package during test collection.
- `license` field updated to SPDX string format (`"MIT"`) per PEP 639,
  removing the deprecated `{ text = "MIT" }` table syntax.

### Fixed
- `ruff format --check` removed from CI lint job; formatting is enforced
  locally only, preventing version-skew failures between environments.

## [0.1.0] - 2026-03-17

### Added
- Initial release.
- `StructureGenerator` class API for programmatic use.
- `generate()` functional API as a thin convenience wrapper.
- `Structure` dataclass with `.to_xyz()` and `.write_xyz()` helpers.
- Three placement modes: `gas`, `chain`, `shell`.
- Ten disorder metrics: `H_atom`, `H_spatial`, `H_total`, `RDF_dev`,
  `shape_aniso`, `Q4`, `Q6`, `Q8`, `graph_lcc`, `graph_cc`.
- `pasted` CLI entry-point (identical behaviour to the original script).
- `src/` layout with separated concerns:
  `_atoms`, `_placement`, `_metrics`, `_io`, `_generator`, `cli`.
- `pyproject.toml` with `ruff`, `mypy`, and `pytest` dev extras.
