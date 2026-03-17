# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
