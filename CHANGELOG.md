# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
