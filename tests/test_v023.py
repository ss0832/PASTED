"""
Tests for pasted v0.2.3.

Covers:
- OpenMP symbols are absent from public API
- generate() smoke test
- compute_all_metrics() smoke test
- StructureOptimizer smoke test
- Version string
"""

from __future__ import annotations

import math
import sys

import pytest

import pasted
from pasted import (
    ALL_METRICS,
    GenerationResult,
    Structure,
    StructureGenerator,
    StructureOptimizer,
    compute_all_metrics,
    generate,
)
from pasted._ext import HAS_GRAPH, HAS_RELAX

# ---------------------------------------------------------------------------
# v0.2.3 regression: OpenMP symbols must be gone
# ---------------------------------------------------------------------------

def test_no_has_openmp_in_pasted():
    assert not hasattr(pasted, "HAS_OPENMP"), (
        "HAS_OPENMP must not be exported from pasted in v0.2.3"
    )


def test_no_set_num_threads_in_pasted():
    assert not hasattr(pasted, "set_num_threads"), (
        "set_num_threads must not be exported from pasted in v0.2.3"
    )


def test_no_has_openmp_in_ext():
    import pasted._ext as ext
    assert not hasattr(ext, "HAS_OPENMP"), (
        "HAS_OPENMP must not be in pasted._ext in v0.2.3"
    )


def test_no_set_num_threads_in_ext():
    import pasted._ext as ext
    assert not hasattr(ext, "set_num_threads"), (
        "set_num_threads must not be in pasted._ext in v0.2.3"
    )


def test_no_ctypes_imported_by_ext():
    """_ext.__init__ must not import ctypes (no longer needed)."""
    import pasted._ext as ext
    assert "ctypes" not in sys.modules or ext.__file__ not in str(
        getattr(sys.modules.get("ctypes"), "__file__", "")
    ), "ctypes may be imported elsewhere, but should not be in ext"
    # Simpler: just check the source doesn't re-export it
    assert not hasattr(ext, "ctypes")


# ---------------------------------------------------------------------------
# generate() smoke tests
# ---------------------------------------------------------------------------

def _gas_kw(n_atoms: int, n_samples: int = 5, seed: int = 42) -> dict:
    radius = int(math.ceil((n_atoms / 0.1) ** (1 / 3)))
    return dict(
        n_atoms=n_atoms,
        charge=0,
        mult=1,
        mode="gas",
        region=f"sphere:{radius}",
        elements="6,7,8,1",
        n_samples=n_samples,
        seed=seed,
    )


def test_generate_returns_generation_result():
    result = generate(**_gas_kw(8))
    assert isinstance(result, GenerationResult)


def test_generate_structures_are_structure_instances():
    result = generate(**_gas_kw(8))
    for s in result.structures:
        assert isinstance(s, Structure)


def test_generate_small():
    result = generate(**_gas_kw(8, n_samples=10))
    assert len(result.structures) >= 1


def test_generate_chain():
    result = generate(
        n_atoms=10, charge=0, mult=1,
        mode="chain", elements="6,7,8,1",
        n_samples=5, seed=0,
    )
    assert isinstance(result, GenerationResult)
    assert len(result.structures) >= 1


def test_structure_has_expected_attrs():
    result = generate(**_gas_kw(8, n_samples=10))
    s = result.structures[0]
    assert hasattr(s, "atoms")
    assert hasattr(s, "positions")
    assert hasattr(s, "metrics")
    assert hasattr(s, "charge")
    assert hasattr(s, "mult")
    assert hasattr(s, "mode")
    assert len(s.atoms) == len(s.positions)


def test_structure_metrics_keys():
    result = generate(**_gas_kw(8, n_samples=10))
    s = result.structures[0]
    assert set(s.metrics.keys()) == set(ALL_METRICS)


# ---------------------------------------------------------------------------
# compute_all_metrics() smoke tests
# ---------------------------------------------------------------------------

METRICS_KW = dict(n_bins=20, w_atom=0.5, w_spatial=0.5, cutoff=6.0)


def test_compute_all_metrics_returns_dict():
    result = generate(**_gas_kw(8, n_samples=10))
    s = result.structures[0]
    m = compute_all_metrics(s.atoms, s.positions, **METRICS_KW)
    assert isinstance(m, dict)


def test_compute_all_metrics_keys():
    result = generate(**_gas_kw(8, n_samples=10))
    s = result.structures[0]
    m = compute_all_metrics(s.atoms, s.positions, **METRICS_KW)
    assert set(m.keys()) == set(ALL_METRICS)


def test_compute_all_metrics_values_are_finite():
    result = generate(**_gas_kw(20, n_samples=10))
    s = result.structures[0]
    m = compute_all_metrics(s.atoms, s.positions, **METRICS_KW)
    for k, v in m.items():
        assert math.isfinite(v), f"metric {k} = {v} is not finite"


@pytest.mark.parametrize("n_atoms", [30, 100, 500])
def test_compute_all_metrics_scaling(n_atoms):
    result = generate(**_gas_kw(n_atoms, n_samples=10))
    if not result.structures:
        pytest.skip(f"no structures generated for n_atoms={n_atoms}")
    s = result.structures[0]
    m = compute_all_metrics(s.atoms, s.positions, **METRICS_KW)
    assert set(m.keys()) == set(ALL_METRICS)


# ---------------------------------------------------------------------------
# StructureGenerator class API
# ---------------------------------------------------------------------------

def test_structure_generator_class():
    gen = StructureGenerator(**_gas_kw(8, n_samples=5))
    result = gen.generate()
    assert isinstance(result, GenerationResult)


# ---------------------------------------------------------------------------
# StructureOptimizer smoke test
# ---------------------------------------------------------------------------

def test_optimizer_run_returns_result():
    from pasted import OptimizationResult
    opt = StructureOptimizer(
        n_atoms=8, charge=0, mult=1,
        objective={"H_total": 1.0},
        elements="1-9",
        max_steps=100, n_restarts=1, n_replicas=1, seed=0,
    )
    result = opt.run()
    assert isinstance(result, OptimizationResult)


def test_optimizer_best_is_structure_or_none():
    opt = StructureOptimizer(
        n_atoms=8, charge=0, mult=1,
        objective={"H_total": 1.0},
        elements="1-9",
        max_steps=100, n_restarts=1, n_replicas=1, seed=0,
    )
    result = opt.run()
    assert result.best is None or isinstance(result.best, Structure)


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

def test_version_is_023():
    assert pasted.__version__ == "0.2.3", (
        f"Expected 0.2.3, got {pasted.__version__!r}"
    )


# ---------------------------------------------------------------------------
# C++ extensions present
# ---------------------------------------------------------------------------

def test_has_relax():
    assert HAS_RELAX, "HAS_RELAX should be True (prebuilt .so present)"


def test_has_graph():
    assert HAS_GRAPH, "HAS_GRAPH should be True (prebuilt .so present)"
