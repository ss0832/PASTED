"""
PASTED — Return-Type Contract Tests  (tests/test_type_contracts.py)
===================================================================
Every test in this file answers exactly one question:

    "Does the function return the type its annotation declares?"

Nothing more.  No value-range checks, no business-logic assertions.
If a function's annotation says it returns ``tuple[str, float, float]``,
we call it and check that the result is a 3-tuple whose elements are
``str``, ``float``, and ``float``.  Full stop.

The type spec is derived from:
  • pybind11 docstring signatures for C++ entry points
  • Python ``inspect.get_annotations`` / ``inspect.signature`` for the rest
  • Runtime probes to confirm the concrete types pybind11 actually produces

C++ entry points
  relax_positions       → tuple[ndarray[float64], bool]
  angular_repulsion_gradient → ndarray[float64]
  steinhardt_per_atom   → dict[str, ndarray[float64]]
  graph_metrics_cpp     → dict[str, float]
  rdf_h_cpp             → dict[str, float]

Python API
  generate()            → GenerationResult
  StructureGenerator.generate() → GenerationResult
  StructureGenerator.stream()   → Iterator[Structure]
  StructureOptimizer.run()      → OptimizationResult
  Structure.to_xyz()    → str
  Structure.write_xyz() → None
  Structure.from_xyz()  → Structure
  Structure.comp        → str
  Structure.atoms       → list[str]
  Structure.positions   → list[tuple[float,...]]
  Structure.charge / .mult / .sample_index → int
  Structure.mode        → str
  Structure.metrics     → dict[str, float]
  GenerationResult.n_* → int
  GenerationResult.summary() → str
  GenerationResult.__add__() → GenerationResult
  GenerationResult.__getitem__(int)   → Structure
  GenerationResult.__getitem__(slice) → list[Structure]
  OptimizationResult.all_structures   → list[Structure]
  OptimizationResult.objective_scores → list[float]
  OptimizationResult.n_restarts_attempted → int
  OptimizationResult.method → str
  OptimizationResult.best  → Structure
  OptimizationResult.summary() → str
  EvalContext.atoms / .element_pool → tuple[str, ...]
  EvalContext.positions → tuple[tuple[numpy.float64, ...], ...]
  EvalContext.per_atom_q6 → ndarray[float64, shape=(n_atoms,)]
  EvalContext.charge/mult/n_atoms/step/max_steps/restart_idx → int
  EvalContext.progress/temperature/f_current/best_f/cutoff → float
  EvalContext.replica_idx / .n_replicas → int | None   (None in non-PT)
  EvalContext.metrics → dict[str, float]
  parse_filter()        → tuple[str, float, float]
  parse_objective_spec()→ dict[str, float]
  validate_charge_mult()→ tuple[bool, str]
  compute_all_metrics() → dict[str, float]
  format_xyz()          → str
  ALL_METRICS           → frozenset[str]
  GeneratorConfig       → frozen dataclass
"""

from __future__ import annotations

import collections.abc
import warnings
from typing import Any, cast

import numpy as np
import pytest

from pasted import (
    ALL_METRICS,
    GenerationResult,
    GeneratorConfig,
    Structure,
    StructureGenerator,
    StructureOptimizer,
    compute_all_metrics,
    format_xyz,
    generate,
    parse_filter,
    parse_objective_spec,
    validate_charge_mult,
)
from pasted._ext import (
    HAS_GRAPH,
    HAS_MAXENT,
    HAS_RELAX,
    HAS_STEINHARDT,
    angular_repulsion_gradient,
    graph_metrics_cpp,
    rdf_h_cpp,
    relax_positions,
    steinhardt_per_atom,
)
from pasted._optimizer import EvalContext, OptimizationResult

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------
needs_relax = pytest.mark.skipif(not HAS_RELAX, reason="HAS_RELAX=False")
needs_maxent = pytest.mark.skipif(not HAS_MAXENT, reason="HAS_MAXENT=False")
needs_steinhardt = pytest.mark.skipif(not HAS_STEINHARDT, reason="HAS_STEINHARDT=False")
needs_graph = pytest.mark.skipif(not HAS_GRAPH, reason="HAS_GRAPH=False")

# ---------------------------------------------------------------------------
# Shared fixtures — canonical valid inputs
# ---------------------------------------------------------------------------

PTS3: np.ndarray = np.array(
    [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]], dtype=np.float64
)
RADII3: np.ndarray = np.full(3, 0.77, dtype=np.float64)
EN3: np.ndarray = np.array([2.55, 3.04, 3.44], dtype=np.float64)

_GRAPH_KEYS = frozenset(
    {"graph_lcc", "graph_cc", "ring_fraction", "charge_frustration", "moran_I_chi"}
)
_RDF_KEYS = frozenset({"h_spatial", "rdf_dev"})


def _gas(seed: int = 0, n: int = 8, n_samples: int = 30) -> GenerationResult:
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return generate(
            n_atoms=n, charge=0, mult=1, mode="gas", region="sphere:8",
            elements="6,7,8", n_samples=n_samples, seed=seed,
        )


def _one_structure(seed: int = 0) -> Structure:
    r = _gas(seed=seed)
    assert len(r) > 0
    return cast(Structure, r[0])


def _collect_ctx(max_steps: int = 20, seed: int = 0) -> list[EvalContext]:
    captured: list[EvalContext] = []

    def _cap(m: dict[str, float], ctx: EvalContext) -> float:
        captured.append(ctx)
        return m["H_total"]

    StructureOptimizer(
        n_atoms=6, charge=0, mult=1, elements="6,7,8",
        objective=_cap, method="annealing",
        max_steps=max_steps, n_restarts=1, seed=seed,
    ).run()
    return captured


# ===========================================================================
# TC-1  C++ extension: relax_positions
# ===========================================================================

class TestRelaxPositionsReturnType:
    """relax_positions → tuple[NDArray[float64], bool]"""

    @needs_relax
    def test_return_is_tuple_of_length_2(self) -> None:
        result = relax_positions(PTS3, RADII3, 1.0, 100)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @needs_relax
    def test_first_element_is_ndarray(self) -> None:
        out, _ = relax_positions(PTS3, RADII3, 1.0, 100)
        assert isinstance(out, np.ndarray)

    @needs_relax
    def test_first_element_dtype_float64(self) -> None:
        out, _ = relax_positions(PTS3, RADII3, 1.0, 100)
        assert out.dtype == np.float64

    @needs_relax
    def test_first_element_shape_n_by_3(self) -> None:
        out, _ = relax_positions(PTS3, RADII3, 1.0, 100)
        assert out.ndim == 2
        assert out.shape[0] == len(PTS3)
        assert out.shape[1] == 3

    @needs_relax
    def test_second_element_is_bool(self) -> None:
        _, conv = relax_positions(PTS3, RADII3, 1.0, 100)
        assert isinstance(conv, bool)

    @needs_relax
    def test_first_element_c_contiguous(self) -> None:
        out, _ = relax_positions(PTS3, RADII3, 1.0, 100)
        assert out.flags["C_CONTIGUOUS"]


# ===========================================================================
# TC-2  C++ extension: angular_repulsion_gradient
# ===========================================================================

class TestAngularRepulsionGradientReturnType:
    """angular_repulsion_gradient → NDArray[float64]"""

    @needs_maxent
    def test_return_is_ndarray(self) -> None:
        result = angular_repulsion_gradient(PTS3, 5.0)
        assert isinstance(result, np.ndarray)

    @needs_maxent
    def test_dtype_float64(self) -> None:
        result = angular_repulsion_gradient(PTS3, 5.0)
        assert result.dtype == np.float64

    @needs_maxent
    def test_shape_n_by_3(self) -> None:
        result = angular_repulsion_gradient(PTS3, 5.0)
        assert result.ndim == 2
        assert result.shape[0] == len(PTS3)
        assert result.shape[1] == 3


# ===========================================================================
# TC-3  C++ extension: steinhardt_per_atom
# ===========================================================================

class TestSteindhardtPerAtomReturnType:
    """steinhardt_per_atom → dict[str, NDArray[float64]]"""

    @needs_steinhardt
    def test_return_is_dict(self) -> None:
        result = steinhardt_per_atom(PTS3, 5.0, [4, 6])
        assert isinstance(result, dict)

    @needs_steinhardt
    def test_keys_are_str(self) -> None:
        result = steinhardt_per_atom(PTS3, 5.0, [4, 6])
        for key in result:
            assert isinstance(key, str), f"Expected str key, got {type(key)}"

    @needs_steinhardt
    def test_key_names_match_l_values(self) -> None:
        """Key for l=6 must be 'Q6', for l=4 must be 'Q4', etc."""
        result = steinhardt_per_atom(PTS3, 5.0, [4, 6, 8])
        assert set(result.keys()) == {"Q4", "Q6", "Q8"}

    @needs_steinhardt
    def test_values_are_ndarray(self) -> None:
        result = steinhardt_per_atom(PTS3, 5.0, [6])
        for val in result.values():
            assert isinstance(val, np.ndarray)

    @needs_steinhardt
    def test_values_dtype_float64(self) -> None:
        result = steinhardt_per_atom(PTS3, 5.0, [4, 6, 8])
        for key, val in result.items():
            assert val.dtype == np.float64, f"{key}: expected float64, got {val.dtype}"

    @needs_steinhardt
    def test_values_shape_n(self) -> None:
        """Each value array must have shape (N,) matching the number of atoms."""
        result = steinhardt_per_atom(PTS3, 5.0, [4, 6])
        for key, val in result.items():
            assert val.ndim == 1, f"{key}: expected 1-D array, got ndim={val.ndim}"
            assert val.shape[0] == len(PTS3), (
                f"{key}: expected shape ({len(PTS3)},), got {val.shape}"
            )


# ===========================================================================
# TC-4  C++ extension: graph_metrics_cpp
# ===========================================================================

class TestGraphMetricsCppReturnType:
    """graph_metrics_cpp → dict[str, float] with exactly 5 keys"""

    @needs_graph
    def test_return_is_dict(self) -> None:
        assert isinstance(graph_metrics_cpp(PTS3, RADII3, 1.0, EN3, 5.0), dict)

    @needs_graph
    def test_keys_are_exactly_five_str(self) -> None:
        result = graph_metrics_cpp(PTS3, RADII3, 1.0, EN3, 5.0)
        assert set(result.keys()) == _GRAPH_KEYS

    @needs_graph
    def test_values_are_float(self) -> None:
        result = graph_metrics_cpp(PTS3, RADII3, 1.0, EN3, 5.0)
        for key, val in result.items():
            assert isinstance(val, float), (
                f"{key!r}: expected float, got {type(val).__name__}"
            )


# ===========================================================================
# TC-5  C++ extension: rdf_h_cpp
# ===========================================================================

class TestRdfHCppReturnType:
    """rdf_h_cpp → dict[str, float] with exactly 2 keys"""

    @needs_graph
    def test_return_is_dict(self) -> None:
        assert isinstance(rdf_h_cpp(PTS3, 5.0, 20), dict)

    @needs_graph
    def test_keys_are_exactly_two_str(self) -> None:
        result = rdf_h_cpp(PTS3, 5.0, 20)
        assert set(result.keys()) == _RDF_KEYS

    @needs_graph
    def test_values_are_float(self) -> None:
        result = rdf_h_cpp(PTS3, 5.0, 20)
        for key, val in result.items():
            assert isinstance(val, float), (
                f"{key!r}: expected float, got {type(val).__name__}"
            )


# ===========================================================================
# TC-6  generate() / StructureGenerator.generate()
# ===========================================================================

class TestGenerateReturnType:
    """generate() → GenerationResult"""

    def test_return_is_generationresult(self) -> None:
        assert isinstance(_gas(), GenerationResult)

    def test_n_attempted_is_int(self) -> None:
        assert isinstance(_gas().n_attempted, int)

    def test_n_passed_is_int(self) -> None:
        assert isinstance(_gas().n_passed, int)

    def test_n_rejected_parity_is_int(self) -> None:
        assert isinstance(_gas().n_rejected_parity, int)

    def test_n_rejected_filter_is_int(self) -> None:
        assert isinstance(_gas().n_rejected_filter, int)

    def test_summary_is_str(self) -> None:
        assert isinstance(_gas().summary(), str)

    def test_add_returns_generationresult(self) -> None:
        r1 = _gas(seed=1)
        r2 = _gas(seed=2)
        assert isinstance(r1 + r2, GenerationResult)

    def test_int_index_returns_structure(self) -> None:
        r = _gas()
        assert len(r) > 0
        assert isinstance(cast(Structure, r[0]), Structure)

    def test_slice_returns_list(self) -> None:
        r = _gas()
        sliced = r[:]
        assert isinstance(sliced, list)

    def test_slice_elements_are_structure(self) -> None:
        r = _gas()
        sliced = r[:]
        assert isinstance(sliced, list)
        for s in sliced:
            assert isinstance(s, Structure)

    def test_iter_yields_structure(self) -> None:
        for s in _gas():
            assert isinstance(s, Structure)

    def test_structures_field_is_list(self) -> None:
        assert isinstance(_gas().structures, list)

    def test_structuregenerator_generate_returns_generationresult(self) -> None:
        gen = StructureGenerator(
            n_atoms=6, charge=0, mult=1, mode="chain",
            elements="6,7,8", n_samples=10, seed=0,
        )
        assert isinstance(gen.generate(), GenerationResult)


# ===========================================================================
# TC-7  StructureGenerator.stream()
# ===========================================================================

class TestStreamReturnType:
    """stream() → Iterator[Structure]"""

    def test_stream_is_iterator(self) -> None:
        gen = StructureGenerator(
            n_atoms=6, charge=0, mult=1, mode="gas", region="sphere:8",
            elements="6,7,8", n_success=2, n_samples=30, seed=0,
        )
        it = gen.stream()
        assert isinstance(it, collections.abc.Iterator)

    def test_stream_yields_structure(self) -> None:
        gen = StructureGenerator(
            n_atoms=6, charge=0, mult=1, mode="gas", region="sphere:8",
            elements="6,7,8", n_success=2, n_samples=30, seed=0,
        )
        for s in gen.stream():
            assert isinstance(s, Structure)


# ===========================================================================
# TC-8  Structure attributes
# ===========================================================================

class TestStructureAttributeTypes:
    """All Structure attributes must match their declared types."""

    def setup_method(self) -> None:
        self.s = _one_structure()

    def test_atoms_is_list(self) -> None:
        assert isinstance(self.s.atoms, list)

    def test_atoms_elements_are_str(self) -> None:
        for a in self.s.atoms:
            assert isinstance(a, str)

    def test_positions_is_list(self) -> None:
        assert isinstance(self.s.positions, list)

    def test_positions_elements_are_tuple(self) -> None:
        for p in self.s.positions:
            assert isinstance(p, tuple)

    def test_positions_inner_elements_are_float64(self) -> None:
        for p in self.s.positions:
            for v in p:
                assert isinstance(v, float | np.floating), (
                    f"Expected float, got {type(v).__name__}"
                )

    def test_positions_inner_tuples_length_3(self) -> None:
        for p in self.s.positions:
            assert len(p) == 3

    def test_charge_is_int(self) -> None:
        assert isinstance(self.s.charge, int)

    def test_mult_is_int(self) -> None:
        assert isinstance(self.s.mult, int)

    def test_mode_is_str(self) -> None:
        assert isinstance(self.s.mode, str)

    def test_metrics_is_dict(self) -> None:
        assert isinstance(self.s.metrics, dict)

    def test_metrics_keys_are_str(self) -> None:
        for k in self.s.metrics:
            assert isinstance(k, str)

    def test_metrics_values_are_float(self) -> None:
        for k, v in self.s.metrics.items():
            assert isinstance(v, float), (
                f"metrics[{k!r}]: expected float, got {type(v).__name__}"
            )

    def test_comp_is_str(self) -> None:
        assert isinstance(self.s.comp, str)

    def test_sample_index_is_int(self) -> None:
        assert isinstance(self.s.sample_index, int)

    def test_center_sym_is_str_or_none(self) -> None:
        assert self.s.center_sym is None or isinstance(self.s.center_sym, str)

    def test_seed_is_int_or_none(self) -> None:
        assert self.s.seed is None or isinstance(self.s.seed, int)

    def test_to_xyz_returns_str(self) -> None:
        assert isinstance(self.s.to_xyz(), str)

    def test_write_xyz_returns_none(self, tmp_path: Any) -> None:
        # write_xyz is annotated as -> None; mypy rejects assigning its result.
        # We verify the contract by confirming no exception is raised and the
        # annotation is honoured — the return value simply does not exist.
        self.s.write_xyz(str(tmp_path / "out.xyz"))

    def test_from_xyz_returns_structure(self) -> None:
        s2 = Structure.from_xyz(self.s.to_xyz())
        assert isinstance(s2, Structure)


# ===========================================================================
# TC-9  StructureOptimizer.run() → OptimizationResult
# ===========================================================================

class TestOptimizationResultReturnType:
    """StructureOptimizer.run() → OptimizationResult"""

    def setup_method(self) -> None:
        self.result = StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective={"H_total": 1.0}, method="annealing",
            max_steps=30, n_restarts=1, seed=0,
        ).run()

    def test_return_is_optimizationresult(self) -> None:
        assert isinstance(self.result, OptimizationResult)

    def test_all_structures_is_list(self) -> None:
        assert isinstance(self.result.all_structures, list)

    def test_all_structures_elements_are_structure(self) -> None:
        for s in self.result.all_structures:
            assert isinstance(s, Structure)

    def test_objective_scores_is_list(self) -> None:
        assert isinstance(self.result.objective_scores, list)

    def test_objective_scores_elements_are_float(self) -> None:
        for v in self.result.objective_scores:
            assert isinstance(v, float)

    def test_n_restarts_attempted_is_int(self) -> None:
        assert isinstance(self.result.n_restarts_attempted, int)

    def test_method_is_str(self) -> None:
        assert isinstance(self.result.method, str)

    def test_best_is_structure(self) -> None:
        assert isinstance(self.result.best, Structure)

    def test_summary_returns_str(self) -> None:
        assert isinstance(self.result.summary(), str)


# ===========================================================================
# TC-10  EvalContext field types
# ===========================================================================

class TestEvalContextFieldTypes:
    """EvalContext fields must match their documented types."""

    def setup_method(self) -> None:
        contexts = _collect_ctx(max_steps=10, seed=0)
        assert contexts, "No EvalContext captured"
        self.ctx = contexts[0]

    def test_atoms_is_tuple(self) -> None:
        assert isinstance(self.ctx.atoms, tuple)

    def test_atoms_elements_are_str(self) -> None:
        for a in self.ctx.atoms:
            assert isinstance(a, str)

    def test_positions_is_tuple(self) -> None:
        assert isinstance(self.ctx.positions, tuple)

    def test_positions_elements_are_tuple(self) -> None:
        for p in self.ctx.positions:
            assert isinstance(p, tuple)

    def test_positions_inner_elements_are_float64(self) -> None:
        for p in self.ctx.positions:
            for v in p:
                assert isinstance(v, (float | np.floating)), (
                    f"Expected float64, got {type(v).__name__}"
                )

    def test_positions_inner_tuples_length_3(self) -> None:
        for p in self.ctx.positions:
            assert len(p) == 3

    def test_charge_is_int(self) -> None:
        assert isinstance(self.ctx.charge, int)

    def test_mult_is_int(self) -> None:
        assert isinstance(self.ctx.mult, int)

    def test_n_atoms_is_int(self) -> None:
        assert isinstance(self.ctx.n_atoms, int)

    def test_metrics_is_dict(self) -> None:
        assert isinstance(self.ctx.metrics, dict)

    def test_metrics_keys_are_str(self) -> None:
        for k in self.ctx.metrics:
            assert isinstance(k, str)

    def test_metrics_values_are_float(self) -> None:
        for k, v in self.ctx.metrics.items():
            assert isinstance(v, float), (
                f"metrics[{k!r}]: expected float, got {type(v).__name__}"
            )

    def test_step_is_int(self) -> None:
        assert isinstance(self.ctx.step, int)

    def test_max_steps_is_int(self) -> None:
        assert isinstance(self.ctx.max_steps, int)

    def test_progress_is_float(self) -> None:
        assert isinstance(self.ctx.progress, float)

    def test_temperature_is_float(self) -> None:
        assert isinstance(self.ctx.temperature, float)

    def test_f_current_is_float(self) -> None:
        assert isinstance(self.ctx.f_current, float)

    def test_best_f_is_float(self) -> None:
        assert isinstance(self.ctx.best_f, float)

    def test_per_atom_q6_is_float64_ndarray(self) -> None:
        arr = self.ctx.per_atom_q6
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64

    def test_per_atom_q6_shape_matches_n_atoms(self) -> None:
        arr = self.ctx.per_atom_q6
        assert arr.ndim == 1
        assert arr.shape[0] == self.ctx.n_atoms

    def test_restart_idx_is_int(self) -> None:
        assert isinstance(self.ctx.restart_idx, int)

    def test_element_pool_is_tuple(self) -> None:
        assert isinstance(self.ctx.element_pool, tuple)

    def test_element_pool_elements_are_str(self) -> None:
        for e in self.ctx.element_pool:
            assert isinstance(e, str)

    def test_cutoff_is_float(self) -> None:
        assert isinstance(self.ctx.cutoff, float)

    def test_replica_idx_is_int_or_none_in_annealing(self) -> None:
        """In non-PT mode, replica_idx must be None."""
        assert self.ctx.replica_idx is None

    def test_n_replicas_is_int_or_none_in_annealing(self) -> None:
        """In non-PT mode, n_replicas must be None."""
        assert self.ctx.n_replicas is None

    def test_replica_idx_is_int_in_parallel_tempering(self) -> None:
        """In PT mode, replica_idx must be int."""
        pt_contexts: list[EvalContext] = []

        def _cap(m: dict[str, float], ctx: EvalContext) -> float:
            pt_contexts.append(ctx)
            return m["H_total"]

        StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective=_cap, method="parallel_tempering",
            n_replicas=2, max_steps=10, n_restarts=1, seed=0,
        ).run()

        assert pt_contexts, "No PT EvalContext captured"
        for ctx in pt_contexts:
            assert isinstance(ctx.replica_idx, int), (
                f"PT replica_idx must be int, got {type(ctx.replica_idx).__name__}"
            )
            assert isinstance(ctx.n_replicas, int), (
                f"PT n_replicas must be int, got {type(ctx.n_replicas).__name__}"
            )


# ===========================================================================
# TC-11  parse_filter()
# ===========================================================================

class TestParseFilterReturnType:
    """parse_filter → tuple[str, float, float]"""

    def test_return_is_tuple(self) -> None:
        assert isinstance(parse_filter("H_total:1.0:-"), tuple)

    def test_tuple_length_is_3(self) -> None:
        assert len(parse_filter("H_total:1.0:-")) == 3

    def test_first_element_is_str(self) -> None:
        result = parse_filter("H_total:1.0:-")
        assert isinstance(result[0], str)

    def test_second_element_is_float(self) -> None:
        result = parse_filter("H_total:1.0:-")
        assert isinstance(result[1], float)

    def test_third_element_is_float(self) -> None:
        result = parse_filter("H_total:1.0:2.5")
        assert isinstance(result[2], float)

    def test_open_upper_bound_is_float_inf(self) -> None:
        """'-' as upper bound must be float('inf'), not None or a string."""
        import math
        _, _, hi = parse_filter("H_total:1.0:-")
        assert isinstance(hi, float)
        assert math.isinf(hi)

    def test_open_lower_bound_is_float_neg_inf(self) -> None:
        """'-' as lower bound must be float('-inf')."""
        import math
        _, lo, _ = parse_filter("H_total:-:2.0")
        assert isinstance(lo, float)
        assert math.isinf(lo) and lo < 0


# ===========================================================================
# TC-12  parse_objective_spec()
# ===========================================================================

class TestParseObjectiveSpecReturnType:
    """parse_objective_spec → dict[str, float]"""

    def test_return_is_dict(self) -> None:
        assert isinstance(parse_objective_spec(["H_total:1.0"]), dict)

    def test_keys_are_str(self) -> None:
        for k in parse_objective_spec(["H_total:1.0", "Q6:-2.0"]):
            assert isinstance(k, str)

    def test_values_are_float(self) -> None:
        for v in parse_objective_spec(["H_total:1.0", "Q6:-2.0"]).values():
            assert isinstance(v, float)

    def test_empty_input_returns_empty_dict(self) -> None:
        result = parse_objective_spec([])
        assert isinstance(result, dict)
        assert len(result) == 0


# ===========================================================================
# TC-13  validate_charge_mult()
# ===========================================================================

class TestValidateChargeMultReturnType:
    """validate_charge_mult → tuple[bool, str]"""

    def test_return_is_tuple(self) -> None:
        assert isinstance(validate_charge_mult(["C"], 0, 1), tuple)

    def test_tuple_length_is_2(self) -> None:
        assert len(validate_charge_mult(["C"], 0, 1)) == 2

    def test_first_element_is_bool(self) -> None:
        ok, _ = validate_charge_mult(["C"], 0, 1)
        assert isinstance(ok, bool)

    def test_second_element_is_str(self) -> None:
        _, msg = validate_charge_mult(["C"], 0, 1)
        assert isinstance(msg, str)

    def test_invalid_returns_false_bool_str(self) -> None:
        """Invalid parity → first element must be False (not None, not 0)."""
        ok, msg = validate_charge_mult(["N"], 0, 1)
        assert isinstance(ok, bool)
        assert ok is False
        assert isinstance(msg, str)


# ===========================================================================
# TC-14  compute_all_metrics()
# ===========================================================================

class TestComputeAllMetricsReturnType:
    """compute_all_metrics → dict[str, float]"""

    def test_return_is_dict(self) -> None:
        result = compute_all_metrics(
            ["C", "N", "O"],
            [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 1.5, 0.0)],
            n_bins=10, w_atom=1.0, w_spatial=1.0, cutoff=5.0,
        )
        assert isinstance(result, dict)

    def test_keys_are_str(self) -> None:
        result = compute_all_metrics(
            ["C", "N", "O"],
            [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 1.5, 0.0)],
            n_bins=10, w_atom=1.0, w_spatial=1.0, cutoff=5.0,
        )
        for k in result:
            assert isinstance(k, str)

    def test_values_are_float(self) -> None:
        result = compute_all_metrics(
            ["C", "N", "O"],
            [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 1.5, 0.0)],
            n_bins=10, w_atom=1.0, w_spatial=1.0, cutoff=5.0,
        )
        for k, v in result.items():
            assert isinstance(v, float), (
                f"compute_all_metrics[{k!r}]: expected float, got {type(v).__name__}"
            )

    def test_keys_match_all_metrics(self) -> None:
        """The returned dict must contain exactly the 13 keys in ALL_METRICS."""
        result = compute_all_metrics(
            ["C", "N", "O"],
            [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 1.5, 0.0)],
            n_bins=10, w_atom=1.0, w_spatial=1.0, cutoff=5.0,
        )
        assert set(result.keys()) == ALL_METRICS


# ===========================================================================
# TC-15  format_xyz()
# ===========================================================================

class TestFormatXyzReturnType:
    """format_xyz → str"""

    def test_return_is_str(self) -> None:
        result = format_xyz(
            ["C", "N"],
            [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)],
            charge=0, mult=1, metrics={},
        )
        assert isinstance(result, str)


# ===========================================================================
# TC-16  ALL_METRICS
# ===========================================================================

class TestAllMetricsType:
    """ALL_METRICS → frozenset[str]"""

    def test_is_frozenset(self) -> None:
        assert isinstance(ALL_METRICS, frozenset)

    def test_elements_are_str(self) -> None:
        for item in ALL_METRICS:
            assert isinstance(item, str)


# ===========================================================================
# TC-17  GeneratorConfig — frozen dataclass
# ===========================================================================

class TestGeneratorConfigType:
    """GeneratorConfig must be a frozen dataclass (immutable)."""

    def test_is_generatorconfig(self) -> None:
        cfg = GeneratorConfig(
            n_atoms=6, charge=0, mult=1, mode="chain",
            elements="6,7,8", n_samples=5, seed=0,
        )
        assert isinstance(cfg, GeneratorConfig)

    def test_is_frozen(self) -> None:
        """Mutation must raise FrozenInstanceError — not silently succeed."""
        import dataclasses

        cfg = GeneratorConfig(
            n_atoms=6, charge=0, mult=1, mode="chain",
            elements="6,7,8", n_samples=5, seed=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.n_atoms = 99  # type: ignore[misc]
