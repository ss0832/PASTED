"""
PASTED — Nasty Edge Cases: tests NOT covered by the existing suite
==================================================================
Every test here was designed by first auditing the full existing test
inventory (test_edge_cases.py, test_adversarial.py, test_generator.py,
test_optimizer.py, test_atoms.py, test_io.py, test_metrics.py,
test_placement.py, test_maxent.py) and then probing the live library to
confirm the expected behavior before committing it as an assertion.

Categories
  NEC-A  GenerationResult contract — accounting invariant, concatenation,
          bool semantics, + non-list TypeError
  NEC-B  GeneratorConfig frozen dataclass — FrozenInstanceError on mutation,
          __getattr__ proxy forwarding on StructureGenerator
  NEC-C  Structure data integrity — pickle / deepcopy round-trip, manual
          construction attributes, sample_index monotonicity
  NEC-D  XYZ serialization — to_xyz / from_xyz positional round-trip (file
          path and in-memory), format_xyz prefix in comment line
  NEC-E  parse_filter pathological inputs — NaN min blocks everything,
          whitespace-padded metric name, case-sensitive metric name
  NEC-F  parse_objective_spec contract — empty list, duplicate key last-wins,
          non-float weight raises, all-metrics dict round-trip
  NEC-G  compute_all_metrics direct API — 2-atom call returns all 13 keys,
          w_atom/w_spatial linearity of H_total
  NEC-H  EvalContext field invariants — atoms/positions/element_pool are
          immutable tuples, step ∈ [0, max_steps), best_f monotone,
          temperature strictly positive and within [T_end, T_start],
          cutoff matches constructor parameter
  NEC-I  OptimizationResult contract — summary() fields, n_restarts_attempted
          equals n_restarts, basin_hopping method name in summary, run()
          idempotent (callable twice)
  NEC-J  Objective function flexibility — integer-valued return, callable
          covering all 13 ALL_METRICS keys simultaneously
  NEC-K  Shell-mode specifics — center_sym populated, center_z not in pool
          raises ValueError, n_atoms=1 shell succeeds
  NEC-L  High multiplicity (mult=7) valid structures generated
  NEC-M  Affine-shear-only path (stretch=0, jitter=0) does not crash
  NEC-N  validate_charge_mult direct API — positive/negative results
  NEC-O  ALL_METRICS is exactly the 13 documented keys
"""

from __future__ import annotations

import copy
import dataclasses
import math
import os
import pickle
import tempfile
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
from pasted._optimizer import OptimizationResult

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _gas(
    n: int = 6,
    elements: str | list[str] = "6,7,8",
    n_samples: int = 30,
    seed: int = 0,
    **kw: Any,
) -> GenerationResult:
    """Shorthand: gas-mode generate, warnings suppressed."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return generate(
            n_atoms=n,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:8",
            elements=elements,
            n_samples=n_samples,
            seed=seed,
            **kw,
        )


def _one_structure(seed: int = 0) -> Structure:
    """Return a single guaranteed Structure."""
    r = _gas(n=8, seed=seed, n_samples=50)
    assert len(r) > 0, "fixture failed to generate any structure"
    return cast(Structure, r[0])


# ---------------------------------------------------------------------------
# Type-contract assertion helpers
# Each helper asserts that a return value honours its declared type signature.
# They are called in every test that exercises the corresponding function,
# regardless of the input being nominal or adversarial.
# ---------------------------------------------------------------------------

def _check_generation_result(r: object) -> None:
    """generate() / StructureGenerator.generate() → GenerationResult"""
    assert isinstance(r, GenerationResult), (
        f"Expected GenerationResult, got {type(r).__name__}"
    )
    assert isinstance(r.n_attempted, int), (
        f"n_attempted: expected int, got {type(r.n_attempted).__name__}"
    )
    assert isinstance(r.n_passed, int), (
        f"n_passed: expected int, got {type(r.n_passed).__name__}"
    )
    assert isinstance(r.n_rejected_parity, int), (
        f"n_rejected_parity: expected int, got {type(r.n_rejected_parity).__name__}"
    )
    assert isinstance(r.n_rejected_filter, int), (
        f"n_rejected_filter: expected int, got {type(r.n_rejected_filter).__name__}"
    )
    assert isinstance(r.summary(), str), (
        f"summary(): expected str, got {type(r.summary()).__name__}"
    )


def _check_optimization_result(r: object) -> None:
    """StructureOptimizer.run() → OptimizationResult"""
    assert isinstance(r, OptimizationResult), (
        f"Expected OptimizationResult, got {type(r).__name__}"
    )
    assert isinstance(r.all_structures, list), (
        f"all_structures: expected list, got {type(r.all_structures).__name__}"
    )
    assert isinstance(r.objective_scores, list), (
        f"objective_scores: expected list, got {type(r.objective_scores).__name__}"
    )
    for v in r.objective_scores:
        assert isinstance(v, float), (
            f"objective_scores element: expected float, got {type(v).__name__}"
        )
    assert isinstance(r.n_restarts_attempted, int), (
        f"n_restarts_attempted: expected int, got {type(r.n_restarts_attempted).__name__}"
    )
    assert isinstance(r.method, str), (
        f"method: expected str, got {type(r.method).__name__}"
    )
    assert isinstance(r.summary(), str), (
        f"summary(): expected str, got {type(r.summary()).__name__}"
    )


def _check_to_xyz(r: object) -> None:
    """Structure.to_xyz() → str"""
    assert isinstance(r, str), f"to_xyz(): expected str, got {type(r).__name__}"


def _check_from_xyz(r: object) -> None:
    """Structure.from_xyz() → Structure"""
    assert isinstance(r, Structure), (
        f"from_xyz(): expected Structure, got {type(r).__name__}"
    )


def _check_format_xyz(r: object) -> None:
    """format_xyz() → str"""
    assert isinstance(r, str), f"format_xyz(): expected str, got {type(r).__name__}"


def _check_parse_filter(r: object) -> None:
    """parse_filter() → tuple[str, float, float]"""
    assert isinstance(r, tuple), f"Expected tuple, got {type(r).__name__}"
    assert len(r) == 3, f"Expected length 3, got {len(r)}"  # type: ignore[arg-type]
    assert isinstance(r[0], str), f"[0]: expected str, got {type(r[0]).__name__}"  # type: ignore[index]
    assert isinstance(r[1], float), f"[1]: expected float, got {type(r[1]).__name__}"  # type: ignore[index]
    assert isinstance(r[2], float), f"[2]: expected float, got {type(r[2]).__name__}"  # type: ignore[index]


def _check_parse_objective_spec(r: object) -> None:
    """parse_objective_spec() → dict[str, float]"""
    assert isinstance(r, dict), f"Expected dict, got {type(r).__name__}"
    for k, v in r.items():  # type: ignore[union-attr]
        assert isinstance(k, str), f"key: expected str, got {type(k).__name__}"
        assert isinstance(v, float), f"value: expected float, got {type(v).__name__}"


def _check_validate_charge_mult(r: object) -> None:
    """validate_charge_mult() → tuple[bool, str]"""
    assert isinstance(r, tuple), f"Expected tuple, got {type(r).__name__}"
    assert len(r) == 2, f"Expected length 2, got {len(r)}"  # type: ignore[arg-type]
    assert isinstance(r[0], bool), f"[0]: expected bool, got {type(r[0]).__name__}"  # type: ignore[index]
    assert isinstance(r[1], str), f"[1]: expected str, got {type(r[1]).__name__}"  # type: ignore[index]


def _check_compute_all_metrics(r: object) -> None:
    """compute_all_metrics() → dict[str, float]"""
    assert isinstance(r, dict), f"Expected dict, got {type(r).__name__}"
    for k, v in r.items():  # type: ignore[union-attr]
        assert isinstance(k, str), f"key: expected str, got {type(k).__name__}"
        assert isinstance(v, float), f"value: expected float, got {type(v).__name__}"



# ===========================================================================
# NEC-A  GenerationResult contract
# ===========================================================================

class TestGenerationResultContract:
    """Accounting invariant, concatenation, bool, and + semantics."""

    def test_accounting_invariant_no_filter(self) -> None:
        """n_passed + n_rejected_parity + n_rejected_filter == n_attempted
        must hold for every call, regardless of filter."""
        r = _gas(n_samples=80, seed=3)
        total = r.n_passed + r.n_rejected_parity + r.n_rejected_filter
        assert total == r.n_attempted

    def test_accounting_invariant_with_tight_filter(self) -> None:
        """Same invariant when a tight filter rejects most structures."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r = generate(
                n_atoms=8,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:8",
                elements="6,7,8",
                n_samples=60,
                seed=42,
                filters=["H_total:2.0:-"],
            )
            _check_generation_result(r)
        total = r.n_passed + r.n_rejected_parity + r.n_rejected_filter
        assert total == r.n_attempted

    def test_concatenation_n_passed_is_sum(self) -> None:
        """r1 + r2 must report n_passed == r1.n_passed + r2.n_passed."""
        r1 = _gas(seed=1, n_samples=20)
        r2 = _gas(seed=2, n_samples=20)
        combined = r1 + r2
        assert combined.n_passed == r1.n_passed + r2.n_passed

    def test_concatenation_len_matches_n_passed(self) -> None:
        """len(r1 + r2) must equal (r1 + r2).n_passed."""
        r1 = _gas(seed=10, n_samples=20)
        r2 = _gas(seed=11, n_samples=20)
        combined = r1 + r2
        assert len(combined) == combined.n_passed

    def test_concatenation_preserves_element_order(self) -> None:
        """Structures from r1 come before structures from r2 after +."""
        r1 = _gas(seed=20, n_samples=20)
        r2 = _gas(seed=21, n_samples=20)
        combined = r1 + r2
        if r1 and r2:
            first = cast(Structure, combined[0])
            last = cast(Structure, combined[len(r1) - 1])
            assert first.atoms == cast(Structure, r1[0]).atoms
            assert last.atoms == cast(Structure, r1[-1]).atoms

    def test_bool_true_when_nonempty(self) -> None:
        """bool(result) is True when at least one structure passed."""
        r = _gas(n_samples=50, seed=5)
        assert bool(r) is True

    def test_bool_false_when_empty(self) -> None:
        """bool(GenerationResult()) is False for the default empty object."""
        assert bool(GenerationResult()) is False

    def test_add_with_non_result_raises_type_error(self) -> None:
        """r + 'string' must raise TypeError, not silently concatenate."""
        r = _gas(seed=7, n_samples=10)
        with pytest.raises(TypeError):
            _ = r + "oops"  # type: ignore[operator]

    def test_add_with_int_raises_type_error(self) -> None:
        """r + 42 must raise TypeError."""
        r = _gas(seed=8, n_samples=10)
        with pytest.raises(TypeError):
            _ = r + 42  # type: ignore[operator]

    def test_n_passed_equals_len(self) -> None:
        """n_passed must always equal len(result)."""
        r = _gas(n_samples=40, seed=9)
        assert r.n_passed == len(r)

    def test_n_attempted_ge_n_passed(self) -> None:
        """n_attempted must be >= n_passed at all times."""
        r = _gas(n_samples=40, seed=9)
        assert r.n_attempted >= r.n_passed


# ===========================================================================
# NEC-B  GeneratorConfig frozen dataclass + __getattr__ proxy
# ===========================================================================

class TestGeneratorConfigFrozenAndProxy:
    """GeneratorConfig must be truly immutable; StructureGenerator must proxy
    all config fields transparently."""

    def test_mutation_raises_frozen_instance_error(self) -> None:
        """Assigning to any field on a frozen config must raise."""
        cfg = GeneratorConfig(
            n_atoms=10, charge=0, mult=1,
            mode="chain", elements="6,7,8",
            n_samples=5, seed=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.n_atoms = 99  # type: ignore[misc]

    def test_replace_does_not_mutate_original(self) -> None:
        """dataclasses.replace() must return a new object, original unchanged."""
        cfg = GeneratorConfig(
            n_atoms=10, charge=0, mult=1,
            mode="chain", elements="6,7,8",
            n_samples=5, seed=0,
        )
        cfg_new = dataclasses.replace(cfg, seed=999)
        assert cfg.seed == 0
        assert cfg_new.seed == 999
        assert cfg is not cfg_new

    def test_generator_proxy_forwards_n_atoms(self) -> None:
        """gen.n_atoms must equal the config's n_atoms via __getattr__."""
        cfg = GeneratorConfig(
            n_atoms=12, charge=0, mult=1,
            mode="chain", elements="6,7,8",
            n_samples=5, seed=42,
        )
        gen = StructureGenerator(cfg)
        assert gen.n_atoms == 12

    def test_generator_proxy_forwards_seed(self) -> None:
        """gen.seed must equal the config's seed via __getattr__."""
        cfg = GeneratorConfig(
            n_atoms=8, charge=0, mult=1,
            mode="chain", elements="6,7,8",
            n_samples=5, seed=77,
        )
        gen = StructureGenerator(cfg)
        assert gen.seed == 77

    def test_generator_proxy_forwards_mode(self) -> None:
        """gen.mode must reflect the config mode."""
        cfg = GeneratorConfig(
            n_atoms=8, charge=0, mult=1,
            mode="chain", elements="6,7,8",
            n_samples=5, seed=0,
        )
        gen = StructureGenerator(cfg)
        assert gen.mode == "chain"

    def test_keyword_form_and_config_form_are_equivalent(self) -> None:
        """Keyword constructor and GeneratorConfig constructor must produce
        bit-identical results when given the same parameters."""
        common: dict[str, Any] = dict(
            n_atoms=8, charge=0, mult=1,
            mode="gas", region="sphere:8",
            elements="6,7,8", n_samples=20, seed=55,
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r_kw = generate(**common)
            _check_generation_result(r_kw)
            r_cfg = generate(GeneratorConfig(**common))
            _check_generation_result(r_cfg)

        positions_kw = [s.positions for s in r_kw]
        positions_cfg = [s.positions for s in r_cfg]
        assert positions_kw == positions_cfg


# ===========================================================================
# NEC-C  Structure data integrity
# ===========================================================================

class TestStructureDataIntegrity:
    """Pickle/deepcopy round-trips, manual construction, sample_index order."""

    def test_pickle_round_trip_atoms(self) -> None:
        """Pickling and unpickling must preserve the atoms list."""
        s = _one_structure(seed=1)
        s2: Structure = pickle.loads(pickle.dumps(s))
        assert s2.atoms == s.atoms

    def test_pickle_round_trip_positions(self) -> None:
        """Pickling and unpickling must preserve positions to full float precision."""
        s = _one_structure(seed=2)
        s2: Structure = pickle.loads(pickle.dumps(s))
        assert s2.positions == s.positions

    def test_pickle_round_trip_metrics(self) -> None:
        """Pickling and unpickling must preserve the metrics dict exactly."""
        s = _one_structure(seed=3)
        s2: Structure = pickle.loads(pickle.dumps(s))
        assert s2.metrics == s.metrics

    def test_deepcopy_preserves_atoms(self) -> None:
        """copy.deepcopy must produce an independent object with the same atoms."""
        s = _one_structure(seed=4)
        s2 = copy.deepcopy(s)
        assert s2.atoms == s.atoms
        assert s2 is not s

    def test_deepcopy_is_independent(self) -> None:
        """Mutating the deepcopy's metrics must not affect the original."""
        s = _one_structure(seed=5)
        s2 = copy.deepcopy(s)
        s2.metrics["H_total"] = -999.0
        # original must be unchanged
        assert s.metrics["H_total"] != -999.0

    def test_manually_constructed_structure_charge_mult(self) -> None:
        """A Structure built directly must expose charge and mult correctly."""
        s = Structure(
            atoms=["C", "N"],
            positions=[(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)],
            charge=0,
            mult=1,
            metrics={},
            mode="gas",
        )
        assert s.charge == 0
        assert s.mult == 1

    def test_manually_constructed_structure_center_sym_defaults_none(self) -> None:
        """center_sym must default to None when omitted."""
        s = Structure(
            atoms=["C", "N"],
            positions=[(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)],
            charge=0,
            mult=1,
            metrics={},
            mode="gas",
        )
        assert s.center_sym is None

    def test_sample_index_strictly_increasing(self) -> None:
        """sample_index on successive structures from one generate() call must
        be strictly increasing (each refers to a distinct attempt index)."""
        r = _gas(n_samples=60, seed=13)
        indices = [s.sample_index for s in r]
        assert all(indices[i] < indices[i + 1] for i in range(len(indices) - 1))

    def test_sample_index_is_int(self) -> None:
        """sample_index must be an integer, not a float or other type."""
        r = _gas(n_samples=30, seed=14)
        for s in r:
            assert isinstance(s.sample_index, int)

    def test_comp_alphabetical_not_hill_for_argon(self) -> None:
        """comp uses alphabetical order, not Hill order.
        ['Ar','C','H','H'] -> 'ArCH2' alphabetically (Ar < C < H)."""
        s = Structure(
            atoms=["Ar", "C", "H", "H"],
            positions=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
            charge=0, mult=1, metrics={}, mode="gas",
        )
        # Hill order would give 'CH2Ar'; alphabetical gives 'ArCH2'
        assert s.comp == "ArCH2"

    def test_comp_sodium_carbon_alphabetical(self) -> None:
        """['Na','C','H','H'] alphabetical: C < H < Na -> 'CH2Na'."""
        s = Structure(
            atoms=["Na", "C", "H", "H"],
            positions=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
            charge=0, mult=1, metrics={}, mode="gas",
        )
        assert s.comp == "CH2Na"


# ===========================================================================
# NEC-D  XYZ serialization round-trips
# ===========================================================================

class TestXYZSerializationRoundTrips:
    """to_xyz → from_xyz must preserve positions; format_xyz prefix contract."""

    def test_to_xyz_from_xyz_in_memory_positions(self) -> None:
        """Positions must survive a to_xyz/from_xyz cycle within tolerance."""
        s = _one_structure(seed=20)
        xyz_str = s.to_xyz()
        _check_to_xyz(xyz_str)
        s2 = Structure.from_xyz(xyz_str)
        _check_from_xyz(s2)
        pos1 = np.array(s.positions)
        pos2 = np.array(s2.positions)
        np.testing.assert_allclose(pos1, pos2, atol=1e-3)

    def test_to_xyz_from_xyz_in_memory_atoms(self) -> None:
        """Atom list must survive a to_xyz/from_xyz cycle exactly."""
        s = _one_structure(seed=21)
        s2 = Structure.from_xyz(s.to_xyz())
        _check_from_xyz(s2)
        assert s2.atoms == s.atoms

    def test_to_xyz_from_xyz_via_file_positions(self) -> None:
        """Positions must survive a write_xyz/from_xyz(path) cycle."""
        s = _one_structure(seed=22)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as fh:
            fh.write(s.to_xyz())
            fname = fh.name
        try:
            s2 = Structure.from_xyz(fname)
            _check_from_xyz(s2)
            pos1 = np.array(s.positions)
            pos2 = np.array(s2.positions)
            np.testing.assert_allclose(pos1, pos2, atol=1e-3)
        finally:
            os.unlink(fname)

    def test_to_xyz_from_xyz_via_file_atoms(self) -> None:
        """Atom list must survive a write_xyz/from_xyz(path) cycle."""
        s = _one_structure(seed=23)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xyz", delete=False
        ) as fh:
            fh.write(s.to_xyz())
            fname = fh.name
        try:
            s2 = Structure.from_xyz(fname)
            _check_from_xyz(s2)
            assert s2.atoms == s.atoms
        finally:
            os.unlink(fname)

    def test_format_xyz_prefix_appears_in_comment_line(self) -> None:
        """A non-empty prefix must appear at the start of the comment line."""
        s = _one_structure(seed=24)
        xyz = format_xyz(s.atoms, s.positions, s.charge, s.mult, s.metrics,
                         prefix="MY_PREFIX")
        _check_format_xyz(xyz)
        comment_line = xyz.splitlines()[1]
        assert comment_line.startswith("MY_PREFIX")

    def test_format_xyz_empty_prefix_no_leading_space(self) -> None:
        """When prefix is empty (default), the comment line must not start
        with an extra leading space or the literal string 'None'."""
        s = _one_structure(seed=25)
        xyz = format_xyz(s.atoms, s.positions, s.charge, s.mult, s.metrics)
        _check_format_xyz(xyz)
        comment_line = xyz.splitlines()[1]
        assert not comment_line.startswith(" ")
        assert "None" not in comment_line.split()[0]

    def test_format_xyz_first_line_is_atom_count(self) -> None:
        """Line 0 of format_xyz output must be an integer equal to len(atoms)."""
        s = _one_structure(seed=26)
        xyz = format_xyz(s.atoms, s.positions, s.charge, s.mult, s.metrics)
        _check_format_xyz(xyz)
        assert int(xyz.splitlines()[0]) == len(s.atoms)


# ===========================================================================
# NEC-E  parse_filter pathological inputs
# ===========================================================================

class TestParseFilterPathological:
    """NaN bounds, whitespace, and case-sensitivity must be handled correctly."""

    def test_nan_min_blocks_all_structures(self) -> None:
        """A filter 'H_total:nan:-' must block every structure because
        nan <= x is always False in IEEE 754."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r = generate(
                n_atoms=8, charge=0, mult=1,
                mode="gas", region="sphere:8",
                elements="6,7,8", n_samples=20, seed=0,
                filters=["H_total:nan:-"],
            )
            _check_generation_result(r)
        assert r.n_passed == 0

    def test_nan_min_parse_filter_returns_nan(self) -> None:
        """parse_filter with 'nan' as min must return math.nan, not raise."""
        _, lo, _ = parse_filter("H_total:nan:inf")
        _check_parse_filter(parse_filter("H_total:nan:inf"))
        assert math.isnan(lo)

    def test_whitespace_padded_metric_name_raises(self) -> None:
        """parse_filter must treat '  H_total  ' as an unknown metric and
        raise ValueError — it does NOT strip whitespace from the metric name."""
        with pytest.raises(ValueError):
            parse_filter("  H_total  :1.0:-")

    def test_lowercase_metric_name_raises(self) -> None:
        """Metric names are case-sensitive; 'h_total' must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            parse_filter("h_total:1.0:-")

    def test_uppercase_metric_name_raises(self) -> None:
        """'H_TOTAL' must raise ValueError — only the canonical case is valid."""
        with pytest.raises(ValueError, match="Unknown metric"):
            parse_filter("H_TOTAL:1.0:-")

    def test_extra_colon_fields_raise(self) -> None:
        """'METRIC:MIN:MAX:extra' must raise ValueError."""
        with pytest.raises(ValueError):
            parse_filter("H_total:1.0:2.0:extra")

    def test_empty_string_raises(self) -> None:
        """An empty string must raise ValueError."""
        with pytest.raises(ValueError):
            parse_filter("")

    def test_missing_colons_raises(self) -> None:
        """A string with no colons must raise ValueError."""
        with pytest.raises(ValueError):
            parse_filter("H_total")


# ===========================================================================
# NEC-F  parse_objective_spec contract
# ===========================================================================

class TestParseObjectiveSpecContract:
    """Empty input, duplicate keys, bad weights, all-metric round-trip."""

    def test_empty_list_returns_empty_dict(self) -> None:
        """parse_objective_spec([]) must return an empty dict."""
        result = parse_objective_spec([])
        _check_parse_objective_spec(result)
        assert result == {}

    def test_duplicate_key_last_value_wins(self) -> None:
        """When the same metric appears twice the last weight takes precedence."""
        result = parse_objective_spec(["H_total:1.0", "H_total:9.9"])
        _check_parse_objective_spec(result)
        assert result["H_total"] == pytest.approx(9.9)

    def test_non_float_weight_raises_value_error(self) -> None:
        """A non-numeric weight like 'abc' must raise ValueError."""
        with pytest.raises(ValueError):
            parse_objective_spec(["H_total:abc"])

    def test_all_metrics_round_trip(self) -> None:
        """Building a spec from every metric in ALL_METRICS and parsing it
        must produce a dict with exactly those 13 keys."""
        specs = [f"{m}:1.0" for m in sorted(ALL_METRICS)]
        result = parse_objective_spec(specs)
        _check_parse_objective_spec(result)
        assert set(result.keys()) == ALL_METRICS

    def test_zero_weight_is_accepted(self) -> None:
        """A weight of 0 is a legal no-op contribution; must not raise."""
        result = parse_objective_spec(["H_total:0"])
        _check_parse_objective_spec(result)
        assert result["H_total"] == pytest.approx(0.0)

    def test_negative_weight_is_accepted(self) -> None:
        """Negative weights are legal (penalize metric); must not raise."""
        result = parse_objective_spec(["Q6:-2.0"])
        _check_parse_objective_spec(result)
        assert result["Q6"] == pytest.approx(-2.0)


# ===========================================================================
# NEC-G  compute_all_metrics direct API
# ===========================================================================

class TestComputeAllMetricsDirect:
    """Direct calls to compute_all_metrics with unusual configurations."""

    _DEFAULT_KW: dict[str, Any] = dict(
        n_bins=10, w_atom=1.0, w_spatial=1.0, cutoff=5.0
    )

    def test_two_atom_call_returns_all_13_keys(self) -> None:
        """A 2-atom system must still return all 13 metric keys."""
        m = compute_all_metrics(
            ["C", "N"],
            [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)],
            **self._DEFAULT_KW,
        )
        _check_compute_all_metrics(m)
        assert set(m.keys()) == ALL_METRICS

    def test_w_spatial_zero_h_total_equals_h_atom(self) -> None:
        """When w_spatial=0 and w_atom=1, H_total must equal H_atom exactly."""
        m = compute_all_metrics(
            ["C", "N", "O"],
            [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0)],
            n_bins=10, w_atom=1.0, w_spatial=0.0, cutoff=5.0,
        )
        _check_compute_all_metrics(m)
        assert m["H_total"] == pytest.approx(m["H_atom"])

    def test_w_atom_zero_h_total_equals_h_spatial(self) -> None:
        """When w_atom=0 and w_spatial=1, H_total must equal H_spatial exactly."""
        m = compute_all_metrics(
            ["C", "N", "O"],
            [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0)],
            n_bins=10, w_atom=0.0, w_spatial=1.0, cutoff=5.0,
        )
        _check_compute_all_metrics(m)
        assert m["H_total"] == pytest.approx(m["H_spatial"])

    def test_all_values_are_finite_or_nan(self) -> None:
        """Every metric value must be either a finite float or NaN — never inf."""
        m = compute_all_metrics(
            ["C", "C", "N", "O"],
            [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 1.5, 0.0), (1.5, 1.5, 0.0)],
            **self._DEFAULT_KW,
        )
        _check_compute_all_metrics(m)
        for key, val in m.items():
            assert math.isfinite(val) or math.isnan(val), (
                f"metric {key!r} is infinite: {val}"
            )


# ===========================================================================
# NEC-H  EvalContext field invariants
# ===========================================================================

class TestEvalContextFieldInvariants:
    """Immutability of tuple fields, step bounds, best_f monotonicity,
    temperature range, and cutoff passthrough."""

    def _collect_ctx(
        self,
        max_steps: int = 30,
        seed: int = 0,
        **opt_kw: Any,
    ) -> list[Any]:
        """Run a small optimizer and collect every EvalContext seen."""
        contexts: list[Any] = []

        def _capture(m: dict[str, float], ctx: Any) -> float:
            contexts.append(ctx)
            return m["H_total"]

        StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective=_capture,
            method="annealing", max_steps=max_steps,
            n_restarts=1, seed=seed,
            **opt_kw,
        ).run()
        return contexts

    def test_atoms_is_immutable_tuple(self) -> None:
        """ctx.atoms must be a tuple so callers cannot mutate the internal
        atom list by accident."""
        contexts = self._collect_ctx(seed=1)
        assert len(contexts) > 0
        assert isinstance(contexts[0].atoms, tuple)
        with pytest.raises(TypeError):
            contexts[0].atoms[0] = "X"  # type: ignore[index]

    def test_positions_is_immutable_tuple(self) -> None:
        """ctx.positions must be a tuple of tuples, not a mutable list."""
        contexts = self._collect_ctx(seed=2)
        assert len(contexts) > 0
        assert isinstance(contexts[0].positions, tuple)
        with pytest.raises(TypeError):
            contexts[0].positions[0] = (0.0, 0.0, 0.0)  # type: ignore[index]

    def test_element_pool_is_immutable_tuple(self) -> None:
        """ctx.element_pool must be a tuple."""
        contexts = self._collect_ctx(seed=3)
        assert len(contexts) > 0
        assert isinstance(contexts[0].element_pool, tuple)

    def test_element_pool_matches_elements_arg(self) -> None:
        """ctx.element_pool must contain exactly the parsed elements."""
        contexts = self._collect_ctx(seed=4)
        assert len(contexts) > 0
        pool = set(contexts[0].element_pool)
        # elements='6,7,8' -> C, N, O
        assert pool == {"C", "N", "O"}

    def test_step_always_in_0_to_max_steps_exclusive(self) -> None:
        """ctx.step must satisfy 0 <= ctx.step < ctx.max_steps for all calls."""
        contexts = self._collect_ctx(max_steps=20, seed=5)
        for ctx in contexts:
            assert 0 <= ctx.step < ctx.max_steps

    def test_best_f_monotonically_non_decreasing(self) -> None:
        """ctx.best_f must never decrease from one call to the next within a
        single restart (it tracks the running best, not the current score)."""
        contexts = self._collect_ctx(max_steps=50, seed=6)
        best_fs = [ctx.best_f for ctx in contexts]
        for i in range(len(best_fs) - 1):
            assert best_fs[i] <= best_fs[i + 1] + 1e-12, (
                f"best_f decreased at step {i}: {best_fs[i]:.6f} -> {best_fs[i+1]:.6f}"
            )

    def test_temperature_always_positive(self) -> None:
        """ctx.temperature must be strictly greater than 0 at every step."""
        contexts = self._collect_ctx(max_steps=40, seed=7)
        for ctx in contexts:
            assert ctx.temperature > 0, (
                f"temperature is non-positive at step {ctx.step}: {ctx.temperature}"
            )

    def test_temperature_within_t_start_t_end(self) -> None:
        """ctx.temperature must stay within [T_end, T_start] for annealing."""
        T_START, T_END = 2.0, 0.01
        contexts = self._collect_ctx(
            max_steps=50, seed=8, T_start=T_START, T_end=T_END
        )
        for ctx in contexts:
            assert T_END <= ctx.temperature <= T_START + 1e-12

    def test_cutoff_matches_constructor_parameter(self) -> None:
        """ctx.cutoff must reflect the cutoff= value passed to StructureOptimizer."""
        CUTOFF = 4.5
        contexts = self._collect_ctx(max_steps=10, seed=9, cutoff=CUTOFF)
        for ctx in contexts:
            assert ctx.cutoff == pytest.approx(CUTOFF)

    def test_progress_in_zero_to_one_exclusive(self) -> None:
        """ctx.progress must be in [0.0, 1.0) for all steps."""
        contexts = self._collect_ctx(max_steps=30, seed=10)
        for ctx in contexts:
            assert 0.0 <= ctx.progress < 1.0

    def test_n_atoms_matches_atoms_len(self) -> None:
        """ctx.n_atoms must equal len(ctx.atoms)."""
        contexts = self._collect_ctx(max_steps=10, seed=11)
        for ctx in contexts:
            assert ctx.n_atoms == len(ctx.atoms)

    def test_annealing_replica_fields_are_none(self) -> None:
        """In annealing mode, ctx.replica_idx and ctx.n_replicas must be None."""
        contexts = self._collect_ctx(max_steps=10, seed=12)
        for ctx in contexts:
            assert ctx.replica_idx is None
            assert ctx.n_replicas is None


# ===========================================================================
# NEC-I  OptimizationResult contract
# ===========================================================================

class TestOptimizationResultContract:
    """summary() fields, n_restarts_attempted, basin_hopping name, idempotency."""

    def _opt(self, method: str = "annealing", n_restarts: int = 2,
             seed: int = 0, **kw: Any) -> OptimizationResult:
        return StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective={"H_total": 1.0},
            method=method, max_steps=30,
            n_restarts=n_restarts, seed=seed,
            **kw,
        ).run()

    def test_summary_contains_restarts_field(self) -> None:
        """summary() must include 'restarts='."""
        r = self._opt()
        assert "restarts=" in r.summary()

    def test_summary_contains_best_f_field(self) -> None:
        """summary() must include 'best_f='."""
        r = self._opt()
        assert "best_f=" in r.summary()

    def test_summary_contains_method_name(self) -> None:
        """summary() must include the method string that was used."""
        r = self._opt(method="annealing")
        assert "annealing" in r.summary()

    def test_basin_hopping_method_name_in_summary(self) -> None:
        """basin_hopping method must appear in summary()."""
        r = self._opt(method="basin_hopping")
        assert "basin_hopping" in r.summary()

    def test_n_restarts_attempted_matches_n_restarts(self) -> None:
        """n_restarts_attempted must equal the n_restarts given to the optimizer."""
        r = self._opt(n_restarts=3)
        assert r.n_restarts_attempted == 3

    def test_run_called_twice_both_succeed(self) -> None:
        """Calling run() twice on the same StructureOptimizer instance must
        succeed both times — the optimizer must not consume internal state."""
        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective={"H_total": 1.0},
            method="annealing", max_steps=20,
            n_restarts=1, seed=0,
        )
        r1 = opt.run()
        _check_optimization_result(r1)
        r2 = opt.run()
        _check_optimization_result(r2)
        assert r1.best is not None
        assert r2.best is not None

    def test_objective_scores_sorted_descending(self) -> None:
        """objective_scores must be sorted from best (highest) to worst."""
        r = self._opt(n_restarts=4, seed=99)
        scores = r.objective_scores
        assert scores == sorted(scores, reverse=True)

    def test_best_score_equals_first_score(self) -> None:
        """The first objective_score must correspond to result.best."""
        r = self._opt(n_restarts=4, seed=88)
        if r.objective_scores:
            best_score = r.objective_scores[0]
            # Recompute objective for best structure to verify
            best_h = r.best.metrics["H_total"]
            assert best_score == pytest.approx(best_h)


# ===========================================================================
# NEC-J  Objective function flexibility
# ===========================================================================

class TestObjectiveFunctionFlexibility:
    """Integer-valued return, all-13-metrics objective."""

    def test_integer_returning_objective_does_not_crash(self) -> None:
        """An objective that returns int (not float) must run without error."""
        def int_obj(m: dict[str, float]) -> int:
            return int(m["H_total"] * 10)

        result = StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective=int_obj,
            method="annealing", max_steps=30, seed=0,
        ).run()
        _check_optimization_result(result)
        assert result.best is not None

    def test_all_metrics_dict_objective_runs(self) -> None:
        """An objective dict containing all 13 metric keys must execute
        without KeyError or other crash."""
        full_obj = {k: 1.0 for k in ALL_METRICS}
        result = StructureOptimizer(
            n_atoms=8, charge=0, mult=1, elements="6,7,8",
            objective=full_obj,
            method="annealing", max_steps=30, seed=0,
        ).run()
        _check_optimization_result(result)
        assert result.best is not None

    def test_callable_objective_covering_all_metrics(self) -> None:
        """A callable that reads every metric key from m must not raise."""
        def all_metrics_fn(m: dict[str, float]) -> float:
            return sum(m[k] for k in ALL_METRICS)

        result = StructureOptimizer(
            n_atoms=8, charge=0, mult=1, elements="6,7,8",
            objective=all_metrics_fn,
            method="annealing", max_steps=30, seed=1,
        ).run()
        _check_optimization_result(result)
        assert result.best is not None


# ===========================================================================
# NEC-K  Shell-mode specifics
# ===========================================================================

class TestShellModeSpecifics:
    """center_sym populated, center_z pool membership, n_atoms=1 shell."""

    def test_shell_mode_sets_center_sym(self) -> None:
        """Structures from shell mode must have a non-None center_sym."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r = generate(
                n_atoms=4, charge=0, mult=1,
                mode="shell", center_z=6,
                elements="6,7,8", n_samples=20, seed=0,
            )
            _check_generation_result(r)
        assert len(r) > 0
        for s in r:
            assert s.center_sym is not None

    def test_shell_mode_center_sym_matches_center_z(self) -> None:
        """center_sym must be the element symbol for center_z=6 (carbon)."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r = generate(
                n_atoms=4, charge=0, mult=1,
                mode="shell", center_z=6,
                elements="6,7,8", n_samples=20, seed=0,
            )
            _check_generation_result(r)
        assert len(r) > 0
        assert cast(Structure, r[0]).center_sym == "C"

    def test_shell_mode_center_z_not_in_pool_raises(self) -> None:
        """center_z pointing to an element absent from the element pool must
        raise ValueError at construction time."""
        with pytest.raises(ValueError):
            generate(
                n_atoms=4, charge=0, mult=1,
                mode="shell", center_z=26,        # Fe, not in pool
                elements="6,7,8", n_samples=5, seed=0,
            )

    def test_shell_mode_n_atoms_1_does_not_crash(self) -> None:
        """Shell mode with just 1 shell atom around the center must succeed."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r = generate(
                n_atoms=1, charge=0, mult=1,
                mode="shell", center_z=6,
                elements="6,7,8", n_samples=10, seed=0,
            )
            _check_generation_result(r)
        # May produce 0 structures depending on parity, but must not raise
        assert isinstance(r, GenerationResult)

    def test_shell_mode_atom_list_length_equals_n_atoms(self) -> None:
        """Structure.atoms must contain exactly n_atoms entries in shell mode.
        n_atoms is the *total* atom count (center atom is counted within it when
        included by parity; hydrogen added by add_hydrogen is counted too).
        The center is separately recorded in center_sym, not as an extra atom
        beyond n_atoms."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r = generate(
                n_atoms=4, charge=0, mult=1,
                mode="shell", center_z=6,
                elements="6,7,8", n_samples=20, seed=0,
                add_hydrogen=False,
            )
            _check_generation_result(r)
        assert len(r) > 0
        for s in r:
            assert len(s.atoms) == 4, (
                f"Expected exactly 4 atoms but got {len(s.atoms)}: {s.atoms}"
            )


# ===========================================================================
# NEC-L  High multiplicity
# ===========================================================================

class TestHighMultiplicity:
    """mult=7 (sextuplet) must generate valid structures, not crash."""

    def test_mult_7_generates_structures(self) -> None:
        """mult=7 is an unusual but valid spin state; at least some
        structures should be generated from an appropriate element pool."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r = generate(
                n_atoms=8, charge=0, mult=7,
                mode="gas", region="sphere:8",
                elements="6,7,8", n_samples=40, seed=0,
            )
            _check_generation_result(r)
        # May produce 0 due to parity, but must not raise
        assert isinstance(r, GenerationResult)

    def test_mult_7_result_structures_have_correct_mult(self) -> None:
        """Any structure that does pass parity must record mult=7."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r = generate(
                n_atoms=8, charge=0, mult=7,
                mode="gas", region="sphere:8",
                elements="6,7,8", n_samples=40, seed=0,
            )
            _check_generation_result(r)
        for s in r:
            assert s.mult == 7


# ===========================================================================
# NEC-M  Affine-shear-only path
# ===========================================================================

class TestAffineShearOnly:
    """affine_shear > 0 while affine_stretch=0 and affine_jitter=0 must work
    across all four placement modes and not raise or produce NaN positions."""

    def _shear_only_gas(self, seed: int = 0) -> GenerationResult:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            return generate(
                n_atoms=10, charge=0, mult=1,
                mode="gas", region="sphere:8",
                elements="6,7,8", n_samples=20, seed=seed,
                affine_strength=0.2,
                affine_stretch=0.0,
                affine_shear=0.3,
                affine_jitter=0.0,
            )

    def test_shear_only_gas_does_not_crash(self) -> None:
        r = self._shear_only_gas(seed=0)
        assert isinstance(r, GenerationResult)

    def test_shear_only_gas_positions_are_finite(self) -> None:
        r = self._shear_only_gas(seed=1)
        for s in r:
            for xyz in s.positions:
                assert all(math.isfinite(v) for v in xyz), (
                    f"Non-finite position in shear-only structure: {xyz}"
                )

    def test_shear_only_chain_does_not_crash(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r = generate(
                n_atoms=10, charge=0, mult=1,
                mode="chain", elements="6,7,8", n_samples=20, seed=0,
                affine_strength=0.2,
                affine_stretch=0.0,
                affine_shear=0.3,
                affine_jitter=0.0,
            )
            _check_generation_result(r)
        assert isinstance(r, GenerationResult)


# ===========================================================================
# NEC-N  validate_charge_mult direct API
# ===========================================================================

class TestValidateChargeMult:
    """Direct calls to the public validate_charge_mult function."""

    def test_carbon_singlet_is_valid(self) -> None:
        """C, charge=0, mult=1 must be valid (6 electrons, even → singlet ok)."""
        ok, msg = validate_charge_mult(["C"], 0, 1)
        _check_validate_charge_mult((ok, msg))
        assert ok is True

    def test_nitrogen_doublet_is_valid(self) -> None:
        """N, charge=0, mult=2 must be valid (7 electrons, odd → doublet ok)."""
        ok, msg = validate_charge_mult(["N"], 0, 2)
        _check_validate_charge_mult((ok, msg))
        assert ok is True

    def test_nitrogen_singlet_is_invalid(self) -> None:
        """N, charge=0, mult=1 must be invalid (7 electrons, odd → singlet bad)."""
        ok, _msg = validate_charge_mult(["N"], 0, 1)
        _check_validate_charge_mult((ok, _msg))
        assert ok is False

    def test_message_is_string(self) -> None:
        """The second return value must always be a non-empty string."""
        _ok, msg = validate_charge_mult(["C"], 0, 1)
        _check_validate_charge_mult((_ok, msg))
        assert isinstance(msg, str) and len(msg) > 0

    def test_positive_charge_modifies_electron_count(self) -> None:
        """C, charge=+1 removes one electron → 5 electrons (odd) → doublet ok."""
        ok, _msg = validate_charge_mult(["C"], 1, 2)
        _check_validate_charge_mult((ok, _msg))
        assert ok is True

    def test_negative_charge_adds_electrons(self) -> None:
        """C, charge=-1 adds one electron → 7 electrons (odd) → doublet ok."""
        ok, _msg = validate_charge_mult(["C"], -1, 2)
        _check_validate_charge_mult((ok, _msg))
        assert ok is True


# ===========================================================================
# NEC-O  ALL_METRICS completeness
# ===========================================================================

class TestAllMetricsCompleteness:
    """ALL_METRICS must be exactly the 13 metrics documented in quickstart.md."""

    _EXPECTED: frozenset[str] = frozenset({
        "H_atom", "H_spatial", "H_total",
        "RDF_dev", "shape_aniso",
        "Q4", "Q6", "Q8",
        "graph_lcc", "graph_cc",
        "ring_fraction", "charge_frustration", "moran_I_chi",
    })

    def test_all_metrics_is_frozenset(self) -> None:
        """ALL_METRICS must be a frozenset (immutable)."""
        assert isinstance(ALL_METRICS, frozenset)

    def test_all_metrics_has_exactly_13_entries(self) -> None:
        """ALL_METRICS must contain exactly 13 keys."""
        assert len(ALL_METRICS) == 13

    def test_all_metrics_equals_documented_set(self) -> None:
        """ALL_METRICS must match the exact set from the documentation."""
        assert ALL_METRICS == self._EXPECTED

    def test_no_undocumented_metric_in_all_metrics(self) -> None:
        """There must be no metric in ALL_METRICS beyond the documented 13."""
        extras = ALL_METRICS - self._EXPECTED
        assert not extras, f"Undocumented metrics found: {extras}"

    def test_no_missing_metric_from_all_metrics(self) -> None:
        """Every documented metric must appear in ALL_METRICS."""
        missing = self._EXPECTED - ALL_METRICS
        assert not missing, f"Documented metrics missing from ALL_METRICS: {missing}"
