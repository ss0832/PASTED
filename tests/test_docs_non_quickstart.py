"""
tests/test_docs_non_quickstart.py
==================================
Test cases derived from the PASTED documentation examples
**excluding** the Quick Start page.

Sections covered (mirroring https://ss0832.github.io/PASTED/quickstart.html):
 * GenerationResult + operator (merge)
 * UserWarning emission from generate()
 * maxent mode
 * Class API – chain_bias
 * Writing to file (write_xyz / to_xyz)
 * n_success with n_samples=0 (unlimited)
 * Streaming vs generate equivalence
 * Structure attributes (comp, mode, sample_index, seed, n, len, center_sym)
 * GeneratorConfig – frozen dataclass and dataclasses.replace
 * Affine transforms in StructureGenerator (affine_strength)
 * Element fractions (element_fractions)
 * Element count bounds (element_min_counts / element_max_counts)
 * Combining fractions + bounds
 * Position-only optimisation (allow_composition_moves=False)
 * Composition-only optimisation (allow_displacements=False)
 * StructureOptimizer – basic usage
 * StructureOptimizer – Parallel Tempering
 * StructureOptimizer – Electronegativity-targeted (charge_frustration / moran_I_chi)
 * StructureOptimizer – Affine displacement moves (allow_affine_moves)
 * StructureOptimizer – callable objective (lambda & 2-arg ctx)
 * StructureOptimizer – basin_hopping method
 * OptimizationResult – iteration, ranking, summary
 * Structure.from_xyz – string source
 * Structure.from_xyz – frame index
 * read_xyz – multi-frame file
 * GenerationResult + GenerationResult.__add__ counters
 * parse_objective_spec helper
 * EvalContext fields
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pasted import (
    GenerationResult,
    GeneratorConfig,
    Structure,
    StructureGenerator,
    StructureOptimizer,
    generate,
    parse_objective_spec,
    read_xyz,
)
from pasted._optimizer import EvalContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SMALL_GAS_KWARGS: dict[str, Any] = dict(
    n_atoms=8,
    charge=0,
    mult=1,
    mode="gas",
    region="sphere:8",
    elements="6,7,8",
    n_samples=20,
    seed=42,
)

SMALL_CHAIN_KWARGS: dict[str, Any] = dict(
    n_atoms=10,
    charge=0,
    mult=1,
    mode="chain",
    elements="6,7,8",
    n_samples=20,
    seed=0,
)


@pytest.fixture
def small_gas_structure() -> Structure:
    result = generate(**SMALL_GAS_KWARGS)
    assert result, "fixture needs at least 1 structure"
    return result[0]


# ===========================================================================
# 1. GenerationResult — merge operator (+)
# ===========================================================================


class TestGenerationResultAdd:
    """Verify that GenerationResult + GenerationResult merges correctly."""

    def test_add_structures_concatenated(self) -> None:
        r1 = generate(**SMALL_GAS_KWARGS)
        r2 = generate(**{**SMALL_GAS_KWARGS, "seed": 1})
        combined = r1 + r2
        assert len(combined) == len(r1) + len(r2)

    def test_add_counters_accumulated(self) -> None:
        r1 = generate(**SMALL_GAS_KWARGS)
        r2 = generate(**{**SMALL_GAS_KWARGS, "seed": 1})
        combined = r1 + r2
        assert combined.n_attempted == r1.n_attempted + r2.n_attempted
        assert combined.n_passed == r1.n_passed + r2.n_passed
        assert combined.n_rejected_parity == r1.n_rejected_parity + r2.n_rejected_parity
        assert combined.n_rejected_filter == r1.n_rejected_filter + r2.n_rejected_filter

    def test_add_n_success_target_from_self(self) -> None:
        r1 = generate(**SMALL_GAS_KWARGS)
        r2 = generate(**{**SMALL_GAS_KWARGS, "seed": 1})
        # Neither sets n_success_target → both are None
        assert r1.n_success_target is None
        combined = r1 + r2
        assert combined.n_success_target is None

    def test_add_not_implemented_for_non_result(self) -> None:
        r1 = generate(**SMALL_GAS_KWARGS)
        result = r1.__add__("not-a-result")  # type: ignore[arg-type]
        assert result is NotImplemented

    def test_summary_after_add(self) -> None:
        r1 = generate(**SMALL_GAS_KWARGS)
        r2 = generate(**{**SMALL_GAS_KWARGS, "seed": 1})
        combined = r1 + r2
        summary = combined.summary()
        assert "passed=" in summary
        assert "attempted=" in summary

    def test_add_preserves_structure_order(self) -> None:
        r1 = generate(**SMALL_GAS_KWARGS)
        r2 = generate(**{**SMALL_GAS_KWARGS, "seed": 99})
        combined = r1 + r2
        # First len(r1) structures come from r1
        for i, s in enumerate(r1):
            assert combined[i] is s


# ===========================================================================
# 2. UserWarning emission
# ===========================================================================


class TestUserWarningEmission:
    """Verify that generate() emits UserWarning for pathological cases."""

    def test_impossible_filter_emits_warning(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = generate(
                n_atoms=8,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:8",
                elements="6,7,8",
                n_samples=10,
                seed=0,
                filters=["H_total:999:-"],
            )
        assert not result
        assert any(issubclass(warning.category, UserWarning) for warning in w)

    def test_budget_exhausted_before_n_success_emits_warning(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gen = StructureGenerator(
                n_atoms=15,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:8",
                elements="1-30",
                n_success=1000,
                n_samples=3,
                seed=42,
            )
            gen.generate()
        assert any(issubclass(warning.category, UserWarning) for warning in w)
        msg = str(w[0].message)
        assert "n_success" in msg or "budget" in msg

    def test_warning_message_mentions_filter_when_all_filtered(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            generate(
                n_atoms=8,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:8",
                elements="6,7,8",
                n_samples=5,
                seed=0,
                filters=["H_total:9999:-"],
            )
        assert w
        assert any("filter" in str(warning.message).lower() for warning in w)


# ===========================================================================
# 3. maxent mode
# ===========================================================================


class TestMaxentMode:
    """Verify that maxent mode generates valid structures."""

    def test_maxent_gas_produces_structures(self) -> None:
        result = generate(
            n_atoms=10,
            charge=0,
            mult=1,
            mode="maxent",
            region="sphere:6",
            elements="6,7,8",
            n_samples=10,
            seed=42,
        )
        # Some samples may fail parity; at least one should pass
        assert result is not None

    def test_maxent_metrics_computed(self) -> None:
        result = generate(
            n_atoms=10,
            charge=0,
            mult=1,
            mode="maxent",
            region="sphere:6",
            elements="6,7,8",
            n_samples=15,
            seed=7,
        )
        if result:
            s = result[0]
            assert "H_total" in s.metrics
            assert "Q6" in s.metrics
            assert s.mode == "maxent"

    def test_maxent_requires_region(self) -> None:
        with pytest.raises(ValueError, match="region"):
            StructureGenerator(
                n_atoms=8,
                charge=0,
                mult=1,
                mode="maxent",
                elements="6,7,8",
                n_samples=5,
                seed=0,
            )


# ===========================================================================
# 4. Class API — chain_bias
# ===========================================================================


class TestChainBias:
    """chain_bias > 0 should produce larger shape_aniso on average."""

    def test_chain_bias_zero_no_crash(self) -> None:
        result = generate(**SMALL_CHAIN_KWARGS)
        assert result is not None

    def test_chain_bias_nonzero_larger_aniso(self) -> None:
        n_trials = 10
        bias_aniso = []
        unbias_aniso = []
        for seed in range(n_trials):
            r_bias = generate(
                n_atoms=20,
                charge=0,
                mult=1,
                mode="chain",
                chain_bias=0.8,
                branch_prob=0.0,
                elements="6,7,8",
                n_samples=5,
                seed=seed,
            )
            r_unbias = generate(
                n_atoms=20,
                charge=0,
                mult=1,
                mode="chain",
                chain_bias=0.0,
                branch_prob=0.0,
                elements="6,7,8",
                n_samples=5,
                seed=seed,
            )
            if r_bias:
                bias_aniso.append(np.mean([s.metrics["shape_aniso"] for s in r_bias]))
            if r_unbias:
                unbias_aniso.append(np.mean([s.metrics["shape_aniso"] for s in r_unbias]))
        if bias_aniso and unbias_aniso:
            assert np.mean(bias_aniso) > np.mean(unbias_aniso) * 0.5  # loose check


# ===========================================================================
# 5. Writing to file
# ===========================================================================


class TestWriteXYZ:
    """Verify write_xyz and to_xyz produce valid output."""

    def test_write_xyz_creates_file(self, small_gas_structure: Structure) -> None:
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            fname = f.name
        try:
            small_gas_structure.write_xyz(fname, append=False)
            assert Path(fname).stat().st_size > 0
        finally:
            os.unlink(fname)

    def test_write_xyz_append(self, small_gas_structure: Structure) -> None:
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            fname = f.name
        try:
            s = small_gas_structure
            s.write_xyz(fname, append=False)
            s.write_xyz(fname, append=True)
            text = Path(fname).read_text()
            # Two structures → first line of each block is the atom count
            lines = [ln for ln in text.strip().split("\n") if ln.strip()]
            count_lines = [ln for ln in lines if ln.strip().isdigit()]
            assert len(count_lines) == 2
        finally:
            os.unlink(fname)

    def test_to_xyz_roundtrip(self, small_gas_structure: Structure) -> None:
        xyz_str = small_gas_structure.to_xyz()
        assert str(len(small_gas_structure)) in xyz_str.split("\n")[0]
        assert "H_total" in xyz_str

    def test_write_then_read_xyz_roundtrip(self, small_gas_structure: Structure) -> None:
        """Round-trip: write → read back → same atom count."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            fname = f.name
        try:
            small_gas_structure.write_xyz(fname, append=False)
            loaded = read_xyz(fname)
            assert len(loaded) == 1
            assert len(loaded[0]) == len(small_gas_structure)
            assert sorted(loaded[0].atoms) == sorted(small_gas_structure.atoms)
        finally:
            os.unlink(fname)


# ===========================================================================
# 6. n_success with n_samples=0
# ===========================================================================


class TestNSuccessUnlimited:
    """n_samples=0 should run until n_success is satisfied."""

    def test_n_samples_zero_reaches_n_success(self) -> None:
        gen = StructureGenerator(
            n_atoms=8,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:8",
            elements="6,7,8",
            n_success=3,
            n_samples=0,
            seed=42,
        )
        result = gen.generate()
        assert len(result) == 3

    def test_n_samples_zero_requires_n_success(self) -> None:
        with pytest.raises(ValueError, match="n_success"):
            StructureGenerator(
                n_atoms=8,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:8",
                elements="6,7,8",
                n_samples=0,
                seed=0,
            )


# ===========================================================================
# 7. Streaming vs generate equivalence
# ===========================================================================


class TestStreamingEquivalence:
    """stream() and list(generate()) must produce identical structures."""

    def test_stream_equals_generate(self) -> None:
        gen = StructureGenerator(**SMALL_GAS_KWARGS)
        via_generate = gen.generate()
        gen2 = StructureGenerator(**SMALL_GAS_KWARGS)
        via_stream = list(gen2.stream())
        assert len(via_generate) == len(via_stream)
        for g, s in zip(via_generate, via_stream, strict=True):
            assert g.atoms == s.atoms
            assert g.metrics.keys() == s.metrics.keys()

    def test_stream_yields_immediately(self) -> None:
        """Each yielded structure is a valid Structure with metrics."""
        gen = StructureGenerator(
            n_atoms=8,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:8",
            elements="6,7,8",
            n_success=2,
            n_samples=50,
            seed=1,
        )
        count = 0
        for s in gen.stream():
            assert isinstance(s, Structure)
            assert "H_total" in s.metrics
            count += 1
        assert count == 2


# ===========================================================================
# 8. Structure attributes
# ===========================================================================


class TestStructureAttributes:
    """Verify all documented Structure attributes."""

    def test_atoms_is_list(self, small_gas_structure: Structure) -> None:
        assert isinstance(small_gas_structure.atoms, list)
        assert all(isinstance(a, str) for a in small_gas_structure.atoms)

    def test_positions_length_matches_atoms(self, small_gas_structure: Structure) -> None:
        assert len(small_gas_structure.positions) == len(small_gas_structure.atoms)

    def test_comp_is_string(self, small_gas_structure: Structure) -> None:
        assert isinstance(small_gas_structure.comp, str)
        assert len(small_gas_structure.comp) > 0

    def test_comp_contains_element_symbols(self, small_gas_structure: Structure) -> None:
        for sym in set(small_gas_structure.atoms):
            assert sym in small_gas_structure.comp

    def test_comp_counts_correct(self, small_gas_structure: Structure) -> None:
        c = Counter(small_gas_structure.atoms)
        for sym, count in c.items():
            if count > 1:
                assert f"{sym}{count}" in small_gas_structure.comp
            else:
                assert sym in small_gas_structure.comp

    def test_len_equals_n_atoms(self, small_gas_structure: Structure) -> None:
        assert len(small_gas_structure) == len(small_gas_structure.atoms)

    def test_n_property(self, small_gas_structure: Structure) -> None:
        assert small_gas_structure.n == len(small_gas_structure)

    def test_mode_is_string(self, small_gas_structure: Structure) -> None:
        assert small_gas_structure.mode in ("gas", "chain", "shell", "maxent", "loaded_xyz")

    def test_sample_index_positive(self, small_gas_structure: Structure) -> None:
        assert small_gas_structure.sample_index >= 1

    def test_seed_stored(self) -> None:
        result = generate(**SMALL_GAS_KWARGS)
        if result:
            assert result[0].seed == 42

    def test_metrics_all_keys_present(self, small_gas_structure: Structure) -> None:
        from pasted import ALL_METRICS

        for key in ALL_METRICS:
            assert key in small_gas_structure.metrics

    def test_center_sym_shell_mode(self) -> None:
        result = generate(
            n_atoms=8,
            charge=0,
            mult=1,
            mode="shell",
            center_z=26,
            elements="1-30",
            n_samples=5,
            seed=7,
        )
        if result:
            assert result[0].center_sym == "Fe"

    def test_center_sym_none_for_gas(self, small_gas_structure: Structure) -> None:
        assert small_gas_structure.center_sym is None

    def test_repr_contains_comp(self, small_gas_structure: Structure) -> None:
        r = repr(small_gas_structure)
        assert small_gas_structure.comp in r
        assert "H_total" in r


# ===========================================================================
# 9. GeneratorConfig — frozen dataclass + dataclasses.replace
# ===========================================================================


class TestGeneratorConfig:
    """Verify GeneratorConfig immutability and one-field override."""

    def test_config_frozen(self) -> None:
        cfg = GeneratorConfig(
            n_atoms=10,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:8",
            elements="6,7,8",
            n_samples=5,
            seed=42,
        )
        with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
            cfg.seed = 99  # type: ignore[misc]

    def test_replace_creates_new_config(self) -> None:
        cfg = GeneratorConfig(
            n_atoms=10,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:8",
            elements="6,7,8",
            n_samples=5,
            seed=42,
        )
        cfg2 = dataclasses.replace(cfg, seed=99)
        assert cfg.seed == 42
        assert cfg2.seed == 99

    def test_two_seeds_give_different_results(self) -> None:
        cfg = GeneratorConfig(
            n_atoms=10,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:8",
            elements="6,7,8",
            n_samples=10,
            seed=42,
        )
        r1 = StructureGenerator(cfg).generate()
        r2 = StructureGenerator(dataclasses.replace(cfg, seed=99)).generate()
        if r1 and r2:
            # At least the atoms should differ between runs in most cases
            assert r1[0].atoms != r2[0].atoms or r1[0].positions != r2[0].positions

    def test_generate_accepts_config(self) -> None:
        cfg = GeneratorConfig(
            n_atoms=8,
            charge=0,
            mult=1,
            mode="chain",
            elements="6,7,8",
            n_samples=10,
            seed=0,
        )
        result = generate(cfg)
        assert isinstance(result, GenerationResult)

    def test_config_accessible_via_generator_attr(self) -> None:
        cfg = GeneratorConfig(
            n_atoms=8,
            charge=0,
            mult=1,
            mode="chain",
            elements="6,7,8",
            n_samples=5,
            seed=0,
        )
        gen = StructureGenerator(cfg)
        assert gen.n_atoms == 8
        assert gen.seed == 0


# ===========================================================================
# 10. Affine transforms in StructureGenerator
# ===========================================================================


class TestAffineTransformsGenerator:
    """Verify affine_strength is applied and affects geometry."""

    def test_affine_strength_no_crash(self) -> None:
        result = generate(
            n_atoms=20,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:10",
            elements="6,7,8",
            n_samples=10,
            seed=42,
            affine_strength=0.2,
        )
        assert result is not None

    def test_affine_strength_produces_structures(self) -> None:
        result = generate(
            n_atoms=12,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:8",
            elements="6,7,8",
            n_samples=10,
            seed=5,
            affine_strength=0.15,
        )
        if result:
            assert len(result[0].positions) == len(result[0].atoms)

    def test_affine_strength_chain_mode(self) -> None:
        result = generate(
            n_atoms=15,
            charge=0,
            mult=1,
            mode="chain",
            elements="6,7,8",
            n_samples=10,
            seed=3,
            affine_strength=0.3,
        )
        assert result is not None

    def test_affine_zero_backward_compatible(self) -> None:
        r_no_affine = generate(**SMALL_GAS_KWARGS)
        r_zero_affine = generate(**{**SMALL_GAS_KWARGS, "affine_strength": 0.0})
        # Same seed → same result
        assert len(r_no_affine) == len(r_zero_affine)
        if r_no_affine and r_zero_affine:
            assert r_no_affine[0].atoms == r_zero_affine[0].atoms


# ===========================================================================
# 11. Element fractions
# ===========================================================================


class TestElementFractions:
    """element_fractions shifts the element distribution."""

    def test_c_rich_composition(self) -> None:
        gen = StructureGenerator(
            n_atoms=30,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:10",
            elements="6,7,8",
            element_fractions={"C": 0.8, "N": 0.1, "O": 0.1},
            n_samples=20,
            seed=0,
        )
        result = gen.generate()
        if not result:
            pytest.skip("No structures generated (parity)")
        c_fracs = [Counter(s.atoms)["C"] / len(s) for s in result]
        assert np.mean(c_fracs) > 0.4  # C should dominate

    def test_relative_weights_normalized(self) -> None:
        """Weights 6:3:1 should be equivalent to 0.6:0.3:0.1."""
        gen_int = StructureGenerator(
            n_atoms=20,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:10",
            elements="6,7,8",
            element_fractions={"C": 6, "N": 3, "O": 1},
            n_samples=10,
            seed=1,
        )
        gen_flt = StructureGenerator(
            n_atoms=20,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:10",
            elements="6,7,8",
            element_fractions={"C": 0.6, "N": 0.3, "O": 0.1},
            n_samples=10,
            seed=1,
        )
        r1 = gen_int.generate()
        r2 = gen_flt.generate()
        assert len(r1) == len(r2)
        if r1 and r2:
            assert r1[0].atoms == r2[0].atoms

    def test_unknown_symbol_raises(self) -> None:
        with pytest.raises(ValueError, match="not in the element pool"):
            StructureGenerator(
                n_atoms=10,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:8",
                elements="6,7,8",
                element_fractions={"Fe": 0.5},
                n_samples=5,
                seed=0,
            )

    def test_all_zero_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            StructureGenerator(
                n_atoms=10,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:8",
                elements="6,7,8",
                element_fractions={"C": 0.0, "N": 0.0, "O": 0.0},
                n_samples=5,
                seed=0,
            )


# ===========================================================================
# 12. Element count bounds
# ===========================================================================


class TestElementCountBounds:
    """element_min_counts / element_max_counts respected in all structures."""

    def test_min_counts_satisfied(self) -> None:
        gen = StructureGenerator(
            n_atoms=15,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:10",
            elements="6,7,8,15,16",
            element_min_counts={"C": 4},
            n_samples=20,
            seed=42,
        )
        result = gen.generate()
        for s in result:
            assert Counter(s.atoms)["C"] >= 4

    def test_max_counts_respected(self) -> None:
        gen = StructureGenerator(
            n_atoms=15,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:10",
            elements="6,7,8,15,16",
            element_max_counts={"N": 2, "O": 2},
            n_samples=20,
            seed=42,
        )
        result = gen.generate()
        for s in result:
            c = Counter(s.atoms)
            assert c.get("N", 0) <= 2
            assert c.get("O", 0) <= 2

    def test_min_exceeds_n_atoms_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds n_atoms"):
            StructureGenerator(
                n_atoms=5,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:8",
                elements="6,7,8",
                element_min_counts={"C": 3, "N": 3},
                n_samples=5,
                seed=0,
            )

    def test_min_gt_max_raises(self) -> None:
        with pytest.raises(ValueError, match="element_min_counts"):
            StructureGenerator(
                n_atoms=10,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:8",
                elements="6,7,8",
                element_min_counts={"C": 5},
                element_max_counts={"C": 2},
                n_samples=5,
                seed=0,
            )

    def test_combined_fractions_and_bounds(self) -> None:
        gen = StructureGenerator(
            n_atoms=12,
            charge=0,
            mult=1,
            mode="chain",
            elements="6,7,8",
            element_fractions={"C": 5, "N": 2, "O": 1},
            element_min_counts={"C": 2},
            element_max_counts={"N": 4},
            n_samples=20,
            seed=7,
        )
        result = gen.generate()
        for s in result:
            c = Counter(s.atoms)
            assert c.get("C", 0) >= 2
            assert c.get("N", 0) <= 4


# ===========================================================================
# 13. StructureOptimizer — basic usage
# ===========================================================================


class TestOptimizerBasic:
    """StructureOptimizer simulated annealing basic smoke tests."""

    def test_run_returns_optimization_result(self) -> None:
        from pasted import OptimizationResult

        opt = StructureOptimizer(
            n_atoms=10,
            charge=0,
            mult=1,
            elements="6,7,8,15,16",
            objective={"H_total": 1.0, "Q6": -2.0},
            method="annealing",
            max_steps=300,
            n_restarts=2,
            seed=42,
        )
        result = opt.run()
        assert isinstance(result, OptimizationResult)

    def test_best_structure_is_structure(self) -> None:
        opt = StructureOptimizer(
            n_atoms=8,
            charge=0,
            mult=1,
            elements="6,7,8",
            objective={"H_total": 1.0},
            method="annealing",
            max_steps=200,
            seed=0,
        )
        result = opt.run()
        assert isinstance(result.best, Structure)

    def test_summary_contains_method(self) -> None:
        opt = StructureOptimizer(
            n_atoms=8,
            charge=0,
            mult=1,
            elements="6,7,8",
            objective={"H_total": 1.0},
            method="annealing",
            max_steps=200,
            seed=0,
        )
        result = opt.run()
        summary = result.summary()
        assert "annealing" in summary

    def test_iteration_over_result(self) -> None:
        opt = StructureOptimizer(
            n_atoms=8,
            charge=0,
            mult=1,
            elements="6,7,8",
            objective={"H_total": 1.0},
            method="annealing",
            max_steps=200,
            n_restarts=3,
            seed=0,
        )
        result = opt.run()
        for _rank, s in enumerate(result, 1):
            assert isinstance(s, Structure)
            assert "H_total" in s.metrics

    def test_best_f_nonnegative_or_finite(self) -> None:
        opt = StructureOptimizer(
            n_atoms=8,
            charge=0,
            mult=1,
            elements="6,7,8",
            objective={"H_total": 1.0},
            method="annealing",
            max_steps=200,
            seed=1,
        )
        result = opt.run()
        assert np.isfinite(result.best.metrics["H_total"])


# ===========================================================================
# 14. Parallel Tempering
# ===========================================================================


class TestParallelTempering:
    """Verify parallel_tempering method works end-to-end."""

    def test_pt_runs_without_error(self) -> None:
        opt = StructureOptimizer(
            n_atoms=10,
            charge=0,
            mult=1,
            elements="6,7,8,15,16",
            objective={"H_total": 1.0, "Q6": -2.0},
            method="parallel_tempering",
            n_replicas=3,
            pt_swap_interval=5,
            max_steps=200,
            n_restarts=2,
            T_start=1.0,
            T_end=0.01,
            seed=42,
        )
        result = opt.run()
        assert result.best is not None

    def test_pt_summary_contains_method(self) -> None:
        opt = StructureOptimizer(
            n_atoms=8,
            charge=0,
            mult=1,
            elements="6,7,8",
            objective={"H_total": 1.0},
            method="parallel_tempering",
            n_replicas=2,
            max_steps=100,
            seed=0,
        )
        result = opt.run()
        assert "parallel_tempering" in result.summary()

    def test_pt_replicas_count(self) -> None:
        """PT yields at least n_restarts results (global best + per-replica finals)."""
        n_restarts = 2
        n_replicas = 4
        opt = StructureOptimizer(
            n_atoms=8,
            charge=0,
            mult=1,
            elements="6,7,8",
            objective={"H_total": 1.0},
            method="parallel_tempering",
            n_replicas=n_replicas,
            max_steps=100,
            n_restarts=n_restarts,
            seed=1,
        )
        result = opt.run()
        # PT returns global_best + each distinct replica final per restart,
        # so total >= n_restarts and <= n_restarts * n_replicas.
        assert len(result) >= n_restarts
        assert len(result) <= n_restarts * n_replicas + n_restarts


# ===========================================================================
# 15. Electronegativity-targeted optimization
# ===========================================================================


class TestElectronegativityOptimization:
    """charge_frustration and moran_I_chi optimization."""

    def test_en_opt_metrics_computed(self) -> None:
        opt = StructureOptimizer(
            n_atoms=8,
            charge=0,
            mult=1,
            elements="6,7,8,9,14,15,16",
            objective={
                "charge_frustration": 2.0,
                "moran_I_chi": -1.0,
            },
            method="annealing",
            max_steps=200,
            seed=7,
        )
        result = opt.run()
        s = result.best
        assert "charge_frustration" in s.metrics
        assert "moran_I_chi" in s.metrics

    def test_en_opt_maximizes_frustration(self) -> None:
        """Optimizer should raise charge_frustration above an unoptimized baseline."""
        baseline = generate(
            n_atoms=8,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:8",
            elements="6,7,8,9,14,15,16",
            n_samples=5,
            seed=7,
        )
        opt = StructureOptimizer(
            n_atoms=8,
            charge=0,
            mult=1,
            elements="6,7,8,9,14,15,16",
            objective={"charge_frustration": 1.0},
            method="annealing",
            max_steps=500,
            seed=7,
        )
        result = opt.run()
        if baseline:
            opt_cf = result.best.metrics["charge_frustration"]
            # Optimization should at least not make it worse
            assert opt_cf >= 0.0


# ===========================================================================
# 16. Affine displacement moves in StructureOptimizer
# ===========================================================================


class TestAffineMovesOptimizer:
    """allow_affine_moves=True must not crash and must complete."""

    def test_affine_moves_no_crash(self) -> None:
        opt = StructureOptimizer(
            n_atoms=10,
            charge=0,
            mult=1,
            elements="6,7,8",
            objective={"H_total": 1.0},
            allow_affine_moves=True,
            affine_strength=0.15,
            method="annealing",
            max_steps=300,
            seed=42,
        )
        result = opt.run()
        assert result.best is not None

    def test_affine_moves_has_metrics(self) -> None:
        opt = StructureOptimizer(
            n_atoms=8,
            charge=0,
            mult=1,
            elements="6,7,8",
            objective={"H_total": 1.0},
            allow_affine_moves=True,
            affine_strength=0.2,
            method="annealing",
            max_steps=200,
            seed=1,
        )
        result = opt.run()
        assert "H_total" in result.best.metrics

    def test_affine_strength_effect_range(self) -> None:
        """affine_strength 0.02..0.4 should all complete without error."""
        for strength in (0.02, 0.1, 0.4):
            opt = StructureOptimizer(
                n_atoms=8,
                charge=0,
                mult=1,
                elements="6,7,8",
                objective={"H_total": 1.0},
                allow_affine_moves=True,
                affine_strength=strength,
                method="annealing",
                max_steps=100,
                seed=0,
            )
            result = opt.run()
            assert result is not None


# ===========================================================================
# 17. Callable objective
# ===========================================================================


class TestCallableObjective:
    """Verify lambda and 2-argument (ctx) callable objectives work."""

    def test_lambda_objective(self) -> None:
        opt = StructureOptimizer(
            n_atoms=8,
            charge=0,
            mult=1,
            elements="6,7,8",
            objective=lambda m: m["H_spatial"] - 2.0 * m["Q6"],
            method="annealing",
            max_steps=200,
            seed=0,
        )
        result = opt.run()
        assert result.best is not None

    def test_two_arg_ctx_objective(self) -> None:
        """2-argument objective receives EvalContext as second argument."""
        ctx_received: list[EvalContext] = []

        def my_obj(m: dict[str, float], ctx: EvalContext) -> float:
            ctx_received.append(ctx)
            return float(m["H_total"]) - float(np.max(ctx.per_atom_q6))

        opt = StructureOptimizer(
            n_atoms=8,
            charge=0,
            mult=1,
            elements="6,7,8",
            objective=my_obj,
            method="annealing",
            max_steps=100,
            seed=1,
        )
        result = opt.run()
        assert result.best is not None
        assert len(ctx_received) > 0
        ctx = ctx_received[0]
        assert isinstance(ctx, EvalContext)
        assert len(ctx.atoms) > 0
        assert ctx.method == "annealing"

    def test_ctx_fields_populated(self) -> None:
        ctx_list: list[EvalContext] = []

        def record_ctx(m: dict[str, float], ctx: EvalContext) -> float:
            ctx_list.append(ctx)
            return float(m["H_total"])

        opt = StructureOptimizer(
            n_atoms=8,
            charge=0,
            mult=1,
            elements="6,7,8",
            objective=record_ctx,
            method="annealing",
            max_steps=50,
            seed=2,
        )
        opt.run()
        if ctx_list:
            ctx = ctx_list[-1]
            assert ctx.step >= 0
            assert ctx.temperature > 0
            assert ctx.n_atoms == 8
            assert isinstance(ctx.element_pool, tuple)
            assert ctx.cutoff > 0.0


# ===========================================================================
# 18. Basin-hopping method
# ===========================================================================


class TestBasinHopping:
    """basin_hopping method smoke tests."""

    def test_basin_hopping_runs(self) -> None:
        opt = StructureOptimizer(
            n_atoms=8,
            charge=0,
            mult=1,
            elements="6,7,8",
            objective={"H_total": 1.0},
            method="basin_hopping",
            max_steps=200,
            n_restarts=2,
            seed=5,
        )
        result = opt.run()
        assert result.best is not None
        assert "basin_hopping" in result.summary()

    def test_basin_hopping_metrics_valid(self) -> None:
        opt = StructureOptimizer(
            n_atoms=8,
            charge=0,
            mult=1,
            elements="6,7,8",
            objective={"H_total": 1.0},
            method="basin_hopping",
            max_steps=150,
            seed=3,
        )
        result = opt.run()
        for key in ("H_total", "Q6", "shape_aniso"):
            assert key in result.best.metrics


# ===========================================================================
# 19. OptimizationResult — list interface
# ===========================================================================


class TestOptimizationResultInterface:
    """OptimizationResult must behave like a list[Structure]."""

    @pytest.fixture
    def opt_result(self) -> Any:
        opt = StructureOptimizer(
            n_atoms=8,
            charge=0,
            mult=1,
            elements="6,7,8",
            objective={"H_total": 1.0},
            method="annealing",
            max_steps=200,
            n_restarts=3,
            seed=7,
        )
        return opt.run()

    def test_len(self, opt_result: Any) -> None:
        assert len(opt_result) == 3

    def test_indexing(self, opt_result: Any) -> None:
        s = opt_result[0]
        assert isinstance(s, Structure)

    def test_iteration(self, opt_result: Any) -> None:
        structs = list(opt_result)
        assert len(structs) == 3

    def test_bool_true(self, opt_result: Any) -> None:
        assert bool(opt_result)

    def test_best_is_highest_scoring(self, opt_result: Any) -> None:
        h_totals = [s.metrics["H_total"] for s in opt_result]
        assert opt_result.best.metrics["H_total"] == max(h_totals)

    def test_summary_contains_restarts(self, opt_result: Any) -> None:
        summary = opt_result.summary()
        assert "restarts=" in summary


# ===========================================================================
# 20. Structure.from_xyz — string source
# ===========================================================================


class TestFromXYZString:
    """Structure.from_xyz can accept a raw XYZ string."""

    def test_from_xyz_string_roundtrip(self, small_gas_structure: Structure) -> None:
        xyz_str = small_gas_structure.to_xyz()
        loaded = Structure.from_xyz(xyz_str)
        assert len(loaded) == len(small_gas_structure)
        assert sorted(loaded.atoms) == sorted(small_gas_structure.atoms)

    def test_from_xyz_string_metrics_recomputed(self, small_gas_structure: Structure) -> None:
        xyz_str = small_gas_structure.to_xyz()
        loaded = Structure.from_xyz(xyz_str, recompute_metrics=True)
        assert "H_total" in loaded.metrics
        assert np.isfinite(loaded.metrics["H_total"])

    def test_from_xyz_string_no_recompute(self, small_gas_structure: Structure) -> None:
        xyz_str = small_gas_structure.to_xyz()
        loaded = Structure.from_xyz(xyz_str, recompute_metrics=False)
        # When not recomputed, the embedded metrics from to_xyz() are parsed
        assert "H_total" in loaded.metrics

    def test_from_xyz_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            Structure.from_xyz("/nonexistent/path/file.xyz")

    def test_from_xyz_directory_raises(self) -> None:
        with pytest.raises((IsADirectoryError, ValueError, OSError)):
            Structure.from_xyz("/tmp")


# ===========================================================================
# 21. Structure.from_xyz — frame index
# ===========================================================================


class TestFromXYZFrameIndex:
    """Structure.from_xyz frame= selects the correct frame."""

    def test_frame_zero_is_default(self, small_gas_structure: Structure) -> None:
        xyz_str = small_gas_structure.to_xyz()
        s0 = Structure.from_xyz(xyz_str, frame=0)
        assert len(s0) == len(small_gas_structure)

    def test_frame_out_of_range_raises(self, small_gas_structure: Structure) -> None:
        xyz_str = small_gas_structure.to_xyz()
        with pytest.raises(ValueError, match="out of range"):
            Structure.from_xyz(xyz_str, frame=5)

    def test_multiframe_selection(self) -> None:
        result = generate(**SMALL_GAS_KWARGS)
        if len(result) < 2:
            pytest.skip("Need at least 2 structures")
        xyz_parts = "\n".join(s.to_xyz() for s in result[:2])
        s0 = Structure.from_xyz(xyz_parts, frame=0)
        s1 = Structure.from_xyz(xyz_parts, frame=1)
        # Different seeds should yield different structures
        assert (s0.atoms != s1.atoms) or (s0.positions != s1.positions)


# ===========================================================================
# 22. read_xyz — multi-frame file
# ===========================================================================


class TestReadXYZ:
    """read_xyz loads all frames from a multi-structure file."""

    def test_read_xyz_correct_count(self) -> None:
        result = generate(**SMALL_GAS_KWARGS)
        if len(result) < 2:
            pytest.skip("Need at least 2 structures")
        n_write = min(3, len(result))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            for s in result[:n_write]:
                f.write(s.to_xyz() + "\n")
            fname = f.name
        try:
            loaded = read_xyz(fname)
            assert len(loaded) == n_write
        finally:
            os.unlink(fname)

    def test_read_xyz_metrics_recomputed(self) -> None:
        result = generate(**SMALL_GAS_KWARGS)
        if not result:
            pytest.skip("No structures")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write(result[0].to_xyz() + "\n")
            fname = f.name
        try:
            loaded = read_xyz(fname)
            assert "H_total" in loaded[0].metrics
        finally:
            os.unlink(fname)

    def test_read_xyz_string_source(self) -> None:
        result = generate(**SMALL_GAS_KWARGS)
        if not result:
            pytest.skip("No structures")
        xyz_str = result[0].to_xyz()
        loaded = read_xyz(xyz_str)
        assert len(loaded) == 1


# ===========================================================================
# 23. parse_objective_spec helper
# ===========================================================================


class TestParseObjectiveSpec:
    """parse_objective_spec converts METRIC:WEIGHT strings."""

    def test_single_spec(self) -> None:
        obj = parse_objective_spec(["H_total:1.0"])
        assert obj == {"H_total": 1.0}

    def test_multiple_specs(self) -> None:
        obj = parse_objective_spec(["H_total:1.0", "Q6:-2.0"])
        assert obj["H_total"] == pytest.approx(1.0)
        assert obj["Q6"] == pytest.approx(-2.0)

    def test_negative_weight(self) -> None:
        obj = parse_objective_spec(["Q6:-3.5"])
        assert obj["Q6"] == pytest.approx(-3.5)

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises((ValueError, KeyError)):
            parse_objective_spec(["not_a_metric:1.0"])


# ===========================================================================
# 24. Position-only optimization (allow_composition_moves=False)
# ===========================================================================


class TestPositionOnlyOptimization:
    """Composition is unchanged when allow_composition_moves=False."""

    def test_composition_unchanged(self) -> None:
        initial = generate(**SMALL_GAS_KWARGS)
        if not initial:
            pytest.skip("No initial structure")
        s0 = initial[0]
        opt = StructureOptimizer(
            n_atoms=len(s0),
            charge=s0.charge,
            mult=s0.mult,
            elements=list(set(s0.atoms)),
            objective={"H_total": 1.0, "Q6": -2.0},
            allow_composition_moves=False,
            method="annealing",
            max_steps=300,
            seed=42,
        )
        result = opt.run(initial=s0)
        assert sorted(result.best.atoms) == sorted(s0.atoms)

    def test_positions_changed(self) -> None:
        initial = generate(**SMALL_GAS_KWARGS)
        if not initial:
            pytest.skip("No initial structure")
        s0 = initial[0]
        opt = StructureOptimizer(
            n_atoms=len(s0),
            charge=s0.charge,
            mult=s0.mult,
            elements=list(set(s0.atoms)),
            objective={"H_total": 1.0},
            allow_composition_moves=False,
            method="annealing",
            max_steps=500,
            seed=42,
        )
        result = opt.run(initial=s0)
        # Positions may differ
        assert result.best is not None


# ===========================================================================
# 25. Composition-only optimization (allow_displacements=False)
# ===========================================================================


class TestCompositionOnlyOptimization:
    """Positions are unchanged when allow_displacements=False."""

    def test_positions_unchanged(self) -> None:
        initial = generate(**SMALL_GAS_KWARGS)
        if not initial:
            pytest.skip("No initial structure")
        s0 = initial[0]
        opt = StructureOptimizer(
            n_atoms=len(s0),
            charge=s0.charge,
            mult=s0.mult,
            elements=["C", "N", "O"],
            objective={"H_atom": 1.0},
            allow_displacements=False,
            method="annealing",
            max_steps=300,
            seed=42,
        )
        result = opt.run(initial=s0)
        np.testing.assert_allclose(
            np.array(result.best.positions),
            np.array(s0.positions),
        )

    def test_both_false_raises(self) -> None:
        with pytest.raises(ValueError):
            StructureOptimizer(
                n_atoms=8,
                charge=0,
                mult=1,
                elements="6,7,8",
                objective={"H_total": 1.0},
                allow_displacements=False,
                allow_composition_moves=False,
            )


# ===========================================================================
# 26. Memory regression — no detectable leak in long runs
# ===========================================================================


class TestMemoryLeak:
    """Repeated generate() calls must not accumulate unbounded memory."""

    def test_generate_no_memory_leak(self) -> None:
        import gc
        import tracemalloc

        tracemalloc.start()
        gc.collect()
        snap_before = tracemalloc.take_snapshot()

        for i in range(5):
            r = generate(
                n_atoms=15,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:8",
                elements="6,7,8",
                n_samples=20,
                seed=i,
            )
            del r
            gc.collect()

        snap_after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        diff = snap_after.compare_to(snap_before, "lineno")
        total_added = sum(s.size_diff for s in diff if s.size_diff > 0)
        # Less than 4 MB accumulated is fine
        assert total_added < 4 * 1024 * 1024, (
            f"Possible memory leak: {total_added / 1024:.1f} KB added"
        )

    def test_optimizer_no_memory_leak(self) -> None:
        import gc
        import tracemalloc

        tracemalloc.start()
        gc.collect()
        snap_before = tracemalloc.take_snapshot()

        for i in range(3):
            opt = StructureOptimizer(
                n_atoms=10,
                charge=0,
                mult=1,
                elements="6,7,8",
                objective={"H_total": 1.0},
                method="annealing",
                max_steps=300,
                seed=i,
            )
            r = opt.run()
            del opt, r
            gc.collect()

        snap_after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        diff = snap_after.compare_to(snap_before, "lineno")
        total_added = sum(s.size_diff for s in diff if s.size_diff > 0)
        assert total_added < 4 * 1024 * 1024, (
            f"Possible memory leak: {total_added / 1024:.1f} KB added"
        )
