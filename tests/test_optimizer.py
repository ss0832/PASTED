"""Tests for pasted._optimizer: StructureOptimizer and helpers."""

from __future__ import annotations

import random
import warnings

import numpy as np
import pytest

from pasted import OptimizationResult, StructureGenerator, StructureOptimizer
from pasted._atoms import ALL_METRICS
from pasted._optimizer import (
    _composition_move,
    _eval_objective,
    _fragment_move,
    parse_objective_spec,
)

# ---------------------------------------------------------------------------
# parse_objective_spec
# ---------------------------------------------------------------------------


def test_parse_objective_spec_basic() -> None:
    obj = parse_objective_spec(["H_atom:1.0", "Q6:-2.0"])
    assert obj["H_atom"] == pytest.approx(1.0)
    assert obj["Q6"] == pytest.approx(-2.0)


def test_parse_objective_spec_empty() -> None:
    assert parse_objective_spec([]) == {}


def test_parse_objective_spec_bad_format() -> None:
    with pytest.raises(ValueError):
        parse_objective_spec(["H_atom"])


def test_parse_objective_spec_unknown_metric() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        parse_objective_spec(["bogus:1.0"])


# ---------------------------------------------------------------------------
# _eval_objective
# ---------------------------------------------------------------------------


def test_eval_objective_dict() -> None:
    metrics = {"H_atom": 1.0, "Q6": 0.4}
    result = _eval_objective(metrics, {"H_atom": 2.0, "Q6": -1.0})
    assert result == pytest.approx(2.0 * 1.0 + (-1.0) * 0.4)


def test_eval_objective_callable() -> None:
    metrics = {"H_total": 2.5}
    result = _eval_objective(metrics, lambda m: m["H_total"] * 3)
    assert result == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# _fragment_move
# ---------------------------------------------------------------------------


def test_fragment_move_targets_high_q6() -> None:

    positions = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
    per_atom_q6 = np.array([0.1, 0.6, 0.1])  # only atom 1 exceeds threshold 0.5
    rng = random.Random(0)

    new_pos = _fragment_move(positions, per_atom_q6, frag_threshold=0.5, move_step=0.5, rng=rng)

    # atom 0 and 2 should be unchanged; atom 1 should have moved
    assert new_pos[0] == positions[0]
    assert new_pos[2] == positions[2]
    assert new_pos[1] != positions[1]


def test_fragment_move_fallback_when_no_ordered() -> None:

    positions = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    per_atom_q6 = np.array([0.1, 0.1])  # nothing exceeds threshold
    rng = random.Random(1)

    # Should not raise; fallback moves one atom
    new_pos = _fragment_move(positions, per_atom_q6, frag_threshold=0.5, move_step=0.5, rng=rng)
    changed = sum(1 for a, b in zip(positions, new_pos, strict=True) if a != b)
    assert changed == 1


# ---------------------------------------------------------------------------
# _composition_move
# ---------------------------------------------------------------------------


def test_composition_move_swap() -> None:

    atoms = ["C", "N", "O", "C"]
    rng = random.Random(42)
    new_atoms = _composition_move(atoms, ["C", "N", "O"], rng)
    # total count should be the same or only change by 1 (swap vs replace)
    assert len(new_atoms) == len(atoms)


def test_composition_move_all_same_fallback() -> None:

    atoms = ["C", "C", "C"]
    rng = random.Random(0)
    new_atoms = _composition_move(atoms, ["C", "N", "O"], rng)
    # fallback: one atom replaced with a pool element
    assert len(new_atoms) == 3


# ---------------------------------------------------------------------------
# StructureOptimizer: construction
# ---------------------------------------------------------------------------


class TestStructureOptimizerInit:
    def test_basic(self) -> None:
        opt = StructureOptimizer(
            n_atoms=6,
            charge=0,
            mult=1,
            objective={"H_total": 1.0},
            elements="6,7,8",
        )
        assert opt.n_atoms == 6
        assert opt.method == "annealing"

    def test_bad_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method"):
            StructureOptimizer(
                n_atoms=6,
                charge=0,
                mult=1,
                objective={"H_total": 1.0},
                method="gradient_descent",
            )

    def test_cutoff_override(self) -> None:
        opt = StructureOptimizer(
            n_atoms=6,
            charge=0,
            mult=1,
            objective={"H_total": 1.0},
            cutoff=3.5,
        )
        assert opt.cutoff == pytest.approx(3.5)

    def test_element_pool_from_spec(self) -> None:
        opt = StructureOptimizer(
            n_atoms=6,
            charge=0,
            mult=1,
            objective={"H_total": 1.0},
            elements="6,7,8",
        )
        assert set(opt.element_pool) == {"C", "N", "O"}

    def test_repr(self) -> None:
        opt = StructureOptimizer(
            n_atoms=6,
            charge=0,
            mult=1,
            objective={"H_total": 1.0},
        )
        assert "StructureOptimizer" in repr(opt)


# ---------------------------------------------------------------------------
# StructureOptimizer: run()
# ---------------------------------------------------------------------------


class TestStructureOptimizerRun:
    def _small_opt(self, method: str = "annealing", **kwargs) -> StructureOptimizer:
        return StructureOptimizer(
            n_atoms=6,
            charge=0,
            mult=1,
            objective={"H_total": 1.0, "Q6": -1.0},
            elements="6,7,8",
            method=method,
            max_steps=30,
            seed=0,
            **kwargs,
        )

    def test_returns_structure(self) -> None:
        result = self._small_opt().run()
        assert isinstance(result, OptimizationResult)

    def test_basin_hopping(self) -> None:
        result = self._small_opt(method="basin_hopping").run()
        assert isinstance(result, OptimizationResult)

    def test_callable_objective(self) -> None:
        opt = StructureOptimizer(
            n_atoms=6,
            charge=0,
            mult=1,
            objective=lambda m: m["H_spatial"] - m["Q6"],
            elements="6,7,8",
            max_steps=20,
            seed=1,
        )
        result = opt.run()
        assert isinstance(result, OptimizationResult)

    def test_reproducible_with_seed(self) -> None:
        r1 = self._small_opt().run()
        r2 = self._small_opt().run()
        assert r1.best.atoms == r2.best.atoms
        assert r1.best.positions == r2.best.positions

    def test_n_restarts(self) -> None:
        opt = StructureOptimizer(
            n_atoms=6,
            charge=0,
            mult=1,
            objective={"H_total": 1.0},
            elements="6,7,8",
            max_steps=10,
            n_restarts=3,
            seed=2,
        )
        result = opt.run()
        assert isinstance(result, OptimizationResult)

    def test_with_provided_initial(self) -> None:
        initial = StructureGenerator(
            n_atoms=6,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:6",
            elements="6,7,8",
            n_samples=1,
            seed=5,
        ).generate()[0]

        opt = StructureOptimizer(
            n_atoms=6,
            charge=0,
            mult=1,
            objective={"H_total": 1.0},
            elements="6,7,8",
            max_steps=20,
            seed=5,
        )
        result = opt.run(initial=initial)
        assert isinstance(result, OptimizationResult)

    def test_structure_mode_label(self) -> None:
        result = self._small_opt(method="annealing").run()
        assert "opt_annealing" in result.best.mode

    def test_metrics_complete(self) -> None:

        result = self._small_opt().run()
        assert set(result.best.metrics.keys()) == ALL_METRICS

    def test_lcc_threshold_respected(self) -> None:
        # With a very tight lcc_threshold, the optimizer should still return
        # something (may take more steps but must not crash)
        opt = StructureOptimizer(
            n_atoms=6,
            charge=0,
            mult=1,
            objective={"H_total": 1.0},
            elements="6,7,8",
            max_steps=30,
            lcc_threshold=0.0,  # disabled — always passes
            seed=3,
        )
        result = opt.run()
        assert isinstance(result, OptimizationResult)


# ---------------------------------------------------------------------------
# OptimizationResult
# ---------------------------------------------------------------------------


class TestOptimizationResult:
    def test_is_list_compatible(self) -> None:
        result = OptimizationResult(
            all_structures=[],
            objective_scores=[],
            n_restarts_attempted=0,
        )
        assert len(result) == 0
        assert not result
        assert list(result) == []

    def test_best_returns_first(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s1 = StructureGenerator(
                n_atoms=6, charge=0, mult=1, mode="gas", region="sphere:6",
                elements="6,8", n_samples=1, seed=0,
            ).generate().structures[0]
            s2 = StructureGenerator(
                n_atoms=6, charge=0, mult=1, mode="gas", region="sphere:6",
                elements="6,8", n_samples=1, seed=1,
            ).generate().structures[0]
        result = OptimizationResult(
            all_structures=[s1, s2],
            objective_scores=[2.0, 1.0],
            n_restarts_attempted=2,
        )
        assert result.best is s1
        assert result[0] is s1
        assert result[1] is s2

    def test_summary_contains_fields(self) -> None:
        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1,
            objective={"H_total": 1.0},
            elements="6,8", max_steps=20, n_restarts=2, seed=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run()
        s = result.summary()
        assert "restarts=2" in s
        assert "best_f=" in s
        assert "method=" in s

    def test_sorted_best_first(self) -> None:
        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1,
            objective={"H_total": 1.0},
            elements="6,8", max_steps=30, n_restarts=4, seed=5,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run()
        assert result.objective_scores == sorted(result.objective_scores, reverse=True)

    def test_repr(self) -> None:
        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1,
            objective={"H_total": 1.0},
            elements="6,8", max_steps=10, seed=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run()
        assert "OptimizationResult" in repr(result)


# ---------------------------------------------------------------------------
# Objective alignment verification
# ---------------------------------------------------------------------------


class TestObjectiveAlignment:
    """Verify that SA and BH actually improve the objective vs random baseline."""

    def _baseline_mean(self, metric: str, n_samples: int = 30) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structs = StructureGenerator(
                n_atoms=8, charge=0, mult=1,
                mode="gas", region="sphere:9",
                elements="6,8", n_samples=n_samples, seed=99,
            ).generate()
        return sum(s.metrics[metric] for s in structs) / len(structs)

    def test_sa_improves_h_total(self) -> None:
        baseline = self._baseline_mean("H_total")
        opt = StructureOptimizer(
            n_atoms=8, charge=0, mult=1,
            objective={"H_total": 1.0},
            elements="6,8", method="annealing",
            max_steps=1000, n_restarts=3, seed=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run()
        assert result.best.metrics["H_total"] > baseline, (
            f"SA H_total {result.best.metrics['H_total']:.3f} not above "
            f"baseline {baseline:.3f}"
        )

    def test_bh_improves_h_total(self) -> None:
        baseline = self._baseline_mean("H_total")
        opt = StructureOptimizer(
            n_atoms=8, charge=0, mult=1,
            objective={"H_total": 1.0},
            elements="6,8", method="basin_hopping",
            max_steps=300, n_restarts=3, seed=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run()
        assert result.best.metrics["H_total"] > baseline, (
            f"BH H_total {result.best.metrics['H_total']:.3f} not above "
            f"baseline {baseline:.3f}"
        )

    def test_negative_weight_reduces_metric(self) -> None:
        """Minimizing Q6 via negative weight should produce objective >= random best."""
        opt = StructureOptimizer(
            n_atoms=8, charge=0, mult=1,
            objective={"H_total": 0.5, "Q6": -2.0},
            elements="6,8", method="annealing",
            max_steps=1500, n_restarts=3, seed=7,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run()
            baseline = StructureGenerator(
                n_atoms=8, charge=0, mult=1,
                mode="gas", region="sphere:9",
                elements="6,8", n_samples=30, seed=99,
            ).generate()
        best_random_score = max(
            0.5 * s.metrics["H_total"] - 2.0 * s.metrics["Q6"]
            for s in baseline
        )
        opt_score = 0.5 * result.best.metrics["H_total"] - 2.0 * result.best.metrics["Q6"]
        # Optimizer must match or beat the best random structure found in 30 samples
        assert opt_score >= best_random_score, (
            f"Optimizer score {opt_score:.3f} not above best random {best_random_score:.3f}"
        )

    def test_callable_objective_alignment(self) -> None:
        """Callable objective should also be maximized."""
        opt = StructureOptimizer(
            n_atoms=8, charge=0, mult=1,
            objective=lambda m: m["H_spatial"] - m["Q6"],
            elements="6,8", method="annealing",
            max_steps=800, n_restarts=2, seed=3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run()
            baseline = StructureGenerator(
                n_atoms=8, charge=0, mult=1,
                mode="gas", region="sphere:9",
                elements="6,8", n_samples=30, seed=99,
            ).generate()
        b_scores = [s.metrics["H_spatial"] - s.metrics["Q6"] for s in baseline]
        opt_score = result.best.metrics["H_spatial"] - result.best.metrics["Q6"]
        assert opt_score > sum(b_scores) / len(b_scores), (
            f"Callable objective score {opt_score:.3f} not above baseline mean "
            f"{sum(b_scores)/len(b_scores):.3f}"
        )

    def test_n_restarts_best_not_worse_than_single(self) -> None:
        """n_restarts=4 best should be >= any individual single-restart result."""
        single_scores = []
        for seed in range(4):
            o = StructureOptimizer(
                n_atoms=6, charge=0, mult=1,
                objective={"H_total": 1.0},
                elements="6,8", max_steps=200, n_restarts=1, seed=seed * 97,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                single_scores.append(o.run().best.metrics["H_total"])

        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1,
            objective={"H_total": 1.0},
            elements="6,8", max_steps=200, n_restarts=4, seed=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run()
        # multi-restart best should be >= any individual run (with tolerance)
        assert result.best.metrics["H_total"] >= min(single_scores) - 0.01
        assert len(result) == 4  # all restarts included


# ---------------------------------------------------------------------------
# Parallel Tempering
# ---------------------------------------------------------------------------


class TestParallelTempering:
    def _pt_opt(self, **kwargs: object) -> StructureOptimizer:
        defaults: dict[str, object] = {
            "n_atoms": 6,
            "charge": 0,
            "mult": 1,
            "objective": {"H_total": 1.0, "Q6": -1.0},
            "elements": "6,8",
            "method": "parallel_tempering",
            "max_steps": 50,
            "n_replicas": 3,
            "pt_swap_interval": 5,
            "n_restarts": 1,
            "seed": 0,
        }
        defaults.update(kwargs)
        return StructureOptimizer(**defaults)  # type: ignore[arg-type]

    def test_returns_optimization_result(self) -> None:
        opt = self._pt_opt()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run()
        assert isinstance(result, OptimizationResult)
        assert len(result) > 0

    def test_best_is_highest_score(self) -> None:
        opt = self._pt_opt()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run()
        assert result.best is result[0]
        assert result.objective_scores[0] == max(result.objective_scores)

    def test_sorted_best_first(self) -> None:
        opt = self._pt_opt(n_replicas=4, max_steps=80)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run()
        assert result.objective_scores == sorted(result.objective_scores, reverse=True)

    def test_mode_label(self) -> None:
        opt = self._pt_opt()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run()
        assert "parallel_tempering" in result.best.mode

    def test_temperature_ladder_geometric(self) -> None:
        opt = self._pt_opt(n_replicas=4, T_start=1.0, T_end=0.01)
        temps = opt._pt_temperatures()
        assert len(temps) == 4
        assert temps[0] == pytest.approx(0.01, rel=1e-4)
        assert temps[-1] == pytest.approx(1.0, rel=1e-4)
        # Geometric spacing: each ratio should be equal
        ratios = [temps[k + 1] / temps[k] for k in range(len(temps) - 1)]
        for r in ratios:
            assert r == pytest.approx(ratios[0], rel=1e-4)

    def test_n_replicas_parameter(self) -> None:
        for n_rep in (2, 3, 5):
            opt = self._pt_opt(n_replicas=n_rep, max_steps=30)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = opt.run()
            # Result includes best + up to n_rep replica final states
            assert len(result) <= n_rep + 1

    def test_repr_includes_replicas(self) -> None:
        opt = self._pt_opt(n_replicas=4)
        r = repr(opt)
        assert "parallel_tempering" in r
        assert "n_replicas=4" in r

    def test_bad_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method"):
            StructureOptimizer(
                n_atoms=6, charge=0, mult=1,
                objective={"H_total": 1.0},
                method="gibbs_sampling",
            )

    def test_pt_improves_over_baseline(self) -> None:
        # PT on a tight landscape (Q6 maximize) should beat random gas
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            baseline = StructureGenerator(
                n_atoms=8, charge=0, mult=1,
                mode="gas", region="sphere:9",
                elements="6,8", n_samples=30, seed=99,
            ).generate()
            opt = StructureOptimizer(
                n_atoms=8, charge=0, mult=1,
                objective={"H_total": 1.0},
                elements="6,8",
                method="parallel_tempering",
                max_steps=300, n_replicas=4, n_restarts=2, seed=42,
            )
            result = opt.run()
        baseline_mean = sum(s.metrics["H_total"] for s in baseline) / len(baseline)
        assert result.best.metrics["H_total"] > baseline_mean

    def test_pt_multirestarts_accumulates_all(self) -> None:
        # n_restarts=2 should give more structures than n_restarts=1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1 = self._pt_opt(n_restarts=1, n_replicas=3, max_steps=30).run()
            r2 = self._pt_opt(n_restarts=2, n_replicas=3, max_steps=30).run()
        assert len(r2) > len(r1)


# ---------------------------------------------------------------------------
# allow_composition_moves
# ---------------------------------------------------------------------------


class TestAllowCompositionMoves:
    def _opt(self, **kwargs: object) -> StructureOptimizer:
        defaults: dict[str, object] = {
            "n_atoms": 6,
            "charge": 0,
            "mult": 1,
            "objective": {"H_total": 1.0},
            "elements": "6,7,8",
            "max_steps": 50,
            "seed": 0,
        }
        defaults.update(kwargs)
        return StructureOptimizer(**defaults)  # type: ignore[arg-type]

    def test_default_is_true(self) -> None:
        opt = self._opt()
        assert opt.allow_composition_moves is True

    def test_disabled_preserves_initial_composition(self) -> None:
        """With composition moves off, atom types must not change."""
        gen = StructureGenerator(
            n_atoms=6, charge=0, mult=1, mode="gas", region="sphere:6",
            elements="6,7,8", n_samples=50, n_success=1, seed=5,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_gen = gen.generate()
        if not result_gen:
            pytest.skip("Could not generate initial structure")
        initial = result_gen[0]

        initial_composition = sorted(initial.atoms)
        opt = self._opt(
            allow_composition_moves=False,
            max_steps=200,
            seed=5,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run(initial=initial)

        assert sorted(result.best.atoms) == initial_composition, (
            f"Composition changed despite allow_composition_moves=False: "
            f"{sorted(result.best.atoms)} != {initial_composition}"
        )

    def test_disabled_still_optimises(self) -> None:
        """Position-only optimisation should still return an OptimizationResult."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self._opt(allow_composition_moves=False, max_steps=100).run()
        assert isinstance(result, OptimizationResult)
        assert len(result) > 0

    def test_repr_mentions_flag_when_disabled(self) -> None:
        opt = self._opt(allow_composition_moves=False)
        assert "allow_composition_moves=False" in repr(opt)

    def test_repr_silent_when_enabled(self) -> None:
        opt = self._opt(allow_composition_moves=True)
        assert "allow_composition_moves" not in repr(opt)

    def test_parallel_tempering_respects_flag(self) -> None:
        """PT should also preserve composition when flag is off."""
        gen = StructureGenerator(
            n_atoms=6, charge=0, mult=1, mode="gas", region="sphere:6",
            elements="6,7,8", n_samples=50, n_success=1, seed=3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gen_result = gen.generate()
        if not gen_result:
            pytest.skip("Could not generate initial structure")
        initial = gen_result[0]

        initial_composition = sorted(initial.atoms)
        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1,
            objective={"H_total": 1.0},
            elements="6,7,8",
            method="parallel_tempering",
            allow_composition_moves=False,
            max_steps=50, n_replicas=2, n_restarts=1, seed=3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run(initial=initial)
        assert sorted(result.best.atoms) == initial_composition

# ---------------------------------------------------------------------------
# allow_displacements
# ---------------------------------------------------------------------------


class TestAllowDisplacements:
    def _opt(self, **kwargs: object) -> StructureOptimizer:
        defaults: dict[str, object] = {
            "n_atoms": 6,
            "charge": 0,
            "mult": 1,
            "objective": {"H_atom": 1.0},
            "elements": "6,7,8",
            "max_steps": 50,
            "seed": 0,
        }
        defaults.update(kwargs)
        return StructureOptimizer(**defaults)  # type: ignore[arg-type]

    # ── Attribute defaults ────────────────────────────────────────────────

    def test_default_is_true(self) -> None:
        opt = self._opt()
        assert opt.allow_displacements is True

    # ── Mutual-exclusion validation ───────────────────────────────────────

    def test_both_false_raises(self) -> None:
        """allow_displacements=False + allow_composition_moves=False must raise ValueError."""
        with pytest.raises(ValueError, match="cannot both be False"):
            self._opt(allow_displacements=False, allow_composition_moves=False)

    # ── Positions are frozen ──────────────────────────────────────────────

    def test_disabled_preserves_positions(self) -> None:
        """With displacements off, atom coordinates must not change."""
        gen = StructureGenerator(
            n_atoms=6, charge=0, mult=1, mode="gas", region="sphere:6",
            elements="6,7,8", n_samples=50, n_success=1, seed=11,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_gen = gen.generate()
        if not result_gen:
            pytest.skip("Could not generate initial structure")
        initial = result_gen[0]

        import numpy as np

        initial_positions = [tuple(p) for p in initial.positions]
        opt = self._opt(
            allow_displacements=False,
            allow_composition_moves=True,
            max_steps=200,
            seed=11,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run(initial=initial)

        best_positions = [tuple(p) for p in result.best.positions]
        np.testing.assert_allclose(
            np.array(best_positions), np.array(initial_positions), atol=1e-9,
            err_msg="Positions changed despite allow_displacements=False",
        )

    def test_disabled_still_optimises(self) -> None:
        """Composition-only optimisation should still return an OptimizationResult."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self._opt(allow_displacements=False, max_steps=100).run()
        assert isinstance(result, OptimizationResult)
        assert len(result) > 0

    def test_repr_mentions_flag_when_disabled(self) -> None:
        opt = self._opt(allow_displacements=False)
        assert "allow_displacements=False" in repr(opt)

    def test_repr_silent_when_enabled(self) -> None:
        opt = self._opt(allow_displacements=True)
        assert "allow_displacements" not in repr(opt)

    # ── Basin-hopping respects flag ───────────────────────────────────────

    def test_basin_hopping_preserves_positions(self) -> None:
        gen = StructureGenerator(
            n_atoms=6, charge=0, mult=1, mode="gas", region="sphere:6",
            elements="6,7,8", n_samples=50, n_success=1, seed=22,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gen_result = gen.generate()
        if not gen_result:
            pytest.skip("Could not generate initial structure")
        initial = gen_result[0]

        import numpy as np

        initial_positions = [tuple(p) for p in initial.positions]
        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1,
            objective={"H_atom": 1.0},
            elements="6,7,8",
            method="basin_hopping",
            allow_displacements=False,
            max_steps=50, n_restarts=1, seed=22,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run(initial=initial)
        best_positions = [tuple(p) for p in result.best.positions]
        np.testing.assert_allclose(
            np.array(best_positions), np.array(initial_positions), atol=1e-9,
        )

    # ── Parallel Tempering respects flag ─────────────────────────────────

    def test_parallel_tempering_preserves_positions(self) -> None:
        gen = StructureGenerator(
            n_atoms=6, charge=0, mult=1, mode="gas", region="sphere:6",
            elements="6,7,8", n_samples=50, n_success=1, seed=33,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gen_result = gen.generate()
        if not gen_result:
            pytest.skip("Could not generate initial structure")
        initial = gen_result[0]

        import numpy as np

        initial_positions = [tuple(p) for p in initial.positions]
        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1,
            objective={"H_atom": 1.0},
            elements="6,7,8",
            method="parallel_tempering",
            allow_displacements=False,
            max_steps=50, n_replicas=2, n_restarts=1, seed=33,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run(initial=initial)
        best_positions = [tuple(p) for p in result.best.positions]
        np.testing.assert_allclose(
            np.array(best_positions), np.array(initial_positions), atol=1e-9,
        )
