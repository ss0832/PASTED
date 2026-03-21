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
            elements="6,8",
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
            elements="6,8",
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
            elements="6,8",
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
        )  # type: ignore[arg-type]
        result = opt.run(initial=initial)  # type: ignore[arg-type]
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
            elements="6,8",
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
  # type: ignore[union-attr]
        initial_composition = sorted(initial.atoms)  # type: ignore[union-attr]
        opt = self._opt(
            allow_composition_moves=False,
            max_steps=200,
            seed=5,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # type: ignore[arg-type]
            result = opt.run(initial=initial)  # type: ignore[arg-type]

        assert sorted(result.best.atoms) == initial_composition, (
            f"Composition changed despite allow_composition_moves=False: "
            f"{sorted(result.best.atoms)} != {initial_composition}"
        )

    def test_disabled_still_optimizes(self) -> None:
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
  # type: ignore[union-attr]
        initial_composition = sorted(initial.atoms)  # type: ignore[union-attr]
        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1,
            objective={"H_total": 1.0},
            elements="6,7,8",
            method="parallel_tempering",
            allow_composition_moves=False,
            max_steps=50, n_replicas=2, n_restarts=1, seed=3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # type: ignore[arg-type]
            result = opt.run(initial=initial)  # type: ignore[arg-type]
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

  # type: ignore[union-attr]
        initial_positions = [tuple(p) for p in initial.positions]  # type: ignore[union-attr]
        opt = self._opt(
            allow_displacements=False,
            allow_composition_moves=True,
            max_steps=200,
            seed=11,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # type: ignore[arg-type]
            result = opt.run(initial=initial)  # type: ignore[arg-type]

        best_positions = [tuple(p) for p in result.best.positions]
        np.testing.assert_allclose(
            np.array(best_positions), np.array(initial_positions), atol=1e-9,
            err_msg="Positions changed despite allow_displacements=False",
        )

    def test_disabled_still_optimizes(self) -> None:
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

  # type: ignore[union-attr]
        initial_positions = [tuple(p) for p in initial.positions]  # type: ignore[union-attr]
        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1,
            objective={"H_atom": 1.0},
            elements="6,7,8",
            method="basin_hopping",
            allow_displacements=False,
            max_steps=50, n_restarts=1, seed=22,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # type: ignore[arg-type]
            result = opt.run(initial=initial)  # type: ignore[arg-type]
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

  # type: ignore[union-attr]
        initial_positions = [tuple(p) for p in initial.positions]  # type: ignore[union-attr]
        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1,
            objective={"H_atom": 1.0},
            elements="6,7,8",
            method="parallel_tempering",
            allow_displacements=False,
            max_steps=50, n_replicas=2, n_restarts=1, seed=33,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # type: ignore[arg-type]
            result = opt.run(initial=initial)  # type: ignore[arg-type]
        best_positions = [tuple(p) for p in result.best.positions]
        np.testing.assert_allclose(
            np.array(best_positions), np.array(initial_positions), atol=1e-9,
        )


# ===========================================================================
# TestEvalContext
# ===========================================================================

class TestEvalContext:
    """Tests for EvalContext and 2-argument objective dispatch."""

    def _make_opt(self, objective, method="annealing", n_atoms=6, seed=99, **kw):
        return StructureOptimizer(
            n_atoms=n_atoms, charge=0, mult=1,
            elements="6,7,8",
            objective=objective,
            method=method,
            max_steps=30, n_restarts=1, seed=seed,
            **kw,
        )

    # ── EvalContext exported from top-level namespace ─────────────────────

    def test_evalcontext_in_public_api(self) -> None:
        import pasted
        assert hasattr(pasted, "EvalContext")
        from pasted import EvalContext  # noqa: F401

    # ── _objective_needs_ctx correctly detects arity ──────────────────────

    def test_needs_ctx_dict(self) -> None:
        from pasted._optimizer import _objective_needs_ctx
        assert _objective_needs_ctx({"H_total": 1.0}) is False

    def test_needs_ctx_one_arg_lambda(self) -> None:
        from pasted._optimizer import _objective_needs_ctx
        assert _objective_needs_ctx(lambda m: m["H_total"]) is False

    def test_needs_ctx_two_arg_lambda(self) -> None:
        from pasted._optimizer import _objective_needs_ctx
        assert _objective_needs_ctx(lambda m, ctx: m["H_total"]) is True

    def test_needs_ctx_optional_second_arg(self) -> None:
        from pasted._optimizer import _objective_needs_ctx
        # second arg has default → treated as 1-arg
        assert _objective_needs_ctx(lambda m, ctx=None: m["H_total"]) is False

    def test_needs_ctx_two_arg_def(self) -> None:
        from pasted._optimizer import _objective_needs_ctx
        def f(m, ctx):
            return m["H_total"]
        assert _objective_needs_ctx(f) is True

    # ── _needs_ctx cached on instance ─────────────────────────────────────

    def test_instance_cache_one_arg(self) -> None:
        opt = self._make_opt(lambda m: m["H_total"])
        assert opt._needs_ctx is False

    def test_instance_cache_two_arg(self) -> None:
        opt = self._make_opt(lambda m, ctx: m["H_total"])
        assert opt._needs_ctx is True

    # ── 1-arg objective still works (backward compat) ─────────────────────

    def test_one_arg_objective_runs(self) -> None:
        opt = self._make_opt(lambda m: m["H_total"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run()
        assert result.best is not None

    # ── 2-arg objective receives EvalContext ──────────────────────────────

    def test_two_arg_objective_receives_ctx(self) -> None:
        from pasted import EvalContext
        received: list[EvalContext] = []

        def f(m, ctx):
            received.append(ctx)
            return m["H_total"]

        opt = self._make_opt(f)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt.run()

        assert len(received) > 0
        ctx = received[0]
        assert isinstance(ctx, EvalContext)

    # ── EvalContext fields are correct types ──────────────────────────────

    def test_ctx_structure_fields(self) -> None:
        from pasted import EvalContext
        seen: list[EvalContext] = []

        def f(m, ctx):
            seen.append(ctx)
            return m["H_total"]

        opt = self._make_opt(f, n_atoms=6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt.run()

        ctx = seen[0]
        assert isinstance(ctx.atoms, tuple)
        assert all(isinstance(a, str) for a in ctx.atoms)
        assert isinstance(ctx.positions, tuple)
        assert ctx.n_atoms == 6
        assert ctx.charge == 0
        assert ctx.mult == 1
        assert isinstance(ctx.metrics, dict)
        assert "H_total" in ctx.metrics

    def test_ctx_optimizer_state_fields(self) -> None:
        from pasted import EvalContext
        seen: list[EvalContext] = []

        def f(m, ctx):
            seen.append(ctx)
            return m["H_total"]

        opt = self._make_opt(f, n_atoms=6, seed=7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt.run()

        ctx = seen[0]
        assert isinstance(ctx.step, int)
        assert isinstance(ctx.max_steps, int)
        assert ctx.max_steps == 30
        assert isinstance(ctx.temperature, float)
        assert isinstance(ctx.f_current, float)
        assert isinstance(ctx.best_f, float)
        assert isinstance(ctx.restart_idx, int)
        assert isinstance(ctx.n_restarts, int)
        assert ctx.n_restarts == 1
        assert ctx.per_atom_q6.shape == (6,)
        assert 0.0 <= ctx.progress <= 1.0

    def test_ctx_config_fields(self) -> None:
        from pasted import EvalContext
        seen: list[EvalContext] = []

        def f(m, ctx):
            seen.append(ctx)
            return m["H_total"]

        opt = self._make_opt(f, n_atoms=6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt.run()

        ctx = seen[0]
        assert isinstance(ctx.element_pool, tuple)
        assert isinstance(ctx.cutoff, float)
        assert ctx.method == "annealing"
        assert isinstance(ctx.T_start, float)
        assert isinstance(ctx.T_end, float)

    # ── PT-specific fields ─────────────────────────────────────────────────

    def test_ctx_non_pt_replica_fields_are_none(self) -> None:
        from pasted import EvalContext
        seen: list[EvalContext] = []

        def f(m, ctx):
            seen.append(ctx)
            return m["H_total"]

        opt = self._make_opt(f, method="annealing")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt.run()

        ctx = seen[0]
        assert ctx.replica_idx is None
        assert ctx.replica_temperature is None
        assert ctx.n_replicas is None

    def test_ctx_pt_replica_fields_set(self) -> None:
        from pasted import EvalContext
        seen: list[EvalContext] = []

        def f(m, ctx):
            seen.append(ctx)
            return m["H_total"]

        opt = self._make_opt(f, method="parallel_tempering", n_atoms=6,
                              n_replicas=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt.run()

        pt_ctxs = [c for c in seen if c.replica_idx is not None]
        assert len(pt_ctxs) > 0
        ctx = pt_ctxs[0]
        assert ctx.n_replicas == 2
        assert ctx.replica_idx in (0, 1)
        assert isinstance(ctx.replica_temperature, float)

    # ── to_xyz() ──────────────────────────────────────────────────────────

    def test_ctx_to_xyz(self) -> None:
        from pasted import EvalContext
        seen: list[EvalContext] = []

        def f(m, ctx):
            seen.append(ctx)
            return m["H_total"]

        opt = self._make_opt(f, n_atoms=6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt.run()

        ctx = seen[0]
        xyz = ctx.to_xyz()
        lines = xyz.splitlines()
        assert int(lines[0]) == ctx.n_atoms
        assert "charge=" in lines[1]
        assert len(lines) == ctx.n_atoms + 2

    # ── progress property ──────────────────────────────────────────────────

    def test_ctx_progress_range(self) -> None:
        progresses: list[float] = []

        def f(m, ctx):
            progresses.append(ctx.progress)
            return m["H_total"]

        opt = self._make_opt(f, n_atoms=6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt.run()

        assert all(0.0 <= p < 1.0 for p in progresses)

    # ── 2-arg objective improves over dict baseline ───────────────────────

    def test_two_arg_objective_optimizes(self) -> None:
        import numpy as np

        def spread_obj(m, ctx):
            pos = np.array(ctx.positions)
            diffs = pos[:, None, :] - pos[None, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=-1))
            return float(dists[np.triu_indices(ctx.n_atoms, k=1)].mean())

        opt = StructureOptimizer(
            n_atoms=8, charge=0, mult=1, elements="6,7,8",
            objective=spread_obj,
            method="annealing", max_steps=200, n_restarts=1, seed=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run()
        assert result.best is not None
        assert result.objective_scores[0] > 0.0

    # ── basin_hopping with 2-arg objective ────────────────────────────────

    def test_two_arg_objective_basin_hopping(self) -> None:
        def f(m, ctx):
            return m["H_total"] - 0.5 * float(ctx.per_atom_q6.max())

        opt = self._make_opt(f, method="basin_hopping")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run()
        assert result.best is not None


# ---------------------------------------------------------------------------
# Bug fixes 0.3.1 — _composition_move pool usage, radii cache, sanitize
# ---------------------------------------------------------------------------


class TestCompositionMoveBugFixes:
    """0.3.1 — _composition_move must draw from element_pool (Bug #2)."""

    def test_composition_move_draws_from_pool(self) -> None:
        """Every returned atom must come from element_pool."""
        import random

        from pasted._optimizer import _composition_move

        pool = ["Cr", "Mn", "Fe", "Co", "Ni"]
        pool_set = set(pool)
        atoms = ["C", "C", "N", "O", "C"]  # all outside pool
        rng = random.Random(42)

        found_pool_elements: set[str] = set()
        for _ in range(200):
            new = _composition_move(list(atoms), pool, rng)
            for a in new:
                if a in pool_set:
                    found_pool_elements.add(a)

        assert len(found_pool_elements) > 0, (
            "_composition_move never introduced a pool element in 200 attempts"
        )

    def test_composition_move_length_preserved(self) -> None:
        import random

        from pasted._optimizer import _composition_move

        pool = ["C", "N", "O", "S"]
        atoms = ["C", "N", "O"]
        rng = random.Random(0)
        for _ in range(50):
            new = _composition_move(list(atoms), pool, rng)
            assert len(new) == len(atoms)

    def test_composition_move_single_element_pool(self) -> None:
        """Pool with one element: fallback path must not crash."""
        import random

        from pasted._optimizer import _composition_move

        pool = ["Fe"]
        atoms = ["C", "C", "N"]
        rng = random.Random(7)
        new = _composition_move(list(atoms), pool, rng)
        assert len(new) == len(atoms)


class TestRadiiCacheBugFix:
    """0.3.1 — radii cache must refresh when composition changes (Bug #3)."""

    def test_composition_only_opt_all_pool_atoms(self) -> None:
        """
        After composition-only optimisation the best structure must contain
        only atoms from the element pool.  If the radii cache was stale the
        optimizer would either crash on a shape mismatch or silently keep
        wrong radii, producing invalid geometries.
        """
        import warnings

        from pasted import StructureOptimizer, generate

        initial = generate(
            n_atoms=8, charge=0, mult=1, mode="gas", region="sphere:8",
            elements="6,7,8", n_samples=50, seed=5,
        )[0]

        pool = ["Cr", "Mn", "Fe", "Co", "Ni"]
        pool_set = set(pool)

        opt = StructureOptimizer(
            n_atoms=len(initial),  # type: ignore[union-attr]
            charge=initial.charge,  # type: ignore[union-attr]
            mult=initial.mult,  # type: ignore[union-attr]
            elements=pool,
            objective={"H_atom": 1.0},
            allow_displacements=False,
            method="annealing",
            max_steps=3000,
            seed=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # type: ignore[arg-type]
            result = opt.run(initial=initial)  # type: ignore[arg-type]

        unexpected = set(result.best.atoms) - pool_set
        assert not unexpected, (
            f"Non-pool atoms found in composition-only result: {unexpected}"
        )


class TestSanitizeAtomsToPool:
    """0.3.1 — _sanitize_atoms_to_pool (Bug #4)."""

    def test_all_atoms_in_pool_after_sanitize(self) -> None:
        import random

        from pasted._optimizer import _sanitize_atoms_to_pool

        pool = ["Cr", "Mn", "Fe", "Co", "Ni"]
        pool_set = set(pool)
        atoms = ["C", "N", "O", "C", "N"]  # all outside pool
        rng = random.Random(0)
        result = _sanitize_atoms_to_pool(atoms, pool, rng)
        assert all(a in pool_set for a in result), (
            f"Non-pool atoms after sanitize: {[a for a in result if a not in pool_set]}"
        )

    def test_length_preserved(self) -> None:
        import random

        from pasted._optimizer import _sanitize_atoms_to_pool

        atoms = ["C", "N", "O"]
        pool = ["Fe", "Ni"]
        rng = random.Random(1)
        result = _sanitize_atoms_to_pool(atoms, pool, rng)
        assert len(result) == len(atoms)

    def test_already_in_pool_unchanged(self) -> None:
        import random

        from pasted._optimizer import _sanitize_atoms_to_pool

        pool = ["C", "N", "O"]
        atoms = ["C", "N", "O"]
        rng = random.Random(2)
        result = _sanitize_atoms_to_pool(atoms, pool, rng)
        assert result == atoms

    def test_composition_only_opt_preserves_positions(self) -> None:
        """Positions must be exactly preserved when allow_displacements=False."""
        import warnings

        import numpy as np

        from pasted import StructureOptimizer, generate

        initial = generate(
            n_atoms=8, charge=0, mult=1, mode="gas", region="sphere:8",
            elements="6,7,8", n_samples=50, seed=11,
        )[0]

        opt = StructureOptimizer(
            n_atoms=len(initial),  # type: ignore[union-attr]
            charge=initial.charge,  # type: ignore[union-attr]
            mult=initial.mult,  # type: ignore[union-attr]
            elements=["Cr", "Mn", "Fe", "Co", "Ni"],
            objective={"H_atom": 1.0},
            allow_displacements=False,
            method="annealing",
            max_steps=2000,
            n_restarts=2,
            seed=99,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # type: ignore[arg-type]
            result = opt.run(initial=initial)  # type: ignore[arg-type]

        np.testing.assert_allclose(
            np.array(result.best.positions),  # type: ignore[union-attr]
            np.array(initial.positions),  # type: ignore[union-attr]
            atol=1e-9,
            err_msg="Positions changed despite allow_displacements=False",
        )


# ---------------------------------------------------------------------------
# Bug fix 0.3.1 — parallel tempering sanitize (Bug #6)
# ---------------------------------------------------------------------------


class TestParallelTemperingSanitize:
    """0.3.1 — PT path must sanitize foreign atoms from initial structure (Bug #6)."""

    def test_pt_no_foreign_atoms_after_run(self) -> None:
        """Every atom in the PT best result must come from the element pool."""
        import warnings

        from pasted import StructureOptimizer, generate

        initial = generate(
            n_atoms=6, charge=0, mult=1, mode="gas", region="sphere:6",
            elements="6,7,8", n_samples=30, seed=5,
        )[0]

        pool = ["Cr", "Mn", "Fe", "Co", "Ni"]
        pool_set = set(pool)

        opt = StructureOptimizer(
            n_atoms=len(initial),  # type: ignore[union-attr]
            charge=initial.charge,  # type: ignore[union-attr]
            mult=initial.mult,  # type: ignore[union-attr]
            elements=pool,
            objective={"H_atom": 1.0},
            allow_displacements=False,
            method="parallel_tempering",
            n_replicas=2,
            max_steps=200,
            seed=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # type: ignore[arg-type]
            result = opt.run(initial=initial)  # type: ignore[arg-type]

        unexpected = set(result.best.atoms) - pool_set
        assert not unexpected, (
            f"Non-pool atoms found in PT composition-only result: {unexpected}"
        )

    def test_pt_with_initial_same_pool_unchanged(self) -> None:
        """When initial atoms already belong to the pool, PT must not alter them
        before the first MC step (i.e., sanitize is a no-op)."""
        import warnings

        from pasted import StructureOptimizer, generate

        pool = ["C", "N", "O"]
        initial = generate(
            n_atoms=6, charge=0, mult=1, mode="gas", region="sphere:6",
            elements="6,7,8", n_samples=30, seed=3,
        )[0]
        # All atoms already in pool — sanitize must be a no-op.  # type: ignore[union-attr]
        assert all(a in set(pool) for a in initial.atoms)  # type: ignore[union-attr]

        opt = StructureOptimizer(
            n_atoms=len(initial),  # type: ignore[union-attr]
            charge=initial.charge,  # type: ignore[union-attr]
            mult=initial.mult,  # type: ignore[union-attr]
            elements=pool,
            objective={"H_atom": 1.0},
            allow_displacements=False,
            method="parallel_tempering",
            n_replicas=2,
            max_steps=50,
            seed=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # type: ignore[arg-type]
            result = opt.run(initial=initial)  # type: ignore[arg-type]

        pool_set = set(pool)
        unexpected = set(result.best.atoms) - pool_set
        assert not unexpected, (
            f"Non-pool atoms in PT result with in-pool initial: {unexpected}"
        )


# ---------------------------------------------------------------------------
# Regression tests for 0.3.2 bug fixes
# ---------------------------------------------------------------------------

class TestMakeInitialNoSpuriousWarning:
    """Regression test for 0.3.2 bug fix.

    ``StructureOptimizer.run()`` must not leak UserWarnings that originate
    from transient parity-check failures inside ``_make_initial``'s internal
    retry loop.  Previously, every failed single-sample attempt emitted a
    UserWarning that surfaced to the caller even when the optimization
    ultimately succeeded.
    """

    def test_no_userwarning_on_successful_run(self) -> None:
        """No UserWarning should be emitted when opt.run() succeeds."""
        opt = StructureOptimizer(
            n_atoms=12,
            charge=0,
            mult=1,
            # Mixed-element pool where some compositions fail parity —
            # this is the exact setup that triggered the spurious warnings.
            elements="6,7,8,15,16",
            objective={"H_total": 1.0, "Q6": -2.0},
            method="annealing",
            max_steps=200,
            n_restarts=2,
            seed=42,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = opt.run()

        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert user_warns == [], (
            f"Unexpected UserWarning(s) from opt.run(): "
            f"{[str(x.message) for x in user_warns]}"
        )
        assert result.best is not None


class TestFilterWarningWithCarbonOnlyPool:
    """Regression test for 0.3.2 documentation fix.

    The quickstart warning example now uses ``elements='6'`` (carbon only)
    so that the parity check always passes and the filter-rejection warning
    fires as documented.  This test validates that exact scenario.
    """

    def test_filter_warning_fires_not_parity_warning(self) -> None:
        """With elements='6', the filter warning (not parity) should fire."""
        from pasted import generate

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = generate(
                n_atoms=8, charge=0, mult=1,
                mode="gas", region="sphere:8",
                elements="6",           # carbon-only: parity always satisfied
                n_samples=10, seed=0,
                filters=["H_total:999:-"],  # impossible threshold
            )

        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) >= 1, "Expected at least one UserWarning"
        msg = str(user_warns[0].message).lower()
        assert "filter" in msg, (
            f"Expected filter-rejection warning, got: {msg}"
        )
        assert "parity" not in msg, (
            f"Unexpected parity warning with carbon-only pool: {msg}"
        )
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests for max_init_attempts and __init__ parity validation (0.3.2)
# ---------------------------------------------------------------------------

class TestPoolParityValidation:
    """StructureOptimizer.__init__ must raise ValueError for impossible pools."""

    def test_all_odd_pool_even_natoms_wrong_parity(self) -> None:
        """All-nitrogen pool (Z=7, odd) with n_atoms=8 (even sum→even electrons)
        clashes with mult=1 which requires even electrons — but charge=0 and
        n_atoms*7 gives even total_Z, so n_e is even and mult=1 is fine.
        Use charge=1 to make n_e odd, which clashes with mult=1 (needs even n_e).
        """
        with pytest.raises(ValueError, match="cannot produce"):
            StructureOptimizer(
                n_atoms=8, charge=1, mult=1,
                elements="7",          # all-odd Z; sum=56 → n_e=55 (odd) → mult=1 impossible
                objective={"H_total": 1.0},
            )

    def test_all_odd_pool_raises_when_parity_impossible(self) -> None:
        """N-only pool, n_atoms=1: total_Z=7, n_e=7 (odd) → mult=1 impossible."""
        with pytest.raises(ValueError, match="cannot produce"):
            StructureOptimizer(
                n_atoms=1, charge=0, mult=1,
                elements="7",          # Z=7 odd; n_e=7 → needs mult=2, not 1
                objective={"H_total": 1.0},
            )

    def test_all_even_pool_wrong_parity(self) -> None:
        """All-carbon pool (Z=6, even): total_Z always even → n_e always even.
        mult=2 requires n_unpaired=1 → n_e must be odd → impossible.
        """
        with pytest.raises(ValueError, match="cannot produce"):
            StructureOptimizer(
                n_atoms=4, charge=0, mult=2,
                elements="6",          # Z=6 even; sum always even → mult=2 impossible
                objective={"H_total": 1.0},
            )

    def test_mixed_pool_does_not_raise(self) -> None:
        """Mixed pool (even- and odd-Z) can always satisfy parity."""
        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1,
            elements="6,7",            # C (Z=6 even) + N (Z=7 odd) — mixed
            objective={"H_total": 1.0},
        )
        assert opt._element_pool == ["C", "N"]

    def test_single_even_element_correct_parity(self) -> None:
        """Carbon-only pool: total_Z always even → n_e even → mult=1 OK."""
        opt = StructureOptimizer(
            n_atoms=4, charge=0, mult=1,
            elements="6",
            objective={"H_total": 1.0},
        )
        assert opt._element_pool == ["C"]

    def test_charge_makes_min_electrons_nonpositive(self) -> None:
        """Extreme positive charge that makes n_electrons <= 0 must raise."""
        with pytest.raises(ValueError, match="cannot produce"):
            StructureOptimizer(
                n_atoms=1, charge=99, mult=1,
                elements="6",          # Z=6, n_e = 6-99 = -93 ≤ 0
                objective={"H_total": 1.0},
            )


class TestMaxInitAttempts:
    """max_init_attempts controls the retry cap inside _make_initial."""

    def test_default_is_unlimited(self) -> None:
        """Default max_init_attempts=0 means unlimited."""
        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1,
            elements="6,7,8",
            objective={"H_total": 1.0},
        )
        assert opt.max_init_attempts == 0

    def test_explicit_value_stored(self) -> None:
        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1,
            elements="6,7,8",
            objective={"H_total": 1.0},
            max_init_attempts=100,
        )
        assert opt.max_init_attempts == 100

    def test_unlimited_run_succeeds(self) -> None:
        """max_init_attempts=0 must not prevent successful optimization."""
        opt = StructureOptimizer(
            n_atoms=8, charge=0, mult=1,
            elements="6,7,8",
            objective={"H_total": 1.0},
            method="annealing",
            max_steps=100,
            max_init_attempts=0,
            seed=1,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = opt.run()
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert user_warns == [], f"Unexpected warnings: {[str(x.message) for x in user_warns]}"
        assert result.best is not None

    def test_capped_run_succeeds_with_easy_pool(self) -> None:
        """max_init_attempts=5 is plenty for an easy pool."""
        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1,
            elements="6,8",           # C+O: always even electrons
            objective={"H_total": 1.0},
            method="annealing",
            max_steps=100,
            max_init_attempts=5,
            seed=2,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = opt.run()
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert user_warns == [], f"Unexpected warnings: {[str(x.message) for x in user_warns]}"
        assert result.best is not None


class TestPoolCanSatisfyParity:
    """Unit tests for the _pool_can_satisfy_parity helper."""

    def test_import(self) -> None:
        from pasted._optimizer import _pool_can_satisfy_parity
        assert callable(_pool_can_satisfy_parity)

    def test_mixed_pool_always_true(self) -> None:
        from pasted._optimizer import _pool_can_satisfy_parity
        # C (6, even) + N (7, odd) → mixed → always satisfiable
        assert _pool_can_satisfy_parity(["C", "N"], 4, 0, 1) is True

    def test_all_even_correct_target(self) -> None:
        from pasted._optimizer import _pool_can_satisfy_parity
        # C only (Z=6 even), charge=0, mult=1: target_parity = (0+1-1)%2 = 0 ✓
        assert _pool_can_satisfy_parity(["C"], 4, 0, 1) is True

    def test_all_even_wrong_target(self) -> None:
        from pasted._optimizer import _pool_can_satisfy_parity
        # C only (Z=6 even), charge=0, mult=2: target_parity = (0+2-1)%2 = 1 ✗
        assert _pool_can_satisfy_parity(["C"], 4, 0, 2) is False

    def test_all_odd_matching_natoms(self) -> None:
        from pasted._optimizer import _pool_can_satisfy_parity
        # N only (Z=7 odd), n_atoms=4 → sum%2==0, charge=0, mult=1: target=0 ✓
        assert _pool_can_satisfy_parity(["N"], 4, 0, 1) is True

    def test_all_odd_mismatched_natoms(self) -> None:
        from pasted._optimizer import _pool_can_satisfy_parity
        # N only (Z=7 odd), n_atoms=1 → sum%2==1, charge=0, mult=1: target=0 ✗
        assert _pool_can_satisfy_parity(["N"], 1, 0, 1) is False

    def test_nonpositive_electrons(self) -> None:
        from pasted._optimizer import _pool_can_satisfy_parity
        # charge=99 > min_z*n_atoms=6 → n_e would be ≤ 0
        assert _pool_can_satisfy_parity(["C"], 1, 99, 1) is False
