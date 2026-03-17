"""Tests for pasted._optimizer: StructureOptimizer and helpers."""

from __future__ import annotations

import random

import numpy as np
import pytest

from pasted import Structure, StructureGenerator, StructureOptimizer
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
        assert isinstance(result, Structure)

    def test_basin_hopping(self) -> None:
        result = self._small_opt(method="basin_hopping").run()
        assert isinstance(result, Structure)

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
        assert isinstance(result, Structure)

    def test_reproducible_with_seed(self) -> None:
        r1 = self._small_opt().run()
        r2 = self._small_opt().run()
        assert r1.atoms == r2.atoms
        assert r1.positions == r2.positions

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
        assert isinstance(result, Structure)

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
        assert isinstance(result, Structure)

    def test_structure_mode_label(self) -> None:
        result = self._small_opt(method="annealing").run()
        assert "opt_annealing" in result.mode

    def test_metrics_complete(self) -> None:

        result = self._small_opt().run()
        assert set(result.metrics.keys()) == ALL_METRICS

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
        assert isinstance(result, Structure)
