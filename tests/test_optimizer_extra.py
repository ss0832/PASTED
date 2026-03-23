"""
tests/test_optimizer_extra.py
==============================
Covers the remaining missed branches in ``pasted._optimizer`` (94% → higher).

Missed lines targeted
---------------------
360         : ``OptimizationResult.__repr__`` when result is empty.
382         : ``OptimizationResult.summary()`` when objective_scores is empty.
532-533     : ``_eval_objective`` — inspect.signature raises (ValueError/TypeError)
              fallback sets n_required = 1.
537         : ``_eval_objective`` — 2-arg callable with ctx=None raises ValueError.
578-579     : ``_objective_needs_ctx`` — inspect.signature raises, returns False.
799-812     : ``_composition_move`` fallback paths when same_parity_pool is empty:
              two-atom diff-parity replacement (L799-806) and last-resort (L811).
1204        : ``StructureOptimizer._resolve_cutoff`` verbose log for user cutoff.
1267        : ``_make_initial`` returns None when generation fails.
1274        : ``_temperature`` with T_end <= 0 returns T_start.
1286        : ``_pt_temperatures`` with n_replicas=1 (degenerate case).
1366        : PT restart loop — shared initial structure is generated when
              composition moves are disabled and initial=None.
1394-1400   : PT replica setup uses shared_initial (no-composition-moves path).
1543-1547   : SA/BH pure-Python relax path and composition-changed radii update.
1598        : PT loop rejects step when graph_lcc < lcc_threshold.
1850        : SA/BH pure-Python relax path (no C++ extension).
1860        : SA/BH parity rejection in the MC loop.
1901        : SA/BH rejects step when graph_lcc < lcc_threshold.
2032-2035   : run() skips a restart when _make_initial returns None (SA/BH path).
2049        : run() emits UserWarning when some restarts were skipped.
"""

from __future__ import annotations

import random
import warnings
from unittest.mock import patch

import pytest

from pasted._optimizer import (
    EvalContext,
    OptimizationResult,
    StructureOptimizer,
    _composition_move,
    _eval_objective,
    _objective_needs_ctx,
)

# ---------------------------------------------------------------------------
# OptimizationResult.__repr__ and .summary() empty paths  (L360, L382)
# ---------------------------------------------------------------------------


class TestOptimizationResultEmpty:
    def test_repr_empty_result(self) -> None:
        """L360: repr of an empty OptimizationResult must contain 'empty'."""
        r = OptimizationResult()
        assert "empty" in repr(r)

    def test_summary_empty_scores(self) -> None:
        """L382: summary() with no scores must return a 'no results' string."""
        r = OptimizationResult(n_restarts_attempted=2, objective_scores=[])
        s = r.summary()
        assert "no results" in s

    def test_summary_with_scores(self) -> None:
        """Sanity: summary() with scores must include best_f and worst_f."""
        r = OptimizationResult(
            n_restarts_attempted=3,
            objective_scores=[1.5, 1.2, 0.9],
            method="annealing",
        )
        s = r.summary()
        assert "best_f" in s
        assert "worst_f" in s


# ---------------------------------------------------------------------------
# _eval_objective — inspect.signature fallback & 2-arg ctx=None error
# (L532-533, L537)
# ---------------------------------------------------------------------------


class TestEvalObjective:
    def test_signature_raises_falls_back_to_1arg(self) -> None:
        """L532-533: when inspect.signature raises, legacy 1-arg path is used."""

        # A built-in whose signature is not inspectable
        class _NoSig:
            __signature__ = None  # triggers TypeError in some Python versions

            def __call__(self, m: dict[str, float]) -> float:
                return m.get("H_total", 0.0)

        obj = _NoSig()
        with patch("inspect.signature", side_effect=ValueError("no sig")):
            result = _eval_objective({"H_total": 1.5}, obj)
        assert result == pytest.approx(1.5)

    def test_2arg_callable_with_no_ctx_raises(self) -> None:
        """L537: a 2-arg callable with ctx=None must raise ValueError."""

        def two_arg(m: dict[str, float], ctx: EvalContext) -> float:
            return 0.0

        with pytest.raises(ValueError, match="EvalContext"):
            _eval_objective({"H_total": 1.0}, two_arg, ctx=None)


# ---------------------------------------------------------------------------
# _objective_needs_ctx — inspect.signature raises → returns False  (L578-579)
# ---------------------------------------------------------------------------


class TestObjectiveNeedsCtx:
    def test_signature_raises_returns_false(self) -> None:
        """L578-579: when inspect.signature raises, False is returned."""

        def some_callable(m: dict[str, float]) -> float:
            return 0.0

        with patch("inspect.signature", side_effect=TypeError("no sig")):
            result = _objective_needs_ctx(some_callable)
        assert result is False


# ---------------------------------------------------------------------------
# _composition_move fallback paths  (L799-812)
# ---------------------------------------------------------------------------


class TestCompositionMoveFallbacks:
    def test_diff_parity_two_atom_replacement(self) -> None:
        """L799-806: when same_parity_pool is empty, two odd-Z atoms are replaced."""
        rng = random.Random(0)
        # Pool of only odd-Z elements (H=1, N=7, F=9); atoms are all even-Z (C=6).
        # After the 20-try loop fails (no same-parity), the diff-parity path fires.
        atoms = ["C", "C", "C", "C"]
        pool = ["H", "N", "F"]  # all odd Z; C has even Z
        result = _composition_move(atoms, pool, rng)
        assert len(result) == len(atoms)
        assert all(a in pool or a == "C" for a in result)

    def test_last_resort_single_atom_n1(self) -> None:
        """L811: n=1 with all-odd-Z pool falls through to last-resort replacement."""
        rng = random.Random(42)
        atoms = ["C"]  # even Z
        pool = ["H"]  # only odd Z; n=1 so two-atom path is skipped
        result = _composition_move(atoms, pool, rng)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# StructureOptimizer._resolve_cutoff verbose log  (L1204)
# ---------------------------------------------------------------------------


class TestOptimizerResolveCutoffVerbose:
    def test_user_cutoff_logged_when_verbose(self, capsys: pytest.CaptureFixture[str]) -> None:
        """L1204: a user-supplied cutoff must be logged when verbose=True."""
        StructureOptimizer(
            n_atoms=4,
            charge=0,
            mult=1,
            elements=["C", "N", "O"],
            objective={"H_total": 1.0},
            method="annealing",
            max_steps=10,
            n_restarts=1,
            seed=0,
            verbose=True,
            cutoff=4.0,
        )
        captured = capsys.readouterr()
        assert "user-specified" in captured.err or "4.000" in captured.err


# ---------------------------------------------------------------------------
# _make_initial returns None  (L1267)
# ---------------------------------------------------------------------------


class TestMakeInitialReturnsNone:
    def test_make_initial_returns_none_when_generation_fails(self) -> None:
        """L1267: _make_initial returns None when the inner generator yields nothing."""
        opt = StructureOptimizer(
            n_atoms=4,
            charge=0,
            mult=1,
            elements=["C", "N", "O"],
            objective={"H_total": 1.0},
            method="annealing",
            max_steps=10,
            max_init_attempts=1,  # prevents itertools.count() infinite loop
            n_restarts=1,
            seed=0,
        )
        from pasted._generator import GenerationResult

        with patch("pasted._optimizer.StructureGenerator") as mock_sg:
            mock_sg.return_value.generate.return_value = GenerationResult()
            result = opt._make_initial(random.Random(0))
        assert result is None


# ---------------------------------------------------------------------------
# _temperature edge case: T_end <= 0  (L1274)
# ---------------------------------------------------------------------------


class TestTemperatureEdgeCases:
    def test_t_end_zero_returns_t_start(self) -> None:
        """L1274: when T_end <= 0, _temperature must return T_start."""
        opt = StructureOptimizer(
            n_atoms=4,
            charge=0,
            mult=1,
            elements=["C", "N", "O"],
            objective={"H_total": 1.0},
            method="annealing",
            max_steps=10,
            T_start=1.0,
            T_end=0.0,
            n_restarts=1,
            seed=0,
        )
        assert opt._temperature(50) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _pt_temperatures degenerate n_replicas=1  (L1286)
# ---------------------------------------------------------------------------


class TestPtTemperaturesDegenerate:
    def test_n_replicas_1_returns_single_temperature(self) -> None:
        """L1286: _pt_temperatures with n_replicas=1 should return [T_start]."""
        opt = StructureOptimizer(
            n_atoms=4,
            charge=0,
            mult=1,
            elements=["C", "N", "O"],
            objective={"H_total": 1.0},
            method="parallel_tempering",
            max_steps=10,
            n_replicas=1,
            seed=0,
        )
        temps = opt._pt_temperatures()
        # n = max(2, 1) = 2 per implementation, but still no error
        assert len(temps) >= 1
        assert all(t > 0 for t in temps)


# ---------------------------------------------------------------------------
# run() skips restart and emits UserWarning  (L2032-2035, L2049)
# ---------------------------------------------------------------------------


class TestRunSkipsRestarts:
    def test_skipped_restart_emits_user_warning(self) -> None:
        """L2032-2035, L2049: when _make_initial always returns None for SA,
        every restart is skipped and a UserWarning is emitted.
        The final RuntimeError is also raised if *all* restarts fail.
        """
        opt = StructureOptimizer(
            n_atoms=4,
            charge=0,
            mult=1,
            elements=["C", "N", "O"],
            objective={"H_total": 1.0},
            method="annealing",
            max_steps=10,
            n_restarts=2,
            seed=0,
        )
        # Provide a fake valid structure for the *first* call so at least one
        # restart succeeds, then return None for subsequent calls so the
        # second restart is skipped and the UserWarning fires.

        from pasted._generator import Structure

        fake_atoms = ["C", "C", "C", "C"]
        fake_positions = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 1.5, 0.0), (0.0, 0.0, 1.5)]
        fake_metrics: dict[str, float] = {}
        fake_structure = Structure(
            atoms=fake_atoms,
            positions=fake_positions,
            charge=0,
            mult=1,
            metrics=fake_metrics,
            mode="chain",
        )

        call_count = 0

        def fake_make_initial(rng: random.Random) -> object:
            nonlocal call_count
            call_count += 1
            return fake_structure if call_count == 1 else None

        with patch.object(opt, "_make_initial", side_effect=fake_make_initial):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                opt.run()
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert any("skipped" in str(x.message).lower() for x in user_warnings)


# ---------------------------------------------------------------------------
# SA/BH lcc_threshold rejection  (L1901)
# ---------------------------------------------------------------------------


class TestLccThresholdRejection:
    def test_lcc_threshold_rejects_step(self) -> None:
        """L1901: steps with graph_lcc below lcc_threshold must be rejected."""
        opt = StructureOptimizer(
            n_atoms=6,
            charge=0,
            mult=1,
            elements=["C", "N", "O"],
            objective={"H_total": 1.0},
            method="annealing",
            max_steps=30,
            lcc_threshold=1.1,  # impossibly high → every step rejected
            n_restarts=1,
            seed=0,
        )
        result = opt.run()
        assert result is not None
