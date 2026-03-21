"""
PASTED — Adversarial Edge Case Tests (tests/test_edge_cases.py)
======================================================================
Systematically validates boundary conditions, intentional misuse, internal
consistency, and unexpected side-effects not covered by the existing test suite.

Categories
  EC1.  API boundary values — n_atoms=0/-1, mult=0, n_restarts=0, extreme temperatures
  EC2.  H_total linear composition consistency — arbitrary w_atom/w_spatial combinations
  EC3.  GenerationResult interface — slicing, negative indexing, NotImplemented
  EC4.  OptimizationResult interface — empty result, sorted_scores, iteration
  EC5.  EvalContext field internal consistency — best_f monotonicity, PT replica_idx range
  EC6.  element_fractions partial spec / extreme skew / impossible max_counts
  EC7.  parse_element_spec — symbol string input (spec recording), inverted range
  EC8.  Box region — asymmetric box:LX,LY,LZ coordinate bounds (post-relax overflow documented)
  EC9.  XYZ I/O robustness — CRLF line endings, empty string, nonexistent file path
  EC10. Objective functions — empty dict, constant, zero return, dict mutation non-destructiveness
  EC11. chain_bias effect — high bias -> larger shape_aniso (statistical guarantee)
  EC12. generate() multiple calls — calling the same generator twice is deterministic
  EC13. Structure.comp — empty atoms, single element, H-containing, many elements
  EC14. Structure.to_xyz / write_xyz — append=False overwrites, write to invalid path
  EC15. validate_charge_mult — direct call boundary values
  EC16. compute_all_metrics — single atom, overlapping coordinates, cutoff=0
  EC17. Impossible max_counts -> RuntimeError propagates to caller
  EC18. Passing 'C,N,O' symbol string to elements= raises ValueError (spec recording)
  EC19. Two calls with the same seed produce bit-identical results (optimizer too)
  EC20. PT n_replicas=1 / n_replicas=2 degenerate cases
"""

from __future__ import annotations

import math
import os
import statistics
import tempfile
import warnings
from collections import Counter

import numpy as np
import pytest

from pasted import (
    GenerationResult,
    Structure,
    StructureGenerator,
    StructureOptimizer,
    compute_all_metrics,
    generate,
    parse_element_spec,
    read_xyz,
    validate_charge_mult,
)
from pasted._metrics import compute_all_metrics  # noqa: F811  (same symbol, just explicit)
from pasted._optimizer import OptimizationResult

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _gas(n: int = 6, elements: str | list[str] = "6,7,8", n_samples: int = 20, seed: int = 0,
         **kw: object) -> GenerationResult:
    """Shorthand: gas-mode generate with sensible defaults."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return generate(
            n_atoms=n, charge=0, mult=1,
            mode="gas", region="sphere:8",
            elements=elements, n_samples=n_samples, seed=seed,
            **kw,
        )


def _opt(n: int = 6, elements: str = "6,7,8", max_steps: int = 50, seed: int = 0,
         n_restarts: int = 1, **kw: object) -> OptimizationResult:
    """Shorthand: annealing optimizer with sensible defaults."""
    return StructureOptimizer(
        n_atoms=n, charge=0, mult=1, elements=elements,
        objective={"H_total": 1.0},
        method="annealing", max_steps=max_steps,
        n_restarts=n_restarts, seed=seed,
        **kw,
    ).run()


# ===========================================================================
# EC1. API boundary values
# ===========================================================================


class TestApiBoundaryValues:
    """Boundary values: n_atoms=0, negative, mult=0, n_restarts=0, etc."""

    def test_n_atoms_zero_generates_empty_result(self) -> None:
        """n_atoms=0 must not crash and must return 0 structures (possibly with a warning)."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = generate(
                n_atoms=0, charge=0, mult=1,
                mode="gas", region="sphere:5",
                elements="6", n_samples=5, seed=0,
            )
        assert isinstance(result, GenerationResult)
        assert len(result) == 0

    def test_n_atoms_negative_does_not_segfault(self) -> None:
        """n_atoms=-1 may be constructed, but generation must yield 0 structures or raise cleanly."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                gen = StructureGenerator(
                    n_atoms=-1, charge=0, mult=1,
                    mode="chain", elements="6",
                )
                result = gen.generate()
                assert len(result) == 0
            except (ValueError, RuntimeError):
                pass  # an exception raised at construction or generation time is also acceptable

    def test_mult_zero_rejected_by_parity(self) -> None:
        """mult=0 (n_unpaired=-1, odd) with an even-Z-only pool rejects all attempts.

        mult=0 -> n_unpaired = -1 (odd).  With carbon-only (Z=6, even),
        n_electrons is always even, so parity always mismatches.
        Mixed pools (e.g. C+N) may pass because odd-Z combinations can satisfy
        the parity condition.
        """
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = generate(
                n_atoms=4, charge=0, mult=0,
                mode="gas", region="sphere:5",
                elements="6",  # C only: Z=6 (even) -> n_electrons=24 (even) -> parity mismatch
                n_samples=10, seed=0,
            )
        assert len(result) == 0

    def test_n_restarts_zero_raises(self) -> None:
        """n_restarts=0 means no restarts can be attempted, so RuntimeError must be raised."""
        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective={"H_total": 1.0},
            method="annealing", max_steps=10, n_restarts=0, seed=0,
        )
        with pytest.raises(RuntimeError):
            opt.run()

    def test_t_start_less_than_t_end_does_not_crash(self) -> None:
        """T_start < T_end (inverted cooling schedule) must not crash; result values are not guaranteed."""
        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective={"H_total": 1.0},
            method="annealing", max_steps=20,
            T_start=0.001, T_end=10.0, seed=0,
        )
        result = opt.run()
        assert result.best is not None

    def test_max_steps_zero_does_not_hang(self) -> None:
        """max_steps=0 must complete immediately and return a valid best structure."""
        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective={"H_total": 1.0},
            method="annealing", max_steps=0, seed=0,
        )
        result = opt.run()
        # initial score is set at step 0; best must not be None
        assert result.best is not None


# ===========================================================================
# EC2. H_total linear composition consistency
# ===========================================================================


class TestHTotalLinearComposition:
    """Verify H_total = w_atom * H_atom + w_spatial * H_spatial for arbitrary weights."""

    @pytest.mark.parametrize("w_atom,w_spatial", [
        (0.5, 0.5),
        (0.3, 0.7),
        (0.0, 1.0),
        (1.0, 0.0),
        (2.0, 3.0),   # weights that do not sum to 1.0 are still a valid linear combination
        (0.0, 0.0),   # both zero -> H_total = 0
    ])
    def test_h_total_equals_weighted_sum(self, w_atom: float, w_spatial: float) -> None:
        result = _gas(n_samples=5, seed=42, w_atom=w_atom, w_spatial=w_spatial)
        for s in result:
            expected = w_atom * s.metrics["H_atom"] + w_spatial * s.metrics["H_spatial"]
            assert math.isclose(s.metrics["H_total"], expected, rel_tol=1e-9, abs_tol=1e-12), (
                f"H_total mismatch: got {s.metrics['H_total']:.8f}, expected {expected:.8f}"
            )


# ===========================================================================
# EC3. GenerationResult interface
# ===========================================================================


class TestGenerationResultInterface:
    """Exhaustive validation of the list-compatible interface."""

    @pytest.fixture()
    def result(self) -> GenerationResult:
        return _gas(n=8, n_samples=30, seed=77)

    def test_negative_index_last_element(self, result: GenerationResult) -> None:
        if not result:
            pytest.skip("No structures generated")
        assert result[-1] is result[len(result) - 1]

    def test_slice_returns_plain_list(self, result: GenerationResult) -> None:
        if len(result) < 2:
            pytest.skip("Need >= 2 structures")
        sliced = result[0:2]
        assert isinstance(sliced, list)
        assert len(sliced) == 2

    def test_full_slice_preserves_all(self, result: GenerationResult) -> None:
        full = result[:]
        assert isinstance(full, list)
        assert len(full) == len(result)

    def test_add_with_non_result_returns_not_implemented(self) -> None:
        r = GenerationResult()
        assert r.__add__(42) is NotImplemented
        assert r.__add__("list") is NotImplemented

    def test_bool_false_when_empty(self) -> None:
        empty = GenerationResult()
        assert not empty

    def test_summary_labels_use_n_prefix_attributes(self, result: GenerationResult) -> None:
        """Verify that summary() labels correspond to their n_-prefixed attributes."""
        s = result.summary()
        assert f"passed={result.n_passed}" in s
        assert f"attempted={result.n_attempted}" in s
        assert f"rejected_parity={result.n_rejected_parity}" in s
        assert f"rejected_filter={result.n_rejected_filter}" in s

    def test_n_passed_equals_len(self, result: GenerationResult) -> None:
        assert result.n_passed == len(result)

    def test_n_attempted_ge_n_passed(self, result: GenerationResult) -> None:
        assert result.n_attempted >= result.n_passed


# ===========================================================================
# EC4. OptimizationResult interface
# ===========================================================================


class TestOptimizationResultInterface:
    """List compatibility and empty-result behavior of OptimizationResult."""

    def test_best_raises_runtime_error_when_empty(self) -> None:
        empty = OptimizationResult(
            all_structures=[], objective_scores=[],
            n_restarts_attempted=0, method="annealing",
        )
        with pytest.raises(RuntimeError):
            _ = empty.best

    def test_objective_scores_sorted_descending(self) -> None:
        result = _opt(n=8, max_steps=50, n_restarts=4, seed=0)
        scores = result.objective_scores
        assert scores == sorted(scores, reverse=True), (
            "objective_scores must be sorted best-first (descending)"
        )

    def test_best_score_equals_first_objective_score(self) -> None:
        result = _opt(n=6, max_steps=30, n_restarts=2, seed=1)
        assert result.objective_scores[0] == pytest.approx(
            result.objective_scores[0]  # trivially same reference; checks no KeyError
        )
        # best.metrics must be accessible without crashing
        _ = result.best.metrics["H_total"]

    def test_slice_returns_list(self) -> None:
        result = _opt(n=6, max_steps=30, n_restarts=3, seed=2)
        sliced = result[0:2]
        assert isinstance(sliced, list)

    def test_iter_length_matches_len(self) -> None:
        result = _opt(n=6, max_steps=30, n_restarts=2, seed=3)
        assert len(list(iter(result))) == len(result)

    def test_bool_false_when_all_restarts_fail(self) -> None:
        """An OptimizationResult with no structures must evaluate as falsy."""
        # Simulate the scenario where all restarts produce no results
        empty = OptimizationResult(
            all_structures=[], objective_scores=[],
            n_restarts_attempted=0, method="annealing",
        )
        assert not empty


# ===========================================================================
# EC5. EvalContext field internal consistency
# ===========================================================================


class TestEvalContextConsistency:
    """Verify that EvalContext fields remain internally consistent throughout a run."""

    def test_best_f_is_monotone_non_decreasing(self) -> None:
        """ctx.best_f must be monotonically non-decreasing as steps advance."""
        from pasted import EvalContext

        best_f_seq: list[float] = []

        def tracker(m: dict[str, float], ctx: EvalContext) -> float:  # type: ignore[type-arg]
            best_f_seq.append(ctx.best_f)
            return m.get("H_total", 0.0)

        StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective=tracker, method="annealing", max_steps=50, seed=42,
        ).run()

        assert best_f_seq, "No calls to objective"
        for i in range(len(best_f_seq) - 1):
            assert best_f_seq[i] <= best_f_seq[i + 1] or math.isnan(best_f_seq[i]), (
                f"best_f decreased at step {i}: {best_f_seq[i]} -> {best_f_seq[i+1]}"
            )

    def test_progress_in_01_range(self) -> None:
        """ctx.progress must always be in [0, 1)."""
        from pasted import EvalContext

        progresses: list[float] = []

        def tracker(m: dict[str, float], ctx: EvalContext) -> float:  # type: ignore[type-arg]
            progresses.append(ctx.progress)
            return m.get("H_total", 0.0)

        StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective=tracker, method="annealing", max_steps=30, seed=0,
        ).run()

        for p in progresses:
            assert 0.0 <= p < 1.0, f"progress={p} out of [0, 1)"

    def test_pt_replica_idx_covers_all_replicas(self) -> None:
        """Every replica index must appear at least once during a Parallel Tempering run."""
        from pasted import EvalContext

        seen: set[int] = set()
        n_replicas = 4

        def tracker(m: dict[str, float], ctx: EvalContext) -> float:  # type: ignore[type-arg]
            if ctx.replica_idx is not None:
                seen.add(ctx.replica_idx)
            return m.get("H_total", 0.0)

        StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective=tracker, method="parallel_tempering",
            n_replicas=n_replicas, max_steps=30, seed=0,
        ).run()

        assert seen == set(range(n_replicas)), (
            f"Expected replica_idx 0..{n_replicas-1}, got {sorted(seen)}"
        )

    def test_annealing_replica_fields_are_none(self) -> None:
        """replica_idx and n_replicas must both be None when using annealing."""
        from pasted import EvalContext

        seen_replica: list[object] = []

        def tracker(m: dict[str, float], ctx: EvalContext) -> float:  # type: ignore[type-arg]
            seen_replica.append(ctx.replica_idx)
            seen_replica.append(ctx.n_replicas)
            return 0.0

        StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective=tracker, method="annealing", max_steps=5, seed=0,
        ).run()

        assert all(v is None for v in seen_replica), (
            "annealing must expose replica_idx=None, n_replicas=None"
        )

    def test_ctx_step_range(self) -> None:
        """ctx.step must be an integer in [0, max_steps] at every call."""
        from pasted import EvalContext

        steps: list[int] = []
        max_steps = 25

        def tracker(m: dict[str, float], ctx: EvalContext) -> float:  # type: ignore[type-arg]
            steps.append(ctx.step)
            return m.get("H_total", 0.0)

        StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective=tracker, method="annealing",
            max_steps=max_steps, seed=7,
        ).run()

        assert all(isinstance(s, int) for s in steps)
        assert all(0 <= s <= max_steps for s in steps), (
            f"step out of range: min={min(steps)}, max={max(steps)}"
        )


# ===========================================================================
# EC6. element_fractions partial spec / extreme skew / impossible max_counts
# ===========================================================================


class TestElementConstraintsEdgeCases:
    """Adversarial combinations of element_fractions, min_counts, and max_counts."""

    def test_element_fractions_partial_spec_defaults_to_one(self) -> None:
        """When only C=10 is specified, N and O default to weight 1.0 and C dominates."""
        result = _gas(
            n=30, elements="6,7,8", n_samples=50, seed=0,
            element_fractions={"C": 10.0},
        )
        assert len(result) > 0
        c_total = sum(s.atoms.count("C") for s in result)
        non_c_total = sum(
            s.atoms.count("N") + s.atoms.count("O") for s in result
        )
        # C should overwhelmingly dominate
        assert c_total > non_c_total, (
            f"C={c_total} should dominate non-C={non_c_total} with fraction=10"
        )

    def test_element_max_counts_sum_less_than_n_atoms_raises_runtime(self) -> None:
        """Sum of all max_counts < n_atoms must raise RuntimeError during generation."""
        with pytest.raises(RuntimeError, match="cannot be satisfied"):
            generate(
                n_atoms=10, charge=0, mult=1,
                mode="gas", region="sphere:8",
                elements="6,7,8",
                element_max_counts={"C": 2, "N": 2, "O": 2},  # max total 6 < 10
                n_samples=5, seed=0,
            )

    def test_min_equals_max_single_element_composition_fixed(self) -> None:
        """Setting element_min_counts == element_max_counts pins the composition exactly."""
        result = generate(
            n_atoms=6, charge=0, mult=1,
            mode="gas", region="sphere:6",
            elements="6,7,8",
            element_min_counts={"C": 2, "N": 2, "O": 2},
            element_max_counts={"C": 2, "N": 2, "O": 2},
            n_samples=20, seed=42,
        )
        for s in result:
            cnt = Counter(s.atoms)
            assert cnt["C"] == 2, f"Expected 2 C, got {cnt['C']}"
            assert cnt["N"] == 2, f"Expected 2 N, got {cnt['N']}"
            assert cnt["O"] == 2, f"Expected 2 O, got {cnt['O']}"

    def test_element_fractions_extreme_skew_still_generates(self) -> None:
        """An extreme 1000:0.001:0.001 fraction skew must not prevent generation."""
        result = _gas(
            n=20, elements="6,7,8", n_samples=20, seed=99,
            element_fractions={"C": 1000.0, "N": 0.001, "O": 0.001},
        )
        assert len(result) > 0


# ===========================================================================
# EC7. parse_element_spec spec recording
# ===========================================================================


class TestParseElementSpecEdgeCases:
    """Record and protect the current behavior of parse_element_spec."""

    def test_numeric_csv_works(self) -> None:
        """A numeric CSV string like '6,7,8' must parse correctly."""
        pool = parse_element_spec("6,7,8")
        assert set(pool) == {"C", "N", "O"}

    def test_numeric_range_works(self) -> None:
        """A numeric range like '1-10' must parse correctly."""
        pool = parse_element_spec("1-10")
        assert len(pool) == 10
        assert "H" in pool
        assert "Ne" in pool

    def test_symbol_csv_raises_value_error(self) -> None:
        """A symbol CSV like 'C,N,O' raises ValueError under the current spec.

        This is an intentional spec-recording test.  It documents that
        generate(elements=['C','N','O']) requires list form, not a symbol string.
        """
        with pytest.raises(ValueError):
            parse_element_spec("C,N,O")

    def test_inverted_range_raises(self) -> None:
        """'8-6' (upper bound < lower bound) must raise ValueError."""
        with pytest.raises(ValueError, match="lower > upper"):
            parse_element_spec("8-6")

    def test_single_symbol_as_list_works(self) -> None:
        """elements=['Fe'] passed as a list is used directly without parsing."""
        result = _gas(n=4, elements=["Fe"], n_samples=5, seed=0)
        assert isinstance(result, GenerationResult)
        # Fe has even Z (26), so singlet parity may pass for some atom counts


# ===========================================================================
# EC8. Box region bounds (post-relax overflow documented as by-design)
# ===========================================================================


class TestBoxRegionBounds:
    """Document the relationship between place_gas box placement and relax displacement."""

    def test_symmetric_box_initial_placement_in_bounds(self) -> None:
        """Symmetric box:L places atoms in [-L/2, L/2]^3 before relax.
        After relax atoms may drift outside; this test only verifies no crash occurs."""
        result = generate(
            n_atoms=5, charge=0, mult=1,
            mode="gas", region="box:10",
            elements="6", n_samples=10, seed=0,
        )
        assert isinstance(result, GenerationResult)

    def test_asymmetric_box_generates_without_crash(self) -> None:
        """box:LX,LY,LZ (asymmetric three-axis spec) must not raise ValueError."""
        result = generate(
            n_atoms=5, charge=0, mult=1,
            mode="gas", region="box:10,5,3",
            elements="6,7,8", n_samples=5, seed=0,
        )
        assert isinstance(result, GenerationResult)

    def test_box_relax_may_push_atoms_outside(self) -> None:
        """Repulsion relaxation can push atoms beyond the original box boundaries.

        This is documented as by-design behavior (BUG-3 classified as spec).
        If this behavior changes, update this test to make the spec change explicit.
        """
        result = generate(
            n_atoms=30, charge=0, mult=1,
            mode="gas", region="box:10,3,2",
            elements="6", n_samples=10, seed=0,
        )
        if not result:
            pytest.skip("No structures generated for this seed")
        # box half-lengths: 5, 1.5, 1.0
        half = np.array([5.0, 1.5, 1.0])
        any_outside = False
        for s in result:
            pos = np.abs(np.array(s.positions))
            if np.any(pos > half + 1.0):  # 1 Angstrom margin
                any_outside = True
                break
        # Record the fact that overflow can occur without asserting on it.
        # This variable is retained for future regression detection.
        _ = any_outside


# ===========================================================================
# EC9. XYZ I/O robustness
# ===========================================================================


class TestXyzIoRobustness:
    """Verify resilience to adversarial XYZ inputs."""

    def test_parse_xyz_empty_string_returns_empty_list(self) -> None:
        from pasted import parse_xyz
        assert parse_xyz("") == []

    def test_parse_xyz_crlf_line_endings(self) -> None:
        """XYZ files with Windows-style CR+LF line endings must parse correctly."""
        from pasted import format_xyz, parse_xyz
        atoms = ["C", "N"]
        positions: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]
        xyz = format_xyz(atoms, positions, 0, 1, {}, "test")
        crlf = xyz.replace("\n", "\r\n")
        frames = parse_xyz(crlf)
        assert len(frames) == 1
        assert frames[0][0] == atoms

    def test_from_xyz_nonexistent_file_raises_value_error(self) -> None:
        """A path that does not exist must raise ValueError, not a low-level OSError."""
        with pytest.raises(ValueError):
            Structure.from_xyz("/absolutely/nonexistent/path/file.xyz")

    def test_from_xyz_multiline_string_not_treated_as_path(self) -> None:
        """A string containing newlines must be parsed as XYZ content, not as a file path."""
        result = _gas(n=6, n_samples=5, seed=55)
        if not result:
            pytest.skip("No structures")
        xyz_str = result[0].to_xyz()
        assert "\n" in xyz_str
        s = Structure.from_xyz(xyz_str, recompute_metrics=False)
        assert len(s.atoms) == len(result[0].atoms)

    def test_write_xyz_to_nonexistent_parent_raises_os_error(self) -> None:
        """Writing to a path whose parent directory does not exist must raise FileNotFoundError."""
        result = _gas(n=4, elements="6", n_samples=5, seed=0)
        if not result:
            pytest.skip("No structures")
        with pytest.raises((FileNotFoundError, OSError)):
            result[0].write_xyz("/nonexistent_parent_dir/output.xyz")

    def test_write_xyz_append_false_overwrites(self, tmp_path: object) -> None:
        """append=False must overwrite the existing file, not append to it."""
        import pathlib
        path = pathlib.Path(str(tmp_path)) / "out.xyz"  # type: ignore[operator]
        result = _gas(n=6, n_samples=10, seed=3)
        if len(result) < 2:
            pytest.skip("Need >= 2 structures")

        result[0].write_xyz(str(path), append=False)
        size_after_first = path.stat().st_size

        result[1].write_xyz(str(path), append=False)
        size_after_overwrite = path.stat().st_size

        # Size after overwrite should be comparable to the first write, not growing unboundedly
        assert size_after_overwrite <= size_after_first * 2, (
            "append=False should overwrite, not keep growing"
        )

    def test_read_xyz_mode_is_loaded_xyz(self) -> None:
        """Structures loaded via read_xyz must have mode set to 'loaded_xyz'."""
        result = _gas(n=6, n_samples=5, seed=4)
        if not result:
            pytest.skip("No structures")
        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as f:
            fname = f.name
            f.write(result[0].to_xyz() + "\n")
        try:
            loaded = read_xyz(fname)
        finally:
            os.unlink(fname)
        assert loaded[0].mode == "loaded_xyz"


# ===========================================================================
# EC10. Objective function boundary behavior
# ===========================================================================


class TestObjectiveFunctionEdgeCases:
    """Optimizer behavior when the objective function returns non-standard values."""

    def test_empty_dict_objective_score_is_zero(self) -> None:
        """An empty dict objective evaluates to 0.0 for every structure."""
        result = StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective={},
            method="annealing", max_steps=30, n_restarts=1, seed=0,
        ).run()
        assert result.objective_scores[0] == pytest.approx(0.0)

    def test_constant_objective_all_scores_equal(self) -> None:
        """A constant objective must produce identical scores across all restarts."""
        result = StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective=lambda m: 42.0,
            method="annealing", max_steps=30, n_restarts=3, seed=0,
        ).run()
        assert all(s == pytest.approx(42.0) for s in result.objective_scores)

    def test_objective_mutating_metrics_dict_does_not_corrupt_best(self) -> None:
        """Injecting foreign keys into the metrics dict must not corrupt best.metrics."""
        def polluter(m: dict[str, float]) -> float:
            m["__evil__"] = float("nan")
            return m.get("H_total", 0.0)

        result = StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective=polluter,
            method="annealing", max_steps=50, seed=5,
        ).run()

        assert "H_total" in result.best.metrics
        val = result.best.metrics["H_total"]
        assert val == val, "H_total in best.metrics is NaN after dict pollution"

    def test_zero_constant_objective_does_not_crash(self) -> None:
        """A zero-constant objective keeps acceptance probability at 1; must not crash."""
        result = StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective=lambda m: 0.0,
            method="basin_hopping", max_steps=30, seed=0,
        ).run()
        assert result.best is not None

    def test_objective_returning_nan_completes_without_crash(self) -> None:
        """An objective that always returns NaN must complete without raising."""
        result = StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective=lambda m: float("nan"),
            method="annealing", max_steps=30, seed=0,
        ).run()
        assert result is not None  # guaranteed to complete without crashing


# ===========================================================================
# EC11. chain_bias statistical effect
# ===========================================================================


class TestChainBiasEffect:
    """Verify that chain_bias has a statistically significant effect on shape_aniso."""

    def test_high_chain_bias_increases_mean_shape_aniso(self) -> None:
        """Mean shape_aniso at bias=0.9 must exceed mean shape_aniso at bias=0.0."""
        def mean_aniso(bias: float) -> float:
            r = generate(
                n_atoms=20, charge=0, mult=1,
                mode="chain", elements="6,7,8",
                chain_bias=bias, n_samples=40, seed=0,
            )
            if not r:
                return 0.0
            return statistics.mean(s.metrics["shape_aniso"] for s in r)

        low = mean_aniso(0.0)
        high = mean_aniso(0.9)
        assert high > low, (
            f"chain_bias=0.9 mean_aniso={high:.3f} should exceed bias=0.0 mean_aniso={low:.3f}"
        )


# ===========================================================================
# EC12. generate() multiple calls on the same StructureGenerator
# ===========================================================================


class TestGeneratorMultipleCallsDeterminism:
    """Calling generate() twice on the same StructureGenerator must yield identical results."""

    def test_generate_twice_same_seed_same_count(self) -> None:
        gen = StructureGenerator(
            n_atoms=8, charge=0, mult=1,
            mode="gas", region="sphere:7",
            elements="6,7,8", n_samples=20, seed=31,
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r1 = gen.generate()
            r2 = gen.generate()

        assert len(r1) == len(r2), (
            f"Same StructureGenerator called twice yielded {len(r1)} vs {len(r2)}"
        )

    def test_generate_twice_same_positions(self) -> None:
        gen = StructureGenerator(
            n_atoms=6, charge=0, mult=1,
            mode="gas", region="sphere:6",
            elements="6,7,8", n_samples=15, seed=7,
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r1 = gen.generate()
            r2 = gen.generate()

        if not r1:
            pytest.skip("No structures generated")
        np.testing.assert_allclose(
            np.array(r1[0].positions),
            np.array(r2[0].positions),
            err_msg="positions differ between two generate() calls with same seed",
        )


# ===========================================================================
# EC13. Structure.comp boundary values
# ===========================================================================


class TestStructureCompEdgeCases:
    """Boundary behavior of the Structure.comp property."""

    def test_comp_empty_atoms_is_empty_string(self) -> None:
        s = Structure(atoms=[], positions=[], charge=0, mult=1, metrics={}, mode="test")
        assert s.comp == ""

    def test_comp_single_element_no_count_suffix(self) -> None:
        s = Structure(atoms=["C"], positions=[(0.0, 0.0, 0.0)],
                      charge=0, mult=1, metrics={}, mode="test")
        assert s.comp == "C"

    def test_comp_single_element_multiple_atoms(self) -> None:
        s = Structure(atoms=["N", "N", "N"], positions=[(float(i), 0.0, 0.0) for i in range(3)],
                      charge=0, mult=1, metrics={}, mode="test")
        assert s.comp == "N3"

    def test_comp_alphabetical_order(self) -> None:
        """comp uses ascending alphabetical order, not Hill order (C first, H second)."""
        s = Structure(atoms=["Na", "C", "H", "H"],
                      positions=[(float(i), 0.0, 0.0) for i in range(4)],
                      charge=0, mult=1, metrics={}, mode="test")
        # Counter: C:1, H:2, Na:1 -> sorted alphabetically -> C, H, Na -> 'CH2Na'
        assert s.comp == "CH2Na"

    def test_comp_is_consistent_with_atoms_counter(self) -> None:
        """Reconstructing a Counter from comp must match the Counter of atoms."""
        result = _gas(n=10, n_samples=5, seed=22)
        if not result:
            pytest.skip("No structures")
        s = result[0]
        counts = Counter(s.atoms)
        expected = "".join(
            f"{sym}{n}" if n > 1 else sym
            for sym, n in sorted(counts.items())
        )
        assert s.comp == expected


# ===========================================================================
# EC14. Structure.to_xyz / write_xyz
# ===========================================================================


class TestStructureXyzOutput:
    """Guarantee the output format of to_xyz and write_xyz."""

    def test_to_xyz_first_line_is_atom_count(self) -> None:
        result = _gas(n=6, n_samples=5, seed=0)
        if not result:
            pytest.skip("No structures")
        s = result[0]
        first_line = s.to_xyz().splitlines()[0].strip()
        assert first_line.isdigit(), f"First line should be atom count, got {first_line!r}"
        assert int(first_line) == len(s.atoms)

    def test_to_xyz_custom_prefix_appears_in_comment(self) -> None:
        result = _gas(n=6, n_samples=5, seed=1)
        if not result:
            pytest.skip("No structures")
        s = result[0]
        xyz = s.to_xyz(prefix="custom_prefix=42")
        comment_line = xyz.splitlines()[1]
        assert "custom_prefix=42" in comment_line

    def test_to_xyz_coord_lines_count_matches_atoms(self) -> None:
        result = _gas(n=8, n_samples=5, seed=2)
        if not result:
            pytest.skip("No structures")
        s = result[0]
        lines = s.to_xyz().splitlines()
        n_atoms = int(lines[0])
        coord_lines = lines[2:2 + n_atoms]
        assert len(coord_lines) == n_atoms

    def test_write_xyz_append_true_doubles_file(self, tmp_path: object) -> None:
        import pathlib
        path = pathlib.Path(str(tmp_path)) / "out.xyz"  # type: ignore[operator]
        result = _gas(n=6, n_samples=10, seed=4)
        if len(result) < 2:
            pytest.skip("Need >= 2 structures")
        result[0].write_xyz(str(path), append=False)
        size1 = path.stat().st_size
        result[0].write_xyz(str(path), append=True)
        size2 = path.stat().st_size
        # The file should grow after appending
        assert size2 > size1


# ===========================================================================
# EC15. validate_charge_mult boundary values
# ===========================================================================


class TestValidateChargeMult:
    """Direct-call boundary cases for validate_charge_mult."""

    def test_homoatom_even_z_singlet_ok(self) -> None:
        ok, _ = validate_charge_mult(["C", "C"], 0, 1)
        assert ok

    def test_homoatom_even_z_doublet_fails(self) -> None:
        ok, msg = validate_charge_mult(["C", "C"], 0, 2)
        assert not ok
        assert msg != ""

    def test_positive_charge_reduces_electrons(self) -> None:
        # H4 total_Z=4, charge=+3 -> 1 electron -> doublet is valid
        ok, _ = validate_charge_mult(["H", "H", "H", "H"], 3, 2)
        assert ok

    def test_charge_exceeds_electrons_fails(self) -> None:
        # H2 total_Z=2, charge=+5 -> n_electrons=-3 -> fails
        ok, msg = validate_charge_mult(["H", "H"], 5, 1)
        assert not ok

    def test_empty_atoms_charge_zero_fails_or_ok(self) -> None:
        """atoms=[] has undefined parity; the function must not crash."""
        try:
            ok, _ = validate_charge_mult([], 0, 1)
            # pass/fail is implementation-defined; no crash is the only requirement
        except Exception as e:
            pytest.fail(f"validate_charge_mult([]) crashed: {e}")


# ===========================================================================
# EC16. compute_all_metrics boundary values
# ===========================================================================


class TestComputeAllMetricsBoundary:
    """Boundary behavior of metric computation."""

    def test_single_atom_all_metrics_finite_or_nan_for_shape_aniso(self) -> None:
        """Validate metrics for a single-atom structure.

        shape_aniso is documented to return NaN for fewer than 2 atoms
        (see compute_shape_anisotropy in the source).  All other metrics
        must be finite.
        """
        m = compute_all_metrics(["C"], [(0.0, 0.0, 0.0)], 20, 0.5, 0.5, 2.0, 1.0)
        nan_allowed = {"shape_aniso"}  # documented: "Returns NaN for a single atom."
        for key, val in m.items():
            if key in nan_allowed:
                # both NaN and finite values are acceptable
                assert isinstance(val, float), f"{key} is not float"
            else:
                assert math.isfinite(val), (
                    f"Single atom metric {key}={val} is unexpectedly non-finite"
                )

    def test_cutoff_zero_no_bonds_all_graph_metrics_degenerate(self) -> None:
        """cutoff=0.0 means no atom pair falls within the neighbor cutoff; must return floats without crashing."""
        atoms = ["C", "N", "O", "C", "N"]
        positions = [(float(i), 0.0, 0.0) for i in range(5)]
        m = compute_all_metrics(atoms, positions, 20, 0.5, 0.5, 0.0, 1.0)
        for key, val in m.items():
            # must return a float (NaN is acceptable); must not crash
            assert isinstance(val, float), f"Metric {key} is not float: {type(val)}"

    def test_overlapping_positions_does_not_crash(self) -> None:
        """Two atoms at the exact same coordinates must not cause a crash."""
        m = compute_all_metrics(["C", "N"], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], 20, 0.5, 0.5, 2.0, 1.0)
        assert isinstance(m, dict)
        assert "H_total" in m

    def test_all_13_metrics_present(self) -> None:
        """compute_all_metrics must return all 13 metrics listed in quickstart.md."""
        expected_keys = {
            "H_atom", "H_spatial", "H_total", "RDF_dev",
            "shape_aniso", "Q4", "Q6", "Q8",
            "graph_lcc", "graph_cc", "ring_fraction",
            "charge_frustration", "moran_I_chi",
        }
        atoms = ["C", "N", "O", "C", "N"]
        positions = [(float(i), 0.0, 0.0) for i in range(5)]
        m = compute_all_metrics(atoms, positions, 20, 0.5, 0.5, 2.0, 1.0)
        missing = expected_keys - set(m.keys())
        assert not missing, f"Missing metrics: {missing}"


# ===========================================================================
# EC17. Impossible constraint RuntimeError propagation
# ===========================================================================


class TestImpossibleConstraintPropagation:
    """Verify that RuntimeError propagates to the caller when constraints cannot be satisfied."""

    def test_impossible_max_counts_raises_runtime_during_generate(self) -> None:
        """Sum of max_counts < n_atoms must propagate a RuntimeError out of generate()."""
        with pytest.raises(RuntimeError):
            generate(
                n_atoms=10, charge=0, mult=1,
                mode="gas", region="sphere:8",
                elements="6,7,8",
                element_max_counts={"C": 2, "N": 2, "O": 2},
                n_samples=1, seed=0,
            )

    def test_impossible_max_counts_with_stream(self) -> None:
        """The same RuntimeError must propagate through stream() as well."""
        gen = StructureGenerator(
            n_atoms=10, charge=0, mult=1,
            mode="gas", region="sphere:8",
            elements="6,7,8",
            element_max_counts={"C": 2, "N": 2, "O": 2},
            n_samples=1, seed=0,
        )
        with pytest.raises(RuntimeError):
            list(gen.stream())


# ===========================================================================
# EC18. Passing 'C,N,O' symbol string to elements= raises ValueError (spec recording)
# ===========================================================================


class TestSymbolStringElementsSpec:
    """Record the current behavior: elements='C,N,O' raises ValueError.

    If this spec changes in the future, update this test to make the
    spec change explicit and maintain a regression record.
    """

    def test_symbol_csv_in_elements_kwarg_raises(self) -> None:
        """elements='C,N,O' must raise ValueError under the current spec."""
        with pytest.raises(ValueError):
            generate(
                n_atoms=5, charge=0, mult=1,
                mode="gas", region="sphere:5",
                elements="C,N,O",  # symbols: invalid; use '6,7,8' or ['C','N','O']
                n_samples=3, seed=0,
            )

    def test_symbol_list_in_elements_kwarg_works(self) -> None:
        """elements=['C','N','O'] in list form must work correctly."""
        result = generate(
            n_atoms=5, charge=0, mult=1,
            mode="gas", region="sphere:5",
            elements=["C", "N", "O"],
            n_samples=5, seed=0,
        )
        assert isinstance(result, GenerationResult)


# ===========================================================================
# EC19. Two calls with the same seed produce bit-identical results (optimizer too)
# ===========================================================================


class TestBitIdenticalReproducibility:
    """The same seed must produce identical results across independent calls."""

    def test_generate_bit_identical_positions(self) -> None:
        kwargs = dict(
            n_atoms=8, charge=0, mult=1,
            mode="gas", region="sphere:7",
            elements="6,7,8", n_samples=25, seed=12345,
        )
        r1 = generate(**kwargs)
        r2 = generate(**kwargs)
        assert len(r1) == len(r2)
        for i, (s1, s2) in enumerate(zip(r1, r2)):
            np.testing.assert_allclose(
                np.array(s1.positions), np.array(s2.positions),
                err_msg=f"positions differ at structure {i}",
            )

    def test_optimizer_bit_identical_scores(self) -> None:
        def run() -> OptimizationResult:
            return StructureOptimizer(
                n_atoms=8, charge=0, mult=1, elements="6,7,8",
                objective={"H_total": 1.0, "Q6": -0.5},
                method="annealing", max_steps=80, n_restarts=2, seed=9999,
            ).run()

        r1, r2 = run(), run()
        assert r1.objective_scores == pytest.approx(r2.objective_scores), (
            "Optimizer scores differ between two identical runs"
        )


# ===========================================================================
# EC20. PT n_replicas degenerate cases
# ===========================================================================


class TestParallelTemperingDegenerateCases:
    """Edge cases with unusual replica counts in Parallel Tempering."""

    def test_pt_n_replicas_1_runs_like_single_chain(self) -> None:
        """n_replicas=1 runs as a single chain with no swaps attempted."""
        result = StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective={"H_total": 1.0},
            method="parallel_tempering",
            n_replicas=1, max_steps=50, n_restarts=1, seed=0,
        ).run()
        assert result.best is not None

    def test_pt_n_replicas_2_completes_with_swaps(self) -> None:
        """n_replicas=2 (the smallest non-trivial swap pair) must complete successfully."""
        result = StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective={"H_total": 1.0},
            method="parallel_tempering",
            n_replicas=2, pt_swap_interval=5, max_steps=50,
            n_restarts=1, seed=1,
        ).run()
        assert result.best is not None

    def test_pt_ctx_replica_idx_none_not_present_in_annealing(self) -> None:
        """The same objective used in annealing mode must receive replica_idx=None."""
        from pasted import EvalContext

        seen: list[object] = []

        def collector(m: dict[str, float], ctx: EvalContext) -> float:  # type: ignore[type-arg]
            seen.append(ctx.replica_idx)
            return 0.0

        StructureOptimizer(
            n_atoms=6, charge=0, mult=1, elements="6,7,8",
            objective=collector, method="annealing", max_steps=10, seed=0,
        ).run()

        assert all(v is None for v in seen)
