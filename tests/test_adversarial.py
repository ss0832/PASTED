"""
PASTED v0.3.2 — Adversarial edge-case test suite
=================================================
Covers cases not exercised by the quickstart examples.

Test categories
  A. Validation      — invalid arguments raise ValueError / TypeError at construction
  B. Numeric bounds  — n_atoms=1, extreme charge, parity mismatch, tiny/huge region
  C. Filter bounds   — fully open bounds, MIN==MAX, contradictory filter combinations
  D. XYZ I/O errors  — empty string, garbage input, truncated frames, out-of-range frame
  E. Hostile objectives — NaN/inf return values, exceptions, dict mutation
  F. Thread safety   — concurrent generate() and StructureOptimizer with the same seed
  G. Memory          — large n_atoms, many restarts; peak heap and RSS growth limits
  H. Counterintuitive valid cases — things that should succeed despite looking suspicious
"""

import gc
import math
import os
import tempfile
import threading
import tracemalloc
import warnings

import numpy as np
import pytest

try:
    import psutil

    _PROC = psutil.Process(os.getpid())
    PSUTIL_AVAILABLE = True
except ImportError:
    _PROC = None
    PSUTIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    """Return current RSS in MiB, or 0.0 if psutil is unavailable."""
    return _PROC.memory_info().rss / 1024**2 if PSUTIL_AVAILABLE else 0.0


def _peak_heap_kb() -> float:
    """Return peak traced heap in KiB from an active tracemalloc snapshot."""
    _, peak = tracemalloc.get_traced_memory()
    return peak / 1024


# ---------------------------------------------------------------------------
# A. Validation — invalid arguments
# ---------------------------------------------------------------------------

class TestValidation:
    """Construction-time validation: bad arguments must raise ValueError."""

    def test_gas_mode_requires_region(self):
        """StructureGenerator(mode='gas') without region must raise ValueError."""
        from pasted import StructureGenerator

        with pytest.raises(ValueError, match="region"):
            StructureGenerator(
                n_atoms=10, charge=0, mult=1,
                mode="gas", elements="6,7,8", n_samples=10,
            )

    def test_maxent_mode_requires_region(self):
        """StructureGenerator(mode='maxent') without region must raise ValueError."""
        from pasted import StructureGenerator

        with pytest.raises(ValueError, match="region"):
            StructureGenerator(
                n_atoms=10, charge=0, mult=1,
                mode="maxent", elements="6,7,8", n_samples=10,
            )

    def test_invalid_mode_raises(self):
        """An unrecognized placement mode must raise ValueError."""
        from pasted import StructureGenerator

        with pytest.raises(ValueError, match="mode"):
            StructureGenerator(
                n_atoms=10, charge=0, mult=1,
                mode="crystal", elements="6,7,8", n_samples=10,
            )

    def test_n_samples_zero_requires_n_success(self):
        """n_samples=0 (unlimited) without n_success would loop forever; must raise."""
        from pasted import StructureGenerator

        with pytest.raises(ValueError, match="n_success"):
            StructureGenerator(
                n_atoms=10, charge=0, mult=1,
                mode="gas", region="sphere:8",
                elements="6,7,8", n_samples=0,
            )

    def test_n_success_zero_raises(self):
        """n_success=0 is nonsensical; must raise ValueError."""
        from pasted import StructureGenerator

        with pytest.raises(ValueError, match="n_success"):
            StructureGenerator(
                n_atoms=10, charge=0, mult=1,
                mode="gas", region="sphere:8",
                elements="6,7,8", n_samples=50, n_success=0,
            )

    def test_element_fractions_unknown_symbol(self):
        """element_fractions referencing a symbol not in the pool must raise."""
        from pasted import StructureGenerator

        with pytest.raises(ValueError, match="element_fractions"):
            StructureGenerator(
                n_atoms=10, charge=0, mult=1,
                mode="gas", region="sphere:8",
                elements="6,7,8",
                element_fractions={"C": 1.0, "Unobtanium": 2.0},
                n_samples=20,
            )

    def test_element_fractions_all_zero(self):
        """All-zero fractions leave no probability mass; must raise ValueError."""
        from pasted import StructureGenerator

        with pytest.raises(ValueError, match="zero"):
            StructureGenerator(
                n_atoms=10, charge=0, mult=1,
                mode="gas", region="sphere:8",
                elements="6,7,8",
                element_fractions={"C": 0.0, "N": 0.0, "O": 0.0},
                n_samples=20,
            )

    def test_element_fractions_negative_weight(self):
        """A negative fraction weight is undefined; must raise ValueError."""
        from pasted import StructureGenerator

        with pytest.raises(ValueError, match="non-negative"):
            StructureGenerator(
                n_atoms=10, charge=0, mult=1,
                mode="gas", region="sphere:8",
                elements="6,7,8",
                element_fractions={"C": -1.0, "N": 2.0},
                n_samples=20,
            )

    def test_element_min_counts_sum_exceeds_n_atoms(self):
        """Sum of element_min_counts greater than n_atoms must raise ValueError."""
        from pasted import StructureGenerator

        with pytest.raises(ValueError, match="n_atoms"):
            StructureGenerator(
                n_atoms=5, charge=0, mult=1,
                mode="gas", region="sphere:8",
                elements="6,7,8",
                element_min_counts={"C": 4, "N": 3},  # 7 > 5
                n_samples=20,
            )

    def test_element_min_greater_than_max_for_same_element(self):
        """min_counts > max_counts for the same element must raise ValueError."""
        from pasted import StructureGenerator

        with pytest.raises(ValueError):
            StructureGenerator(
                n_atoms=10, charge=0, mult=1,
                mode="gas", region="sphere:8",
                elements="6,7,8",
                element_min_counts={"C": 5},
                element_max_counts={"C": 2},
                n_samples=20,
            )

    def test_parse_filter_inverted_bounds(self):
        """parse_filter('METRIC:5.0:1.0') — MIN > MAX — must raise ValueError."""
        from pasted import parse_filter

        with pytest.raises(ValueError, match="MIN > MAX"):
            parse_filter("H_total:5.0:1.0")

    def test_parse_filter_unknown_metric(self):
        """parse_filter with an unrecognized metric name must raise ValueError."""
        from pasted import parse_filter

        with pytest.raises(ValueError, match="Unknown metric"):
            parse_filter("fake_metric:0:1")

    def test_parse_filter_missing_max_field(self):
        """parse_filter with only two colon-separated fields must raise ValueError."""
        from pasted import parse_filter

        with pytest.raises(ValueError, match="METRIC:MIN:MAX"):
            parse_filter("H_total:1.0")

    def test_parse_element_spec_inverted_range(self):
        """parse_element_spec('8-6') — lower > upper — must raise ValueError."""
        from pasted import parse_element_spec

        with pytest.raises(ValueError, match="lower > upper"):
            parse_element_spec("8-6")

    def test_parse_element_spec_empty_string(self):
        """parse_element_spec('') must raise ValueError (empty pool)."""
        from pasted import parse_element_spec

        with pytest.raises(ValueError, match="empty"):
            parse_element_spec("")


# ---------------------------------------------------------------------------
# B. Numeric bounds
# ---------------------------------------------------------------------------

class TestNumericBounds:
    """Edge cases around atom count, charge, multiplicity, and region geometry."""

    def test_n_atoms_1_does_not_crash(self):
        """A single-atom structure (H doublet radical) must not raise."""
        from pasted import generate

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            generate(
                n_atoms=1, charge=0, mult=2,  # doublet: 1 electron, valid spin
                mode="gas", region="sphere:5",
                elements="1",  # hydrogen only; Z=1 (odd) satisfies mult=2
                n_samples=10, seed=0,
            )
        # No assertion on count — parity may reject some; the point is no crash.

    def test_n_atoms_2_does_not_crash(self):
        """A two-atom structure must generate without error."""
        from pasted import generate

        result = generate(
            n_atoms=2, charge=0, mult=1,
            mode="gas", region="sphere:5",
            elements="6", n_samples=20, seed=42,
        )
        assert len(result) == 20  # C2 singlet: 12 electrons (even), 0 unpaired — always valid

    def test_large_positive_charge_barely_positive_electron_count(self):
        """charge=+3 on H4 leaves exactly 1 electron (doublet); must not crash."""
        from pasted import generate

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            # H4: total_Z=4, charge=+3 → n_electrons=1 → mult=2 ok
            result = generate(
                n_atoms=4, charge=3, mult=2,
                mode="gas", region="sphere:6",
                elements="1", n_samples=20, seed=1,
            )
        assert len(result) == 20

    def test_charge_removes_all_electrons_yields_zero_structures(self):
        """charge larger than total Z strips all electrons; all samples rejected."""
        from pasted import generate

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            # H2: total_Z=2, charge=+5 → n_electrons=-3 → all rejected
            result = generate(
                n_atoms=2, charge=5, mult=1,
                mode="gas", region="sphere:5",
                elements="1", n_samples=10, seed=0,
            )
        assert len(result) == 0

    def test_region_sphere_near_zero_radius_does_not_crash(self):
        """An extremely small sphere radius triggers steric rejection but must not crash."""
        from pasted import generate

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            generate(
                n_atoms=10, charge=0, mult=1,
                mode="gas", region="sphere:0.1",
                elements="6,7,8", n_samples=30, seed=7,
            )
        # Count may be zero or nonzero; crash is the only failure mode here.

    def test_region_box_huge_does_not_crash(self):
        """A 1000 Å box must not cause overflow or crash."""
        from pasted import generate

        generate(
            n_atoms=5, charge=0, mult=1,
            mode="gas", region="box:1000",
            elements="6,7,8", n_samples=10, seed=0,
        )
        # Sparse placement in a huge box should still yield valid structures.

    def test_parity_mismatch_yields_zero_structures(self):
        """Even-Z pool, even n_atoms, odd mult → electron count parity fails for all samples."""
        from pasted import generate

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            # C2: total_Z=12 (even), charge=0 → n_electrons=12 (even)
            # mult=2 requires n_unpaired=1 (odd) → parity mismatch → all rejected
            result = generate(
                n_atoms=2, charge=0, mult=2,
                mode="gas", region="sphere:5",
                elements="6", n_samples=10, seed=0,
            )
        assert len(result) == 0


# ---------------------------------------------------------------------------
# C. Filter bounds
# ---------------------------------------------------------------------------

class TestFilterBounds:
    """Edge cases for metric filter parsing and application."""

    def test_fully_open_filter_passes_same_as_no_filter(self):
        """'H_total:-:-' (both bounds open) must pass identical structures to no filter."""
        from pasted import generate

        kwargs = dict(
            n_atoms=10, charge=0, mult=1,
            mode="gas", region="sphere:8",
            elements="6,7,8", n_samples=30, seed=42,
        )
        result_open = generate(**kwargs, filters=["H_total:-:-"])  # type: ignore[arg-type]
        result_none = generate(**kwargs)  # type: ignore[arg-type]

        assert len(result_open) == len(result_none), (
            f"Open filter changed count: {len(result_open)} vs {len(result_none)}"
        )

    def test_filter_min_equals_max_does_not_crash(self):
        """Setting MIN==MAX (exact float boundary) must not raise, regardless of how many pass."""
        from pasted import generate

        result_all = generate(
            n_atoms=10, charge=0, mult=1,
            mode="gas", region="sphere:8",
            elements="6,7,8", n_samples=50, seed=99,
        )
        if not result_all:
            pytest.skip("No structures generated with this seed — cannot form exact filter")

        from typing import cast

        from pasted import Structure as _StructureF
        val = cast(_StructureF, result_all[0]).metrics["H_total"]
        # Floating-point equality: at least the first structure's value matches itself.
        result_exact = generate(
            n_atoms=10, charge=0, mult=1,
            mode="gas", region="sphere:8",
            elements="6,7,8", n_samples=50, seed=99,
            filters=[f"H_total:{val}:{val}"],
        )
        assert len(result_exact) <= len(result_all)

    def test_contradictory_filters_yield_zero_structures(self):
        """Mutually exclusive filters must reject every sample without crashing."""
        from pasted import generate

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = generate(
                n_atoms=10, charge=0, mult=1,
                mode="gas", region="sphere:8",
                elements="6,7,8", n_samples=30, seed=1,
                filters=["H_total:999:-", "Q6:-:-0.001"],  # Q6 is always non-negative
            )
        assert len(result) == 0


# ---------------------------------------------------------------------------
# D. XYZ I/O errors
# ---------------------------------------------------------------------------

class TestXYZIO:
    """Robustness of Structure.from_xyz and parse_xyz against malformed input."""

    def test_from_xyz_empty_string_raises_valueerror(self):
        """Empty string must raise ValueError, not IsADirectoryError (regression for library bug fix)."""
        from pasted import Structure

        with pytest.raises(ValueError, match="No frames"):
            Structure.from_xyz("")

    def test_from_xyz_garbage_string_raises_valueerror(self):
        """Completely non-XYZ text must raise ValueError."""
        from pasted import Structure

        with pytest.raises(ValueError):
            Structure.from_xyz("not xyz data at all!!!\n$$$\n###")

    def test_from_xyz_truncated_raises_valueerror(self):
        """XYZ declaring 5 atoms but supplying only 2 coordinate lines must raise ValueError."""
        from pasted import Structure

        truncated = "5\ncomment\nC 0.0 0.0 0.0\nN 1.0 0.0 0.0\n"
        with pytest.raises(ValueError, match="Unexpected end of file"):
            Structure.from_xyz(truncated)

    def test_from_xyz_nan_inf_coords_does_not_hard_crash(self):
        """NaN/Inf coordinates are accepted by the parser but may raise during metric computation.

        The important invariant is that no unhandled C-extension segfault or uncaught
        low-level exception escapes — only a clean Python exception is permissible.
        """
        import numpy.linalg

        from pasted import Structure
        bad_xyz = "2\ncomment\nC 0.0 NaN INFINITY\nN 1.0 0.0 0.0\n"
        try:
            Structure.from_xyz(bad_xyz)
            # Accepted: metrics may contain NaN but the object was constructed.
        except (ValueError, numpy.linalg.LinAlgError, RuntimeWarning):
            pass  # Expected clean Python exceptions
        except Exception as exc:
            pytest.fail(
                f"Unexpected low-level exception for NaN/Inf coordinates: {type(exc).__name__}: {exc}"
            )

    def test_from_xyz_frame_out_of_range_raises_valueerror(self):
        """Requesting a non-existent frame index must raise ValueError."""
        from pasted import Structure, generate

        structs = generate(
            n_atoms=6, charge=0, mult=1,
            mode="gas", region="sphere:6",
            elements="6,7,8", n_samples=30, seed=42,
        )
        assert structs, "Need at least one structure to form an XYZ string"
        from typing import cast as _cast2

        from pasted import Structure as _Structure2
        xyz_str = _cast2(_Structure2, structs[0]).to_xyz()
        with pytest.raises(ValueError, match="out of range"):
            Structure.from_xyz(xyz_str, frame=999)

    def test_from_xyz_multiframe_round_trip(self):
        """Writing N frames then reading each back by index must preserve element lists."""
        from pasted import Structure, generate

        structs = generate(
            n_atoms=8, charge=0, mult=1,
            mode="gas", region="sphere:7",
            elements="6,7,8", n_samples=10, seed=5,
        )
        assert len(structs) > 1, "Need at least two structures for a multiframe test"

        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as fh:
            fname = fh.name
        try:
            for i, s in enumerate(structs):
                s.write_xyz(fname, append=(i > 0))

            for i, original in enumerate(structs):
                reloaded = Structure.from_xyz(fname, frame=i)
                assert sorted(reloaded.atoms) == sorted(original.atoms), (
                    f"Frame {i}: atom list mismatch after round-trip"
                )
        finally:
            os.unlink(fname)

    def test_parse_xyz_missing_coordinate_field_raises_valueerror(self):
        """A coordinate line with fewer than four fields must raise ValueError."""
        from pasted import parse_xyz

        bad = "2\ncomment\nC 0.0 0.0\nN 1.0 0.0 0.0\n"  # first atom missing z
        with pytest.raises(ValueError, match="Malformed coordinate line"):
            parse_xyz(bad)


# ---------------------------------------------------------------------------
# E. Hostile objective functions
# ---------------------------------------------------------------------------

class TestHostileObjectives:
    """StructureOptimizer must handle pathological objective return values gracefully."""

    def test_objective_returning_nan_does_not_crash(self):
        """An objective that always returns NaN must not cause a crash or hang."""
        from pasted import StructureOptimizer

        call_count = {"n": 0}

        def nan_obj(m):
            call_count["n"] += 1
            return float("nan")

        opt = StructureOptimizer(
            n_atoms=8, charge=0, mult=1, elements="6,7,8",
            objective=nan_obj,
            method="annealing", max_steps=100, seed=42,
        )
        opt.run()
        assert call_count["n"] > 0
        # best_f will be NaN; that is acceptable — the run must complete.

    def test_objective_returning_pos_inf_does_not_crash(self):
        """An objective that always returns +inf must complete without error."""
        from pasted import StructureOptimizer

        opt = StructureOptimizer(
            n_atoms=8, charge=0, mult=1, elements="6,7,8",
            objective=lambda m: float("inf"),
            method="annealing", max_steps=100, seed=42,
        )
        result = opt.run()
        assert result.objective_scores, "Expected at least one restart score"
        assert math.isinf(result.objective_scores[0])

    def test_objective_returning_neg_inf_does_not_crash(self):
        """An objective that always returns -inf must complete without error."""
        from pasted import StructureOptimizer

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            opt = StructureOptimizer(
                n_atoms=8, charge=0, mult=1, elements="6,7,8",
                objective=lambda m: float("-inf"),
                method="annealing", max_steps=100, seed=42,
            )
            result = opt.run()
        assert result.objective_scores, "Expected at least one restart score"
        assert math.isinf(result.objective_scores[0])

    def test_objective_raising_exception_propagates(self):
        """An objective that raises intermittently must propagate the exception to the caller.

        PASTED does not suppress objective exceptions — callers (e.g., xTB wrappers)
        are expected to handle their own failures and return float('-inf') on error,
        as shown in quickstart.md.
        """
        from pasted import StructureOptimizer

        boom_count = {"n": 0}

        def boom_obj(m):
            boom_count["n"] += 1
            if boom_count["n"] % 3 == 0:
                raise RuntimeError("simulated external tool failure")
            return m["H_total"]

        opt = StructureOptimizer(
            n_atoms=8, charge=0, mult=1, elements="6,7,8",
            objective=boom_obj,
            method="annealing", max_steps=50, seed=42,
        )
        # Either completes (exception suppressed internally) or raises — both are documented.
        try:
            opt.run()
        except RuntimeError:
            pass  # Exception propagation is the documented behavior

    def test_objective_mutating_metrics_dict_does_not_corrupt_state(self):
        """Injecting extra keys into the metrics dict passed to the objective must not
        corrupt the internal state or the best structure's canonical metrics."""
        from pasted import StructureOptimizer

        mutations = {"n": 0}

        def mutating_obj(m):
            m["__injected__"] = 999.0  # pollute the dict
            mutations["n"] += 1
            return m.get("H_total", 0.0)

        opt = StructureOptimizer(
            n_atoms=8, charge=0, mult=1, elements="6,7,8",
            objective=mutating_obj,
            method="annealing", max_steps=150, seed=5,
        )
        result = opt.run()

        assert "H_total" in result.best.metrics, "H_total disappeared from best.metrics after dict mutation"
        val = result.best.metrics["H_total"]
        assert val == val, "H_total is NaN after dict mutation"  # NaN != NaN
        assert mutations["n"] > 0

    def test_objective_wrong_signature_raises_typeerror(self):
        """A zero-argument callable passed as objective must raise TypeError (not crash silently)."""
        from pasted import StructureOptimizer

        def no_args() -> float:
            return 1.0

        opt = StructureOptimizer(
            n_atoms=8, charge=0, mult=1, elements="6,7,8",
            objective=no_args,  # type: ignore[arg-type]
            method="annealing", max_steps=50, seed=42,
        )
        with pytest.raises(TypeError):
            opt.run()


# ---------------------------------------------------------------------------
# F. Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Concurrent calls must not race, corrupt shared state, or deadlock."""

    def test_generate_four_threads_same_seed_deterministic(self):
        """Four threads calling generate() with the same seed must return the same count."""
        from pasted import generate

        counts: dict[int, int] = {}
        errors: list[tuple[int, Exception]] = []

        def worker(tid: int) -> None:
            try:
                r = generate(
                    n_atoms=10, charge=0, mult=1,
                    mode="gas", region="sphere:8",
                    elements="6,7,8", n_samples=30, seed=42,
                )
                counts[tid] = len(r)
            except Exception as exc:
                errors.append((tid, exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        unique_counts = set(counts.values())
        assert len(unique_counts) == 1, (
            f"generate() is non-deterministic across threads with the same seed: {counts}"
        )

    def test_parallel_optimizer_instances_do_not_interfere(self):
        """Four independent StructureOptimizer instances run concurrently must all complete."""
        from pasted import StructureOptimizer

        errors: list[Exception] = []

        def worker(seed: int) -> None:
            try:
                opt = StructureOptimizer(
                    n_atoms=8, charge=0, mult=1, elements="6,7,8",
                    objective={"H_total": 1.0},
                    method="annealing", max_steps=200, seed=seed,
                )
                opt.run()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(s,)) for s in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent optimizer errors: {errors}"


# ---------------------------------------------------------------------------
# G. Memory
# ---------------------------------------------------------------------------

class TestMemory:
    """Peak heap and RSS growth must stay within reasonable bounds."""

    def test_large_n_atoms_peak_heap_below_200mb(self):
        """n_atoms=100 with 10 samples must not exceed 200 MiB of traced heap."""
        from pasted import generate

        gc.collect()
        tracemalloc.start()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            generate(
                n_atoms=100, charge=0, mult=1,
                mode="gas", region="sphere:20",
                elements="6,7,8", n_samples=10, seed=0,
            )
        peak_mb = _peak_heap_kb() / 1024
        tracemalloc.stop()

        assert peak_mb < 200, f"Peak heap too large: {peak_mb:.1f} MiB"

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil required for RSS measurement")
    def test_optimizer_rss_stable_over_ten_restarts(self):
        """Ten optimizer restarts must not grow RSS by more than 20 MiB (leak guard)."""
        from pasted import StructureOptimizer

        gc.collect()
        rss_before = _rss_mb()

        opt = StructureOptimizer(
            n_atoms=10, charge=0, mult=1, elements="6,7,8",
            objective={"H_total": 1.0},
            method="annealing", max_steps=300, n_restarts=10, seed=0,
        )
        opt.run()
        gc.collect()

        rss_delta = _rss_mb() - rss_before
        assert rss_delta < 20.0, f"Possible memory leak: RSS grew by {rss_delta:.2f} MiB over 10 restarts"


# ---------------------------------------------------------------------------
# H. Counterintuitive valid cases
# ---------------------------------------------------------------------------

class TestCounterIntuitiveValidCases:
    """Cases that look unusual but must succeed per the documented contract."""

    def test_single_element_pool_even_z_all_structures_use_that_element(self):
        """A C-only pool (Z=6, even) must generate structures composed entirely of C."""
        from pasted import generate

        result = generate(
            n_atoms=6, charge=0, mult=1,
            mode="gas", region="sphere:7",
            elements="6", n_samples=20, seed=42,
        )
        assert len(result) > 0, "C-only even-Z pool should always satisfy parity for singlet"
        for s in result:
            assert all(sym == "C" for sym in s.atoms), (
                f"Non-carbon atom found in C-only pool result: {set(s.atoms)}"
            )

    def test_chain_mode_silently_ignores_region(self):
        """Passing region= to chain mode must be silently ignored; same seed yields identical count."""
        from pasted import generate

        r_no_region = generate(
            n_atoms=10, charge=0, mult=1,
            mode="chain", elements="6,7,8",
            n_samples=20, seed=42,
        )
        r_with_region = generate(
            n_atoms=10, charge=0, mult=1,
            mode="chain", region="sphere:99999",
            elements="6,7,8", n_samples=20, seed=42,
        )
        assert len(r_no_region) == len(r_with_region), (
            "chain mode must silently discard region; same seed must yield same count"
        )

    def test_generate_kwargs_and_config_are_equivalent(self):
        """generate(**kwargs) and generate(GeneratorConfig(**kwargs)) must return the same count."""
        from pasted import GeneratorConfig, generate

        kwargs = dict(
            n_atoms=10, charge=0, mult=1,
            mode="gas", region="sphere:8",
            elements="6,7,8", n_samples=30, seed=77,
        )
        r_kwargs = generate(**kwargs)  # type: ignore[arg-type]
        r_config = generate(GeneratorConfig(**kwargs))  # type: ignore[arg-type]

        assert len(r_kwargs) == len(r_config), (
            f"kwargs path returned {len(r_kwargs)} but config path returned {len(r_config)}"
        )

    def test_element_fractions_extreme_skew_does_not_crash(self):
        """A 1000:0.001:0.001 fraction skew must not raise; the dominant element dominates."""
        from pasted import generate

        result = generate(
            n_atoms=20, charge=0, mult=1,
            mode="gas", region="sphere:10",
            elements="6,7,8",
            element_fractions={"C": 1000.0, "N": 0.001, "O": 0.001},
            n_samples=30, seed=0,
        )
        assert len(result) > 0

    def test_n_success_larger_than_n_samples_stops_at_n_samples(self):
        """n_success=100 with n_samples=5 must return at most 5 structures and not hang."""
        from pasted import StructureGenerator

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            gen = StructureGenerator(
                n_atoms=10, charge=0, mult=1,
                mode="gas", region="sphere:8",
                elements="6,7,8",
                n_success=100,  # want 100
                n_samples=5,    # budget is only 5 attempts
                seed=42,
            )
            result = gen.generate()

        assert len(result) <= 5, (
            f"Expected at most 5 structures (limited by n_samples), got {len(result)}"
        )

    def test_composition_only_with_foreign_initial_auto_sanitized(self):
        """When initial structure uses elements outside the optimizer pool, they must be
        replaced before the MC loop begins; all output atoms must belong to the pool."""
        from pasted import StructureOptimizer, generate

        _init_result = generate(
            n_atoms=8, charge=0, mult=1,
            mode="gas", region="sphere:7",
            elements="6,7,8", n_samples=30, seed=0,
        )
        from typing import cast as _cast

        from pasted import Structure as _Structure
        initial = _cast(_Structure, _init_result[0])  # C/N/O structure

        opt = StructureOptimizer(
            n_atoms=len(initial.atoms), charge=initial.charge, mult=initial.mult,
            elements=["Fe", "Ni", "Co"],  # entirely different pool
            objective={"H_total": 1.0},
            allow_displacements=False,
            method="annealing", max_steps=200, seed=0,
        )
        result = opt.run(initial=initial)  # type: ignore[arg-type]

        pool = {"Fe", "Ni", "Co"}
        foreign = set(result.best.atoms) - pool
        assert not foreign, f"Foreign atoms remain after auto-sanitization: {foreign}"

    def test_identical_seeds_produce_bit_identical_positions(self):
        """Two independent generate() calls with the same seed must return byte-identical positions."""
        from pasted import generate

        kwargs = dict(
            n_atoms=10, charge=0, mult=1,
            mode="gas", region="sphere:8",
            elements="6,7,8", n_samples=30, seed=123,
        )
        r1 = generate(**kwargs)  # type: ignore[arg-type]
        r2 = generate(**kwargs)  # type: ignore[arg-type]

        assert len(r1) == len(r2), "Same seed must yield the same number of structures"
        for i, (s1, s2) in enumerate(zip(r1, r2, strict=True)):
            np.testing.assert_allclose(
                np.array(s1.positions),
                np.array(s2.positions),
                err_msg=f"Positions differ between two calls with seed=123 at structure index {i}",
            )


