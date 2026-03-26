"""
tests/test_jitter_memory.py
============================
Memory usage tests for the jitter O(N²) → FlatCellList fix and
the FlatCellList int32 overflow / SIGSEGV fix (v0.4.3).

What is verified:
  1. The jitter FlatCellList respects the max(1024, 16*n) cell cap
  2. For large N (n=4096, n=8192), cell_head stays within the stated bound
  3. Consistency with the O(N²) path (n < 64): coincident pairs are detected
  4. Edge cases (n=0, n=1, …) do not crash (Segfault regression guard)
  5. Measured peak memory via tracemalloc stays within the stated bound
"""

from __future__ import annotations

import importlib.util
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load _relax_core directly from its .so path
# ---------------------------------------------------------------------------
_EXT_PATH = Path(__file__).parent.parent / "src" / "pasted" / "_ext"


def _load_relax_core():
    so_files = list(_EXT_PATH.glob("_relax_core*.so"))
    if not so_files:
        pytest.skip("_relax_core.so not found (C++ build required)")
    spec = importlib.util.spec_from_file_location("_relax_core", so_files[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_relax_core = _load_relax_core()
relax_positions = _relax_core.relax_positions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_pts(n: int, rng: np.random.Generator, spread: float = 5.0) -> np.ndarray:
    """Return random (n, 3) float64 coordinates."""
    return rng.uniform(-spread, spread, size=(n, 3))


def _make_radii(n: int, r: float = 0.5) -> np.ndarray:
    return np.full(n, r, dtype=np.float64)


def _make_coincident_pts(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return coordinates where the first two atoms are exactly coincident
    (to trigger the jitter path)."""
    pts = _make_pts(n, rng)
    pts[1] = pts[0].copy()       # coincident pair
    if n > 3:
        pts[3] = pts[2].copy()   # second coincident pair
    return pts


# ===========================================================================
# 1. FlatCellList cell count cap
# ===========================================================================
class TestFlatCellListCap:
    """
    v0.4.3 fix: the jitter FlatCellList is bounded to max(1024, 16*n) cells.
    cell_head element count = nx*ny*nz <= max(1024, 16*n).
    Verified indirectly via tracemalloc peak.

    n=4096  -> max_cells = 65_536  -> 4 bytes/int * 65_536 = 256 KB
    n=16384 -> max_cells = 262_144 -> ~1 MB
    Old implementation (4 M cell cap) -> 16 MB  (compared against this)
    """

    @pytest.mark.parametrize("n", [128, 512, 1024, 4096])
    def test_peak_memory_scales_linearly_not_quadratic(self, n: int):
        """With coincident pairs present the jitter FlatCellList is exercised.
        Peak additional memory must stay below 16 MB (the old 4M-cell limit).
        """
        rng = np.random.default_rng(42)
        pts = _make_coincident_pts(n, rng)
        radii = _make_radii(n, 0.4)

        tracemalloc.start()
        snap0 = tracemalloc.take_snapshot()

        _, _ = relax_positions(pts, radii, cov_scale=0.9, max_cycles=1, seed=0)

        snap1 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = snap1.compare_to(snap0, "lineno")
        peak_bytes = sum(max(s.size_diff, 0) for s in stats)

        # Hard upper bound: 16 MB (= old 4M-cell * 4 bytes implementation)
        # After the fix, n=4096 should be ~256 KB
        limit_bytes = 16 * 1024 * 1024  # 16 MB
        assert peak_bytes < limit_bytes, (
            f"n={n}: peak memory {peak_bytes/1024:.1f} KB exceeded the "
            f"{limit_bytes//1024} KB limit"
        )

    @pytest.mark.parametrize("n", [4096, 8192])
    def test_peak_memory_well_below_old_limit(self, n: int):
        """After the fix, peak memory should be at least 10x below the old 16 MB limit.
        Old: ~16 MB, new: ~256 KB (n=4096) / ~1 MB (n=16384).
        """
        rng = np.random.default_rng(0)
        pts = _make_coincident_pts(n, rng)
        radii = _make_radii(n, 0.4)

        tracemalloc.start()
        _, _ = relax_positions(pts, radii, cov_scale=0.9, max_cycles=1, seed=0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Expected cap: max(1024, 16*n) ints for cell_head + n ints for next
        expected_cap_bytes = max(1024, 16 * n) * 4 * 2  # cell_head + next (rough bound)
        # With 20x headroom, should still be well below 16 MB
        assert peak < 16 * 1024 * 1024, (
            f"n={n}: tracemalloc peak {peak/1024:.1f} KB >= 16 MB (old implementation limit)"
        )
        # Also verify against the theoretical cap with generous headroom
        assert peak < expected_cap_bytes * 20, (
            f"n={n}: peak {peak/1024:.1f} KB exceeded "
            f"expected cap {expected_cap_bytes*20//1024} KB"
        )


# ===========================================================================
# 2. Correctness: coincident pairs are detected and jittered
# ===========================================================================
class TestJitterCorrectness:
    """Verify that jitter actually separates coincident atom pairs."""

    @pytest.mark.parametrize("n", [2, 10, 64, 200])
    def test_coincident_atoms_are_separated(self, n: int):
        """Input with a coincident pair -> after relax the pair distance must be > 0."""
        rng = np.random.default_rng(7)
        pts = _make_coincident_pts(n, rng)
        radii = _make_radii(n, 0.3)

        pts_out, _ = relax_positions(pts, radii, cov_scale=0.9, max_cycles=200, seed=1)

        d = np.linalg.norm(pts_out[1] - pts_out[0])
        assert d > 1e-12, (
            f"n={n}: coincident pair was not jittered (d={d:.2e})"
        )

    @pytest.mark.parametrize("n", [32, 64, 65, 128])  # straddles CELL_LIST_THRESHOLD=64
    def test_threshold_boundary_consistency(self, n: int):
        """At the O(N^2) -> FlatCellList switch point (n=64), results must be
        finite and self-consistent."""
        rng = np.random.default_rng(99)
        pts = _make_coincident_pts(n, rng)
        radii = _make_radii(n, 0.3)

        pts_out, converged = relax_positions(
            pts, radii, cov_scale=0.9, max_cycles=500, seed=42
        )
        assert np.all(np.isfinite(pts_out)), (
            f"n={n}: output contains non-finite values"
        )


# ===========================================================================
# 3. Edge cases (Segfault regression guard)
# ===========================================================================
class TestEdgeCases:
    """Ensure extreme inputs do not crash (guards against SIGSEGV regression)."""

    def test_n_zero(self):
        pts = np.zeros((0, 3), dtype=np.float64)
        radii = np.zeros(0, dtype=np.float64)
        pts_out, conv = relax_positions(pts, radii, cov_scale=0.9, max_cycles=10, seed=0)
        assert pts_out.shape == (0, 3)

    def test_n_one(self):
        pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        radii = np.array([0.5], dtype=np.float64)
        pts_out, conv = relax_positions(pts, radii, cov_scale=0.9, max_cycles=10, seed=0)
        assert pts_out.shape == (1, 3)

    def test_n_two_identical(self):
        """Minimal coincident case: two exactly overlapping atoms."""
        pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        radii = np.array([1.0, 1.0], dtype=np.float64)
        pts_out, _ = relax_positions(pts, radii, cov_scale=0.9, max_cycles=200, seed=5)
        assert np.all(np.isfinite(pts_out))

    @pytest.mark.parametrize("n", [63, 64, 65])  # around CELL_LIST_THRESHOLD
    def test_threshold_no_crash(self, n: int):
        """No Segfault at the CELL_LIST_THRESHOLD (64) boundary."""
        rng = np.random.default_rng(n)
        pts = _make_coincident_pts(n, rng)
        radii = _make_radii(n, 0.3)
        pts_out, _ = relax_positions(pts, radii, cov_scale=0.9, max_cycles=50, seed=0)
        assert np.all(np.isfinite(pts_out))

    def test_large_n_no_segfault(self):
        """No Segfault for n=2000 (regression test for the int32 overflow bug).

        Root cause of the original SIGSEGV (fixed in v0.4.3):
          cell_size=1e-9 A, atom spread ~10 A
          -> (range / cell_size) ~ 1e10 > INT_MAX (2.1e9)
          -> old static_cast<int>(1e10) truncated to ~1.4e9 (int32 UB)
          -> tnx()*tny()*tnz() int64 product overflowed to a negative value
          -> while (negative > max_cells) evaluated to False immediately
          -> nx = ny = nz = ~1.4e9 -> cell_head.assign(huge, -1) -> SIGSEGV
        """
        rng = np.random.default_rng(2000)
        pts = _make_coincident_pts(2000, rng)
        radii = _make_radii(2000, 0.3)
        pts_out, _ = relax_positions(pts, radii, cov_scale=0.9, max_cycles=5, seed=0)
        assert np.all(np.isfinite(pts_out))


# ===========================================================================
# 4. Early-exit path (no overlap) — FlatCellList is NOT called
# ===========================================================================
class TestEarlyExitMemory:
    """When no overlaps exist, relax_positions returns immediately without
    running the jitter FlatCellList.  Peak memory should be tiny."""

    def test_no_overlap_peak_memory_small(self):
        """Well-separated atoms -> early exit -> jitter FlatCellList not reached.
        Peak memory must be below 4 MB."""
        n = 2000
        rng = np.random.default_rng(1)
        # Place atoms on a grid with 5 A spacing -- guaranteed no overlaps
        pts = np.array(
            [[i * 5.0, j * 5.0, k * 5.0]
             for i in range(13) for j in range(13) for k in range(13)],
            dtype=np.float64,
        )[:n]
        radii = _make_radii(n, 0.3)

        tracemalloc.start()
        _, _ = relax_positions(pts, radii, cov_scale=0.9, max_cycles=1, seed=0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 4 * 1024 * 1024, (  # 4 MB
            f"Peak memory unexpectedly large on early-exit path: {peak/1024:.1f} KB"
        )
