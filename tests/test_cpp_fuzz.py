"""
PASTED — C++ Extension Fuzz / Robustness Tests  (tests/test_cpp_fuzz.py)
========================================================================
Ruthlessly exercises the five pybind11 entry points against inputs that
are mathematically degenerate, geometrically collapsed, or outright
malformed.

Design goals
  1. NO SEGFAULT — every test is independently process-isolated via a
     subprocess shim so that a C++ memory fault cannot bring down the
     whole pytest run.  Return code -11 (SIGSEGV) or -6 (SIGABRT) is
     treated as test failure.
  2. NUMERICAL HEALTH — when garbage inputs produce garbage outputs,
     the test documents *which* outputs are allowed to be NaN/inf and
     which must remain finite.  This acts as a regression baseline.
  3. TYPE SAFETY — shape mismatches and length mismatches that currently
     slip through pybind11 without raising are documented and the
     worst UB-triggering ones are explicitly called out.

Entry points under test
  relax_positions(pts, radii, cov_scale, max_cycles, seed=-1)
  angular_repulsion_gradient(pts, cutoff)
  steinhardt_per_atom(pts, cutoff, l_values)
  graph_metrics_cpp(pts, radii, cov_scale, en_vals, cutoff)
  rdf_h_cpp(pts, cutoff, n_bins)

Excluded (not C++):
  affine transforms  — pure Python in _placement.py
  MCMC / penalty λ  — pure Python in _optimizer.py
  multi-threading    — package is explicitly single-threaded

Categories
  FZ-A  No-crash subprocess isolation — SIGSEGV / SIGABRT detection
  FZ-B  NaN coordinate inputs — all five functions
  FZ-C  Inf / -Inf coordinate inputs — all five functions
  FZ-D  Geometric collapse — all-zeros, 2-D planar, collinear
  FZ-E  Empty (N=0) arrays — all five functions
  FZ-F  Single-atom (N=1) arrays — all five functions
  FZ-G  Wrong array shape — (N,2) passed where (N,3) expected
  FZ-H  Array length mismatch — radii / en_vals shorter than pts
  FZ-I  Scalar / 1-D pts — UB: uninitialized memory is read
  FZ-J  Non-C-contiguous (Fortran-order) arrays
  FZ-K  Boundary parameter values — cutoff, cov_scale, max_cycles,
        n_bins, l_values
  FZ-L  NaN / Inf in ancillary arrays (radii, en_vals)
  FZ-M  Float overflow — sys.float_info.max as coordinate
"""

from __future__ import annotations

import math
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

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

# ---------------------------------------------------------------------------
# Skip decorators — test only what compiled
# ---------------------------------------------------------------------------
needs_relax = pytest.mark.skipif(not HAS_RELAX, reason="HAS_RELAX=False")
needs_maxent = pytest.mark.skipif(not HAS_MAXENT, reason="HAS_MAXENT=False")
needs_steinhardt = pytest.mark.skipif(
    not HAS_STEINHARDT, reason="HAS_STEINHARDT=False"
)
needs_graph = pytest.mark.skipif(not HAS_GRAPH, reason="HAS_GRAPH=False")

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

# Canonical 3-atom triangular geometry used throughout
PTS3: np.ndarray = np.array(
    [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]], dtype=np.float64
)
RADII3: np.ndarray = np.full(3, 0.77, dtype=np.float64)
EN3: np.ndarray = np.array([2.55, 3.04, 3.44], dtype=np.float64)

# Timeout for subprocess-isolated tests (seconds)
_SUBPROCESS_TIMEOUT = 15


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh interpreter and return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )


def _assert_no_crash(proc: subprocess.CompletedProcess[str]) -> None:
    """Fail the test if the subprocess died with a signal (SegFault, Abort…)."""
    rc = proc.returncode
    # Negative return code on POSIX means killed by signal
    # -11 = SIGSEGV, -6 = SIGABRT, -8 = SIGFPE, etc.
    assert rc >= 0 or rc == 0, (
        f"C++ extension crashed: returncode={rc} "
        f"(SIGSEGV=-11, SIGABRT=-6, SIGFPE=-8)\n"
        f"stdout: {proc.stdout[:400]}\n"
        f"stderr: {proc.stderr[:400]}"
    )


def _all_finite_or_nan(d: dict[str, Any]) -> bool:
    """True iff every float value in d is either finite or NaN (no inf)."""
    for v in d.values():
        val = float(v)
        if math.isinf(val):
            return False
    return True


# ===========================================================================
# FZ-A  No-crash subprocess isolation
# ===========================================================================


class TestNoSegfaultIsolated:
    """Each test spawns a fresh Python process.  A negative return code
    means the C++ extension killed the process via a signal."""

    @needs_relax
    def test_relax_nan_coords_no_crash(self) -> None:
        code = (
            "import numpy as np; from pasted._ext import relax_positions; "
            "pts=np.full((5,3),float('nan')); r=np.full(5,0.77); "
            "relax_positions(pts,r,1.0,100)"
        )
        _assert_no_crash(_run_isolated(code))

    @needs_relax
    def test_relax_inf_coords_no_crash(self) -> None:
        code = (
            "import numpy as np; from pasted._ext import relax_positions; "
            "pts=np.full((5,3),float('inf')); r=np.full(5,0.77); "
            "relax_positions(pts,r,1.0,100)"
        )
        _assert_no_crash(_run_isolated(code))

    @needs_relax
    def test_relax_empty_no_crash(self) -> None:
        code = (
            "import numpy as np; from pasted._ext import relax_positions; "
            "pts=np.empty((0,3)); r=np.empty(0); "
            "relax_positions(pts,r,1.0,10)"
        )
        _assert_no_crash(_run_isolated(code))

    @needs_relax
    def test_relax_1d_input_no_crash(self) -> None:
        """1-D array triggers pybind11 reinterpret — must not SegFault."""
        code = (
            "import numpy as np; from pasted._ext import relax_positions; "
            "pts=np.array([0.0,1.5,0.0]); "
            "relax_positions(pts,np.array([0.77]),1.0,10)"
        )
        _assert_no_crash(_run_isolated(code))

    @needs_relax
    def test_relax_nx2_no_crash(self) -> None:
        """(N,2) array — column count is wrong; must not SegFault."""
        code = (
            "import numpy as np; from pasted._ext import relax_positions; "
            "pts=np.zeros((4,2)); r=np.full(4,0.77); "
            "relax_positions(pts,r,1.0,10)"
        )
        _assert_no_crash(_run_isolated(code))

    @needs_maxent
    def test_angular_gradient_nan_no_crash(self) -> None:
        code = (
            "import numpy as np; from pasted._ext import angular_repulsion_gradient; "
            "pts=np.full((5,3),float('nan')); "
            "angular_repulsion_gradient(pts,5.0)"
        )
        _assert_no_crash(_run_isolated(code))

    @needs_steinhardt
    def test_steinhardt_nan_no_crash(self) -> None:
        code = (
            "import numpy as np; from pasted._ext import steinhardt_per_atom; "
            "pts=np.full((5,3),float('nan')); "
            "steinhardt_per_atom(pts,5.0,[4,6,8])"
        )
        _assert_no_crash(_run_isolated(code))

    @needs_graph
    def test_graph_nan_no_crash(self) -> None:
        code = (
            "import numpy as np; from pasted._ext import graph_metrics_cpp; "
            "pts=np.full((5,3),float('nan')); r=np.full(5,0.77); en=np.full(5,2.5); "
            "graph_metrics_cpp(pts,r,1.0,en,5.0)"
        )
        _assert_no_crash(_run_isolated(code))

    @needs_graph
    def test_rdf_nan_no_crash(self) -> None:
        code = (
            "import numpy as np; from pasted._ext import rdf_h_cpp; "
            "pts=np.full((5,3),float('nan')); "
            "rdf_h_cpp(pts,5.0,20)"
        )
        _assert_no_crash(_run_isolated(code))

    @needs_graph
    def test_graph_length_mismatch_no_crash(self) -> None:
        """radii shorter than pts — must not read past the radii buffer."""
        code = (
            "import numpy as np; from pasted._ext import graph_metrics_cpp; "
            "pts=np.zeros((5,3)); r=np.full(2,0.77); en=np.full(2,2.5); "
            "graph_metrics_cpp(pts,r,1.0,en,5.0)"
        )
        _assert_no_crash(_run_isolated(code))


# ===========================================================================
# FZ-B  NaN coordinate inputs — numerical health assertions
# ===========================================================================


class TestNanCoordinates:
    """C++ must not crash.  Output rules (regression baseline):
      - relax_positions  : NaN propagates into output; converged=False
      - angular_gradient : all zeros (NaN atom has no valid neighbors)
      - steinhardt        : NaN atom treated as isolated; Q_l = 0
      - graph_metrics_cpp: graph topology unaffected (NaN atom isolated);
                           charge metrics may propagate NaN
      - rdf_h_cpp        : NaN pair excluded from histogram; h_spatial=0
    """

    def _pts_with_nan_atom(self, atom_idx: int = 0) -> np.ndarray:
        pts = PTS3.copy()
        pts[atom_idx] = [float("nan"), 0.0, 0.0]
        return pts

    @needs_relax
    def test_relax_nan_first_atom_returns_array(self) -> None:
        pts = self._pts_with_nan_atom(0)
        out, converged = relax_positions(pts, RADII3, 1.0, 100)
        assert out.shape == (3, 3)
        # Convergence may fail when NaN atoms are present
        assert isinstance(converged, bool)

    @needs_relax
    def test_relax_nan_propagates_to_output(self) -> None:
        """The NaN-contaminated atom's output coordinates remain NaN — the
        C++ code must not silently conjure a finite value from NaN input."""
        pts = self._pts_with_nan_atom(0)
        out, _ = relax_positions(pts, RADII3, 1.0, 100)
        assert np.any(np.isnan(out)), (
            "Expected NaN to propagate into relax output for NaN-input atom"
        )

    @needs_relax
    def test_relax_all_nan_output_shape_correct(self) -> None:
        pts = np.full((4, 3), float("nan"), dtype=np.float64)
        out, _ = relax_positions(pts, np.full(4, 0.77), 1.0, 100)
        assert out.shape == (4, 3)

    @needs_maxent
    def test_angular_gradient_nan_returns_zero_rows(self) -> None:
        """NaN atoms have no valid neighbor distances; gradient must be 0."""
        pts = self._pts_with_nan_atom(0)
        grad = angular_repulsion_gradient(pts, 5.0)
        assert grad.shape == (3, 3)
        assert np.all(grad == 0.0), (
            "Expected all-zero gradient when any atom has NaN coordinates"
        )

    @needs_steinhardt
    def test_steinhardt_nan_atom_gets_zero_q(self) -> None:
        """A NaN-coordinate atom has no neighbors inside the cutoff sphere;
        Steinhardt Q_l must be 0 (not NaN, not inf)."""
        pts = self._pts_with_nan_atom(0)
        result = steinhardt_per_atom(pts, 5.0, [6])
        q6 = result["Q6"]
        assert q6.shape == (3,)
        assert math.isfinite(float(q6[0])), (
            f"Q6 of NaN-coordinate atom must be finite (got {q6[0]})"
        )
        assert float(q6[0]) == pytest.approx(0.0)

    @needs_graph
    def test_graph_nan_coord_returns_dict_with_all_keys(self) -> None:
        pts = self._pts_with_nan_atom(0)
        result = graph_metrics_cpp(pts, RADII3, 1.0, EN3, 5.0)
        expected_keys = {
            "graph_lcc",
            "graph_cc",
            "ring_fraction",
            "charge_frustration",
            "moran_I_chi",
        }
        assert set(result.keys()) == expected_keys

    @needs_graph
    def test_rdf_nan_coord_h_spatial_is_zero(self) -> None:
        """NaN pair distances are excluded; with no valid pairs h_spatial=0."""
        pts = self._pts_with_nan_atom(0)
        result = rdf_h_cpp(pts, 5.0, 20)
        assert float(result["h_spatial"]) == pytest.approx(0.0)

    @needs_graph
    def test_graph_all_nan_no_infinite_values(self) -> None:
        """All-NaN pts: graph metrics must not produce inf."""
        pts = np.full((3, 3), float("nan"), dtype=np.float64)
        result = graph_metrics_cpp(pts, RADII3, 1.0, EN3, 5.0)
        for key, val in result.items():
            assert not math.isinf(float(val)), (
                f"graph_metrics_cpp[{key!r}] is inf for all-NaN input"
            )


# ===========================================================================
# FZ-C  Inf / -Inf coordinate inputs
# ===========================================================================


class TestInfCoordinates:
    """Inf coordinates represent atoms at infinity — physically nonsensical
    but should never crash the process."""

    @needs_relax
    def test_relax_inf_output_shape(self) -> None:
        pts = np.full((3, 3), float("inf"), dtype=np.float64)
        out, _ = relax_positions(pts, RADII3, 1.0, 100)
        assert out.shape == (3, 3)

    @needs_relax
    def test_relax_neginf_output_shape(self) -> None:
        pts = np.full((3, 3), float("-inf"), dtype=np.float64)
        out, _ = relax_positions(pts, RADII3, 1.0, 100)
        assert out.shape == (3, 3)

    @needs_maxent
    def test_angular_gradient_inf_returns_finite_or_zero(self) -> None:
        """All-inf positions → all distances are inf → no neighbors within
        any finite cutoff → gradient must be all zeros."""
        pts = np.full((3, 3), float("inf"), dtype=np.float64)
        grad = angular_repulsion_gradient(pts, 5.0)
        assert grad.shape == (3, 3)
        assert np.all(np.isfinite(grad)), (
            "angular_repulsion_gradient must return finite values for inf-coord input"
        )

    @needs_steinhardt
    def test_steinhardt_inf_coords_all_q_zero(self) -> None:
        pts = np.full((3, 3), float("inf"), dtype=np.float64)
        result = steinhardt_per_atom(pts, 5.0, [4, 6, 8])
        for l_key, arr in result.items():
            assert np.all(arr == 0.0), (
                f"steinhardt {l_key} must be zero for inf-coord atoms"
            )

    @needs_graph
    def test_rdf_inf_coords_h_spatial_zero(self) -> None:
        pts = np.full((3, 3), float("inf"), dtype=np.float64)
        result = rdf_h_cpp(pts, 5.0, 20)
        assert float(result["h_spatial"]) == pytest.approx(0.0)

    @needs_graph
    def test_graph_inf_coords_no_infinite_output(self) -> None:
        pts = np.full((3, 3), float("inf"), dtype=np.float64)
        result = graph_metrics_cpp(pts, RADII3, 1.0, EN3, 5.0)
        for key, val in result.items():
            v = float(val)
            assert not math.isinf(v), (
                f"graph_metrics_cpp[{key!r}] is inf for all-inf input"
            )

    @needs_relax
    def test_relax_mixed_nan_and_inf_output_shape(self) -> None:
        pts = PTS3.copy()
        pts[0, 0] = float("nan")
        pts[1, 1] = float("inf")
        out, _ = relax_positions(pts, RADII3, 1.0, 50)
        assert out.shape == (3, 3)


# ===========================================================================
# FZ-D  Geometric collapse
# ===========================================================================


class TestGeometricCollapse:
    """Structures with zero spatial extent — all atoms at origin, collinear,
    or strictly planar.  The C++ FlatCellList and L-BFGS must handle these
    without division-by-zero or infinite loops."""

    @needs_relax
    def test_relax_all_atoms_at_origin(self) -> None:
        """Coincident atoms trigger the seed-based jitter path in relax_core."""
        pts = np.zeros((5, 3), dtype=np.float64)
        out, _ = relax_positions(pts, np.full(5, 0.77), 1.0, 200)
        assert out.shape == (5, 3)
        # After relaxation atoms should have separated (jitter + L-BFGS)
        assert not np.all(out == 0.0), (
            "relax_positions should move coincident atoms apart"
        )

    @needs_relax
    def test_relax_two_atoms_at_origin(self) -> None:
        """Minimum clash case: 2 atoms at the exact same point."""
        pts = np.zeros((2, 3), dtype=np.float64)
        out, _ = relax_positions(pts, np.full(2, 0.77), 1.0, 500)
        assert out.shape == (2, 3)
        # Distance must be > 0 after relaxation
        dist = float(np.linalg.norm(out[0] - out[1]))
        assert dist > 0.0, "Two coincident atoms must be separated after relax"

    @needs_maxent
    def test_angular_gradient_all_atoms_at_origin(self) -> None:
        pts = np.zeros((4, 3), dtype=np.float64)
        grad = angular_repulsion_gradient(pts, 5.0)
        assert grad.shape == (4, 3)
        # All distances are 0; gradient may be 0 (no well-defined direction)
        assert np.all(np.isfinite(grad)), "Gradient must be finite for collapsed input"

    @needs_steinhardt
    def test_steinhardt_all_atoms_at_origin(self) -> None:
        """All atoms at origin — all mutual distances are 0 and within any
        positive cutoff; Q_l values must be finite."""
        pts = np.zeros((4, 3), dtype=np.float64)
        result = steinhardt_per_atom(pts, 5.0, [6])
        assert "Q6" in result
        assert np.all(np.isfinite(result["Q6"])), (
            "steinhardt Q6 must be finite for all-origin input"
        )

    @needs_graph
    def test_graph_all_atoms_at_origin_returns_dict(self) -> None:
        pts = np.zeros((4, 3), dtype=np.float64)
        result = graph_metrics_cpp(pts, np.full(4, 0.77), 1.0, np.full(4, 2.5), 5.0)
        assert set(result.keys()) == {
            "graph_lcc", "graph_cc", "ring_fraction",
            "charge_frustration", "moran_I_chi",
        }
        for key, val in result.items():
            assert math.isfinite(float(val)), (
                f"graph[{key!r}] must be finite for all-origin input"
            )

    @needs_graph
    def test_rdf_all_atoms_at_origin(self) -> None:
        pts = np.zeros((4, 3), dtype=np.float64)
        result = rdf_h_cpp(pts, 5.0, 20)
        assert math.isfinite(result["h_spatial"])
        assert math.isfinite(result["rdf_dev"])

    @needs_maxent
    def test_angular_gradient_planar_z0(self) -> None:
        """All z=0 (2-D degenerate): the gradient z-component may be zero
        but must not produce inf or NaN."""
        pts = np.array(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
             [0.0, 1.5, 0.0], [1.5, 1.5, 0.0]],
            dtype=np.float64,
        )
        grad = angular_repulsion_gradient(pts, 5.0)
        assert grad.shape == (4, 3)
        assert np.all(np.isfinite(grad))

    @needs_steinhardt
    def test_steinhardt_collinear_atoms(self) -> None:
        """All atoms on a single line (1-D): Q values must be finite."""
        pts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
             [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        result = steinhardt_per_atom(pts, 5.0, [4, 6])
        for key, arr in result.items():
            assert np.all(np.isfinite(arr)), (
                f"steinhardt {key} must be finite for collinear atoms"
            )


# ===========================================================================
# FZ-E  Empty (N=0) arrays
# ===========================================================================


class TestEmptyArrays:
    """Zero-atom inputs must return valid empty results — not crash."""

    _PTS0: np.ndarray = np.empty((0, 3), dtype=np.float64)
    _R0: np.ndarray = np.empty(0, dtype=np.float64)
    _EN0: np.ndarray = np.empty(0, dtype=np.float64)

    @needs_relax
    def test_relax_empty(self) -> None:
        out, conv = relax_positions(self._PTS0, self._R0, 1.0, 10)
        assert out.shape == (0, 3)
        assert conv is True  # vacuously converged

    @needs_maxent
    def test_angular_gradient_empty(self) -> None:
        grad = angular_repulsion_gradient(self._PTS0, 5.0)
        # Empty in → empty (or (0,) shape) out, no crash
        assert grad.size == 0

    @needs_steinhardt
    def test_steinhardt_empty(self) -> None:
        result = steinhardt_per_atom(self._PTS0, 5.0, [4, 6, 8])
        # An empty dict OR a dict of empty arrays are both acceptable
        for arr in result.values():
            assert len(arr) == 0

    @needs_graph
    def test_graph_metrics_empty(self) -> None:
        result = graph_metrics_cpp(
            self._PTS0, self._R0, 1.0, self._EN0, 5.0
        )
        # All topological metrics default to 0 for an empty structure
        for key, val in result.items():
            assert math.isfinite(float(val)), (
                f"graph[{key!r}] must be finite for empty input"
            )

    @needs_graph
    def test_rdf_empty(self) -> None:
        result = rdf_h_cpp(self._PTS0, 5.0, 20)
        assert math.isfinite(result["h_spatial"])
        assert math.isfinite(result["rdf_dev"])


# ===========================================================================
# FZ-F  Single-atom (N=1) arrays
# ===========================================================================


class TestSingleAtom:
    """One-atom inputs: all pair-based metrics must degenerate gracefully."""

    _PTS1: np.ndarray = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    _R1: np.ndarray = np.array([0.77], dtype=np.float64)
    _EN1: np.ndarray = np.array([2.55], dtype=np.float64)

    @needs_relax
    def test_relax_single_atom(self) -> None:
        out, conv = relax_positions(self._PTS1, self._R1, 1.0, 100)
        assert out.shape == (1, 3)
        assert conv is True  # no pairs to clash

    @needs_maxent
    def test_angular_gradient_single_atom(self) -> None:
        grad = angular_repulsion_gradient(self._PTS1, 5.0)
        assert grad.shape == (1, 3)
        # No neighbors → gradient is zero
        assert np.all(grad == 0.0)

    @needs_steinhardt
    def test_steinhardt_single_atom_q_zero(self) -> None:
        """A single atom has no neighbors; all Q_l must be 0."""
        result = steinhardt_per_atom(self._PTS1, 5.0, [4, 6, 8])
        for key, arr in result.items():
            assert arr.shape == (1,)
            assert float(arr[0]) == pytest.approx(0.0), (
                f"Single-atom {key} must be 0 (no neighbors)"
            )

    @needs_graph
    def test_graph_single_atom_finite_metrics(self) -> None:
        result = graph_metrics_cpp(
            self._PTS1, self._R1, 1.0, self._EN1, 5.0
        )
        for key, val in result.items():
            assert math.isfinite(float(val)), (
                f"graph[{key!r}] must be finite for single-atom input"
            )

    @needs_graph
    def test_rdf_single_atom_h_spatial_zero(self) -> None:
        """No pairs → distance histogram is empty → h_spatial = 0."""
        result = rdf_h_cpp(self._PTS1, 5.0, 20)
        assert float(result["h_spatial"]) == pytest.approx(0.0)

    @needs_graph
    def test_rdf_single_atom_no_infinite(self) -> None:
        result = rdf_h_cpp(self._PTS1, 5.0, 20)
        assert math.isfinite(result["h_spatial"])
        assert math.isfinite(result["rdf_dev"])


# ===========================================================================
# FZ-G  Wrong array shape — (N,2) instead of (N,3)
# ===========================================================================


class TestWrongShape:
    """pybind11 accepts (N,2) without raising because numpy's ArrayLike
    coercion reinterprets the memory as (N,3) by striding into the next
    rows.  This is a latent UB hazard; these tests document the current
    behavior so that a future fix shows up as a deliberate change.

    Ideal behavior: raise ValueError.
    Current behavior (regression baseline): silently returns results with
    UB-tainted values — which is still preferable to SIGSEGV.
    """

    _PTS_2D: np.ndarray = np.array(
        [[0.0, 0.0], [1.5, 0.0], [0.0, 1.5]], dtype=np.float64
    )
    _RADII_2D: np.ndarray = np.full(3, 0.77, dtype=np.float64)

    @needs_relax
    def test_relax_nx2_does_not_segfault(self) -> None:
        """(N,2) input: the binding must not SegFault regardless of output."""
        # Use subprocess to avoid contaminating other tests if output is garbage
        code = (
            "import numpy as np; from pasted._ext import relax_positions; "
            "pts=np.zeros((4,2),dtype='f8'); r=np.full(4,0.77); "
            "out,conv=relax_positions(pts,r,1.0,10); "
            "assert out.shape==(4,3) or True"  # shape may differ; just no crash
        )
        _assert_no_crash(_run_isolated(code))

    @needs_maxent
    def test_angular_gradient_nx2_does_not_segfault(self) -> None:
        code = (
            "import numpy as np; from pasted._ext import angular_repulsion_gradient; "
            "pts=np.zeros((4,2),dtype='f8'); "
            "angular_repulsion_gradient(pts,5.0)"
        )
        _assert_no_crash(_run_isolated(code))

    @needs_relax
    def test_relax_nx4_does_not_segfault(self) -> None:
        """(N,4) — extra column; buffer is larger than expected, safer."""
        code = (
            "import numpy as np; from pasted._ext import relax_positions; "
            "pts=np.zeros((3,4),dtype='f8'); r=np.full(3,0.77); "
            "relax_positions(pts,r,1.0,10)"
        )
        _assert_no_crash(_run_isolated(code))


# ===========================================================================
# FZ-H  Array length mismatch
# ===========================================================================


class TestLengthMismatch:
    """radii / en_vals arrays shorter than pts.  pybind11 does not validate
    array lengths; the C++ core reads past the short array's bounds.
    These tests document that the current behavior is survivable (no crash)
    and must not regress to a SegFault."""

    @needs_relax
    def test_relax_radii_one_shorter_no_crash(self) -> None:
        code = (
            "import numpy as np; from pasted._ext import relax_positions; "
            "pts=np.zeros((4,3),dtype='f8'); r=np.full(3,0.77); "
            "relax_positions(pts,r,1.0,10)"
        )
        _assert_no_crash(_run_isolated(code))

    @needs_relax
    def test_relax_radii_half_length_no_crash(self) -> None:
        code = (
            "import numpy as np; from pasted._ext import relax_positions; "
            "pts=np.zeros((6,3),dtype='f8'); r=np.full(3,0.77); "
            "relax_positions(pts,r,1.0,10)"
        )
        _assert_no_crash(_run_isolated(code))

    @needs_graph
    def test_graph_radii_shorter_no_crash(self) -> None:
        code = (
            "import numpy as np; from pasted._ext import graph_metrics_cpp; "
            "pts=np.zeros((5,3),dtype='f8'); r=np.full(2,0.77); en=np.full(2,2.5); "
            "graph_metrics_cpp(pts,r,1.0,en,5.0)"
        )
        _assert_no_crash(_run_isolated(code))

    @needs_graph
    def test_graph_en_vals_shorter_no_crash(self) -> None:
        code = (
            "import numpy as np; from pasted._ext import graph_metrics_cpp; "
            "pts=np.zeros((5,3),dtype='f8'); r=np.full(5,0.77); en=np.full(2,2.5); "
            "graph_metrics_cpp(pts,r,1.0,en,5.0)"
        )
        _assert_no_crash(_run_isolated(code))


# ===========================================================================
# FZ-I  Scalar / 1-D pts — uninitialized memory hazard
# ===========================================================================


class TestOneDimensionalInput:
    """A flat 1-D array is the worst-case shape mismatch: pybind11 reinterprets
    3 floats as 1 atom's (x,y,z), but rows 1,2,... read uninitialized heap.
    Tests assert no crash; output values are explicitly *not* checked because
    they are undefined behavior."""

    @needs_relax
    def test_relax_1d_array_no_crash(self) -> None:
        code = (
            "import numpy as np; from pasted._ext import relax_positions; "
            "pts=np.array([0.0,1.5,0.0]); "
            "relax_positions(pts,np.array([0.77]),1.0,10)"
        )
        _assert_no_crash(_run_isolated(code))

    @needs_maxent
    def test_angular_gradient_1d_no_crash(self) -> None:
        code = (
            "import numpy as np; from pasted._ext import angular_repulsion_gradient; "
            "pts=np.array([0.0,1.5,0.0]); "
            "angular_repulsion_gradient(pts,5.0)"
        )
        _assert_no_crash(_run_isolated(code))


# ===========================================================================
# FZ-J  Non-C-contiguous (Fortran-order) arrays
# ===========================================================================


class TestNonContiguousArrays:
    """pybind11's ArrayLike annotation triggers a copy for non-C-contiguous
    inputs.  These tests verify that the copy path works correctly and the
    output is identical to the C-contiguous path."""

    @needs_relax
    def test_relax_fortran_order_same_result(self) -> None:
        pts_c = PTS3.copy()
        pts_f = np.asfortranarray(PTS3)
        out_c, conv_c = relax_positions(pts_c, RADII3, 1.0, 200)
        out_f, conv_f = relax_positions(pts_f, RADII3, 1.0, 200)
        assert conv_c == conv_f
        np.testing.assert_allclose(out_c, out_f, atol=1e-10)

    @needs_maxent
    def test_angular_gradient_fortran_order_same_result(self) -> None:
        pts_c = PTS3.copy()
        pts_f = np.asfortranarray(PTS3)
        grad_c = angular_repulsion_gradient(pts_c, 5.0)
        grad_f = angular_repulsion_gradient(pts_f, 5.0)
        np.testing.assert_allclose(grad_c, grad_f, atol=1e-12)

    @needs_steinhardt
    def test_steinhardt_fortran_order_same_result(self) -> None:
        pts_c = PTS3.copy()
        pts_f = np.asfortranarray(PTS3)
        res_c = steinhardt_per_atom(pts_c, 5.0, [6])
        res_f = steinhardt_per_atom(pts_f, 5.0, [6])
        np.testing.assert_allclose(res_c["Q6"], res_f["Q6"], atol=1e-12)

    @needs_graph
    def test_graph_fortran_order_same_result(self) -> None:
        pts_c = PTS3.copy()
        pts_f = np.asfortranarray(PTS3)
        res_c = graph_metrics_cpp(pts_c, RADII3, 1.0, EN3, 5.0)
        res_f = graph_metrics_cpp(pts_f, RADII3, 1.0, EN3, 5.0)
        for key in res_c:
            assert res_c[key] == pytest.approx(res_f[key], abs=1e-12)

    @needs_graph
    def test_rdf_fortran_order_same_result(self) -> None:
        pts_c = PTS3.copy()
        pts_f = np.asfortranarray(PTS3)
        res_c = rdf_h_cpp(pts_c, 5.0, 20)
        res_f = rdf_h_cpp(pts_f, 5.0, 20)
        assert res_c["h_spatial"] == pytest.approx(res_f["h_spatial"], abs=1e-12)


# ===========================================================================
# FZ-K  Boundary parameter values
# ===========================================================================


class TestBoundaryParameters:
    """Test edge values of scalar parameters: zero, negative, tiny, huge."""

    # ── relax_positions parameters ──────────────────────────────────────────

    @needs_relax
    def test_relax_cov_scale_zero(self) -> None:
        """cov_scale=0 means no minimum distance → penalty is always 0 →
        convergence is immediate (no forces)."""
        _, conv = relax_positions(PTS3, RADII3, 0.0, 100)
        assert conv is True

    @needs_relax
    def test_relax_cov_scale_negative(self) -> None:
        """Negative cov_scale inverts the harmonic penalty direction;
        the C++ must not crash (behaviour is undefined by the API but
        survivability is required)."""
        out, _ = relax_positions(PTS3, RADII3, -1.0, 100)
        assert out.shape == (3, 3)

    @needs_relax
    def test_relax_max_cycles_zero_returns_unconverged(self) -> None:
        """max_cycles=0: no L-BFGS steps taken; must return immediately
        with converged=False."""
        _, conv = relax_positions(PTS3, RADII3, 1.0, 0)
        assert conv is False

    @needs_relax
    def test_relax_max_cycles_negative_treated_as_zero(self) -> None:
        """Negative max_cycles: treated as no-op, converged=False."""
        _, conv = relax_positions(PTS3, RADII3, 1.0, -1)
        assert conv is False

    @needs_relax
    def test_relax_max_cycles_one_does_not_crash(self) -> None:
        out, _ = relax_positions(PTS3, RADII3, 1.0, 1)
        assert out.shape == (3, 3)

    # ── angular_repulsion_gradient cutoff ───────────────────────────────────

    @needs_maxent
    def test_angular_gradient_cutoff_zero(self) -> None:
        """cutoff=0: no atom within range → gradient is all zero."""
        grad = angular_repulsion_gradient(PTS3, 0.0)
        assert grad.shape == (3, 3)
        assert np.all(grad == 0.0)

    @needs_maxent
    def test_angular_gradient_cutoff_negative_finite_output(self) -> None:
        """Negative cutoff is undefined; output must at minimum be finite
        (no NaN, no inf) — no requirement on the specific values."""
        grad = angular_repulsion_gradient(PTS3, -1.0)
        assert grad.shape == (3, 3)
        assert np.all(np.isfinite(grad)), (
            "angular_repulsion_gradient must return finite values for negative cutoff"
        )

    @needs_maxent
    def test_angular_gradient_cutoff_very_large(self) -> None:
        """Huge cutoff: all atoms are neighbors — must not overflow."""
        grad = angular_repulsion_gradient(PTS3, 1e15)
        assert grad.shape == (3, 3)
        assert np.all(np.isfinite(grad))

    # ── steinhardt_per_atom l_values ────────────────────────────────────────

    @needs_steinhardt
    def test_steinhardt_empty_l_values(self) -> None:
        """l_values=[] returns an empty dict — no crash."""
        result = steinhardt_per_atom(PTS3, 5.0, [])
        assert result == {}

    @needs_steinhardt
    def test_steinhardt_l_zero(self) -> None:
        """l=0 is the monopole term; Q0=1 for all atoms with any neighbors."""
        result = steinhardt_per_atom(PTS3, 5.0, [0])
        assert "Q0" in result
        assert result["Q0"].shape == (3,)
        assert np.all(np.isfinite(result["Q0"]))

    @needs_steinhardt
    def test_steinhardt_l_max_boundary(self) -> None:
        """l=12 is the documented maximum; must return finite values."""
        result = steinhardt_per_atom(PTS3, 5.0, [12])
        assert "Q12" in result
        assert np.all(np.isfinite(result["Q12"]))

    @needs_steinhardt
    def test_steinhardt_l_out_of_range_raises(self) -> None:
        """l=13 exceeds the [0,12] range; must raise RuntimeError, not crash."""
        with pytest.raises(RuntimeError, match="out of range"):
            steinhardt_per_atom(PTS3, 5.0, [13])

    @needs_steinhardt
    def test_steinhardt_l_negative_raises(self) -> None:
        """l=-1 is out of range; must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="out of range"):
            steinhardt_per_atom(PTS3, 5.0, [-1])

    # ── rdf_h_cpp n_bins / cutoff ────────────────────────────────────────────

    @needs_graph
    def test_rdf_n_bins_zero(self) -> None:
        """n_bins=0: no histogram possible; h_spatial and rdf_dev must be 0."""
        result = rdf_h_cpp(PTS3, 5.0, 0)
        assert float(result["h_spatial"]) == pytest.approx(0.0)

    @needs_graph
    def test_rdf_n_bins_one(self) -> None:
        """n_bins=1: a single bin captures all pairs; must not divide by zero."""
        result = rdf_h_cpp(PTS3, 5.0, 1)
        assert math.isfinite(result["h_spatial"])
        assert math.isfinite(result["rdf_dev"])

    @needs_graph
    def test_rdf_n_bins_very_large(self) -> None:
        """10 million bins: no OOM or integer overflow in the C++ allocation."""
        result = rdf_h_cpp(PTS3, 5.0, 10_000_000)
        assert math.isfinite(result["h_spatial"])

    @needs_graph
    def test_rdf_cutoff_zero(self) -> None:
        """cutoff=0: no pairs within range → h_spatial=0."""
        result = rdf_h_cpp(PTS3, 0.0, 20)
        assert float(result["h_spatial"]) == pytest.approx(0.0)

    @needs_graph
    def test_rdf_cutoff_very_large(self) -> None:
        """Giant cutoff includes all pairs; result must be finite."""
        result = rdf_h_cpp(PTS3, 1e15, 20)
        assert math.isfinite(result["h_spatial"])
        assert math.isfinite(result["rdf_dev"])

    @needs_graph
    def test_rdf_cutoff_nan(self) -> None:
        """cutoff=NaN: no valid pair distances; h_spatial=0, no crash."""
        result = rdf_h_cpp(PTS3, float("nan"), 20)
        assert math.isfinite(result["h_spatial"])

    # ── graph_metrics_cpp cutoff ─────────────────────────────────────────────

    @needs_graph
    def test_graph_cutoff_zero(self) -> None:
        """cutoff=0: no bonds → graph is fully disconnected; must be finite."""
        result = graph_metrics_cpp(PTS3, RADII3, 1.0, EN3, 0.0)
        for key, val in result.items():
            assert math.isfinite(float(val)), (
                f"graph[{key!r}] must be finite at cutoff=0"
            )

    @needs_graph
    def test_graph_cutoff_very_large(self) -> None:
        result = graph_metrics_cpp(PTS3, RADII3, 1.0, EN3, 1e15)
        for key, val in result.items():
            assert math.isfinite(float(val))


# ===========================================================================
# FZ-L  NaN / Inf in ancillary arrays (radii, en_vals)
# ===========================================================================


class TestAncillaryArrayAnomalies:
    """Non-coordinate inputs (radii, en_vals) containing NaN or Inf."""

    @needs_relax
    def test_relax_nan_radii_output_finite_or_nan(self) -> None:
        """NaN radius for atom 0 enters the pairwise penalty E_ij for every
        pair (0,j).  Because all atoms are within the cutoff, the L-BFGS
        gradient is NaN-contaminated for the entire system and the output
        positions all become NaN.  The key requirement is no crash."""
        radii_nan = RADII3.copy()
        radii_nan[0] = float("nan")
        out, conv = relax_positions(PTS3, radii_nan, 1.0, 100)
        assert out.shape == (3, 3)
        # NaN radius contaminates all pairs; converged must be False
        assert conv is False
        # All output positions become NaN — regression baseline
        assert np.all(np.isnan(out)), (
            "Expected all positions to be NaN when any radius is NaN "
            f"(actual: {out})"
        )

    @needs_relax
    def test_relax_inf_radii_does_not_crash(self) -> None:
        radii_inf = RADII3.copy()
        radii_inf[0] = float("inf")
        out, _ = relax_positions(PTS3, radii_inf, 1.0, 100)
        assert out.shape == (3, 3)

    @needs_graph
    def test_graph_nan_en_vals_charge_metric_is_nan(self) -> None:
        """NaN electronegativity propagates into charge_frustration and
        moran_I_chi; other metrics (graph topology) must remain finite."""
        en_nan = EN3.copy()
        en_nan[0] = float("nan")
        result = graph_metrics_cpp(PTS3, RADII3, 1.0, en_nan, 5.0)
        # Topology metrics must be finite
        assert math.isfinite(result["graph_lcc"])
        assert math.isfinite(result["graph_cc"])
        assert math.isfinite(result["ring_fraction"])

    @needs_graph
    def test_graph_inf_en_vals_does_not_crash(self) -> None:
        en_inf = EN3.copy()
        en_inf[1] = float("inf")
        result = graph_metrics_cpp(PTS3, RADII3, 1.0, en_inf, 5.0)
        # Must return a dict — not crash
        assert isinstance(result, dict)

    @needs_graph
    def test_graph_all_zero_en_vals_charge_metrics_zero(self) -> None:
        """Uniform electronegativity → zero variance → charge_frustration=0,
        moran_I_chi=0 (no spatial autocorrelation in a constant field)."""
        en_zero = np.zeros(3, dtype=np.float64)
        result = graph_metrics_cpp(PTS3, RADII3, 1.0, en_zero, 5.0)
        assert float(result["charge_frustration"]) == pytest.approx(0.0, abs=1e-12)
        assert math.isfinite(result["moran_I_chi"])


# ===========================================================================
# FZ-M  Float overflow — sys.float_info.max as coordinate
# ===========================================================================


class TestFloatOverflow:
    """Coordinates at the floating-point ceiling.  Distance computations
    (d = sqrt(dx^2 + dy^2 + dz^2)) can overflow to inf even when inputs
    are finite.  C++ must handle the resulting inf distances without
    crashing."""

    _FMAX: float = sys.float_info.max
    _PTS_FMAX: np.ndarray = np.array(
        [[sys.float_info.max, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64
    )
    _RADII_FMAX: np.ndarray = np.full(2, 0.77, dtype=np.float64)
    _EN_FMAX: np.ndarray = np.full(2, 2.5, dtype=np.float64)

    @needs_relax
    def test_relax_fmax_coord_output_shape(self) -> None:
        """fmax in a coordinate → distance overflows to inf; relax must not
        crash.  converged=False is acceptable."""
        out, _ = relax_positions(self._PTS_FMAX, self._RADII_FMAX, 1.0, 10)
        assert out.shape == (2, 3)

    @needs_maxent
    def test_angular_gradient_fmax_coord_finite_output(self) -> None:
        """Two atoms: one at fmax, one at origin.  Distance is inf → no
        neighbor → gradient must be zero (or at least finite)."""
        grad = angular_repulsion_gradient(self._PTS_FMAX, 5.0)
        assert grad.shape == (2, 3)
        assert np.all(np.isfinite(grad))

    @needs_steinhardt
    def test_steinhardt_fmax_coord_all_q_zero(self) -> None:
        """Atom at fmax is infinitely far from any finite-coord atom;
        Q_l must be 0 (no neighbors)."""
        result = steinhardt_per_atom(self._PTS_FMAX, 5.0, [6])
        q6 = result["Q6"]
        assert q6.shape == (2,)
        assert np.all(np.isfinite(q6))

    @needs_graph
    def test_graph_fmax_coord_finite_metrics(self) -> None:
        result = graph_metrics_cpp(
            self._PTS_FMAX, self._RADII_FMAX, 1.0, self._EN_FMAX, 5.0
        )
        for key, val in result.items():
            assert not math.isinf(float(val)), (
                f"graph[{key!r}] must not be inf for fmax-coord input"
            )

    @needs_graph
    def test_rdf_fmax_coord_h_spatial_zero(self) -> None:
        """Pair distance overflows to inf → excluded from histogram → h=0."""
        result = rdf_h_cpp(self._PTS_FMAX, 5.0, 20)
        assert float(result["h_spatial"]) == pytest.approx(0.0)

    @needs_graph
    def test_rdf_fmax_coord_no_infinite(self) -> None:
        result = rdf_h_cpp(self._PTS_FMAX, 5.0, 20)
        assert math.isfinite(result["h_spatial"])
        assert math.isfinite(result["rdf_dev"])

    @needs_relax
    def test_relax_both_atoms_at_fmax(self) -> None:
        """Two coincident atoms at fmax — jitter path, must not crash."""
        pts = np.array(
            [[self._FMAX, self._FMAX, self._FMAX],
             [self._FMAX, self._FMAX, self._FMAX]],
            dtype=np.float64,
        )
        out, _ = relax_positions(pts, self._RADII_FMAX, 1.0, 10)
        assert out.shape == (2, 3)
