"""
PASTED — C++ Extension Fuzz / Robustness Tests  (tests/test_cpp_fuzz.py)
========================================================================
Aggressively exercises the five pybind11 entry points against inputs that
are mathematically degenerate, geometrically collapsed, or outright
malformed.

Design goals
  1. NO SEGFAULT — every class that passes malformed memory (wrong shape,
     length mismatch, 1-D reinterpret) is subprocess-isolated.  A negative
     return code means a signal killed the process; the test fails.
  2. HONEST OUTPUT VERIFICATION — every expected output value is derived
     from a concrete probe run and documented with a reason.  We do not
     soften assertions to make tests pass.
  3. NO INTENTIONAL FAILURES — tests only cover inputs we send; we do not
     assert that the library *should* raise when it currently does not.

Entry points under test (all in pasted._ext)
  relax_positions(pts, radii, cov_scale, max_cycles, seed=-1)
  angular_repulsion_gradient(pts, cutoff)
  steinhardt_per_atom(pts, cutoff, l_values)
  graph_metrics_cpp(pts, radii, cov_scale, en_vals, cutoff)
  rdf_h_cpp(pts, cutoff, n_bins)

Excluded (not C++)
  affine transforms  — pure Python in _placement.py
  MCMC / penalty λ  — pure Python in _optimizer.py
  multi-threading    — package is explicitly single-threaded

Categories
  FZ-A  Subprocess SIGSEGV isolation
          · (N,2) wrong shape
          · (N,) 1-D reinterpret
          · radii / en_vals shorter than pts
  FZ-B  NaN coordinate inputs
  FZ-C  Inf / -Inf coordinate inputs
  FZ-D  Geometric collapse (all-origin, collinear, planar)
  FZ-E  Empty (N=0) arrays
  FZ-F  Single-atom (N=1) arrays
  FZ-G  Non-C-contiguous (Fortran-order) arrays
  FZ-H  Boundary scalar parameters
          · cov_scale: 0, -1
          · max_cycles: 0, -1, 1
          · cutoff: 0, -1, 1e15, NaN
          · n_bins: 0, -1, 1, 10 000 000
          · l_values: [], [0], [12], [-1]→raises, [13]→raises
  FZ-I  NaN / Inf in ancillary arrays (radii, en_vals)
  FZ-J  Float overflow — sys.float_info.max as coordinate
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
# Skip decorators
# ---------------------------------------------------------------------------
needs_relax = pytest.mark.skipif(not HAS_RELAX, reason="HAS_RELAX=False")
needs_maxent = pytest.mark.skipif(not HAS_MAXENT, reason="HAS_MAXENT=False")
needs_steinhardt = pytest.mark.skipif(not HAS_STEINHARDT, reason="HAS_STEINHARDT=False")
needs_graph = pytest.mark.skipif(not HAS_GRAPH, reason="HAS_GRAPH=False")

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Canonical 3-atom right-angle triangle
PTS3: np.ndarray = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]], dtype=np.float64)
RADII3: np.ndarray = np.full(3, 0.77, dtype=np.float64)
EN3: np.ndarray = np.array([2.55, 3.04, 3.44], dtype=np.float64)

_SUBPROCESS_TIMEOUT = 15


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )


def _assert_no_signal(proc: subprocess.CompletedProcess[str]) -> None:
    """Fail if the subprocess was killed by a signal (SIGSEGV, SIGABRT …).
    On POSIX, subprocess.returncode is negative when terminated by signal."""
    rc = proc.returncode
    assert rc >= 0, (
        f"C++ extension crashed with signal (returncode={rc}; "
        f"SIGSEGV=-11, SIGABRT=-6, SIGFPE=-8)\n"
        f"stdout: {proc.stdout[:400]}\n"
        f"stderr: {proc.stderr[:400]}"
    )


_GRAPH_KEYS = frozenset(
    {"graph_lcc", "graph_cc", "ring_fraction", "charge_frustration", "moran_I_chi"}
)
_RDF_KEYS = frozenset({"h_spatial", "rdf_dev"})

# ---------------------------------------------------------------------------
# Type-contract helpers
# Called in EVERY test that receives output from a C++ function.
# The C++ binding must honour its declared signature even for anomalous input.
# ---------------------------------------------------------------------------


def _assert_relax_contract(result: Any, n: int) -> tuple[np.ndarray, bool]:
    """relax_positions → tuple[NDArray[float64, (N,3)], bool]"""
    assert isinstance(result, tuple), f"Expected tuple, got {type(result).__name__}"
    assert len(result) == 2, f"Expected 2-tuple, got length {len(result)}"
    out, conv = result
    assert isinstance(out, np.ndarray), f"out: expected ndarray, got {type(out).__name__}"
    assert out.dtype == np.float64, f"out.dtype: expected float64, got {out.dtype}"
    assert out.ndim == 2, f"out.ndim: expected 2, got {out.ndim}"
    assert out.shape[0] == n, f"out.shape[0]: expected {n}, got {out.shape[0]}"
    assert out.shape[1] == 3, f"out.shape[1]: expected 3, got {out.shape[1]}"
    assert isinstance(conv, bool), f"conv: expected bool, got {type(conv).__name__}"
    return out, conv


def _assert_grad_contract(result: Any, n: int) -> np.ndarray:
    """angular_repulsion_gradient → NDArray[float64, (N,3)]"""
    assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result).__name__}"
    assert result.dtype == np.float64, f"dtype: expected float64, got {result.dtype}"
    assert result.ndim == 2, f"ndim: expected 2, got {result.ndim}"
    assert result.shape[0] == n, f"shape[0]: expected {n}, got {result.shape[0]}"
    assert result.shape[1] == 3, f"shape[1]: expected 3, got {result.shape[1]}"
    return result


def _assert_steinhardt_contract(result: Any, n: int) -> dict[str, np.ndarray]:
    """steinhardt_per_atom → dict[str, NDArray[float64, (N,)]]"""
    assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}"
    for k, v in result.items():
        assert isinstance(k, str), f"Key: expected str, got {type(k).__name__}"
        assert isinstance(v, np.ndarray), f"{k!r}: expected ndarray, got {type(v).__name__}"
        assert v.dtype == np.float64, f"{k!r}.dtype: expected float64, got {v.dtype}"
        assert v.ndim == 1, f"{k!r}.ndim: expected 1, got {v.ndim}"
        assert v.shape[0] == n, f"{k!r}.shape[0]: expected {n}, got {v.shape[0]}"
    return result


def _assert_graph_contract(result: Any) -> dict[str, float]:
    """graph_metrics_cpp → dict[str, float] with exactly 5 keys"""
    assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}"
    assert set(result.keys()) == _GRAPH_KEYS, (
        f"Keys mismatch: {set(result.keys())} != {_GRAPH_KEYS}"
    )
    for k, v in result.items():
        assert isinstance(v, float), f"{k!r}: expected float, got {type(v).__name__}"
    return result


def _assert_rdf_contract(result: Any) -> dict[str, float]:
    """rdf_h_cpp → dict[str, float] with exactly 2 keys"""
    assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}"
    assert set(result.keys()) == _RDF_KEYS, f"Keys mismatch: {set(result.keys())} != {_RDF_KEYS}"
    for k, v in result.items():
        assert isinstance(v, float), f"{k!r}: expected float, got {type(v).__name__}"
    return result


# ===========================================================================
# FZ-A  Subprocess SIGSEGV isolation
# ===========================================================================
# These inputs put the C++ core into undefined-behaviour territory because
# pybind11 does not validate array shapes or lengths before passing pointers.
# We can only assert "no fatal signal" — output values are meaningless UB.
# ===========================================================================


class TestSubprocessIsolation:
    """Inputs that trigger UB in the C++ layer.  Verified by spawning an
    isolated subprocess and checking that it exited cleanly (rc >= 0)."""

    @needs_relax
    def test_relax_nx2_no_signal(self) -> None:
        """(N,2) array: pybind11 reinterprets as (floor(2N/3), 3)."""
        _assert_no_signal(
            _run_isolated(
                "import numpy as np; from pasted._ext import relax_positions; "
                "pts=np.zeros((4,2),dtype='f8'); r=np.full(4,0.77); "
                "relax_positions(pts,r,1.0,10)"
            )
        )

    @needs_relax
    def test_relax_nx4_no_signal(self) -> None:
        """(N,4) array: buffer is wider than expected."""
        _assert_no_signal(
            _run_isolated(
                "import numpy as np; from pasted._ext import relax_positions; "
                "pts=np.zeros((3,4),dtype='f8'); r=np.full(3,0.77); "
                "relax_positions(pts,r,1.0,10)"
            )
        )

    @needs_maxent
    def test_angular_gradient_nx2_no_signal(self) -> None:
        _assert_no_signal(
            _run_isolated(
                "import numpy as np; from pasted._ext import angular_repulsion_gradient; "
                "pts=np.zeros((4,2),dtype='f8'); "
                "angular_repulsion_gradient(pts,5.0)"
            )
        )

    @needs_relax
    def test_relax_1d_array_no_signal(self) -> None:
        """1-D array: C++ reads uninitialized rows beyond the array bounds."""
        _assert_no_signal(
            _run_isolated(
                "import numpy as np; from pasted._ext import relax_positions; "
                "pts=np.array([0.0,1.5,0.0]); "
                "relax_positions(pts,np.array([0.77]),1.0,10)"
            )
        )

    @needs_maxent
    def test_angular_gradient_1d_no_signal(self) -> None:
        _assert_no_signal(
            _run_isolated(
                "import numpy as np; from pasted._ext import angular_repulsion_gradient; "
                "pts=np.array([0.0,1.5,0.0]); "
                "angular_repulsion_gradient(pts,5.0)"
            )
        )

    @needs_relax
    def test_relax_radii_shorter_than_pts_no_signal(self) -> None:
        """radii length < pts rows: C++ reads past the radii buffer."""
        _assert_no_signal(
            _run_isolated(
                "import numpy as np; from pasted._ext import relax_positions; "
                "pts=np.zeros((6,3),dtype='f8'); r=np.full(3,0.77); "
                "relax_positions(pts,r,1.0,10)"
            )
        )

    @needs_graph
    def test_graph_radii_shorter_than_pts_no_signal(self) -> None:
        _assert_no_signal(
            _run_isolated(
                "import numpy as np; from pasted._ext import graph_metrics_cpp; "
                "pts=np.zeros((5,3),dtype='f8'); r=np.full(2,0.77); en=np.full(2,2.5); "
                "graph_metrics_cpp(pts,r,1.0,en,5.0)"
            )
        )

    @needs_graph
    def test_graph_en_vals_shorter_than_pts_no_signal(self) -> None:
        _assert_no_signal(
            _run_isolated(
                "import numpy as np; from pasted._ext import graph_metrics_cpp; "
                "pts=np.zeros((5,3),dtype='f8'); r=np.full(5,0.77); en=np.full(2,2.5); "
                "graph_metrics_cpp(pts,r,1.0,en,5.0)"
            )
        )


# ===========================================================================
# FZ-B  NaN coordinate inputs
# ===========================================================================
# Probed outputs (3-atom triangle, atom-0 NaN):
#   relax_positions  → out[0]=[nan,0,0]  out[1]/[2] unchanged; conv=False
#   angular_gradient → all zeros  (NaN atom has no valid neighbours)
#   steinhardt Q6    → [0., 1., 1.]  (atom-0 isolated; atoms 1,2 connected)
#   graph_metrics    → lcc=2/3  ring_fraction=0  moran finite
#   rdf_h_cpp        → h_spatial=0  rdf_dev=0  (NaN pair excluded)
#
# All-NaN probed outputs:
#   relax            → all NaN; conv=False
#   angular_gradient → all zeros
#   steinhardt Q6    → all zeros
#   graph            → lcc=1/3  (each atom is its own component)
#   rdf              → h_spatial=0  rdf_dev=0
# ===========================================================================


class TestNanCoordinates:
    def _pts_nan0(self) -> np.ndarray:
        pts = PTS3.copy()
        pts[0] = [float("nan"), 0.0, 0.0]
        return pts

    # ── relax_positions ──────────────────────────────────────────────────────

    @needs_relax
    def test_relax_nan_atom0_out_shape(self) -> None:
        out, _ = relax_positions(self._pts_nan0(), RADII3, 1.0, 100)
        assert out.shape == (3, 3)

    @needs_relax
    def test_relax_nan_atom0_conv_false(self) -> None:
        _, conv = relax_positions(self._pts_nan0(), RADII3, 1.0, 100)
        assert conv is False

    @needs_relax
    def test_relax_nan_atom0_row0_is_nan(self) -> None:
        """The NaN x-coordinate of atom 0 propagates into out[0][0].
        The y and z components were 0.0 and remain 0.0 (they were never
        NaN-contaminated because the gradient only touches each coordinate
        independently in the L-BFGS line-search)."""
        out, _ = relax_positions(self._pts_nan0(), RADII3, 1.0, 100)
        assert math.isnan(float(out[0, 0]))
        assert float(out[0, 1]) == pytest.approx(0.0)
        assert float(out[0, 2]) == pytest.approx(0.0)

    @needs_relax
    def test_relax_nan_atom0_other_rows_finite(self) -> None:
        """Atoms 1 and 2 have valid coordinates; their mutual penalty does
        not involve atom 0.  Their output rows remain finite."""
        out, _ = relax_positions(self._pts_nan0(), RADII3, 1.0, 100)
        assert np.all(np.isfinite(out[1]))
        assert np.all(np.isfinite(out[2]))

    @needs_relax
    def test_relax_all_nan_output_all_nan_conv_false(self) -> None:
        """All NaN → gradient is NaN from the first step → all output NaN,
        conv=False."""
        pts = np.full((3, 3), float("nan"), dtype=np.float64)
        out, conv = relax_positions(pts, RADII3, 1.0, 100)
        out, conv = _assert_relax_contract((out, conv), len(pts))
        assert out.shape == (3, 3)
        assert np.all(np.isnan(out))
        assert conv is False

    # ── angular_repulsion_gradient ───────────────────────────────────────────

    @needs_maxent
    def test_angular_gradient_nan_atom0_all_zero(self) -> None:
        """NaN in any coordinate makes all inter-atom distances NaN → no
        valid neighbour for any atom → gradient collapses to all zeros."""
        grad = angular_repulsion_gradient(self._pts_nan0(), 5.0)
        grad = _assert_grad_contract(grad, len(self._pts_nan0()))
        assert grad.shape == (3, 3)
        assert np.all(grad == 0.0)

    @needs_maxent
    def test_angular_gradient_all_nan_all_zero(self) -> None:
        pts = np.full((3, 3), float("nan"), dtype=np.float64)
        grad = angular_repulsion_gradient(pts, 5.0)
        grad = _assert_grad_contract(grad, len(pts))
        assert np.all(grad == 0.0)

    # ── steinhardt_per_atom ──────────────────────────────────────────────────

    @needs_steinhardt
    def test_steinhardt_nan_atom0_q6_isolated(self) -> None:
        """Atom 0 is at NaN; its distance to every other atom is NaN and
        outside any finite cutoff.  Q6[0]=0 (isolated); atoms 1 and 2
        remain connected so Q6[1]=Q6[2]=1.0 (2-atom neighbour shell)."""
        result = steinhardt_per_atom(self._pts_nan0(), 5.0, [6])
        result = _assert_steinhardt_contract(result, len(self._pts_nan0()))
        q6 = result["Q6"]
        assert q6.shape == (3,)
        assert float(q6[0]) == pytest.approx(0.0)
        assert float(q6[1]) == pytest.approx(1.0)
        assert float(q6[2]) == pytest.approx(1.0)

    @needs_steinhardt
    def test_steinhardt_all_nan_all_q_zero(self) -> None:
        pts = np.full((3, 3), float("nan"), dtype=np.float64)
        result = steinhardt_per_atom(pts, 5.0, [6])
        result = _assert_steinhardt_contract(result, len(pts))
        assert np.all(result["Q6"] == 0.0)

    # ── graph_metrics_cpp ────────────────────────────────────────────────────

    @needs_graph
    def test_graph_nan_atom0_keys_present(self) -> None:
        result = graph_metrics_cpp(self._pts_nan0(), RADII3, 1.0, EN3, 5.0)
        result = _assert_graph_contract(result)
        assert set(result.keys()) == {
            "graph_lcc",
            "graph_cc",
            "ring_fraction",
            "charge_frustration",
            "moran_I_chi",
        }

    @needs_graph
    def test_graph_nan_atom0_lcc_two_thirds(self) -> None:
        """Atom 0 is isolated; atoms 1 and 2 form a 2-node component.
        graph_lcc = 2/3 (fraction of nodes in the largest component)."""
        result = graph_metrics_cpp(self._pts_nan0(), RADII3, 1.0, EN3, 5.0)
        result = _assert_graph_contract(result)
        assert float(result["graph_lcc"]) == pytest.approx(2 / 3)

    @needs_graph
    def test_graph_nan_atom0_no_rings(self) -> None:
        """A 2-node component cannot form a ring."""
        result = graph_metrics_cpp(self._pts_nan0(), RADII3, 1.0, EN3, 5.0)
        result = _assert_graph_contract(result)
        assert float(result["ring_fraction"]) == pytest.approx(0.0)

    @needs_graph
    def test_graph_nan_atom0_moran_finite(self) -> None:
        result = graph_metrics_cpp(self._pts_nan0(), RADII3, 1.0, EN3, 5.0)
        result = _assert_graph_contract(result)
        assert math.isfinite(result["moran_I_chi"])

    @needs_graph
    def test_graph_all_nan_lcc_one_third(self) -> None:
        """All atoms isolated → largest component is 1 node → lcc = 1/3."""
        pts = np.full((3, 3), float("nan"), dtype=np.float64)
        result = graph_metrics_cpp(pts, RADII3, 1.0, EN3, 5.0)
        result = _assert_graph_contract(result)
        assert float(result["graph_lcc"]) == pytest.approx(1 / 3)

    # ── rdf_h_cpp ────────────────────────────────────────────────────────────

    @needs_graph
    def test_rdf_nan_atom0_h_spatial_zero_rdf_finite(self) -> None:
        """The one valid pair (atoms 1-2) lands in a single bin → h=0.
        rdf_dev is non-zero (spike deviates from ideal-gas baseline)."""
        result = rdf_h_cpp(self._pts_nan0(), 5.0, 20)
        result = _assert_rdf_contract(result)
        assert float(result["h_spatial"]) == pytest.approx(0.0)
        assert math.isfinite(result["rdf_dev"])

    @needs_graph
    def test_rdf_all_nan_both_zero(self) -> None:
        """No valid pairs → empty histogram → both metrics 0."""
        pts = np.full((3, 3), float("nan"), dtype=np.float64)
        result = rdf_h_cpp(pts, 5.0, 20)
        result = _assert_rdf_contract(result)
        assert float(result["h_spatial"]) == pytest.approx(0.0)
        assert float(result["rdf_dev"]) == pytest.approx(0.0)


# ===========================================================================
# FZ-C  Inf / -Inf coordinate inputs
# ===========================================================================
# All-inf probed outputs:
#   relax      → output retains inf values; conv=False
#   ang_grad   → all zeros  (inf atoms have no finite-cutoff neighbours)
#   steinhardt → all zeros  (no neighbours)
#   graph      → lcc=1/3  (no bonds, each node isolated)
#   rdf        → h_spatial=0  rdf_dev=0
# ===========================================================================


class TestInfCoordinates:
    def _pts_all_inf(self, val: float = float("inf")) -> np.ndarray:
        return np.full((3, 3), val, dtype=np.float64)

    @needs_relax
    def test_relax_all_inf_conv_false_shape_correct(self) -> None:
        out, conv = relax_positions(self._pts_all_inf(), RADII3, 1.0, 100)
        out, conv = _assert_relax_contract((out, conv), len(self._pts_all_inf()))
        assert out.shape == (3, 3)
        assert conv is False

    @needs_relax
    def test_relax_all_neginf_conv_false(self) -> None:
        _, conv = relax_positions(self._pts_all_inf(float("-inf")), RADII3, 1.0, 100)
        assert conv is False

    @needs_relax
    def test_relax_mixed_nan_inf_has_nan_conv_false(self) -> None:
        """Mixed NaN + inf: any NaN gradient makes L-BFGS fail immediately;
        conv=False and NaN appears in output."""
        pts = PTS3.copy()
        pts[0, 0] = float("nan")
        pts[1, 1] = float("inf")
        out, conv = relax_positions(pts, RADII3, 1.0, 50)
        out, conv = _assert_relax_contract((out, conv), len(pts))
        assert out.shape == (3, 3)
        assert conv is False
        assert np.any(np.isnan(out))

    @needs_maxent
    def test_angular_gradient_all_inf_all_zero(self) -> None:
        """All atoms at +inf: pairwise distances are NaN (inf-inf=NaN) →
        no valid neighbour within finite cutoff → gradient = 0."""
        grad = angular_repulsion_gradient(self._pts_all_inf(), 5.0)
        grad = _assert_grad_contract(grad, len(self._pts_all_inf()))
        assert grad.shape == (3, 3)
        assert np.all(grad == 0.0)

    @needs_maxent
    def test_angular_gradient_all_neginf_all_zero(self) -> None:
        grad = angular_repulsion_gradient(self._pts_all_inf(float("-inf")), 5.0)
        grad = _assert_grad_contract(grad, len(self._pts_all_inf(float("-inf"))))
        assert np.all(grad == 0.0)

    @needs_steinhardt
    def test_steinhardt_all_inf_all_q_zero(self) -> None:
        result = steinhardt_per_atom(self._pts_all_inf(), 5.0, [4, 6, 8])
        result = _assert_steinhardt_contract(result, len(self._pts_all_inf()))
        for key, arr in result.items():
            assert np.all(arr == 0.0), f"{key} must be 0 for all-inf input"

    @needs_graph
    def test_graph_all_inf_no_bonds_lcc_one_third(self) -> None:
        """All atoms at inf → no bonds → each is its own component →
        lcc = 1/3."""
        result = graph_metrics_cpp(self._pts_all_inf(), RADII3, 1.0, EN3, 5.0)
        result = _assert_graph_contract(result)
        assert float(result["graph_lcc"]) == pytest.approx(1 / 3)
        assert float(result["ring_fraction"]) == pytest.approx(0.0)

    @needs_graph
    def test_rdf_all_inf_both_zero(self) -> None:
        result = rdf_h_cpp(self._pts_all_inf(), 5.0, 20)
        result = _assert_rdf_contract(result)
        assert float(result["h_spatial"]) == pytest.approx(0.0)
        assert float(result["rdf_dev"]) == pytest.approx(0.0)


# ===========================================================================
# FZ-D  Geometric collapse
# ===========================================================================
# Probed:
#   all-origin 4 atoms after relax: all pairwise distances > 0; conv=True
#   2 atoms at origin after relax:  distance ≈ 1.54 Å
#   planar z=0: ang_grad z-component = 0 exactly (symmetry), all finite
#   collinear:  steinhardt Q4=Q6=1.0 for all atoms (perfect 1-D rod)
#   all-origin steinhardt: Q6=[0,0,0,0] and finite
# ===========================================================================


class TestGeometricCollapse:
    @needs_relax
    def test_relax_all_origin_moves_all_atoms_apart(self) -> None:
        """Coincident atoms trigger jitter+L-BFGS; all pairwise distances
        must be > 0 after relaxation."""
        pts = np.zeros((4, 3), dtype=np.float64)
        out, conv = relax_positions(pts, np.full(4, 0.77), 1.0, 500)
        out, conv = _assert_relax_contract((out, conv), len(pts))
        assert out.shape == (4, 3)
        assert conv is True
        assert not np.all(out == 0.0)
        for i in range(4):
            for j in range(i + 1, 4):
                d = float(np.linalg.norm(out[i] - out[j]))
                assert d > 0.0, f"atoms {i} and {j} remain coincident after relax"

    @needs_relax
    def test_relax_two_atoms_at_origin_distance_positive(self) -> None:
        """After relax the 2-atom distance must be > 0."""
        pts = np.zeros((2, 3), dtype=np.float64)
        out, conv = relax_positions(pts, np.full(2, 0.77), 1.0, 500)
        out, conv = _assert_relax_contract((out, conv), len(pts))
        assert conv is True
        assert float(np.linalg.norm(out[0] - out[1])) > 0.0

    @needs_maxent
    def test_angular_gradient_all_origin_finite(self) -> None:
        """All atoms at origin: zero-distance vectors → gradient must be
        finite (implementation uses a zero-guard for direction)."""
        pts = np.zeros((4, 3), dtype=np.float64)
        grad = angular_repulsion_gradient(pts, 5.0)
        grad = _assert_grad_contract(grad, len(pts))
        assert grad.shape == (4, 3)
        assert np.all(np.isfinite(grad))

    @needs_steinhardt
    def test_steinhardt_all_origin_finite(self) -> None:
        """Zero-distance neighbours: Q_l must be finite (not NaN, not inf)."""
        pts = np.zeros((4, 3), dtype=np.float64)
        result = steinhardt_per_atom(pts, 5.0, [6])
        result = _assert_steinhardt_contract(result, len(pts))
        assert np.all(np.isfinite(result["Q6"]))

    @needs_graph
    def test_graph_all_origin_all_metrics_finite(self) -> None:
        pts = np.zeros((4, 3), dtype=np.float64)
        result = graph_metrics_cpp(pts, np.full(4, 0.77), 1.0, np.full(4, 2.5), 5.0)
        result = _assert_graph_contract(result)
        for key, val in result.items():
            assert math.isfinite(float(val)), f"graph[{key!r}] must be finite for all-origin input"

    @needs_graph
    def test_rdf_all_origin_finite(self) -> None:
        pts = np.zeros((4, 3), dtype=np.float64)
        result = rdf_h_cpp(pts, 5.0, 20)
        result = _assert_rdf_contract(result)
        assert math.isfinite(result["h_spatial"])
        assert math.isfinite(result["rdf_dev"])

    @needs_maxent
    def test_angular_gradient_planar_z_component_exactly_zero(self) -> None:
        """All atoms in z=0: by symmetry the z-component of the angular
        repulsion gradient must be exactly 0."""
        pts = np.array(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [1.5, 1.5, 0.0]], dtype=np.float64
        )
        grad = angular_repulsion_gradient(pts, 5.0)
        grad = _assert_grad_contract(grad, len(pts))
        assert grad.shape == (4, 3)
        assert np.all(np.isfinite(grad))
        np.testing.assert_array_equal(grad[:, 2], np.zeros(4))

    @needs_steinhardt
    def test_steinhardt_collinear_q4_q6_all_one(self) -> None:
        """Four atoms on a line: perfect 1-D rod → Q4=Q6=1 for all."""
        pts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64
        )
        result = steinhardt_per_atom(pts, 5.0, [4, 6])
        result = _assert_steinhardt_contract(result, len(pts))
        np.testing.assert_array_almost_equal(result["Q4"], np.ones(4))
        np.testing.assert_array_almost_equal(result["Q6"], np.ones(4))


# ===========================================================================
# FZ-E  Empty (N=0) arrays
# ===========================================================================
# Probed outputs:
#   relax      → shape (0,3); conv=True  (vacuously converged)
#   ang_grad   → size 0
#   steinhardt → {} (empty dict)
#   graph      → all metrics 0.0 and finite
#   rdf        → h_spatial=0; rdf_dev=0
# ===========================================================================


class TestEmptyArrays:
    _PTS0: np.ndarray = np.empty((0, 3), dtype=np.float64)
    _R0: np.ndarray = np.empty(0, dtype=np.float64)
    _EN0: np.ndarray = np.empty(0, dtype=np.float64)

    @needs_relax
    def test_relax_empty_shape_conv_true(self) -> None:
        out, conv = relax_positions(self._PTS0, self._R0, 1.0, 10)
        out, conv = _assert_relax_contract((out, conv), len(self._PTS0))
        assert out.shape == (0, 3)
        assert conv is True

    @needs_maxent
    def test_angular_gradient_empty_size_zero(self) -> None:
        grad = angular_repulsion_gradient(self._PTS0, 5.0)
        grad = _assert_grad_contract(grad, len(self._PTS0))
        assert grad.size == 0

    @needs_steinhardt
    def test_steinhardt_empty_returns_empty_dict(self) -> None:
        assert steinhardt_per_atom(self._PTS0, 5.0, [4, 6, 8]) == {}

    @needs_graph
    def test_graph_empty_all_metrics_zero_finite(self) -> None:
        result = graph_metrics_cpp(self._PTS0, self._R0, 1.0, self._EN0, 5.0)
        result = _assert_graph_contract(result)
        for key, val in result.items():
            v = float(val)
            assert math.isfinite(v), f"graph[{key!r}] not finite for empty input"
            assert v == pytest.approx(0.0)

    @needs_graph
    def test_rdf_empty_h_zero_rdf_zero(self) -> None:
        result = rdf_h_cpp(self._PTS0, 5.0, 20)
        result = _assert_rdf_contract(result)
        assert float(result["h_spatial"]) == pytest.approx(0.0)
        assert float(result["rdf_dev"]) == pytest.approx(0.0)


# ===========================================================================
# FZ-F  Single-atom (N=1) arrays
# ===========================================================================
# Probed outputs:
#   relax      → shape (1,3); conv=True; position unchanged
#   ang_grad   → [[0,0,0]]
#   steinhardt → Q4/Q6/Q8=[0.0]
#   graph      → lcc=1.0; ring_fraction=0; all finite
#   rdf        → h_spatial=0; rdf_dev=0
# ===========================================================================


class TestSingleAtom:
    _PTS1: np.ndarray = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    _R1: np.ndarray = np.array([0.77], dtype=np.float64)
    _EN1: np.ndarray = np.array([2.55], dtype=np.float64)

    @needs_relax
    def test_relax_single_shape_conv_true_position_unchanged(self) -> None:
        """No clash partner → no penalty → position must not change."""
        out, conv = relax_positions(self._PTS1, self._R1, 1.0, 100)
        out, conv = _assert_relax_contract((out, conv), len(self._PTS1))
        assert out.shape == (1, 3)
        assert conv is True
        np.testing.assert_array_equal(out[0], self._PTS1[0])

    @needs_maxent
    def test_angular_gradient_single_zero_vector(self) -> None:
        grad = angular_repulsion_gradient(self._PTS1, 5.0)
        grad = _assert_grad_contract(grad, len(self._PTS1))
        assert grad.shape == (1, 3)
        assert np.all(grad == 0.0)

    @needs_steinhardt
    def test_steinhardt_single_all_q_zero(self) -> None:
        result = steinhardt_per_atom(self._PTS1, 5.0, [4, 6, 8])
        result = _assert_steinhardt_contract(result, len(self._PTS1))
        for _key, arr in result.items():
            assert arr.shape == (1,)
            assert float(arr[0]) == pytest.approx(0.0)

    @needs_graph
    def test_graph_single_lcc_one_no_rings_all_finite(self) -> None:
        result = graph_metrics_cpp(self._PTS1, self._R1, 1.0, self._EN1, 5.0)
        result = _assert_graph_contract(result)
        assert float(result["graph_lcc"]) == pytest.approx(1.0)
        assert float(result["ring_fraction"]) == pytest.approx(0.0)
        for _key, val in result.items():
            assert math.isfinite(float(val))

    @needs_graph
    def test_rdf_single_h_zero_rdf_zero(self) -> None:
        result = rdf_h_cpp(self._PTS1, 5.0, 20)
        result = _assert_rdf_contract(result)
        assert float(result["h_spatial"]) == pytest.approx(0.0)
        assert float(result["rdf_dev"]) == pytest.approx(0.0)


# ===========================================================================
# FZ-G  Non-C-contiguous (Fortran-order) arrays
# ===========================================================================
# pybind11 copies Fortran-order arrays to a temporary C-contiguous buffer.
# Results must be bit-identical to the C-order path.
# ===========================================================================


class TestFortranOrderArrays:
    @needs_relax
    def test_relax_fortran_bit_identical(self) -> None:
        out_c, conv_c = relax_positions(PTS3.copy(), RADII3, 1.0, 200)
        out_f, conv_f = relax_positions(np.asfortranarray(PTS3), RADII3, 1.0, 200)
        assert conv_c == conv_f
        np.testing.assert_array_equal(out_c, out_f)

    @needs_maxent
    def test_angular_gradient_fortran_bit_identical(self) -> None:
        gc = angular_repulsion_gradient(PTS3.copy(), 5.0)
        gf = angular_repulsion_gradient(np.asfortranarray(PTS3), 5.0)
        np.testing.assert_array_equal(gc, gf)

    @needs_steinhardt
    def test_steinhardt_fortran_bit_identical(self) -> None:
        rc = steinhardt_per_atom(PTS3.copy(), 5.0, [6])
        rf = steinhardt_per_atom(np.asfortranarray(PTS3), 5.0, [6])
        np.testing.assert_array_equal(rc["Q6"], rf["Q6"])

    @needs_graph
    def test_graph_fortran_bit_identical(self) -> None:
        rc = graph_metrics_cpp(PTS3.copy(), RADII3, 1.0, EN3, 5.0)
        rf = graph_metrics_cpp(np.asfortranarray(PTS3), RADII3, 1.0, EN3, 5.0)
        for key in rc:
            assert rc[key] == pytest.approx(rf[key], abs=0.0)

    @needs_graph
    def test_rdf_fortran_bit_identical(self) -> None:
        rc = rdf_h_cpp(PTS3.copy(), 5.0, 20)
        rf = rdf_h_cpp(np.asfortranarray(PTS3), 5.0, 20)
        assert rc["h_spatial"] == pytest.approx(rf["h_spatial"], abs=0.0)
        assert rc["rdf_dev"] == pytest.approx(rf["rdf_dev"], abs=0.0)


# ===========================================================================
# FZ-H  Boundary scalar parameters
# ===========================================================================


class TestBoundaryScalarParameters:
    # ── relax_positions: cov_scale ───────────────────────────────────────────

    @needs_relax
    def test_relax_cov_scale_zero_positions_unchanged_conv_true(self) -> None:
        """cov_scale=0 → minimum distance = 0 → penalty always 0 →
        no force → L-BFGS terminates immediately; conv=True, positions
        unchanged."""
        out, conv = relax_positions(PTS3, RADII3, 0.0, 100)
        out, conv = _assert_relax_contract((out, conv), len(PTS3))
        assert conv is True
        np.testing.assert_array_equal(out, PTS3)

    @needs_relax
    def test_relax_cov_scale_negative_conv_true_finite_moved(self) -> None:
        """Negative cov_scale inverts the penalty (atoms attracted toward
        overlap).  L-BFGS finds a local minimum; conv=True, output finite,
        positions differ from input."""
        out, conv = relax_positions(PTS3, RADII3, -1.0, 100)
        out, conv = _assert_relax_contract((out, conv), len(PTS3))
        assert conv is True
        assert out.shape == (3, 3)
        assert np.all(np.isfinite(out))
        assert not np.allclose(out, PTS3)

    # ── relax_positions: max_cycles ──────────────────────────────────────────

    @needs_relax
    def test_relax_max_cycles_zero_conv_false_positions_unchanged(self) -> None:
        """max_cycles=0: no iterations run → conv=False; positions untouched."""
        out, conv = relax_positions(PTS3, RADII3, 1.0, 0)
        out, conv = _assert_relax_contract((out, conv), len(PTS3))
        assert conv is False
        np.testing.assert_array_equal(out, PTS3)

    @needs_relax
    def test_relax_max_cycles_negative_conv_false(self) -> None:
        """Negative max_cycles treated as 0 (no-op) → conv=False."""
        _, conv = relax_positions(PTS3, RADII3, 1.0, -1)
        assert conv is False

    @needs_relax
    def test_relax_max_cycles_one_conv_true_finite(self) -> None:
        """One L-BFGS step on a triangle that has no clashes: converges
        immediately; output is finite."""
        out, conv = relax_positions(PTS3, RADII3, 1.0, 1)
        out, conv = _assert_relax_contract((out, conv), len(PTS3))
        assert conv is True
        assert np.all(np.isfinite(out))

    # ── angular_repulsion_gradient: cutoff ───────────────────────────────────

    @needs_maxent
    def test_angular_gradient_cutoff_zero_all_zero(self) -> None:
        """cutoff=0: no pair within range → no repulsion → gradient = 0."""
        grad = angular_repulsion_gradient(PTS3, 0.0)
        grad = _assert_grad_contract(grad, len(PTS3))
        assert grad.shape == (3, 3)
        assert np.all(grad == 0.0)

    @needs_maxent
    def test_angular_gradient_cutoff_negative_all_zero_finite(self) -> None:
        """cutoff=-1: negative cutoff forms no valid pairs → gradient = 0
        and finite."""
        grad = angular_repulsion_gradient(PTS3, -1.0)
        grad = _assert_grad_contract(grad, len(PTS3))
        assert grad.shape == (3, 3)
        assert np.all(np.isfinite(grad))
        assert np.all(grad == 0.0)

    @needs_maxent
    def test_angular_gradient_cutoff_very_large_nonzero_finite(self) -> None:
        """cutoff=1e15: all atoms are neighbours → non-zero repulsion →
        gradient has nonzero entries; all values finite."""
        grad = angular_repulsion_gradient(PTS3, 1e15)
        grad = _assert_grad_contract(grad, len(PTS3))
        assert grad.shape == (3, 3)
        assert np.all(np.isfinite(grad))
        assert np.any(grad != 0.0)

    # ── rdf_h_cpp: n_bins ────────────────────────────────────────────────────

    @needs_graph
    def test_rdf_n_bins_zero_both_zero(self) -> None:
        """n_bins=0: empty histogram → h_spatial=0; rdf_dev=0."""
        result = rdf_h_cpp(PTS3, 5.0, 0)
        result = _assert_rdf_contract(result)
        assert float(result["h_spatial"]) == pytest.approx(0.0)
        assert float(result["rdf_dev"]) == pytest.approx(0.0)

    @needs_graph
    def test_rdf_n_bins_negative_both_zero(self) -> None:
        """n_bins=-1: treated same as 0 → both metrics 0."""
        result = rdf_h_cpp(PTS3, 5.0, -1)
        result = _assert_rdf_contract(result)
        assert float(result["h_spatial"]) == pytest.approx(0.0)
        assert float(result["rdf_dev"]) == pytest.approx(0.0)

    @needs_graph
    def test_rdf_n_bins_one_h_zero_rdf_dev_positive_finite(self) -> None:
        """n_bins=1: one bin captures all pairs → uniform probability →
        h_spatial=0 (single-bin entropy).  rdf_dev > 0 because a spike
        deviates from the ideal-gas plateau."""
        result = rdf_h_cpp(PTS3, 5.0, 1)
        result = _assert_rdf_contract(result)
        assert float(result["h_spatial"]) == pytest.approx(0.0)
        assert math.isfinite(result["rdf_dev"])
        assert result["rdf_dev"] > 0.0

    @needs_graph
    def test_rdf_n_bins_large_finite(self) -> None:
        """10 million bins: no OOM or integer overflow."""
        result = rdf_h_cpp(PTS3, 5.0, 10_000_000)
        result = _assert_rdf_contract(result)
        assert math.isfinite(result["h_spatial"])
        assert math.isfinite(result["rdf_dev"])

    # ── rdf_h_cpp: cutoff ────────────────────────────────────────────────────

    @needs_graph
    def test_rdf_cutoff_zero_both_zero(self) -> None:
        result = rdf_h_cpp(PTS3, 0.0, 20)
        result = _assert_rdf_contract(result)
        assert float(result["h_spatial"]) == pytest.approx(0.0)
        assert float(result["rdf_dev"]) == pytest.approx(0.0)

    @needs_graph
    def test_rdf_cutoff_nan_both_zero(self) -> None:
        """cutoff=NaN: no pair satisfies d < NaN → empty histogram."""
        result = rdf_h_cpp(PTS3, float("nan"), 20)
        result = _assert_rdf_contract(result)
        assert float(result["h_spatial"]) == pytest.approx(0.0)
        assert float(result["rdf_dev"]) == pytest.approx(0.0)

    @needs_graph
    def test_rdf_cutoff_large_h_zero_rdf_dev_one(self) -> None:
        """cutoff=1e15: all pairs included; all land in a tiny fraction of
        bins near the origin → h_spatial=0; rdf_dev=1.0 (delta spike
        relative to ideal-gas)."""
        result = rdf_h_cpp(PTS3, 1e15, 20)
        result = _assert_rdf_contract(result)
        assert float(result["h_spatial"]) == pytest.approx(0.0)
        assert float(result["rdf_dev"]) == pytest.approx(1.0, abs=1e-9)

    # ── steinhardt_per_atom: l_values ────────────────────────────────────────

    @needs_steinhardt
    def test_steinhardt_empty_l_values_empty_dict(self) -> None:
        assert steinhardt_per_atom(PTS3, 5.0, []) == {}

    @needs_steinhardt
    def test_steinhardt_l_zero_all_one(self) -> None:
        """Q_0 = 1 for every atom that has any neighbours (monopole)."""
        result = steinhardt_per_atom(PTS3, 5.0, [0])
        result = _assert_steinhardt_contract(result, len(PTS3))
        assert "Q0" in result
        np.testing.assert_array_almost_equal(result["Q0"], np.ones(3))

    @needs_steinhardt
    def test_steinhardt_l_twelve_finite(self) -> None:
        """l=12 is the documented maximum; output must be finite."""
        result = steinhardt_per_atom(PTS3, 5.0, [12])
        result = _assert_steinhardt_contract(result, len(PTS3))
        assert "Q12" in result
        assert np.all(np.isfinite(result["Q12"]))

    @needs_steinhardt
    def test_steinhardt_l_negative_raises_runtime_error(self) -> None:
        with pytest.raises(RuntimeError, match="out of range"):
            steinhardt_per_atom(PTS3, 5.0, [-1])

    @needs_steinhardt
    def test_steinhardt_l_thirteen_raises_runtime_error(self) -> None:
        with pytest.raises(RuntimeError, match="out of range"):
            steinhardt_per_atom(PTS3, 5.0, [13])

    # ── graph_metrics_cpp: cutoff ─────────────────────────────────────────────

    @needs_graph
    def test_graph_cutoff_zero_no_bonds_lcc_one_third(self) -> None:
        """cutoff=0: no bonds → every node is its own component → lcc=1/3."""
        result = graph_metrics_cpp(PTS3, RADII3, 1.0, EN3, 0.0)
        result = _assert_graph_contract(result)
        assert float(result["graph_lcc"]) == pytest.approx(1 / 3)
        assert float(result["ring_fraction"]) == pytest.approx(0.0)
        for _key, val in result.items():
            assert math.isfinite(float(val))

    @needs_graph
    def test_graph_cutoff_large_all_bonded_lcc_one(self) -> None:
        """cutoff=1e15: all atoms bonded → lcc=1; all metrics finite."""
        result = graph_metrics_cpp(PTS3, RADII3, 1.0, EN3, 1e15)
        result = _assert_graph_contract(result)
        assert float(result["graph_lcc"]) == pytest.approx(1.0)
        for _key, val in result.items():
            assert math.isfinite(float(val))


# ===========================================================================
# FZ-I  NaN / Inf in ancillary arrays (radii, en_vals)
# ===========================================================================
# Probed outputs:
#   nan_radii → all positions NaN; conv=False
#     (NaN radius poisons penalty for every pair involving atom 0;
#      all atoms are within range → gradient NaN globally)
#   inf_radii → mix of inf/nan; conv=False
#   nan_en_vals[0] → topology finite; charge_frustration=NaN; moran=0
#   inf_en_vals[1] → same
#   zero_en_vals   → charge_frustration=0; moran=0
# ===========================================================================


class TestAncillaryArrayAnomalies:
    @needs_relax
    def test_relax_nan_first_radius_all_nan_conv_false(self) -> None:
        """NaN in radii[0] enters the pairwise penalty for every pair (0,j).
        All atoms are within cutoff of each other → gradient NaN globally
        → all output positions NaN; conv=False."""
        radii_nan = RADII3.copy()
        radii_nan[0] = float("nan")
        out, conv = relax_positions(PTS3, radii_nan, 1.0, 100)
        out, conv = _assert_relax_contract((out, conv), len(PTS3))
        assert out.shape == (3, 3)
        assert conv is False
        assert np.all(np.isnan(out))

    @needs_relax
    def test_relax_inf_first_radius_conv_false(self) -> None:
        """Infinite radius → clamped penalty distance is always negative →
        penalty diverges → L-BFGS cannot converge."""
        radii_inf = RADII3.copy()
        radii_inf[0] = float("inf")
        out, conv = relax_positions(PTS3, radii_inf, 1.0, 100)
        out, conv = _assert_relax_contract((out, conv), len(PTS3))
        assert out.shape == (3, 3)
        assert conv is False

    @needs_graph
    def test_graph_nan_en_topology_finite_charge_nan(self) -> None:
        """NaN in en_vals[0] contaminates charge_frustration but must not
        affect pure graph topology (lcc, cc, ring_fraction)."""
        en_nan = EN3.copy()
        en_nan[0] = float("nan")
        result = graph_metrics_cpp(PTS3, RADII3, 1.0, en_nan, 5.0)
        result = _assert_graph_contract(result)
        assert float(result["graph_lcc"]) == pytest.approx(1.0)
        assert float(result["graph_cc"]) == pytest.approx(1.0)
        assert math.isfinite(result["ring_fraction"])
        assert math.isnan(result["charge_frustration"])

    @needs_graph
    def test_graph_inf_en_topology_finite_charge_nan(self) -> None:
        """Infinite EN causes EN-difference overflow → charge_frustration=NaN;
        topology metrics unaffected."""
        en_inf = EN3.copy()
        en_inf[1] = float("inf")
        result = graph_metrics_cpp(PTS3, RADII3, 1.0, en_inf, 5.0)
        result = _assert_graph_contract(result)
        assert float(result["graph_lcc"]) == pytest.approx(1.0)
        assert math.isnan(result["charge_frustration"])

    @needs_graph
    def test_graph_all_zero_en_vals_charge_zero_moran_zero(self) -> None:
        """Uniform EN → zero EN variance → charge_frustration=0 and
        moran_I_chi=0 (no spatial autocorrelation in a constant field)."""
        result = graph_metrics_cpp(PTS3, RADII3, 1.0, np.zeros(3), 5.0)
        result = _assert_graph_contract(result)
        assert float(result["charge_frustration"]) == pytest.approx(0.0)
        assert float(result["moran_I_chi"]) == pytest.approx(0.0)


# ===========================================================================
# FZ-J  Float overflow — sys.float_info.max as coordinate
# ===========================================================================
# 2-atom probe: [fmax,0,0] + [0,0,0]
#   relax:      dist >> cov_threshold → no clash → conv=True
#   ang_grad:   pair dist ~inf → no neighbour within 5 Å → all zeros
#   steinhardt: same reasoning → Q6=[0,0]
#   graph:      lcc=0.5 (two isolated nodes)
#   rdf:        pair excluded → h=0; rdf_dev=0
#
# 2-atom probe both at fmax:
#   relax:      jitter fires; fmax+jitter overflows → gradient NaN → conv=False
# ===========================================================================


class TestFloatOverflow:
    _FMAX: float = sys.float_info.max

    def _pts_fmax(self) -> np.ndarray:
        return np.array([[self._FMAX, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)

    @needs_relax
    def test_relax_fmax_coord_conv_true_shape_correct(self) -> None:
        """One atom at fmax, one at origin: distance >> cov threshold →
        no clash → L-BFGS converges immediately."""
        out, conv = relax_positions(self._pts_fmax(), np.full(2, 0.77), 1.0, 10)
        out, conv = _assert_relax_contract((out, conv), len(self._pts_fmax()))
        assert out.shape == (2, 3)
        assert conv is True

    @needs_maxent
    def test_angular_gradient_fmax_coord_all_zero(self) -> None:
        """Pair distance overflows to inf → no neighbour within 5 Å →
        gradient = 0."""
        grad = angular_repulsion_gradient(self._pts_fmax(), 5.0)
        grad = _assert_grad_contract(grad, len(self._pts_fmax()))
        assert grad.shape == (2, 3)
        assert np.all(grad == 0.0)

    @needs_steinhardt
    def test_steinhardt_fmax_coord_all_q_zero(self) -> None:
        """Same reasoning: fmax atom has no neighbours → Q_l = 0."""
        result = steinhardt_per_atom(self._pts_fmax(), 5.0, [6])
        result = _assert_steinhardt_contract(result, len(self._pts_fmax()))
        np.testing.assert_array_equal(result["Q6"], np.zeros(2))

    @needs_graph
    def test_graph_fmax_coord_lcc_half_no_infinite(self) -> None:
        """Two disconnected atoms (distance > cutoff): each is its own
        component of size 1 → lcc = 1/2; no metric is infinite."""
        result = graph_metrics_cpp(self._pts_fmax(), np.full(2, 0.77), 1.0, np.full(2, 2.5), 5.0)
        assert float(result["graph_lcc"]) == pytest.approx(0.5)
        for key, val in result.items():
            assert not math.isinf(float(val)), (
                f"graph[{key!r}] must not be inf for fmax-coord input"
            )

    @needs_graph
    def test_rdf_fmax_coord_h_zero_rdf_zero(self) -> None:
        """Pair distance > cutoff → excluded → h=0; rdf_dev=0."""
        result = rdf_h_cpp(self._pts_fmax(), 5.0, 20)
        result = _assert_rdf_contract(result)
        assert float(result["h_spatial"]) == pytest.approx(0.0)
        assert float(result["rdf_dev"]) == pytest.approx(0.0)

    @needs_relax
    def test_relax_both_atoms_at_fmax_conv_false_shape_correct(self) -> None:
        """Two coincident atoms at fmax: jitter fires, fmax+delta overflows
        → gradient becomes NaN → conv=False.  No crash; shape is correct."""
        pts = np.full((2, 3), self._FMAX, dtype=np.float64)
        out, conv = relax_positions(pts, np.full(2, 0.77), 1.0, 10)
        out, conv = _assert_relax_contract((out, conv), len(pts))
        assert out.shape == (2, 3)
        assert conv is False
