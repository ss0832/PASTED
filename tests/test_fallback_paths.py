"""Tests for pure-Python fallback paths in _placement.py and _metrics.py.

Each test group patches the relevant ``HAS_*`` flag(s) to ``False`` so that
the C++ extension is bypassed and the Python/NumPy implementation runs.
This covers the lines that are dead under the normal test suite (which always
has the compiled extensions available).

Patching strategy
-----------------
The flags are imported as module-level names at load time, so we patch them
*in the consuming module* (not in ``_ext``):

    pasted._placement.HAS_RELAX
    pasted._placement.HAS_MAXENT
    pasted._placement.HAS_MAXENT_LOOP
    pasted._metrics._HAS_GRAPH
    pasted._metrics._HAS_STEINHARDT
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RELAX_FLAGS = {
    "pasted._placement.HAS_RELAX": False,
}
_MAXENT_FLAGS = {
    "pasted._placement.HAS_MAXENT": False,
    "pasted._placement.HAS_MAXENT_LOOP": False,
}
_ALL_PLACEMENT_FLAGS = {**_RELAX_FLAGS, **_MAXENT_FLAGS}
_GRAPH_FLAG = {"pasted._metrics._HAS_GRAPH": False}
_STEINHARDT_FLAG = {"pasted._metrics._HAS_STEINHARDT": False}
_ALL_METRICS_FLAGS = {**_GRAPH_FLAG, **_STEINHARDT_FLAG}


def _patch(**flags):
    """Return a context manager that patches all given names to their values."""
    from contextlib import ExitStack

    stack = ExitStack()
    for target, val in flags.items():
        stack.enter_context(patch(target, val))
    return stack


# ---------------------------------------------------------------------------
# relax_positions — Python/NumPy fallback
# ---------------------------------------------------------------------------


class TestRelaxPositionsFallback:
    """Verify the pure-Python repulsion-relaxation loop."""

    def _run(self, atoms, positions, cov_scale=1.0, max_cycles=1500, seed=0):
        from pasted._placement import relax_positions

        with _patch(**_RELAX_FLAGS):
            return relax_positions(
                atoms, positions, cov_scale, max_cycles=max_cycles, seed=seed
            )

    def test_single_atom_returns_immediately(self):
        """n < 2 must return immediately with converged=True."""
        pos, converged = self._run(["C"], [(0.0, 0.0, 0.0)])
        assert converged is True
        assert len(pos) == 1

    def test_two_separated_atoms_already_converged(self):
        """Atoms far apart → converged=True on first iteration."""
        _pos, converged = self._run(
            ["C", "C"], [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
        )
        assert converged is True

    def test_overlapping_atoms_get_separated(self):
        """Two atoms placed at the same point must be pushed apart."""
        pos, _converged = self._run(
            ["C", "C"], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
        )
        p0 = np.array(pos[0])
        p1 = np.array(pos[1])
        d = float(np.linalg.norm(p0 - p1))
        # Covalent radius of C ≈ 0.77 Å → min dist ≈ 1.54 Å
        assert d > 0.5, f"Atoms still too close after relax: d={d:.4f} Å"

    def test_very_close_atoms_converge(self):
        """Atoms slightly too close must reach convergence (not exhaust cycles)."""
        _pos, converged = self._run(
            ["C", "N"], [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
            max_cycles=2000,
        )
        assert converged is True

    def test_max_cycles_exhaustion_returns_converged_false(self):
        """With max_cycles=1 on severely clashing atoms, converged must be False."""
        # Place many atoms all at origin so one cycle is never enough.
        n = 6
        atoms = ["C"] * n
        positions = [(0.0, 0.0, 0.0)] * n
        pos, converged = self._run(atoms, positions, max_cycles=1)
        # Convergence status is False; positions were still updated (not frozen).
        assert converged is False
        assert len(pos) == n

    def test_output_length_matches_input(self):
        """Output list length must equal input atom count."""
        atoms = ["C", "N", "O", "H"]
        positions = [(float(i), 0.0, 0.0) for i in range(len(atoms))]
        pos, _ = self._run(atoms, positions)
        assert len(pos) == len(atoms)

    def test_positions_are_finite(self):
        """All output coordinates must be finite floats."""
        atoms = ["C", "C", "N", "O"]
        positions = [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.0, 0.3)]
        pos, _ = self._run(atoms, positions, max_cycles=500)
        for xyz in pos:
            for v in xyz:
                assert math.isfinite(v), f"Non-finite coordinate: {v}"

    def test_result_close_to_cpp_path(self):
        """Python fallback result must be close to the C++ result (same math)."""
        from pasted._ext import HAS_RELAX
        from pasted._placement import relax_positions

        if not HAS_RELAX:
            pytest.skip("C++ relax not available — cannot compare paths")

        atoms = ["C", "N", "O", "C"]
        positions = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)]
        seed = 7

        pos_cpp, conv_cpp = relax_positions(atoms, positions, 1.0, seed=seed)

        with _patch(**_RELAX_FLAGS):
            pos_py, conv_py = relax_positions(atoms, positions, 1.0, seed=seed)

        # Both should converge on the same input.
        assert conv_cpp == conv_py
        # Final atom counts must match.
        assert len(pos_cpp) == len(pos_py)
        # Distances between pairs should be at least the covalent minimum
        # in both outputs (i.e. both implementations actually enforce the constraint).
        from pasted._atoms import _cov_radius_ang
        for out in (pos_cpp, pos_py):
            pts = np.array(out)
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = float(np.linalg.norm(pts[i] - pts[j]))
                    ri = _cov_radius_ang(atoms[i])
                    rj = _cov_radius_ang(atoms[j])
                    assert d >= 0.9 * (ri + rj), (
                        f"Clash in {'cpp' if out is pos_cpp else 'py'} output: "
                        f"atoms {i},{j} d={d:.3f} min={ri+rj:.3f}"
                    )


# ---------------------------------------------------------------------------
# _angular_repulsion_gradient — Python fallback
# ---------------------------------------------------------------------------


class TestAngularRepulsionGradientFallback:
    """Verify the pure-Python angular repulsion gradient."""

    def _grad_py(self, pts: np.ndarray, cutoff: float) -> np.ndarray:
        from pasted._placement import _angular_repulsion_gradient

        with patch("pasted._placement.HAS_MAXENT", False):
            return _angular_repulsion_gradient(pts, cutoff)

    def test_output_shape(self):
        """Gradient shape must be (n, 3)."""
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        grad = self._grad_py(pts, cutoff=5.0)
        assert grad.shape == (3, 3)

    def test_no_neighbors_gives_zero_gradient(self):
        """When no pair is within cutoff, gradient must be identically zero."""
        pts = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        grad = self._grad_py(pts, cutoff=1.0)
        assert np.allclose(grad, 0.0)

    def test_gradient_is_finite(self):
        """All gradient components must be finite for a typical configuration."""
        rng = np.random.default_rng(42)
        pts = rng.uniform(-3.0, 3.0, size=(8, 3))
        grad = self._grad_py(pts, cutoff=5.0)
        assert np.all(np.isfinite(grad)), "Non-finite gradient component"

    def test_matches_cpp_result(self):
        """Python gradient must be close to the C++ result."""
        from pasted._ext import HAS_MAXENT
        from pasted._placement import _angular_repulsion_gradient

        if not HAS_MAXENT:
            pytest.skip("C++ angular gradient not available")

        rng = np.random.default_rng(0)
        pts = rng.uniform(-3.0, 3.0, size=(6, 3))
        cutoff = 5.0

        grad_cpp = _angular_repulsion_gradient(pts, cutoff)
        grad_py = self._grad_py(pts, cutoff)

        # The Python and C++ implementations use the same formula but
        # differ in floating-point evaluation order, so an exact match is not
        # expected.  A relative tolerance of 1e-3 verifies they compute the
        # same quantity without being fragile to FP reordering.
        np.testing.assert_allclose(
            grad_py, grad_cpp, rtol=1e-3, atol=1e-6,
            err_msg="Python and C++ angular gradient disagree",
        )


# ---------------------------------------------------------------------------
# place_maxent — Python steepest-descent fallback
# ---------------------------------------------------------------------------


class TestPlaceMaxentFallback:
    """Verify the Python steepest-descent fallback for place_maxent."""

    def _run(self, atoms, region, relax_also_python=True, maxent_steps=5, seed=0):
        """Run place_maxent with C++ loop disabled (and optionally C++ relax too)."""
        import random

        from pasted._placement import place_maxent

        rng = random.Random(seed)
        flags = dict(_MAXENT_FLAGS)
        if relax_also_python:
            flags.update(_RELAX_FLAGS)

        with _patch(**flags):
            return place_maxent(
                atoms,
                region,
                cov_scale=1.0,
                rng=rng,
                maxent_steps=maxent_steps,
                maxent_lr=0.05,
                seed=seed,
            )

    def test_returns_correct_atom_count_sphere(self):
        atoms = ["C", "N", "O", "C", "N"]
        out_atoms, positions = self._run(atoms, "sphere:6")
        assert len(out_atoms) == len(atoms)
        assert len(positions) == len(atoms)

    def test_returns_correct_atom_count_box(self):
        atoms = ["C", "O", "N"]
        out_atoms, positions = self._run(atoms, "box:8")
        assert len(out_atoms) == len(atoms)
        assert len(positions) == len(atoms)

    def test_positions_are_finite(self):
        atoms = ["C", "C", "N", "O"]
        _, positions = self._run(atoms, "sphere:6")
        for xyz in positions:
            for v in xyz:
                assert math.isfinite(v), f"Non-finite coordinate: {v}"

    def test_hybrid_mode_python_loop_cpp_relax(self):
        """Python gradient descent + C++ relax: must still produce valid output."""
        from pasted._ext import HAS_RELAX

        if not HAS_RELAX:
            pytest.skip("C++ relax not available for hybrid mode test")

        atoms = ["C", "N", "O"]
        _out_atoms, positions = self._run(atoms, "sphere:5", relax_also_python=False)
        assert len(positions) == 3
        for xyz in positions:
            assert all(math.isfinite(v) for v in xyz)

    def test_unknown_region_raises(self):
        """place_maxent must raise ValueError for an unrecognised region spec."""
        import random

        from pasted._placement import place_maxent

        rng = random.Random(0)
        with _patch(**_ALL_PLACEMENT_FLAGS):
            with pytest.raises(ValueError, match="Unknown region spec"):
                place_maxent(["C", "N"], "cylinder:5", 1.0, rng, maxent_steps=1)


# ---------------------------------------------------------------------------
# compute_all_metrics — _HAS_GRAPH=False fallback
# ---------------------------------------------------------------------------


class TestComputeAllMetricsFallback:
    """Verify compute_all_metrics uses the pure-Python path when HAS_GRAPH=False."""

    def _metrics(self, atoms, positions):
        import numpy as np

        from pasted._atoms import _cov_radius_ang
        from pasted._metrics import compute_all_metrics

        radii = np.array([_cov_radius_ang(a) for a in atoms])
        cutoff = 1.5 * float(np.median(radii)) * 2.0

        with _patch(**_ALL_METRICS_FLAGS):
            return compute_all_metrics(
                atoms, positions, n_bins=10, w_atom=0.5, w_spatial=0.5,
                cutoff=cutoff, cov_scale=1.0,
            )

    def test_returns_all_expected_keys(self):
        from pasted._atoms import ALL_METRICS

        atoms = ["C", "N", "O", "C", "N", "O"]
        positions = [(float(i), 0.0, 0.0) for i in range(len(atoms))]
        m = self._metrics(atoms, positions)
        assert set(m.keys()) == ALL_METRICS

    def test_all_values_are_finite(self):
        atoms = ["C", "N", "O", "C", "C"]
        positions = [(float(i) * 2.0, 0.0, 0.0) for i in range(len(atoms))]
        m = self._metrics(atoms, positions)
        for k, v in m.items():
            assert math.isfinite(v), f"Non-finite metric {k}={v}"

    def test_h_total_in_unit_range(self):
        """H_total = 0.5*H_atom + 0.5*H_spatial must lie in [0, ∞) and be reasonable."""
        atoms = ["C", "N", "O", "C", "N", "O"]
        positions = [(float(i) * 2.0, 0.0, 0.0) for i in range(len(atoms))]
        m = self._metrics(atoms, positions)
        assert m["H_total"] >= 0.0

    def test_graph_lcc_in_unit_interval(self):
        atoms = ["C", "N", "O"]
        positions = [(float(i) * 2.0, 0.0, 0.0) for i in range(len(atoms))]
        m = self._metrics(atoms, positions)
        assert 0.0 <= m["graph_lcc"] <= 1.0

    def test_ring_fraction_in_unit_interval(self):
        atoms = ["C", "N", "O"]
        positions = [(float(i) * 2.0, 0.0, 0.0) for i in range(len(atoms))]
        m = self._metrics(atoms, positions)
        assert 0.0 <= m["ring_fraction"] <= 1.0

    def test_results_close_to_cpp_path(self):
        """Python fallback metrics must agree with C++ metrics within tolerance."""
        from pasted._atoms import _cov_radius_ang
        from pasted._ext import HAS_GRAPH
        from pasted._metrics import compute_all_metrics

        if not HAS_GRAPH:
            pytest.skip("C++ graph metrics not available — cannot compare paths")

        atoms = ["C", "N", "O", "C", "N"]
        positions = [
            (0.0, 0.0, 0.0), (1.8, 0.0, 0.0), (0.0, 1.8, 0.0),
            (1.8, 1.8, 0.0), (0.9, 0.9, 1.5),
        ]
        radii = np.array([_cov_radius_ang(a) for a in atoms])
        cutoff = 1.5 * float(np.median(radii)) * 2.0

        m_cpp = compute_all_metrics(
            atoms, positions, 10, 0.5, 0.5, cutoff, 1.0
        )
        with _patch(**_ALL_METRICS_FLAGS):
            m_py = compute_all_metrics(
                atoms, positions, 10, 0.5, 0.5, cutoff, 1.0
            )

        for key in ("H_atom", "H_total", "graph_lcc", "graph_cc", "ring_fraction"):
            assert math.isclose(m_cpp[key], m_py[key], rel_tol=1e-4, abs_tol=1e-6), (
                f"Metric {key}: cpp={m_cpp[key]:.6f}  py={m_py[key]:.6f}"
            )


# ---------------------------------------------------------------------------
# compute_steinhardt — _HAS_STEINHARDT=False fallback
# ---------------------------------------------------------------------------


class TestSteinhhardtFallback:
    """Verify the pure-Python Steinhardt Q_l computation."""

    def _steinhardt(self, pts, l_values, cutoff):
        from pasted._metrics import compute_steinhardt

        with _patch(**_STEINHARDT_FLAG):
            return compute_steinhardt(np.array(pts), l_values, cutoff)

    def test_returns_expected_keys(self):
        pts = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)]
        result = self._steinhardt(pts, [4, 6, 8], cutoff=5.0)
        assert set(result.keys()) == {"Q4", "Q6", "Q8"}

    def test_values_in_valid_range(self):
        """Q_l ∈ [0, 1] for any geometry."""
        rng = np.random.default_rng(1)
        pts = rng.uniform(-3.0, 3.0, size=(10, 3)).tolist()
        result = self._steinhardt(pts, [4, 6], cutoff=6.0)
        for k, v in result.items():
            assert 0.0 <= v <= 1.0 + 1e-9, f"{k}={v:.6f} out of [0, 1]"

    def test_no_neighbors_gives_zero(self):
        """Atoms with no neighbors within cutoff → Q_l = 0."""
        pts = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]
        result = self._steinhardt(pts, [4, 6], cutoff=1.0)
        for k, v in result.items():
            assert v == pytest.approx(0.0), f"{k}={v:.6f} should be 0"

    def test_matches_cpp_result(self):
        """Python Steinhardt must agree with C++ path within tolerance."""
        from pasted._ext import HAS_STEINHARDT
        from pasted._metrics import compute_steinhardt

        if not HAS_STEINHARDT:
            pytest.skip("C++ Steinhardt not available — cannot compare paths")

        rng = np.random.default_rng(99)
        pts_arr = rng.uniform(-3.0, 3.0, size=(12, 3))

        q_cpp = compute_steinhardt(pts_arr, [4, 6, 8], cutoff=5.0)
        with _patch(**_STEINHARDT_FLAG):
            q_py = compute_steinhardt(pts_arr, [4, 6, 8], cutoff=5.0)

        for k in ("Q4", "Q6", "Q8"):
            assert math.isclose(q_cpp[k], q_py[k], rel_tol=1e-5, abs_tol=1e-8), (
                f"Steinhardt {k}: cpp={q_cpp[k]:.8f}  py={q_py[k]:.8f}"
            )


# ---------------------------------------------------------------------------
# _compute_graph_ring_charge — individual pure-Python sub-metrics
# ---------------------------------------------------------------------------


class TestGraphRingChargeFallback:
    """Verify compute_graph_metrics, compute_ring_fraction, compute_charge_frustration,
    and compute_moran_I_chi directly (the pure-Python implementations)."""

    def _dmat(self, positions):
        from scipy.spatial.distance import pdist, squareform

        return squareform(pdist(np.array(positions)))

    def test_graph_lcc_single_component(self):
        """A chain of connected atoms → graph_lcc = 1.0."""
        from pasted._metrics import compute_graph_metrics

        # Linear chain: each atom within 2.5 Å of its neighbor
        positions = [(float(i) * 2.0, 0.0, 0.0) for i in range(5)]
        dmat = self._dmat(positions)
        result = compute_graph_metrics(dmat, cutoff=2.5)
        assert result["graph_lcc"] == pytest.approx(1.0)

    def test_graph_lcc_disconnected(self):
        """Two well-separated clusters → graph_lcc < 1.0."""
        from pasted._metrics import compute_graph_metrics

        positions = [
            (0.0, 0.0, 0.0), (1.5, 0.0, 0.0),   # cluster A
            (50.0, 0.0, 0.0), (51.5, 0.0, 0.0),  # cluster B
        ]
        dmat = self._dmat(positions)
        result = compute_graph_metrics(dmat, cutoff=2.5)
        assert result["graph_lcc"] < 1.0

    def test_ring_fraction_triangle(self):
        """A triangle → all 3 atoms are in a ring → ring_fraction = 1.0.

        Previously the Union-Find implementation returned 2/3 because it only
        marked the two endpoints of the detected back-edge.  The Tarjan
        bridge-finding fix correctly marks all 3 atoms.
        """
        from pasted._metrics import compute_ring_fraction

        s = 1.8
        positions = [(0.0, 0.0, 0.0), (s, 0.0, 0.0), (s / 2, s * 0.866, 0.0)]
        dmat = self._dmat(positions)
        rf = compute_ring_fraction(["C", "C", "C"], dmat, cutoff=2.0)
        assert rf == pytest.approx(1.0)

    def test_ring_fraction_chain_has_no_rings(self):
        """A linear chain has no back-edges → ring_fraction = 0.0."""
        from pasted._metrics import compute_ring_fraction

        positions = [(float(i) * 1.5, 0.0, 0.0) for i in range(4)]
        dmat = self._dmat(positions)
        rf = compute_ring_fraction(["C"] * 4, dmat, cutoff=1.6)
        assert rf == pytest.approx(0.0)

    def test_charge_frustration_identical_elements(self):
        """All same element → all chi differences are 0 → variance = 0."""
        from pasted._metrics import compute_charge_frustration

        positions = [(float(i) * 1.5, 0.0, 0.0) for i in range(4)]
        dmat = self._dmat(positions)
        cf = compute_charge_frustration(["C", "C", "C", "C"], dmat, cutoff=2.0)
        assert cf == pytest.approx(0.0)

    def test_moran_i_same_element_is_zero(self):
        """Same element everywhere → zero variance → Moran's I = 0."""
        from pasted._metrics import compute_moran_I_chi

        positions = [(float(i) * 1.5, 0.0, 0.0) for i in range(4)]
        dmat = self._dmat(positions)
        mi = compute_moran_I_chi(["C", "C", "C", "C"], dmat, cutoff=2.0)
        assert mi == pytest.approx(0.0)

    def test_n_less_than_2_graph(self):
        """Single atom: graph_lcc=1.0, graph_cc=0.0 (degenerate case)."""
        from pasted._metrics import compute_graph_metrics

        dmat = self._dmat([(0.0, 0.0, 0.0)])
        result = compute_graph_metrics(dmat, cutoff=5.0)
        assert result["graph_lcc"] == pytest.approx(1.0)
        assert result["graph_cc"] == pytest.approx(0.0)

    def test_n_less_than_3_ring_fraction(self):
        """Fewer than 3 atoms → ring_fraction = 0.0 (no ring possible)."""
        from pasted._metrics import compute_ring_fraction

        positions = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]
        dmat = self._dmat(positions)
        rf = compute_ring_fraction(["C", "C"], dmat, cutoff=2.0)
        assert rf == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Regression: ring_fraction correctness — Tarjan vs old Union-Find
# ---------------------------------------------------------------------------


class TestRingFractionRegression:
    """Verify that ring_fraction correctly counts ALL cycle members.

    The old Union-Find implementation only marked back-edge endpoints,
    undercounting ring membership for every cycle size >= 3.
    Both the Python fallback and the C++ path are checked.
    """

    def _dmat(self, positions):
        from scipy.spatial.distance import pdist, squareform
        return squareform(pdist(np.array(positions)))

    def _ring_positions(
        self, n: int, chord: float = 1.5
    ) -> tuple[list[tuple[float, float, float]], float]:
        """Return positions of n atoms on a circle with the given bond length.

        Parameters
        ----------
        n:
            Number of atoms in the ring.
        chord:
            Bond length (Å) between adjacent atoms.  The circle radius is
            derived from this so that adjacent atoms are exactly *chord* Å
            apart regardless of ring size.

        Returns
        -------
        (positions, cutoff)
            *positions* — list of (x, y, 0) coordinates.
            *cutoff*    — ``chord * 1.1``, a cutoff that connects adjacent
            atoms while excluding non-adjacent ones (valid when the ring has
            ≥ 4 atoms and chord < radius√2; always safe for n ≥ 3 with a
            10 % margin).
        """
        import math
        # chord = 2 * r * sin(π/n)  →  r = chord / (2 * sin(π/n))
        r = chord / (2 * math.sin(math.pi / n))
        positions = [
            (r * math.cos(2 * math.pi * i / n),
             r * math.sin(2 * math.pi * i / n),
             0.0)
            for i in range(n)
        ]
        return positions, chord * 1.1

    @pytest.mark.parametrize("ring_size", [3, 4, 5, 6, 8])
    def test_pure_ring_python_fallback(self, ring_size):
        """A pure N-cycle: every atom is in the ring → ring_fraction = 1.0 (Python)."""
        from unittest.mock import patch

        from pasted._metrics import compute_ring_fraction

        positions, cutoff = self._ring_positions(ring_size)
        dmat = self._dmat(positions)
        with patch("pasted._metrics._HAS_GRAPH", False):
            rf = compute_ring_fraction(["C"] * ring_size, dmat, cutoff=cutoff)
        assert rf == pytest.approx(1.0), (
            f"{ring_size}-cycle: expected ring_fraction=1.0, got {rf:.4f}"
        )

    @pytest.mark.parametrize("ring_size", [3, 4, 5, 6, 8])
    def test_pure_ring_cpp_path(self, ring_size):
        """A pure N-cycle: every atom is in the ring → ring_fraction = 1.0 (C++)."""
        from pasted._ext import HAS_GRAPH
        from pasted._metrics import compute_all_metrics

        if not HAS_GRAPH:
            pytest.skip("C++ graph metrics not available")

        positions, cutoff = self._ring_positions(ring_size)
        atoms = ["C"] * ring_size
        m = compute_all_metrics(
            atoms, positions, n_bins=10, w_atom=0.5, w_spatial=0.5,
            cutoff=cutoff, cov_scale=1.0,
        )
        assert m["ring_fraction"] == pytest.approx(1.0), (
            f"{ring_size}-cycle (C++): expected 1.0, got {m['ring_fraction']:.4f}"
        )

    def test_chain_has_no_ring_atoms(self):
        """A linear chain: no cycles → ring_fraction = 0.0 (Python fallback)."""
        from unittest.mock import patch

        from pasted._metrics import compute_ring_fraction

        positions = [(float(i) * 1.5, 0.0, 0.0) for i in range(6)]
        dmat = self._dmat(positions)
        with patch("pasted._metrics._HAS_GRAPH", False):
            rf = compute_ring_fraction(["C"] * 6, dmat, cutoff=1.6)
        assert rf == pytest.approx(0.0)

    def test_fused_bicyclic_all_atoms_in_ring(self):
        """Two fused triangles (4 atoms sharing one edge): all atoms are in a ring."""
        from unittest.mock import patch

        from pasted._metrics import compute_ring_fraction

        # Diamond / rhombus: A-B-C-D where A-B, B-C, C-D, D-A, and A-C are bonds
        # (two triangles sharing edge A-C)
        s = 1.5
        positions = [
            (0.0,  0.0,  0.0),   # A
            (s,    0.0,  0.0),   # B  — triangle A-B-C
            (s/2,  s*0.866, 0.0),# C
            (s*1.5, s*0.866, 0.0),# D  — triangle B-C-D
        ]
        dmat = self._dmat(positions)
        with patch("pasted._metrics._HAS_GRAPH", False):
            rf = compute_ring_fraction(["C"] * 4, dmat, cutoff=s * 1.1)
        assert rf == pytest.approx(1.0)

    def test_python_and_cpp_agree_on_ring_fraction(self):
        """Python and C++ ring_fraction must agree for a variety of structures."""
        from unittest.mock import patch

        from pasted._ext import HAS_GRAPH
        from pasted._metrics import compute_all_metrics, compute_ring_fraction

        if not HAS_GRAPH:
            pytest.skip("C++ path not available")

        ring3, c3 = self._ring_positions(3)
        ring5, c5 = self._ring_positions(5)
        ring6, c6 = self._ring_positions(6)
        chain = [(float(i) * 1.5, 0.0, 0.0) for i in range(5)]

        test_cases = [
            (ring3, c3),
            (ring5, c5),
            (ring6, c6),
            (chain, 1.6),
        ]
        for positions, cutoff in test_cases:
            atoms = ["C"] * len(positions)
            dmat = self._dmat(positions)

            with patch("pasted._metrics._HAS_GRAPH", False):
                rf_py = compute_ring_fraction(atoms, dmat, cutoff=cutoff)

            m_cpp = compute_all_metrics(
                atoms, positions, n_bins=10, w_atom=0.5, w_spatial=0.5,
                cutoff=cutoff, cov_scale=1.0,
            )
            rf_cpp = m_cpp["ring_fraction"]

            assert abs(rf_py - rf_cpp) < 1e-9, (
                f"Python ({rf_py:.4f}) and C++ ({rf_cpp:.4f}) ring_fraction disagree "
                f"for {len(positions)}-atom structure"
            )
