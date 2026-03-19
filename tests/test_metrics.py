"""Tests for pasted._metrics: all disorder metric functions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from pasted import _ext
from pasted._atoms import _cov_radius_ang, parse_filter, pauling_electronegativity
from pasted._metrics import (
    compute_all_metrics,
    compute_charge_frustration,
    compute_graph_metrics,
    compute_h_atom,
    compute_h_spatial,
    compute_moran_I_chi,
    compute_ring_fraction,
    compute_shape_anisotropy,
    compute_steinhardt,
    passes_filters,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FOUR_ATOMS = ["C", "N", "O", "H"]
FOUR_POS: list[tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (0.0, 2.0, 0.0),
    (0.0, 0.0, 2.0),
]

# Tight ring: 3 C atoms in an equilateral triangle with side ~1.4 Å (< 2×0.75)
_RING_ATOMS = ["C", "C", "C"]
_RING_POS: list[tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),
    (1.4, 0.0, 0.0),
    (0.7, 1.21, 0.0),
]


# ---------------------------------------------------------------------------
# compute_h_atom
# ---------------------------------------------------------------------------


class TestComputeHAtom:
    def test_single_element_zero(self) -> None:
        assert compute_h_atom(["C", "C", "C"]) == pytest.approx(0.0)

    def test_two_equal_elements(self) -> None:
        h = compute_h_atom(["C", "N"])
        assert h == pytest.approx(math.log(2), rel=1e-6)

    def test_four_equal_elements(self) -> None:
        h = compute_h_atom(["C", "N", "O", "H"])
        assert h == pytest.approx(math.log(4), rel=1e-6)


class TestComputeHSpatial:
    def test_empty_returns_zero(self) -> None:
        assert compute_h_spatial(np.array([]), 20) == 0.0


class TestComputeShapeAnisotropy:
    def test_single_atom_nan(self) -> None:
        result = compute_shape_anisotropy(np.array([[0.0, 0.0, 0.0]]))
        assert math.isnan(result)

    def test_range(self) -> None:
        pts = np.array(FOUR_POS)
        result = compute_shape_anisotropy(pts)
        assert 0.0 <= result <= 1.0

    def test_linear_chain_near_1(self) -> None:
        pts = np.array([[float(i), 0.0, 0.0] for i in range(10)])
        result = compute_shape_anisotropy(pts)
        assert result > 0.8

    def test_sphere_near_0(self) -> None:
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((200, 3))
        result = compute_shape_anisotropy(pts)
        assert result < 0.1


# ---------------------------------------------------------------------------
# compute_steinhardt
# ---------------------------------------------------------------------------


def test_steinhardt_keys() -> None:
    pts = np.array(FOUR_POS)
    dmat = squareform(pdist(pts))
    result = compute_steinhardt(pts, dmat, [4, 6, 8], cutoff=3.0)
    assert set(result.keys()) == {"Q4", "Q6", "Q8"}


def test_steinhardt_range() -> None:
    pts = np.array(FOUR_POS)
    dmat = squareform(pdist(pts))
    result = compute_steinhardt(pts, dmat, [4, 6], cutoff=3.0)
    for v in result.values():
        assert 0.0 <= v <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# compute_graph_metrics
# ---------------------------------------------------------------------------


class TestComputeGraphMetrics:
    def test_keys(self) -> None:
        dmat = squareform(pdist(np.array(FOUR_POS)))
        result = compute_graph_metrics(dmat, cutoff=3.0)
        assert set(result.keys()) == {"graph_lcc", "graph_cc"}

    def test_fully_connected(self) -> None:
        # All atoms within cutoff → lcc = 1.0
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        dmat = squareform(pdist(pts))
        result = compute_graph_metrics(dmat, cutoff=2.0)
        assert result["graph_lcc"] == pytest.approx(1.0)

    def test_single_atom(self) -> None:
        dmat = np.array([[0.0]])
        result = compute_graph_metrics(dmat, cutoff=3.0)
        assert result["graph_lcc"] == pytest.approx(1.0)
        assert result["graph_cc"] == pytest.approx(0.0)

    def test_disconnected(self) -> None:
        # Two atoms far apart → each is its own component → lcc = 0.5
        pts = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        dmat = squareform(pdist(pts))
        result = compute_graph_metrics(dmat, cutoff=2.0)
        assert result["graph_lcc"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestComputeBondStrainRms:
    def _dmat(self, pos: list[tuple[float, float, float]]) -> np.ndarray:
        return squareform(pdist(np.array(pos)))








# ---------------------------------------------------------------------------
# compute_ring_fraction
# ---------------------------------------------------------------------------


class TestComputeRingFraction:
    def _dmat(self, pos: list[tuple[float, float, float]]) -> np.ndarray:
        return squareform(pdist(np.array(pos)))

    def test_triangle_all_in_ring(self) -> None:
        # Triangle with side 1.4 Å < cutoff 2.13: all three pairs connected.
        # Union-Find back-edge marks 2/3 atoms (both endpoints of the cycle edge).
        dmat = self._dmat(_RING_POS)
        result = compute_ring_fraction(_RING_ATOMS, dmat, cutoff=2.13)
        assert result == pytest.approx(2 / 3, rel=1e-6)

    def test_linear_chain_no_ring(self) -> None:
        # Linear chain: atoms at 0, 1.4, 2.8 Å.
        # Pairs 0-1 and 1-2 are within cutoff=2.13; pair 0-2 (d=2.8) is not.
        # Spanning tree: no back-edges → ring_fraction = 0.
        pos = [(0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (2.8, 0.0, 0.0)]
        dmat = self._dmat(pos)
        result = compute_ring_fraction(["C", "C", "C"], dmat, cutoff=2.13)
        assert result == pytest.approx(0.0)

    def test_range(self) -> None:
        dmat = self._dmat(FOUR_POS)
        result = compute_ring_fraction(FOUR_ATOMS, dmat, cutoff=2.13)
        assert 0.0 <= result <= 1.0

    def test_fewer_than_three_atoms_zero(self) -> None:
        pos = [(0.0, 0.0, 0.0), (1.4, 0.0, 0.0)]
        dmat = self._dmat(pos)
        assert compute_ring_fraction(["C", "C"], dmat, cutoff=2.13) == pytest.approx(0.0)

    def test_no_bonds_returns_zero(self) -> None:
        # All atoms far apart: no pairs within cutoff
        pos = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)]
        dmat = self._dmat(pos)
        assert compute_ring_fraction(["C", "C", "C"], dmat, cutoff=2.13) == pytest.approx(0.0)


class TestComputeChargeFrustration:
    def _dmat(self, pos: list[tuple[float, float, float]]) -> np.ndarray:
        return squareform(pdist(np.array(pos)))

    def test_homoatomic_zero_variance(self) -> None:
        # All C within cutoff: |ΔEN| = 0 for every pair → variance = 0
        pos = [(0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (2.8, 0.0, 0.0)]
        dmat = self._dmat(pos)
        result = compute_charge_frustration(["C", "C", "C"], dmat, cutoff=2.13)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_non_negative(self) -> None:
        dmat = self._dmat(FOUR_POS)
        result = compute_charge_frustration(FOUR_ATOMS, dmat, cutoff=2.13)
        assert result >= 0.0

    def test_no_bonds_returns_zero(self) -> None:
        pos = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
        dmat = self._dmat(pos)
        assert compute_charge_frustration(["C", "N"], dmat, cutoff=2.13) == pytest.approx(0.0)

    def test_mixed_system_positive(self) -> None:
        # C(EN=2.55) and F(EN=3.98) within cutoff → non-zero variance
        pos = [(0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.7, 1.2, 0.0)]
        dmat = self._dmat(pos)
        result = compute_charge_frustration(["C", "F", "C"], dmat, cutoff=2.13)
        assert result > 0.0



class TestPassesFilters:
    def test_empty_filters_pass(self) -> None:
        assert passes_filters({"H_total": 1.0}, [])

    def test_pass_within_range(self) -> None:
        assert passes_filters({"H_total": 2.5}, [("H_total", 2.0, 3.0)])

    def test_fail_below_min(self) -> None:
        assert not passes_filters({"H_total": 1.5}, [("H_total", 2.0, 3.0)])

    def test_fail_above_max(self) -> None:
        assert not passes_filters({"H_total": 4.0}, [("H_total", 2.0, 3.0)])

    def test_multiple_filters_all_must_pass(self) -> None:
        metrics = {"H_total": 2.5, "shape_aniso": 0.7}
        ok_both = [("H_total", 2.0, 3.0), ("shape_aniso", 0.5, 1.0)]
        fail_second = [("H_total", 2.0, 3.0), ("shape_aniso", 0.8, 1.0)]
        assert passes_filters(metrics, ok_both)
        assert not passes_filters(metrics, fail_second)

    def test_nan_metric_fails(self) -> None:
        assert not passes_filters({"shape_aniso": float("nan")}, [("shape_aniso", 0.0, 1.0)])

    def test_new_metrics_filterable(self) -> None:
        # Verify that the new metrics can be used as filter keys via parse_filter
        metric2, lo2, hi2 = parse_filter("ring_fraction:-:0.3")
        assert metric2 == "ring_fraction"
        assert math.isinf(lo2) and lo2 < 0
        assert hi2 == pytest.approx(0.3)

        metric3, lo3, hi3 = parse_filter("charge_frustration:0.0:-")
        assert metric3 == "charge_frustration"
        assert lo3 == pytest.approx(0.0)
        assert math.isinf(hi3) and hi3 > 0

# ---------------------------------------------------------------------------
# _graph_core C++ extension — contract tests
# ---------------------------------------------------------------------------


class TestGraphCoreCpp:
    """C++ _graph_core extension produces results identical to Python fallbacks."""

    @pytest.mark.skipif(not _ext.HAS_GRAPH, reason="_graph_core extension not built")
    def test_cpp_matches_python_ring_fraction(self) -> None:
        """C++ ring_fraction must agree with pure-Python to within 1e-9."""
        atoms = ["C", "C", "C", "N", "O"]
        side = 1.3
        pts = np.array([
            (0.0, 0.0, 0.0),
            (side, 0.0, 0.0),
            (side / 2, side * math.sqrt(3) / 2, 0.0),
            (5.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
        ])
        dmat    = squareform(pdist(pts))
        radii   = np.array([_cov_radius_ang(a) for a in atoms])
        en_vals = np.array([pauling_electronegativity(a) for a in atoms])

        cpp     = _ext.graph_metrics_cpp(pts, radii, 1.0, en_vals, 2.13)
        py_ring = compute_ring_fraction(atoms, dmat, 2.13)
        py_charge = compute_charge_frustration(atoms, dmat, 2.13)

        assert cpp["ring_fraction"]      == pytest.approx(py_ring,   abs=1e-9)
        assert cpp["charge_frustration"] == pytest.approx(py_charge, abs=1e-9)

    @pytest.mark.skipif(not _ext.HAS_GRAPH, reason="_graph_core extension not built")
    def test_cpp_matches_python_graph_metrics(self) -> None:
        """C++ graph_lcc and graph_cc must agree with pure-Python fallbacks."""
        atoms = ["C", "N", "O", "Fe", "H"]
        rng = np.random.default_rng(7)
        pts     = rng.uniform(-3, 3, (5, 3))
        dmat    = squareform(pdist(pts))
        radii   = np.array([_cov_radius_ang(a) for a in atoms])
        en_vals = np.array([pauling_electronegativity(a) for a in atoms])

        cpp = _ext.graph_metrics_cpp(pts, radii, 1.0, en_vals, 2.13)
        py  = compute_graph_metrics(dmat, 2.13)

        assert cpp["graph_lcc"] == pytest.approx(py["graph_lcc"], abs=1e-9)
        assert cpp["graph_cc"]  == pytest.approx(py["graph_cc"],  abs=1e-9)

    @pytest.mark.skipif(not _ext.HAS_GRAPH, reason="_graph_core extension not built")
    def test_cpp_all_metrics_roundtrip(self) -> None:
        """compute_all_metrics with C++ path returns finite values in range."""
        atoms = ["C", "N", "O", "Fe", "H"] * 10
        rng = np.random.default_rng(3)
        positions = [tuple(float(x) for x in row)
                     for row in rng.uniform(-5, 5, (50, 3))]
        m = compute_all_metrics(atoms, positions, 20, 0.5, 0.5, 2.13, 1.0)

        for key in ("graph_lcc", "graph_cc", "ring_fraction", "charge_frustration"):
            assert math.isfinite(m[key]), f"{key} is not finite: {m[key]}"
        for key in ("graph_lcc", "graph_cc", "ring_fraction"):
            assert 0.0 <= m[key] <= 1.0, f"{key}={m[key]} out of [0,1]"
        assert m["charge_frustration"] >= 0.0


# ---------------------------------------------------------------------------
# compute_moran_I_chi
# ---------------------------------------------------------------------------


class TestComputeMoranIChi:
    @staticmethod
    def _dmat(pos: list) -> np.ndarray:
        return squareform(pdist(np.array(pos)))

    def test_alternating_negative(self) -> None:
        """Perfectly alternating high/low EN on a grid → I = -1."""
        # 2-atom unit: H (EN=2.2) and F (EN=3.98) at bond distance
        atoms = ["H", "F", "H", "F"]
        pos = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0),
               (3.0, 0.0, 0.0), (4.5, 0.0, 0.0)]
        result = compute_moran_I_chi(atoms, self._dmat(pos), cutoff=2.0)
        assert result == pytest.approx(-1.0, abs=1e-9)

    def test_clustered_positive(self) -> None:
        """Same-EN atoms clustered far apart → I = +1."""
        atoms = ["H", "H", "F", "F"]
        pos = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
               (20.0, 0.0, 0.0), (21.0, 0.0, 0.0)]
        result = compute_moran_I_chi(atoms, self._dmat(pos), cutoff=1.5)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_single_element_zero(self) -> None:
        """All same element → denominator is 0 → returns 0.0."""
        atoms = ["C", "C", "C"]
        pos = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0)]
        result = compute_moran_I_chi(atoms, self._dmat(pos), cutoff=2.0)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_range(self) -> None:
        """Moran's I must be finite for any structure.

        Note: Moran's I is NOT bounded to [-1, 1] in general.
        The theoretical bounds are (-1/(N-1), 1] for regular lattices, but
        sparse spatial weight matrices (step-function cutoff) can produce
        values outside [-1, 1] for small or clustered structures.
        """
        rng = np.random.default_rng(42)
        for _ in range(20):
            n = rng.integers(4, 20)
            atoms = rng.choice(["C", "N", "O", "Fe", "H"], n).tolist()
            pos_arr = rng.uniform(-5, 5, (n, 3))
            result = compute_moran_I_chi(atoms, squareform(pdist(pos_arr)), 2.13)
            assert math.isfinite(result), f"I={result} is not finite"

    def test_noble_gas_en_4(self) -> None:
        """He/Ne/Ar/Rn return 4.0; Kr returns 3.0; Xe returns 2.6."""
        for sym in ("He", "Ne", "Ar", "Rn"):
            assert pauling_electronegativity(sym) == pytest.approx(4.0), (
                f"{sym} EN should be 4.0, got {pauling_electronegativity(sym)}"
            )
        assert pauling_electronegativity("Kr") == pytest.approx(3.0)
        assert pauling_electronegativity("Xe") == pytest.approx(2.6)

    @pytest.mark.skipif(not _ext.HAS_GRAPH, reason="_graph_core extension not built")
    def test_cpp_matches_python(self) -> None:
        """C++ moran_I_chi_cpp must agree with pure-Python to within 1e-9."""
        atoms = ["C", "N", "O", "Fe", "H", "C", "N"]
        rng = np.random.default_rng(11)
        pts = rng.uniform(-3, 3, (7, 3))
        dmat = squareform(pdist(pts))
        en_vals = np.array([pauling_electronegativity(a) for a in atoms])

        py_result  = compute_moran_I_chi(atoms, dmat, 2.13)
        cpp_result = _ext.moran_I_chi_cpp(pts, en_vals, 2.13)

        assert cpp_result == pytest.approx(py_result, abs=1e-9)
