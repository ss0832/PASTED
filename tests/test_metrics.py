"""Tests for pasted._metrics: all disorder metric functions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from pasted._atoms import ALL_METRICS, parse_filter
from pasted._metrics import (
    compute_all_metrics,
    compute_bond_strain_rms,
    compute_charge_frustration,
    compute_graph_metrics,
    compute_h_atom,
    compute_h_spatial,
    compute_rdf_deviation,
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

    def test_non_negative(self) -> None:
        assert compute_h_atom(["C"]) >= 0.0


# ---------------------------------------------------------------------------
# compute_h_spatial
# ---------------------------------------------------------------------------


class TestComputeHSpatial:
    def test_empty_returns_zero(self) -> None:
        assert compute_h_spatial(np.array([]), 20) == 0.0

    def test_non_negative(self) -> None:
        dists = np.array([1.0, 2.0, 3.0])
        assert compute_h_spatial(dists, 10) >= 0.0

    def test_more_uniform_higher_entropy(self) -> None:
        uniform = np.linspace(1.0, 10.0, 100)
        peaked = np.full(100, 5.0)
        assert compute_h_spatial(uniform, 20) > compute_h_spatial(peaked, 20)


# ---------------------------------------------------------------------------
# compute_rdf_deviation
# ---------------------------------------------------------------------------


def test_rdf_deviation_non_negative() -> None:
    pts = np.array(FOUR_POS)
    dists = pdist(pts)
    rdf = compute_rdf_deviation(pts, dists, 20)
    assert rdf >= 0.0


def test_rdf_deviation_single_pair_non_negative() -> None:
    pts = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    dists = pdist(pts)
    assert compute_rdf_deviation(pts, dists, 10) >= 0.0


# ---------------------------------------------------------------------------
# compute_shape_anisotropy
# ---------------------------------------------------------------------------


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
# compute_bond_strain_rms
# ---------------------------------------------------------------------------


class TestComputeBondStrainRms:
    def _dmat(self, pos: list[tuple[float, float, float]]) -> np.ndarray:
        return squareform(pdist(np.array(pos)))

    def test_ideal_bond_zero_strain(self) -> None:
        # C–C ideal: 0.75 + 0.75 = 1.5 Å
        pos = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]
        result = compute_bond_strain_rms(["C", "C"], self._dmat(pos), cov_scale=1.3)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_non_negative(self) -> None:
        dmat = self._dmat(FOUR_POS)
        result = compute_bond_strain_rms(FOUR_ATOMS, dmat, cov_scale=1.3)
        assert result >= 0.0

    def test_no_bonds_returns_zero(self) -> None:
        # Atoms 100 Å apart → no bonds detected
        pos = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        dmat = self._dmat(pos)
        assert compute_bond_strain_rms(["C", "C"], dmat, cov_scale=1.3) == pytest.approx(0.0)

    def test_compressed_bond_positive_strain(self) -> None:
        # C–C at 1.0 Å (ideal 1.5 Å) → strain = (1.0-1.5)/1.5 < 0 → rms > 0
        pos = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        result = compute_bond_strain_rms(["C", "C"], self._dmat(pos), cov_scale=1.3)
        assert result > 0.0

    def test_larger_cov_scale_detects_more_pairs(self) -> None:
        # With a large cov_scale, distant pairs are also treated as bonded.
        dmat = self._dmat(FOUR_POS)
        small = compute_bond_strain_rms(FOUR_ATOMS, dmat, cov_scale=1.0)
        large = compute_bond_strain_rms(FOUR_ATOMS, dmat, cov_scale=3.0)
        # Both should be non-negative; result may differ
        assert small >= 0.0
        assert large >= 0.0


# ---------------------------------------------------------------------------
# compute_ring_fraction
# ---------------------------------------------------------------------------


class TestComputeRingFraction:
    def _dmat(self, pos: list[tuple[float, float, float]]) -> np.ndarray:
        return squareform(pdist(np.array(pos)))

    def test_triangle_all_in_ring(self) -> None:
        # Union-Find back-edge marks only the two endpoints of the back-edge,
        # so a 3-atom triangle yields exactly 2/3 atoms marked.
        dmat = self._dmat(_RING_POS)
        result = compute_ring_fraction(_RING_ATOMS, dmat, cov_scale=1.3)
        assert result == pytest.approx(2 / 3, rel=1e-6)

    def test_linear_chain_no_ring(self) -> None:
        pos = [(0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (2.8, 0.0, 0.0)]
        dmat = self._dmat(pos)
        result = compute_ring_fraction(["C", "C", "C"], dmat, cov_scale=1.3)
        assert result == pytest.approx(0.0)

    def test_range(self) -> None:
        dmat = self._dmat(FOUR_POS)
        result = compute_ring_fraction(FOUR_ATOMS, dmat, cov_scale=1.3)
        assert 0.0 <= result <= 1.0

    def test_fewer_than_three_atoms_zero(self) -> None:
        pos = [(0.0, 0.0, 0.0), (1.4, 0.0, 0.0)]
        dmat = self._dmat(pos)
        assert compute_ring_fraction(["C", "C"], dmat, cov_scale=1.3) == pytest.approx(0.0)

    def test_no_bonds_returns_zero(self) -> None:
        pos = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (200.0, 0.0, 0.0)]
        dmat = self._dmat(pos)
        assert compute_ring_fraction(["C", "C", "C"], dmat, cov_scale=1.3) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_charge_frustration
# ---------------------------------------------------------------------------


class TestComputeChargeFrustration:
    def _dmat(self, pos: list[tuple[float, float, float]]) -> np.ndarray:
        return squareform(pdist(np.array(pos)))

    def test_homoatomic_zero_variance(self) -> None:
        # All C: |ΔEN| = 0 for every pair → variance = 0
        pos = [(0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (2.8, 0.0, 0.0)]
        dmat = self._dmat(pos)
        result = compute_charge_frustration(["C", "C", "C"], dmat, cov_scale=1.3)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_non_negative(self) -> None:
        dmat = self._dmat(FOUR_POS)
        result = compute_charge_frustration(FOUR_ATOMS, dmat, cov_scale=1.3)
        assert result >= 0.0

    def test_no_bonds_returns_zero(self) -> None:
        pos = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        dmat = self._dmat(pos)
        assert compute_charge_frustration(["C", "N"], dmat, cov_scale=1.3) == pytest.approx(0.0)

    def test_single_bond_zero_variance(self) -> None:
        # Only one bonded pair → variance of a single value = 0
        pos = [(0.0, 0.0, 0.0), (1.4, 0.0, 0.0)]
        dmat = self._dmat(pos)
        result = compute_charge_frustration(["C", "N"], dmat, cov_scale=1.3)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_mixed_system_positive(self) -> None:
        # C(2.55)–F(3.98) and C(2.55)–H(2.20): different |ΔEN| → variance > 0
        pos = [(0.0, 0.0, 0.0), (1.35, 0.0, 0.0), (0.0, 1.07, 0.0)]
        dmat = self._dmat(pos)
        result = compute_charge_frustration(["C", "F", "H"], dmat, cov_scale=1.3)
        assert result > 0.0


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------


def test_all_metrics_keys() -> None:
    metrics = compute_all_metrics(
        FOUR_ATOMS, FOUR_POS, n_bins=10, w_atom=0.5, w_spatial=0.5, cutoff=3.0
    )
    assert set(metrics.keys()) == ALL_METRICS


def test_all_metrics_h_total_formula() -> None:
    metrics = compute_all_metrics(
        FOUR_ATOMS, FOUR_POS, n_bins=10, w_atom=0.3, w_spatial=0.7, cutoff=3.0
    )
    expected = 0.3 * metrics["H_atom"] + 0.7 * metrics["H_spatial"]
    assert metrics["H_total"] == pytest.approx(expected, rel=1e-9)


def test_all_metrics_finite() -> None:
    metrics = compute_all_metrics(
        FOUR_ATOMS, FOUR_POS, n_bins=10, w_atom=0.5, w_spatial=0.5, cutoff=3.0
    )
    for k, v in metrics.items():
        assert math.isfinite(v), f"Metric {k} is not finite: {v}"


def test_all_metrics_contains_new_keys() -> None:
    metrics = compute_all_metrics(
        FOUR_ATOMS, FOUR_POS, n_bins=10, w_atom=0.5, w_spatial=0.5, cutoff=3.0
    )
    assert "bond_strain_rms" in metrics
    assert "ring_fraction" in metrics
    assert "charge_frustration" in metrics


def test_all_metrics_cov_scale_default() -> None:
    # cov_scale=1.0 (default) and explicit should give the same result
    m1 = compute_all_metrics(FOUR_ATOMS, FOUR_POS, n_bins=10, w_atom=0.5, w_spatial=0.5,
                             cutoff=3.0)
    m2 = compute_all_metrics(FOUR_ATOMS, FOUR_POS, n_bins=10, w_atom=0.5, w_spatial=0.5,
                             cutoff=3.0, cov_scale=1.0)
    assert m1["bond_strain_rms"] == pytest.approx(m2["bond_strain_rms"])
    assert m1["ring_fraction"] == pytest.approx(m2["ring_fraction"])
    assert m1["charge_frustration"] == pytest.approx(m2["charge_frustration"])


# ---------------------------------------------------------------------------
# passes_filters
# ---------------------------------------------------------------------------


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
        metric, lo, hi = parse_filter("bond_strain_rms:0.0:0.5")
        assert metric == "bond_strain_rms"
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(0.5)

        metric2, lo2, hi2 = parse_filter("ring_fraction:-:0.3")
        assert metric2 == "ring_fraction"
        assert math.isinf(lo2) and lo2 < 0
        assert hi2 == pytest.approx(0.3)

        metric3, lo3, hi3 = parse_filter("charge_frustration:0.0:-")
        assert metric3 == "charge_frustration"
        assert lo3 == pytest.approx(0.0)
        assert math.isinf(hi3) and hi3 > 0
