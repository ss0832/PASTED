"""Tests for pasted._metrics: all disorder metric functions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from pasted._atoms import ALL_METRICS
from pasted._metrics import (
    compute_all_metrics,
    compute_graph_metrics,
    compute_h_atom,
    compute_h_spatial,
    compute_rdf_deviation,
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
