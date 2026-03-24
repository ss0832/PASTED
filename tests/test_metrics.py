"""Tests for pasted._metrics: all disorder metric functions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from pasted import _ext
from pasted._atoms import cov_radius_ang as _cov_radius_ang
from pasted._atoms import parse_filter, pauling_electronegativity
from pasted._metrics import (
    _compute_adversarial,
    _compute_bond_angle_entropy,
    _compute_coordination_variance,
    _compute_local_anisotropy,
    _compute_radial_variance,
    _steinhardt_per_atom_sparse,
    compute_all_metrics,
    compute_charge_frustration,
    compute_graph_metrics,
    compute_h_atom,
    compute_h_spatial,
    compute_moran_I_chi,
    compute_rdf_deviation,
    compute_ring_fraction,
    compute_shape_anisotropy,
    compute_steinhardt,
    compute_steinhardt_per_atom,
    passes_filters,
)
from pasted.neighbor_list import NeighborList

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

# Tight ring: 3 C atoms in an equilateral triangle with side ~1.4 A (< 2x0.75)
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


# ---------------------------------------------------------------------------
# compute_h_spatial  (new signature: pts, cutoff, n_bins)
# ---------------------------------------------------------------------------


class TestComputeHSpatial:
    def test_single_atom_zero(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0]])
        assert compute_h_spatial(pts, cutoff=3.0, n_bins=20) == 0.0

    def test_no_pairs_within_cutoff_zero(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        assert compute_h_spatial(pts, cutoff=3.0, n_bins=20) == 0.0

    def test_returns_non_negative(self) -> None:
        pts = np.array(FOUR_POS)
        result = compute_h_spatial(pts, cutoff=3.0, n_bins=20)
        assert result >= 0.0

    def test_more_uniform_higher_entropy(self) -> None:
        """Uniformly spaced atoms should have higher h_spatial than clustered."""
        rng = np.random.default_rng(42)
        uniform = rng.uniform(-5.0, 5.0, (30, 3))
        clustered = rng.normal(0.0, 0.3, (30, 3))
        h_uniform = compute_h_spatial(uniform, cutoff=4.0, n_bins=20)
        h_clustered = compute_h_spatial(clustered, cutoff=4.0, n_bins=20)
        assert h_uniform >= h_clustered


# ---------------------------------------------------------------------------
# compute_rdf_deviation  (new signature: pts, cutoff, n_bins)
# ---------------------------------------------------------------------------


class TestComputeRdfDeviation:
    def test_single_atom_zero(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0]])
        assert compute_rdf_deviation(pts, cutoff=3.0, n_bins=20) == 0.0

    def test_no_pairs_within_cutoff_zero(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        assert compute_rdf_deviation(pts, cutoff=3.0, n_bins=20) == 0.0

    def test_non_negative(self) -> None:
        pts = np.array(FOUR_POS)
        result = compute_rdf_deviation(pts, cutoff=3.0, n_bins=20)
        assert result >= 0.0

    def test_ideal_gas_near_zero(self) -> None:
        """Uniform random cloud inside a sphere should have low RDF_dev.

        ``compute_rdf_deviation`` normalizes by the spherical volume defined
        by ``r_bound`` (maximum distance from the centroid).  This assumption
        holds when atoms are drawn from a sphere -- which is exactly how
        PASTED generates structures with ``region='sphere:R'``.  A uniform
        cubic distribution would inflate ``r_bound`` and raise RDF_dev.
        """
        rng = np.random.default_rng(0)
        n = 1000
        # Uniform distribution inside a sphere of radius 20 A
        u = rng.uniform(0.0, 1.0, n)
        theta = np.arccos(1.0 - 2.0 * rng.uniform(size=n))
        phi = rng.uniform(0.0, 2.0 * math.pi, n)
        radii_ = 20.0 * u ** (1.0 / 3.0)
        pts = np.column_stack(
            [
                radii_ * np.sin(theta) * np.cos(phi),
                radii_ * np.sin(theta) * np.sin(phi),
                radii_ * np.cos(theta),
            ]
        )
        result = compute_rdf_deviation(pts, cutoff=8.0, n_bins=20)
        assert result < 0.5


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
# compute_steinhardt  (new signature: pts, l_values, cutoff -- no dmat)
# ---------------------------------------------------------------------------


def test_steinhardt_keys() -> None:
    pts = np.array(FOUR_POS)
    result = compute_steinhardt(pts, [4, 6, 8], cutoff=3.0)
    assert set(result.keys()) == {"Q4", "Q6", "Q8"}


def test_steinhardt_range() -> None:
    pts = np.array(FOUR_POS)
    result = compute_steinhardt(pts, [4, 6], cutoff=3.0)
    for v in result.values():
        assert 0.0 <= v <= 1.0 + 1e-9


def test_steinhardt_fcc_theoretical() -> None:
    """FCC 12-neighbor shell: Q4 ~ 0.1909, Q6 ~ 0.5745."""
    a = 2.87
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [a / 2, a / 2, 0.0],
            [-a / 2, a / 2, 0.0],
            [a / 2, -a / 2, 0.0],
            [-a / 2, -a / 2, 0.0],
            [a / 2, 0.0, a / 2],
            [-a / 2, 0.0, a / 2],
            [a / 2, 0.0, -a / 2],
            [-a / 2, 0.0, -a / 2],
            [0.0, a / 2, a / 2],
            [0.0, -a / 2, a / 2],
            [0.0, a / 2, -a / 2],
            [0.0, -a / 2, -a / 2],
        ]
    )
    per_atom = compute_steinhardt_per_atom(pts, [4, 6], cutoff=3.0)
    assert per_atom["Q4"][0] == pytest.approx(0.19094, abs=1e-4)
    assert per_atom["Q6"][0] == pytest.approx(0.57452, abs=1e-4)


# ---------------------------------------------------------------------------
# compute_graph_metrics  (Python fallback -- still takes dmat)
# ---------------------------------------------------------------------------


class TestComputeGraphMetrics:
    def test_keys(self) -> None:
        dmat = squareform(pdist(np.array(FOUR_POS)))
        result = compute_graph_metrics(dmat, cutoff=3.0)
        assert set(result.keys()) == {"graph_lcc", "graph_cc"}

    def test_fully_connected(self) -> None:
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
        pts = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        dmat = squareform(pdist(pts))
        result = compute_graph_metrics(dmat, cutoff=2.0)
        assert result["graph_lcc"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_ring_fraction
# ---------------------------------------------------------------------------


class TestComputeRingFraction:
    def _dmat(self, pos: list[tuple[float, float, float]]) -> np.ndarray:
        return squareform(pdist(np.array(pos)))

    def test_triangle_all_in_ring(self) -> None:
        # All 3 atoms of a triangle must be ring members.
        # The old Union-Find implementation returned 2/3 (only back-edge
        # endpoints were marked).  The Tarjan bridge-finding fix correctly
        # returns 1.0 for all ring sizes.
        dmat = self._dmat(_RING_POS)
        result = compute_ring_fraction(_RING_ATOMS, dmat, cutoff=2.13)
        assert result == pytest.approx(1.0)

    def test_linear_chain_no_ring(self) -> None:
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
        pos = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)]
        dmat = self._dmat(pos)
        assert compute_ring_fraction(["C", "C", "C"], dmat, cutoff=2.13) == pytest.approx(0.0)


class TestComputeChargeFrustration:
    def _dmat(self, pos: list[tuple[float, float, float]]) -> np.ndarray:
        return squareform(pdist(np.array(pos)))

    def test_homoatomic_zero_variance(self) -> None:
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
        metric2, lo2, hi2 = parse_filter("ring_fraction:-:0.3")
        assert metric2 == "ring_fraction"
        assert math.isinf(lo2)
        assert lo2 < 0
        assert hi2 == pytest.approx(0.3)

        metric3, lo3, hi3 = parse_filter("charge_frustration:0.0:-")
        assert metric3 == "charge_frustration"
        assert lo3 == pytest.approx(0.0)
        assert math.isinf(hi3)
        assert hi3 > 0


# ---------------------------------------------------------------------------
# _graph_core C++ extension -- contract tests
# ---------------------------------------------------------------------------


class TestGraphCoreCpp:
    """C++ _graph_core extension produces results consistent with Python fallbacks."""

    @pytest.mark.skipif(not _ext.HAS_GRAPH, reason="_graph_core extension not built")
    def test_cpp_matches_python_ring_fraction(self) -> None:
        """C++ ring_fraction must agree with pure-Python to within 1e-9."""
        atoms = ["C", "C", "C", "N", "O"]
        side = 1.3
        pts = np.array(
            [
                (0.0, 0.0, 0.0),
                (side, 0.0, 0.0),
                (side / 2, side * math.sqrt(3) / 2, 0.0),
                (5.0, 0.0, 0.0),
                (10.0, 0.0, 0.0),
            ]
        )
        dmat = squareform(pdist(pts))
        radii = np.array([_cov_radius_ang(a) for a in atoms])
        en_vals = np.array([pauling_electronegativity(a) for a in atoms])

        cpp = _ext.graph_metrics_cpp(pts, radii, 1.0, en_vals, 2.13)
        py_ring = compute_ring_fraction(atoms, dmat, 2.13)
        py_charge = compute_charge_frustration(atoms, dmat, 2.13)

        assert cpp["ring_fraction"] == pytest.approx(py_ring, abs=1e-9)
        assert cpp["charge_frustration"] == pytest.approx(py_charge, abs=1e-9)

    @pytest.mark.skipif(not _ext.HAS_GRAPH, reason="_graph_core extension not built")
    def test_cpp_matches_python_graph_metrics(self) -> None:
        """C++ graph_lcc and graph_cc must agree with pure-Python fallbacks."""
        atoms = ["C", "N", "O", "Fe", "H"]
        rng = np.random.default_rng(7)
        pts = rng.uniform(-3, 3, (5, 3))
        dmat = squareform(pdist(pts))
        radii = np.array([_cov_radius_ang(a) for a in atoms])
        en_vals = np.array([pauling_electronegativity(a) for a in atoms])

        cpp = _ext.graph_metrics_cpp(pts, radii, 1.0, en_vals, 2.13)
        py = compute_graph_metrics(dmat, 2.13)

        assert cpp["graph_lcc"] == pytest.approx(py["graph_lcc"], abs=1e-9)
        assert cpp["graph_cc"] == pytest.approx(py["graph_cc"], abs=1e-9)

    @pytest.mark.skipif(not _ext.HAS_GRAPH, reason="_graph_core extension not built")
    def test_rdf_h_cpp_non_negative(self) -> None:
        """rdf_h_cpp must return finite non-negative values."""
        rng = np.random.default_rng(5)
        pts = rng.uniform(-5.0, 5.0, (50, 3))
        result = dict(_ext.rdf_h_cpp(pts, 4.0, 20))
        assert math.isfinite(result["h_spatial"])
        assert math.isfinite(result["rdf_dev"])
        assert result["h_spatial"] >= 0.0
        assert result["rdf_dev"] >= 0.0

    @pytest.mark.skipif(not _ext.HAS_GRAPH, reason="_graph_core extension not built")
    def test_rdf_h_cpp_matches_python(self) -> None:
        """rdf_h_cpp h_spatial must agree with Python cKDTree fallback."""
        rng = np.random.default_rng(99)
        pts = rng.uniform(-8.0, 8.0, (80, 3))
        cutoff = 4.0
        cpp = dict(_ext.rdf_h_cpp(pts, cutoff, 20))
        py_h = compute_h_spatial(pts, cutoff, 20)
        py_rdf = compute_rdf_deviation(pts, cutoff, 20)
        assert cpp["h_spatial"] == pytest.approx(py_h, abs=1e-9)
        assert cpp["rdf_dev"] == pytest.approx(py_rdf, abs=1e-9)

    @pytest.mark.skipif(not _ext.HAS_GRAPH, reason="_graph_core extension not built")
    def test_cpp_all_metrics_roundtrip(self) -> None:
        """compute_all_metrics with C++ path returns finite values in range."""
        atoms = ["C", "N", "O", "Fe", "H"] * 10
        rng = np.random.default_rng(3)
        positions = [tuple(float(x) for x in row) for row in rng.uniform(-5, 5, (50, 3))]  # type: ignore[arg-type]
        m = compute_all_metrics(atoms, positions, 20, 0.5, 0.5, 2.13, 1.0)  # type: ignore[arg-type]

        for key in (
            "graph_lcc",
            "graph_cc",
            "ring_fraction",
            "charge_frustration",
            "H_spatial",
            "RDF_dev",
        ):
            assert math.isfinite(m[key]), f"{key} is not finite: {m[key]}"
        for key in ("graph_lcc", "graph_cc", "ring_fraction"):
            assert 0.0 <= m[key] <= 1.0, f"{key}={m[key]} out of [0,1]"
        assert m["charge_frustration"] >= 0.0
        assert m["H_spatial"] >= 0.0
        assert m["RDF_dev"] >= 0.0


# ---------------------------------------------------------------------------
# compute_moran_I_chi
# ---------------------------------------------------------------------------


class TestComputeMoranIChi:
    @staticmethod
    def _dmat(pos: list) -> np.ndarray:
        return squareform(pdist(np.array(pos)))

    def test_alternating_negative(self) -> None:
        """Perfectly alternating high/low EN on a grid -> I = -1."""
        atoms = ["H", "F", "H", "F"]
        pos = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0), (4.5, 0.0, 0.0)]
        result = compute_moran_I_chi(atoms, self._dmat(pos), cutoff=2.0)
        assert result == pytest.approx(-1.0, abs=1e-9)

    def test_clustered_positive(self) -> None:
        """Same-EN atoms clustered far apart -> I = +1."""
        atoms = ["H", "H", "F", "F"]
        pos = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (20.0, 0.0, 0.0), (21.0, 0.0, 0.0)]
        result = compute_moran_I_chi(atoms, self._dmat(pos), cutoff=1.5)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_single_element_zero(self) -> None:
        """All same element -> denominator is 0 -> returns 0.0."""
        atoms = ["C", "C", "C"]
        pos = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0)]
        result = compute_moran_I_chi(atoms, self._dmat(pos), cutoff=2.0)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_range(self) -> None:
        """Moran's I must be finite for any structure."""
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

        py_result = compute_moran_I_chi(atoms, dmat, 2.13)
        cpp_result = _ext.moran_I_chi_cpp(pts, en_vals, 2.13)

        assert cpp_result == pytest.approx(py_result, abs=1e-9)

    def test_moran_I_chi_never_exceeds_one(self) -> None:
        """Regression test for v0.3.8 bug: moran_I_chi must be <= 1.0.

        With binary (0/1) weights and a very sparse cutoff graph (W < N),
        the raw formula (n/W) * numer/denom can exceed +1 because the
        n/W prefactor is > 1.  Both the Python fallback and the C++ path
        must clamp the result to 1.0 from above.
        """
        import warnings

        from pasted import generate

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = generate(
                n_atoms=12,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:8",
                elements="6,7,8,16",
                n_samples=30,
                seed=42,
            )

        for s in result:
            mI = s.metrics["moran_I_chi"]
            assert mI <= 1.0 + 1e-12, (
                f"moran_I_chi={mI:.6f} > 1.0 (regression: v0.3.8 clamp not applied)"
            )

    @pytest.mark.skipif(not _ext.HAS_GRAPH, reason="_graph_core extension not built")
    def test_moran_I_chi_clamp_cpp(self) -> None:
        """C++ moran_I_chi is clamped to <= 1.0 on a deliberately sparse graph."""
        # Two isolated atoms of very different EN, cutoff just large enough
        # to connect them: W=2, N=2 → n/W=1 (no inflation here).
        # Force W < N by using 3 atoms but a cutoff that only connects 2 of them.
        atoms = ["C", "N", "O"]  # EN: 2.55, 3.04, 3.44
        pts = np.array(
            [
                [0, 0, 0],
                [1.0, 0, 0],  # C-N: 1.0 Å — within cutoff
                [9.9, 0, 0],
            ],  # O far away — outside cutoff
            dtype=float,
        )
        en_vals = np.array([pauling_electronegativity(a) for a in atoms])
        # W=2 (one directed edge each way), N=3 → n/W=1.5 → raw may exceed 1
        result = _ext.moran_I_chi_cpp(pts, en_vals, cutoff=2.0)
        assert result <= 1.0 + 1e-12, f"C++ moran_I_chi={result:.6f} > 1.0 after v0.3.8 clamp"


class TestSteinhardtFastPath:
    """Regression tests for ④ real-SH hardcoded fast-path (v0.3.8)."""

    @pytest.mark.skipif(not _ext.HAS_STEINHARDT, reason="_steinhardt_core not built")
    @pytest.mark.parametrize("n", [5, 50, 200, 1000])
    def test_fast_path_matches_python_sparse(self, n: int) -> None:
        """④ fast-path [4,6,8] must agree with Python sparse to atol=1e-12."""
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((n, 3)) * 2.5
        fast = compute_steinhardt(pts, [4, 6, 8], 3.5)
        ref_pa = _steinhardt_per_atom_sparse(pts, [4, 6, 8], 3.5)
        for l_val in [4, 6, 8]:
            key = f"Q{l_val}"
            ref_val = float(np.mean(ref_pa[key]))
            assert fast[key] == pytest.approx(ref_val, abs=1e-12), (
                f"N={n} {key}: fast={fast[key]:.10f} ref={ref_val:.10f}"
            )

    @pytest.mark.skipif(not _ext.HAS_STEINHARDT, reason="_steinhardt_core not built")
    def test_fast_path_vs_generic_order(self) -> None:
        """[4,6,8] fast-path and [6,4,8] generic path must give same Q values."""
        rng = np.random.default_rng(7)
        pts = rng.standard_normal((300, 3)) * 2.5
        fast = compute_steinhardt(pts, [4, 6, 8], 3.5)
        generic = compute_steinhardt(pts, [6, 4, 8], 3.5)
        for key in ["Q4", "Q6", "Q8"]:
            assert fast[key] == pytest.approx(generic[key], abs=1e-12), (
                f"{key}: fast={fast[key]:.10f} generic={generic[key]:.10f}"
            )


# ---------------------------------------------------------------------------
# Adversarial metrics (v0.4.0): shared fixtures
# ---------------------------------------------------------------------------

# Tetrahedral-ish cluster: 4 atoms, every pair within cutoff=2.5
_ADV_PTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 1.5],
    ]
)
_ADV_CUTOFF = 2.5


# ---------------------------------------------------------------------------
# _compute_bond_angle_entropy
# ---------------------------------------------------------------------------


class TestComputeBondAngleEntropy:
    """Tests for _compute_bond_angle_entropy (v0.4.0 adversarial metric)."""

    def test_no_pairs_returns_zero(self) -> None:
        """n_pairs == 0 must short-circuit to 0.0."""
        nl = NeighborList(_ADV_PTS, cutoff=0.1)
        assert nl.n_pairs == 0
        assert _compute_bond_angle_entropy(nl) == pytest.approx(0.0)

    def test_returns_non_negative(self) -> None:
        """Result must be >= 0 for any valid structure."""
        nl = NeighborList(_ADV_PTS, cutoff=_ADV_CUTOFF)
        assert _compute_bond_angle_entropy(nl) >= 0.0

    def test_range_upper_bound(self) -> None:
        """Result must be <= ln(36) (maximum entropy for 36-bin histogram)."""
        nl = NeighborList(_ADV_PTS, cutoff=_ADV_CUTOFF)
        assert _compute_bond_angle_entropy(nl) <= math.log(36) + 1e-9

    def test_numpy_fallback_matches_cpp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pure-NumPy fallback must agree with C++ path to within 1e-9."""
        import pasted._metrics as mod

        nl = NeighborList(_ADV_PTS, cutoff=_ADV_CUTOFF)
        cpp_result = _compute_bond_angle_entropy(nl)

        monkeypatch.setattr(mod, "_HAS_BA_CPP", False)
        numpy_result = _compute_bond_angle_entropy(nl)

        assert numpy_result == pytest.approx(cpp_result, abs=1e-9)

    def test_numpy_fallback_no_pairs_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NumPy fallback with no pairs must return 0.0."""
        import pasted._metrics as mod

        monkeypatch.setattr(mod, "_HAS_BA_CPP", False)
        nl = NeighborList(_ADV_PTS, cutoff=0.1)
        assert _compute_bond_angle_entropy(nl) == pytest.approx(0.0)

    def test_numpy_fallback_all_atoms_have_one_neighbor_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Atoms with fewer than 2 neighbors are skipped; empty entropies list returns 0.0."""
        import pasted._metrics as mod

        monkeypatch.setattr(mod, "_HAS_BA_CPP", False)
        # Two atoms connected to each other only -> each has degree 1
        pts = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [50.0, 0.0, 0.0]])
        nl = NeighborList(pts, cutoff=2.0)
        assert nl.deg[0] == pytest.approx(1.0)
        assert _compute_bond_angle_entropy(nl) == pytest.approx(0.0)

    def test_numpy_fallback_positive_for_multi_neighbor_atoms(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NumPy fallback returns > 0 when atoms have >= 2 neighbors."""
        import pasted._metrics as mod

        monkeypatch.setattr(mod, "_HAS_BA_CPP", False)
        nl = NeighborList(_ADV_PTS, cutoff=_ADV_CUTOFF)
        assert _compute_bond_angle_entropy(nl) > 0.0


# ---------------------------------------------------------------------------
# _compute_coordination_variance
# ---------------------------------------------------------------------------


class TestComputeCoordinationVariance:
    """Tests for _compute_coordination_variance (v0.4.0 adversarial metric)."""

    def test_no_pairs_returns_zero(self) -> None:
        """n_pairs == 0 must short-circuit to 0.0."""
        nl = NeighborList(_ADV_PTS, cutoff=0.1)
        assert _compute_coordination_variance(nl) == pytest.approx(0.0)

    def test_single_atom_returns_zero(self) -> None:
        """n_atoms < 2 must short-circuit to 0.0."""
        pts = np.array([[0.0, 0.0, 0.0]])
        nl = NeighborList(pts, cutoff=2.0)
        assert _compute_coordination_variance(nl) == pytest.approx(0.0)

    def test_uniform_coordination_returns_zero(self) -> None:
        """All atoms with identical coordination number -> variance = 0."""
        nl = NeighborList(_ADV_PTS, cutoff=_ADV_CUTOFF)
        assert nl.deg.tolist() == pytest.approx([3.0, 3.0, 3.0, 3.0])
        assert _compute_coordination_variance(nl) == pytest.approx(0.0)

    def test_varying_coordination_returns_positive(self) -> None:
        """Non-uniform coordination numbers -> variance > 0."""
        # Center atom connected to 4 neighbors; outer atoms each connected to 1
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [-1.5, 0.0, 0.0],
                [0.0, 1.5, 0.0],
                [0.0, -1.5, 0.0],
            ]
        )
        nl = NeighborList(pts, cutoff=2.0)
        assert _compute_coordination_variance(nl) > 0.0

    def test_result_is_non_negative(self) -> None:
        """Variance is always >= 0."""
        rng = np.random.default_rng(0)
        pts = rng.uniform(-3.0, 3.0, (20, 3))
        nl = NeighborList(pts, cutoff=2.0)
        assert _compute_coordination_variance(nl) >= 0.0


# ---------------------------------------------------------------------------
# _compute_radial_variance
# ---------------------------------------------------------------------------


class TestComputeRadialVariance:
    """Tests for _compute_radial_variance (v0.4.0 adversarial metric)."""

    def test_no_pairs_returns_zero(self) -> None:
        """n_pairs == 0 must short-circuit to 0.0."""
        nl = NeighborList(_ADV_PTS, cutoff=0.1)
        assert _compute_radial_variance(nl) == pytest.approx(0.0)

    def test_single_atom_returns_zero(self) -> None:
        """n_atoms < 2 must short-circuit to 0.0."""
        pts = np.array([[0.0, 0.0, 0.0]])
        nl = NeighborList(pts, cutoff=2.0)
        assert _compute_radial_variance(nl) == pytest.approx(0.0)

    def test_equidistant_neighbors_returns_zero(self) -> None:
        """All neighbors at identical distance -> per-atom variance = 0."""
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [-1.5, 0.0, 0.0],
                [0.0, 1.5, 0.0],
                [0.0, -1.5, 0.0],
            ]
        )
        nl = NeighborList(pts, cutoff=2.0)
        # Center atom has 4 neighbors all at distance 1.5 -> var = 0
        assert _compute_radial_variance(nl) == pytest.approx(0.0, abs=1e-12)

    def test_mixed_distances_returns_positive(self) -> None:
        """Atoms whose neighbors span near and far distances -> variance > 0."""
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        nl = NeighborList(pts, cutoff=2.5)
        # atom 0: neighbors at d=1.0 and d=2.0 -> var > 0
        assert _compute_radial_variance(nl) > 0.0

    def test_result_is_non_negative(self) -> None:
        """Result must be >= 0 for any valid structure."""
        nl = NeighborList(_ADV_PTS, cutoff=_ADV_CUTOFF)
        assert _compute_radial_variance(nl) >= 0.0


# ---------------------------------------------------------------------------
# _compute_local_anisotropy
# ---------------------------------------------------------------------------


class TestComputeLocalAnisotropy:
    """Tests for _compute_local_anisotropy (v0.4.0 adversarial metric)."""

    def test_no_pairs_returns_zero(self) -> None:
        """n_pairs == 0 must short-circuit to 0.0."""
        nl = NeighborList(_ADV_PTS, cutoff=0.1)
        assert _compute_local_anisotropy(nl) == pytest.approx(0.0)

    def test_single_atom_returns_zero(self) -> None:
        """n_atoms < 2 must short-circuit to 0.0."""
        pts = np.array([[0.0, 0.0, 0.0]])
        nl = NeighborList(pts, cutoff=2.0)
        assert _compute_local_anisotropy(nl) == pytest.approx(0.0)

    def test_linear_chain_near_one(self) -> None:
        """Linear chain: each interior atom has two collinear neighbors -> kappa^2 = 1."""
        pts = np.array([[float(i), 0.0, 0.0] for i in range(5)])
        nl = NeighborList(pts, cutoff=1.5)
        result = _compute_local_anisotropy(nl)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_range(self) -> None:
        """Result must lie in [0, 1]."""
        nl = NeighborList(_ADV_PTS, cutoff=_ADV_CUTOFF)
        result = _compute_local_anisotropy(nl)
        assert 0.0 <= result <= 1.0

    def test_isotropic_structure_lower_than_linear(self) -> None:
        """Isotropic environment (atoms around a sphere) is less anisotropic than linear."""
        rng = np.random.default_rng(42)
        pts_iso = rng.standard_normal((50, 3))  # roughly isotropic
        nl_iso = NeighborList(pts_iso, cutoff=1.5)

        pts_lin = np.array([[float(i), 0.0, 0.0] for i in range(50)])
        nl_lin = NeighborList(pts_lin, cutoff=1.5)

        assert _compute_local_anisotropy(nl_iso) < _compute_local_anisotropy(nl_lin)


# ---------------------------------------------------------------------------
# _compute_adversarial (wrapper)
# ---------------------------------------------------------------------------


class TestComputeAdversarial:
    """Tests for _compute_adversarial, the shared-NeighborList wrapper."""

    def test_returns_all_four_keys(self) -> None:
        """Wrapper must return exactly the four adversarial metric keys."""
        nl = NeighborList(_ADV_PTS, cutoff=_ADV_CUTOFF)
        result = _compute_adversarial(nl)
        assert set(result.keys()) == {
            "bond_angle_entropy",
            "coordination_variance",
            "radial_variance",
            "local_anisotropy",
        }

    def test_all_values_are_finite(self) -> None:
        """Every adversarial metric value must be finite."""
        nl = NeighborList(_ADV_PTS, cutoff=_ADV_CUTOFF)
        for key, val in _compute_adversarial(nl).items():
            assert math.isfinite(val), f"{key} is not finite: {val}"

    def test_no_pairs_all_zero(self) -> None:
        """With no pairs within cutoff, all four metrics must be 0.0."""
        nl = NeighborList(_ADV_PTS, cutoff=0.1)
        for key, val in _compute_adversarial(nl).items():
            assert val == pytest.approx(0.0), f"{key} expected 0.0, got {val}"

    def test_values_match_individual_functions(self) -> None:
        """Wrapper result must equal direct per-function calls on the same NeighborList."""
        nl = NeighborList(_ADV_PTS, cutoff=_ADV_CUTOFF)
        result = _compute_adversarial(nl)
        assert result["bond_angle_entropy"] == pytest.approx(
            _compute_bond_angle_entropy(nl), abs=1e-12
        )
        assert result["coordination_variance"] == pytest.approx(
            _compute_coordination_variance(nl), abs=1e-12
        )
        assert result["radial_variance"] == pytest.approx(
            _compute_radial_variance(nl), abs=1e-12
        )
        assert result["local_anisotropy"] == pytest.approx(
            _compute_local_anisotropy(nl), abs=1e-12
        )


# ---------------------------------------------------------------------------
# compute_all_metrics: adversarial metrics included in output
# ---------------------------------------------------------------------------


class TestComputeAllMetricsAdversarial:
    """Verify that compute_all_metrics exposes all four v0.4.0 adversarial metrics."""

    @pytest.fixture
    def all_metrics(self) -> dict[str, float]:
        atoms = ["C", "N", "O", "H"]
        positions = list(_ADV_PTS)  # type: ignore[arg-type]
        return compute_all_metrics(atoms, positions)

    def test_bond_angle_entropy_present(self, all_metrics: dict[str, float]) -> None:
        assert "bond_angle_entropy" in all_metrics

    def test_coordination_variance_present(self, all_metrics: dict[str, float]) -> None:
        assert "coordination_variance" in all_metrics

    def test_radial_variance_present(self, all_metrics: dict[str, float]) -> None:
        assert "radial_variance" in all_metrics

    def test_local_anisotropy_present(self, all_metrics: dict[str, float]) -> None:
        assert "local_anisotropy" in all_metrics

    def test_bond_angle_entropy_non_negative(self, all_metrics: dict[str, float]) -> None:
        assert all_metrics["bond_angle_entropy"] >= 0.0

    def test_coordination_variance_non_negative(self, all_metrics: dict[str, float]) -> None:
        assert all_metrics["coordination_variance"] >= 0.0

    def test_radial_variance_non_negative(self, all_metrics: dict[str, float]) -> None:
        assert all_metrics["radial_variance"] >= 0.0

    def test_local_anisotropy_in_range(self, all_metrics: dict[str, float]) -> None:
        assert 0.0 <= all_metrics["local_anisotropy"] <= 1.0
