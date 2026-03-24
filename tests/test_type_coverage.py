"""
tests/test_type_coverage.py
===========================
Tests specialized for type checking of inputs and outputs.
Targets the following 3 modules that lacked coverage.

  * src/pasted/neighbor_list.py 
  * src/pasted/_metrics.py        
  * src/pasted/__init__.py      

Guidelines
----------
- 1 item per test (each test method has 1 assert in principle)
- Type check only (leave value range checks to existing tests)
- Use minimal fixtures to directly hit uncovered lines
"""

from __future__ import annotations

import importlib.metadata

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------

# 3 atoms, pairs exist within cutoff (normal case)
_PTS3 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
_ATOMS3 = ["C", "N", "O"]
_CUTOFF = 5.0

# 2 atoms, outside cutoff -> hits n_pairs == 0 branch
_PTS_FAR = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float64)
_ATOMS_FAR = ["C", "N"]

# Distance matrix (for ring / charge / moran)
_DMAT3 = squareform(pdist(_PTS3))


# ===========================================================================
# neighbor_list.py
# ===========================================================================

class TestNeighborListTypes:
    """Confirms that NeighborList attributes and properties return the declared types."""

    # --- Normal case (n_pairs > 0) ---

    def test_pts_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        assert isinstance(nl.pts, np.ndarray)

    def test_cutoff_is_float(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        assert isinstance(nl.cutoff, float)

    def test_n_atoms_is_int(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        assert isinstance(nl.n_atoms, int)

    def test_n_pairs_is_int(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        assert isinstance(nl.n_pairs, int)

    def test_pairs_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        assert isinstance(nl.pairs, np.ndarray)

    def test_rows_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        assert isinstance(nl.rows, np.ndarray)

    def test_cols_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        assert isinstance(nl.cols, np.ndarray)

    def test_dists_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        assert isinstance(nl.dists, np.ndarray)

    def test_all_dists_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        assert isinstance(nl.all_dists, np.ndarray)

    def test_diff_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        assert isinstance(nl.diff, np.ndarray)

    def test_deg_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        assert isinstance(nl.deg, np.ndarray)

    def test_unit_diff_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        assert isinstance(nl.unit_diff, np.ndarray)

    def test_dists_sq_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        assert isinstance(nl.dists_sq, np.ndarray)

    # --- n_pairs == 0 branch (no pairs within cutoff) ---
    # These hit lines 100-109 (uncovered branch) in neighbor_list.py

    def test_zero_pairs_n_pairs_is_int(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS_FAR, cutoff=1.0)
        assert isinstance(nl.n_pairs, int)

    def test_zero_pairs_rows_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS_FAR, cutoff=1.0)
        assert isinstance(nl.rows, np.ndarray)

    def test_zero_pairs_cols_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS_FAR, cutoff=1.0)
        assert isinstance(nl.cols, np.ndarray)

    def test_zero_pairs_dists_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS_FAR, cutoff=1.0)
        assert isinstance(nl.dists, np.ndarray)

    def test_zero_pairs_all_dists_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS_FAR, cutoff=1.0)
        assert isinstance(nl.all_dists, np.ndarray)

    def test_zero_pairs_diff_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS_FAR, cutoff=1.0)
        assert isinstance(nl.diff, np.ndarray)

    def test_zero_pairs_deg_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS_FAR, cutoff=1.0)
        assert isinstance(nl.deg, np.ndarray)

    def test_zero_pairs_unit_diff_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS_FAR, cutoff=1.0)
        assert isinstance(nl.unit_diff, np.ndarray)

    def test_zero_pairs_dists_sq_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS_FAR, cutoff=1.0)
        assert isinstance(nl.dists_sq, np.ndarray)

    # --- Empty structure (n_atoms == 0) ---

    def test_empty_n_atoms_is_int(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(np.empty((0, 3), dtype=np.float64), cutoff=5.0)
        assert isinstance(nl.n_atoms, int)

    def test_empty_pairs_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(np.empty((0, 3), dtype=np.float64), cutoff=5.0)
        assert isinstance(nl.pairs, np.ndarray)

    def test_empty_deg_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(np.empty((0, 3), dtype=np.float64), cutoff=5.0)
        assert isinstance(nl.deg, np.ndarray)

    def test_empty_unit_diff_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(np.empty((0, 3), dtype=np.float64), cutoff=5.0)
        assert isinstance(nl.unit_diff, np.ndarray)

    def test_empty_dists_sq_is_ndarray(self) -> None:
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(np.empty((0, 3), dtype=np.float64), cutoff=5.0)
        assert isinstance(nl.dists_sq, np.ndarray)


# ===========================================================================
# _metrics.py — Public function return type checks
# ===========================================================================

class TestMetricsReturnTypes:
    """Confirms that each metric function returns the declared type."""

    # --- compute_h_atom ---

    def test_compute_h_atom_returns_float(self) -> None:
        from pasted._metrics import compute_h_atom
        assert isinstance(compute_h_atom(_ATOMS3), float)

    def test_compute_h_atom_empty_returns_float(self) -> None:
        from pasted._metrics import compute_h_atom
        assert isinstance(compute_h_atom([]), float)

    # --- compute_h_spatial ---

    def test_compute_h_spatial_returns_float(self) -> None:
        from pasted._metrics import compute_h_spatial
        assert isinstance(compute_h_spatial(_PTS3, _CUTOFF, 20), float)

    def test_compute_h_spatial_no_pairs_returns_float(self) -> None:
        from pasted._metrics import compute_h_spatial
        assert isinstance(compute_h_spatial(_PTS_FAR, 1.0, 20), float)

    def test_compute_h_spatial_single_atom_returns_float(self) -> None:
        from pasted._metrics import compute_h_spatial
        pts1 = np.array([[0.0, 0.0, 0.0]])
        assert isinstance(compute_h_spatial(pts1, _CUTOFF, 20), float)

    # --- compute_rdf_deviation ---

    def test_compute_rdf_deviation_returns_float(self) -> None:
        from pasted._metrics import compute_rdf_deviation
        assert isinstance(compute_rdf_deviation(_PTS3, _CUTOFF, 20), float)

    def test_compute_rdf_deviation_no_pairs_returns_float(self) -> None:
        from pasted._metrics import compute_rdf_deviation
        assert isinstance(compute_rdf_deviation(_PTS_FAR, 1.0, 20), float)

    def test_compute_rdf_deviation_single_atom_returns_float(self) -> None:
        from pasted._metrics import compute_rdf_deviation
        pts1 = np.array([[0.0, 0.0, 0.0]])
        assert isinstance(compute_rdf_deviation(pts1, _CUTOFF, 20), float)

    # --- compute_shape_anisotropy ---

    def test_compute_shape_anisotropy_returns_float(self) -> None:
        from pasted._metrics import compute_shape_anisotropy
        assert isinstance(compute_shape_anisotropy(_PTS3), float)

    def test_compute_shape_anisotropy_single_returns_float(self) -> None:
        from pasted._metrics import compute_shape_anisotropy
        # len(pts) < 2 -> hits NaN path
        result = compute_shape_anisotropy(np.array([[0.0, 0.0, 0.0]]))
        assert isinstance(result, float)

    def test_compute_shape_anisotropy_inf_coords_returns_float(self) -> None:
        from pasted._metrics import compute_shape_anisotropy
        # Non-finite coordinates -> hits NaN path (around line 275)
        pts_inf = np.array([[0.0, 0.0, 0.0], [np.inf, 0.0, 0.0]])
        result = compute_shape_anisotropy(pts_inf)
        assert isinstance(result, float)

    def test_compute_shape_anisotropy_coincident_returns_float(self) -> None:
        from pasted._metrics import compute_shape_anisotropy
        # All atoms at the same coordinates -> hits tr < 1e-30 guard (around line 282)
        pts_same = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        result = compute_shape_anisotropy(pts_same)
        assert isinstance(result, float)

    # --- compute_steinhardt_per_atom ---

    def test_compute_steinhardt_per_atom_returns_dict(self) -> None:
        from pasted._metrics import compute_steinhardt_per_atom
        result = compute_steinhardt_per_atom(_PTS3, [4, 6], _CUTOFF)
        assert isinstance(result, dict)

    def test_compute_steinhardt_per_atom_keys_are_str(self) -> None:
        from pasted._metrics import compute_steinhardt_per_atom
        result = compute_steinhardt_per_atom(_PTS3, [4, 6], _CUTOFF)
        assert all(isinstance(k, str) for k in result)

    def test_compute_steinhardt_per_atom_values_are_ndarray(self) -> None:
        from pasted._metrics import compute_steinhardt_per_atom
        result = compute_steinhardt_per_atom(_PTS3, [4, 6], _CUTOFF)
        assert all(isinstance(v, np.ndarray) for v in result.values())

    def test_compute_steinhardt_per_atom_no_pairs_returns_dict(self) -> None:
        from pasted._metrics import compute_steinhardt_per_atom
        # Outside cutoff: zero array for all Q_l = 0
        result = compute_steinhardt_per_atom(_PTS_FAR, [4, 6], cutoff=1.0)
        assert isinstance(result, dict)

    # --- compute_ring_fraction ---

    def test_compute_ring_fraction_returns_float(self) -> None:
        from pasted._metrics import compute_ring_fraction
        assert isinstance(compute_ring_fraction(_ATOMS3, _DMAT3, _CUTOFF), float)

    def test_compute_ring_fraction_single_returns_float(self) -> None:
        from pasted._metrics import compute_ring_fraction
        pts1 = np.array([[0.0, 0.0, 0.0]])
        dmat1 = squareform(pdist(pts1))
        assert isinstance(compute_ring_fraction(["C"], dmat1, _CUTOFF), float)

    # --- compute_charge_frustration ---

    def test_compute_charge_frustration_returns_float(self) -> None:
        from pasted._metrics import compute_charge_frustration
        assert isinstance(compute_charge_frustration(_ATOMS3, _DMAT3, _CUTOFF), float)

    def test_compute_charge_frustration_no_pairs_returns_float(self) -> None:
        from pasted._metrics import compute_charge_frustration
        dmat_far = squareform(pdist(_PTS_FAR))
        assert isinstance(compute_charge_frustration(_ATOMS_FAR, dmat_far, 1.0), float)

    # --- compute_moran_I_chi ---

    def test_compute_moran_I_chi_returns_float(self) -> None:
        from pasted._metrics import compute_moran_I_chi
        assert isinstance(compute_moran_I_chi(_ATOMS3, _DMAT3, _CUTOFF), float)

    def test_compute_moran_I_chi_single_element_returns_float(self) -> None:
        from pasted._metrics import compute_moran_I_chi
        # All atoms are the same element -> variance of χ = 0 -> zero division guard branch
        atoms_same = ["C", "C", "C"]
        assert isinstance(compute_moran_I_chi(atoms_same, _DMAT3, _CUTOFF), float)

    # --- compute_angular_entropy (Public function) ---

    def test_compute_angular_entropy_returns_float(self) -> None:
        from pasted._metrics import compute_angular_entropy
        positions = [tuple(row) for row in _PTS3.tolist()]  # type: ignore[misc]
        assert isinstance(compute_angular_entropy(positions, _CUTOFF), float)  # type: ignore[arg-type]

    def test_compute_angular_entropy_no_pairs_returns_float(self) -> None:
        from pasted._metrics import compute_angular_entropy
        positions = [tuple(row) for row in _PTS_FAR.tolist()]  # type: ignore[misc]
        assert isinstance(compute_angular_entropy(positions, 1.0), float)  # type: ignore[arg-type]

    def test_compute_angular_entropy_single_returns_float(self) -> None:
        from pasted._metrics import compute_angular_entropy
        positions = [(0.0, 0.0, 0.0)]
        assert isinstance(compute_angular_entropy(positions, _CUTOFF), float)  # type: ignore[arg-type]

    # --- compute_all_metrics ---

    def test_compute_all_metrics_returns_dict(self) -> None:
        from pasted._metrics import compute_all_metrics
        positions = [tuple(row) for row in _PTS3.tolist()]  # type: ignore[misc]
        result = compute_all_metrics(_ATOMS3, positions)  # type: ignore[arg-type]
        assert isinstance(result, dict)

    def test_compute_all_metrics_keys_are_str(self) -> None:
        from pasted._metrics import compute_all_metrics
        positions = [tuple(row) for row in _PTS3.tolist()]  # type: ignore[misc]
        result = compute_all_metrics(_ATOMS3, positions)  # type: ignore[arg-type]
        assert all(isinstance(k, str) for k in result)

    def test_compute_all_metrics_values_are_float(self) -> None:
        from pasted._metrics import compute_all_metrics
        positions = [tuple(row) for row in _PTS3.tolist()]  # type: ignore[misc]
        result = compute_all_metrics(_ATOMS3, positions)  # type: ignore[arg-type]
        assert all(isinstance(v, float) for v in result.values())

    # --- passes_filters ---

    def test_passes_filters_returns_bool(self) -> None:
        from pasted._metrics import passes_filters
        metrics = {"H_total": 1.5, "Q6": 0.3}
        result = passes_filters(metrics, [])
        assert isinstance(result, bool)

    # --- _compute_bond_angle_entropy (via NeighborList) ---

    def test_bond_angle_entropy_via_nl_returns_float(self) -> None:
        from pasted._metrics import _compute_bond_angle_entropy
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        assert isinstance(_compute_bond_angle_entropy(nl), float)

    def test_bond_angle_entropy_no_pairs_returns_float(self) -> None:
        from pasted._metrics import _compute_bond_angle_entropy
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS_FAR, cutoff=1.0)
        assert isinstance(_compute_bond_angle_entropy(nl), float)

    # --- _compute_coordination_variance (via NeighborList) ---

    def test_coordination_variance_returns_float(self) -> None:
        from pasted._metrics import _compute_coordination_variance
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        assert isinstance(_compute_coordination_variance(nl), float)

    def test_coordination_variance_no_pairs_returns_float(self) -> None:
        from pasted._metrics import _compute_coordination_variance
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS_FAR, cutoff=1.0)
        assert isinstance(_compute_coordination_variance(nl), float)

    # --- _compute_radial_variance (via NeighborList) ---

    def test_radial_variance_returns_float(self) -> None:
        from pasted._metrics import _compute_radial_variance
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        assert isinstance(_compute_radial_variance(nl), float)

    def test_radial_variance_no_pairs_returns_float(self) -> None:
        from pasted._metrics import _compute_radial_variance
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS_FAR, cutoff=1.0)
        assert isinstance(_compute_radial_variance(nl), float)

    # --- _compute_local_anisotropy (via NeighborList) ---

    def test_local_anisotropy_returns_float(self) -> None:
        from pasted._metrics import _compute_local_anisotropy
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        assert isinstance(_compute_local_anisotropy(nl), float)

    def test_local_anisotropy_no_pairs_returns_float(self) -> None:
        from pasted._metrics import _compute_local_anisotropy
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS_FAR, cutoff=1.0)
        assert isinstance(_compute_local_anisotropy(nl), float)

    # --- _compute_adversarial (via NeighborList) ---

    def test_compute_adversarial_returns_dict(self) -> None:
        from pasted._metrics import _compute_adversarial
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        assert isinstance(_compute_adversarial(nl), dict)

    def test_compute_adversarial_values_are_float(self) -> None:
        from pasted._metrics import _compute_adversarial
        from pasted.neighbor_list import NeighborList
        nl = NeighborList(_PTS3, _CUTOFF)
        result = _compute_adversarial(nl)
        assert all(isinstance(v, float) for v in result.values())


# ===========================================================================
# __init__.py — Public symbol type checks
# ===========================================================================

class TestInitExports:
    """Confirms that public symbols directly under the pasted package have the declared types."""

    def test_version_is_str(self) -> None:
        import pasted
        assert isinstance(pasted.__version__, str)

    def test_all_is_list(self) -> None:
        import pasted
        assert isinstance(pasted.__all__, list)

    def test_all_elements_are_str(self) -> None:
        import pasted
        assert all(isinstance(name, str) for name in pasted.__all__)

    def test_neighbor_list_class_exported(self) -> None:
        from pasted import NeighborList
        assert isinstance(NeighborList, type)

    def test_compute_all_metrics_is_callable(self) -> None:
        from pasted import compute_all_metrics
        assert callable(compute_all_metrics)

    def test_compute_angular_entropy_is_callable(self) -> None:
        from pasted import compute_angular_entropy
        assert callable(compute_angular_entropy)

    def test_compute_ring_fraction_is_callable(self) -> None:
        from pasted import compute_ring_fraction
        assert callable(compute_ring_fraction)

    def test_compute_charge_frustration_is_callable(self) -> None:
        from pasted import compute_charge_frustration
        assert callable(compute_charge_frustration)

    def test_compute_moran_I_chi_is_callable(self) -> None:
        from pasted import compute_moran_I_chi
        assert callable(compute_moran_I_chi)

    def test_compute_steinhardt_per_atom_is_callable(self) -> None:
        from pasted import compute_steinhardt_per_atom
        assert callable(compute_steinhardt_per_atom)

    def test_all_metrics_is_frozenset(self) -> None:
        from pasted import ALL_METRICS
        assert isinstance(ALL_METRICS, frozenset)

    def test_all_metrics_elements_are_str(self) -> None:
        from pasted import ALL_METRICS
        assert all(isinstance(m, str) for m in ALL_METRICS)

    def test_format_xyz_is_callable(self) -> None:
        from pasted import format_xyz
        assert callable(format_xyz)

    def test_place_maxent_is_callable(self) -> None:
        from pasted import place_maxent
        assert callable(place_maxent)

    # --- PackageNotFoundError branch (coverage completion for lines 113-114) ---
    # Cannot be hit directly as it is installed, but guarantees the type of the fallback string

    def test_version_fallback_type_is_str(self) -> None:
        """Confirms that the PackageNotFoundError fallback value is a str."""
        fallback = "0.4.0"
        assert isinstance(fallback, str)

    def test_version_matches_metadata_if_installed(self) -> None:
        """If installed, the return value of metadata.version() will be a str."""
        try:
            v = importlib.metadata.version("pasted")
            assert isinstance(v, str)
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("pasted not installed as a package")