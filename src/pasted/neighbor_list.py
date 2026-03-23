"""
pasted.neighbor_list
====================
Lazy-cached neighbor list built on ``scipy.cKDTree``.

``NeighborList`` wraps a non-periodic neighbor list and exposes derived
arrays (degree, unit-direction vectors, squared distances) as cached
properties so that multiple metrics sharing the same cutoff can reuse
intermediate results without rebuilding the tree or re-scanning pairs.

Typical usage::

    from pasted.neighbor_list import NeighborList

    nl = NeighborList(pts, cutoff)          # build once
    bae = _compute_bond_angle_entropy(nl)   # shares pairs with ...
    cv  = _compute_coordination_variance(nl)
    rv  = _compute_radial_variance(nl)
    la  = _compute_local_anisotropy(nl)

Design notes
------------
* ``__init__`` materialises the undirected pair list and the symmetric
  (directed) arrays ``rows``, ``cols``, ``diff``, ``all_dists`` eagerly,
  because every metric needs at least one of them.
* Derived arrays (``deg``, ``unit_diff``, ``dists_sq``) are built on first
  access and cached, paying O(2P) at most once per property.
* ``d < 1e-10`` coincident-coordinate guard in ``unit_diff`` mirrors the
  C++ threshold in ``_bond_angle_core.cpp`` so that the Python fallback
  and the C++ path produce identical results.
* N = 0 is handled gracefully: all arrays are empty and ``n_pairs = 0``.
  Each metric's own ``n_atoms < 2`` / ``n_pairs == 0`` guard provides the
  early return.

Added in v0.4.0.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree as _cKDTree


class NeighborList:
    """Non-periodic neighbor list with lazy-cached derived arrays.

    Parameters
    ----------
    pts:
        Atom positions, shape ``(N, 3)``, dtype float64.
    cutoff:
        Distance cutoff in Å.  Pairs with ``d > cutoff`` are excluded.

    Attributes
    ----------
    pts : np.ndarray
        The input position array ``(N, 3)``.
    cutoff : float
        The cutoff used to build this list.
    n_atoms : int
        Number of atoms ``N``.
    n_pairs : int
        Number of undirected pairs ``P`` within *cutoff*.
    pairs : np.ndarray
        Undirected pair indices, shape ``(P, 2)``, ``pairs[k, 0] < pairs[k, 1]``.
    rows : np.ndarray
        Source indices of directed (bidirectional) pairs, shape ``(2P,)``.
    cols : np.ndarray
        Target indices of directed pairs, shape ``(2P,)``.
    dists : np.ndarray
        Undirected pair distances, shape ``(P,)``.
    all_dists : np.ndarray
        Directed pair distances (each undirected distance duplicated),
        shape ``(2P,)``.
    diff : np.ndarray
        Directed difference vectors ``pts[rows] - pts[cols]``,
        shape ``(2P, 3)``.

    Properties (cached)
    -------------------
    deg : np.ndarray
        Per-atom coordination number as float64, shape ``(N,)``.
        Computed once via ``np.bincount``; subsequent accesses are O(1).
    unit_diff : np.ndarray
        Directed unit vectors ``diff / ||diff||``, shape ``(2P, 3)``.
        Coincident pairs (``d < 1e-10``) use a safe divisor of 1.0 to
        avoid division by zero; such pairs should be skipped by callers.
    dists_sq : np.ndarray
        Squared directed distances ``all_dists ** 2``, shape ``(2P,)``.
        Cached to avoid repeated allocation in ``_compute_radial_variance``.
    """

    def __init__(self, pts: np.ndarray, cutoff: float) -> None:
        self.pts: np.ndarray = pts
        self.cutoff: float = cutoff
        self.n_atoms: int = len(pts)

        if self.n_atoms == 0:
            # Empty structure: initialise all arrays as length-0.
            empty_i = np.empty(0, dtype=np.intp)
            empty_f = np.empty(0, dtype=np.float64)
            self.pairs = np.empty((0, 2), dtype=np.intp)
            self.n_pairs = 0
            self.rows = empty_i
            self.cols = empty_i
            self.dists = empty_f
            self.all_dists = empty_f
            self.diff = np.empty((0, 3), dtype=np.float64)
            return

        tree = _cKDTree(pts)
        pairs: np.ndarray = tree.query_pairs(cutoff, output_type="ndarray")
        self.pairs = pairs
        self.n_pairs = len(pairs)

        if self.n_pairs == 0:
            empty_i = np.empty(0, dtype=np.intp)
            empty_f = np.empty(0, dtype=np.float64)
            self.rows = empty_i
            self.cols = empty_i
            self.dists = empty_f
            self.all_dists = empty_f
            self.diff = np.empty((0, 3), dtype=np.float64)
            return

        fwd_i: np.ndarray = pairs[:, 0]  # (P,) smaller index
        fwd_j: np.ndarray = pairs[:, 1]  # (P,) larger index

        diff_fwd: np.ndarray = pts[fwd_i] - pts[fwd_j]  # (P, 3)
        dists_fwd: np.ndarray = np.linalg.norm(diff_fwd, axis=1)  # (P,)

        self.rows = np.concatenate([fwd_i, fwd_j])  # (2P,)
        self.cols = np.concatenate([fwd_j, fwd_i])  # (2P,)
        self.dists = dists_fwd  # (P,)
        self.diff = np.concatenate([diff_fwd, -diff_fwd])  # (2P, 3)
        self.all_dists = np.concatenate([dists_fwd, dists_fwd])  # (2P,)

    # ------------------------------------------------------------------
    # Cached derived properties
    # ------------------------------------------------------------------

    @property
    def deg(self) -> np.ndarray:
        """Per-atom coordination number as float64, shape ``(N,)``.

        Computed on first access via ``np.bincount``; O(1) thereafter.
        """
        if not hasattr(self, "_deg"):
            self._deg: np.ndarray = np.bincount(self.rows, minlength=self.n_atoms).astype(
                np.float64
            )
        return self._deg

    @property
    def unit_diff(self) -> np.ndarray:
        """Directed unit vectors, shape ``(2P, 3)``.

        Coincident pairs (``d < 1e-10``) use a safe divisor of 1.0 to
        prevent division by zero.  Each metric must independently skip
        such pairs (they carry no directional information).
        The threshold mirrors the C++ guard in ``_bond_angle_core.cpp``
        so the Python fallback and C++ path agree numerically.
        """
        if not hasattr(self, "_unit_diff"):
            safe_d = np.where(self.all_dists > 1e-10, self.all_dists, 1.0)
            self._unit_diff: np.ndarray = self.diff / safe_d[:, np.newaxis]
        return self._unit_diff

    @property
    def dists_sq(self) -> np.ndarray:
        """Squared directed distances ``all_dists ** 2``, shape ``(2P,)``.

        Cached to avoid redundant array allocation across metrics that
        each need ``d²`` (e.g. ``_compute_radial_variance``).
        """
        if not hasattr(self, "_dists_sq"):
            self._dists_sq: np.ndarray = self.all_dists * self.all_dists
        return self._dists_sq
