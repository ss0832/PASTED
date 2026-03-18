"""
pasted._metrics
===============
Disorder-metric computations.  All functions accept pre-computed numpy
arrays where possible so that the pairwise distance matrix is built only
once per structure.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix as _csr_matrix
from scipy.sparse.csgraph import connected_components as _connected_components
from scipy.spatial.distance import pdist as _pdist
from scipy.spatial.distance import squareform as _squareform

# scipy >= 1.15 renamed sph_harm → sph_harm_y with a different argument order.
# The except branch is kept for environments that still run scipy < 1.15;
# warn_unused_ignores is suppressed for this file in pyproject.toml [tool.mypy.overrides].
try:
    from scipy.special import sph_harm_y as _sph_harm_raw

    def _sph_harm(
        l: int,  # noqa: E741
        m: int,
        phi_azimuth: float | np.ndarray,
        theta_polar: float | np.ndarray,
    ) -> np.ndarray:
        return np.asarray(_sph_harm_raw(l, m, theta_polar, phi_azimuth))

except ImportError:
    from scipy.special import sph_harm as _sph_harm_raw  # type: ignore[no-redef,attr-defined]

    def _sph_harm(  # type: ignore[misc,no-redef]
        l: int,  # noqa: E741
        m: int,
        phi_azimuth: float | np.ndarray,
        theta_polar: float | np.ndarray,
    ) -> np.ndarray:
        return np.asarray(_sph_harm_raw(m, l, phi_azimuth, theta_polar))


if TYPE_CHECKING:
    from ._placement import Vec3

from ._atoms import cov_radius_ang as _cov_radius_ang
from ._atoms import pauling_electronegativity as _pauling_en

# ---------------------------------------------------------------------------
# Low-level entropy helper
# ---------------------------------------------------------------------------


def _shannon_np(counts: np.ndarray) -> float:
    """Shannon entropy from a raw (un-normalised) count array."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def compute_h_atom(atoms: list[str]) -> float:
    """Shannon entropy of the element composition.

    Range: 0 (pure single element) to ln(*k*) for *k* distinct elements.
    """
    counts_dict = Counter(atoms)
    return _shannon_np(np.array(list(counts_dict.values()), dtype=float))


def compute_h_spatial(dists: np.ndarray, n_bins: int) -> float:
    """Shannon entropy of the pairwise-distance histogram.

    Higher values indicate a more uniform distribution of distances.

    Parameters
    ----------
    dists:
        Condensed distance array from :func:`scipy.spatial.distance.pdist`.
    n_bins:
        Number of histogram bins.
    """
    if len(dists) < 1:
        return 0.0
    counts, _ = np.histogram(dists, bins=n_bins)
    return _shannon_np(counts.astype(float))


def compute_rdf_deviation(pts: np.ndarray, dists: np.ndarray, n_bins: int) -> float:
    """RMS deviation of the empirical *g*(*r*) from an ideal-gas baseline.

    A value of 0 indicates a perfectly random (ideal-gas-like) distribution.

    Parameters
    ----------
    pts:
        Positions array of shape ``(n, 3)``.
    dists:
        Condensed distance array (pre-computed from *pts*).
    n_bins:
        Number of histogram bins.
    """
    if len(dists) < 1:
        return 0.0
    n = len(pts)
    r_max = float(dists.max())
    r_bound = float(np.sqrt((pts**2).sum(axis=1)).max())
    if r_bound == 0 or r_max == 0:
        return 0.0
    rho = n / (4 / 3 * math.pi * r_bound**3)
    counts, edges = np.histogram(dists, bins=n_bins, range=(0.0, r_max))
    centres = (edges[:-1] + edges[1:]) / 2
    bw = edges[1] - edges[0]
    ideal = rho * 4 * math.pi * centres**2 * bw * n / 2
    mask = ideal > 0
    if not mask.any():
        return 0.0
    return float(np.sqrt(np.mean(((counts[mask] / ideal[mask]) - 1.0) ** 2)))


def compute_shape_anisotropy(pts: np.ndarray) -> float:
    """Relative shape anisotropy from the gyration tensor.

    Range: 0 (spherical) to 1 (rod-like).
    Returns NaN for a single atom.
    """
    if len(pts) < 2:
        return float("nan")
    p = pts - pts.mean(axis=0)
    T = (p.T @ p) / len(p)
    lam = np.linalg.eigvalsh(T)
    s = float(lam.sum())
    if s == 0:
        return 0.0
    return float(np.clip(1.5 * float(np.sum(lam**2)) / s**2 - 0.5, 0.0, 1.0))


def compute_steinhardt_per_atom(
    pts: np.ndarray,
    dmat: np.ndarray,
    l_values: list[int],
    cutoff: float,
) -> dict[str, np.ndarray]:
    """Per-atom Steinhardt Q_l values.

    Parameters
    ----------
    pts:
        Positions array of shape ``(n, 3)``.
    dmat:
        Full n×n pairwise distance matrix.
    l_values:
        List of *l* values (e.g. ``[4, 6, 8]``).
    cutoff:
        Neighbour distance cutoff (Å).

    Returns
    -------
    dict mapping ``"Q{l}"`` to a :class:`numpy.ndarray` of shape ``(n,)``.
    Atoms with no neighbours within *cutoff* are assigned Q_l = 0.
    """
    n = len(pts)
    result: dict[str, np.ndarray] = {}

    mask = dmat <= cutoff
    np.fill_diagonal(mask, False)
    deg = mask.sum(axis=1).astype(float)  # (n,)
    safe_deg = np.where(deg > 0, deg, 1.0)
    mask_f = mask.astype(float)  # (n, n)

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    safe_r = np.where(dmat[:, :, np.newaxis] > 0, dmat[:, :, np.newaxis], 1.0)
    d_hat = diff / safe_r  # (n, n, 3)

    theta = np.arccos(np.clip(d_hat[:, :, 2], -1.0, 1.0))  # polar (n, n)
    phi = np.arctan2(d_hat[:, :, 1], d_hat[:, :, 0])  # azimuthal (n, n)

    for l in l_values:  # noqa: E741
        qlm_sq = np.zeros(n, dtype=float)
        for m in range(-l, l + 1):
            ylm = _sph_harm(l, m, phi, theta)  # (n, n) complex
            avg = (ylm * mask_f).sum(axis=1) / safe_deg  # (n,) complex
            qlm_sq += np.abs(avg) ** 2

        ql = np.sqrt(4 * math.pi / (2 * l + 1) * qlm_sq)
        result[f"Q{l}"] = np.where(deg > 0, ql, 0.0)

    return result


def compute_steinhardt(
    pts: np.ndarray,
    dmat: np.ndarray,
    l_values: list[int],
    cutoff: float,
) -> dict[str, float]:
    """Steinhardt Q_l averaged over all atoms.

    Delegates to :func:`compute_steinhardt_per_atom` and returns the
    per-structure mean for each *l*.

    Parameters
    ----------
    pts:
        Positions array of shape ``(n, 3)``.
    dmat:
        Full n×n pairwise distance matrix.
    l_values:
        List of *l* values (e.g. ``[4, 6, 8]``).
    cutoff:
        Neighbour distance cutoff (Å).

    Returns
    -------
    dict mapping ``"Q{l}"`` to its global average value.
    """
    per_atom = compute_steinhardt_per_atom(pts, dmat, l_values, cutoff)
    return {k: float(v.mean()) for k, v in per_atom.items()}


def compute_graph_metrics(dmat: np.ndarray, cutoff: float) -> dict[str, float]:
    """Largest connected-component fraction and mean clustering coefficient.

    Parameters
    ----------
    dmat:
        Full n×n pairwise distance matrix.
    cutoff:
        Adjacency distance cutoff (Å).

    Returns
    -------
    dict with keys ``"graph_lcc"`` and ``"graph_cc"``.
    """
    n = len(dmat)
    if n < 2:
        return {"graph_lcc": 1.0, "graph_cc": 0.0}

    adj = dmat <= cutoff
    np.fill_diagonal(adj, False)

    _, labels = _connected_components(_csr_matrix(adj), directed=False, return_labels=True)
    graph_lcc = float(np.bincount(labels).max()) / n

    deg = adj.sum(axis=1).astype(float)
    A = adj.astype(float)
    tri = (A @ A * A).sum(axis=1) / 2.0
    max_tri = deg * (deg - 1) / 2.0
    mask = max_tri > 0
    graph_cc = float(np.mean(tri[mask] / max_tri[mask])) if mask.any() else 0.0

    return {"graph_lcc": graph_lcc, "graph_cc": graph_cc}


# ---------------------------------------------------------------------------
# MM-level structural descriptors (added in 0.1.9)
# ---------------------------------------------------------------------------


def compute_bond_strain_rms(
    atoms: list[str],
    dmat: np.ndarray,
    cov_scale: float,
) -> float:
    """RMS relative deviation of bonded-pair distances from their ideal lengths.

    A pair (i, j) is considered *bonded* when its distance satisfies
    ``d_ij < cov_scale × (r_i + r_j)``, where r values are Pyykkö
    single-bond covalent radii.  The strain of each bonded pair is defined as

    .. math::

        \\varepsilon_{ij} = \\frac{d_{ij} - (r_i + r_j)}{r_i + r_j}

    and the metric is the root-mean-square of all per-pair strains.  A value
    of 0 indicates every bonded pair sits exactly at its ideal length; values
    above ~0.1 indicate structurally distorted geometries.

    Parameters
    ----------
    atoms:
        Element symbols.
    dmat:
        Full n×n pairwise distance matrix (Å).
    cov_scale:
        Bond detection threshold scale factor.  A pair is counted as bonded
        when ``d_ij < cov_scale × (r_i + r_j)``.

    Returns
    -------
    float
        RMS relative bond-length deviation.  Returns 0.0 when no bonded
        pairs are detected.
    """
    n = len(atoms)
    strains: list[float] = []
    for i in range(n):
        ri = _cov_radius_ang(atoms[i])
        for j in range(i + 1, n):
            ideal = ri + _cov_radius_ang(atoms[j])
            if dmat[i, j] < cov_scale * ideal:
                strains.append((dmat[i, j] - ideal) / ideal)
    if not strains:
        return 0.0
    arr = np.array(strains, dtype=float)
    return float(np.sqrt(np.mean(arr**2)))


def compute_ring_fraction(
    atoms: list[str],
    dmat: np.ndarray,
    cov_scale: float,
) -> float:
    """Fraction of atoms that belong to at least one ring.

    Builds a bond graph using the same ``cov_scale × (r_i + r_j)`` threshold
    as :func:`compute_bond_strain_rms`, then detects rings via a Union-Find
    spanning-tree construction: every back-edge (an edge between two vertices
    already in the same component) indicates a cycle, and both its endpoints
    are marked as ring members.

    Parameters
    ----------
    atoms:
        Element symbols.
    dmat:
        Full n×n pairwise distance matrix (Å).
    cov_scale:
        Bond detection threshold scale factor.

    Returns
    -------
    float
        Fraction of atoms in at least one ring, in [0, 1].  Returns 0.0
        for structures with fewer than three atoms or no cycles.
    """
    n = len(atoms)
    if n < 3:
        return 0.0

    parent = list(range(n))
    rank = [0] * n

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def _union(a: int, b: int) -> bool:
        """Union by rank.  Returns False (back-edge) when already in same set."""
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    in_ring = [False] * n
    for i in range(n):
        ri = _cov_radius_ang(atoms[i])
        for j in range(i + 1, n):
            ideal = ri + _cov_radius_ang(atoms[j])
            if dmat[i, j] < cov_scale * ideal:
                if not _union(i, j):  # back-edge → cycle detected
                    in_ring[i] = True
                    in_ring[j] = True

    return float(sum(in_ring) / n)


def compute_charge_frustration(
    atoms: list[str],
    dmat: np.ndarray,
    cov_scale: float,
) -> float:
    """Variance of Pauling electronegativity differences across bonded pairs.

    For each bonded pair (i, j) — defined by the same
    ``cov_scale × (r_i + r_j)`` threshold — the absolute electronegativity
    difference ``|χ_i − χ_j|`` is computed.  The metric is the *variance*
    of these differences over all bonded pairs.

    A high value indicates a structure where electronegativity differences
    are inconsistently distributed across bonds: some neighbours are well
    matched while others are highly mismatched.  This is analogous to
    *charge frustration* in disordered materials, where local charge
    neutrality cannot be satisfied simultaneously at every site.

    Noble gases and elements without a Pauling value use the module-level
    fallback of 1.0 (see :func:`~pasted._atoms.pauling_electronegativity`).

    Parameters
    ----------
    atoms:
        Element symbols.
    dmat:
        Full n×n pairwise distance matrix (Å).
    cov_scale:
        Bond detection threshold scale factor.

    Returns
    -------
    float
        Variance of |Δχ| across all bonded pairs.  Returns 0.0 when fewer
        than two bonded pairs are detected (variance is undefined for a
        single observation).
    """
    n = len(atoms)
    en = [_pauling_en(sym) for sym in atoms]
    diffs: list[float] = []
    for i in range(n):
        ri = _cov_radius_ang(atoms[i])
        for j in range(i + 1, n):
            ideal = ri + _cov_radius_ang(atoms[j])
            if dmat[i, j] < cov_scale * ideal:
                diffs.append(abs(en[i] - en[j]))
    if len(diffs) < 2:
        return 0.0
    arr = np.array(diffs, dtype=float)
    return float(np.var(arr))


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def compute_all_metrics(
    atoms: list[str],
    positions: list[Vec3],
    n_bins: int,
    w_atom: float,
    w_spatial: float,
    cutoff: float,
    cov_scale: float = 1.0,
) -> dict[str, float]:
    """Compute all thirteen disorder metrics for a single structure.

    The pairwise distance matrix is constructed once and shared across all
    individual metric functions.

    Parameters
    ----------
    atoms:
        Element symbols.
    positions:
        Cartesian coordinates (Å).
    n_bins:
        Histogram bins for :func:`compute_h_spatial` and
        :func:`compute_rdf_deviation`.
    w_atom:
        Weight of ``H_atom`` in ``H_total``.
    w_spatial:
        Weight of ``H_spatial`` in ``H_total``.
    cutoff:
        Distance cutoff (Å) for Steinhardt and graph metrics.
    cov_scale:
        Bond detection threshold scale factor for the MM-level descriptors
        :func:`compute_bond_strain_rms`, :func:`compute_ring_fraction`, and
        :func:`compute_charge_frustration`.  Default: ``1.0``.

    Returns
    -------
    dict with keys matching :data:`pasted._atoms.ALL_METRICS`.
    """
    pts = np.array(positions, dtype=float)  # (n, 3)
    dists = _pdist(pts)  # condensed (n*(n-1)/2,)
    dmat = _squareform(dists)  # full (n, n)

    ha = compute_h_atom(atoms)
    hs = compute_h_spatial(dists, n_bins)
    return {
        "H_atom": ha,
        "H_spatial": hs,
        "H_total": w_atom * ha + w_spatial * hs,
        "RDF_dev": compute_rdf_deviation(pts, dists, n_bins),
        "shape_aniso": compute_shape_anisotropy(pts),
        **compute_steinhardt(pts, dmat, [4, 6, 8], cutoff),
        **compute_graph_metrics(dmat, cutoff),
        "bond_strain_rms": compute_bond_strain_rms(atoms, dmat, cov_scale),
        "ring_fraction": compute_ring_fraction(atoms, dmat, cov_scale),
        "charge_frustration": compute_charge_frustration(atoms, dmat, cov_scale),
    }


# ---------------------------------------------------------------------------
# Filter helper
# ---------------------------------------------------------------------------


def passes_filters(
    metrics: dict[str, float],
    filters: list[tuple[str, float, float]],
) -> bool:
    """Return ``True`` iff *metrics* satisfies every (metric, lo, hi) filter."""
    for metric, lo, hi in filters:
        v = metrics.get(metric, float("nan"))
        if math.isnan(v) or not (lo <= v <= hi):
            return False
    return True


# ---------------------------------------------------------------------------
# Angular entropy (diagnostic — not in ALL_METRICS, not in XYZ comment)
# ---------------------------------------------------------------------------


def compute_angular_entropy(
    positions: list[Vec3],
    cutoff: float,
    n_bins: int = 20,
) -> float:
    """Mean per-atom angular entropy of neighbour direction distributions.

    For each atom *i*, the directions to its neighbours within *cutoff* are
    projected onto the unit sphere.  The polar angle θ distribution is
    histogrammed and its Shannon entropy is computed.  The result is averaged
    over all atoms that have at least one neighbour.

    A value close to ln(*n_bins*) indicates a near-uniform (maximum-entropy)
    angular distribution — neighbours are spread evenly over the sphere.
    A low value indicates clustering of neighbours in certain directions,
    i.e. accidental local order.

    This metric is intended as a diagnostic for the ``maxent`` placement mode
    and is **not** included in ``ALL_METRICS`` or in XYZ comment lines.

    Parameters
    ----------
    positions:
        Cartesian coordinates (Å).
    cutoff:
        Neighbour distance cutoff (Å).
    n_bins:
        Number of histogram bins for the θ distribution (default: 20).

    Returns
    -------
    float
        Mean per-atom angular Shannon entropy.  Returns 0.0 for structures
        with fewer than two atoms or no neighbours within *cutoff*.
    """
    pts = np.array(positions, dtype=float)
    n = len(pts)
    if n < 2:
        return 0.0

    dmat = _squareform(_pdist(pts))
    np.fill_diagonal(dmat, np.inf)

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    safe_d = np.where(dmat[:, :, np.newaxis] > 0, dmat[:, :, np.newaxis], 1.0)
    d_hat = diff / safe_d                                   # (n, n, 3) unit vectors

    # Polar angle of each neighbour direction
    theta = np.arccos(np.clip(d_hat[:, :, 2], -1.0, 1.0))  # (n, n)

    mask = dmat <= cutoff  # (n, n)
    entropies: list[float] = []

    for i in range(n):
        nb_theta = theta[i, mask[i]]
        if len(nb_theta) < 2:
            continue
        counts, _ = np.histogram(nb_theta, bins=n_bins, range=(0.0, math.pi))
        entropies.append(_shannon_np(counts.astype(float)))

    return float(np.mean(entropies)) if entropies else 0.0
