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
# Unified entry point
# ---------------------------------------------------------------------------


def compute_all_metrics(
    atoms: list[str],
    positions: list[Vec3],
    n_bins: int,
    w_atom: float,
    w_spatial: float,
    cutoff: float,
) -> dict[str, float]:
    """Compute all ten disorder metrics for a single structure.

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
