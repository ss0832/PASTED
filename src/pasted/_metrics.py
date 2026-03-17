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
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.sparse import csr_matrix as _csr_matrix
from scipy.sparse.csgraph import connected_components as _connected_components
from scipy.spatial.distance import pdist as _pdist
from scipy.spatial.distance import squareform as _squareform

# scipy >= 1.15 renamed sph_harm → sph_harm_y with a different argument order.
try:
    from scipy.special import sph_harm_y as _sph_harm_raw

    def _sph_harm(
        l: int,  # noqa: E741
        m: int,
        phi_azimuth: float | np.ndarray,
        theta_polar: float | np.ndarray,
    ) -> complex | np.ndarray:
        return cast(
            "complex | np.ndarray", _sph_harm_raw(l, m, theta_polar, phi_azimuth)
        )

except ImportError:
    from scipy.special import sph_harm as _sph_harm_raw  # type: ignore[no-redef]

    def _sph_harm(  # type: ignore[misc]
        l: int,  # noqa: E741
        m: int,
        phi_azimuth: float | np.ndarray,
        theta_polar: float | np.ndarray,
    ) -> complex | np.ndarray:
        return cast(
            "complex | np.ndarray", _sph_harm_raw(m, l, phi_azimuth, theta_polar)
        )


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


def compute_steinhardt(
    pts: np.ndarray,
    dmat: np.ndarray,
    l_values: list[int],
    cutoff: float,
) -> dict[str, float]:
    """Steinhardt bond-orientational order parameters *Q_l*, averaged over atoms.

    Vectorised over atoms and neighbours: one :func:`sph_harm` call per
    *(l, m)* processes all n×n angles simultaneously.

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
    dict mapping ``"Q{l}"`` to its value for each *l*.
    """
    n = len(pts)
    result: dict[str, float] = {}

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
        result[f"Q{l}"] = float(np.where(deg > 0, ql, 0.0).mean())

    return result


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
