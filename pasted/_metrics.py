"""
pasted._metrics
===============
Disorder-metric computations.

All public metric functions accept ``pts`` (position array) plus a ``cutoff``
parameter and build their own neighbor lists internally.

**C++ path** (``HAS_GRAPH = True``): ``rdf_h_cpp`` and ``graph_metrics_cpp``
use a single ``FlatCellList`` pass for O(N*k) pair enumeration.
``scipy.spatial.distance.pdist`` / ``squareform`` are **not called** on this
path.

**Pure-Python fallback** (``HAS_GRAPH = False``): ``_compute_graph_ring_charge``
falls back to ``_squareform(_pdist(pts))`` — an O(N²) operation.  This path
is active when the C++17 extensions did not compile at install time (e.g. no
compiler available).  For N ≳ 500 the O(N²) cost becomes significant; see
the *Installation* section of the quickstart for performance guidance.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix as _csr_matrix
from scipy.sparse.csgraph import connected_components as _connected_components
from scipy.spatial import cKDTree as _cKDTree
from scipy.spatial.distance import pdist as _pdist
from scipy.spatial.distance import squareform as _squareform

# scipy >= 1.15 renamed sph_harm -> sph_harm_y with a different argument order.
# The except branch is kept for environments that still run scipy < 1.15;
# warn_unused_ignores is suppressed for this file in pyproject.toml
# [tool.mypy.overrides].
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

# Optional C++ acceleration
from ._ext import HAS_GRAPH as _HAS_GRAPH
from ._ext import HAS_STEINHARDT as _HAS_STEINHARDT
from ._ext import graph_metrics_cpp as _cpp_graph_metrics
from ._ext import rdf_h_cpp as _rdf_h_cpp

if _HAS_STEINHARDT:
    from ._ext import steinhardt_per_atom as _steinhardt_per_atom_cpp

# ---------------------------------------------------------------------------
# Low-level entropy helper
# ---------------------------------------------------------------------------


def _shannon_np(counts: np.ndarray) -> float:
    """Shannon entropy from a raw (un-normalized) count array."""
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


def compute_h_spatial(pts: np.ndarray, cutoff: float, n_bins: int) -> float:
    """Shannon entropy of the pair-distance histogram within *cutoff*.

    Only pairs with ``d_ij <= cutoff`` are included, matching the locality
    assumption used by all other metrics.  Higher values indicate a more
    uniform distribution of short-range distances.

    Parameters
    ----------
    pts:
        Positions array of shape ``(n, 3)``.
    cutoff:
        Neighbor distance cutoff (Å).
    n_bins:
        Number of histogram bins over ``[0, cutoff]``.
    """
    n = len(pts)
    if n < 2:
        return 0.0
    tree = _cKDTree(pts)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")  # type: ignore[call-arg]
    if len(pairs) == 0:
        return 0.0
    dists = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
    counts, _ = np.histogram(dists, bins=n_bins, range=(0.0, cutoff))
    return _shannon_np(counts.astype(float))


def compute_rdf_deviation(pts: np.ndarray, cutoff: float, n_bins: int) -> float:
    """RMS deviation of the empirical *g*(*r*) from an ideal-gas baseline.

    A value of 0 indicates a perfectly random (ideal-gas-like) distribution
    of pair distances within *cutoff*.  Only pairs with ``d_ij <= cutoff`` are
    included in the histogram, consistent with the other local metrics.

    Parameters
    ----------
    pts:
        Positions array of shape ``(n, 3)``.
    cutoff:
        Neighbor distance cutoff (Å).  The histogram range is ``[0, cutoff]``.
    n_bins:
        Number of histogram bins.
    """
    n = len(pts)
    if n < 2:
        return 0.0
    centroid = pts.mean(axis=0)
    r_bound = float(np.sqrt(((pts - centroid) ** 2).sum(axis=1)).max())
    if r_bound == 0.0:
        return 0.0
    tree = _cKDTree(pts)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")  # type: ignore[call-arg]
    if len(pairs) == 0:
        return 0.0
    dists = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
    rho = n / (4.0 / 3.0 * math.pi * r_bound**3)
    counts, edges = np.histogram(dists, bins=n_bins, range=(0.0, cutoff))
    centres = (edges[:-1] + edges[1:]) / 2.0
    bw = edges[1] - edges[0]
    ideal = rho * 4.0 * math.pi * centres**2 * bw * n / 2.0
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


def _steinhardt_per_atom_sparse(
    pts: np.ndarray,
    l_values: list[int],
    cutoff: float,
) -> dict[str, np.ndarray]:
    """Pure-Python O(N*k) fallback for :func:`compute_steinhardt_per_atom`.

    Enumerates neighbor pairs via ``scipy.spatial.cKDTree`` and evaluates
    spherical harmonics only for actual neighbor pairs (``d_ij <= cutoff``),
    giving O(N*k) complexity (k = mean neighbor count).

    Parameters
    ----------
    pts:
        Positions array of shape ``(n, 3)``.
    l_values:
        List of *l* values (e.g. ``[4, 6, 8]``).
    cutoff:
        Neighbor distance cutoff (Å).
    """
    n = len(pts)
    result: dict[str, np.ndarray] = {}

    tree = _cKDTree(pts)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")  # type: ignore[call-arg]

    if len(pairs) == 0:
        for lv in l_values:
            result[f"Q{lv}"] = np.zeros(n, dtype=float)
        return result

    # Build directed (both-way) neighbor index from undirected pairs
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])

    diff_nb = pts[rows] - pts[cols]                             # (n_bonds, 3)
    r_nb = np.linalg.norm(diff_nb, axis=1)
    safe_r_nb = np.where(r_nb > 0, r_nb, 1.0)
    d_hat_nb = diff_nb / safe_r_nb[:, np.newaxis]               # (n_bonds, 3)
    theta_nb = np.arccos(np.clip(d_hat_nb[:, 2], -1.0, 1.0))   # (n_bonds,)
    phi_nb = np.arctan2(d_hat_nb[:, 1], d_hat_nb[:, 0])         # (n_bonds,)

    deg = np.bincount(rows, minlength=n).astype(float)
    safe_deg = np.where(deg > 0, deg, 1.0)

    for lv in l_values:  # noqa: E741
        qlm_sq = np.zeros(n, dtype=float)
        for m in range(-lv, lv + 1):
            ylm_nb = _sph_harm(lv, m, phi_nb, theta_nb)             # (n_bonds,) complex
            re_sum = np.bincount(rows, weights=ylm_nb.real, minlength=n)
            im_sum = np.bincount(rows, weights=ylm_nb.imag, minlength=n)
            avg_sq = (re_sum / safe_deg) ** 2 + (im_sum / safe_deg) ** 2
            qlm_sq += avg_sq
        ql = np.sqrt(4 * math.pi / (2 * lv + 1) * qlm_sq)
        result[f"Q{lv}"] = np.where(deg > 0, ql, 0.0)

    return result


def compute_steinhardt_per_atom(
    pts: np.ndarray,
    l_values: list[int],
    cutoff: float,
) -> dict[str, np.ndarray]:
    """Per-atom Steinhardt Q_l values.

    When the C++ extension ``pasted._ext._steinhardt_core`` is available
    (``HAS_STEINHARDT = True``), the computation uses a sparse neighbor list
    built internally by the extension, evaluating spherical harmonics only for
    actual neighbor pairs.  This gives O(N*k) complexity (k = mean neighbor
    count).

    When the extension is absent the function falls back to a sparse
    Python/NumPy implementation using ``scipy.spatial.cKDTree`` for neighbor
    enumeration and ``np.bincount`` for accumulation.  Both paths have the
    same O(N*k) complexity.

    Parameters
    ----------
    pts:
        Positions array of shape ``(n, 3)``.
    l_values:
        List of *l* values (e.g. ``[4, 6, 8]``).
    cutoff:
        Neighbor distance cutoff (Å).

    Returns
    -------
    dict mapping ``"Q{l}"`` to a :class:`numpy.ndarray` of shape ``(n,)``.
    Atoms with no neighbors within *cutoff* are assigned Q_l = 0.
    """
    if _HAS_STEINHARDT:
        raw: dict[str, np.ndarray] = _steinhardt_per_atom_cpp(  # type: ignore[operator]
            pts, cutoff, l_values
        )
        return raw
    return _steinhardt_per_atom_sparse(pts, l_values, cutoff)


def compute_steinhardt(
    pts: np.ndarray,
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
    l_values:
        List of *l* values (e.g. ``[4, 6, 8]``).
    cutoff:
        Neighbor distance cutoff (Å).

    Returns
    -------
    dict mapping ``"Q{l}"`` to its global average value.
    """
    per_atom = compute_steinhardt_per_atom(pts, l_values, cutoff)
    return {k: float(v.mean()) for k, v in per_atom.items()}


def compute_graph_metrics(dmat: np.ndarray, cutoff: float) -> dict[str, float]:
    """Largest connected-component fraction and mean clustering coefficient.

    Pure-Python fallback used when ``HAS_GRAPH`` is ``False``.  The C++ path
    in ``graph_metrics_cpp`` is preferred and is invoked automatically by
    :func:`_compute_graph_ring_charge`.

    Parameters
    ----------
    dmat:
        Full n x n pairwise distance matrix.
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
# MM-level structural descriptors
# ---------------------------------------------------------------------------


def compute_ring_fraction(
    atoms: list[str],
    dmat: np.ndarray,
    cutoff: float,
) -> float:
    """Fraction of atoms that belong to at least one ring.

    Builds a neighbor graph using the *cutoff* distance threshold, then
    detects rings via a Union-Find spanning-tree construction: every
    back-edge (i.e. an edge between two vertices already in the same
    component) indicates a cycle, and both its endpoints are marked as
    ring members.

    Parameters
    ----------
    atoms:
        Element symbols (unused; retained for API symmetry with other metrics).
    dmat:
        Full n x n pairwise distance matrix (Å).
    cutoff:
        Distance cutoff (Å).  A pair is counted as connected when
        ``d_ij <= cutoff``.

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
        for j in range(i + 1, n):
            if 1e-6 < dmat[i, j] <= cutoff:
                if not _union(i, j):  # back-edge -> cycle detected
                    in_ring[i] = True
                    in_ring[j] = True

    return float(sum(in_ring) / n)


def compute_charge_frustration(
    atoms: list[str],
    dmat: np.ndarray,
    cutoff: float,
) -> float:
    """Variance of Pauling electronegativity differences across neighbor pairs.

    For each neighbor pair (i, j) within *cutoff*, the absolute
    electronegativity difference ``abs(chi_i - chi_j)`` is computed.  The metric
    is the *variance* of these differences over all neighbor pairs.

    A high value indicates a structure where electronegativity differences are
    inconsistently distributed across bonds: some neighbors are well matched
    while others are highly mismatched.  This is analogous to *charge
    frustration* in disordered materials, where local charge neutrality cannot
    be satisfied simultaneously at every site.

    Noble gases and elements without a Pauling value use the module-level
    fallback of 1.0 (see :func:`~pasted._atoms.pauling_electronegativity`).

    Parameters
    ----------
    atoms:
        Element symbols.
    dmat:
        Full n x n pairwise distance matrix (Å).
    cutoff:
        Distance cutoff (Å).  A pair is counted as connected when
        ``d_ij <= cutoff``.

    Returns
    -------
    float
        Variance of ``abs(delta-chi)`` across all neighbor pairs.  Returns 0.0 when
        fewer than two neighbor pairs are detected (variance is undefined for a
        single observation).
    """
    n = len(atoms)
    en = [_pauling_en(sym) for sym in atoms]
    diffs: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if 1e-6 < dmat[i, j] <= cutoff:
                diffs.append(abs(en[i] - en[j]))
    if len(diffs) < 2:
        return 0.0
    arr = np.array(diffs, dtype=float)
    return float(np.var(arr))


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def compute_moran_I_chi(
    atoms: list[str],
    dmat: np.ndarray,
    cutoff: float,
) -> float:
    r"""Moran\'s I spatial autocorrelation for Pauling electronegativity.

    Measures whether atoms with similar electronegativity cluster spatially.

    .. math::

        I = \frac{N}{W} \frac{\sum_{i \neq j} w_{ij}(\chi_i - \bar{\chi})
        (\chi_j - \bar{\chi})}{\sum_i (\chi_i - \bar{\chi})^2}

    where :math:`w_{ij} = 1` when :math:`d_{ij} \leq` *cutoff* and 0
    otherwise.

    Parameters
    ----------
    atoms:
        Element symbols.
    dmat:
        Full n x n pairwise distance matrix (Å).
    cutoff:
        Distance cutoff for the step-function weight matrix (Å).

    Returns
    -------
    float
        Moran\'s I in (-1, 1].  Returns 0.0 when all atoms share the same
        electronegativity or no pair falls within *cutoff*.

        * I ~= 0 : random spatial arrangement (target for disordered structures)
        * I > 0  : same-electronegativity atoms cluster spatially
        * I < 0  : alternating high/low electronegativity (ionic-crystal-like)
    """
    chi = np.array([_pauling_en(a) for a in atoms], dtype=float)
    chi_bar = chi.mean()
    dev = chi - chi_bar
    denom = float(np.sum(dev**2))
    if denom < 1e-30:
        return 0.0
    n = len(atoms)
    W = (dmat > 1e-6) & (dmat <= cutoff)
    W_sum = float(W.sum())
    if W_sum < 1e-30:
        return 0.0
    numer = float((W * np.outer(dev, dev)).sum())
    return float((n / W_sum) * (numer / denom))


def _compute_graph_ring_charge(
    atoms: list[str],
    pts: np.ndarray,
    radii: np.ndarray,
    cov_scale: float,
    cutoff: float,
    en_vals: np.ndarray,
) -> dict[str, float]:
    """Dispatch graph_lcc/cc, ring_fraction, charge_frustration, moran_I_chi.

    Uses the C++ ``graph_metrics_cpp`` (``HAS_GRAPH = True``) when available;
    this path builds a ``FlatCellList`` once and computes all five metrics in
    O(N*k).

    When ``HAS_GRAPH = False`` (C++ extension unavailable), falls back to the
    pure-Python implementations in this module.  The fallback constructs a
    full N×N distance matrix via ``scipy.spatial.distance.pdist`` /
    ``squareform`` — an **O(N²) operation** — and passes it to
    :func:`compute_graph_metrics`, :func:`compute_ring_fraction`,
    :func:`compute_charge_frustration`, and :func:`compute_moran_I_chi`.

    .. note::
        The O(N²) fallback can be slow for N ≳ 500.  Install with a C++17
        compiler to enable the O(N*k) C++ path (``HAS_GRAPH = True``).
    """
    if _HAS_GRAPH:
        # Single C++ call: FlatCellList built once, all 5 metrics computed.
        return dict(_cpp_graph_metrics(pts, radii, cov_scale, en_vals, cutoff))  # type: ignore[misc]
    # Pure-Python fallback
    dmat = _squareform(_pdist(pts))
    return {
        **compute_graph_metrics(dmat, cutoff),
        "ring_fraction": compute_ring_fraction(atoms, dmat, cutoff),
        "charge_frustration": compute_charge_frustration(atoms, dmat, cutoff),
        "moran_I_chi": compute_moran_I_chi(atoms, dmat, cutoff),
    }


def compute_all_metrics(
    atoms: list[str],
    positions: list[Vec3],
    n_bins: int,
    w_atom: float,
    w_spatial: float,
    cutoff: float,
    cov_scale: float = 1.0,
) -> dict[str, float]:
    """Compute all disorder metrics for a single structure.

    The exact count is ``len(ALL_METRICS)`` (currently
    :data:`~pasted._atoms.ALL_METRICS`).

    **C++ path** (``HAS_GRAPH = True``): all pair-enumeration uses a single
    ``FlatCellList`` built in C++, giving O(N*k) complexity throughout.
    All computation is single-threaded (OpenMP removed in v0.2.3).
    ``scipy.spatial.distance.pdist`` / ``squareform`` are not called.

    **Pure-Python fallback** (``HAS_GRAPH = False``): :func:`compute_h_spatial`
    and :func:`compute_rdf_deviation` use ``scipy.spatial.cKDTree`` (O(N*k)),
    but the five graph/ring/charge/Moran metrics are computed via
    :func:`_compute_graph_ring_charge`, which builds a full N×N distance
    matrix with ``pdist`` / ``squareform`` — an **O(N²) operation**.

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
        Distance cutoff (Å) for all local metrics.
    cov_scale:
        Retained for API compatibility; no longer used internally.
        Defaults to ``1.0``.

    Returns
    -------
    dict with keys matching :data:`pasted._atoms.ALL_METRICS`.
    """
    pts = np.array(positions, dtype=float)  # (n, 3)
    radii = np.array([_cov_radius_ang(a) for a in atoms])
    en_vals = np.array([_pauling_en(a) for a in atoms])

    ha = compute_h_atom(atoms)

    if _HAS_GRAPH:
        rdf_h = dict(_rdf_h_cpp(pts, cutoff, n_bins))  # type: ignore[misc]
        hs = float(rdf_h["h_spatial"])
        rdf_dev = float(rdf_h["rdf_dev"])
        graph_result = dict(_cpp_graph_metrics(pts, radii, cov_scale, en_vals, cutoff))  # type: ignore[misc]
    else:
        hs = compute_h_spatial(pts, cutoff, n_bins)
        rdf_dev = compute_rdf_deviation(pts, cutoff, n_bins)
        graph_result = _compute_graph_ring_charge(atoms, pts, radii, cov_scale, cutoff, en_vals)

    return {
        "H_atom": ha,
        "H_spatial": hs,
        "H_total": w_atom * ha + w_spatial * hs,
        "RDF_dev": rdf_dev,
        "shape_aniso": compute_shape_anisotropy(pts),
        **compute_steinhardt(pts, [4, 6, 8], cutoff),
        **graph_result,
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
# Angular entropy (diagnostic -- not in ALL_METRICS, not in XYZ comment)
# ---------------------------------------------------------------------------


def compute_angular_entropy(
    positions: list[Vec3],
    cutoff: float,
    n_bins: int = 20,
) -> float:
    """Mean per-atom angular entropy of neighbor direction distributions.

    For each atom *i*, the directions to its neighbors within *cutoff* are
    projected onto the unit sphere.  The polar angle theta distribution is
    histogrammed and its Shannon entropy is computed.  The result is averaged
    over all atoms that have at least one neighbor.

    A value close to ln(*n_bins*) indicates a near-uniform (maximum-entropy)
    angular distribution -- neighbors are spread evenly over the sphere.
    A low value indicates clustering of neighbors in certain directions,
    i.e. accidental local order.

    This metric is intended as a diagnostic for the ``maxent`` placement mode
    and is **not** included in ``ALL_METRICS`` or in XYZ comment lines.

    Uses ``scipy.spatial.cKDTree`` for O(N*k) pair enumeration instead of
    a full O(N^2) distance matrix.

    Parameters
    ----------
    positions:
        Cartesian coordinates (Å).
    cutoff:
        Neighbor distance cutoff (Å).
    n_bins:
        Number of histogram bins for the theta distribution (default: 20).

    Returns
    -------
    float
        Mean per-atom angular Shannon entropy.  Returns 0.0 for structures
        with fewer than two atoms or no neighbors within *cutoff*.
    """
    pts = np.array(positions, dtype=float)
    n = len(pts)
    if n < 2:
        return 0.0

    tree = _cKDTree(pts)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")  # type: ignore[call-arg]
    if len(pairs) == 0:
        return 0.0

    # Build directed (both-way) neighbor index
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])

    diff = pts[rows] - pts[cols]                                    # (n_bonds, 3)
    r = np.linalg.norm(diff, axis=1)
    safe_r = np.where(r > 0, r, 1.0)
    d_hat = diff / safe_r[:, np.newaxis]                            # (n_bonds, 3)
    theta = np.arccos(np.clip(d_hat[:, 2], -1.0, 1.0))             # (n_bonds,)

    entropies: list[float] = []
    for i in range(n):
        nb_mask = rows == i
        nb_theta = theta[nb_mask]
        if len(nb_theta) < 2:
            continue
        counts, _ = np.histogram(nb_theta, bins=n_bins, range=(0.0, math.pi))
        entropies.append(_shannon_np(counts.astype(float)))

    return float(np.mean(entropies)) if entropies else 0.0
