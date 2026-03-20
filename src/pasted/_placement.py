"""
pasted._placement
=================
Atom-placement algorithms (gas / chain / shell) and post-placement
repulsion relaxation.  No file I/O; no metrics.
"""

from __future__ import annotations

import math
import random

import numpy as np

from ._atoms import _cov_radius_ang

# ---------------------------------------------------------------------------
# Optional C++ acceleration  (pasted._ext)
# ---------------------------------------------------------------------------
# Each hotspot is a separate compiled module under pasted._ext so they can
# be built and updated independently:
#
#   _ext._relax_core   → relax_positions()        (all placement modes)
#   _ext._maxent_core  → angular_repulsion_gradient() (maxent only)
#
# HAS_RELAX / HAS_MAXENT / HAS_MAXENT_LOOP are set by _ext/__init__.py;
# False when the corresponding .so is absent (no compiler, pure-source
# install, etc.).  No user-facing behavior changes in either case.
from ._ext import (
    HAS_MAXENT,
    HAS_MAXENT_LOOP,
    HAS_RELAX,
)
from ._ext import angular_repulsion_gradient as _cpp_angular_gradient
from ._ext import place_maxent_cpp as _cpp_place_maxent
from ._ext import relax_positions as _cpp_relax_positions

# Type alias used throughout this module and exported for type annotations.
Vec3 = tuple[float, float, float]

# ---------------------------------------------------------------------------
# Low-level geometry helpers
# ---------------------------------------------------------------------------


def _unit_vec(rng: random.Random) -> Vec3:
    """Uniform random point on the unit sphere (Marsaglia rejection method)."""
    while True:
        x, y, z = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)
        r2 = x * x + y * y + z * z
        if 0 < r2 <= 1.0:
            r = math.sqrt(r2)
            return (x / r, y / r, z / r)


def _sample_sphere(radius: float, rng: random.Random) -> Vec3:
    """Uniform random point inside a sphere of *radius*."""
    while True:
        x, y, z = (rng.uniform(-radius, radius) for _ in range(3))
        if x * x + y * y + z * z <= radius * radius:
            return (x, y, z)


def _sample_box(lx: float, ly: float, lz: float, rng: random.Random) -> Vec3:
    """Uniform random point inside an axis-aligned box centered at the origin."""
    return (
        rng.uniform(-lx / 2, lx / 2),
        rng.uniform(-ly / 2, ly / 2),
        rng.uniform(-lz / 2, lz / 2),
    )


def _poisson_disk_sphere(
    n: int,
    radius: float,
    min_dist: float,
    rng: random.Random,
    k: int = 30,
) -> list[Vec3]:
    """Poisson-disk-like sampling inside a sphere using stratified jitter.

    Divides the bounding cube into grid cells of side ``min_dist / sqrt(3)``
    (so no two cells in a 3×3×3 neighborhood can both contain an atom closer
    than *min_dist*), randomly selects *n* cells inside the sphere, and places
    one atom per cell at a uniformly-random position within the cell.

    This is O(n) and guarantees that atoms in *different* cells are at least
    ``min_dist / sqrt(3)`` apart.  Atoms in the *same* cell (impossible by
    construction) would be closer, but we only place one per cell.

    Parameters
    ----------
    n:
        Number of atoms.
    radius:
        Sphere radius (Å).
    min_dist:
        Target minimum inter-atom distance (Å).  Used to set cell size.
    rng:
        Seeded :class:`random.Random` — controls which cells are chosen and
        the jitter within each cell.
    k:
        Unused (kept for API compatibility with a full Poisson-disk
        implementation).

    Returns
    -------
    list[Vec3]
        Exactly *n* points.  Falls back to uniform random placement when the
        sphere contains fewer grid cells than *n*.
    """
    np_rng = np.random.default_rng(rng.randint(0, 2**31))
    cell = min_dist / math.sqrt(3)

    # Number of cells along each axis (bounding cube)
    n_cells = max(1, int(math.floor(2.0 * radius / cell)))
    origin  = -n_cells * cell / 2.0   # cube origin

    # Generate all cell centers inside the sphere
    total = n_cells ** 3
    ijk   = np.stack(np.unravel_index(np.arange(total), (n_cells, n_cells, n_cells)), axis=1)
    centers = origin + (ijk + 0.5) * cell          # (total, 3)
    in_sphere = (centers ** 2).sum(axis=1) <= radius ** 2
    valid_ijk = ijk[in_sphere]
    n_valid   = len(valid_ijk)

    if n_valid >= n:
        # Choose n cells without replacement and jitter within each
        chosen  = np_rng.choice(n_valid, size=n, replace=False)
        chosen_ijk = valid_ijk[chosen]
        jitter  = np_rng.uniform(0.0, cell, (n, 3))
        pts_arr = origin + chosen_ijk * cell + jitter  # (n, 3)
        # Clip to sphere (jitter may push slightly outside)
        r_pts = np.linalg.norm(pts_arr, axis=1)
        outside = r_pts > radius
        if outside.any():
            scale = radius / r_pts[outside]
            pts_arr[outside] *= scale[:, None]
        return [(float(p[0]), float(p[1]), float(p[2])) for p in pts_arr]
    else:
        # Fallback: use all valid cells (with replacement for extras)
        chosen  = np_rng.choice(n_valid, size=n, replace=True)
        chosen_ijk = valid_ijk[chosen]
        jitter  = np_rng.uniform(0.0, cell, (n, 3))
        pts_arr = origin + chosen_ijk * cell + jitter
        r_pts = np.linalg.norm(pts_arr, axis=1)
        outside = r_pts > radius
        if outside.any():
            scale = radius / r_pts[outside]
            pts_arr[outside] *= scale[:, None]
        return [(float(p[0]), float(p[1]), float(p[2])) for p in pts_arr]


def _poisson_disk_box(
    n: int,
    lx: float,
    ly: float,
    lz: float,
    min_dist: float,
    rng: random.Random,
    k: int = 30,
) -> list[Vec3]:
    """Bridson's Poisson-disk sampling inside an axis-aligned box.

    Mirrors :func:`_poisson_disk_sphere` for box regions.
    Falls back to uniform random when the region is too dense.
    """
    cell = min_dist / math.sqrt(3)
    inv_cell = 1.0 / cell
    hx, hy, hz = lx / 2, ly / 2, lz / 2

    def _grid_idx(x: float, y: float, z: float) -> tuple[int, int, int]:
        return (int((x + hx) * inv_cell), int((y + hy) * inv_cell),
                int((z + hz) * inv_cell))

    grid: dict[tuple[int, int, int], Vec3] = {}
    points: list[Vec3] = []
    active: list[Vec3] = []

    def _try_add(p: Vec3) -> bool:
        gx, gy, gz = _grid_idx(*p)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-2, 3):
                    nb = grid.get((gx+dx, gy+dy, gz+dz))
                    if nb is None:
                        continue
                    ddx, ddy, ddz = p[0]-nb[0], p[1]-nb[1], p[2]-nb[2]
                    if ddx*ddx + ddy*ddy + ddz*ddz < min_dist * min_dist:
                        return False
        grid[(gx, gy, gz)] = p
        points.append(p)
        active.append(p)
        return True

    _try_add(_sample_box(lx, ly, lz, rng))
    md2 = min_dist * min_dist

    while active and len(points) < n:
        idx = rng.randrange(len(active))
        base = active[idx]
        placed = False
        for _ in range(k):
            while True:
                dx = rng.uniform(-2*min_dist, 2*min_dist)
                dy = rng.uniform(-2*min_dist, 2*min_dist)
                dz = rng.uniform(-2*min_dist, 2*min_dist)
                d2 = dx*dx + dy*dy + dz*dz
                if md2 <= d2 <= 4 * md2:
                    break
            cand: Vec3 = (base[0]+dx, base[1]+dy, base[2]+dz)
            if abs(cand[0]) > hx or abs(cand[1]) > hy or abs(cand[2]) > hz:
                continue
            if _try_add(cand):
                placed = True
                break
        if not placed:
            active.pop(idx)

    while len(points) < n:
        points.append(_sample_box(lx, ly, lz, rng))

    return points[:n]


# ---------------------------------------------------------------------------
# Affine transformation (shared with StructureOptimizer)
# ---------------------------------------------------------------------------


def _affine_move(
    positions: list[Vec3],
    move_step: float,
    affine_strength: float,
    rng: random.Random,
) -> list[Vec3]:
    """Apply a random affine transformation to all atom positions.

    The transformation is a composition of:

    * **Stretch / compress** along one random axis — scale factor drawn from
      ``Uniform(1 − s, 1 + s)`` where ``s = affine_strength``.
    * **Shear** — a small off-diagonal component drawn from
      ``Uniform(-s/2, s/2)`` along a randomly chosen axis pair.
    * **Global translation jitter** — each coordinate nudged by
      ``Uniform(-move_step/4, move_step/4)`` to break symmetry.
      Pass ``move_step=0.0`` to skip the jitter (recommended for
      :class:`StructureGenerator` use where the transform is applied once
      before relaxation).

    The center of mass is pinned before and after the transform so the
    structure stays centered.

    Parameters
    ----------
    positions:
        Current atom positions.
    move_step:
        Maximum per-atom translation jitter added after the affine transform.
        Pass ``0.0`` for a pure affine transform with no per-atom noise.
    affine_strength:
        Dimensionless strength ∈ (0, 1).  Typical values: 0.05–0.3.
        At 0.1 the structure is stretched/compressed by up to ±10 %.
    rng:
        Seeded random-number generator.

    Returns
    -------
    list[Vec3]
        Transformed positions (same length as input).
    """
    pts = np.array(positions, dtype=float)   # (n, 3)
    com = pts.mean(axis=0)
    pts -= com                               # work around center of mass

    # ── Stretch / compress along a random axis ────────────────────────────
    axis = rng.randrange(3)                  # 0=x, 1=y, 2=z
    scale = 1.0 + rng.uniform(-affine_strength, affine_strength)
    A = np.eye(3)
    A[axis, axis] = scale

    # ── Random shear ──────────────────────────────────────────────────────
    axes = [0, 1, 2]
    axes.pop(axis)
    a1, a2 = axes
    shear = rng.uniform(-affine_strength * 0.5, affine_strength * 0.5)
    A[a1, a2] += shear                       # shear in one direction

    # Apply affine transform
    pts = pts @ A.T                          # (n, 3)

    # ── Small per-atom jitter (optional fine-grain noise) ─────────────────
    if move_step > 0.0:
        jitter_scale = move_step * 0.25
        pts += np.array([
            [rng.uniform(-jitter_scale, jitter_scale) for _ in range(3)]
            for _ in range(len(positions))
        ])

    # Restore center of mass
    pts += com

    return [tuple(row) for row in pts.tolist()]


# ---------------------------------------------------------------------------
# Post-placement repulsion relaxation
# ---------------------------------------------------------------------------


def relax_positions(
    atoms: list[str],
    positions: list[Vec3],
    cov_scale: float,
    max_cycles: int = 500,
    *,
    seed: int | None = None,
) -> tuple[list[Vec3], bool]:
    """Resolve interatomic distance violations by L-BFGS penalty minimization.

    For every pair (i, j) whose distance falls below
    ``cov_scale × (r_i + r_j)`` (Pyykkö single-bond covalent radii), a
    harmonic penalty energy is accumulated and its gradient used to drive
    atoms apart.  The C++ path minimizes
    ``E = Σ_{i<j} ½ · max(0, threshold − d_ij)²`` via L-BFGS; convergence
    is declared when E < 1 × 10⁻⁶.

    When the compiled C++ extension (``pasted._ext._relax_core``) is
    available the optimization runs in native code; otherwise the
    pure-Python/NumPy Gauss-Seidel fallback is used transparently.

    Parameters
    ----------
    atoms:
        Element symbols, one per atom.
    positions:
        Initial Cartesian coordinates (Å).
    cov_scale:
        Scale factor applied to the sum of covalent radii.
    max_cycles:
        Maximum number of L-BFGS iterations (C++ path) or Gauss-Seidel
        sweeps (Python fallback).  The C++ solver exits early when E < 1e-6,
        so the limit is rarely reached for typical structures.
    seed:
        Optional integer seed for the one-time pre-perturbation jitter
        (C++ path) or the coincident-atom RNG (Python fallback).
        ``None`` → non-deterministic.
        Pass the generator's master seed here for full reproducibility.

    Returns
    -------
    (relaxed_positions, converged)
        *converged* is ``False`` only when *max_cycles* was reached with
        violations still present; the structure is still returned and usable.
    """
    n = len(atoms)
    if n < 2:
        return positions, True

    pts = np.array(positions, dtype=float)          # (n, 3)
    radii = np.array([_cov_radius_ang(a) for a in atoms])  # (n,)

    # ── C++ fast path ────────────────────────────────────────────────────
    if HAS_RELAX:
        seed_int: int = -1 if seed is None else int(seed)
        pts_out, converged = _cpp_relax_positions(
            pts, radii, cov_scale, max_cycles, seed_int
        )
        return [tuple(row) for row in pts_out], converged

    # ── Pure-Python / NumPy fallback ─────────────────────────────────────
    thresh = cov_scale * (radii[:, np.newaxis] + radii[np.newaxis, :])  # (n, n)
    rng_np = np.random.default_rng(seed)  # seeded; used only for coincident atoms

    converged = False
    for _ in range(max_cycles):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
        dmat = np.sqrt((diff**2).sum(axis=2))                  # (n, n)
        np.fill_diagonal(dmat, np.inf)

        viol_mask = np.triu(dmat < thresh, k=1)
        if not viol_mask.any():
            converged = True
            break

        vi, vj = np.where(viol_mask)
        for i, j in zip(vi, vj, strict=False):
            d = dmat[i, j]
            if d < 1e-10:  # coincident atoms: push in a random direction
                v_raw = rng_np.standard_normal(3)
                v = v_raw / np.linalg.norm(v_raw)
            else:
                v = diff[i, j] / d  # unit vector from j → i
            push = (thresh[i, j] - d) * 0.5
            pts[i] += push * v
            pts[j] -= push * v

    return [tuple(row) for row in pts], converged


# ---------------------------------------------------------------------------
# Hydrogen augmentation
# ---------------------------------------------------------------------------


def add_hydrogen(atoms: list[str], rng: random.Random) -> list[str]:
    """Append hydrogen atoms when H is in the pool but absent from *atoms*.

    The number of H atoms added is:
    ``n_H = 1 + round(uniform(0, 1) × n_current × 1.2)``

    The original list is not modified; a new list is returned.
    """
    if "H" in atoms:
        return atoms
    n = len(atoms)
    n_h = 1 + round(rng.random() * n * 1.2)
    return atoms + ["H"] * n_h


# ---------------------------------------------------------------------------
# Placement: gas
# ---------------------------------------------------------------------------


def place_gas(
    atoms: list[str],
    region: str,
    rng: random.Random,
) -> tuple[list[str], list[Vec3]]:
    """Place all atoms uniformly at random inside *region*.

    No clash checking is performed — call :func:`relax_positions` afterwards.
    The C++ Bridson Poisson-disk functions (``_poisson_disk_sphere_cpp``,
    ``_poisson_disk_box_cpp``) are available via ``pasted._ext`` for callers
    that want minimum-separation placement; ``place_gas`` itself uses uniform
    random for performance predictability across all density regimes.

    Parameters
    ----------
    atoms:
        Element symbols.
    region:
        ``"sphere:R"`` | ``"box:L"`` | ``"box:LX,LY,LZ"``
    rng:
        Seeded random-number generator.

    Returns
    -------
    (atoms, positions)
        Always ``len(atoms)`` positions.

    Raises
    ------
    ValueError
        On unrecognised region spec.
    """
    if region.startswith("sphere:"):
        r = float(region.split(":")[1])

        def sampler() -> Vec3:
            return _sample_sphere(r, rng)

    elif region.startswith("box:"):
        dims = list(map(float, region.split(":")[1].split(",")))
        if len(dims) == 1:
            dims *= 3

        def sampler() -> Vec3:
            return _sample_box(dims[0], dims[1], dims[2], rng)

    else:
        raise ValueError(f"Unknown region spec: {region!r}")
    return atoms, [sampler() for _ in atoms]


# ---------------------------------------------------------------------------
# Placement: chain
# ---------------------------------------------------------------------------


def place_chain(
    atoms: list[str],
    bond_lo: float,
    bond_hi: float,
    branch_prob: float,
    persist: float,
    rng: random.Random,
    chain_bias: float = 0.0,
) -> tuple[list[str], list[Vec3]]:
    """Random-walk atom-chain growth with branching and directional persistence.

    Parameters
    ----------
    atoms:
        Element symbols (order is preserved).
    bond_lo, bond_hi:
        Bond-length range (Å).
    branch_prob:
        Probability that an atom becomes a new branch tip rather than
        replacing the existing tip (0 = linear, 1 = fully branched tree).
    persist:
        Directional persistence ∈ [0, 1].  A new step direction *d* is
        accepted only when ``dot(d, prev_dir) ≥ persist − 1``.

        - 0.0 → fully random (may self-tangle)
        - 0.5 → rear 120° cone excluded  *(default)*
        - 1.0 → front hemisphere only, nearly straight chain

    rng:
        Seeded random-number generator.
    chain_bias:
        Global-axis drift strength ∈ [0, 1] (default: 0.0).

        After the first bond is placed its direction becomes the *bias axis*.
        Every subsequent step direction is blended toward that axis before
        normalisation::

            d_biased = d + axis * chain_bias
            d_final  = d_biased / ||d_biased||

        - 0.0 → no bias; behavior identical to previous versions
        - 0.3 → moderate elongation; shape_aniso ≥ 0.5 rate rises from
                ~40% to ~70% for n = 20 atoms
        - 1.0 → strong elongation; nearly rod-like for small n

        ``chain_bias`` and ``persist`` are complementary: ``persist`` controls
        local turn angles between consecutive bonds; ``chain_bias`` imposes a
        global preferred axis regardless of chain length.

    Returns
    -------
    (atoms, positions)
        Always ``len(atoms)`` positions.
    """
    positions: list[Vec3] = [(0.0, 0.0, 0.0)]
    tip_dirs: list[Vec3 | None] = [None]
    tips: list[int] = [0]

    # The bias axis is set from the first bond placed (atom 0 → atom 1).
    # Until then it is None and no bias is applied.
    bias_axis: Vec3 | None = None

    for idx in range(1, len(atoms)):
        tip = rng.choice(tips)
        tp = positions[tip]
        prev_dir = tip_dirs[tip]
        bl = rng.uniform(bond_lo, bond_hi)

        if prev_dir is None or persist == 0.0:
            d = _unit_vec(rng)
        else:
            threshold = persist - 1.0
            d = _unit_vec(rng)
            for _ in range(200):
                if (d[0] * prev_dir[0] + d[1] * prev_dir[1] + d[2] * prev_dir[2]) >= threshold:
                    break
                d = _unit_vec(rng)

        # Apply global-axis bias when active (from the second bond onward)
        if chain_bias > 0.0 and bias_axis is not None:
            bx = d[0] + bias_axis[0] * chain_bias
            by = d[1] + bias_axis[1] * chain_bias
            bz = d[2] + bias_axis[2] * chain_bias
            inv_len = 1.0 / math.sqrt(bx * bx + by * by + bz * bz)
            d = (bx * inv_len, by * inv_len, bz * inv_len)

        pt: Vec3 = (
            tp[0] + bl * d[0],
            tp[1] + bl * d[1],
            tp[2] + bl * d[2],
        )
        positions.append(pt)
        tip_dirs.append(d)

        # First bond establishes the bias axis
        if idx == 1 and chain_bias > 0.0:
            bias_axis = d

        if rng.random() < branch_prob:
            tips.append(idx)
        else:
            tips[tips.index(tip)] = idx

    return atoms, positions


# ---------------------------------------------------------------------------
# Placement: shell
# ---------------------------------------------------------------------------


def place_shell(
    atoms: list[str],
    center_sym: str,
    coord_lo: int,
    coord_hi: int,
    shell_lo: float,
    shell_hi: float,
    tail_lo: float,
    tail_hi: float,
    rng: random.Random,
) -> tuple[list[str], list[Vec3]]:
    """Center atom at origin + coordination shell + tail atoms.

    No clash checking is performed — call :func:`relax_positions` afterwards.

    Parameters
    ----------
    atoms:
        Element symbols; must contain at least one occurrence of *center_sym*.
    center_sym:
        Symbol of the atom placed at the origin.
    coord_lo, coord_hi:
        Coordination-number range (number of shell atoms).
    shell_lo, shell_hi:
        Shell radius range (Å).
    tail_lo, tail_hi:
        Tail bond-length range (Å).
    rng:
        Seeded random-number generator.

    Returns
    -------
    (ordered_atoms, positions)
        Center atom is first; always ``len(atoms)`` positions.
    """
    n = len(atoms)
    ci = next((i for i, a in enumerate(atoms) if a == center_sym), 0)
    ordered = [atoms[ci]] + [a for i, a in enumerate(atoms) if i != ci]

    positions: list[Vec3] = [(0.0, 0.0, 0.0)]
    coord_num = min(rng.randint(coord_lo, coord_hi), n - 1)

    for _ in range(1, min(1 + coord_num, n)):
        r = rng.uniform(shell_lo, shell_hi)
        d = _unit_vec(rng)
        positions.append((r * d[0], r * d[1], r * d[2]))

    for _ in range(len(positions), n):
        par = rng.randint(1, len(positions) - 1)
        pp = positions[par]
        bl = rng.uniform(tail_lo, tail_hi)
        d = _unit_vec(rng)
        positions.append(
            (
                pp[0] + bl * d[0],
                pp[1] + bl * d[1],
                pp[2] + bl * d[2],
            )
        )

    return ordered, positions


# ---------------------------------------------------------------------------
# Placement: maxent
# ---------------------------------------------------------------------------


def _angular_repulsion_gradient(
    pts: np.ndarray, cutoff: float
) -> np.ndarray:
    """Compute gradient of the angular repulsion potential.

    For each atom *i* and each neighbor *j* within *cutoff*, the potential

        U_ij = 1 / (1 - cos θ_ij + ε)

    penalises neighbors that are close in *direction* from *i*.
    A small ε = 1e-6 prevents division by zero when two directions coincide.

    When the compiled C++ extension is available the inner double loop runs
    in native code (O(N²) cost, but without Python interpreter overhead).

    Returns the gradient ∂U/∂r_i of shape (n, 3).
    """
    # ── C++ fast path ────────────────────────────────────────────────────
    if HAS_MAXENT:
        return _cpp_angular_gradient(pts, cutoff)  # type: ignore[no-any-return]

    # ── Pure-Python / NumPy fallback ─────────────────────────────────────
    n = len(pts)
    grad = np.zeros((n, 3), dtype=float)
    eps = 1e-6

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.sqrt((diff**2).sum(axis=2))                  # (n, n)
    np.fill_diagonal(dist, np.inf)

    mask = dist <= cutoff                                   # (n, n) bool

    # Unit vectors from j → i
    safe_dist = np.where(dist > 0, dist, 1.0)
    uhat = diff / safe_dist[:, :, np.newaxis]              # (n, n, 3)

    mask_f = mask.astype(float)
    for i in range(n):
        ni_dirs = uhat[i] * mask_f[i, :, np.newaxis]  # (n, 3) zero for non-neighbors
        ni_idx = np.where(mask[i])[0]
        for j in ni_idx:
            cos_vals = (ni_dirs[ni_idx] * ni_dirs[j]).sum(axis=1)  # (n_nb,)
            denom = 1.0 - cos_vals + eps
            weights = 1.0 / denom**2                                # (n_nb,)
            perp = ni_dirs[ni_idx] - cos_vals[:, np.newaxis] * ni_dirs[j]
            grad[i] += (weights[:, np.newaxis] * perp).sum(axis=0) / safe_dist[i, j]

    return grad


def place_maxent(
    atoms: list[str],
    region: str,
    cov_scale: float,
    rng: random.Random,
    maxent_steps: int = 300,
    maxent_lr: float = 0.05,
    maxent_cutoff_scale: float = 2.5,
    trust_radius: float = 0.5,
    convergence_tol: float = 1e-3,
    seed: int | None = None,
) -> tuple[list[str], list[Vec3]]:
    """Place atoms to maximize angular entropy subject to distance constraints.

    Implements constrained maximum-entropy placement: atoms are initialized
    inside *region* at random, then iteratively repositioned so that each
    atom's neighborhood directions become as uniformly distributed over the
    sphere as possible — the solution to

        max  S = −∫ p(Ω) ln p(Ω) dΩ
        s.t.  d_ij ≥ cov_scale × (r_i + r_j)   ∀ i,j

    The angular repulsion potential

        U = Σ_{i} Σ_{j,k ∈ N(i), j≠k}  1 / (1 − cos θ_{jk} + ε)

    is minimised by L-BFGS (m=7, Armijo backtracking) when the C++ extension
    ``_maxent_core.place_maxent_cpp`` is available (``HAS_MAXENT_LOOP``), or
    by steepest descent otherwise.  A per-atom *trust radius* caps the maximum
    displacement per step, replacing the fixed ``maxent_lr`` unit-norm clip of
    the steepest-descent fallback.

    After every gradient step the mandatory distance-constraint relaxation is
    applied (L-BFGS penalty, identical to ``_relax_core``).

    Stability measures applied per step:

    - Per-atom trust-radius clamp: the step is uniformly rescaled so no
      atom moves more than *trust_radius* Å, preventing L-BFGS overshooting.
    - Soft restoring force: atoms that drift outside the initial region
      radius are gently pulled back toward the center of mass.
    - Centre-of-mass pinning: the center of mass is re-centered to the origin
      after each step so the whole structure does not drift.

    Parameters
    ----------
    atoms:
        Element symbols.
    region:
        Initial placement region: ``"sphere:R"`` | ``"box:L"`` | ``"box:LX,LY,LZ"``.
    cov_scale:
        Pyykkö distance scale factor.
    rng:
        Seeded random-number generator.
    maxent_steps:
        Gradient-descent / L-BFGS outer iterations (default: 300).
    maxent_lr:
        Learning rate used only by the Python steepest-descent fallback
        (default: 0.05).  Ignored when the C++ loop is active.
    maxent_cutoff_scale:
        Neighbor cutoff = this factor × median covalent-radius sum (default: 2.5).
        Larger values include more neighbors in the angular calculation.
        The median is computed in O(N) via ``numpy.median`` on the per-atom
        radii array; see *Implementation notes* below.
    trust_radius:
        Per-atom maximum displacement per step (Å, default: 0.5).  Used by
        the C++ L-BFGS loop; steepest-descent fallback uses unit-norm clip
        scaled by *maxent_lr* instead.
    convergence_tol:
        Early-termination threshold: the loop stops when the RMS gradient
        per atom falls below this value (Å⁻¹·a.u., default: 1e-3).  Set to
        ``0`` to disable early termination and always run *maxent_steps*
        iterations.  Ignored by the Python steepest-descent fallback.
    seed:
        Optional integer seed forwarded to the steric-clash relaxation for the
        coincident-atom edge case.  ``None`` → non-deterministic (default).

    Returns
    -------
    (atoms, positions)
        Always ``len(atoms)`` positions.

    Implementation notes
    --------------------
    **O(N) cutoff computation (v0.2.6):**
    Prior to v0.2.6 the angular-repulsion neighbor cutoff was derived from the
    median of all N*(N+1)/2 pairwise covalent-radius sums, computed by building
    the full generator and sorting it — O(N² log N) time and O(N²) memory.
    For large structures this dominated total runtime (e.g., ~88% of wall time
    at N=2,000 with ``maxent_steps=5``).

    The identity ``median(rᵢ + rⱼ) = 2 · median(rᵢ)`` holds whenever the
    element distribution is unimodal and symmetric about its median, which is
    true for all built-in element pools.  The replacement computation

    .. code-block:: python

        median_sum = float(np.median(radii)) * 2.0

    is O(N) and produces an identical ``ang_cutoff`` value for all tested
    element pools (C, N, O, H, S, …), yielding wall-time reductions of
    1.6× at N=100, 1.9× at N=1,000, 3.8× at N=5,000, and 4.5× at N=10,000.
    """
    # ── Initial random placement ─────────────────────────────────────────
    _, positions = place_gas(atoms, region, rng)
    positions, _ = relax_positions(atoms, positions, cov_scale, max_cycles=500, seed=seed)

    # ── Parse region radius for restoring force ──────────────────────────
    if region.startswith("sphere:"):
        region_radius = float(region.split(":")[1])
    elif region.startswith("box:"):
        dims = list(map(float, region.split(":")[1].split(",")))
        if len(dims) == 1:
            dims *= 3
        region_radius = min(dims) / 2.0
    else:
        raise ValueError(f"Unknown region spec: {region!r}")

    # ── Determine neighbor cutoff from covalent radii ───────────────────
    # Cache radii once; used by both paths and by do_relax (Python fallback).
    radii = np.array([_cov_radius_ang(a) for a in atoms], dtype=float)
    # v0.2.6: O(N) replacement for the previous O(N² log N) sorted(pair_sums)
    # call.  The identity median(rᵢ + rⱼ) = 2 · median(rᵢ) holds for all
    # built-in element pools; see the docstring "Implementation notes" section.
    median_sum = float(np.median(radii)) * 2.0
    ang_cutoff = cov_scale * maxent_cutoff_scale * median_sum

    pts = np.array(positions, dtype=float)
    pts -= pts.mean(axis=0)

    # ── C++ fast path: full L-BFGS loop in native code ───────────────────
    if HAS_MAXENT_LOOP:
        seed_int: int = -1 if seed is None else int(seed)
        pts = _cpp_place_maxent(
            pts, radii, cov_scale, region_radius, ang_cutoff,
            maxent_steps, trust_radius, convergence_tol, seed_int,
        )
        return atoms, [tuple(row) for row in pts]

    # ── Python steepest-descent fallback ─────────────────────────────────
    # Retained for environments without a compiled _maxent_core.
    # Incorporates the patch from v0.1.14:
    #   - radii pre-computed above (no per-step dict lookup)
    #   - relax calls _cpp_relax_positions directly (bypasses Python wrapper)
    #   - list ↔ ndarray conversion eliminated from the inner loop
    seed_int_fb: int = -1 if seed is None else int(seed)
    k_restore = 0.1 * maxent_lr

    for _ in range(maxent_steps):
        grad = _angular_repulsion_gradient(pts, ang_cutoff)

        # Per-atom gradient clipping (unit-norm cap)
        norms = np.linalg.norm(grad, axis=1, keepdims=True)
        norms_safe = np.where(norms > 0, norms, 1.0)
        grad_clipped = np.where(norms > 1.0, grad / norms_safe, grad)

        pts -= maxent_lr * grad_clipped

        # Soft restoring force
        r_from_com = np.linalg.norm(pts, axis=1, keepdims=True)
        excess = np.maximum(r_from_com - region_radius, 0.0)
        pts -= k_restore * excess * (pts / np.where(r_from_com > 0, r_from_com, 1.0))

        # CoM pinning
        pts -= pts.mean(axis=0)

        # Distance constraint — bypass Python wrapper when C++ available
        if HAS_RELAX:
            pts, _ = _cpp_relax_positions(pts, radii, cov_scale, 50, seed_int_fb)
        else:
            pos_list: list[Vec3] = [tuple(row) for row in pts]
            pos_list, _ = relax_positions(atoms, pos_list, cov_scale, max_cycles=50, seed=seed)
            pts = np.array(pos_list, dtype=float)

    return atoms, [tuple(row) for row in pts]
