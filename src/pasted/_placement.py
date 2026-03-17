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
    """Uniform random point inside an axis-aligned box centred at the origin."""
    return (
        rng.uniform(-lx / 2, lx / 2),
        rng.uniform(-ly / 2, ly / 2),
        rng.uniform(-lz / 2, lz / 2),
    )

# ---------------------------------------------------------------------------
# Post-placement repulsion relaxation
# ---------------------------------------------------------------------------

def relax_positions(
    atoms: list[str],
    positions: list[Vec3],
    cov_scale: float,
    max_cycles: int = 500,
) -> tuple[list[Vec3], bool]:
    """Resolve interatomic distance violations by iterative pair repulsion.

    For every pair (i, j) whose distance falls below
    ``cov_scale × (r_i + r_j)`` (Pyykkö single-bond covalent radii), both
    atoms are displaced along their connecting vector by half the deficit.
    The loop repeats until no violations remain or *max_cycles* is exhausted.

    Parameters
    ----------
    atoms:
        Element symbols, one per atom.
    positions:
        Initial Cartesian coordinates (Å).
    cov_scale:
        Scale factor applied to the sum of covalent radii.
    max_cycles:
        Maximum number of relaxation iterations.

    Returns
    -------
    (relaxed_positions, converged)
        *converged* is ``False`` only when *max_cycles* was reached with
        violations still present; the structure is still returned and usable.
    """
    n = len(atoms)
    if n < 2:
        return positions, True

    pts = np.array(positions, dtype=float)                   # (n, 3)
    radii = np.array([_cov_radius_ang(a) for a in atoms])   # (n,)
    thresh = cov_scale * (radii[:, np.newaxis] + radii[np.newaxis, :])  # (n, n)

    converged = False
    for _ in range(max_cycles):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
        dmat = np.sqrt((diff ** 2).sum(axis=2))               # (n, n)
        np.fill_diagonal(dmat, np.inf)

        viol_mask = np.triu(dmat < thresh, k=1)
        if not viol_mask.any():
            converged = True
            break

        vi, vj = np.where(viol_mask)
        for i, j in zip(vi, vj, strict=False):
            d = dmat[i, j]
            if d < 1e-10:  # coincident atoms: push in a random direction
                v_raw = np.random.default_rng().standard_normal(3)
                v = v_raw / np.linalg.norm(v_raw)
            else:
                v = diff[i, j] / d  # unit vector from j → i
            push = (thresh[i, j] - d) * 0.5
            pts[i] += push * v
            pts[j] -= push * v

    return [tuple(row) for row in pts], converged  # type: ignore[return-value]

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

        def sampler() -> Vec3:  # type: ignore[misc]
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

    Returns
    -------
    (atoms, positions)
        Always ``len(atoms)`` positions.
    """
    positions: list[Vec3] = [(0.0, 0.0, 0.0)]
    tip_dirs: list[Vec3 | None] = [None]
    tips: list[int] = [0]

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
                if (
                    d[0] * prev_dir[0]
                    + d[1] * prev_dir[1]
                    + d[2] * prev_dir[2]
                ) >= threshold:
                    break
                d = _unit_vec(rng)

        pt: Vec3 = (
            tp[0] + bl * d[0],
            tp[1] + bl * d[1],
            tp[2] + bl * d[2],
        )
        positions.append(pt)
        tip_dirs.append(d)

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
        positions.append((
            pp[0] + bl * d[0],
            pp[1] + bl * d[1],
            pp[2] + bl * d[2],
        ))

    return ordered, positions
