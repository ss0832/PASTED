#!/usr/bin/env python3
"""
PASTED - Pointless Atom STructure with Entropy Diagnostics

Placement modes
---------------
  gas    Random placement inside a specified region (default).
  chain  Atom-by-atom growth from a seed via random walk + branching.
  shell  Center atom + coordination shell + tail atoms.

Disorder metrics
----------------
  H_atom        Compositional Shannon entropy
  H_spatial     Pairwise-distance Shannon entropy
  H_total       Weighted sum of the above two
  RDF_dev       RMS deviation of g(r) from ideal gas
  shape_aniso   Relative shape anisotropy (gyration tensor), 0=sphere 1=rod
  Q4 Q6 Q8      Steinhardt bond-orientational order parameters
  graph_lcc     Largest connected-component fraction (at --cutoff)
  graph_cc      Mean clustering coefficient (at --cutoff)

numpy + scipy are required.
"""

import argparse
import math
import random
import sys
from collections import Counter

try:
    import numpy as np
    # scipy >= 1.15 renamed sph_harm -> sph_harm_y(l, m, theta_polar, phi_azimuth)
    try:
        from scipy.special import sph_harm_y as _sph_harm_raw
        def _sph_harm(l: int, m: int, phi_azimuth: float, theta_polar: float) -> complex:
            return _sph_harm_raw(l, m, theta_polar, phi_azimuth)
    except ImportError:
        from scipy.special import sph_harm as _sph_harm_raw  # type: ignore[no-redef]
        def _sph_harm(l: int, m: int, phi_azimuth: float, theta_polar: float) -> complex:  # type: ignore[misc]
            return _sph_harm_raw(m, l, phi_azimuth, theta_polar)
    from scipy.spatial.distance import pdist as _pdist, squareform as _squareform
    from scipy.sparse import csr_matrix as _csr_matrix
    from scipy.sparse.csgraph import connected_components as _connected_components
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    np = None

# ---------------------------------------------------------------------------
# Atomic data  Z = 1-106
# ---------------------------------------------------------------------------

ATOMIC_NUMBERS: dict[str, int] = {
    "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
    "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
    "K":19,"Ca":20,
    "Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,
    "Ga":31,"Ge":32,"As":33,"Se":34,"Br":35,"Kr":36,
    "Rb":37,"Sr":38,
    "Y":39,"Zr":40,"Nb":41,"Mo":42,"Tc":43,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,
    "In":49,"Sn":50,"Sb":51,"Te":52,"I":53,"Xe":54,
    "Cs":55,"Ba":56,
    "La":57,"Ce":58,"Pr":59,"Nd":60,"Pm":61,"Sm":62,"Eu":63,"Gd":64,
    "Tb":65,"Dy":66,"Ho":67,"Er":68,"Tm":69,"Yb":70,"Lu":71,
    "Hf":72,"Ta":73,"W":74,"Re":75,"Os":76,"Ir":77,"Pt":78,"Au":79,"Hg":80,
    "Tl":81,"Pb":82,"Bi":83,"Po":84,"At":85,"Rn":86,
    "Fr":87,"Ra":88,
    "Ac":89,"Th":90,"Pa":91,"U":92,"Np":93,"Pu":94,"Am":95,"Cm":96,
    "Bk":97,"Cf":98,"Es":99,"Fm":100,"Md":101,"No":102,"Lr":103,
    "Rf":104,"Db":105,"Sg":106,
}

_Z_TO_SYM: dict[int, str] = {v: k for k, v in ATOMIC_NUMBERS.items()}
_ALL_Z: list[int] = sorted(_Z_TO_SYM)

_ALL_METRICS = {
    "H_atom","H_spatial","H_total","RDF_dev",
    "shape_aniso","Q4","Q6","Q8",
    "graph_lcc","graph_cc",
}

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_element_spec(spec: str) -> list[str]:
    z_set: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"Range {token!r}: lower > upper.")
            z_set.update(range(lo, hi + 1))
        else:
            z_set.add(int(token))
    symbols = []
    for z in sorted(z_set):
        if z not in _Z_TO_SYM:
            raise ValueError(f"Z={z} not supported (range {_ALL_Z[0]}-{_ALL_Z[-1]})")
        symbols.append(_Z_TO_SYM[z])
    if not symbols:
        raise ValueError("Element specification resolved to an empty pool.")
    return symbols


def default_element_pool() -> list[str]:
    return [_Z_TO_SYM[z] for z in _ALL_Z]


def parse_lo_hi(s: str, name: str) -> tuple[float, float]:
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"--{name} must be 'LO:HI', got {s!r}")
    return float(parts[0]), float(parts[1])


def parse_int_range(s: str) -> tuple[int, int]:
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Must be 'MIN:MAX', got {s!r}")
    lo, hi = int(parts[0]), int(parts[1])
    if lo < 1 or lo > hi:
        raise ValueError(f"MIN must be >=1 and <= MAX, got {s!r}")
    return lo, hi


def parse_filter(f: str) -> tuple[str, float, float]:
    parts = f.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected 'METRIC:MIN:MAX', got {f!r}")
    metric, lo_s, hi_s = parts
    if metric not in _ALL_METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Valid: {sorted(_ALL_METRICS)}")
    lo = -math.inf if lo_s.strip() == "-" else float(lo_s)
    hi =  math.inf if hi_s.strip() == "-" else float(hi_s)
    if lo > hi:
        raise ValueError(f"Filter {f!r}: MIN > MAX.")
    return metric, lo, hi

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_charge_mult(
    atoms_list: list[str], charge: int, mult: int
) -> tuple[bool, str]:
    total_z = sum(ATOMIC_NUMBERS[a] for a in atoms_list)
    n_e = total_z - charge
    if n_e <= 0:
        return False, f"n_electrons={n_e} (total_Z={total_z}, charge={charge:+d})."
    n_up = mult - 1
    if (n_e % 2) != (n_up % 2):
        return False, (
            f"parity mismatch: n_electrons={n_e} (charge={charge:+d}), "
            f"mult={mult} -> n_unpaired={n_up}."
        )
    comp = " ".join(f"{s}:{c}" for s, c in sorted(Counter(atoms_list).items()))
    return True, (
        f"n_atoms={len(atoms_list)} total_Z={total_z} "
        f"n_electrons={n_e} charge={charge:+d} mult={mult} comp=[{comp}]"
    )

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

Vec3 = tuple[float, float, float]


def _dist(p: Vec3, q: Vec3) -> float:
    return math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2)


def _unit_vec(rng: random.Random) -> Vec3:
    """Uniform point on the unit sphere (Marsaglia)."""
    while True:
        x, y, z = rng.uniform(-1,1), rng.uniform(-1,1), rng.uniform(-1,1)
        r2 = x*x + y*y + z*z
        if 0 < r2 <= 1.0:
            r = math.sqrt(r2)
            return (x/r, y/r, z/r)


def _sample_sphere(radius: float, rng: random.Random) -> Vec3:
    while True:
        x, y, z = (rng.uniform(-radius, radius) for _ in range(3))
        if x*x + y*y + z*z <= radius*radius:
            return (x, y, z)


def _sample_box(lx: float, ly: float, lz: float, rng: random.Random) -> Vec3:
    return (rng.uniform(-lx/2,lx/2), rng.uniform(-ly/2,ly/2), rng.uniform(-lz/2,lz/2))


def _no_clash(pt: Vec3, positions: list[Vec3], min_dist: float) -> bool:
    """Return True if pt is at least min_dist away from every position in the list."""
    if not positions:
        return True
    # np.asarray on a list of tuples is faster than np.array and avoids a copy
    arr = np.asarray(positions, dtype=float)
    p   = pt[0] - arr[:, 0]; q = pt[1] - arr[:, 1]; r = pt[2] - arr[:, 2]
    return bool(np.min(p*p + q*q + r*r) >= min_dist * min_dist)

# ---------------------------------------------------------------------------
# Placement: gas
# ---------------------------------------------------------------------------

def place_gas(
    atoms: list[str], region: str, min_dist: float,
    max_att: int, rng: random.Random
) -> tuple[list[str], list[Vec3]]:
    if region.startswith("sphere:"):
        r = float(region.split(":")[1])
        gen = lambda: _sample_sphere(r, rng)
    elif region.startswith("box:"):
        dims = list(map(float, region.split(":")[1].split(",")))
        if len(dims) == 1: dims *= 3
        gen = lambda: _sample_box(dims[0], dims[1], dims[2], rng)
    else:
        raise ValueError(f"Unknown region: {region!r}")
    positions: list[Vec3] = []
    for idx, atom in enumerate(atoms):
        placed = False
        for _ in range(max_att):
            pt = gen()
            if _no_clash(pt, positions, min_dist):
                positions.append(pt); placed = True; break
        if not placed:
            raise RuntimeError(
                f"gas: cannot place atom #{idx+1} ({atom}) after {max_att} attempts.")
    return atoms, positions

# ---------------------------------------------------------------------------
# Placement: chain
# ---------------------------------------------------------------------------

def place_chain(
    atoms: list[str], bond_lo: float, bond_hi: float,
    branch_prob: float, min_dist: float, max_att: int, rng: random.Random
) -> tuple[list[str], list[Vec3]]:
    """
    Random-walk growth with branching.
    No region constraint; the chain grows freely.
    """
    positions: list[Vec3] = [(0.0, 0.0, 0.0)]
    tips: list[int] = [0]
    for idx in range(1, len(atoms)):
        placed = False
        for _ in range(max_att):
            tip = rng.choice(tips)
            tp = positions[tip]
            bl = rng.uniform(bond_lo, bond_hi)
            d = _unit_vec(rng)
            pt: Vec3 = (tp[0]+bl*d[0], tp[1]+bl*d[1], tp[2]+bl*d[2])
            if _no_clash(pt, positions, min_dist):
                positions.append(pt)
                if rng.random() < branch_prob:
                    tips.append(idx)          # branch: keep parent tip
                else:
                    tips[tips.index(tip)] = idx  # linear: advance tip
                placed = True; break
        if not placed:
            raise RuntimeError(
                f"chain: cannot place atom #{idx+1} after {max_att} attempts.")
    return atoms, positions

# ---------------------------------------------------------------------------
# Placement: shell
# ---------------------------------------------------------------------------

def place_shell(
    atoms: list[str], center_sym: str,
    coord_lo: int, coord_hi: int,
    shell_lo: float, shell_hi: float,
    tail_lo: float, tail_hi: float,
    min_dist: float, max_att: int, rng: random.Random
) -> tuple[list[str], list[Vec3]]:
    """
    Center atom at origin + coordination shell + tail atoms.
    """
    n = len(atoms)
    # Reorder so center atom is first
    ci = next((i for i, a in enumerate(atoms) if a == center_sym), 0)
    ordered = [atoms[ci]] + [a for i, a in enumerate(atoms) if i != ci]

    positions: list[Vec3] = [(0.0, 0.0, 0.0)]
    coord_num = min(rng.randint(coord_lo, coord_hi), n - 1)

    # Coordination shell
    for idx in range(1, min(1 + coord_num, n)):
        placed = False
        for _ in range(max_att):
            r = rng.uniform(shell_lo, shell_hi)
            d = _unit_vec(rng)
            pt: Vec3 = (r*d[0], r*d[1], r*d[2])
            if _no_clash(pt, positions, min_dist):
                positions.append(pt); placed = True; break
        if not placed:
            raise RuntimeError(f"shell: cannot place coord atom #{idx} after {max_att} attempts.")

    # Tail atoms grow from any non-center atom
    for idx in range(len(positions), n):
        placed = False
        for _ in range(max_att):
            par = rng.randint(1, len(positions) - 1)
            pp = positions[par]
            bl = rng.uniform(tail_lo, tail_hi)
            d = _unit_vec(rng)
            pt = (pp[0]+bl*d[0], pp[1]+bl*d[1], pp[2]+bl*d[2])
            if _no_clash(pt, positions, min_dist):
                positions.append(pt); placed = True; break
        if not placed:
            raise RuntimeError(f"shell: cannot place tail atom #{idx-coord_num} after {max_att} attempts.")

    return ordered, positions

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _shannon_np(counts: np.ndarray) -> float:
    """Shannon entropy from a raw count array (not yet normalised)."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log(p)))


def compute_h_atom(atoms: list[str]) -> float:
    counts = Counter(atoms)
    n = len(atoms)
    return _shannon_np(np.array(list(counts.values()), dtype=float))


def compute_h_spatial(dists: np.ndarray, n_bins: int) -> float:
    """Spatial Shannon entropy from precomputed condensed distance array."""
    if len(dists) < 1:
        return 0.0
    counts, _ = np.histogram(dists, bins=n_bins)
    return _shannon_np(counts.astype(float))


def compute_rdf_deviation(
    pts: np.ndarray, dists: np.ndarray, n_bins: int
) -> float:
    """
    RMS deviation of empirical g(r) from ideal-gas baseline.
    Uses precomputed pts (n,3) and condensed distances.
    """
    if len(dists) < 1:
        return 0.0
    n = len(pts)
    r_max = float(dists.max())
    r_bound = float(np.sqrt((pts ** 2).sum(axis=1)).max())
    if r_bound == 0 or r_max == 0:
        return 0.0
    rho = n / (4 / 3 * math.pi * r_bound ** 3)
    counts, edges = np.histogram(dists, bins=n_bins, range=(0.0, r_max))
    centres = (edges[:-1] + edges[1:]) / 2
    bw = edges[1] - edges[0]
    ideal = rho * 4 * math.pi * centres ** 2 * bw * n / 2
    mask = ideal > 0
    if not mask.any():
        return 0.0
    return float(np.sqrt(np.mean(((counts[mask] / ideal[mask]) - 1.0) ** 2)))


def compute_shape_anisotropy(pts: np.ndarray) -> float:
    """
    Relative shape anisotropy from the gyration tensor.
    Range: 0 (spherical) to 1 (rod-like).
    """
    if len(pts) < 2:
        return float("nan")
    p = pts - pts.mean(axis=0)
    T = (p.T @ p) / len(p)
    lam = np.linalg.eigvalsh(T)
    s = float(lam.sum())
    if s == 0:
        return 0.0
    return float(np.clip(1.5 * float(np.sum(lam ** 2)) / s ** 2 - 0.5, 0.0, 1.0))


def compute_steinhardt(
    pts: np.ndarray, dmat: np.ndarray,
    l_values: list[int], cutoff: float,
) -> dict[str, float]:
    """
    Steinhardt Q_l averaged over all atoms.
    Vectorized over atoms and neighbours: one sph_harm_y call per (l, m)
    processes all n×n angles at once.
    dmat is the full n×n distance matrix.
    """
    n = len(pts)
    result: dict[str, float] = {}

    # Neighbour mask (n, n) — no self-loops
    mask = (dmat <= cutoff)
    np.fill_diagonal(mask, False)
    deg      = mask.sum(axis=1).astype(float)       # (n,)
    safe_deg = np.where(deg > 0, deg, 1.0)
    mask_f   = mask.astype(float)                   # (n, n)

    # Normalised direction vectors (n, n, 3)
    diff   = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]   # (n, n, 3)
    safe_r = np.where(dmat[:, :, np.newaxis] > 0,
                      dmat[:, :, np.newaxis], 1.0)
    d_hat  = diff / safe_r                                     # (n, n, 3)

    # Spherical angles (n, n)
    theta = np.arccos(np.clip(d_hat[:, :, 2], -1.0, 1.0))    # polar
    phi   = np.arctan2(d_hat[:, :, 1], d_hat[:, :, 0])        # azimuthal

    for l in l_values:
        qlm_sq = np.zeros(n, dtype=float)
        for m in range(-l, l + 1):
            # One call handles all n×n angles simultaneously
            ylm = _sph_harm(l, m, phi, theta)                  # (n, n) complex
            avg = (ylm * mask_f).sum(axis=1) / safe_deg        # (n,) complex
            qlm_sq += np.abs(avg) ** 2

        ql = np.sqrt(4 * math.pi / (2 * l + 1) * qlm_sq)
        result[f"Q{l}"] = float(np.where(deg > 0, ql, 0.0).mean())

    return result


def compute_graph_metrics(dmat: np.ndarray, cutoff: float) -> dict[str, float]:
    """
    graph_lcc (largest connected-component fraction) and
    graph_cc  (mean clustering coefficient).
    Uses scipy connected_components and matrix multiplication for triangles.
    """
    n = len(dmat)
    if n < 2:
        return {"graph_lcc": 1.0, "graph_cc": 0.0}

    # Boolean adjacency matrix (no self-loops)
    adj = (dmat <= cutoff)
    np.fill_diagonal(adj, False)

    # Largest connected component via scipy sparse
    _, labels = _connected_components(
        _csr_matrix(adj), directed=False, return_labels=True
    )
    graph_lcc = float(np.bincount(labels).max()) / n

    # Mean clustering coefficient
    # tri[i] = number of closed triangles at i
    #         = 0.5 * (A @ A)[i] · A[i]  (summed over j)
    # max_tri[i] = k_i * (k_i - 1) / 2
    deg = adj.sum(axis=1).astype(float)                     # (n,)
    A   = adj.astype(float)
    tri = (A @ A * A).sum(axis=1) / 2.0                    # (n,)
    max_tri = deg * (deg - 1) / 2.0
    mask = max_tri > 0
    graph_cc = float(np.mean(tri[mask] / max_tri[mask])) if mask.any() else 0.0

    return {"graph_lcc": graph_lcc, "graph_cc": graph_cc}


def compute_all_metrics(
    atoms: list[str], positions: list[Vec3],
    n_bins: int, w_atom: float, w_spatial: float, cutoff: float,
) -> dict[str, float]:
    """
    Compute all disorder metrics.
    The positions-to-numpy conversion and pairwise distance matrix are
    computed once here and shared across all metric functions.
    """
    pts   = np.array(positions, dtype=float)   # (n, 3)
    dists = _pdist(pts)                         # condensed (n*(n-1)/2,)
    dmat  = _squareform(dists)                  # full (n, n)

    ha = compute_h_atom(atoms)
    hs = compute_h_spatial(dists, n_bins)
    return {
        "H_atom":      ha,
        "H_spatial":   hs,
        "H_total":     w_atom * ha + w_spatial * hs,
        "RDF_dev":     compute_rdf_deviation(pts, dists, n_bins),
        "shape_aniso": compute_shape_anisotropy(pts),
        **compute_steinhardt(pts, dmat, [4, 6, 8], cutoff),
        **compute_graph_metrics(dmat, cutoff),
    }


def passes_filters(metrics: dict[str, float], filters: list[tuple[str, float, float]]) -> bool:
    for metric, lo, hi in filters:
        v = metrics.get(metric, float("nan"))
        if math.isnan(v) or not (lo <= v <= hi):
            return False
    return True

# ---------------------------------------------------------------------------
# XYZ output
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return "nan" if math.isnan(v) else f"{v:.4f}"


def format_xyz(
    atoms: list[str], positions: list[Vec3],
    charge: int, mult: int, metrics: dict[str, float], prefix: str = ""
) -> str:
    comp = ",".join(f"{s}:{c}" for s, c in sorted(Counter(atoms).items()))
    metric_str = "  ".join(f"{k}={_fmt(v)}" for k, v in metrics.items())
    comment = f"{prefix} charge={charge:+d} mult={mult} comp=[{comp}]  {metric_str}".strip()
    lines = [str(len(atoms)), comment]
    for atom, (x, y, z) in zip(atoms, positions):
        lines.append(f"{atom:<4s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pasted",
        description=(
            "PASTED: Pointless Atom STructure with Entropy Diagnostics\n"
            "Elements Z=1-106. Modes: gas / chain / shell.\n"
            "Metrics: H_atom H_spatial H_total RDF_dev shape_aniso "
            "Q4 Q6 Q8 graph_lcc graph_cc\n"
            "Requires: numpy scipy"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
element-spec syntax: ranges and/or comma lists of atomic numbers.
  '1-30'       H to Zn
  '6,7,8'      C, N, O
  '1-10,26'    H-Ne plus Fe
  '72-80'      5d metals Hf-Hg
  (omitted)    all Z=1-106

examples
--------
  # gas mode
  python pasted.py --n-atoms 12 --elements 1-30 --charge 0 --mult 1 \\
      --mode gas --region sphere:9 --n-samples 50 --filter H_total:1.8:3.5

  # chain mode (organic-like)
  python pasted.py --n-atoms 15 --elements 6,7,8 --charge 0 --mult 1 \\
      --mode chain --branch-prob 0.4 --n-samples 20

  # shell mode (coordination-complex-like)
  python pasted.py --n-atoms 12 --elements 6,7,8,26 --charge 0 --mult 1 \\
      --mode shell --center-z 26 --coord-range 4:6 --n-samples 10

  # filter on new structural metrics
  python pasted.py --n-atoms 10 --elements 6,7,8 --charge 0 --mult 1 \\
      --mode chain --n-samples 100 \\
      --filter shape_aniso:0.4:- --filter graph_lcc:0.8:- -o out.xyz
""",
    )

    req = p.add_argument_group("required")
    req.add_argument("--n-atoms", type=int, required=True,
                     help="Total number of atoms per structure.")
    req.add_argument("--charge", type=int, required=True,
                     help="Total system charge.")
    req.add_argument("--mult", type=int, required=True,
                     help="Spin multiplicity 2S+1 (HS/LS not enforced).")

    mg = p.add_argument_group("placement mode")
    mg.add_argument("--mode", choices=["gas","chain","shell"], default="gas")
    mg.add_argument("--region",
                    help="[gas] 'sphere:R' | 'box:L' | 'box:LX,LY,LZ' (Angstrom). Required for gas.")
    mg.add_argument("--branch-prob", type=float, default=0.3,
                    help="[chain] Branching probability (default: 0.3).")
    mg.add_argument("--bond-range", default="1.2:1.6", metavar="LO:HI",
                    help="[chain/shell-tails] Bond length range Angstrom (default: 1.2:1.6).")
    mg.add_argument("--center-z", type=int, default=None, metavar="Z",
                    help="[shell] Atomic number of center atom. Default: max Z>20 in pool.")
    mg.add_argument("--coord-range", default="4:8", metavar="MIN:MAX",
                    help="[shell] Coordination number range (default: 4:8).")
    mg.add_argument("--shell-radius", default="1.8:2.5", metavar="LO:HI",
                    help="[shell] Shell radius range Angstrom (default: 1.8:2.5).")

    eg = p.add_argument_group("elements")
    eg.add_argument("--elements", default=None, metavar="SPEC",
                    help="Element pool by atomic number (default: all Z=1-106).")

    pg = p.add_argument_group("placement")
    pg.add_argument("--min-dist", type=float, default=0.8,
                    help="Minimum interatomic distance Angstrom (default: 0.8).")
    pg.add_argument("--max-attempts", type=int, default=10000,
                    help="Max placement attempts per atom (default: 10000).")

    sg = p.add_argument_group("sampling")
    sg.add_argument("--n-samples", type=int, default=1)
    sg.add_argument("--seed", type=int, default=None)

    xg = p.add_argument_group("metrics")
    xg.add_argument("--n-bins", type=int, default=20,
                    help="Histogram bins for H_spatial/RDF_dev (default: 20).")
    xg.add_argument("--w-atom", type=float, default=0.5)
    xg.add_argument("--w-spatial", type=float, default=0.5)
    xg.add_argument("--cutoff", type=float, default=2.0,
                    help="Distance cutoff Angstrom for graph_* and Q_l (default: 2.0).")

    fg = p.add_argument_group("filtering")
    fg.add_argument("--filter", action="append", default=[], dest="filters",
                    metavar="METRIC:MIN:MAX",
                    help=(
                        "Keep structures where METRIC in [MIN,MAX]. "
                        "Use '-' for open bound. Repeatable."
                    ))

    og = p.add_argument_group("output")
    og.add_argument("--validate", action="store_true",
                    help="Validate charge/mult against one random composition, then exit.")
    og.add_argument("-o","--output", default=None,
                    help="Output XYZ file (default: stdout).")
    return p


def _validate_center_z(center_z: int | None, element_pool: list[str]) -> str | None:
    """
    Validate --center-z if provided and return its symbol.
    Returns None when --center-z is omitted: center atom is chosen randomly
    per sample from the element pool (fully random, chaos-first).
    """
    if center_z is None:
        return None
    if center_z not in _Z_TO_SYM:
        raise ValueError(f"--center-z {center_z}: unknown atomic number.")
    sym = _Z_TO_SYM[center_z]
    if sym not in element_pool:
        raise ValueError(f"--center-z {center_z} ({sym}) is not in the element pool.")
    return sym


def main() -> None:
    if not _HAS_SCIPY:
        print("[ERROR] numpy + scipy required: pip install numpy scipy", file=sys.stderr)
        sys.exit(1)

    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "gas" and not args.region:
        parser.error("--region is required for --mode gas")

    # Parse range arguments
    try:
        bond_lo, bond_hi = parse_lo_hi(args.bond_range, "bond-range")
        shell_lo, shell_hi = parse_lo_hi(args.shell_radius, "shell-radius")
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr); sys.exit(1)

    # Element pool
    try:
        element_pool = parse_element_spec(args.elements) if args.elements else default_element_pool()
    except ValueError as e:
        print(f"[ERROR] --elements: {e}", file=sys.stderr); sys.exit(1)

    pool_label = (
        f"--elements {args.elements!r} -> {len(element_pool)} elements"
        if args.elements else f"all Z=1-106 ({len(element_pool)} elements)"
    )
    print(f"[pool] {pool_label}", file=sys.stderr)

    # Shell center: None means "random per sample"; fixed sym if --center-z given
    fixed_center_sym: str | None = None
    if args.mode == "shell":
        try:
            fixed_center_sym = _validate_center_z(args.center_z, element_pool)
        except ValueError as e:
            print(f"[ERROR] {e}", file=sys.stderr); sys.exit(1)
        if fixed_center_sym is not None:
            print(f"[shell] center fixed: {fixed_center_sym} (Z={ATOMIC_NUMBERS[fixed_center_sym]})",
                  file=sys.stderr)
        else:
            print("[shell] center: random per sample (chaos mode)", file=sys.stderr)

    coord_lo, coord_hi = 4, 8
    if args.mode == "shell":
        try:
            coord_lo, coord_hi = parse_int_range(args.coord_range)
        except ValueError as e:
            print(f"[ERROR] --coord-range: {e}", file=sys.stderr); sys.exit(1)

    rng = random.Random(args.seed)

    # Validate mode
    if args.validate:
        trial = [rng.choice(element_pool) for _ in range(args.n_atoms)]
        ok, msg = validate_charge_mult(trial, args.charge, args.mult)
        print(f"[validate:{'OK' if ok else 'FAIL'}] {msg}", file=sys.stderr)
        sys.exit(0 if ok else 1)

    # Filters
    filters: list[tuple[str, float, float]] = []
    for f_str in args.filters:
        try:
            filters.append(parse_filter(f_str))
        except ValueError as e:
            print(f"[ERROR] --filter: {e}", file=sys.stderr); sys.exit(1)
    if filters:
        print("[filter] " + ",  ".join(f"{m} in [{lo:.4g},{hi:.4g}]" for m,lo,hi in filters),
              file=sys.stderr)

    # Generation loop
    xyz_blocks: list[str] = []
    n_passed = n_invalid = 0
    width = len(str(args.n_samples))

    for i in range(args.n_samples):
        atoms_list = [rng.choice(element_pool) for _ in range(args.n_atoms)]

        ok, val_msg = validate_charge_mult(atoms_list, args.charge, args.mult)
        if not ok:
            n_invalid += 1
            print(f"[{i+1:>{width}}/{args.n_samples}:invalid] {val_msg}", file=sys.stderr)
            continue

        try:
            if args.mode == "gas":
                atoms_out, positions = place_gas(
                    atoms_list, args.region, args.min_dist, args.max_attempts, rng)
            elif args.mode == "chain":
                atoms_out, positions = place_chain(
                    atoms_list, bond_lo, bond_hi,
                    args.branch_prob, args.min_dist, args.max_attempts, rng)
            else:
                # Center atom: fixed override OR random from this sample's composition
                center_sym = (
                    fixed_center_sym
                    if fixed_center_sym is not None
                    else rng.choice(atoms_list)
                )
                atoms_out, positions = place_shell(
                    atoms_list, center_sym,
                    coord_lo, coord_hi, shell_lo, shell_hi,
                    bond_lo, bond_hi, args.min_dist, args.max_attempts, rng)
        except RuntimeError as e:
            print(f"[ERROR] sample {i+1}: {e}", file=sys.stderr); sys.exit(1)

        metrics = compute_all_metrics(
            atoms_out, positions,
            args.n_bins, args.w_atom, args.w_spatial, args.cutoff)

        passed = passes_filters(metrics, filters)
        flag = "PASS" if passed else "skip"
        print(
            f"[{i+1:>{width}}/{args.n_samples}:{flag}]  "
            + "  ".join(f"{k}={_fmt(v)}" for k,v in metrics.items()),
            file=sys.stderr)

        if not passed:
            continue
        n_passed += 1
        prefix = f"sample={n_passed} mode={args.mode}"
        if args.mode == "shell":
            prefix += f" center={center_sym}(Z={ATOMIC_NUMBERS[center_sym]})"
        if args.seed is not None:
            prefix += f" seed={args.seed}"
        xyz_blocks.append(format_xyz(atoms_out, positions, args.charge, args.mult, metrics, prefix))

    n_skip = args.n_samples - n_passed - n_invalid
    print(f"[summary] attempted={args.n_samples}  passed={n_passed}  "
          f"filtered_out={n_skip}  invalid_charge_mult={n_invalid}", file=sys.stderr)
    if not xyz_blocks:
        print("[warning] No structures passed. Try relaxing --filter or increasing --n-samples.",
              file=sys.stderr)

    output_text = "\n".join(xyz_blocks) + ("\n" if xyz_blocks else "")
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(output_text)
        print(f"[info] {n_passed} structure(s) written to '{args.output}'", file=sys.stderr)
    else:
        sys.stdout.write(output_text)


if __name__ == "__main__":
    main()