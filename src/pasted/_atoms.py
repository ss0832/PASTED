"""
pasted._atoms
=============
Atomic data (Z → symbol mapping, Pyykkö covalent radii) and all
input-parsing / validation helpers that do not depend on numpy.
"""

from __future__ import annotations

import math
from collections import Counter

# ---------------------------------------------------------------------------
# Atomic data  Z = 1–106
# ---------------------------------------------------------------------------

ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
}

_Z_TO_SYM: dict[int, str] = {v: k for k, v in ATOMIC_NUMBERS.items()}
_ALL_Z: list[int] = sorted(_Z_TO_SYM)

ALL_METRICS: frozenset[str] = frozenset(
    {
        "H_atom",
        "H_spatial",
        "H_total",
        "RDF_dev",
        "shape_aniso",
        "Q4",
        "Q6",
        "Q8",
        "graph_lcc",
        "graph_cc",
    }
)

# ---------------------------------------------------------------------------
# Pyykkö single-bond covalent radii (Å), Z = 1–86
# Reference: Pyykkö & Atsumi, Chem. Eur. J. 15 (2009) 186–197
# ---------------------------------------------------------------------------

_COV_RADII_ANG: dict[str, float] = {
    "H": 0.32,
    "He": 0.46,
    "Li": 1.33,
    "Be": 1.02,
    "B": 0.85,
    "C": 0.75,
    "N": 0.71,
    "O": 0.63,
    "F": 0.64,
    "Ne": 0.67,
    "Na": 1.55,
    "Mg": 1.39,
    "Al": 1.26,
    "Si": 1.16,
    "P": 1.11,
    "S": 1.03,
    "Cl": 0.99,
    "Ar": 0.96,
    "K": 1.96,
    "Ca": 1.71,
    "Sc": 1.48,
    "Ti": 1.36,
    "V": 1.34,
    "Cr": 1.22,
    "Mn": 1.19,
    "Fe": 1.16,
    "Co": 1.11,
    "Ni": 1.10,
    "Cu": 1.12,
    "Zn": 1.18,
    "Ga": 1.24,
    "Ge": 1.24,
    "As": 1.21,
    "Se": 1.16,
    "Br": 1.14,
    "Kr": 1.17,
    "Rb": 2.10,
    "Sr": 1.85,
    "Y": 1.63,
    "Zr": 1.54,
    "Nb": 1.47,
    "Mo": 1.38,
    "Tc": 1.28,
    "Ru": 1.25,
    "Rh": 1.25,
    "Pd": 1.20,
    "Ag": 1.28,
    "Cd": 1.36,
    "In": 1.42,
    "Sn": 1.40,
    "Sb": 1.40,
    "Te": 1.36,
    "I": 1.33,
    "Xe": 1.31,
    "Cs": 2.32,
    "Ba": 1.96,
    "La": 1.80,
    "Ce": 1.63,
    "Pr": 1.76,
    "Nd": 1.74,
    "Pm": 1.73,
    "Sm": 1.72,
    "Eu": 1.68,
    "Gd": 1.69,
    "Tb": 1.68,
    "Dy": 1.67,
    "Ho": 1.66,
    "Er": 1.65,
    "Tm": 1.64,
    "Yb": 1.70,
    "Lu": 1.62,
    "Hf": 1.52,
    "Ta": 1.46,
    "W": 1.37,
    "Re": 1.31,
    "Os": 1.29,
    "Ir": 1.22,
    "Pt": 1.23,
    "Au": 1.24,
    "Hg": 1.33,
    "Tl": 1.44,
    "Pb": 1.44,
    "Bi": 1.51,
    "Po": 1.45,
    "At": 1.47,
    "Rn": 1.42,
}

# Z > 86: no literature single-bond radii available.
# Proxy: same-group nearest lighter element.
_COV_RADII_PROXY: dict[str, str] = {
    "Fr": "Cs",  # group  1
    "Ra": "Ba",  # group  2
    "Ac": "La",  # group  3
    # Actinides (Th–Lr) → corresponding lanthanides (Ce–Lu)
    "Th": "Ce",
    "Pa": "Pr",
    "U": "Nd",
    "Np": "Pm",
    "Pu": "Sm",
    "Am": "Eu",
    "Cm": "Gd",
    "Bk": "Tb",
    "Cf": "Dy",
    "Es": "Ho",
    "Fm": "Er",
    "Md": "Tm",
    "No": "Yb",
    "Lr": "Lu",
    # Period-7 d-block → Period-6 d-block (same group)
    "Rf": "Hf",
    "Db": "Ta",
    "Sg": "W",
}


def cov_radius_ang(sym: str) -> float:
    """Return the Pyykkö single-bond covalent radius in Å for *sym*.

    For Z > 86 the same-group nearest lighter element is used as a proxy
    (e.g. Fr → Cs, U → Nd, Rf → Hf).
    """
    r = _COV_RADII_ANG.get(sym)
    if r is not None:
        return r
    proxy = _COV_RADII_PROXY.get(sym)
    if proxy is not None:
        return _COV_RADII_ANG[proxy]
    return 1.50  # ultimate fallback (should never be reached for Z ≤ 106)


# Keep the private alias used in other modules that were not yet updated.
_cov_radius_ang = cov_radius_ang

# ---------------------------------------------------------------------------
# Element-pool helpers
# ---------------------------------------------------------------------------


def parse_element_spec(spec: str) -> list[str]:
    """Parse an atomic-number spec string into a sorted list of element symbols.

    Syntax
    ------
    ``"1-30"``       Z = 1 through 30
    ``"6,7,8"``      Z = 6, 7, 8
    ``"1-10,26,28"`` Z = 1–10 plus Z = 26 and 28

    Raises
    ------
    ValueError
        On malformed input or unsupported Z values.
    """
    z_set: set[int] = set()
    for token in spec.split(","):
        token = token.strip()  # noqa: PLW2901
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
    symbols: list[str] = []
    for z in sorted(z_set):
        if z not in _Z_TO_SYM:
            raise ValueError(f"Z={z} not supported (supported range: {_ALL_Z[0]}–{_ALL_Z[-1]})")
        symbols.append(_Z_TO_SYM[z])
    if not symbols:
        raise ValueError("Element specification resolved to an empty pool.")
    return symbols


def default_element_pool() -> list[str]:
    """Return all supported element symbols (Z = 1–106), sorted by Z."""
    return [_Z_TO_SYM[z] for z in _ALL_Z]


# ---------------------------------------------------------------------------
# Range / filter parsers
# ---------------------------------------------------------------------------


def parse_lo_hi(s: str, name: str = "range") -> tuple[float, float]:
    """Parse ``"LO:HI"`` → ``(float, float)``."""
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"--{name} must be 'LO:HI', got {s!r}")
    return float(parts[0]), float(parts[1])


def parse_int_range(s: str) -> tuple[int, int]:
    """Parse ``"MIN:MAX"`` → ``(int, int)`` with MIN ≥ 1 and MIN ≤ MAX."""
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Must be 'MIN:MAX', got {s!r}")
    lo, hi = int(parts[0]), int(parts[1])
    if lo < 1 or lo > hi:
        raise ValueError(f"MIN must be ≥ 1 and ≤ MAX, got {s!r}")
    return lo, hi


def parse_filter(f: str) -> tuple[str, float, float]:
    """Parse ``"METRIC:MIN:MAX"`` → ``(metric, lo, hi)``.

    Use ``"-"`` for an open bound.

    Raises
    ------
    ValueError
        On unknown metric or malformed string.
    """
    parts = f.split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected 'METRIC:MIN:MAX', got {f!r}")
    metric, lo_s, hi_s = parts
    if metric not in ALL_METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Valid metrics: {sorted(ALL_METRICS)}")
    lo = -math.inf if lo_s.strip() == "-" else float(lo_s)
    hi = math.inf if hi_s.strip() == "-" else float(hi_s)
    if lo > hi:
        raise ValueError(f"Filter {f!r}: MIN > MAX.")
    return metric, lo, hi


# ---------------------------------------------------------------------------
# Charge / multiplicity validation
# ---------------------------------------------------------------------------


def validate_charge_mult(atoms_list: list[str], charge: int, mult: int) -> tuple[bool, str]:
    """Check electron count and spin-parity for *atoms_list*.

    Returns
    -------
    (ok, message)
        *ok* is ``True`` when both conditions pass.
    """
    total_z = sum(ATOMIC_NUMBERS[a] for a in atoms_list)
    n_e = total_z - charge
    if n_e <= 0:
        return False, (f"n_electrons={n_e} (total_Z={total_z}, charge={charge:+d}).")
    n_up = mult - 1
    if (n_e % 2) != (n_up % 2):
        return False, (
            f"parity mismatch: n_electrons={n_e} (charge={charge:+d}), "
            f"mult={mult} → n_unpaired={n_up}."
        )
    comp = " ".join(f"{s}:{c}" for s, c in sorted(Counter(atoms_list).items()))
    return True, (
        f"n_atoms={len(atoms_list)} total_Z={total_z} "
        f"n_electrons={n_e} charge={charge:+d} mult={mult} comp=[{comp}]"
    )
