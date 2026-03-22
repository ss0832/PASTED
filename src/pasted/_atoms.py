"""
pasted._atoms
=============
Atomic data (Z → symbol mapping, Pyykkö covalent radii) and all
input-parsing / validation helpers that do not depend on numpy.
"""

from __future__ import annotations

import math

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
        # MM-level structural descriptors (added in 0.1.9)
        "ring_fraction",
        "charge_frustration",
        "moran_I_chi",
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
# Pauling electronegativity  (Allen scale not used — Pauling per user choice)
# Reference: Pauling, L. The Nature of the Chemical Bond, 3rd ed. (1960).
#            IUPAC 2016 recommended values used for updates.
# Noble gases and most actinides lack Pauling values; fallback = 1.0.
# ---------------------------------------------------------------------------

_PAULING_EN: dict[str, float] = {
    "H": 2.20,
    "Li": 0.98,
    "Be": 1.57,
    "B": 2.04,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
    "F": 3.98,
    "Na": 0.93,
    "Mg": 1.31,
    "Al": 1.61,
    "Si": 1.90,
    "P": 2.19,
    "S": 2.58,
    "Cl": 3.16,
    "K": 0.82,
    "Ca": 1.00,
    "Sc": 1.36,
    "Ti": 1.54,
    "V": 1.63,
    "Cr": 1.66,
    "Mn": 1.55,
    "Fe": 1.83,
    "Co": 1.88,
    "Ni": 1.91,
    "Cu": 1.90,
    "Zn": 1.65,
    "Ga": 1.81,
    "Ge": 2.01,
    "As": 2.18,
    "Se": 2.55,
    "Br": 2.96,
    "Rb": 0.82,
    "Sr": 0.95,
    "Y": 1.22,
    "Zr": 1.33,
    "Nb": 1.60,
    "Mo": 2.16,
    "Tc": 1.90,
    "Ru": 2.20,
    "Rh": 2.28,
    "Pd": 2.20,
    "Ag": 1.93,
    "Cd": 1.69,
    "In": 1.78,
    "Sn": 1.96,
    "Sb": 2.05,
    "Te": 2.10,
    "I": 2.66,
    "Cs": 0.79,
    "Ba": 0.89,
    "La": 1.10,
    "Ce": 1.12,
    "Pr": 1.13,
    "Nd": 1.14,
    "Pm": 1.13,
    "Sm": 1.17,
    "Eu": 1.20,
    "Gd": 1.20,
    "Tb": 1.10,
    "Dy": 1.22,
    "Ho": 1.23,
    "Er": 1.24,
    "Tm": 1.25,
    "Yb": 1.10,
    "Lu": 1.27,
    "Hf": 1.30,
    "Ta": 1.50,
    "W": 2.36,
    "Re": 1.90,
    "Os": 2.20,
    "Ir": 2.20,
    "Pt": 2.28,
    "Au": 2.54,
    "Hg": 2.00,
    "Tl": 1.62,
    "Pb": 2.33,
    "Bi": 2.02,
    "Po": 2.00,
    "At": 2.20,
    # Z > 86: no reliable Pauling values — use group-based proxies
    # Fr, Ra: alkali/alkaline-earth → Cs, Ba values
    "Fr": 0.70,
    "Ra": 0.90,
    # Ac-series: approximate from lanthanide analogues
    "Ac": 1.10,
    "Th": 1.30,
    "Pa": 1.50,
    "U": 1.38,
    "Np": 1.36,
    "Pu": 1.28,
    "Am": 1.13,
    "Cm": 1.28,
    "Bk": 1.30,
    "Cf": 1.30,
    "Es": 1.30,
    "Fm": 1.30,
    "Md": 1.30,
    "No": 1.30,
    "Lr": 1.30,
    # Period-7 d-block: no published values — use Period-6 analogue
    "Rf": 1.30,
    "Db": 1.50,
    "Sg": 2.36,
    # Noble gases: He, Ne, Ar — no experimental Pauling value;
    # assigned 4.0 (maximum) to model complete resistance to electron donation.
    # Kr and Xe can form compounds (e.g. XeF2, KrF2) and have literature
    # estimates on the Allen/Allred-Rochow scale: Kr ≈ 3.0, Xe ≈ 2.6.
    # Rn: no reliable data; conservatively set to 4.0.
    "He": 4.0,
    "Ne": 4.0,
    "Ar": 4.0,
    "Kr": 3.0,
    "Xe": 2.6,
    "Rn": 4.0,
}

#: Fallback Pauling electronegativity for elements without a literature value
#: (any symbol not in the table; Kr/Xe/other noble gases have explicit entries).
PAULING_EN_FALLBACK: float = 1.0


def pauling_electronegativity(sym: str) -> float:
    """Return the Pauling electronegativity for element *sym*.

    Values follow Pauling (1960) with IUPAC 2016 updates.  Noble gases
    with no known compounds (He, Ne, Ar, Rn) are assigned 4.0 to model
    complete resistance to electron donation.  Kr (≈ 3.0) and Xe (≈ 2.6)
    use literature estimates from the Allen / Allred-Rochow scale, reflecting
    their known tendency to form compounds (KrF₂, XeF₂, etc.).
    Any symbol not in the table returns :data:`PAULING_EN_FALLBACK` (1.0).

    Parameters
    ----------
    sym:
        Element symbol (case-sensitive, e.g. ``"Fe"``).

    Returns
    -------
    float
        Pauling electronegativity.  Noble gases return 4.0;
        any other element without a value returns 1.0.
    """
    return _PAULING_EN.get(sym, PAULING_EN_FALLBACK)


# ---------------------------------------------------------------------------
# Element-pool helpers
# ---------------------------------------------------------------------------


def parse_element_spec(spec: str | list[str]) -> list[str]:
    """Parse an element specification into a sorted list of element symbols.

    Three input forms are accepted:

    **Atomic-number string** (most common)
        ``"1-30"``       — Z = 1 through 30
        ``"6,7,8"``      — Z = 6, 7, 8  (C, N, O)
        ``"1-10,26,28"`` — Z = 1–10 plus Z = 26 and 28

    **Symbol list**
        ``["C", "N", "O"]`` — explicit element-symbol list.  Symbols must
        be present in the built-in atomic-number table (Z = 1–106).

    Parameters
    ----------
    spec:
        An atomic-number spec string **or** a list of element symbol strings.

    Returns
    -------
    list[str]
        Element symbols sorted by atomic number (ascending Z).

    Raises
    ------
    ValueError
        On malformed input, unknown element symbols, or unsupported Z values.
    """
    # ------------------------------------------------------------------ #
    # Branch A — explicit symbol list, e.g. ["C", "N", "O"]              #
    # ------------------------------------------------------------------ #
    if isinstance(spec, list):
        _sym_to_z: dict[str, int] = ATOMIC_NUMBERS
        z_set: set[int] = set()
        for sym in spec:
            if not isinstance(sym, str):
                raise ValueError(
                    f"Symbol list entries must be strings, got {type(sym).__name__!r}: {sym!r}"
                )
            if sym not in _sym_to_z:
                raise ValueError(
                    f"Unknown element symbol {sym!r}. "
                    'Use atomic-number notation (e.g. "6,7,8") or a valid symbol list.'
                )
            z_set.add(_sym_to_z[sym])
        if not z_set:
            raise ValueError("Element specification resolved to an empty pool.")
        return [_Z_TO_SYM[z] for z in sorted(z_set)]

    # ------------------------------------------------------------------ #
    # Branch B — atomic-number spec string, e.g. "6,7,8" or "1-30"      #
    # ------------------------------------------------------------------ #
    z_set_str: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"Range {token!r}: lower > upper.")
            z_set_str.update(range(lo, hi + 1))
        else:
            z_set_str.add(int(token))
    symbols: list[str] = []
    for z in sorted(z_set_str):
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
        *ok* is ``True`` when both conditions pass.  On the success path the
        message is an empty string ``""``; callers in the hot optimizer loop
        discard it (``ok, _ = validate_charge_mult(...)``), so building the
        full diagnostic string is unnecessary work.  On failure the message
        contains the full diagnostic as before.
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
    # Success: skip Counter construction and f-string formatting — the message
    # is discarded by every hot-loop caller.  (~2.2× faster on the ok=True path)
    return True, ""
