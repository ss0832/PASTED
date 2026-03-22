"""
pasted._io
==========
XYZ format serialization and deserialization helpers.

Public API
----------
format_xyz(atoms, positions, charge, mult, metrics, prefix="") → str
    Serialize a structure to an extended-XYZ string.  The second line
    (the XYZ comment line) includes charge, multiplicity, and all metric
    values as ``key=value`` pairs.

parse_xyz(text) → list[dict]
    Parse one or more XYZ frames from *text*.  Each frame is returned as
    a dict with keys ``atoms``, ``positions``, ``charge``, ``mult``,
    ``metrics``, and ``prefix``.  Blank lines between frames are silently
    skipped.

Notes
-----
*  The extended-XYZ comment line written by :func:`format_xyz` is
   machine-readable: all fields use ``=`` as a separator with no spaces,
   allowing downstream tools to extract metrics without regex.
*  :func:`parse_xyz` accepts files produced by any tool that writes
   standard XYZ (atom-count line, then free-form comment line, then N
   coordinate lines).  Unrecognized comment-line content is stored as
   ``prefix`` and does not raise an error.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._placement import Vec3


def _fmt(v: float) -> str:
    """Format a metric value: ``nan`` as the string ``'nan'``, else 4 d.p."""
    return "nan" if math.isnan(v) else f"{v:.4f}"


def format_xyz(
    atoms: list[str],
    positions: list[Vec3],
    charge: int,
    mult: int,
    metrics: dict[str, float],
    prefix: str = "",
) -> str:
    """Serialise a structure to the extended XYZ format.

    The second line (comment line) encodes *prefix*, charge, multiplicity,
    composition, and all metric values.

    Parameters
    ----------
    atoms:
        Element symbols.
    positions:
        Cartesian coordinates (Å), one per atom.
    charge:
        Total system charge.
    mult:
        Spin multiplicity 2S+1.
    metrics:
        Dict of computed disorder metrics.
    prefix:
        Prepended to the comment line (e.g. ``"sample=1 mode=gas"``).

    Returns
    -------
    A multi-line string (no trailing newline).
    """
    comp = ",".join(f"{s}:{c}" for s, c in sorted(Counter(atoms).items()))
    metric_str = "  ".join(f"{k}={_fmt(v)}" for k, v in metrics.items())
    comment = (f"{prefix} charge={charge:+d} mult={mult} comp=[{comp}]  {metric_str}").strip()
    lines = [str(len(atoms)), comment]
    for atom, (x, y, z) in zip(atoms, positions, strict=False):
        lines.append(f"{atom:<4s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
    return "\n".join(lines)


def parse_xyz(text: str) -> list[tuple[list[str], list[Vec3], int, int, dict[str, float]]]:
    """Parse a (possibly multi-frame) XYZ string — standard or extended format.

    Supports both:

    * **Standard XYZ** — atom count line, comment line, then coordinate lines.
      ``charge`` defaults to 0, ``mult`` to 1, ``metrics`` is empty.
    * **Extended XYZ** (as written by PASTED) — the comment line may contain
      ``charge=+0``, ``mult=1``, and ``KEY=VALUE`` metric tokens.

    Parameters
    ----------
    text:
        Full contents of one or more XYZ frames (concatenated).

    Returns
    -------
    list of ``(atoms, positions, charge, mult, metrics)`` tuples, one per frame.

    Raises
    ------
    ValueError
        When the atom-count line or a coordinate line cannot be parsed.
    """
    frames = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        # Skip blank lines between frames
        if not lines[i].strip():
            i += 1
            continue

        # --- atom count line ---
        try:
            n_atoms = int(lines[i].strip())
        except ValueError as exc:
            raise ValueError(f"Expected atom count on line {i + 1}, got {lines[i]!r}") from exc
        i += 1

        if i >= len(lines):
            raise ValueError("Unexpected end of file after atom count line.")

        # --- comment line (extended XYZ fields) ---
        comment = lines[i]
        i += 1

        charge = 0
        mult = 1
        metrics: dict[str, float] = {}

        m_charge = re.search(r"\bcharge=([+-]?\d+)", comment)
        if m_charge:
            charge = int(m_charge.group(1))
        m_mult = re.search(r"\bmult=(\d+)", comment)
        if m_mult:
            mult = int(m_mult.group(1))

        # Parse KEY=FLOAT tokens for metrics (skip charge/mult already captured)
        pat = r"\b([A-Za-z_][A-Za-z0-9_]*)=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
        for tok in re.findall(pat, comment):
            key, val_str = tok
            if key in ("charge", "mult"):
                continue
            try:
                metrics[key] = float(val_str)
            except ValueError:
                pass

        # --- coordinate lines ---
        atoms: list[str] = []
        positions: list[tuple[float, float, float]] = []
        for _ in range(n_atoms):
            # Skip blank lines inside the coordinate block (some tools emit them)
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines):
                raise ValueError(f"Unexpected end of file: expected {n_atoms} coordinate lines.")
            parts = lines[i].split()
            i += 1
            if len(parts) < 4:
                raise ValueError(f"Malformed coordinate line: {lines[i - 1]!r}")
            atoms.append(parts[0])
            try:
                positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError as exc:
                raise ValueError(f"Non-numeric coordinate in: {lines[i - 1]!r}") from exc

        frames.append((atoms, positions, charge, mult, metrics))

    return frames
