"""
pasted._io
==========
XYZ format serialisation helpers.
"""

from __future__ import annotations

import math
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
