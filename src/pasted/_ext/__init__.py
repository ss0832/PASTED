"""
pasted._ext
===========
Optional C++ acceleration for inner-loop hotspots.

Each sub-module is a separate pybind11 extension compiled from its own
``.cpp`` source.  Modules are imported independently so that a partial build
(e.g. only ``_relax_core`` compiled) degrades gracefully rather than
disabling all acceleration.

Public names
------------
HAS_RELAX  : bool  – True when _relax_core is available
HAS_MAXENT : bool  – True when _maxent_core is available

relax_positions(pts, radii, cov_scale, max_cycles, seed=-1)
    Available when HAS_RELAX is True.

angular_repulsion_gradient(pts, cutoff)
    Available when HAS_MAXENT is True.

Callers should check the relevant flag before calling and fall back to the
pure-Python implementations in _placement.py when False.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# _relax_core  — repulsion-relaxation inner loop
# ---------------------------------------------------------------------------
try:
    from ._relax_core import relax_positions  # type: ignore[import-untyped]
    HAS_RELAX: bool = True
except ImportError:
    relax_positions: Any = None  # type: ignore[no-redef]
    HAS_RELAX = False

# ---------------------------------------------------------------------------
# _maxent_core  — angular repulsion gradient (maxent placement only)
# ---------------------------------------------------------------------------
try:
    from ._maxent_core import angular_repulsion_gradient  # type: ignore[import-untyped]
    HAS_MAXENT: bool = True
except ImportError:
    angular_repulsion_gradient: Any = None  # type: ignore[no-redef]
    HAS_MAXENT = False

__all__ = [
    "HAS_RELAX",
    "HAS_MAXENT",
    "relax_positions",
    "angular_repulsion_gradient",
]
