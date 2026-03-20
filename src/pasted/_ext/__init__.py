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
HAS_RELAX       : bool  -- True when _relax_core is available
HAS_MAXENT      : bool  -- True when _maxent_core angular gradient is available
HAS_MAXENT_LOOP : bool  -- True when _maxent_core full C++ L-BFGS loop is available
                           (implies HAS_MAXENT; enables the fast place_maxent_cpp path)
HAS_STEINHARDT  : bool  -- True when _steinhardt_core is available
HAS_GRAPH       : bool  -- True when _graph_core is available
HAS_OPENMP      : bool  -- True when the C++ extensions were compiled with -fopenmp
                           (Linux only; False on all other platforms or when
                           PASTED_DISABLE_OPENMP=1 was set at build time)

relax_positions(pts, radii, cov_scale, max_cycles, seed=-1)
    Available when HAS_RELAX is True.

angular_repulsion_gradient(pts, cutoff)
    Available when HAS_MAXENT is True.

steinhardt_per_atom(pts, cutoff, l_values)
    Available when HAS_STEINHARDT is True.

graph_metrics_cpp(pts, radii, cov_scale, en_vals, cutoff)
    Available when HAS_GRAPH is True.
    Returns dict with graph_lcc, graph_cc, ring_fraction,
    charge_frustration, moran_I_chi -- all in one FlatCellList pass.

rdf_h_cpp(pts, cutoff, n_bins)
    Available when HAS_GRAPH is True.
    Returns dict with h_spatial and rdf_dev computed in O(N*k) via
    FlatCellList pair enumeration.  Replaces the O(N^2) pdist path.

Callers should check the relevant flag before calling and fall back to the
pure-Python implementations in _placement.py / _metrics.py when False.
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# _relax_core  -- repulsion-relaxation inner loop
# ---------------------------------------------------------------------------
try:
    from ._relax_core import relax_positions  # type: ignore[import-untyped]
    HAS_RELAX: bool = True
except ImportError:
    relax_positions: Any = None  # type: ignore[no-redef]
    HAS_RELAX = False

# ---------------------------------------------------------------------------
# _maxent_core  -- angular repulsion gradient + full loop (maxent only)
# ---------------------------------------------------------------------------
try:
    from ._maxent_core import (  # type: ignore[import-untyped]
        angular_repulsion_gradient,
        place_maxent_cpp,
    )
    HAS_MAXENT: bool = True
    HAS_MAXENT_LOOP: bool = True
except ImportError:
    angular_repulsion_gradient: Any = None  # type: ignore[no-redef]
    place_maxent_cpp: Any = None  # type: ignore[no-redef]
    HAS_MAXENT = False
    HAS_MAXENT_LOOP = False

# ---------------------------------------------------------------------------
# _steinhardt_core  -- sparse Steinhardt Q_l (all modes)
# ---------------------------------------------------------------------------
try:
    from ._steinhardt_core import steinhardt_per_atom  # type: ignore[import-untyped]
    HAS_STEINHARDT: bool = True
except ImportError:
    steinhardt_per_atom: Any = None  # type: ignore[no-redef]
    HAS_STEINHARDT = False

# ---------------------------------------------------------------------------
# _graph_core  -- O(N*k) graph / ring / charge / Moran / RDF metrics
# ---------------------------------------------------------------------------
try:
    from ._graph_core import (  # type: ignore[import-untyped]
        graph_metrics_cpp,
        moran_I_chi_cpp,
        rdf_h_cpp,
    )
    HAS_GRAPH: bool = True
except ImportError:
    graph_metrics_cpp: Any = None  # type: ignore[no-redef]
    moran_I_chi_cpp:   Any = None  # type: ignore[no-redef]
    rdf_h_cpp:         Any = None  # type: ignore[no-redef]
    HAS_GRAPH = False

# ---------------------------------------------------------------------------
# HAS_OPENMP  -- runtime detection of OpenMP support
# ---------------------------------------------------------------------------
# True only on Linux when the extensions were compiled with -fopenmp AND
# the omp runtime reports more than one thread available.
# PASTED_DISABLE_OPENMP=1 also suppresses it at import time.

def _detect_openmp() -> bool:
    if sys.platform != "linux":
        return False
    if os.environ.get("PASTED_DISABLE_OPENMP", "0") == "1":
        return False
    if not HAS_RELAX:
        return False
    try:
        libomp = ctypes.CDLL("libgomp.so.1", use_errno=True)
        return bool(libomp.omp_get_max_threads() > 1)
    except Exception:
        pass
    return False


HAS_OPENMP: bool = _detect_openmp()


def set_num_threads(n: int) -> None:
    """Set the number of OpenMP threads used by all C++ extensions.

    This is a no-op when :data:`HAS_OPENMP` is ``False``.

    Parameters
    ----------
    n:
        Number of threads.  ``0`` or negative values are silently ignored.
        Equivalent to setting the ``OMP_NUM_THREADS`` environment variable
        before import, but takes effect immediately at runtime.

    Notes
    -----
    The change is global and affects all subsequent C++ extension calls in
    the current process.  It does **not** override ``OMP_NUM_THREADS`` for
    child processes.

    Supported on Linux only (where OpenMP is available).
    """
    if not HAS_OPENMP or n <= 0:
        return
    try:
        libomp = ctypes.CDLL("libgomp.so.1", use_errno=True)
        libomp.omp_set_num_threads(ctypes.c_int(n))
    except Exception:
        pass


__all__ = [
    "HAS_RELAX",
    "HAS_MAXENT",
    "HAS_MAXENT_LOOP",
    "HAS_STEINHARDT",
    "HAS_GRAPH",
    "HAS_OPENMP",
    "relax_positions",
    "angular_repulsion_gradient",
    "place_maxent_cpp",
    "steinhardt_per_atom",
    "graph_metrics_cpp",
    "moran_I_chi_cpp",
    "rdf_h_cpp",
    "set_num_threads",
]
