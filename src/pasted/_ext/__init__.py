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

relax_positions(pts, radii, cov_scale, max_cycles, seed=-1)
    Available when HAS_RELAX is True.

angular_repulsion_gradient(pts, cutoff)
    Available when HAS_MAXENT is True.

steinhardt_per_atom(pts, cutoff, l_values)
    Available when HAS_STEINHARDT is True.

Memory notes (v0.2.1)
---------------------
``_relax_core`` and ``_maxent_core`` were refactored in v0.2.1 to eliminate
repeated heap allocation inside their hot L-BFGS loops.  Gradient scratch
buffers and neighbor lists are now held as persistent members / outer-scope
variables and reused across iterations.

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

Changes in v0.2.3
-----------------
Removed ``HAS_OPENMP`` and ``set_num_threads``.  Benchmarking showed that the
OpenMP thread-pool overhead in ``compute_all_metrics`` outweighed the
parallelism benefit for all practically relevant structure sizes (n < 30 000),
causing a 1.4--2.5x regression versus v0.1.17.  All C++ extensions now run
single-threaded; the ``libgomp`` dependency is dropped.

Changes in v0.2.10
------------------
Removed ``HAS_POISSON``, ``_poisson_disk_sphere_cpp``, and
``_poisson_disk_box_cpp``.  The Python-only Poisson-disk helpers
(``_poisson_disk_sphere``, ``_poisson_disk_box``) were similarly removed from
``_placement.py``.  These functions were never called by any internal path
and were not part of the stable public API.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# _relax_core  -- repulsion-relaxation inner loop
# ---------------------------------------------------------------------------
try:
    from ._relax_core import (  # type: ignore[import-untyped]
        relax_positions,
    )
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

__all__ = [
    "HAS_RELAX",
    "HAS_MAXENT",
    "HAS_MAXENT_LOOP",
    "HAS_STEINHARDT",
    "HAS_GRAPH",
    "relax_positions",
    "angular_repulsion_gradient",
    "place_maxent_cpp",
    "steinhardt_per_atom",
    "graph_metrics_cpp",
    "moran_I_chi_cpp",
    "rdf_h_cpp",
]
