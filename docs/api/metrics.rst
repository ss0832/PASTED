Metric functions
================

.. automodule:: pasted._metrics
   :members:
   :undoc-members: False

MM-level structural descriptors
---------------------------------

The following two functions were added in v0.1.9 and revised in v0.1.13.
They use the same *cutoff* distance threshold as ``graph_lcc``,
``graph_cc``, and ``moran_I_chi``, so all five cutoff-based metrics share
a single unified adjacency definition.

All metrics are included in :data:`~pasted._atoms.ALL_METRICS` and can
therefore be used as ``--filter`` targets on the CLI and in the
:class:`~pasted._generator.StructureGenerator` ``filters=`` parameter.

.. autofunction:: pasted._metrics.compute_ring_fraction
.. autofunction:: pasted._metrics.compute_charge_frustration
.. autofunction:: pasted._metrics.compute_moran_I_chi

.. rubric:: Metric overview

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Key
     - Description
     - Range
   * - ``H_atom``
     - Shannon entropy of the element composition histogram.
     - 0 – ln(*k*) for *k* distinct elements
   * - ``H_spatial``
     - Shannon entropy of the radial-distribution histogram within *cutoff*.
     - 0 – ln(*n_bins*)
   * - ``H_total``
     - ``w_atom · H_atom + w_spatial · H_spatial``
     - 0 – depends on weights
   * - ``RDF_dev``
     - RMS deviation of the normalized RDF from a flat distribution.
       High = crystalline; low = amorphous.
     - ≥ 0
   * - ``shape_aniso``
     - Relative shape anisotropy κ² from the gyration tensor
       (κ² = 1.5·Σλᵢ²/(Σλᵢ)² − 0.5).
       0 = perfectly spherical; 1 = perfectly rod-like.
     - [0, 1]
   * - ``Q4``, ``Q6``, ``Q8``
     - Steinhardt bond-orientational order parameters (*l* = 4, 6, 8).
       0 = amorphous; 1 = crystalline.
     - [0, 1]
   * - ``graph_lcc``
     - Fraction of atoms in the largest connected component of the
       covalent-bond graph.  1.0 = fully connected.
     - [0, 1]
   * - ``graph_cc``
     - Mean local clustering coefficient of the covalent-bond graph.
       0 = no triangles among neighbors; 1 = all neighbor pairs are
       mutually bonded.
     - [0, 1]
   * - ``ring_fraction``
     - Fraction of atoms that participate in at least one ring, determined
       by Tarjan's iterative bridge-finding algorithm (O(N + E)).  A bond is
       a *bridge* if its removal disconnects the graph; an atom is counted
       as a ring member when at least one of its incident bonds is a
       non-bridge.
     - [0, 1]
   * - ``charge_frustration``
     - Mean EN variance across bonded atom pairs (Pauling scale).
     - ≥ 0
   * - ``moran_I_chi``
     - Moran's spatial autocorrelation of electronegativity on the
       bond graph.  +1 = clustered; −1 = alternating; 0 = no pattern.
     - ≈ [−1, 1]

.. note::

   **C++ acceleration flags.**

   :data:`~pasted._ext.HAS_GRAPH` enables O(N·k) pair enumeration for
   ``graph_lcc``, ``graph_cc``, ``ring_fraction``, ``charge_frustration``,
   ``moran_I_chi``, ``H_spatial``, and ``RDF_dev``.  The C++ implementation
   uses a single shared adjacency list (no duplicate allocations), sorted
   adjacency for O(log k) triangle lookup in ``graph_cc``, and streaming
   histogram construction in ``rdf_h_cpp`` (no intermediate distance vector).

   :data:`~pasted._ext.HAS_STEINHARDT` enables sparse per-atom Steinhardt
   computation for ``Q4``, ``Q6``, and ``Q8`` (~2000× vs. the dense Python
   fallback).

   .. note::

      **Superlinear wall-time scaling of** ``compute_steinhardt`` **(known
      limitation).**  Although the algorithm is O(N·k·l²), measured latency
      grows faster than linear from N ≈ 1 000.  The cause is a CPU cache
      pressure effect in the C++ accumulator buffer (layout
      ``(n_l, l_max+1, N)`` — atom index innermost), whose strides of
      N × 8 bytes exceed the L2 cache at N ≈ 1 000 and cause ~5–10×
      higher write latency.  ``compute_steinhardt`` is therefore the
      dominant cost in ``compute_all_metrics`` for N ≳ 200.
      See ``docs/architecture.md`` → *Superlinear wall-time scaling* for the
      full analysis and the planned buffer-transpose fix.

   .. warning::

      When ``HAS_GRAPH = False``, the five graph/ring/charge/Moran metrics
      fall back to a pure-Python path that builds a full **N×N distance
      matrix** (O(N²) memory and time).  This is **~100× slower** than the
      C++ path at N=500 (~100 ms vs. ~1 ms) and is intended only for
      environments where the C++ extension cannot be compiled.  Reinstall
      with a C++17 compiler (``pip install pybind11 && pip install -e .``)
      to enable ``HAS_GRAPH = True``.

.. note::

   **Distance cutoff.**

   All local metrics (``H_spatial``, ``RDF_dev``, ``graph_*``, ``Q*``,
   ``ring_fraction``, ``charge_frustration``, ``moran_I_chi``) share a
   single *cutoff* threshold.  When ``cutoff=None`` (the default) it is
   auto-computed as ``1.5 × median(rᵢ + rⱼ)`` over covalent radii.

   For reproducible comparisons — especially with reference data — always
   pass ``cutoff=`` explicitly to
   :class:`~pasted._generator.StructureGenerator`,
   :class:`~pasted._optimizer.StructureOptimizer`, or
   :func:`~pasted._metrics.compute_all_metrics`.

   When calling :func:`~pasted._metrics.compute_all_metrics` directly,
   ``n_bins``, ``w_atom``, ``w_spatial``, and ``cutoff`` all have
   sensible defaults (``20``, ``0.5``, ``0.5``, ``None``), so the
   minimal explicit-cutoff form is::

       from pasted._metrics import compute_all_metrics
       metrics = compute_all_metrics(atoms, positions, cutoff=4.5)
