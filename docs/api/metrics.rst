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
     - Population variance of |Δχ| (Pauling electronegativity difference)
       across all cutoff-adjacent atom pairs.  High = inconsistent
       bond-polarity landscape; 0 = all bonds equally polar (or fewer than
       two pairs detected).
     - ≥ 0
   * - ``moran_I_chi``
     - Moran's I spatial autocorrelation of electronegativity on the
       bond graph.  +1 = clustered; −1 = alternating; 0 = no pattern.
       Clamped to 1.0 from above: binary-weight graphs with fewer edges
       than atoms can produce raw values > 1 due to the n/W prefactor.
     - (-∞, 1]

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

      **Steinhardt optimisations (v0.3.6 + v0.3.7).**

      *v0.3.6 — accumulator buffer transpose.*  Layout changed from
      ``(n_l, l_max+1, N)`` to ``(N, n_l, l_max+1)`` (atom index outermost),
      making every bond's writes contiguous (stride 8 B) and eliminating the
      L2→L3 spill that caused superlinear wall-time growth at N ≈ 1 000.

      *v0.3.7 — per-bond arithmetic.*  ``atan2`` replaced by ``sqrt + div``
      (``cos_phi``/``sin_phi``); ``cos(m·phi)``/``sin(m·phi)`` via Chebyshev
      recurrence (2 mults + 1 sub each) instead of 18 libm calls per bond;
      P_lm table stack-allocated (``double[13][13]``) instead of heap per bond.

      Combined speedup: **~2.1–2.3×** on ``compute_steinhardt`` and
      **~1.3×** on ``compute_all_metrics`` at N = 500–1 000.
      See ``docs/architecture.md`` → *Per-bond arithmetic optimisations*.

   .. note::

      **Bug fix — ``moran_I_chi`` upper-bound clamp (v0.3.8).**

      Prior to v0.3.8, ``moran_I_chi`` could return values above +1.0 on
      structures whose cutoff graph was very sparse (fewer edges than atoms,
      i.e. ``W < N``).  The ``N / W`` prefactor in Moran's I formula inflated
      the result when un-normalised binary weights were used.  Both the C++
      path (``graph_metrics_cpp``) and the Python fallback
      (``compute_moran_I_chi``) now clamp the result to ``min(raw, 1.0)``
      before returning.  Results for connected graphs (``graph_lcc ≈ 1.0``)
      are unaffected.

   .. note::

      **Performance — real spherical harmonics fast-path for l=4,6,8 (v0.3.8, ④).**

      When ``l_values = [4, 6, 8]`` (the default), ``compute_steinhardt`` uses
      hardcoded Cartesian polynomial arithmetic instead of the
      associated-Legendre recurrence.  Every real spherical harmonic
      ``S_lm(x,y,z)`` is a pure integer-coefficient polynomial on the unit
      sphere; SymPy joint CSE across all three ``l`` values yields 84
      intermediates + 39 accumulation lines with no ``sqrt``, no ``atan2``, and
      no ``std::pow``.  Speedup: **1.4–1.6×** at N = 100–1 000 vs. the
      ①②③ generic path.  Other ``l`` combinations are unaffected.

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
