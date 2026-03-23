Atom utilities
==============

.. automodule:: pasted._atoms
   :members:
   :undoc-members: False

Electronegativity data
----------------------

.. autofunction:: pasted._atoms.pauling_electronegativity
.. autodata:: pasted._atoms.PAULING_EN_FALLBACK

.. rubric:: ALL_METRICS

.. autodata:: pasted._atoms.ALL_METRICS

``ALL_METRICS`` is a ``frozenset[str]`` containing all 17 metric names accepted
by ``filters=``, ``--filter``, and the ``objective=`` dict.  They are also the
keys present in :attr:`Structure.metrics <pasted._generator.Structure.metrics>`
and :attr:`EvalContext.metrics <pasted._optimizer.EvalContext.metrics>`.
Because ``frozenset`` is unordered, use ``sorted(ALL_METRICS)`` when a
deterministic listing is required.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - What it measures
   * - ``H_atom``
     - Shannon entropy of the element composition (0 = pure, ln *k* = *k* equal elements)
   * - ``H_spatial``
     - Shannon entropy of the pairwise-distance histogram within *cutoff*
   * - ``H_total``
     - ``w_atom · H_atom + w_spatial · H_spatial``
   * - ``RDF_dev``
     - RMS deviation of the normalized RDF from a flat distribution
   * - ``shape_aniso``
     - Relative shape anisotropy κ² from the gyration tensor: 0 = spherical, 1 = rod-like ([0, 1])
   * - ``Q4``
     - Steinhardt bond-orientational order parameter *l* = 4
   * - ``Q6``
     - Steinhardt bond-orientational order parameter *l* = 6
   * - ``Q8``
     - Steinhardt bond-orientational order parameter *l* = 8
   * - ``graph_lcc``
     - Fraction of atoms in the largest connected component of the covalent-bond graph
   * - ``graph_cc``
     - Mean local clustering coefficient of the covalent-bond graph: 0 = no triangles, 1 = fully triangulated ([0, 1])
   * - ``ring_fraction``
     - Fraction of atoms participating in at least one ring
   * - ``charge_frustration``
     - Population variance of |Δχ| (Pauling electronegativity difference) across cutoff-adjacent atom pairs
   * - ``moran_I_chi``
     - Moran's spatial autocorrelation of electronegativity on the bond graph
   * - ``bond_angle_entropy``
     - Mean per-atom Shannon entropy of the bond-angle distribution (added in v0.4.0)
   * - ``coordination_variance``
     - Population variance of per-atom coordination numbers (added in v0.4.0)
   * - ``radial_variance``
     - Mean per-atom variance of neighbor distances in Å² (added in v0.4.0)
   * - ``local_anisotropy``
     - Mean per-atom local covariance-tensor anisotropy [0, 1] (added in v0.4.0)

.. rubric:: Element pool specification

.. autofunction:: pasted._atoms.parse_element_spec

.. note::

   When passing ``elements=`` as a *string*, it must contain **atomic numbers
   (integers)**, not element symbols.  ``elements="C,N,O"`` raises
   ``ValueError``; use ``elements="6,7,8"`` or ``elements=["C","N","O"]``
   instead.

   :func:`parse_element_spec` accepts **two** input forms:

   .. list-table::
      :header-rows: 1
      :widths: 30 30 40

      * - Form
        - Example
        - Meaning
      * - Comma-separated atomic numbers
        - ``"6,7,8"``
        - C, N, O
      * - Atomic-number range
        - ``"1-30"``
        - H through Zn
      * - Mixed ranges and singles
        - ``"1-10,26,28"``
        - H–Ne plus Fe and Ni
      * - Symbol list *(new in v0.3.5)*
        - ``["C", "N", "O"]``
        - Explicit list of element symbols

   .. versionchanged:: 0.3.5

      ``parse_element_spec`` now accepts a ``list[str]`` of element symbols
      in addition to atomic-number strings.  Previously, passing a list
      raised :exc:`AttributeError` because the function tried to call
      ``.split(",")`` on the list object.  The fix aligns the public API
      with the behavior already documented in :doc:`/quickstart`.

   .. versionchanged:: 0.3.5

      :meth:`Structure.from_xyz` now raises :exc:`FileNotFoundError` when
      given a path that does not exist, instead of silently treating the
      path string as XYZ content and raising a confusing :exc:`ValueError`.

   .. versionchanged:: 0.4.0

      :func:`~pasted._generator.read_xyz` now raises :exc:`FileNotFoundError`
      (or :exc:`IsADirectoryError`) for invalid paths, matching
      :meth:`Structure.from_xyz` behavior.  Previously it raised a confusing
      :exc:`ValueError` by attempting to parse the path string as XYZ text.


.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - What it measures
   * - ``H_atom``
     - Shannon entropy of the element composition (0 = pure, ln *k* = *k* equal elements)
   * - ``H_spatial``
     - Shannon entropy of the pairwise-distance histogram within *cutoff*
   * - ``H_total``
     - ``w_atom · H_atom + w_spatial · H_spatial``
   * - ``RDF_dev``
     - RMS deviation of the normalized RDF from a flat distribution
   * - ``shape_aniso``
     - Relative shape anisotropy κ² from the gyration tensor: 0 = spherical, 1 = rod-like ([0, 1])
   * - ``Q4``
     - Steinhardt bond-orientational order parameter *l* = 4
   * - ``Q6``
     - Steinhardt bond-orientational order parameter *l* = 6
   * - ``Q8``
     - Steinhardt bond-orientational order parameter *l* = 8
   * - ``graph_lcc``
     - Fraction of atoms in the largest connected component of the covalent-bond graph
   * - ``graph_cc``
     - Mean local clustering coefficient of the covalent-bond graph: 0 = no triangles, 1 = fully triangulated ([0, 1])
   * - ``ring_fraction``
     - Fraction of atoms participating in at least one ring
   * - ``charge_frustration``
     - Population variance of |Δχ| (Pauling electronegativity difference) across cutoff-adjacent atom pairs
   * - ``moran_I_chi``
     - Moran's spatial autocorrelation of electronegativity on the bond graph

.. rubric:: Element pool specification

.. autofunction:: pasted._atoms.parse_element_spec

.. note::

   When passing ``elements=`` as a *string*, it must contain **atomic numbers
   (integers)**, not element symbols.  ``elements="C,N,O"`` raises
   ``ValueError``; use ``elements="6,7,8"`` or ``elements=["C","N","O"]``
   instead.

   :func:`parse_element_spec` accepts **two** input forms:

   .. list-table::
      :header-rows: 1
      :widths: 30 30 40

      * - Form
        - Example
        - Meaning
      * - Comma-separated atomic numbers
        - ``"6,7,8"``
        - C, N, O
      * - Atomic-number range
        - ``"1-30"``
        - H through Zn
      * - Mixed ranges and singles
        - ``"1-10,26,28"``
        - H–Ne plus Fe and Ni
      * - Symbol list *(new in v0.3.5)*
        - ``["C", "N", "O"]``
        - Explicit list of element symbols

   .. versionchanged:: 0.3.5

      ``parse_element_spec`` now accepts a ``list[str]`` of element symbols
      in addition to atomic-number strings.  Previously, passing a list
      raised :exc:`AttributeError` because the function tried to call
      ``.split(",")`` on the list object.  The fix aligns the public API
      with the behavior already documented in :doc:`/quickstart`.

   .. versionchanged:: 0.3.5

      :meth:`Structure.from_xyz` now raises :exc:`FileNotFoundError` when
      given a path that does not exist, instead of silently treating the
      path string as XYZ content and raising a confusing :exc:`ValueError`.
