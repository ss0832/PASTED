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

The 13 keys in ``ALL_METRICS`` are the only names accepted by ``filters=``,
``--filter``, and the ``objective=`` dict.  They are also the keys present
in :attr:`Structure.metrics <pasted._generator.Structure.metrics>` and
:attr:`EvalContext.metrics <pasted._optimizer.EvalContext.metrics>`.

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
     - Mean EN variance across bonded atom pairs (Pauling scale)
   * - ``moran_I_chi``
     - Moran's spatial autocorrelation of electronegativity on the bond graph

.. rubric:: Element pool specification

.. autofunction:: pasted._atoms.parse_element_spec

.. note::

   When passing ``elements=`` as a *string*, it must contain **atomic numbers
   (integers)**, not element symbols.  ``elements="C,N,O"`` raises
   ``ValueError``; use ``elements="6,7,8"`` or ``elements=["C","N","O"]``
   instead.

   Three string forms are accepted by :func:`parse_element_spec`:

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
