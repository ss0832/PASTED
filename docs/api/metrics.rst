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
