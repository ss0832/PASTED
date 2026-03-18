Metric functions
================

.. automodule:: pasted._metrics
   :members:
   :undoc-members: False

MM-level descriptors
--------------------

The following three functions were added in version 0.1.9.  They share the
same ``cov_scale × (r_i + r_j)`` bond-detection threshold as
:func:`~pasted._placement.relax_positions`, ensuring that the concept of a
"bond" is consistent across placement, relaxation, and metric computation.

All three are included in :data:`~pasted._atoms.ALL_METRICS` and can
therefore be used as ``--filter`` targets on the CLI and in the
:class:`~pasted._generator.StructureGenerator` ``filters=`` parameter.

.. autofunction:: pasted._metrics.compute_bond_strain_rms
.. autofunction:: pasted._metrics.compute_ring_fraction
.. autofunction:: pasted._metrics.compute_charge_frustration
