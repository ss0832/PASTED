Placement functions
===================

.. automodule:: pasted._placement
   :members:
   :undoc-members: False
   :exclude-members: Vec3

.. note::

   **v0.2.6 — O(N) neighbor-cutoff computation in** ``place_maxent``

   The angular-repulsion neighbor cutoff (``ang_cutoff``) is now derived via
   ``float(numpy.median(radii)) * 2.0`` instead of sorting all N*(N+1)/2
   pairwise radius sums.  The identity ``median(rᵢ + rⱼ) = 2 · median(rᵢ)``
   holds for all built-in element pools and produces a numerically identical
   result, while reducing the pre-loop setup from O(N² log N) to O(N).
   See the :func:`~pasted._placement.place_maxent` docstring for benchmark
   numbers.
