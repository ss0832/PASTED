StructureGenerator & generate()
================================

.. autoclass:: pasted._generator.StructureGenerator
   :members:
   :show-inheritance:

.. autofunction:: pasted._generator.generate

.. autofunction:: pasted._generator.read_xyz

Structure
---------

.. autoclass:: pasted._generator.Structure
   :members:
   :show-inheritance:
   :no-index:

GenerationResult
----------------

.. autoclass:: pasted._generator.GenerationResult
   :members:
   :show-inheritance:
   :no-index:

.. note::

   **Attribute naming — always use the** ``n_`` **prefix.**

   The one-line string returned by :meth:`~pasted._generator.GenerationResult.summary`
   uses short labels (``passed``, ``attempted``, ``rejected_parity``,
   ``rejected_filter``).  The corresponding Python attributes carry an ``n_``
   prefix: ``result.n_passed``, ``result.n_attempted``,
   ``result.n_rejected_parity``, ``result.n_rejected_filter``.
   Accessing ``result.passed`` or ``result.attempted`` directly raises
   ``AttributeError``.

.. note::

   **Automatic** ``UserWarning`` **signals.**

   Both :func:`~pasted._generator.generate` and
   :meth:`~pasted._generator.StructureGenerator.generate` emit a
   ``UserWarning`` (via Python's :mod:`warnings` module) whenever:

   * any attempt is rejected by the parity check
     (``n_rejected_parity > 0``),
   * no structures pass the metric filters, or
   * the attempt budget is exhausted before ``n_success`` is reached.

   These warnings fire regardless of ``verbose`` so that downstream tools
   receive a machine-visible signal even when PASTED is silent::

       import warnings
       from pasted import generate

       with warnings.catch_warnings(record=True) as w:
           warnings.simplefilter("always")
           result = generate(
               n_atoms=8, charge=0, mult=1,
               mode="gas", region="sphere:8",
               elements="6",
               n_samples=10, seed=0,
               filters=["H_total:999:-"],   # impossible — nothing will pass
           )
       if w:
           print(w[0].message)
