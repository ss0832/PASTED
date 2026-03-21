StructureOptimizer
==================

.. autoclass:: pasted._optimizer.StructureOptimizer
   :members:
   :show-inheritance:

.. autofunction:: pasted._optimizer.parse_objective_spec

.. note::

   **Parity validation at construction time.**

   :class:`StructureOptimizer` checks at ``__init__`` time that the element
   pool can produce at least one composition of ``n_atoms`` atoms that
   satisfies the charge/multiplicity parity constraint.  If it cannot,
   ``ValueError`` is raised immediately — before any call to
   :meth:`~pasted._optimizer.StructureOptimizer.run`.  This makes
   ``max_init_attempts=0`` (unlimited retries) safe: if construction
   succeeds, a valid initial structure is guaranteed to eventually be found.

.. note::

   **Move-type constraints.**

   ``allow_displacements=False`` and ``allow_composition_moves=False`` cannot
   both be set at the same time unless ``allow_affine_moves=True``.  Setting
   all three to ``False`` raises ``ValueError``.

.. _affine-moves:

.. rubric:: Affine moves

When ``allow_affine_moves=True``, half of all displacement moves are
replaced by random affine transforms (stretch / compress along one axis,
shear one axis pair, and per-atom jitter).  This lets the optimizer
explore anisotropic configurations that fragment moves cannot reach
efficiently.

Unlike in :class:`~pasted._generator.StructureGenerator`, the
``affine_jitter`` term **does** have a visible effect here because
``move_step`` is non-zero during MC steps.

.. rubric:: Position-only optimization

Set ``allow_composition_moves=False`` to fix the stoichiometry and only
move atoms::

    result = opt.run(initial=my_structure)
    assert sorted(result.best.atoms) == sorted(my_structure.atoms)

.. rubric:: Composition-only optimization

Set ``allow_displacements=False`` to fix the atomic coordinates and
only swap element labels.  Atoms outside the pool are automatically
replaced by parity-compatible pool elements before the first MC step,
so cross-pool starting structures work with all three methods::

    result = opt.run(initial=my_structure)
    import numpy as np
    np.testing.assert_allclose(
        np.array(result.best.positions), np.array(my_structure.positions)
    )

OptimizationResult
------------------

.. autoclass:: pasted._optimizer.OptimizationResult
   :members:
   :show-inheritance:
   :no-index:

EvalContext
-----------

.. autoclass:: pasted._optimizer.EvalContext
   :members:
   :show-inheritance:

   Full evaluation context passed as the second argument to a 2-parameter
   objective callable.  Consolidates the current candidate structure, all
   pre-computed disorder metrics, and the live optimizer runtime state.

   **Calling conventions**

   Two calling conventions are supported for the ``objective`` parameter of
   :class:`StructureOptimizer`:

   * **1-argument** ``f(m)`` — ``m`` is a ``dict[str, float]`` of disorder
     metrics.  Fully backward-compatible with all existing code.
   * **2-argument** ``f(m, ctx)`` — ``m`` is the same metrics dict; ``ctx``
     is an :class:`EvalContext`.  Dispatch is based on the number of
     *required* positional parameters via :func:`inspect.signature`.
     A callable with a default for the second argument
     (``lambda m, ctx=None:``) is treated as 1-argument.

   **ObjectiveType alias**

   .. code-block:: python

      ObjectiveType = (
          dict[str, float]
          | Callable[[dict[str, float]], float]
          | Callable[[dict[str, float], EvalContext], float]
      )

   :class:`EvalContext` is exported from the top-level ``pasted`` namespace::

       from pasted import EvalContext

   **Example — adaptive curriculum objective**

   .. code-block:: python

       def curriculum_objective(m: dict, ctx: EvalContext) -> float:
           """Broad exploration early, strong Q6 penalty late."""
           base = m["H_total"]
           if ctx.progress < 0.5:
               return base
           else:
               return base - 3.0 * m["Q6"]

       opt = StructureOptimizer(
           n_atoms=15, charge=0, mult=1, elements="6,7,8,16",
           objective=curriculum_objective,
           method="annealing", max_steps=4000, seed=7,
       )

   **Example — per-atom Q6 locality penalty**

   .. code-block:: python

       import numpy as np

       def local_disorder_objective(m: dict, ctx: EvalContext) -> float:
           q6_var = float(np.var(ctx.per_atom_q6))
           q6_max = float(np.max(ctx.per_atom_q6))
           return m["H_total"] + q6_var * 0.5 - q6_max * 1.0
