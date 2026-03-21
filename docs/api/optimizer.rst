StructureOptimizer
==================

.. autoclass:: pasted._optimizer.StructureOptimizer
   :members:
   :show-inheritance:

.. autofunction:: pasted._optimizer.parse_objective_spec

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
