GeneratorConfig
===============

.. autoclass:: pasted._config.GeneratorConfig
   :members:
   :show-inheritance:

.. rubric:: Required fields

The three fields without defaults — **n_atoms**, **charge**, **mult** —
must always be supplied explicitly.  All other fields carry sensible
defaults and are optional.

.. rubric:: One-field override pattern

:class:`GeneratorConfig` is ``frozen=True``, so instances are immutable
and hashable.  Use :func:`dataclasses.replace` to derive a new config
that differs in exactly one field without mutating the original::

    import dataclasses
    from pasted import GeneratorConfig, StructureGenerator

    base = GeneratorConfig(
        n_atoms=20, charge=0, mult=1,
        mode="gas", region="sphere:10",
        elements="6,7,8", n_samples=100, seed=42,
    )

    for seed in range(10):
        cfg = dataclasses.replace(base, seed=seed)
        result = StructureGenerator(cfg).generate()

.. rubric:: Passing to the functional API

:func:`~pasted._generator.generate` also accepts a
:class:`GeneratorConfig` as its first positional argument::

    from pasted import generate, GeneratorConfig

    result = generate(
        GeneratorConfig(n_atoms=12, charge=0, mult=1,
                        mode="chain", elements="6,7,8",
                        n_samples=50, seed=0)
    )

The original keyword-argument style is **fully backward-compatible**
and continues to work without modification.

.. rubric:: Affine-transform fields (StructureGenerator)

.. note::

   ``affine_jitter`` has no visible effect when used inside
   :class:`~pasted._generator.StructureGenerator` because the internal
   ``move_step`` is ``0.0`` at generation time.  The jitter term is only
   meaningful in :class:`~pasted._optimizer.StructureOptimizer`, where
   ``move_step`` is set per MC step.
   See :ref:`affine-moves` in the optimizer documentation for details.
