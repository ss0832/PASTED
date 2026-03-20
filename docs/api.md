# API reference

## Top-level (`pasted`)

```{eval-rst}
.. automodule:: pasted
   :members:
   :undoc-members:
```

## Structure generation

```{eval-rst}
.. autoclass:: pasted.StructureGenerator
   :members:

.. autofunction:: pasted.generate

.. autoclass:: pasted.GenerationResult
   :members:

.. autoclass:: pasted.Structure
   :members:
```

## Structure optimisation

```{eval-rst}
.. autoclass:: pasted.StructureOptimizer
   :members:

.. autoclass:: pasted.OptimizationResult
   :members:
```

## Metrics

```{eval-rst}
.. autofunction:: pasted.compute_all_metrics
.. autofunction:: pasted.compute_angular_entropy
.. autofunction:: pasted.compute_ring_fraction
.. autofunction:: pasted.compute_charge_frustration
.. autofunction:: pasted.compute_moran_I_chi
.. autofunction:: pasted.compute_steinhardt_per_atom
```

## Utilities

```{eval-rst}
.. autofunction:: pasted.generate
.. autofunction:: pasted.read_xyz
.. autofunction:: pasted.parse_xyz
.. autofunction:: pasted.format_xyz
.. autofunction:: pasted.parse_element_spec
.. autofunction:: pasted.parse_filter
.. autofunction:: pasted.validate_charge_mult
```

## C++ extensions (`pasted._ext`)

```{eval-rst}
.. automodule:: pasted._ext
   :members:
```

### Available flags

| Flag | Meaning |
|------|---------|
| `HAS_RELAX` | `_relax_core` repulsion-relaxation extension available |
| `HAS_POISSON` | Bridson Poisson-disk placement available |
| `HAS_MAXENT` | `_maxent_core` angular-gradient extension available |
| `HAS_MAXENT_LOOP` | Full C++ L-BFGS maxent loop available |
| `HAS_STEINHARDT` | Sparse Steinhardt Q_l extension available |
| `HAS_GRAPH` | O(N·k) graph/metrics extension available |

> **Note (v0.2.3):** `HAS_OPENMP` and `set_num_threads` were removed in
> v0.2.3.  All computation is single-threaded.  See the
> [Changelog](changelog.md) for details.
