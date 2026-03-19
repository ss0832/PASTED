# Architecture

## Design philosophy

PASTED has one job: produce atomic structures that are strange enough to
stress-test quantum-chemistry (QC) and machine-learning potential (MLP)
codes, but not so malformed that they cannot be parsed at all.

Three principles guide every design decision:

**The tool does not judge.**
Physical validity is not checked beyond the minimum-distance constraint.
Charge/multiplicity parity is validated; everything else is the engine's
problem.  The point is to find the engine's breaking points, not to
generate textbook molecules.

**Fail gracefully, not loudly.**
The C++ acceleration layer is optional.  If the extensions are not compiled,
the package falls back to pure Python/NumPy with no change in behavior.
If `relax_positions` does not converge within `max_cycles`, the best
available structure is returned with `converged=False` — the run continues.

**Small surface, stable interface.**
New parameters are added only when they cannot be reproduced by composing
existing ones.  `chain_bias=0.0` is the default precisely because it
changes nothing for existing users.  Internals can be restructured freely
as long as the public API stays the same.

---

## Source layout

```
src/pasted/
├── __init__.py          Public API exports and package docstring
├── _atoms.py            Element data, covalent radii, pool/filter parsing
├── _generator.py        StructureGenerator class and generate() function
├── _io.py               XYZ serialization (format_xyz)
├── _metrics.py          All disorder metrics: entropy, RDF, Steinhardt, graph
├── _optimizer.py        StructureOptimizer — basin-hopping on existing structures
├── _placement.py        Placement algorithms + relax_positions dispatcher
├── cli.py               argparse CLI entry point
└── _ext/
    ├── __init__.py      HAS_RELAX / HAS_MAXENT / HAS_STEINHARDT flags; None fallbacks
    ├── _relax.cpp       C++17: relax_positions with flat Cell List (N≥64)
    ├── _maxent.cpp      C++17: angular_repulsion_gradient with Cell List (N≥64)
    └── _steinhardt.cpp  C++17: Steinhardt Q_l with sparse neighbor list (N≥64)
```

---

## Placement modes

### gas

Atoms are placed uniformly at random inside a sphere or axis-aligned box.
No clash checking at placement time — `relax_positions` is called afterwards
to resolve any distance violations.

### chain

A random-walk chain grows from the origin.  Each atom is attached to a
randomly chosen existing tip.  Two parameters control shape:

- `chain_persist` — local directional memory (cone constraint on turn angle)
- `chain_bias` — global axis drift; the first bond direction becomes a bias
  axis and all subsequent steps are blended toward it

`chain_bias=0.0` (the default) reproduces the behavior of versions prior
to 0.1.6 exactly.

### shell

A central atom is placed at the origin, surrounded by `coord_num` shell
atoms at a random radial distance.  Remaining atoms are attached as tails
via a short random walk from shell atoms.

### maxent

Atoms start from a random gas placement and are iteratively repositioned by
gradient descent on an angular repulsion potential:

$$U = \sum_i \sum_{j,k \in N(i)} \frac{1}{1 - \cos\theta_{jk} + \varepsilon}$$

The result is the constrained maximum-entropy solution: neighbor directions
spread as uniformly over the sphere as the distance constraints allow.

---

## Disorder metrics

All metrics are computed once per structure in `compute_all_metrics` and
stored in `Structure.metrics`.

| Metric | Range | Description |
|---|---|---|
| `H_atom` | ≥ 0 | Shannon entropy of element composition |
| `H_spatial` | ≥ 0 | Shannon entropy of pairwise-distance histogram |
| `H_total` | ≥ 0 | `w_atom × H_atom + w_spatial × H_spatial` |
| `RDF_dev` | ≥ 0 | RMS deviation of empirical g(r) from ideal-gas baseline |
| `shape_aniso` | [0, 1] | Relative shape anisotropy from gyration tensor (0=sphere, 1=rod) |
| `Q4`, `Q6`, `Q8` | [0, 1] | Steinhardt bond-order parameters |
| `graph_lcc` | [0, 1] | Largest connected-component fraction |
| `graph_cc` | [0, 1] | Mean clustering coefficient |
| `bond_strain_rms` | ≥ 0 | RMS relative deviation of bonded-pair distances from Pyykkö ideal lengths |
| `ring_fraction` | [0, 1] | Fraction of atoms belonging to at least one ring (Union-Find detection) |
| `charge_frustration` | ≥ 0 | Variance of Pauling electronegativity differences across bonded pairs |

---

## C++ acceleration layer

The `_ext` sub-package contains three independently compiled pybind11 modules.
Each can be absent without affecting the others.

### `_relax_core` — `relax_positions`

Resolves distance violations by L-BFGS minimization of the harmonic
steric-clash penalty energy:

$$E = \sum_{i<j} \tfrac{1}{2} \max\!\bigl(0,\; \text{cov\_scale}\cdot(r_i+r_j) - d_{ij}\bigr)^2$$

Gradients are computed analytically; pair enumeration uses `FlatCellList`
for N ≥ 64 (O(N) per energy evaluation) and an O(N²) full-pair loop for
N < 64.  The L-BFGS solver (history depth m = 7, Armijo backtracking) is
implemented in C++17 standard library with no external dependencies.

Convergence criterion: E < 1 × 10⁻⁶.  Typical iteration count: 50–300
for dense 5000-atom structures.

### `_maxent_core` — `angular_repulsion_gradient`

Computes ∂U/∂rᵢ for the angular repulsion potential used by `place_maxent`.

- **N < 32**: O(N³) full neighbor search
- **N ≥ 32**: O(N²) Cell List neighbor search (cell width = `cutoff`)

### `_steinhardt_core` — `compute_steinhardt_per_atom`

Computes per-atom Steinhardt Q_l bond-order parameters using a sparse
neighbor list, replacing the dense O(N²) Python/scipy path.

- **N < 64**: O(N²) full-pair loop
- **N ≥ 64**: O(N·k) flat Cell List (k = mean neighbor count)

Spherical harmonics are evaluated via the associated Legendre polynomial
three-term recurrence with no scipy call in the hot loop.  The symmetry
`|Y_l^{-m}|² = |Y_l^m|²` halves harmonic evaluations by computing only
m = 0..l.  When absent, `_steinhardt_per_atom_sparse` provides the same
O(N·k) complexity using `np.bincount` for accumulation.

| Path | N=2000 | Speed-up vs dense Python |
|---|:---:|:---:|
| Dense Python (original) | ~35 s | 1× |
| Sparse Python fallback | ~0.21 s | ~164× |
| C++ (`_steinhardt_core`) | ~17 ms | **~2 000×** |

---

## Generation flow

`StructureGenerator.stream()` is the single implementation of the generation
loop.  It yields each passing structure immediately, enabling incremental
file output and early stopping via `n_success`.

`generate()` delegates to `stream()` and collects results into a list.
`__iter__` also delegates to `stream()`, so all three call sites share the
same loop logic.

`n_success` and `n_samples` together control termination:

- `n_samples > 0`, `n_success = None` — attempt exactly `n_samples` times
  (original behavior).
- `n_samples > 0`, `n_success = N` — stop at N successes or `n_samples`
  attempts, whichever comes first.
- `n_samples = 0`, `n_success = N` — unlimited attempts; stop only when N
  structures have passed.

---

## Reproducibility

All random decisions pass through a single `random.Random(seed)` instance
created at the start of each `stream()` call.  The C++ extensions accept
an integer `seed` for the coincident-atom RNG edge case.  With the same
`seed`, the same `n_atoms`, and the same parameters, `stream()` (and
therefore `generate()`) returns bit-for-bit identical output.
