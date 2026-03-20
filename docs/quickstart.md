# Quick start

## Installation

```bash
pip install pasted
```

Requires Python 3.10 or later. Python 3.10, 3.11, 3.12, and 3.13 are
officially supported and tested in CI.

A C++17 compiler is required to build the optional acceleration extensions.
If none is available the package still installs and runs on pure Python/NumPy.

Verify that the C++ extensions compiled successfully:

```python
from pasted._ext import HAS_RELAX, HAS_POISSON, HAS_MAXENT, HAS_MAXENT_LOOP, HAS_STEINHARDT, HAS_GRAPH
print(HAS_RELAX, HAS_POISSON, HAS_MAXENT, HAS_MAXENT_LOOP, HAS_STEINHARDT, HAS_GRAPH)
# True True True True True True  (all extensions compiled)
```

| Flag | What it enables |
|---|---|
| `HAS_RELAX` | C++ L-BFGS steric-clash relaxation (~20–50× vs Python) |
| `HAS_POISSON` | C++ Bridson Poisson-disk sampling functions (`_poisson_disk_sphere_cpp`, `_poisson_disk_box_cpp`). Same extension as `HAS_RELAX`; both are `True` together. `place_gas()` uses uniform random — call these functions directly when minimum-separation placement is needed. |
| `HAS_MAXENT` | C++ angular-repulsion gradient for `maxent` mode |
| `HAS_MAXENT_LOOP` | Full C++ L-BFGS loop for `place_maxent` (~10–22× vs `HAS_MAXENT_LOOP=False`) |
| `HAS_STEINHARDT` | C++ sparse Steinhardt Q_l (~2000× vs dense Python) |
| `HAS_GRAPH` | C++ O(N·k) graph / ring / charge / Moran / RDF metrics (~25× vs Python) |

When `HAS_GRAPH` is `True`, both `graph_metrics_cpp` (graph / ring / charge /
Moran metrics) and `rdf_h_cpp` (`H_spatial` and `RDF_dev`) are active.
`compute_all_metrics` selects the C++ path automatically — no configuration
required.

> **Performance note (HAS_GRAPH = False):** when the `_graph_core` extension
> did not compile, `compute_all_metrics` computes graph, ring, charge, and
> Moran metrics via a full O(N²) `scipy.spatial.distance.pdist` call.  This
> is significantly slower for N ≳ 500 (e.g. ~3 s at N=1000 vs ~17 ms with
> `HAS_GRAPH=True`).  If you see slow metric computation, confirm that
> `HAS_GRAPH` is `True` and reinstall with a C++17 compiler if not.

---

## CLI

The simplest invocation generates one gas-phase structure with 10 atoms drawn
from elements Z = 1–30:

```bash
pasted --n-atoms 10 --charge 0 --mult 1 \
       --mode gas --region sphere:8 \
       --elements 1-30 --n-samples 1 --seed 42
```

### Placement modes

| Flag | Description | `region` required? |
|---|---|:---:|
| `--mode gas` | Atoms placed uniformly at random inside a sphere or box | ✓ |
| `--mode chain` | Random-walk chain with branching and directional persistence | — |
| `--mode shell` | Central atom surrounded by a coordination shell | — |
| `--mode maxent` | Maximum-entropy placement — neighbors spread as uniformly as possible | ✓ |

> **`region` is required for `gas` and `maxent` modes.**  Use `--region sphere:R`
> (radius R Å) or `--region box:L` (L×L×L Å cube).  The `chain` and `shell`
> modes use their own geometry parameters (`--bond-range`, `--shell-radius`) and
> ignore `region`.

### Generating elongated chain structures

Use `--chain-bias` to bias the chain toward a global axis.
Higher values produce more rod-like structures with larger `shape_aniso`:

```bash
pasted --n-atoms 20 --charge 0 --mult 1 \
       --mode chain --branch-prob 0.0 --chain-bias 0.6 \
       --elements 6,7,8 --n-samples 50 --seed 0 \
       -o chains.xyz
```

### Filtering by disorder metrics

Use `--filter METRIC:MIN:MAX` to keep only structures within a metric range.
Use `-` for an open bound:

```bash
# Keep structures with H_total >= 2.0 and Q6 <= 0.3
pasted --n-atoms 15 --charge 0 --mult 1 \
       --mode gas --region sphere:8 \
       --elements 6,7,8,16 --n-samples 200 --seed 1 \
       --filter "H_total:2.0:-" \
       --filter "Q6:-:0.3" \
       -o filtered.xyz
```

Available metrics: `H_atom`, `H_spatial`, `H_total`, `RDF_dev`,
`shape_aniso`, `Q4`, `Q6`, `Q8`, `graph_lcc`, `graph_cc`,
`ring_fraction`, `charge_frustration`, `moran_I_chi`.

### Shell mode with a fixed center atom

```bash
pasted --n-atoms 8 --charge 0 --mult 1 \
       --mode shell --center-z 26 \
       --elements 1-30 --n-samples 20 --seed 7 \
       -o shell_fe.xyz
```

### Maximum-entropy mode (`maxent`)

`maxent` places atoms so that their angular distribution around each center is
as uniform as possible.  Like `gas`, it requires a `region` spec:

```bash
pasted --n-atoms 12 --charge 0 --mult 1 \
       --mode maxent --region sphere:6 \
       --elements 6,7,8 --n-samples 20 --seed 42 \
       -o maxent.xyz
```

> **Tip:** `maxent` is slower than `gas` or `chain` because it runs an
> iterative angular-repulsion optimisation per structure.  For large N, the
> C++ extension (`HAS_MAXENT_LOOP=True`) is strongly recommended.



### Functional API

`generate()` returns a `GenerationResult` — a list-compatible object that
also carries metadata about how many attempts were made and why samples were
rejected.  All existing code that treats the return value as a list continues
to work without modification:

```python
from pasted import generate

result = generate(
    n_atoms=12,
    charge=0,
    mult=1,
    mode="gas",
    region="sphere:9",
    elements="1-30",
    n_samples=50,
    seed=42,
    filters=["H_total:2.0:-"],
)

for s in result:
    print(s)           # Structure(n=14, comp='C2H8N2O2', mode='gas', H_total=2.341)
    print(s.to_xyz())  # extended-XYZ string
```

To inspect rejection statistics — useful when integrating PASTED into
automated pipelines such as ASE or high-throughput workflows:

```python
print(result.summary())
# e.g. "passed=5  attempted=50  rejected_parity=8  rejected_filter=37"

if result.n_rejected_parity > 0:
    print(f"{result.n_rejected_parity} attempt(s) failed the parity check.")
if not result:
    print("No structures passed — try relaxing filters or increasing n_samples.")
```

`generate()` and `StructureGenerator.generate()` also emit a
`UserWarning` automatically (via Python's `warnings` module) whenever
attempts are rejected by the parity check, no structures pass the filters,
or the attempt budget is exhausted before `n_success` is reached.  These
warnings fire regardless of `verbose` so that downstream tools receive a
machine-visible signal even when PASTED is silent:

```python
import warnings
from pasted import generate

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = generate(
        n_atoms=8, charge=0, mult=1,
        mode="gas", region="sphere:8",
        elements="6,7,8", n_samples=10, seed=0,
        filters=["H_total:999:-"],      # impossible — nothing will pass
    )

if w:
    print(w[0].message)
    # "No structures passed the metric filters after 10 attempt(s) ..."
```

### `maxent` mode

`maxent` requires a `region` argument — the same `"sphere:R"` / `"box:L"` string
used by `gas`:

```python
from pasted import generate

result = generate(
    n_atoms=12,
    charge=0,
    mult=1,
    mode="maxent",
    region="sphere:6",   # required for maxent, just like gas
    elements="6,7,8",
    n_samples=20,
    seed=42,
)
for s in result:
    print(s)
```

> **Note:** `chain` and `shell` do **not** accept a `region` argument — they
> use `bond_range` / `shell_radius` to control geometry.  Passing `region` to
> those modes has no effect and will raise a `TypeError`.  Only `gas` and
> `maxent` require `region`.

### Class API

```python
from pasted import StructureGenerator

gen = StructureGenerator(
    n_atoms=12,
    charge=0,
    mult=1,
    mode="chain",
    chain_bias=0.5,       # elongation bias
    elements="6,7,8",
    n_samples=100,
    seed=0,
)

result = gen.generate()   # returns GenerationResult
```

### Writing to file

```python
for i, s in enumerate(structures):
    s.write_xyz("out.xyz", append=(i > 0))
```

### Collecting a target number of structures

Use `n_success` to stop as soon as enough structures have passed all
filters, without exhausting the full attempt budget:

```python
from pasted import StructureGenerator

gen = StructureGenerator(
    n_atoms=15,
    charge=0,
    mult=1,
    mode="gas",
    region="sphere:8",
    elements="1-30",
    n_success=10,     # stop when 10 structures have passed
    n_samples=500,    # give up after 500 attempts at most
    filters=["H_total:2.0:-"],
    seed=42,
)
structures = gen.generate()   # returns up to 10 structures
```

Use `n_samples=0` for unlimited attempts — generation runs until
`n_success` is reached:

```python
gen = StructureGenerator(
    ...,
    n_samples=0,      # no attempt limit
    n_success=10,
)
```

### Streaming output (write as you go)

`stream()` yields each passing structure immediately.  This is useful when
you want to write results to disk without waiting for all attempts to
complete, or when an interrupted run should not lose collected structures:

```python
gen = StructureGenerator(
    n_atoms=12,
    charge=0,
    mult=1,
    mode="gas",
    region="sphere:9",
    elements="1-30",
    n_success=10,
    n_samples=500,
    seed=42,
)

for s in gen.stream():
    s.write_xyz("out.xyz")   # written immediately on each PASS
```

`generate()` delegates to `stream()` internally, so the two are equivalent
when you want a list:

```python
structures = gen.generate()        # list
structures = list(gen.stream())    # same result
```

### Accessing metrics

Each `Structure` carries a `metrics` dict:

```python
s = structures[0]
print(s.metrics["H_total"])
print(s.metrics["shape_aniso"])
print(s.metrics["Q6"])
```

---

## Optimizer

`StructureOptimizer` maximizes a user-defined objective function by running
Markov Chain Monte Carlo over atom positions and compositions.  Three methods
are available: Simulated Annealing, Basin-Hopping, and Parallel Tempering.

### Basic usage

```python
from pasted import StructureOptimizer

opt = StructureOptimizer(
    n_atoms=12,
    charge=0,
    mult=1,
    elements="6,7,8,15,16",          # C, N, O, P, S
    objective={"H_total": 1.0, "Q6": -2.0},  # maximize disorder, minimize order
    method="annealing",
    max_steps=5000,
    n_restarts=4,
    seed=42,
)

result = opt.run()
print(result.best)          # best structure found
print(result.summary())     # "restarts=4  best_f=…  worst_f=…  method='annealing'"

for rank, s in enumerate(result, 1):
    print(f"rank {rank}: H_total={s.metrics['H_total']:.3f}  Q6={s.metrics['Q6']:.3f}")
```

### Parallel Tempering

Parallel Tempering runs multiple replicas at different temperatures and
periodically exchanges states, allowing hot replicas to cross energy barriers
while cold replicas refine the best solutions found.

```python
opt = StructureOptimizer(
    n_atoms=12,
    charge=0,
    mult=1,
    elements="6,7,8,15,16",
    objective={"H_total": 1.0, "Q6": -2.0},
    method="parallel_tempering",
    n_replicas=4,               # temperatures: T_end, …, T_start (geometric)
    pt_swap_interval=10,        # attempt replica exchange every 10 steps
    max_steps=2000,
    n_restarts=2,
    T_start=1.0,
    T_end=0.01,
    seed=42,
)

result = opt.run()
print(result.summary())
```

### Electronegativity-targeted optimization

To find structures with frustrated or random electronegativity arrangements:

```python
opt = StructureOptimizer(
    n_atoms=10,
    charge=0,
    mult=1,
    elements="6,7,8,9,14,15,16",     # C, N, O, F, Si, P, S — wide EN range
    objective={
        "charge_frustration": 2.0,    # maximize EN variance across neighbors
        "moran_I_chi": -1.0,          # minimize EN spatial autocorrelation
    },
    method="parallel_tempering",
    n_replicas=4,
    max_steps=3000,
    n_restarts=3,
    seed=7,
)

result = opt.run()
s = result.best
print(f"charge_frustration: {s.metrics['charge_frustration']:.4f}")
print(f"moran_I_chi:        {s.metrics['moran_I_chi']:.3f}")
```

### Affine displacement moves

By default the optimizer uses fragment moves (displacing individual atoms).
Enable **affine moves** to also stretch, compress, and shear the entire
structure — useful for exploring anisotropic configurations:

```python
opt = StructureOptimizer(
    n_atoms=12,
    charge=0,
    mult=1,
    elements="6,7,8",
    objective={"H_total": 1.0},
    allow_affine_moves=True,   # half of displacement moves become affine
    affine_strength=0.15,      # stretch / compress up to ±15 %
    method="annealing",
    max_steps=3000,
    seed=42,
)
result = opt.run()
```

`affine_strength` controls the scale of the transform (default: `0.1`).
Practical range: 0.02–0.4.  Has no effect when `allow_affine_moves=False`.

### Objective function

The objective is **maximized**.  Use negative weights to penalize a metric:

```python
# Dict form: f = sum(w * metric)
objective = {"H_atom": 1.0, "H_spatial": 1.0, "Q6": -2.0}

# Callable form
objective = lambda m: m["H_spatial"] - 2.0 * m["Q6"]
```

Available metrics: all keys in `pasted.ALL_METRICS` —
`H_atom`, `H_spatial`, `H_total`, `RDF_dev`, `shape_aniso`,
`Q4`, `Q6`, `Q8`, `graph_lcc`, `graph_cc`,
`ring_fraction`, `charge_frustration`, `moran_I_chi`.

---

## GeneratorConfig — immutable configuration object

`GeneratorConfig` is a `frozen=True` dataclass that encapsulates every
parameter of `StructureGenerator`.  It gives full mypy / IDE type-checking
and allows safe one-field overrides via `dataclasses.replace`.

```python
import dataclasses
from pasted import GeneratorConfig, StructureGenerator

# Build once
cfg = GeneratorConfig(
    n_atoms=20, charge=0, mult=1,
    mode="gas", region="sphere:10",
    elements="6,7,8", n_samples=100, seed=42,
)

# Pass to the class API
result = StructureGenerator(cfg).generate()

# One-field override — creates a new config, does not mutate the original
cfg_new_seed = dataclasses.replace(cfg, seed=99)
result2 = StructureGenerator(cfg_new_seed).generate()
```

`generate()` also accepts a config directly:

```python
from pasted import generate, GeneratorConfig

result = generate(GeneratorConfig(n_atoms=12, charge=0, mult=1,
                                  mode="chain", elements="6,7,8",
                                  n_samples=50, seed=0))
```

The original keyword-argument style (`StructureGenerator(n_atoms=..., ...)`
and `generate(n_atoms=..., ...)`) is **fully backward-compatible** and
continues to work unchanged.  All instance attributes
(`gen.n_atoms`, `gen.seed`, …) are still accessible via `__getattr__` proxy.

---

## Affine transforms in StructureGenerator

Set `affine_strength > 0` to apply a random affine transformation
(stretch/compress one axis + shear one axis pair) to each generated
structure **before** `relax_positions`.  This creates more anisotropic
initial geometries while still guaranteeing clash-free output after relax.

```python
from pasted import generate

result = generate(
    n_atoms=20, charge=0, mult=1,
    mode="gas", region="sphere:10",
    elements="6,7,8", n_samples=50, seed=42,
    affine_strength=0.2,   # ±20 % stretch + ±10 % shear before relax
)
```

| `affine_strength` | Effect |
|---|---|
| `0.0` (default) | disabled — backward-compatible, no transform |
| `0.05`–`0.1` | subtle anisotropy, negligible overhead |
| `0.2`–`0.4` | strong anisotropy; useful for chain/shell modes |

Works across **all placement modes** (`gas`, `chain`, `shell`, `maxent`).
CLI: `--affine-strength S`.

> **Relationship to `StructureOptimizer`:** both use the same `_affine_move`
> function from `pasted._placement`.  In the Generator the transform is applied
> once per structure before relax; in the Optimizer it is applied per MC step
> when `allow_affine_moves=True`.

---

## Element sampling control (StructureGenerator)

### Biased element fractions

By default each element is sampled with equal probability.  Pass
`element_fractions` to shift the distribution:

```python
from pasted import StructureGenerator

gen = StructureGenerator(
    n_atoms=20, charge=0, mult=1,
    mode="gas", region="sphere:10",
    elements="6,7,8",
    element_fractions={"C": 0.6, "N": 0.3, "O": 0.1},  # C-rich
    n_samples=50, seed=0,
)
result = gen.generate()
```

Weights are *relative* — they are normalised internally.
`{"C": 6, "N": 3, "O": 1}` is equivalent to the above.
Elements absent from the dict receive weight `1.0`.

### Element count bounds

Use `element_min_counts` and `element_max_counts` to guarantee or cap the
number of specific atoms in each generated structure:

```python
gen = StructureGenerator(
    n_atoms=15, charge=0, mult=1,
    mode="gas", region="sphere:10",
    elements="6,7,8,15,16",          # C, N, O, P, S
    element_min_counts={"C": 4},      # always at least 4 carbons
    element_max_counts={"N": 3, "O": 3},  # cap nitrogen and oxygen
    n_samples=100, seed=42,
)
result = gen.generate()

from collections import Counter
for s in result:
    c = Counter(s.atoms)
    assert c["C"] >= 4
    assert c.get("N", 0) <= 3
    assert c.get("O", 0) <= 3
```

Bounds are enforced during atom sampling, before the parity check.
A `ValueError` is raised at construction time when constraints are
inconsistent (sum of mins > `n_atoms`, or min > max for any element).

### Combining fractions and bounds

All three parameters can be used together:

```python
gen = StructureGenerator(
    n_atoms=12, charge=0, mult=1,
    mode="chain",
    elements="6,7,8",
    element_fractions={"C": 5, "N": 2, "O": 1},
    element_min_counts={"C": 2},
    element_max_counts={"N": 4},
    n_samples=30, seed=7,
)
```

---

## Position-only optimisation (StructureOptimizer)

Set `allow_composition_moves=False` to fix the composition and only
optimise atomic positions.  This is useful when the stoichiometry is
predetermined:

```python
from pasted import StructureOptimizer, Structure

# Load a structure with a fixed composition
initial = Structure.from_xyz("my_structure.xyz")

opt = StructureOptimizer(
    n_atoms=len(initial),
    charge=initial.charge,
    mult=initial.mult,
    elements=list(set(initial.atoms)),
    objective={"H_total": 1.0, "Q6": -2.0},
    allow_composition_moves=False,   # position-only
    method="annealing",
    max_steps=5000,
    seed=42,
)

result = opt.run(initial=initial)
print(result.best)
# Composition is identical to initial; only positions have changed
assert sorted(result.best.atoms) == sorted(initial.atoms)
```

---

## Composition-only optimisation (StructureOptimizer)

Set `allow_displacements=False` to fix the atomic coordinates and only
optimise element types.  This is useful when exploring compositional
disorder on a pre-relaxed geometry (e.g. a fixed lattice):

```python
from pasted import StructureOptimizer, Structure

# Load a geometry with a fixed set of coordinates
initial = Structure.from_xyz("fixed_geometry.xyz")

opt = StructureOptimizer(
    n_atoms=len(initial),
    charge=initial.charge,
    mult=initial.mult,
    elements=["Cr", "Mn", "Fe", "Co", "Ni"],  # Cantor alloy pool
    objective={"H_atom": 1.0, "Q6": -2.0},
    allow_displacements=False,   # composition-only; coordinates fixed
    method="annealing",
    max_steps=5000,
    seed=42,
)

result = opt.run(initial=initial)
print(result.best)
# Positions are identical to initial; only element labels have changed
import numpy as np
np.testing.assert_allclose(
    np.array(result.best.positions), np.array(initial.positions)
)
```

> **Note**: `allow_displacements=False` and `allow_composition_moves=False`
> cannot both be set — at least one move type must be enabled.  Attempting
> to do so raises a `ValueError`.

---

## OpenMP parallelization (Linux only)

When PASTED is installed on Linux with a GCC or Clang toolchain, the C++
extensions are compiled with `-fopenmp` automatically.  All four inner-loop
modules (`_relax_core`, `_steinhardt_core`, `_graph_core`, `_maxent_core`)
benefit from parallelization.

### Checking availability

```python
import pasted

print(pasted.HAS_OPENMP)   # True on a -fopenmp build, False otherwise
```

### Setting the thread count at runtime

```python
import pasted

pasted.set_num_threads(4)  # use 4 threads for all subsequent calls
```

`set_num_threads` is a no-op when `HAS_OPENMP` is `False`, so it is safe to
call unconditionally.  The standard `OMP_NUM_THREADS` environment variable is
also respected; `set_num_threads` overrides it when called after import.

### CLI

```
pasted --n-atoms 50000 --mode gas --region sphere:250 \
    --charge 0 --mult 1 --n-threads 4 -o out.xyz
```

> **Default thread count is 1 (single-threaded) since v0.2.2.**  Pass `--n-threads N` to enable OpenMP parallelism.  On machines with ≤ 2 cores the default of 1 avoids thread-spawn overhead exceeding the parallelism gain.

### Opting out

To build without OpenMP (e.g. for reproducibility testing):

```
PASTED_DISABLE_OPENMP=1 pip install -e .
```

> **Platform note**: OpenMP is supported on Linux only.  On macOS and Windows
> the extensions build without `-fopenmp` regardless of toolchain, and
> `HAS_OPENMP` will be `False`.
