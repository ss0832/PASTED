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
from pasted._ext import HAS_RELAX, HAS_MAXENT, HAS_STEINHARDT, HAS_GRAPH
print(HAS_RELAX, HAS_MAXENT, HAS_STEINHARDT, HAS_GRAPH)  # True True True True
```

When `HAS_GRAPH` is `True`, both `graph_metrics_cpp` (graph / ring / charge /
Moran metrics) and `rdf_h_cpp` (`H_spatial` and `RDF_dev`) are active.
`compute_all_metrics` selects the C++ path automatically — no configuration
required.

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

| Flag | Description |
|---|---|
| `--mode gas` | Atoms placed uniformly at random inside a sphere or box |
| `--mode chain` | Random-walk chain with branching and directional persistence |
| `--mode shell` | Central atom surrounded by a coordination shell |
| `--mode maxent` | Maximum-entropy placement — neighbors spread as uniformly as possible |

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

---

## Python API

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
