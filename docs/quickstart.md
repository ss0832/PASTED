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
from pasted._ext import HAS_RELAX, HAS_MAXENT, HAS_STEINHARDT
print(HAS_RELAX, HAS_MAXENT, HAS_STEINHARDT)   # True True True  → all acceleration active
```

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
`bond_strain_rms`, `ring_fraction`, `charge_frustration`.

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

```python
from pasted import generate

structures = generate(
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

for s in structures:
    print(s)           # Structure(n=14, comp='C2H8N2O2', mode='gas', H_total=2.341)
    print(s.to_xyz())  # extended-XYZ string
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

structures = gen.generate()
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
