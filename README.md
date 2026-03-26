# PASTED

[![CI](https://github.com/ss0832/PASTED/actions/workflows/ci.yml/badge.svg)](https://github.com/ss0832/PASTED/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/pasted.svg)](https://img.shields.io/pypi/v/pasted.svg)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pasted.svg)](https://img.shields.io/pypi/pyversions/pasted.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://img.shields.io/badge/License-MIT-yellow.svg)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pasted?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=RED&left_text=downloads)](https://pepy.tech/projects/pasted)

**P**ointless **A**tom **ST**ructure with **E**ntropy **D**iagnostics

PASTED is a structure fuzzer for quantum chemistry (QC) and machine-learning
potential (MLP) codes. It generates intentionally random, physically
meaningless atomic structures and quantifies their disorder through a suite
of 17 structural metrics. Useful for stress-testing structure optimizers,
generating worst-case inputs for QC codes, or exploring what "maximum chaos"
looks like in structural space.

*This program was developed using LLMs.

## Features

- **Four placement modes** — random gas (`gas`), chain-growth (`chain`),
  coordination-complex-like (`shell`), and maximum-entropy (`maxent`)
- **13 disorder metrics** computed per structure, all usable as output filters
- **Element pool** specified by atomic number (Z = 1–106); composition sampled
  randomly per structure
- **Guaranteed atom count** — post-placement L-BFGS repulsion relaxation
  ensures `--n-atoms` atoms are always delivered regardless of initial density
- **Auto-scaled `--cutoff`** — defaults to `cov_scale × 1.5 × median(r_i + r_j)`
  over the element pool; all graph, Steinhardt, ring, and charge metrics share
  this single cutoff
- **Structure optimizer** — `StructureOptimizer` runs simulated annealing or
  basin-hopping on an existing structure to maximize a user-defined disorder
  objective
- Charge/multiplicity parity validation, reproducible via `--seed`, incremental
  output via `stream()`

## Requirements

```
Linux (tested)
Python >= 3.10
numpy
scipy
C++ compiler (GCC/Clang)
pybind11 >= 2.12
```

A C++17 compiler is required to build the optional acceleration extensions. If no compiler is
available, the package falls back to pure Python/NumPy transparently.

## Installation

```bash
pip install pasted
```

or from source:

```bash
# Building from source requires pybind11 >= 2.12
git clone https://github.com/ss0832/pasted.git
cd pasted
pip install -e .
# or run directly without installing:
python pasted.py --help
```

Verify that the C++ extensions compiled successfully:

```python
from pasted._ext import HAS_RELAX, HAS_MAXENT, HAS_STEINHARDT, HAS_GRAPH
print(HAS_RELAX, HAS_MAXENT, HAS_STEINHARDT, HAS_GRAPH)
# True True True True  ->  all acceleration active
```

## Quick Start

```bash
# 10 atoms drawn from H-Zn, placed randomly in a sphere
pasted --n-atoms 10 --elements 1-30 --charge 0 --mult 1 \
    --mode gas --region sphere:8

# Chain structure (C/N/O), 20 samples, filter by disorder
pasted --n-atoms 15 --elements 6,7,8 --charge 0 --mult 1 \
    --mode chain --branch-prob 0.4 --n-samples 20 \
    --filter "H_total:2.0:-" -o organic_junk.xyz

# Coordination-complex-like structure with Fe center
pasted --n-atoms 12 --elements 6,7,8,26 --charge 0 --mult 1 \
    --mode shell --center-z 26 --coord-range 4:6 --n-samples 10

# Stop as soon as 10 disordered structures are found
pasted --n-atoms 15 --elements 1-30 --charge 0 --mult 1 \
    --mode gas --region sphere:8 \
    --filter "H_total:2.0:-" --n-success 10 --n-samples 500 \
    -o disordered.xyz

# Select spatially random electronegativity arrangements (Moran's I near 0)
pasted --n-atoms 50 --elements 1-30 --charge 0 --mult 1 \
    --mode gas --region sphere:12 --n-samples 200 \
    --filter "moran_I_chi:-0.1:0.1" -o random_en.xyz
```

## Placement Modes

### `gas` (default)

Atoms placed uniformly at random inside a sphere or box. No clash checking
at placement time — repulsion relaxation resolves all violations afterward.

```
--region sphere:R       sphere of radius R Angstrom
--region box:L          cube of side L Angstrom
--region box:LX,LY,LZ   orthorhombic box
```

### `chain`

Atoms grow one by one from a seed via a random walk with directional
persistence. Produces elongated, tree-like structures.

```
--branch-prob FLOAT     branching probability (default: 0.3)
--chain-persist FLOAT   directional persistence 0.0-1.0 (default: 0.5)
--chain-bias FLOAT      global axis drift; higher -> more rod-like (default: 0.0)
--bond-range LO:HI      bond length range Angstrom (default: 1.2:1.6)
```

### `shell`

One center atom surrounded by a coordination shell, plus tail atoms grown
from shell members. Resembles coordination complexes.

```
--center-z Z            atomic number of center atom (default: random)
--coord-range MIN:MAX   coordination number range (default: 4:8)
--shell-radius LO:HI    shell radius range Angstrom (default: 1.8:2.5)
--bond-range LO:HI      tail bond length range Angstrom (default: 1.2:1.6)
```

### `maxent`

Atoms start from a random gas placement and are repositioned by gradient
descent on an angular repulsion potential, spreading neighbor directions as
uniformly over the sphere as the distance constraints allow.

```
--region SPEC           same as gas mode (required)
--maxent-steps N        gradient-descent iterations (default: 300)
--maxent-lr LR          learning rate (default: 0.05)
--maxent-cutoff-scale S neighbor cutoff scale factor (default: 2.5)
```

## Element Pool

```
--elements SPEC
```

| Syntax | Meaning |
|---|---|
| `1-30` | Z = 1 through 30 (H to Zn) |
| `6,7,8` | Z = 6, 7, 8 (C, N, O) |
| `1-10,26,28` | Z = 1-10 plus Fe(26) and Ni(28) |
| *(omitted)* | all Z = 1-106 |

> **Note:** `--elements` (CLI) and the `elements=` keyword (Python API) accept
> **atomic-number integers only** when given as a string — e.g. `"6,7,8"` for
> C, N, O.  Symbol strings such as `"C,N,O"` raise `ValueError`.  To pass
> symbols from Python, use a list: `elements=["C", "N", "O"]`.

If H (Z = 1) is in the pool and the sampled composition contains no hydrogen,
a random number of H atoms is automatically appended. Disable with
`--no-add-hydrogen`.

## Interatomic Distance Control

PASTED enforces a minimum interatomic distance using Pyykkoe single-bond
covalent radii (Pyykkoe & Atsumi, *Chem. Eur. J.* **15**, 186-197, 2009):

```
d_min(i, j) = cov_scale x (r_i + r_j)
```

Default `--cov-scale 1.0`. Post-placement relaxation uses L-BFGS to minimize
a harmonic penalty energy until all violations are resolved (or
`--relax-cycles` is exhausted).

## Disorder Metrics

All 13 metrics are computed for every structure and embedded in the XYZ
comment line. All are usable in `--filter`.

| Metric | Description | Range |
|---|---|---|
| `H_atom` | Shannon entropy of element composition | >= 0 |
| `H_spatial` | Shannon entropy of pairwise-distance histogram | >= 0 |
| `H_total` | `w_atom * H_atom + w_spatial * H_spatial` | >= 0 |
| `RDF_dev` | RMS deviation of empirical g(r) from ideal-gas baseline | >= 0 |
| `shape_aniso` | Relative shape anisotropy from gyration tensor | [0, 1] |
| `Q4`, `Q6`, `Q8` | Steinhardt bond-order parameters | [0, 1] |
| `graph_lcc` | Largest connected-component fraction at `cutoff` | [0, 1] |
| `graph_cc` | Mean clustering coefficient at `cutoff` | [0, 1] |
| `ring_fraction` | Fraction of atoms in at least one cycle in the cutoff-adjacency graph | [0, 1] |
| `charge_frustration` | Variance of |delta-chi| across cutoff-adjacent pairs | >= 0 |
| `moran_I_chi` | Moran's I spatial autocorrelation for Pauling electronegativity | unbounded |

### Unified cutoff

Five metrics share a single adjacency definition: a pair (i, j) is
"adjacent" when `d_ij <= cutoff`. These are `graph_lcc`, `graph_cc`,
`ring_fraction`, `charge_frustration`, and `moran_I_chi`. Using a unified
cutoff prevents the zero-value pathology that occurs when a covalent-radius
threshold is used for bond detection in relaxed structures
(`relax_positions` guarantees `d_ij >= cov_scale * (r_i + r_j)`).

The auto cutoff is printed to stderr:

```
[cutoff] 2.130 Ang (auto: cov_scale=1.0 x 1.5 x median(r_i+r_j)=1.420 Ang)
```

Override with `--cutoff FLOAT` when needed.

### Moran's I interpretation

`moran_I_chi` measures how randomly Pauling electronegativity is distributed
in space:

| Value | Meaning |
|---|---|
| I near 0 | Random spatial arrangement — the target for disordered structures |
| I > 0 | Atoms of similar electronegativity cluster spatially (phase separation) |
| I < 0 | Alternating high/low electronegativity (NaCl-like ionic order) |

Note: Moran's I is not bounded to [-1, 1] for sparse weight matrices.

### ring_fraction and charge_frustration

`ring_fraction` counts the fraction of atoms that belong to at least one
cycle in the cutoff-adjacency graph (detected via Union-Find spanning tree).
`charge_frustration` measures the variance of |delta-chi| across all
adjacent pairs — high values indicate strongly heterogeneous electrostatic
environments.

## Filtering

```
--filter METRIC:MIN:MAX
```

Use `-` for an open bound. Multiple flags are ANDed together.

```bash
--filter "H_total:2.0:-"          # H_total >= 2.0
--filter "Q6:-:0.3"               # Q6 <= 0.3
--filter "shape_aniso:0.5:-"      # rod-like structures
--filter "graph_lcc:0.8:-"        # well-connected graph
--filter "moran_I_chi:-0.1:0.1"   # spatially random electronegativity
```

## Output Format

```
12
sample=3 mode=chain charge=+0 mult=1 comp=[C:4,N:5,O:3]  H_atom=1.0986 ... moran_I_chi=-0.0312
C       1.234567    -0.987654     2.345678
N      -1.456789     3.210987    -0.123456
...
```

```bash
pasted ... -o out.xyz          # XYZ to file, progress to terminal
pasted ... 2>/dev/null | tool  # pipe XYZ, discard progress
pasted ... -o /dev/null        # progress only (check filter hit rate)
```

## Python API

### Functional API

```python
from pasted import generate

structures = generate(
    n_atoms=12, charge=0, mult=1,
    mode="gas", region="sphere:9",
    elements="1-30", n_samples=50, seed=42,
    filters=["H_total:2.0:-"],
)
for s in structures:
    print(s)           # Structure(n=14, comp='C2H8N2O2', mode='gas', H_total=2.341)
    print(s.to_xyz())
```

### Class API with n_success

```python
from pasted import StructureGenerator

gen = StructureGenerator(
    n_atoms=15, charge=0, mult=1,
    mode="gas", region="sphere:8",
    elements="1-30",
    n_success=10,   # stop when 10 structures pass
    n_samples=500,  # give up after 500 attempts
    filters=["H_total:2.0:-"],
    seed=42,
)
structures = gen.generate()
```

### Streaming output

```python
for s in gen.stream():
    s.write_xyz("out.xyz")   # written immediately on each PASS
```

### Structure attributes

```python
s = structures[0]
s.atoms        # ['C', 'N', 'H', ...]
s.positions    # [(x, y, z), ...]
s.comp         # 'C4N3O3'  — alphabetically-sorted composition string (new in 0.3.1)
s.metrics      # {'H_atom': 1.09, 'moran_I_chi': -0.03, ...}
s.charge       # 0
s.mult         # 1
s.mode         # 'gas'
s.sample_index # 1
len(s)         # 12
```

### Structure optimizer

```python
from pasted import StructureOptimizer

opt = StructureOptimizer(
    n_atoms=50, charge=0, mult=1,
    objective={"H_total": 1.0, "Q6": -2.0},
    elements="24,25,26,27,28",   # Cantor alloy
    method="annealing",
    max_steps=5000,
    lcc_threshold=0.8,
    seed=42,
)
result = opt.run()
best = result.best        # highest-scoring Structure
print(best)               # Structure(n=50, comp='Cr11Fe13Mn10Co8Ni8', ...)
print(result.summary())   # restarts=1  best_f=…  method='annealing'
```

### Composition-only optimization (fix coordinates, vary elements)

Pass an initial structure and set `allow_displacements=False` to optimize
element types while keeping atomic coordinates fixed.  The element pool
does not need to overlap with the initial composition — any foreign atoms
are replaced by parity-compatible pool elements before the MC loop begins.
This sanitization applies to all three methods (`"annealing"`,
`"basin_hopping"`, and `"parallel_tempering"`).

```python
from pasted import StructureOptimizer, generate

# Generate a fixed geometry from any element set
initial = generate(
    n_atoms=10, charge=0, mult=1,
    mode="gas", region="sphere:8",
    elements="6,7,8",          # C / N / O starting geometry
    n_samples=50, seed=5,
)[0]

# Optimize element types on that geometry using a different pool
opt = StructureOptimizer(
    n_atoms=len(initial),
    charge=initial.charge,
    mult=initial.mult,
    elements=["Cr", "Mn", "Fe", "Co", "Ni"],   # Cantor alloy pool
    objective={"H_atom": 1.0, "Q6": -2.0},
    allow_displacements=False,   # coordinates fixed
    method="annealing",          # or "basin_hopping" / "parallel_tempering"
    max_steps=5000,
    seed=42,
)
result = opt.run(initial=initial)
print(result.best.comp)          # e.g. 'CoCr3Fe3MnNi2'
```

### Accessing metrics

```python
s = result.best           # or: s = structures[0]
print(s.metrics["H_total"])
print(s.metrics["moran_I_chi"])
print(s.metrics["ring_fraction"])
print(s.comp)             # alphabetically-sorted composition string, e.g. 'C5N2O3'
```

## Full Option Reference

```
required:
  --n-atoms N           number of atoms per structure
  --charge INT          total system charge
  --mult INT            spin multiplicity 2S+1

placement mode:
  --mode {gas,chain,shell,maxent}
  --region SPEC         [gas/maxent] sphere:R | box:L | box:LX,LY,LZ
  --branch-prob FLOAT   [chain] branching probability (default: 0.3)
  --chain-persist FLOAT [chain] directional persistence 0.0-1.0 (default: 0.5)
  --chain-bias FLOAT    [chain] global axis drift (default: 0.0)
  --bond-range LO:HI    [chain/shell] bond length range Ang (default: 1.2:1.6)
  --center-z Z          [shell] fix center atom by atomic number
  --coord-range MIN:MAX [shell] coordination number range (default: 4:8)
  --shell-radius LO:HI  [shell] shell radius range Ang (default: 1.8:2.5)
  --maxent-steps N      [maxent] gradient-descent iterations (default: 300)
  --maxent-lr LR        [maxent] learning rate (default: 0.05)
  --maxent-cutoff-scale S [maxent] neighbor cutoff scale (default: 2.5)

elements:
  --elements SPEC       atomic-number spec (default: all Z=1-106)

physical constraints:
  --cov-scale FLOAT     d_min = cov_scale x (r_i + r_j) (default: 1.0)
  --relax-cycles INT    max L-BFGS iterations for repulsion relaxation (default: 1500)
  --no-add-hydrogen     disable automatic H augmentation

sampling:
  --n-samples INT       number of structures to attempt (default: 1)
  --n-success INT       stop after this many passing structures
  --seed INT            random seed

metrics:
  --n-bins INT          histogram bins for H_spatial and RDF_dev (default: 20)
  --w-atom FLOAT        H_atom weight in H_total (default: 0.5)
  --w-spatial FLOAT     H_spatial weight in H_total (default: 0.5)
  --cutoff FLOAT        unified adjacency cutoff Ang for all five cutoff-based
                        metrics (default: auto = cov_scale x 1.5 x median(r_i+r_j))

filtering:
  --filter METRIC:MIN:MAX   repeatable; use - for open bound

output:
  --validate            check charge/mult against one random composition, then exit
  -o / --output FILE    XYZ output file (default: stdout)
  --verbose             print per-sample metrics to stderr
```

## Notes and Limitations

- **Repulsion relaxation** uses L-BFGS (harmonic penalty energy, convergence
  criterion E < 1e-12). If `[warn] relax_positions did not converge` appears,
  the structure may contain marginal distance violations. Increase
  `--relax-cycles`.
- **Unified cutoff**: the five cutoff-based metrics all use the same `cutoff`
  parameter. Ring detection and charge frustration are computed on the
  cutoff-adjacency graph, not the covalent-radius bond graph, so they yield
  informative non-zero values in relaxed structures.
- **Moran's I range**: not bounded to [-1, 1] for sparse weight matrices. Use
  it as a relative indicator.
- **Pyykkoe radii**: for Z > 86 (Fr through Sg), same-group proxies are used.
- **Noble gas EN**: He/Ne/Ar/Rn = 4.0; Kr = 3.0; Xe = 2.6 (literature
  estimates from Allen/Allred-Rochow scale).

## License

MIT License. See [LICENSE](LICENSE).
