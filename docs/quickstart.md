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
from pasted._ext import HAS_RELAX, HAS_MAXENT, HAS_MAXENT_LOOP, HAS_STEINHARDT, HAS_GRAPH
print(HAS_RELAX, HAS_MAXENT, HAS_MAXENT_LOOP, HAS_STEINHARDT, HAS_GRAPH)
# True True True True True  (all extensions compiled)
```

| Flag | What it enables |
|---|---|
| `HAS_RELAX` | C++ L-BFGS steric-clash relaxation (~20–50× vs Python) |
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

### `compute_all_metrics` latency by atom count (all C++ extensions active)

Measured on a single CPU core with all five C++ extensions compiled
(`HAS_GRAPH = HAS_STEINHARDT = … = True`), element pool C/N/O/S/P,
`mode="gas"`, uniform random positions scaled so density is constant across N.
Each value is the median of ≥ 3 repeats filling a 1.5 s budget.

| N atoms | `shape_aniso` | `validate` ok=True | `rdf_h_cpp` | `graph_cpp` | `steinhardt` (v0.3.6) | **`all_metrics` total** |
|--------:|------:|------:|------:|------:|------:|------:|
| 5 | 18 µs | 0.6 µs | 0.002 ms | 0.004 ms | 0.020 ms | **0.33 ms** |
| 10 | 17 µs | 0.7 µs | 0.002 ms | 0.004 ms | 0.020 ms | **0.31 ms** |
| 20 | 17 µs | 1.1 µs | 0.002 ms | 0.004 ms | 0.020 ms | **0.32 ms** |
| 50 | 19 µs | 2.2 µs | 0.011 ms | 0.020 ms | 0.076 ms | **0.37 ms** |
| 100 | 21 µs | 4.1 µs | 0.011 ms | 0.020 ms | 0.076 ms | **0.26 ms** |
| 200 | 23 µs | 6.9 µs | 0.053 ms | 0.099 ms | 0.180 ms | **0.55 ms** |
| 500 | 32 µs | 16.7 µs | 0.053 ms | 0.099 ms | 0.306 ms | **0.92 ms** |
| 1 000 | 63 µs | 32.8 µs | 0.115 ms | 0.287 ms | 0.654 ms | **1.68 ms** |
| 2 000 | 80 µs | 63.7 µs | 0.264 ms | 0.576 ms | 2.307 ms | **3.99 ms** |
| 5 000 | 184 µs | 177 µs | 0.678 ms | 1.524 ms | 5.605 ms | **9.72 ms** |

Peak RSS across the full N = 5–5 000 sweep: **152 MB** (growth < 3 MB total;
no memory leak detected over 500 repeated calls at each size).

**`compute_all_metrics` latency is roughly linear in N** (k ≈ 0.7 at the
default cutoff, so each sub-metric is O(N·k)).  Prior to v0.3.6 the
`compute_steinhardt` step showed superlinear growth due to a CPU cache
pressure effect in the C++ accumulator buffer (former layout `(n_l, l_max+1,
N)` — atom index innermost, stride N×8 B per m-step, causing L2→L3 spill at
N ≈ 1 000).  The buffer has been transposed to `(N, n_l, l_max+1)` in
v0.3.6, making every bond's writes contiguous (stride 8 B).  See
`docs/architecture.md` → *Accumulator buffer layout* for the full analysis.

---

## Python API

### Functional API (`generate`)

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
    print(s)           # Structure(n=14, comp='C2H2N2O8', mode='gas', H_total=2.341)
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

> **Note — `summary()` labels vs attribute names:** the one-line string
> returned by `result.summary()` uses short labels (`passed`, `attempted`,
> `rejected_parity`, `rejected_filter`).  The corresponding Python attributes
> carry an `n_` prefix: `result.n_passed`, `result.n_attempted`,
> `result.n_rejected_parity`, `result.n_rejected_filter`.
> Accessing `result.passed` or `result.attempted` directly raises
> `AttributeError`.

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
        elements="6",       # carbon-only pool: parity always passes, so the
        n_samples=10, seed=0,  # filter warning fires cleanly
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

> **Note:** `chain` and `shell` **ignore** the `region` argument — they
> use `bond_range` / `shell_radius` to control geometry and silently discard
> any `region` value that is passed.  Only `gas` and `maxent` require `region`.

### Class API (`StructureGenerator`)

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
for i, s in enumerate(result):
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

### The `comp` property

`Structure.comp` returns a compact composition label derived from the atom
list.  Elements are sorted in **alphabetical order** (Python `sorted()` on
symbol strings) and counts above one are appended as a suffix:

```python
s.comp          # e.g. 'C5N2O3'
repr(s)         # "Structure(n=10, comp='C5N2O3', mode='gas', H_total=2.031)"
```

> **Sort order note:** `comp` uses **alphabetical** order, not Hill order
> (which would put C first, H second, then all others alphabetically).  For
> pools of only C, H, N, O the two orderings are identical, but other elements
> appear at their alphabetical position.  For example:
>
> | atoms | `comp` | Hill order would give |
> |---|---|---|
> | `['C','H','H','O']` | `'CH2O'` | `'CH2O'` (same) |
> | `['Ar','C','H','H']` | `'ArCH2'` | `'CH2Ar'` (different) |
> | `['Na','C','H','H']` | `'CH2Na'` | `'CH2Na'` (same here) |

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
# Keep structures with H_total >= 1.0 and Q6 <= 0.3
pasted --n-atoms 15 --charge 0 --mult 1 \
       --mode gas --region sphere:8 \
       --elements 6,7,8,16 --n-samples 500 --seed 1 \
       --filter "H_total:1.0:-" \
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

> **Atom count in shell mode:** the number of atoms written to the output file
> may be larger than `--n-atoms` for two reasons.  First, when `--center-z Z`
> is set the center atom is prepended to the coordinate block, adding one atom
> regardless of `n_atoms`.  Second, `add_hydrogen` is enabled by default and
> appends hydrogen atoms to satisfy open valences.  The `n_atoms` parameter
> controls only the number of *shell* atoms placed around the center.  Use
> `--no-add-hydrogen` to suppress hydrogen addition, or inspect
> `Structure.center_sym` to identify the center atom in the output.

### Maximum-entropy mode (`maxent`)

`maxent` places atoms so that their angular distribution around each center is
as uniform as possible.  Like `gas`, it requires a `region` spec:

```bash
pasted --n-atoms 12 --charge 0 --mult 1 \
       --mode maxent --region sphere:6 \
       --elements 6,7,8 --n-samples 50 --seed 42 \
       -o maxent.xyz
```

> **Tip:** `maxent` is slower than `gas` or `chain` because it runs an
> iterative angular-repulsion optimisation per structure.  For large N, the
> C++ extension (`HAS_MAXENT_LOOP=True`) is strongly recommended.

For a complete reference of all CLI options, flags, and optimizer mode, see
[CLI reference](cli.md).

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

### Extended objective with EvalContext: full structure and optimizer state

For objectives that need atomic coordinates, charge/multiplicity, or
optimizer runtime information, use the **2-argument calling convention**.
Any callable with two required positional parameters receives an
[`EvalContext`](api/optimizer.rst) as its second argument:

```python
from pasted import EvalContext, StructureOptimizer
```

The `EvalContext` exposes:

| Field | Type | Description |
|---|---|---|
| `ctx.atoms` | `tuple[str, ...]` | Element symbols |
| `ctx.positions` | `tuple[tuple[float,float,float], ...]` | Coordinates (Å) |
| `ctx.charge` / `ctx.mult` | `int` | Charge and multiplicity |
| `ctx.n_atoms` | `int` | Atom count |
| `ctx.metrics` | `dict[str, float]` | Same as `m` |
| `ctx.to_xyz()` | `str` | XYZ-format string |
| `ctx.step` / `ctx.max_steps` | `int` | MC step index and total |
| `ctx.progress` | `float` | `step / max_steps` ∈ [0, 1) |
| `ctx.temperature` | `float` | Current temperature |
| `ctx.f_current` / `ctx.best_f` | `float` | Current and best scores |
| `ctx.per_atom_q6` | `np.ndarray` | Per-atom Q6, shape `[n_atoms]` |
| `ctx.restart_idx` | `int` | Restart index (0-based) |
| `ctx.element_pool` | `tuple[str, ...]` | Available elements |
| `ctx.cutoff` | `float` | Distance cutoff (Å) |
| `ctx.replica_idx` / `ctx.n_replicas` | `int \| None` | PT-only fields |

**Example 1 — NumPy geometry objective (no external dependencies)**

```python
import numpy as np
from pasted import StructureOptimizer

# 1-arg form: unchanged from before
opt1 = StructureOptimizer(
    n_atoms=10, charge=0, mult=1, elements="6,7,8",
    objective=lambda m: m["H_total"] - 2.0 * m["Q6"],
)

# 2-arg form: use ctx.positions for a purely geometric objective
def rms_spread_objective(m, ctx):
    """Maximize mean pairwise distance."""
    pos = np.array(ctx.positions)
    diffs = pos[:, None, :] - pos[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1))
    return float(dists[np.triu_indices(ctx.n_atoms, k=1)].mean())

opt2 = StructureOptimizer(
    n_atoms=10, charge=0, mult=1, elements="6,7,8",
    objective=rms_spread_objective,
    method="annealing", max_steps=2000, seed=42,
)
result = opt2.run()
```

**Example 2 — Adaptive curriculum objective**

```python
def curriculum_objective(m, ctx):
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
```

**Example 3 — Per-atom Q6 locality penalty**

```python
import numpy as np

def local_disorder_objective(m, ctx):
    """Reward Q6 variance; penalize the most locally ordered atom."""
    q6_var = float(np.var(ctx.per_atom_q6))
    q6_max = float(np.max(ctx.per_atom_q6))
    return m["H_total"] + q6_var * 0.5 - q6_max * 1.0

opt = StructureOptimizer(
    n_atoms=20, charge=0, mult=1, elements="6,7,8,15,16",
    objective=local_disorder_objective,
    method="basin_hopping", max_steps=3000, seed=21,
)
```

**Example 4 — xTB single-point energy as objective**

> **External dependency:** `xtb` — GFN-xTB semiempirical tight-binding code (standalone binary).
> Install: `conda install -c conda-forge xtb` (recommended) or `pip install xtb-python`
> Reference: [xtb documentation](https://xtb-docs.readthedocs.io/) | [GitHub](https://github.com/grimme-lab/xtb) | DOI: `10.1021/acs.jctc.8b01176`

```python
import os, subprocess, tempfile

def xtb_energy_objective(m, ctx):
    """Minimize GFN2-xTB single-point energy (negated for maximization).

    Uses ctx.to_xyz() to serialize the structure and ctx.charge / ctx.mult
    for the --chrg / --uhf flags.  Requires the xtb binary on PATH.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".xyz", mode="w", delete=False
    ) as fh:
        fh.write(ctx.to_xyz())
        fname = fh.name
    try:
        proc = subprocess.run(
            ["xtb", fname, "--sp", "--gfn", "2",
             "--chrg", str(ctx.charge),
             "--uhf",  str(ctx.mult - 1)],
            capture_output=True, text=True, timeout=60,
        )
        for line in proc.stdout.splitlines():
            if "TOTAL ENERGY" in line:
                return -float(line.split()[3])   # negate → maximize
        return float("-inf")
    finally:
        os.unlink(fname)

opt = StructureOptimizer(
    n_atoms=8, charge=0, mult=1, elements="6,7,8",
    objective=xtb_energy_objective,
    method="basin_hopping", max_steps=100, seed=7,
)
```

**Example 5 — ASE EMT potential as objective**

> **External dependency:** `ase` — Atomic Simulation Environment.
> Install: `pip install ase`
> Reference: [ASE documentation](https://wiki.fysik.dtu.dk/ase/) | [PyPI](https://pypi.org/project/ase/) | DOI: `10.1088/1361-648X/aa680e`
>
> Note: The EMT calculator below supports only Al, Cu, Ag, Au, Ni, Pd, Pt, H, C, N, O.
> Replace with any ASE-compatible calculator (GPAW, NequIP, MACE, etc.) as needed.

```python
from ase import Atoms
from ase.calculators.emt import EMT

def ase_emt_objective(m, ctx):
    """Use ASE/EMT energy as objective (negated for maximization)."""
    structure = Atoms(
        symbols=list(ctx.atoms),
        positions=list(ctx.positions),
    )
    structure.calc = EMT()
    return -structure.get_potential_energy()   # negate → maximize

opt = StructureOptimizer(
    n_atoms=12, charge=0, mult=1,
    elements="29,79",   # Cu, Au — both supported by EMT
    objective=ase_emt_objective,
    method="annealing", max_steps=500, seed=42,
)
```

---

### Controlling initial-structure generation retries (`max_init_attempts`)

By default `StructureOptimizer` retries generating the initial structure for
each restart **without a limit** (`max_init_attempts=0`).  This is safe
because the constructor validates at build time that the element pool can
satisfy the charge/multiplicity parity constraint — if construction succeeds,
a valid structure is guaranteed to eventually be found.

Pass a positive integer to cap the number of attempts per restart.  If the
cap is reached without success the restart is skipped and a
`UserWarning` is emitted.  This is useful in automated pipelines where a
wall-time budget matters more than exhaustive search:

```python
from pasted import StructureOptimizer

# At most 100 tries per restart — useful in time-constrained pipelines
opt = StructureOptimizer(
    n_atoms=20,
    charge=0,
    mult=1,
    elements="6,7,8,15,16",
    objective={"H_total": 1.0},
    method="annealing",
    max_steps=3000,
    n_restarts=4,
    max_init_attempts=100,   # cap retries; 0 (default) = unlimited
    seed=42,
)
result = opt.run()
```

Passing an element pool that can *never* satisfy the parity constraint raises
`ValueError` immediately at construction time — no retries are attempted:

```python
# Raises ValueError: all-nitrogen pool cannot satisfy charge=0, mult=1
# N has Z=7 (odd). With an all-odd-Z pool, sum(Z) parity == n_atoms % 2.
# mult=1 requires an even number of electrons, i.e. sum(Z) must be even,
# so n_atoms must be even — but here n_atoms=7 (odd) makes it impossible.
StructureOptimizer(
    n_atoms=7, charge=0, mult=1,
    elements="7",          # nitrogen only — all-odd-Z pool
    objective={"H_total": 1.0},
)
```

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

Setting `affine_strength > 0` applies a random affine transformation —
stretch/compress along one axis, shear one axis pair, and optionally a
small per-atom jitter — to each generated structure **before**
`relax_positions`.  The center of mass is pinned throughout, so the
structure stays centered.  The transform creates more anisotropic initial
geometries while still guaranteeing clash-free output after relaxation.

Three sub-operations can be controlled independently via
`affine_stretch`, `affine_shear`, and `affine_jitter`.  When any of
these is `None` (the default), it falls back to `affine_strength`.
Setting one of them to `0.0` disables that operation entirely regardless
of `affine_strength`.

### Basic usage

```python
from pasted import generate

# Global control: all three operations use the same strength
result = generate(
    n_atoms=20, charge=0, mult=1,
    mode="gas", region="sphere:10",
    elements="6,7,8", n_samples=50, seed=42,
    affine_strength=0.2,   # ±20 % stretch, ±10 % shear, jitter ∝ 0.2
)
```

### Per-operation control

```python
from pasted import generate

# Strong axial elongation only — no shear, no per-atom noise
result_stretch = generate(
    n_atoms=20, charge=0, mult=1,
    mode="chain", elements="6,7,8",
    n_samples=50, seed=0,
    affine_strength=0.2,
    affine_stretch=0.4,   # large stretch/compress (±40 %)
    affine_shear=0.0,     # shear disabled
    affine_jitter=0.0,    # per-atom jitter disabled
)

# Pure shear distortion — no axial scaling
result_shear = generate(
    n_atoms=20, charge=0, mult=1,
    mode="gas", region="sphere:10",
    elements="6,7,8", n_samples=50, seed=0,
    affine_strength=0.2,
    affine_stretch=0.0,   # stretch disabled
    affine_shear=0.3,     # shear only
    affine_jitter=0.0,
)

# Per-atom jitter only — break symmetry without global distortion
result_jitter = generate(
    n_atoms=20, charge=0, mult=1,
    mode="gas", region="sphere:10",
    elements="6,7,8", n_samples=50, seed=0,
    affine_strength=0.1,
    affine_stretch=0.0,
    affine_shear=0.0,
    affine_jitter=0.3,    # fine-grain per-atom noise only
)
```

### Parameter reference

| Parameter | Controls | Range | Default |
|---|---|---|---|
| `affine_strength` | Global fallback for all three operations | 0.0–0.4 | `0.0` (disabled) |
| `affine_stretch` | Scale factor along one random axis: `Uniform(1−s, 1+s)` | 0.0–0.4 | `None` (→ `affine_strength`) |
| `affine_shear` | Off-diagonal matrix element: `Uniform(-s/2, s/2)` | 0.0–0.4 | `None` (→ `affine_strength`) |
| `affine_jitter` | Per-atom translation noise ∝ `move_step × s` | 0.0–0.4 | `None` (→ `affine_strength`) |

> **Note on `affine_jitter` in StructureGenerator:** when used inside
> `generate()` or `StructureGenerator`, the internal `move_step` is `0.0`,
> so `affine_jitter` has no visible effect — the jitter term is gated on
> `move_step > 0`.  `affine_jitter` is therefore only meaningful in
> `StructureOptimizer` (where `move_step` is set per MC step).

Practical `affine_strength` guide:

| Value | Effect |
|---|---|
| `0.0` (default) | disabled — backward-compatible, no transform |
| `0.05`–`0.1` | subtle anisotropy, negligible overhead |
| `0.2`–`0.3` | noticeable elongation or shear |
| `0.4` | strong distortion; may require larger `region` to avoid clashes |

Works across **all placement modes** (`gas`, `chain`, `shell`, `maxent`).
CLI flags: `--affine-strength S`, `--affine-stretch S`, `--affine-shear S`,
`--affine-jitter S`.

```bash
# CLI: axial elongation only, no shear
pasted --n-atoms 20 --charge 0 --mult 1 \
       --mode chain --elements 6,7,8 --n-samples 50 --seed 0 \
       --affine-strength 0.2 --affine-stretch 0.4 \
       --affine-shear 0.0 --affine-jitter 0.0
```

### Affine moves in StructureOptimizer

`StructureOptimizer` supports the same four parameters through
`allow_affine_moves=True`.  When enabled, half of all displacement moves
are replaced by affine moves, allowing the optimizer to explore
elongated, compressed, or sheared configurations that fragment moves
cannot reach efficiently.

```python
from pasted import StructureOptimizer

# Strong axial stretch — useful for maximizing shape_aniso
opt = StructureOptimizer(
    n_atoms=20, charge=0, mult=1,
    elements="6,7,8",
    objective={"shape_aniso": 2.0, "H_total": 1.0},
    allow_affine_moves=True,
    affine_strength=0.2,
    affine_stretch=0.4,   # dominant stretch moves
    affine_shear=0.0,     # no shear
    affine_jitter=0.0,    # no per-atom noise
    method="annealing",
    max_steps=2000, n_restarts=2, seed=0,
)
result = opt.run()
print(f"shape_aniso = {result.best.metrics['shape_aniso']:.4f}")

# Combined stretch + shear for complex anisotropy
opt2 = StructureOptimizer(
    n_atoms=20, charge=0, mult=1,
    elements="6,7,8",
    objective={"shape_aniso": 2.0, "H_total": 1.0},
    allow_affine_moves=True,
    affine_strength=0.2,
    affine_stretch=0.25,
    affine_shear=0.15,
    affine_jitter=0.0,    # disable noise for pure geometric distortion
    method="annealing",
    max_steps=2000, n_restarts=2, seed=0,
)
result2 = opt2.run()
print(f"shape_aniso = {result2.best.metrics['shape_aniso']:.4f}")
```

> **Relationship to `StructureGenerator`:** both classes use the same
> `_affine_move` function from `pasted._placement`.  In `StructureGenerator`
> the transform is applied once per structure before `relax_positions`; in
> `StructureOptimizer` it is applied at each accepted MC step when
> `allow_affine_moves=True`.  The `affine_jitter` parameter only has a
> visible effect in the Optimizer (where `move_step > 0`).

---

## Element sampling control (StructureGenerator)

### Element pool specification

The `elements=` parameter (Python API) and `--elements` (CLI) control which
elements are available for random sampling.  Three forms are supported:

| Form | Example | Meaning |
|---|---|---|
| Atomic-number spec string | `"6,7,8"` | C, N, O (by atomic number Z) |
| Atomic-number range string | `"1-30"` | H through Zn |
| Combined ranges + singles | `"1-10,26,28"` | H–Ne plus Fe and Ni |
| List of element symbols | `["C", "N", "O"]` | explicit symbol list |
| `None` (omitted) | — | all Z = 1–106 |

> **Important:** when passing a *string*, it must contain **atomic numbers
> (integers)**, not element symbols.  `elements="C,N,O"` raises `ValueError`;
> use `elements="6,7,8"` or `elements=["C", "N", "O"]` instead.

> **Note (v0.3.5):** `parse_element_spec()` now accepts a `list[str]` of
> element symbols directly (e.g. `["C", "N", "O"]`).  Previously, calling
> `parse_element_spec(["C", "N", "O"])` raised `AttributeError` because
> the function attempted to call `.split(",")` on the list.  The fix has
> no effect on the `StructureGenerator` / `generate()` keyword API, which
> already handled symbol lists correctly via an internal branch.

```python
# Correct — numeric atomic-number string
gen = StructureGenerator(n_atoms=10, charge=0, mult=1, mode="chain",
                         elements="6,7,8")

# Correct — explicit symbol list (passed to StructureGenerator or parse_element_spec)
gen = StructureGenerator(n_atoms=10, charge=0, mult=1, mode="chain",
                         elements=["C", "N", "O"])

from pasted._atoms import parse_element_spec
assert parse_element_spec(["C", "N", "O"]) == ["C", "N", "O"]  # now works

# WRONG — symbol string raises ValueError
# gen = StructureGenerator(..., elements="C,N,O")  # ← ValueError!
```

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

Weights are *relative* — they are normalized internally.
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

## Position-only optimization (StructureOptimizer)

Set `allow_composition_moves=False` to fix the composition and only
optimize atomic positions.  This is useful when the stoichiometry is
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

## Composition-only optimization (StructureOptimizer)

Set `allow_displacements=False` to fix the atomic coordinates and only
optimize element types.  This is useful when exploring compositional
disorder on a pre-relaxed geometry (e.g. a fixed lattice).

The element pool does not need to match the initial structure's composition.
Any atoms outside the pool are automatically replaced by parity-compatible
pool elements before the MC loop begins, so cross-pool optimization works
reliably with all three methods (`"annealing"`, `"basin_hopping"`, and
`"parallel_tempering"`):

```python
from pasted import StructureOptimizer, generate

# Starting geometry can use any element set — the optimizer will sanitize
# foreign atoms to the target pool before the first MC step.
initial = generate(
    n_atoms=10, charge=0, mult=1,
    mode="gas", region="sphere:8",
    elements="6,7,8",          # C / N / O geometry
    n_samples=50, seed=0,
)[0]

opt = StructureOptimizer(
    n_atoms=len(initial),
    charge=initial.charge,
    mult=initial.mult,
    elements=["Cr", "Mn", "Fe", "Co", "Ni"],  # Cantor alloy pool
    objective={
        "moran_I_chi": -1.0,          # minimize EN spatial autocorrelation
        "charge_frustration": 2.0,    # maximize EN variance across neighbors
    },
    allow_displacements=False,   # composition-only; coordinates fixed
    method="annealing",
    max_steps=5000,
    seed=42,
)

result = opt.run(initial=initial)
print(result.best.comp)   # e.g. 'CoCr3Fe3MnNi2'
# Positions are identical to initial; only element labels have changed
import numpy as np
np.testing.assert_allclose(
    np.array(result.best.positions), np.array(initial.positions)
)
```

> **Choosing an objective for composition-only runs:** each composition move
> selects a random atom and replaces it with a different element drawn from
> the pool, which changes the element-count histogram.  However, metrics
> that capture the *spatial arrangement* of elements — `moran_I_chi`
> (electronegativity autocorrelation) and `charge_frustration` (EN variance
> across neighbor pairs) — respond most sensitively to which element sits
> where and are therefore the recommended choices for composition-only
> optimization.  Pure composition-count metrics such as `H_atom` tend to
> plateau quickly once the pool coverage is broad.

> **Note**: `allow_displacements=False` and `allow_composition_moves=False`
> cannot both be set — at least one move type must be enabled.  Attempting
> to do so raises a `ValueError`.

---

## Optimizer case studies

### Case study 1 — Reproducing a target disorder profile

This example shows how to drive `StructureOptimizer` toward a reference
structure defined by a known set of disorder metrics.  The target is a
dense 40-atom C/N sphere with high Moran's I spatial autocorrelation
(`moran_I_chi ≈ 0.94`), a fully connected graph (`graph_lcc = 1.0`), and
a high ring fraction (`ring_fraction = 0.875`).  The `cutoff` parameter is
set to match the value used when the reference metrics were computed.

```python
from pasted import StructureOptimizer

# Target metrics (from moran1_dense_sphere.xyz, computed at cutoff=5.0 Å):
#   moran_I_chi   = 0.9446   graph_lcc     = 1.0000
#   ring_fraction = 0.8750   charge_frustration = 0.0058
#   H_spatial     = 3.4420   Q6            = 0.3335

opt = StructureOptimizer(
    n_atoms=40,
    charge=0,
    mult=1,
    elements=["C", "N"],
    objective={
        "moran_I_chi":        3.0,   # primary target: high spatial EN autocorrelation
        "ring_fraction":      2.0,   # dense ring connectivity
        "graph_lcc":          2.0,   # fully connected graph
        "H_spatial":          1.0,   # spatial disorder
        "charge_frustration": -5.0,  # suppress (target ≈ 0)
    },
    cutoff=5.0,                      # match the reference metric cutoff
    method="parallel_tempering",
    n_replicas=4,
    pt_swap_interval=10,
    max_steps=3000,
    n_restarts=3,
    T_start=1.0,
    T_end=0.01,
    seed=42,
)

result = opt.run()
best = result.best
print(result.summary())
print(f"moran_I_chi   = {best.metrics['moran_I_chi']:.4f}  (target 0.9446)")
print(f"ring_fraction = {best.metrics['ring_fraction']:.4f}  (target 0.8750)")
print(f"graph_lcc     = {best.metrics['graph_lcc']:.4f}  (target 1.0000)")
best.write_xyz("similar_to_target.xyz")
```

**Tips for target-matching runs:**

- Always pass `cutoff=` explicitly so that PASTED uses the same distance
  threshold as was used to compute the reference metrics.
- Use `n_restarts ≥ 3` with `method="parallel_tempering"` — the landscape
  is rugged and hot replicas help escape local optima.
- Negative weights (`charge_frustration: -5.0`) suppress metrics that
  should stay near zero; large magnitude is important to counteract the
  positive objectives.

---

### Case study 2 — Geometry search with an ASE calculator

`EvalContext` lets you call any ASE-compatible calculator inside the
objective function, connecting PASTED's stochastic search to the entire
ASE ecosystem (xTB, MACE, NequIP, GPAW, …).  This example uses the
built-in EMT potential to search for a low-energy methane-like (CH₄)
geometry.

> **External dependency:** `ase` — Atomic Simulation Environment.
> Install: `pip install ase`
>
> **Note:** the EMT calculator supports only Al, Cu, Ag, Au, Ni, Pd, Pt,
> H, C, N, O.  Replace with any ASE-compatible calculator as needed.

```python
from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from pasted import EvalContext, StructureGenerator, StructureOptimizer

# ── Step 1: generate C₁H₄ initial geometries with PASTED ────────────────
initial_structs = StructureGenerator(
    n_atoms=5,
    charge=0,
    mult=1,
    mode="gas",
    region="sphere:3",
    elements=["C", "H"],
    element_min_counts={"C": 1},
    element_max_counts={"C": 1},
    n_samples=100,
    seed=0,
).generate()

initial = initial_structs[0]   # pick the first C₁H₄ structure

# ── Step 2: basin-hopping with ASE/EMT objective ─────────────────────────
def ase_emt_objective(m: dict, ctx: EvalContext) -> float:
    """Maximize stability — lower EMT potential energy is better."""
    structure = Atoms(
        symbols=list(ctx.atoms),
        positions=list(ctx.positions),
    )
    structure.calc = EMT()
    try:
        return -structure.get_potential_energy()   # negate: maximize = minimize energy
    except Exception:
        return float("-inf")

opt = StructureOptimizer(
    n_atoms=len(initial),
    charge=initial.charge,
    mult=initial.mult,
    elements=list(set(initial.atoms)),
    objective=ase_emt_objective,
    allow_composition_moves=False,   # fix C₁H₄ stoichiometry
    method="basin_hopping",
    max_steps=500,
    n_restarts=3,
    seed=7,
)
result = opt.run(initial=initial)
best = result.best
print(result.summary())

# ── Step 3: final geometry relaxation with ASE BFGS ──────────────────────
ase_mol = Atoms(
    symbols=list(best.atoms),
    positions=list(best.positions),
)
ase_mol.calc = EMT()
bfgs = BFGS(ase_mol, logfile=None)
bfgs.run(fmax=0.01)

print(f"EMT energy after BFGS: {ase_mol.get_potential_energy():.4f} eV")
for sym, pos in zip(ase_mol.get_chemical_symbols(), ase_mol.positions):
    print(f"  {sym:2s}  {pos[0]:8.4f}  {pos[1]:8.4f}  {pos[2]:8.4f}")
```

**Workflow summary:**

1. `StructureGenerator` samples random clash-free starting geometries
   with the required stoichiometry.
2. `StructureOptimizer` explores configuration space using PASTED's MC
   moves, calling the ASE calculator at each accepted step.
3. The best PASTED geometry is passed to ASE `BFGS` for a final
   gradient-based refinement.

This three-stage pipeline (sample → stochastic search → local minimize)
is a general pattern for any external potential.

---

### Case study 3 — Two-phase curriculum objective

`EvalContext.progress` returns the fractional progress of the current
restart (range `[0.0, 1.0)`), enabling objectives that change behavior
over the course of a run.  This is useful for annealing-style curricula
where early exploration and late refinement require different signals.

```python
from pasted import EvalContext, StructureOptimizer

def curriculum_objective(m: dict, ctx: EvalContext) -> float:
    """
    Two-phase curriculum driven by optimizer progress:

    Phase 1 (first 50% of steps):
        Maximize spatial disorder only — broad exploration of
        configuration space without penalizing ordered regions.

    Phase 2 (last 50% of steps):
        Continue maximizing spatial disorder AND actively suppress
        crystalline order (Q6).  Switching late avoids trapping
        the search in disordered-but-ordered local minima early on.
    """
    if ctx.progress < 0.5:
        return m["H_spatial"] * 2.0
    else:
        return m["H_spatial"] * 2.0 - m["Q6"] * 3.0

opt = StructureOptimizer(
    n_atoms=20,
    charge=0,
    mult=1,
    elements="6,7,8,15,16",          # C, N, O, P, S
    objective=curriculum_objective,
    method="annealing",
    max_steps=2000,
    n_restarts=4,
    seed=99,
)

result = opt.run()
best = result.best
print(result.summary())
print(f"H_spatial = {best.metrics['H_spatial']:.4f}")
print(f"Q6        = {best.metrics['Q6']:.4f}")
```

**When to use curriculum objectives:**

- When the search landscape has a conflict between early exploration and
  late refinement (e.g., maximizing disorder while also minimizing a
  specific metric).
- When you want to prevent the optimizer from committing too early to a
  particular region of metric space.
- Use `ctx.step` for step-based schedules and `ctx.temperature` to tie
  the curriculum to the annealing schedule directly.
