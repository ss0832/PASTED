# CLI reference

PASTED is invoked as `pasted` from the command line.

```
pasted [OPTIONS]
```

## Core options

| Option | Type | Default | Description |
|---|---|---|---|
| `--n-atoms N` | int | — | Number of non-hydrogen atoms per structure (**required**) |
| `--charge C` | int | — | Total system charge (**required**) |
| `--mult M` | int | — | Spin multiplicity 2S+1 (**required**) |
| `--elements SPEC` | str | all Z=1–106 | Element pool: range `"1-30"`, list `"6,7,8"`, or symbol list |
| `--n-samples N` | int | 1 | Maximum number of placement attempts. Use `0` for unlimited (requires `--n-success`) |
| `--n-success N` | int | None | Stop as soon as N structures pass all filters |
| `--seed S` | int | None | RNG seed for reproducibility |
| `--n-threads N` | int | 1 | Number of OpenMP threads for C++ extensions (Linux only). Default: **1** (single-threaded) since v0.2.2 — pass a higher value to enable parallelism. No effect when `HAS_OPENMP` is `False` |
| `--output FILE` / `-o` | path | stdout | Output XYZ file (append mode) |
| `--verbose` | flag | off | Print per-sample metrics to stderr |

## Element sampling control

These flags control how elements are sampled from the pool defined by
`--elements`.  They can be combined freely.

| Option | Type | Default | Description |
|---|---|---|---|
| `--element-fractions SYM:WEIGHT` | repeatable | uniform | Relative sampling weight for one element. Weights are normalised; elements not listed receive weight 1.0 |
| `--element-min-counts SYM:N` | repeatable | none | Minimum number of atoms guaranteed for one element |
| `--element-max-counts SYM:N` | repeatable | none | Maximum number of atoms allowed for one element |

### Examples

```bash
# Carbon-rich sampling from C/N/O pool
pasted --n-atoms 20 --elements 6,7,8 --charge 0 --mult 1 \
    --mode gas --region sphere:10 --n-samples 50 \
    --element-fractions C:0.6 --element-fractions N:0.3 --element-fractions O:0.1

# Guarantee at least 4 C atoms, cap N and O at 3 each
pasted --n-atoms 15 --elements 6,7,8,15,16 --charge 0 --mult 1 \
    --mode gas --region sphere:10 --n-samples 100 \
    --element-min-counts C:4 \
    --element-max-counts N:3 --element-max-counts O:3

# Combine fractions and bounds
pasted --n-atoms 12 --elements 6,7,8 --charge 0 --mult 1 \
    --mode chain --n-samples 30 \
    --element-fractions C:5 --element-fractions N:2 --element-fractions O:1 \
    --element-min-counts C:2 --element-max-counts N:4
```

## Placement mode

| Option | Description |
|---|---|
| `--mode {gas,chain,shell,maxent}` | Placement algorithm (default: `gas`). `gas` and `maxent` require `--region`; `chain` and `shell` do not. |

### gas mode

| Option | Default | Description |
|---|---|---|
| `--region SPEC` | — | `sphere:R`, `box:L`, or `box:LX,LY,LZ` (**required**) |

### chain mode

| Option | Default | Description |
|---|---|---|
| `--branch-prob P` | 0.3 | Branching probability [0, 1] |
| `--chain-persist P` | 0.5 | Directional persistence [0, 1] |
| `--chain-bias B` | 0.0 | Global-axis drift strength [0, 1]; higher values produce more elongated structures |
| `--bond-range LO:HI` | 1.2:1.6 | Bond length range (Å) |

### shell mode

| Option | Default | Description |
|---|---|---|
| `--center-z Z` | random | Atomic number of center atom |
| `--coord-range MIN:MAX` | 4:8 | Coordination number range |
| `--shell-radius LO:HI` | 1.8:2.5 | Shell radius range (Å) |
| `--bond-range LO:HI` | 1.2:1.6 | Tail bond length range (Å) |

### maxent mode

| Option | Default | Description |
|---|---|---|
| `--region SPEC` | — | Same as gas mode (**required**) |
| `--maxent-steps N` | 300 | Gradient-descent iterations |
| `--maxent-lr LR` | 0.05 | Learning rate |
| `--maxent-cutoff-scale S` | 2.5 | Neighbour cutoff scale factor |

## Metric options

| Option | Default | Description |
|---|---|---|
| `--n-bins N` | 20 | Histogram bins for `H_spatial` and `RDF_dev` |
| `--w-atom W` | 0.5 | Weight of `H_atom` in `H_total` |
| `--w-spatial W` | 0.5 | Weight of `H_spatial` in `H_total` |
| `--cutoff R` | auto | Distance cutoff (Å) for Steinhardt and graph metrics |

## Filters

```
--filter "METRIC:MIN:MAX"
```

Use `-` for an open bound. Multiple `--filter` flags are ANDed together.

```bash
--filter "H_total:2.0:-"      # H_total >= 2.0
--filter "Q6:-:0.3"           # Q6 <= 0.3
--filter "shape_aniso:0.5:-"  # shape_aniso >= 0.5
--filter "moran_I_chi:-0.1:0.1"  # |I| < 0.1 — spatially random EN
```

Available metrics: `H_atom`, `H_spatial`, `H_total`, `RDF_dev`,
`shape_aniso`, `Q4`, `Q6`, `Q8`, `graph_lcc`, `graph_cc`,
`ring_fraction`, `charge_frustration`, `moran_I_chi`.

### Collecting a fixed number of passing structures

Combine `--filter` with `--n-success` to stop as soon as enough structures
have passed, rather than running all `--n-samples` attempts:

```bash
# Collect exactly 10 structures with H_total >= 2.0, try up to 500 times
pasted --n-atoms 15 --charge 0 --mult 1 \
       --mode gas --region sphere:8 \
       --elements 1-30 \
       --filter "H_total:2.0:-" \
       --n-success 10 --n-samples 500 \
       -o out.xyz
```

Use `--n-samples 0` for unlimited attempts:

```bash
pasted --n-atoms 15 --charge 0 --mult 1 \
       --mode gas --region sphere:8 \
       --elements 1-30 \
       --filter "H_total:2.5:-" \
       --n-success 10 --n-samples 0 \
       -o out.xyz
```

Output is written to `-o` immediately each time a structure passes.
An interrupted run always produces a valid XYZ file.

## Physical constraints

| Option | Default | Description |
|---|---|---|
| `--cov-scale S` | 1.0 | Minimum-distance scale factor: `d_min = S × (r_i + r_j)` |
| `--relax-cycles N` | 1500 | Maximum repulsion-relaxation iterations |
| `--no-add-hydrogen` | flag | Disable automatic H augmentation |

## Optimizer mode

Pass `--optimize` to maximise a disorder objective instead of sampling
randomly.  All sampling-mode options (`--n-atoms`, `--charge`, `--mult`,
`--elements`, element sampling flags, `--cov-scale`, `--relax-cycles`,
`--cutoff`, `--seed`) are shared.

| Option | Default | Description |
|---|---|---|
| `--optimize` | flag | Enable optimization mode |
| `--objective METRIC:WEIGHT` | `H_total:1.0 Q6:-1.0` | Objective term (repeatable). Optimizer **maximises** the weighted sum |
| `--method {annealing,basin_hopping,parallel_tempering}` | `annealing` | Optimization algorithm |
| `--max-steps N` | 5000 | MC steps per restart |
| `--n-samples N` | 1 | Number of independent restarts |
| `--T-start T` | 1.0 | Initial temperature |
| `--T-end T` | 0.01 | Final temperature (SA only) |
| `--frag-threshold Q` | 0.3 | Local Q6 threshold for fragment selection |
| `--move-step A` | 0.5 | Maximum displacement per step (Å) |
| `--no-composition-moves` | flag | Disable element-type swaps; only atomic positions are moved |
| `--no-displacements` | flag | Disable atomic-position moves; only element-type swaps are performed. Cannot be combined with `--no-composition-moves` |
| `--lcc-threshold L` | 0.0 | Minimum `graph_lcc` for step acceptance (0 = disabled) |
| `--n-replicas N` | 4 | Temperature replicas for `parallel_tempering` |
| `--pt-swap-interval N` | 10 | Replica-exchange attempt interval |
| `--initial-xyz FILE` | — | XYZ file to use as the starting structure |

```bash
# Position-only SA — composition fixed, only geometry explored
pasted --n-atoms 50 --charge 0 --mult 1 --elements 24,25,26,27,28 \
    --optimize --method annealing --max-steps 5000 \
    --no-composition-moves \
    --objective H_atom:1.0 --objective H_spatial:1.0 --objective Q6:-2.0 \
    --seed 42 -o best.xyz

# Composition-only SA — coordinates fixed, only element types explored
pasted --n-atoms 50 --charge 0 --mult 1 --elements 24,25,26,27,28 \
    --optimize --method annealing --max-steps 5000 \
    --no-displacements \
    --initial-xyz fixed_geometry.xyz \
    --objective H_atom:1.0 --objective Q6:-2.0 \
    --seed 42 -o best.xyz

# Cantor-alloy optimisation with composition moves (default)
pasted --n-atoms 50 --charge 0 --mult 1 --elements 24,25,26,27,28 \
    --optimize --method parallel_tempering \
    --n-samples 2 --n-replicas 4 --max-steps 2000 \
    --objective H_total:1.0 --objective Q6:-2.0 \
    --seed 0 -o best.xyz
```
