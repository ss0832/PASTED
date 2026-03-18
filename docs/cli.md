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
| `--n-samples N` | int | 1 | Number of structures to attempt |
| `--seed S` | int | None | RNG seed for reproducibility |
| `--output FILE` / `-o` | path | stdout | Output XYZ file (append mode) |
| `--verbose` | flag | off | Print per-sample metrics to stderr |

## Placement mode

| Option | Description |
|---|---|
| `--mode {gas,chain,shell,maxent}` | Placement algorithm (default: `gas`) |

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
--filter "H_total:2.0:-"     # H_total >= 2.0
--filter "Q6:-:0.3"          # Q6 <= 0.3
--filter "shape_aniso:0.5:-" # shape_aniso >= 0.5
```

Available metrics: `H_atom`, `H_spatial`, `H_total`, `RDF_dev`,
`shape_aniso`, `Q4`, `Q6`, `Q8`, `graph_lcc`, `graph_cc`.

## Physical constraints

| Option | Default | Description |
|---|---|---|
| `--cov-scale S` | 1.0 | Minimum-distance scale factor: `d_min = S × (r_i + r_j)` |
| `--relax-cycles N` | 1500 | Maximum repulsion-relaxation iterations |
| `--no-add-hydrogen` | flag | Disable automatic H augmentation |

## Optimiser mode

Pass `--optimize` to run basin-hopping optimisation on an existing structure
instead of de-novo generation:

```bash
pasted --optimize input.xyz \
       --objective "H_total:max" \
       --opt-steps 500 --seed 0
```
