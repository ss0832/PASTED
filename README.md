# PASTED

**P**ointless **A**tom **ST**ructure with **E**ntropy **D**iagnostics

A CLI tool that generates intentionally random, physically meaningless atomic structures and evaluates their degree of disorder through a suite of structural metrics. Useful for stress-testing structure optimizers, generating worst-case inputs for quantum chemistry codes, or studying what "maximum chaos" looks like in structural space.

## Features

- **Three placement modes** — fully random (`gas`), chain-growth (`chain`), coordination-complex-like (`shell`)
- **10 disorder metrics** computed per structure, all usable as output filters
- **Element pool** specified by atomic number (Z = 1–106, H through Sg); composition sampled randomly per structure
- Charge/multiplicity parity validation before any geometry is generated
- Multi-structure batch generation with `--n-samples`; per-structure progress on stderr, XYZ on stdout
- Reproducible runs via `--seed`

## Requirements

```
Python >= 3.10
numpy
scipy
```

```bash
pip install numpy scipy
```

## Installation

```bash
git clone https://github.com/yourname/pasted.git
cd pasted
# no build step required; run directly
python pasted.py --help
```

## Quick Start

```bash
# 10 atoms drawn from H–Zn, placed randomly in a sphere of radius 8 Å
python pasted.py --n-atoms 10 --elements 1-30 --charge 0 --mult 1 \
    --mode gas --region sphere:8

# Organic-looking chain structure (C/N/O only)
python pasted.py --n-atoms 15 --elements 6,7,8 --charge 0 --mult 1 \
    --mode chain --branch-prob 0.4 --n-samples 20 -o organic_junk.xyz

# Coordination-complex-like structure with Fe as center
python pasted.py --n-atoms 12 --elements 6,7,8,26 --charge 0 --mult 1 \
    --mode shell --center-z 26 --coord-range 4:6 --n-samples 10

# Generate 100 structures, keep only the most disordered ones
python pasted.py --n-atoms 12 --elements 1-30 --charge 0 --mult 1 \
    --mode gas --region sphere:9 --n-samples 100 \
    --filter H_total:2.0:- --filter shape_aniso:0.3:- -o disordered.xyz
```

## Placement Modes

### `gas` (default)

Atoms are placed independently and uniformly at random inside the specified region.
Closest to true spatial randomness; highest expected `H_spatial`.

```
--region sphere:R       sphere of radius R Å
--region box:L          cube of side L Å
--region box:LX,LY,LZ   orthorhombic box
```

### `chain`

Atoms grow one by one from a seed atom via a random walk.
At each step, a random active tip is selected and the new atom is placed at a random bond length in a random direction.
A branching probability controls whether the old tip is kept (branch) or replaced (linear advance).
Produces elongated, tree-like structures reminiscent of organic molecules — except with no regard for valence.

```
--branch-prob FLOAT     branching probability (default: 0.3)
--bond-range LO:HI      bond length range in Å (default: 1.2:1.6)
```

### `shell`

One atom is placed at the origin as the "center", surrounded by a coordination shell at a random radius,
followed by tail atoms growing from shell members.
Produces structures that superficially resemble coordination complexes.

```
--center-z Z            atomic number of center atom
                        (default: random from the sample's composition)
--coord-range MIN:MAX   coordination number range (default: 4:8)
--shell-radius LO:HI    shell radius range in Å (default: 1.8:2.5)
--bond-range LO:HI      tail bond length range in Å (default: 1.2:1.6)
```

The center atom and its Z are recorded in the XYZ comment line as `center=Fe(Z=26)`.

## Element Pool

```
--elements SPEC
```

Elements are specified by atomic number. Omit to use all supported elements (Z = 1–106).

| Syntax | Meaning |
|--------|---------|
| `1-30` | Z = 1 through 30 (H to Zn) |
| `6,7,8` | Z = 6, 7, 8 (C, N, O) |
| `1-10,26,28` | Z = 1–10 plus Fe(26) and Ni(28) |
| `72-80` | 5d metals Hf through Hg |
| *(omitted)* | all Z = 1–106 |

For each structure, `--n-atoms` elements are drawn independently and uniformly from this pool.
The resulting composition varies per sample.

## Charge and Multiplicity

`--charge` and `--mult` are required and apply to every generated structure.
Before placement, PASTED checks two conditions against the randomly sampled composition:

1. Total electron count `N_e = Σ Z − charge > 0`
2. Parity: `N_e % 2 == (mult − 1) % 2`

Structures that fail either check are logged as `[invalid]` and skipped.
High-spin vs. low-spin selection is **not enforced**; that is the user's responsibility.

Because composition is random, parity failures are common when the element pool contains many odd-Z elements and `mult=1` is specified.
Increasing `--n-samples` or using `--mult 2` reduces this.

## Disorder Metrics

All metrics are computed for every structure and embedded in the XYZ comment line.
All are usable in `--filter`.

| Metric | Description | Range |
|--------|-------------|-------|
| `H_atom` | Shannon entropy of element composition | 0 (single element) to ln(*k*) |
| `H_spatial` | Shannon entropy of the pairwise-distance histogram | higher = more uniform distances |
| `H_total` | Weighted sum: `w_atom · H_atom + w_spatial · H_spatial` | — |
| `RDF_dev` | RMS deviation of empirical *g*(*r*) from ideal-gas baseline | 0 = perfectly random |
| `shape_aniso` | Relative shape anisotropy from the gyration tensor | 0 = spherical, 1 = rod-like |
| `Q4`, `Q6`, `Q8` | Steinhardt bond-orientational order parameters (averaged over atoms) | 0 = disordered |
| `graph_lcc` | Fraction of atoms in the largest connected component at `--cutoff` | 0–1 |
| `graph_cc` | Mean clustering coefficient at `--cutoff` | 0–1 |

### Tuning metric behaviour

```
--n-bins N          histogram bins for H_spatial and RDF_dev (default: 20)
--w-atom FLOAT      weight of H_atom in H_total (default: 0.5)
--w-spatial FLOAT   weight of H_spatial in H_total (default: 0.5)
--cutoff FLOAT      distance cutoff in Å for Q_l and graph_* (default: 2.0)
```

## Filtering

```
--filter METRIC:MIN:MAX
```

Only structures whose metric falls in [MIN, MAX] are written to output.
Use `-` for an open bound.
The flag is repeatable; all conditions must be satisfied simultaneously.

```bash
# Keep structures with high total entropy
--filter H_total:2.0:-

# Keep elongated structures (rod-like)
--filter shape_aniso:0.5:-

# Keep well-connected chains
--filter graph_lcc:0.8:- --filter graph_cc:0.4:-

# Keep structures with low local order (no accidental crystallinity)
--filter Q6:-:0.4
```

## Output Format

Structures are written as a concatenated multi-structure XYZ file.
Progress and statistics are written to stderr; the XYZ data goes to stdout (or `--output`).

```
12
sample=3 mode=chain charge=+0 mult=1 comp=[C:4,N:5,O:3]  H_atom=1.0986  H_spatial=2.7812  H_total=1.9399  RDF_dev=3.2451  shape_aniso=0.5123  Q4=0.5210  Q6=0.5880  Q8=0.6014  graph_lcc=1.0000  graph_cc=0.5714
C       1.234567    -0.987654     2.345678
N      -1.456789     3.210987    -0.123456
...
```

```bash
# XYZ to file, progress to terminal
python pasted.py ... -o out.xyz

# pipe XYZ, discard progress
python pasted.py ... 2>/dev/null | downstream_tool

# progress only (dry run to check filter hit rate)
python pasted.py ... -o /dev/null
```

## Full Option Reference

```
required:
  --n-atoms N           total atoms per structure
  --charge INT          total system charge
  --mult INT            spin multiplicity 2S+1

placement mode:
  --mode {gas,chain,shell}
  --region SPEC         [gas] sphere:R | box:L | box:LX,LY,LZ
  --branch-prob FLOAT   [chain] branching probability (default: 0.3)
  --bond-range LO:HI    [chain/shell] bond length range Å (default: 1.2:1.6)
  --center-z Z          [shell] fix center atom by atomic number
  --coord-range MIN:MAX [shell] coordination number range (default: 4:8)
  --shell-radius LO:HI  [shell] shell radius range Å (default: 1.8:2.5)

elements:
  --elements SPEC       atomic-number spec (default: all Z=1-106)

placement:
  --min-dist FLOAT      minimum interatomic distance Å (default: 0.8)
  --max-attempts INT    placement attempts per atom (default: 10000)

sampling:
  --n-samples INT       number of structures to attempt (default: 1)
  --seed INT            random seed

metrics:
  --n-bins INT          histogram bins (default: 20)
  --w-atom FLOAT        H_atom weight in H_total (default: 0.5)
  --w-spatial FLOAT     H_spatial weight in H_total (default: 0.5)
  --cutoff FLOAT        distance cutoff Å for Q_l / graph_* (default: 2.0)

filtering:
  --filter METRIC:MIN:MAX   repeatable; use - for open bound

output:
  --validate            check charge/mult against one random composition, then exit
  -o / --output FILE    XYZ output file (default: stdout)
```

## Notes and Limitations

- **RDF_dev** is a finite-system approximation. Treat it as a relative indicator, not an absolute measure.
- **Q4/Q6/Q8** are meaningful only when `--cutoff` is set to a value that gives a reasonable number of neighbours. The default 2.0 Å is appropriate for light elements; increase it (e.g. `--cutoff 3.0`) for heavier elements with longer bonds.
- Charge/mult parity is checked per sample against the randomly drawn composition. A large `invalid_charge_mult` count in the summary means the element pool and charge/mult combination produce many odd-electron systems. Either widen `--mult` or restrict `--elements` to even-Z elements.

## License

MIT License. See [LICENSE](LICENSE).