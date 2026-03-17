"""
pasted.cli
==========
Command-line interface.  Parses arguments and delegates all generation logic
to :class:`~pasted._generator.StructureGenerator`.
"""

from __future__ import annotations

import argparse
import random
import sys

from ._atoms import (
    default_element_pool,
    parse_element_spec,
    parse_int_range,
    parse_lo_hi,
    validate_charge_mult,
)
from ._generator import StructureGenerator

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pasted",
        description=(
            "PASTED: Pointless Atom STructure with Entropy Diagnostics\n"
            "Elements Z=1-106.  Modes: gas / chain / shell.\n"
            "Metrics: H_atom H_spatial H_total RDF_dev shape_aniso "
            "Q4 Q6 Q8 graph_lcc graph_cc\n"
            "Requires: numpy scipy"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
element-spec syntax: ranges and/or comma lists of atomic numbers.
  '1-30'       H to Zn
  '6,7,8'      C, N, O
  '1-10,26'    H-Ne plus Fe
  '72-80'      5d metals Hf-Hg
  (omitted)    all Z=1-106

examples
--------
  # gas mode
  python -m pasted --n-atoms 12 --elements 1-30 --charge 0 --mult 1 \\
      --mode gas --region sphere:9 --n-samples 50 --filter H_total:1.8:3.5

  # chain mode (organic-like)
  python -m pasted --n-atoms 15 --elements 6,7,8 --charge 0 --mult 1 \\
      --mode chain --branch-prob 0.4 --n-samples 20

  # shell mode (coordination-complex-like)
  python -m pasted --n-atoms 12 --elements 6,7,8,26 --charge 0 --mult 1 \\
      --mode shell --center-z 26 --coord-range 4:6 --n-samples 10

  # filter on structural metrics
  python -m pasted --n-atoms 10 --elements 6,7,8 --charge 0 --mult 1 \\
      --mode chain --n-samples 100 \\
      --filter shape_aniso:0.4:- --filter graph_lcc:0.8:- -o out.xyz
""",
    )

    req = p.add_argument_group("required")
    req.add_argument(
        "--n-atoms", type=int, required=True, help="Total number of atoms per structure."
    )
    req.add_argument("--charge", type=int, required=True, help="Total system charge.")
    req.add_argument(
        "--mult", type=int, required=True, help="Spin multiplicity 2S+1 (HS/LS not enforced)."
    )

    mg = p.add_argument_group("placement mode")
    mg.add_argument("--mode", choices=["gas", "chain", "shell"], default="gas")
    mg.add_argument(
        "--region", help="[gas] 'sphere:R' | 'box:L' | 'box:LX,LY,LZ' (Å). Required for gas."
    )
    mg.add_argument(
        "--branch-prob",
        type=float,
        default=0.3,
        help="[chain] Branching probability (default: 0.3).",
    )
    mg.add_argument(
        "--chain-persist",
        type=float,
        default=0.5,
        help=(
            "[chain] Directional persistence (0.0–1.0, default: 0.5). "
            "0.0 = fully random; 0.5 = rear 120° cone excluded; "
            "1.0 = nearly straight."
        ),
    )
    mg.add_argument(
        "--bond-range",
        default="1.2:1.6",
        metavar="LO:HI",
        help="[chain/shell-tails] Bond length range Å (default: 1.2:1.6).",
    )
    mg.add_argument(
        "--center-z",
        type=int,
        default=None,
        metavar="Z",
        help="[shell] Atomic number of center atom. Default: random per sample.",
    )
    mg.add_argument(
        "--coord-range",
        default="4:8",
        metavar="MIN:MAX",
        help="[shell] Coordination number range (default: 4:8).",
    )
    mg.add_argument(
        "--shell-radius",
        default="1.8:2.5",
        metavar="LO:HI",
        help="[shell] Shell radius range Å (default: 1.8:2.5).",
    )

    eg = p.add_argument_group("elements")
    eg.add_argument(
        "--elements",
        default=None,
        metavar="SPEC",
        help="Element pool by atomic number (default: all Z=1-106).",
    )

    pg = p.add_argument_group("placement")
    pg.add_argument(
        "--cov-scale",
        type=float,
        default=1.0,
        help=(
            "Minimum distance = cov_scale × (r_i + r_j), Pyykkö (2009) radii. "
            "Default 1.0 = exact sum of covalent radii."
        ),
    )
    pg.add_argument(
        "--relax-cycles",
        type=int,
        default=1500,
        help="Max cycles for post-placement repulsion relaxation (default: 1500).",
    )
    pg.add_argument(
        "--no-add-hydrogen",
        action="store_true",
        help=(
            "Disable automatic H augmentation. "
            "By default, if H(Z=1) is in the element pool and the sampled "
            "composition contains no H, H atoms are appended "
            "(n_H ≈ 1 + uniform(0,1) × n_atoms × 1.2)."
        ),
    )

    sg = p.add_argument_group("sampling")
    sg.add_argument("--n-samples", type=int, default=1)
    sg.add_argument("--seed", type=int, default=None)

    xg = p.add_argument_group("metrics")
    xg.add_argument(
        "--n-bins", type=int, default=20, help="Histogram bins for H_spatial/RDF_dev (default: 20)."
    )
    xg.add_argument("--w-atom", type=float, default=0.5)
    xg.add_argument("--w-spatial", type=float, default=0.5)
    xg.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help=(
            "Distance cutoff in Å for Q_l and graph_* metrics. "
            "Default: auto = cov_scale × 1.5 × median(r_i + r_j) over "
            "the element pool. Set explicitly to override."
        ),
    )

    fg = p.add_argument_group("filtering")
    fg.add_argument(
        "--filter",
        action="append",
        default=[],
        dest="filters",
        metavar="METRIC:MIN:MAX",
        help=("Keep structures where METRIC ∈ [MIN, MAX]. Use '-' for open bound. Repeatable."),
    )

    og = p.add_argument_group("output")
    og.add_argument(
        "--validate",
        action="store_true",
        help="Validate charge/mult against one random composition, then exit.",
    )
    og.add_argument("-o", "--output", default=None, help="Output XYZ file (default: stdout).")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "gas" and not args.region:
        parser.error("--region is required for --mode gas")

    # Parse range arguments
    try:
        bond_range = parse_lo_hi(args.bond_range, "bond-range")
        shell_radius = parse_lo_hi(args.shell_radius, "shell-radius")
        coord_range = parse_int_range(args.coord_range)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    # Resolve element pool
    try:
        element_pool: list[str] | None = (
            parse_element_spec(args.elements) if args.elements else None
        )
    except ValueError as exc:
        print(f"[ERROR] --elements: {exc}", file=sys.stderr)
        sys.exit(1)

    # Print pool info (CLI-only: includes the original spec string)
    if element_pool is not None:
        pool_label = f"--elements {args.elements!r} → {len(element_pool)} elements"
    else:
        pool_label = f"all Z=1-106 ({len(default_element_pool())} elements)"
    print(f"[pool] {pool_label}", file=sys.stderr)

    # --validate: quick sanity check then exit
    if args.validate:
        pool = element_pool or default_element_pool()
        rng = random.Random(args.seed)
        trial = [rng.choice(pool) for _ in range(args.n_atoms)]
        ok, msg = validate_charge_mult(trial, args.charge, args.mult)
        print(f"[validate:{'OK' if ok else 'FAIL'}] {msg}", file=sys.stderr)
        sys.exit(0 if ok else 1)

    # Build generator (verbose=True → all progress printed to stderr)
    try:
        gen = StructureGenerator(
            n_atoms=args.n_atoms,
            charge=args.charge,
            mult=args.mult,
            mode=args.mode,
            region=args.region,
            branch_prob=args.branch_prob,
            chain_persist=args.chain_persist,
            bond_range=bond_range,
            center_z=args.center_z,
            coord_range=coord_range,
            shell_radius=shell_radius,
            elements=element_pool,  # already a list[str] or None
            cov_scale=args.cov_scale,
            relax_cycles=args.relax_cycles,
            add_hydrogen=not args.no_add_hydrogen,
            n_samples=args.n_samples,
            seed=args.seed,
            n_bins=args.n_bins,
            w_atom=args.w_atom,
            w_spatial=args.w_spatial,
            cutoff=args.cutoff,
            filters=args.filters,
            verbose=True,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    structures = gen.generate()

    # Serialise output
    xyz_blocks = [s.to_xyz() for s in structures]
    output_text = "\n".join(xyz_blocks) + ("\n" if xyz_blocks else "")

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(output_text)
        print(
            f"[info] {len(structures)} structure(s) written to {args.output!r}",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(output_text)


if __name__ == "__main__":
    main()
