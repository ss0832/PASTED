"""
PASTED — Pointless Atom STructure with Entropy Diagnostics
==========================================================
A structure fuzzer for quantum-chemistry and machine-learning potential codes.

PASTED generates random atomic clusters with configurable disorder metrics and
feeds them to QC/ML engines to expose edge-case failures.

Quick start
-----------
**Functional API** — one call to get a list of structures::

    from pasted import generate

    result = generate(
        n_atoms=12, charge=0, mult=1,
        mode="gas", region="sphere:9",
        elements="1-30", n_samples=50, seed=42,
    )
    for s in result:
        print(s)            # Structure(n=14, comp='C2H8N2O2', mode='gas', H_total=2.341)
        print(s.to_xyz())   # extended-XYZ string

**Class API** — more control over generation::

    from pasted import StructureGenerator

    gen = StructureGenerator(
        n_atoms=15, charge=0, mult=1,
        mode="chain", elements="6,7,8",
        n_success=5,       # stop after 5 passing structures
        n_samples=200,
        filters=["H_total:2.0:-"],
        seed=0,
    )
    result = gen.generate()
    print(result.summary())   # passed=5  attempted=73  rejected_filter=68

**Streaming output** — write each result immediately::

    for s in gen.stream():
        s.write_xyz("out.xyz")   # appended on each pass

**Optimizer** — maximize a disorder objective::

    from pasted import StructureOptimizer

    opt = StructureOptimizer(
        n_atoms=12, charge=0, mult=1,
        elements="6,7,8,15,16",
        objective={"H_total": 1.0, "Q6": -2.0},
        method="annealing",
        max_steps=5000,
        n_restarts=4,
        seed=42,
    )
    result = opt.run()
    print(result.best)           # highest-scoring structure
    print(result.summary())      # restarts=4  best_f=…  method='annealing'

**Immutable config** — type-safe, hashable, one-field override::

    import dataclasses
    from pasted import GeneratorConfig, StructureGenerator

    cfg = GeneratorConfig(
        n_atoms=20, charge=0, mult=1,
        mode="gas", region="sphere:10",
        elements="6,7,8", n_samples=100, seed=42,
    )
    result1 = StructureGenerator(cfg).generate()
    result2 = StructureGenerator(dataclasses.replace(cfg, seed=99)).generate()

**CLI**::

    pasted --n-atoms 12 --elements 1-30 --charge 0 --mult 1 \\
           --mode gas --region sphere:9 --n-samples 50 -o out.xyz

See ``docs/quickstart.md`` for the full guide; ``docs/cli.md`` for all
command-line options.
"""

import importlib.metadata

from ._atoms import (
    ALL_METRICS,
    ATOMIC_NUMBERS,
    PAULING_EN_FALLBACK,
    cov_radius_ang,
    default_element_pool,
    parse_element_spec,
    parse_filter,
    pauling_electronegativity,
    validate_charge_mult,
)
from ._config import GeneratorConfig
from ._generator import GenerationResult, Structure, StructureGenerator, generate, read_xyz
from ._io import format_xyz, parse_xyz
from ._metrics import (
    compute_all_metrics,
    compute_angular_entropy,
    compute_charge_frustration,
    compute_moran_I_chi,
    compute_ring_fraction,
    compute_steinhardt_per_atom,
)
from ._optimizer import OptimizationResult, StructureOptimizer, parse_objective_spec
from ._placement import place_maxent

try:
    __version__: str = importlib.metadata.version("pasted")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.2.9"

__all__ = [
    # High-level API
    "GenerationResult",
    "GeneratorConfig",
    "OptimizationResult",
    "Structure",
    "StructureGenerator",
    "StructureOptimizer",
    "generate",
    "parse_objective_spec",
    "read_xyz",
    "parse_xyz",
    # Atomic data / utilities
    "ATOMIC_NUMBERS",
    "ALL_METRICS",
    "PAULING_EN_FALLBACK",
    "cov_radius_ang",
    "pauling_electronegativity",
    "default_element_pool",
    "parse_element_spec",
    "parse_filter",
    "validate_charge_mult",
    # Metric / IO utilities
    "compute_all_metrics",
    "compute_angular_entropy",
    "compute_ring_fraction",
    "compute_charge_frustration",
    "compute_moran_I_chi",
    "compute_steinhardt_per_atom",
    "format_xyz",
    # Placement utilities
    "place_maxent",
    # Package metadata
    "__version__",
]
