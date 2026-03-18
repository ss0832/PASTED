"""
PASTED — Pointless Atom STructure with Entropy Diagnostics
==========================================================
A structure fuzzer for quantum-chemistry and machine-learning potential codes.

Quick start
-----------
**Class API**::

    from pasted import StructureGenerator

    gen = StructureGenerator(
        n_atoms=12, charge=0, mult=1,
        mode="gas", region="sphere:9",
        elements="1-30", n_samples=50, seed=42,
    )
    structures = gen.generate()
    for s in structures:
        print(s)               # Structure(n=14, comp='C2H8N2O2', mode='gas', H_total=2.341)
        print(s.to_xyz())      # extended-XYZ string

**Functional API**::

    from pasted import generate

    structures = generate(
        n_atoms=10, charge=0, mult=1,
        mode="chain", elements="6,7,8",
        n_samples=20, seed=0,
    )

**CLI**::

    pasted --n-atoms 12 --elements 1-30 --charge 0 --mult 1 \\
           --mode gas --region sphere:9 --n-samples 50 -o out.xyz
"""

import importlib.metadata

from ._atoms import (
    ALL_METRICS,
    ATOMIC_NUMBERS,
    cov_radius_ang,
    default_element_pool,
    parse_element_spec,
    parse_filter,
    validate_charge_mult,
)
from ._generator import Structure, StructureGenerator, generate
from ._io import format_xyz
from ._metrics import compute_all_metrics, compute_angular_entropy, compute_steinhardt_per_atom
from ._optimizer import StructureOptimizer, parse_objective_spec
from ._placement import place_maxent

try:
    __version__: str = importlib.metadata.version("pasted")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    # High-level API
    "Structure",
    "StructureGenerator",
    "StructureOptimizer",
    "generate",
    "parse_objective_spec",
    # Atomic data / utilities
    "ATOMIC_NUMBERS",
    "ALL_METRICS",
    "cov_radius_ang",
    "default_element_pool",
    "parse_element_spec",
    "parse_filter",
    "validate_charge_mult",
    # Metric / IO utilities
    "compute_all_metrics",
    "compute_angular_entropy",
    "compute_steinhardt_per_atom",
    "format_xyz",
    # Placement utilities
    "place_maxent",
    # Package metadata
    "__version__",
]
