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
    PAULING_EN_FALLBACK,
    cov_radius_ang,
    default_element_pool,
    parse_element_spec,
    parse_filter,
    pauling_electronegativity,
    validate_charge_mult,
)
from ._ext import HAS_OPENMP, set_num_threads
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
    __version__ = "unknown"

__all__ = [
    # High-level API
    "GenerationResult",
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
    # OpenMP / threading
    "HAS_OPENMP",
    "set_num_threads",
    # Package metadata
    "__version__",
]
