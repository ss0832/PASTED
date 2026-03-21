"""
pasted._config
==============
:class:`GeneratorConfig` — frozen dataclass that captures every parameter
accepted by :class:`~pasted._generator.StructureGenerator`.

Using a frozen dataclass instead of individual keyword arguments gives:

* Full mypy / IDE type-checking on every field
* Hashable, immutable config objects (safe to store, copy, and compare)
* ``dataclasses.replace(cfg, seed=99)`` for convenient one-field overrides
* A single variable to pass around instead of 30+ scattered kwargs

Relationship to the existing kwargs API
----------------------------------------
:func:`~pasted._generator.generate` still accepts plain keyword arguments
for backward compatibility — it constructs a :class:`GeneratorConfig`
internally.  New code is encouraged to build the config explicitly::

    from pasted import GeneratorConfig, StructureGenerator

    cfg = GeneratorConfig(n_atoms=12, charge=0, mult=1,
                          mode="gas", region="sphere:8",
                          elements="6,7,8", n_samples=50, seed=42)
    gen = StructureGenerator(cfg)
    result = gen.generate()

    # One-field override via dataclasses.replace (no mutation):
    cfg2 = dataclasses.replace(cfg, seed=99)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GeneratorConfig:
    """Immutable configuration for :class:`~pasted._generator.StructureGenerator`.

    All fields correspond exactly to the keyword parameters of
    :class:`~pasted._generator.StructureGenerator` and
    :func:`~pasted._generator.generate`.  See those docstrings for full
    per-field documentation.

    The three fields without defaults (**n_atoms**, **charge**, **mult**)
    must always be supplied explicitly.

    Examples
    --------
    Construct and pass to the class API::

        from dataclasses import replace
        from pasted import GeneratorConfig, StructureGenerator

        cfg = GeneratorConfig(
            n_atoms=20, charge=0, mult=1,
            mode="gas", region="sphere:10",
            elements="6,7,8", n_samples=100, seed=0,
        )
        result = StructureGenerator(cfg).generate()

        # Reuse with a different seed:
        result2 = StructureGenerator(replace(cfg, seed=1)).generate()

    Pass directly to the functional API::

        from pasted import generate, GeneratorConfig

        cfg = GeneratorConfig(n_atoms=12, charge=0, mult=1,
                              mode="chain", elements="6,7,8",
                              n_samples=50, seed=42)
        result = generate(cfg)
    """

    # ── Required fields (no default) ─────────────────────────────────────
    n_atoms: int
    charge: int
    mult: int

    # ── Placement mode ───────────────────────────────────────────────────
    mode: str = "gas"
    region: str | None = None

    # ── Chain-mode parameters ─────────────────────────────────────────────
    branch_prob: float = 0.3
    chain_persist: float = 0.5
    chain_bias: float = 0.0
    bond_range: tuple[float, float] = (1.2, 1.6)

    # ── Shell-mode parameters ─────────────────────────────────────────────
    center_z: int | None = None
    coord_range: tuple[int, int] = (4, 8)
    shell_radius: tuple[float, float] = (1.8, 2.5)

    # ── Element pool ─────────────────────────────────────────────────────
    elements: str | list[str] | None = None
    element_fractions: dict[str, float] | None = None
    element_min_counts: dict[str, int] | None = None
    element_max_counts: dict[str, int] | None = None

    # ── Physics / relaxation ─────────────────────────────────────────────
    cov_scale: float = 1.0
    relax_cycles: int = 1500
    #: Global affine transform strength. ``0.0`` disables (default).
    #: Individual operations can be overridden via *affine_stretch*,
    #: *affine_shear*, and *affine_jitter*.
    affine_strength: float = 0.0
    #: Stretch/compress strength; falls back to *affine_strength* when ``None``.
    affine_stretch: float | None = None
    #: Shear strength; falls back to *affine_strength* when ``None``.
    affine_shear: float | None = None
    #: Per-atom jitter scale; falls back to *affine_strength* when ``None``.
    affine_jitter: float | None = None

    # ── Maxent-mode parameters ────────────────────────────────────────────
    maxent_steps: int = 300
    maxent_lr: float = 0.05
    maxent_cutoff_scale: float = 2.5
    trust_radius: float = 0.5
    convergence_tol: float = 1e-3

    # ── Hydrogen augmentation ─────────────────────────────────────────────
    add_hydrogen: bool = True

    # ── Sampling control ─────────────────────────────────────────────────
    n_samples: int = 1
    n_success: int | None = None
    seed: int | None = None

    # ── Metric computation ────────────────────────────────────────────────
    n_bins: int = 20
    w_atom: float = 0.5
    w_spatial: float = 0.5
    cutoff: float | None = None

    # ── Filters & verbosity ───────────────────────────────────────────────
    filters: list[str] | None = field(default=None)
    verbose: bool = False
