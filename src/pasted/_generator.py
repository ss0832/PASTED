"""
pasted._generator
=================
High-level API:

- :class:`Structure`          — dataclass holding one generated structure.
- :class:`StructureGenerator` — stateful generator (class API).
- :func:`generate`            — convenience functional wrapper.

Internal design
---------------
The generation loop lives in a single private method,
:meth:`StructureGenerator._stream_with_stats`, which returns a
``(structures_iterator, stats_dict)`` pair.  The public methods
:meth:`~StructureGenerator.stream` and
:meth:`~StructureGenerator.generate` are thin wrappers around it:

- ``stream()`` forwards the iterator directly to the caller.
- ``generate()`` exhausts the iterator, reads the populated *stats_dict*,
  and wraps everything in a :class:`GenerationResult`.

This two-layer design eliminates the hidden coupling that previously
existed via the ``_last_run_stats`` instance variable: interrupting
``stream()`` early can no longer leave stale counters for a subsequent
``generate()`` call.

Verbose log output is routed through three focused helpers
(:meth:`~StructureGenerator._log_filter_header`,
:meth:`~StructureGenerator._log_sample_result`,
:meth:`~StructureGenerator._log_summary`) so that the generation loop
itself contains only placement logic.
"""

from __future__ import annotations

import random
import sys
import warnings
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._atoms import (
    _Z_TO_SYM,
    ATOMIC_NUMBERS,
    _cov_radius_ang,
    default_element_pool,
    parse_element_spec,
    parse_filter,
    validate_charge_mult,
)
from ._config import GeneratorConfig
from ._io import _fmt, format_xyz, parse_xyz
from ._metrics import compute_all_metrics, passes_filters
from ._placement import (
    Vec3,
    _affine_move,
    add_hydrogen,
    place_chain,
    place_gas,
    place_maxent,
    place_shell,
    relax_positions,
)

# ---------------------------------------------------------------------------
# Structure dataclass
# ---------------------------------------------------------------------------


@dataclass
class Structure:
    """A single generated atomic structure with its computed disorder metrics.

    Attributes
    ----------
    atoms:
        Element symbols, one per atom.
    positions:
        Cartesian coordinates in Å, one ``(x, y, z)`` tuple per atom.
    charge:
        Total system charge.
    mult:
        Spin multiplicity 2S+1.
    metrics:
        Computed disorder metrics (see :data:`pasted._atoms.ALL_METRICS`).
    mode:
        Placement mode used (``"gas"``, ``"chain"``, ``"shell"``, ``"maxent"``,
        or ``"opt_<method>"`` for optimizer results).
    sample_index:
        1-based index within the batch of structures that passed filters.
    center_sym:
        Element symbol of the shell center atom (shell mode only).
    seed:
        Random seed used for generation (``None`` if unseeded).

    Properties
    ----------
    comp:
        Read-only composition string derived from :attr:`atoms`, sorted in
        alphabetical order by element symbol, e.g. ``'C5N2O3'``.
        Computed on access; not stored as a field.

        .. note::
            The sort order is **alphabetical** (``sorted()`` on symbol strings),
            not Hill order (C first, H second, then alphabetical).  Structures
            containing only C, H, N, O will look identical to Hill order, but
            others — e.g. ``['Na', 'C', 'H']`` → ``'CH2Na'`` — differ.

    Examples
    --------
    Access the composition string directly::

        s = generate(n_atoms=10, charge=0, mult=1, mode="gas",
                     region="sphere:8", elements="6,7,8", n_samples=5, seed=0)[0]
        print(s.comp)          # e.g. 'C4N3O3'
        print(repr(s))         # Structure(n=10, comp='C4N3O3', mode='gas', H_total=…)
    """

    atoms: list[str]
    positions: list[Vec3]
    charge: int
    mult: int
    metrics: dict[str, float]
    mode: str
    sample_index: int = 0
    center_sym: str | None = None
    seed: int | None = None

    # ------------------------------------------------------------------ #
    # XYZ output                                                           #
    # ------------------------------------------------------------------ #

    def to_xyz(self, prefix: str = "") -> str:
        """Serialise to extended XYZ format.

        Parameters
        ----------
        prefix:
            Custom prefix for the comment line.  When omitted the standard
            ``"sample=N mode=M …"`` string is generated automatically.

        Returns
        -------
        Multi-line string (no trailing newline).
        """
        if not prefix:
            prefix = f"sample={self.sample_index} mode={self.mode}"
            if self.mode == "shell" and self.center_sym:
                prefix += f" center={self.center_sym}(Z={ATOMIC_NUMBERS[self.center_sym]})"
            if self.seed is not None:
                prefix += f" seed={self.seed}"
        return format_xyz(
            self.atoms,
            self.positions,
            self.charge,
            self.mult,
            self.metrics,
            prefix,
        )

    def write_xyz(self, path: str | Path, *, append: bool = True) -> None:
        """Write this structure to an XYZ file.

        Parameters
        ----------
        path:
            Output file path.
        append:
            If ``True`` (default) the file is opened in append mode so that
            multiple structures can be written in sequence.  Use
            ``append=False`` to overwrite.
        """
        mode = "a" if append else "w"
        with Path(path).open(mode) as fh:
            fh.write(self.to_xyz() + "\n")

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.atoms)

    @property
    def n(self) -> int:
        """Number of atoms in the structure."""
        return len(self.atoms)

    # ------------------------------------------------------------------ #
    # XYZ import                                                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_xyz(
        cls,
        source: str | Path,
        *,
        frame: int = 0,
        recompute_metrics: bool = True,
        cutoff: float | None = None,
        n_bins: int = 20,
        w_atom: float = 0.5,
        w_spatial: float = 0.5,
        cov_scale: float = 1.0,
    ) -> Structure:
        """Load a :class:`Structure` from an XYZ file or string.

        Supports both plain XYZ and PASTED extended XYZ (with ``charge=``,
        ``mult=``, and metric tokens on the comment line).  When
        *recompute_metrics* is ``True`` (default), all disorder metrics are
        recomputed from the loaded geometry so that the returned structure
        is fully usable as optimizer input or for filtering.

        Parameters
        ----------
        source:
            Path to an XYZ file **or** a raw XYZ string.
        frame:
            Zero-based frame index when *source* contains multiple
            concatenated structures (default: first frame).
        recompute_metrics:
            Recompute all disorder metrics after loading.  Set to ``False``
            to skip the recomputation and return the structure with whatever
            metric values were embedded in the extended XYZ comment (or an
            empty dict for plain XYZ).
        cutoff:
            Distance cutoff (Å) for metric computation.  Auto-computed from
            the element pool when ``None``.
        n_bins:
            Histogram bins for ``H_spatial`` / ``RDF_dev`` (default: 20).
        w_atom:
            Weight of ``H_atom`` in ``H_total`` (default: 0.5).
        w_spatial:
            Weight of ``H_spatial`` in ``H_total`` (default: 0.5).
        cov_scale:
            Minimum distance scale factor used for metrics (default: 1.0).

        Returns
        -------
        Structure

        Raises
        ------
        FileNotFoundError
            When *source* looks like a file path (no newlines) but the path
            does not exist on disk.
        IsADirectoryError
            When *source* is a path that points to a directory rather than
            a regular file.
        ValueError
            When the XYZ content cannot be parsed, or *frame* is out of
            range.

        Examples
        --------
        Load and immediately use as optimizer initial structure::

            from pasted import Structure, StructureOptimizer

            s = Structure.from_xyz("my_structure.xyz")
            opt = StructureOptimizer(
                n_atoms=len(s), charge=s.charge, mult=s.mult,
                objective={"H_total": 1.0},
                elements=[sym for sym in set(s.atoms)],
                max_steps=2000, seed=42,
            )
            result = opt.run(initial=s)
        """
        # Determine whether *source* looks like a file path or raw XYZ text.
        # Heuristic: a string containing no newlines is treated as a path;
        # a multi-line string is treated as raw XYZ content.
        _looks_like_path = not isinstance(source, str) or (
            "\n" not in str(source) and str(source).strip()
        )
        p = Path(source) if _looks_like_path else None

        if p is not None:
            # *source* looks like a file path — enforce explicit errors.
            if not p.exists():
                raise FileNotFoundError(f"XYZ file not found: {p!s}")
            if not p.is_file():
                raise IsADirectoryError(f"Expected a file path, but {p!s} is a directory.")
            text = p.read_text()
        else:
            text = str(source)

        frames = parse_xyz(text)
        if not frames:
            raise ValueError("No frames found in XYZ source.")
        if frame < 0 or frame >= len(frames):
            raise ValueError(
                f"frame={frame} out of range; source contains {len(frames)} frame(s)."
            )

        atoms, positions, charge, mult, embedded_metrics = frames[frame]

        if recompute_metrics:
            if cutoff is None:
                radii = np.array([_cov_radius_ang(a) for a in atoms])
                # O(N) approximation: median(r_i + r_j) ≈ 2 × median(r_i).
                # Avoids O(N² log N) pair enumeration for large structures.
                cutoff = cov_scale * 1.5 * float(np.median(radii)) * 2.0
            metrics = compute_all_metrics(
                atoms, positions, n_bins, w_atom, w_spatial, cutoff, cov_scale
            )
        else:
            metrics = embedded_metrics

        return cls(
            atoms=list(atoms),
            positions=list(positions),
            charge=charge,
            mult=mult,
            metrics=metrics,
            mode="loaded_xyz",
        )

    @property
    def comp(self) -> str:
        """Alphabetically-sorted composition string derived from :attr:`atoms`.

        Elements are sorted in ascending alphabetical order by symbol and
        counts above one are appended as a suffix, e.g. ``'C5N2O3'``.
        Single-atom elements are written without a count, e.g. ``'C'``
        rather than ``'C1'``.

        .. note::
            The sort order is **alphabetical** (Python ``sorted()``), **not**
            Hill order (which would place C first, H second, then all other
            elements alphabetically).  For structures containing only C, H,
            N, O the two orderings coincide, but elements such as Na, Fe, or
            Ar will appear at their alphabetical position rather than after H.
            For example ``['Na', 'C', 'H', 'H']`` yields ``'CH2Na'``
            (alphabetical) rather than ``'CH2Na'`` (which happens to match
            Hill here) but ``['Ar', 'C', 'H']`` yields ``'ArCH2'``
            (alphabetical) not ``'CH2Ar'`` (Hill).

        This property is computed on each access and is not persisted as a
        dataclass field.

        Returns
        -------
        str
            Compact composition label, e.g. ``'C5N2O3'``.

        Examples
        --------
        ::

            s.comp          # 'C5N2O3'
            s.comp in repr(s)  # True
        """
        counts = Counter(self.atoms)
        return "".join(f"{sym}{n}" if n > 1 else sym for sym, n in sorted(counts.items()))

    def __repr__(self) -> str:
        h_total = self.metrics.get("H_total", float("nan"))
        return f"Structure(n={len(self)}, comp={self.comp!r}, mode={self.mode!r}, H_total={h_total:.3f})"


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """Return value of :func:`generate` and :meth:`StructureGenerator.generate`.

    Behaves like a ``list[Structure]`` in all normal usage (indexing,
    iteration, ``len``, boolean test, ``for s in result``) while also
    carrying metadata about how many attempts were made and why samples
    were rejected.  This metadata is especially useful when integrating
    PASTED into automated pipelines such as ASE or high-throughput
    workflows, where a silent empty list would be indistinguishable from
    a successful run that just produced no results.

    Attributes
    ----------
    structures:
        Structures that passed all filters.
    n_attempted:
        Total placement attempts made.
    n_passed:
        Number of structures that passed all filters (equals
        ``len(structures)`` unless the caller mutates the list).
    n_rejected_parity:
        Attempts rejected by the charge/multiplicity parity check.
    n_rejected_filter:
        Attempts rejected by user-supplied metric filters.
    n_success_target:
        The ``n_success`` value that was in effect during generation
        (``None`` when not set).

    Examples
    --------
    Drop-in replacement for ``list[Structure]``::

        result = generate(n_atoms=10, charge=0, mult=1,
                          mode="gas", region="sphere:8",
                          elements="6,7,8", n_samples=20, seed=0)
        for s in result:          # iterates like a list
            print(s.to_xyz())
        print(len(result))        # number that passed

    Inspect rejection metadata::

        if result.n_rejected_parity > 0:
            print(f"{result.n_rejected_parity} samples failed parity check")
        print(result.summary())

    Notes
    -----
    ``GenerationResult`` is a :func:`~dataclasses.dataclass`; downstream
    code should treat it as immutable.  The ``structures`` field is a
    plain ``list`` and may be sorted or sliced freely.
    """

    structures: list[Structure] = field(default_factory=list)
    n_attempted: int = 0
    n_passed: int = 0
    n_rejected_parity: int = 0
    n_rejected_filter: int = 0
    n_success_target: int | None = None

    # ------------------------------------------------------------------ #
    # list-compatible interface                                            #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.structures)

    def __iter__(self) -> Iterator[Structure]:
        return iter(self.structures)

    def __getitem__(self, index: int | slice) -> Structure | list[Structure]:
        if isinstance(index, slice):
            return self.structures[index]
        return self.structures[index]

    def __bool__(self) -> bool:
        return bool(self.structures)

    def __add__(self, other: GenerationResult) -> GenerationResult:
        """Merge two :class:`GenerationResult` objects into one.

        Combines structures and accumulates all counters so that batch
        workflows can collect results across multiple calls and treat them
        as a single result::

            r1 = generate(..., n_samples=20, seed=0)
            r2 = generate(..., n_samples=20, seed=1)
            combined = r1 + r2
            print(len(combined))          # up to 40
            print(combined.summary())

        Parameters
        ----------
        other:
            Another :class:`GenerationResult` to merge into this one.

        Returns
        -------
        GenerationResult
            New result containing all structures from both operands.
            ``n_success_target`` is taken from *self* when set, otherwise
            from *other*.
        """
        if not isinstance(other, GenerationResult):
            return NotImplemented
        return GenerationResult(
            structures=self.structures + other.structures,
            n_attempted=self.n_attempted + other.n_attempted,
            n_passed=self.n_passed + other.n_passed,
            n_rejected_parity=self.n_rejected_parity + other.n_rejected_parity,
            n_rejected_filter=self.n_rejected_filter + other.n_rejected_filter,
            n_success_target=self.n_success_target
            if self.n_success_target is not None
            else other.n_success_target,
        )

    def __repr__(self) -> str:
        return (
            f"GenerationResult("
            f"passed={self.n_passed}, "
            f"attempted={self.n_attempted}, "
            f"rejected_parity={self.n_rejected_parity}, "
            f"rejected_filter={self.n_rejected_filter})"
        )

    # ------------------------------------------------------------------ #
    # Metadata helpers                                                     #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        """Return a human-readable one-line summary of the generation run.

        Returns
        -------
        str
            E.g. ``"passed=5  attempted=20  rejected_parity=2  rejected_filter=13"``.
        """
        parts = [
            f"passed={self.n_passed}",
            f"attempted={self.n_attempted}",
            f"rejected_parity={self.n_rejected_parity}",
            f"rejected_filter={self.n_rejected_filter}",
        ]
        if self.n_success_target is not None:
            parts.append(f"n_success_target={self.n_success_target}")
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# StructureGenerator
# ---------------------------------------------------------------------------


class StructureGenerator:
    """Generate random atomic structures with disorder metrics.

    All parameters use Python snake_case names that correspond 1-to-1 with
    their CLI ``--flag`` counterparts.

    Parameters
    ----------
    n_atoms:
        Number of atoms per structure (before optional H augmentation).
    charge:
        Total system charge (applied to every structure).
    mult:
        Spin multiplicity 2S+1.
    mode:
        Placement mode: ``"gas"`` (default), ``"chain"``, ``"shell"``, or
        ``"maxent"``.
    region:
        Bounding-region spec: ``"sphere:R"`` | ``"box:L"`` | ``"box:LX,LY,LZ"``.
        **Required when** *mode* **is** ``"gas"`` **or** ``"maxent"``; ignored
        for ``"chain"`` and ``"shell"`` (those modes use their own geometry
        parameters such as *shell_radius* and *bond_range*).
        Example: ``region="sphere:8"`` places atoms inside an 8 Å-radius sphere.
    branch_prob:
        [chain] Branching probability (default: 0.3).
    chain_persist:
        [chain] Directional persistence ∈ [0, 1] (default: 0.5).
    chain_bias:
        [chain] Global-axis drift strength ∈ [0, 1] (default: 0.0).
        The direction of the first bond becomes the bias axis; each
        subsequent step is blended toward that axis before normalisation.
        0.0 → no bias (backwards-compatible); higher values produce more
        elongated structures with larger ``shape_aniso``.
    bond_range:
        [chain / shell tails] Bond-length range in Å (default: ``(1.2, 1.6)``).
    center_z:
        [shell] Atomic number of center atom.  ``None`` → random per sample.
    coord_range:
        [shell] Coordination-number range (default: ``(4, 8)``).
    shell_radius:
        [shell] Shell-radius range in Å (default: ``(1.8, 2.5)``).
    elements:
        Element pool.  Three forms are accepted:

        * **Atomic-number spec string** — a comma-separated list of integers
          and/or integer ranges, e.g. ``"6,7,8"`` (C, N, O) or ``"1-30"``
          (H to Zn) or ``"1-10,26,28"`` (H–Ne plus Fe and Ni).
          Ranges are inclusive.  **Symbol strings such as** ``"C,N,O"``
          **are not accepted** and will raise :exc:`ValueError`; use the
          numeric form ``"6,7,8"`` or pass a list instead.
        * **Explicit list of element symbols** — e.g. ``["C", "N", "O"]``
          or ``["Cr", "Mn", "Fe", "Co", "Ni"]``.  Symbols must be valid
          two-character-or-less IUPAC symbols recognised by PASTED.
        * ``None`` — all Z = 1–106 (default).
    element_fractions:
        Relative sampling weights for elements in the pool, as a
        ``{symbol: weight}`` dict (e.g. ``{"C": 0.5, "N": 0.3, "O": 0.2}``).
        Weights are *relative* — they are normalized internally and need not
        sum to 1.  Elements absent from the dict receive a weight of 1.0.
        When ``None`` (default), every element in the pool is sampled with
        equal probability.
    element_min_counts:
        Minimum number of atoms per element guaranteed in every generated
        structure (e.g. ``{"C": 2, "N": 1}``).  The required atoms are
        placed first; remaining slots are filled by weighted random sampling.
        ``None`` (default) → no lower bounds.  The sum of all minimum counts
        must not exceed ``n_atoms``.
    element_max_counts:
        Maximum number of atoms allowed per element
        (e.g. ``{"N": 5, "O": 3}``).  Elements that have reached their
        cap are excluded from sampling for the remaining slots.
        ``None`` (default) → no upper bounds.

        .. note::
            When both *element_min_counts* and *element_max_counts* are
            given, each element's min must be ≤ its max.

        .. note::
            The automatic hydrogen augmentation step (``add_hydrogen=True``)
            runs *after* the constrained sampling and may temporarily exceed
            *element_max_counts* for H.  Set ``add_hydrogen=False`` if H
            count limits are critical.
    cov_scale:
        Minimum-distance scale factor: ``d_min(i,j) = cov_scale × (r_i + r_j)``
        using Pyykkö (2009) single-bond covalent radii.  Default: ``1.0``.
    relax_cycles:
        Maximum repulsion-relaxation iterations (default: 1500).
    add_hydrogen:
        Automatically append H atoms when H is in the pool but the sampled
        composition contains none (default: ``True``).
    affine_strength:
        Global dimensionless scale of the affine transformation applied to
        every generated structure **before** :func:`relax_positions` (default:
        ``0.0`` = disabled).  When > 0 a random stretch/compress + shear is
        applied once per structure, creating more anisotropic initial
        geometries before the repulsion-relaxation step.  Practical range:
        0.05–0.4.  At 0.1 the structure is stretched / compressed by up to
        ±10 % along a random axis and sheared by up to ±5 %.  Works
        identically across all placement modes (``gas``, ``chain``,
        ``shell``, ``maxent``).  ``0.0`` preserves the behavior of all
        versions prior to v0.2.3.

        Use *affine_stretch*, *affine_shear*, and *affine_jitter* to override
        individual operation strengths independently.
    affine_stretch:
        Strength of the stretch/compress operation only ∈ (0, 1).  When
        ``None`` (default) *affine_strength* is used.  Set to ``0.0`` to
        disable stretching while keeping shear and jitter active.
    affine_shear:
        Strength of the shear operation only ∈ (0, 1).  When ``None``
        (default) *affine_strength* is used.  Set to ``0.0`` to disable
        shearing while keeping stretch and jitter active.
    affine_jitter:
        Per-atom jitter scale ∈ (0, 1) relative to the move step.  When
        ``None`` (default) *affine_strength* is used.  For
        :class:`StructureGenerator` the move step is always ``0.0``, so
        jitter is never applied during generation regardless of this value;
        the parameter exists for symmetry with :class:`StructureOptimizer`.
    n_samples:
        Maximum number of placement attempts (default: 1).
        Use ``0`` to allow unlimited attempts (only valid when *n_success*
        is also set, otherwise a :exc:`ValueError` is raised).
    n_success:
        Target number of structures that must pass all filters before
        generation stops (default: ``None``).

        - ``None`` → generate exactly *n_samples* attempts and return all
          that passed (original behavior).
        - ``N > 0`` with ``n_samples > 0`` → stop as soon as *N* structures
          pass **or** *n_samples* attempts are exhausted, whichever comes
          first.  Returns the structures collected so far with a warning if
          fewer than *N* were found.
        - ``N > 0`` with ``n_samples = 0`` → unlimited attempts; stop only
          when *N* structures have passed.
    seed:
        Random seed for reproducibility (``None`` → non-deterministic).
    n_bins:
        Histogram bins for ``H_spatial`` and ``RDF_dev`` (default: 20).
    w_atom:
        Weight of ``H_atom`` in ``H_total`` (default: 0.5).
    w_spatial:
        Weight of ``H_spatial`` in ``H_total`` (default: 0.5).
    cutoff:
        Distance cutoff in Å for Steinhardt and graph metrics.
        ``None`` → auto-computed as ``cov_scale × 1.5 × median(r_i + r_j)``
        over the element pool.
    filters:
        Filter strings of the form ``"METRIC:MIN:MAX"`` (use ``"-"`` for an
        open bound).  Only structures satisfying *all* filters are returned.
    verbose:
        Print progress and statistics to *stderr* (default: ``False``).
        The CLI always passes ``True``; library callers usually leave it off.

    Examples
    --------
    Class API (config-based, recommended)::

        from pasted import GeneratorConfig, StructureGenerator

        cfg = GeneratorConfig(
            n_atoms=12, charge=0, mult=1,
            mode="gas", region="sphere:9",
            elements="1-30", n_samples=50, seed=42,
            filters=["H_total:2.0:-"],
        )
        gen = StructureGenerator(cfg)
        structures = gen.generate()
        for s in structures:
            print(s)

    Functional API (keyword-based, backward-compatible)::

        from pasted import generate

        structures = generate(
            n_atoms=12, charge=0, mult=1,
            mode="chain", elements="6,7,8",
            n_samples=20, seed=0,
        )
    """

    def __init__(
        self,
        config: GeneratorConfig | None = None,
        *,
        n_atoms: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Construct a :class:`StructureGenerator`.

        Two calling conventions are accepted:

        **Config-based (recommended):**

            gen = StructureGenerator(GeneratorConfig(n_atoms=12, charge=0, mult=1, ...))

        **Keyword-based (backward-compatible):**

            gen = StructureGenerator(n_atoms=12, charge=0, mult=1, ...)

        When *config* is given all other arguments are ignored.
        When *config* is ``None`` a :class:`GeneratorConfig` is built from
        the keyword arguments; ``n_atoms``, ``charge``, and ``mult`` are
        required.
        """
        if config is None:
            if n_atoms is None:
                raise TypeError(
                    "StructureGenerator requires either a GeneratorConfig as the "
                    "first argument, or keyword arguments including n_atoms=, "
                    "charge=, and mult=."
                )
            config = GeneratorConfig(n_atoms=n_atoms, **kwargs)
        self._cfg = config
        cfg = config  # local alias for brevity inside __init__

        # ── Mode / region validation ─────────────────────────────────────
        if cfg.mode not in ("gas", "chain", "shell", "maxent"):
            raise ValueError(
                f"mode must be 'gas', 'chain', 'shell', or 'maxent'; got {cfg.mode!r}"
            )
        if cfg.mode in ("gas", "maxent") and cfg.region is None:
            raise ValueError(
                f"region is required when mode={cfg.mode!r}. "
                'Pass e.g. region="sphere:8" (radius 8 Å) or '
                'region="box:10" (10×10×10 Å box).'
            )

        # ── n_samples / n_success validation ────────────────────────────
        if cfg.n_samples == 0 and cfg.n_success is None:
            raise ValueError(
                "n_samples=0 (unlimited) requires n_success to be set; "
                "otherwise generation would run forever."
            )
        if cfg.n_success is not None and cfg.n_success < 1:
            raise ValueError(f"n_success must be >= 1; got {cfg.n_success}.")

        # ── Element pool ────────────────────────────────────────────────
        if cfg.elements is None:
            self._element_pool: list[str] = default_element_pool()
        elif isinstance(cfg.elements, str):
            self._element_pool = parse_element_spec(cfg.elements)
        else:
            self._element_pool = list(cfg.elements)

        # ── Element fractions ────────────────────────────────────────────
        if cfg.element_fractions is not None:
            unknown = set(cfg.element_fractions) - set(self._element_pool)
            if unknown:
                raise ValueError(
                    f"element_fractions contains symbols not in the element pool: "
                    f"{sorted(unknown)}"
                )
            weights = [float(cfg.element_fractions.get(sym, 1.0)) for sym in self._element_pool]
            if any(w < 0 for w in weights):
                raise ValueError("element_fractions weights must be non-negative.")
            total = sum(weights)
            if total == 0:
                raise ValueError("element_fractions weights must not all be zero.")
            self._element_weights: list[float] = [w / total for w in weights]
        else:
            n = len(self._element_pool)
            self._element_weights = [1.0 / n] * n

        # ── Element min/max counts ───────────────────────────────────────
        if cfg.element_min_counts is not None:
            unknown_min = set(cfg.element_min_counts) - set(self._element_pool)
            if unknown_min:
                raise ValueError(
                    f"element_min_counts contains symbols not in the element pool: "
                    f"{sorted(unknown_min)}"
                )
            if any(v < 0 for v in cfg.element_min_counts.values()):
                raise ValueError("element_min_counts values must be non-negative.")
            total_min = sum(cfg.element_min_counts.values())
            if total_min > cfg.n_atoms:
                raise ValueError(
                    f"Sum of element_min_counts ({total_min}) exceeds n_atoms ({cfg.n_atoms})."
                )
        if cfg.element_max_counts is not None:
            unknown_max = set(cfg.element_max_counts) - set(self._element_pool)
            if unknown_max:
                raise ValueError(
                    f"element_max_counts contains symbols not in the element pool: "
                    f"{sorted(unknown_max)}"
                )
            if any(v < 0 for v in cfg.element_max_counts.values()):
                raise ValueError("element_max_counts values must be non-negative.")
        if cfg.element_min_counts is not None and cfg.element_max_counts is not None:
            for sym in cfg.element_min_counts:
                lo = cfg.element_min_counts[sym]
                hi = cfg.element_max_counts.get(sym, lo)
                if lo > hi:
                    raise ValueError(
                        f"element_min_counts[{sym!r}]={lo} > element_max_counts[{sym!r}]={hi}."
                    )
        self._element_min_counts: dict[str, int] = dict(cfg.element_min_counts or {})
        self._element_max_counts: dict[str, int] = dict(cfg.element_max_counts or {})

        # ── Filters ─────────────────────────────────────────────────────
        self._filters: list[tuple[str, float, float]] = [
            parse_filter(f) for f in (cfg.filters or [])
        ]

        # ── Cutoff ──────────────────────────────────────────────────────
        self._cutoff: float = self._resolve_cutoff(cfg.cutoff)

        # ── Shell center ─────────────────────────────────────────────────
        self._fixed_center_sym: str | None = None
        if cfg.mode == "shell" and cfg.center_z is not None:
            if cfg.center_z not in _Z_TO_SYM:
                raise ValueError(f"center_z={cfg.center_z}: unknown atomic number.")
            sym = _Z_TO_SYM[cfg.center_z]
            if sym not in self._element_pool:
                raise ValueError(f"center_z={cfg.center_z} ({sym}) is not in the element pool.")
            self._fixed_center_sym = sym

        if cfg.verbose:
            self._log(f"[pool] {len(self._element_pool)} elements in pool")
            if cfg.mode == "shell":
                if self._fixed_center_sym:
                    self._log(
                        f"[shell] center fixed: {self._fixed_center_sym} "
                        f"(Z={ATOMIC_NUMBERS[self._fixed_center_sym]})"
                    )
                else:
                    self._log("[shell] center: random per sample (chaos mode)")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _log(self, msg: str) -> None:
        """Print *msg* to stderr when verbose mode is active."""
        print(msg, file=sys.stderr)

    # ------------------------------------------------------------------ #
    # Verbose logging helpers                                              #
    # ------------------------------------------------------------------ #

    def _log_filter_header(self) -> None:
        """Log the active filter bounds to stderr (verbose mode only).

        Emits a single ``[filter]`` line listing every active filter in
        the form ``METRIC in [lo, hi]``.  Does nothing when there are no
        filters or verbose mode is off.
        """
        if self._cfg.verbose and self._filters:
            self._log(
                "[filter] "
                + ",  ".join(f"{m} in [{lo:.4g},{hi:.4g}]" for m, lo, hi in self._filters)
            )

    def _log_sample_result(
        self,
        i: int,
        width: int,
        denom: str,
        flag: str,
        *,
        metrics: dict[str, float] | None = None,
        msg: str | None = None,
    ) -> None:
        """Log one sample outcome to stderr (verbose mode only).

        Parameters
        ----------
        i:
            Zero-based attempt index.
        width:
            Field width for left-padding the attempt counter.
        denom:
            Denominator string (e.g. ``"20"`` or ``"∞"``).
        flag:
            Short status tag: ``"PASS"``, ``"skip"``, ``"invalid"``, or
            ``"warn"``.
        metrics:
            Metric dict to append when *flag* is ``"PASS"`` or ``"skip"``.
        msg:
            Free-form message to append (used for ``"invalid"`` and
            ``"warn"``).
        """
        if not self._cfg.verbose:
            return
        prefix = f"[{i + 1:>{width}}/{denom}:{flag}]"
        if metrics is not None:
            self._log(prefix + "  " + "  ".join(f"{k}={_fmt(v)}" for k, v in metrics.items()))
        elif msg is not None:
            self._log(f"{prefix} {msg}")
        else:
            self._log(prefix)

    def _log_summary(
        self,
        n_attempted: int,
        n_passed: int,
        n_invalid: int,
        n_rejected_filter: int,
    ) -> None:
        """Log the end-of-run summary line to stderr (verbose mode only).

        Parameters
        ----------
        n_attempted:
            Total placement attempts made.
        n_passed:
            Number of structures that passed all filters.
        n_invalid:
            Attempts rejected by the charge/multiplicity parity check.
        n_rejected_filter:
            Attempts rejected by metric filters.
        """
        if not self._cfg.verbose:
            return
        self._log(
            f"[summary] attempted={n_attempted}  passed={n_passed}  "
            f"rejected_parity={n_invalid}  rejected_filter={n_rejected_filter}"
        )

    def _resolve_cutoff(self, override: float | None) -> float:
        if override is not None:
            if self._cfg.verbose:
                self._log(f"[cutoff] {override:.3f} Å (user-specified)")
            return override
        radii = np.array([_cov_radius_ang(s) for s in self._element_pool])
        # O(N) approximation: median(r_i + r_j) ≈ 2 × median(r_i).
        # The element pool is at most 106 elements, so O(N²) would be fast here
        # too; we still use the O(N) form for consistency with _metrics.py and
        # to match the formula documented in architecture.md (v0.2.6).
        median_sum = float(np.median(radii)) * 2.0
        cutoff = self._cfg.cov_scale * 1.5 * median_sum
        if self._cfg.verbose:
            self._log(
                f"[cutoff] {cutoff:.3f} Å (auto: cov_scale={self._cfg.cov_scale} × 1.5 × "
                f"median(r_i+r_j)≈{median_sum:.3f} Å)"
            )
        return cutoff

    def _sample_atoms(self, rng: random.Random) -> list[str]:
        """Sample *n_atoms* element symbols respecting fractions and count bounds.

        Algorithm
        ---------
        1. If no fractions/min/max are configured, falls back to the
           original uniform ``rng.choice`` per atom (preserves seed parity).
        2. Otherwise: place the guaranteed minimum-count atoms first
           (``element_min_counts``), fill remaining slots by weighted random
           sampling (``element_fractions``), excluding elements that have
           reached their ``element_max_counts`` cap, then shuffle.

        Raises
        ------
        RuntimeError
            When the constraints cannot be satisfied (e.g. all remaining
            elements are capped and there are still slots to fill).
        """
        pool = self._element_pool
        min_c = self._element_min_counts
        max_c = self._element_max_counts
        n = len(pool)
        uniform = n > 0 and all(abs(w - 1.0 / n) < 1e-12 for w in self._element_weights)

        # Fast path: uniform weights, no bounds → identical to original behavior
        if uniform and not min_c and not max_c:
            return [rng.choice(pool) for _ in range(self._cfg.n_atoms)]

        weights = self._element_weights

        # ── Step 1: fill guaranteed minimum counts ──────────────────────
        counts: dict[str, int] = {sym: min_c.get(sym, 0) for sym in pool}
        atoms: list[str] = []
        for sym in pool:
            atoms.extend([sym] * counts[sym])

        remaining = self._cfg.n_atoms - len(atoms)

        # ── Step 2: weighted sampling for remaining slots ────────────────
        for _ in range(remaining):
            # Build eligible pool (not yet capped)
            eligible: list[str] = []
            eligible_w: list[float] = []
            for sym, w in zip(pool, weights, strict=True):
                cap = max_c.get(sym, None)
                if cap is None or counts.get(sym, 0) < cap:
                    eligible.append(sym)
                    eligible_w.append(w)

            if not eligible:
                raise RuntimeError(
                    "element_max_counts constraints cannot be satisfied: "
                    "all elements are capped before n_atoms is reached."
                )

            # Normalise eligible weights and do a weighted choice
            total_w = sum(eligible_w)
            cum: list[float] = []
            acc = 0.0
            for w in eligible_w:
                acc += w / total_w
                cum.append(acc)

            r = rng.random()
            chosen = eligible[-1]
            for sym, c in zip(eligible, cum, strict=True):
                if r <= c:
                    chosen = sym
                    break

            counts[chosen] = counts.get(chosen, 0) + 1
            atoms.append(chosen)

        # ── Step 3: shuffle so forced atoms don't cluster at front ───────
        rng.shuffle(atoms)
        return atoms

    # ------------------------------------------------------------------ #
    # Public properties                                                    #
    # ------------------------------------------------------------------ #

    @property
    def element_pool(self) -> list[str]:
        """A copy of the resolved element pool (list of symbols)."""
        return list(self._element_pool)

    @property
    def cutoff(self) -> float:
        """Distance cutoff in Å used for Steinhardt and graph metrics."""
        return self._cutoff

    @property
    def config(self) -> GeneratorConfig:
        """The :class:`GeneratorConfig` that was used to construct this generator."""
        return self._cfg

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to ``_cfg`` for all :class:`GeneratorConfig` fields.

        This allows code written against the old kwargs-based API
        (e.g. ``gen.n_atoms``, ``gen.seed``) to continue working without
        modification after the migration to config-based construction.
        """
        # Avoid infinite recursion on _cfg itself during __init__
        if name == "_cfg":
            raise AttributeError(name)
        try:
            return getattr(self._cfg, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None

    # ------------------------------------------------------------------ #
    # Generation                                                           #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Internal placement dispatch                                          #
    # ------------------------------------------------------------------ #

    def _place_one(
        self,
        atoms_list: list[str],
        rng: random.Random,
    ) -> tuple[list[str], list[Vec3], str | None]:
        """Run the mode-specific placement and return (atoms, positions, center_sym).

        Raises
        ------
        RuntimeError, ValueError
            Propagated from the underlying placement functions.
        """
        bond_lo, bond_hi = self._cfg.bond_range
        shell_lo, shell_hi = self._cfg.shell_radius
        coord_lo, coord_hi = self._cfg.coord_range

        center_sym: str | None = None
        if self._cfg.mode == "gas":
            assert self._cfg.region is not None  # guaranteed by __init__ validation
            atoms_out, positions = place_gas(
                atoms_list,
                self._cfg.region,
                rng,
            )
        elif self._cfg.mode == "chain":
            atoms_out, positions = place_chain(
                atoms_list,
                bond_lo,
                bond_hi,
                self._cfg.branch_prob,
                self._cfg.chain_persist,
                rng,
                chain_bias=self._cfg.chain_bias,
            )
        elif self._cfg.mode == "maxent":
            assert self._cfg.region is not None
            atoms_out, positions = place_maxent(
                atoms_list,
                self._cfg.region,
                self._cfg.cov_scale,
                rng,
                maxent_steps=self._cfg.maxent_steps,
                maxent_lr=self._cfg.maxent_lr,
                maxent_cutoff_scale=self._cfg.maxent_cutoff_scale,
                trust_radius=self._cfg.trust_radius,
                convergence_tol=self._cfg.convergence_tol,
                seed=self._cfg.seed,
            )
        else:  # shell
            center_sym = (
                self._fixed_center_sym
                if self._fixed_center_sym is not None
                else rng.choice(atoms_list)
            )
            atoms_out, positions = place_shell(
                atoms_list,
                center_sym,
                coord_lo,
                coord_hi,
                shell_lo,
                shell_hi,
                bond_lo,
                bond_hi,
                rng,
            )
        return atoms_out, positions, center_sym

    # ------------------------------------------------------------------ #
    # Generation                                                           #
    # ------------------------------------------------------------------ #

    def _stream_with_stats(self) -> tuple[Iterator[Structure], dict[str, int]]:
        """Run the generation loop and expose both structures and run statistics.

        Returns a ``(structures_iterator, stats_dict)`` pair.  The
        *stats_dict* is a mutable mapping that is populated **in-place** as
        the iterator is consumed; callers that need statistics must exhaust
        the iterator before reading from it.

        This is the single source of truth for all generation logic.
        :meth:`stream` and :meth:`generate` are thin wrappers around it,
        which eliminates the hidden coupling that previously existed via the
        ``_last_run_stats`` instance variable.

        Returns
        -------
        tuple[Iterator[Structure], dict[str, int]]
            ``(it, stats)`` where *it* yields passing structures and *stats*
            is populated with ``n_attempted``, ``n_passed``,
            ``n_rejected_parity``, and ``n_rejected_filter`` once *it* is
            exhausted.
        """
        stats: dict[str, int] = {}

        def _inner() -> Iterator[Structure]:
            rng = random.Random(self._cfg.seed)
            self._log_filter_header()

            do_add_h = ("H" in self._element_pool) and self._cfg.add_hydrogen
            n_passed = n_invalid = n_attempted = n_rejected_filter = 0
            unlimited = self._cfg.n_samples == 0
            denom = "∞" if unlimited else str(self._cfg.n_samples)
            width = len(denom)

            while True:
                # Stop conditions
                if not unlimited and n_attempted >= self._cfg.n_samples:
                    break
                if self._cfg.n_success is not None and n_passed >= self._cfg.n_success:
                    break

                i = n_attempted
                n_attempted += 1

                atoms_list = self._sample_atoms(rng)
                if do_add_h:
                    atoms_list = add_hydrogen(atoms_list, rng)

                ok, val_msg = validate_charge_mult(atoms_list, self._cfg.charge, self._cfg.mult)
                if not ok:
                    n_invalid += 1
                    self._log_sample_result(i, width, denom, "invalid", msg=val_msg)
                    continue

                try:
                    atoms_out, positions, center_sym = self._place_one(atoms_list, rng)
                except (RuntimeError, ValueError) as exc:
                    if self._cfg.verbose:
                        self._log(f"[ERROR] sample {i + 1}: {exc}")
                    raise

                # ── Optional affine transform (applied once, before relax) ──
                if self._cfg.affine_strength > 0.0:
                    positions = _affine_move(
                        positions,
                        0.0,
                        self._cfg.affine_strength,
                        rng,
                        affine_stretch=self._cfg.affine_stretch,
                        affine_shear=self._cfg.affine_shear,
                        affine_jitter=self._cfg.affine_jitter,
                    )

                positions, converged = relax_positions(
                    atoms_out,
                    positions,
                    self._cfg.cov_scale,
                    self._cfg.relax_cycles,
                    seed=self._cfg.seed,
                )
                if not converged:
                    self._log_sample_result(
                        i,
                        width,
                        denom,
                        "warn",
                        msg=(
                            f"relax_positions did not converge in {self._cfg.relax_cycles} cycles."
                        ),
                    )

                metrics = compute_all_metrics(
                    atoms_out,
                    positions,
                    self._cfg.n_bins,
                    self._cfg.w_atom,
                    self._cfg.w_spatial,
                    self._cutoff,
                    self._cfg.cov_scale,
                )
                passed = passes_filters(metrics, self._filters)
                self._log_sample_result(
                    i,
                    width,
                    denom,
                    "PASS" if passed else "skip",
                    metrics=metrics,
                )
                if not passed:
                    n_rejected_filter += 1
                    continue

                n_passed += 1
                yield Structure(
                    atoms=atoms_out,
                    positions=positions,
                    charge=self._cfg.charge,
                    mult=self._cfg.mult,
                    metrics=metrics,
                    mode=self._cfg.mode,
                    sample_index=n_passed,
                    center_sym=center_sym if self._cfg.mode == "shell" else None,
                    seed=self._cfg.seed,
                )

            n_skip = n_attempted - n_passed - n_invalid
            self._log_summary(n_attempted, n_passed, n_invalid, n_skip)

            # ── warnings.warn for noteworthy outcomes ──────────────────────
            # Fires regardless of verbose so that downstream consumers
            # (ASE, HT pipelines) receive machine-visible signals even when
            # PASTED is not in verbose mode.
            #
            # Parity warnings fire only when n_passed == 0 (complete failure).
            # Partial parity rejection where some structures still passed is
            # expected behavior for mixed-element pools and does not require
            # a warning — the verbose summary line already reports the counts.
            if n_invalid > 0 and n_passed == 0:
                if n_rejected_filter == 0:
                    # Pure parity failure — no attempt reached the filter stage.
                    warnings.warn(
                        f"All {n_attempted} attempt(s) were rejected by the charge/"
                        f"multiplicity parity check ({n_invalid} invalid). "
                        f"No structures were generated. "
                        f"Check that your element pool can satisfy "
                        f"charge={self._cfg.charge}, mult={self._cfg.mult}.",
                        UserWarning,
                        stacklevel=4,
                    )
                else:
                    # Mixed failure: some attempts failed parity AND some failed
                    # filters.  Report both causes so users don't only debug the
                    # element pool.
                    warnings.warn(
                        f"{n_invalid} of {n_attempted} attempt(s) were rejected by the "
                        f"charge/multiplicity parity check, and the remaining "
                        f"{n_rejected_filter} that passed parity were rejected by metric "
                        f"filters. No structures were generated. "
                        f"Check your element pool (charge={self._cfg.charge}, "
                        f"mult={self._cfg.mult}) and relax --filter thresholds.",
                        UserWarning,
                        stacklevel=4,
                    )

            if n_passed == 0 and n_invalid == 0:
                warnings.warn(
                    f"No structures passed the metric filters after "
                    f"{n_attempted} attempt(s) "
                    f"({n_skip} rejected by filters). "
                    f"Try relaxing the --filter thresholds or increasing n_samples.",
                    UserWarning,
                    stacklevel=4,
                )
            elif (
                self._cfg.n_success is not None
                and n_passed < self._cfg.n_success
                and not unlimited
            ):
                warnings.warn(
                    f"Attempt budget exhausted ({n_attempted} attempts) before "
                    f"reaching n_success={self._cfg.n_success}; "
                    f"only {n_passed} structure(s) collected. "
                    f"Increase n_samples or relax filters.",
                    UserWarning,
                    stacklevel=4,
                )

            # Populate the shared stats dict now that the loop is complete.
            stats.update(
                {
                    "n_attempted": n_attempted,
                    "n_passed": n_passed,
                    "n_rejected_parity": n_invalid,
                    "n_rejected_filter": n_rejected_filter,
                }
            )

        return _inner(), stats

    def stream(self) -> Iterator[Structure]:
        """Generate structures one by one, yielding each that passes all filters.

        Unlike :meth:`generate`, structures are yielded immediately as they
        pass, so callers can write output or stop early without waiting for
        all attempts to complete.

        Respects both *n_samples* (maximum attempts) and *n_success* (target
        number of passing structures):

        - If *n_success* is set, the iterator stops as soon as that many
          structures have been yielded — even if *n_samples* attempts have
          not been exhausted.
        - If *n_samples* is ``0`` (unlimited), the iterator runs until
          *n_success* structures have been yielded.
        - If *n_samples* attempts are exhausted before *n_success* is
          reached, a warning is emitted to *stderr* and the iterator ends.

        Each call creates a fresh :class:`random.Random` seeded with
        ``self._cfg.seed``, so repeated calls with the same seed are
        reproducible.

        Yields
        ------
        Structure
            Each structure that passed all filters, in generation order.

        Examples
        --------
        Write structures to a file as they are found::

            gen = StructureGenerator(
                n_atoms=12, charge=0, mult=1,
                mode="gas", region="sphere:9",
                elements="1-30", n_success=10, n_samples=500, seed=42,
            )
            for s in gen.stream():
                s.write_xyz("out.xyz")
        """
        it, _ = self._stream_with_stats()
        return it

    def generate(self) -> GenerationResult:
        """Generate structures and return a :class:`GenerationResult`.

        Collects all structures yielded by the internal generation loop,
        attaches generation metadata (attempt counts, rejection breakdowns),
        and returns a :class:`GenerationResult` that behaves like a
        ``list[Structure]`` in all normal usage while also carrying the
        diagnostics needed for automated pipelines.

        Run statistics (``n_attempted``, ``n_passed``, etc.) are obtained
        directly from :meth:`_stream_with_stats` rather than via a shared
        instance variable, so there is no hidden coupling between
        :meth:`stream` and :meth:`generate`.  Calling one does not affect
        the other, and partial iteration of :meth:`stream` cannot leave
        stale counters for a subsequent :meth:`generate` call.

        :class:`GenerationResult` supports the full ``list`` interface
        (indexing, iteration, ``len``, ``bool``) so existing code that
        does ``result[0]`` or ``for s in result`` continues to work
        without modification.

        Warnings are also emitted via :func:`warnings.warn` (category
        :class:`UserWarning`) when:

        - Any attempts are rejected by the charge/multiplicity parity check.
        - No structures pass the metric filters.
        - The attempt budget is exhausted before ``n_success`` is reached.

        Each call creates a fresh :class:`random.Random` seeded with
        ``self._cfg.seed``, so repeated calls with the same seed are
        reproducible.

        Returns
        -------
        GenerationResult
            Wraps the list of passing structures together with generation
            metadata.  Use ``result.structures`` for the raw list or
            ``result.summary()`` for a one-line diagnostic string.

        Examples
        --------
        Drop-in list usage::

            result = gen.generate()
            for s in result:
                print(s.to_xyz())

        Metadata access::

            result = gen.generate()
            if result.n_rejected_parity > 0:
                print(result.summary())
        """
        it, stats = self._stream_with_stats()
        structures = list(it)  # exhausts the iterator, populating stats
        return GenerationResult(
            structures=structures,
            n_attempted=stats.get("n_attempted", len(structures)),
            n_passed=stats.get("n_passed", len(structures)),
            n_rejected_parity=stats.get("n_rejected_parity", 0),
            n_rejected_filter=stats.get("n_rejected_filter", 0),
            n_success_target=self._cfg.n_success,
        )

    # ------------------------------------------------------------------ #
    # Iteration support                                                    #
    # ------------------------------------------------------------------ #

    def __iter__(self) -> Iterator[Structure]:
        """Iterate over generated structures (delegates to :meth:`stream`)."""
        return self.stream()

    def __repr__(self) -> str:
        return (
            f"StructureGenerator("
            f"n_atoms={self._cfg.n_atoms}, mode={self._cfg.mode!r}, "
            f"charge={self._cfg.charge:+d}, mult={self._cfg.mult}, "
            f"n_samples={self._cfg.n_samples}, "
            f"n_success={self._cfg.n_success}, "
            f"pool_size={len(self._element_pool)})"
        )


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def read_xyz(
    source: str | Path,
    *,
    recompute_metrics: bool = True,
    cutoff: float | None = None,
    n_bins: int = 20,
    w_atom: float = 0.5,
    w_spatial: float = 0.5,
    cov_scale: float = 1.0,
) -> list[Structure]:
    """Read one or more structures from an XYZ file or string.

    Convenience wrapper around :meth:`Structure.from_xyz` that reads **all
    frames** from a (possibly multi-frame) XYZ source and returns them as a
    list.  Both plain XYZ and PASTED extended XYZ are supported.

    Parameters
    ----------
    source:
        Path to an XYZ file **or** a raw XYZ string.
    recompute_metrics:
        Recompute all disorder metrics after loading each structure
        (default: ``True``).
    cutoff:
        Distance cutoff (Å) for metric computation.  Auto-computed from
        each structure's element pool when ``None``.
    n_bins:
        Histogram bins for ``H_spatial`` / ``RDF_dev`` (default: 20).
    w_atom:
        Weight of ``H_atom`` in ``H_total`` (default: 0.5).
    w_spatial:
        Weight of ``H_spatial`` in ``H_total`` (default: 0.5).
    cov_scale:
        Minimum distance scale factor used for metrics (default: 1.0).

    Returns
    -------
    list[Structure]
        One :class:`Structure` per frame, in file order.

    Examples
    --------
    Load a PASTED output file and pass the first structure to the
    optimizer::

        from pasted import read_xyz, StructureOptimizer

        structs = read_xyz("results.xyz")
        opt = StructureOptimizer(
            n_atoms=len(structs[0]),
            charge=structs[0].charge,
            mult=structs[0].mult,
            objective={"H_total": 1.0},
            elements=list(set(structs[0].atoms)),
            max_steps=3000,
            seed=42,
        )
        result = opt.run(initial=structs[0])

    Compose with :class:`GenerationResult` via ``+``::

        from pasted import read_xyz, generate

        existing = generate(n_atoms=10, charge=0, mult=1,
                            mode="gas", region="sphere:9",
                            elements="6,7,8", n_samples=5, seed=0)
        loaded   = read_xyz("previous_run.xyz")
        # loaded is a list[Structure]; wrap manually if needed:
        from pasted import GenerationResult
        all_structs = existing + GenerationResult(structures=loaded,
                                                  n_passed=len(loaded),
                                                  n_attempted=len(loaded))
    """
    p = Path(source) if not isinstance(source, str) or "\n" not in str(source) else None
    text = p.read_text() if (p is not None and p.exists()) else str(source)

    frames = parse_xyz(text)
    result: list[Structure] = []
    for atoms, positions, charge, mult, embedded_metrics in frames:
        if recompute_metrics:
            cut = cutoff
            if cut is None:
                radii = np.array([_cov_radius_ang(a) for a in atoms])
                # O(N) approximation: median(r_i + r_j) ≈ 2 × median(r_i).
                cut = cov_scale * 1.5 * float(np.median(radii)) * 2.0
            metrics = compute_all_metrics(
                atoms, positions, n_bins, w_atom, w_spatial, cut, cov_scale
            )
        else:
            metrics = embedded_metrics

        result.append(
            Structure(
                atoms=list(atoms),
                positions=list(positions),
                charge=charge,
                mult=mult,
                metrics=metrics,
                mode="loaded_xyz",
            )
        )
    return result


def generate(
    config: GeneratorConfig | None = None,
    *,
    n_atoms: int | None = None,
    charge: int | None = None,
    mult: int | None = None,
    **kwargs: Any,
) -> GenerationResult:
    """Create a :class:`StructureGenerator` and immediately call
    :meth:`~StructureGenerator.generate`.

    Two calling conventions are supported:

    **Config-based (recommended for new code):**
    Pass a :class:`GeneratorConfig` as the first positional argument.
    Provides full mypy / IDE type-checking on every field::

        from pasted import generate, GeneratorConfig

        cfg = GeneratorConfig(n_atoms=10, charge=0, mult=1,
                              mode="gas", region="sphere:8",
                              elements="6,7,8", n_samples=20, seed=0)
        result = generate(cfg)

    **Keyword-based (backward-compatible, original API):**
    Pass all parameters as keyword arguments.  ``n_atoms``, ``charge``,
    and ``mult`` are required; all others are optional::

        result = generate(
            n_atoms=10, charge=0, mult=1,
            mode="gas", region="sphere:8",
            elements="6,7,8", n_samples=20, seed=0,
        )

    Both forms may not be mixed: if *config* is given, all other keyword
    arguments are ignored.

    Parameters
    ----------
    config:
        A fully-populated :class:`GeneratorConfig` instance.  When given,
        all other keyword arguments are ignored.
    n_atoms:
        Number of atoms per structure (**required** when *config* is ``None``).
    charge:
        Total system charge (**required** when *config* is ``None``).
    mult:
        Spin multiplicity 2S+1 (**required** when *config* is ``None``).
    **kwargs:
        Any optional :class:`GeneratorConfig` field, e.g.
        ``mode``, ``region``, ``elements``, ``n_samples``, ``seed``,
        ``filters``, ``affine_strength``, …
        Ignored when *config* is provided.

    Returns
    -------
    GenerationResult
        A list-compatible object containing the structures that passed all
        filters plus metadata about the generation run.
    """
    if config is not None:
        return StructureGenerator(config).generate()

    # Backward-compatible kwargs path
    if n_atoms is None or charge is None or mult is None:
        raise TypeError(
            "generate() requires n_atoms, charge, and mult when no GeneratorConfig is given."
        )
    cfg = GeneratorConfig(n_atoms=n_atoms, charge=charge, mult=mult, **kwargs)
    return StructureGenerator(cfg).generate()
