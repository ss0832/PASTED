"""
pasted._generator
=================
High-level API:

- :class:`Structure`          — dataclass holding one generated structure.
- :class:`StructureGenerator` — stateful generator (class API).
- :func:`generate`            — convenience functional wrapper.
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ._atoms import (
    _Z_TO_SYM,
    ATOMIC_NUMBERS,
    _cov_radius_ang,
    default_element_pool,
    parse_element_spec,
    parse_filter,
    validate_charge_mult,
)
from ._io import _fmt, format_xyz
from ._metrics import compute_all_metrics, passes_filters
from ._placement import (
    Vec3,
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
        Placement mode used (``"gas"``, ``"chain"``, or ``"shell"``).
    sample_index:
        1-based index within the batch of structures that passed filters.
    center_sym:
        Element symbol of the shell center atom (shell mode only).
    seed:
        Random seed used for generation (``None`` if unseeded).
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

    def __repr__(self) -> str:
        counts = Counter(self.atoms)
        comp = "".join(f"{sym}{n}" if n > 1 else sym for sym, n in sorted(counts.items()))
        h_total = self.metrics.get("H_total", float("nan"))
        return f"Structure(n={len(self)}, comp={comp!r}, mode={self.mode!r}, H_total={h_total:.3f})"


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
        Placement mode: ``"gas"`` (default), ``"chain"``, or ``"shell"``.
    region:
        [gas] Region spec: ``"sphere:R"`` | ``"box:L"`` | ``"box:LX,LY,LZ"``.
        Required when *mode="gas"*.
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
        Element pool.  A spec string such as ``"1-30"`` or ``"6,7,8"``, an
        explicit list of element symbols, or ``None`` for all Z = 1–106.
    cov_scale:
        Minimum-distance scale factor: ``d_min(i,j) = cov_scale × (r_i + r_j)``
        using Pyykkö (2009) single-bond covalent radii.  Default: ``1.0``.
    relax_cycles:
        Maximum repulsion-relaxation iterations (default: 1500).
    add_hydrogen:
        Automatically append H atoms when H is in the pool but the sampled
        composition contains none (default: ``True``).
    n_samples:
        Maximum number of placement attempts (default: 1).
        Use ``0`` to allow unlimited attempts (only valid when *n_success*
        is also set, otherwise a :exc:`ValueError` is raised).
    n_success:
        Target number of structures that must pass all filters before
        generation stops (default: ``None``).

        - ``None`` → generate exactly *n_samples* attempts and return all
          that passed (original behaviour).
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
    Class API::

        from pasted import StructureGenerator

        gen = StructureGenerator(
            n_atoms=12, charge=0, mult=1,
            mode="gas", region="sphere:9",
            elements="1-30", n_samples=50, seed=42,
            filters=["H_total:2.0:-"],
        )
        structures = gen.generate()
        for s in structures:
            print(s)

    Functional API::

        from pasted import generate

        structures = generate(
            n_atoms=12, charge=0, mult=1,
            mode="chain", elements="6,7,8",
            n_samples=20, seed=0,
        )
    """

    def __init__(
        self,
        *,
        n_atoms: int,
        charge: int,
        mult: int,
        mode: str = "gas",
        region: str | None = None,
        branch_prob: float = 0.3,
        chain_persist: float = 0.5,
        chain_bias: float = 0.0,
        bond_range: tuple[float, float] = (1.2, 1.6),
        center_z: int | None = None,
        coord_range: tuple[int, int] = (4, 8),
        shell_radius: tuple[float, float] = (1.8, 2.5),
        elements: str | list[str] | None = None,
        cov_scale: float = 1.0,
        relax_cycles: int = 1500,
        add_hydrogen: bool = True,
        n_samples: int = 1,
        n_success: int | None = None,
        seed: int | None = None,
        n_bins: int = 20,
        w_atom: float = 0.5,
        w_spatial: float = 0.5,
        cutoff: float | None = None,
        filters: list[str] | None = None,
        verbose: bool = False,
    ) -> None:
        if mode not in ("gas", "chain", "shell", "maxent"):
            raise ValueError(
                f"mode must be 'gas', 'chain', 'shell', or 'maxent'; got {mode!r}"
            )
        if mode in ("gas", "maxent") and region is None:
            raise ValueError("region is required when mode='gas' or mode='maxent'")

        self.n_atoms = n_atoms
        self.charge = charge
        self.mult = mult
        self.mode = mode
        self.region = region
        self.branch_prob = branch_prob
        self.chain_persist = chain_persist
        self.chain_bias = chain_bias
        self.bond_range = bond_range
        self.center_z = center_z
        self.coord_range = coord_range
        self.shell_radius = shell_radius
        self.cov_scale = cov_scale
        self.relax_cycles = relax_cycles
        self._add_hydrogen = add_hydrogen
        self.n_samples = n_samples
        self.n_success = n_success
        self.seed = seed
        self.n_bins = n_bins
        self.w_atom = w_atom
        self.w_spatial = w_spatial
        self.verbose = verbose

        # ── n_samples / n_success validation ────────────────────────────
        if n_samples == 0 and n_success is None:
            raise ValueError(
                "n_samples=0 (unlimited) requires n_success to be set; "
                "otherwise generation would run forever."
            )
        if n_success is not None and n_success < 1:
            raise ValueError(f"n_success must be >= 1; got {n_success}.")

        # ── Element pool ────────────────────────────────────────────────
        if elements is None:
            self._element_pool: list[str] = default_element_pool()
        elif isinstance(elements, str):
            self._element_pool = parse_element_spec(elements)
        else:
            self._element_pool = list(elements)

        # ── Filters ─────────────────────────────────────────────────────
        self._filters: list[tuple[str, float, float]] = [parse_filter(f) for f in (filters or [])]

        # ── Cutoff ──────────────────────────────────────────────────────
        self._cutoff: float = self._resolve_cutoff(cutoff)

        # ── Shell center ─────────────────────────────────────────────────
        self._fixed_center_sym: str | None = None
        if mode == "shell" and center_z is not None:
            if center_z not in _Z_TO_SYM:
                raise ValueError(f"center_z={center_z}: unknown atomic number.")
            sym = _Z_TO_SYM[center_z]
            if sym not in self._element_pool:
                raise ValueError(f"center_z={center_z} ({sym}) is not in the element pool.")
            self._fixed_center_sym = sym

        if self.verbose:
            self._log(f"[pool] {len(self._element_pool)} elements in pool")
            if mode == "shell":
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

    def _resolve_cutoff(self, override: float | None) -> float:
        if override is not None:
            if self.verbose:
                self._log(f"[cutoff] {override:.3f} Å (user-specified)")
            return override
        radii = [_cov_radius_ang(s) for s in self._element_pool]
        pair_sums = sorted(ra + rb for i, ra in enumerate(radii) for rb in radii[i:])
        median_sum = pair_sums[len(pair_sums) // 2]
        cutoff = self.cov_scale * 1.5 * median_sum
        if self.verbose:
            self._log(
                f"[cutoff] {cutoff:.3f} Å (auto: cov_scale={self.cov_scale} × 1.5 × "
                f"median(r_i+r_j)={median_sum:.3f} Å)"
            )
        return cutoff

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
        bond_lo, bond_hi = self.bond_range
        shell_lo, shell_hi = self.shell_radius
        coord_lo, coord_hi = self.coord_range

        center_sym: str | None = None
        if self.mode == "gas":
            assert self.region is not None  # guaranteed by __init__ validation
            atoms_out, positions = place_gas(
                atoms_list,
                self.region,
                rng,
            )
        elif self.mode == "chain":
            atoms_out, positions = place_chain(
                atoms_list,
                bond_lo,
                bond_hi,
                self.branch_prob,
                self.chain_persist,
                rng,
                chain_bias=self.chain_bias,
            )
        elif self.mode == "maxent":
            assert self.region is not None
            atoms_out, positions = place_maxent(
                atoms_list,
                self.region,
                self.cov_scale,
                rng,
                seed=self.seed,
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
        ``self.seed``, so repeated calls with the same seed are reproducible.

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
        rng = random.Random(self.seed)

        if self.verbose and self._filters:
            self._log(
                "[filter] "
                + ",  ".join(f"{m} in [{lo:.4g},{hi:.4g}]" for m, lo, hi in self._filters)
            )

        do_add_h = ("H" in self._element_pool) and self._add_hydrogen
        n_passed = n_invalid = n_attempted = 0
        unlimited = (self.n_samples == 0)
        denom = "∞" if unlimited else str(self.n_samples)
        width = len(denom)

        while True:
            # Stop conditions
            if not unlimited and n_attempted >= self.n_samples:
                break
            if self.n_success is not None and n_passed >= self.n_success:
                break

            i = n_attempted
            n_attempted += 1

            atoms_list = [rng.choice(self._element_pool) for _ in range(self.n_atoms)]
            if do_add_h:
                atoms_list = add_hydrogen(atoms_list, rng)

            ok, val_msg = validate_charge_mult(atoms_list, self.charge, self.mult)
            if not ok:
                n_invalid += 1
                if self.verbose:
                    self._log(f"[{i + 1:>{width}}/{denom}:invalid] {val_msg}")
                continue

            try:
                atoms_out, positions, center_sym = self._place_one(atoms_list, rng)
            except (RuntimeError, ValueError) as exc:
                if self.verbose:
                    self._log(f"[ERROR] sample {i + 1}: {exc}")
                raise

            positions, converged = relax_positions(
                atoms_out, positions, self.cov_scale, self.relax_cycles, seed=self.seed
            )
            if not converged and self.verbose:
                self._log(
                    f"[{i + 1:>{width}}/{denom}:warn] "
                    f"relax_positions did not converge in {self.relax_cycles} cycles."
                )

            metrics = compute_all_metrics(
                atoms_out,
                positions,
                self.n_bins,
                self.w_atom,
                self.w_spatial,
                self._cutoff,
            )
            passed = passes_filters(metrics, self._filters)
            if self.verbose:
                flag = "PASS" if passed else "skip"
                self._log(
                    f"[{i + 1:>{width}}/{denom}:{flag}]  "
                    + "  ".join(f"{k}={_fmt(v)}" for k, v in metrics.items())
                )
            if not passed:
                continue

            n_passed += 1
            yield Structure(
                atoms=atoms_out,
                positions=positions,
                charge=self.charge,
                mult=self.mult,
                metrics=metrics,
                mode=self.mode,
                sample_index=n_passed,
                center_sym=center_sym if self.mode == "shell" else None,
                seed=self.seed,
            )

        if self.verbose:
            n_skip = n_attempted - n_passed - n_invalid
            self._log(
                f"[summary] attempted={n_attempted}  passed={n_passed}  "
                f"filtered_out={n_skip}  invalid_charge_mult={n_invalid}"
            )
            if self.n_success is not None and n_passed < self.n_success:
                self._log(
                    f"[warning] Reached attempt limit ({n_attempted}) before collecting "
                    f"n_success={self.n_success} structures ({n_passed} collected). "
                    f"Try increasing n_samples or relaxing filters."
                )
            elif n_passed == 0:
                self._log(
                    "[warning] No structures passed. Try relaxing filters or increasing n_samples."
                )

    def generate(self) -> list[Structure]:
        """Generate structures and return those that pass all filters.

        Delegates to :meth:`stream` and collects results into a list.
        Each call creates a fresh :class:`random.Random` seeded with
        ``self.seed``, so repeated calls with the same seed are reproducible.

        Returns
        -------
        list[Structure]
            Structures that passed all filters, in generation order.
        """
        return list(self.stream())

    # ------------------------------------------------------------------ #
    # Iteration support                                                    #
    # ------------------------------------------------------------------ #

    def __iter__(self) -> Iterator[Structure]:
        """Iterate over generated structures (delegates to :meth:`stream`)."""
        return self.stream()

    def __repr__(self) -> str:
        return (
            f"StructureGenerator("
            f"n_atoms={self.n_atoms}, mode={self.mode!r}, "
            f"charge={self.charge:+d}, mult={self.mult}, "
            f"n_samples={self.n_samples}, "
            f"n_success={self.n_success}, "
            f"pool_size={len(self._element_pool)})"
        )


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def generate(
    *,
    n_atoms: int,
    charge: int,
    mult: int,
    mode: str = "gas",
    region: str | None = None,
    branch_prob: float = 0.3,
    chain_persist: float = 0.5,
    chain_bias: float = 0.0,
    bond_range: tuple[float, float] = (1.2, 1.6),
    center_z: int | None = None,
    coord_range: tuple[int, int] = (4, 8),
    shell_radius: tuple[float, float] = (1.8, 2.5),
    elements: str | list[str] | None = None,
    cov_scale: float = 1.0,
    relax_cycles: int = 1500,
    add_hydrogen: bool = True,
    n_samples: int = 1,
    n_success: int | None = None,
    seed: int | None = None,
    n_bins: int = 20,
    w_atom: float = 0.5,
    w_spatial: float = 0.5,
    cutoff: float | None = None,
    filters: list[str] | None = None,
    verbose: bool = False,
) -> list[Structure]:
    """Create a :class:`StructureGenerator` and immediately call
    :meth:`~StructureGenerator.generate`.

    All parameters are forwarded unchanged.  See :class:`StructureGenerator`
    for full documentation.

    Returns
    -------
    list[Structure]
        Structures that passed all filters, in generation order.

    Examples
    --------
    ::

        from pasted import generate

        # 20 random gas-phase structures drawn from C/N/O
        structures = generate(
            n_atoms=10, charge=0, mult=1,
            mode="gas", region="sphere:8",
            elements="6,7,8", n_samples=20, seed=0,
        )

        # Write all to a single XYZ file
        for i, s in enumerate(structures):
            s.write_xyz("out.xyz", append=(i > 0))
    """
    gen = StructureGenerator(
        n_atoms=n_atoms,
        charge=charge,
        mult=mult,
        mode=mode,
        region=region,
        branch_prob=branch_prob,
        chain_persist=chain_persist,
        chain_bias=chain_bias,
        bond_range=bond_range,
        center_z=center_z,
        coord_range=coord_range,
        shell_radius=shell_radius,
        elements=elements,
        cov_scale=cov_scale,
        relax_cycles=relax_cycles,
        add_hydrogen=add_hydrogen,
        n_samples=n_samples,
        n_success=n_success,
        seed=seed,
        n_bins=n_bins,
        w_atom=w_atom,
        w_spatial=w_spatial,
        cutoff=cutoff,
        filters=filters,
        verbose=verbose,
    )
    return gen.generate()
