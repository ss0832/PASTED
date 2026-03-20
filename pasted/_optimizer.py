"""
pasted._optimizer
=================
Objective-based structure optimization.

Three methods
-------------
``"annealing"``
    Simulated Annealing with exponential cooling from *T_start* to *T_end*.
``"basin_hopping"``
    Basin-Hopping: each step applies a more thorough relaxation (3x relax
    cycles) before the Metropolis acceptance test.  Temperature is held
    constant at *T_start*.
``"parallel_tempering"``
    Parallel Tempering (replica exchange): *n_replicas* independent Markov
    chains run at geometrically spaced temperatures between *T_start* and
    *T_pt_high*.  Every *swap_interval* steps, adjacent replicas attempt a
    swap via the Metropolis exchange criterion.  The lowest-temperature
    replica benefits from high-temperature replicas crossing energy barriers.
    All replicas final structures are returned in ``OptimizationResult``.

Move types (chosen with equal probability each step)
----------------------------------------------------
Fragment coordinate move
    Compute per-atom Q6.  Atoms whose local Q6 exceeds *frag_threshold*
    are considered "accidentally ordered" and are displaced by a random
    vector of magnitude <= *move_step* Ang.  If no atom exceeds the threshold
    (structure is already fully disordered), a single random atom is moved.
Composition move
    Parity-preserving composition change: swap two atoms with different
    elements (always parity-safe), or replace two atoms simultaneously
    so the net electron-count change is even (parity-preserving fallback).

Objective function
------------------
The objective is **maximised**.  Pass a weight dict or any callable::

    # dict: f = sum(w * metric)
    objective = {"H_atom": 1.0, "H_spatial": 1.0, "Q6": -2.0}

    # callable
    objective = lambda m: m["H_spatial"] - 2.0 * m["Q6"]

Use negative weights to penalise a metric.
"""

from __future__ import annotations

import math
import random
import sys
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

from ._atoms import (
    ALL_METRICS,
    ATOMIC_NUMBERS,
    _cov_radius_ang,
    default_element_pool,
    parse_element_spec,
    validate_charge_mult,
)
from ._ext import HAS_RELAX
from ._ext import relax_positions as _cpp_relax_positions
from ._generator import Structure, StructureGenerator
from ._io import _fmt
from ._metrics import compute_all_metrics, compute_steinhardt_per_atom
from ._placement import Vec3, _affine_move, place_gas, relax_positions

# ---------------------------------------------------------------------------
# Public type alias
# ---------------------------------------------------------------------------

ObjectiveType = dict[str, float] | Callable[[dict[str, float]], float]

# ---------------------------------------------------------------------------
# OptimizationResult
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    """Return value of :meth:`StructureOptimizer.run`.

    Wraps all per-restart results and exposes the best structure as a
    first-class attribute.  Behaves like a ``list[Structure]`` — indexing,
    iteration, ``len``, and ``bool`` all work — so callers that only want
    the best result can access it without changing existing code::

        result = opt.run()
        best   = result.best        # highest-scoring Structure
        best   = result[0]          # same — index 0 is always the best
        for s in result:            # iterate all restarts, best first
            print(s.metrics["H_total"])

    Attributes
    ----------
    all_structures:
        All structures produced by each restart, sorted by objective value
        (highest first).  ``all_structures[0]`` is always the best.
    objective_scores:
        Scalar objective values corresponding to each entry in
        ``all_structures``.
    n_restarts_attempted:
        Number of restarts that were actually run (may be less than
        ``n_restarts`` when initial-structure generation fails).
    method:
        The optimization method used (``"annealing"``,
        ``"basin_hopping"``, or ``"parallel_tempering"``).

    Examples
    --------
    Single-structure usage (backward-compatible)::

        result = opt.run()
        result.best.to_xyz()      # best structure
        result[0].to_xyz()        # same

    All-restarts usage::

        result = opt.run()
        print(result.summary())
        for rank, s in enumerate(result, 1):
            print(f"rank {rank}: H_total={s.metrics['H_total']:.3f}")
    """

    all_structures: list[Structure] = field(default_factory=list)
    objective_scores: list[float] = field(default_factory=list)
    n_restarts_attempted: int = 0
    method: str = "annealing"

    # ------------------------------------------------------------------ #
    # list-compatible interface (best-first order)                        #
    # ------------------------------------------------------------------ #

    @property
    def best(self) -> Structure:
        """The structure with the highest objective value."""
        if not self.all_structures:
            raise RuntimeError("OptimizationResult is empty — all restarts failed.")
        return self.all_structures[0]

    def __len__(self) -> int:
        return len(self.all_structures)

    def __iter__(self) -> Iterator[Structure]:
        return iter(self.all_structures)

    def __getitem__(self, index: int | slice) -> Structure | list[Structure]:
        if isinstance(index, slice):
            return self.all_structures[index]
        return self.all_structures[index]

    def __bool__(self) -> bool:
        return bool(self.all_structures)

    def __repr__(self) -> str:
        if not self.all_structures:
            return f"OptimizationResult(empty, method={self.method!r})"
        best_f = self.objective_scores[0] if self.objective_scores else float("nan")
        return (
            f"OptimizationResult("
            f"restarts={self.n_restarts_attempted}, "
            f"best_f={best_f:.4f}, "
            f"method={self.method!r})"
        )

    # ------------------------------------------------------------------ #
    # Metadata helpers                                                     #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        """Return a human-readable one-line summary of the optimization run.

        Returns
        -------
        str
            E.g. ``"restarts=5  best_f=1.2294  worst_f=0.7823  method='annealing'"``.
        """
        if not self.objective_scores:
            return f"restarts={self.n_restarts_attempted}  no results  method={self.method!r}"
        return (
            f"restarts={self.n_restarts_attempted}"
            f"  best_f={self.objective_scores[0]:.4f}"
            f"  worst_f={self.objective_scores[-1]:.4f}"
            f"  method={self.method!r}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_objective_spec(specs: list[str]) -> dict[str, float]:
    """Parse ``["METRIC:WEIGHT", ...]`` into a weight dict.

    Parameters
    ----------
    specs:
        Each string must be of the form ``"METRIC:WEIGHT"``,
        e.g. ``["H_atom:1.0", "Q6:-2.0"]``.

    Returns
    -------
    dict[str, float]

    Raises
    ------
    ValueError
        On malformed strings or unknown metric names.
    """
    result: dict[str, float] = {}
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Expected 'METRIC:WEIGHT', got {spec!r}")
        metric, weight_s = parts
        if metric not in ALL_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}. Valid: {sorted(ALL_METRICS)}"
            )
        result[metric] = float(weight_s)
    return result


def _eval_objective(
    metrics: dict[str, float],
    objective: ObjectiveType,
) -> float:
    """Evaluate the scalar objective from a metrics dict."""
    if callable(objective):
        return float(objective(metrics))
    return float(sum(w * metrics.get(k, 0.0) for k, w in objective.items()))


def _fragment_move(
    positions: list[Vec3],
    per_atom_q6: np.ndarray,
    frag_threshold: float,
    move_step: float,
    rng: random.Random,
) -> list[Vec3]:
    """Displace atoms whose local Q6 exceeds *frag_threshold*.

    Falls back to moving a single random atom when no atom exceeds the
    threshold (structure is already maximally disordered).
    """
    candidates = [i for i, q in enumerate(per_atom_q6) if q > frag_threshold]
    if not candidates:
        candidates = [rng.randrange(len(positions))]
    new_pos = list(positions)
    for i in candidates:
        x, y, z = positions[i]
        new_pos[i] = (
            x + rng.uniform(-move_step, move_step),
            y + rng.uniform(-move_step, move_step),
            z + rng.uniform(-move_step, move_step),
        )
    return new_pos


def _composition_move(
    atoms: list[str],
    element_pool: list[str],
    rng: random.Random,
    charge: int = 0,
    mult: int = 1,
) -> list[str]:
    """Propose a parity-preserving composition change.

    Two move types are attempted in order:

    1. **Swap**: exchange the element types of two atoms with different
       elements.  Since the total electron count is unchanged, parity is
       always preserved.
    2. **Parity-preserving replace**: when all atoms are the same element
       (so no swap is possible), replace *two* atoms simultaneously so that
       the net change in electron count is even, keeping the charge/
       multiplicity parity invariant.  Both replacement elements are drawn
       independently from *element_pool*; if no valid pair is found in
       *max_tries* attempts the best single-replacement is used with a
       fallback parity correction.

    The ``charge`` and ``mult`` parameters are used only to determine the
    required electron-count parity (even or odd); they do not affect the
    swap path.

    Parameters
    ----------
    atoms:
        Current element list (modified copy is returned).
    element_pool:
        Elements available for replacement.
    rng:
        Seeded random-number generator.
    charge:
        System charge (used to infer required parity).
    mult:
        Spin multiplicity 2S+1 (used to infer required parity).

    Returns
    -------
    list[str]
        New atom list with the same length as *atoms*.
    """
    new_atoms = list(atoms)
    n = len(new_atoms)

    # ── Path 1: swap two atoms with different elements ────────────────────
    # Electron count is preserved → always parity-valid.
    for _ in range(20):
        i, j = rng.sample(range(n), 2)
        if new_atoms[i] != new_atoms[j]:
            new_atoms[i], new_atoms[j] = new_atoms[j], new_atoms[i]
            return new_atoms

    # ── Path 2: parity-preserving two-atom replacement ───────────────────
    # All atoms are the same element (swap path found no diverse pair).
    # Replace atom i with X and atom j with Y such that
    #   ΔZ_total = (Z(X) - Z(atoms[i])) + (Z(Y) - Z(atoms[j]))
    # is even, preserving the electron-count parity required by charge/mult.
    #
    # Required parity of total electrons:
    #   n_electrons = sum(Z) - charge
    #   For mult=1 (singlet), n_electrons must be even → sum(Z) parity = charge parity
    i = rng.randrange(n)
    zi = ATOMIC_NUMBERS[new_atoms[i]]

    # Separate pool into elements whose Z has the same parity as zi (ΔZ even)
    # vs different parity (ΔZ odd → need a compensating second replacement).
    same_parity_pool = [e for e in element_pool if ATOMIC_NUMBERS[e] % 2 == zi % 2]
    diff_parity_pool = [e for e in element_pool if ATOMIC_NUMBERS[e] % 2 != zi % 2]

    if same_parity_pool:
        # Replace one atom with same-parity element: ΔZ is even → parity kept.
        new_atoms[i] = rng.choice(same_parity_pool)
        return new_atoms

    if diff_parity_pool and n >= 2:
        # Replace two atoms with opposite-parity elements so ΔZ_total is even.
        j = rng.choice([k for k in range(n) if k != i])
        # Both from diff_parity_pool give ΔZ each odd → ΔZ_total even. ✓
        # Both from diff_parity_pool gives ΔZ each odd → ΔZ_total even. ✓
        new_atoms[i] = rng.choice(diff_parity_pool)
        new_atoms[j] = rng.choice(diff_parity_pool)
        return new_atoms

    # ── Last-resort fallback ──────────────────────────────────────────────
    # Pool has only one parity class and n == 1, or pool is empty.
    # Replace one atom; caller's validate_charge_mult will reject if invalid.
    new_atoms[i] = rng.choice(element_pool)
    return new_atoms


# ---------------------------------------------------------------------------
# StructureOptimizer
# ---------------------------------------------------------------------------


class StructureOptimizer:
    """Optimise a single structure to maximise a disorder objective.

    Parameters
    ----------
    n_atoms:
        Number of atoms.
    charge:
        Total system charge.
    mult:
        Spin multiplicity 2S+1.
    objective:
        Weight dict ``{"METRIC": weight, ...}`` or any callable
        ``(metrics: dict[str, float]) -> float``.
        The optimizer **maximises** the scalar value.
        Use negative weights to penalise a metric.
    elements:
        Element pool — spec string (``"6,7,8"``), list of symbols, or
        ``None`` for all Z = 1–106.
    method:
        ``"annealing"`` (default), ``"basin_hopping"``, or
        ``"parallel_tempering"``.
    max_steps:
        Number of MC steps per restart (or per replica per restart for
        ``"parallel_tempering"``; default: 5000).
    T_start:
        Initial temperature (default: 1.0).  For
        ``"parallel_tempering"`` this is the *highest* replica
        temperature.
    T_end:
        Final temperature for SA (default: 0.01).  For
        ``"parallel_tempering"`` this is the *lowest* replica
        temperature (the coldest, most selective replica).
        BH uses *T_start* throughout.
    n_replicas:
        Number of temperature replicas for ``"parallel_tempering"``
        (default: 4).  Ignored for other methods.  Temperatures are
        spaced geometrically between *T_end* and *T_start*.
    pt_swap_interval:
        Attempt a replica-exchange swap every this many MC steps
        (default: 10).  Ignored for other methods.
    allow_displacements:
        When ``True`` (default), atomic-position moves (fragment moves and,
        optionally, affine moves) are included in the MC step pool.
        When ``False``, only composition moves are performed — atomic
        coordinates are held fixed and only element types are optimised.
        Cannot be ``False`` simultaneously with *allow_composition_moves*.
    allow_composition_moves:
        When ``True`` (default), each MC step randomly chooses between a
        displacement move and a composition move (element-type swap) with
        equal probability.  When ``False``, only displacement moves are
        performed — element types are held fixed.
        Cannot be ``False`` simultaneously with *allow_displacements*.
    allow_affine_moves:
        When ``True``, half of the displacement moves are replaced by
        **affine moves** — a random stretch, compress, or shear applied to
        the entire structure, followed by a small per-atom jitter.  Affine
        moves allow the optimiser to explore elongated or compressed
        configurations that fragment moves cannot reach efficiently.
        Default: ``False`` (backward-compatible).
    affine_strength:
        Dimensionless scale of the affine transform (default: 0.1).
        At 0.1 the structure is stretched / compressed by up to ±10 % along
        a random axis and sheared by up to ±5 %.  Practical range: 0.02–0.4.
        Has no effect when *allow_affine_moves* is ``False``.
    frag_threshold:
        Local Q6 threshold for fragment selection (default: 0.3).
        Atoms with local Q6 > threshold are preferentially displaced.
    move_step:
        Maximum displacement magnitude per coordinate step (Å, default: 0.5).
        Also used as the per-atom jitter scale in affine moves (× 0.25).
    lcc_threshold:
        Minimum ``graph_lcc`` required to accept a step (default: 0.0,
        i.e. no connectivity constraint).  Set to 0.8 to enforce that at
        least 80 % of atoms remain connected.
    cov_scale:
        Minimum distance scale factor for :func:`relax_positions`.
    relax_cycles:
        Max repulsion-relaxation cycles per step.  Basin-Hopping uses
        3× this value for its local-minimisation step.
    cutoff:
        Distance cutoff (Å) for Steinhardt / graph metrics.  Auto-computed
        from the element pool when ``None``.
    n_bins:
        Histogram bins for ``H_spatial`` / ``RDF_dev`` (default: 20).
    w_atom:
        Weight of ``H_atom`` in ``H_total`` (default: 0.5).
    w_spatial:
        Weight of ``H_spatial`` in ``H_total`` (default: 0.5).
    n_restarts:
        Independent optimisation runs (default: 1).  The best result
        across all restarts is returned.
    seed:
        Random seed (``None`` → non-deterministic).
    verbose:
        Print per-step progress to stderr (default: ``False``).

    Examples
    --------
    Class API::

        from pasted import StructureOptimizer

        opt = StructureOptimizer(
            n_atoms=50,
            charge=0, mult=1,
            elements="24,25,26,27,28",      # Cantor alloy
            objective={"H_atom": 1.0, "H_spatial": 1.0, "Q6": -2.0},
            method="annealing",
            max_steps=5000,
            lcc_threshold=0.8,
            seed=42,
        )
        best = opt.run()

    Callable objective::

        opt = StructureOptimizer(
            ...,
            objective=lambda m: m["H_spatial"] - 2.0 * m["Q6"],
        )
    """

    def __init__(
        self,
        *,
        n_atoms: int,
        charge: int,
        mult: int,
        objective: ObjectiveType,
        elements: str | list[str] | None = None,
        method: str = "annealing",
        max_steps: int = 5000,
        T_start: float = 1.0,
        T_end: float = 0.01,
        frag_threshold: float = 0.3,
        move_step: float = 0.5,
        allow_composition_moves: bool = True,
        allow_displacements: bool = True,
        allow_affine_moves: bool = False,
        affine_strength: float = 0.1,
        lcc_threshold: float = 0.0,
        cov_scale: float = 1.0,
        relax_cycles: int = 1500,
        cutoff: float | None = None,
        n_bins: int = 20,
        w_atom: float = 0.5,
        w_spatial: float = 0.5,
        n_restarts: int = 1,
        n_replicas: int = 4,
        pt_swap_interval: int = 10,
        seed: int | None = None,
        verbose: bool = False,
    ) -> None:
        if method not in ("annealing", "basin_hopping", "parallel_tempering"):
            raise ValueError(
                f"method must be 'annealing', 'basin_hopping', or "
                f"'parallel_tempering', got {method!r}"
            )

        if not allow_displacements and not allow_composition_moves:
            raise ValueError(
                "allow_displacements and allow_composition_moves cannot both be False: "
                "at least one move type must be enabled."
            )

        self.n_atoms = n_atoms
        self.charge = charge
        self.mult = mult
        self.objective = objective
        self.method = method
        self.max_steps = max_steps
        self.T_start = T_start
        self.T_end = T_end
        self.frag_threshold = frag_threshold
        self.move_step = move_step
        self.allow_composition_moves = allow_composition_moves
        self.allow_displacements = allow_displacements
        self.allow_affine_moves = allow_affine_moves
        self.affine_strength = affine_strength
        self.lcc_threshold = lcc_threshold
        self.cov_scale = cov_scale
        self.relax_cycles = relax_cycles
        self.n_bins = n_bins
        self.w_atom = w_atom
        self.w_spatial = w_spatial
        self.n_restarts = n_restarts
        self.n_replicas = n_replicas
        self.pt_swap_interval = pt_swap_interval
        self.seed = seed
        self.verbose = verbose

        # Element pool
        if elements is None:
            self._element_pool: list[str] = default_element_pool()
        elif isinstance(elements, str):
            self._element_pool = parse_element_spec(elements)
        else:
            self._element_pool = list(elements)

        # Cutoff
        self._cutoff: float = self._resolve_cutoff(cutoff)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)

    def _resolve_cutoff(self, override: float | None) -> float:
        if override is not None:
            if self.verbose:
                self._log(f"[cutoff] {override:.3f} Å (user-specified)")
            return override
        radii = [_cov_radius_ang(s) for s in self._element_pool]
        pair_sums = sorted(
            ra + rb for i, ra in enumerate(radii) for rb in radii[i:]
        )
        median_sum = pair_sums[len(pair_sums) // 2]
        cutoff = self.cov_scale * 1.5 * median_sum
        if self.verbose:
            self._log(
                f"[cutoff] {cutoff:.3f} Å (auto: cov_scale={self.cov_scale} × 1.5 × "
                f"median(r_i+r_j)={median_sum:.3f} Å)"
            )
        return cutoff

    def _auto_region(self) -> str:
        """Estimate a sphere radius that gives approximately bulk density."""
        v_per_atom = 4 / 3 * math.pi * 1.3**3  # Å³, mean atomic radius ~1.3 Å
        r = (3 * self.n_atoms * v_per_atom / (4 * math.pi * 0.74)) ** (1 / 3)
        return f"sphere:{r * 1.2:.1f}"  # 20 % margin

    def _make_initial(self, rng: random.Random) -> Structure | None:
        """Generate a valid initial structure using StructureGenerator."""
        region = self._auto_region()
        for _ in range(50):
            seed = rng.randint(0, 2**31)
            structs = StructureGenerator(
                n_atoms=self.n_atoms,
                charge=self.charge,
                mult=self.mult,
                mode="gas",
                region=region,
                elements=self._element_pool,
                cov_scale=self.cov_scale,
                relax_cycles=self.relax_cycles,
                cutoff=self._cutoff,
                n_bins=self.n_bins,
                w_atom=self.w_atom,
                w_spatial=self.w_spatial,
                n_samples=1,
                seed=seed,
            ).generate()
            if structs:
                return structs.structures[0]
        return None

    def _temperature(self, step: int) -> float:
        """Return temperature at *step*.  BH and PT use constant T_start."""
        if self.method in ("basin_hopping", "parallel_tempering"):
            return self.T_start
        if self.T_end <= 0 or self.T_start <= 0:
            return self.T_start
        ratio = self.T_end / self.T_start
        return float(self.T_start * ratio ** (step / max(self.max_steps - 1, 1)))

    def _relax_cycles_for_method(self) -> int:
        """BH uses 3× relax cycles as local minimisation."""
        return self.relax_cycles * 3 if self.method == "basin_hopping" else self.relax_cycles

    def _pt_temperatures(self) -> list[float]:
        """Geometric temperature ladder from T_end (coldest) to T_start (hottest)."""
        n = max(2, self.n_replicas)
        if n == 1:
            return [self.T_start]
        t_lo = max(self.T_end, 1e-6)
        t_hi = max(self.T_start, t_lo * 1.01)
        ratio = (t_hi / t_lo) ** (1.0 / (n - 1))
        return [round(t_lo * ratio ** k, 8) for k in range(n)]  # cold→hot

    # ------------------------------------------------------------------ #
    # Parallel Tempering                                                  #
    # ------------------------------------------------------------------ #

    def _run_parallel_tempering(
        self, initial: Structure | None, restart_idx: int
    ) -> list[tuple[float, Structure]]:
        """Run one Parallel Tempering restart; return (score, structure) per replica.

        Algorithm
        ---------
        1. Build N replicas at temperatures T_0 < T_1 < … < T_{N-1}
           (geometric ladder from T_end to T_start).
        2. Every step: each replica independently proposes a move and
           applies Metropolis acceptance at its own temperature.
        3. Every ``pt_swap_interval`` steps: for each adjacent pair (k, k+1)
           attempt a replica exchange with the Metropolis criterion::

               ΔE = (β_k − β_{k+1}) × (f_{k+1} − f_k)

           where β = 1/T and f is the objective value (higher is better).
           Accept with probability min(1, exp(ΔE)).
        4. Track the global best across all replicas and all steps.

        Returns all replica final states sorted best-first so that
        ``run()`` can incorporate them into ``OptimizationResult``.
        """
        rng = random.Random(
            None if self.seed is None else self.seed + restart_idx * 97
        )
        temps = self._pt_temperatures()
        n_rep = len(temps)
        rc = self.relax_cycles

        # ── Initialise one state per replica ────────────────────────────
        # Replica 0 = coldest (T_end), Replica N-1 = hottest (T_start).
        # All replicas start from the same initial structure (or independent
        # gas-mode structures when initial is None) so that they explore
        # different regions from the start due to different acceptance rates.
        states_atoms:     list[list[str]]   = []
        states_positions: list[list[Vec3]]  = []
        states_metrics:   list[dict[str, float]] = []
        states_f:         list[float]       = []
        states_q6:        list[np.ndarray]  = []

        for k in range(n_rep):
            if initial is not None:
                a = list(initial.atoms)
                p = list(initial.positions)
            else:
                # Generate a fresh random structure for each replica
                rng_k = random.Random(
                    None if self.seed is None else self.seed + restart_idx * 97 + k * 13
                )
                _, p_raw = place_gas(
                    [rng.choice(self._element_pool) for _ in range(self.n_atoms)],
                    self._auto_region(), rng_k,
                )
                a_raw = [rng.choice(self._element_pool) for _ in range(self.n_atoms)]
                ok, _ = validate_charge_mult(a_raw, self.charge, self.mult)
                if not ok:
                    a_raw = list(a_raw)
                    # fallback: use first-valid replacement
                a = a_raw
                p_list, _ = relax_positions(a, p_raw, self.cov_scale, rc)
                p = list(p_list)

            m = compute_all_metrics(
                a, p, self.n_bins, self.w_atom, self.w_spatial,
                self._cutoff, self.cov_scale,
            )
            f = _eval_objective(m, self.objective)
            pts = np.array(p)
            q6: np.ndarray = compute_steinhardt_per_atom(pts, [6], self._cutoff)["Q6"]

            states_atoms.append(a)
            states_positions.append(p)
            states_metrics.append(m)
            states_f.append(f)
            states_q6.append(q6)

        best_atoms     = list(states_atoms[0])
        best_positions = list(states_positions[0])
        best_metrics   = dict(states_metrics[0])
        best_f         = states_f[0]
        for k in range(n_rep):
            if states_f[k] > best_f:
                best_f         = states_f[k]
                best_atoms     = list(states_atoms[k])
                best_positions = list(states_positions[k])
                best_metrics   = dict(states_metrics[k])

        # Exchange-attempt counts for diagnostics
        n_swap_attempted = 0
        n_swap_accepted  = 0

        # Precompute radii per replica (updated on composition moves)
        replicas_radii: list[np.ndarray] = [
            np.array([_cov_radius_ang(a) for a in atoms], dtype=float)
            for atoms in states_atoms
        ]
        seed_int = -1 if self.seed is None else int(self.seed + restart_idx * 97)

        log_interval = max(1, self.max_steps // 20)
        width = len(str(self.max_steps))

        for step in range(self.max_steps):
            # ── Each replica: one Metropolis step ────────────────────────
            for k in range(n_rep):
                T_k = temps[k]
                atoms = states_atoms[k]
                positions = states_positions[k]
                f_k = states_f[k]
                q6_k = states_q6[k]
                radii_k = replicas_radii[k]

                # Propose move
                _do_displacement = (
                    self.allow_displacements
                    and (not self.allow_composition_moves or rng.random() < 0.5)
                )
                if _do_displacement:
                    if self.allow_affine_moves and rng.random() < 0.5:
                        new_positions = _affine_move(
                            positions, self.move_step, self.affine_strength, rng
                        )
                    else:
                        new_positions = _fragment_move(
                            positions, q6_k, self.frag_threshold, self.move_step, rng
                        )
                    new_atoms = list(atoms)
                else:
                    new_atoms = _composition_move(
                        atoms, self._element_pool, rng, self.charge, self.mult
                    )
                    new_positions = list(positions)

                # Relax — skip when allow_displacements=False (positions must stay fixed)
                radii_new = radii_k
                if self.allow_displacements:
                    if HAS_RELAX:
                        if new_atoms != atoms:
                            radii_new = np.array(
                                [_cov_radius_ang(a) for a in new_atoms], dtype=float
                            )
                        pts_arr = np.array(new_positions, dtype=float)
                        pts_arr, _ = _cpp_relax_positions(
                            pts_arr, radii_new, self.cov_scale, rc, seed_int
                        )
                        new_positions = [tuple(row) for row in pts_arr]
                    else:
                        new_positions, _ = relax_positions(
                            new_atoms, new_positions, self.cov_scale, rc
                        )
                        if new_atoms != atoms:
                            radii_new = np.array(
                                [_cov_radius_ang(a) for a in new_atoms], dtype=float
                            )

                ok_parity, _ = validate_charge_mult(new_atoms, self.charge, self.mult)
                if not ok_parity:
                    continue

                new_metrics = compute_all_metrics(
                    new_atoms, new_positions, self.n_bins,
                    self.w_atom, self.w_spatial, self._cutoff, self.cov_scale,
                )
                f_new = _eval_objective(new_metrics, self.objective)

                if new_metrics.get("graph_lcc", 0.0) < self.lcc_threshold:
                    continue

                delta = f_new - f_k
                accept = delta >= 0 or (
                    T_k > 1e-12 and rng.random() < math.exp(delta / T_k)
                )

                if accept:
                    states_atoms[k]     = new_atoms
                    states_positions[k] = new_positions
                    states_metrics[k]   = new_metrics
                    states_f[k]         = f_new
                    replicas_radii[k]   = radii_new
                    new_pts = np.array(new_positions)
                    states_q6[k] = compute_steinhardt_per_atom(
                        new_pts, [6], self._cutoff
                    )["Q6"]

                    if f_new > best_f:
                        best_f         = f_new
                        best_atoms     = list(new_atoms)
                        best_positions = list(new_positions)
                        best_metrics   = dict(new_metrics)

            # ── Replica-exchange swaps every pt_swap_interval steps ───────
            if (step + 1) % self.pt_swap_interval == 0:
                # Attempt swaps between adjacent replica pairs in random order
                pairs = list(range(n_rep - 1))
                rng.shuffle(pairs)
                for k in pairs:
                    f_k  = states_f[k]
                    f_k1 = states_f[k + 1]
                    T_k  = temps[k]
                    T_k1 = temps[k + 1]
                    # Metropolis criterion for maximization:
                    #   ΔE = (β_k − β_{k+1}) × (f_{k+1} − f_k)
                    #   β = 1/T, higher T = more permissive
                    beta_k  = 1.0 / T_k  if T_k  > 1e-12 else 1e12
                    beta_k1 = 1.0 / T_k1 if T_k1 > 1e-12 else 1e12
                    delta_swap = (beta_k - beta_k1) * (f_k1 - f_k)
                    n_swap_attempted += 1
                    if delta_swap >= 0 or rng.random() < math.exp(delta_swap):
                        # Exchange states k ↔ k+1
                        (
                            states_atoms[k],     states_atoms[k + 1],
                            states_positions[k], states_positions[k + 1],
                            states_metrics[k],   states_metrics[k + 1],
                            states_f[k],         states_f[k + 1],
                            states_q6[k],        states_q6[k + 1],
                            replicas_radii[k],   replicas_radii[k + 1],
                        ) = (
                            states_atoms[k + 1],     states_atoms[k],
                            states_positions[k + 1], states_positions[k],
                            states_metrics[k + 1],   states_metrics[k],
                            states_f[k + 1],         states_f[k],
                            states_q6[k + 1],        states_q6[k],
                            replicas_radii[k + 1],   replicas_radii[k],
                        )
                        n_swap_accepted += 1

            # ── Logging ──────────────────────────────────────────────────
            if self.verbose and (step + 1) % log_interval == 0:
                swap_rate = (
                    n_swap_accepted / n_swap_attempted
                    if n_swap_attempted > 0 else 0.0
                )
                self._log(
                    f"[restart={restart_idx + 1} step={step + 1:>{width}}/{self.max_steps}] "
                    f"best_f={best_f:.4f}  "
                    f"replica_f=[{', '.join(f'{f:.3f}' for f in states_f)}]  "
                    f"T=[{', '.join(f'{t:.3f}' for t in temps)}]  "
                    f"swap_rate={swap_rate:.2f}"
                )

        if self.verbose:
            swap_rate = n_swap_accepted / max(n_swap_attempted, 1)
            self._log(
                f"[PT restart={restart_idx + 1}] best_f={best_f:.4f}  "
                f"swap_accept_rate={swap_rate:.3f}  "
                f"({n_swap_accepted}/{n_swap_attempted} swaps)"
            )

        # Return (score, structure) for each replica's final state + global best
        all_results: list[tuple[float, Structure]] = []
        # Global best first
        all_results.append((best_f, Structure(
            atoms=best_atoms,
            positions=best_positions,
            charge=self.charge,
            mult=self.mult,
            metrics=best_metrics,
            mode="opt_parallel_tempering",
            sample_index=restart_idx + 1,
            seed=self.seed,
        )))
        # Add each replica's final state (if not identical to best)
        for k in range(n_rep):
            f_k = states_f[k]
            if abs(f_k - best_f) > 1e-10 or states_atoms[k] != best_atoms:
                all_results.append((f_k, Structure(
                    atoms=list(states_atoms[k]),
                    positions=list(states_positions[k]),
                    charge=self.charge,
                    mult=self.mult,
                    metrics=dict(states_metrics[k]),
                    mode=f"opt_parallel_tempering_T{temps[k]:.4f}",
                    sample_index=restart_idx * n_rep + k + 1,
                    seed=self.seed,
                )))
        return all_results

    # ------------------------------------------------------------------ #
    # Single restart                                                       #
    # ------------------------------------------------------------------ #

    def _run_one(self, initial: Structure, restart_idx: int) -> Structure:
        rng = random.Random(
            None if self.seed is None else self.seed + restart_idx * 97
        )

        atoms: list[str] = list(initial.atoms)
        positions: list[Vec3] = list(initial.positions)

        # Pre-compute radii and seed_int once; reused every step.
        radii = np.array([_cov_radius_ang(a) for a in atoms], dtype=float)
        seed_int: int = -1 if self.seed is None else int(self.seed + restart_idx * 97)

        # Initial evaluation
        pts = np.array(positions)
        metrics = compute_all_metrics(
            atoms, positions, self.n_bins, self.w_atom, self.w_spatial, self._cutoff,
            self.cov_scale,
        )
        f_current = _eval_objective(metrics, self.objective)
        per_atom_q6: np.ndarray = compute_steinhardt_per_atom(pts, [6], self._cutoff)[
            "Q6"
        ]

        best_atoms = list(atoms)
        best_positions = list(positions)
        best_f = f_current
        best_metrics = dict(metrics)

        rc = self._relax_cycles_for_method()
        log_interval = max(1, self.max_steps // 20)
        width = len(str(self.max_steps))

        for step in range(self.max_steps):
            T = self._temperature(step)

            # ── Move ─────────────────────────────────────────────────────
            _do_displacement = (
                self.allow_displacements
                and (not self.allow_composition_moves or rng.random() < 0.5)
            )
            if _do_displacement:
                if self.allow_affine_moves and rng.random() < 0.5:
                    new_positions = _affine_move(
                        positions, self.move_step, self.affine_strength, rng
                    )
                else:
                    new_positions = _fragment_move(
                        positions, per_atom_q6, self.frag_threshold, self.move_step, rng
                    )
                new_atoms = list(atoms)
            else:
                new_atoms = _composition_move(
                    atoms, self._element_pool, rng, self.charge, self.mult
                )
                new_positions = list(positions)

            # ── Relax (distance constraint) ───────────────────────────────
            # Skip relax when allow_displacements=False: the caller expects
            # positions to be exactly preserved after composition-only moves.
            if self.allow_displacements:
                if HAS_RELAX:
                    # radii may need updating if atom types changed (composition move)
                    if new_atoms != atoms:
                        new_radii = np.array([_cov_radius_ang(a) for a in new_atoms], dtype=float)
                    else:
                        new_radii = radii
                    new_pts_arr = np.array(new_positions, dtype=float)
                    new_pts_arr, _ = _cpp_relax_positions(
                        new_pts_arr, new_radii, self.cov_scale, rc, seed_int
                    )
                    new_positions = [tuple(row) for row in new_pts_arr]
                else:
                    new_positions, _ = relax_positions(
                        new_atoms, new_positions, self.cov_scale, rc
                    )

            # ── Charge/mult validity ──────────────────────────────────────
            ok, _ = validate_charge_mult(new_atoms, self.charge, self.mult)
            if not ok:
                continue

            # ── Evaluate ─────────────────────────────────────────────────
            new_metrics = compute_all_metrics(
                new_atoms,
                new_positions,
                self.n_bins,
                self.w_atom,
                self.w_spatial,
                self._cutoff,
                self.cov_scale,
            )
            f_new = _eval_objective(new_metrics, self.objective)

            # ── Hard connectivity constraint ──────────────────────────────
            if new_metrics.get("graph_lcc", 0.0) < self.lcc_threshold:
                continue

            # ── Accept / reject (Metropolis) ─────────────────────────────
            delta = f_new - f_current
            accept = delta >= 0 or (
                T > 1e-12 and rng.random() < math.exp(delta / T)
            )

            if accept:
                atoms = new_atoms
                positions = new_positions
                metrics = new_metrics
                f_current = f_new
                # Update radii cache when composition changed
                if atoms is not new_atoms or new_atoms != list(atoms):
                    radii = np.array([_cov_radius_ang(a) for a in atoms], dtype=float)
                # Update per_atom_q6 for next fragment selection
                new_pts = np.array(positions)
                per_atom_q6 = compute_steinhardt_per_atom(
                    new_pts, [6], self._cutoff
                )["Q6"]

                if f_current > best_f:
                    best_f = f_current
                    best_atoms = list(atoms)
                    best_positions = list(positions)
                    best_metrics = dict(metrics)

            # ── Logging ──────────────────────────────────────────────────
            if self.verbose and step % log_interval == 0:
                self._log(
                    f"[restart={restart_idx + 1} "
                    f"step={step + 1:>{width}}/{self.max_steps}] "
                    f"T={T:.4f}  f={f_current:.4f}  best={best_f:.4f}  "
                    + "  ".join(f"{k}={_fmt(v)}" for k, v in metrics.items())
                )

        return Structure(
            atoms=best_atoms,
            positions=best_positions,
            charge=self.charge,
            mult=self.mult,
            metrics=best_metrics,
            mode=f"opt_{self.method}",
            sample_index=restart_idx + 1,
            seed=self.seed,
        )

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(self, initial: Structure | None = None) -> OptimizationResult:
        """Run ``n_restarts`` optimizations and return an :class:`OptimizationResult`.

        Each restart begins from an independently generated random gas-mode
        structure (or from *initial* if provided).  All per-restart results
        are collected, sorted by objective value (highest first), and returned
        together in an :class:`OptimizationResult`.

        :class:`OptimizationResult` is list-compatible: ``result[0]`` and
        ``result.best`` both return the highest-scoring structure, and
        ``for s in result`` iterates all restarts in rank order.  Existing
        code that calls ``opt.run()`` and uses the return value as a single
        ``Structure`` should switch to ``opt.run().best`` or ``opt.run()[0]``.

        A :class:`UserWarning` is emitted when one or more restarts fail to
        produce a valid initial structure.

        Parameters
        ----------
        initial:
            Starting structure.  When ``None`` (default), a random gas-mode
            structure is generated automatically for each restart.

        Returns
        -------
        OptimizationResult
            All per-restart structures sorted by objective value (highest
            first), plus summary metadata.  Raises :class:`RuntimeError` if
            every restart fails to produce a valid initial structure.

        Raises
        ------
        RuntimeError
            When all restarts fail to produce a valid initial structure.

        Examples
        --------
        Best structure only::

            result = opt.run()
            print(result.best)          # highest-scoring structure
            print(result[0])            # same — index 0 is always the best
            print(result.summary())     # one-line diagnostic

        All restarts::

            result = opt.run()
            for rank, s in enumerate(result, 1):
                print(f"rank {rank}: f={result.objective_scores[rank-1]:.4f}  {s}")
        """
        rng = random.Random(self.seed)
        all_results: list[tuple[float, Structure]] = []
        n_attempted = 0

        for r in range(self.n_restarts):
            self._log(f"[optimize] restart {r + 1}/{self.n_restarts} start")

            # ── Parallel Tempering path ───────────────────────────────────
            if self.method == "parallel_tempering":
                n_attempted += 1
                pt_results = self._run_parallel_tempering(initial, r)
                all_results.extend(pt_results)
                best_pt = max(pt_results, key=lambda x: x[0])
                self._log(
                    f"[optimize] restart {r + 1}/{self.n_restarts} done: "
                    f"best_f={best_pt[0]:.4f} ({len(pt_results)} replica states)"
                )
                continue

            # ── SA / BH path ──────────────────────────────────────────────
            init = initial
            if init is None:
                init = self._make_initial(rng)
            if init is None:
                self._log(
                    f"[optimize] restart {r + 1}: "
                    "could not generate initial structure, skipping"
                )
                continue

            n_attempted += 1
            result = self._run_one(init, r)
            f = _eval_objective(result.metrics, self.objective)
            self._log(f"[optimize] restart {r + 1}/{self.n_restarts} done: f={f:.4f}")
            all_results.append((f, result))

        if not all_results:
            raise RuntimeError(
                "Optimization failed: no valid structure found across all restarts."
            )

        n_skipped = self.n_restarts - n_attempted
        if n_skipped > 0:
            warnings.warn(
                f"{n_skipped} of {self.n_restarts} restart(s) were skipped because "
                f"no valid initial structure could be generated.  "
                f"Try narrowing the element pool to satisfy "
                f"charge={self.charge}, mult={self.mult}.",
                UserWarning,
                stacklevel=2,
            )

        all_results.sort(key=lambda x: x[0], reverse=True)
        best_f = all_results[0][0]
        self._log(f"[optimize] best f={best_f:.4f}")

        return OptimizationResult(
            all_structures=[s for _, s in all_results],
            objective_scores=[f for f, _ in all_results],
            n_restarts_attempted=n_attempted,
            method=self.method,
        )

    # ------------------------------------------------------------------ #
    # Properties / dunder                                                  #
    # ------------------------------------------------------------------ #

    @property
    def element_pool(self) -> list[str]:
        """A copy of the resolved element pool."""
        return list(self._element_pool)

    @property
    def cutoff(self) -> float:
        """Distance cutoff (Å) used for Steinhardt and graph metrics."""
        return self._cutoff

    def __repr__(self) -> str:
        pt_info = (
            f", n_replicas={self.n_replicas}, pt_swap_interval={self.pt_swap_interval}"
            if self.method == "parallel_tempering"
            else ""
        )
        comp_info  = "" if self.allow_composition_moves else ", allow_composition_moves=False"
        disp_info  = "" if self.allow_displacements else ", allow_displacements=False"
        affine_info = (
            f", affine_strength={self.affine_strength}"
            if self.allow_affine_moves
            else ""
        )
        return (
            f"StructureOptimizer("
            f"n_atoms={self.n_atoms}, method={self.method!r}, "
            f"max_steps={self.max_steps}, "
            f"T_start={self.T_start}, T_end={self.T_end}, "
            f"pool_size={len(self._element_pool)}"
            f"{pt_info}{comp_info}{disp_info}{affine_info})"
        )
