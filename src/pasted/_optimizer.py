"""
pasted._optimizer
=================
Objective-based structure optimisation.

Two methods
-----------
``"annealing"``
    Simulated Annealing with exponential cooling from *T_start* to *T_end*.
``"basin_hopping"``
    Basin-Hopping: each step applies a more thorough relaxation (3× relax
    cycles) before the Metropolis acceptance test.  Temperature is held
    constant at *T_start*.

Move types (chosen with equal probability each step)
----------------------------------------------------
Fragment coordinate move
    Compute per-atom Q6.  Atoms whose local Q6 exceeds *frag_threshold*
    are considered "accidentally ordered" and are displaced by a random
    vector of magnitude ≤ *move_step* Å.  If no atom exceeds the threshold
    (structure is already fully disordered), a single random atom is moved.
Composition move
    Two atoms of different element types are swapped.  If no such pair
    exists, one atom is replaced by a random element from the pool.

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
from collections.abc import Callable

import numpy as np
from scipy.spatial.distance import pdist as _pdist
from scipy.spatial.distance import squareform as _squareform

from ._atoms import (
    ALL_METRICS,
    _cov_radius_ang,
    default_element_pool,
    parse_element_spec,
    validate_charge_mult,
)
from ._generator import Structure, StructureGenerator
from ._io import _fmt
from ._metrics import compute_all_metrics, compute_steinhardt_per_atom
from ._placement import Vec3, relax_positions

# ---------------------------------------------------------------------------
# Public type alias
# ---------------------------------------------------------------------------

ObjectiveType = dict[str, float] | Callable[[dict[str, float]], float]

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
) -> list[str]:
    """Swap element types of two atoms (prefer different elements).

    Tries up to 20 times to find a pair of different elements.  Falls
    back to replacing a random atom with a random pool element if no
    suitable pair is found.
    """
    new_atoms = list(atoms)
    n = len(new_atoms)
    for _ in range(20):
        i, j = rng.sample(range(n), 2)
        if new_atoms[i] != new_atoms[j]:
            new_atoms[i], new_atoms[j] = new_atoms[j], new_atoms[i]
            return new_atoms
    # Fallback: replace one atom
    i = rng.randrange(n)
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
        ``"annealing"`` (default) or ``"basin_hopping"``.
    max_steps:
        Number of MC steps per restart (default: 5000).
    T_start:
        Initial temperature (default: 1.0).
    T_end:
        Final temperature for SA (default: 0.01).
        BH uses *T_start* throughout.
    frag_threshold:
        Local Q6 threshold for fragment selection (default: 0.3).
        Atoms with local Q6 > threshold are preferentially displaced.
    move_step:
        Maximum displacement magnitude per coordinate step (Å, default: 0.5).
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
        lcc_threshold: float = 0.0,
        cov_scale: float = 1.0,
        relax_cycles: int = 1500,
        cutoff: float | None = None,
        n_bins: int = 20,
        w_atom: float = 0.5,
        w_spatial: float = 0.5,
        n_restarts: int = 1,
        seed: int | None = None,
        verbose: bool = False,
    ) -> None:
        if method not in ("annealing", "basin_hopping"):
            raise ValueError(
                f"method must be 'annealing' or 'basin_hopping', got {method!r}"
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
        self.lcc_threshold = lcc_threshold
        self.cov_scale = cov_scale
        self.relax_cycles = relax_cycles
        self.n_bins = n_bins
        self.w_atom = w_atom
        self.w_spatial = w_spatial
        self.n_restarts = n_restarts
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
                return structs[0]
        return None

    def _temperature(self, step: int) -> float:
        """Return temperature at *step*.  BH uses constant T_start."""
        if self.method == "basin_hopping":
            return self.T_start
        if self.T_end <= 0 or self.T_start <= 0:
            return self.T_start
        ratio = self.T_end / self.T_start
        return float(self.T_start * ratio ** (step / max(self.max_steps - 1, 1)))

    def _relax_cycles_for_method(self) -> int:
        """BH uses 3× relax cycles as local minimisation."""
        return self.relax_cycles * 3 if self.method == "basin_hopping" else self.relax_cycles

    # ------------------------------------------------------------------ #
    # Single restart                                                       #
    # ------------------------------------------------------------------ #

    def _run_one(self, initial: Structure, restart_idx: int) -> Structure:
        rng = random.Random(
            None if self.seed is None else self.seed + restart_idx * 97
        )

        atoms: list[str] = list(initial.atoms)
        positions: list[Vec3] = list(initial.positions)

        # Initial evaluation
        pts = np.array(positions)
        dmat = _squareform(_pdist(pts))
        metrics = compute_all_metrics(
            atoms, positions, self.n_bins, self.w_atom, self.w_spatial, self._cutoff,
            self.cov_scale,
        )
        f_current = _eval_objective(metrics, self.objective)
        per_atom_q6: np.ndarray = compute_steinhardt_per_atom(pts, dmat, [6], self._cutoff)[
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
            if rng.random() < 0.5:
                # Fragment coordinate move
                new_positions = _fragment_move(
                    positions, per_atom_q6, self.frag_threshold, self.move_step, rng
                )
                new_atoms = list(atoms)
            else:
                # Composition move
                new_atoms = _composition_move(atoms, self._element_pool, rng)
                new_positions = list(positions)

            # ── Relax (distance constraint) ───────────────────────────────
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
                # Update per_atom_q6 for next fragment selection
                new_pts = np.array(positions)
                new_dmat = _squareform(_pdist(new_pts))
                per_atom_q6 = compute_steinhardt_per_atom(
                    new_pts, new_dmat, [6], self._cutoff
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

    def run(self, initial: Structure | None = None) -> Structure:
        """Run ``n_restarts`` optimisations and return the best structure.

        Parameters
        ----------
        initial:
            Starting structure.  When ``None`` (default), a random gas-mode
            structure is generated automatically for each restart.

        Returns
        -------
        Structure
            The structure with the highest objective value across all
            restarts.

        Raises
        ------
        RuntimeError
            When all restarts fail to produce a valid initial structure.
        """
        rng = random.Random(self.seed)
        best_structure: Structure | None = None
        best_f = -math.inf

        for r in range(self.n_restarts):
            self._log(f"[optimize] restart {r + 1}/{self.n_restarts} start")

            init = initial
            if init is None:
                init = self._make_initial(rng)
            if init is None:
                self._log(
                    f"[optimize] restart {r + 1}: "
                    "could not generate initial structure, skipping"
                )
                continue

            result = self._run_one(init, r)
            f = _eval_objective(result.metrics, self.objective)
            self._log(f"[optimize] restart {r + 1}/{self.n_restarts} done: f={f:.4f}")

            if f > best_f:
                best_f = f
                best_structure = result

        if best_structure is None:
            raise RuntimeError(
                "Optimization failed: no valid structure found across all restarts."
            )

        self._log(f"[optimize] best f={best_f:.4f}")
        return best_structure

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
        return (
            f"StructureOptimizer("
            f"n_atoms={self.n_atoms}, method={self.method!r}, "
            f"max_steps={self.max_steps}, "
            f"T_start={self.T_start}, T_end={self.T_end}, "
            f"pool_size={len(self._element_pool)})"
        )
