"""Tests for pasted._placement: all placement modes and relax_positions."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from pasted import _ext
from pasted._atoms import cov_radius_ang
from pasted._ext import HAS_RELAX
from pasted._metrics import compute_shape_anisotropy
from pasted._placement import (
    add_hydrogen,
    place_chain,
    place_gas,
    place_shell,
    relax_positions,
)

RNG = random.Random(99)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _min_dist(positions: list) -> float:
    n = len(positions)
    dmin = math.inf
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = positions[i], positions[j]
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(pi, pj, strict=True)))
            dmin = min(dmin, d)
    return dmin


# ---------------------------------------------------------------------------
# place_gas
# ---------------------------------------------------------------------------


class TestPlaceGas:
    def test_sphere_count(self) -> None:
        atoms = ["C", "N", "O"]
        rng = random.Random(0)
        atoms_out, pos = place_gas(atoms, "sphere:5", rng)
        assert len(pos) == 3

    def test_sphere_bounds(self) -> None:
        atoms = ["C"] * 20
        rng = random.Random(1)
        _, pos = place_gas(atoms, "sphere:5", rng)
        for p in pos:
            r = math.sqrt(sum(x * x for x in p))
            assert r <= 5.0 + 1e-9

    def test_box_count(self) -> None:
        rng = random.Random(2)
        _, pos = place_gas(["H"] * 10, "box:4", rng)
        assert len(pos) == 10

    def test_box_bounds(self) -> None:
        rng = random.Random(3)
        _, pos = place_gas(["H"] * 20, "box:4,6,8", rng)
        for x, y, z in pos:
            assert abs(x) <= 2.0 + 1e-9
            assert abs(y) <= 3.0 + 1e-9
            assert abs(z) <= 4.0 + 1e-9

    def test_atoms_preserved(self) -> None:
        atoms = ["C", "N", "O", "Fe"]
        rng = random.Random(4)
        atoms_out, _ = place_gas(atoms, "sphere:5", rng)
        assert atoms_out == atoms

    def test_unknown_region_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown region"):
            place_gas(["C"], "cylinder:3", random.Random(0))


# ---------------------------------------------------------------------------
# place_chain
# ---------------------------------------------------------------------------


class TestPlaceChain:
    def test_count(self) -> None:
        atoms = ["C"] * 10
        rng = random.Random(5)
        _, pos = place_chain(atoms, 1.2, 1.6, 0.3, 0.5, rng)
        assert len(pos) == 10

    def test_seed_reproducible(self) -> None:
        atoms = ["C", "N", "O"] * 4
        _, pos1 = place_chain(atoms, 1.2, 1.6, 0.3, 0.5, random.Random(7))
        _, pos2 = place_chain(atoms, 1.2, 1.6, 0.3, 0.5, random.Random(7))
        for p1, p2 in zip(pos1, pos2, strict=True):
            assert p1 == p2

    def test_first_atom_at_origin(self) -> None:
        _, pos = place_chain(["C", "N"], 1.4, 1.4, 0.0, 0.0, random.Random(8))
        assert pos[0] == (0.0, 0.0, 0.0)

    def test_atoms_order_preserved(self) -> None:
        atoms = ["C", "N", "O"]
        atoms_out, _ = place_chain(atoms, 1.2, 1.6, 0.3, 0.5, random.Random(9))
        assert atoms_out == atoms

    def test_chain_bias_zero_unchanged(self) -> None:
        """chain_bias=0.0 must produce exactly the same output as the default."""
        atoms = ["C"] * 12
        _, pos_default = place_chain(atoms, 1.2, 1.6, 0.3, 0.5, random.Random(20))
        _, pos_bias0 = place_chain(atoms, 1.2, 1.6, 0.3, 0.5, random.Random(20),
                                   chain_bias=0.0)
        for p1, p2 in zip(pos_default, pos_bias0, strict=True):
            assert p1 == p2

    def test_chain_bias_elongates(self) -> None:
        """High chain_bias should produce more elongated structures on average."""
        atoms = ["C"] * 20
        sa_nobias, sa_bias = [], []
        for seed in range(100):
            _, pos0 = place_chain(atoms, 1.2, 1.6, 0.0, 0.5, random.Random(seed),
                                  chain_bias=0.0)
            _, pos1 = place_chain(atoms, 1.2, 1.6, 0.0, 0.5, random.Random(seed),
                                  chain_bias=0.8)
            sa_nobias.append(compute_shape_anisotropy(np.array(pos0)))
            sa_bias.append(compute_shape_anisotropy(np.array(pos1)))

        assert np.mean(sa_bias) > np.mean(sa_nobias) + 0.05

    def test_chain_bias_seed_reproducible(self) -> None:
        """chain_bias results are reproducible with the same seed."""
        atoms = ["C"] * 15
        _, pos1 = place_chain(atoms, 1.2, 1.6, 0.3, 0.5, random.Random(42),
                              chain_bias=0.5)
        _, pos2 = place_chain(atoms, 1.2, 1.6, 0.3, 0.5, random.Random(42),
                              chain_bias=0.5)
        for p1, p2 in zip(pos1, pos2, strict=True):
            assert p1 == p2


# ---------------------------------------------------------------------------
# place_shell
# ---------------------------------------------------------------------------


class TestPlaceShell:
    def test_count(self) -> None:
        atoms = ["Fe", "C", "C", "N", "N", "O", "O", "O"]
        rng = random.Random(10)
        atoms_out, pos = place_shell(atoms, "Fe", 4, 6, 1.8, 2.5, 1.2, 1.6, rng)
        assert len(pos) == len(atoms)

    def test_center_at_origin(self) -> None:
        atoms = ["Fe"] + ["C"] * 5
        rng = random.Random(11)
        atoms_out, pos = place_shell(atoms, "Fe", 4, 4, 2.0, 2.0, 1.4, 1.4, rng)
        assert atoms_out[0] == "Fe"
        assert pos[0] == (0.0, 0.0, 0.0)

    def test_center_first_in_output(self) -> None:
        atoms = ["C", "N", "Fe", "O"]
        rng = random.Random(12)
        atoms_out, _ = place_shell(atoms, "Fe", 2, 2, 2.0, 2.0, 1.4, 1.4, rng)
        assert atoms_out[0] == "Fe"


# ---------------------------------------------------------------------------
# relax_positions
# ---------------------------------------------------------------------------


class TestRelaxPositions:
    def test_single_atom_converges(self) -> None:
        pos = [(0.0, 0.0, 0.0)]
        result, converged = relax_positions(["C"], pos, 1.0)
        assert converged
        assert result == pos

    def test_two_atoms_coincident_converges(self) -> None:
        atoms = ["C", "C"]
        pos = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
        result, converged = relax_positions(atoms, pos, 1.0, max_cycles=500)
        assert converged
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(result[0], result[1], strict=True)))
        r_sum = cov_radius_ang("C") + cov_radius_ang("C")
        assert d >= r_sum - 1e-6

    def test_well_separated_unchanged(self) -> None:
        atoms = ["H", "H"]
        pos = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        result, converged = relax_positions(atoms, pos, 1.0, max_cycles=10)
        assert converged
        # Far-apart atoms should not move
        assert result[0] == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
        assert result[1] == pytest.approx((100.0, 0.0, 0.0), abs=1e-9)

    def test_min_distance_enforced(self) -> None:
        atoms = ["C", "N", "O"]
        # Place all atoms at origin
        pos = [(0.0, 0.0, 0.0)] * 3
        result, _ = relax_positions(atoms, pos, 1.0, max_cycles=2000)
        n = len(atoms)
        for i in range(n):
            for j in range(i + 1, n):
                d = math.sqrt(sum((result[i][k] - result[j][k]) ** 2 for k in range(3)))
                threshold = cov_radius_ang(atoms[i]) + cov_radius_ang(atoms[j])
                assert d >= threshold - 1e-5

    def test_seed_reproducible(self) -> None:
        """Same seed → same result; different seed may differ (coincident case)."""
        atoms = ["C", "C"]
        pos = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]  # coincident: needs RNG

        r1, ok1 = relax_positions(atoms, pos, 1.0, max_cycles=500, seed=7)
        r2, ok2 = relax_positions(atoms, pos, 1.0, max_cycles=500, seed=7)
        assert ok1 and ok2
        for p1, p2 in zip(r1, r2, strict=True):
            assert p1 == pytest.approx(p2, abs=1e-12)

    def test_seed_none_still_converges(self) -> None:
        """seed=None (non-deterministic) must still produce a valid structure."""
        atoms = ["C", "C"]
        pos = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
        result, converged = relax_positions(atoms, pos, 1.0, max_cycles=500, seed=None)
        assert converged
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(result[0], result[1], strict=True)))
        r_sum = cov_radius_ang("C") + cov_radius_ang("C")
        assert d >= r_sum - 1e-6

    @pytest.mark.skipif(not HAS_RELAX, reason="_relax_core extension not built")
    def test_cpp_and_python_agree(self) -> None:
        """C++ and Python paths must both converge and enforce min distances.

        Exact coordinate agreement is *not* expected: the Python fallback uses
        a Jacobi-style update (distance matrix frozen for the whole cycle),
        while the C++ path uses a Gauss-Seidel-style update (positions updated
        immediately within each cycle).  Both strategies are correct but produce
        different trajectories.

        We verify the shared contract: all min-distance constraints are
        satisfied by both implementations.
        """
        atoms = ["C", "N", "O", "Fe", "H"]
        rng = np.random.default_rng(42)
        pos = [tuple(float(x) for x in rng.uniform(0.1, 0.5, 3)) for _ in atoms]

        original = _ext.HAS_RELAX
        try:
            _ext.HAS_RELAX = False
            py_result, py_conv = relax_positions(atoms, pos, 1.0, max_cycles=1000, seed=123)
        finally:
            _ext.HAS_RELAX = original

        cpp_result, cpp_conv = relax_positions(atoms, pos, 1.0, max_cycles=1000, seed=123)

        # Both must converge
        assert py_conv and cpp_conv

        # Both must satisfy all pairwise minimum distances
        for result, label in [(py_result, "python"), (cpp_result, "cpp")]:
            n = len(atoms)
            for i in range(n):
                for j in range(i + 1, n):
                    d = math.sqrt(
                        sum((result[i][k] - result[j][k]) ** 2 for k in range(3))
                    )
                    thr = cov_radius_ang(atoms[i]) + cov_radius_ang(atoms[j])
                    assert d >= thr - 1e-5, (
                        f"{label}: pair ({atoms[i]},{atoms[j]}) d={d:.6f} < thr={thr:.6f}"
                    )

    @pytest.mark.skipif(not HAS_RELAX, reason="_relax_core extension not built")
    def test_cpp_coincident_converges(self) -> None:
        """C++ path must converge for coincident atoms regardless of RNG path."""
        atoms = ["C", "N", "O"]
        pos = [(0.0, 0.0, 0.0)] * 3
        result, converged = relax_positions(atoms, pos, 1.0, max_cycles=2000, seed=99)
        assert converged
        n = len(atoms)
        for i in range(n):
            for j in range(i + 1, n):
                d = math.sqrt(sum((result[i][k] - result[j][k]) ** 2 for k in range(3)))
                threshold = cov_radius_ang(atoms[i]) + cov_radius_ang(atoms[j])
                assert d >= threshold - 1e-5


# ---------------------------------------------------------------------------
# add_hydrogen
# ---------------------------------------------------------------------------


class TestAddHydrogen:
    def test_already_has_h(self) -> None:
        atoms = ["C", "H", "O"]
        result = add_hydrogen(atoms, random.Random(0))
        assert result is atoms  # same object returned unchanged

    def test_adds_h_when_missing(self) -> None:
        atoms = ["C", "N", "O"]
        result = add_hydrogen(atoms, random.Random(42))
        assert "H" in result
        assert len(result) > len(atoms)

    def test_original_not_modified(self) -> None:
        atoms = ["C", "N"]
        original = list(atoms)
        add_hydrogen(atoms, random.Random(0))
        assert atoms == original
