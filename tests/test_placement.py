"""Tests for pasted._placement: all placement modes and relax_positions."""

from __future__ import annotations

import math
import random

import pytest

from pasted._atoms import cov_radius_ang
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
                d = math.sqrt(
                    sum((result[i][k] - result[j][k]) ** 2 for k in range(3))
                )
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
