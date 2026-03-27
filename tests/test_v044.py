"""Tests for new features introduced in PASTED v0.4.4.

New features covered:
1. add_hydrogen() – region-based volume cap and parity-aware H count
2. StructureGenerator._adjust_parity() – nudges atom list ±1 electron
3. StructureGenerator._validate_density() – auto-scales region when packing
   fraction exceeds safe thresholds (warn: 0.50, hard: 0.64)
"""

from __future__ import annotations

import math
import random
import warnings

import numpy as np
import pytest

from pasted import StructureGenerator
from pasted._atoms import ATOMIC_NUMBERS, _cov_radius_ang, parse_element_spec, validate_charge_mult
from pasted._placement import (
    _H_COV_RADIUS,
    _PACKING_HARD,
    _PACKING_WARN,
    add_hydrogen,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sphere_vol(r: float) -> float:
    return (4 / 3) * math.pi * r ** 3


def _atom_vol(r: float) -> float:
    return (4 / 3) * math.pi * r ** 3


def _sphere_region_at_packing(n_atoms: int, pf: float, elements: str = "6") -> str:
    """Return a sphere region spec that gives exactly *pf* packing fraction.

    Uses the same mean-radius formula as _validate_density internally.
    """
    pool = parse_element_spec(elements)
    mean_r = float(np.mean([_cov_radius_ang(s) for s in pool]))
    atom_v = _atom_vol(mean_r)
    region_vol = n_atoms * atom_v / pf
    r = ((3 * region_vol) / (4 * math.pi)) ** (1 / 3)
    return f"sphere:{r:.4f}"


def _make_gen(**kwargs: object) -> StructureGenerator:
    """Return a minimal gas-mode generator, merging caller kwargs."""
    defaults: dict[str, object] = dict(
        n_atoms=6, charge=0, mult=1, mode="gas",
        region="sphere:8", elements="6,7,8", n_samples=1, seed=0,
    )
    defaults.update(kwargs)
    return StructureGenerator(**defaults)  # type: ignore[arg-type]


# ===========================================================================
# 1. add_hydrogen()
# ===========================================================================

class TestAddHydrogen:
    """add_hydrogen() with new region / charge / mult parameters."""

    def test_no_h_in_atoms_returns_augmented(self) -> None:
        atoms = ["C", "N", "O"]
        rng = random.Random(0)
        result = add_hydrogen(atoms, rng)
        assert "H" in result
        assert result[:3] == atoms

    def test_h_already_present_returns_unchanged(self) -> None:
        atoms = ["C", "H", "O"]
        rng = random.Random(0)
        result = add_hydrogen(atoms, rng)
        assert result is atoms  # exact same object

    def test_parity_singlet_charge0(self) -> None:
        """With charge=0, mult=1 the electron count must be even."""
        rng = random.Random(42)
        for _ in range(50):
            atoms = ["C", "N"]  # 6+7=13 electrons, odd → needs one H
            result = add_hydrogen(atoms, rng, charge=0, mult=1)
            total_z = sum(ATOMIC_NUMBERS[a] for a in result)
            n_e = total_z - 0
            assert n_e % 2 == 0, f"Expected even electron count, got {n_e} for {result}"

    def test_parity_doublet_charge0(self) -> None:
        """With charge=0, mult=2 the electron count must be odd."""
        rng = random.Random(7)
        atoms = ["C", "C"]  # 12 electrons, even → needs one H for odd
        result = add_hydrogen(atoms, rng, charge=0, mult=2)
        total_z = sum(ATOMIC_NUMBERS[a] for a in result)
        n_e = total_z - 0
        assert n_e % 2 == 1, f"Expected odd electron count, got {n_e}"

    def test_volume_cap_sphere_limits_h_count(self) -> None:
        """H count must not exceed the volume cap for a small sphere."""
        atoms = ["C"] * 3
        region = "sphere:2.0"
        region_vol = _sphere_vol(2.0)
        mean_r = _cov_radius_ang("C")
        heavy_vol = 3 * _atom_vol(mean_r)
        h_vol = _atom_vol(_H_COV_RADIUS)
        max_h = max(0, int((region_vol * _PACKING_HARD - heavy_vol) / h_vol))

        results = [add_hydrogen(list(atoms), random.Random(i), region=region) for i in range(30)]
        h_counts = [r.count("H") for r in results]
        assert all(n <= max_h for n in h_counts), (
            f"H count exceeded volume cap {max_h}: max seen = {max(h_counts)}"
        )

    def test_volume_cap_box_limits_h_count(self) -> None:
        """Volume cap works for box region too."""
        atoms = ["N", "O"]
        region = "box:3.0"
        region_vol = 3.0 ** 3
        mean_r = float(np.mean([_cov_radius_ang(a) for a in atoms]))
        heavy_vol = 2 * _atom_vol(mean_r)
        h_vol = _atom_vol(_H_COV_RADIUS)
        max_h = max(0, int((region_vol * _PACKING_HARD - heavy_vol) / h_vol))

        results = [add_hydrogen(list(atoms), random.Random(i), region=region) for i in range(20)]
        h_counts = [r.count("H") for r in results]
        assert all(n <= max_h for n in h_counts)

    def test_no_region_uncapped_original_behaviour(self) -> None:
        """Without region, H count follows original distribution (uncapped)."""
        rng = random.Random(0)
        atoms = ["C"] * 5
        result = add_hydrogen(atoms, rng)
        assert result.count("H") >= 1

    def test_result_not_mutated(self) -> None:
        """The input list must not be mutated."""
        atoms = ["C", "N"]
        original = list(atoms)
        add_hydrogen(atoms, random.Random(0))
        assert atoms == original

    def test_parity_with_negative_charge(self) -> None:
        """Parity check works for negative charges."""
        rng = random.Random(99)
        atoms = ["O", "O"]  # 16 electrons, even; charge=-1 → 17 electrons
        result = add_hydrogen(atoms, rng, charge=-1, mult=2)
        total_z = sum(ATOMIC_NUMBERS[a] for a in result)
        n_e = total_z - (-1)
        target_parity = (2 - 1) % 2  # mult=2 → 1, odd
        assert n_e % 2 == target_parity


# ===========================================================================
# 2. StructureGenerator._adjust_parity()
# ===========================================================================

class TestAdjustParity:
    """_adjust_parity() nudges the atom list to satisfy charge/mult parity."""

    def _gen(self, charge: int = 0, mult: int = 1, elements: str = "1,6,7,8") -> StructureGenerator:
        return StructureGenerator(
            n_atoms=5, charge=charge, mult=mult, mode="gas",
            region="sphere:8", elements=elements, n_samples=1, seed=0,
        )

    def test_already_correct_unchanged(self) -> None:
        """No change when parity is already satisfied."""
        gen = self._gen()
        rng = random.Random(0)
        # 2×C → 12 electrons, even → valid for mult=1, charge=0
        atoms = ["C", "C"]
        result = gen._adjust_parity(atoms, rng)
        assert validate_charge_mult(result, 0, 1)[0]

    def test_adds_h_when_parity_wrong_and_h_in_pool(self) -> None:
        """Adds one H when parity is wrong and H is in the pool."""
        gen = self._gen(charge=0, mult=1, elements="1,6,7,8")
        rng = random.Random(0)
        # C+N = 13 electrons (odd) → fails mult=1 (even needed)
        atoms = ["C", "N"]
        result = gen._adjust_parity(atoms, rng)
        ok, _ = validate_charge_mult(result, 0, 1)
        assert ok, f"Parity should pass after adjust, got atoms={result}"

    def test_adjust_changes_at_most_one_atom(self) -> None:
        """Only ±1 atom change allowed."""
        gen = self._gen(charge=0, mult=1, elements="1,6,7,8")
        rng = random.Random(5)
        atoms = ["C", "N", "O", "C", "N"]
        result = gen._adjust_parity(atoms, rng)
        assert abs(len(result) - len(atoms)) <= 1

    def test_adjust_no_h_in_pool_swaps_element(self) -> None:
        """When H not in pool, swaps one atom to fix parity."""
        gen = StructureGenerator(
            n_atoms=3, charge=0, mult=1, mode="gas",
            region="sphere:8", elements="6,7", n_samples=1, seed=0,
        )
        rng = random.Random(3)
        # C+C+N = 6+6+7=19 electrons (odd) → fails mult=1
        atoms = ["C", "C", "N"]
        result = gen._adjust_parity(atoms, rng)
        ok, _ = validate_charge_mult(result, 0, 1)
        assert ok, f"Parity should pass after no-H swap, got {result}"

    def test_result_is_list(self) -> None:
        gen = self._gen()
        rng = random.Random(0)
        result = gen._adjust_parity(["C", "O"], rng)
        assert isinstance(result, list)

    def test_all_elements_from_pool_or_original(self) -> None:
        """Every symbol in the result must be from the pool or original atoms."""
        gen = self._gen(elements="1,6,7,8")
        pool_set = set(gen.element_pool)
        rng = random.Random(11)
        result = gen._adjust_parity(["C", "N", "C"], rng)
        for sym in result:
            assert sym in pool_set, f"Unexpected symbol {sym!r} not in pool {pool_set}"


# ===========================================================================
# 3. StructureGenerator._validate_density() — auto-scaling behaviour
# ===========================================================================

class TestValidateDensity:
    """Density validation auto-scales region instead of raising errors."""

    def test_low_packing_no_warning_no_scaling(self) -> None:
        """Packing well below WARN threshold → no warning, region unchanged."""
        region = _sphere_region_at_packing(5, pf=0.20)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gen = StructureGenerator(
                n_atoms=5, charge=0, mult=1, mode="gas",
                region=region, elements="6", n_samples=1, seed=0,
            )
        density_warns = [x for x in w if "Packing fraction" in str(x.message)]
        assert len(density_warns) == 0
        # effective_region must equal the original
        assert gen._effective_region == region

    def test_warn_threshold_auto_scales_and_warns(self) -> None:
        """pf ∈ (WARN, HARD] → UserWarning + region auto-scaled to target."""
        pf = (_PACKING_WARN + _PACKING_HARD) / 2  # ~0.57
        region = _sphere_region_at_packing(10, pf=pf)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gen = StructureGenerator(
                n_atoms=10, charge=0, mult=1, mode="gas",
                region=region, elements="6", n_samples=1, seed=0,
            )
        density_warns = [x for x in w if "Packing fraction" in str(x.message)]
        assert len(density_warns) >= 1, "Expected a UserWarning about packing fraction"
        # auto-scaled message must mention "auto-scaled"
        assert "auto-scaled" in str(density_warns[0].message)
        # effective_region must differ from the (too-small) original
        assert gen._effective_region != region
        assert gen._effective_region.startswith("sphere:")

    def test_hard_threshold_auto_scales_and_warns(self) -> None:
        """pf > HARD → UserWarning (not ValueError) + region auto-scaled."""
        pf = _PACKING_HARD + 0.10  # 0.74
        region = _sphere_region_at_packing(20, pf=pf)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Must NOT raise; auto-scale instead
            gen = StructureGenerator(
                n_atoms=20, charge=0, mult=1, mode="gas",
                region=region, elements="6", n_samples=1, seed=0,
            )
        density_warns = [x for x in w if "Packing fraction" in str(x.message)]
        assert len(density_warns) >= 1
        assert "auto-scaled" in str(density_warns[0].message)
        assert gen._effective_region != region

    def test_auto_scaled_region_is_physically_valid(self) -> None:
        """The auto-scaled region must yield pf ≤ PACKING_WARN."""
        pf = _PACKING_HARD + 0.10
        region = _sphere_region_at_packing(20, pf=pf)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            gen = StructureGenerator(
                n_atoms=20, charge=0, mult=1, mode="gas",
                region=region, elements="6", n_samples=1, seed=0,
            )
        # Verify the scaled region's packing fraction
        scaled = gen._effective_region
        pool = parse_element_spec("6")
        mean_r = float(np.mean([_cov_radius_ang(s) for s in pool]))
        atom_v = _atom_vol(mean_r)
        r = float(scaled.split(":")[1])
        scaled_vol = _sphere_vol(r)
        scaled_pf = 20 * atom_v / scaled_vol
        assert scaled_pf <= _PACKING_WARN, (
            f"Scaled region {scaled!r} still has pf={scaled_pf:.2f} > WARN={_PACKING_WARN}"
        )

    def test_box_region_auto_scales(self) -> None:
        """Box regions are auto-scaled the same as sphere regions."""
        pf = _PACKING_HARD + 0.10
        pool = parse_element_spec("6")
        mean_r = float(np.mean([_cov_radius_ang(s) for s in pool]))
        atom_v = _atom_vol(mean_r)
        n_atoms = 15
        region_vol = n_atoms * atom_v / pf
        L = region_vol ** (1 / 3)
        region = f"box:{L:.4f}"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gen = StructureGenerator(
                n_atoms=n_atoms, charge=0, mult=1, mode="gas",
                region=region, elements="6", n_samples=1, seed=0,
            )
        density_warns = [x for x in w if "Packing fraction" in str(x.message)]
        assert len(density_warns) >= 1
        assert gen._effective_region != region

    def test_maxent_mode_also_auto_scales(self) -> None:
        """Density auto-scaling also runs for maxent mode."""
        pf = _PACKING_HARD + 0.10
        region = _sphere_region_at_packing(10, pf=pf)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gen = StructureGenerator(
                n_atoms=10, charge=0, mult=1, mode="maxent",
                region=region, elements="6", n_samples=1, seed=0,
            )
        density_warns = [x for x in w if "Packing fraction" in str(x.message)]
        assert len(density_warns) >= 1
        assert gen._effective_region != region

    def test_chain_mode_skips_density_validation(self) -> None:
        """Chain mode has no region so no density check / scaling is run."""
        gen = StructureGenerator(
            n_atoms=50, charge=0, mult=1, mode="chain",
            elements="6", n_samples=1, seed=0,
        )
        assert gen is not None

    def test_unknown_region_spec_unchanged(self) -> None:
        """Unrecognised region spec passes through without scaling."""
        gen = _make_gen()
        result = gen._validate_density(10, "cylinder:5.0")
        assert result == "cylinder:5.0"

    def test_region_volume_sphere(self) -> None:
        gen = _make_gen()
        result = gen._region_volume("sphere:5.0")
        assert result is not None
        shape, vol = result
        assert shape == "sphere"
        assert abs(vol - _sphere_vol(5.0)) < 1e-6

    def test_region_volume_box_cubic(self) -> None:
        gen = _make_gen()
        result = gen._region_volume("box:4.0")
        assert result is not None
        shape, vol = result
        assert shape == "box"
        assert abs(vol - 4.0 ** 3) < 1e-6

    def test_region_volume_box_rectangular(self) -> None:
        gen = _make_gen()
        result = gen._region_volume("box:2.0,3.0,4.0")
        assert result is not None
        shape, vol = result
        assert shape == "box"
        assert abs(vol - 24.0) < 1e-6

    def test_region_volume_unknown_returns_none(self) -> None:
        gen = _make_gen()
        assert gen._region_volume("cylinder:5.0") is None

    def test_recommend_region_sphere(self) -> None:
        gen = _make_gen()
        rec = gen._recommend_region(10, mean_r=0.75, shape="sphere")
        assert rec.startswith("sphere:")
        r = float(rec.split(":")[1])
        assert r > 0

    def test_recommend_region_box(self) -> None:
        gen = _make_gen()
        rec = gen._recommend_region(10, mean_r=0.75, shape="box")
        assert rec.startswith("box:")
        L = float(rec.split(":")[1])
        assert L > 0

    def test_generation_proceeds_after_auto_scale(self) -> None:
        """Generator with an over-dense region must still produce structures."""
        pf = _PACKING_HARD + 0.10
        region = _sphere_region_at_packing(8, pf=pf)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            gen = StructureGenerator(
                n_atoms=8, charge=0, mult=1, mode="gas",
                region=region, elements="6,7,8", n_samples=3, seed=0,
            )
            result = gen.generate()
        # Should produce at least some structures
        assert len(result) > 0


# ===========================================================================
# 4. Integration: full generation with all v0.4.4 guards active
# ===========================================================================

class TestIntegrationV044:
    """End-to-end generation tests exercising v0.4.4 code paths."""

    def test_generation_with_hydrogen_pool_produces_valid_parity(self) -> None:
        """Structures generated with H in pool must all pass parity check."""
        gen = StructureGenerator(
            n_atoms=8, charge=0, mult=1, mode="gas",
            region="sphere:10", elements="1,6,7,8",
            n_samples=20, seed=42,
        )
        result = gen.generate()
        for s in result:
            ok, msg = validate_charge_mult(s.atoms, s.charge, s.mult)
            assert ok, f"Parity failed for {s.comp}: {msg}"

    def test_generation_normal_density_no_warnings(self) -> None:
        """A reasonably-sized region generates without density warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            StructureGenerator(
                n_atoms=5, charge=0, mult=1, mode="gas",
                region="sphere:10", elements="6,7,8",
                n_samples=5, seed=0,
            ).generate()
        density_warns = [x for x in w if "Packing fraction" in str(x.message)]
        assert len(density_warns) == 0

    def test_parity_gate_all_structures_valid(self) -> None:
        """All returned structures must pass the parity check."""
        gen = StructureGenerator(
            n_atoms=10, charge=0, mult=1, mode="gas",
            region="sphere:12", elements="1,6,7,8",
            n_samples=50, seed=7,
        )
        result = gen.generate()
        for s in result:
            ok, _ = validate_charge_mult(s.atoms, s.charge, s.mult)
            assert ok

    def test_auto_scaled_generation_produces_valid_structures(self) -> None:
        """Even when region is auto-scaled, all structures must be parity-valid."""
        pf = _PACKING_HARD + 0.05
        region = _sphere_region_at_packing(6, pf=pf)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            gen = StructureGenerator(
                n_atoms=6, charge=0, mult=1, mode="gas",
                region=region, elements="1,6,7,8",
                n_samples=10, seed=99,
            )
            result = gen.generate()
        for s in result:
            ok, msg = validate_charge_mult(s.atoms, s.charge, s.mult)
            assert ok, f"Parity failed after auto-scale: {s.comp}: {msg}"
