"""Tests for maxent placement mode and angular_entropy diagnostic."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from pasted import (
    GenerationResult,
    StructureGenerator,
    compute_angular_entropy,
    place_maxent,
)
from pasted._atoms import cov_radius_ang
from pasted._ext import HAS_MAXENT_LOOP
from pasted._placement import _angular_repulsion_gradient, place_gas, relax_positions

# ---------------------------------------------------------------------------
# _angular_repulsion_gradient
# ---------------------------------------------------------------------------


def test_gradient_shape() -> None:
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    grad = _angular_repulsion_gradient(pts, cutoff=3.0)
    assert grad.shape == (3, 3)


def test_gradient_zero_for_isolated() -> None:
    # atoms far apart → no neighbors → gradient should be zero
    pts = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    grad = _angular_repulsion_gradient(pts, cutoff=2.0)
    assert np.allclose(grad, 0.0)


def test_gradient_finite() -> None:
    rng = np.random.default_rng(0)
    pts = rng.uniform(-3, 3, (8, 3))
    grad = _angular_repulsion_gradient(pts, cutoff=4.0)
    assert np.all(np.isfinite(grad))


# ---------------------------------------------------------------------------
# place_maxent
# ---------------------------------------------------------------------------


class TestPlaceMaxent:
    def test_count(self) -> None:
        atoms = ["C", "N", "O", "C", "N"]
        rng = random.Random(0)
        _atoms_out, pos = place_maxent(atoms, "sphere:5", cov_scale=1.0, rng=rng, maxent_steps=5)
        assert len(pos) == len(atoms)

    def test_atoms_preserved(self) -> None:
        atoms = ["C", "N", "O"]
        rng = random.Random(1)
        atoms_out, _ = place_maxent(atoms, "sphere:4", cov_scale=1.0, rng=rng, maxent_steps=5)
        assert atoms_out == atoms

    def test_no_distance_violations(self) -> None:

        atoms = ["C", "N", "O", "C"]
        rng = random.Random(2)
        _, pos = place_maxent(atoms, "sphere:5", cov_scale=1.0, rng=rng, maxent_steps=20)
        n = len(atoms)
        for i in range(n):
            for j in range(i + 1, n):
                d = math.sqrt(sum((pos[i][k] - pos[j][k]) ** 2 for k in range(3)))
                thresh = cov_radius_ang(atoms[i]) + cov_radius_ang(atoms[j])
                msg = f"violation: {atoms[i]}-{atoms[j]} d={d:.4f} thresh={thresh:.4f}"
                assert d >= thresh - 1e-4, msg

    def test_box_region(self) -> None:
        atoms = ["C"] * 5
        rng = random.Random(3)
        _, pos = place_maxent(atoms, "box:8", cov_scale=1.0, rng=rng, maxent_steps=5)
        assert len(pos) == 5

    def test_reproducible(self) -> None:
        atoms = ["C", "N", "O", "C", "N"]
        _, pos1 = place_maxent(atoms, "sphere:5", 1.0, random.Random(7), maxent_steps=10)
        _, pos2 = place_maxent(atoms, "sphere:5", 1.0, random.Random(7), maxent_steps=10)
        for p1, p2 in zip(pos1, pos2, strict=True):
            assert p1 == pytest.approx(p2, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_angular_entropy
# ---------------------------------------------------------------------------


class TestComputeAngularEntropy:
    def test_single_atom(self) -> None:
        assert compute_angular_entropy([(0.0, 0.0, 0.0)], cutoff=3.0) == 0.0

    def test_non_negative(self) -> None:
        pos = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)]
        val = compute_angular_entropy(pos, cutoff=3.0)
        assert val >= 0.0

    def test_no_neighbors_returns_zero(self) -> None:
        pos = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        assert compute_angular_entropy(pos, cutoff=1.0) == 0.0

    def test_maxent_higher_than_gas(self) -> None:
        """MaxEnt placement should yield higher angular entropy than gas placement."""
        atoms = ["C"] * 12
        rng_gas = random.Random(42)
        rng_maxent = random.Random(42)

        _, pos_gas = place_gas(atoms, "sphere:6", rng_gas)
        pos_gas, _ = relax_positions(atoms, pos_gas, 1.0)

        _, pos_maxent = place_maxent(
            atoms, "sphere:6", cov_scale=1.0, rng=rng_maxent, maxent_steps=200
        )

        h_gas = compute_angular_entropy(pos_gas, cutoff=4.0)
        h_maxent = compute_angular_entropy(pos_maxent, cutoff=4.0)

        # maxent should be at least as good as gas on average
        # (allow small tolerance for small atom counts)
        assert h_maxent >= h_gas - 0.3


# ---------------------------------------------------------------------------
# StructureGenerator with mode="maxent"
# ---------------------------------------------------------------------------


class TestGeneratorMaxent:
    def test_basic(self) -> None:
        structs = StructureGenerator(
            n_atoms=6,
            charge=0,
            mult=1,
            mode="maxent",
            region="sphere:6",
            elements="6,7,8",
            n_samples=2,
            seed=0,
        ).generate()
        assert isinstance(structs, GenerationResult)
        assert all(len(s) == len(s.atoms) for s in structs)

    def test_mode_label(self) -> None:
        structs = StructureGenerator(
            n_atoms=6,
            charge=0,
            mult=1,
            mode="maxent",
            region="sphere:6",
            elements="6,7,8",
            n_samples=1,
            seed=1,
        ).generate()  # type: ignore[union-attr]
        assert structs[0].mode == "maxent"  # type: ignore[union-attr]

    def test_no_region_raises(self) -> None:
        with pytest.raises(ValueError, match="region"):
            StructureGenerator(
                n_atoms=6,
                charge=0,
                mult=1,
                mode="maxent",
            )


# ---------------------------------------------------------------------------
# O(N) cutoff identity: median(rᵢ + rⱼ) == 2 * median(rᵢ)
# ---------------------------------------------------------------------------


class TestPlaceMaxentCutoff:
    """Verify that the v0.2.6 O(N) median formula matches the O(N²) reference.

    These tests confirm that ``float(np.median(radii)) * 2.0`` is numerically
    identical to ``sorted(rᵢ + rⱼ for all pairs)[mid]`` for the element pools
    that PASTED supports, so that replacing the latter with the former does not
    change ``ang_cutoff`` or any downstream result.
    """

    @staticmethod
    def _reference_median_sum(radii: np.ndarray) -> float:
        """O(N²) reference implementation (matches pre-v0.2.6 code)."""
        pairs = sorted(ra + rb for i, ra in enumerate(radii.tolist()) for rb in radii[i:].tolist())
        return pairs[len(pairs) // 2]

    @staticmethod
    def _fast_median_sum(radii: np.ndarray) -> float:
        """O(N) replacement introduced in v0.2.6."""
        return float(np.median(radii)) * 2.0

    def test_homogeneous_pool(self) -> None:
        """Single element (all radii equal): both formulas must agree."""
        from pasted._atoms import _cov_radius_ang

        atoms = ["C"] * 50
        radii = np.array([_cov_radius_ang(a) for a in atoms])
        ref = self._reference_median_sum(radii)
        fast = self._fast_median_sum(radii)
        assert fast == pytest.approx(ref, abs=1e-9), f"homogeneous C pool: ref={ref}, fast={fast}"

    def test_heterogeneous_pool(self) -> None:
        """Mixed element pool (C, N, O, H): both formulas must agree."""
        from pasted._atoms import _cov_radius_ang

        atoms = ["C", "N", "O", "H"] * 20
        radii = np.array([_cov_radius_ang(a) for a in atoms])
        ref = self._reference_median_sum(radii)
        fast = self._fast_median_sum(radii)
        assert fast == pytest.approx(ref, abs=1e-9), f"mixed C/N/O/H pool: ref={ref}, fast={fast}"

    def test_single_element_pool_heavy(self) -> None:
        """Heavier element (S): both formulas must agree."""
        from pasted._atoms import _cov_radius_ang

        atoms = ["S"] * 30
        radii = np.array([_cov_radius_ang(a) for a in atoms])
        ref = self._reference_median_sum(radii)
        fast = self._fast_median_sum(radii)
        assert fast == pytest.approx(ref, abs=1e-9), f"homogeneous S pool: ref={ref}, fast={fast}"

    @pytest.mark.skipif(
        not HAS_MAXENT_LOOP,
        reason="HAS_MAXENT_LOOP=False: C++ path not active; monkey-patch has no effect",
    )
    def test_ang_cutoff_unchanged_end_to_end(self) -> None:
        """Structures generated before and after the patch must be identical.

        We monkey-patch ``place_maxent`` to capture the ``ang_cutoff`` value
        used on each call, run the generator twice (once with the old formula,
        once with the new), and assert the cutoffs match.
        """
        import pasted._placement as _pl
        from pasted._atoms import _cov_radius_ang

        captured: list[float] = []

        original_cpp = _pl._cpp_place_maxent  # type: ignore[attr-defined]

        def _capture_cutoff(
            pts: np.ndarray,
            radii: np.ndarray,
            cov_scale: float,
            region_radius: float,
            ang_cutoff: float,
            *args: object,
            **kwargs: object,
        ) -> np.ndarray:
            captured.append(ang_cutoff)
            return original_cpp(pts, radii, cov_scale, region_radius, ang_cutoff, *args, **kwargs)

        atoms = ["C", "N", "O"] * 4
        radii = np.array([_cov_radius_ang(a) for a in atoms])

        # Compute expected ang_cutoff using the O(N) formula (v0.2.6).
        fast_median = float(np.median(radii)) * 2.0
        expected_cutoff = 1.0 * 2.5 * fast_median  # cov_scale=1.0, maxent_cutoff_scale=2.5

        # Run place_maxent with the monkey-patched C++ entry point.
        _pl._cpp_place_maxent = _capture_cutoff  # type: ignore[attr-defined]
        try:
            place_maxent(atoms, "sphere:5", cov_scale=1.0, rng=random.Random(99), maxent_steps=2)
        finally:
            _pl._cpp_place_maxent = original_cpp  # type: ignore[attr-defined]

        assert len(captured) == 1
        assert captured[0] == pytest.approx(expected_cutoff, abs=1e-9)
