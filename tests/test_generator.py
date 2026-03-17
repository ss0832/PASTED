"""Tests for pasted._generator: StructureGenerator, Structure, generate()."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pasted import Structure, StructureGenerator, generate
from pasted._atoms import ALL_METRICS

# ---------------------------------------------------------------------------
# StructureGenerator: construction
# ---------------------------------------------------------------------------

class TestStructureGeneratorInit:
    def test_basic_gas(self) -> None:
        gen = StructureGenerator(
            n_atoms=6, charge=0, mult=1,
            mode="gas", region="sphere:6",
            elements="6,7,8", seed=0,
        )
        assert gen.n_atoms == 6
        assert gen.mode == "gas"

    def test_bad_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            StructureGenerator(
                n_atoms=5, charge=0, mult=1,
                mode="invalid",
            )

    def test_gas_without_region_raises(self) -> None:
        with pytest.raises(ValueError, match="region"):
            StructureGenerator(
                n_atoms=5, charge=0, mult=1,
                mode="gas",
            )

    def test_bad_center_z_raises(self) -> None:
        with pytest.raises(ValueError):
            StructureGenerator(
                n_atoms=5, charge=0, mult=1,
                mode="shell", elements="6,7,8",
                center_z=26,   # Fe not in pool
            )

    def test_element_pool_from_spec(self) -> None:
        gen = StructureGenerator(
            n_atoms=5, charge=0, mult=1,
            mode="chain", elements="6,7,8",
        )
        assert set(gen.element_pool) == {"C", "N", "O"}

    def test_element_pool_from_list(self) -> None:
        gen = StructureGenerator(
            n_atoms=5, charge=0, mult=1,
            mode="chain", elements=["C", "N", "O"],
        )
        assert "C" in gen.element_pool

    def test_element_pool_default(self) -> None:
        gen = StructureGenerator(
            n_atoms=5, charge=0, mult=1,
            mode="chain",
        )
        assert len(gen.element_pool) == 106

    def test_cutoff_positive(self) -> None:
        gen = StructureGenerator(
            n_atoms=5, charge=0, mult=1,
            mode="chain", elements="6,7,8",
        )
        assert gen.cutoff > 0

    def test_cutoff_override(self) -> None:
        gen = StructureGenerator(
            n_atoms=5, charge=0, mult=1,
            mode="chain", elements="6,7,8",
            cutoff=3.5,
        )
        assert gen.cutoff == pytest.approx(3.5)

    def test_repr(self) -> None:
        gen = StructureGenerator(
            n_atoms=6, charge=0, mult=1,
            mode="gas", region="sphere:5",
            elements="6,7,8",
        )
        r = repr(gen)
        assert "StructureGenerator" in r
        assert "gas" in r


# ---------------------------------------------------------------------------
# StructureGenerator: generate()
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_returns_list(self, gas_gen: StructureGenerator) -> None:
        results = gas_gen.generate()
        assert isinstance(results, list)

    def test_all_structures(self, gas_gen: StructureGenerator) -> None:
        results = gas_gen.generate()
        assert all(isinstance(s, Structure) for s in results)

    def test_reproducible_with_seed(self) -> None:
        def run() -> list[Structure]:
            return StructureGenerator(
                n_atoms=6, charge=0, mult=1,
                mode="gas", region="sphere:6",
                elements="6,7,8", n_samples=5, seed=42,
            ).generate()

        r1, r2 = run(), run()
        assert len(r1) == len(r2)
        for s1, s2 in zip(r1, r2, strict=True):
            assert s1.atoms == s2.atoms
            assert s1.positions == s2.positions

    def test_filter_applied(self) -> None:
        # Request only structures with H_total > 100 (impossible) → 0 results
        results = StructureGenerator(
            n_atoms=6, charge=0, mult=1,
            mode="gas", region="sphere:6",
            elements="6,7,8", n_samples=10, seed=0,
            filters=["H_total:100:-"],
        ).generate()
        assert results == []

    def test_chain_mode(self, chain_gen: StructureGenerator) -> None:
        results = chain_gen.generate()
        assert isinstance(results, list)

    def test_shell_mode(self, shell_gen: StructureGenerator) -> None:
        results = shell_gen.generate()
        assert isinstance(results, list)
        for s in results:
            assert s.center_sym is not None

    def test_iteration_protocol(self, gas_gen: StructureGenerator) -> None:
        structures = list(gas_gen)
        assert all(isinstance(s, Structure) for s in structures)

    def test_sample_index_sequential(self) -> None:
        results = StructureGenerator(
            n_atoms=6, charge=0, mult=1,
            mode="gas", region="sphere:6",
            elements="6,7,8", n_samples=10, seed=5,
        ).generate()
        for i, s in enumerate(results, start=1):
            assert s.sample_index == i


# ---------------------------------------------------------------------------
# Structure dataclass
# ---------------------------------------------------------------------------

class TestStructure:
    def _make_structure(self) -> Structure:
        results = StructureGenerator(
            n_atoms=6, charge=0, mult=1,
            mode="gas", region="sphere:6",
            elements="6,7,8", n_samples=3, seed=10,
        ).generate()
        assert results, "Need at least one structure"
        return results[0]

    def test_len(self) -> None:
        s = self._make_structure()
        assert len(s) == len(s.atoms)

    def test_metrics_complete(self) -> None:
        s = self._make_structure()
        assert set(s.metrics.keys()) == ALL_METRICS

    def test_metrics_finite(self) -> None:
        s = self._make_structure()
        for k, v in s.metrics.items():
            assert math.isfinite(v), f"{k} = {v}"

    def test_to_xyz_first_line(self) -> None:
        s = self._make_structure()
        xyz = s.to_xyz()
        lines = xyz.split("\n")
        assert lines[0] == str(len(s.atoms))

    def test_to_xyz_custom_prefix(self) -> None:
        s = self._make_structure()
        xyz = s.to_xyz(prefix="my_prefix")
        assert "my_prefix" in xyz.split("\n")[1]

    def test_to_xyz_contains_mode(self) -> None:
        s = self._make_structure()
        xyz = s.to_xyz()
        assert "mode=gas" in xyz

    def test_to_xyz_coord_count(self) -> None:
        s = self._make_structure()
        lines = s.to_xyz().strip().split("\n")
        # First line = count, second = comment, rest = coordinates
        assert len(lines) == len(s.atoms) + 2

    def test_write_xyz_creates_file(self, tmp_path: Path) -> None:
        s = self._make_structure()
        out = tmp_path / "test.xyz"
        s.write_xyz(out, append=False)
        assert out.exists()
        content = out.read_text()
        assert str(len(s.atoms)) in content

    def test_write_xyz_append(self, tmp_path: Path) -> None:
        structures = StructureGenerator(
            n_atoms=6, charge=0, mult=1,
            mode="gas", region="sphere:6",
            elements="6,7,8", n_samples=5, seed=11,
        ).generate()
        out = tmp_path / "multi.xyz"
        for i, s in enumerate(structures):
            s.write_xyz(out, append=(i > 0))
        lines = out.read_text().splitlines()
        # Should have (n_atoms+2) lines per structure
        expected_lines = sum(len(s.atoms) + 2 for s in structures)
        assert len(lines) == expected_lines

    def test_repr(self) -> None:
        s = self._make_structure()
        r = repr(s)
        assert "Structure" in r
        assert "H_total" in r


# ---------------------------------------------------------------------------
# generate() functional API
# ---------------------------------------------------------------------------

class TestGenerateFunction:
    def test_basic_call(self) -> None:
        results = generate(
            n_atoms=6, charge=0, mult=1,
            mode="gas", region="sphere:6",
            elements="6,7,8", n_samples=3, seed=0,
        )
        assert isinstance(results, list)

    def test_chain_mode(self) -> None:
        results = generate(
            n_atoms=8, charge=0, mult=1,
            mode="chain", elements="6,7,8",
            n_samples=3, seed=1,
        )
        assert isinstance(results, list)

    def test_same_output_as_class(self) -> None:
        kwargs: dict = {
            "n_atoms": 6, "charge": 0, "mult": 1,
            "mode": "gas", "region": "sphere:6",
            "elements": "6,7,8", "n_samples": 5, "seed": 99,
        }
        r_func = generate(**kwargs)
        r_class = StructureGenerator(**kwargs).generate()
        assert len(r_func) == len(r_class)
        for sf, sc in zip(r_func, r_class, strict=True):
            assert sf.atoms == sc.atoms
