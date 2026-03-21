"""Tests for pasted._generator: StructureGenerator, Structure, generate()."""

from __future__ import annotations

import math
import warnings
from collections import Counter
from pathlib import Path

import pytest

from pasted import GenerationResult, Structure, StructureGenerator, generate
from pasted._atoms import ALL_METRICS

# ---------------------------------------------------------------------------
# StructureGenerator: construction
# ---------------------------------------------------------------------------


class TestStructureGeneratorInit:
    def test_basic_gas(self) -> None:
        gen = StructureGenerator(
            n_atoms=6,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:6",
            elements="6,7,8",
            seed=0,
        )
        assert gen.n_atoms == 6
        assert gen.mode == "gas"

    def test_bad_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            StructureGenerator(
                n_atoms=5,
                charge=0,
                mult=1,
                mode="invalid",
            )

    def test_gas_without_region_raises(self) -> None:
        with pytest.raises(ValueError, match="region"):
            StructureGenerator(
                n_atoms=5,
                charge=0,
                mult=1,
                mode="gas",
            )

    def test_bad_center_z_raises(self) -> None:
        with pytest.raises(ValueError):
            StructureGenerator(
                n_atoms=5,
                charge=0,
                mult=1,
                mode="shell",
                elements="6,7,8",
                center_z=26,  # Fe not in pool
            )

    def test_element_pool_from_spec(self) -> None:
        gen = StructureGenerator(
            n_atoms=5,
            charge=0,
            mult=1,
            mode="chain",
            elements="6,7,8",
        )
        assert set(gen.element_pool) == {"C", "N", "O"}

    def test_element_pool_from_list(self) -> None:
        gen = StructureGenerator(
            n_atoms=5,
            charge=0,
            mult=1,
            mode="chain",
            elements=["C", "N", "O"],
        )
        assert "C" in gen.element_pool

    def test_element_pool_default(self) -> None:
        gen = StructureGenerator(
            n_atoms=5,
            charge=0,
            mult=1,
            mode="chain",
        )
        assert len(gen.element_pool) == 106

    def test_cutoff_positive(self) -> None:
        gen = StructureGenerator(
            n_atoms=5,
            charge=0,
            mult=1,
            mode="chain",
            elements="6,7,8",
        )
        assert gen.cutoff > 0

    def test_cutoff_override(self) -> None:
        gen = StructureGenerator(
            n_atoms=5,
            charge=0,
            mult=1,
            mode="chain",
            elements="6,7,8",
            cutoff=3.5,
        )
        assert gen.cutoff == pytest.approx(3.5)

    def test_repr(self) -> None:
        gen = StructureGenerator(
            n_atoms=6,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:5",
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
        assert isinstance(results, GenerationResult)

    def test_all_structures(self, gas_gen: StructureGenerator) -> None:
        results = gas_gen.generate()
        assert all(isinstance(s, Structure) for s in results)

    def test_reproducible_with_seed(self) -> None:
        def run() -> GenerationResult:
            return StructureGenerator(
                n_atoms=6,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:6",
                elements="6,7,8",
                n_samples=5,
                seed=42,
            ).generate()

        r1, r2 = run(), run()
        assert len(r1) == len(r2)
        for s1, s2 in zip(r1, r2, strict=True):
            assert s1.atoms == s2.atoms
            assert s1.positions == s2.positions

    def test_filter_applied(self) -> None:
        # Request only structures with H_total > 100 (impossible) → 0 results
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = StructureGenerator(
                n_atoms=6,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:6",
                elements="6,7,8",
                n_samples=10,
                seed=0,
                filters=["H_total:100:-"],
            ).generate()
        assert len(results) == 0

    def test_chain_mode(self, chain_gen: StructureGenerator) -> None:
        results = chain_gen.generate()
        assert isinstance(results, GenerationResult)

    def test_shell_mode(self, shell_gen: StructureGenerator) -> None:
        results = shell_gen.generate()
        assert isinstance(results, GenerationResult)
        for s in results:
            assert s.center_sym is not None

    def test_iteration_protocol(self, gas_gen: StructureGenerator) -> None:
        structures = list(gas_gen)
        assert all(isinstance(s, Structure) for s in structures)

    def test_sample_index_sequential(self) -> None:
        results = StructureGenerator(
            n_atoms=6,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:6",
            elements="6,7,8",
            n_samples=10,
            seed=5,
        ).generate()
        for i, s in enumerate(results, start=1):
            assert s.sample_index == i


# ---------------------------------------------------------------------------
# Structure dataclass
# ---------------------------------------------------------------------------


class TestStructure:
    def _make_structure(self) -> Structure:
        results = StructureGenerator(
            n_atoms=6,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:6",
            elements="6,7,8",
            n_samples=3,
            seed=10,
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
            n_atoms=6,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:6",
            elements="6,7,8",
            n_samples=5,
            seed=11,
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
            n_atoms=6,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:6",
            elements="6,7,8",
            n_samples=3,
            seed=0,
        )
        assert isinstance(results, GenerationResult)

    def test_chain_mode(self) -> None:
        results = generate(
            n_atoms=8,
            charge=0,
            mult=1,
            mode="chain",
            elements="6,7,8",
            n_samples=3,
            seed=1,
        )
        assert isinstance(results, GenerationResult)

    def test_same_output_as_class(self) -> None:
        kwargs: dict = {
            "n_atoms": 6,
            "charge": 0,
            "mult": 1,
            "mode": "gas",
            "region": "sphere:6",
            "elements": "6,7,8",
            "n_samples": 5,
            "seed": 99,
        }
        r_func = generate(**kwargs)
        r_class = StructureGenerator(**kwargs).generate()
        assert len(r_func) == len(r_class)
        for sf, sc in zip(r_func, r_class, strict=True):
            assert sf.atoms == sc.atoms


# ---------------------------------------------------------------------------
# n_success and stream()
# ---------------------------------------------------------------------------


class TestNSuccess:
    def _gen(self, n_success: int | None, n_samples: int) -> StructureGenerator:
        return StructureGenerator(
            n_atoms=6,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:6",
            elements="6,7,8",
            n_samples=n_samples,
            n_success=n_success,
            seed=0,
        )

    def test_n_success_stops_early(self) -> None:
        """stream() stops as soon as n_success structures have been yielded."""
        gen = self._gen(n_success=2, n_samples=200)
        results = list(gen.stream())
        assert len(results) == 2

    def test_n_success_returns_partial_on_exhaustion(self) -> None:
        """When n_samples is exhausted before n_success, return what was collected."""
        gen = self._gen(n_success=1000, n_samples=3)
        results = list(gen.stream())
        assert len(results) <= 3

    def test_n_success_none_uses_n_samples(self) -> None:
        """Without n_success, stream() behaves identically to the original generate()."""
        gen = self._gen(n_success=None, n_samples=5)
        via_stream = list(gen.stream())
        via_generate = gen.generate()
        assert len(via_stream) == len(via_generate)
        for a, b in zip(via_stream, via_generate, strict=True):
            assert a.atoms == b.atoms

    def test_n_samples_zero_requires_n_success(self) -> None:
        """n_samples=0 without n_success must raise ValueError."""
        with pytest.raises(ValueError, match="n_success"):
            StructureGenerator(
                n_atoms=6, charge=0, mult=1,
                mode="gas", region="sphere:6",
                elements="6,7,8",
                n_samples=0,
            )

    def test_n_samples_zero_unlimited(self) -> None:
        """n_samples=0 with n_success runs until n_success is reached."""
        gen = StructureGenerator(
            n_atoms=6, charge=0, mult=1,
            mode="gas", region="sphere:6",
            elements="6,7,8",
            n_samples=0,
            n_success=3,
            seed=1,
        )
        results = list(gen.stream())
        assert len(results) == 3

    def test_sample_index_sequential_with_n_success(self) -> None:
        """sample_index must be 1-based and sequential even when stopping early."""
        gen = self._gen(n_success=3, n_samples=200)
        results = list(gen.stream())
        for expected, s in enumerate(results, start=1):
            assert s.sample_index == expected

    def test_generate_delegates_to_stream(self) -> None:
        """generate() and stream() must yield the same structures in the same order."""
        gen = self._gen(n_success=3, n_samples=100)
        via_stream = list(gen.stream())
        via_generate = gen.generate()
        assert len(via_stream) == len(via_generate)
        for a, b in zip(via_stream, via_generate, strict=True):
            assert a.atoms == b.atoms

    def test_stream_write_xyz(self, tmp_path: Path) -> None:
        """Each yielded structure should be writable to XYZ immediately."""
        out = tmp_path / "stream.xyz"
        gen = self._gen(n_success=2, n_samples=100)
        for s in gen.stream():
            s.write_xyz(str(out))
        lines = out.read_text().splitlines()
        # Each XYZ block starts with atom count; we should have 2 blocks
        atom_count_lines = [ln for ln in lines if ln.strip().isdigit()]
        assert len(atom_count_lines) == 2


# ---------------------------------------------------------------------------
# GenerationResult
# ---------------------------------------------------------------------------


class TestGenerationResult:
    def test_is_list_compatible(self) -> None:
        result = GenerationResult(
            structures=[],
            n_attempted=10,
            n_passed=0,
            n_rejected_parity=0,
            n_rejected_filter=10,
        )
        assert len(result) == 0
        assert not result
        assert list(result) == []

    def test_indexing(self) -> None:
        s = Structure(
            atoms=["C"],
            positions=[(0.0, 0.0, 0.0)],
            charge=0,
            mult=1,
            metrics=dict.fromkeys(ALL_METRICS, 0.0),
            mode="gas",
        )
        result = GenerationResult(structures=[s], n_attempted=1, n_passed=1)
        assert result[0] is s

    def test_bool_true_when_structures(self) -> None:
        s = Structure(
            atoms=["C"],
            positions=[(0.0, 0.0, 0.0)],
            charge=0,
            mult=1,
            metrics=dict.fromkeys(ALL_METRICS, 0.0),
            mode="gas",
        )
        result = GenerationResult(structures=[s], n_attempted=1, n_passed=1)
        assert result

    def test_summary_contains_key_fields(self) -> None:
        result = GenerationResult(
            structures=[],
            n_attempted=20,
            n_passed=0,
            n_rejected_parity=5,
            n_rejected_filter=15,
        )
        s = result.summary()
        assert "attempted=20" in s
        assert "rejected_parity=5" in s
        assert "rejected_filter=15" in s

    def test_repr(self) -> None:
        result = GenerationResult(
            structures=[], n_attempted=5, n_passed=0,
            n_rejected_parity=0, n_rejected_filter=5,
        )
        assert "GenerationResult" in repr(result)

    def test_generate_returns_generation_result(self) -> None:
        result = generate(
            n_atoms=6, charge=0, mult=1,
            mode="gas", region="sphere:6",
            elements="6,8",  # C + O: even electrons, no parity warnings
            n_samples=5, seed=0,
        )
        assert isinstance(result, GenerationResult)

    def test_metadata_counts_sum_correctly(self) -> None:
        # Use C+O (even electrons) so parity never fires; all rejections
        # come from filters or pass cleanly.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = generate(
                n_atoms=6, charge=0, mult=1,
                mode="gas", region="sphere:6",
                elements="6,8",  # C + O only
                n_samples=20, seed=42,
            )
        assert result.n_passed == len(result.structures)
        assert result.n_attempted == (
            result.n_passed + result.n_rejected_parity + result.n_rejected_filter
        )

    def test_warn_on_filter_rejection(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = generate(
                n_atoms=8, charge=0, mult=1,
                mode="gas", region="sphere:8",
                elements="6,8",  # C + O: always even electrons, no parity failures
                n_samples=10, seed=0,
                filters=["H_total:999:-"],  # impossible threshold
            )
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) >= 1
        assert "filter" in str(user_warns[0].message).lower()
        assert result.n_rejected_filter == 10
        assert len(result) == 0

    def test_warn_on_parity_rejection(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # charge=99 makes parity impossible for any C/N/O composition
            result = generate(
                n_atoms=6, charge=99, mult=1,
                mode="gas", region="sphere:6",
                elements="6,7,8", n_samples=10, seed=0,
            )
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) >= 1
        assert "parity" in str(user_warns[0].message).lower()
        assert result.n_rejected_parity == result.n_attempted

    def test_no_spurious_warn_on_clean_run(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Use C + O only (both even electrons) so the parity check
            # never fires; with no filters all structures pass cleanly.
            generate(
                n_atoms=6, charge=0, mult=1,
                mode="gas", region="sphere:6",
                elements="6,8",  # C + O: always satisfies charge=0, mult=1
                n_samples=10, seed=0,
            )
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warns) == 0


# ---------------------------------------------------------------------------
# GenerationResult.__add__
# ---------------------------------------------------------------------------


class TestGenerationResultAdd:
    def test_add_combines_structures(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1 = generate(
                n_atoms=6, charge=0, mult=1, mode="gas", region="sphere:6",
                elements="6,8", n_samples=3, seed=0,
            )
            r2 = generate(
                n_atoms=6, charge=0, mult=1, mode="gas", region="sphere:6",
                elements="6,8", n_samples=4, seed=1,
            )
        combined = r1 + r2
        assert len(combined) == len(r1) + len(r2)
        assert isinstance(combined, GenerationResult)

    def test_add_accumulates_counters(self) -> None:
        r1 = GenerationResult(
            structures=[], n_attempted=10, n_passed=4,
            n_rejected_parity=2, n_rejected_filter=4,
        )
        r2 = GenerationResult(
            structures=[], n_attempted=8, n_passed=3,
            n_rejected_parity=1, n_rejected_filter=4,
        )
        combined = r1 + r2
        assert combined.n_attempted == 18
        assert combined.n_passed == 7
        assert combined.n_rejected_parity == 3
        assert combined.n_rejected_filter == 8

    def test_add_preserves_n_success_target_from_left(self) -> None:
        r1 = GenerationResult(structures=[], n_success_target=10)
        r2 = GenerationResult(structures=[], n_success_target=20)
        assert (r1 + r2).n_success_target == 10

    def test_add_takes_right_when_left_is_none(self) -> None:
        r1 = GenerationResult(structures=[], n_success_target=None)
        r2 = GenerationResult(structures=[], n_success_target=5)
        assert (r1 + r2).n_success_target == 5

    def test_add_returns_notimplemented_for_non_result(self) -> None:
        r = GenerationResult(structures=[])
        result = r.__add__([])  # type: ignore[arg-type]
        assert result is NotImplemented


# ---------------------------------------------------------------------------
# parity warning threshold
# ---------------------------------------------------------------------------


class TestParityWarningThreshold:
    def test_no_warn_on_partial_parity_failure(self) -> None:
        # C/N/O mix: N has odd Z → some parity failures, but some pass
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = generate(
                n_atoms=6, charge=0, mult=1, mode="gas", region="sphere:6",
                elements="6,7,8", n_samples=20, seed=0,
            )
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        # Warning should fire only when n_passed == 0; partial failures are silent
        assert len(result) > 0, "Expected some structures to pass"
        assert len(user_warns) == 0, f"Unexpected warnings: {[str(x.message) for x in user_warns]}"

    def test_warn_on_complete_parity_failure(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = generate(
                n_atoms=6, charge=99, mult=1, mode="gas", region="sphere:6",
                elements="6,7,8", n_samples=5, seed=0,
            )
        user_warns = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(result) == 0
        assert len(user_warns) >= 1
        assert "parity" in str(user_warns[0].message).lower()


# ---------------------------------------------------------------------------
# Element fractions / min counts / max counts
# ---------------------------------------------------------------------------


class TestElementFractions:
    def _gen(self, **kwargs: object) -> StructureGenerator:
        defaults: dict[str, object] = {
            "n_atoms": 12,
            "charge": 0,
            "mult": 1,
            "mode": "gas",
            "region": "sphere:8",
            "elements": "6,7,8",
            "n_samples": 10,
            "seed": 42,
        }
        defaults.update(kwargs)
        return StructureGenerator(**defaults)  # type: ignore[arg-type]

    def test_fractions_biases_toward_heavy_element(self) -> None:
        """C-heavy fractions should produce more C atoms than uniform sampling."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_biased = self._gen(
                element_fractions={"C": 10.0, "N": 1.0, "O": 1.0},
                n_samples=30,
            ).generate()
            result_uniform = self._gen(n_samples=30).generate()

        def c_fraction(r: GenerationResult) -> float:
            total = sum(len(s.atoms) for s in r)
            c_count = sum(Counter(s.atoms).get("C", 0) for s in r)
            return c_count / total if total else 0.0

        assert c_fraction(result_biased) > c_fraction(result_uniform), (
            "C-heavy fractions should produce more C than uniform"
        )

    def test_fractions_unknown_element_raises(self) -> None:
        with pytest.raises(ValueError, match="element pool"):
            self._gen(element_fractions={"Fe": 1.0})

    def test_fractions_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            self._gen(element_fractions={"C": -1.0, "N": 1.0})

    def test_fractions_all_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="all be zero"):
            self._gen(element_fractions={"C": 0.0, "N": 0.0, "O": 0.0})

    def test_fractions_uniform_equivalent_to_none(self) -> None:
        """Equal weights should behave the same as no fractions (same atoms)."""
        gen_none = self._gen(seed=7)
        gen_uniform = self._gen(
            seed=7, element_fractions={"C": 1.0, "N": 1.0, "O": 1.0}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_none = gen_none.generate()
            r_uniform = gen_uniform.generate()
        # Same seed + equal weights → same atoms lists
        assert [s.atoms for s in r_none] == [s.atoms for s in r_uniform]

    def test_fractions_passed_to_generate_functional(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = generate(
                n_atoms=8,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:7",
                elements="6,7,8",
                element_fractions={"C": 5.0, "N": 1.0, "O": 1.0},
                n_samples=10,
                seed=0,
            )
        assert len(result) >= 0  # no crash; fractions accepted


class TestElementMinMaxCounts:
    def _gen(self, **kwargs: object) -> StructureGenerator:
        defaults: dict[str, object] = {
            "n_atoms": 10,
            "charge": 0,
            "mult": 1,
            "mode": "gas",
            "region": "sphere:8",
            "elements": "6,7,8",
            "n_samples": 10,
            "seed": 0,
        }
        defaults.update(kwargs)
        return StructureGenerator(**defaults)  # type: ignore[arg-type]

    def test_min_counts_respected(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self._gen(element_min_counts={"C": 3}, n_samples=20).generate()
        assert len(result) > 0
        for s in result:
            assert Counter(s.atoms).get("C", 0) >= 3, (
                f"C min violated: {Counter(s.atoms)}"
            )

    def test_max_counts_respected(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self._gen(element_max_counts={"N": 2}, n_samples=20).generate()
        assert len(result) > 0
        for s in result:
            assert Counter(s.atoms).get("N", 0) <= 2, (
                f"N max violated: {Counter(s.atoms)}"
            )

    def test_min_and_max_together(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self._gen(
                element_min_counts={"C": 2},
                element_max_counts={"C": 4, "N": 3},
                n_samples=20,
            ).generate()
        assert len(result) > 0
        for s in result:
            c = Counter(s.atoms)
            assert c.get("C", 0) >= 2
            assert c.get("C", 0) <= 4
            assert c.get("N", 0) <= 3

    def test_min_exceeds_n_atoms_raises(self) -> None:
        with pytest.raises(ValueError, match="n_atoms"):
            self._gen(element_min_counts={"C": 8, "N": 5})

    def test_min_gt_max_raises(self) -> None:
        with pytest.raises(ValueError, match="element_min_counts"):
            self._gen(
                element_min_counts={"C": 5},
                element_max_counts={"C": 3},
            )

    def test_unknown_min_element_raises(self) -> None:
        with pytest.raises(ValueError, match="element pool"):
            self._gen(element_min_counts={"Fe": 1})

    def test_unknown_max_element_raises(self) -> None:
        with pytest.raises(ValueError, match="element pool"):
            self._gen(element_max_counts={"Fe": 2})

    def test_impossible_cap_raises_runtime_error(self) -> None:
        """Cap every element to 0 → RuntimeError during sampling."""
        gen = self._gen(element_max_counts={"C": 0, "N": 0, "O": 0})
        with pytest.raises(RuntimeError, match="capped"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                list(gen.stream())


# ---------------------------------------------------------------------------
# GeneratorConfig dataclass tests (v0.2.3)
# ---------------------------------------------------------------------------

class TestGeneratorConfig:
    """Tests for the GeneratorConfig frozen dataclass and dual-mode StructureGenerator."""

    def _cfg(self, **kw):
        from pasted import GeneratorConfig
        defaults = dict(n_atoms=8, charge=0, mult=1, mode="gas",
                        region="sphere:6", elements="6,7,8", n_samples=3, seed=0)
        defaults.update(kw)
        return GeneratorConfig(**defaults)

    def test_config_exported_from_pasted(self) -> None:
        from pasted import GeneratorConfig
        assert GeneratorConfig is not None

    def test_frozen_prevents_mutation(self) -> None:
        import dataclasses
        cfg = self._cfg()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.n_atoms = 99  # type: ignore[misc]

    def test_dataclasses_replace(self) -> None:
        import dataclasses
        cfg = self._cfg(seed=1)
        cfg2 = dataclasses.replace(cfg, seed=42)
        assert cfg2.seed == 42
        assert cfg.seed == 1  # original unchanged

    def test_config_based_construction(self) -> None:
        from pasted import StructureGenerator
        cfg = self._cfg()
        gen = StructureGenerator(cfg)
        assert gen.config is cfg

    def test_kwargs_construction_backward_compat(self) -> None:
        """Old-style StructureGenerator(n_atoms=..., ...) must still work."""
        gen = StructureGenerator(
            n_atoms=8, charge=0, mult=1, mode="gas",
            region="sphere:6", elements="6,7,8", n_samples=3, seed=0,
        )
        assert gen.n_atoms == 8

    def test_config_and_kwargs_produce_same_result(self) -> None:
        """Config-based and kwargs-based construction must yield identical outputs."""
        from pasted import StructureGenerator
        cfg = self._cfg(n_samples=5, seed=77)
        r_cfg = StructureGenerator(cfg).generate()
        r_kw = StructureGenerator(
            n_atoms=8, charge=0, mult=1, mode="gas", region="sphere:6",
            elements="6,7,8", n_samples=5, seed=77,
        ).generate()
        assert r_cfg.n_passed == r_kw.n_passed

    def test_getattr_proxy_all_fields(self) -> None:
        """gen.seed, gen.mode, etc. should all proxy to _cfg."""
        from pasted import StructureGenerator
        cfg = self._cfg(seed=123)
        gen = StructureGenerator(cfg)
        assert gen.seed == 123
        assert gen.mode == "gas"
        assert gen.n_atoms == 8

    def test_config_missing_required_fields_raises(self) -> None:
        from pasted import GeneratorConfig
        with pytest.raises(TypeError):
            GeneratorConfig()  # type: ignore[call-arg]  # n_atoms required

    def test_generate_func_accepts_config(self) -> None:
        """generate(cfg) config-based call must work."""
        from pasted import generate
        cfg = self._cfg(n_samples=2)
        result = generate(cfg)
        assert len(result) >= 0  # at least ran without error


# ---------------------------------------------------------------------------
# affine_strength tests (v0.2.3)
# ---------------------------------------------------------------------------

class TestAffineStrength:
    """Tests for affine_strength in StructureGenerator."""

    def _gen(self, affine_strength: float, mode: str = "gas", **kw):
        defaults = dict(n_atoms=10, charge=0, mult=1, mode=mode,
                        region="sphere:7", elements="6,8",
                        n_samples=5, seed=42, affine_strength=affine_strength)
        if mode != "gas":
            defaults.pop("region", None)
        defaults.update(kw)
        if mode == "chain":
            defaults.pop("region", None)
        return StructureGenerator(**defaults)

    def test_zero_strength_is_default(self) -> None:
        """affine_strength=0.0 must behave identically to no-affine (backward compat)."""
        r_no = StructureGenerator(
            n_atoms=10, charge=0, mult=1, mode="gas", region="sphere:7",
            elements="6,8", n_samples=5, seed=99,
        ).generate()
        r_zero = StructureGenerator(
            n_atoms=10, charge=0, mult=1, mode="gas", region="sphere:7",
            elements="6,8", n_samples=5, seed=99, affine_strength=0.0,
        ).generate()
        assert r_no.n_passed == r_zero.n_passed

    def test_nonzero_strength_changes_positions(self) -> None:
        """With affine_strength > 0, positions must differ from strength=0."""
        import numpy as np
        def get_pos(strength):
            gen = StructureGenerator(
                n_atoms=10, charge=0, mult=1, mode="gas", region="sphere:7",
                elements="6,8", n_samples=1, seed=7, affine_strength=strength,
            )
            r = gen.generate()
            if not r:
                return None
            return np.array(r[0].positions)

        p0 = get_pos(0.0)
        p1 = get_pos(0.3)
        if p0 is not None and p1 is not None:
            assert not np.allclose(p0, p1, atol=1e-6), \
                "affine_strength=0.3 should change positions"

    def test_affine_applies_to_chain_mode(self) -> None:
        gen = StructureGenerator(
            n_atoms=10, charge=0, mult=1, mode="chain",
            elements="6,8", n_samples=3, seed=5, affine_strength=0.15,
        )
        r = gen.generate()
        assert r.n_passed >= 0  # just checks it runs without error

    def test_affine_applies_to_shell_mode(self) -> None:
        gen = StructureGenerator(
            n_atoms=8, charge=0, mult=1, mode="shell",
            elements="6,8,26", n_samples=3, seed=5, affine_strength=0.1,
        )
        r = gen.generate()
        assert r.n_passed >= 0

    def test_affine_strength_stored_in_config(self) -> None:
        gen = StructureGenerator(
            n_atoms=8, charge=0, mult=1, mode="gas", region="sphere:6",
            elements="6,8", n_samples=1, seed=0, affine_strength=0.2,
        )
        assert gen.affine_strength == 0.2

    def test_relax_runs_after_affine(self) -> None:
        """Structures generated with affine should be clash-free (relax ran)."""
        import numpy as np
        gen = StructureGenerator(
            n_atoms=12, charge=0, mult=1, mode="gas", region="sphere:7",
            elements="6,8", n_samples=3, seed=42, affine_strength=0.3,
        )
        for s in gen.stream():
            pts = np.array(s.positions)
            n = len(pts)
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(pts[i] - pts[j])
                    # cov_scale=1.0, C radius ≈ 0.77 Å → min dist ≈ 1.54 Å
                    assert d > 0.5, f"Atoms {i},{j} overlap: d={d:.3f} Å"


# ---------------------------------------------------------------------------
# Structure.comp property  (fix: Bug #1 — 0.3.1)
# ---------------------------------------------------------------------------


class TestStructureCompProperty:
    """Tests for the Structure.comp property added in 0.3.1."""

    def test_comp_returns_string(self) -> None:
        gen = StructureGenerator(
            n_atoms=6, charge=0, mult=1, mode="gas", region="sphere:6",
            elements="6,7,8", n_samples=5, seed=0,
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = gen.generate()
        assert result, "No structures generated"
        s = result[0]
        assert isinstance(s.comp, str)
        assert len(s.comp) > 0

    def test_comp_matches_repr(self) -> None:
        gen = StructureGenerator(
            n_atoms=6, charge=0, mult=1, mode="gas", region="sphere:6",
            elements="6,7,8", n_samples=5, seed=1,
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = gen.generate()
        assert result
        s = result[0]
        assert s.comp in repr(s), f"{s.comp!r} not found in {repr(s)!r}"

    def test_comp_consistent_with_atoms(self) -> None:
        from collections import Counter
        gen = StructureGenerator(
            n_atoms=6, charge=0, mult=1, mode="gas", region="sphere:6",
            elements="6,7,8", n_samples=5, seed=2,
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = gen.generate()
        assert result
        s = result[0]
        counts = Counter(s.atoms)
        expected = "".join(
            f"{sym}{n}" if n > 1 else sym
            for sym, n in sorted(counts.items())
        )
        assert s.comp == expected

    def test_comp_accessible_on_optimizer_result(self) -> None:
        """comp must work on structures returned by StructureOptimizer."""
        import warnings

        from pasted import StructureOptimizer
        opt = StructureOptimizer(
            n_atoms=6, charge=0, mult=1,
            objective={"H_total": 1.0},
            elements="6,7,8",
            max_steps=20,
            seed=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opt.run()
        assert isinstance(result.best.comp, str)
        assert len(result.best.comp) > 0
