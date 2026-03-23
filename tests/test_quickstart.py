"""
tests/test_quickstart.py
========================
Comprehensive test suite for PASTED v0.3.10, derived from docs/quickstart.md.

Covers:
- Extension availability
- Functional API (generate)
- GenerationResult metadata and warnings
- Class API (StructureGenerator)
- maxent mode
- chain / shell modes
- XYZ I/O (write_xyz / from_xyz / read_xyz)
- n_success / streaming
- Metrics access
- comp property
- CLI basics
- StructureOptimizer (all three methods)
- EvalContext / 2-arg objectives
- Affine transforms
- Element sampling (fractions, min/max counts)
- GeneratorConfig
- Position-only / composition-only optimization
- parse_element_spec
- Timing and memory-leak guard (via tracemalloc)
"""

from __future__ import annotations

import dataclasses
import sys
import tracemalloc
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import pasted
from pasted import (
    ALL_METRICS,
    EvalContext,
    GenerationResult,
    GeneratorConfig,
    Structure,
    StructureGenerator,
    StructureOptimizer,
    generate,
    read_xyz,
)
from pasted._atoms import parse_element_spec, parse_filter, validate_charge_mult
from pasted._io import format_xyz, parse_xyz

# ===========================================================================
# 1. Extension availability
# ===========================================================================


def test_extensions_importable() -> None:
    """_ext module must be importable and expose the five HAS_* flags."""
    from pasted._ext import (
        HAS_GRAPH,
        HAS_MAXENT,
        HAS_MAXENT_LOOP,
        HAS_RELAX,
        HAS_STEINHARDT,
    )

    for flag in (HAS_RELAX, HAS_MAXENT, HAS_MAXENT_LOOP, HAS_STEINHARDT, HAS_GRAPH):
        assert isinstance(flag, bool), f"Expected bool, got {type(flag)}"


# ===========================================================================
# 2. Functional API — generate()
# ===========================================================================


def test_generate_basic_gas() -> None:
    """Quickstart example: generate gas-phase structures, iterate over them."""
    result = generate(
        n_atoms=12,
        charge=0,
        mult=1,
        mode="gas",
        region="sphere:9",
        elements="1-30",
        n_samples=50,
        seed=42,
        filters=["H_total:2.0:-"],
    )
    assert isinstance(result, GenerationResult)
    # Must be list-compatible
    for s in result:
        assert isinstance(s, Structure)
        assert s.to_xyz()  # non-empty string
    # Structures that passed the filter must actually pass it
    for s in result:
        assert s.metrics["H_total"] >= 2.0


def test_generate_returns_generation_result() -> None:
    """generate() must return a GenerationResult, not a plain list."""
    result = generate(
        n_atoms=8,
        charge=0,
        mult=1,
        mode="gas",
        region="sphere:7",
        elements="6,7,8",
        n_samples=10,
        seed=0,
    )
    assert isinstance(result, GenerationResult)
    assert isinstance(result.n_attempted, int)
    assert isinstance(result.n_passed, int)
    assert result.n_attempted >= result.n_passed


def test_generate_summary_labels() -> None:
    """summary() string must contain the four documented keys."""
    result = generate(
        n_atoms=6, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=5, seed=1
    )
    s = result.summary()
    for token in ("passed=", "attempted=", "rejected_parity=", "rejected_filter="):
        assert token in s, f"Missing token {token!r} in summary: {s!r}"


def test_generate_n_prefix_attributes() -> None:
    """Accessing result.passed or result.attempted (no n_ prefix) must raise AttributeError."""
    result = generate(
        n_atoms=6, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=5, seed=1
    )
    with pytest.raises(AttributeError):
        _ = result.passed  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = result.attempted  # type: ignore[attr-defined]
    # The n_ versions must work
    assert isinstance(result.n_passed, int)
    assert isinstance(result.n_attempted, int)


def test_generate_warning_no_pass() -> None:
    """A UserWarning must fire when no structures pass the filters."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = generate(
            n_atoms=8,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:8",
            elements="6",  # carbon-only: parity always passes
            n_samples=10,
            seed=0,
            filters=["H_total:999:-"],  # impossible — nothing will pass
        )
    assert not result
    assert any(issubclass(x.category, UserWarning) for x in w), (
        "Expected UserWarning when no structures pass"
    )


def test_generate_bool_empty_result() -> None:
    """Empty GenerationResult must be falsy; non-empty must be truthy."""
    result = generate(
        n_atoms=6, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=5, seed=0
    )
    # n_samples=5 will almost certainly yield at least one result for chain/CNO
    assert bool(result) == (len(result) > 0)


# ===========================================================================
# 3. maxent mode
# ===========================================================================


def test_generate_maxent() -> None:
    """maxent mode requires region= and should produce valid structures."""
    result = generate(
        n_atoms=12,
        charge=0,
        mult=1,
        mode="maxent",
        region="sphere:6",
        elements="6,7,8",
        n_samples=5,
        seed=42,
    )
    for s in result:
        assert len(s.atoms) >= 1
        assert "H_total" in s.metrics


def test_generate_maxent_missing_region_raises() -> None:
    """maxent without region= must raise ValueError."""
    with pytest.raises(ValueError, match="region"):
        generate(n_atoms=6, charge=0, mult=1, mode="maxent", elements="6,7,8", n_samples=3, seed=0)


# ===========================================================================
# 4. Class API — StructureGenerator
# ===========================================================================


def test_struct_gen_chain() -> None:
    """StructureGenerator with chain mode and chain_bias should work."""
    gen = StructureGenerator(
        n_atoms=12,
        charge=0,
        mult=1,
        mode="chain",
        chain_bias=0.5,
        elements="6,7,8",
        n_samples=10,
        seed=0,
    )
    result = gen.generate()
    assert isinstance(result, GenerationResult)
    for s in result:
        assert len(s.atoms) >= 1


def test_struct_gen_getattr_proxy() -> None:
    """Attribute access on StructureGenerator should proxy to _cfg."""
    gen = StructureGenerator(
        n_atoms=10, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=5, seed=7
    )
    assert gen.n_atoms == 10
    assert gen.seed == 7
    assert gen.mode == "chain"


def test_struct_gen_element_pool_property() -> None:
    """element_pool property must return a copy of the pool list."""
    gen = StructureGenerator(
        n_atoms=5, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=1, seed=0
    )
    pool = gen.element_pool
    assert isinstance(pool, list)
    assert set(pool) == {"C", "N", "O"}


# ===========================================================================
# 5. Writing to file
# ===========================================================================


def test_write_xyz_append(tmp_path: Path) -> None:
    """write_xyz(append=True) should produce a multi-frame file."""
    result = generate(
        n_atoms=6, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=5, seed=3
    )
    if not result:
        pytest.skip("No structures generated; skip I/O test.")
    out = tmp_path / "out.xyz"
    for i, s in enumerate(result):
        s.write_xyz(str(out), append=(i > 0))
    text = out.read_text()
    # Number of frames = number of atom-count lines at the start of lines
    frame_count = sum(1 for line in text.splitlines() if line.strip().isdigit())
    assert frame_count == len(result)


def test_write_xyz_overwrite(tmp_path: Path) -> None:
    """write_xyz(append=False) should produce a single-frame file."""
    result = generate(
        n_atoms=6, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=5, seed=4
    )
    if not result:
        pytest.skip("No structures generated.")
    out = tmp_path / "single.xyz"
    result.structures[0].write_xyz(str(out), append=False)
    result.structures[0].write_xyz(str(out), append=False)  # overwrite again
    text = out.read_text()
    frame_count = sum(1 for line in text.splitlines() if line.strip().isdigit())
    assert frame_count == 1


# ===========================================================================
# 6. n_success / streaming
# ===========================================================================


def test_n_success_stops_early() -> None:
    """n_success=5 should return at most 5 structures."""
    gen = StructureGenerator(
        n_atoms=10,
        charge=0,
        mult=1,
        mode="gas",
        region="sphere:8",
        elements="6,7,8",
        n_success=3,
        n_samples=200,
        seed=42,
    )
    result = gen.generate()
    assert len(result) <= 3


def test_n_samples_zero_needs_n_success() -> None:
    """n_samples=0 without n_success must raise ValueError."""
    with pytest.raises(ValueError, match="n_success"):
        StructureGenerator(
            n_atoms=5, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=0
        )


def test_stream_same_as_generate() -> None:
    """list(gen.stream()) and gen.generate() must yield the same atoms."""
    gen1 = StructureGenerator(
        n_atoms=8, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=10, seed=9
    )
    gen2 = StructureGenerator(
        n_atoms=8, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=10, seed=9
    )
    streamed = list(gen1.stream())
    generated = gen2.generate().structures
    assert len(streamed) == len(generated)
    for s1, s2 in zip(streamed, generated, strict=True):
        assert s1.atoms == s2.atoms


# ===========================================================================
# 7. Structure properties and metrics
# ===========================================================================


def test_structure_metrics_keys() -> None:
    """Each generated Structure must have all 17 expected metric keys."""
    result = generate(
        n_atoms=8, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=3, seed=11
    )
    if not result:
        pytest.skip("No structures generated.")
    s = result.structures[0]
    for key in ALL_METRICS:
        assert key in s.metrics, f"Missing metric key {key!r}"


def test_structure_comp_property() -> None:
    """comp must be an alphabetically-sorted composition string."""
    result = generate(
        n_atoms=10, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=5, seed=12
    )
    if not result:
        pytest.skip("No structures generated.")
    for s in result:
        comp = s.comp
        assert isinstance(comp, str)
        # Reconstruct from comp and compare with Counter
        c = Counter(s.atoms)
        expected = "".join(f"{sym}{n}" if n > 1 else sym for sym, n in sorted(c.items()))
        assert comp == expected, f"{comp!r} != {expected!r}"


def test_structure_comp_alphabetical_ar() -> None:
    """Quickstart example: ['Ar','C','H','H'] -> 'ArCH2' (alphabetical)."""
    s = Structure(
        atoms=["Ar", "C", "H", "H"],
        positions=[(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.5, 0.5, 0.0), (2.5, -0.5, 0.0)],
        charge=0,
        mult=1,
        metrics={k: 0.0 for k in ALL_METRICS},
        mode="test",
    )
    assert s.comp == "ArCH2"


def test_structure_repr_contains_comp() -> None:
    """repr(s) must contain the comp string."""
    result = generate(
        n_atoms=6, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=5, seed=2
    )
    if not result:
        pytest.skip()
    s = result.structures[0]
    assert s.comp in repr(s)
    assert "Structure(" in repr(s)


def test_structure_len() -> None:
    """len(s) and s.n must equal the number of atoms."""
    result = generate(
        n_atoms=8, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=3, seed=5
    )
    if not result:
        pytest.skip()
    s = result.structures[0]
    assert len(s) == len(s.atoms)
    assert s.n == len(s.atoms)


# ===========================================================================
# 8. GenerationResult.__add__
# ===========================================================================


def test_generation_result_add() -> None:
    """r1 + r2 must combine structures and accumulate counters."""
    r1 = generate(
        n_atoms=6, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=5, seed=100
    )
    r2 = generate(
        n_atoms=6, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=5, seed=101
    )
    combined = r1 + r2
    assert isinstance(combined, GenerationResult)
    assert combined.n_attempted == r1.n_attempted + r2.n_attempted
    assert len(combined) == len(r1) + len(r2)


# ===========================================================================
# 9. GeneratorConfig
# ===========================================================================


def test_generator_config_immutable() -> None:
    """GeneratorConfig must be frozen (immutable)."""
    cfg = GeneratorConfig(
        n_atoms=10, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=5, seed=0
    )
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        cfg.seed = 99  # type: ignore[misc]


def test_generator_config_replace() -> None:
    """dataclasses.replace on a config must not mutate the original."""
    cfg = GeneratorConfig(
        n_atoms=10, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=5, seed=0
    )
    cfg2 = dataclasses.replace(cfg, seed=99)
    assert cfg.seed == 0
    assert cfg2.seed == 99


def test_generator_config_with_struct_gen() -> None:
    """StructureGenerator must accept a GeneratorConfig as first arg."""
    cfg = GeneratorConfig(
        n_atoms=8, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=5, seed=77
    )
    result = StructureGenerator(cfg).generate()
    assert isinstance(result, GenerationResult)


def test_generate_accepts_config() -> None:
    """generate() must accept a GeneratorConfig as its first positional arg."""
    cfg = GeneratorConfig(
        n_atoms=8, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=5, seed=88
    )
    result = generate(cfg)
    assert isinstance(result, GenerationResult)


# ===========================================================================
# 10. parse_element_spec
# ===========================================================================


def test_parse_element_spec_list_of_symbols() -> None:
    """parse_element_spec([\"C\",\"N\",\"O\"]) must return [\"C\",\"N\",\"O\"]."""
    assert parse_element_spec(["C", "N", "O"]) == ["C", "N", "O"]


def test_parse_element_spec_numeric_string() -> None:
    """parse_element_spec(\"6,7,8\") must return [\"C\",\"N\",\"O\"]."""
    assert parse_element_spec("6,7,8") == ["C", "N", "O"]


def test_parse_element_spec_range() -> None:
    """parse_element_spec(\"1-3\") must return [\"H\",\"He\",\"Li\"]."""
    pool = parse_element_spec("1-3")
    assert pool == ["H", "He", "Li"]


def test_parse_element_spec_symbol_string_raises() -> None:
    """parse_element_spec(\"C,N,O\") (symbol string) must raise ValueError."""
    with pytest.raises(ValueError):
        parse_element_spec("C,N,O")


# ===========================================================================
# 11. validate_charge_mult
# ===========================================================================


def test_validate_charge_mult_valid() -> None:
    """Simple valid case should return (True, '')."""
    ok, msg = validate_charge_mult(["C", "C"], 0, 1)
    assert ok
    assert msg == ""


def test_validate_charge_mult_invalid() -> None:
    """An impossible parity case should return (False, <msg>)."""
    # Single N (Z=7, odd): total electrons = 7 - 0 = 7 (odd) → mult=1 (even) impossible
    ok, msg = validate_charge_mult(["N"], 0, 1)
    assert not ok
    assert isinstance(msg, str)


# ===========================================================================
# 12. parse_filter
# ===========================================================================


def test_parse_filter_open_bounds() -> None:
    """parse_filter with '-' for open bound should return ±inf."""
    metric, lo, hi = parse_filter("H_total:2.0:-")
    assert metric == "H_total"
    assert lo == pytest.approx(2.0)
    assert hi == float("inf")


def test_parse_filter_both_open() -> None:
    """Both bounds open: -inf to +inf."""
    metric, lo, hi = parse_filter("Q6:-:-")
    assert metric == "Q6"
    assert lo == float("-inf")
    assert hi == float("inf")


# ===========================================================================
# 13. XYZ I/O — format_xyz / parse_xyz
# ===========================================================================


def test_format_parse_roundtrip() -> None:
    """format_xyz → parse_xyz roundtrip must preserve atoms and positions."""
    atoms = ["C", "N", "O"]
    positions = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0)]
    metrics = {k: 0.123 for k in ALL_METRICS}
    xyz = format_xyz(atoms, positions, 0, 1, metrics)
    frames = parse_xyz(xyz)
    assert len(frames) == 1
    p_atoms, p_pos, p_charge, p_mult, _p_metrics = frames[0]
    assert p_atoms == atoms
    assert p_charge == 0
    assert p_mult == 1
    for i, (x, _y, _z) in enumerate(p_pos):
        assert abs(x - positions[i][0]) < 1e-5


def test_parse_xyz_multi_frame() -> None:
    """parse_xyz must handle two concatenated XYZ frames."""
    atoms = ["C", "N"]
    positions = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]
    metrics = {k: 0.0 for k in ALL_METRICS}
    block = (
        format_xyz(atoms, positions, 0, 1, metrics)
        + "\n"
        + format_xyz(atoms, positions, 0, 1, metrics)
    )
    frames = parse_xyz(block)
    assert len(frames) == 2


def test_structure_from_xyz_roundtrip(tmp_path: Path) -> None:
    """Structure.from_xyz(write_xyz(...)) must produce a valid Structure."""
    result = generate(
        n_atoms=6, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=5, seed=20
    )
    if not result:
        pytest.skip()
    s_orig = result.structures[0]
    p = tmp_path / "test.xyz"
    s_orig.write_xyz(str(p), append=False)
    s_loaded = Structure.from_xyz(str(p))
    assert s_loaded.atoms == s_orig.atoms
    assert len(s_loaded.metrics) > 0


def test_structure_from_xyz_file_not_found() -> None:
    """from_xyz on a non-existent file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        Structure.from_xyz("/tmp/does_not_exist_xyz_test_pasted.xyz")


def test_read_xyz_multi(tmp_path: Path) -> None:
    """read_xyz must return all frames as a list of Structure objects."""
    result = generate(
        n_atoms=6, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=10, seed=22
    )
    if len(result) < 2:
        pytest.skip("Need at least 2 structures for multi-frame test.")
    p = tmp_path / "multi.xyz"
    for i, s in enumerate(result):
        s.write_xyz(str(p), append=(i > 0))
    loaded = read_xyz(str(p))
    assert len(loaded) == len(result)
    for s in loaded:
        assert isinstance(s, Structure)


# ===========================================================================
# 14. Shell mode
# ===========================================================================


def test_shell_mode_fixed_center() -> None:
    """Shell mode with center_z should produce a center_sym attribute."""
    result = generate(
        n_atoms=6,
        charge=0,
        mult=1,
        mode="shell",
        center_z=26,
        elements="1-30",
        n_samples=5,
        seed=7,
    )
    for s in result:
        assert s.center_sym == "Fe"


def test_shell_mode_n_atoms_expands() -> None:
    """Shell mode output may have more atoms than n_atoms (center + H)."""
    result = generate(
        n_atoms=4,
        charge=0,
        mult=1,
        mode="shell",
        elements="1-30",
        n_samples=5,
        seed=7,
    )
    for s in result:
        # At minimum the shell atoms + optional center
        assert len(s.atoms) >= 1


# ===========================================================================
# 15. Element sampling: fractions and min/max counts
# ===========================================================================


def test_element_fractions() -> None:
    """Biased element fractions should shift the distribution toward C."""
    gen = StructureGenerator(
        n_atoms=20,
        charge=0,
        mult=1,
        mode="gas",
        region="sphere:10",
        elements="6,7,8",
        element_fractions={"C": 0.6, "N": 0.3, "O": 0.1},
        n_samples=20,
        seed=0,
    )
    result = gen.generate()
    if not result:
        pytest.skip()
    # With 0.6 weight, C should be the most common element overall
    all_atoms = [a for s in result for a in s.atoms]
    c_count = all_atoms.count("C")
    o_count = all_atoms.count("O")
    assert c_count >= o_count, "Expected C to be more common than O with 0.6 weight"


def test_element_min_counts() -> None:
    """element_min_counts must guarantee minimum atom presence."""
    gen = StructureGenerator(
        n_atoms=10,
        charge=0,
        mult=1,
        mode="gas",
        region="sphere:8",
        elements="6,7,8",
        element_min_counts={"C": 3},
        n_samples=10,
        seed=42,
    )
    result = gen.generate()
    for s in result:
        assert Counter(s.atoms)["C"] >= 3


def test_element_max_counts() -> None:
    """element_max_counts must enforce upper bounds per element."""
    gen = StructureGenerator(
        n_atoms=10,
        charge=0,
        mult=1,
        mode="gas",
        region="sphere:8",
        elements="6,7,8",
        element_max_counts={"N": 2},
        n_samples=10,
        seed=42,
    )
    result = gen.generate()
    for s in result:
        assert Counter(s.atoms).get("N", 0) <= 2


def test_element_min_max_inconsistency_raises() -> None:
    """min > max for the same element must raise ValueError."""
    with pytest.raises(ValueError):
        StructureGenerator(
            n_atoms=10,
            charge=0,
            mult=1,
            mode="chain",
            elements="6,7,8",
            element_min_counts={"C": 5},
            element_max_counts={"C": 2},
            n_samples=5,
        )


def test_element_min_sum_exceeds_n_atoms_raises() -> None:
    """Sum of min counts > n_atoms must raise ValueError."""
    with pytest.raises(ValueError):
        StructureGenerator(
            n_atoms=5,
            charge=0,
            mult=1,
            mode="chain",
            elements="6,7,8",
            element_min_counts={"C": 4, "N": 4},  # 8 > 5
            n_samples=5,
        )


# ===========================================================================
# 16. Affine transforms
# ===========================================================================


def test_affine_strength_runs() -> None:
    """affine_strength > 0 should produce valid structures without errors."""
    result = generate(
        n_atoms=15,
        charge=0,
        mult=1,
        mode="gas",
        region="sphere:10",
        elements="6,7,8",
        n_samples=10,
        seed=42,
        affine_strength=0.2,
    )
    for s in result:
        assert len(s.atoms) >= 1


def test_affine_per_operation_control() -> None:
    """Per-operation affine parameters should run without errors."""
    result = generate(
        n_atoms=12,
        charge=0,
        mult=1,
        mode="chain",
        elements="6,7,8",
        n_samples=5,
        seed=0,
        affine_strength=0.2,
        affine_stretch=0.4,
        affine_shear=0.0,
        affine_jitter=0.0,
    )
    assert isinstance(result, GenerationResult)


# ===========================================================================
# 17. StructureOptimizer — basic annealing
# ===========================================================================


def test_optimizer_annealing_basic() -> None:
    """Basic annealing optimizer should return an OptimizationResult with a best."""
    opt = StructureOptimizer(
        n_atoms=8,
        charge=0,
        mult=1,
        elements="6,7,8,15,16",
        objective={"H_total": 1.0, "Q6": -2.0},
        method="annealing",
        max_steps=200,
        n_restarts=2,
        seed=42,
    )
    result = opt.run()
    assert result.best is not None
    assert isinstance(result.best, Structure)
    summary = result.summary()
    assert "restarts=" in summary
    assert "method=" in summary


def test_optimizer_summary_format() -> None:
    """OptimizationResult.summary() must contain the documented keys."""
    opt = StructureOptimizer(
        n_atoms=6,
        charge=0,
        mult=1,
        elements="6,7,8",
        objective={"H_total": 1.0},
        method="annealing",
        max_steps=100,
        n_restarts=1,
        seed=0,
    )
    result = opt.run()
    s = result.summary()
    for key in ("restarts=", "best_f=", "method="):
        assert key in s, f"Missing key {key!r} in summary: {s!r}"


def test_optimizer_iteration() -> None:
    """Iterating over OptimizationResult must yield Structure objects."""
    opt = StructureOptimizer(
        n_atoms=6,
        charge=0,
        mult=1,
        elements="6,7,8",
        objective={"H_total": 1.0},
        method="annealing",
        max_steps=100,
        n_restarts=2,
        seed=1,
    )
    result = opt.run()
    for _rank, s in enumerate(result, 1):
        assert isinstance(s, Structure)


# ===========================================================================
# 18. StructureOptimizer — basin_hopping and parallel_tempering
# ===========================================================================


def test_optimizer_basin_hopping() -> None:
    """basin_hopping must complete and produce a valid best structure."""
    opt = StructureOptimizer(
        n_atoms=6,
        charge=0,
        mult=1,
        elements="6,7,8",
        objective={"H_total": 1.0},
        method="basin_hopping",
        max_steps=100,
        n_restarts=1,
        seed=3,
    )
    result = opt.run()
    assert result.best is not None


def test_optimizer_parallel_tempering() -> None:
    """parallel_tempering must complete and produce a valid best structure."""
    opt = StructureOptimizer(
        n_atoms=6,
        charge=0,
        mult=1,
        elements="6,7,8",
        objective={"H_total": 1.0},
        method="parallel_tempering",
        n_replicas=2,
        pt_swap_interval=5,
        max_steps=100,
        n_restarts=1,
        T_start=1.0,
        T_end=0.01,
        seed=5,
    )
    result = opt.run()
    assert result.best is not None


# ===========================================================================
# 19. StructureOptimizer — callable objectives
# ===========================================================================


def test_optimizer_callable_1arg() -> None:
    """1-arg callable objective (lambda) should work correctly."""
    opt = StructureOptimizer(
        n_atoms=6,
        charge=0,
        mult=1,
        elements="6,7,8",
        objective=lambda m: m["H_total"] - 2.0 * m["Q6"],
        method="annealing",
        max_steps=100,
        seed=9,
    )
    result = opt.run()
    assert result.best is not None


def test_optimizer_callable_2arg_ctx() -> None:
    """2-arg callable objective receives an EvalContext as second arg."""
    ctx_seen: list[EvalContext] = []

    def obj(m: dict, ctx: EvalContext) -> float:
        ctx_seen.append(ctx)
        return float(m["H_total"])

    opt = StructureOptimizer(
        n_atoms=6,
        charge=0,
        mult=1,
        elements="6,7,8",
        objective=obj,
        method="annealing",
        max_steps=50,
        seed=10,
    )
    opt.run()
    assert len(ctx_seen) > 0
    c = ctx_seen[0]
    assert hasattr(c, "atoms")
    assert hasattr(c, "positions")
    assert hasattr(c, "metrics")
    assert hasattr(c, "step")
    assert hasattr(c, "progress")
    assert 0.0 <= c.progress < 1.0


def test_eval_context_progress_field() -> None:
    """EvalContext.progress must be in [0, 1)."""
    progresses: list[float] = []

    def obj(m: dict, ctx: EvalContext) -> float:
        progresses.append(ctx.progress)
        return float(m["H_total"])

    opt = StructureOptimizer(
        n_atoms=6,
        charge=0,
        mult=1,
        elements="6,7,8",
        objective=obj,
        method="annealing",
        max_steps=50,
        seed=11,
    )
    opt.run()
    assert all(0.0 <= p < 1.0 for p in progresses), (
        f"Some progress values out of [0, 1): {progresses[:5]}"
    )


# ===========================================================================
# 20. StructureOptimizer — allow_composition_moves=False (position-only)
# ===========================================================================


def test_optimizer_position_only() -> None:
    """allow_composition_moves=False must preserve the composition."""
    initial_structs = generate(
        n_atoms=8,
        charge=0,
        mult=1,
        mode="gas",
        region="sphere:7",
        elements="6,7,8",
        n_samples=20,
        seed=0,
    )
    if not initial_structs:
        pytest.skip()
    initial = initial_structs.structures[0]
    opt = StructureOptimizer(
        n_atoms=len(initial),
        charge=initial.charge,
        mult=initial.mult,
        elements=list(set(initial.atoms)),
        objective={"H_total": 1.0, "Q6": -2.0},
        allow_composition_moves=False,
        method="annealing",
        max_steps=100,
        seed=42,
    )
    result = opt.run(initial=initial)
    assert sorted(result.best.atoms) == sorted(initial.atoms)


# ===========================================================================
# 21. StructureOptimizer — allow_displacements=False (composition-only)
# ===========================================================================


def test_optimizer_composition_only() -> None:
    """allow_displacements=False must preserve the positions."""
    initial_structs = generate(
        n_atoms=8,
        charge=0,
        mult=1,
        mode="gas",
        region="sphere:7",
        elements="6,7,8",
        n_samples=20,
        seed=0,
    )
    if not initial_structs:
        pytest.skip()
    initial = initial_structs.structures[0]
    opt = StructureOptimizer(
        n_atoms=len(initial),
        charge=initial.charge,
        mult=initial.mult,
        elements=["C", "N"],
        objective={"moran_I_chi": -1.0},
        allow_displacements=False,
        method="annealing",
        max_steps=100,
        seed=42,
    )
    result = opt.run(initial=initial)
    np.testing.assert_allclose(
        np.array(result.best.positions),
        np.array(initial.positions),
        err_msg="Positions changed in composition-only mode",
    )


def test_optimizer_both_disabled_raises() -> None:
    """allow_displacements=False AND allow_composition_moves=False must raise ValueError."""
    with pytest.raises(ValueError):
        StructureOptimizer(
            n_atoms=6,
            charge=0,
            mult=1,
            elements="6,7,8",
            objective={"H_total": 1.0},
            allow_displacements=False,
            allow_composition_moves=False,
        )


# ===========================================================================
# 22. StructureOptimizer — max_init_attempts
# ===========================================================================


def test_optimizer_max_init_attempts_positive() -> None:
    """max_init_attempts > 0 should not break a normal run."""
    opt = StructureOptimizer(
        n_atoms=8,
        charge=0,
        mult=1,
        elements="6,7,8",
        objective={"H_total": 1.0},
        method="annealing",
        max_steps=100,
        n_restarts=2,
        max_init_attempts=50,
        seed=99,
    )
    result = opt.run()
    assert result.best is not None


def test_optimizer_impossible_parity_raises() -> None:
    """All-nitrogen pool with n_atoms=7, charge=0, mult=1 must raise ValueError."""
    with pytest.raises(ValueError):
        StructureOptimizer(
            n_atoms=7,
            charge=0,
            mult=1,
            elements="7",  # nitrogen only
            objective={"H_total": 1.0},
        )


# ===========================================================================
# 23. Affine moves in StructureOptimizer
# ===========================================================================


def test_optimizer_affine_moves() -> None:
    """allow_affine_moves=True should produce a valid best structure."""
    opt = StructureOptimizer(
        n_atoms=10,
        charge=0,
        mult=1,
        elements="6,7,8",
        objective={"shape_aniso": 2.0, "H_total": 1.0},
        allow_affine_moves=True,
        affine_strength=0.2,
        affine_stretch=0.4,
        affine_shear=0.0,
        affine_jitter=0.0,
        method="annealing",
        max_steps=100,
        n_restarts=1,
        seed=0,
    )
    result = opt.run()
    assert result.best is not None
    assert "shape_aniso" in result.best.metrics


# ===========================================================================
# 24. Electronegativity metrics
# ===========================================================================


def test_en_metrics_in_result() -> None:
    """charge_frustration and moran_I_chi must be present in metrics."""
    opt = StructureOptimizer(
        n_atoms=8,
        charge=0,
        mult=1,
        elements="6,7,8,9,14,15,16",
        objective={
            "charge_frustration": 2.0,
            "moran_I_chi": -1.0,
        },
        method="annealing",
        max_steps=100,
        n_restarts=1,
        seed=7,
    )
    result = opt.run()
    assert "charge_frustration" in result.best.metrics
    assert "moran_I_chi" in result.best.metrics


# ===========================================================================
# 25. Timing and memory-leak guard
# ===========================================================================


@pytest.mark.parametrize("n_atoms", [5, 50, 200])
def test_compute_all_metrics_timing(n_atoms: int) -> None:
    """compute_all_metrics must complete quickly and not leak memory."""
    import time

    from pasted._metrics import compute_all_metrics

    rng = np.random.default_rng(0)
    atoms = ["C", "N", "O"] * (n_atoms // 3 + 1)
    atoms = atoms[:n_atoms]
    positions = [tuple(rng.uniform(-5, 5, 3).tolist()) for _ in range(n_atoms)]

    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(5):
        compute_all_metrics(atoms, positions, 20, 0.5, 0.5, 5.0, 1.0)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Generous budget: 5 seconds per batch of 5 calls
    assert elapsed < 5.0, f"compute_all_metrics too slow at N={n_atoms}: {elapsed:.2f}s"
    # Peak memory growth should be < 100 MB
    assert peak < 100 * 1024 * 1024, f"Peak memory too high at N={n_atoms}: {peak / 1e6:.1f} MB"


def test_no_memory_leak_repeated_generate() -> None:
    """Repeated generate() calls must not exhibit unbounded memory growth."""
    import tracemalloc

    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()
    for i in range(10):
        generate(n_atoms=8, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=5, seed=i)
    snap1 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snap1.compare_to(snap0, "lineno")
    total_diff = sum(s.size_diff for s in stats)
    # Allow up to 50 MB total growth for 10 calls
    assert total_diff < 50 * 1024 * 1024, (
        f"Suspicious memory growth: {total_diff / 1e6:.1f} MB over 10 calls"
    )


# ===========================================================================
# 26. CLI smoke test
# ===========================================================================


def test_cli_help(capsys: pytest.CaptureFixture) -> None:  # type: ignore[type-arg]
    """pasted --help must exit cleanly (exit code 0)."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "pasted", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"--help exited with {proc.returncode}"
    assert "pasted" in proc.stdout.lower() or "pasted" in proc.stderr.lower()


def test_cli_basic_gas(tmp_path: Path) -> None:
    """CLI invocation must produce a valid XYZ file."""
    import subprocess

    out_file = tmp_path / "cli_out.xyz"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pasted",
            "--n-atoms",
            "6",
            "--charge",
            "0",
            "--mult",
            "1",
            "--mode",
            "gas",
            "--region",
            "sphere:6",
            "--elements",
            "6,7,8",
            "--n-samples",
            "3",
            "--seed",
            "0",
            "-o",
            str(out_file),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"CLI failed: {proc.stderr}"
    assert out_file.exists(), "Output file not created"
    content = out_file.read_text()
    assert len(content) > 0, "Output file is empty"


# ===========================================================================
# 27. version attribute
# ===========================================================================


def test_version_attribute() -> None:
    """pasted.__version__ must be a non-empty string."""
    assert isinstance(pasted.__version__, str)
    assert pasted.__version__


# ===========================================================================
# 28. Structure.from_xyz — edge cases
# ===========================================================================


def test_from_xyz_directory_raises(tmp_path: Path) -> None:
    """from_xyz on a directory must raise IsADirectoryError."""
    with pytest.raises(IsADirectoryError):
        Structure.from_xyz(str(tmp_path))


def test_from_xyz_frame_out_of_range(tmp_path: Path) -> None:
    """Requesting frame=5 from a 1-frame file must raise ValueError."""
    result = generate(
        n_atoms=6, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=5, seed=50
    )
    if not result:
        pytest.skip()
    p = tmp_path / "single.xyz"
    result.structures[0].write_xyz(str(p), append=False)
    with pytest.raises(ValueError, match="frame"):
        Structure.from_xyz(str(p), frame=5)


# ===========================================================================
# 29. n_success warning when budget exhausted
# ===========================================================================


def test_n_success_warning_on_budget_exhaust() -> None:
    """A UserWarning must fire when n_success is not reached within n_samples."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = generate(
            n_atoms=8,
            charge=0,
            mult=1,
            mode="gas",
            region="sphere:7",
            elements="6,7,8",
            n_success=999,  # essentially impossible to reach in 3 tries
            n_samples=3,
            seed=1,
        )
    budget_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    # Either fewer structures were generated OR a warning was emitted
    assert len(result) <= 999
    if len(result) < 999:
        assert len(budget_warnings) > 0, "Expected UserWarning when n_success not reached"


# ===========================================================================
# 30. Reproducibility — same seed => same output
# ===========================================================================


def test_reproducibility_same_seed() -> None:
    """Two generate() calls with the same seed must produce identical atoms."""
    kwargs = dict(
        n_atoms=8, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=10, seed=314
    )
    r1 = generate(**kwargs)  # type: ignore[arg-type]
    r2 = generate(**kwargs)  # type: ignore[arg-type]
    assert len(r1) == len(r2)
    for s1, s2 in zip(r1, r2, strict=True):
        assert s1.atoms == s2.atoms


# ===========================================================================
# 31. compute_all_metrics shape_anisotropy guard (v0.3.10 fix)
# ===========================================================================


def test_shape_anisotropy_colinear_atoms() -> None:
    """compute_all_metrics on a perfectly colinear structure must not raise ZeroDivisionError."""
    from pasted._metrics import compute_all_metrics

    # All atoms on the X axis → moment-of-inertia tensor is singular
    atoms = ["C", "C", "C", "C", "C"]
    positions = [(float(i) * 1.5, 0.0, 0.0) for i in range(5)]
    # Should not raise
    m = compute_all_metrics(atoms, positions, 20, 0.5, 0.5, 5.0, 1.0)
    assert "shape_aniso" in m
    # shape_aniso must be a float, potentially NaN but not an error
    assert isinstance(m["shape_aniso"], float)


# ===========================================================================
# 32. moran_I_chi clamped to 1.0 (v0.3.8 fix)
# ===========================================================================


def test_moran_i_chi_clamped() -> None:
    """moran_I_chi must be <= 1.0 for all generated structures."""
    result = generate(
        n_atoms=10, charge=0, mult=1, mode="chain", elements="6,7,8", n_samples=10, seed=42
    )
    for s in result:
        val = s.metrics.get("moran_I_chi", 0.0)
        assert val <= 1.0 + 1e-9, f"moran_I_chi = {val} exceeds 1.0"
