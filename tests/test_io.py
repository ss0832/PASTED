"""Tests for pasted._io: parse_xyz, read_xyz, Structure.from_xyz."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest

from pasted import Structure, StructureOptimizer, generate, parse_xyz, read_xyz
from pasted._atoms import ALL_METRICS

# ---------------------------------------------------------------------------
# parse_xyz
# ---------------------------------------------------------------------------

PLAIN_XYZ = """\
3
comment line
C   0.000000   0.000000   0.000000
O   1.200000   0.000000   0.000000
C   2.400000   0.000000   0.000000
"""

EXTENDED_XYZ = """\
3
sample=1 mode=gas charge=0 mult=1 comp=[C:2,O:1]  H_total=0.8000
C   0.000000   0.000000   0.000000
O   1.200000   0.000000   0.000000
C   2.400000   0.000000   0.000000
"""

MULTI_FRAME = PLAIN_XYZ + PLAIN_XYZ.replace("comment line", "frame 2")


class TestParseXyz:
    def test_plain_xyz_parsed(self) -> None:
        frames = parse_xyz(PLAIN_XYZ)
        assert len(frames) == 1
        atoms, positions, charge, mult, metrics = frames[0]
        assert atoms == ["C", "O", "C"]
        assert len(positions) == 3
        assert charge == 0
        assert mult == 1
        assert metrics == {}

    def test_extended_xyz_charge_mult(self) -> None:
        frames = parse_xyz(EXTENDED_XYZ)
        _, _, charge, mult, _metrics = frames[0]
        assert charge == 0
        assert mult == 1

    def test_extended_xyz_metrics(self) -> None:
        frames = parse_xyz(EXTENDED_XYZ)
        _, _, _, _, metrics = frames[0]
        assert "H_total" in metrics
        assert metrics["H_total"] == pytest.approx(0.8)

    def test_multi_frame(self) -> None:
        frames = parse_xyz(MULTI_FRAME)
        assert len(frames) == 2

    def test_positions_correct(self) -> None:
        frames = parse_xyz(PLAIN_XYZ)
        _, positions, _, _, _ = frames[0]
        assert positions[0] == pytest.approx((0.0, 0.0, 0.0))
        assert positions[1] == pytest.approx((1.2, 0.0, 0.0))

    def test_bad_atom_count_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_xyz("not_a_number\ncomment\nC 0 0 0\n")

    def test_bad_coordinate_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_xyz("1\ncomment\nC abc 0 0\n")

    def test_empty_string_returns_empty(self) -> None:
        assert parse_xyz("") == []


# ---------------------------------------------------------------------------
# Structure.from_xyz
# ---------------------------------------------------------------------------


class TestStructureFromXyz:
    def test_from_xyz_string_plain(self) -> None:
        s = Structure.from_xyz(PLAIN_XYZ, recompute_metrics=False)
        assert s.atoms == ["C", "O", "C"]
        assert len(s.positions) == 3
        assert s.mode == "loaded_xyz"

    def test_from_xyz_string_extended_no_recompute(self) -> None:
        s = Structure.from_xyz(EXTENDED_XYZ, recompute_metrics=False)
        assert "H_total" in s.metrics
        assert s.metrics["H_total"] == pytest.approx(0.8)

    def test_from_xyz_recomputes_metrics(self) -> None:
        s = Structure.from_xyz(PLAIN_XYZ, recompute_metrics=True)
        assert set(s.metrics.keys()) == ALL_METRICS
        for v in s.metrics.values():
            assert math.isfinite(v)

    def test_from_xyz_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.xyz"
        f.write_text(PLAIN_XYZ)
        s = Structure.from_xyz(f, recompute_metrics=False)
        assert len(s.atoms) == 3

    def test_from_xyz_frame_index(self) -> None:
        s0 = Structure.from_xyz(MULTI_FRAME, frame=0, recompute_metrics=False)
        s1 = Structure.from_xyz(MULTI_FRAME, frame=1, recompute_metrics=False)
        assert s0.atoms == s1.atoms  # same atoms in both frames

    def test_from_xyz_bad_frame_raises(self) -> None:
        with pytest.raises(ValueError, match="frame=5"):
            Structure.from_xyz(PLAIN_XYZ, frame=5)

    def test_from_xyz_roundtrip(self) -> None:
        """Write via to_xyz, read back — atoms, charge, mult exact; stable metrics close."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structs = generate(
                n_atoms=6,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:6",
                elements="6,8",
                n_samples=1,
                seed=0,
            )
        original = structs[0]  # type: ignore[union-attr]
        xyz_str = original.to_xyz()  # type: ignore[union-attr]
        loaded = Structure.from_xyz(xyz_str, recompute_metrics=True)  # type: ignore[union-attr]
        assert loaded.atoms == original.atoms  # type: ignore[union-attr]
        assert loaded.charge == original.charge  # type: ignore[union-attr]
        assert loaded.mult == original.mult  # type: ignore[union-attr]
        # Full metric set is present
        assert set(loaded.metrics.keys()) == ALL_METRICS
        # Composition-based metrics are cutoff-independent → exact match
        stable = ("H_atom",)
        for key in stable:  # type: ignore[union-attr]
            assert loaded.metrics[key] == pytest.approx(original.metrics[key], abs=1e-4), key  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# read_xyz
# ---------------------------------------------------------------------------


class TestReadXyz:
    def test_read_single_frame_string(self) -> None:
        structs = read_xyz(PLAIN_XYZ, recompute_metrics=False)
        assert len(structs) == 1
        assert structs[0].atoms == ["C", "O", "C"]

    def test_read_multi_frame_string(self) -> None:
        structs = read_xyz(MULTI_FRAME, recompute_metrics=False)
        assert len(structs) == 2

    def test_read_file(self, tmp_path: Path) -> None:
        f = tmp_path / "out.xyz"
        f.write_text(EXTENDED_XYZ)
        structs = read_xyz(f, recompute_metrics=False)
        assert len(structs) == 1

    def test_read_recomputes_metrics(self) -> None:
        structs = read_xyz(PLAIN_XYZ, recompute_metrics=True)
        assert set(structs[0].metrics.keys()) == ALL_METRICS

    def test_read_pasted_output_roundtrip(self, tmp_path: Path) -> None:
        """Write PASTED output, read back, get same number of structures."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = generate(
                n_atoms=6,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:6",
                elements="6,8",
                n_samples=3,
                seed=42,
            )
        f = tmp_path / "batch.xyz"
        for i, s in enumerate(result):
            s.write_xyz(f, append=(i > 0))
        loaded = read_xyz(f, recompute_metrics=False)
        assert len(loaded) == len(result)

    def test_read_useful_as_optimizer_initial(self) -> None:
        """Loaded structure can be passed to StructureOptimizer.run()."""
        s = Structure.from_xyz(PLAIN_XYZ, recompute_metrics=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt = StructureOptimizer(
                n_atoms=len(s),
                charge=s.charge,
                mult=s.mult,
                objective={"H_total": 1.0},
                elements=list(set(s.atoms)),
                max_steps=20,
                seed=0,
            )
            result = opt.run(initial=s)
        assert result.best is not None

    def test_read_xyz_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """read_xyz raises FileNotFoundError for a non-existent path (Bug fix v0.4.0).

        Previously this raised a confusing ValueError because the path string was
        silently parsed as XYZ text.  Now it raises FileNotFoundError, matching
        the behavior of Structure.from_xyz().
        """
        missing = tmp_path / "does_not_exist.xyz"
        with pytest.raises(FileNotFoundError, match="XYZ file not found"):
            read_xyz(missing)

    def test_read_xyz_directory_raises_is_a_directory(self, tmp_path: Path) -> None:
        """read_xyz raises IsADirectoryError when given a directory path (Bug fix v0.4.0)."""
        with pytest.raises(IsADirectoryError):
            read_xyz(tmp_path)
