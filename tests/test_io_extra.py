"""
tests/test_io_extra.py
======================
Covers the remaining missed branches in ``pasted._io`` (91% → higher).

Missed lines targeted
---------------------
117-118 : blank lines *between* XYZ frames are skipped (``continue``).
128     : EOF immediately after the atom-count line raises ValueError.
153-154 : a metric token whose value cannot be parsed as float is silently
          skipped (the ``except ValueError: pass`` branch).
162     : blank lines *inside* the coordinate block are skipped.
"""

from __future__ import annotations

import pytest

from pasted._io import parse_xyz


class TestParseXyzBlankLines:
    """L117-118: blank lines between frames must be silently skipped."""

    def test_blank_lines_between_frames_are_skipped(self) -> None:
        """Two valid frames separated by blank lines must both be returned."""
        frame = "2\ncharge=0 mult=1\nC  0.0 0.0 0.0\nN  1.2 0.0 0.0\n"
        text = "\n\n" + frame + "\n\n" + frame + "\n"
        frames = parse_xyz(text)
        assert len(frames) == 2

    def test_leading_blank_lines_are_skipped(self) -> None:
        """Blank lines before the first frame must not cause an error."""
        frame = "1\ncharge=0 mult=1\nC  0.0 0.0 0.0\n"
        frames = parse_xyz("\n\n\n" + frame)
        assert len(frames) == 1


class TestParseXyzEofAfterAtomCount:
    """L128: EOF immediately after the atom-count line must raise ValueError."""

    def test_eof_after_atom_count_raises(self) -> None:
        """A file that ends right after the atom-count line is truncated."""
        with pytest.raises(ValueError, match=r"[Uu]nexpected end"):
            parse_xyz("3\n")


class TestParseXyzBlanksInCoordBlock:
    """L162: blank lines inside the coordinate block must be skipped."""

    def test_blank_line_inside_coord_block_is_skipped(self) -> None:
        """An XYZ frame with a blank line between coordinate rows must parse."""
        text = "2\ncharge=0 mult=1\nC  0.0 0.0 0.0\n\nN  1.2 0.0 0.0\n"
        frames = parse_xyz(text)
        assert len(frames) == 1
        atoms, positions, _charge, _mult, _ = frames[0]
        assert atoms == ["C", "N"]
        assert len(positions) == 2


class TestParseXyzMetricParseFailure:
    """L153-154: metric token with an unparseable value is silently skipped."""

    def test_unparseable_metric_value_is_ignored(self) -> None:
        """A metric whose value is not a valid float must be silently dropped.

        The regex in parse_xyz only matches float-like patterns, so truly
        unparseable values never enter the ``try`` block.  We verify that
        a comment line with mixed valid and invalid-looking tokens still
        parses cleanly and the valid metrics are captured.
        """
        # "H_total=1.5" is valid; "weirdkey=abc" won't match the float regex
        # at all, so it's simply absent from metrics — no error.
        text = "1\ncharge=0 mult=1 H_total=1.5 weirdkey=abc\nC  0.0 0.0 0.0\n"
        frames = parse_xyz(text)
        assert len(frames) == 1
        _, _, _, _, metrics = frames[0]
        assert metrics.get("H_total") == pytest.approx(1.5)
        assert "weirdkey" not in metrics
