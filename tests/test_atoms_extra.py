"""
tests/test_atoms_extra.py
=========================
Covers the remaining missed branches in ``pasted._atoms`` (94% → higher).

Missed lines targeted
---------------------
307  : ``cov_radius_ang`` ultimate fallback return 1.50 for a symbol with no
       entry in ``_COV_RADII_ANG`` and no proxy in ``_COV_RADII_PROXY``.
510  : ``parse_element_spec`` (list branch) — non-string entry raises ValueError.
514  : ``parse_element_spec`` (list branch) — unknown symbol raises ValueError.
520  : ``parse_element_spec`` (list branch) — empty list raises ValueError.
571  : ``parse_int_range`` — input without a colon raises ValueError.
"""

from __future__ import annotations

import pytest

from pasted._atoms import cov_radius_ang, parse_element_spec, parse_int_range


class TestCovRadiusAngFallback:
    """L307: ultimate fallback 1.50 Å for a symbol absent from both dicts."""

    def test_unknown_symbol_returns_fallback(self) -> None:
        """A symbol not in _COV_RADII_ANG and not in _COV_RADII_PROXY returns 1.50."""
        # "Xx" is not a real element and exists in neither lookup dict.
        result = cov_radius_ang("Xx")
        assert result == pytest.approx(1.50)

    def test_known_symbol_does_not_use_fallback(self) -> None:
        """A real element symbol must NOT return the fallback value (sanity check)."""
        # Carbon (C) has a well-defined radius; it should differ from the 1.50 fallback.
        result = cov_radius_ang("C")
        assert result != pytest.approx(1.50)


class TestParseElementSpecListBranch:
    """L510, 514, 520: parse_element_spec with an explicit symbol list."""

    def test_non_string_entry_raises(self) -> None:
        """L510: an integer in the symbol list must raise ValueError."""
        with pytest.raises(ValueError, match="must be strings"):
            parse_element_spec([123])  # type: ignore[list-item]

    def test_unknown_symbol_raises(self) -> None:
        """L514: an unrecognised symbol string must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown element symbol"):
            parse_element_spec(["Xx"])

    def test_empty_list_raises(self) -> None:
        """L520: an empty list must raise ValueError."""
        with pytest.raises(ValueError, match="empty pool"):
            parse_element_spec([])

    def test_valid_symbol_list_round_trips(self) -> None:
        """Happy path: a valid symbol list returns the correct elements."""
        result = parse_element_spec(["C", "N", "O"])
        assert set(result) == {"C", "N", "O"}


class TestParseIntRangeErrors:
    """L571: parse_int_range without a colon raises ValueError."""

    def test_missing_colon_raises(self) -> None:
        """L571: input with no ':' separator must raise ValueError."""
        with pytest.raises(ValueError, match="Must be 'MIN:MAX'"):
            parse_int_range("BADFORMAT")

    def test_valid_range_parses(self) -> None:
        """Sanity check: a well-formed range returns the correct tuple."""
        assert parse_int_range("2:6") == (2, 6)
