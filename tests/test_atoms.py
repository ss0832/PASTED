"""Tests for pasted._atoms: parsers, validators, atomic data."""

from __future__ import annotations

import math

import pytest

from pasted._atoms import (
    ATOMIC_NUMBERS,
    cov_radius_ang,
    default_element_pool,
    parse_element_spec,
    parse_filter,
    parse_int_range,
    parse_lo_hi,
    validate_charge_mult,
)

# ---------------------------------------------------------------------------
# parse_element_spec
# ---------------------------------------------------------------------------

class TestParseElementSpec:
    def test_range(self) -> None:
        syms = parse_element_spec("1-3")
        assert syms == ["H", "He", "Li"]

    def test_list(self) -> None:
        syms = parse_element_spec("6,7,8")
        assert syms == ["C", "N", "O"]

    def test_mixed(self) -> None:
        syms = parse_element_spec("1-2,26")
        assert syms == ["H", "He", "Fe"]

    def test_single(self) -> None:
        assert parse_element_spec("6") == ["C"]

    def test_deduplication(self) -> None:
        # 1-3 and 2 overlap; result should still be H, He, Li
        syms = parse_element_spec("1-3,2")
        assert syms == ["H", "He", "Li"]

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty pool"):
            parse_element_spec("")

    def test_bad_range_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_element_spec("10-5")

    def test_unsupported_z_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_element_spec("200")


# ---------------------------------------------------------------------------
# default_element_pool
# ---------------------------------------------------------------------------

def test_default_pool_length() -> None:
    pool = default_element_pool()
    assert len(pool) == 106

def test_default_pool_sorted_by_z() -> None:
    pool = default_element_pool()
    zs = [ATOMIC_NUMBERS[s] for s in pool]
    assert zs == sorted(zs)


# ---------------------------------------------------------------------------
# parse_lo_hi / parse_int_range
# ---------------------------------------------------------------------------

def test_parse_lo_hi_basic() -> None:
    lo, hi = parse_lo_hi("1.2:1.6", "bond-range")
    assert lo == pytest.approx(1.2)
    assert hi == pytest.approx(1.6)

def test_parse_lo_hi_bad_format() -> None:
    with pytest.raises(ValueError):
        parse_lo_hi("1.2", "bond-range")

def test_parse_int_range_basic() -> None:
    lo, hi = parse_int_range("4:8")
    assert lo == 4
    assert hi == 8

def test_parse_int_range_zero_raises() -> None:
    with pytest.raises(ValueError):
        parse_int_range("0:4")

def test_parse_int_range_inverted_raises() -> None:
    with pytest.raises(ValueError):
        parse_int_range("8:4")


# ---------------------------------------------------------------------------
# parse_filter
# ---------------------------------------------------------------------------

class TestParseFilter:
    def test_open_hi(self) -> None:
        metric, lo, hi = parse_filter("H_total:2.0:-")
        assert metric == "H_total"
        assert lo == pytest.approx(2.0)
        assert math.isinf(hi) and hi > 0

    def test_open_lo(self) -> None:
        metric, lo, hi = parse_filter("Q6:-:0.4")
        assert metric == "Q6"
        assert math.isinf(lo) and lo < 0
        assert hi == pytest.approx(0.4)

    def test_closed(self) -> None:
        metric, lo, hi = parse_filter("shape_aniso:0.1:0.9")
        assert metric == "shape_aniso"
        assert lo == pytest.approx(0.1)
        assert hi == pytest.approx(0.9)

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            parse_filter("bogus:0:1")

    def test_bad_format_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_filter("H_total:2.0")

    def test_min_gt_max_raises(self) -> None:
        with pytest.raises(ValueError, match="MIN > MAX"):
            parse_filter("H_total:5.0:1.0")


# ---------------------------------------------------------------------------
# validate_charge_mult
# ---------------------------------------------------------------------------

class TestValidateChargeMult:
    def test_h2_singlet_ok(self) -> None:
        ok, msg = validate_charge_mult(["H", "H"], 0, 1)
        assert ok, msg

    def test_h_doublet_ok(self) -> None:
        # H: Z=1, charge=0, n_e=1, mult=2 (1 unpaired) → parity ok
        ok, msg = validate_charge_mult(["H"], 0, 2)
        assert ok, msg

    def test_parity_fail(self) -> None:
        # H2: n_e=2, mult=2 → n_unpaired=1, but 2%2=0 ≠ 1%2=1
        ok, msg = validate_charge_mult(["H", "H"], 0, 2)
        assert not ok
        assert "parity" in msg

    def test_negative_electrons_fail(self) -> None:
        # Very high positive charge
        ok, msg = validate_charge_mult(["H"], 5, 1)
        assert not ok
        assert "n_electrons" in msg

    def test_composition_in_message(self) -> None:
        # C+C: Z=6+6=12, n_e=12, mult=1 → n_unpaired=0, 12%2==0%2 → OK
        ok, msg = validate_charge_mult(["C", "C"], 0, 1)
        assert ok
        assert "C" in msg


# ---------------------------------------------------------------------------
# cov_radius_ang
# ---------------------------------------------------------------------------

def test_cov_radius_known_elements() -> None:
    assert cov_radius_ang("H") == pytest.approx(0.32)
    assert cov_radius_ang("C") == pytest.approx(0.75)
    assert cov_radius_ang("Fe") == pytest.approx(1.16)

def test_cov_radius_proxy_elements() -> None:
    # Fr → Cs proxy
    assert cov_radius_ang("Fr") == cov_radius_ang("Cs")
    # Rf → Hf proxy
    assert cov_radius_ang("Rf") == cov_radius_ang("Hf")

def test_cov_radius_all_supported() -> None:
    """All supported elements should return a positive radius."""
    for sym in ATOMIC_NUMBERS:
        r = cov_radius_ang(sym)
        assert r > 0, f"Non-positive radius for {sym}: {r}"
