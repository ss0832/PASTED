"""
PASTED — Property-Based Tests  (tests/test_property_based.py)
=============================================================
Uses hypothesis to explore the full input space and verify invariants
that hand-written tests cannot exhaustively cover.

Properties under test
  PB-1  Metric value-range invariants
         · Q4/Q6/Q8 ∈ [0, 1] for any valid positions + cutoff
         · graph_lcc / graph_cc / ring_fraction ∈ [0, 1]
         · charge_frustration ≥ 0
         · moran_I_chi ≤ 1.0  (v0.3.8 clamp)
         · H_atom / H_spatial / H_total ≥ 0
         · shape_aniso ∈ [0, 1]

  PB-2  Translation invariance
         · All metrics unchanged when all coordinates shifted by an
           arbitrary vector.

  PB-3  Rotation invariance
         · All metrics unchanged under an arbitrary 3-D rotation.

  PB-4  C++ ↔ Python sparse consistency
         · compute_steinhardt (C++ fast/generic path) agrees with
           _steinhardt_per_atom_sparse to atol=1e-10 for any positions.

  PB-5  ④ fast-path vs generic path
         · [4,6,8] fast-path agrees with [6,4,8] generic to atol=1e-10
           for any positions and cutoff.

  PB-6  generate() structural invariants
         · len(structure.atoms) == n_atoms (+ possibly center atom in shell)
         · charge/mult parity is satisfied
         · all metric keys documented in ALL_METRICS are present
         · every metric value is finite

  PB-7  compute_all_metrics completeness
         · All 13 keys present in the result for any valid input
         · All values finite for finite, non-degenerate input

  PB-8  Metric monotonicity hints (smoke, not strict)
         · Adding more atoms while keeping density constant does not make
           H_atom decrease when the element pool is fixed (non-strict).

Excluded from property tests (covered by FZ tests or too slow):
  · NaN / Inf coordinates (FZ-B/C)
  · Subprocess SIGSEGV isolation (FZ-A)
  · Benchmark / timing properties
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pasted import _ext, generate
from pasted._atoms import ALL_METRICS
from pasted._metrics import (
    _steinhardt_per_atom_sparse,
    compute_all_metrics,
    compute_steinhardt,
)

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------
needs_steinhardt = pytest.mark.skipif(not _ext.HAS_STEINHARDT, reason="HAS_STEINHARDT=False")
needs_graph = pytest.mark.skipif(not _ext.HAS_GRAPH, reason="HAS_GRAPH=False")

# ---------------------------------------------------------------------------
# Hypothesis settings profiles
# ---------------------------------------------------------------------------
# "fast"  — CI-friendly: 50 examples, no DB, short deadline
# "thorough" — local: 300 examples (not used in CI by default)
fast_settings = settings(
    max_examples=60,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=5000,
)

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# Atom count: small enough to keep tests fast, large enough to be interesting
_N_ATOMS = st.integers(min_value=2, max_value=40)

# Finite, bounded coordinate values (avoid overflow and NaN)
_COORD = st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False)


@st.composite
def positions_strategy(draw: Any, n: int | None = None) -> np.ndarray:
    """Draw an (N, 3) float64 array of finite coordinates."""
    n_atoms = draw(_N_ATOMS) if n is None else n
    flat = draw(
        st.lists(
            _COORD,
            min_size=n_atoms * 3,
            max_size=n_atoms * 3,
        )
    )
    return np.array(flat, dtype=np.float64).reshape(n_atoms, 3)


@st.composite
def atoms_and_positions(draw: Any) -> tuple[list[str], np.ndarray]:
    """Draw (atoms, positions) pair with consistent length."""
    pool = ["C", "N", "O", "S", "P", "Fe", "Na", "Cl"]
    pts = draw(positions_strategy())
    n = len(pts)
    atoms = draw(st.lists(st.sampled_from(pool), min_size=n, max_size=n))
    return atoms, pts


# cutoff: keep reasonable range to avoid trivially empty / trivially full graphs
_CUTOFF = st.floats(min_value=1.0, max_value=8.0, allow_nan=False, allow_infinity=False)
_L_VALUES_VALID = [[4], [6], [8], [4, 6], [4, 6, 8], [6, 4, 8], [2, 4, 6, 8]]


# ---------------------------------------------------------------------------
# PB-1  Metric value-range invariants
# ---------------------------------------------------------------------------


class TestMetricRangeInvariants:
    """All metrics must stay within their documented ranges."""

    @needs_steinhardt
    @fast_settings
    @given(pts=positions_strategy(), cutoff=_CUTOFF)
    def test_steinhardt_q_in_unit_interval(self, pts: np.ndarray, cutoff: float) -> None:
        """Q4, Q6, Q8 ∈ [0, 1] for any finite positions and cutoff."""
        result = compute_steinhardt(pts, [4, 6, 8], cutoff)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0 + 1e-12, (
                f"{key}={val:.6f} outside [0, 1] for N={len(pts)}, cutoff={cutoff}"
            )

    @needs_graph
    @fast_settings
    @given(atoms_pos=atoms_and_positions(), cutoff=_CUTOFF)
    def test_graph_metrics_in_unit_interval(
        self, atoms_pos: tuple[list[str], np.ndarray], cutoff: float
    ) -> None:
        """graph_lcc, graph_cc, ring_fraction ∈ [0, 1]."""
        atoms, pts = atoms_pos
        result = compute_all_metrics(atoms, pts.tolist(), cutoff=cutoff)
        for key in ("graph_lcc", "graph_cc", "ring_fraction"):
            val = result[key]
            assert 0.0 - 1e-12 <= val <= 1.0 + 1e-12, f"{key}={val:.6f} outside [0, 1]"

    @needs_graph
    @fast_settings
    @given(atoms_pos=atoms_and_positions(), cutoff=_CUTOFF)
    def test_charge_frustration_non_negative(
        self, atoms_pos: tuple[list[str], np.ndarray], cutoff: float
    ) -> None:
        """charge_frustration ≥ 0 (it is a variance)."""
        atoms, pts = atoms_pos
        result = compute_all_metrics(atoms, pts.tolist(), cutoff=cutoff)
        assert result["charge_frustration"] >= -1e-12, (
            f"charge_frustration={result['charge_frustration']:.6e} < 0"
        )

    @needs_graph
    @fast_settings
    @given(atoms_pos=atoms_and_positions(), cutoff=_CUTOFF)
    def test_moran_I_chi_at_most_one(
        self, atoms_pos: tuple[list[str], np.ndarray], cutoff: float
    ) -> None:
        """moran_I_chi ≤ 1.0 (v0.3.8 clamp; was violated on sparse graphs)."""
        atoms, pts = atoms_pos
        result = compute_all_metrics(atoms, pts.tolist(), cutoff=cutoff)
        assert result["moran_I_chi"] <= 1.0 + 1e-12, (
            f"moran_I_chi={result['moran_I_chi']:.6f} > 1.0"
        )

    @fast_settings
    @given(atoms_pos=atoms_and_positions(), cutoff=_CUTOFF)
    def test_entropy_metrics_non_negative(
        self, atoms_pos: tuple[list[str], np.ndarray], cutoff: float
    ) -> None:
        """H_atom, H_spatial, H_total ≥ 0."""
        atoms, pts = atoms_pos
        result = compute_all_metrics(atoms, pts.tolist(), cutoff=cutoff)
        for key in ("H_atom", "H_spatial", "H_total"):
            assert result[key] >= -1e-12, f"{key}={result[key]:.6e} < 0"

    @fast_settings
    @given(atoms_pos=atoms_and_positions())
    def test_shape_aniso_in_unit_interval(self, atoms_pos: tuple[list[str], np.ndarray]) -> None:
        """shape_aniso ∈ [0, 1]."""
        atoms, pts = atoms_pos
        result = compute_all_metrics(atoms, pts.tolist())
        val = result["shape_aniso"]
        assert -1e-12 <= val <= 1.0 + 1e-12, f"shape_aniso={val:.6f} outside [0, 1]"

    @needs_graph
    @fast_settings
    @given(atoms_pos=atoms_and_positions(), cutoff=_CUTOFF)
    def test_rdf_dev_non_negative(
        self, atoms_pos: tuple[list[str], np.ndarray], cutoff: float
    ) -> None:
        """RDF_dev ≥ 0 (it is an RMS deviation)."""
        atoms, pts = atoms_pos
        result = compute_all_metrics(atoms, pts.tolist(), cutoff=cutoff)
        assert result["RDF_dev"] >= -1e-12, f"RDF_dev={result['RDF_dev']:.6e} < 0"


# ---------------------------------------------------------------------------
# PB-2  Translation invariance
# ---------------------------------------------------------------------------


class TestTranslationInvariance:
    """Disorder metrics are defined on relative geometry and must be
    invariant to rigid translation of all coordinates."""

    @needs_steinhardt
    @fast_settings
    @given(
        pts=positions_strategy(),
        shift=st.tuples(_COORD, _COORD, _COORD),
        cutoff=_CUTOFF,
    )
    def test_steinhardt_translation_invariant(
        self,
        pts: np.ndarray,
        shift: tuple[float, float, float],
        cutoff: float,
    ) -> None:
        sv = np.array(shift)
        pts_shifted = pts + sv
        r_orig = compute_steinhardt(pts, [4, 6, 8], cutoff)
        r_shift = compute_steinhardt(pts_shifted, [4, 6, 8], cutoff)
        for key in r_orig:
            np.testing.assert_allclose(
                r_shift[key],
                r_orig[key],
                atol=1e-9,
                err_msg=f"{key}: not translation-invariant (shift={sv})",
            )

    @needs_graph
    @fast_settings
    @given(
        atoms_pos=atoms_and_positions(),
        shift=st.tuples(_COORD, _COORD, _COORD),
        cutoff=_CUTOFF,
    )
    def test_graph_metrics_translation_invariant(
        self,
        atoms_pos: tuple[list[str], np.ndarray],
        shift: tuple[float, float, float],
        cutoff: float,
    ) -> None:
        atoms, pts = atoms_pos
        sv = np.array(shift)
        pts_shifted = pts + sv
        r_orig = compute_all_metrics(atoms, pts.tolist(), cutoff=cutoff)
        r_shift = compute_all_metrics(atoms, pts_shifted.tolist(), cutoff=cutoff)
        for key in (
            "graph_lcc",
            "graph_cc",
            "ring_fraction",
            "charge_frustration",
            "moran_I_chi",
            "H_spatial",
            "RDF_dev",
            "Q4",
            "Q6",
            "Q8",
        ):
            np.testing.assert_allclose(
                r_shift[key],
                r_orig[key],
                atol=1e-9,
                err_msg=f"{key}: not translation-invariant",
            )


# ---------------------------------------------------------------------------
# PB-3  Rotation invariance
# ---------------------------------------------------------------------------


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Return a uniformly random 3x3 rotation matrix via QR decomposition."""
    H = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(H)
    Q *= np.sign(np.diag(R))  # ensure det=+1
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


@st.composite
def rotation_matrix(draw: Any) -> np.ndarray:
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    return _random_rotation(np.random.default_rng(seed))


class TestRotationInvariance:
    """Steinhardt Q_l and graph metrics are fully isotropic and must be
    unchanged under any rigid rotation of all coordinates."""

    @needs_steinhardt
    @fast_settings
    @given(pts=positions_strategy(), rot=rotation_matrix(), cutoff=_CUTOFF)
    def test_steinhardt_rotation_invariant(
        self, pts: np.ndarray, rot: np.ndarray, cutoff: float
    ) -> None:
        pts_rot = pts @ rot.T
        r_orig = compute_steinhardt(pts, [4, 6, 8], cutoff)
        r_rot = compute_steinhardt(pts_rot, [4, 6, 8], cutoff)
        for key in r_orig:
            np.testing.assert_allclose(
                r_rot[key],
                r_orig[key],
                atol=1e-8,
                err_msg=f"{key}: not rotation-invariant",
            )

    @needs_graph
    @fast_settings
    @given(
        atoms_pos=atoms_and_positions(),
        rot=rotation_matrix(),
        cutoff=_CUTOFF,
    )
    def test_graph_metrics_rotation_invariant(
        self,
        atoms_pos: tuple[list[str], np.ndarray],
        rot: np.ndarray,
        cutoff: float,
    ) -> None:
        atoms, pts = atoms_pos
        pts_rot = pts @ rot.T
        r_orig = compute_all_metrics(atoms, pts.tolist(), cutoff=cutoff)
        r_rot = compute_all_metrics(atoms, pts_rot.tolist(), cutoff=cutoff)
        for key in (
            "graph_lcc",
            "graph_cc",
            "ring_fraction",
            "Q4",
            "Q6",
            "Q8",
            "H_spatial",
            "RDF_dev",
        ):
            np.testing.assert_allclose(
                r_rot[key],
                r_orig[key],
                atol=1e-8,
                err_msg=f"{key}: not rotation-invariant",
            )


# ---------------------------------------------------------------------------
# PB-4  C++ ↔ Python sparse consistency
# ---------------------------------------------------------------------------


class TestCppVsPythonConsistency:
    """C++ steinhardt_per_atom_cpp and Python _steinhardt_per_atom_sparse
    must agree to within floating-point round-off for any valid input."""

    @needs_steinhardt
    @fast_settings
    @given(pts=positions_strategy(), cutoff=_CUTOFF)
    def test_cpp_vs_python_sparse_l468(self, pts: np.ndarray, cutoff: float) -> None:
        """l=[4,6,8]: C++ fast-path vs Python sparse, atol=1e-10."""
        cpp = compute_steinhardt(pts, [4, 6, 8], cutoff)
        py_pa = _steinhardt_per_atom_sparse(pts, [4, 6, 8], cutoff)
        for l_val in [4, 6, 8]:
            key = f"Q{l_val}"
            py_mean = float(np.mean(py_pa[key]))
            np.testing.assert_allclose(
                cpp[key],
                py_mean,
                atol=1e-10,
                err_msg=f"{key}: C++ vs Python mismatch (N={len(pts)}, cutoff={cutoff})",
            )

    @needs_steinhardt
    @fast_settings
    @given(pts=positions_strategy(), cutoff=_CUTOFF)
    def test_cpp_vs_python_sparse_l6_only(self, pts: np.ndarray, cutoff: float) -> None:
        """l=[6] (generic path): C++ vs Python sparse."""
        cpp = compute_steinhardt(pts, [6], cutoff)
        py_pa = _steinhardt_per_atom_sparse(pts, [6], cutoff)
        py_mean = float(np.mean(py_pa["Q6"]))
        np.testing.assert_allclose(
            cpp["Q6"],
            py_mean,
            atol=1e-10,
            err_msg=f"Q6: C++ vs Python mismatch (N={len(pts)}, cutoff={cutoff})",
        )


# ---------------------------------------------------------------------------
# PB-5  ④ fast-path vs generic path consistency
# ---------------------------------------------------------------------------


class TestFastPathVsGeneric:
    """④ fast-path ([4,6,8] in order) must agree with ①②③ generic path
    ([6,4,8] forces the generic branch) to within numerical round-off."""

    @needs_steinhardt
    @fast_settings
    @given(pts=positions_strategy(), cutoff=_CUTOFF)
    def test_fast_path_vs_generic_all_n(self, pts: np.ndarray, cutoff: float) -> None:
        fast = compute_steinhardt(pts, [4, 6, 8], cutoff)
        generic = compute_steinhardt(pts, [6, 4, 8], cutoff)
        for key in ("Q4", "Q6", "Q8"):
            np.testing.assert_allclose(
                fast[key],
                generic[key],
                atol=1e-10,
                err_msg=(f"{key}: fast-path vs generic mismatch (N={len(pts)}, cutoff={cutoff})"),
            )


# ---------------------------------------------------------------------------
# PB-6  generate() structural invariants
# ---------------------------------------------------------------------------

_GENERATE_ELEMENTS = ["6,7,8", "6,7,8,16", "6,7,8,15,16"]
_GENERATE_N = [5, 10, 15, 20]


class TestGenerateInvariants:
    """generate() must always produce structures satisfying hard constraints."""

    @fast_settings
    @given(
        n_atoms=st.sampled_from(_GENERATE_N),
        elements=st.sampled_from(_GENERATE_ELEMENTS),
        seed=st.integers(min_value=0, max_value=9999),
    )
    def test_generate_gas_atom_count(self, n_atoms: int, elements: str, seed: int) -> None:
        """gas mode: every generated structure has exactly n_atoms atoms."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = generate(
                n_atoms=n_atoms,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:9",
                elements=elements,
                n_samples=3,
                seed=seed,
            )
        for s in result:
            assert len(s.atoms) == n_atoms, f"Expected {n_atoms} atoms, got {len(s.atoms)}"

    @fast_settings
    @given(
        n_atoms=st.sampled_from(_GENERATE_N),
        elements=st.sampled_from(_GENERATE_ELEMENTS),
        seed=st.integers(min_value=0, max_value=9999),
    )
    def test_generate_gas_parity(self, n_atoms: int, elements: str, seed: int) -> None:
        """Every generated structure must satisfy the electron-count parity
        constraint: sum(Z_i) − charge must have the correct even/odd parity
        for the given multiplicity."""
        from pasted._atoms import ATOMIC_NUMBERS

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = generate(
                n_atoms=n_atoms,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:9",
                elements=elements,
                n_samples=3,
                seed=seed,
            )
        for s in result:
            n_electrons = sum(ATOMIC_NUMBERS[a] for a in s.atoms) - s.charge
            # mult=1 (singlet) requires even number of electrons
            # mult=2 (doublet) requires odd, etc.
            expected_parity = (s.mult - 1) % 2  # 0=even, 1=odd
            assert n_electrons % 2 == expected_parity, (
                f"Parity violation: n_e={n_electrons}, mult={s.mult}, comp={s.comp}"
            )

    @fast_settings
    @given(
        n_atoms=st.sampled_from(_GENERATE_N),
        seed=st.integers(min_value=0, max_value=9999),
    )
    def test_generate_metrics_all_keys_finite(self, n_atoms: int, seed: int) -> None:
        """Every structure's metrics dict contains all ALL_METRICS keys
        with finite values."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = generate(
                n_atoms=n_atoms,
                charge=0,
                mult=1,
                mode="gas",
                region="sphere:9",
                elements="6,7,8",
                n_samples=3,
                seed=seed,
            )
        for s in result:
            for key in ALL_METRICS:
                assert key in s.metrics, f"Missing metric key {key!r} in structure metrics"
                assert np.isfinite(s.metrics[key]), (
                    f"metrics[{key!r}]={s.metrics[key]} is not finite"
                )


# ---------------------------------------------------------------------------
# PB-7  compute_all_metrics completeness and finiteness
# ---------------------------------------------------------------------------


class TestComputeAllMetricsCompleteness:
    """compute_all_metrics must return all documented keys with finite values
    for any well-formed (finite, non-empty) input."""

    @fast_settings
    @given(atoms_pos=atoms_and_positions(), cutoff=_CUTOFF)
    def test_all_keys_present(
        self, atoms_pos: tuple[list[str], np.ndarray], cutoff: float
    ) -> None:
        """Result must contain every key in ALL_METRICS."""
        atoms, pts = atoms_pos
        result = compute_all_metrics(atoms, pts.tolist(), cutoff=cutoff)
        for key in ALL_METRICS:
            assert key in result, f"Missing key {key!r} in compute_all_metrics output"

    @fast_settings
    @given(atoms_pos=atoms_and_positions(), cutoff=_CUTOFF)
    def test_all_values_finite(
        self, atoms_pos: tuple[list[str], np.ndarray], cutoff: float
    ) -> None:
        """Every value in the result must be a finite float."""
        atoms, pts = atoms_pos
        result = compute_all_metrics(atoms, pts.tolist(), cutoff=cutoff)
        for key, val in result.items():
            assert np.isfinite(val), (
                f"compute_all_metrics[{key!r}]={val} is not finite "
                f"(N={len(atoms)}, cutoff={cutoff})"
            )

    @fast_settings
    @given(atoms_pos=atoms_and_positions(), cutoff=_CUTOFF)
    def test_result_is_dict_of_floats(
        self, atoms_pos: tuple[list[str], np.ndarray], cutoff: float
    ) -> None:
        """All values must be plain Python floats."""
        atoms, pts = atoms_pos
        result = compute_all_metrics(atoms, pts.tolist(), cutoff=cutoff)
        assert isinstance(result, dict)
        for key, val in result.items():
            assert isinstance(val, float), (
                f"result[{key!r}] is {type(val).__name__}, expected float"
            )


# ---------------------------------------------------------------------------
# PB-8  Steinhardt per-atom values also in [0,1]
# ---------------------------------------------------------------------------


class TestPerAtomRanges:
    """Per-atom Q_l values (not just their mean) must be in [0, 1]."""

    @needs_steinhardt
    @fast_settings
    @given(pts=positions_strategy(), cutoff=_CUTOFF)
    def test_per_atom_q_in_unit_interval(self, pts: np.ndarray, cutoff: float) -> None:
        from pasted._ext import steinhardt_per_atom

        result = steinhardt_per_atom(pts, cutoff, [4, 6, 8])
        for key, arr in result.items():
            assert np.all(arr >= -1e-12), f"per-atom {key} has values < 0: min={arr.min():.6f}"
            assert np.all(arr <= 1.0 + 1e-12), (
                f"per-atom {key} has values > 1: max={arr.max():.6f}"
            )


# ---------------------------------------------------------------------------
# PB-9  Idempotency: calling twice with same input gives same result
# ---------------------------------------------------------------------------


class TestIdempotency:
    """compute_all_metrics is a pure function: same input → same output."""

    @fast_settings
    @given(atoms_pos=atoms_and_positions(), cutoff=_CUTOFF)
    def test_compute_all_metrics_idempotent(
        self, atoms_pos: tuple[list[str], np.ndarray], cutoff: float
    ) -> None:
        atoms, pts = atoms_pos
        r1 = compute_all_metrics(atoms, pts.tolist(), cutoff=cutoff)
        r2 = compute_all_metrics(atoms, pts.tolist(), cutoff=cutoff)
        for key in r1:
            assert r1[key] == r2[key], (
                f"compute_all_metrics not idempotent for {key!r}: {r1[key]} != {r2[key]}"
            )

    @needs_steinhardt
    @fast_settings
    @given(pts=positions_strategy(), cutoff=_CUTOFF)
    def test_compute_steinhardt_idempotent(self, pts: np.ndarray, cutoff: float) -> None:
        r1 = compute_steinhardt(pts, [4, 6, 8], cutoff)
        r2 = compute_steinhardt(pts, [4, 6, 8], cutoff)
        for key in r1:
            assert r1[key] == r2[key], f"compute_steinhardt not idempotent for {key!r}"
