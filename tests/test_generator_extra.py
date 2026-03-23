"""
tests/test_generator_extra.py
==============================
Covers the remaining missed branches in ``pasted._generator`` (96% → higher).

Missed lines targeted
---------------------
515          : ``GenerationResult.summary()`` with ``n_success_target`` set includes
               that value in the output string.
731          : ``StructureGenerator.__init__`` with no config and no ``n_atoms``
               raises ``TypeError``.
797          : ``element_min_counts`` with a negative value raises ValueError.
811          : ``element_max_counts`` with a negative value raises ValueError.
835          : ``center_z`` not in the element pool raises ValueError.
914          : ``_log_sample_result`` with neither metrics nor msg hits the bare
               ``self._log(prefix)`` branch.
946          : ``_resolve_cutoff`` with a user-specified override logs the value.
1067, 1070-1071: ``__getattr__`` raises AttributeError for ``"_cfg"`` (recursion
               guard) and for an unknown attribute name.
1209-1212    : ``_place_one`` raising an error while verbose=True logs the message
               before re-raising.
1234         : relax_positions not converging emits a "warn" log entry.
1651         : ``generate()`` with no config and missing required kwargs raises
               TypeError.
"""

from __future__ import annotations

import pytest

from pasted._config import GeneratorConfig
from pasted._generator import GenerationResult, StructureGenerator, generate

# ---------------------------------------------------------------------------
# GenerationResult.summary() with n_success_target set  (L515)
# ---------------------------------------------------------------------------


class TestGenerationResultSummary:
    def test_summary_includes_n_success_target_when_set(self) -> None:
        """L515: summary() must include 'n_success_target=N' when the field is set."""
        r = GenerationResult(n_success_target=3)
        s = r.summary()
        assert "n_success_target=3" in s

    def test_summary_omits_n_success_target_when_none(self) -> None:
        """n_success_target=None must not appear in summary()."""
        r = GenerationResult()
        assert "n_success_target" not in r.summary()


# ---------------------------------------------------------------------------
# StructureGenerator.__init__ TypeError when n_atoms is missing  (L731)
# ---------------------------------------------------------------------------


class TestStructureGeneratorInit:
    def test_no_config_no_n_atoms_raises_type_error(self) -> None:
        """L731: constructing without config and without n_atoms must raise TypeError."""
        with pytest.raises(TypeError, match="n_atoms"):
            StructureGenerator(charge=0, mult=1)


# ---------------------------------------------------------------------------
# element_min_counts / element_max_counts negative values  (L797, L811)
# ---------------------------------------------------------------------------


class TestNegativeCountsRaiseError:
    def test_negative_element_min_counts_raises(self) -> None:
        """L797: a negative min-count value must raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            StructureGenerator(
                n_atoms=6,
                charge=0,
                mult=1,
                mode="chain",
                elements="6,7,8",
                element_min_counts={"C": -1},
            )

    def test_negative_element_max_counts_raises(self) -> None:
        """L811: a negative max-count value must raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            StructureGenerator(
                n_atoms=6,
                charge=0,
                mult=1,
                mode="chain",
                elements="6,7,8",
                element_max_counts={"C": -1},
            )


# ---------------------------------------------------------------------------
# center_z not in element pool  (L835)
# ---------------------------------------------------------------------------


class TestCenterZNotInPool:
    def test_center_z_absent_from_pool_raises(self) -> None:
        """L835/838: center_z pointing to an element not in the pool must raise."""
        # Pool is C, N, O (Z=6,7,8); center_z=26 (Fe) is absent.
        with pytest.raises(ValueError, match="not in the element pool"):
            StructureGenerator(
                n_atoms=4,
                charge=0,
                mult=1,
                mode="shell",
                elements="6,7,8",
                center_z=26,
            )

    def test_center_z_unknown_atomic_number_raises(self) -> None:
        """L834: center_z with an unsupported Z raises ValueError."""
        with pytest.raises(ValueError, match=r"unknown atomic number|not in the element pool"):
            StructureGenerator(
                n_atoms=4,
                charge=0,
                mult=1,
                mode="shell",
                elements="1-106",
                center_z=999,
            )


# ---------------------------------------------------------------------------
# _log_sample_result bare branch (neither metrics nor msg)  (L914)
# ---------------------------------------------------------------------------


class TestLogSampleResultBareBranch:
    def test_log_sample_result_bare_branch(self, capsys: pytest.CaptureFixture[str]) -> None:
        """L914: calling _log_sample_result with no metrics and no msg logs bare prefix."""
        gen = StructureGenerator(
            n_atoms=4,
            charge=0,
            mult=1,
            mode="chain",
            elements="6,7,8",
            n_samples=1,
            seed=0,
            verbose=True,
        )
        # Call the private helper directly with neither metrics nor msg.
        gen._log_sample_result(0, 1, "1", "PASS", metrics=None, msg=None)
        captured = capsys.readouterr()
        assert "[1/1:PASS]" in captured.err


# ---------------------------------------------------------------------------
# _resolve_cutoff with user-specified override  (L946)
# ---------------------------------------------------------------------------


class TestResolveCutoffVerbose:
    def test_user_cutoff_logged_when_verbose(self, capsys: pytest.CaptureFixture[str]) -> None:
        """L946: when verbose=True, a user-supplied cutoff must be logged."""
        StructureGenerator(
            n_atoms=4,
            charge=0,
            mult=1,
            mode="chain",
            elements="6,7,8",
            n_samples=1,
            seed=0,
            verbose=True,
            cutoff=4.0,
        )
        captured = capsys.readouterr()
        assert "user-specified" in captured.err or "4.000" in captured.err


# ---------------------------------------------------------------------------
# __getattr__ recursion guard and unknown attribute  (L1067, L1070-1071)
# ---------------------------------------------------------------------------


class TestGetAttr:
    def test_getattr_unknown_raises_attribute_error(self) -> None:
        """L1070-1071: accessing a nonexistent attribute must raise AttributeError."""
        gen = StructureGenerator(
            n_atoms=4,
            charge=0,
            mult=1,
            mode="chain",
            elements="6,7,8",
        )
        with pytest.raises(AttributeError, match="no attribute"):
            _ = gen.this_does_not_exist_at_all

    def test_getattr_cfg_guard_raises_attribute_error(self) -> None:
        """L1067: accessing '_cfg' via __getattr__ must raise AttributeError."""
        gen = StructureGenerator(
            n_atoms=4,
            charge=0,
            mult=1,
            mode="chain",
            elements="6,7,8",
        )
        # Delete _cfg to force __getattr__ to be called for it.
        object.__delattr__(gen, "_cfg")
        with pytest.raises(AttributeError):
            _ = gen._cfg


# ---------------------------------------------------------------------------
# _place_one error re-raised with verbose log  (L1209-1212)
# ---------------------------------------------------------------------------


class TestPlaceOneErrorVerbose:
    def test_place_one_error_is_reraised(self, capsys: pytest.CaptureFixture[str]) -> None:
        """L1209-1212: a RuntimeError from _place_one must be logged and re-raised."""
        from unittest.mock import patch

        # C-only pool (Z=6, even) with even n_atoms guarantees parity passes,
        # so _place_one is actually reached before the mock fires.
        gen = StructureGenerator(
            n_atoms=4,
            charge=0,
            mult=1,
            mode="chain",
            elements="6",
            n_samples=1,
            seed=0,
            verbose=True,
        )
        with patch.object(gen, "_place_one", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                gen.generate()
        captured = capsys.readouterr()
        assert "boom" in captured.err


# ---------------------------------------------------------------------------
# relax_positions non-convergence warning  (L1234)
# ---------------------------------------------------------------------------


class TestRelaxNotConverged:
    def test_relax_non_convergence_logs_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        """L1234: when relax_positions returns converged=False, a warning is logged."""
        from unittest.mock import patch

        # C-only pool (Z=6, even) with even n_atoms guarantees parity passes,
        # so relax_positions is actually reached before the mock fires.
        gen = StructureGenerator(
            n_atoms=4,
            charge=0,
            mult=1,
            mode="chain",
            elements="6",
            n_samples=1,
            seed=0,
            verbose=True,
        )

        import pasted._generator as _gen_mod

        original_relax = _gen_mod.relax_positions

        Vec3 = tuple[float, float, float]

        def fake_relax(
            atoms: list[str],
            positions: list[Vec3],
            cov_scale: float,
            max_cycles: int = 500,
            *,
            seed: int | None = None,
        ) -> tuple[list[Vec3], bool]:
            result, _ = original_relax(atoms, positions, cov_scale, max_cycles, seed=seed)
            return result, False  # force converged=False

        with patch.object(_gen_mod, "relax_positions", fake_relax):
            gen.generate()

        captured = capsys.readouterr()
        assert "converge" in captured.err.lower() or "warn" in captured.err.lower()


# ---------------------------------------------------------------------------
# generate() top-level helper — TypeError path  (L1651)
# ---------------------------------------------------------------------------


class TestGenerateFunction:
    def test_missing_required_kwargs_raises_type_error(self) -> None:
        """L1651: generate() without n_atoms/charge/mult must raise TypeError."""
        with pytest.raises(TypeError, match="n_atoms"):
            generate()

    def test_generate_with_config_works(self) -> None:
        """L1646-1647: generate() accepting a GeneratorConfig object."""
        cfg = GeneratorConfig(
            n_atoms=4,
            charge=0,
            mult=1,
            mode="chain",
            elements="6,7,8",
            n_samples=3,
            seed=0,
        )
        result = generate(config=cfg)
        assert result is not None
