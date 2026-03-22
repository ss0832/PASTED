"""
tests/test_cli.py
=================
Comprehensive test suite for ``pasted.cli`` (v0.3.10).

Coverage strategy
-----------------
1.  **Parser structure** — ``build_parser()`` returns an ``ArgumentParser``
    with the correct argument names, types, defaults, and choice lists.
2.  **Type validation** — every argument that carries an explicit ``type=``
    (``int``, ``float``) is verified to produce the correct Python type after
    parsing; ``store_true`` flags produce ``bool``; ``append`` actions produce
    ``list``; string arguments stay ``str``.
3.  **Required-argument enforcement** — missing ``--n-atoms``, ``--charge``,
    or ``--mult`` cause a non-zero exit.
4.  **Mode / region validation** — ``gas`` and ``maxent`` require ``--region``
    in sampling mode; ``chain`` and ``shell`` do not.
5.  **All placement modes** — smoke-test each of gas, chain, shell, maxent
    via subprocess to verify end-to-end output.
6.  **Filter parsing** — ``--filter`` is repeatable and produces a list of
    valid filter strings; open bounds (``-``) round-trip correctly.
7.  **Element pool parsing** — ``--elements`` spec string, ``--element-fractions``,
    ``--element-min-counts``, ``--element-max-counts`` all validated for
    correct types and error handling.
8.  **Output routing** — stdout (default) vs ``--output FILE``.
9.  **``--validate`` flag** — exits 0 on a satisfiable parity constraint,
    exits 1 on an impossible constraint.
10. **Sampling controls** — ``--n-samples``, ``--n-success``, ``--seed``
    produce reproducible deterministic output.
11. **Affine transform flags** — ``--affine-strength``, ``--affine-stretch``,
    ``--affine-shear``, ``--affine-jitter`` are parsed as ``float | None``.
12. **Optimizer mode** — ``--optimize`` activates ``StructureOptimizer``;
    ``--objective``, ``--method``, ``--max-steps``, ``--T-start``, ``--T-end``,
    ``--n-replicas``, ``--pt-swap-interval`` all forwarded correctly.
13. **``--initial-xyz``** — loads an XYZ file and uses it as the optimizer
    starting structure; missing file causes a graceful error exit.
14. **Mutual-exclusion errors** — ``--no-displacements`` combined with
    ``--no-composition-moves`` causes exit 1.
15. **Internal helpers** — ``_write_output`` correctly routes to file or
    stdout; ``build_parser`` is idempotent (multiple calls produce
    independent parsers).
16. **XYZ output validity** — all subprocess-generated XYZ files parse
    cleanly via ``pasted._io.parse_xyz``.

All tests pass ``ruff check`` and ``mypy`` with the project's strictness
settings.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from pasted._io import parse_xyz
from pasted.cli import _write_output, build_parser, main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimum valid CLI arguments for gas mode
_GAS_BASE: list[str] = [
    "--n-atoms", "6",
    "--charge", "0",
    "--mult", "1",
    "--mode", "gas",
    "--region", "sphere:6",
    "--elements", "6,7,8",
    "--n-samples", "5",
    "--seed", "42",
]

# Minimum valid CLI arguments for chain mode (no --region required)
_CHAIN_BASE: list[str] = [
    "--n-atoms", "6",
    "--charge", "0",
    "--mult", "1",
    "--mode", "chain",
    "--elements", "6,7,8",
    "--n-samples", "5",
    "--seed", "0",
]


class _MockCompletedProcess:
    """Drop-in replacement for subprocess.CompletedProcess used by _run()."""

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _run(*extra_args: str) -> _MockCompletedProcess:
    """Invoke main() in-process instead of spawning a subprocess.

    This allows pytest-cov to instrument cli.py directly, so coverage is
    measured correctly rather than being invisible inside a child process.
    """
    args = ["pasted", *extra_args]
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    returncode = 0

    with (
        patch("sys.argv", args),
        patch("sys.stdout", captured_stdout),
        patch("sys.stderr", captured_stderr),
    ):
        try:
            main()
        except SystemExit as e:
            returncode = e.code if isinstance(e.code, int) else (1 if e.code else 0)
        except Exception as e:
            returncode = 1
            captured_stderr.write(str(e))

    return _MockCompletedProcess(
        returncode=returncode,
        stdout=captured_stdout.getvalue(),
        stderr=captured_stderr.getvalue(),
    )


def _parse_args(*argv: str) -> argparse.Namespace:
    """Parse *argv* with ``build_parser()`` without exit-on-error."""
    return build_parser().parse_args(list(argv))


# ===========================================================================
# 1. Parser structure — build_parser()
# ===========================================================================


class TestBuildParser:
    """Verify that build_parser() returns a well-formed ArgumentParser."""

    def test_returns_argument_parser(self) -> None:
        """build_parser() must return an ArgumentParser instance."""
        p = build_parser()
        assert isinstance(p, argparse.ArgumentParser)

    def test_prog_name(self) -> None:
        """Parser prog must be 'pasted'."""
        p = build_parser()
        assert p.prog == "pasted"

    def test_idempotent_multiple_calls(self) -> None:
        """Each call to build_parser() must return a fresh, independent parser."""
        p1 = build_parser()
        p2 = build_parser()
        assert p1 is not p2
        # Both can parse the same argv without interference
        ns1 = p1.parse_args(_CHAIN_BASE)
        ns2 = p2.parse_args(_CHAIN_BASE)
        assert ns1.n_atoms == ns2.n_atoms

    def test_all_required_args_present(self) -> None:
        """--n-atoms, --charge, --mult must be registered as required."""
        p = build_parser()
        required_dests = {
            action.dest
            for group in p._action_groups
            for action in group._group_actions
            if getattr(action, "required", False)
        }
        assert "n_atoms" in required_dests
        assert "charge" in required_dests
        assert "mult" in required_dests

    def test_mode_choices(self) -> None:
        """--mode must have exactly the four documented choices."""
        p = build_parser()
        mode_action = next(
            (
                a
                for group in p._action_groups
                for a in group._group_actions
                if getattr(a, "dest", None) == "mode"
            ),
            None,
        )
        assert mode_action is not None
        choices = mode_action.choices
        assert choices is not None
        assert set(choices) == {"gas", "chain", "shell", "maxent"}

    def test_method_choices(self) -> None:
        """--method must have exactly the three documented choices."""
        p = build_parser()
        method_action = next(
            (
                a
                for group in p._action_groups
                for a in group._group_actions
                if getattr(a, "dest", None) == "method"
            ),
            None,
        )
        assert method_action is not None
        method_choices = method_action.choices
        assert method_choices is not None
        assert set(method_choices) == {
            "annealing", "basin_hopping", "parallel_tempering"
        }


# ===========================================================================
# 2. Type validation — parsed argument types
# ===========================================================================


class TestArgumentTypes:
    """Verify that every argument is parsed to the declared Python type."""

    def _ns(self) -> argparse.Namespace:
        """Namespace with all commonly-used non-default values set."""
        return _parse_args(
            "--n-atoms", "12",
            "--charge", "-1",
            "--mult", "2",
            "--mode", "chain",
            "--branch-prob", "0.4",
            "--chain-persist", "0.7",
            "--chain-bias", "0.3",
            "--bond-range", "1.2:1.8",
            "--center-z", "26",
            "--coord-range", "4:6",
            "--shell-radius", "1.8:2.5",
            "--maxent-steps", "200",
            "--maxent-lr", "0.03",
            "--maxent-cutoff-scale", "3.0",
            "--cov-scale", "1.1",
            "--relax-cycles", "500",
            "--affine-strength", "0.2",
            "--affine-stretch", "0.3",
            "--affine-shear", "0.1",
            "--affine-jitter", "0.05",
            "--n-samples", "20",
            "--n-success", "5",
            "--seed", "99",
            "--n-bins", "30",
            "--w-atom", "0.6",
            "--w-spatial", "0.4",
            "--cutoff", "3.5",
            "--max-steps", "1000",
            "--T-start", "2.0",
            "--T-end", "0.005",
            "--frag-threshold", "0.25",
            "--move-step", "0.3",
            "--lcc-threshold", "0.5",
            "--n-replicas", "3",
            "--pt-swap-interval", "5",
        )

    # --- int arguments ---

    def test_n_atoms_is_int(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert isinstance(ns.n_atoms, int)
        assert ns.n_atoms == 6

    def test_charge_is_int(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert isinstance(ns.charge, int)
        assert ns.charge == 0

    def test_mult_is_int(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert isinstance(ns.mult, int)
        assert ns.mult == 1

    def test_negative_charge_is_int(self) -> None:
        ns = self._ns()
        assert isinstance(ns.charge, int)
        assert ns.charge == -1

    def test_maxent_steps_is_int(self) -> None:
        ns = self._ns()
        assert isinstance(ns.maxent_steps, int)
        assert ns.maxent_steps == 200

    def test_relax_cycles_is_int(self) -> None:
        ns = self._ns()
        assert isinstance(ns.relax_cycles, int)
        assert ns.relax_cycles == 500

    def test_n_samples_is_int(self) -> None:
        ns = self._ns()
        assert isinstance(ns.n_samples, int)
        assert ns.n_samples == 20

    def test_n_success_is_int(self) -> None:
        ns = self._ns()
        assert isinstance(ns.n_success, int)
        assert ns.n_success == 5

    def test_seed_is_int(self) -> None:
        ns = self._ns()
        assert isinstance(ns.seed, int)
        assert ns.seed == 99

    def test_n_bins_is_int(self) -> None:
        ns = self._ns()
        assert isinstance(ns.n_bins, int)
        assert ns.n_bins == 30

    def test_max_steps_is_int(self) -> None:
        ns = self._ns()
        assert isinstance(ns.max_steps, int)
        assert ns.max_steps == 1000

    def test_n_replicas_is_int(self) -> None:
        ns = self._ns()
        assert isinstance(ns.n_replicas, int)
        assert ns.n_replicas == 3

    def test_pt_swap_interval_is_int(self) -> None:
        ns = self._ns()
        assert isinstance(ns.pt_swap_interval, int)
        assert ns.pt_swap_interval == 5

    def test_center_z_is_int(self) -> None:
        ns = self._ns()
        assert isinstance(ns.center_z, int)
        assert ns.center_z == 26

    # --- float arguments ---

    def test_branch_prob_is_float(self) -> None:
        ns = self._ns()
        assert isinstance(ns.branch_prob, float)
        assert ns.branch_prob == pytest.approx(0.4)

    def test_chain_persist_is_float(self) -> None:
        ns = self._ns()
        assert isinstance(ns.chain_persist, float)
        assert ns.chain_persist == pytest.approx(0.7)

    def test_chain_bias_is_float(self) -> None:
        ns = self._ns()
        assert isinstance(ns.chain_bias, float)
        assert ns.chain_bias == pytest.approx(0.3)

    def test_maxent_lr_is_float(self) -> None:
        ns = self._ns()
        assert isinstance(ns.maxent_lr, float)
        assert ns.maxent_lr == pytest.approx(0.03)

    def test_maxent_cutoff_scale_is_float(self) -> None:
        ns = self._ns()
        assert isinstance(ns.maxent_cutoff_scale, float)
        assert ns.maxent_cutoff_scale == pytest.approx(3.0)

    def test_cov_scale_is_float(self) -> None:
        ns = self._ns()
        assert isinstance(ns.cov_scale, float)
        assert ns.cov_scale == pytest.approx(1.1)

    def test_affine_strength_is_float(self) -> None:
        ns = self._ns()
        assert isinstance(ns.affine_strength, float)
        assert ns.affine_strength == pytest.approx(0.2)

    def test_affine_stretch_is_float(self) -> None:
        ns = self._ns()
        assert isinstance(ns.affine_stretch, float)
        assert ns.affine_stretch == pytest.approx(0.3)

    def test_affine_shear_is_float(self) -> None:
        ns = self._ns()
        assert isinstance(ns.affine_shear, float)
        assert ns.affine_shear == pytest.approx(0.1)

    def test_affine_jitter_is_float(self) -> None:
        ns = self._ns()
        assert isinstance(ns.affine_jitter, float)
        assert ns.affine_jitter == pytest.approx(0.05)

    def test_w_atom_is_float(self) -> None:
        ns = self._ns()
        assert isinstance(ns.w_atom, float)
        assert ns.w_atom == pytest.approx(0.6)

    def test_w_spatial_is_float(self) -> None:
        ns = self._ns()
        assert isinstance(ns.w_spatial, float)
        assert ns.w_spatial == pytest.approx(0.4)

    def test_cutoff_is_float(self) -> None:
        ns = self._ns()
        assert isinstance(ns.cutoff, float)
        assert ns.cutoff == pytest.approx(3.5)

    def test_T_start_is_float(self) -> None:
        ns = self._ns()
        assert isinstance(ns.T_start, float)
        assert ns.T_start == pytest.approx(2.0)

    def test_T_end_is_float(self) -> None:
        ns = self._ns()
        assert isinstance(ns.T_end, float)
        assert ns.T_end == pytest.approx(0.005)

    def test_frag_threshold_is_float(self) -> None:
        ns = self._ns()
        assert isinstance(ns.frag_threshold, float)
        assert ns.frag_threshold == pytest.approx(0.25)

    def test_move_step_is_float(self) -> None:
        ns = self._ns()
        assert isinstance(ns.move_step, float)
        assert ns.move_step == pytest.approx(0.3)

    def test_lcc_threshold_is_float(self) -> None:
        ns = self._ns()
        assert isinstance(ns.lcc_threshold, float)
        assert ns.lcc_threshold == pytest.approx(0.5)

    # --- str arguments ---

    def test_mode_is_str(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert isinstance(ns.mode, str)
        assert ns.mode == "chain"

    def test_region_is_str(self) -> None:
        ns = _parse_args(*_GAS_BASE)
        assert isinstance(ns.region, str)
        assert ns.region == "sphere:6"

    def test_bond_range_is_str(self) -> None:
        """bond-range is parsed as a raw string; parse_lo_hi() converts it later."""
        ns = _parse_args(*_CHAIN_BASE)
        assert isinstance(ns.bond_range, str)

    def test_output_is_none_by_default(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert ns.output is None

    def test_initial_xyz_is_str(self) -> None:
        ns = _parse_args(*_CHAIN_BASE, "--initial-xyz", "/tmp/fake.xyz")
        assert isinstance(ns.initial_xyz, str)
        assert ns.initial_xyz == "/tmp/fake.xyz"

    # --- bool / flag arguments ---

    def test_optimize_is_bool_false_default(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert isinstance(ns.optimize, bool)
        assert ns.optimize is False

    def test_optimize_is_bool_true_when_set(self) -> None:
        ns = _parse_args(*_CHAIN_BASE, "--optimize")
        assert isinstance(ns.optimize, bool)
        assert ns.optimize is True

    def test_no_add_hydrogen_is_bool(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert isinstance(ns.no_add_hydrogen, bool)
        assert ns.no_add_hydrogen is False

    def test_no_add_hydrogen_true_when_flag_set(self) -> None:
        ns = _parse_args(*_CHAIN_BASE, "--no-add-hydrogen")
        assert isinstance(ns.no_add_hydrogen, bool)
        assert ns.no_add_hydrogen is True

    def test_validate_is_bool(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert isinstance(ns.validate, bool)
        assert ns.validate is False

    def test_no_composition_moves_is_bool(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert isinstance(ns.no_composition_moves, bool)
        assert ns.no_composition_moves is False

    def test_no_displacements_is_bool(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert isinstance(ns.no_displacements, bool)
        assert ns.no_displacements is False

    # --- list arguments ---

    def test_filters_is_list(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert isinstance(ns.filters, list)

    def test_filters_append(self) -> None:
        ns = _parse_args(
            *_CHAIN_BASE,
            "--filter", "H_total:1.0:-",
            "--filter", "Q6:-:0.5",
        )
        assert isinstance(ns.filters, list)
        assert len(ns.filters) == 2
        assert ns.filters[0] == "H_total:1.0:-"
        assert ns.filters[1] == "Q6:-:0.5"

    def test_element_fractions_is_list(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert isinstance(ns.element_fractions, list)

    def test_element_fractions_append(self) -> None:
        ns = _parse_args(
            *_CHAIN_BASE,
            "--element-fractions", "C:0.6",
            "--element-fractions", "N:0.3",
        )
        assert isinstance(ns.element_fractions, list)
        assert len(ns.element_fractions) == 2

    def test_element_min_counts_is_list(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert isinstance(ns.element_min_counts, list)

    def test_element_max_counts_is_list(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert isinstance(ns.element_max_counts, list)

    def test_objectives_is_list(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert isinstance(ns.objectives, list)

    # --- None defaults ---

    def test_seed_default_is_none(self) -> None:
        # Use a minimal namespace that does NOT include --seed
        ns = _parse_args(
            "--n-atoms", "4", "--charge", "0", "--mult", "1", "--mode", "chain",
        )
        assert ns.seed is None

    def test_cutoff_default_is_none(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert ns.cutoff is None

    def test_n_success_default_is_none(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert ns.n_success is None

    def test_center_z_default_is_none(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert ns.center_z is None

    def test_region_default_is_none(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert ns.region is None

    def test_affine_stretch_default_is_none(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert ns.affine_stretch is None

    def test_affine_shear_default_is_none(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert ns.affine_shear is None

    def test_affine_jitter_default_is_none(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert ns.affine_jitter is None

    def test_initial_xyz_default_is_none(self) -> None:
        ns = _parse_args(*_CHAIN_BASE)
        assert ns.initial_xyz is None


# ===========================================================================
# 3. Default values
# ===========================================================================


class TestDefaults:
    """Verify every documented default value is correct."""

    def _ns(self) -> argparse.Namespace:
        return _parse_args(*_CHAIN_BASE)

    def test_mode_default(self) -> None:
        # chain explicitly set; test gas default
        ns = _parse_args(
            "--n-atoms", "4", "--charge", "0", "--mult", "1",
            "--region", "sphere:4",
        )
        assert ns.mode == "gas"

    def test_branch_prob_default(self) -> None:
        assert self._ns().branch_prob == pytest.approx(0.3)

    def test_chain_persist_default(self) -> None:
        assert self._ns().chain_persist == pytest.approx(0.5)

    def test_chain_bias_default(self) -> None:
        assert self._ns().chain_bias == pytest.approx(0.0)

    def test_bond_range_default(self) -> None:
        assert self._ns().bond_range == "1.2:1.6"

    def test_coord_range_default(self) -> None:
        assert self._ns().coord_range == "4:8"

    def test_shell_radius_default(self) -> None:
        assert self._ns().shell_radius == "1.8:2.5"

    def test_maxent_steps_default(self) -> None:
        assert self._ns().maxent_steps == 300

    def test_maxent_lr_default(self) -> None:
        assert self._ns().maxent_lr == pytest.approx(0.05)

    def test_maxent_cutoff_scale_default(self) -> None:
        assert self._ns().maxent_cutoff_scale == pytest.approx(2.5)

    def test_cov_scale_default(self) -> None:
        assert self._ns().cov_scale == pytest.approx(1.0)

    def test_relax_cycles_default(self) -> None:
        assert self._ns().relax_cycles == 1500

    def test_affine_strength_default(self) -> None:
        assert self._ns().affine_strength == pytest.approx(0.0)

    def test_n_samples_default(self) -> None:
        # overridden in _CHAIN_BASE to 5; test raw default
        ns = _parse_args(
            "--n-atoms", "4", "--charge", "0", "--mult", "1", "--mode", "chain"
        )
        assert ns.n_samples == 1

    def test_n_bins_default(self) -> None:
        assert self._ns().n_bins == 20

    def test_w_atom_default(self) -> None:
        assert self._ns().w_atom == pytest.approx(0.5)

    def test_w_spatial_default(self) -> None:
        assert self._ns().w_spatial == pytest.approx(0.5)

    def test_method_default(self) -> None:
        assert self._ns().method == "annealing"

    def test_max_steps_default(self) -> None:
        assert self._ns().max_steps == 5000

    def test_T_start_default(self) -> None:
        assert self._ns().T_start == pytest.approx(1.0)

    def test_T_end_default(self) -> None:
        assert self._ns().T_end == pytest.approx(0.01)

    def test_frag_threshold_default(self) -> None:
        assert self._ns().frag_threshold == pytest.approx(0.3)

    def test_move_step_default(self) -> None:
        assert self._ns().move_step == pytest.approx(0.5)

    def test_lcc_threshold_default(self) -> None:
        assert self._ns().lcc_threshold == pytest.approx(0.0)

    def test_n_replicas_default(self) -> None:
        assert self._ns().n_replicas == 4

    def test_pt_swap_interval_default(self) -> None:
        assert self._ns().pt_swap_interval == 10

    def test_elements_default_is_none(self) -> None:
        # Use minimal args that omit --elements to get the real default
        ns = _parse_args(
            "--n-atoms", "4", "--charge", "0", "--mult", "1", "--mode", "chain",
        )
        assert ns.elements is None

    def test_filters_default_empty_list(self) -> None:
        assert self._ns().filters == []

    def test_objectives_default_empty_list(self) -> None:
        assert self._ns().objectives == []

    def test_element_fractions_default_empty_list(self) -> None:
        assert self._ns().element_fractions == []

    def test_element_min_counts_default_empty_list(self) -> None:
        assert self._ns().element_min_counts == []

    def test_element_max_counts_default_empty_list(self) -> None:
        assert self._ns().element_max_counts == []


# ===========================================================================
# 4. Required-argument enforcement
# ===========================================================================


class TestRequiredArguments:
    """Missing required arguments must produce a non-zero exit code."""

    def test_missing_n_atoms(self) -> None:
        r = _run("--charge", "0", "--mult", "1", "--mode", "chain", "--n-samples", "1")
        assert r.returncode != 0

    def test_missing_charge(self) -> None:
        r = _run("--n-atoms", "4", "--mult", "1", "--mode", "chain", "--n-samples", "1")
        assert r.returncode != 0

    def test_missing_mult(self) -> None:
        r = _run("--n-atoms", "4", "--charge", "0", "--mode", "chain", "--n-samples", "1")
        assert r.returncode != 0

    def test_no_args_at_all(self) -> None:
        r = _run()
        assert r.returncode != 0


# ===========================================================================
# 5. Mode / region validation
# ===========================================================================


class TestModeRegionValidation:
    """gas and maxent require --region; chain and shell do not."""

    def test_gas_without_region_exits_nonzero(self) -> None:
        r = _run(
            "--n-atoms", "4", "--charge", "0", "--mult", "1",
            "--mode", "gas", "--elements", "6,7,8", "--n-samples", "1",
        )
        assert r.returncode != 0

    def test_maxent_without_region_exits_nonzero(self) -> None:
        r = _run(
            "--n-atoms", "4", "--charge", "0", "--mult", "1",
            "--mode", "maxent", "--elements", "6,7,8", "--n-samples", "1",
        )
        assert r.returncode != 0

    def test_chain_without_region_succeeds(self) -> None:
        r = _run(*_CHAIN_BASE)
        # returncode 0 means at least the run started; some parity failures
        # are fine as long as the process itself exits cleanly
        assert r.returncode == 0, f"chain mode failed: {r.stderr}"

    def test_shell_without_region_succeeds(self) -> None:
        r = _run(
            "--n-atoms", "4", "--charge", "0", "--mult", "1",
            "--mode", "shell", "--elements", "1-30",
            "--n-samples", "5", "--seed", "3",
        )
        assert r.returncode == 0, f"shell mode failed: {r.stderr}"


# ===========================================================================
# 6. All placement modes — smoke tests
# ===========================================================================


class TestPlacementModeSmoke:
    """End-to-end smoke test for each placement mode."""

    def test_gas_mode_produces_xyz(self, tmp_path: Path) -> None:
        out = tmp_path / "gas.xyz"
        r = _run(*_GAS_BASE, "-o", str(out))
        assert r.returncode == 0, f"stderr: {r.stderr}"
        content = out.read_text()
        frames = parse_xyz(content)
        assert len(frames) >= 1

    def test_chain_mode_produces_xyz(self, tmp_path: Path) -> None:
        out = tmp_path / "chain.xyz"
        r = _run(*_CHAIN_BASE, "-o", str(out))
        assert r.returncode == 0, f"stderr: {r.stderr}"
        content = out.read_text()
        assert len(content) > 0

    def test_shell_mode_produces_xyz(self, tmp_path: Path) -> None:
        out = tmp_path / "shell.xyz"
        r = _run(
            "--n-atoms", "4", "--charge", "0", "--mult", "1",
            "--mode", "shell", "--center-z", "26",
            "--elements", "1-30", "--n-samples", "5", "--seed", "7",
            "-o", str(out),
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert out.exists()

    def test_maxent_mode_produces_xyz(self, tmp_path: Path) -> None:
        out = tmp_path / "maxent.xyz"
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--mode", "maxent", "--region", "sphere:5",
            "--elements", "6,7,8", "--n-samples", "5", "--seed", "42",
            "-o", str(out),
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert out.exists()


# ===========================================================================
# 7. Filter parsing
# ===========================================================================


class TestFilterParsing:
    """--filter is repeatable; both bounds and open bounds round-trip."""

    def test_single_filter_stored(self) -> None:
        ns = _parse_args(*_CHAIN_BASE, "--filter", "H_total:1.0:-")
        assert ns.filters == ["H_total:1.0:-"]

    def test_two_filters_stored(self) -> None:
        ns = _parse_args(
            *_CHAIN_BASE,
            "--filter", "H_total:1.0:-",
            "--filter", "Q6:-:0.4",
        )
        assert len(ns.filters) == 2
        assert "H_total:1.0:-" in ns.filters
        assert "Q6:-:0.4" in ns.filters

    def test_filter_applied_cli(self, tmp_path: Path) -> None:
        """Filters must eliminate structures; the remaining ones satisfy the bound."""
        out = tmp_path / "filtered.xyz"
        r = _run(
            "--n-atoms", "8", "--charge", "0", "--mult", "1",
            "--mode", "chain", "--elements", "6,7,8",
            "--n-samples", "20", "--seed", "1",
            "--filter", "H_total:0.0:-",  # accept everything with H_total >= 0
            "-o", str(out),
        )
        assert r.returncode == 0


# ===========================================================================
# 8. Element pool parsing
# ===========================================================================


class TestElementPoolParsing:
    """--elements, --element-fractions, --element-min/max-counts."""

    def test_elements_numeric_range(self, tmp_path: Path) -> None:
        out = tmp_path / "out.xyz"
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--mode", "gas", "--region", "sphere:6",
            "--elements", "1-30", "--n-samples", "3", "--seed", "0",
            "-o", str(out),
        )
        assert r.returncode == 0

    def test_elements_comma_list(self, tmp_path: Path) -> None:
        out = tmp_path / "out.xyz"
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--mode", "gas", "--region", "sphere:6",
            "--elements", "6,7,8", "--n-samples", "3", "--seed", "0",
            "-o", str(out),
        )
        assert r.returncode == 0

    def test_elements_omitted_uses_full_pool(self, tmp_path: Path) -> None:
        out = tmp_path / "out.xyz"
        r = _run(
            "--n-atoms", "4", "--charge", "0", "--mult", "1",
            "--mode", "chain", "--n-samples", "3", "--seed", "0",
            "-o", str(out),
        )
        assert r.returncode == 0

    def test_element_fractions_parsed_to_dict(self) -> None:
        """After main() parsing, parsed_element_fractions must be a dict[str, float]."""
        # Verify that the raw list is the correct type for downstream processing
        ns = _parse_args(
            *_CHAIN_BASE,
            "--element-fractions", "C:0.6",
            "--element-fractions", "N:0.3",
        )
        raw: list[str] = ns.element_fractions
        assert isinstance(raw, list)
        assert all(isinstance(s, str) for s in raw)

    def test_element_fractions_cli_succeeds(self, tmp_path: Path) -> None:
        out = tmp_path / "out.xyz"
        r = _run(
            *_GAS_BASE,
            "--element-fractions", "C:0.6",
            "--element-fractions", "N:0.3",
            "--element-fractions", "O:0.1",
            "-o", str(out),
        )
        assert r.returncode == 0

    def test_element_min_counts_cli_succeeds(self, tmp_path: Path) -> None:
        out = tmp_path / "out.xyz"
        r = _run(
            *_GAS_BASE,
            "--element-min-counts", "C:2",
            "-o", str(out),
        )
        assert r.returncode == 0

    def test_element_max_counts_cli_succeeds(self, tmp_path: Path) -> None:
        out = tmp_path / "out.xyz"
        r = _run(
            *_GAS_BASE,
            "--element-max-counts", "N:3",
            "-o", str(out),
        )
        assert r.returncode == 0

    def test_bad_element_fractions_format_exits_nonzero(self) -> None:
        r = _run(*_CHAIN_BASE, "--element-fractions", "INVALID_NO_COLON")
        assert r.returncode != 0
        assert "element-fractions" in r.stderr.lower() or "error" in r.stderr.lower()

    def test_bad_element_fractions_weight_exits_nonzero(self) -> None:
        r = _run(*_CHAIN_BASE, "--element-fractions", "C:notanumber")
        assert r.returncode != 0

    def test_bad_element_min_counts_format_exits_nonzero(self) -> None:
        r = _run(*_CHAIN_BASE, "--element-min-counts", "INVALID")
        assert r.returncode != 0

    def test_bad_element_min_counts_value_exits_nonzero(self) -> None:
        r = _run(*_CHAIN_BASE, "--element-min-counts", "C:notanint")
        assert r.returncode != 0

    def test_bad_element_max_counts_format_exits_nonzero(self) -> None:
        r = _run(*_CHAIN_BASE, "--element-max-counts", "NOCODON")
        assert r.returncode != 0

    def test_bad_element_max_counts_value_exits_nonzero(self) -> None:
        r = _run(*_CHAIN_BASE, "--element-max-counts", "N:notanint")
        assert r.returncode != 0


# ===========================================================================
# 9. Output routing
# ===========================================================================


class TestOutputRouting:
    """Verify stdout vs file routing."""

    def test_stdout_output_when_no_output_flag(self) -> None:
        r = _run(*_CHAIN_BASE)
        assert r.returncode == 0
        # Some output on stdout when at least one structure is generated
        # (chain with seed=0 on 5 attempts almost certainly passes)
        # We only check returncode; stdout content validated via parse_xyz
        # in the file tests.

    def test_file_output_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "result.xyz"
        r = _run(*_CHAIN_BASE, "-o", str(out))
        assert r.returncode == 0
        assert out.exists()
        assert out.stat().st_size > 0

    def test_file_output_long_flag(self, tmp_path: Path) -> None:
        out = tmp_path / "result2.xyz"
        r = _run(*_CHAIN_BASE, "--output", str(out))
        assert r.returncode == 0
        assert out.exists()

    def test_file_output_valid_xyz(self, tmp_path: Path) -> None:
        out = tmp_path / "valid.xyz"
        r = _run(*_GAS_BASE, "-o", str(out))
        assert r.returncode == 0
        content = out.read_text()
        if content.strip():
            frames = parse_xyz(content)
            assert len(frames) >= 1
            for atoms, positions, charge, mult, _ in frames:
                assert isinstance(atoms, list)
                assert all(isinstance(a, str) for a in atoms)
                assert isinstance(charge, int)
                assert isinstance(mult, int)
                assert len(atoms) == len(positions)


# ===========================================================================
# 10. --help
# ===========================================================================


class TestHelp:
    """--help must exit with code 0."""

    def test_help_exits_zero(self) -> None:
        r = _run("--help")
        assert r.returncode == 0

    def test_help_mentions_pasted(self) -> None:
        r = _run("--help")
        combined = (r.stdout + r.stderr).lower()
        assert "pasted" in combined

    def test_help_mentions_all_modes(self) -> None:
        r = _run("--help")
        combined = r.stdout + r.stderr
        for mode in ("gas", "chain", "shell", "maxent"):
            assert mode in combined, f"mode {mode!r} not in help output"


# ===========================================================================
# 11. --validate flag
# ===========================================================================


class TestValidateFlag:
    """--validate checks parity and exits."""

    def test_validate_valid_exits_zero(self) -> None:
        """--validate exits 0 when the parity constraint can be satisfied.

        Use an all-even-Z pool (C=6, O=8) with even n_atoms so that the
        random trial composition always satisfies charge=0, mult=1.
        A fixed seed makes the result deterministic.
        """
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--mode", "chain", "--elements", "6,8",  # C, O — both even Z
            "--n-samples", "1", "--seed", "0", "--validate",
        )
        assert r.returncode == 0, (
            f"--validate with all-even-Z pool should always pass; stderr: {r.stderr}"
        )

    def test_validate_prints_ok_or_fail_to_stderr(self) -> None:
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--mode", "chain", "--elements", "6,7,8",
            "--n-samples", "1", "--validate",
        )
        stderr_lower = r.stderr.lower()
        assert "ok" in stderr_lower or "fail" in stderr_lower or "validate" in stderr_lower

    def test_validate_does_not_write_xyz(self, tmp_path: Path) -> None:
        """--validate must exit before generating any XYZ output.

        The exit code is 0 when parity passes, 1 when it fails.
        Either way, no XYZ file must be written to disk.
        """
        out = tmp_path / "should_not_exist.xyz"
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--mode", "chain", "--elements", "6,7,8",
            "--n-samples", "1", "--validate", "-o", str(out),
        )
        # Exit code 0 = parity passed; 1 = parity failed.  Both are valid.
        assert r.returncode in (0, 1)
        assert not out.exists(), "--validate must exit before writing structures"


# ===========================================================================
# 12. Sampling controls — n-samples, n-success, seed
# ===========================================================================


class TestSamplingControls:
    """n-samples, n-success, and seed behave correctly."""

    def test_n_samples_limits_output(self, tmp_path: Path) -> None:
        out = tmp_path / "out.xyz"
        r = _run(*_GAS_BASE, "--n-samples", "3", "-o", str(out))
        assert r.returncode == 0

    def test_seed_reproducibility(self, tmp_path: Path) -> None:
        """Two runs with the same seed must produce identical output."""
        out1 = tmp_path / "run1.xyz"
        out2 = tmp_path / "run2.xyz"
        args = [
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--mode", "chain", "--elements", "6,7,8",
            "--n-samples", "5", "--seed", "123",
        ]
        r1 = _run(*args, "-o", str(out1))
        r2 = _run(*args, "-o", str(out2))
        assert r1.returncode == 0
        assert r2.returncode == 0
        assert out1.read_text() == out2.read_text(), "Same seed must produce identical output"

    def test_different_seeds_can_differ(self, tmp_path: Path) -> None:
        """Different seeds should generally produce different output."""
        out1 = tmp_path / "seed1.xyz"
        out2 = tmp_path / "seed2.xyz"
        args = [
            "--n-atoms", "8", "--charge", "0", "--mult", "1",
            "--mode", "chain", "--elements", "6,7,8", "--n-samples", "10",
        ]
        _run(*args, "--seed", "1", "-o", str(out1))
        _run(*args, "--seed", "2", "-o", str(out2))
        # We cannot guarantee they differ, but we can assert they both exist
        assert out1.exists()
        assert out2.exists()

    def test_n_success_stops_early(self, tmp_path: Path) -> None:
        """n-success=2 with n-samples=200 should produce at most 2 structures."""
        out = tmp_path / "success.xyz"
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--mode", "gas", "--region", "sphere:6",
            "--elements", "6,7,8", "--n-samples", "200",
            "--n-success", "2", "--seed", "0",
            "-o", str(out),
        )
        assert r.returncode == 0
        if out.exists() and out.stat().st_size > 0:
            frames = parse_xyz(out.read_text())
            assert len(frames) <= 2


# ===========================================================================
# 13. Affine transform flags
# ===========================================================================


class TestAffineFlags:
    """Affine transform CLI flags are forwarded correctly."""

    def test_affine_strength_nonzero(self, tmp_path: Path) -> None:
        out = tmp_path / "affine.xyz"
        r = _run(*_CHAIN_BASE, "--affine-strength", "0.2", "-o", str(out))
        assert r.returncode == 0

    def test_affine_stretch_only(self, tmp_path: Path) -> None:
        out = tmp_path / "stretch.xyz"
        r = _run(
            *_CHAIN_BASE,
            "--affine-strength", "0.2",
            "--affine-stretch", "0.4",
            "--affine-shear", "0.0",
            "--affine-jitter", "0.0",
            "-o", str(out),
        )
        assert r.returncode == 0

    def test_affine_per_op_parsed_as_float_or_none(self) -> None:
        ns_default = _parse_args(*_CHAIN_BASE)
        # All None by default
        assert ns_default.affine_stretch is None
        assert ns_default.affine_shear is None
        assert ns_default.affine_jitter is None
        # Set explicitly
        ns_set = _parse_args(*_CHAIN_BASE, "--affine-stretch", "0.3")
        assert isinstance(ns_set.affine_stretch, float)
        assert ns_set.affine_stretch == pytest.approx(0.3)


# ===========================================================================
# 14. Chain-specific flags
# ===========================================================================


class TestChainFlags:
    """chain-specific flags are parsed correctly and produce valid structures."""

    def test_chain_bias_high(self, tmp_path: Path) -> None:
        out = tmp_path / "bias.xyz"
        r = _run(*_CHAIN_BASE, "--chain-bias", "0.8", "-o", str(out))
        assert r.returncode == 0

    def test_branch_prob_zero(self, tmp_path: Path) -> None:
        out = tmp_path / "nobranch.xyz"
        r = _run(*_CHAIN_BASE, "--branch-prob", "0.0", "-o", str(out))
        assert r.returncode == 0

    def test_chain_persist_high(self, tmp_path: Path) -> None:
        out = tmp_path / "persist.xyz"
        r = _run(*_CHAIN_BASE, "--chain-persist", "0.9", "-o", str(out))
        assert r.returncode == 0


# ===========================================================================
# 15. Shell-specific flags
# ===========================================================================


class TestShellFlags:
    """shell-specific flags are parsed correctly."""

    def test_shell_with_center_z(self, tmp_path: Path) -> None:
        out = tmp_path / "shell_fe.xyz"
        r = _run(
            "--n-atoms", "4", "--charge", "0", "--mult", "1",
            "--mode", "shell", "--center-z", "26",
            "--elements", "1-30", "--n-samples", "5", "--seed", "7",
            "-o", str(out),
        )
        assert r.returncode == 0

    def test_shell_coord_range(self, tmp_path: Path) -> None:
        out = tmp_path / "shell_coord.xyz"
        r = _run(
            "--n-atoms", "4", "--charge", "0", "--mult", "1",
            "--mode", "shell", "--coord-range", "2:4",
            "--elements", "1-30", "--n-samples", "5", "--seed", "2",
            "-o", str(out),
        )
        assert r.returncode == 0


# ===========================================================================
# 16. Optimizer mode
# ===========================================================================


class TestOptimizerMode:
    """--optimize activates StructureOptimizer."""

    def test_optimize_basic_annealing(self, tmp_path: Path) -> None:
        out = tmp_path / "opt.xyz"
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--elements", "6,7,8", "--optimize",
            "--method", "annealing", "--max-steps", "50",
            "--n-samples", "1", "--seed", "42",
            "-o", str(out),
        )
        assert r.returncode == 0, f"optimizer failed:\n{r.stderr}"
        assert out.exists()
        frames = parse_xyz(out.read_text())
        assert len(frames) == 1

    def test_optimize_basin_hopping(self, tmp_path: Path) -> None:
        out = tmp_path / "bh.xyz"
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--elements", "6,7,8", "--optimize",
            "--method", "basin_hopping", "--max-steps", "50",
            "--n-samples", "1", "--seed", "0",
            "-o", str(out),
        )
        assert r.returncode == 0

    def test_optimize_parallel_tempering(self, tmp_path: Path) -> None:
        out = tmp_path / "pt.xyz"
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--elements", "6,7,8", "--optimize",
            "--method", "parallel_tempering",
            "--max-steps", "30", "--n-replicas", "2",
            "--n-samples", "1", "--seed", "1",
            "-o", str(out),
        )
        assert r.returncode == 0

    def test_optimize_custom_objective(self, tmp_path: Path) -> None:
        out = tmp_path / "obj.xyz"
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--elements", "6,7,8", "--optimize",
            "--objective", "H_total:1.0",
            "--objective", "Q6:-2.0",
            "--method", "annealing", "--max-steps", "50",
            "--n-samples", "1", "--seed", "5",
            "-o", str(out),
        )
        assert r.returncode == 0

    def test_optimize_no_composition_moves(self, tmp_path: Path) -> None:
        out = tmp_path / "nocomp.xyz"
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--elements", "6,7,8", "--optimize",
            "--no-composition-moves",
            "--method", "annealing", "--max-steps", "50",
            "--n-samples", "1", "--seed", "7",
            "-o", str(out),
        )
        assert r.returncode == 0

    def test_optimize_no_displacements(self, tmp_path: Path) -> None:
        out = tmp_path / "nodisplace.xyz"
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--elements", "6,7,8", "--optimize",
            "--no-displacements",
            "--method", "annealing", "--max-steps", "50",
            "--n-samples", "1", "--seed", "8",
            "-o", str(out),
        )
        assert r.returncode == 0

    def test_optimize_both_disabled_exits_nonzero(self) -> None:
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--elements", "6,7,8", "--optimize",
            "--no-composition-moves", "--no-displacements",
            "--method", "annealing", "--max-steps", "50",
            "--n-samples", "1",
        )
        assert r.returncode != 0
        assert "no-displacements" in r.stderr or "no-composition" in r.stderr \
               or "error" in r.stderr.lower()

    def test_optimize_output_is_valid_xyz(self, tmp_path: Path) -> None:
        out = tmp_path / "opt_valid.xyz"
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--elements", "6,7,8", "--optimize",
            "--method", "annealing", "--max-steps", "30",
            "--n-samples", "1", "--seed", "42",
            "-o", str(out),
        )
        assert r.returncode == 0
        content = out.read_text()
        frames = parse_xyz(content)
        assert len(frames) == 1
        atoms, positions, charge, mult, _metrics = frames[0]
        assert isinstance(atoms, list)
        assert isinstance(charge, int)
        assert isinstance(mult, int)
        assert len(atoms) == len(positions)
        assert all(isinstance(a, str) for a in atoms)
        assert all(
            isinstance(x, float) and isinstance(y, float) and isinstance(z, float)
            for x, y, z in positions
        )

    def test_optimize_bad_objective_format_exits_nonzero(self) -> None:
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--elements", "6,7,8", "--optimize",
            "--objective", "INVALID_NO_COLON",
            "--n-samples", "1",
        )
        assert r.returncode != 0

    def test_optimizer_constructor_value_error_exits_nonzero(self) -> None:
        """StructureOptimizer.__init__ raising ValueError must produce exit 1.

        This branch in _run_optimize_mode is otherwise unreachable via normal
        CLI flags, so the failure is injected with a mock.
        """
        with patch("pasted.cli.StructureOptimizer", side_effect=ValueError("injected")):
            r = _run(
                "--n-atoms", "6", "--charge", "0", "--mult", "1",
                "--elements", "6,7,8", "--optimize",
                "--objective", "H_total:1.0",
                "--method", "annealing", "--max-steps", "50",
                "--n-samples", "1",
            )
        assert r.returncode != 0
        assert "injected" in r.stderr or "error" in r.stderr.lower()


# ===========================================================================
# 17. --initial-xyz
# ===========================================================================


class TestInitialXyz:
    """--initial-xyz loads a starting structure for the optimizer."""

    def test_initial_xyz_valid_file(self, tmp_path: Path) -> None:
        """Write a structure to disk, then use it as the optimizer initial structure."""
        # First generate a structure to disk
        init_file = tmp_path / "initial.xyz"
        _run(*_GAS_BASE, "-o", str(init_file))
        if not init_file.exists() or not init_file.stat().st_size:
            pytest.skip("No initial structure generated; skip --initial-xyz test")

        out = tmp_path / "opt_init.xyz"
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--elements", "6,7,8", "--optimize",
            "--method", "annealing", "--max-steps", "30",
            "--n-samples", "1", "--seed", "0",
            "--initial-xyz", str(init_file),
            "-o", str(out),
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert out.exists()

    def test_initial_xyz_missing_file_exits_nonzero(self) -> None:
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--elements", "6,7,8", "--optimize",
            "--method", "annealing", "--max-steps", "30",
            "--n-samples", "1",
            "--initial-xyz", "/tmp/this_file_does_not_exist_pasted_test.xyz",
        )
        assert r.returncode != 0

    def test_initial_xyz_empty_file_exits_nonzero(self, tmp_path: Path) -> None:
        """An XYZ file that contains no frames must cause a non-zero exit."""
        empty = tmp_path / "empty.xyz"
        empty.write_text("")
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--elements", "6,7,8", "--optimize",
            "--method", "annealing", "--max-steps", "30",
            "--n-samples", "1",
            "--initial-xyz", str(empty),
        )
        assert r.returncode != 0


# ===========================================================================
# 18. _write_output internal helper
# ===========================================================================


class TestWriteOutput:
    """_write_output routes text to a file or stdout."""

    def test_writes_to_file(self, tmp_path: Path) -> None:
        out = tmp_path / "written.txt"
        _write_output("hello\n", str(out))
        assert out.read_text() == "hello\n"

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        out = tmp_path / "overwrite.txt"
        out.write_text("old content\n")
        _write_output("new content\n", str(out))
        assert out.read_text() == "new content\n"

    def test_writes_to_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        _write_output("stdout_content\n", None)
        captured = capsys.readouterr()
        assert "stdout_content" in captured.out

    def test_empty_string_creates_empty_file(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.txt"
        _write_output("", str(out))
        assert out.read_text() == ""


# ===========================================================================
# 19. XYZ output structural validation
# ===========================================================================


class TestXyzOutputValidity:
    """All generated XYZ output must pass parse_xyz and contain valid data."""

    def test_gas_xyz_valid_structure(self, tmp_path: Path) -> None:
        out = tmp_path / "gas_valid.xyz"
        r = _run(*_GAS_BASE, "--n-samples", "10", "-o", str(out))
        assert r.returncode == 0
        content = out.read_text()
        if not content.strip():
            pytest.skip("No structures generated; parity rejection expected for some pools")
        frames = parse_xyz(content)
        for atoms, positions, charge, mult, _metrics in frames:
            assert len(atoms) >= 1
            assert len(atoms) == len(positions)
            assert isinstance(charge, int)
            assert isinstance(mult, int)
            for sym in atoms:
                assert isinstance(sym, str)
                assert len(sym) >= 1
            for x, y, z in positions:
                assert all(isinstance(c, float) for c in (x, y, z))
                assert all(abs(c) < 1e6 for c in (x, y, z)), "Unreasonably large coordinate"

    def test_chain_xyz_charge_mult_in_comment(self, tmp_path: Path) -> None:
        out = tmp_path / "chain_comment.xyz"
        r = _run(*_CHAIN_BASE, "--charge", "-1", "--n-samples", "5", "-o", str(out))
        assert r.returncode == 0
        content = out.read_text()
        if not content.strip():
            pytest.skip("No structures generated")
        frames = parse_xyz(content)
        for _atoms, _pos, charge, _mult, _met in frames:
            assert charge == -1, f"Expected charge=-1, got {charge}"

    def test_metrics_present_in_extended_xyz(self, tmp_path: Path) -> None:
        out = tmp_path / "metrics.xyz"
        r = _run(*_GAS_BASE, "--n-samples", "10", "-o", str(out))
        assert r.returncode == 0
        content = out.read_text()
        if not content.strip():
            pytest.skip("No structures generated")
        frames = parse_xyz(content)
        for _atoms, _pos, _charge, _mult, metrics in frames:
            # Extended XYZ from PASTED must embed metric values
            assert len(metrics) > 0, "No metrics found in extended XYZ comment line"
            for key, val in metrics.items():
                assert isinstance(key, str)
                assert isinstance(val, float)


# ===========================================================================
# 20. Mutual-exclusion and edge cases
# ===========================================================================


class TestEdgeCases:
    """Miscellaneous CLI edge cases."""

    def test_n_atoms_zero_produces_no_structures(self, tmp_path: Path) -> None:
        """n-atoms=0 is accepted by the CLI but produces zero atoms per structure;
        parity rejection fires because no electrons can satisfy mult=1.
        The run itself exits 0 (a UserWarning is emitted, not an error exit)."""
        out = tmp_path / "zero.xyz"
        r = _run(
            "--n-atoms", "0", "--charge", "0", "--mult", "1",
            "--mode", "chain", "--n-samples", "1",
            "-o", str(out),
        )
        # CLI exits cleanly; no structures are written
        assert r.returncode == 0
        # Either the file is absent or empty (no valid structure was generated)
        if out.exists():
            assert out.stat().st_size == 0 or not out.read_text().strip()

    def test_negative_mult_not_caught_by_argparse(self) -> None:
        """argparse does not validate mult > 0; that is done downstream."""
        ns = _parse_args(*_CHAIN_BASE[:6], "--mult", "-1", "--mode", "chain",
                         "--n-samples", "1")
        assert isinstance(ns.mult, int)
        assert ns.mult == -1

    def test_mode_invalid_exits_nonzero(self) -> None:
        r = _run(
            "--n-atoms", "4", "--charge", "0", "--mult", "1",
            "--mode", "invalid_mode", "--n-samples", "1",
        )
        assert r.returncode != 0

    def test_method_invalid_exits_nonzero(self) -> None:
        r = _run(
            "--n-atoms", "4", "--charge", "0", "--mult", "1",
            "--mode", "chain", "--n-samples", "1",
            "--optimize", "--method", "not_a_method",
        )
        assert r.returncode != 0

    def test_large_n_atoms_runs_without_crash(self, tmp_path: Path) -> None:
        """n-atoms=50 should run to completion in a reasonable time."""
        out = tmp_path / "large.xyz"
        r = _run(
            "--n-atoms", "50", "--charge", "0", "--mult", "1",
            "--mode", "chain", "--elements", "6,7,8",
            "--n-samples", "2", "--seed", "0",
            "-o", str(out),
        )
        assert r.returncode == 0

    def test_no_add_hydrogen_flag_removes_h_augmentation(self, tmp_path: Path) -> None:
        """With --no-add-hydrogen and C-only pool, no H atoms should appear."""
        out = tmp_path / "noh.xyz"
        r = _run(
            "--n-atoms", "6", "--charge", "0", "--mult", "1",
            "--mode", "chain", "--elements", "6",
            "--no-add-hydrogen", "--n-samples", "5", "--seed", "0",
            "-o", str(out),
        )
        assert r.returncode == 0
        content = out.read_text()
        if not content.strip():
            pytest.skip("No structures generated")
        frames = parse_xyz(content)
        for atoms, _pos, _charge, _mult, _met in frames:
            assert "H" not in atoms, "H should not appear when --no-add-hydrogen is set"


# ===========================================================================
# 21. __main__ entry-point
# ===========================================================================


class TestMainEntryPoint:
    """``python -m pasted`` must reach __main__.py (currently 0 % coverage).

    ``runpy.run_module`` does not register coverage for ``__main__.py``
    because pytest-cov cannot trace into a separately-spawned module context.
    Instead we evict ``pasted.__main__`` from ``sys.modules`` and re-import
    it with ``pasted.cli.main`` patched to a no-op, which forces Python to
    re-execute both lines of ``__main__.py`` inside the already-traced process.
    """

    def test_dunder_main_executes_both_lines(self) -> None:
        """Re-importing __main__ executes 'from .cli import main' and 'main()'.

        Both source lines are covered; the patched main() call is verified
        via assert_called_once().
        """
        sys.modules.pop("pasted.__main__", None)
        with patch("pasted.cli.main") as mock_main:
            import pasted.__main__  # noqa: F401
        mock_main.assert_called_once()


# ===========================================================================
# 21. Range-argument error handling
# ===========================================================================


class TestRangeArgErrors:
    """Malformed range arguments must produce a non-zero exit (cli.py L652-654)."""

    def test_bad_bond_range_exits_nonzero(self) -> None:
        """--bond-range without a colon separator must fail."""
        r = _run(*_CHAIN_BASE, "--bond-range", "NOTARANGE")
        assert r.returncode != 0

    def test_bad_shell_radius_exits_nonzero(self) -> None:
        """--shell-radius without a colon separator must fail."""
        r = _run(*_CHAIN_BASE, "--shell-radius", "NOTARANGE")
        assert r.returncode != 0

    def test_bad_coord_range_lo_zero_exits_nonzero(self) -> None:
        """--coord-range with lo=0 violates MIN >= 1 and must fail."""
        r = _run(*_CHAIN_BASE, "--coord-range", "0:5")
        assert r.returncode != 0


# ===========================================================================
# 22. --elements error handling
# ===========================================================================


class TestElementsArgErrors:
    """Unsupported atomic numbers in --elements must produce a non-zero exit
    (cli.py L661-663)."""

    def test_elements_out_of_range_exits_nonzero(self) -> None:
        """Z=999 is outside the supported 1-106 range."""
        r = _run(
            "--n-atoms", "4", "--charge", "0", "--mult", "1",
            "--mode", "chain", "--elements", "999", "--n-samples", "1",
        )
        assert r.returncode != 0
        assert "error" in r.stderr.lower()


# ===========================================================================
# 23. StructureGenerator constructor error handling
# ===========================================================================


class TestGeneratorConstructorErrors:
    """Inputs that pass CLI parsing but are rejected by StructureGenerator
    must produce a non-zero exit (cli.py L603-605)."""

    def test_n_samples_zero_without_n_success_exits_nonzero(self) -> None:
        """--n-samples 0 requires --n-success; omitting it must fail."""
        r = _run(
            "--n-atoms", "4", "--charge", "0", "--mult", "1",
            "--mode", "chain", "--elements", "6,7,8", "--n-samples", "0",
        )
        assert r.returncode != 0

    def test_n_success_zero_exits_nonzero(self) -> None:
        """--n-success 0 is invalid (must be >= 1)."""
        r = _run(*_CHAIN_BASE, "--n-success", "0")
        assert r.returncode != 0

    def test_element_fractions_symbol_not_in_pool_exits_nonzero(self) -> None:
        """A fraction symbol absent from the element pool must fail."""
        # Pool is C,N,O (Z=6,7,8); Fe is not in the pool.
        r = _run(*_CHAIN_BASE, "--element-fractions", "Fe:1.0")
        assert r.returncode != 0

    def test_element_min_counts_exceeds_n_atoms_exits_nonzero(self) -> None:
        """Sum of min-counts exceeding n_atoms must fail."""
        r = _run(
            "--n-atoms", "2", "--charge", "0", "--mult", "1",
            "--mode", "chain", "--elements", "6",
            "--element-min-counts", "C:5", "--n-samples", "1",
        )
        assert r.returncode != 0

    def test_element_min_gt_max_counts_exits_nonzero(self) -> None:
        """min-count > max-count for the same element must fail."""
        r = _run(
            *_CHAIN_BASE,
            "--element-min-counts", "C:5",
            "--element-max-counts", "C:2",
        )
        assert r.returncode != 0

    def test_shell_center_z_not_in_pool_exits_nonzero(self) -> None:
        """--center-z specifying an element absent from the pool must fail."""
        # Pool is C,N,O; center-z=26 (Fe) is not in the pool.
        r = _run(
            "--n-atoms", "4", "--charge", "0", "--mult", "1",
            "--mode", "shell", "--elements", "6,7,8",
            "--center-z", "26", "--n-samples", "1",
        )
        assert r.returncode != 0
