"""
tests/test_ext_init.py
======================
Coverage for the ``except ImportError`` fallback branches in
``pasted._ext.__init__``.

The C++ extensions (_relax_core, _maxent_core, _steinhardt_core, _graph_core)
are compiled in the CI environment, so their ``try:`` import blocks always
succeed and the ``except ImportError:`` branches — which set ``HAS_* = False``
and ``name = None`` — are never reached during normal test runs.

Strategy
--------
Each test:
1.  Removes ``pasted._ext`` (and any cached sub-extension) from
    ``sys.modules`` so that the next import triggers a fresh evaluation.
2.  Injects ``None`` for the target extension module(s) via
    ``unittest.mock.patch.dict``, which causes Python to raise
    ``ImportError`` when that name is imported.
3.  Imports ``pasted._ext`` inside the patch context and asserts the
    expected ``HAS_*`` flags and ``None`` sentinels.
4.  Restores ``sys.modules`` automatically when the ``patch.dict`` context
    exits (the ``clear=False`` default keeps all other entries intact).

Notes
-----
* ``patch.dict`` with a ``None`` value makes Python raise ``ImportError``
  for that module name — this is the documented behaviour.
* We must also remove any cached ``pasted._ext.*`` entries so the reload
  uses our injected values rather than the already-imported ``.so`` objects.
* After each test, ``pasted._ext`` is re-imported normally (outside any
  patch) to leave the process in a consistent state for subsequent tests.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXT_SUBMODULES = [
    "pasted._ext._relax_core",
    "pasted._ext._maxent_core",
    "pasted._ext._steinhardt_core",
    "pasted._ext._graph_core",
    "pasted._ext._bond_angle_core",  # added in v0.4.0
    "pasted._ext._combined_core",  # added in v0.4.0 (step 2)
]


def _reload_ext(blocked: dict[str, None]) -> types.ModuleType:
    """Remove ``pasted._ext`` and the named sub-modules from ``sys.modules``,
    inject ``None`` sentinels for *blocked* entries, then re-import the
    package and return the fresh module object.
    """
    # Remove pasted._ext and all sub-extension entries from the module cache.
    for key in list(sys.modules):
        if key == "pasted._ext" or key in _EXT_SUBMODULES:
            del sys.modules[key]

    with patch.dict(sys.modules, blocked):
        ext = importlib.import_module("pasted._ext")

    return ext


def _restore_ext() -> None:
    """Re-import pasted._ext cleanly so subsequent tests see real flags."""
    for key in list(sys.modules):
        if key == "pasted._ext" or key in _EXT_SUBMODULES:
            del sys.modules[key]
    importlib.import_module("pasted._ext")


# ---------------------------------------------------------------------------
# Tests — one extension blocked at a time
# ---------------------------------------------------------------------------


class TestExtInitFallbacks:
    """Each test blocks exactly one C++ extension and verifies the fallback."""

    def test_relax_core_missing_sets_has_relax_false(self) -> None:
        """ImportError on _relax_core must set HAS_RELAX=False and relax_positions=None."""
        ext = _reload_ext({"pasted._ext._relax_core": None})
        try:
            assert ext.HAS_RELAX is False
            assert ext.relax_positions is None
        finally:
            _restore_ext()

    def test_maxent_core_missing_sets_has_maxent_false(self) -> None:
        """ImportError on _maxent_core must set HAS_MAXENT=False, HAS_MAXENT_LOOP=False."""
        ext = _reload_ext({"pasted._ext._maxent_core": None})
        try:
            assert ext.HAS_MAXENT is False
            assert ext.HAS_MAXENT_LOOP is False
            assert ext.angular_repulsion_gradient is None
            assert ext.place_maxent_cpp is None
        finally:
            _restore_ext()

    def test_steinhardt_core_missing_sets_has_steinhardt_false(self) -> None:
        """ImportError on _steinhardt_core must set HAS_STEINHARDT=False."""
        ext = _reload_ext({"pasted._ext._steinhardt_core": None})
        try:
            assert ext.HAS_STEINHARDT is False
            assert ext.steinhardt_per_atom is None
        finally:
            _restore_ext()

    def test_graph_core_missing_sets_has_graph_false(self) -> None:
        """ImportError on _graph_core must set HAS_GRAPH=False."""
        ext = _reload_ext({"pasted._ext._graph_core": None})
        try:
            assert ext.HAS_GRAPH is False
            assert ext.graph_metrics_cpp is None
            assert ext.moran_I_chi_cpp is None
            assert ext.rdf_h_cpp is None
        finally:
            _restore_ext()

    def test_all_extensions_missing(self) -> None:
        """All four extensions missing: every HAS_* flag must be False."""
        blocked = {name: None for name in _EXT_SUBMODULES}
        ext = _reload_ext(blocked)
        try:
            assert ext.HAS_RELAX is False
            assert ext.HAS_MAXENT is False
            assert ext.HAS_MAXENT_LOOP is False
            assert ext.HAS_STEINHARDT is False
            assert ext.HAS_GRAPH is False
            assert ext.HAS_BA_CPP is False  # added in v0.4.0
            assert ext.HAS_COMBINED is False  # added in v0.4.0
        finally:
            _restore_ext()

    def test_partial_build_relax_only(self) -> None:
        """Only _relax_core available: HAS_RELAX=True, all others False."""
        blocked = {name: None for name in _EXT_SUBMODULES if "relax" not in name}
        ext = _reload_ext(blocked)
        try:
            assert ext.HAS_RELAX is True
            assert ext.HAS_MAXENT is False
            assert ext.HAS_STEINHARDT is False
            assert ext.HAS_GRAPH is False
        finally:
            _restore_ext()

    def test_partial_build_relax_only_extended(self) -> None:
        """Only _relax_core available: new HAS_BA_CPP and HAS_COMBINED must also be False."""
        blocked = {name: None for name in _EXT_SUBMODULES if "relax" not in name}
        ext = _reload_ext(blocked)
        try:
            assert ext.HAS_BA_CPP is False
            assert ext.HAS_COMBINED is False
        finally:
            _restore_ext()

    def test_bond_angle_core_missing_sets_has_ba_cpp_false(self) -> None:
        """ImportError on _bond_angle_core must set HAS_BA_CPP=False."""
        ext = _reload_ext({"pasted._ext._bond_angle_core": None})
        try:
            assert ext.HAS_BA_CPP is False
            assert ext.bond_angle_entropy_cpp is None
        finally:
            _restore_ext()

    def test_combined_core_missing_sets_has_combined_false(self) -> None:
        """ImportError on _combined_core must set HAS_COMBINED=False."""
        ext = _reload_ext({"pasted._ext._combined_core": None})
        try:
            assert ext.HAS_COMBINED is False
            assert ext.all_metrics_cpp is None
        finally:
            _restore_ext()

    def test_normal_import_all_flags_true(self) -> None:
        """Sanity check: with all extensions present every HAS_* flag is True."""
        _restore_ext()
        import pasted._ext as ext

        assert ext.HAS_RELAX is True
        assert ext.HAS_MAXENT is True
        assert ext.HAS_MAXENT_LOOP is True
        assert ext.HAS_STEINHARDT is True
        assert ext.HAS_GRAPH is True
        # v0.4.0 extensions: only check attribute existence because CI may
        # not have rebuilt the binaries yet.
        assert hasattr(ext, "HAS_BA_CPP")
        assert hasattr(ext, "HAS_COMBINED")
