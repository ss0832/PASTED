"""
setup.py — C++ extensions for pasted._ext
==========================================
Each hotspot lives in its own extension module so they can be built,
updated, and failed independently.  All project metadata is in pyproject.toml.

Usage
-----
  pip install pybind11        # build-time only
  pip install -e .            # builds all extensions in-place

Extensions
----------
  pasted._ext._relax_core      ← src/pasted/_ext/_relax.cpp
      relax_positions() — L-BFGS distance-violation repair (all modes)

  pasted._ext._maxent_core     ← src/pasted/_ext/_maxent.cpp
      angular_repulsion_gradient() — maxent gradient descent kernel

  pasted._ext._steinhardt_core ← src/pasted/_ext/_steinhardt.cpp
      steinhardt_per_atom() — sparse O(N·k) Steinhardt Q_l

  pasted._ext._graph_core      ← src/pasted/_ext/_graph_core.cpp
      graph_metrics_cpp() / moran_I_chi_cpp() — O(N·k) graph, ring,
      charge, and Moran metrics (unified cutoff adjacency)

Fallback
--------
If any extension fails to build, _ext/__init__.py sets the corresponding
HAS_* flag to False and _placement.py / _metrics.py use the pure-Python
fallback path transparently.  The remaining extensions are still attempted
and installed if they succeed.

Build failure reasons:
  - No C++17 compiler available
  - pybind11 headers not found
  - Platform-specific compiler flags unsupported

In all cases the package installs and runs correctly on pure Python/NumPy.
"""

import sys
import warnings
from setuptools import setup

# ---------------------------------------------------------------------------
# pybind11 import — optional at install time
# ---------------------------------------------------------------------------
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext as _pybind_build_ext
    _PYBIND11_AVAILABLE = True
except ImportError:
    _PYBIND11_AVAILABLE = False

_COMPILE_ARGS = ["-O3", "-std=c++17"]

# ---------------------------------------------------------------------------
# Per-extension build with individual fallback
# ---------------------------------------------------------------------------
# Each entry is (module_name, source_file).
_EXT_SPECS = [
    ("pasted._ext._relax_core",      "src/pasted/_ext/_relax.cpp"),
    ("pasted._ext._maxent_core",     "src/pasted/_ext/_maxent.cpp"),
    ("pasted._ext._steinhardt_core", "src/pasted/_ext/_steinhardt.cpp"),
    ("pasted._ext._graph_core",      "src/pasted/_ext/_graph_core.cpp"),
]


def _make_extensions():
    """Return a list of Pybind11Extension objects, skipping any that fail
    to construct (e.g. missing source files or broken pybind11 headers).
    Returns an empty list when pybind11 is not installed."""
    if not _PYBIND11_AVAILABLE:
        warnings.warn(
            "pybind11 not found — C++ acceleration extensions will not be "
            "built.  Install pybind11 and reinstall to enable them:\n"
            "  pip install pybind11 && pip install -e .",
            stacklevel=2,
        )
        return []

    exts = []
    for name, src in _EXT_SPECS:
        try:
            exts.append(
                Pybind11Extension(
                    name,
                    sources=[src],
                    extra_compile_args=_COMPILE_ARGS,
                    cxx_std=17,
                )
            )
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Could not configure extension {name!r} ({exc}). "
                "It will be skipped; the pure-Python fallback will be used.",
                stacklevel=2,
            )
    return exts


# ---------------------------------------------------------------------------
# Custom build_ext: continue on per-extension compile errors
# ---------------------------------------------------------------------------
if _PYBIND11_AVAILABLE:
    class _FallbackBuildExt(_pybind_build_ext):
        """build_ext subclass that logs individual extension failures and
        continues building the remaining extensions instead of aborting."""

        def build_extension(self, ext):
            try:
                super().build_extension(ext)
            except Exception as exc:  # pragma: no cover
                warnings.warn(
                    f"\n[pasted] Failed to build {ext.name!r}:\n  {exc}\n"
                    "  This extension will be unavailable; "
                    "the pure-Python fallback will be used automatically.\n"
                    "  To suppress this warning, install a C++17 compiler "
                    "and re-run: pip install -e .",
                    stacklevel=2,
                )

    _cmdclass = {"build_ext": _FallbackBuildExt}
else:
    _cmdclass = {}

# ---------------------------------------------------------------------------

setup(
    ext_modules=_make_extensions(),
    cmdclass=_cmdclass,
)