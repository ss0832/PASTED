"""
setup.py — C++ extensions for pasted._ext
==========================================
Each hotspot lives in its own extension module so they can be built,
updated, and failed independently.  All project metadata is in pyproject.toml.

Usage
-----
  pip install pybind11        # build-time only
  pip install -e .            # builds all extensions

Platform support
----------------
The C++ extensions are **supported on Linux only**.
macOS and Windows are best-effort: the extensions may compile and run, but
correctness on non-Linux systems is not guaranteed or tested.

Extensions
----------
  pasted._ext._relax_core      <- src/pasted/_ext/_relax.cpp
  pasted._ext._maxent_core     <- src/pasted/_ext/_maxent.cpp
  pasted._ext._steinhardt_core <- src/pasted/_ext/_steinhardt.cpp
  pasted._ext._graph_core      <- src/pasted/_ext/_graph_core.cpp

Fallback
--------
If any extension fails to build, _ext/__init__.py sets the corresponding
HAS_* flag to False and the pure-Python fallback is used transparently.
"""

import warnings

from setuptools import setup

try:
    from pybind11.setup_helpers import Pybind11Extension
    from pybind11.setup_helpers import build_ext as _pybind_build_ext
    _PYBIND11_AVAILABLE = True
except ImportError:
    _PYBIND11_AVAILABLE = False

_COMPILE_ARGS = ["-O3", "-std=c++17"]
_LINK_ARGS: list[str] = []

_EXT_SPECS = [
    ("pasted._ext._relax_core",      "src/pasted/_ext/_relax.cpp"),
    ("pasted._ext._maxent_core",     "src/pasted/_ext/_maxent.cpp"),
    ("pasted._ext._steinhardt_core", "src/pasted/_ext/_steinhardt.cpp"),
    ("pasted._ext._graph_core",      "src/pasted/_ext/_graph_core.cpp"),
]


def _make_extensions():
    if not _PYBIND11_AVAILABLE:
        warnings.warn(
            "pybind11 not found — C++ acceleration extensions will not be built. "
            "Install pybind11 and reinstall: pip install pybind11 && pip install -e .",
            stacklevel=2,
        )
        return []
    exts = []
    for name, src in _EXT_SPECS:
        try:
            exts.append(Pybind11Extension(
                name, sources=[src],
                extra_compile_args=_COMPILE_ARGS,
                extra_link_args=_LINK_ARGS,
                cxx_std=17,
            ))
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Could not configure extension {name!r} ({exc}). "
                "It will be skipped; the pure-Python fallback will be used.",
                stacklevel=2,
            )
    return exts


if _PYBIND11_AVAILABLE:
    class _FallbackBuildExt(_pybind_build_ext):
        def build_extension(self, ext):
            try:
                super().build_extension(ext)
            except Exception as exc:  # pragma: no cover
                warnings.warn(
                    f"\n[pasted] Failed to build {ext.name!r}:\n  {exc}\n"
                    "  This extension will be unavailable; "
                    "the pure-Python fallback will be used automatically.\n"
                    "  To suppress: install a C++17 compiler and re-run pip install -e .",
                    stacklevel=2,
                )
    _cmdclass = {"build_ext": _FallbackBuildExt}
else:
    _cmdclass = {}

setup(ext_modules=_make_extensions(), cmdclass=_cmdclass)
