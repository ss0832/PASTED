"""
setup.py — C++ extensions for pasted._ext
==========================================
Each hotspot lives in its own extension module so they can be built,
updated, and failed independently.  All project metadata is in pyproject.toml.

Usage
-----
  pip install pybind11        # build-time only
  pip install -e .            # builds all extensions; OpenMP enabled when available

OpenMP
------
OpenMP parallelization is enabled automatically when:
  1. The platform is Linux (``sys.platform == "linux"``), AND
  2. A compiler that accepts ``-fopenmp`` is available (GCC or Clang + libomp), AND
  3. The environment variable ``PASTED_DISABLE_OPENMP`` is NOT set to ``"1"``.

To opt out::

    PASTED_DISABLE_OPENMP=1 pip install -e .

When OpenMP is unavailable or disabled, all extensions fall back to
single-threaded execution transparently — no API changes, no errors.

Platform support
----------------
The C++ extensions (including OpenMP) are **supported on Linux only**.
macOS and Windows are best-effort: the extensions may compile and run, but
OpenMP will not be attempted on those platforms, and correctness on non-Linux
systems is not guaranteed or tested.

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

import os
import subprocess
import sys
import tempfile
import warnings

from setuptools import setup

try:
    from pybind11.setup_helpers import Pybind11Extension
    from pybind11.setup_helpers import build_ext as _pybind_build_ext
    _PYBIND11_AVAILABLE = True
except ImportError:
    _PYBIND11_AVAILABLE = False

_COMPILE_ARGS = ["-O3", "-std=c++17"]


def _check_openmp() -> bool:
    if sys.platform != "linux":
        return False
    if os.environ.get("PASTED_DISABLE_OPENMP", "0") == "1":
        warnings.warn("[pasted] OpenMP disabled via PASTED_DISABLE_OPENMP=1.", stacklevel=2)
        return False
    compiler = os.environ.get("CC", "gcc")
    src = "#include <omp.h>\nint main(void){return omp_get_max_threads()>0?0:1;}\n"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "p.c")
            op = os.path.join(tmp, "p")
            with open(sp, "w") as fh:
                fh.write(src)
            r = subprocess.run([compiler, "-fopenmp", sp, "-o", op],
                               capture_output=True, timeout=30, check=False)
            return r.returncode == 0
    except Exception:
        return False


_OPENMP_AVAILABLE = _check_openmp()

if _OPENMP_AVAILABLE:
    _COMPILE_ARGS += ["-fopenmp"]
    _LINK_ARGS = ["-fopenmp"]
else:
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
