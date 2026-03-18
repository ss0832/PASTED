"""
setup.py — C++ extensions for pasted._ext
==========================================
Each hotspot lives in its own extension module so they can be built,
updated, and failed independently.  All project metadata is in pyproject.toml.

Usage
-----
  pip install pybind11        # build-time only
  pip install -e .            # builds both extensions in-place

Extensions
----------
  pasted._ext._relax_core   ← src/pasted/_ext/_relax.cpp
      relax_positions() — distance-violation repair loop (all modes)

  pasted._ext._maxent_core  ← src/pasted/_ext/_maxent.cpp
      angular_repulsion_gradient() — maxent gradient descent kernel

Fallback
--------
If either extension fails to build, _ext/__init__.py sets the corresponding
HAS_* flag to False and _placement.py uses the pure-Python/NumPy path.
"""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

_COMPILE_ARGS = ["-O3", "-std=c++17"]

ext_modules = [
    Pybind11Extension(
        "pasted._ext._relax_core",
        sources=["src/pasted/_ext/_relax.cpp"],
        extra_compile_args=_COMPILE_ARGS,
        cxx_std=17,
    ),
    Pybind11Extension(
        "pasted._ext._maxent_core",
        sources=["src/pasted/_ext/_maxent.cpp"],
        extra_compile_args=_COMPILE_ARGS,
        cxx_std=17,
    ),
    Pybind11Extension(
        "pasted._ext._steinhardt_core",
        sources=["src/pasted/_ext/_steinhardt.cpp"],
        extra_compile_args=_COMPILE_ARGS,
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
