#!/usr/bin/env python3
"""
run_pasted.py — direct-run entry point.

Allows ``python run_pasted.py`` without a prior ``pip install``, as documented
in README.md.  Adds ``src/`` to the import path when the package has not
been installed, then delegates entirely to :func:`pasted.cli.main`.

Usage (without installation)::

    python run_pasted.py --help
    python run_pasted.py --n-atoms 10 --elements 1-30 --charge 0 --mult 1 \\
        --mode gas --region sphere:8

Usage (after ``pip install -e .``)::

    pasted --help        # CLI entry-point installed by pip
    python run_pasted.py     # still works; resolves to the installed package
"""

from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_src  = _here / "src"

# ---------------------------------------------------------------------------
# Import-path fix
# ---------------------------------------------------------------------------
# Python inserts the script directory (and/or '') into sys.path[0] before
# running this file.  Both resolve to the project root, which makes
# ``import pasted`` find *this file* (pasted.py) rather than the real
# src/pasted/ package — causing infinite recursion.
#
# Fix: remove every sys.path entry that points at the project root,
#      clear the stale sys.modules entry, then add src/ so the package
#      is found correctly whether or not it has been pip-installed.
# ---------------------------------------------------------------------------

def _path_points_to_root(p: str) -> bool:
    """Return True if *p* resolves to the project root directory."""
    try:
        return Path(p).resolve() == _here
    except Exception:
        return False


# Remove '' and the project root dir from sys.path (both shadow pasted.py).
sys.path = [p for p in sys.path if not _path_points_to_root(p or ".")]

# Clear any stale reference to this script registered as the "pasted" module.
sys.modules.pop("pasted", None)

# Ensure src/ is on the path (no-op when already installed).
if _src.is_dir():
    _src_str = str(_src)
    if _src_str not in sys.path:
        sys.path.insert(0, _src_str)

# ---------------------------------------------------------------------------
# Delegate to the package CLI
# ---------------------------------------------------------------------------

try:
    from pasted.cli import main  # noqa: PLC0415
except ImportError as exc:
    print(
        f"[ERROR] Could not import pasted: {exc}\n"
        "Make sure numpy and scipy are installed:\n"
        "    pip install numpy scipy",
        file=sys.stderr,
    )
    sys.exit(1)

if __name__ == "__main__":
    main()
