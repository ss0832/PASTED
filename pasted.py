#!/usr/bin/env python3
"""
pasted.py — direct-run entry point.

Allows ``python pasted.py`` without a prior ``pip install``, as documented
in README.md.  Adds ``src/`` to the import path when the package has not
been installed, then delegates entirely to :func:`pasted.cli.main`.

Usage (without installation)::

    python pasted.py --help
    python pasted.py --n-atoms 10 --elements 1-30 --charge 0 --mult 1 \\
        --mode gas --region sphere:8

Usage (after ``pip install -e .``)::

    pasted --help        # CLI entry-point installed by pip
    python pasted.py     # still works; resolves to the installed package
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import-path fix
# ---------------------------------------------------------------------------
# This script is named ``pasted.py``.  Python automatically inserts the
# script's directory into sys.path[0], which makes ``import pasted`` resolve
# to *this file* instead of the real ``src/pasted/`` package.
#
# Fix: temporarily remove the project root from sys.path, add src/ instead,
# clear any stale ``pasted`` entry from sys.modules, then import normally.
# ---------------------------------------------------------------------------

_here = str(Path(__file__).resolve().parent)
_src  = str(Path(__file__).resolve().parent / "src")

# 1. Clear any stale reference to this script masquerading as the package.
sys.modules.pop("pasted", None)

# 2. Remove the project root so pasted.py is no longer findable as "pasted".
if _here in sys.path:
    sys.path.remove(_here)

# 3. Add src/ (only if it exists — bare checkout) so the real package is found.
if Path(_src).is_dir() and _src not in sys.path:
    sys.path.insert(0, _src)

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
