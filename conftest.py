"""
Root conftest.py — import-path fix for src layout.

``pasted.py`` in the project root shadows ``src/pasted/`` when pytest adds
the rootdir to sys.path.  This file runs before any test collection and
ensures the real package is importable.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Remove any stale ``pasted`` entry that points at pasted.py.
sys.modules.pop("pasted", None)

# Guarantee src/ is on the path and comes before the project root.
_src = str(Path(__file__).parent / "src")
_root = str(Path(__file__).parent)

if _root in sys.path:
    sys.path.remove(_root)
if _src not in sys.path:
    sys.path.insert(0, _src)
# Put root back at the end so other non-pasted imports still work.
if _root not in sys.path:
    sys.path.append(_root)
