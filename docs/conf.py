# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import importlib.metadata

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project   = "PASTED"
author    = "PASTED contributors"
copyright = "2026, PASTED contributors"  # noqa: A001

try:
    release = importlib.metadata.version("pasted")
except importlib.metadata.PackageNotFoundError:
    release = "unknown"

version = ".".join(release.split(".")[:2])

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # NumPy / Google docstring styles
    "sphinx.ext.viewcode",       # [source] links in API pages
    "sphinx.ext.intersphinx",    # cross-links to NumPy, Python docs
    "myst_parser",               # Markdown support (.md pages)
]

templates_path  = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# Autodoc
# ---------------------------------------------------------------------------

autodoc_default_options = {
    "members":          True,
    "undoc-members":    False,
    "show-inheritance": True,
    "special-members":  "__iter__, __len__, __repr__",
}

autosummary_generate = False
napoleon_numpy_docstring  = True
napoleon_google_docstring = False

suppress_warnings = [
    "ref.duplicate",
    "intersphinx.fetch_inventory",   # network unavailable in some CI envs
]

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
}

# ---------------------------------------------------------------------------
# HTML output — sphinx-book-theme
# ---------------------------------------------------------------------------

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url":        "https://github.com/YOUR_USERNAME/pasted",
    "use_repository_button": True,
    "use_issues_button":     True,
    "use_download_button":   True,
    "show_navbar_depth":     2,
    "navigation_with_keys":  True,
    "logo": {
        "text": "PASTED",
        "alt":  "PASTED",
    },
}

html_title       = f"PASTED {version}"
html_static_path = ["_static"]

# ---------------------------------------------------------------------------
# MyST (Markdown)
# ---------------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",    # ::: directive syntax
    "deflist",        # definition lists
    "dollarmath",     # $...$ inline math
]
