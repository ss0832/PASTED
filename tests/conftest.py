"""Shared pytest fixtures for the PASTED test suite."""

from __future__ import annotations

import pytest

from pasted import StructureGenerator


@pytest.fixture
def gas_gen() -> StructureGenerator:
    """A small gas-mode generator with a fixed seed."""
    return StructureGenerator(
        n_atoms=6,
        charge=0,
        mult=1,
        mode="gas",
        region="sphere:6",
        elements="6,7,8",
        n_samples=5,
        seed=0,
    )


@pytest.fixture
def chain_gen() -> StructureGenerator:
    """A small chain-mode generator with a fixed seed."""
    return StructureGenerator(
        n_atoms=8,
        charge=0,
        mult=1,
        mode="chain",
        elements="6,7,8",
        n_samples=5,
        seed=1,
    )


@pytest.fixture
def shell_gen() -> StructureGenerator:
    """A small shell-mode generator (Fe center) with a fixed seed."""
    return StructureGenerator(
        n_atoms=8,
        charge=0,
        mult=1,
        mode="shell",
        elements="6,7,8,26",
        center_z=26,
        n_samples=5,
        seed=2,
    )
