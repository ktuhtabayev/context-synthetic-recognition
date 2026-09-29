"""Shared paths and fixtures."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
EXPERIMENTS = ROOT / "resources" / "experiments" / "context-synthetic-model"


@pytest.fixture
def root() -> Path:
    return ROOT
