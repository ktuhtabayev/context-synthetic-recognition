"""Shared paths and fixtures."""

import sys
from pathlib import Path
from types import ModuleType

import pytest

from context_synthetic_recognition.core.context import ContextModel, fit_context
from context_synthetic_recognition.data import Dataset, load_builtin, load_dataset

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
EXPERIMENTS = ROOT / "resources" / "experiments" / "context-synthetic-model"
WORKBOOK = EXPERIMENTS / (
    "Context-Synthetic Model – Full Experiment "
    "[My Experiment on Heart-Disease (10, 13, 2)] - Opus 5.5.xlsx"
)
"""The full Excel experiment: the specification and its golden values."""
KNN_WORKBOOK = (
    EXPERIMENTS / "Forming Synthetic Features based on k-NN (Heart-Disease 10,13,2) - Opus 5.5.xlsx"
)
REFERENCE_ENGINE = ROOT / "docs" / "handoff" / "reference-engine"
GOLDEN_VALUES = REFERENCE_ENGINE / "golden_values.json"


@pytest.fixture
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def experiment() -> Dataset:
    """Heart-Disease (10, 13, 2) read from the workbook's Dataset sheet."""
    return load_dataset(WORKBOOK)


@pytest.fixture(scope="session")
def experiment_model(experiment: Dataset) -> ContextModel:
    """Steps 1–8 fitted on the experiment with the default (template) configuration."""
    return fit_context(experiment)


@pytest.fixture(scope="session")
def heart270() -> Dataset:
    return load_builtin("heart-disease-270")


def reference_engine() -> ModuleType:
    """The reference engine's ``csmodel.core`` (dependency-free test oracle, read-only)."""
    if str(REFERENCE_ENGINE) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ENGINE))
    from csmodel import core

    return core
