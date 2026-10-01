"""Shared paths and fixtures."""

import sys
from pathlib import Path
from types import ModuleType

import pytest

from context_synthetic_recognition.core.context import ContextModel, fit_context
from context_synthetic_recognition.core.model import CSModel, fit_model
from context_synthetic_recognition.data import Dataset, load_builtin, load_dataset
from context_synthetic_recognition.export import RunView, build_view
from context_synthetic_recognition.services.runner import ExperimentResult, run_experiment
from context_synthetic_recognition.services.sensitivity import switch_sensitivity

EXPERIMENT_RUN_ID = "20260930_120000_0a1b2c3d"
"""Run id of the :func:`experiment_view` fixture."""
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
TEMPLATES = ROOT / "tests" / "data" / "templates"
HAG_TEMPLATE = (
    TEMPLATES / "RegularizedStackingEnsembleWithHAG [Heart-Disease (10, 13, 2)] - Opus 5.5.xlsx"
)
"""Copy of the HAG template workbook (ADR-029): SET {x₃, x₆, x₁₃, x₄, x₉}, r₁ … r₄."""
META_TEMPLATE = TEMPLATES / "Meta-algorithm [Heart-Disease (10, 13, 2)] - Opus 5.5.xlsx"
"""Copy of the meta-algorithm template workbook (ADR-029): the new object is Class 2."""
TEMPLATE_PROJECT = ROOT.parent / "hag-regularized-stacking-boosting-meta"
"""The author's earlier project (read-only): present on the author's machine, absent in CI."""


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
def experiment_cs_model(experiment: Dataset) -> CSModel:
    """The whole CS-model fitted on the experiment with the default (template) configuration."""
    return fit_model(experiment)


@pytest.fixture(scope="session")
def heart270() -> Dataset:
    return load_builtin("heart-disease-270")


@pytest.fixture(scope="session")
def experiment_result(experiment: Dataset) -> ExperimentResult:
    """The experiment evaluated with the default (template) configuration."""
    return run_experiment(experiment)


@pytest.fixture(scope="session")
def experiment_view(experiment: Dataset, experiment_result: ExperimentResult) -> RunView:
    """What the exporters show of the experiment, with the four switch settings."""
    return build_view(
        experiment_result, sensitivity=switch_sensitivity(experiment), run_id=EXPERIMENT_RUN_ID
    )


def reference_engine() -> ModuleType:
    """The reference engine's ``csmodel.core`` (dependency-free test oracle, read-only)."""
    if str(REFERENCE_ENGINE) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ENGINE))
    from csmodel import core

    return core


def reference_pipeline() -> ModuleType:
    """The reference engine's ``csmodel.pipeline``: fit (Steps 1–11), represent, predict."""
    reference_engine()
    from csmodel import pipeline

    return pipeline
