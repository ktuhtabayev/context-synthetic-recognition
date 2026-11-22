"""Shared paths and fixtures."""

import os
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import BaseModel

from context_synthetic_recognition.config import (
    AdditionalFeaturesConfig,
    ClassMeans,
    ExperimentConfig,
    preset,
)
from context_synthetic_recognition.core.model import CSModel, fit_model
from context_synthetic_recognition.core.synthetic_features.psi import PsiOperator, fit_psi
from context_synthetic_recognition.data import Dataset, load_builtin, load_dataset
from context_synthetic_recognition.export import RunView, build_view
from context_synthetic_recognition.services.runner import ExperimentResult, run_experiment
from context_synthetic_recognition.services.sensitivity import switch_sensitivity

EXPERIMENT_RUN_ID = "20260930_120000_0a1b2c3d"
"""Run id of the :func:`experiment_view` fixture."""
# GUI tests draw into memory: no window appears and no display is needed (set before Qt loads)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
WORKBOOKS = ROOT / "tests" / "data" / "workbooks"
"""Copies of the author's workbooks the tests read (byte for byte)."""
WORKBOOK = WORKBOOKS / "Excel experiment [Heart-Disease (10, 13, 2)].xlsx"
"""The Excel experiment: its cell values are acceptance tests (with its own settings)."""
KNN_WORKBOOK = WORKBOOKS / "Synthetic features by k-NN [Heart-Disease (10, 13, 2)].xlsx"
ADDITIONAL_FEATURES_WORKBOOK = WORKBOOKS / "Additional features [Heart-Disease (10, 13, 2)].xlsx"
"""The additional-features check workbook (ADR-029): SET {x₃, x₆, x₁₃, x₄, x₉}, d₁ … d₄."""
META_ALGORITHM_WORKBOOK = WORKBOOKS / "Meta-algorithm [Heart-Disease (10, 13, 2)].xlsx"
"""The meta-algorithm check workbook (ADR-029): the new object is Class 2."""
GOLDEN_VALUES = ROOT / "tests" / "golden" / "golden_values.json"
"""Values of the independent re-implementation on Heart-Disease (10, 13, 2)."""


def excel_experiment() -> ExperimentConfig:
    """The settings of the Excel experiment workbook (they reproduce its cell values)."""
    return preset("excel-experiment")


def _plain(value: object) -> object:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def workbook_config(**sections: object) -> ExperimentConfig:
    """The workbook's settings with some sections replaced (``local_context=…`` and so on)."""
    data = excel_experiment().model_dump(mode="json")
    data.update({key: _plain(value) for key, value in sections.items()})
    return ExperimentConfig.model_validate(data)


def workbook_additional(**fields: object) -> AdditionalFeaturesConfig:
    """The workbook's settings of the agglomerative algorithm with some fields replaced."""
    data = excel_experiment().additional_features.model_dump(mode="json")
    data.update({key: _plain(value) for key, value in fields.items()})
    return AdditionalFeaturesConfig.model_validate(data)


def article_switches() -> ExperimentConfig:
    """The workbook's settings with the article's two switches: final class means, one pass."""
    base = excel_experiment()
    additional = base.additional_features.model_copy(
        update={"class_means": ClassMeans.FINAL, "majorizing_passes": 1}
    )
    return base.model_copy(update={"additional_features": additional})


@pytest.fixture
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def experiment() -> Dataset:
    """Heart-Disease (10, 13, 2) read from the workbook's Dataset sheet."""
    return load_dataset(WORKBOOK)


@pytest.fixture(scope="session")
def experiment_psi(experiment: Dataset) -> PsiOperator:
    """Steps 1–8 fitted on the experiment with the workbook's settings."""
    return fit_psi(experiment, excel_experiment())


@pytest.fixture(scope="session")
def experiment_cs_model(experiment: Dataset) -> CSModel:
    """The whole CS-model fitted on the experiment with the workbook's settings."""
    return fit_model(experiment, excel_experiment())


@pytest.fixture(scope="session")
def heart270() -> Dataset:
    return load_builtin("heart-disease-270")


@pytest.fixture(scope="session")
def experiment_result(experiment: Dataset) -> ExperimentResult:
    """The experiment evaluated with the workbook's settings."""
    return run_experiment(experiment, excel_experiment())


@pytest.fixture(scope="session")
def experiment_view(experiment: Dataset, experiment_result: ExperimentResult) -> RunView:
    """What the exporters show of the experiment, with the four switch settings."""
    return build_view(
        experiment_result,
        sensitivity=switch_sensitivity(experiment, excel_experiment()),
        run_id=EXPERIMENT_RUN_ID,
    )


def independent_core() -> ModuleType:
    """The independent re-implementation's ``core`` (dependency-free test oracle)."""
    from .golden.independent import core

    return core


def independent_pipeline() -> ModuleType:
    """The independent re-implementation's ``pipeline``: fit (Steps 1–11), represent, predict."""
    from .golden.independent import pipeline

    return pipeline
