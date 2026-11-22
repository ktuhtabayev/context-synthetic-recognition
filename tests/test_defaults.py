"""The defaults follow the article (ADR-062, ADR-063): the numbers before and after the change.

*Before* is the Excel experiment's setting (preset ``excel-experiment``: odd k = 3 … 2·min|Kᵢ| − 3,
running class means, two majorizing passes); *after* is the default configuration (odd k = 3 …
min|Kᵢ|, final class means, one pass). Every number comes from a run of the package.
"""

import pytest

from context_synthetic_recognition.config import ExperimentConfig, PresetName, load_config, preset
from context_synthetic_recognition.core.model import CSModel, fit_model
from context_synthetic_recognition.data import Dataset
from context_synthetic_recognition.evaluation.protocols import run_protocol

from .conftest import CONFIGS


def tuplam(model: CSModel) -> list[str]:
    """TUPLAM as operator·k."""
    features = model.trace.features
    return [f"{features[u].operator_label}·{features[u].k}" for u in model.tuplam]


def accuracy(dataset: Dataset, config: ExperimentConfig, protocol: str) -> tuple[float, int]:
    """Accuracy (refusals counted as errors) and refusals under ``protocol``."""
    metrics = run_protocol(dataset, config, protocol).predictions.metrics()
    return round(metrics.accuracy, 4), metrics.refusals


def test_the_experiment_before_the_change(experiment: Dataset) -> None:
    config = preset(PresetName.EXCEL_EXPERIMENT)
    model = fit_model(experiment, config)
    assert model.psi.ks == (3, 5)
    assert model.trace.r == 6
    assert tuplam(model) == ["ρ_J·5", "ρ_I·3", "ρ·3", "ρ·5", "ρ_I·5"]
    assert accuracy(experiment, config, "resubstitution") == (0.7, 0)
    assert accuracy(experiment, config, "leave-one-out") == (0.2, 4)


def test_the_experiment_after_the_change(experiment: Dataset) -> None:
    config = ExperimentConfig()
    model = fit_model(experiment, config)
    assert model.psi.ks == (3,)
    assert model.trace.r == 3
    assert tuplam(model) == ["ρ·3", "ρ_J·3", "ρ_I·3"]
    assert accuracy(experiment, config, "resubstitution") == (0.5, 0)
    assert accuracy(experiment, config, "leave-one-out") == (0.0, 5)


@pytest.mark.slow
def test_heart_disease_270_before_and_after_the_change(heart270: Dataset) -> None:
    before = preset(PresetName.EXCEL_EXPERIMENT)
    model = fit_model(heart270, before)
    assert (model.psi.ks[0], model.psi.ks[-1], len(model.psi.ks)) == (3, 237, 118)
    assert model.trace.r == 354
    assert accuracy(heart270, before, "resubstitution")[0] == 0.4481
    assert accuracy(heart270, before, "leave-one-out") == (0.0704, 8)

    after = ExperimentConfig()
    model = fit_model(heart270, after)
    assert (model.psi.ks[0], model.psi.ks[-1], len(model.psi.ks)) == (3, 119, 59)
    assert model.trace.r == 177
    assert tuplam(model) == ["ρ_I·119", "ρ·93", "ρ·31", "ρ_I·3", "ρ_J·5"]
    assert accuracy(heart270, after, "resubstitution") == (0.8444, 4)
    assert accuracy(heart270, after, "leave-one-out") == (0.8074, 10)


@pytest.mark.slow
def test_the_large_data_configuration_on_the_article_settings(heart270: Dataset) -> None:
    # ADR-076: skip_constant on the article's settings; no synthetic feature of Heart-Disease
    # (270, 13, 2) is constant there, so the model and its results are the default's
    large = load_config(CONFIGS / "heart-disease-270-large-data.yaml")
    model = fit_model(heart270, large)
    default = fit_model(heart270, ExperimentConfig())
    assert model.trace.r == default.trace.r == 177
    assert tuplam(model) == tuplam(default)
    assert accuracy(heart270, large, "leave-one-out") == (0.8074, 10)
