"""The CS-model end to end: fit, the meta-dataset Y, the Theorem (no label), resubstitution."""

import dataclasses
import inspect

import numpy as np
import pytest

from context_synthetic_recognition.config import (
    ContextConfig,
    ExperimentConfig,
    HAGConfig,
    MetaConfig,
    OperatorConfig,
    plugin,
    preset,
)
from context_synthetic_recognition.core.meta import REFUSAL
from context_synthetic_recognition.core.model import CSModel, fit_model
from context_synthetic_recognition.data import Dataset, FeatureType
from context_synthetic_recognition.errors import ModelUndefinedError, RegistryError

RESUBSTITUTION = [1, 2, 2, 1, 1, 2, 2, 2, 2, 2]


def test_the_experiment_model(experiment_cs_model: CSModel) -> None:
    model = experiment_cs_model
    assert model.tuplam == (5, 2, 0, 1, 3)
    assert model.p == 4
    assert model.classes == (1, 2)
    Y = model.meta_dataset
    assert Y.p == 4
    assert Y.Y.shape == (10, 9)
    assert Y.columns == (
        "y₀ (a₆)",
        "y₁ (a₃)",
        "y₂ (a₁)",
        "y₃ (a₂)",
        "y₄ (a₄)",
        "r₁",
        "r₂",
        "r₃",
        "r₄",
    )
    assert np.array_equal(Y.initial, model.trace.contributions[:, [5, 2, 0, 1, 3]])
    assert np.array_equal(Y.latent, model.hag.latent)
    assert np.array_equal(model.description.gradations, model.trace.values[:, [5, 2, 0, 1, 3]])
    assert model.description.gradations[0].tolist() == [2, 1, 2, 2, 2]  # Brace, row S₁


def test_resubstitution(experiment_cs_model: CSModel) -> None:
    training = experiment_cs_model.classify_training()
    assert training.decisions.tolist() == RESUBSTITUTION
    assert training.labels == tuple(RESUBSTITUTION)
    assert training.refusals == 0
    assert training.representation is None
    assert training.meta.scores1.tolist() == [0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    assert training.scores[1] == pytest.approx(0.5 - 4 / 6)


def test_the_new_object_path_reproduces_the_training_rows(
    experiment: Dataset, experiment_cs_model: CSModel
) -> None:
    left_out = experiment_cs_model.classify(experiment.X, exclude=np.arange(experiment.m))
    training = experiment_cs_model.classify_training()
    assert left_out.representation is not None
    assert np.array_equal(
        left_out.representation.description, experiment_cs_model.description.gradations
    )
    assert np.array_equal(left_out.decisions, training.decisions)
    assert np.array_equal(left_out.meta.b1_sizes, training.meta.b1_sizes)
    assert experiment_cs_model.predict(experiment.X[0], exclude=[0]).tolist() == [1]


def test_no_label_argument_anywhere_on_the_new_object_path() -> None:
    for method in (CSModel.represent, CSModel.classify, CSModel.predict):
        assert list(inspect.signature(method).parameters) == ["self", "X", "exclude"]


def test_the_class_of_a_left_out_object_is_never_read(
    experiment: Dataset, experiment_cs_model: CSModel
) -> None:
    # poison the class of Sᵢ in the context model: its own description and decision must not move
    for i in range(experiment.m):
        classes = experiment_cs_model.context.reference_classes.copy()
        classes[i] = 1 - classes[i]
        poisoned = dataclasses.replace(
            experiment_cs_model,
            context=dataclasses.replace(experiment_cs_model.context, reference_classes=classes),
        )
        honest = experiment_cs_model.classify(experiment.X[i], exclude=[i])
        attacked = poisoned.classify(experiment.X[i], exclude=[i])
        assert honest.representation is not None
        assert attacked.representation is not None
        assert np.array_equal(
            honest.representation.description, attacked.representation.description
        )
        assert np.array_equal(honest.decisions, attacked.decisions)


def test_a_new_object(experiment_cs_model: CSModel) -> None:
    new = np.array([58.0, 1, 3, 125, 250, 0, 2, 150, 0, 1.0, 2, 1, 7])
    result = experiment_cs_model.classify(new)
    assert result.representation is not None
    assert result.representation.description.tolist() == [[2, 2, 2, 2, 2]]
    assert result.decisions.tolist() == [2]
    assert result.labels == (2,)


def test_both_presets(experiment: Dataset) -> None:
    article = fit_model(experiment, preset("article"))
    assert article.tuplam == (5, 0, 1, 3, 4)
    decisions = article.classify_training().decisions
    assert decisions.tolist() == [1, 1, 2, 1, 1, 1, 1, 1, 1, 1]  # 5 of 10 correct


def test_text_labels_and_refusals() -> None:
    X = np.array([[0.0, 1], [0.1, 1], [0.2, 2], [0.9, 2], [1.0, 1], [0.8, 2]])
    dataset = Dataset(
        X=X,
        y=np.array(["absent", "absent", "absent", "present", "present", "present"]),
        feature_types=(FeatureType.QUANTITATIVE, FeatureType.NOMINAL),
    )
    model = fit_model(dataset)
    assert model.classes == ("absent", "present")
    result = model.classify(X, exclude=np.arange(6))
    for decision, label in zip(result.decisions, result.labels, strict=True):
        if decision == REFUSAL:
            assert label is None
        else:
            assert label == model.classes[decision - 1]


def test_a_single_synthetic_feature() -> None:
    # one operator and one permitted k: r = 1, TUPLAM = {a₁}, p = 0
    X = np.array([[0.0], [0.1], [0.2], [0.9], [1.0], [0.8], [0.15]])
    dataset = Dataset(
        X=X, y=np.array([1, 1, 1, 2, 2, 2, 1]), feature_types=(FeatureType.QUANTITATIVE,)
    )
    config = ExperimentConfig(
        context=ContextConfig(operators=(OperatorConfig(label="ρ"),)),
        k=plugin("explicit", values=[3]),
    )
    model = fit_model(dataset, config)
    assert model.tuplam == (0,)
    assert model.p == 0
    assert model.meta_dataset.Y.shape == (7, 1)
    assert model.classify_training().decisions.tolist() == [1, 1, 1, 2, 2, 2, 1]


def test_bad_plug_ins_fail_before_steps_1_to_8(experiment: Dataset) -> None:
    with pytest.raises(RegistryError, match="majorizers"):
        fit_model(experiment, ExperimentConfig(hag=HAGConfig(majorizer=plugin("relu"))))
    with pytest.raises(RegistryError, match="decision rules"):
        fit_model(experiment, ExperimentConfig(meta=MetaConfig(decision_rule=plugin("vote"))))


def test_undefined_models_are_reported() -> None:
    dataset = Dataset(
        X=np.array([[0.0], [1.0], [2.0]]),
        y=np.array([1, 2, 2]),
        feature_types=(FeatureType.QUANTITATIVE,),
    )
    with pytest.raises(ModelUndefinedError, match="no k is permitted"):
        fit_model(dataset)
