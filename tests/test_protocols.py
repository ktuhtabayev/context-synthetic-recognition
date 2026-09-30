"""Evaluation protocols and baselines: folds, re-fitting, undefined folds, cancellation."""

import importlib
import json
from typing import Any

import numpy as np
import pytest

from context_synthetic_recognition.config import ExperimentConfig, plugin, preset
from context_synthetic_recognition.data import Dataset, FeatureType
from context_synthetic_recognition.errors import ConfigError, ModelUndefinedError
from context_synthetic_recognition.evaluation import baselines as baselines_module
from context_synthetic_recognition.evaluation.baselines import (
    BASELINES,
    BaselineInput,
    encode_features,
    knn_vote,
)
from context_synthetic_recognition.evaluation.protocols import (
    PROTOCOLS,
    EvaluationCancelledError,
    run_protocol,
)

from .conftest import GOLDEN_VALUES

Y = np.array([0] * 6 + [1] * 9)


def test_the_registry() -> None:
    assert PROTOCOLS.names() == [
        "resubstitution",
        "leave-one-out",
        "stratified-k-fold",
        "repeated-k-fold",
        "hold-out",
    ]
    assert PROTOCOLS.info("loo").name == "leave-one-out"


def test_leave_one_out_splits() -> None:
    splits = PROTOCOLS.get("leave-one-out")(Y, 0, None)
    assert len(splits) == Y.size
    for i, split in enumerate(splits):
        assert split.test.tolist() == [i]
        assert split.train.tolist() == [j for j in range(Y.size) if j != i]


def test_stratified_folds_are_balanced_and_seeded() -> None:
    params = PROTOCOLS.make_params("stratified-k-fold", {"folds": 3})
    splits = PROTOCOLS.get("stratified-k-fold")(Y, 7, params)
    tested = np.sort(np.concatenate([s.test for s in splits]))
    assert tested.tolist() == list(range(Y.size))  # every object held out exactly once
    assert [int((Y[s.test] == 0).sum()) for s in splits] == [2, 2, 2]
    assert [int((Y[s.test] == 1).sum()) for s in splits] == [3, 3, 3]
    again = PROTOCOLS.get("stratified-k-fold")(Y, 7, params)
    assert all(np.array_equal(a.test, b.test) for a, b in zip(splits, again, strict=True))
    other = PROTOCOLS.get("stratified-k-fold")(Y, 8, params)
    assert any(not np.array_equal(a.test, b.test) for a, b in zip(splits, other, strict=True))
    unshuffled = PROTOCOLS.make_params("stratified-k-fold", {"folds": 3, "shuffle": False})
    assert PROTOCOLS.get("stratified-k-fold")(Y, 0, unshuffled)[0].test.tolist() == [0, 3, 6, 9, 12]
    with pytest.raises(ModelUndefinedError, match="16 folds"):
        PROTOCOLS.get("stratified-k-fold")(
            Y, 0, PROTOCOLS.make_params("stratified-k-fold", {"folds": 16})
        )


def test_repeated_and_hold_out_splits() -> None:
    params = PROTOCOLS.make_params("repeated-k-fold", {"folds": 3, "repeats": 2})
    splits = PROTOCOLS.get("repeated-k-fold")(Y, 0, params)
    assert [s.repeat for s in splits] == [0, 0, 0, 1, 1, 1]
    hold = PROTOCOLS.get("hold-out")(Y, 0, PROTOCOLS.make_params("hold-out", {"test_size": 0.3}))
    assert len(hold) == 1
    assert int((Y[hold[0].test] == 0).sum()) == 2  # round(0.3 · 6)
    assert int((Y[hold[0].test] == 1).sum()) == 3  # round(0.3 · 9)
    assert np.intersect1d(hold[0].train, hold[0].test).size == 0
    with pytest.raises(ConfigError):
        PROTOCOLS.make_params("hold-out", {"test_size": 1.5})
    resub = PROTOCOLS.get("resubstitution")(Y, 0, None)
    assert resub[0].resubstitution
    assert resub[0].test.size == Y.size


# ---------------------------------------------------------------- running a protocol


def test_leave_one_out_of_the_experiment(experiment: Dataset) -> None:
    with GOLDEN_VALUES.open(encoding="utf-8") as handle:
        golden = json.load(handle)
    result = run_protocol(experiment, None, "leave-one-out", baselines=[plugin("knn-vote")])
    assert result.protocol == "leave-one-out"
    assert result.predictions.by_object().tolist() == [f["cls"] for f in golden["loo"]]
    assert [f.permitted_k for f in result.folds] == [tuple(f["ks"]) for f in golden["loo"]]
    assert result.folds[0].tuplam_names == ("ρ·3", "ρ_J·5", "ρ_I·5", "ρ·5", "ρ_J·3")
    assert result.undefined_folds == ()
    keys = ["Z3", "Z5", "I3", "I5", "J3", "J5"]
    assert [b.method for b in result.baselines] == [
        "k-NN vote ρ, k = 3",
        "k-NN vote ρ, k = 5",
        "k-NN vote ρ_I, k = 3",
        "k-NN vote ρ_I, k = 5",
        "k-NN vote ρ_J, k = 3",
        "k-NN vote ρ_J, k = 5",
    ]
    for baseline, key in zip(result.baselines, keys, strict=True):
        assert baseline.by_object().tolist() == golden["base"][key]
    assert result.predictions.roc().auc == pytest.approx(0.2708333333333333)
    assert result.methods[0] is result.predictions


def test_resubstitution_equals_classify_training(experiment: Dataset) -> None:
    result = run_protocol(experiment, None, "resubstitution")
    assert result.predictions.decisions.tolist() == [1, 2, 2, 1, 1, 2, 2, 2, 2, 2]
    assert result.folds[0].descriptions is not None


def test_repeated_k_fold_metrics(experiment: Dataset) -> None:
    spec = plugin("repeated-k-fold", folds=2, repeats=3)
    result = run_protocol(experiment, None, spec, baselines=[plugin("knn-vote", ks=[3])])
    assert result.predictions.objects.size == 3 * experiment.m
    assert len(result.predictions.repeat_metrics()) == 3
    assert len(result.baselines) == 3  # three operators, one k


def _small_classes() -> Dataset:
    # |K1| = 2: holding out a K1 object leaves one — no permitted k (ADR-005)
    X = np.array([[0.0], [0.1], [0.9], [1.0], [0.8], [0.7], [0.95]])
    return Dataset(
        X=X, y=np.array([1, 1, 2, 2, 2, 2, 2]), feature_types=(FeatureType.QUANTITATIVE,)
    )


def test_undefined_folds_refuse_and_are_flagged() -> None:
    config = ExperimentConfig(
        context=preset("template").context.model_copy(
            update={"operators": preset("template").context.operators[:1]}
        )
    )
    result = run_protocol(
        _small_classes(), config, "leave-one-out", baselines=[plugin("knn-vote", ks=[1])]
    )
    undefined = result.undefined_folds
    assert [f.number for f in undefined] == [0, 1]
    assert "no k is permitted" in (undefined[0].undefined or "")
    assert result.predictions.decisions[:2].tolist() == [0, 0]
    assert undefined[0].descriptions is None
    assert result.baselines[0].decisions.size == 7


def test_a_single_class_training_part_refuses_the_baselines() -> None:
    X = np.array([[0.0], [0.1], [0.2], [0.9]])
    dataset = Dataset(X=X, y=np.array([1, 1, 1, 2]), feature_types=(FeatureType.QUANTITATIVE,))
    result = run_protocol(dataset, None, "leave-one-out", baselines=[plugin("knn-vote", ks=[1])])
    assert result.predictions.decisions[3] == 0  # the fold without K2
    assert result.baselines[0].decisions[3] == 0


def test_progress_and_cancellation(experiment: Dataset) -> None:
    seen: list[tuple[int, int]] = []
    run_protocol(
        experiment, None, "leave-one-out", progress=lambda done, total: seen.append((done, total))
    )
    assert seen[-1] == (10, 10)
    calls = iter([False, False, True])
    with pytest.raises(EvaluationCancelledError, match="after 2 of 10"):
        run_protocol(experiment, None, "leave-one-out", cancelled=lambda: next(calls))


# ---------------------------------------------------------------- baselines


def test_knn_vote_parameters_and_refusal(experiment: Dataset) -> None:
    with pytest.raises(ConfigError, match="odd"):
        BASELINES.make_params("knn-vote", {"ks": [4]})
    params = BASELINES.make_params("knn-vote", {"ks": [3, 11], "operators": ["ρ_I"]})
    data = BaselineInput(experiment, experiment.X[:2], None, ExperimentConfig())
    outputs = knn_vote(data, params)
    assert [o.method for o in outputs] == ["k-NN vote ρ_I, k = 3", "k-NN vote ρ_I, k = 11"]
    assert outputs[1].decisions.tolist() == [0, 0]  # 11 > the 10 training objects
    bad = BASELINES.make_params("knn-vote", {"operators": ["ρ_X"]})
    with pytest.raises(ConfigError, match="unknown operators"):
        knn_vote(data, bad)


@pytest.mark.parametrize(
    "name", ["logistic-regression", "random-forest", "svm", "decision-tree", "naive-bayes"]
)
def test_sklearn_baselines(experiment: Dataset, name: str) -> None:
    pytest.importorskip("sklearn")
    result = run_protocol(
        experiment, ExperimentConfig(seed=3), "leave-one-out", baselines=[plugin(name)]
    )
    (baseline,) = result.baselines
    assert baseline.method.endswith("(scikit-learn)")
    assert set(np.unique(baseline.decisions)) <= {1, 2}
    assert np.isfinite(baseline.scores).all()


def test_sklearn_options_and_missing_package(
    experiment: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("sklearn")
    spec = plugin("logistic-regression", options={"C": 0.5})
    assert run_protocol(experiment, None, "resubstitution", baselines=[spec]).baselines
    real = importlib.import_module

    def missing(name: str, *args: Any) -> Any:
        if name.startswith("sklearn"):
            raise ImportError(name)
        return real(name, *args)

    monkeypatch.setattr(baselines_module.importlib, "import_module", missing)
    with pytest.raises(ConfigError, match=r"\[sklearn\]"):
        run_protocol(experiment, None, "resubstitution", baselines=[plugin("svm")])


def test_one_hot_encoding(experiment: Dataset) -> None:
    from context_synthetic_recognition.core.normalizers import minmax

    scaling = minmax(experiment.X, experiment.quantitative)
    train, target = encode_features(experiment, experiment.X[:1], scaling)
    assert train.shape[0] == experiment.m
    nominal = int(
        sum(np.unique(experiment.X[:, j]).size for j in np.flatnonzero(experiment.nominal))
    )
    assert train.shape[1] == int(experiment.quantitative.sum()) + nominal
    assert np.array_equal(target[0], train[0])
