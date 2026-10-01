"""Runs of other shapes than the experiment, for the exporters' generic layouts.

Every case differs from Heart-Disease (10, 13, 2) in what the layouts depend on: the number of
objects, features, operators, permitted k, synthetic and latent features, the class labels, the
protocols and the baselines.
"""

from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np

from context_synthetic_recognition.config import (
    ContextConfig,
    ExperimentConfig,
    HAGConfig,
    OperatorConfig,
    load_config,
    plugin,
    preset,
)
from context_synthetic_recognition.config.models import EvaluationConfig
from context_synthetic_recognition.data import Dataset, FeatureType, load_builtin
from context_synthetic_recognition.export import RunView, build_view
from context_synthetic_recognition.services.runner import run_experiment

from .conftest import CONFIGS


def evaluation(*protocols: Any, baselines: tuple[Any, ...] = ()) -> EvaluationConfig:
    """The template's evaluation with other protocols and baselines."""
    return preset("template").evaluation.model_copy(
        update={"protocols": protocols, "baselines": baselines}
    )


def numeric_view() -> RunView:
    """24 objects, 4 quantitative features, text labels, article switches, three protocols.

    No nominal feature (no ρ_J, no *Nominal* sheet), ϰ = 3 (two HAG iterations at most), a typed
    new object that is not a training object, a hold-out protocol and two kinds of baselines.
    """
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, 4))
    X[:9] += 0.9
    dataset = Dataset(
        X=X,
        y=np.array(["healthy"] * 9 + ["sick"] * 15),
        feature_types=(FeatureType.QUANTITATIVE,) * 4,
        name="numeric-24",
        feature_names=("age", "pressure", "sugar", "pulse"),
    )
    config = preset("article").model_copy(
        update={
            "evaluation": evaluation(
                plugin("resubstitution"),
                plugin("leave-one-out"),
                plugin("hold-out"),
                baselines=(plugin("knn-vote", ks=[1, 3]), plugin("naive-bayes")),
            ).model_copy(update={"positive_class": "sick"}),
            "hag": HAGConfig(kappa=3, centres="final", step4_passes=1),  # type: ignore[arg-type]
        }
    )
    return build_view(run_experiment(dataset, config), new_object=[0.1, 0.2, 0.3, 0.4])


def nominal_view() -> RunView:
    """12 objects, 3 nominal features, one operator, one k: r = 1, so p = 0 (no latent feature)."""
    rng = np.random.default_rng(7)
    dataset = Dataset(
        X=rng.integers(0, 3, size=(12, 3)).astype(float),
        y=np.array([1] * 5 + [2] * 7),
        feature_types=(FeatureType.NOMINAL,) * 3,
        name="nominal-12",
    )
    config = ExperimentConfig(
        context=ContextConfig(operators=(OperatorConfig(label="ρ_J", features="nominal"),)),
        k=plugin("explicit", values=[3]),
        evaluation=evaluation(
            plugin("resubstitution"), plugin("repeated-k-fold", folds=3, repeats=2)
        ),
    )
    return build_view(run_experiment(dataset, config))


def tiny_view() -> RunView:
    """6 objects with |K1| = 2: the only permitted k is 1."""
    rng = np.random.default_rng(7)
    dataset = Dataset(
        X=rng.normal(size=(6, 2)),
        y=np.array([1, 1, 2, 2, 2, 2]),
        feature_types=(FeatureType.QUANTITATIVE,) * 2,
        name="tiny-6",
    )
    config = ExperimentConfig(
        evaluation=evaluation(plugin("resubstitution"), plugin("leave-one-out"))
    )
    return build_view(run_experiment(dataset, config))


def heart270_view() -> RunView:
    """Heart-Disease (270, 13, 2) with the large-data configuration and a stratified 5-fold."""
    large = load_config(CONFIGS / "heart-disease-270-large-data.yaml")
    config = large.model_copy(
        update={
            "evaluation": evaluation(
                plugin("resubstitution"),
                plugin("stratified-k-fold", folds=5),
                baselines=(plugin("knn-vote"),),
            )
        }
    )
    return build_view(run_experiment(load_builtin("heart-disease-270"), config))


def literal_view() -> RunView:
    """Heart-Disease (270, 13, 2) with the literal k rule: 118 permitted k, r = 354."""
    config = ExperimentConfig(evaluation=evaluation(plugin("resubstitution")))
    return build_view(run_experiment(load_builtin("heart-disease-270"), config))


SHAPES: dict[str, Callable[[], RunView]] = {
    "numeric": numeric_view,
    "nominal": nominal_view,
    "tiny": tiny_view,
    "heart270": heart270_view,
    "literal": literal_view,
}
SMALL = ("numeric", "nominal", "tiny")
"""The shapes that are quick to export in every format."""


@cache
def shape(name: str) -> RunView:
    """The run of a shape (computed once per test session)."""
    return SHAPES[name]()
