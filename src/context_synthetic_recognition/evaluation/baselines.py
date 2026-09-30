"""Baselines to compare the CS-model with, evaluated under the same protocol and folds.

Registry :data:`BASELINES`:

``knn-vote`` (default)
    Plain k-nearest-neighbour majority vote for every base operator and k: the object gets the
    class of the majority of its k nearest training objects, with the fold's scale unification,
    the same operators, rounding and tie rule (sheet *Leave-One-Out*, columns M–R). The k are
    3 and 5 by default, independent of the fold's permitted k (ADR-009).
``logistic-regression``, ``random-forest``, ``svm``, ``decision-tree``, ``naive-bayes``
    scikit-learn classifiers (optional extra ``[sklearn]``) on the quantitative features scaled by
    the fold's normalizer and the nominal features one-hot encoded (categories of the training
    part; an unseen category encodes as all zeros). ``options`` are passed to the estimator; a
    ``random_state`` is set from the experiment seed when the estimator has one.

Every baseline returns decisions (1 = K1, 2 = K2, 0 = refusal) and a score that is larger for K1
(for the AUC): (χ₁ − χ₂)/k for the vote, P(K1) − P(K2) or the decision function otherwise.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from pydantic import Field, JsonValue, field_validator
from pydantic.dataclasses import dataclass as params_dataclass

from context_synthetic_recognition.config.models import ExperimentConfig
from context_synthetic_recognition.core.arrays import FloatArray, IntArray
from context_synthetic_recognition.core.encoders import k1_counts
from context_synthetic_recognition.core.meta import K1_DECISION, K2_DECISION, REFUSAL
from context_synthetic_recognition.core.neighbours import neighbour_order
from context_synthetic_recognition.core.normalizers import NORMALIZERS, Scaling
from context_synthetic_recognition.core.operators import resolve_operators
from context_synthetic_recognition.core.registry import PARAMS_CONFIG, Registry
from context_synthetic_recognition.data.schema import Dataset
from context_synthetic_recognition.errors import ConfigError


@dataclass(frozen=True)
class BaselineInput:
    """What a baseline gets for one fold."""

    train: Dataset
    """The fold's training part (two classes)."""
    X: FloatArray
    """(q × n) original values of the objects to classify."""
    exclude: IntArray | None
    """Per object, a training index to leave out (resubstitution), or ``None``."""
    config: ExperimentConfig
    """The experiment configuration (normalizer, operators, rounding, seed)."""


@dataclass(frozen=True, eq=False)
class BaselineOutput:
    """Decisions and scores of one baseline method on one fold."""

    method: str
    """Display name, e.g. ``k-NN vote ρ_I, k = 3``."""
    decisions: IntArray
    """(q,) 1, 2 or 0."""
    scores: FloatArray = field(repr=False)
    """(q,) larger for K1."""


class Baseline(Protocol):
    """A comparison method: fitted on the fold's training part, applied to its objects."""

    def __call__(self, data: BaselineInput, params: Any, /) -> list[BaselineOutput]:
        """Return one output per method variant (e.g. per operator and k)."""
        ...


BASELINES: Registry[Baseline] = Registry("baselines")
"""Baselines by name."""


def _scaling(data: BaselineInput) -> Scaling:
    spec = data.config.preprocessing.normalizer
    return NORMALIZERS.get(spec.name)(
        data.train.X, data.train.quantitative, NORMALIZERS.make_params(spec.name, spec.params)
    )


# ---------------------------------------------------------------- k-NN vote


@params_dataclass(frozen=True, config=PARAMS_CONFIG)
class KnnVoteParams:
    """Parameters of the ``knn-vote`` baseline."""

    ks: tuple[int, ...] = (3, 5)
    """Neighbourhood sizes (odd)."""
    operators: tuple[str, ...] | None = None
    """Labels of the base operators to use; ``None``: every configured operator."""

    @field_validator("ks")
    @classmethod
    def _odd(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if not values or any(k < 1 or k % 2 == 0 for k in values):
            raise ValueError("give at least one k; every k must be odd and positive")
        return values


@BASELINES.register(
    "knn-vote",
    aliases=("knn",),
    summary="k-NN majority vote per operator and k (default: k = 3, 5)",
    params_type=KnnVoteParams,
)
def knn_vote(data: BaselineInput, params: KnnVoteParams | None = None, /) -> list[BaselineOutput]:
    """Majority class of the k nearest training objects, for every operator and k.

    A k larger than the number of available neighbours gives a refusal.
    """
    chosen = params or KnnVoteParams()
    train = data.train
    scaling = _scaling(data)
    reference = scaling.transform(train.X)
    targets = scaling.transform(np.asarray(data.X, dtype=np.float64))
    operators, _ = resolve_operators(data.config.context, train.feature_names, train.quantitative)
    if chosen.operators is not None:
        unknown = sorted(set(chosen.operators) - {o.label for o in operators})
        if unknown:
            raise ConfigError(f"knn-vote: unknown operators {', '.join(unknown)}")
        operators = tuple(o for o in operators if o.label in chosen.operators)
    y = train.class_index
    outputs = []
    for operator in operators:
        D = operator.distances(targets, reference, decimals=data.config.context.distance_decimals)
        order = neighbour_order(D, data.exclude)
        usable = [k for k in chosen.ks if k <= order.shape[1]]
        chi1 = k1_counts(order, y, usable) if usable else np.empty((targets.shape[0], 0))
        for k in chosen.ks:
            name = f"k-NN vote {operator.label}, k = {k}"
            if k not in usable:
                outputs.append(
                    BaselineOutput(
                        name, np.zeros(targets.shape[0], dtype=np.int64), np.zeros(targets.shape[0])
                    )
                )
                continue
            c = chi1[:, usable.index(k)]
            decisions = np.where(c > k // 2, K1_DECISION, K2_DECISION).astype(np.int64)
            outputs.append(BaselineOutput(name, decisions, (2 * c - k) / k))
    return outputs


# ---------------------------------------------------------------- scikit-learn


@params_dataclass(frozen=True, config=PARAMS_CONFIG)
class SklearnParams:
    """Parameters of the scikit-learn baselines."""

    options: dict[str, JsonValue] = Field(default_factory=dict)
    """Keyword arguments of the estimator (e.g. ``C``, ``n_estimators``)."""


@dataclass(frozen=True)
class _Estimator:
    module: str
    name: str
    title: str
    defaults: dict[str, Any] = field(default_factory=dict)


_ESTIMATORS = {
    "logistic-regression": _Estimator(
        "sklearn.linear_model", "LogisticRegression", "logistic regression", {"max_iter": 1000}
    ),
    "random-forest": _Estimator(
        "sklearn.ensemble", "RandomForestClassifier", "random forest", {"n_estimators": 200}
    ),
    "svm": _Estimator("sklearn.svm", "SVC", "SVM (RBF)"),
    "decision-tree": _Estimator("sklearn.tree", "DecisionTreeClassifier", "decision tree"),
    "naive-bayes": _Estimator("sklearn.naive_bayes", "GaussianNB", "naive Bayes (Gaussian)"),
}


def encode_features(
    train: Dataset, X: npt.ArrayLike, scaling: Scaling
) -> tuple[FloatArray, FloatArray]:
    """Quantitative features scaled, nominal features one-hot (categories of the training part)."""
    target = np.asarray(X, dtype=np.float64)
    parts_train: list[FloatArray] = []
    parts_target: list[FloatArray] = []
    quantitative = train.quantitative
    if quantitative.any():
        parts_train.append(scaling.transform(train.X)[:, quantitative])
        parts_target.append(scaling.transform(target)[:, quantitative])
    for j in np.flatnonzero(~quantitative):
        categories = np.unique(train.X[:, j])
        parts_train.append((train.X[:, j, None] == categories[None, :]).astype(np.float64))
        parts_target.append((target[:, j, None] == categories[None, :]).astype(np.float64))
    return np.hstack(parts_train), np.hstack(parts_target)


def _sklearn_baseline(key: str) -> Callable[[BaselineInput, Any], list[BaselineOutput]]:
    spec = _ESTIMATORS[key]

    def run(data: BaselineInput, params: SklearnParams | None = None, /) -> list[BaselineOutput]:
        try:
            module = importlib.import_module(spec.module)
        except ImportError as error:
            raise ConfigError(
                f"baseline '{key}' needs scikit-learn: pip install "
                "'context-synthetic-recognition[sklearn]'"
            ) from error
        estimator_type = getattr(module, spec.name)
        options = {**spec.defaults, **(params.options if params else {})}
        if "random_state" in estimator_type().get_params() and "random_state" not in options:
            options["random_state"] = data.config.seed
        estimator = estimator_type(**options)
        train_x, target_x = encode_features(data.train, data.X, _scaling(data))
        codes = data.train.class_index + 1
        estimator.fit(train_x, codes)
        decisions = np.asarray(estimator.predict(target_x), dtype=np.int64)
        if hasattr(estimator, "predict_proba"):
            probability = estimator.predict_proba(target_x)
            column = {int(c): i for i, c in enumerate(estimator.classes_)}
            scores = probability[:, column[K1_DECISION]] - probability[:, column[K2_DECISION]]
        else:
            scores = -np.asarray(estimator.decision_function(target_x), dtype=np.float64)
        return [BaselineOutput(f"{spec.title} (scikit-learn)", decisions, np.asarray(scores))]

    run.__doc__ = f"The scikit-learn {spec.title} ({spec.module}.{spec.name})."
    return run


for _key, _spec in _ESTIMATORS.items():
    BASELINES.add(
        _key,
        _sklearn_baseline(_key),
        summary=f"scikit-learn {_spec.title} (optional extra [sklearn])",
        params_type=SklearnParams,
    )


def refusals(q: int) -> tuple[IntArray, FloatArray]:
    """Decisions and scores of a method that refuses ``q`` objects."""
    return np.full(q, REFUSAL, dtype=np.int64), np.zeros(q)
