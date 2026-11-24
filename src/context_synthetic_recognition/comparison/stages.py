"""The comparison in three stages: training, test and evaluation (ADR-084, ADR-085).

- :func:`fit_split` — *Training*: every method is fitted on the training sample, and the training
  objects are classified (resubstitution as in Definition 2: a k-NN training object is not its
  own neighbour, a CS-model training object is left out of its own context).
- :func:`classify_split` — *Test*: every fitted method classifies the test objects from an
  :class:`~context_synthetic_recognition.data.samples.UnlabelledSample`, which has no class.
- :func:`evaluate_split` — *Evaluation*: the decisions are compared with the true classes of an
  :class:`~context_synthetic_recognition.data.samples.AnswerKey`, which only this stage reads.

The k-NN vote keeps the training sample as its model and the CS-model keeps its fitted
:class:`~context_synthetic_recognition.core.model.CSModel`. SVM and random forest live in the
project's baselines, which fit and predict in one call: their fitted state is the training sample
with the fixed options, and the *Test* stage fits them again — deterministically, so the decisions
are those of a single fit (ADR-085). :func:`compare_split` is the three stages in a row.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from context_synthetic_recognition.comparison.cs_model import (
    ALPHAS,
    CSModelResult,
    cs_model_config,
    cs_model_metrics,
)
from context_synthetic_recognition.comparison.methods import (
    MethodResult,
    SplitResult,
    comparison_config,
    knn_key,
)
from context_synthetic_recognition.comparison.settings import (
    KNN_KS,
    KNN_METRICS,
    METHOD_TITLES,
    METRIC_TITLES,
    SKLEARN_OPTIONS,
    TABLE_7,
    TABLE_8,
    TABLE_CS_MODEL,
    SplitSpec,
)
from context_synthetic_recognition.config import ExperimentConfig
from context_synthetic_recognition.core.arrays import IntArray, readonly
from context_synthetic_recognition.core.model import CSModel, fit_model
from context_synthetic_recognition.data.samples import (
    AnswerKey,
    Decisions,
    UnlabelledSample,
    require_unlabelled,
)
from context_synthetic_recognition.data.schema import Dataset, FeatureType
from context_synthetic_recognition.data.splits import FixedSplit
from context_synthetic_recognition.errors import ModelUndefinedError
from context_synthetic_recognition.evaluation.baselines import (
    BASELINES,
    BaselineInput,
    KnnVoteParams,
    SklearnParams,
    knn_vote,
)
from context_synthetic_recognition.evaluation.metrics import (
    ClassificationMetrics,
    classification_metrics,
)

NOT_PRINTED = "Not printed"
"""Group of the k-NN rows that Tables 7 and 8 do not print (the Zhuravlyov metric)."""


@dataclass(frozen=True)
class MethodSpec:
    """A method of the comparison: what it is, the features it uses, where it is printed."""

    key: str
    """``svm``, ``random-forest``, ``knn-<metric>-<k>`` or ``cs-model-<metric>-<α>``."""
    title: str
    """``SVM``, ``k-NN Euclidean, k = 3``, ``CS-model Euclidean, α = 0.1`` and so on."""
    method: str
    """``svm``, ``random-forest``, ``knn`` or ``cs-model``."""
    group: str
    """``Table 7``, ``Table 8``, ``Not printed`` or ``CS-model``."""
    features: str
    """``all`` or ``quantitative``."""
    n_features: int = 0
    """Number of original features the method uses (0: not known yet)."""
    metric: str | None = None
    k: int | None = None
    alpha: float | None = None

    @property
    def table(self) -> str | None:
        """``Table 7`` or ``Table 8`` for the rows the tables print, otherwise ``None``."""
        return self.group if self.group in (TABLE_7, TABLE_8) else None


def method_specs(spec: SplitSpec, n: int = 0, n_quantitative: int = 0) -> tuple[MethodSpec, ...]:
    """The methods of a split, in the order of the results: SVM, random forest, k-NN, CS-model.

    Args:
        spec: The split.
        n: Its number of features (0 if not known).
        n_quantitative: Its number of quantitative features.
    """
    found = [MethodSpec(key, METHOD_TITLES[key], key, TABLE_7, "all", n) for key in SKLEARN_OPTIONS]
    for metric in (*KNN_METRICS, *spec.extra_knn):
        printed = metric in KNN_METRICS
        found += [
            MethodSpec(
                knn_key(metric, k),
                f"k-NN {METRIC_TITLES[metric]}, k = {k}",
                "knn",
                TABLE_8 if printed else NOT_PRINTED,
                "quantitative" if printed else "all",
                n_quantitative if printed else n,
                metric=metric,
                k=k,
            )
            for k in KNN_KS
        ]
    for metric in cs_model_metrics(spec):
        every = metric in spec.extra_knn
        found += [
            MethodSpec(
                f"cs-model-{metric}-{alpha}",
                f"CS-model {METRIC_TITLES[metric]}, α = {alpha}",
                "cs-model",
                TABLE_CS_MODEL,
                "all" if every else "quantitative",
                n if every else n_quantitative,
                metric=metric,
                alpha=alpha,
            )
            for alpha in ALPHAS
        ]
    return tuple(found)


# ---------------------------------------------------------------- fitted methods


class FittedMethod(ABC):
    """A method fitted on a training sample, ready to classify objects without their class."""

    spec: MethodSpec
    training_decisions: IntArray | None
    """(training objects,) resubstitution decisions; ``None`` if the method is undefined."""
    undefined: str
    """Why the method is undefined on this training sample (empty if it is defined)."""

    def classify(self, unlabelled: UnlabelledSample) -> Decisions:
        """Classify objects from their feature values only (the Theorem).

        Raises:
            TypeError: ``unlabelled`` is not an :class:`UnlabelledSample` — a dataset carries the
                classes (ADR-085).
            ValueError: The objects do not have the training sample's features.
        """
        sample = require_unlabelled(unlabelled)
        names, types = self._features()
        if sample.feature_names != names or sample.feature_types != types:
            raise ValueError(
                f"{sample.name}: the objects do not have the features of the training sample"
            )
        return self._decide(sample)

    @abstractmethod
    def _features(self) -> tuple[tuple[str, ...], tuple[FeatureType, ...]]:
        """Names and types of the features the method was fitted on."""

    @abstractmethod
    def _decide(self, sample: UnlabelledSample) -> Decisions:
        """The decisions for objects with the training sample's features."""


@dataclass(frozen=True, eq=False)
class FittedBaseline(FittedMethod):
    """A baseline whose model is the training sample with the fixed settings."""

    spec: MethodSpec
    training: Dataset
    """The training sample, with its classes."""
    config: ExperimentConfig
    training_decisions: IntArray | None = field(repr=False)
    undefined: str = ""
    options: Mapping[str, Any] = field(default_factory=dict)
    """Options of a scikit-learn estimator."""

    def _features(self) -> tuple[tuple[str, ...], tuple[FeatureType, ...]]:
        return self.training.feature_names, self.training.feature_types

    def _decide(self, sample: UnlabelledSample) -> Decisions:
        data = BaselineInput(self.training, sample.X, None, self.config)
        if self.spec.method == "knn":
            params = KnnVoteParams(ks=KNN_KS, operators=(METRIC_TITLES[str(self.spec.metric)],))
            output = knn_vote(data, params)[KNN_KS.index(int(self.spec.k or 0))]
        else:  # SVM, random forest: fitted again on the training sample (deterministic)
            baseline = BASELINES.get(self.spec.method)
            (output,) = baseline(data, SklearnParams(options=dict(self.options)))
        return Decisions(output.decisions, output.scores)


@dataclass(frozen=True, eq=False)
class FittedCSModel(FittedMethod):
    """The CS-model fitted on the training sample; ``model`` is ``None`` if it is undefined."""

    spec: MethodSpec
    model: CSModel | None
    names: tuple[str, ...]
    types: tuple[FeatureType, ...]
    training_decisions: IntArray | None = field(default=None, repr=False)
    undefined: str = ""
    r: int | None = None
    """Number of synthetic features kept."""
    left_out_zero: int = 0
    """Synthetic features left out because they take 0 on a training object (Table 1)."""
    tuplam: tuple[str, ...] = ()

    def _features(self) -> tuple[tuple[str, ...], tuple[FeatureType, ...]]:
        return self.names, self.types

    def _decide(self, sample: UnlabelledSample) -> Decisions:
        if self.model is None:
            return Decisions(None, None, self.undefined)
        classified = self.model.classify(sample.X)
        return Decisions(classified.decisions, classified.scores)


# ---------------------------------------------------------------- the stages


@dataclass(frozen=True, eq=False)
class FittedSplit:
    """*Training* on one split: every method fitted on the training sample."""

    spec: SplitSpec
    training: Dataset
    """The training sample, with its classes."""
    methods: tuple[FittedMethod, ...]

    def training_metrics(self, method: FittedMethod) -> ClassificationMetrics | None:
        """Training correctness (Definition 2) of one method; ``None`` if it is undefined."""
        if method.training_decisions is None:
            return None
        return classification_metrics(self.training.class_index + 1, method.training_decisions)


@dataclass(frozen=True, eq=False)
class SplitDecisions:
    """*Test* on one split: the decisions of every method for the test objects."""

    fitted: FittedSplit
    objects: UnlabelledSample
    decisions: Mapping[str, Decisions]
    """Method key → its decisions, in the order of the fitted methods."""


def fit_split(spec: SplitSpec, training: Dataset) -> FittedSplit:
    """*Training*: fit every method and the CS-model on the training sample.

    Raises:
        ValueError: The training sample does not have two classes.
    """
    if len(training.classes) != 2:
        raise ValueError(f"{spec.key}: the comparison needs two classes")
    config = comparison_config(spec)
    on_training = BaselineInput(training, training.X, np.arange(training.m, dtype=np.int64), config)
    specs = method_specs(spec, training.n, int(training.quantitative.sum()))
    knn = {
        metric: knn_vote(on_training, KnnVoteParams(ks=KNN_KS, operators=(METRIC_TITLES[metric],)))
        for metric in (*KNN_METRICS, *spec.extra_knn)
    }
    names, types = training.feature_names, training.feature_types
    methods: list[FittedMethod] = []
    for method in specs:
        if method.method in SKLEARN_OPTIONS:
            options = SKLEARN_OPTIONS[method.method]
            baseline = BASELINES.get(method.method)
            (trained,) = baseline(on_training, SklearnParams(options=dict(options)))
            methods.append(
                FittedBaseline(method, training, config, trained.decisions, options=options)
            )
        elif method.method == "knn":
            output = knn[str(method.metric)][KNN_KS.index(int(method.k or 0))]
            methods.append(FittedBaseline(method, training, config, output.decisions))
        else:
            methods.append(_fit_cs_model(method, training, spec, names, types))
    return FittedSplit(spec, training, tuple(methods))


def _fit_cs_model(
    method: MethodSpec,
    training: Dataset,
    spec: SplitSpec,
    names: tuple[str, ...],
    types: tuple[FeatureType, ...],
) -> FittedCSModel:
    metric, alpha = str(method.metric), float(method.alpha or 0.0)
    config = cs_model_config(metric, alpha, all_features=metric in spec.extra_knn)
    try:
        model = fit_model(training, config)
    except ModelUndefinedError as error:
        return FittedCSModel(method, None, names, types, undefined=str(error))
    trace = model.trace
    return FittedCSModel(
        method,
        model,
        names,
        types,
        training_decisions=readonly(model.classify_training().decisions),
        r=trace.r,
        left_out_zero=sum(1 for s in trace.skipped_features if s.reason == "zero"),
        tuplam=tuple(trace.features[u].label for u in model.tuplam),
    )


def classify_split(fitted: FittedSplit, unlabelled: UnlabelledSample) -> SplitDecisions:
    """*Test*: every fitted method classifies the objects without their class."""
    sample = require_unlabelled(unlabelled)
    decisions = {m.spec.key: m.classify(sample) for m in fitted.methods}
    return SplitDecisions(fitted, sample, MappingProxyType(decisions))


def _sizes(training: Dataset) -> tuple[int, int, int]:
    k1, k2 = (int(c) for c in np.bincount(training.class_index, minlength=2))
    return training.m, k1, k2


def evaluate_split(decided: SplitDecisions, answers: AnswerKey) -> SplitResult:
    """*Evaluation*: the decisions against the true classes of the test objects.

    Raises:
        ValueError: The answer key is not for the objects that were classified.
    """
    if not answers.matches(decided.objects):
        raise ValueError(f"{answers.name}: the answer key is for other objects")
    fitted = decided.fitted
    training = fitted.training
    results: list[MethodResult] = []
    models: list[CSModelResult] = []
    for method in fitted.methods:
        spec = method.spec
        test = decided.decisions[spec.key]
        tested = (
            None
            if test.decisions is None
            else classification_metrics(answers.classes, test.decisions)
        )
        trained = fitted.training_metrics(method)
        if isinstance(method, FittedCSModel):
            models.append(
                CSModelResult(
                    str(spec.metric),
                    float(spec.alpha or 0.0),
                    spec.features,
                    spec.n_features,
                    r=method.r,
                    left_out_zero=method.left_out_zero,
                    tuplam=method.tuplam,
                    test_decisions=test.decisions,
                    training_decisions=method.training_decisions,
                    test=tested,
                    training=trained,
                    undefined=method.undefined,
                )
            )
            continue
        if (
            test.decisions is None
            or method.training_decisions is None
            or tested is None
            or trained is None
        ):
            raise ValueError(f"{spec.key}: a baseline gave no decisions")
        results.append(
            MethodResult(
                key=spec.key,
                method=spec.method,
                metric=spec.metric,
                k=spec.k,
                features=spec.features,
                n_features=spec.n_features,
                table=spec.table,
                test_decisions=test.decisions,
                training_decisions=method.training_decisions,
                test=tested,
                training=trained,
            )
        )
    size, k1, k2 = answers.sizes
    return SplitResult(
        spec=fitted.spec,
        training_sizes=_sizes(training),
        test_sizes=(size, k1, k2),
        n_features=training.n,
        n_quantitative=int(training.quantitative.sum()),
        methods=tuple(results),
        cs_model=tuple(models),
    )


def compare_split(spec: SplitSpec, split: FixedSplit) -> SplitResult:
    """Fit every method and the CS-model on the training sample; classify the test sample.

    The three stages in a row: :func:`fit_split` on the training sample, :func:`classify_split`
    on the test objects without their class, :func:`evaluate_split` against their classes.

    Raises:
        ValueError: The split does not have two classes.
    """
    if len(split.dataset.classes) != 2:
        raise ValueError(f"{spec.key}: the comparison needs two classes")
    fitted = fit_split(spec, split.training_part)
    decided = classify_split(fitted, split.test_objects())
    return evaluate_split(decided, split.test_answers())
