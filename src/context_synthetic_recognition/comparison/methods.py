"""Standard classifiers on the article's training and test samples (Tables 7 and 8).

Every setting is fixed here, before any run, and nothing is chosen on the test objects (ADR-079):

- SVM and random forest: the project's ``svm`` and ``random-forest`` baselines with explicit
  options — ``SVC(kernel="rbf", C=1.0, gamma="scale")`` and
  ``RandomForestClassifier(n_estimators=200, random_state=42)`` — on every feature of the file:
  quantitative features min–max scaled on the training sample, nominal features one-hot encoded
  with the categories of the training sample (an unseen category encodes as all zeros);
- k-NN: the project's ``knn-vote`` baseline, majority vote with k = 3 and k = 5, on the
  quantitative features min–max scaled on the training sample, with the Euclidean, Chebyshev and
  Canberra metrics (Table 8) and, where a split lists it, the Zhuravlyov metric on all features
  (an extra row, not printed). Distances are rounded to 10 decimals and equal distances go to the
  smaller training index (ADR-008).

Each method is fitted on the training sample only and classifies the test objects without their
class. Reported per method: the accuracy on the test sample (the tables' cells), the balanced
accuracy (mean recall of the two classes), the accuracy on the training sample — resubstitution as
in Definition 2, so a k-NN training object is not its own neighbour — and, per split, the sample
sizes and the majority-class rate of the test sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from context_synthetic_recognition.comparison.cs_model import CSModelResult, run_cs_model
from context_synthetic_recognition.comparison.settings import (
    DISTANCE_DECIMALS,
    KNN_KS,
    KNN_METRICS,
    METHOD_TITLES,
    METRIC_TITLES,
    SEED,
    SKLEARN_OPTIONS,
    TABLE_7,
    TABLE_8,
    SplitSpec,
)
from context_synthetic_recognition.config import (
    ExperimentConfig,
    LocalContextConfig,
    OperatorConfig,
    plugin,
)
from context_synthetic_recognition.core.arrays import IntArray, readonly
from context_synthetic_recognition.data.schema import Dataset
from context_synthetic_recognition.data.splits import FixedSplit
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


@dataclass(frozen=True, eq=False)
class MethodResult:
    """One method on one split."""

    key: str
    """``svm``, ``random-forest`` or ``knn-<metric>-<k>``."""
    method: str
    """Registry name of the baseline: ``svm``, ``random-forest`` or ``knn``."""
    metric: str | None
    """The k-NN metric (``None`` for SVM and random forest)."""
    k: int | None
    """The k of the k-NN vote."""
    features: str
    """``all`` or ``quantitative``."""
    n_features: int
    """Number of original features the method uses."""
    table: str | None
    """``Table 7``, ``Table 8`` or ``None`` (not printed)."""
    test_decisions: IntArray = field(repr=False)
    """(test objects,) decisions: 1 = K1, 2 = K2, 0 = refusal."""
    training_decisions: IntArray = field(repr=False)
    """(training objects,) resubstitution decisions."""
    test: ClassificationMetrics = field(repr=False)
    training: ClassificationMetrics = field(repr=False)

    def __post_init__(self) -> None:
        """Freeze the decisions."""
        readonly(self.test_decisions)
        readonly(self.training_decisions)

    @property
    def title(self) -> str:
        """Display name: ``SVM``, ``Random Forest``, ``k-NN Euclidean, k = 3`` and so on."""
        if self.metric is None:
            return METHOD_TITLES[self.method]
        return f"k-NN {METRIC_TITLES[self.metric]}, k = {self.k}"


@dataclass(frozen=True, eq=False)
class SplitResult:
    """Every method on one split."""

    spec: SplitSpec
    training_sizes: tuple[int, int, int]
    """|training|, training objects of K1 and of K2."""
    test_sizes: tuple[int, int, int]
    """|test|, test objects of K1 and of K2."""
    n_features: int
    n_quantitative: int
    methods: tuple[MethodResult, ...]
    cs_model: tuple[CSModelResult, ...] = ()
    """The CS-model on the same samples, per metric and α (ADR-080)."""

    @property
    def majority_rate(self) -> float:
        """Share of the larger class in the test sample."""
        return max(self.test_sizes[1:]) / self.test_sizes[0]

    @property
    def column(self) -> str:
        """``336 (196, 140)`` — the training sample's size and class split."""
        size, k1, k2 = self.training_sizes
        return f"{size} ({k1}, {k2})"

    def method(self, key: str) -> MethodResult:
        """The result of a method by its key.

        Raises:
            KeyError: No such method on this split.
        """
        for result in self.methods:
            if result.key == key:
                return result
        raise KeyError(key)


def knn_key(metric: str, k: int) -> str:
    """Key of a k-NN result: ``knn-euclidean-3``."""
    return f"knn-{metric}-{k}"


def comparison_config(spec: SplitSpec) -> ExperimentConfig:
    """The configuration the baselines read: min–max, the k-NN operators, rounding, seed."""
    operators = [
        OperatorConfig(label=METRIC_TITLES[m], metric=plugin(m), features="quantitative")
        for m in KNN_METRICS
    ]
    operators += [
        OperatorConfig(label=METRIC_TITLES[m], metric=plugin(m), features="all")
        for m in spec.extra_knn
    ]
    return ExperimentConfig(
        name=f"comparison-{spec.key}",
        seed=SEED,
        local_context=LocalContextConfig(
            normalizer=plugin("minmax"),
            operators=tuple(operators),
            distance_decimals=DISTANCE_DECIMALS,
        ),
    )


def _sizes(dataset: Dataset) -> tuple[int, int, int]:
    k1, k2 = (int(c) for c in np.bincount(dataset.class_index, minlength=2))
    return dataset.m, k1, k2


def compare_split(spec: SplitSpec, split: FixedSplit) -> SplitResult:
    """Fit every method and the CS-model on the training sample; classify the test sample.

    Raises:
        ValueError: The split does not have two classes.
    """
    if len(split.dataset.classes) != 2:
        raise ValueError(f"{spec.key}: the comparison needs two classes")
    config = comparison_config(spec)
    train, test = split.training_part, split.test_part
    on_test = BaselineInput(train, test.X, None, config)
    on_training = BaselineInput(train, train.X, np.arange(train.m, dtype=np.int64), config)
    truth_test = test.class_index + 1
    truth_training = train.class_index + 1
    n_quantitative = int(train.quantitative.sum())
    results: list[MethodResult] = []

    for key, options in SKLEARN_OPTIONS.items():
        baseline = BASELINES.get(key)
        params = SklearnParams(options=dict(options))
        (tested,) = baseline(on_test, params)
        (trained,) = baseline(on_training, params)
        results.append(
            MethodResult(
                key=key,
                method=key,
                metric=None,
                k=None,
                features="all",
                n_features=train.n,
                table=TABLE_7,
                test_decisions=tested.decisions,
                training_decisions=trained.decisions,
                test=classification_metrics(truth_test, tested.decisions),
                training=classification_metrics(truth_training, trained.decisions),
            )
        )

    for metric in (*KNN_METRICS, *spec.extra_knn):
        printed = metric in KNN_METRICS
        params_knn = KnnVoteParams(ks=KNN_KS, operators=(METRIC_TITLES[metric],))
        tested_knn = knn_vote(on_test, params_knn)
        trained_knn = knn_vote(on_training, params_knn)
        for k, t, r in zip(KNN_KS, tested_knn, trained_knn, strict=True):
            results.append(
                MethodResult(
                    key=knn_key(metric, k),
                    method="knn",
                    metric=metric,
                    k=k,
                    features="quantitative" if printed else "all",
                    n_features=n_quantitative if printed else train.n,
                    table=TABLE_8 if printed else None,
                    test_decisions=t.decisions,
                    training_decisions=r.decisions,
                    test=classification_metrics(truth_test, t.decisions),
                    training=classification_metrics(truth_training, r.decisions),
                )
            )

    return SplitResult(
        spec=spec,
        training_sizes=_sizes(train),
        test_sizes=_sizes(test),
        n_features=train.n,
        n_quantitative=n_quantitative,
        methods=tuple(results),
        cs_model=run_cs_model(spec, split),
    )
