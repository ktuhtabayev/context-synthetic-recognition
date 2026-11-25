"""Standard classifiers on the article's training and test samples (ADR-079).

Every setting is fixed here, before any run, and nothing is chosen on the test objects (ADR-079):

- SVM and Random Forest: the project's ``svm`` and ``random-forest`` baselines with explicit
  options — ``SVC(kernel="rbf", C=1.0, gamma="scale")`` and
  ``RandomForestClassifier(n_estimators=200, random_state=42)`` — on every feature of the file:
  quantitative features min–max scaled on the training sample, nominal features one-hot encoded
  with the categories of the training sample (an unseen category encodes as all zeros);
- k-NN: the project's ``knn-vote`` baseline, majority vote with k = 3 and k = 5, on the
  quantitative features min–max scaled on the training sample, with the Euclidean, Chebyshev and
  Canberra metrics (the k-NN by metric table) and, where a split lists it, the Zhuravlyov metric
  on all features (an extra row, in neither table). Distances are rounded to 10 decimals and equal
  distances go to the smaller training index (ADR-008).

Each method is fitted on the training sample only and classifies the test objects without their
class: :mod:`.stages` runs the comparison in three stages (training, test, evaluation), and
:func:`~.stages.compare_split` is the three in a row. Reported per method: the accuracy on the
test sample (the tables' cells), the balanced accuracy (mean recall of the two classes), the
accuracy on the training sample — resubstitution as in Definition 2, so a k-NN training object
is not its own neighbour — and, per split, the sample sizes and the majority-class rate of the
test sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from context_synthetic_recognition.comparison.cs_model import CSModelResult
from context_synthetic_recognition.comparison.settings import (
    DISTANCE_DECIMALS,
    KNN_METRICS,
    METHOD_TITLES,
    SEED,
    SplitSpec,
    metric_title,
)
from context_synthetic_recognition.config import (
    ExperimentConfig,
    LocalContextConfig,
    OperatorConfig,
    plugin,
)
from context_synthetic_recognition.core.arrays import IntArray, readonly
from context_synthetic_recognition.evaluation.metrics import (
    ClassificationMetrics,
)


@dataclass(frozen=True, eq=False)
class MethodResult:
    """One method on one split."""

    key: str
    """``svm``, ``random-forest`` or ``knn-<metric>-<k>``."""
    method: str
    """Registry name of the baseline: ``svm``, ``random-forest`` or ``knn``."""
    metric: str | None
    """The k-NN metric (``None`` for SVM and Random Forest)."""
    k: int | None
    """The k of the k-NN vote."""
    features: str
    """``all`` or ``quantitative``."""
    n_features: int
    """Number of original features the method uses."""
    table: str | None
    """``Standard classifiers``, ``k-NN by metric`` or ``None`` (in neither table)."""
    test_decisions: IntArray | None = field(repr=False)
    """(test objects,) decisions: 1 = K1, 2 = K2, 0 = refusal; ``None`` if unavailable."""
    training_decisions: IntArray | None = field(repr=False)
    """(training objects,) resubstitution decisions."""
    test: ClassificationMetrics | None = field(repr=False)
    training: ClassificationMetrics | None = field(repr=False)
    undefined: str = ""
    """Why the split cannot support the method (empty if it can, ADR-090)."""

    def __post_init__(self) -> None:
        """Freeze the decisions."""
        for decisions in (self.test_decisions, self.training_decisions):
            if decisions is not None:
                readonly(decisions)

    @property
    def title(self) -> str:
        """Display name: ``SVM``, ``Random Forest``, ``k-NN Euclidean, k = 3`` and so on."""
        if self.metric is None:
            return METHOD_TITLES[self.method]
        return f"k-NN {metric_title(self.metric)}, k = {self.k}"


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
    def label(self) -> str:
        """``German · Zhuravlyov split · training 336 (196, 140)`` — the split's name (ADR-090)."""
        return self.spec.label(self.training_sizes)

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
        OperatorConfig(label=metric_title(m), metric=plugin(m), features="quantitative")
        for m in KNN_METRICS
    ]
    operators += [
        OperatorConfig(label=metric_title(m), metric=plugin(m), features="all")
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
