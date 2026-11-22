"""The CS-model on the article's training and test samples, beside Tables 7 and 8 (ADR-080).

Settings approved by the author before any run; nothing is chosen on the test objects:

- one base operator per metric, as the k-NN rows: Euclidean, Chebyshev and Canberra on the
  quantitative features of every split, and — where a split lists it — the Zhuravlyov metric on all
  features (the 336 split); every operator min–max scaled on the training sample;
- the article's experiments: odd k = 3 … 41, synthetic features {0, 1, 2} with the features that
  take 0 on a training object left out (Table 1), α = 0.1 and α = 0.2;
- otherwise the article's defaults: final class means, one majorizing pass, no ⚠ switch on.

The model is fitted on the training sample; the test objects are classified without their class
(the Theorem). Reported: the accuracy on the test sample with refusals counted as errors
(Definition 3), the accuracy on the answered objects, the refusals, the balanced accuracy and the
accuracy on the training sample (Definition 2: an object is left out of its own context).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from context_synthetic_recognition.comparison.settings import (
    KNN_METRICS,
    METRIC_TITLES,
    SplitSpec,
)
from context_synthetic_recognition.config import ExperimentConfig, OperatorConfig, plugin
from context_synthetic_recognition.core.arrays import IntArray, readonly
from context_synthetic_recognition.core.model import fit_model
from context_synthetic_recognition.data.splits import FixedSplit
from context_synthetic_recognition.errors import ModelUndefinedError
from context_synthetic_recognition.evaluation.metrics import (
    ClassificationMetrics,
    classification_metrics,
)

ALPHAS = (0.1, 0.2)
"""Values of the margin regularization parameter α of the article's experiments."""
K_START, K_STOP = 3, 41
"""Odd k = 3 … 41 (the article's experiments, Table 1)."""
NORMALIZER = "minmax"
"""Scaling of every operator, fitted on the training sample."""


@dataclass(frozen=True, eq=False)
class CSModelResult:
    """The CS-model with one metric and one α on one split."""

    metric: str
    """Registry name of the operator's metric."""
    alpha: float
    features: str
    """``all`` or ``quantitative``."""
    n_features: int
    """Number of original features the operator uses."""
    r: int | None = None
    """Number of synthetic features kept (``None`` if the model is undefined)."""
    left_out_zero: int = 0
    """Synthetic features left out because they take 0 on a training object (Table 1)."""
    tuplam: tuple[str, ...] = ()
    """The synthetic features of TUPLAM, by label."""
    test_decisions: IntArray | None = field(default=None, repr=False)
    training_decisions: IntArray | None = field(default=None, repr=False)
    test: ClassificationMetrics | None = field(default=None, repr=False)
    training: ClassificationMetrics | None = field(default=None, repr=False)
    undefined: str = ""
    """Why the model is undefined on this split (empty if it is defined)."""

    def __post_init__(self) -> None:
        """Freeze the decisions."""
        for decisions in (self.test_decisions, self.training_decisions):
            if decisions is not None:
                readonly(decisions)

    @property
    def key(self) -> str:
        """``cs-model-euclidean-0.1``."""
        return f"cs-model-{self.metric}-{self.alpha}"

    @property
    def title(self) -> str:
        """``CS-model Euclidean, α = 0.1``."""
        return f"CS-model {METRIC_TITLES[self.metric]}, α = {self.alpha}"


def cs_model_metrics(spec: SplitSpec) -> tuple[str, ...]:
    """The metrics of the CS-model on a split: those of the k-NN rows, in the same order."""
    return (*KNN_METRICS, *spec.extra_knn)


def cs_model_config(metric: str, alpha: float, *, all_features: bool) -> ExperimentConfig:
    """The article's defaults with one operator, k = 3 … 41, the zero value and α."""
    base = ExperimentConfig()
    operator = OperatorConfig(
        label="ρ",
        metric=plugin(metric),
        features="all" if all_features else "quantitative",
        normalizer=plugin(NORMALIZER),
    )
    return base.model_copy(
        update={
            "name": f"comparison-cs-model-{metric}-alpha-{alpha}",
            "local_context": base.local_context.model_copy(update={"operators": (operator,)}),
            "admissible_k": plugin("range", start=K_START, stop=K_STOP),
            "synthetic_features": base.synthetic_features.model_copy(update={"zero_value": True}),
            "additional_features": base.additional_features.model_copy(update={"alpha": alpha}),
        }
    )


def run_cs_model(spec: SplitSpec, split: FixedSplit) -> tuple[CSModelResult, ...]:
    """Fit the CS-model on the training sample for every metric and α; classify the test sample."""
    train, test = split.training_part, split.test_part
    truth_test = test.class_index + 1
    truth_training = train.class_index + 1
    results = []
    for metric in cs_model_metrics(spec):
        all_features = metric in spec.extra_knn
        features = "all" if all_features else "quantitative"
        n_features = train.n if all_features else int(train.quantitative.sum())
        for alpha in ALPHAS:
            config = cs_model_config(metric, alpha, all_features=all_features)
            try:
                model = fit_model(train, config)
            except ModelUndefinedError as error:
                results.append(
                    CSModelResult(metric, alpha, features, n_features, undefined=str(error))
                )
                continue
            tested = model.predict(test.X)
            trained = model.classify_training().decisions
            trace = model.trace
            results.append(
                CSModelResult(
                    metric,
                    alpha,
                    features,
                    n_features,
                    r=trace.r,
                    left_out_zero=sum(1 for s in trace.skipped_features if s.reason == "zero"),
                    tuplam=tuple(trace.features[u].label for u in model.tuplam),
                    test_decisions=tested,
                    training_decisions=trained,
                    test=classification_metrics(truth_test, tested),
                    training=classification_metrics(truth_training, trained),
                )
            )
    return tuple(results)
