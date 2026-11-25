"""The fixed settings of the comparison and the rule that gives a split its methods.

Fixed before any run; nothing is tuned and nothing is chosen on the test objects (ADR-079,
ADR-080). The splits themselves are not listed here: every split file of the folder is a sample
(ADR-090), and only the comparison set — German's three splits, the columns of the comparison
tables — is named in code (:data:`COMPARISON_SPLITS`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

SEED = 42
"""Seed of Random Forest (and of the experiment configuration)."""
DISTANCE_DECIMALS = 10
"""Rounding of the k-NN distances (the project's convention)."""
KNN_KS = (3, 5)
"""Neighbourhood sizes of the k-NN vote."""
KNN_METRICS = ("euclidean", "chebyshev", "canberra")
"""Metrics of the k-NN rows of the k-NN by metric table (on the quantitative features)."""
SKLEARN_OPTIONS: dict[str, dict[str, str | int | float]] = {
    "svm": {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
    "random-forest": {"n_estimators": 200, "random_state": SEED},
}
"""Options of the scikit-learn baselines, in the order of the Standard classifiers table's rows."""
METHOD_TITLES = {"svm": "SVM", "random-forest": "Random Forest", "knn": "k-NN"}
"""Display names of the methods."""
METRIC_TITLES = {
    "zhuravlyov": "Zhuravlyov",
    "euclidean": "Euclidean",
    "chebyshev": "Chebyshev",
    "canberra": "Canberra",
}
"""Display names of the metrics (of the k-NN and of the splits)."""
TABLE_STANDARD = "Standard classifiers"
"""Name of the table of SVM and Random Forest."""
TABLE_KNN = "k-NN by metric"
"""Name of the table of the k-NN vote, one row per metric."""
TABLE_CS_MODEL = "CS-model"
"""Name of the CS-model's table, beside the comparison tables."""
COMPARISON_SPLITS = ("German-336", "German-341", "German-367")
"""The comparison set, in column order: German's Zhuravlyov, Euclidean and Chebyshev splits — the
columns of the comparison tables and the content of ``experiments/comparison``. The only list of
splits kept in code (ADR-090)."""


def metric_title(metric: str) -> str:
    """``Euclidean`` for ``euclidean``; a metric without a title is capitalised."""
    return METRIC_TITLES.get(metric, metric.replace("-", " ").capitalize())


def svm_settings() -> str:
    """SVM's own settings: ``SVC, RBF kernel, C = 1.0, gamma = scale`` (ADR-094)."""
    options = SKLEARN_OPTIONS["svm"]
    return (
        f"SVC, {str(options['kernel']).upper()} kernel, C = {options['C']}, "
        f"gamma = {options['gamma']}"
    )


def random_forest_settings() -> str:
    """Random Forest's own settings: ``200 trees, random_state = 42`` (ADR-094)."""
    options = SKLEARN_OPTIONS["random-forest"]
    return f"{options['n_estimators']} trees, random_state = {options['random_state']}"


def knn_settings(metric: str, k: int, features: str) -> str:
    """A k-NN row's own settings: its metric, k, the features it uses and the tie rule."""
    return (
        f"{metric_title(metric)} metric, k = {k}, on the {features} features; equal distances "
        "go to the smaller training index"
    )


def own_metrics(key: str, metric: str) -> tuple[str, ...]:
    """Further metrics of a split's k-NN vote and CS-model, on every feature (ADR-090).

    The rule: a split of the comparison set whose own metric is none of the k-NN metrics also
    runs with its own metric — German's Zhuravlyov split. Every other split runs with the k-NN
    metrics only.
    """
    return (metric,) if key in COMPARISON_SPLITS and metric not in KNN_METRICS else ()


@dataclass(frozen=True)
class SplitSpec:
    """A split as the comparison reads it: its key, dataset, file and the metric of the split."""

    key: str
    """``German-336`` (ADR-090): internal, never shown in the desktop application."""
    dataset: str
    """Name of the dataset, e.g. ``German``."""
    file: str
    """The split file, relative to the split folder."""
    split_metric: str
    """Proximity measure the split was made with (registry name)."""
    training_size: int = 0
    """Objects of the training sample, as the file's name gives it."""
    extra_knn: tuple[str, ...] = ()
    """Further metrics of the k-NN vote and the CS-model, on every feature (:func:`own_metrics`)."""

    @classmethod
    def of(cls, key: str, dataset: str, file: str, metric: str, training_size: int) -> SplitSpec:
        """A split with the methods of the rule (:func:`own_metrics`)."""
        return cls(key, dataset, file, metric, training_size, own_metrics(key, metric))

    @property
    def title(self) -> str:
        """``Zhuravlyov split`` and so on."""
        return f"{metric_title(self.split_metric)} split"

    @property
    def name(self) -> str:
        """``German · Zhuravlyov split · training 336``."""
        return f"{self.dataset} · {self.title} · training {self.training_size}"

    def label(self, sizes: Sequence[int]) -> str:
        """``German · Zhuravlyov split · training 336 (196, 140)`` from (|T|, |T ∩ K1|, …)."""
        size, *classes = sizes
        return f"{self.dataset} · {self.title} · training {size} ({', '.join(map(str, classes))})"


def percent(value: float) -> float:
    """A share as a percentage with two decimals (the tables' precision)."""
    return round(100 * value, 2)
