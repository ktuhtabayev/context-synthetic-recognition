"""The fixed settings and the article's splits of the comparison (ADR-079, ADR-080).

Fixed before any run; nothing is tuned and nothing is chosen on the test objects.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class SplitSpec:
    """One of the article's training/test splits."""

    key: str
    """Short name, e.g. ``336``."""
    dataset: str
    """Name of the dataset, e.g. ``German``."""
    file: str
    """The split file, relative to ``datasets/splits``."""
    split_metric: str
    """Proximity measure the split was made with (registry name)."""
    extra_knn: tuple[str, ...] = ()
    """Further k-NN metrics on all features (rows in neither comparison table)."""

    @property
    def title(self) -> str:
        """``Zhuravlyov split`` and so on."""
        return f"{METRIC_TITLES[self.split_metric]} split"


SPLITS = (
    SplitSpec(
        "336",
        "German",
        "German/German (1000, 20, 2) — training 336 (zhuravlyov).dat",
        "zhuravlyov",
        extra_knn=("zhuravlyov",),
    ),
    SplitSpec(
        "341", "German", "German/German (1000, 7, 2) — training 341 (euclidean).dat", "euclidean"
    ),
    SplitSpec(
        "367", "German", "German/German (1000, 7, 2) — training 367 (chebyshev).dat", "chebyshev"
    ),
)
"""The splits of the comparison tables, in column order."""
HEART_SPLIT = SplitSpec(
    "76",
    "Heart",
    "Heart-Disease/Heart-Disease (270, 13, 2) — training 76 (zhuravlyov).dat",
    "zhuravlyov",
)
"""Heart's split: readable, not part of the comparison tables."""
SPAMBASE_SPLITS = (
    SplitSpec(
        "601",
        "Spambase",
        "Spambase/Spambase (2102, 57, 2) — training 601 (euclidean).dat",
        "euclidean",
    ),
    SplitSpec(
        "722",
        "Spambase",
        "Spambase/Spambase (2102, 57, 2) — training 722 (euclidean).dat",
        "euclidean",
    ),
)
"""Spambase's splits (ADR-088): readable, not part of the comparison tables. Both hold the same 2102
objects; the 722 file's values are min–max scaled over them."""
TABLE_CS_MODEL = "CS-model"
"""Name of the CS-model's table, beside the comparison tables."""


def percent(value: float) -> float:
    """A share as a percentage with two decimals (the tables' precision)."""
    return round(100 * value, 2)
