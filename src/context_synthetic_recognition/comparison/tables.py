"""The comparison tables, the CS-model's table, the full results table and the summary — no I/O.

The tables are :class:`~context_synthetic_recognition.export.tables.Table` objects, so the
project's renderers write them as CSV, Markdown and LaTeX. Their cells are percentages with two
decimals; the notes below them give the majority-class rates of the test samples and the settings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from context_synthetic_recognition.comparison.cs_model import (
    ALPHAS,
    K_START,
    K_STOP,
    CSModelResult,
    cs_model_metrics,
)
from context_synthetic_recognition.comparison.methods import MethodResult, SplitResult, knn_key
from context_synthetic_recognition.comparison.settings import (
    DISTANCE_DECIMALS,
    KNN_KS,
    KNN_METRICS,
    METHOD_TITLES,
    METRIC_TITLES,
    SEED,
    SKLEARN_OPTIONS,
    TABLE_CS_MODEL,
    TABLE_KNN,
    TABLE_STANDARD,
    percent,
)
from context_synthetic_recognition.evaluation.metrics import ClassificationMetrics
from context_synthetic_recognition.export.tables import Cell, Table

GROUP = "Comparison"
"""Group of the comparison's tables."""
STANDARD_KEY = "standard-classifiers"
KNN_KEY = "knn-by-metric"
CS_MODEL_KEY = "cs-model"
RESULTS_KEY = "results"
DECIMALS = 2
"""Decimals of the percentages in Markdown and LaTeX."""

SETTINGS_SKLEARN = (
    "SVM: SVC with the RBF kernel, C = 1.0, gamma = scale; Random Forest: 200 trees, "
    f"random_state = {SEED}. Every feature of the file: the quantitative ones min–max scaled on "
    "the training sample, the nominal ones one-hot encoded with the categories of the training "
    "sample (an unseen category encodes as all zeros)."
)
SETTINGS_KNN = (
    "Majority vote of the k nearest training objects on the quantitative features, min–max "
    f"scaled on the training sample; distances rounded to {DISTANCE_DECIMALS} decimals, equal "
    "distances to the smaller training index."
)
TRAINING_ACCURACY = (
    "resubstitution (Definition 2): a k-NN training object is not its own neighbour, a CS-model "
    "training object is left out of its own context; SVM and Random Forest classify the training "
    "objects with the model fitted on them"
)
SETTINGS_CS_MODEL = (
    "One base operator per metric, min–max scaled on the training sample: Euclidean, Chebyshev "
    "and Canberra on the quantitative features, Zhuravlyov on all features (the 336 split only); "
    f"odd k = {K_START} … {K_STOP}; synthetic features 0, 1, 2 with the features that take 0 on a "
    "training object left out; final class means, one majorizing pass. Refusals count as errors "
    "(Definition 3)."
)


def _column(result: SplitResult) -> str:
    return f"{result.column} — {result.spec.title}"


def _majority(results: Sequence[SplitResult]) -> str:
    rates = ", ".join(f"{percent(r.majority_rate):.{DECIMALS}f} % ({r.spec.key})" for r in results)
    return f"Majority-class rate of the test sample: {rates}."


def standard_classifiers_table(results: Sequence[SplitResult]) -> Table:
    """Standard classifiers: accuracy of SVM and Random Forest on the test samples, %."""
    rows: tuple[tuple[Cell, ...], ...] = tuple(
        (METHOD_TITLES[key], *(percent(r.method(key).test.accuracy) for r in results))
        for key in SKLEARN_OPTIONS
    )
    return Table(
        key=STANDARD_KEY,
        title=f"{TABLE_STANDARD}: accuracy of SVM and Random Forest on the test samples, %",
        columns=("Method", *(_column(r) for r in results)),
        rows=rows,
        group=GROUP,
        note=(
            "Each method is fitted on the training sample and classifies the test sample. "
            f"{_majority(results)} {SETTINGS_SKLEARN}"
        ),
    )


def knn_by_metric_table(results: Sequence[SplitResult]) -> Table:
    """k-NN by metric: accuracy of the k-NN vote on the test samples, k = 3 / k = 5, %."""
    rows: list[tuple[Cell, ...]] = []
    for metric in KNN_METRICS:
        cells = []
        for result in results:
            values = (percent(result.method(knn_key(metric, k)).test.accuracy) for k in KNN_KS)
            cells.append(" / ".join(f"{v:.{DECIMALS}f}" for v in values))
        rows.append((METRIC_TITLES[metric], *cells))
    ks = " / ".join(f"k = {k}" for k in KNN_KS)
    return Table(
        key=KNN_KEY,
        title=f"{TABLE_KNN}: accuracy of the k-NN vote on the test samples, {ks}, %",
        columns=("Metric", *(_column(r) for r in results)),
        rows=tuple(rows),
        group=GROUP,
        note=(
            f"Each cell: {ks}. {_majority(results)} {SETTINGS_KNN} Training: the article's "
            "training samples; nothing is chosen on the test samples."
        ),
    )


def _alpha_cell(results: Sequence[CSModelResult]) -> Cell:
    values = [None if r.test is None else percent(r.test.accuracy) for r in results]
    if not values:
        return None
    return " / ".join("—" if v is None else f"{v:.{DECIMALS}f}" for v in values)


def cs_model_table(results: Sequence[SplitResult]) -> Table:
    """The CS-model on the same samples: accuracy on the test samples, α = 0.1 / α = 0.2, %."""
    alphas = " / ".join(f"α = {a}" for a in ALPHAS)
    metrics = [*KNN_METRICS]
    for result in results:
        metrics += [m for m in cs_model_metrics(result.spec) if m not in metrics]
    rows = tuple(
        (
            METRIC_TITLES[metric],
            *(_alpha_cell([c for c in r.cs_model if c.metric == metric]) for r in results),
        )
        for metric in metrics
    )
    return Table(
        key=CS_MODEL_KEY,
        title=f"{TABLE_CS_MODEL} on the same samples: accuracy on the test samples, {alphas}, %",
        columns=("Metric", *(_column(r) for r in results)),
        rows=rows,
        group=GROUP,
        note=(
            f"Each cell: {alphas}; — = not run on that split. {_majority(results)} "
            f"{SETTINGS_CS_MODEL} Fitted on the article's training samples; the test objects are "
            "classified without their class."
        ),
    )


RESULT_COLUMNS = (
    "Split",
    "Split metric",
    "Method",
    "Features",
    "Table",
    "Training",
    "Training K1",
    "Training K2",
    "Test",
    "Test K1",
    "Test K2",
    "Majority rate, %",
    "Test accuracy, %",
    "Balanced accuracy, %",
    "Test correct",
    "Test refusals",
    "Answered accuracy, %",
    "Training accuracy, %",
    "Training correct",
)
"""Columns of the full results table."""


def _result_row(split: SplitResult, method: MethodResult) -> tuple[Cell, ...]:
    return (
        split.spec.key,
        METRIC_TITLES[split.spec.split_metric],
        method.title,
        f"{method.features} ({method.n_features})",
        method.table,
        *split.training_sizes,
        *split.test_sizes,
        percent(split.majority_rate),
        percent(method.test.accuracy),
        percent(method.test.macro_recall),
        method.test.correct,
        method.test.refusals,
        _answered(method.test),
        percent(method.training.accuracy),
        method.training.correct,
    )


def _answered(metrics: ClassificationMetrics) -> Cell:
    answered = metrics.accuracy_answered
    return None if answered is None else percent(answered)


def _cs_model_row(split: SplitResult, model: CSModelResult) -> tuple[Cell, ...]:
    test, training = model.test, model.training
    return (
        split.spec.key,
        METRIC_TITLES[split.spec.split_metric],
        model.title,
        f"{model.features} ({model.n_features})",
        TABLE_CS_MODEL,
        *split.training_sizes,
        *split.test_sizes,
        percent(split.majority_rate),
        None if test is None else percent(test.accuracy),
        None if test is None else percent(test.macro_recall),
        None if test is None else test.correct,
        None if test is None else test.refusals,
        None if test is None else _answered(test),
        None if training is None else percent(training.accuracy),
        None if training is None else training.correct,
    )


def results_table(results: Sequence[SplitResult]) -> Table:
    """Every split × method: accuracy, balanced accuracy, majority rate, training accuracy."""
    return Table(
        key=RESULTS_KEY,
        title="Standard classifiers on the article's training and test samples",
        columns=RESULT_COLUMNS,
        rows=tuple(
            row
            for s in results
            for row in (
                *(_result_row(s, m) for m in s.methods),
                *(_cs_model_row(s, c) for c in s.cs_model),
            )
        ),
        group=GROUP,
        note=(
            "Balanced accuracy: the mean recall of the two classes. Majority rate: the share of "
            "the larger class in the test sample. Test accuracy: refusals count as errors; "
            "answered accuracy: on the objects that received a class. Training accuracy: "
            f"{TRAINING_ACCURACY}. Table — = in neither the {TABLE_STANDARD} nor the {TABLE_KNN} "
            "table."
        ),
        labels=5,
    )


def comparison_tables(results: Sequence[SplitResult]) -> tuple[Table, Table, Table, Table]:
    """Standard classifiers, k-NN by metric, the CS-model's table and the full results table."""
    return (
        standard_classifiers_table(results),
        knn_by_metric_table(results),
        cs_model_table(results),
        results_table(results),
    )


def _scores(metrics: ClassificationMetrics) -> dict[str, Any]:
    return {
        "accuracy": percent(metrics.accuracy),
        "balanced_accuracy": percent(metrics.macro_recall),
        "correct": metrics.correct,
        "n": metrics.n,
        "refusals": metrics.refusals,
    }


def _cs_model_data(model: CSModelResult) -> dict[str, Any]:
    if model.test is None or model.training is None:
        return {"key": model.key, "title": model.title, "undefined": model.undefined}
    return {
        "key": model.key,
        "title": model.title,
        "metric": model.metric,
        "alpha": model.alpha,
        "features": model.features,
        "n_features": model.n_features,
        "r": model.r,
        "left_out_zero": model.left_out_zero,
        "tuplam": list(model.tuplam),
        "test": {**_scores(model.test), "accuracy_answered": _answered(model.test)},
        "training": {**_scores(model.training), "accuracy_answered": _answered(model.training)},
    }


def _sizes(sizes: tuple[int, int, int]) -> dict[str, int]:
    return {"n": sizes[0], "K1": sizes[1], "K2": sizes[2]}


def summary_data(
    results: Sequence[SplitResult],
    *,
    versions: Mapping[str, str],
    inputs: Mapping[str, str],
) -> dict[str, Any]:
    """Settings, library versions and every result, for ``summary.json``.

    Args:
        results: The splits in the tables' column order.
        versions: Versions of the libraries the numbers depend on (scikit-learn, numpy).
        inputs: SHA-256 of every split file, by its path relative to ``datasets/splits``.
    """
    return {
        "description": (
            "Standard classifiers on the article's training and test samples: the "
            f"{TABLE_STANDARD} and {TABLE_KNN} tables "
            "(ADR-079), and the CS-model on the same samples (ADR-080). Percentages with two "
            "decimals."
        ),
        "settings": {
            "svm": dict(SKLEARN_OPTIONS["svm"]),
            "random-forest": dict(SKLEARN_OPTIONS["random-forest"]),
            "svm_and_random_forest_features": SETTINGS_SKLEARN,
            "knn": {
                "ks": list(KNN_KS),
                "metrics": list(KNN_METRICS),
                "features": "quantitative",
                "normalization": "min–max fitted on the training sample",
                "distance_decimals": DISTANCE_DECIMALS,
                "ties": "the smaller training index",
            },
            "cs_model": {
                "alphas": list(ALPHAS),
                "k": f"odd k = {K_START} … {K_STOP}",
                "settings": SETTINGS_CS_MODEL,
            },
            "training_accuracy": TRAINING_ACCURACY,
            "balanced_accuracy": "mean recall of the two classes",
        },
        "versions": dict(versions),
        "splits": [
            {
                "key": r.spec.key,
                "dataset": r.spec.dataset,
                "file": r.spec.file,
                "file_sha256": inputs[r.spec.file],
                "split_metric": r.spec.split_metric,
                "features": r.n_features,
                "quantitative_features": r.n_quantitative,
                "training": _sizes(r.training_sizes),
                "test": _sizes(r.test_sizes),
                "majority_rate": percent(r.majority_rate),
                "methods": [
                    {
                        "key": m.key,
                        "title": m.title,
                        "table": m.table,
                        "features": m.features,
                        "n_features": m.n_features,
                        "test": _scores(m.test),
                        "training": _scores(m.training),
                    }
                    for m in r.methods
                ],
                "cs_model": [_cs_model_data(c) for c in r.cs_model],
            }
            for r in results
        ],
    }
