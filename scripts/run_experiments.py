r"""The computational experiment of the article, rebuilt from real runs.

Run from the repository root::

    .\.venv\Scripts\python.exe scripts\run_experiments.py

It writes ``experiments/results/*.csv``, ``experiments/results/summary.json`` and
``experiments/checksums.sha256`` (SHA-256 of every result file). Nothing is tuned: the settings
below are fixed in advance, and where the article leaves a setting open the choice is flagged in
:data:`OPEN_SETTINGS` and in the results (ADR-070).

The experiment, as the article describes it:

- datasets Heart (270, 13, 2) — with the slope quantitative, the article's 7 quantitative and 6
  nominal features — and German (1000, 20, 2);
- R-IPS splits each into a training sample (the objects left in E₁) and a test sample (the
  moved set D): with the Zhuravlyov metric on all features, the quantitative ones normalized to
  [0; 1]; and for German also with the Euclidean and the Chebyshev metric on the 7 quantitative
  features without normalization (Table 6);
- the CS-model with the Zhuravlyov, Euclidean, Chebyshev and Canberra metrics, k ∈ {3, 5, …, 41},
  synthetic features {0, 1, 2} with the features that take 0 left out (Table 1), and the
  regularization parameter α = 0.1 and 0.2 (Tables 2–6): accuracy on the training sample
  (Definition 2) and on the test sample (Definition 3);
- k-NN on the same splits for comparison.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from context_synthetic_recognition import __version__
from context_synthetic_recognition.config import (
    DatasetConfig,
    ExperimentConfig,
    LocalContextConfig,
    OperatorConfig,
    RIPSConfig,
    plugin,
)
from context_synthetic_recognition.core.model import fit_model
from context_synthetic_recognition.core.rips.selection import select_precedents
from context_synthetic_recognition.data import Dataset, load_from_config
from context_synthetic_recognition.errors import ModelUndefinedError
from context_synthetic_recognition.evaluation.baselines import (
    BaselineInput,
    KnnVoteParams,
    knn_vote,
)
from context_synthetic_recognition.evaluation.metrics import classification_metrics
from context_synthetic_recognition.export.text import csv_rows, write_csv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments"

DATASETS = {
    "Heart": DatasetConfig(path="heart-disease-270", feature_types={"x₁₁": "quantitative"}),
    "German": DatasetConfig(path="german-1000"),
}
"""The two datasets of the article (Heart with the slope x₁₁ quantitative: 7 quantitative)."""

ZHURAVLYOV_SPLIT = RIPSConfig()
SPLITS = {
    "zhuravlyov": RIPSConfig(metric=plugin("zhuravlyov")),
    "euclidean": RIPSConfig(
        metric=plugin("euclidean"), normalizer=plugin("none"), features="quantitative"
    ),
    "chebyshev": RIPSConfig(
        metric=plugin("chebyshev"), normalizer=plugin("none"), features="quantitative"
    ),
}
"""R-IPS proximity measures: Zhuravlyov (all features, min–max) and, for German, Euclidean and
Chebyshev on the quantitative features without normalization."""
READINGS = {"literal": False, "keep-covering": True}
"""``rips.keep_blocked_cover``: the article's text, and blocked objects that keep covering."""

METRICS = ("zhuravlyov", "euclidean", "chebyshev", "canberra")
ALPHAS = (0.1, 0.2)
KS = plugin("range", start=3, stop=41)
KNN_KS = (1, 3, 5)

ARTICLE = {
    "samples": {
        ("Heart", "zhuravlyov"): (76, 36, 40),
        ("German", "zhuravlyov"): (336, 196, 140),
        ("German", "euclidean"): (341, 173, 168),
        ("German", "chebyshev"): (367, 190, 177),
    },
    "binary_features": {  # Table 1, out of 31
        ("Heart", "zhuravlyov"): 30,
        ("Heart", "euclidean"): 30,
        ("Heart", "chebyshev"): 30,
        ("Heart", "canberra"): 29,
        ("German", "zhuravlyov"): 21,
        ("German", "euclidean"): 25,
        ("German", "chebyshev"): 23,
        ("German", "canberra"): 27,
    },
    "accuracy": {  # Tables 2–5 (Zhuravlyov split): (training, test) for α = 0.1 and 0.2, in %
        ("Heart", "zhuravlyov"): ((97.37, 100.0), (100.0, 100.0)),
        ("German", "zhuravlyov"): ((97.32, 100.0), (100.0, 100.0)),
        ("Heart", "euclidean"): ((97.37, 100.0), (100.0, 100.0)),
        ("German", "euclidean"): ((100.0, 100.0), (100.0, 100.0)),
        ("Heart", "chebyshev"): ((96.05, 100.0), (100.0, 100.0)),
        ("German", "chebyshev"): ((100.0, 100.0), (100.0, 100.0)),
        ("Heart", "canberra"): ((98.68, 100.0), (100.0, 100.0)),
        ("German", "canberra"): ((100.0, 100.0), (100.0, 100.0)),
    },
    "table6": {  # German splits by Euclid and Chebyshev: training / test accuracy, %
        "euclidean": (100.0, 100.0),
        "chebyshev": (100.0, 100.0),
    },
    "knn": "k-NN not above 80 %",
}
"""The numbers the article prints, for the comparison (never used by the computation)."""

OPEN_SETTINGS = [
    (
        "Normalization of the CS-model for the Euclidean, Chebyshev and Canberra metrics: none "
        "(the article states it only for the R-IPS splits)."
    ),
    "Features of the Canberra metric: the quantitative ones (the metric is defined on them).",
    "k-NN of the comparison: majority vote with k = 1, 3, 5 and the same metric as the CS-model.",
    "Table 6 gives one pair of accuracies for two values of α; both α are computed here.",
    (
        "The article reports 31 synthetic features for k ∈ {3, …, 41}, which has 20 values; the "
        "20 values give 20 synthetic features per metric here."
    ),
    "R-IPS: both readings of blocking are computed; the training sample is the remaining set E₁.",
]
"""Settings the article leaves open, chosen before any run (flagged in the results)."""


def operator(metric: str) -> OperatorConfig:
    """The one base operator of a metric: Zhuravlyov on all features, the others raw on I."""
    if metric == "zhuravlyov":
        return OperatorConfig(label="ρ", metric=plugin(metric), features="all")
    return OperatorConfig(
        label="ρ", metric=plugin(metric), features="quantitative", normalizer=plugin("none")
    )


def study_config(metric: str, alpha: float) -> ExperimentConfig:
    """The default configuration with the experiment's metric, k range, zero value and α."""
    base = ExperimentConfig()
    return base.model_copy(
        update={
            "name": f"study-{metric}-alpha-{alpha}",
            "local_context": LocalContextConfig(operators=(operator(metric),)),
            "admissible_k": KS,
            "synthetic_features": base.synthetic_features.model_copy(update={"zero_value": True}),
            "additional_features": base.additional_features.model_copy(update={"alpha": alpha}),
        }
    )


def counts(dataset: Dataset, indices: np.ndarray) -> tuple[int, int, int]:
    """Size and class split of a set of objects."""
    y = dataset.class_index[indices]
    return int(indices.size), int((y == 0).sum()), int((y == 1).sum())


def scores(truth: np.ndarray, decisions: np.ndarray) -> dict[str, Any]:
    """Accuracy (refusals as errors), accuracy on the answered objects, refusals, in %."""
    metrics = classification_metrics(truth, decisions, 1)
    answered = metrics.accuracy_answered
    return {
        "accuracy": round(100 * metrics.accuracy, 2),
        "accuracy_answered": None if answered is None else round(100 * answered, 2),
        "refusals": metrics.refusals,
        "n": metrics.n,
    }


def evaluate(
    dataset: Dataset, train: np.ndarray, test: np.ndarray, metric: str, alpha: float
) -> dict[str, Any]:
    """Fit on the training sample; accuracy there (Definition 2) and on the test sample (3)."""
    config = study_config(metric, alpha)
    part = dataset.subset(train)
    try:
        model = fit_model(part, config)
    except ModelUndefinedError as error:
        return {"undefined": str(error)}
    training = scores(part.class_index + 1, model.classify_training().decisions)
    tested = scores(dataset.class_index[test] + 1, model.predict(dataset.X[test]))
    trace = model.trace
    return {
        "r": trace.r,
        "left_out_zero": sum(1 for s in trace.skipped_features if s.reason == "zero"),
        "tuplam": [trace.features[u].label for u in model.tuplam],
        "training": training,
        "test": tested,
    }


def knn(dataset: Dataset, train: np.ndarray, test: np.ndarray, metric: str) -> dict[str, float]:
    """k-NN majority vote on the same split, with the same metric."""
    config = study_config(metric, 0.1)
    data = BaselineInput(dataset.subset(train), dataset.X[test], None, config)
    outputs = knn_vote(data, KnnVoteParams(ks=KNN_KS))
    truth = dataset.class_index[test] + 1
    return {
        o.method: round(100 * classification_metrics(truth, o.decisions, 1).accuracy, 2)
        for o in outputs
    }


def write_result(results: Path, name: str, header: list[str], rows: list[list[Any]]) -> Path:
    """Write one result table: CSV with a byte-order mark, UTF-8, LF line ends (ADR-083)."""
    return write_csv(results / name, csv_rows(header, rows))


def run(out: Path = OUT) -> list[Path]:
    """Run the experiment and write its results under ``out``; returns the files written."""
    results = out / "results"
    results.mkdir(parents=True, exist_ok=True)
    samples_rows: list[list[Any]] = []
    binary_rows: list[list[Any]] = []
    accuracy_rows: list[list[Any]] = []
    knn_rows: list[list[Any]] = []
    summary: dict[str, Any] = {"version": __version__, "open_settings": OPEN_SETTINGS, "runs": []}

    for name, source in DATASETS.items():
        dataset = load_from_config(source, ROOT)
        split_metrics = ("zhuravlyov",) if name == "Heart" else tuple(SPLITS)
        for split_metric in split_metrics:
            for reading, keep in READINGS.items():
                rips = SPLITS[split_metric].model_copy(update={"keep_blocked_cover": keep})
                result = select_precedents(dataset, rips)
                train, test = result.training, result.test
                article = ARTICLE["samples"].get((name, split_metric))
                samples_rows.append(
                    [
                        name,
                        split_metric,
                        reading,
                        *counts(dataset, train),
                        *counts(dataset, test),
                        *(article or (None, None, None)),
                    ]
                )
                # Tables 1–5 on the Zhuravlyov split; Table 6 on the Euclid / Chebyshev splits
                metrics = (
                    METRICS
                    if split_metric == "zhuravlyov"
                    else ("euclidean", "chebyshev", "canberra")
                )
                for metric in metrics:
                    runs = {
                        alpha: evaluate(dataset, train, test, metric, alpha) for alpha in ALPHAS
                    }
                    first = runs[ALPHAS[0]]
                    if split_metric == "zhuravlyov" and "r" in first:
                        binary_rows.append(
                            [
                                name,
                                reading,
                                metric,
                                first["r"] + first["left_out_zero"],
                                first["r"],
                                ARTICLE["binary_features"].get((name, metric)),
                            ]
                        )
                    for alpha, run in runs.items():
                        if "undefined" in run:
                            accuracy_rows.append(
                                [
                                    name,
                                    split_metric,
                                    reading,
                                    metric,
                                    alpha,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    run["undefined"],
                                ]
                            )
                            continue
                        accuracy_rows.append(
                            [
                                name,
                                split_metric,
                                reading,
                                metric,
                                alpha,
                                run["training"]["accuracy"],
                                run["training"]["refusals"],
                                run["test"]["accuracy"],
                                run["test"]["accuracy_answered"],
                                run["test"]["refusals"],
                                " ".join(run["tuplam"]),
                                "",
                            ]
                        )
                        summary["runs"].append(
                            {
                                "dataset": name,
                                "split": split_metric,
                                "reading": reading,
                                "metric": metric,
                                "alpha": alpha,
                                **run,
                            }
                        )
                    for method, accuracy in knn(dataset, train, test, metric).items():
                        knn_rows.append([name, split_metric, reading, metric, method, accuracy])

    files = [
        write_result(
            results,
            "rips_samples.csv",
            [
                "dataset",
                "split_metric",
                "reading",
                "training",
                "training_K1",
                "training_K2",
                "test",
                "test_K1",
                "test_K2",
                "article_training",
                "article_K1",
                "article_K2",
            ],
            samples_rows,
        ),
        write_result(
            results,
            "table1_binary_features.csv",
            [
                "dataset",
                "reading",
                "metric",
                "synthetic_features",
                "binary_features",
                "article_binary_features",
            ],
            binary_rows,
        ),
        write_result(
            results,
            "tables2_6_accuracy.csv",
            [
                "dataset",
                "split_metric",
                "reading",
                "metric",
                "alpha",
                "training_accuracy",
                "training_refusals",
                "test_accuracy",
                "test_accuracy_answered",
                "test_refusals",
                "tuplam",
                "undefined",
            ],
            accuracy_rows,
        ),
        write_result(
            results,
            "knn_comparison.csv",
            ["dataset", "split_metric", "reading", "metric", "method", "test_accuracy"],
            knn_rows,
        ),
    ]
    summary["article"] = {
        "samples": {f"{k[0]}/{k[1]}": v for k, v in ARTICLE["samples"].items()},
        "binary_features": {f"{k[0]}/{k[1]}": v for k, v in ARTICLE["binary_features"].items()},
        "accuracy": {f"{k[0]}/{k[1]}": v for k, v in ARTICLE["accuracy"].items()},
        "table6": ARTICLE["table6"],
        "knn": ARTICLE["knn"],
    }
    summary_path = results / "summary.json"
    summary_path.write_bytes(
        (json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode()
    )
    files.append(summary_path)
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  results/{path.name}" for path in files
    ]
    checksums = out / "checksums.sha256"
    checksums.write_bytes(("\n".join(lines) + "\n").encode())
    return [*files, checksums]


def main() -> int:
    """Write the results to ``experiments/`` (or to the folder given as the first argument)."""
    started = time.perf_counter()
    files = run(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
    sys.stdout.write(f"experiments written in {time.perf_counter() - started:.1f} s\n")
    for path in files:
        sys.stdout.write(f"  {path.as_posix()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
