"""The experiment runner: fit, evaluate under every configured protocol, save a run folder.

:func:`run_experiment` fits the CS-model on the whole dataset (margins, model properties) and
evaluates it — with the configured baselines — under every protocol of ``evaluation.protocols``.
:func:`save_run` writes ``runs/<YYYYMMDD_HHMMSS>_<hash>/`` with the manifest (configuration,
hashes, seed, package versions, timings), ``results.json`` (metrics, TUPLAM, margins, properties),
``predictions.csv`` and ``folds.csv``. The CLI and the GUI call the same functions.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from context_synthetic_recognition.config.io import config_hash
from context_synthetic_recognition.config.models import ExperimentConfig
from context_synthetic_recognition.config.presets import active_deviations
from context_synthetic_recognition.core.meta import REFUSAL
from context_synthetic_recognition.core.model import CSModel, fit_model
from context_synthetic_recognition.core.properties import ModelProperties, model_properties
from context_synthetic_recognition.data.schema import Dataset, Label
from context_synthetic_recognition.errors import ConfigError
from context_synthetic_recognition.evaluation.margins import MarginAnalysis, margin_analysis
from context_synthetic_recognition.evaluation.metrics import ClassificationMetrics
from context_synthetic_recognition.evaluation.protocols import (
    MethodPredictions,
    ProtocolResult,
    run_protocol,
)
from context_synthetic_recognition.services.manifest import (
    build_manifest,
    create_run_dir,
    write_manifest,
)

Progress = Callable[[str, int, int], None]
"""Called with (stage, done, total) — e.g. ("leave-one-out", 3, 10)."""


def positive_code(classes: tuple[Label, ...], positive_class: Label) -> int:
    """The decision code (1 = K1, 2 = K2) of the configured positive class.

    Raises:
        ConfigError: The label is not one of the two classes.
    """
    for index, label in enumerate(classes):
        if label == positive_class or str(label) == str(positive_class):
            return index + 1
    raise ConfigError(
        f"evaluation.positive_class = {positive_class!r} is not a class of the dataset "
        f"(classes: {', '.join(map(str, classes))})"
    )


@dataclass(frozen=True, eq=False)
class ExperimentResult:
    """A fitted model and its evaluation."""

    config: ExperimentConfig
    dataset: Dataset
    model: CSModel
    """The CS-model fitted on every object."""
    protocols: tuple[ProtocolResult, ...]
    margins: MarginAnalysis
    properties: ModelProperties
    positive: int
    """Decision code of the positive class."""
    timings: dict[str, float] = field(default_factory=dict)
    """Seconds per stage."""

    def protocol(self, name: str) -> ProtocolResult:
        """The result of a protocol by its canonical name.

        Raises:
            KeyError: The protocol was not run.
        """
        for result in self.protocols:
            if result.protocol == name:
                return result
        raise KeyError(name)

    def metrics(self, predictions: MethodPredictions) -> ClassificationMetrics:
        """Metrics of a method with the configured positive class."""
        return predictions.metrics(self.positive)

    def auc(self, predictions: MethodPredictions) -> float:
        """AUC of a method (scores rounded to ``evaluation.score_decimals``)."""
        return predictions.roc(self.positive, self.config.evaluation.score_decimals).auc


def run_experiment(
    dataset: Dataset,
    config: ExperimentConfig | None = None,
    *,
    progress: Progress | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> ExperimentResult:
    """Fit the CS-model and evaluate it under every configured protocol.

    Raises:
        ModelUndefinedError: The model is undefined on the whole dataset.
        ConfigError: Invalid plug-in parameters or positive class.
        RegistryError: Unknown plug-in.
        EvaluationCancelledError: ``cancelled`` returned ``True``.
    """
    cfg = config or ExperimentConfig()
    evaluation = cfg.evaluation
    timings: dict[str, float] = {}
    started = time.perf_counter()
    model = fit_model(dataset, cfg)
    positive = positive_code(model.classes, evaluation.positive_class)
    timings["fit"] = round(time.perf_counter() - started, 3)
    results = []
    for spec in evaluation.protocols:
        began = time.perf_counter()

        def report(done: int, total: int, stage: str = spec.name) -> None:
            if progress is not None:
                progress(stage, done, total)

        results.append(
            run_protocol(
                dataset,
                cfg,
                spec,
                baselines=evaluation.baselines,
                progress=report,
                cancelled=cancelled,
            )
        )
        timings[results[-1].protocol] = round(time.perf_counter() - began, 3)
    began = time.perf_counter()
    margins = margin_analysis(model)
    properties = model_properties(model)
    timings["margins and properties"] = round(time.perf_counter() - began, 3)
    return ExperimentResult(
        cfg, dataset, model, tuple(results), margins, properties, positive, timings
    )


# ---------------------------------------------------------------- persistence


def _metrics_dict(result: ExperimentResult, predictions: MethodPredictions) -> dict[str, Any]:
    metrics = result.metrics(predictions)
    auc = result.auc(predictions)
    return {
        "accuracy": metrics.accuracy,
        "coverage": metrics.coverage,
        "accuracy_answered": metrics.accuracy_answered,
        "tp": metrics.tp,
        "tn": metrics.tn,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "refusals": metrics.refusals,
        "precision": [c.precision for c in metrics.classes],
        "recall": [c.recall for c in metrics.classes],
        "f1": [c.f1 for c in metrics.classes],
        "macro_f1": metrics.macro_f1,
        "auc": None if np.isnan(auc) else auc,
    }


def results_dict(result: ExperimentResult) -> dict[str, Any]:
    """The run's results as JSON-ready data (``results.json``)."""
    model, dataset = result.model, result.dataset
    return {
        "dataset": {
            "name": dataset.name,
            "hash": dataset.content_hash(),
            "m": dataset.m,
            "n": dataset.n,
            "classes": [str(c) for c in model.classes],
            "class_sizes": list(model.trace.class_sizes),
        },
        "config_hash": config_hash(result.config),
        "template_deviations": [d.adr for d in active_deviations(result.config)],
        "model": {
            "permitted_k": list(model.trace.permitted_k.ks),
            "r": model.trace.r,
            "tuplam": list(model.hag.names),
            "crit": list(model.hag.crit),
            "stop": model.hag.stop.value,
            "p": model.p,
        },
        "positive_class": str(model.classes[result.positive - 1]),
        "protocols": {
            p.protocol: {
                "undefined_folds": [f.number for f in p.undefined_folds],
                "methods": {m.method: _metrics_dict(result, m) for m in p.methods},
            }
            for p in result.protocols
        },
        "margins": [
            {"feature": f"r{j + 1}", "with_majorizer": a.width, "without_majorizer": b.width}
            for j, (a, b) in enumerate(
                zip(result.margins.with_majorizer, result.margins.without_majorizer, strict=True)
            )
        ],
        "properties": {
            "defined": result.properties.determinacy.defined,
            "tuplam_conflicts": result.properties.tuplam_conflicts,
            "psi_conflicts": result.properties.psi_conflicts,
        },
        "timings": result.timings,
    }


def _label(result: ExperimentResult, decision: int) -> str:
    return "" if decision == REFUSAL else str(result.model.classes[decision - 1])


def save_run(result: ExperimentResult, runs_dir: Path | None = None) -> Path:
    """Write the run folder and return it."""
    root = Path(runs_dir if runs_dir is not None else result.config.output.runs_dir)
    run_dir = create_run_dir(root, config_hash(result.config))
    manifest = build_manifest(
        result.config,
        run_id=run_dir.name,
        dataset_hash=result.dataset.content_hash(),
        timings=result.timings,
    )
    write_manifest(manifest, run_dir)
    text = json.dumps(results_dict(result), indent=2, ensure_ascii=False)
    (run_dir / "results.json").write_text(text + "\n", encoding="utf-8", newline="\n")
    ids = result.dataset.object_ids
    labels = result.dataset.y
    with (run_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["protocol", "method", "repeat", "object", "true", "decision", "predicted", "score"]
        )
        for protocol in result.protocols:
            for method in protocol.methods:
                for o, r, d, s in zip(
                    method.objects, method.repeats, method.decisions, method.scores, strict=True
                ):
                    writer.writerow(
                        [
                            protocol.protocol,
                            method.method,
                            int(r),
                            ids[int(o)],
                            labels[int(o)],
                            int(d),
                            _label(result, int(d)),
                            repr(float(s)),
                        ]
                    )
    with (run_dir / "folds.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "protocol",
                "fold",
                "repeat",
                "held out",
                "|K1|",
                "|K2|",
                "k",
                "r",
                "TUPLAM",
                "undefined",
            ]
        )
        for protocol in result.protocols:
            for fold in protocol.folds:
                writer.writerow(
                    [
                        protocol.protocol,
                        fold.number,
                        fold.split.repeat,
                        " ".join(ids[int(i)] for i in fold.split.test),
                        *(fold.class_sizes or ("", "")),
                        " ".join(map(str, fold.permitted_k or ())),
                        fold.r if fold.r is not None else "",
                        " ".join(fold.tuplam_names or ()),
                        fold.undefined or "",
                    ]
                )
    return run_dir
