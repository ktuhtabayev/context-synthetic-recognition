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
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from context_synthetic_recognition.config.io import config_hash
from context_synthetic_recognition.config.models import CentreMode, ExperimentConfig
from context_synthetic_recognition.config.presets import active_deviations
from context_synthetic_recognition.core.meta import REFUSAL
from context_synthetic_recognition.core.model import CSModel, fit_model
from context_synthetic_recognition.core.properties import ModelProperties, model_properties
from context_synthetic_recognition.data.loaders import load_from_config
from context_synthetic_recognition.data.schema import Dataset, Label
from context_synthetic_recognition.data.snapshot import SNAPSHOT_NAME, load_snapshot, save_snapshot
from context_synthetic_recognition.errors import ConfigError, CSRError, DatasetError
from context_synthetic_recognition.evaluation.margins import MarginAnalysis, margin_analysis
from context_synthetic_recognition.evaluation.metrics import ClassificationMetrics
from context_synthetic_recognition.evaluation.protocols import (
    MethodPredictions,
    ProtocolResult,
    run_protocol,
)
from context_synthetic_recognition.services.manifest import (
    RUN_ID_PATTERN,
    RunManifest,
    build_manifest,
    create_run_dir,
    read_manifest,
    write_manifest,
)

if TYPE_CHECKING:
    from context_synthetic_recognition.services.sensitivity import SwitchVariant

Progress = Callable[[str, int, int], None]
"""Called with (stage, done, total) — e.g. ("leave-one-out", 3, 10)."""
RESULTS_NAME = "results.json"
"""The results file of a run folder."""


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


def results_dict(
    result: ExperimentResult, sensitivity: Sequence[SwitchVariant] | None = None
) -> dict[str, Any]:
    """The run's results as JSON-ready data (``results.json``).

    ``sensitivity`` — the four switch settings, if they were evaluated — is stored under the key
    of the same name, so that a later export does not have to evaluate them again.
    """
    model, dataset = result.model, result.dataset
    extra: dict[str, Any] = {}
    if sensitivity is not None:
        extra["sensitivity"] = [
            {**asdict(variant), "centres": variant.centres.value, "crit": list(variant.crit)}
            for variant in sensitivity
        ]
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
        **extra,
    }


def _label(result: ExperimentResult, decision: int) -> str:
    return "" if decision == REFUSAL else str(result.model.classes[decision - 1])


def save_run(
    result: ExperimentResult,
    runs_dir: Path | None = None,
    *,
    sensitivity: Sequence[SwitchVariant] | None = None,
) -> Path:
    """Write the run folder and return it.

    Besides the manifest and the results the folder holds ``dataset.json``, the dataset exactly
    as it was used, so that the run can be repeated and exported later (:func:`load_run`).
    """
    root = Path(runs_dir if runs_dir is not None else result.config.output.runs_dir)
    run_dir = create_run_dir(root, config_hash(result.config))
    manifest = build_manifest(
        result.config,
        run_id=run_dir.name,
        dataset_hash=result.dataset.content_hash(),
        timings=result.timings,
    )
    write_manifest(manifest, run_dir)
    save_snapshot(result.dataset, run_dir)
    text = json.dumps(results_dict(result, sensitivity), indent=2, ensure_ascii=False)
    (run_dir / RESULTS_NAME).write_text(text + "\n", encoding="utf-8", newline="\n")
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


# ---------------------------------------------------------------- reading a run back


class RunError(CSRError, RuntimeError):
    """A run folder cannot be read back (missing files, or the data no longer match)."""


@dataclass(frozen=True, eq=False)
class LoadedRun:
    """A run repeated from its folder."""

    folder: Path
    manifest: RunManifest
    result: ExperimentResult
    sensitivity: tuple[SwitchVariant, ...] | None
    """The four switch settings as stored with the run, or ``None`` if they were not evaluated."""
    warnings: tuple[str, ...] = ()
    """Differences between the stored results and the repeated ones (another package version)."""


def latest_run(runs_dir: Path) -> Path | None:
    """The most recent run folder of ``runs_dir`` (by its time stamp), or ``None``."""
    if not runs_dir.is_dir():
        return None
    runs = sorted(
        folder
        for folder in runs_dir.iterdir()
        if folder.is_dir() and RUN_ID_PATTERN.match(folder.name)
    )
    return runs[-1] if runs else None


def _stored_sensitivity(stored: dict[str, Any]) -> tuple[SwitchVariant, ...] | None:
    # services.sensitivity imports this module, so its type is looked up here
    from context_synthetic_recognition.services.sensitivity import SwitchVariant

    variants = stored.get("sensitivity")
    if not variants:
        return None
    try:
        return tuple(
            SwitchVariant(
                **{
                    **variant,
                    "centres": CentreMode(variant["centres"]),
                    "crit": tuple(variant["crit"]),
                }
            )
            for variant in variants
        )
    except (KeyError, TypeError, ValueError):
        return None


def _differences(stored: dict[str, Any], result: ExperimentResult) -> list[str]:
    """How the repeated run differs from the stored results (empty if it reproduces them)."""
    repeated = results_dict(result)
    differences = []
    for key in ("permitted_k", "r", "tuplam", "p"):
        before, now = stored.get("model", {}).get(key), repeated["model"][key]
        if before is not None and before != now:
            differences.append(f"model.{key}: stored {before}, repeated {now}")
    for protocol, entry in stored.get("protocols", {}).items():
        before = entry.get("methods", {}).get("CS-model", {}).get("accuracy")
        now = repeated["protocols"].get(protocol, {}).get("methods", {}).get("CS-model", {})
        if before is not None and now and abs(before - now["accuracy"]) > 1e-12:
            differences.append(
                f"{protocol}: accuracy stored {before:.6g}, repeated {now['accuracy']:.6g}"
            )
    return differences


def load_run(
    run_dir: Path,
    *,
    progress: Progress | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> LoadedRun:
    """Repeat a run from its folder: the manifest's configuration on the stored dataset.

    The computation is deterministic, so the repeated run has the full trace the exporters need
    (the folder itself keeps only the results). The dataset is read from the folder's
    ``dataset.json``; a folder without it (written by version 0.4) falls back to the
    configuration's ``dataset.path``. In both cases the content hash must equal the manifest's.

    Raises:
        RunError: The folder is not a run folder, its dataset is missing or has changed.
        ConfigError: The manifest cannot be read.
    """
    if not run_dir.is_dir():
        raise RunError(f"{run_dir}: not a run folder")
    manifest = read_manifest(run_dir)
    config = manifest.config
    if (run_dir / SNAPSHOT_NAME).is_file():
        dataset = load_snapshot(run_dir)
    else:
        try:
            dataset = load_from_config(config.dataset)
        except DatasetError as error:
            raise RunError(
                f"{run_dir.name}: the run folder has no {SNAPSHOT_NAME} and its dataset cannot be "
                f"found from the configuration ({error})"
            ) from error
    if manifest.dataset_hash is not None and dataset.content_hash() != manifest.dataset_hash:
        raise RunError(
            f"{run_dir.name}: the dataset is not the one the run was computed on "
            "(its content hash differs from the manifest's)"
        )
    stored: dict[str, Any] = {}
    results_file = run_dir / RESULTS_NAME
    if results_file.is_file():
        try:
            stored = json.loads(results_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RunError(f"{results_file}: cannot read the results ({error})") from error
    result = run_experiment(dataset, config, progress=progress, cancelled=cancelled)
    return LoadedRun(
        folder=run_dir,
        manifest=manifest,
        result=result,
        sensitivity=_stored_sensitivity(stored),
        warnings=tuple(_differences(stored, result)),
    )
