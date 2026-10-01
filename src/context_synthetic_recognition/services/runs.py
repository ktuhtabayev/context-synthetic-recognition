"""Looking at saved runs side by side.

A run folder holds what identifies a run (``manifest.yaml``) and what it found (``results.json``).
:func:`list_runs` reads the folders of a runs directory, :func:`comparison` puts the key facts of
several runs into one table and :func:`config_differences` lists the settings in which their
configurations differ. Nothing is recomputed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from context_synthetic_recognition.config.io import to_data
from context_synthetic_recognition.config.presets import matching_preset
from context_synthetic_recognition.errors import ConfigError
from context_synthetic_recognition.evaluation.protocols import CS_MODEL
from context_synthetic_recognition.export.tables import Cell, Table
from context_synthetic_recognition.services.manifest import (
    RUN_ID_PATTERN,
    RunManifest,
    build_manifest,
    read_manifest,
)
from context_synthetic_recognition.services.runner import (
    RESULTS_NAME,
    ExperimentResult,
    RunError,
    results_dict,
)

CURRENT = "current"
"""Run id of a result that was not saved."""
FACT = "Fact"
SETTING = "Setting"


@dataclass(frozen=True, eq=False)
class RunRecord:
    """A run as its folder describes it."""

    folder: Path | None
    """The run folder, or ``None`` for a result that was not saved."""
    manifest: RunManifest
    results: dict[str, Any]
    """The content of ``results.json`` (empty if the folder has none)."""

    @property
    def run_id(self) -> str:
        """Name of the run."""
        return self.manifest.run_id

    @property
    def created(self) -> datetime:
        """When the run was made (UTC)."""
        return self.manifest.created

    @property
    def dataset(self) -> str:
        """Name of the dataset."""
        return str(self.results.get("dataset", {}).get("name", "—"))

    def accuracy(self, protocol: str) -> float | None:
        """Accuracy of the CS-model under a protocol, or ``None`` if it was not run."""
        methods = self.results.get("protocols", {}).get(protocol, {}).get("methods", {})
        value = methods.get(CS_MODEL, {}).get("accuracy")
        return None if value is None else float(value)

    @property
    def protocols(self) -> tuple[str, ...]:
        """The protocols of the run."""
        return tuple(self.results.get("protocols", {}))


def read_run(folder: Path) -> RunRecord:
    """Read a run folder.

    Raises:
        RunError: Not a folder, or its results cannot be read.
        ConfigError: The manifest is missing or invalid.
    """
    if not folder.is_dir():
        raise RunError(f"{folder}: not a run folder")
    manifest = read_manifest(folder)
    results: dict[str, Any] = {}
    file = folder / RESULTS_NAME
    if file.is_file():
        try:
            results = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RunError(f"{file}: cannot read the results ({error})") from error
    return RunRecord(folder, manifest, results)


def list_runs(runs_dir: Path) -> tuple[list[RunRecord], list[str]]:
    """The runs of a directory, newest first, and why any folder could not be read."""
    records: list[RunRecord] = []
    problems: list[str] = []
    if not runs_dir.is_dir():
        return records, problems
    folders = sorted(
        (f for f in runs_dir.iterdir() if f.is_dir() and RUN_ID_PATTERN.match(f.name)),
        reverse=True,
    )
    for folder in folders:
        try:
            records.append(read_run(folder))
        except (RunError, ConfigError) as error:
            problems.append(str(error))
    return records, problems


def record_of(result: ExperimentResult, run_id: str = CURRENT) -> RunRecord:
    """A result that was not saved, in the form of a run record."""
    manifest = build_manifest(
        result.config,
        run_id=run_id,
        dataset_hash=result.dataset.content_hash(),
        timings=result.timings,
    )
    return RunRecord(None, manifest, results_dict(result))


def _percent(value: Any) -> Cell:
    return None if value is None else float(value)


def run_facts(record: RunRecord) -> list[tuple[str, Cell]]:
    """The key facts of a run, in a fixed order."""
    manifest, results = record.manifest, record.results
    config = manifest.config
    data = results.get("dataset", {})
    model = results.get("model", {})
    preset = matching_preset(config)
    facts: list[tuple[str, Cell]] = [
        ("Dataset", data.get("name")),
        ("Objects m", data.get("m")),
        ("Features n", data.get("n")),
        ("Dataset SHA-256", (manifest.dataset_hash or "")[:12] or None),
        ("Preset", preset.value if preset else "custom"),
        ("⚠ Class centres in θ, γ", config.hag.centres.value),
        ("⚠ STEP 4 passes", config.hag.step4_passes),
        ("k rule", config.k.name),
        ("Permitted k", ", ".join(str(k) for k in model.get("permitted_k", ())) or None),
        ("Synthetic features r", model.get("r")),
        ("TUPLAM", "{" + ", ".join(model["tuplam"]) + "}" if model.get("tuplam") else None),
        ("Latent features p", model.get("p")),
        ("HAG stopped because", model.get("stop")),
    ]
    for protocol, entry in results.get("protocols", {}).items():
        own = entry.get("methods", {}).get(CS_MODEL, {})
        facts += [
            (f"Accuracy · {protocol}", _percent(own.get("accuracy"))),
            (f"Coverage · {protocol}", _percent(own.get("coverage"))),
            (f"Refusals · {protocol}", own.get("refusals")),
            (f"Macro F1 · {protocol}", _percent(own.get("macro_f1"))),
            (f"AUC · {protocol}", _percent(own.get("auc"))),
        ]
        others = {
            name: method.get("accuracy")
            for name, method in entry.get("methods", {}).items()
            if name != CS_MODEL and method.get("accuracy") is not None
        }
        if others:
            best = max(others, key=lambda name: others[name])
            facts.append(
                (f"Best baseline · {protocol}", f"{best}: {100 * float(others[best]):.1f} %")
            )
    properties = results.get("properties", {})
    facts += [
        ("Definition 1: defined", properties.get("defined")),
        ("Definition 4: conflicts on (a₀ … a_p)", properties.get("tuplam_conflicts")),
        ("Seed", manifest.seed),
        ("Configuration SHA-256", manifest.config_hash[:12]),
        ("Package version", manifest.packages.get("context-synthetic-recognition")),
        ("Created (UTC)", manifest.created.strftime("%Y-%m-%d %H:%M")),
    ]
    return [(name, int(value) if isinstance(value, bool) else value) for name, value in facts]


def _columns(records: Sequence[RunRecord]) -> tuple[str, ...]:
    return tuple(record.run_id for record in records)


def comparison(records: Sequence[RunRecord]) -> Table:
    """The key facts of several runs side by side (one column per run)."""
    facts = [dict(run_facts(record)) for record in records]
    names: list[str] = []
    for record_facts in facts:
        names += [name for name in record_facts if name not in names]
    rows = tuple((name, *(record_facts.get(name) for record_facts in facts)) for name in names)
    return Table(
        "run-comparison",
        "Runs side by side",
        (FACT, *_columns(records)),
        rows,
        "Compare",
        "Accuracies and AUC are those of the CS-model; a refusal counts as an error.",
    )


def differing_rows(table: Table) -> set[int]:
    """The rows of a comparison whose values are not the same in every run."""
    return {r for r, row in enumerate(table.rows) if len(set(map(repr, row[1:]))) > 1}


def _flatten(value: Any, prefix: str = "") -> dict[str, Cell]:
    if isinstance(value, dict):
        flat: dict[str, Cell] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    if isinstance(value, list):
        return {prefix: json.dumps(value, ensure_ascii=False)}
    if isinstance(value, bool):
        return {prefix: str(value).lower()}
    return {prefix: value}


def config_differences(records: Sequence[RunRecord]) -> Table:
    """The settings in which the configurations of several runs differ."""
    flat = [_flatten(to_data(record.manifest.config)) for record in records]
    names: list[str] = []
    for settings in flat:
        names += [name for name in settings if name not in names]
    rows = tuple(
        (name, *(settings.get(name) for settings in flat))
        for name in names
        if len({repr(settings.get(name)) for settings in flat}) > 1
    )
    return Table(
        "run-config-differences",
        "Settings that differ",
        (SETTING, *_columns(records)),
        rows,
        "Compare",
        "Only the settings that are not the same in every run; an empty table means the "
        "configurations are identical.",
    )
