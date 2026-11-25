"""The tables and files of the evaluation designs (ADR-095, ADR-096) — no I/O.

For every design and split: accuracy, balanced accuracy, refusals, the majority-class rate of the
classified objects and the sizes; a repeated design adds the standard deviation (n − 1) and the
minimum and maximum over its rounds, and the random control the percentile of the sample as given.
The files of ``experiments/designs/<design>/`` are rendered here and written by the service.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from context_synthetic_recognition.export.tables import Cell, Table
from context_synthetic_recognition.export.text import (
    csv_bytes,
    csv_text,
    json_text,
    latex_table,
    markdown_text,
)
from context_synthetic_recognition.services.designs import (
    DESIGNS,
    CaseResult,
    DesignResult,
    EvaluationDesign,
    SplitDesignResult,
    Statistic,
)

GROUP = "Evaluation designs"
DECIMALS = 2
CHECKSUMS = "checksums.sha256"
ROUNDS_KEY = "rounds"
ALL_DESIGNS_KEY = "all-designs"


def _mean(values: Sequence[float]) -> Cell:
    if not values:
        return None
    mean = sum(values) / len(values)
    return int(mean) if float(mean).is_integer() else round(mean, 1)


def _note(case: CaseResult, method: str) -> str:
    undefined = case.undefined(method)
    if not undefined:
        return ""
    reasons = sorted({r.undefined.get(method, "") for r in case.rounds} - {""})
    where = (
        "undefined"
        if undefined == len(case.rounds)
        else f"undefined on {undefined} of {len(case.rounds)} rounds"
    )
    return f"{where}: {'; '.join(reasons)}" if reasons else where


def _stat_cells(stat: Statistic | None, repeated: bool) -> tuple[Cell, ...]:
    if stat is None:
        return (None, None, None, None) if repeated else (None,)
    return (stat.mean, stat.sd, stat.low, stat.high) if repeated else (stat.mean,)


def _has_cases(result: DesignResult) -> bool:
    return any(c.title for s in result.splits for c in s.cases)


def design_columns(design: EvaluationDesign, result: DesignResult) -> tuple[str, ...]:
    """The columns of a design's table."""
    columns = ["Split"]
    if _has_cases(result):
        columns.append("Case")
    columns.append("Method")
    if design.repeated:
        columns.append("Rounds")
    columns += ["Training", "Classified", "Majority rate, %"]
    if design.repeated:
        columns += [
            "Accuracy, %",
            "SD",
            "Min",
            "Max",
            "Balanced accuracy, %",
            "Balanced SD",
            "Balanced min",
            "Balanced max",
            "Refusals",
        ]
    else:
        columns += ["Accuracy, %", "Balanced accuracy, %", "Refusals"]
    if design.key == "random-control":
        columns.append("Percentile of the given sample")
    columns.append("Note")
    return tuple(columns)


def _rows(design: EvaluationDesign, result: DesignResult) -> list[tuple[Cell, ...]]:
    cases = _has_cases(result)
    width = len(design_columns(design, result))
    rows: list[tuple[Cell, ...]] = []
    for split in result.splits:
        ranks = split.extras.get("percentile", {})
        for case in split.cases:
            lead: list[Cell] = [split.label, *([case.title] if cases else [])]
            if case.unavailable:
                rows.append((*lead, "—", *(None,) * (width - len(lead) - 2), case.unavailable))
                continue
            for key, title in case.methods:
                rounds = [r for r in case.rounds if r.metrics]
                row: list[Cell] = [*lead, title]
                if design.repeated:
                    row.append(len(rounds))
                row += [
                    _mean([r.training_sizes[0] for r in rounds]),
                    _mean([r.classified_sizes[0] for r in rounds]),
                    _mean([round(100 * r.majority_rate, 2) for r in rounds]),
                ]
                row += _stat_cells(Statistic.of(case.accuracies(key)), design.repeated)
                row += _stat_cells(Statistic.of(case.balanced(key)), design.repeated)
                refusals = case.refusals(key)
                row.append(_mean([float(r) for r in refusals]) if refusals else None)
                if design.key == "random-control":
                    row.append(ranks.get(key) if case.key == "random" else None)
                row.append(_note(case, key))
                rows.append(tuple(row))
    return rows


def design_table(design: EvaluationDesign, result: DesignResult) -> Table:
    """A design's table: every split, case and method."""
    repeated = (
        " Repeated: the mean over the rounds, the standard deviation (n − 1), the minimum and the "
        "maximum."
        if design.repeated
        else ""
    )
    return Table(
        key=design.key,
        title=f"{design.title}: accuracy and balanced accuracy, %",
        columns=design_columns(design, result),
        rows=tuple(_rows(design, result)),
        group=GROUP,
        note=(
            f"{design.explanation} {design.settings} Training and Classified: the numbers of "
            "objects (the mean over the rounds). Majority rate: the share of the larger class "
            "among the classified objects. Balanced accuracy: the mean recall of the two classes. "
            f"Refusals count as errors (Definition 3).{repeated}"
        ).replace("  ", " "),
        labels=3 if _has_cases(result) else 2,
    )


def rounds_table(design: EvaluationDesign, result: DesignResult) -> Table:
    """Every round of a repeated design: the sizes and every method's accuracy."""
    rows: list[tuple[Cell, ...]] = []
    for split in result.splits:
        for case in split.cases:
            for r in case.rounds:
                for key, title in case.methods:
                    metrics = r.metrics.get(key)
                    rows.append(
                        (
                            split.label,
                            case.title,
                            r.label,
                            title,
                            r.training_sizes[0] if r.training_sizes else None,
                            r.classified_sizes[0] if r.classified_sizes else None,
                            None if metrics is None else round(100 * metrics.accuracy, 2),
                            None if metrics is None else round(100 * metrics.macro_recall, 2),
                            None if metrics is None else metrics.refusals,
                        )
                    )
    return Table(
        key=ROUNDS_KEY,
        title=f"{design.title}: every round",
        columns=(
            "Split",
            "Case",
            "Round",
            "Method",
            "Training",
            "Classified",
            "Accuracy, %",
            "Balanced accuracy, %",
            "Refusals",
        ),
        rows=tuple(rows),
        group=GROUP,
        note=f"{design.settings} — = the method is undefined on that round.",
        labels=4,
    )


def _main(design: EvaluationDesign, split: SplitDesignResult) -> CaseResult | None:
    found = [c for c in split.cases if not c.unavailable]
    chosen = next((c for c in found if c.key == design.main_case), None)
    return chosen or (found[0] if found else None)


def all_designs_table(results: Mapping[str, DesignResult]) -> Table:
    """Method × design for every split: the accuracy and the balanced accuracy of each design run.

    A design of several cases is shown by its main case (the random samples, the folds, 50 % of
    the training sample, the first target split).
    """
    chosen = [(DESIGNS.get(key), result) for key, result in results.items() if key in DESIGNS]
    columns = ["Split", "Method"]
    for design, _ in chosen:
        columns += [f"{design.title} · accuracy, %", f"{design.title} · balanced, %"]
    order: dict[tuple[str, str], dict[str, Cell]] = {}
    titles: dict[str, str] = {}
    for design, result in chosen:
        for split in result.splits:
            case = _main(design, split)
            if case is None:
                continue
            for key, title in case.methods:
                titles[key] = title
                cells = order.setdefault((split.label, key), {})
                accuracy = Statistic.of(case.accuracies(key))
                balanced = Statistic.of(case.balanced(key))
                cells[f"{design.key}:a"] = None if accuracy is None else accuracy.mean
                cells[f"{design.key}:b"] = None if balanced is None else balanced.mean
    rows = tuple(
        (
            label,
            titles[key],
            *(cells.get(f"{design.key}:{part}") for design, _ in chosen for part in ("a", "b")),
        )
        for (label, key), cells in order.items()
    )
    shown = "; ".join(
        f"{d.title}: {c}"
        for d, c in (
            (
                design,
                next((c.title for s in r.splits for c in s.cases if c.key == design.main_case), ""),
            )
            for design, r in chosen
        )
        if c
    )
    return Table(
        key=ALL_DESIGNS_KEY,
        title="All designs: accuracy and balanced accuracy, %",
        columns=tuple(columns),
        rows=rows,
        group=GROUP,
        note=(
            "The designs run so far, on the ticked splits; a repeated design by its mean."
            + (f" Shown: {shown}." if shown else "")
        ),
        labels=2,
    )


def all_designs_files(results: Mapping[str, DesignResult]) -> dict[str, bytes]:
    """The *All designs* table as files: CSV, Markdown, and their checksums last."""
    table = all_designs_table(results)
    files = {
        f"{ALL_DESIGNS_KEY}.csv": csv_bytes(csv_text(table)),
        f"{ALL_DESIGNS_KEY}.md": markdown_text(table, DECIMALS, level=2).encode("utf-8"),
    }
    files[CHECKSUMS] = "".join(f"{sha256(data)}  {name}\n" for name, data in files.items()).encode()
    return files


def _statistic(stat: Statistic | None) -> dict[str, float | None] | None:
    if stat is None:
        return None
    return {"mean": stat.mean, "sd": stat.sd, "min": stat.low, "max": stat.high}


def summary_data(
    design: EvaluationDesign,
    result: DesignResult,
    *,
    versions: Mapping[str, str],
    inputs: Mapping[str, tuple[str, str]],
) -> dict[str, Any]:
    """Settings, library versions and every number of a design, for ``summary.json``.

    Args:
        design: The design.
        result: Its run.
        versions: Versions of the libraries the numbers depend on.
        inputs: Split key → (its file, the SHA-256 of the file).
    """
    splits = []
    for split in result.splits:
        file, digest = inputs.get(split.key, ("", ""))
        cases: list[dict[str, Any]] = []
        for case in split.cases:
            if case.unavailable:
                cases.append(
                    {"key": case.key, "title": case.title, "unavailable": case.unavailable}
                )
                continue
            rounds = [r for r in case.rounds if r.metrics]
            cases.append(
                {
                    "key": case.key,
                    "title": case.title,
                    "rounds": len(case.rounds),
                    "training": [r.training_sizes[0] for r in rounds],
                    "classified": [r.classified_sizes[0] for r in rounds],
                    "majority_rate": _mean([round(100 * r.majority_rate, 2) for r in rounds]),
                    "methods": [
                        {
                            "key": key,
                            "title": title,
                            "accuracy": _statistic(Statistic.of(case.accuracies(key))),
                            "balanced_accuracy": _statistic(Statistic.of(case.balanced(key))),
                            "refusals": case.refusals(key),
                            "undefined_rounds": case.undefined(key),
                            "note": _note(case, key),
                        }
                        for key, title in case.methods
                    ],
                    "notes": sorted({r.note for r in case.rounds if r.note}),
                }
            )
        splits.append(
            {
                "key": split.key,
                "label": split.label,
                "file": file,
                "file_sha256": digest,
                "cases": cases,
                "extras": dict(split.extras),
            }
        )
    return {
        "design": design.key,
        "title": design.title,
        "group": design.group,
        "explanation": design.explanation,
        "settings": design.settings,
        "description": (
            "An evaluation design on the article's split files (ADR-095): every method of the "
            "comparison with fixed settings, the classified objects without their class. "
            "Percentages with two decimals."
        ),
        "versions": dict(versions),
        "splits": splits,
    }


def sha256(content: bytes) -> str:
    """SHA-256 of some bytes."""
    return hashlib.sha256(content).hexdigest()


def design_files(
    design: EvaluationDesign,
    result: DesignResult,
    *,
    versions: Mapping[str, str],
    inputs: Mapping[str, tuple[str, str]],
) -> dict[str, bytes]:
    """The files of a design's folder (UTF-8, LF; the CSV files with a byte-order mark, ADR-083).

    ``<design>.csv | md | tex``, ``rounds.csv`` (repeated designs), ``summary.json`` and
    ``checksums.sha256`` last.
    """
    table = design_table(design, result)
    texts = {
        f"{design.key}.csv": csv_text(table),
        f"{design.key}.md": markdown_text(table, DECIMALS, level=2),
        f"{design.key}.tex": latex_table(table, DECIMALS),
    }
    if design.repeated:
        texts[f"{ROUNDS_KEY}.csv"] = csv_text(rounds_table(design, result))
    texts["summary.json"] = json_text(
        summary_data(design, result, versions=versions, inputs=inputs)
    )
    files = {
        name: csv_bytes(text) if name.endswith(".csv") else text.encode("utf-8")
        for name, text in texts.items()
    }
    files[CHECKSUMS] = "".join(f"{sha256(data)}  {name}\n" for name, data in files.items()).encode()
    return files
