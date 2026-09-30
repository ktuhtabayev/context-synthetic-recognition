"""Validation of the experiment workbook, Steps 1–12.

The inputs come from the workbook itself: the data from its *Dataset* sheet, the HAG parameters
and the two template/article switches from its *Parameters* sheet, and the new object typed on
*Brace for Meta-algorithm*. The package fits the CS-model, classifies the new object and every
training object, and the map (:func:`workbook_checks`) compares every computed cell with the
workbook's cached values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from context_synthetic_recognition.config.models import (
    CentreMode,
    ExperimentConfig,
    HAGConfig,
    plugin,
)
from context_synthetic_recognition.config.presets import active_deviations
from context_synthetic_recognition.core.model import CSModel, fit_model
from context_synthetic_recognition.core.properties import model_properties
from context_synthetic_recognition.core.trace import ContextTrace
from context_synthetic_recognition.data.loaders import load_dataset
from context_synthetic_recognition.evaluation.margins import margin_analysis
from context_synthetic_recognition.evaluation.protocols import run_protocol
from context_synthetic_recognition.services.sensitivity import switch_sensitivity
from context_synthetic_recognition.services.validation.checks import (
    Check,
    ValidationError,
    ValidationReport,
    read_numbers,
    run_checks,
)
from context_synthetic_recognition.services.validation.context_map import (
    WORKBOOK_KS,
    WORKBOOK_OBJECTS,
    WORKBOOK_OPERATORS,
    context_checks,
)
from context_synthetic_recognition.services.validation.evaluation_map import evaluation_checks
from context_synthetic_recognition.services.validation.model_map import (
    model_checks,
    model_layout_problems,
)
from context_synthetic_recognition.services.validation.subject import (
    ExperimentEvaluation,
    ExperimentSubject,
    NewObject,
)

PARAMETERS = "Parameters"
NEW_OBJECT = "Brace for Meta-algorithm"


@dataclass(frozen=True)
class _Cell:
    ref: str
    what: str


_HAG_CELLS = {
    "alpha": _Cell("B19", "α"),
    "delta": _Cell("B20", "δ"),
    "kappa": _Cell("B21", "ϰ"),
    "cr1": _Cell("B22", "cr1"),
    "centres": _Cell("B24", "⚠1 class centres (1 = running, 2 = final)"),
    "step4_passes": _Cell("B25", "⚠2 majorizer passes in STEP 4 (2 = template, 1 = article)"),
}
_CENTRES = {1: CentreMode.RUNNING, 2: CentreMode.FINAL}


def _trace(subject: ExperimentSubject) -> ContextTrace:
    return subject.trace


def workbook_checks() -> list[Check[ExperimentSubject]]:
    """The map of the experiment workbook — Steps 1–12 and the evaluation — in sheet order."""
    return [
        *(check.on(_trace) for check in context_checks()),
        *model_checks(),
        *evaluation_checks(),
    ]


def read_hag_parameters(workbook: Any) -> dict[str, Any]:
    """α, δ, ϰ, cr1 and the two switches as typed on the *Parameters* sheet.

    Raises:
        ValidationError: A value is missing or not allowed.
    """
    sheet = workbook[PARAMETERS]
    values: dict[str, Any] = {}
    for key, where in _HAG_CELLS.items():
        value = sheet[where.ref].value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValidationError(f"{PARAMETERS}!{where.ref} ({where.what}): {value!r}")
        values[key] = value
    if values["centres"] not in _CENTRES or values["step4_passes"] not in (1, 2):
        raise ValidationError(
            f"{PARAMETERS}!B24:B25: the switches must be 1 or 2, got "
            f"{values['centres']!r} and {values['step4_passes']!r}"
        )
    values["centres"] = _CENTRES[int(values["centres"])]
    values["step4_passes"] = int(values["step4_passes"])
    values["kappa"] = int(values["kappa"])
    return values


def _hag_text(hag: HAGConfig) -> str:
    return (
        f"α = {hag.alpha:g}, δ = {hag.delta:g}, ϰ = {hag.kappa}, cr1₀ = {hag.cr1:g}, "
        f"centres = {hag.centres.value}, STEP 4 passes = {hag.step4_passes}"
    )


def experiment_config(
    workbook: Any, config: ExperimentConfig | None
) -> tuple[ExperimentConfig, list[str]]:
    """The configuration to validate with, and notes for the report.

    Without ``config`` the template preset is used with the workbook's own HAG parameters (the
    cached values were calculated with them). A given ``config`` is used as it is; a note warns
    if its HAG settings differ from the workbook's.
    """
    params = read_hag_parameters(workbook)
    base = config or ExperimentConfig()
    workbook_hag = HAGConfig.model_validate({**base.hag.model_dump(), **params})
    notes = [f"HAG settings of the workbook (Parameters sheet): {_hag_text(workbook_hag)}"]
    if config is None:
        chosen = base.model_copy(update={"hag": workbook_hag})
    else:
        chosen = config
        if config.hag != workbook_hag:
            notes.append(
                "⚠ the configuration's HAG settings differ from the workbook's: "
                f"{_hag_text(config.hag)}"
            )
    notes += [f"⚠ template calculation ({d.adr}): {d.statement}" for d in active_deviations(chosen)]
    if chosen.synthetic.skip_constant:
        notes.append(
            "⚠ synthetic.skip_constant is on: the workbook keeps constant synthetic features "
            "(ADR-030), so leave-one-out folds can differ"
        )
    return chosen, notes


def read_new_object(workbook: Any, n: int, m: int) -> tuple[Any, int | None]:
    """The new object typed on *Brace for Meta-algorithm* and the training object it excludes.

    Returns:
        The n feature values (row "x (input)") and the 0-based training index left out of the
        object's context, or ``None`` for a genuinely new object (cell B35 = 0).

    Raises:
        ValidationError: A value is missing, or the excluded object does not exist.
    """
    sheet = workbook[NEW_OBJECT]
    last = chr(ord("A") + n)
    values = read_numbers(sheet, f"B31:{last}31", "new object")[0]
    exclude = sheet["B35"].value
    if isinstance(exclude, bool) or not isinstance(exclude, int | float) or exclude != int(exclude):
        raise ValidationError(f"{NEW_OBJECT}!B35 (excluded training object): {exclude!r}")
    if not 0 <= int(exclude) <= m:
        raise ValidationError(f"{NEW_OBJECT}!B35: there is no training object № {exclude}")
    return values, (int(exclude) - 1 if exclude else None)


def layout_problems(model: CSModel) -> list[str]:
    """Why the fitted model does not fit the workbook's fixed layout (empty if it does)."""
    trace = model.trace
    labels = tuple(operator.label for operator in trace.operators)
    problems = []
    if trace.m != WORKBOOK_OBJECTS:
        problems.append(f"{trace.m} objects (the layout has {WORKBOOK_OBJECTS})")
    if labels != WORKBOOK_OPERATORS:
        problems.append(f"operators {labels} (the layout has {WORKBOOK_OPERATORS})")
    if trace.permitted_k.count != WORKBOOK_KS:
        problems.append(
            f"{trace.permitted_k.count} permitted k (the layout has a column for {WORKBOOK_KS})"
        )
    return problems or model_layout_problems(model)


def experiment_subject(
    file: Path, workbook: Any, config: ExperimentConfig | None = None
) -> tuple[ExperimentSubject, list[str]]:
    """Fit the CS-model on the workbook's inputs and classify its new object and training objects.

    Returns:
        What the package computes for the workbook, and notes for the report (the settings used).

    Raises:
        ValidationError: The workbook's inputs cannot be read or its layout does not fit.
        DatasetError: The *Dataset* sheet cannot be read.
    """
    for sheet in (PARAMETERS, NEW_OBJECT):
        if sheet not in workbook.sheetnames:
            raise ValidationError(f"{file.name}: sheet '{sheet}' is missing")
    dataset = load_dataset(file, "cs-workbook")
    chosen, notes = experiment_config(workbook, config)
    model = fit_model(dataset, chosen)
    problems = layout_problems(model)
    if problems:
        raise ValidationError("the workbook layout does not match the data: " + "; ".join(problems))
    values, exclude = read_new_object(workbook, dataset.n, dataset.m)
    new_object = NewObject(
        values,
        exclude,
        model.classify(values, exclude=None if exclude is None else [exclude]),
    )
    evaluation = ExperimentEvaluation(
        leave_one_out=run_protocol(
            dataset, chosen, "leave-one-out", baselines=[plugin("knn-vote", ks=[3, 5])]
        ),
        margins=margin_analysis(model),
        properties=model_properties(model),
        sensitivity=switch_sensitivity(dataset, chosen),
    )
    subject = ExperimentSubject(
        model,
        new_object,
        model.classify_training(),
        chosen.evaluation.score_decimals,
        evaluation,
    )
    return subject, notes


def validate_experiment(
    file: Path, workbook: Any, config: ExperimentConfig | None, tolerance: float
) -> ValidationReport:
    """Validate the package against the experiment workbook (Steps 1–12).

    Raises:
        ValidationError: The workbook's inputs cannot be read or its layout does not fit.
        DatasetError: The *Dataset* sheet cannot be read.
    """
    subject, notes = experiment_subject(file, workbook, config)
    results = run_checks(workbook_checks(), workbook, subject, tolerance, file.name)
    return ValidationReport(str(file), tolerance, results, "experiment", tuple(notes))
