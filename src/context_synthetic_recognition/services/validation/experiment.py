"""Validation of the experiment workbook, Steps 1–12.

The inputs come from the workbook itself: the data from its *Dataset* sheet, the parameters
of the agglomerative algorithm and its two switches from the *Parameters* sheet, and the new
object typed on the new-object sheet. The package fits the CS-model, classifies the new object and
every training object, and the map (:func:`workbook_checks`) compares every computed cell with the
workbook's cached values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from context_synthetic_recognition.config.models import (
    AdditionalFeaturesConfig,
    ClassMeans,
    ExperimentConfig,
    plugin,
)
from context_synthetic_recognition.config.presets import PresetName, active_deviations, preset
from context_synthetic_recognition.core.model import CSModel, fit_model
from context_synthetic_recognition.core.properties import model_properties
from context_synthetic_recognition.core.trace import PsiTrace
from context_synthetic_recognition.data.loaders import load_dataset
from context_synthetic_recognition.evaluation.margins import margin_analysis
from context_synthetic_recognition.evaluation.protocols import run_protocol
from context_synthetic_recognition.services.sensitivity import switch_sensitivity
from context_synthetic_recognition.services.validation import sheets
from context_synthetic_recognition.services.validation.checks import (
    Check,
    ValidationError,
    ValidationReport,
    find_sheet,
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
NEW_OBJECT = sheets.NEW_OBJECT


@dataclass(frozen=True)
class _Cell:
    ref: str
    what: str


_AGGLOMERATIVE_CELLS = {
    "alpha": _Cell("B19", "α"),
    "delta": _Cell("B20", "δ"),
    "kappa": _Cell("B21", "ϰ"),
    "cr1": _Cell("B22", "cr1"),
    "class_means": _Cell("B24", "⚠1 class means (1 = running, 2 = final)"),
    "majorizing_passes": _Cell(
        "B25", "⚠2 majorizing passes in Step 4 (1 = the article, 2 = twice)"
    ),
}
_CENTRES = {1: ClassMeans.RUNNING, 2: ClassMeans.FINAL}


def _trace(subject: ExperimentSubject) -> PsiTrace:
    return subject.trace


def workbook_checks() -> list[Check[ExperimentSubject]]:
    """The map of the experiment workbook — Steps 1–12 and the evaluation — in sheet order."""
    return [
        *(check.on(_trace) for check in context_checks()),
        *model_checks(),
        *evaluation_checks(),
    ]


def read_agglomerative_parameters(workbook: Any) -> dict[str, Any]:
    """α, δ, ϰ, cr1 and the two switches as typed on the *Parameters* sheet.

    Raises:
        ValidationError: A value is missing or not allowed.
    """
    sheet = workbook[find_sheet(workbook.sheetnames, PARAMETERS) or PARAMETERS]
    values: dict[str, Any] = {}
    for key, where in _AGGLOMERATIVE_CELLS.items():
        value = sheet[where.ref].value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValidationError(f"{PARAMETERS}!{where.ref} ({where.what}): {value!r}")
        values[key] = value
    if values["class_means"] not in _CENTRES or values["majorizing_passes"] not in (1, 2):
        raise ValidationError(
            f"{PARAMETERS}!B24:B25: the switches must be 1 or 2, got "
            f"{values['class_means']!r} and {values['majorizing_passes']!r}"
        )
    values["class_means"] = _CENTRES[int(values["class_means"])]
    values["majorizing_passes"] = int(values["majorizing_passes"])
    values["kappa"] = int(values["kappa"])
    return values


def _agglomerative_text(agglomerative: AdditionalFeaturesConfig) -> str:
    return (
        f"α = {agglomerative.alpha:g}, δ = {agglomerative.delta:g}, ϰ = {agglomerative.kappa}, "
        f"cr1₀ = {agglomerative.cr1:g}, class means = {agglomerative.class_means.value}, "
        f"majorizing passes = {agglomerative.majorizing_passes}"
    )


def experiment_config(
    workbook: Any, config: ExperimentConfig | None
) -> tuple[ExperimentConfig, list[str]]:
    """The configuration to validate with, and notes for the report.

    Without ``config`` the ``excel-experiment`` preset is used with the workbook's own parameters
    of the agglomerative algorithm (its cached values were calculated with them). A given
    ``config`` is used as it is; a note warns if its settings of the agglomerative algorithm differ
    from the workbook's.
    """
    params = read_agglomerative_parameters(workbook)
    base = config or preset(PresetName.EXCEL_EXPERIMENT)
    workbook_agglomerative = AdditionalFeaturesConfig.model_validate(
        {**base.additional_features.model_dump(), **params}
    )
    notes = [
        (
            "agglomerative algorithm settings of the workbook (Parameters sheet): "
            f"{_agglomerative_text(workbook_agglomerative)}"
        )
    ]
    if config is None:
        chosen = base.model_copy(update={"additional_features": workbook_agglomerative})
    else:
        chosen = config
        if config.additional_features != workbook_agglomerative:
            notes.append(
                "⚠ the configuration's agglomerative algorithm settings differ from the "
                f"workbook's: {_agglomerative_text(config.additional_features)}"
            )
    notes += [
        f"⚠ deviates from the article ({d.adr}): {d.statement}" for d in active_deviations(chosen)
    ]
    if chosen.synthetic_features.skip_constant:
        notes.append(
            "⚠ synthetic.skip_constant is on: the workbook keeps constant synthetic features "
            "(ADR-030), so leave-one-out folds can differ"
        )
    return chosen, notes


def read_new_object(workbook: Any, n: int, m: int) -> tuple[Any, int | None]:
    """The new object typed on the workbook's new-object sheet and the training object it excludes.

    Returns:
        The n feature values (row "x (input)") and the 0-based training index left out of the
        object's context, or ``None`` for a genuinely new object (cell B35 = 0).

    Raises:
        ValidationError: A value is missing, or the excluded object does not exist.
    """
    sheet = workbook[find_sheet(workbook.sheetnames, NEW_OBJECT) or NEW_OBJECT]
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
    if trace.admissible_k.count != WORKBOOK_KS:
        problems.append(
            f"{trace.admissible_k.count} admissible k (the layout has a column for {WORKBOOK_KS})"
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
        if find_sheet(workbook.sheetnames, sheet) is None:
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
