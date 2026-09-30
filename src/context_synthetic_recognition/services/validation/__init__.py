"""Validation against the author's workbooks: cached cell values vs. what the package computes.

Three layouts are recognised from their sheet names (:class:`WorkbookKind`):

- the **experiment** workbook (Steps 1–12, :mod:`.experiment`);
- the **HAG template** (``RegularizedStackingEnsembleWithHAG […]``, :mod:`.templates`);
- the **meta-algorithm template** (``Meta-algorithm […]``, :mod:`.templates`).

A declarative map pairs sheet ranges with values computed by the package; numbers must agree to
a tolerance (1e-9 by default), text exactly. ``csr validate`` and the golden tests use the same
maps.
"""

from __future__ import annotations

import importlib
import warnings
from enum import StrEnum
from pathlib import Path
from typing import Any

from context_synthetic_recognition.config.models import ExperimentConfig
from context_synthetic_recognition.services.validation.checks import (
    DEFAULT_TOLERANCE,
    Check,
    CheckResult,
    ValidationError,
    ValidationReport,
    run_check,
)
from context_synthetic_recognition.services.validation.experiment import (
    experiment_subject,
    validate_experiment,
    workbook_checks,
)
from context_synthetic_recognition.services.validation.model_map import ExperimentSubject
from context_synthetic_recognition.services.validation.templates import (
    hag_template_checks,
    meta_template_checks,
    validate_hag_template,
    validate_meta_template,
)

__all__ = [
    "DEFAULT_TOLERANCE",
    "Check",
    "CheckResult",
    "ExperimentSubject",
    "ValidationError",
    "ValidationReport",
    "WorkbookKind",
    "experiment_subject",
    "hag_template_checks",
    "meta_template_checks",
    "run_check",
    "validate_workbook",
    "workbook_checks",
    "workbook_kind",
]


class WorkbookKind(StrEnum):
    """Layouts :func:`validate_workbook` knows."""

    EXPERIMENT = "experiment"
    """The full CS-model experiment (Steps 1–12)."""
    HAG_TEMPLATE = "hag-template"
    """The HAG template of the author's earlier project."""
    META_TEMPLATE = "meta-template"
    """The meta-algorithm template of the author's earlier project."""


def workbook_kind(sheet_names: list[str]) -> WorkbookKind:
    """Recognise a layout from its sheet names.

    Raises:
        ValidationError: None of the known layouts.
    """
    names = set(sheet_names)
    if "Parameters" in names and "Dataset" in names:
        return WorkbookKind.EXPERIMENT
    if {"Dataset (Contribution & Weight)", "Greedy upon Weight (1-Latent)"} <= names:
        return WorkbookKind.HAG_TEMPLATE
    if {"Brace for Meta-algorithm", "Meta-algorithm"} <= names:
        return WorkbookKind.META_TEMPLATE
    raise ValidationError(
        "not a known workbook: expected the CS-model experiment or one of the HAG and "
        "meta-algorithm templates"
    )


_VALIDATORS = {
    WorkbookKind.EXPERIMENT: validate_experiment,
    WorkbookKind.HAG_TEMPLATE: validate_hag_template,
    WorkbookKind.META_TEMPLATE: validate_meta_template,
}


def validate_workbook(
    path: Path | str,
    config: ExperimentConfig | None = None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ValidationReport:
    """Validate the package against one of the author's workbooks.

    The inputs are read from the workbook itself; the expected values are its cached cell values
    (as last calculated by Excel or LibreOffice).

    Args:
        path: The experiment workbook or a template workbook.
        config: Configuration; the template preset by default (for the experiment together with
            the HAG parameters of its *Parameters* sheet).
        tolerance: Largest accepted absolute difference of numbers.

    Raises:
        ValidationError: The workbook cannot be read, is not a known layout, or its layout does
            not match the data.
        DatasetError: The experiment's *Dataset* sheet cannot be read.
    """
    file = Path(path)
    workbook = _open(file)
    try:
        kind = workbook_kind(workbook.sheetnames)
        return _VALIDATORS[kind](file, workbook, config, tolerance)
    finally:
        workbook.close()


def _open(file: Path) -> Any:
    openpyxl = importlib.import_module("openpyxl")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return openpyxl.load_workbook(file, data_only=True)
    except Exception as error:  # openpyxl raises many types for damaged files
        raise ValidationError(f"{file.name}: cannot read the workbook ({error})") from error
