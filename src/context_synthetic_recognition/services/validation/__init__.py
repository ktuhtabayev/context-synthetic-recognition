"""Validation against the author's workbooks: cached cell values vs. what the package computes.

Three layouts are recognised from their sheet names (:class:`WorkbookKind`):

- the **experiment** workbook (Steps 1–12, :mod:`.experiment`);
- the **additional-features check workbook** — the agglomerative algorithm on its own data
  (:mod:`.check_workbooks`);
- the **meta-algorithm check workbook** (:mod:`.check_workbooks`).

The workbooks' sheet names are quoted in :mod:`.sheets` only.

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
from context_synthetic_recognition.services.validation import sheets
from context_synthetic_recognition.services.validation.check_workbooks import (
    additional_features_workbook_checks,
    meta_algorithm_workbook_checks,
    validate_additional_features_workbook,
    validate_meta_algorithm_workbook,
)
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
from context_synthetic_recognition.services.validation.subject import ExperimentSubject

__all__ = [
    "DEFAULT_TOLERANCE",
    "Check",
    "CheckResult",
    "ExperimentSubject",
    "ValidationError",
    "ValidationReport",
    "WorkbookKind",
    "additional_features_workbook_checks",
    "experiment_subject",
    "meta_algorithm_workbook_checks",
    "run_check",
    "validate_workbook",
    "workbook_checks",
    "workbook_kind",
]


class WorkbookKind(StrEnum):
    """Layouts :func:`validate_workbook` knows."""

    EXPERIMENT = "experiment"
    """The full CS-model experiment (Steps 1–12)."""
    ADDITIONAL_FEATURES = "additional-features"
    """The additional-features check workbook (the agglomerative algorithm on its own data)."""
    META_ALGORITHM = "meta-algorithm"
    """The meta-algorithm check workbook."""


def workbook_kind(sheet_names: list[str]) -> WorkbookKind:
    """Recognise a layout from its sheet names.

    Raises:
        ValidationError: None of the known layouts.
    """
    names = set(sheet_names)
    if "Parameters" in names and "Dataset" in names:
        return WorkbookKind.EXPERIMENT
    if {sheets.CONTRIBUTION_INPUT, sheets.agglomerative_step(1)} <= names:
        return WorkbookKind.ADDITIONAL_FEATURES
    if {sheets.NEW_OBJECT, sheets.META_ALGORITHM} <= names:
        return WorkbookKind.META_ALGORITHM
    raise ValidationError(
        "not a known workbook: expected the CS-model experiment or one of the check workbooks "
        "(additional features, meta-algorithm)"
    )


_VALIDATORS = {
    WorkbookKind.EXPERIMENT: validate_experiment,
    WorkbookKind.ADDITIONAL_FEATURES: validate_additional_features_workbook,
    WorkbookKind.META_ALGORITHM: validate_meta_algorithm_workbook,
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
        path: The experiment workbook or a check workbook.
        config: Configuration; ``None``: the workbook's own settings (the ``excel-experiment``
            preset, for the experiment with the parameters of its *Parameters* sheet).
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
