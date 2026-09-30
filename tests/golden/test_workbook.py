"""Golden tests: every Step 1–8 range of the Excel experiment against the package (tolerance 1e-9).

The expected values are the workbook's cached cell values; the map is the one ``csr validate``
uses (:func:`~context_synthetic_recognition.services.validation.workbook_checks`).
"""

import warnings
from collections.abc import Iterator
from typing import Any

import openpyxl
import pytest

from context_synthetic_recognition.core.context import ContextModel
from context_synthetic_recognition.services.validation import (
    DEFAULT_TOLERANCE,
    Check,
    run_check,
    validate_workbook,
    workbook_checks,
)

from ..conftest import WORKBOOK

pytestmark = pytest.mark.golden

CHECKS = workbook_checks()


@pytest.fixture(scope="module")
def workbook() -> Iterator[Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        book = openpyxl.load_workbook(WORKBOOK, data_only=True)
    yield book
    book.close()


@pytest.mark.parametrize("check", CHECKS, ids=[f"{c.sheet}!{c.ref}" for c in CHECKS])
def test_range_matches_the_workbook(
    check: Check, workbook: Any, experiment_model: ContextModel
) -> None:
    result = run_check(check, workbook[check.sheet], experiment_model.trace, DEFAULT_TOLERANCE)
    assert result.passed, "\n".join(result.mismatches)


def test_every_step_1_to_8_sheet_is_covered() -> None:
    sheets = {check.sheet for check in CHECKS}
    assert sheets == {
        "Parameters",
        "Normalized Dataset",
        "Zhuravlev Distances",
        "Sorted Neighbors (ρ)",
        "Sorted Neighbors (ρ_I)",
        "Sorted Neighbors (ρ_J)",
        "Synthetic Features (k-NN)",
        "Ψ(r) Binary Features",
        "Membership & Stability",
        "Informativeness ω",
        "Ψ(r) Contribution & Weight",
    }


def test_the_validation_report() -> None:
    report = validate_workbook(WORKBOOK)
    assert report.passed
    assert report.cells == 4022
    assert len(report.results) == len(CHECKS)
    assert max(r.max_difference for r in report.results) < 1e-14
    assert next(iter(report.sheets())) == "Parameters"
