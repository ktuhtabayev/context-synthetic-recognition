"""Golden tests: every Step 1–12 range of the Excel experiment against the package (tolerance 1e-9).

The expected values are the workbook's cached cell values; the map is the one ``csr validate``
uses (:func:`~context_synthetic_recognition.services.validation.workbook_checks`). The inputs —
the data, the HAG parameters with both switches and the new object — are read from the workbook.
"""

import warnings
from collections.abc import Iterator
from typing import Any

import openpyxl
import pytest

from context_synthetic_recognition.services.validation import (
    DEFAULT_TOLERANCE,
    Check,
    ExperimentSubject,
    experiment_subject,
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


@pytest.fixture(scope="module")
def subject(workbook: Any) -> ExperimentSubject:
    return experiment_subject(WORKBOOK, workbook)[0]


@pytest.mark.parametrize("check", CHECKS, ids=[f"{c.sheet}!{c.ref}" for c in CHECKS])
def test_range_matches_the_workbook(
    check: Check[ExperimentSubject], workbook: Any, subject: ExperimentSubject
) -> None:
    assert check.applies(subject)
    result = run_check(check, workbook[check.sheet], subject, DEFAULT_TOLERANCE)
    assert result.passed, "\n".join(result.mismatches)


def test_every_step_1_to_12_sheet_is_covered() -> None:
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
        "Greedy upon Weight (1-Latent)",
        "Greedy upon Weight (2-Latent)",
        "Greedy upon Weight (3-Latent)",
        "Greedy upon Weight (4-Latent)",
        "Dataset for Meta-algorithm",
        "Brace for Meta-algorithm",
        "Meta-algorithm",
        "Meta-algorithm (All Objects)",
    }


def test_the_workbook_inputs(subject: ExperimentSubject) -> None:
    # the Parameters sheet holds the template setting; the new object is S₁ left out of its context
    settings = subject.hag.settings
    assert (settings.centres.value, settings.step4_passes) == ("running", 2)
    assert subject.new_object.exclude == 0
    assert subject.new_object.values.tolist()[:3] == [70.0, 1.0, 4.0]


def test_the_validation_report() -> None:
    report = validate_workbook(WORKBOOK)
    assert report.passed
    assert report.kind == "experiment"
    assert report.cells == 10272
    assert len(report.results) == len(CHECKS)
    assert max(r.max_difference for r in report.results) < 1e-13
    assert next(iter(report.sheets())) == "Parameters"
    assert report.notes[0].startswith("HAG settings of the workbook (Parameters sheet): α = 0.3")
    assert any("ADR-002" in note for note in report.notes)
