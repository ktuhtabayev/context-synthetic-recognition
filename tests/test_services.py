"""Services: workbook validation (failure paths) and dataset summaries."""

import warnings
from pathlib import Path

import openpyxl
import pytest

from context_synthetic_recognition.config import ExperimentConfig, plugin
from context_synthetic_recognition.data import Dataset, FeatureType, load_builtin
from context_synthetic_recognition.services.datasets import summarize
from context_synthetic_recognition.services.validation import (
    ValidationError,
    validate_workbook,
)

from .conftest import WORKBOOK


def values_copy(tmp_path: Path, edits: dict[tuple[str, str], object]) -> Path:
    """A copy of the workbook with cached values only (formulas dropped) and some cells changed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        book = openpyxl.load_workbook(WORKBOOK, data_only=True)
    for (sheet, cell), value in edits.items():
        book[sheet][cell] = value
    path = tmp_path / "copy.xlsx"
    book.save(path)
    book.close()
    return path


def test_a_changed_cell_is_reported(tmp_path: Path) -> None:
    path = values_copy(
        tmp_path,
        {("Zhuravlev Distances", "C5"): 6.2, ("Ψ(r) Binary Features", "B40"): "x"},
    )
    report = validate_workbook(path)
    assert not report.passed
    failed = [r for r in report.results if not r.passed]
    assert [(r.sheet, r.ref) for r in failed] == [
        ("Zhuravlev Distances", "B5:K14"),
        ("Ψ(r) Binary Features", "B40:G49"),
    ]
    assert failed[0].mismatches == ("C5: workbook 6.2 ≠ package 6.1621313645",)
    assert failed[1].mismatches == ("B40: workbook 'x' ≠ package 2",)


def test_a_layout_that_does_not_fit_the_data(tmp_path: Path) -> None:
    # S₂ moved from K1 to K2: |K1| = 3, so only k = 3 is permitted
    path = values_copy(tmp_path, {("Dataset", "O4"): 2})
    with pytest.raises(ValidationError, match="1 permitted k"):
        validate_workbook(path)


def test_a_missing_sheet(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        book = openpyxl.load_workbook(WORKBOOK, data_only=True)
    del book["Informativeness ω"]
    path = tmp_path / "short.xlsx"
    book.save(path)
    book.close()
    with pytest.raises(ValidationError, match="sheet 'Informativeness ω' is missing"):
        validate_workbook(path)


# ---------------------------------------------------------------- dataset summaries


def test_summary_of_heart_disease_270() -> None:
    summary = summarize(load_builtin("heart-disease-270"))
    assert summary.permitted_k is not None
    assert summary.permitted_k.count == 118
    assert summary.r == 354
    assert [op.label for op in summary.operators] == ["ρ", "ρ_I", "ρ_J"]
    assert summary.operators[2].features == ("x₂", "x₃", "x₆", "x₇", "x₉", "x₁₁", "x₁₃")
    assert summary.problems == ()


def test_summary_reports_why_the_model_is_undefined() -> None:
    three = Dataset(
        X=[[1.0], [2.0], [3.0], [4.0]], y=[1, 2, 3, 3], feature_types=(FeatureType.QUANTITATIVE,)
    )
    summary = summarize(three)
    assert summary.permitted_k is None
    assert summary.r is None
    assert any("binary" in p for p in summary.problems)
    assert any("no k is permitted" in p for p in summary.problems)
    assert [s.label for s in summary.skipped_operators] == ["ρ_J"]


def test_summary_with_a_capped_k_range() -> None:
    config = ExperimentConfig(k=plugin("formula", k_max_cap=7))
    summary = summarize(load_builtin("heart-disease-270"), config)
    assert summary.permitted_k is not None
    assert summary.permitted_k.ks == (3, 5, 7)
    assert summary.r == 9
