"""The author's two check workbooks: additional features and meta-algorithm (ADR-029, ADR-060).

The workbooks are byte-for-byte copies in ``tests/data/workbooks``; a test pins their checksums.
"""

import hashlib
import json
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pytest

from context_synthetic_recognition.config import ClassMeans
from context_synthetic_recognition.core.additional_features.agglomerative import (
    StopReason,
    agglomerative,
)
from context_synthetic_recognition.core.meta_algorithm.algorithm import K2_DECISION, meta_classify
from context_synthetic_recognition.notation import feature_name
from context_synthetic_recognition.services.validation import sheets as workbook_sheets
from context_synthetic_recognition.services.validation import validate_workbook
from context_synthetic_recognition.services.validation.check_workbooks import (
    read_additional_features_workbook,
    read_meta_algorithm_workbook,
)

from ..conftest import (
    ADDITIONAL_FEATURES_WORKBOOK,
    GOLDEN_VALUES,
    META_ALGORITHM_WORKBOOK,
    WORKBOOKS,
    article_switches,
    excel_experiment,
    workbook_additional,
)

pytestmark = pytest.mark.golden


def _open(path: Path) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return openpyxl.load_workbook(path, data_only=True)


@pytest.fixture(scope="module")
def agglomerative_book() -> Iterator[Any]:
    book = _open(ADDITIONAL_FEATURES_WORKBOOK)
    yield book
    book.close()


@pytest.fixture(scope="module")
def meta_book() -> Iterator[Any]:
    book = _open(META_ALGORITHM_WORKBOOK)
    yield book
    book.close()


def _set(tuplam: tuple[int, ...]) -> str:
    return "{" + ", ".join(feature_name(u) for u in tuplam) + "}"


COPIES = {
    "Additional features [Heart-Disease (10, 13, 2)].xlsx": (
        "da28fb6fbc5be8709e55eceff228300d034e92e634a93b97f3e97e73ca4a3acd"
    ),
    "Excel experiment [Heart-Disease (10, 13, 2)].xlsx": (
        "240cea999ba5134bed622001d6d609a34d7b4968b07e94d653451aa784cee10e"
    ),
    "Meta-algorithm [Heart-Disease (10, 13, 2)].xlsx": (
        "ae22562541f841b6a234f8d8f0d15c0d90ae4ef9e4cc658968e723f14abc645e"
    ),
    "Synthetic features by k-NN [Heart-Disease (10, 13, 2)].xlsx": (
        "4e0888da72010558c82a3af59f02dd738756d1807727010305208e906462ef89"
    ),
}
"""The author's workbooks the tests read, copied byte for byte (ADR-060)."""


@pytest.mark.parametrize(("name", "digest"), sorted(COPIES.items()))
def test_the_workbooks_keep_the_authors_bytes(name: str, digest: str) -> None:
    assert hashlib.sha256((WORKBOOKS / name).read_bytes()).hexdigest() == digest


# ------------------------------------------------------------- additional-features check workbook


def test_the_additional_features_check_workbook_is_reproduced(agglomerative_book: Any) -> None:
    C, w, y, _ = read_additional_features_workbook(agglomerative_book)
    result = agglomerative(C, w, y, excel_experiment().additional_features)
    assert _set(result.tuplam) == "{x₃, x₆, x₁₃, x₄, x₉}"
    assert result.stop is StopReason.KAPPA
    additional = np.array(
        [
            [
                agglomerative_book["Dataset for Meta-algorithm"].cell(r, c).value
                for c in range(10, 14)
            ]
            for r in range(3, 13)
        ]
    )
    assert np.abs(result.additional - additional).max() <= 1e-12
    # θ/γ of every candidate of every iteration (column O of the candidate blocks)
    for j, it in enumerate(result.iterations, start=1):
        sheet = agglomerative_book[workbook_sheets.agglomerative_step(j)]
        cells = [sheet.cell(41 + 15 * c, 15).value for c in range(it.candidates.size)]
        assert np.abs(np.array(cells) - it.ratio).max() <= 1e-12
        assert sheet.cell(30 + 15 * it.candidates.size, 15).value == pytest.approx(
            it.crit, abs=1e-12
        )


def test_the_additional_features_check_workbook_validation_report() -> None:
    report = validate_workbook(ADDITIONAL_FEATURES_WORKBOOK)
    assert report.passed, [r.mismatches[:3] for r in report.results if not r.passed]
    assert report.kind == "additional-features"
    assert report.cells == 6131
    assert max(r.max_difference for r in report.results) < 1e-14
    assert report.notes[0] == "SET = {x₃, x₆, x₁₃, x₄, x₉}; stop: |TUPLAM| = ϰ"


def test_the_switch_variants_on_the_check_workbook_data(agglomerative_book: Any) -> None:
    with GOLDEN_VALUES.open(encoding="utf-8") as handle:
        golden = json.load(handle)["check-workbook"]
    C, w, y, _ = read_additional_features_workbook(agglomerative_book)
    for variant in golden:
        config = workbook_additional(
            class_means=ClassMeans(variant["cent"]), majorizing_passes=variant["pas"]
        )
        assert _set(agglomerative(C, w, y, config).tuplam) == variant["T"], variant["label"]


def test_the_article_setting_does_not_reproduce_the_check_workbook() -> None:
    report = validate_workbook(ADDITIONAL_FEATURES_WORKBOOK, article_switches())
    assert not report.passed
    assert report.notes[0].startswith("SET = {x₃, x₁, x₅, x₉, x₆}")


# ---------------------------------------------------------------- meta-algorithm check workbook


def test_the_meta_algorithm_check_workbook_is_reproduced(meta_book: Any) -> None:
    description, query = read_meta_algorithm_workbook(meta_book)
    assert query.tolist() == [4, 0, 7, 2, 0]
    result = meta_classify(description, query)
    assert result.decisions.tolist() == [K2_DECISION]
    steps = result.steps(0)
    assert steps.b1(4).tolist() == []  # B1(a₄) = ∅
    assert steps.b2(4).tolist() == [8, 9]  # B2(a₄) = {S₉, S₁₀}
    assert meta_book["Meta-algorithm"]["C223"].value == "Class 2"


def test_the_meta_algorithm_check_workbook_validation_report() -> None:
    report = validate_workbook(META_ALGORITHM_WORKBOOK)
    assert report.passed
    assert report.kind == "meta-algorithm"
    assert report.notes == ("B1(a₄) = ∅, B2(a₄) = {S₉, S₁₀}; class: Class 2",)


def test_the_additional_features_feed_the_meta_algorithm_check_workbook(
    agglomerative_book: Any, meta_book: Any
) -> None:
    # the additional features of the first check workbook are the dᵢ of the second
    C, w, y, _ = read_additional_features_workbook(agglomerative_book)
    description, _ = read_meta_algorithm_workbook(meta_book)
    result = agglomerative(C, w, y, excel_experiment().additional_features)
    assert np.abs(result.additional - description.additional).max() <= 1e-12
    assert np.array_equal(result.class_index, description.class_index)


# ----------------------------------------------------------- the switches sheet of the experiment


@pytest.fixture(scope="module")
def experiment_book() -> Iterator[Any]:
    from ..conftest import WORKBOOK

    book = _open(WORKBOOK)
    yield book
    book.close()


def test_the_switches_sheet_tables(agglomerative_book: Any, experiment_book: Any) -> None:
    """Rows 11–22, 32–41 and 47–50: the check workbook's data under both class means."""
    sheet = experiment_book[workbook_sheets.SWITCHES]
    C, w, y, labels = read_additional_features_workbook(agglomerative_book)
    running = agglomerative(C, w, y, excel_experiment().additional_features)
    final = agglomerative(C, w, y, workbook_additional(class_means=ClassMeans.FINAL))
    first_running, first_final = running.iterations[0], final.iterations[0]
    assert np.array_equal(first_running.candidates, first_final.candidates)
    for row, u in enumerate(first_running.candidates, start=11):
        scan = final.scan(0, int(u))
        expected = [
            first_running.theta[row - 11],
            first_running.gamma[row - 11],
            first_running.ratio[row - 11],
            first_final.theta[row - 11],
            first_final.gamma[row - 11],
            first_final.ratio[row - 11],
            scan.centre1[0],
            scan.centre2[0],
        ]
        assert sheet.cell(row, 1).value == feature_name(int(u))
        cells = [sheet.cell(row, c).value for c in range(2, 10)]
        assert np.abs(np.array(cells) - np.array(expected)).max() <= 1e-12
    # d₁ on the check workbook data (q = x₆): R + η_q, one pass (the article), two passes
    assert first_running.q == 5
    once = running.scan(0, 5)
    for t in range(10):
        cells = [sheet.cell(32 + t, c).value for c in range(2, 7)]
        expected = [
            once.b[t],
            once.majorized[t],
            first_running.additional[t],
            first_running.additional[t],
        ]
        assert np.abs(np.array(cells[:4]) - np.array(expected)).max() <= 1e-12
        assert cells[4] == labels[t]
    # the SET of every switch setting
    for row, (class_means, passes) in enumerate(
        [
            (ClassMeans.RUNNING, 2),
            (ClassMeans.RUNNING, 1),
            (ClassMeans.FINAL, 2),
            (ClassMeans.FINAL, 1),
        ],
        start=47,
    ):
        result = agglomerative(
            C, w, y, workbook_additional(class_means=class_means, majorizing_passes=passes)
        )
        assert sheet.cell(row, 4).value == _set(result.tuplam)
