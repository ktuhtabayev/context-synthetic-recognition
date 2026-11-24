"""The Classifiers page: Tables 7 and 8 run, written and checked from the interface (ADR-079)."""

import shutil
from pathlib import Path
from typing import Any

import pytest
from pytestqt.qtbot import QtBot

from context_synthetic_recognition.gui import workers
from context_synthetic_recognition.gui.window import MainWindow
from context_synthetic_recognition.services.comparison import CHECKSUMS

from ..conftest import ROOT
from .conftest import RUN_TIMEOUT

pytestmark = pytest.mark.gui

SPLITS = ROOT / "datasets" / "splits"


def test_the_page_needs_no_dataset_or_run(window: MainWindow) -> None:
    page = window.classifiers_page
    assert window.go_to(page)
    assert page.status.text() == "Not run yet."
    assert not page.write_button.isEnabled()
    assert page.run_button.isEnabled()
    assert [page.tabs.tabText(i) for i in range(page.tabs.count())] == [
        "Table 7",
        "Table 8",
        "CS-model",
        "All results",
        "Check",
    ]
    # the window works in an empty folder: the project's folders are not found from there
    assert page.splits.text() == str(Path("datasets") / "splits")
    assert page.output.text() == str(Path("experiments") / "comparison")


def test_running_writing_and_checking(window: MainWindow, qtbot: QtBot, tmp_path: Path) -> None:
    page = window.classifiers_page
    out = tmp_path / "comparison"
    page.folder_dialog = lambda title, current: {
        "Split files": str(SPLITS),
        "Results folder": str(out),
    }[title]
    page.browse_splits.click()
    page.browse_output.click()
    assert (page.splits.text(), page.output.text()) == (str(SPLITS), str(out))
    assert not page.open_button.isEnabled()  # nothing written yet

    with qtbot.waitSignal(page.finished, timeout=RUN_TIMEOUT):
        page.run_button.click()
        assert page.is_running()
        page.start()  # a second click while running does nothing
    assert page.outcome is not None
    assert page.status.text().startswith("Done: 3 splits from ")
    table_7 = page.table_7.table()
    assert table_7 is not None
    assert table_7.rows == (("SVM", 74.85, 50.68, 61.14), ("Random Forest", 74.25, 56.45, 59.87))
    table_8 = page.table_8.table()
    assert table_8 is not None
    assert table_8.rows[0] == ("Euclidean", "57.38 / 60.09", "58.42 / 54.78", "57.50 / 59.87")
    cs_model = page.cs_model.table()
    assert cs_model is not None
    assert cs_model.rows[-1] == ("Zhuravlyov", "71.54 / 68.07", None, None)
    results = page.results.table()
    assert results is not None
    assert len(results.rows) == 46

    page.write_button.click()
    assert sorted(p.name for p in out.iterdir()) == sorted(page.outcome.files)
    assert page.status.text() == f"13 files written to {out}"
    opened: list[Path] = []
    page.open_folder = opened.append
    assert page.open_button.isEnabled()
    page.open_button.click()
    assert opened == [out]

    with qtbot.waitSignal(page.finished, timeout=RUN_TIMEOUT):
        page.check_button.click()
    assert page.check is not None
    assert page.check.passed
    assert page.status.text() == f"Check passed: {out}"
    check = page.check_panel.table()
    assert check is not None
    assert {row[1] for row in check.rows} == {"✓"}
    assert page.tabs.currentWidget() is page.check_panel

    (out / "table-7.md").write_bytes(b"changed\n")
    (out / CHECKSUMS).write_bytes((out / CHECKSUMS).read_bytes() + b"0  extra.csv\n")
    with qtbot.waitSignal(page.finished, timeout=RUN_TIMEOUT):
        page.start_check()
    assert not page.check.passed
    assert page.status.text() == f"Check failed: 5 problem(s) in {out}"
    check = page.check_panel.table()
    assert check is not None
    marks = {row[0]: (row[1], row[2]) for row in check.rows}
    assert marks["table-7.md"] == (
        "✗",
        "does not match checksums.sha256; differs from a new run",
    )
    assert marks["extra.csv"] == ("✗", "missing; listed, a new run does not write it")


def test_a_comparison_that_fails_is_reported(
    window: MainWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    page = window.classifiers_page
    page.splits.setText(str(tmp_path / "no-splits"))
    with qtbot.waitSignal(page.finished, timeout=RUN_TIMEOUT):
        page.start()
    assert page.status.text() == "Failed."
    assert "split file not found" in page.problems.text()
    assert page.outcome is None
    page.write()  # nothing to write
    assert page.run_button.isEnabled()


def test_a_split_file_that_breaks_the_convention_is_reported(
    window: MainWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    # the old header "m, n + 1, c" counted the marker column (ADR-082)
    splits = tmp_path / "splits"
    shutil.copytree(SPLITS, splits)
    path = splits / "German" / "German (1000, 20, 2) — training 336 (zhuravlyov).dat"
    data = path.read_bytes()
    assert data.startswith(b"1000\t20\t2\t")
    path.write_bytes(data.replace(b"1000\t20\t2\t", b"1000\t21\t2\t", 1))
    message = (
        "⚠ German (1000, 20, 2) — training 336 (zhuravlyov).dat: row 2 (object 1) has 22 values, "
        "expected n + 2 = 23 (21 features, the class, the split marker); the header's n counts "
        "the features only"
    )
    out = tmp_path / "comparison"
    page = window.classifiers_page
    page.splits.setText(str(splits))
    page.output.setText(str(out))

    with qtbot.waitSignal(page.finished, timeout=RUN_TIMEOUT):
        page.run_button.click()
    assert page.status.text() == "Failed."
    assert page.problems.text() == message
    assert not page.problems.isHidden()
    assert page.outcome is None
    assert not page.write_button.isEnabled()
    page.write()  # nothing to write
    assert not out.exists()

    # a check runs the comparison again: the same file and row, and nothing is written
    shutil.copytree(ROOT / "experiments" / "comparison", out)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with qtbot.waitSignal(page.finished, timeout=RUN_TIMEOUT):
        page.check_button.click()
    assert page.status.text() == "Failed."
    assert page.problems.text() == message
    assert page.check is None
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert page.run_button.isEnabled()


def test_a_write_that_fails_is_reported(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = window.classifiers_page

    def fail(*args: Any) -> list[Path]:
        raise OSError("the folder is read-only")

    from context_synthetic_recognition.gui.pages import classifiers

    monkeypatch.setattr(classifiers, "write_comparison", fail)
    page.outcome = object()  # type: ignore[assignment]  # any outcome: writing fails first
    page.write()
    assert page.problems.text() == "⚠ the folder is read-only"


def test_the_jobs_report_one_step_per_split(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(splits: Path | None, *, progress: Any) -> str:
        for name in ("336", "341", "367"):
            if progress is not None:  # without a listener the jobs pass none
                progress(name)
        return f"ran {splits}"

    def fake_check(folder: Path, *, splits: Path | None, progress: Any) -> str:
        return fake(splits, progress=progress).replace("ran", f"checked {folder},")

    monkeypatch.setattr(workers, "run_comparison", fake)
    monkeypatch.setattr(workers, "check_comparison", fake_check)
    steps: list[tuple[str, int, int]] = []
    assert workers.comparison_job(None, progress=lambda *s: steps.append(s)) == "ran None"
    assert steps == [("336", 0, 3), ("341", 1, 3), ("367", 2, 3), ("done", 3, 3)]
    assert workers.comparison_job(SPLITS) == f"ran {SPLITS}"
    steps.clear()
    checked = workers.comparison_check_job(Path("out"), None, progress=lambda *s: steps.append(s))
    assert checked == f"checked {Path('out')}, None"
    assert len(steps) == 4
    assert workers.comparison_check_job(Path("out"), None) == f"checked {Path('out')}, None"
