"""The Classifiers page: classifier parts in three steps — training, test, evaluation (ADR-084)."""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QPushButton
from pytestqt.qtbot import QtBot

from context_synthetic_recognition.gui import workers
from context_synthetic_recognition.gui.window import MainWindow
from context_synthetic_recognition.gui.workbench import PartView, StepState
from context_synthetic_recognition.services.classifier_parts import CLASSIFIER_PARTS
from context_synthetic_recognition.services.comparison import CHECKSUMS, library_versions

from ..conftest import ROOT
from ..dummy_part import DummyPart
from .conftest import RUN_TIMEOUT

pytestmark = pytest.mark.gui

SPLITS = ROOT / "datasets" / "splits"
OUT = ROOT / "experiments" / "comparison"
BROKEN = (
    "⚠ German (1000, 20, 2) — training 336 (zhuravlyov).dat: row 2 (object 1) has 22 values, "
    "expected n + 2 = 23 (21 features, the class, the split marker); the header's n counts "
    "the features only"
)


def ready(view: PartView, qtbot: QtBot, folder: Path = SPLITS) -> PartView:
    """The view with its samples read from ``folder``."""
    view.splits.setText(str(folder))
    with qtbot.waitSignal(view.finished, timeout=RUN_TIMEOUT):
        view._folder_edited() if view._read else view.read_options()
    return view


def run(view: PartView, qtbot: QtBot, action: Any) -> None:
    with qtbot.waitSignal(view.finished, timeout=RUN_TIMEOUT):
        action()


def states(view: PartView) -> list[StepState]:
    return list(view.stepper.states)


def choose(view: PartView, key: str) -> None:
    view.samples.setCurrentIndex([o.key for o in view.options].index(key))


def test_the_page_lists_the_parts(window: MainWindow, qtbot: QtBot) -> None:
    page = window.classifiers_page
    assert window.go_to(page)
    assert page.list.count() == 1
    assert page.list.item(0).text() == "Tables 7 and 8 — standard classifiers and the CS-model"
    # one part: the list starts collapsed to a strip that opens it
    assert page.list.isHidden()
    assert page.toggle.text() == "▶"
    page.toggle.click()
    assert not page.list.isHidden()
    assert page.toggle.toolTip() == "Hide the parts"
    view = page.view
    assert [view.stepper.buttons[i].text() for i in range(3)] == [
        "① Training\n⊘ unavailable",
        "② Test\n⊘ unavailable",
        "③ Evaluation\n⊘ unavailable",
    ]
    assert not view.training_view.action.isEnabled()
    ready(view, qtbot)
    assert [view.samples.itemText(i) for i in range(view.samples.count())] == [
        "All article splits (336, 341, 367) · test 664 / 659 / 633",
        "336 · training 336 (196, 140) · test 664",
        "341 · training 341 (173, 168) · test 659",
        "367 · training 367 (190, 177) · test 633",
        "76 · training 76 (36, 40) · test 194",
        "601 · training 601 (361, 240) · test 1501",
        "722 · training 722 (467, 255) · test 1380",
    ]
    assert states(view) == [StepState.NOT_RUN] * 3
    assert view.training_view.action.text() == "Fit on the training sample"
    assert view.training_view.current_key() == "methods"
    methods = view.training_view.tabs_by_key["methods"].panel.table()
    assert methods is not None
    assert methods.rows[0][:2] == ("Table 7", "SVM")
    assert not view.stepper.buttons[1].isEnabled()
    assert not view.go_to_step(2)
    assert view.stack.currentIndex() == 0


def test_steps_enabling_and_out_of_date(window: MainWindow, qtbot: QtBot) -> None:
    view = ready(window.classifiers_page.view, qtbot)
    choose(view, "341")
    run(view, qtbot, view.training_view.action.click)
    assert states(view) == [StepState.DONE, StepState.NOT_RUN, StepState.NOT_RUN]
    assert view.stepper.buttons[1].isEnabled()
    assert not view.stepper.buttons[2].isEnabled()
    assert [view.training_view.tabs.tabText(i) for i in range(3)] == [
        "Methods",
        "Training sample",
        "Training results",
    ]
    # every tab opens with the decimals of its table (4 when it asks for none)
    tabs = view.training_view.tabs_by_key
    assert tabs["training-sample"].decimals.value() == 4
    assert tabs["training-results"].decimals.value() == 2
    tabs["training-sample"].decimals.setValue(1)
    model = tabs["training-sample"].panel.model()
    assert model is not None
    assert model._decimals == 1  # what is shown changes, the values do not
    # the test objects can be seen before they are classified: without a class column
    objects = view.test_view.tabs_by_key["test-objects"].panel.table()
    assert objects is not None
    assert "Class" not in objects.columns
    assert view.step(1)  # Alt+Right
    assert view.stack.currentIndex() == 1
    assert view.test_view.banner.text().startswith("Classes hidden")
    run(view, qtbot, view.test_view.action.click)
    assert states(view) == [StepState.DONE, StepState.DONE, StepState.NOT_RUN]
    assert view.test_view.current_key() == "predictions"
    run(view, qtbot, view.evaluation_view.action.click)
    assert states(view) == [StepState.DONE] * 3
    assert view.stack.currentIndex() == 2
    assert view.evaluation_view.banner.text().startswith("True classes revealed")
    assert not view.write_button.isEnabled()  # one split: Tables 7 and 8 need all three
    assert "Only the full article set" in view.write_button.toolTip()
    seven = view.evaluation_view.tabs_by_key["table-7"]
    assert seven.panel.isHidden()
    assert "need the three article splits" in seven.missing.text()
    # training again: the later steps are out of date, and the evaluation cannot be opened
    view.go_to_step(0)
    run(view, qtbot, view.training_view.action.click)
    assert states(view) == [StepState.DONE, StepState.OUT_OF_DATE, StepState.OUT_OF_DATE]
    assert "① Training was run again." in view.stepper.buttons[1].toolTip()
    assert "⟳ out of date" in view.stepper.buttons[2].text()
    assert not view.stepper.buttons[2].isEnabled()
    assert not view.evaluation_view.action.isEnabled()
    # other samples: every step is out of date and only the training can be opened
    choose(view, "367")
    assert states(view)[0] is StepState.OUT_OF_DATE
    assert not view.stepper.buttons[1].isEnabled()
    assert not view.step(1)
    assert view.step(-1) is False  # already at the first step


def test_run_all_reproduces_todays_tables(window: MainWindow, qtbot: QtBot, tmp_path: Path) -> None:
    view = ready(window.classifiers_page.view, qtbot)
    option = view.option()
    assert option is not None
    assert option.key == "article"
    formats: list[str] = []
    view.progress.valueChanged.connect(lambda _: formats.append(view.progress.format()))
    run(view, qtbot, view.run_all_button.click)
    assert states(view) == [StepState.DONE] * 3
    assert "① Training · 341" in formats  # the progress names the step and the split
    assert "② Test · 367" in formats
    assert view.status.text() == "Training, test and evaluation done."
    tabs = view.evaluation_view
    assert [tabs.tabs.tabText(i) for i in range(tabs.tabs.count())] == [
        "Table 7",
        "Table 8",
        "CS-model",
        "All results",
        "Per object",
        "Confusion",
        "Check",
    ]
    assert tabs.current_key() == "table-7"
    table_7 = tabs.tabs_by_key["table-7"].panel.table()
    assert table_7 is not None
    assert table_7.rows == (("SVM", 74.85, 50.68, 61.14), ("Random Forest", 74.25, 56.45, 59.87))
    table_8 = tabs.tabs_by_key["table-8"].panel.table()
    assert table_8 is not None
    assert table_8.rows[0] == ("Euclidean", "57.38 / 60.09", "58.42 / 54.78", "57.50 / 59.87")
    cs_model = tabs.tabs_by_key["cs-model"].panel.table()
    assert cs_model is not None
    assert cs_model.rows[-1] == ("Zhuravlyov", "71.54 / 68.07", None, None)
    results = tabs.tabs_by_key["all-results"].panel.table()
    assert results is not None
    assert len(results.rows) == 46

    # write and check from the Evaluation step
    out = tmp_path / "comparison"
    view.folder_dialog = lambda title, current: str(out)
    view.browse_output.click()
    assert view.write_button.isEnabled()
    run(view, qtbot, view.write_button.click)
    assert view.status.text() == f"13 files written to {out}"
    assert sorted(p.name for p in out.iterdir()) == sorted(p.name for p in OUT.iterdir())
    recorded = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))["versions"]
    if recorded == library_versions():
        for path in out.iterdir():
            assert path.read_bytes() == (OUT / path.name).read_bytes(), path.name
    opened: list[Path] = []
    view.open_folder = opened.append
    view.open_button.click()
    assert opened == [out]
    run(view, qtbot, view.check_button.click)
    assert view.status.text() == f"Check passed: {out}"
    assert tabs.current_key() == "check"
    check = tabs.tabs_by_key["check"].panel.table()
    assert check is not None
    assert {row[1] for row in check.rows} == {"✓"}
    (out / "table-7.md").write_bytes(b"changed\n")
    (out / CHECKSUMS).write_bytes((out / CHECKSUMS).read_bytes() + b"0  extra.csv\n")
    run(view, qtbot, view.start_check)
    assert view.status.text() == f"Check failed: 5 problem(s) in {out}"


def test_the_test_step_shows_no_class(window: MainWindow, qtbot: QtBot) -> None:
    view = ready(window.classifiers_page.view, qtbot)
    choose(view, "76")
    run(view, qtbot, view.fit)
    run(view, qtbot, view.classify)
    for key in ("test-objects", "predictions", "predicted-classes"):
        table = view.test_view.tabs_by_key[key].panel.table()
        assert table is not None
        assert not any(c in ("Class", "True class", "Correct methods") for c in table.columns), key
    predictions = view.test_view.tabs_by_key["predictions"]
    # coloured by the predicted class, never green or red: the correctness is not known
    model = predictions.panel.model()
    assert model is not None
    roles = {
        model.style.cell(row, column, value)  # type: ignore[misc]
        for row, values in enumerate(model.table.rows)
        for column, value in enumerate(values)
    }
    assert {r.value for r in roles if r is not None} <= {"k1", "k2", "refusal", "muted"}
    # cross-highlighting within the step
    view.test_view.highlight("S₂")
    assert view.test_view.tabs_by_key["test-objects"].panel.model()._highlight == "S₂"  # type: ignore[union-attr]


def test_a_split_file_that_breaks_the_convention(
    window: MainWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    splits = tmp_path / "splits"
    shutil.copytree(SPLITS, splits)
    view = ready(window.classifiers_page.view, qtbot, splits)
    assert view.training_view.action.isEnabled()
    path = splits / "German" / "German (1000, 20, 2) — training 336 (zhuravlyov).dat"
    data = path.read_bytes()
    path.write_bytes(data.replace(b"1000\t20\t2\t", b"1000\t21\t2\t", 1))  # the old header
    # the files changed after they were read: the training stops with the loader's message
    run(view, qtbot, view.fit)
    assert view.status.text() == "Failed."
    assert view.problems.text() == BROKEN
    assert states(view)[0] is StepState.NOT_RUN
    # read again: the samples with that file are unavailable, and say why
    view.splits.setText(str(tmp_path))
    ready(view, qtbot, splits)
    assert view.option() is not None
    assert not view.option().available  # type: ignore[union-attr]
    assert view.problems.text() == BROKEN
    assert states(view) == [StepState.UNAVAILABLE] * 3
    assert BROKEN[2:] in view.stepper.buttons[0].toolTip()
    assert not view.training_view.action.isEnabled()
    assert not view.run_all_button.isEnabled()
    choose(view, "341")
    assert view.training_view.action.isEnabled()
    assert view.problems.text() == ""
    assert not (tmp_path / "experiments").exists()


@pytest.fixture
def dummy() -> Iterator[DummyPart]:
    part = DummyPart()
    CLASSIFIER_PARTS.add(part.key, part)
    yield part
    CLASSIFIER_PARTS.remove(part.key)


def test_a_part_added_to_the_registry(
    dummy: DummyPart, qtbot: QtBot, settings: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    window = MainWindow(settings)
    window.show_message = lambda title, text: None  # type: ignore[method-assign]
    qtbot.addWidget(window)
    page = window.classifiers_page
    assert [page.list.item(i).text() for i in range(page.list.count())] == [
        "Tables 7 and 8 — standard classifiers and the CS-model",
        "Dummy part",
    ]
    assert not page.list.isHidden()  # two parts: the list is open
    assert page.choose("dummy")
    view = page.view
    assert view.part is dummy
    assert view.files.isHidden()  # it writes and checks nothing
    with qtbot.waitSignal(view.finished, timeout=RUN_TIMEOUT):
        view.read_options()
    assert view.samples.itemText(0) == "Toy · training 4 (2, 2) · test 3"
    run(view, qtbot, view.fit)
    run(view, qtbot, view.classify)
    run(view, qtbot, view.compare)
    assert states(view) == [StepState.DONE] * 3
    assert list(view.evaluation_view.tabs_by_key) == ["per-object", "confusion"]
    per_object = view.evaluation_view.tabs_by_key["per-object"].panel.table()
    assert per_object is not None
    assert [row[3] for row in per_object.rows] == ["✓ 1", "✓ 2", "✗ 1"]
    assert not page.choose("missing")
    window.close()


def test_the_page_fits_the_window(window: MainWindow, qtbot: QtBot) -> None:
    page = window.classifiers_page
    # the page does not make the window wider than the other pages do
    others = [p.minimumSizeHint().width() for p in window.pages if p is not page]
    assert page.minimumSizeHint().width() <= max(others)
    window.go_to(page)
    window.show()
    for size in (
        (1360, 860),
        (window.minimumSizeHint().width(), window.minimumSizeHint().height()),
    ):
        window.resize(*size)
        qtbot.wait(50)
        for button in page.findChildren(QPushButton):
            if button.isVisibleTo(page):
                assert button.width() >= button.minimumSizeHint().width(), button.text()
                right = button.mapTo(page, button.rect().topRight()).x()
                assert right <= page.width(), button.text()


def test_the_jobs_without_a_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    part = DummyPart()
    steps: list[tuple[str, int, int]] = []
    (found,) = workers.part_options_job(part, None, progress=lambda *s: steps.append(s))
    option, methods = found
    assert methods.key == "methods"
    done = workers.part_run_all_job(part, option, None, progress=lambda *s: steps.append(s))
    assert done.evaluation.write_problem == ""
    with pytest.raises(Exception, match="writes no files"):
        workers.part_write_job(part, done.evaluation.evaluation, Path("out"))
    with pytest.raises(Exception, match="checks no files"):
        workers.part_check_job(part, Path("out"), None)
