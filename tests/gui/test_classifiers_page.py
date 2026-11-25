"""The Classifiers page: classifier parts in three steps — training, test, evaluation (ADR-084).

The samples are a tree of datasets and their splits, built from the split files (ADR-091).
"""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication, QPushButton
from pytestqt.qtbot import QtBot

from context_synthetic_recognition.comparison import COMPARISON_SPLITS
from context_synthetic_recognition.gui import workers
from context_synthetic_recognition.gui.settings import Settings
from context_synthetic_recognition.gui.window import MainWindow
from context_synthetic_recognition.gui.workbench import PartView, StepState
from context_synthetic_recognition.services.classifier_parts import CLASSIFIER_PARTS
from context_synthetic_recognition.services.comparison import CHECKSUMS, library_versions

from ..conftest import ROOT
from ..dummy_part import DummyPart
from ..test_split_discovery import toy
from .conftest import RUN_TIMEOUT

pytestmark = pytest.mark.gui

SPLITS = ROOT / "datasets" / "splits"
OUT = ROOT / "experiments" / "comparison"
BROKEN = (
    "⚠ German (1000, 20, 2) — training 336 (zhuravlyov).dat: row 2 (object 1) has 22 values, "
    "expected n + 2 = 23 (21 features, the class, the split marker); the header's n counts "
    "the features only"
)
SAVED = "classifiers/samples/classifier-comparison"
"""Where the page remembers the ticked samples of the comparison part."""


def ready(view: PartView, qtbot: QtBot, folder: Path = SPLITS) -> PartView:
    """The view with its samples read from ``folder``."""
    view.wait()  # a reading the view started when it was shown
    QApplication.processEvents()
    with qtbot.waitSignal(view.finished, timeout=RUN_TIMEOUT):
        if view.splits.text() != str(folder):
            view.splits.setText(str(folder))
            view._folder_edited()
        else:
            view.read_samples()
    return view


def run(view: PartView, qtbot: QtBot, action: Any) -> None:
    with qtbot.waitSignal(view.finished, timeout=RUN_TIMEOUT):
        action()


def states(view: PartView) -> list[StepState]:
    return list(view.stepper.states)


def choose(view: PartView, key: str) -> None:
    """Tick this split alone."""
    view.samples.set_ticked([key])


def texts(view: PartView) -> list[list[str]]:
    """The tree as text: each dataset, then its splits."""
    return [
        [group.text(0), *(group.child(i).text(0) for i in range(group.childCount()))]
        for group in view.samples.groups()
    ]


def test_the_page_lists_the_parts(window: MainWindow, qtbot: QtBot) -> None:
    page = window.classifiers_page
    assert window.go_to(page)
    assert page.list.count() == 1
    assert page.list.item(0).text() == "Comparison with standard classifiers"
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
    assert texts(view)[:3] == [
        [
            "German (1000 objects)",
            "Zhuravlyov split · 20 features · training 336 (196, 140) · test 664",
            "Euclidean split · 7 features · training 341 (173, 168) · test 659",
            "Chebyshev split · 7 features · training 367 (190, 177) · test 633",
        ],
        [
            "Heart-Disease (270 objects)",
            "Zhuravlyov split · 13 features · training 76 (36, 40) · test 194",
        ],
        [
            "Spambase (2102 objects)",
            "Euclidean split · 57 features · training 601 (361, 240) · test 1501",
            "Euclidean split · 57 features · training 722 (467, 255) · test 1380",
        ],
    ]
    # German's three splits are ticked by default
    assert view.samples.ticked() == COMPARISON_SPLITS
    german, heart = view.samples.groups()[:2]
    assert german.checkState(0) == Qt.CheckState.Checked
    assert heart.checkState(0) == Qt.CheckState.Unchecked
    assert (
        german.child(1)
        .toolTip(0)
        .startswith(
            "German · Euclidean split · training 341 (173, 168)\nGerman/German (1000, 7, 2)"
        )
    )
    assert states(view) == [StepState.NOT_RUN] * 3
    assert view.training_view.action.text() == "Fit on the training sample"
    assert view.training_view.current_key() == "methods"
    methods = view.training_view.tabs_by_key["methods"].panel.table()
    assert methods is not None
    assert methods.rows[0][:2] == ("Standard classifiers", "SVM")
    assert not view.stepper.buttons[1].isEnabled()
    assert not view.go_to_step(2)
    assert view.stack.currentIndex() == 0


def test_steps_enabling_and_out_of_date(window: MainWindow, qtbot: QtBot) -> None:
    view = ready(window.classifiers_page.view, qtbot)
    choose(view, "German-341")
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
    assert not view.write_button.isEnabled()  # one split: the comparison tables need all three
    assert "Only German's three splits" in view.write_button.toolTip()
    standard = view.evaluation_view.tabs_by_key["standard-classifiers"]
    assert standard.panel.isHidden()
    assert "need German's three splits" in standard.missing.text()
    # training again: the later steps are out of date, and the evaluation cannot be opened
    view.go_to_step(0)
    run(view, qtbot, view.training_view.action.click)
    assert states(view) == [StepState.DONE, StepState.OUT_OF_DATE, StepState.OUT_OF_DATE]
    assert "① Training was run again." in view.stepper.buttons[1].toolTip()
    assert "⟳ out of date" in view.stepper.buttons[2].text()
    assert not view.stepper.buttons[2].isEnabled()
    assert not view.evaluation_view.action.isEnabled()
    # other samples: every step is out of date and only the training can be opened
    choose(view, "German-367")
    assert states(view)[0] is StepState.OUT_OF_DATE
    assert not view.stepper.buttons[1].isEnabled()
    assert not view.step(1)
    assert view.step(-1) is False  # already at the first step


def test_run_all_reproduces_todays_tables(window: MainWindow, qtbot: QtBot, tmp_path: Path) -> None:
    view = ready(window.classifiers_page.view, qtbot)
    option = view.option()
    assert option is not None
    assert option.splits == COMPARISON_SPLITS
    assert option.complete
    formats: list[str] = []
    view.progress.valueChanged.connect(lambda _: formats.append(view.progress.format()))
    run(view, qtbot, view.run_all_button.click)
    assert states(view) == [StepState.DONE] * 3
    assert (
        "① Training · German · Euclidean split · training 341 (173, 168)" in formats
    )  # the progress names the step and the split
    assert "② Test · German · Chebyshev split · training 367 (190, 177)" in formats
    assert view.status.text() == "Training, test and evaluation done."
    tabs = view.evaluation_view
    assert [tabs.tabs.tabText(i) for i in range(tabs.tabs.count())] == [
        "Standard classifiers",
        "k-NN by metric",
        "CS-model",
        "All results",
        "Per object",
        "Confusion",
        "Check",
    ]
    assert tabs.current_key() == "standard-classifiers"
    standard_classifiers_table = tabs.tabs_by_key["standard-classifiers"].panel.table()
    assert standard_classifiers_table is not None
    assert standard_classifiers_table.rows == (
        ("SVM", 74.85, 50.68, 61.14),
        ("Random Forest", 74.25, 56.45, 59.87),
    )
    knn_by_metric_table = tabs.tabs_by_key["knn-by-metric"].panel.table()
    assert knn_by_metric_table is not None
    assert knn_by_metric_table.rows[0] == (
        "Euclidean",
        "57.38 / 60.09",
        "58.42 / 54.78",
        "57.50 / 59.87",
    )
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
    (out / "standard-classifiers.md").write_bytes(b"changed\n")
    (out / CHECKSUMS).write_bytes((out / CHECKSUMS).read_bytes() + b"0  extra.csv\n")
    run(view, qtbot, view.start_check)
    assert view.status.text() == f"Check failed: 5 problem(s) in {out}"


def test_the_test_step_shows_no_class(window: MainWindow, qtbot: QtBot) -> None:
    view = ready(window.classifiers_page.view, qtbot)
    choose(view, "Heart-Disease-76")
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
    # read again: that file is greyed, without a checkbox, and says why in its tooltip
    ready(view, qtbot, splits)
    item = view.samples.groups()[0].child(0)
    assert not item.flags() & Qt.ItemFlag.ItemIsEnabled
    assert item.data(0, Qt.ItemDataRole.CheckStateRole) is None
    assert item.text(0) == "⊘ Zhuravlyov split · 20 features · training 336"
    assert item.toolTip(0).startswith(f"⊘ {BROKEN[2:]}")
    assert item.toolTip(0).endswith(
        "German/German (1000, 20, 2) — training 336 (zhuravlyov).dat, row 2"
    )
    # it is no longer ticked; the others still are, without the comparison tables
    assert view.samples.ticked() == ("German-341", "German-367")
    assert states(view) == [StepState.NOT_RUN] * 3  # the failed training left nothing done
    assert view.training_view.action.isEnabled()
    assert view.problems.text() == ""
    # nothing ticked: the steps cannot run, and say why
    view.samples.set_ticked([])
    assert states(view) == [StepState.UNAVAILABLE] * 3
    assert "Tick a sample." in view.stepper.buttons[0].toolTip()
    assert not view.run_all_button.isEnabled()
    choose(view, "German-341")
    assert view.training_view.action.isEnabled()
    assert not (tmp_path / "experiments").exists()


def test_the_tree_of_samples(window: MainWindow, qtbot: QtBot, tmp_path: Path) -> None:
    view = ready(window.classifiers_page.view, qtbot)
    tree = view.samples
    german, heart, spambase = tree.groups()[:3]
    # ticking a dataset ticks its splits; some of them ticked: the partial mark
    spambase.setCheckState(0, Qt.CheckState.Checked)
    assert tree.ticked() == (*COMPARISON_SPLITS, "Spambase-601", "Spambase-722")
    german.child(0).setCheckState(0, Qt.CheckState.Unchecked)
    assert german.checkState(0) == Qt.CheckState.PartiallyChecked
    assert tree.ticked() == ("German-341", "German-367", "Spambase-601", "Spambase-722")
    german.setCheckState(0, Qt.CheckState.Unchecked)
    assert tree.ticked() == ("Spambase-601", "Spambase-722")
    # splits of several datasets: no comparison tables, and the page remembers the ticks
    option = view.option()
    assert option is not None
    assert option.available
    assert not option.complete
    assert view.settings is not None
    assert view.settings.recent(SAVED) == ["Spambase-601", "Spambase-722"]
    # the keyboard: the arrows move, Space ticks
    tree.setFocus()
    tree.setCurrentItem(heart)
    heart.setExpanded(True)
    qtbot.keyClick(tree, Qt.Key.Key_Down)
    assert tree.currentItem() is heart.child(0)
    qtbot.keyClick(tree, Qt.Key.Key_Space)
    assert tree.ticked() == ("Heart-Disease-76", "Spambase-601", "Spambase-722")
    qtbot.keyClick(tree, Qt.Key.Key_Space)
    assert tree.ticked() == ("Spambase-601", "Spambase-722")
    # a new file appears with Refresh, without a change to the code; the ticks stay
    splits = tmp_path / "splits"
    shutil.copytree(SPLITS, splits)
    ready(view, qtbot, splits)
    toy(splits / "Toy", "Toy (12, 3, 2) — training 6 (euclidean).dat")
    with qtbot.waitSignal(view.finished, timeout=RUN_TIMEOUT):
        view.refresh_samples.click()
    assert texts(view)[-1] == [
        "Toy (12 objects)",
        "Euclidean split · 3 features · training 6 (3, 3) · test 6",
    ]
    assert tree.ticked() == ("Spambase-601", "Spambase-722")
    # nothing shows a key
    for group in texts(view):
        assert not any("German-3" in line or "Spambase-" in line for line in group)


def test_saved_keys_of_an_earlier_version_fall_back(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(QSettings(str(tmp_path / "old.ini"), QSettings.Format.IniFormat))
    settings.classifier_part = "tables-7-8"
    settings.store.setValue(SAVED, ["article", "336"])
    window = MainWindow(settings)
    window.show_message = lambda title, text: None  # type: ignore[method-assign]
    qtbot.addWidget(window)
    page = window.classifiers_page
    assert page.list.currentRow() == 0  # the first part, silently
    assert settings.classifier_part == "classifier-comparison"
    view = ready(page.view, qtbot)
    assert view.samples.ticked() == COMPARISON_SPLITS  # the default, silently
    assert view.problems.text() == ""
    view.samples.set_ticked(["German-341"])
    window.close()
    # a key that still exists is kept for the next session
    again = MainWindow(settings)
    again.show_message = lambda title, text: None  # type: ignore[method-assign]
    qtbot.addWidget(again)
    view = ready(again.classifiers_page.view, qtbot)
    assert view.samples.ticked() == ("German-341",)
    again.close()


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
        "Comparison with standard classifiers",
        "Dummy part",
    ]
    assert not page.list.isHidden()  # two parts: the list is open
    assert page.choose("dummy")
    view = page.view
    assert view.part is dummy
    assert view.files.isHidden()  # it writes and checks nothing
    with qtbot.waitSignal(view.finished, timeout=RUN_TIMEOUT):
        view.read_samples()
    assert texts(view) == [["Toy (7 objects)", "Toy split · training 4 (2, 2) · test 3"]]
    assert view.samples.ticked() == ("toy",)
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
    tree = workers.part_samples_job(part, None, progress=lambda *s: steps.append(s))
    option = tree.option(tree.default)
    assert option.splits == ("toy",)
    done = workers.part_run_all_job(part, option, None, progress=lambda *s: steps.append(s))
    assert done.evaluation.write_problem == ""
    with pytest.raises(Exception, match="writes no files"):
        workers.part_write_job(part, done.evaluation.evaluation, Path("out"))
    with pytest.raises(Exception, match="checks no files"):
        workers.part_check_job(part, Path("out"), None)
