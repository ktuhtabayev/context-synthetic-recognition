"""The evaluation designs on the Classifiers page: the chooser, the steps, cancel (ADR-096)."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from context_synthetic_recognition.gui.window import MainWindow
from context_synthetic_recognition.gui.workbench import PartView, StepState
from context_synthetic_recognition.services import designs as design_module
from context_synthetic_recognition.services.designs import DESIGNS, BaseDesign, Case, Round

from .conftest import RUN_TIMEOUT
from .test_classifiers_page import SPLITS, choose, ready, run, states

pytestmark = pytest.mark.gui


def test_the_design_chooser(window: MainWindow, qtbot: QtBot) -> None:
    view = ready(window.classifiers_page.view, qtbot)
    chooser = view.designs
    assert view.designable
    texts = [chooser.itemText(i) for i in range(chooser.count())]
    assert texts == [
        "Swapping the subsets",
        "    Training → Test (as given)",
        "    Test → Training (reversed)",
        "    Training → Training (self-check)",
        "    Test → Test (self-check)",
        "Comparison points",
        "    Random training samples (control)",
        "    Whole dataset, cross-validation",
        "Honest estimate",
        "    Selection inside folds",
        "Optional extras",
        "    Training-size curve",
        "    Split to split",
    ]
    model = chooser.model()
    heading = model.index(0, 0)
    assert not model.flags(heading) & Qt.ItemFlag.ItemIsSelectable  # a heading, not a design
    assert chooser.key() == "direct"
    assert view.design_note.text().startswith("Fit on the training sample, classify the test")
    random = chooser.findData("random-control", Qt.ItemDataRole.UserRole)
    tip = chooser.itemData(random, Qt.ItemDataRole.ToolTipRole)
    assert "seeds 42 … 141" in tip
    assert tip.startswith("Random training samples of the same size and classes")
    # another design: the steps run so far are out of date, and the choice is remembered
    choose(view, "German-341")
    run(view, qtbot, view.fit)
    assert chooser.choose("reversed")
    assert states(view)[0] is StepState.OUT_OF_DATE
    assert "The design changed." in view.stepper.buttons[0].toolTip()
    assert view.settings is not None
    assert view.settings.store.value("classifiers/design/classifier-comparison") == "reversed"


def test_a_design_through_the_three_steps(window: MainWindow, qtbot: QtBot) -> None:
    view = ready(window.classifiers_page.view, qtbot)
    choose(view, "Heart-Disease-76")
    view.designs.choose("reversed")
    run(view, qtbot, view.run_all_button.click)
    assert states(view) == [StepState.DONE] * 3
    assert "design-plan" in view.training_view.tabs_by_key
    sample = view.training_view.tabs_by_key["training-sample"].panel.table()
    assert sample is not None
    assert len(sample.rows) == 194  # fitted on the test sample
    objects = view.test_view.tabs_by_key["test-objects"].panel.table()
    assert objects is not None
    assert len(objects.rows) == 76  # the training objects, without their class
    assert "Class" not in objects.columns
    tabs = [view.evaluation_view.tabs.tabText(i) for i in range(view.evaluation_view.tabs.count())]
    assert tabs[:1] == ["Design"]
    assert {"Per object", "Confusion", "All designs"} <= set(tabs)
    assert not view.write_button.isEnabled()
    assert "Only the design as given" in view.write_button.toolTip()
    view.designs.choose("direct")
    run(view, qtbot, view.run_all_button.click)
    everything = view.evaluation_view.tabs_by_key["all-designs"].panel.table()
    assert everything is not None
    assert everything.columns[2:] == (
        "Test → Training (reversed) · accuracy, %",
        "Test → Training (reversed) · balanced, %",
        "Training → Test (as given) · accuracy, %",
        "Training → Test (as given) · balanced, %",
    )
    assert len(everything.rows) == 14  # Heart-Disease's 14 methods


def test_a_long_design_shows_its_progress_and_can_be_cancelled(
    window: MainWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(design_module, "REPETITIONS", 3)
    view = ready(window.classifiers_page.view, qtbot)
    choose(view, "Heart-Disease-76")
    view.designs.choose("random-control")
    formats: list[str] = []
    view.progress.valueChanged.connect(lambda _: formats.append(view.progress.format()))
    run(view, qtbot, view.run_all_button.click)
    assert states(view) == [StepState.DONE] * 3
    assert any("③ Evaluation · Heart-Disease · Zhuravlyov split" in f for f in formats)
    figure = view.evaluation_view.tabs_by_key["design-figure"]
    assert figure.figure_panel.isVisibleTo(figure)
    spec = figure.figure_panel.spec()
    assert spec is not None
    assert spec.key == "random-control"
    # cancelled after its first round: the evaluation is not done, nothing breaks
    monkeypatch.setattr(design_module, "REPETITIONS", 100)
    view.go_to_step(2)
    seen: list[int] = []

    def stop(what: str, done: int, total: int) -> None:
        seen.append(done)
        if len(seen) == 2:
            view.cancel()

    view.progress.valueChanged.connect(lambda value: stop("", value, 0))
    run(view, qtbot, view.compare)
    assert view.status.text() == "Cancelled."
    assert states(view)[2] is StepState.DONE  # the evaluation shown before stays


@pytest.fixture
def every_other() -> Iterator[BaseDesign]:
    class EveryOther(BaseDesign):
        def cases(self, split, others):
            m = split.dataset.m
            planned = Round("", np.arange(0, m, 2), np.arange(1, m, 2))
            return (Case("as-given", "", 1, lambda _: planned),)

    design = EveryOther("every-other", "Every other object", "Optional extras", "For the tests.")
    DESIGNS.add(design.key, design)
    yield design
    DESIGNS.remove(design.key)


def test_a_registered_design_appears_without_a_change(
    every_other: BaseDesign,
    qtbot: QtBot,
    settings: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    window = MainWindow(settings)  # type: ignore[arg-type]
    window.show_message = lambda title, text: None  # type: ignore[method-assign]
    qtbot.addWidget(window)
    view: PartView = window.classifiers_page.view
    assert view.designs.choose("every-other")
    assert view.designs.itemText(view.designs.currentIndex()) == "    Every other object"
    ready(view, qtbot, SPLITS)
    choose(view, "Heart-Disease-76")
    with qtbot.waitSignal(view.finished, timeout=RUN_TIMEOUT):
        view.run_all()
    assert states(view) == [StepState.DONE] * 3
    window.close()


def test_the_page_translates_the_designs_texts() -> None:
    from context_synthetic_recognition.gui.workbench import DESIGN_TEXTS, GROUP_TITLES
    from context_synthetic_recognition.services.designs import GROUPS, designs

    for design in designs()[:9]:
        assert DESIGN_TEXTS[design.key] == (design.title, design.explanation), design.key
    assert tuple(GROUP_TITLES) == GROUPS
