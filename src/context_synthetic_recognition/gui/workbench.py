"""The staged workbench of the *Classifiers* page: one view per classifier part (ADR-084).

A :class:`PartView` shows any part of :data:`~context_synthetic_recognition.services.
classifier_parts.CLASSIFIER_PARTS` from the protocol alone: the sample selector, the stepper
① *Training* → ② *Test* → ③ *Evaluation*, and the tabs each stage returns. Every stage runs in a
worker thread (:class:`~.workers.Task`) through the services; the view computes nothing and
offers no setting to change.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QShowEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from context_synthetic_recognition.export.tables import Cell
from context_synthetic_recognition.export.text import DEFAULT_DECIMALS
from context_synthetic_recognition.gui.i18n import mark, tr
from context_synthetic_recognition.gui.models import TableStyle
from context_synthetic_recognition.gui.theme import LIGHT, CellRole, Palette
from context_synthetic_recognition.gui.widgets import TablePanel, WarningBox, hint, section_title
from context_synthetic_recognition.gui.workers import (
    CheckOutcome,
    EvaluationStep,
    StagesDone,
    Task,
    TestStep,
    TrainingStep,
    part_check_job,
    part_classify_job,
    part_evaluate_job,
    part_options_job,
    part_run_all_job,
    part_train_job,
    part_write_job,
)
from context_synthetic_recognition.services.classifier_parts import (
    STAGES,
    CellMeaning,
    CheckablePart,
    ClassifierPart,
    SampleOption,
    ShownTable,
    StageTable,
    WritablePart,
)

STEPS = (mark("① Training"), mark("② Test"), mark("③ Evaluation"))
"""The three steps, in order (the stages of :data:`~...classifier_parts.STAGES`)."""
TAB_TITLES = {
    "methods": mark("Methods"),
    "training-sample": mark("Training sample"),
    "training-results": mark("Training results"),
    "test-objects": mark("Test objects"),
    "predictions": mark("Predictions"),
    "predicted-classes": mark("Predicted classes"),
    "per-object": mark("Per object"),
    "confusion": mark("Confusion"),
    "check": mark("Check"),
    "standard-classifiers": mark("Standard classifiers"),
    "knn-by-metric": mark("k-NN by metric"),
    "cs-model": mark("CS-model"),
    "all-results": mark("All results"),
}
"""Names of the tabs the page knows (others are shown as the part names them)."""


def step_name(index: int) -> str:
    """A step's (translated) name: ① Training, ② Test, ③ Evaluation."""
    name = STEPS[index]
    return tr(name)


class StepState(Enum):
    """Where a step is; shown by an icon and a word, never by colour alone."""

    NOT_RUN = ("○", mark("not run"))
    RUNNING = ("◔", mark("running"))
    DONE = ("✓", mark("done"))
    OUT_OF_DATE = ("⟳", mark("out of date"))
    UNAVAILABLE = ("⊘", mark("unavailable"))

    @property
    def icon(self) -> str:
        """The step's state as a symbol."""
        return self.value[0]

    @property
    def word(self) -> str:
        """The step's state as a (translated) word."""
        name = self.value[1]
        return tr(name)


# ---------------------------------------------------------------- colours by meaning


def _class_role(value: Cell) -> CellRole | None:
    return {1: CellRole.K1, 2: CellRole.K2}.get(value) if isinstance(value, int) else None


def _decision_role(value: Cell) -> CellRole | None:
    if value is None:
        return CellRole.MUTED
    if value == 0:
        return CellRole.REFUSAL
    return _class_role(value)


def _outcome_role(value: Cell) -> CellRole | None:
    if value is None:
        return CellRole.MUTED
    text = str(value)
    if text.startswith("✓"):
        return CellRole.GOOD
    if text.startswith("✗"):
        return CellRole.BAD
    return CellRole.REFUSAL if text == "0" else None


def _status_role(value: Cell) -> CellRole | None:
    text = "" if value is None else str(value)
    return CellRole.GOOD if text == "✓" else CellRole.BAD if text == "✗" else None


CELL_ROLES: dict[CellMeaning, Callable[[Cell], CellRole | None]] = {
    CellMeaning.CLASS: _class_role,
    CellMeaning.DECISION: _decision_role,
    CellMeaning.OUTCOME: _outcome_role,
    CellMeaning.STATUS: _status_role,
}
HEADER_ROLES = {
    CellMeaning.QUANTITATIVE: CellRole.QUANTITATIVE,
    CellMeaning.NOMINAL: CellRole.NOMINAL,
}


def stage_style(shown: ShownTable) -> TableStyle:
    """The colours of a stage table: by the meaning the services give every column."""
    rules = [CELL_ROLES.get(meaning) for meaning in shown.cells]

    def cell(row: int, column: int, value: Cell) -> CellRole | None:
        rule = rules[column]
        return None if rule is None else rule(value)

    headers = tuple(HEADER_ROLES.get(meaning) for meaning in shown.cells)
    return TableStyle(headers, cell, shown.object_column)


# ---------------------------------------------------------------- one tab


class StageTab(QWidget):
    """One tab of a step: a table, a table per split (chosen above it), or why there is none."""

    objectClicked = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """An empty tab."""
        super().__init__(parent)
        self._colours: Palette = LIGHT
        self.stage_table: StageTable | None = None
        self.variant = QComboBox()
        self.variant.setToolTip(tr("The split whose table is shown"))
        self.variant.currentIndexChanged.connect(self._show_variant)
        self.variant_label = QLabel(tr("Split"))
        self.decimals = QSpinBox()
        self.decimals.setRange(0, 15)
        self.decimals.setToolTip(tr("Decimals shown; the values themselves are not rounded"))
        self.panel = TablePanel()
        self.decimals.valueChanged.connect(self.panel.set_decimals)
        self.panel.objectClicked.connect(self.objectClicked)
        self.missing = hint("")
        self._chooser = QWidget()
        chooser = QHBoxLayout(self._chooser)
        chooser.setContentsMargins(0, 0, 0, 0)
        chooser.addWidget(self.variant_label)
        chooser.addWidget(self.variant)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._chooser)
        row.addStretch(1)
        row.addWidget(QLabel(tr("Decimals")))
        row.addWidget(self.decimals)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(row)
        layout.addWidget(self.missing)
        layout.addWidget(self.panel, 1)

    def set_stage_table(self, stage_table: StageTable) -> None:
        """Show a stage table; the split chosen before stays chosen if it is still there."""
        previous = self.variant.currentText()
        self.stage_table = stage_table
        self.variant.blockSignals(True)
        self.variant.clear()
        for variant in stage_table.variants:
            self.variant.addItem(variant.label)
        index = self.variant.findText(previous)
        self.variant.setCurrentIndex(max(index, 0))
        self.variant.blockSignals(False)
        # the decimals the table asks for (the default otherwise); kept while the split changes
        shown = self.shown()
        wanted = shown.decimals if shown is not None and shown.decimals is not None else None
        self.decimals.blockSignals(True)
        self.decimals.setValue(DEFAULT_DECIMALS if wanted is None else wanted)
        self.decimals.blockSignals(False)
        self._chooser.setVisible(len(stage_table.variants) > 1)
        self.missing.setText(stage_table.missing)
        self.missing.setVisible(bool(stage_table.missing) or not stage_table.variants)
        self.panel.setVisible(bool(stage_table.variants))
        self._show_variant()

    def _show_variant(self) -> None:
        shown = self.shown()
        if shown is None:
            return
        self.panel.set_table(shown.table, stage_style(shown))
        self.panel.set_decimals(self.decimals.value())
        self.panel.set_palette(self._colours)

    def shown(self) -> ShownTable | None:
        """The table on screen."""
        if self.stage_table is None or not self.stage_table.variants:
            return None
        index = max(self.variant.currentIndex(), 0)
        return self.stage_table.variants[min(index, len(self.stage_table.variants) - 1)]

    def set_palette(self, colours: Palette) -> None:
        """Switch the theme."""
        self._colours = colours
        self.panel.set_palette(colours)


def tab_title(stage_table: StageTable) -> str:
    """The tab's name, translated when the page knows it."""
    name = TAB_TITLES.get(stage_table.key)
    return tr(name) if name else stage_table.title


# ---------------------------------------------------------------- the stepper


class Stepper(QWidget):
    """① Training ─ ② Test ─ ③ Evaluation: each step with its state, a step opens its view."""

    stepChosen = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Three step buttons joined by lines."""
        super().__init__(parent)
        self.buttons: list[QPushButton] = []
        self.states = [StepState.NOT_RUN] * len(STEPS)
        self.reasons = [""] * len(STEPS)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for index in range(len(STEPS)):
            if index:
                line = QLabel("─")
                line.setObjectName("stepLine")
                layout.addWidget(line)
            button = QPushButton()
            button.setObjectName("step")
            button.setCheckable(True)
            button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            button.clicked.connect(lambda _=False, i=index: self.stepChosen.emit(i))
            layout.addWidget(button)
            self.buttons.append(button)
        self.current = 0
        self._paint()

    def set_state(self, index: int, state: StepState, reason: str = "") -> None:
        """Show a step's state (``reason``: why it is unavailable or out of date)."""
        self.states[index] = state
        self.reasons[index] = reason
        self._paint()

    def set_current(self, index: int) -> None:
        """Mark the step whose view is shown."""
        self.current = index
        self._paint()

    def can_open(self, index: int) -> bool:
        """A step opens when the step before it is done."""
        return index == 0 or self.states[index - 1] is StepState.DONE

    def _paint(self) -> None:
        for index, button in enumerate(self.buttons):
            state = self.states[index]
            name = step_name(index)
            button.setText(f"{name}\n{state.icon} {state.word}")
            tip = f"{name}: {state.word}"
            if self.reasons[index]:
                tip += f"\n{self.reasons[index]}"
            if not self.can_open(index):
                tip += "\n" + tr("Opens when the step before it is done.")
            button.setToolTip(tip)
            button.setEnabled(self.can_open(index))
            button.setChecked(index == self.current)


# ---------------------------------------------------------------- a step's view


class StepView(QWidget):
    """A step: its banner, its action button, its tabs; objects are highlighted across tabs."""

    def __init__(self, action: str, banner: str = "", parent: QWidget | None = None) -> None:
        """A step with ``action`` on its button and ``banner`` above it (if any)."""
        super().__init__(parent)
        self._colours: Palette = LIGHT
        self.banner = QLabel(banner)
        self.banner.setObjectName("stepBanner")
        self.banner.setWordWrap(True)
        self.action = QPushButton(action)
        self.action.setObjectName("primary")
        self.empty = hint("")
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs_by_key: dict[str, StageTab] = {}
        top = QHBoxLayout()
        top.addWidget(self.action)
        top.addWidget(self.empty, 1)
        self.top = top
        self.rows = QVBoxLayout()
        self.rows.setContentsMargins(0, 0, 0, 0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.banner)
        layout.addLayout(top)
        layout.addLayout(self.rows)
        layout.addWidget(self.tabs, 1)
        self.banner.setVisible(bool(banner))

    def show_tables(self, tables: Sequence[StageTable]) -> None:
        """Show the stage's tabs; the tab and the split chosen before stay chosen."""
        current = self.current_key()
        keep = {t.key for t in tables}
        for key in [k for k in self.tabs_by_key if k not in keep]:
            gone = self.tabs_by_key.pop(key)
            self.tabs.removeTab(self.tabs.indexOf(gone))
            gone.deleteLater()
        for position, stage_table in enumerate(tables):
            tab = self.tabs_by_key.get(stage_table.key)
            if tab is None:
                tab = StageTab()
                tab.set_palette(self._colours)
                tab.objectClicked.connect(self.highlight)
                self.tabs_by_key[stage_table.key] = tab
                self.tabs.insertTab(position, tab, tab_title(stage_table))
            elif self.tabs.indexOf(tab) != position:
                self.tabs.removeTab(self.tabs.indexOf(tab))
                self.tabs.insertTab(position, tab, tab_title(stage_table))
            tab.set_stage_table(stage_table)
        if current in self.tabs_by_key:
            self.tabs.setCurrentWidget(self.tabs_by_key[current])

    def current_key(self) -> str | None:
        """The key of the tab on screen."""
        widget = self.tabs.currentWidget()
        return next((k for k, t in self.tabs_by_key.items() if t is widget), None)

    def open(self, key: str) -> bool:
        """Bring a tab to the front."""
        tab = self.tabs_by_key.get(key)
        if tab is not None:
            self.tabs.setCurrentWidget(tab)
        return tab is not None

    def highlight(self, name: Cell) -> None:
        """Mark an object in every table of the step (cross-highlighting)."""
        for tab in self.tabs_by_key.values():
            tab.panel.highlight(name)

    def set_palette(self, colours: Palette) -> None:
        """Switch the theme."""
        self._colours = colours
        for tab in self.tabs_by_key.values():
            tab.set_palette(colours)


# ---------------------------------------------------------------- a part


class PartView(QWidget):
    """One classifier part: its samples, the three steps and their tabs."""

    finished = Signal()
    """A worker ended (successfully or not)."""
    optionsRead = Signal()
    """The sample options were read."""

    def __init__(self, part: ClassifierPart, parent: QWidget | None = None) -> None:
        """Build the view of ``part`` from its protocol."""
        super().__init__(parent)
        self.part = part
        self.task: Task | None = None
        self.running_step: int | None = None
        self.options: tuple[SampleOption, ...] = ()
        self.methods: dict[str, StageTable] = {}
        self.training: TrainingStep | None = None
        self.test: TestStep | None = None
        self.evaluation: EvaluationStep | None = None
        self.check: CheckOutcome | None = None
        self.base = [StepState.NOT_RUN] * len(STEPS)
        self._before = list(self.base)
        self.stale_reason = [""] * len(STEPS)
        self._read = False
        self._colours: Palette = LIGHT
        self.folder_dialog: Callable[[str, str], str] = self._ask_folder
        self.open_folder: Callable[[Path], None] = self._open_in_file_manager

        self.title = section_title(part.title)
        self.title.setWordWrap(True)
        self.description = hint(part.description)
        self.samples = QComboBox()
        self.samples.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.samples.setMinimumContentsLength(16)
        self.samples.setToolTip(tr("What the part runs on, with the sizes of the samples"))
        self.samples.currentIndexChanged.connect(self._option_changed)
        default = part.default_folder()
        self.splits = QLineEdit("" if default is None else str(default))
        self.splits.setToolTip(tr("The folder the samples are read from"))
        self.splits.editingFinished.connect(self._folder_edited)
        self._folder_text = self.splits.text()
        self.browse_splits = QPushButton(tr("Browse…"))
        self.browse_splits.clicked.connect(self._browse_splits)
        samples_row = QHBoxLayout()
        samples_row.addWidget(QLabel(tr("Samples")))
        samples_row.addWidget(self.samples, 3)
        samples_row.addSpacing(8)
        samples_row.addWidget(QLabel(tr("Split files")))
        samples_row.addWidget(self.splits, 2)
        samples_row.addWidget(self.browse_splits)

        self.stepper = Stepper()
        self.stepper.stepChosen.connect(self.go_to_step)
        self.run_all_button = QPushButton(tr("Run all"))
        self.run_all_button.setToolTip(tr("Training, test and evaluation in a row"))
        self.run_all_button.clicked.connect(self.run_all)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        self.progress.setMinimumWidth(90)
        steps_row = QHBoxLayout()
        steps_row.addWidget(self.stepper)
        steps_row.addStretch(1)
        steps_row.addWidget(self.run_all_button)
        steps_row.addWidget(self.progress, 1)

        self.views = (
            StepView(tr("Fit on the training sample")),
            StepView(
                tr("Classify the test sample"),
                tr(
                    "Classes hidden: the test objects are classified from their feature "
                    "values only."
                ),
            ),
            StepView(
                tr("Compare with the true classes"),
                tr(
                    "True classes revealed: the decisions are compared with the test "
                    "objects' classes."
                ),
            ),
        )
        self.training_view, self.test_view, self.evaluation_view = self.views
        self.training_view.action.setToolTip(
            tr("Fit every method on the training sample, with its fixed settings")
        )
        self.test_view.action.setToolTip(
            tr("Classify the test objects from their feature values; their classes are not used")
        )
        self.evaluation_view.action.setToolTip(
            tr("Compare every decision with the true class of the object")
        )
        self.training_view.action.clicked.connect(self.fit)
        self.test_view.action.clicked.connect(self.classify)
        self.evaluation_view.action.clicked.connect(self.compare)
        self.stack = QStackedWidget()
        for view in self.views:
            self.stack.addWidget(view)

        self.writable = isinstance(part, WritablePart)
        self.checkable = isinstance(part, CheckablePart)
        output: Path | None = None
        if isinstance(part, (WritablePart, CheckablePart)):
            output = part.default_output()
        self.output = QLineEdit("" if output is None else str(output))
        self.output.setToolTip(tr("Where the files are written and checked"))
        self.output.editingFinished.connect(self.refresh)
        self.browse_output = QPushButton(tr("Browse…"))
        self.browse_output.clicked.connect(self._browse_output)
        self.write_button = QPushButton(tr("Write the files"))
        self.write_button.clicked.connect(self.write)
        self.check_button = QPushButton(tr("Check the written files"))
        self.check_button.setToolTip(
            tr("Compare the files of the results folder with their checksums and a new run")
        )
        self.check_button.clicked.connect(self.start_check)
        self.open_button = QPushButton(tr("Open the folder"))
        self.open_button.clicked.connect(self._open_folder)
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel(tr("Results folder")))
        folder_row.addWidget(self.output, 1)
        folder_row.addWidget(self.browse_output)
        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.write_button)
        buttons_row.addWidget(self.check_button)
        buttons_row.addWidget(self.open_button)
        buttons_row.addStretch(1)
        self.files = QWidget()
        files_rows = QVBoxLayout(self.files)
        files_rows.setContentsMargins(0, 0, 0, 0)
        files_rows.addLayout(folder_row)
        files_rows.addLayout(buttons_row)
        self.write_button.setVisible(self.writable)
        self.check_button.setVisible(self.checkable)
        self.evaluation_view.rows.addWidget(self.files)
        self.files.setVisible(self.writable or self.checkable)

        self.status = QLabel(tr("Not run yet."))
        self.status.setWordWrap(True)
        self.problems = WarningBox()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.title)
        layout.addWidget(self.description)
        layout.addLayout(samples_row)
        layout.addLayout(steps_row)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.status)
        layout.addWidget(self.problems)
        self.go_to_step(0)
        self.refresh()

    # ------------------------------------------------------------------ state

    def is_running(self) -> bool:
        """Whether a worker runs."""
        return self.task is not None and self.task.isRunning()

    def wait(self, milliseconds: int = 300_000) -> bool:
        """Block until the worker ends (the tests; closing the window)."""
        return self.task is None or self.task.wait(milliseconds)

    def option(self) -> SampleOption | None:
        """The chosen samples."""
        index = self.samples.currentIndex()
        return self.options[index] if 0 <= index < len(self.options) else None

    def _splits(self) -> Path | None:
        text = self.splits.text().strip()
        return Path(text) if text else None

    def _output(self) -> Path:
        return Path(self.output.text().strip() or ".")

    def _unavailable(self) -> str:
        """Why the steps cannot run now (empty if they can)."""
        option = self.option()
        if not self._read or not self.options:
            return tr("The samples have not been read yet.")
        if option is None:
            return tr("Choose the samples.")
        return "" if option.available else option.reason

    def refresh(self) -> None:
        """Show every step's state and enable what is possible now."""
        running = self.is_running()
        unavailable = self._unavailable()
        for index, state in enumerate(self.base):
            if unavailable and state is not StepState.RUNNING:
                self.stepper.set_state(index, StepState.UNAVAILABLE, unavailable)
            else:
                self.stepper.set_state(index, state, self.stale_reason[index])
        ready = not running and not unavailable
        self.training_view.action.setEnabled(ready)
        self.test_view.action.setEnabled(ready and self.stepper.can_open(1))
        self.evaluation_view.action.setEnabled(ready and self.stepper.can_open(2))
        self.run_all_button.setEnabled(ready)
        for widget in (
            self.samples,
            self.splits,
            self.browse_splits,
            self.output,
            self.browse_output,
        ):
            widget.setEnabled(not running)
        problem = self._write_problem()
        self.write_button.setEnabled(not running and not problem)
        self.write_button.setToolTip(
            problem or tr("The tables as CSV, Markdown and LaTeX, the results and their checksums")
        )
        self.check_button.setEnabled(not running)
        self.open_button.setEnabled(self._output().is_dir())
        if not self.stepper.can_open(self.stack.currentIndex()):
            self.go_to_step(max(i for i in range(len(STEPS)) if self.stepper.can_open(i)))
        self.training_view.empty.setText(
            "" if self.training else tr("Fit the methods to see the training sample and results.")
        )
        self.test_view.empty.setText("" if self.test else tr("Not classified yet."))
        self.evaluation_view.empty.setText("" if self.evaluation else tr("Not compared yet."))

    def _write_problem(self) -> str:
        if not self.writable:
            return ""
        if self.evaluation is None or self.base[2] is not StepState.DONE:
            return tr("Compare with the true classes first.")
        return self.evaluation.write_problem

    def go_to_step(self, index: int) -> bool:
        """Open a step's view (only when the step before it is done)."""
        if not self.stepper.can_open(index):
            self.stepper.set_current(self.stack.currentIndex())
            return False
        self.stack.setCurrentIndex(index)
        self.stepper.set_current(index)
        return True

    def step(self, direction: int) -> bool:
        """Open the next (+1) or the previous (−1) step (Alt+Right / Alt+Left)."""
        target = self.stack.currentIndex() + direction
        return 0 <= target < len(STEPS) and self.go_to_step(target)

    def _stale(self, first: int, reason: str) -> None:
        """Mark the done steps from ``first`` on as out of date."""
        for index in range(first, len(STEPS)):
            if self.base[index] is StepState.DONE:
                self.base[index] = StepState.OUT_OF_DATE
                self.stale_reason[index] = reason

    def _launch(self, task: Task, status: str, step: int | None) -> None:
        self.task = task
        self.running_step = step
        self._before = list(self.base)
        if step is not None:
            self.base[step] = StepState.RUNNING
        task.progressed.connect(self._progress)
        task.failed.connect(self._failed)
        task.finished.connect(self._done)
        self.problems.set_text("")
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("")
        self.status.setText(status)
        task.start()
        self.refresh()

    # ------------------------------------------------------------------ the samples

    def showEvent(self, event: QShowEvent) -> None:
        """Read the sample options the first time the view is shown."""
        super().showEvent(event)
        if not self._read and self.task is None:
            self.read_options()

    def read_options(self) -> None:
        """Read what the part can run on, in a worker (the sizes need the files)."""
        if self.is_running():
            return
        task = Task.with_progress(part_options_job, self.part, self._splits(), parent=self)
        task.succeeded.connect(self._options_read)
        self._launch(task, tr("Reading the samples…"), None)

    def _options_read(self, found: Sequence[tuple[SampleOption, StageTable]]) -> None:
        previous = self.option()
        self._read = True
        self.options = tuple(option for option, _ in found)
        self.methods = {option.key: methods for option, methods in found}
        self.samples.blockSignals(True)
        self.samples.clear()
        for index, option in enumerate(self.options):
            self.samples.addItem(option.label, option.key)
            if not option.available:
                self.samples.setItemData(index, f"⊘ {option.reason}", Qt.ItemDataRole.ToolTipRole)
        keys = [o.key for o in self.options]
        chosen = keys.index(previous.key) if previous is not None and previous.key in keys else 0
        self.samples.setCurrentIndex(chosen)
        self.samples.blockSignals(False)
        self._show_option()
        self.progress.setFormat("")
        self.progress.setValue(0)
        self.status.setText(tr("Samples read from {folder}.").format(folder=self.splits.text()))
        self.optionsRead.emit()

    def _option_changed(self, index: int) -> None:
        self._stale(0, tr("The samples changed."))
        self._show_option()

    def _show_option(self) -> None:
        option = self.option()
        if option is None:
            return
        methods = self.methods.get(option.key)
        others = (
            [] if self.training is None else [t for t in self.training.tabs if t.key != "methods"]
        )
        if methods is not None:
            self.training_view.show_tables([methods, *others])
        self.problems.set_text("" if option.available else f"⚠ {option.reason}")
        self.refresh()

    def _folder_edited(self) -> None:
        if self.splits.text() != self._folder_text:
            self._folder_text = self.splits.text()
            self._stale(0, tr("The split folder changed."))
            self.read_options()

    def _browse_splits(self) -> None:
        folder = self.folder_dialog(tr("Split files"), self.splits.text())
        if folder:
            self.splits.setText(folder)
            self._folder_edited()

    def _browse_output(self) -> None:
        folder = self.folder_dialog(tr("Results folder"), self.output.text())
        if folder:
            self.output.setText(folder)
            self.refresh()

    def _ask_folder(self, title: str, current: str) -> str:
        return QFileDialog.getExistingDirectory(self, title, current)

    def _open_in_file_manager(self, folder: Path) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _open_folder(self) -> None:
        self.open_folder(self._output())

    # ------------------------------------------------------------------ the steps

    def fit(self) -> None:
        """① Fit every method on the training samples, in a worker."""
        option = self.option()
        if self.is_running() or option is None or self._unavailable():
            return
        task = Task.with_progress(part_train_job, self.part, option, self._splits(), parent=self)
        task.succeeded.connect(self._trained)
        self._launch(task, tr("Fitting on the training sample…"), 0)

    def classify(self) -> None:
        """② Classify the test objects without their class, in a worker."""
        if self.is_running() or self.training is None or not self.stepper.can_open(1):
            return
        task = Task.with_progress(
            part_classify_job,
            self.part,
            self.training.trained,
            self.training.samples.test,
            parent=self,
        )
        task.succeeded.connect(self._classified)
        self._launch(task, tr("Classifying the test objects without their class…"), 1)

    def compare(self) -> None:
        """③ Compare the decisions with the true classes, in a worker."""
        if (
            self.is_running()
            or self.training is None
            or self.test is None
            or not self.stepper.can_open(2)
        ):
            return
        task = Task.with_progress(
            part_evaluate_job,
            self.part,
            self.test.predictions,
            self.training.samples.answers,
            parent=self,
        )
        task.succeeded.connect(self._compared)
        self._launch(task, tr("Comparing with the true classes…"), 2)

    def run_all(self) -> None:
        """The three steps in a row, in one worker."""
        option = self.option()
        if self.is_running() or option is None or self._unavailable():
            return
        task = Task.with_progress(part_run_all_job, self.part, option, self._splits(), parent=self)
        task.succeeded.connect(self._all_done)
        self._launch(task, tr("Running the three steps…"), 0)

    def _trained(self, step: TrainingStep) -> None:
        self.training = step
        self.base[0] = StepState.DONE
        self.stale_reason[0] = ""
        self._stale(1, tr("① Training was run again."))
        self.training_view.show_tables(step.tabs)
        if self.test is None:
            self.test_view.show_tables(step.test_tabs)
        self.status.setText(
            tr("Fitted on {n} training sample(s) from {folder}.").format(
                n=len(step.samples.training), folder=self.splits.text()
            )
        )

    def _classified(self, step: TestStep) -> None:
        self.test = step
        self.base[1] = StepState.DONE
        self.stale_reason[1] = ""
        self._stale(2, tr("② Test was run again."))
        self.test_view.show_tables(step.tabs)
        self.test_view.open("predictions")
        self.go_to_step(1)
        self.status.setText(
            tr("{n} test object(s) classified without their class.").format(
                n=sum(s.m for s in step.predictions.objects)
            )
        )

    def _compared(self, step: EvaluationStep) -> None:
        self.evaluation = step
        self.base[2] = StepState.DONE
        self.stale_reason[2] = ""
        self.evaluation_view.show_tables(self._evaluation_tabs())
        self.go_to_step(2)
        self.status.setText(tr("Compared with the true classes."))

    def _all_done(self, done: StagesDone) -> None:
        self.base = [StepState.NOT_RUN] * len(STEPS)
        self._trained(done.training)
        self._classified(done.test)
        self._compared(done.evaluation)
        first = done.evaluation.tabs[0].key if done.evaluation.tabs else None
        if first is not None:
            self.evaluation_view.open(first)
        self.status.setText(tr("Training, test and evaluation done."))

    def _evaluation_tabs(self) -> list[StageTable]:
        tabs = [] if self.evaluation is None else list(self.evaluation.tabs)
        if self.checkable:
            if self.check is None:
                tabs.append(
                    StageTable("check", "Check", missing=tr("Check the written files to see this."))
                )
            else:
                tabs.append(self.check.tab)
        return tabs

    # ------------------------------------------------------------------ writing and checking

    def write(self) -> None:
        """Write the evaluation's files to the results folder, in a worker."""
        if self.is_running() or self.evaluation is None or self._write_problem():
            return
        task = Task.with_progress(
            part_write_job,
            self.part,
            self.evaluation.evaluation,
            self._output(),
            parent=self,
        )
        task.succeeded.connect(self._written)
        self._launch(task, tr("Writing the files…"), None)

    def _written(self, written: list[Path]) -> None:
        self.status.setText(
            tr("{n} files written to {folder}").format(n=len(written), folder=self._output())
        )

    def start_check(self) -> None:
        """Check the results folder against its checksums and a new run, in a worker."""
        if self.is_running() or not self.checkable:
            return
        task = Task.with_progress(
            part_check_job, self.part, self._output(), self._splits(), parent=self
        )
        task.succeeded.connect(self._checked)
        self._launch(task, tr("Checking…"), None)

    def _checked(self, check: CheckOutcome) -> None:
        self.check = check
        self.evaluation_view.show_tables(self._evaluation_tabs())
        self.evaluation_view.open("check")
        self.status.setText(
            tr("Check passed: {folder}").format(folder=check.check.folder)
            if check.check.passed
            else tr("Check failed: {n} problem(s) in {folder}").format(
                n=len(check.check.problems), folder=check.check.folder
            )
        )

    # ------------------------------------------------------------------ progress

    def _progress(self, stage: str, done: int, total: int) -> None:
        step = self.running_step
        what = stage
        prefix, _, rest = stage.partition(" · ")
        if prefix in STAGES:
            step = STAGES.index(prefix)
            what = rest
            for index in range(step):
                if self.base[index] is StepState.RUNNING:
                    self.base[index] = StepState.DONE
            self.base[step] = StepState.RUNNING
            self.refresh()
        label = what if step is None else f"{step_name(step)} · {what}"
        self.progress.setFormat(label)
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        if what != "done":
            self.status.setText(label + "…")

    def _failed(self, message: str) -> None:
        self.base = list(self._before)
        self.status.setText(tr("Failed."))
        self.problems.set_text(f"⚠ {message}")

    def _done(self) -> None:
        if self.running_step is not None:
            for index, state in enumerate(self.base):
                if state is StepState.RUNNING:
                    self.base[index] = self._before[index]
        self.running_step = None
        self.refresh()
        self.finished.emit()

    def set_palette(self, colours: Palette) -> None:
        """Switch the theme."""
        self._colours = colours
        for view in self.views:
            view.set_palette(colours)
