"""Classifiers page: Tables 7 and 8 — standard classifiers on the article's fixed samples.

Apart from the CS-model's workflow (ADR-079): the page needs no dataset and no run. The comparison
runs in a worker thread through :mod:`~context_synthetic_recognition.services.comparison`; the
page shows its tables, writes its files with their checksums and checks written files against a
new run — the same as ``csr compare run`` and ``csr compare check``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QWidget,
)

from context_synthetic_recognition.comparison.tables import DECIMALS
from context_synthetic_recognition.errors import CSRError
from context_synthetic_recognition.gui.i18n import mark, tr
from context_synthetic_recognition.gui.pages.base import Page
from context_synthetic_recognition.gui.settings import Settings
from context_synthetic_recognition.gui.state import AppState
from context_synthetic_recognition.gui.theme import Palette
from context_synthetic_recognition.gui.widgets import TablePanel, WarningBox
from context_synthetic_recognition.gui.workers import (
    Task,
    comparison_check_job,
    comparison_job,
)
from context_synthetic_recognition.services.comparison import (
    OUTPUT_DIR,
    SPLITS_DIR,
    ComparisonCheck,
    ComparisonOutcome,
    check_table,
    default_output,
    splits_folder,
    write_comparison,
)

TABS = (mark("Table 7"), mark("Table 8"), mark("CS-model"), mark("All results"), mark("Check"))
"""The tabs: the article's two tables, the CS-model's, every split × method, the check."""


def _default_splits() -> str:
    try:
        return str(splits_folder())
    except CSRError:
        return str(Path("datasets") / SPLITS_DIR)


def _default_output() -> str:
    try:
        return str(default_output())
    except CSRError:
        return str(Path(*OUTPUT_DIR))


class ClassifiersPage(Page):
    """Run, write and check Tables 7 and 8."""

    title = mark("Classifiers")
    finished = Signal()
    """A run or a check ended (successfully or not)."""

    def __init__(self, state: AppState, settings: Settings, parent: QWidget | None = None) -> None:
        """Build the page."""
        super().__init__(
            state,
            settings,
            tr(
                "Tables 7 and 8: SVM, random forest and k-NN fitted on the article's training "
                "samples and scored on its test samples, with the CS-model on the same samples. "
                "Apart from the workflow: no dataset or run is needed."
            ),
            parent,
        )
        self.task: Task | None = None
        self.outcome: ComparisonOutcome | None = None
        self.check: ComparisonCheck | None = None
        self.open_folder = self._open_in_file_manager
        self.folder_dialog: Callable[[str, str], str] = self._ask_folder

        self.splits = QLineEdit(_default_splits())
        self.splits.setToolTip(tr("The folder of the split files (datasets/splits)"))
        self.output = QLineEdit(_default_output())
        self.output.setToolTip(tr("Where the tables are written (experiments/comparison)"))
        self.output.editingFinished.connect(self.refresh)
        self.browse_splits = QPushButton(tr("Browse…"))
        self.browse_splits.clicked.connect(lambda: self._browse(self.splits, tr("Split files")))
        self.browse_output = QPushButton(tr("Browse…"))
        self.browse_output.clicked.connect(lambda: self._browse(self.output, tr("Results folder")))
        form = QFormLayout()
        for label, edit, button in (
            (tr("Split files"), self.splits, self.browse_splits),
            (tr("Results folder"), self.output, self.browse_output),
        ):
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(button)
            form.addRow(label, row)
        self.body.addLayout(form)

        self.run_button = QPushButton(tr("Run"))
        self.run_button.setObjectName("primary")
        self.run_button.setToolTip(
            tr("Fit every method and the CS-model on the training samples (a few seconds)")
        )
        self.run_button.clicked.connect(self.start)
        self.write_button = QPushButton(tr("Write the files"))
        self.write_button.setToolTip(
            tr("The tables as CSV, Markdown and LaTeX, the results and their checksums")
        )
        self.write_button.clicked.connect(self.write)
        self.check_button = QPushButton(tr("Check the written files"))
        self.check_button.setToolTip(
            tr("Compare the files of the results folder with their checksums and a new run")
        )
        self.check_button.clicked.connect(self.start_check)
        self.open_button = QPushButton(tr("Open the folder"))
        self.open_button.clicked.connect(self._open_folder)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        controls = QHBoxLayout()
        for button in (self.run_button, self.write_button, self.check_button, self.open_button):
            controls.addWidget(button)
        controls.addWidget(self.progress, 1)
        self.body.addLayout(controls)
        self.status = QLabel(tr("Not run yet."))
        self.body.addWidget(self.status)
        self.problems = WarningBox()
        self.body.addWidget(self.problems)

        self.panels = tuple(TablePanel() for _ in TABS)
        self.table_7, self.table_8, self.cs_model, self.results, self.check_panel = self.panels
        self.tabs = QTabWidget()
        for panel, name in zip(self.panels, TABS, strict=True):
            panel.set_decimals(DECIMALS)
            self.tabs.addTab(panel, tr(name))
        self.body.addWidget(self.tabs, 1)

        self.apply_palette(self.colours)
        self.refresh()

    # ------------------------------------------------------------------ state

    def is_running(self) -> bool:
        """Whether a run or a check is in progress."""
        return self.task is not None and self.task.isRunning()

    def refresh(self) -> None:
        """Enable what is possible now."""
        running = self.is_running()
        self.run_button.setEnabled(not running)
        self.check_button.setEnabled(not running)
        self.write_button.setEnabled(not running and self.outcome is not None)
        self.open_button.setEnabled(Path(self.output.text().strip() or ".").is_dir())

    def _ask_folder(self, title: str, current: str) -> str:
        return QFileDialog.getExistingDirectory(self, title, current)

    def _browse(self, edit: QLineEdit, title: str) -> None:
        folder = self.folder_dialog(title, edit.text())
        if folder:
            edit.setText(folder)
            self.refresh()

    def _splits(self) -> Path | None:
        text = self.splits.text().strip()
        return Path(text) if text else None

    def _output(self) -> Path:
        return Path(self.output.text().strip() or Path(*OUTPUT_DIR))

    def wait(self, milliseconds: int = 300_000) -> bool:
        """Block until a run or a check ends (used by the tests and on closing the window)."""
        return self.task is None or self.task.wait(milliseconds)

    def _launch(self, task: Task, status: str) -> None:
        self.task = task
        task.progressed.connect(self._progress)
        task.failed.connect(self._failed)
        task.finished.connect(self._done)
        self.problems.set_text("")
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status.setText(status)
        task.start()
        self.refresh()

    # ------------------------------------------------------------------ run

    def start(self) -> None:
        """Run the comparison in a worker thread."""
        if self.is_running():
            return
        task = Task.with_progress(comparison_job, self._splits(), parent=self)
        task.succeeded.connect(self._ran)
        self._launch(task, tr("Running…"))

    def _ran(self, outcome: ComparisonOutcome) -> None:
        self.outcome = outcome
        table_7, table_8, cs_model, results = outcome.tables
        self.table_7.set_table(table_7)
        self.table_8.set_table(table_8)
        self.cs_model.set_table(cs_model)
        self.results.set_table(results)
        versions = ", ".join(f"{name} {version}" for name, version in outcome.versions.items())
        self.status.setText(
            tr("Done: {n} splits from {folder} ({versions}). Not written yet.").format(
                n=len(outcome.results), folder=outcome.splits_folder, versions=versions
            )
        )
        self.tabs.setCurrentWidget(self.table_7)

    # ------------------------------------------------------------------ write and check

    def write(self) -> None:
        """Write the files of the last run to the results folder."""
        if self.outcome is None or self.is_running():
            return
        folder = self._output()
        try:
            written = write_comparison(self.outcome, folder)
        except OSError as error:
            self.problems.set_text(f"⚠ {error}")
            return
        self.status.setText(
            tr("{n} files written to {folder}").format(n=len(written), folder=folder)
        )
        self.refresh()

    def start_check(self) -> None:
        """Check the results folder against its checksums and a new run, in a worker thread."""
        if self.is_running():
            return
        task = Task.with_progress(comparison_check_job, self._output(), self._splits(), parent=self)
        task.succeeded.connect(self._checked)
        self._launch(task, tr("Checking…"))

    def _checked(self, check: ComparisonCheck) -> None:
        self.check = check
        self.check_panel.set_table(check_table(check))
        self.tabs.setCurrentWidget(self.check_panel)
        self.status.setText(
            tr("Check passed: {folder}").format(folder=check.folder)
            if check.passed
            else tr("Check failed: {n} problem(s) in {folder}").format(
                n=len(check.problems), folder=check.folder
            )
        )

    # ------------------------------------------------------------------ progress

    def _progress(self, stage: str, done: int, total: int) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        if stage != "done":
            self.status.setText(tr("Split {stage}…").format(stage=stage))

    def _failed(self, message: str) -> None:
        self.status.setText(tr("Failed."))
        self.problems.set_text(f"⚠ {message}")

    def _done(self) -> None:
        self.refresh()
        self.finished.emit()

    def _open_in_file_manager(self, folder: Path) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _open_folder(self) -> None:
        self.open_folder(self._output())

    def apply_palette(self, colours: Palette) -> None:
        """Re-colour the tables."""
        for panel in self.panels:
            panel.set_palette(colours)
