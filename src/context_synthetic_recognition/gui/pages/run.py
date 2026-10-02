"""Run page: start the experiment, watch it, cancel it.

The experiment runs in a worker thread (:mod:`..workers`); the page only shows progress, the log
of the package and, at the end, the key results.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from context_synthetic_recognition.config.presets import active_deviations
from context_synthetic_recognition.export.tables import run_tables
from context_synthetic_recognition.export.view import RunView
from context_synthetic_recognition.gui.i18n import mark, tr
from context_synthetic_recognition.gui.models import style_for
from context_synthetic_recognition.gui.pages.base import Page
from context_synthetic_recognition.gui.settings import RECENT_RUNS, Settings
from context_synthetic_recognition.gui.state import AppState
from context_synthetic_recognition.gui.theme import Palette
from context_synthetic_recognition.gui.widgets import (
    TablePanel,
    WarningBox,
    card,
    hint,
    section_title,
)
from context_synthetic_recognition.gui.workers import (
    LogBridge,
    RunOutcome,
    SensitivityChoice,
    Task,
    open_job,
    run_job,
)
from context_synthetic_recognition.services.configs import config_problems
from context_synthetic_recognition.services.datasets import summarize

MAX_LOG_BLOCKS = 5000


class RunPage(Page):
    """Run the experiment in the background."""

    title = mark("Run")
    finished = Signal()
    """A run ended (successfully or not)."""

    def __init__(self, state: AppState, settings: Settings, parent: QWidget | None = None) -> None:
        """Build the page."""
        super().__init__(
            state,
            settings,
            tr(
                "Fit the model on every object and evaluate it under every configured protocol; "
                "leave-one-out re-fits the whole pipeline for every object."
            ),
            parent,
        )
        self.task: Task | None = None
        self._ok = False
        self.folder_dialog = self._ask_folder

        # ---- what will run
        self.plan = hint("", "note")
        self.blockers = WarningBox()
        self.stale = WarningBox()
        self.body.addWidget(self.plan)
        self.body.addWidget(self.blockers)
        self.body.addWidget(self.stale)

        # ---- options
        options, options_layout = card()
        form = QFormLayout()
        self.save = QCheckBox(tr("Save a run folder (manifest, results, dataset snapshot)"))
        self.save.setChecked(True)
        self.runs_dir = QLineEdit(str(settings.runs_dir))
        self.browse = QPushButton(tr("Browse…"))
        self.browse.clicked.connect(self._browse)
        self.save.toggled.connect(self.runs_dir.setEnabled)
        self.save.toggled.connect(self.browse.setEnabled)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.runs_dir, 1)
        folder_row.addWidget(self.browse)
        self.sensitivity = QComboBox()
        self.sensitivity.addItem(tr("automatically (samples of at most 60 objects)"), "auto")
        self.sensitivity.addItem(tr("always"), "yes")
        self.sensitivity.addItem(tr("never"), "no")
        self.sensitivity.setToolTip(
            tr("Evaluate all four settings of the two ⚠ switches (a leave-one-out for each)")
        )
        form.addRow(self.save)
        form.addRow(tr("Runs folder"), folder_row)
        form.addRow(tr("⚠ Switch sensitivity"), self.sensitivity)
        options_layout.addLayout(form)
        self.body.addWidget(options)

        # ---- controls
        self.run_button = QPushButton(tr("Run"))
        self.run_button.setObjectName("primary")
        self.run_button.setToolTip(tr("Run the experiment (F5)"))
        self.cancel_button = QPushButton(tr("Cancel"))
        self.cancel_button.setEnabled(False)
        self.stage = QLabel(tr("Ready."))
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        controls = QHBoxLayout()
        controls.addWidget(self.run_button)
        controls.addWidget(self.cancel_button)
        controls.addWidget(self.progress, 1)
        self.body.addLayout(controls)
        self.body.addWidget(self.stage)

        # ---- result summary | log
        self.summary = TablePanel(compact=True)
        self.summary_note = hint(tr("No run yet."), "note")
        self.explore_button = QPushButton(tr("Explore the results"))
        self.explore_button.setEnabled(False)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.addWidget(section_title(tr("Last run")))
        left_layout.addWidget(self.summary_note)
        left_layout.addWidget(self.summary, 1)
        left_layout.addWidget(self.explore_button)
        self.console = QPlainTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(MAX_LOG_BLOCKS)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.addWidget(section_title(tr("Log")))
        right_layout.addWidget(self.console, 1)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.body.addWidget(splitter, 1)

        self.log_bridge = LogBridge(self)
        self.log_bridge.message.connect(self.console.appendPlainText)
        self.log_bridge.attach()
        self.destroyed.connect(self.log_bridge.detach)

        self.run_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.cancel)
        state.datasetChanged.connect(self.refresh)
        state.configChanged.connect(self.refresh)
        state.resultChanged.connect(self.refresh)
        self.apply_palette(self.colours)
        self.refresh()

    def apply_palette(self, colours: Palette) -> None:
        """Re-colour the summary table."""
        self.summary.set_palette(colours)

    # ------------------------------------------------------------------ status

    def blockers_text(self) -> list[str]:
        """What keeps the experiment from running (empty if it can run)."""
        dataset = self.state.dataset
        if dataset is None:
            return [tr("no dataset is loaded (page Dataset)")]
        config = self.state.config
        problems = [*config_problems(config), *summarize(dataset, config).problems]
        if not config.evaluation.protocols:
            problems.append(tr("no evaluation protocol is chosen (page Configure)"))
        return list(dict.fromkeys(problems))

    def refresh(self) -> None:
        """Show what a run would do and whether it can start."""
        state = self.state
        dataset, config = state.dataset, state.config
        blockers = self.blockers_text()
        self.blockers.set_text("\n".join(f"✗ {b}" for b in blockers))
        running = self.is_running()
        self.run_button.setEnabled(not blockers and not running)
        if dataset is None:
            self.plan.setText(tr("Nothing to run yet."))
        else:
            protocols = ", ".join(spec.name for spec in config.evaluation.protocols) or "—"
            active = active_deviations(config)
            switches = (
                tr("⚠ template calculation for {fields}").format(
                    fields=", ".join(d.field for d in active)
                )
                if active
                else tr("article calculation for both switches")
            )
            self.plan.setText(
                tr("{name} · m = {m}, n = {n} · protocols: {protocols} · {switches}").format(
                    name=dataset.name,
                    m=dataset.m,
                    n=dataset.n,
                    protocols=protocols,
                    switches=switches,
                )
            )
        self.stale.set_text(
            tr(
                "⚠ The results on screen were computed with another configuration. "
                "Run again to update them."
            )
            if state.stale
            else ""
        )
        self._show_summary(state.view)

    def _show_summary(self, view: RunView | None) -> None:
        self.explore_button.setEnabled(view is not None)
        self.summary.setVisible(view is not None)
        if view is None:
            self.summary_note.setText(tr("No run yet."))
            return
        folder = self.state.run_folder
        self.summary_note.setText(
            tr("Not saved.") if folder is None else tr("Run folder: {folder}").format(folder=folder)
        )
        table = run_tables(view, keys=["summary"])["summary"]
        self.summary.set_table(table, style_for(table, view))

    def is_running(self) -> bool:
        """Whether a run is in progress."""
        return self.task is not None and self.task.isRunning()

    # ------------------------------------------------------------------ running

    def _ask_folder(self) -> str:
        return QFileDialog.getExistingDirectory(self, tr("Runs folder"), self.runs_dir.text())

    def _browse(self) -> None:
        folder = self.folder_dialog()
        if folder:
            self.runs_dir.setText(folder)

    def _sensitivity(self) -> SensitivityChoice:
        choice: SensitivityChoice = self.sensitivity.currentData()
        return choice

    def start(self) -> None:
        """Start the experiment in a worker thread."""
        dataset = self.state.dataset
        if dataset is None or self.is_running() or self.blockers_text():
            return
        runs_dir: Path | None = None
        if self.save.isChecked():
            runs_dir = Path(self.runs_dir.text().strip() or "runs")
            self.settings.runs_dir = runs_dir
        self.console.appendPlainText(tr("— run of {name} started —").format(name=dataset.name))
        self._launch(
            Task.with_progress(
                run_job,
                dataset,
                self.state.config,
                sensitivity=self._sensitivity(),
                runs_dir=runs_dir,
                parent=self,
            )
        )

    def open_run(self, folder: Path) -> None:
        """Repeat a saved run from its folder in a worker thread."""
        if self.is_running():
            return
        self.console.appendPlainText(
            tr("— repeating the run {name} from its folder —").format(name=folder.name)
        )
        self._launch(Task.with_progress(open_job, folder, parent=self), opened=True)

    def _launch(self, task: Task, *, opened: bool = False) -> None:
        self.task = task
        self._ok = False
        task.progressed.connect(self._progress)
        task.succeeded.connect(self._opened if opened else self._succeeded)
        task.failed.connect(self._failed)
        task.cancelled.connect(self._cancelled)
        task.finished.connect(self._done)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)  # busy until the first progress report
        self.stage.setText(tr("Fitting the model…"))
        task.start()

    def cancel(self) -> None:
        """Ask the running experiment to stop at the next fold."""
        if self.task is not None and self.task.isRunning():
            self.task.cancel()
            self.stage.setText(tr("Cancelling…"))
            self.cancel_button.setEnabled(False)

    def wait(self, milliseconds: int = 120_000) -> bool:
        """Block until the running task ends (used by the tests and on closing the window)."""
        return self.task is None or self.task.wait(milliseconds)

    def _progress(self, stage: str, done: int, total: int) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        self.stage.setText(
            tr("{stage}: {done} of {total}").format(stage=stage, done=done, total=total)
        )

    def _succeeded(self, outcome: RunOutcome) -> None:
        if outcome.folder is not None:
            self.settings.add_recent(RECENT_RUNS, str(outcome.folder))
        self.state.set_result(outcome.view, outcome.folder)
        self._report(outcome)

    def _opened(self, outcome: RunOutcome) -> None:
        assert outcome.folder is not None
        self.settings.add_recent(RECENT_RUNS, str(outcome.folder))
        self.state.open_run(outcome.view, outcome.folder)
        self._report(outcome)

    def _report(self, outcome: RunOutcome) -> None:
        for warning in outcome.warnings:
            self.console.appendPlainText(
                tr("⚠ the repeated run differs from the stored results — {text}").format(
                    text=warning
                )
            )
        view = outcome.view
        self._ok = True
        self.stage.setText(
            tr("Finished: TUPLAM = {tuplam}, p = {p}.").format(tuplam=view.hag.label, p=view.hag.p)
        )
        self.console.appendPlainText(tr("— finished —"))

    def _failed(self, message: str) -> None:
        self.stage.setText(tr("Failed: {message}").format(message=message))
        self.console.appendPlainText(tr("✗ {message}").format(message=message))

    def _cancelled(self) -> None:
        self.stage.setText(tr("Cancelled."))
        self.console.appendPlainText(tr("— cancelled —"))

    def _done(self) -> None:
        self.cancel_button.setEnabled(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if self._ok else 0)
        self.refresh()
        self.finished.emit()
