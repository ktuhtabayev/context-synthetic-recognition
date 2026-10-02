"""Export page: write the run as a workbook, tables, figures and a report.

The formats are the plug-ins of the exporters' registry, so a format added there appears here.
The export runs in a worker thread and exports what is on screen — including the object of the
New object page.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QWidget,
)

from context_synthetic_recognition.export import EXPORTERS, ExportOptions, ExportSummary
from context_synthetic_recognition.export.figures import DPI
from context_synthetic_recognition.export.text import DEFAULT_DECIMALS
from context_synthetic_recognition.export.theme import THEMES
from context_synthetic_recognition.gui.i18n import mark, tr
from context_synthetic_recognition.gui.pages.base import Page
from context_synthetic_recognition.gui.settings import Settings
from context_synthetic_recognition.gui.state import AppState
from context_synthetic_recognition.gui.widgets import WarningBox, card, hint, section_title
from context_synthetic_recognition.gui.workers import Task, export_job

DEFAULT_FORMATS = ("excel", "csv", "json", "figures", "html")
"""Ticked at first: everything except the article tables and the PDF (an optional extra)."""


class ExportPage(Page):
    """Choose formats and write them."""

    title = mark("Export")
    needs_result = True
    finished = Signal()

    def __init__(self, state: AppState, settings: Settings, parent: QWidget | None = None) -> None:
        """Build the page."""
        super().__init__(
            state,
            settings,
            tr(
                "Write the run on screen: the Excel mirror of the experiment workbook, every "
                "table, the figures and the run report."
            ),
            parent,
        )
        self.task: Task | None = None
        self.summary: ExportSummary | None = None
        self.folder_dialog = self._ask_folder
        self.open_folder = self._open_in_file_manager

        formats, formats_layout = card()
        formats_layout.addWidget(section_title(tr("Formats")))
        grid = QGridLayout()
        self.boxes: dict[str, QCheckBox] = {}
        for index, info in enumerate(EXPORTERS):
            box = QCheckBox(info.name)
            box.setChecked(info.name in DEFAULT_FORMATS)
            box.toggled.connect(self.refresh)
            self.boxes[info.name] = box
            grid.addWidget(box, index, 0)
            grid.addWidget(hint(info.summary, "note"), index, 1)
        grid.setColumnStretch(1, 1)
        formats_layout.addLayout(grid)
        select = QHBoxLayout()
        self.all_button = QPushButton(tr("All"))
        self.none_button = QPushButton(tr("None"))
        self.all_button.clicked.connect(lambda: self.set_formats(list(self.boxes)))
        self.none_button.clicked.connect(lambda: self.set_formats([]))
        select.addWidget(self.all_button)
        select.addWidget(self.none_button)
        select.addStretch(1)
        formats_layout.addLayout(select)
        formats_layout.addStretch(1)

        options, options_layout = card()
        options_layout.addWidget(section_title(tr("Options")))
        form = QFormLayout()
        self.folder = QLineEdit()
        self.browse = QPushButton(tr("Browse…"))
        self.browse.clicked.connect(self._browse)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(self.browse)
        self.decimals = QSpinBox()
        self.decimals.setRange(0, 15)
        self.decimals.setValue(DEFAULT_DECIMALS)
        self.decimals.setToolTip(tr("Decimals in Markdown, LaTeX and the report"))
        self.dpi = QSpinBox()
        self.dpi.setRange(50, 600)
        self.dpi.setValue(DPI)
        self.dpi.setToolTip(tr("Resolution of the PNG figures"))
        self.figure_theme = QComboBox()
        self.figure_theme.addItems(list(THEMES))
        self.figure_theme.setToolTip(tr("light for print and the article"))
        form.addRow(tr("Folder"), folder_row)
        form.addRow(tr("Decimals"), self.decimals)
        form.addRow(tr("Figure resolution (dpi)"), self.dpi)
        form.addRow(tr("Figure colours"), self.figure_theme)
        options_layout.addLayout(form)
        options_layout.addStretch(1)
        cards = QHBoxLayout()
        cards.addWidget(formats, 1)
        cards.addWidget(options, 1)
        self.body.addLayout(cards)

        self.demo = hint("", "note")
        self.body.addWidget(self.demo)
        self.stale = WarningBox()
        self.body.addWidget(self.stale)

        self.export_button = QPushButton(tr("Export"))
        self.export_button.setObjectName("primary")
        self.export_button.clicked.connect(self.start)
        self.open_button = QPushButton(tr("Open the folder"))
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_folder)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        controls = QHBoxLayout()
        controls.addWidget(self.export_button)
        controls.addWidget(self.open_button)
        controls.addWidget(self.progress, 1)
        self.body.addLayout(controls)
        self.status = QLabel("")
        self.body.addWidget(self.status)
        self.report = QPlainTextEdit()
        self.report.setObjectName("console")
        self.report.setReadOnly(True)
        self.body.addWidget(self.report, 1)

        state.resultChanged.connect(self._result_changed)
        state.configChanged.connect(self.refresh)
        self._result_changed()

    # ------------------------------------------------------------------ choices

    def formats(self) -> list[str]:
        """The ticked formats, in the registry's order."""
        return [name for name, box in self.boxes.items() if box.isChecked()]

    def set_formats(self, names: list[str]) -> None:
        """Tick exactly these formats."""
        for name, box in self.boxes.items():
            box.setChecked(name in names)

    def options(self) -> ExportOptions:
        """The export options of the page."""
        return ExportOptions(
            decimals=self.decimals.value(),
            dpi=self.dpi.value(),
            theme=self.figure_theme.currentText(),
        )

    def _ask_folder(self) -> str:
        return QFileDialog.getExistingDirectory(self, tr("Export folder"), self.folder.text())

    def _browse(self) -> None:
        folder = self.folder_dialog()
        if folder:
            self.folder.setText(folder)

    def _result_changed(self) -> None:
        """A run arrived: suggest its folder (or ``exports/<dataset>`` for an unsaved run)."""
        view = self.state.view
        if view is not None:
            if self.state.run_folder is not None:
                self.folder.setText(str(self.state.run_folder))
            elif not self.folder.text():
                self.folder.setText(str(Path("exports") / view.dataset.name))
        self.refresh()

    def refresh(self, *_: object) -> None:
        """Enable the export when there is a run, a format and no export in progress."""
        view = self.state.view
        running = self.task is not None and self.task.isRunning()
        self.export_button.setEnabled(view is not None and bool(self.formats()) and not running)
        if view is None:
            self.demo.setText(tr("Run the experiment first."))
        else:
            demo = view.new_object
            who = (
                tr("a new object")
                if demo.exclude is None
                else tr("{name} left out of its own context").format(
                    name=view.trace.object_ids[demo.exclude]
                )
            )
            switches = (
                tr("with the four ⚠ switch settings")
                if view.sensitivity
                else tr("without the switch sensitivity (not evaluated for this run)")
            )
            self.demo.setText(
                tr("Meta-algorithm sheets show {who}; {switches}.").format(
                    who=who, switches=switches
                )
            )
        self.stale.set_text(
            tr(
                "⚠ The run on screen was computed with another configuration than the one on the "
                "Configure page; the export shows the run, not the edited configuration."
            )
            if self.state.stale
            else ""
        )

    # ------------------------------------------------------------------ exporting

    def start(self) -> None:
        """Write the ticked formats in a worker thread."""
        view = self.state.view
        formats = self.formats()
        if view is None or not formats or (self.task is not None and self.task.isRunning()):
            return
        folder = Path(self.folder.text().strip() or "exports")
        self.report.clear()
        self.summary = None
        options = self.options()  # read in the interface's thread, used in the worker's
        self.task = Task(
            lambda task: export_job(view, folder, formats, options, progress=task.report),
            self,
        )
        self.task.progressed.connect(self._progress)
        self.task.succeeded.connect(self._succeeded)
        self.task.failed.connect(self._failed)
        self.task.finished.connect(self._done)
        self.export_button.setEnabled(False)
        self.progress.setRange(0, len(formats))
        self.progress.setValue(0)
        self.status.setText(tr("Exporting…"))
        self.task.start()

    def wait(self, milliseconds: int = 300_000) -> bool:
        """Block until the export ends (used by the tests and on closing the window)."""
        return self.task is None or self.task.wait(milliseconds)

    def _progress(self, name: str, done: int, total: int) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        if name != "done":
            self.status.setText(tr("Exporting {name}…").format(name=name))

    def _succeeded(self, summary: ExportSummary) -> None:
        self.summary = summary
        lines = []
        for name, files in summary.files.items():
            first = files[0].relative_to(summary.folder).as_posix()
            where = first if len(files) == 1 else f"{Path(first).parent.as_posix()}/"
            lines.append(
                tr("{name}: {where}  ({n} file(s))").format(name=name, where=where, n=len(files))
            )
        lines += [tr("note: {note}").format(note=note) for note in summary.notes]
        self.report.setPlainText("\n".join(lines))
        self.status.setText(
            tr("{n} files written to {folder}").format(n=summary.count, folder=summary.folder)
        )
        self.open_button.setEnabled(True)

    def _failed(self, message: str) -> None:
        self.status.setText(tr("Export failed: {message}").format(message=message))
        self.report.setPlainText(message)

    def _done(self) -> None:
        self.refresh()
        self.finished.emit()

    def _open_in_file_manager(self, folder: Path) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _open_folder(self) -> None:
        if self.summary is not None:
            self.open_folder(self.summary.folder)
