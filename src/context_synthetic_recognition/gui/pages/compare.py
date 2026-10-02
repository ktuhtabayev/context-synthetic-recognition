"""Compare page: saved runs side by side.

The runs come from the run folders (``manifest.yaml`` and ``results.json``); nothing is recomputed.
Rows in which the chosen runs differ are marked, and a second table lists the settings in which
their configurations differ — the quickest way to see what the two ⚠ switches change.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from context_synthetic_recognition.export.tables import Cell, Table
from context_synthetic_recognition.export.text import format_cell
from context_synthetic_recognition.gui.i18n import mark, tr
from context_synthetic_recognition.gui.models import TableStyle
from context_synthetic_recognition.gui.pages.base import Page
from context_synthetic_recognition.gui.settings import Settings
from context_synthetic_recognition.gui.state import AppState
from context_synthetic_recognition.gui.theme import CellRole, Palette
from context_synthetic_recognition.gui.widgets import TablePanel, WarningBox, hint
from context_synthetic_recognition.services.runs import (
    RunRecord,
    comparison,
    config_differences,
    differing_rows,
    list_runs,
    record_of,
    run_facts,
)

COLUMNS = (
    mark("Run"),
    mark("Created (UTC)"),
    mark("Dataset"),
    "m",
    mark("Preset"),
    "TUPLAM",
    mark("Accuracy"),
)
MAX_COMPARED = 6
"""More columns than this do not fit a screen."""


def difference_style(table: Table) -> TableStyle:
    """Mark the rows whose values are not the same in every run."""
    different = differing_rows(table)

    def cell(row: int, column: int, value: Cell) -> CellRole | None:
        if row in different:
            return CellRole.WARNING if column else CellRole.SELECTED
        return None

    return TableStyle((None,) * len(table.columns), cell)


class ComparePage(Page):
    """Choose runs and compare them."""

    title = mark("Compare")
    openRequested = Signal(object)
    """The user asked to open a run folder (a :class:`~pathlib.Path`)."""

    def __init__(self, state: AppState, settings: Settings, parent: QWidget | None = None) -> None:
        """Build the page."""
        super().__init__(
            state,
            settings,
            tr(
                "Tick two or more runs to see them side by side. Differences are marked; the "
                "second tab lists the settings that differ."
            ),
            parent,
        )
        self.records: list[RunRecord] = []
        self.folder_dialog = self._ask_folder

        self.runs_dir = QLineEdit(str(settings.runs_dir))
        self.runs_dir.editingFinished.connect(self._folder_typed)
        self.browse = QPushButton(tr("Browse…"))
        self.browse.clicked.connect(self._browse)
        self.refresh_button = QPushButton(tr("Refresh"))
        self.refresh_button.clicked.connect(self.reload)
        self.open_button = QPushButton(tr("Open the selected run"))
        self.open_button.setToolTip(
            tr("Repeat the run from its folder and show all its tables and figures")
        )
        self.open_button.clicked.connect(self._open)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Runs folder")))
        row.addWidget(self.runs_dir, 1)
        row.addWidget(self.browse)
        row.addWidget(self.refresh_button)
        row.addWidget(self.open_button)
        self.body.addLayout(row)
        self.problems = WarningBox()
        self.body.addWidget(self.problems)

        self.runs = QTableWidget(0, len(COLUMNS))
        self.runs.setHorizontalHeaderLabels([tr(name) for name in COLUMNS])
        self.runs.verticalHeader().setVisible(False)
        self.runs.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.runs.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.runs.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        header = self.runs.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.runs.itemChanged.connect(self._ticked)
        self.runs.itemSelectionChanged.connect(self._selection_changed)

        self.facts = TablePanel()
        self.settings_table = TablePanel()
        self.message = hint(tr("Tick at least two runs."), "note")
        facts_tab = QWidget()
        facts_layout = QVBoxLayout(facts_tab)
        facts_layout.setContentsMargins(8, 8, 8, 8)
        facts_layout.addWidget(self.message)
        facts_layout.addWidget(self.facts, 1)
        facts_layout.addStretch(0)
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        settings_layout.addWidget(self.settings_table)
        self.tabs = QTabWidget()
        self.tabs.addTab(facts_tab, tr("Results side by side"))
        self.tabs.addTab(settings_tab, tr("Settings that differ"))

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.runs)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        self.body.addWidget(splitter, 1)

        state.resultChanged.connect(self._result_changed)
        self.apply_palette(self.colours)
        self.reload()

    # ------------------------------------------------------------------ the list of runs

    def _ask_folder(self) -> str:
        return QFileDialog.getExistingDirectory(self, tr("Runs folder"), self.runs_dir.text())

    def _browse(self) -> None:
        folder = self.folder_dialog()
        if folder:
            self.runs_dir.setText(folder)
            self._folder_typed()

    def _folder_typed(self) -> None:
        """The user chose another runs folder: remember it and read it."""
        self.settings.runs_dir = Path(self.runs_dir.text().strip() or "runs")
        self.reload()

    def _result_changed(self) -> None:
        """A run arrived: show the folder runs are saved to (the Run page may have changed it)."""
        self.runs_dir.setText(str(self.settings.runs_dir))
        self.reload()

    def reload(self) -> None:
        """Read the runs folder again, keeping the ticks of the runs that are still there."""
        ticked = {record.run_id for record in self.checked()}
        folder = Path(self.runs_dir.text().strip() or "runs")
        records, problems = list_runs(folder)
        view = self.state.view
        if view is not None and self.state.run_folder is None:
            records.insert(0, record_of(view.result))  # a result that was not saved
        self.records = records
        self.problems.set_text("\n".join(f"⚠ {problem}" for problem in problems[:5]))
        self.runs.blockSignals(True)
        self.runs.setRowCount(len(records))
        for r, record in enumerate(records):
            facts = dict(run_facts(record))
            protocols = [p for p in record.protocols if p != "resubstitution"] or list(
                record.protocols
            )
            accuracy = record.accuracy(protocols[0]) if protocols else None
            shown = "—" if accuracy is None else f"{100 * accuracy:.1f} % ({protocols[0]})"
            cells = (
                record.run_id,
                record.created.strftime("%Y-%m-%d %H:%M"),
                record.dataset,
                format_cell(facts.get("Objects m")),
                format_cell(facts.get("Preset")),
                format_cell(facts.get("TUPLAM")),
                shown,
            )
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    state = (
                        Qt.CheckState.Checked
                        if record.run_id in ticked
                        else Qt.CheckState.Unchecked
                    )
                    item.setCheckState(state)
                self.runs.setItem(r, c, item)
        self.runs.blockSignals(False)
        # room for the tick box beside the run id
        self.runs.setColumnWidth(0, self.runs.sizeHintForColumn(0) + 36)
        self._selection_changed()
        self.compare()

    def checked(self) -> list[RunRecord]:
        """The ticked runs, in the order of the list."""
        found = []
        for row, record in enumerate(self.records):
            item = self.runs.item(row, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                found.append(record)
        return found

    def set_checked(self, row: int, on: bool = True) -> None:
        """Tick or untick a run."""
        item = self.runs.item(row, 0)
        if item is not None:
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)

    def _ticked(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            self.compare()

    def _selection_changed(self) -> None:
        row = self.runs.currentRow()
        folder = self.records[row].folder if 0 <= row < len(self.records) else None
        self.open_button.setEnabled(folder is not None)

    def _open(self) -> None:
        row = self.runs.currentRow()
        if 0 <= row < len(self.records) and self.records[row].folder is not None:
            self.openRequested.emit(self.records[row].folder)

    # ------------------------------------------------------------------ the comparison

    def compare(self) -> None:
        """Show the ticked runs side by side."""
        chosen = self.checked()
        enough = len(chosen) >= 2
        self.facts.setVisible(enough)
        self.settings_table.setVisible(enough)
        if not enough:
            self.message.setText(tr("Tick at least two runs."))
            self.message.setVisible(True)
            return
        shown = chosen[:MAX_COMPARED]
        self.message.setVisible(len(chosen) > MAX_COMPARED)
        self.message.setText(
            tr("The first {n} of the {total} ticked runs are shown.").format(
                n=MAX_COMPARED, total=len(chosen)
            )
        )
        facts = comparison(shown)
        self.facts.set_table(facts, difference_style(facts))
        settings = config_differences(shown)
        self.settings_table.set_table(settings, difference_style(settings))

    def apply_palette(self, colours: Palette) -> None:
        """Re-colour the tables."""
        self.facts.set_palette(colours)
        self.settings_table.set_palette(colours)
