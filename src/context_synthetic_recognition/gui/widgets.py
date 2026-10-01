"""Reusable widgets: page chrome, the table panel, the figure panel."""

from __future__ import annotations

import csv
import io
from collections.abc import Callable
from pathlib import Path

from matplotlib.figure import Figure
from PySide6.QtCore import QAbstractItemModel, QModelIndex, QRect, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from context_synthetic_recognition.export.figures import FigureSpec, render, save_figure
from context_synthetic_recognition.export.tables import Cell, Table
from context_synthetic_recognition.export.text import DEFAULT_DECIMALS, csv_text, write_text
from context_synthetic_recognition.gui.canvas import Canvas
from context_synthetic_recognition.gui.i18n import tr
from context_synthetic_recognition.gui.models import SortProxy, TableModel, TableStyle
from context_synthetic_recognition.gui.theme import LIGHT, Palette

MAX_COLUMN_WIDTH = 320
"""Columns are sized to their content up to this width (long texts get a tooltip instead)."""
RESIZE_SAMPLE_ROWS = 200


# ---------------------------------------------------------------- page chrome


def page_title(text: str) -> QLabel:
    """The title of a page."""
    label = QLabel(text)
    label.setObjectName("pageTitle")
    return label


def hint(text: str, name: str = "pageHint") -> QLabel:
    """Small secondary text that wraps."""
    label = QLabel(text)
    label.setObjectName(name)
    label.setWordWrap(True)
    return label


def section_title(text: str) -> QLabel:
    """A heading inside a page."""
    label = QLabel(text)
    label.setObjectName("sectionTitle")
    return label


def badge(text: str, *, warning: bool = False) -> QLabel:
    """A small rounded label; ``warning`` gives the ⚠ colours."""
    label = QLabel(text)
    label.setObjectName("warningBadge" if warning else "badge")
    label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return label


def card() -> tuple[QFrame, QVBoxLayout]:
    """A bordered surface with a vertical layout."""
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 10)
    layout.setSpacing(6)
    return frame, layout


class WarningBox(QFrame):
    """A ⚠ box: what the user must not miss (the template deviations, problems of the data)."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        """Create the box; it hides itself while it has no text."""
        super().__init__(parent)
        self.setObjectName("warningBox")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        self._label = QLabel()
        self._label.setWordWrap(True)
        self._label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._label)
        self.set_text(text)

    def set_text(self, text: str) -> None:
        """Show ``text``; an empty text hides the box."""
        self._label.setText(text)
        self.setVisible(bool(text))

    def text(self) -> str:
        """The text shown."""
        return self._label.text()


# ---------------------------------------------------------------- tables


def _model_of(view: QAbstractItemView) -> QAbstractItemModel | None:
    """A view's model; ``None`` before one is set (the type stubs do not say so)."""
    return view.model()


class RoleHeader(QHeaderView):
    """A table header that paints every section in the colour of what the column means."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """A horizontal header with clickable (sortable) sections."""
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.colours: Palette = LIGHT
        self.setSectionsClickable(True)
        self.setHighlightSections(False)
        self.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

    def paintSection(self, painter: QPainter, rect: QRect, logicalIndex: int) -> None:
        """Fill with the role's colour, then draw the name and the sort arrow."""
        model = _model_of(self)
        if model is None or not rect.isValid():
            return
        colours = self.colours
        source = model.sourceModel() if isinstance(model, SortProxy) else model
        role = source.header_role(logicalIndex) if isinstance(source, TableModel) else None
        background, ink = (
            colours.cells[role] if role is not None else (colours.header, colours.text)
        )
        painter.save()
        painter.fillRect(rect, QColor(background))
        painter.setPen(QColor(colours.border))
        painter.drawLine(rect.topRight(), rect.bottomRight())
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        text = str(model.headerData(logicalIndex, Qt.Orientation.Horizontal) or "")
        if isinstance(model, SortProxy) and model.sortColumn() == logicalIndex:
            text += " ▲" if model.sortOrder() == Qt.SortOrder.AscendingOrder else " ▼"
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(ink))
        painter.drawText(
            rect.adjusted(6, 0, -6, 0),
            int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine),
            text,
        )
        painter.restore()


class TableView(QTableView):
    """A sortable table: Ctrl+C copies the selection, clicking a row selects its object."""

    objectClicked = Signal(object)
    """The object named in the clicked row (its name, e.g. ``S₃``)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Set up sorting, row selection and the coloured header."""
        super().__init__(parent)
        self._header = RoleHeader(self)
        self.setHorizontalHeader(self._header)
        # no column is sorted until the user clicks one: tables open in pipeline order
        self._header.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        self.setSortingEnabled(True)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(24)
        self.setWordWrap(False)
        self.clicked.connect(self._clicked)
        QShortcut(QKeySequence.StandardKey.Copy, self, self.copy_selection)

    def set_palette(self, colours: Palette) -> None:
        """Switch the header's theme."""
        self._header.colours = colours
        self._header.viewport().update()

    def table_model(self) -> TableModel | None:
        """The model behind the sort proxy."""
        model = self.model()
        source = model.sourceModel() if isinstance(model, SortProxy) else model
        return source if isinstance(source, TableModel) else None

    def _source_row(self, index: QModelIndex) -> int:
        model = self.model()
        return model.mapToSource(index).row() if isinstance(model, SortProxy) else index.row()

    def _clicked(self, index: QModelIndex) -> None:
        source = self.table_model()
        if source is not None:
            name = source.object_of(self._source_row(index))
            if name is not None:
                self.objectClicked.emit(name)

    def selection_text(self) -> str:
        """The selected rows (all rows if none is selected) as tab-separated text with a header."""
        model = _model_of(self)
        if model is None:
            return ""
        rows = sorted({index.row() for index in self.selectionModel().selectedIndexes()})
        if not rows:
            rows = list(range(model.rowCount()))
        columns = range(model.columnCount())
        lines = ["\t".join(str(model.headerData(c, Qt.Orientation.Horizontal)) for c in columns)]
        lines += ["\t".join(str(model.index(r, c).data() or "") for c in columns) for r in rows]
        return "\n".join(lines)

    def copy_selection(self) -> None:
        """Copy the selection to the clipboard (paste into Excel or a text editor)."""
        QGuiApplication.clipboard().setText(self.selection_text())

    def fit_columns(self) -> None:
        """Size the columns to their content, up to :data:`MAX_COLUMN_WIDTH`."""
        model = _model_of(self)
        if model is None:
            return
        metrics = self.fontMetrics()
        rows = min(model.rowCount(), RESIZE_SAMPLE_ROWS)
        for column in range(model.columnCount()):
            texts = [str(model.headerData(column, Qt.Orientation.Horizontal) or "") + " ▲"]
            texts += [str(model.index(row, column).data() or "") for row in range(rows)]
            width = max(metrics.horizontalAdvance(text) for text in texts) + 22
            self.setColumnWidth(column, min(width, MAX_COLUMN_WIDTH))


class TablePanel(QWidget):
    """A table with its title, its note, a filter box and copy / save buttons."""

    objectClicked = Signal(object)

    def __init__(self, parent: QWidget | None = None, *, compact: bool = False) -> None:
        """Create an empty panel; ``compact`` leaves out the title and the tool row."""
        super().__init__(parent)
        self._colours: Palette = LIGHT
        self._decimals = DEFAULT_DECIMALS
        self._model: TableModel | None = None
        self._proxy = SortProxy(self)
        self.title = section_title("")
        self.note = hint("", "note")
        self.filter = QLineEdit()
        self.filter.setPlaceholderText(tr("Filter rows…"))
        self.filter.setClearButtonEnabled(True)
        self.filter.setMaximumWidth(220)
        self.filter.textChanged.connect(self._proxy.setFilterFixedString)
        self.count = hint("", "note")
        self.copy_button = QPushButton(tr("Copy"))
        self.copy_button.setToolTip(tr("Copy the selected rows (all rows if none is selected)"))
        self.save_button = QPushButton(tr("Save CSV…"))
        self.view = TableView()
        self.view.setModel(self._proxy)
        self.view.objectClicked.connect(self.objectClicked)
        self.copy_button.clicked.connect(self.view.copy_selection)
        self.save_button.clicked.connect(self._save)
        self._proxy.rowsInserted.connect(self._count)
        self._proxy.rowsRemoved.connect(self._count)
        self._proxy.modelReset.connect(self._count)
        self._proxy.layoutChanged.connect(self._count)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        tools = QHBoxLayout()
        tools.addWidget(self.title)
        tools.addStretch(1)
        tools.addWidget(self.count)
        tools.addWidget(self.filter)
        tools.addWidget(self.copy_button)
        tools.addWidget(self.save_button)
        self._tools = QWidget()
        self._tools.setLayout(tools)
        tools.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tools)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.note)
        self._tools.setVisible(not compact)
        self.save_dialog: Callable[[str], str] = self._ask_path

    # ------------------------------------------------------------------ content

    def set_table(self, table: Table, style: TableStyle | None = None) -> None:
        """Show a table."""
        previous = self._model.table.key if self._model is not None else None
        self._model = TableModel(table, style, self._colours, self._decimals, self)
        self._proxy.setSourceModel(self._model)
        if previous != table.key:
            self.view.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
            self.filter.clear()
        self.title.setText(table.title)
        self.note.setText(table.note)
        self.note.setVisible(bool(table.note))
        self.view.fit_columns()
        self._count()

    def table(self) -> Table | None:
        """The table shown."""
        return None if self._model is None else self._model.table

    def model(self) -> TableModel | None:
        """The model of the table shown."""
        return self._model

    def set_palette(self, colours: Palette) -> None:
        """Switch the theme."""
        self._colours = colours
        self.view.set_palette(colours)
        if self._model is not None:
            self._model.set_palette(colours)

    def set_decimals(self, decimals: int) -> None:
        """Change the number of decimals shown."""
        self._decimals = decimals
        if self._model is not None:
            self._model.set_decimals(decimals)
            self.view.fit_columns()

    def highlight(self, name: Cell) -> None:
        """Highlight the rows of an object and scroll to the first of them."""
        if self._model is None:
            return
        self._model.set_highlight(name)
        rows = self._model.rows_of(name)
        if rows:
            index = self._proxy.mapFromSource(self._model.index(rows[0], 0))
            if index.isValid():
                self.view.scrollTo(index, QAbstractItemView.ScrollHint.EnsureVisible)

    def visible_rows(self) -> int:
        """Rows that pass the filter."""
        return self._proxy.rowCount()

    def _count(self) -> None:
        total = 0 if self._model is None else self._model.rowCount()
        shown = self._proxy.rowCount()
        text = tr("{n} rows").format(n=total)
        if shown != total:
            text = tr("{shown} of {n} rows").format(shown=shown, n=total)
        self.count.setText(text)

    # ------------------------------------------------------------------ saving

    def _ask_path(self, suggested: str) -> str:
        path, _ = QFileDialog.getSaveFileName(
            self, tr("Save the table"), suggested, tr("CSV files (*.csv)")
        )
        return path

    def _save(self) -> None:
        table = self.table()
        if table is None:
            return
        path = self.save_dialog(f"{table.key}.csv")
        if path:
            write_text(Path(path), csv_text(table))


def table_as_tsv(table: Table) -> str:
    """A table as tab-separated text at full precision (for the clipboard)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(table.columns)
    writer.writerows(["" if v is None else v for v in row] for row in table.rows)
    return buffer.getvalue()


# ---------------------------------------------------------------- figures


class FigurePanel(QWidget):
    """A matplotlib figure with zoom / pan and "Save as…" (PNG, SVG, PDF)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty panel."""
        super().__init__(parent)
        self._colours: Palette = LIGHT
        self._spec: FigureSpec | None = None
        self.canvas = Canvas()
        self.caption = hint("", "note")
        self.save_button = QPushButton(tr("Save figure…"))
        self.save_button.clicked.connect(self._save)
        self._tools = QHBoxLayout()
        self._tools.setContentsMargins(0, 0, 0, 0)
        self._tools.addStretch(1)
        self._tools.addWidget(self.save_button)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self._layout.addLayout(self._tools)
        self._layout.addWidget(self.canvas, 1)
        self._layout.addWidget(self.caption)
        self.save_dialog: Callable[[str], str] = self._ask_path

    def set_spec(self, spec: FigureSpec | None) -> None:
        """Show a figure (``None`` clears the panel)."""
        self._spec = spec
        self._draw()

    def spec(self) -> FigureSpec | None:
        """The figure shown."""
        return self._spec

    def figure(self) -> Figure | None:
        """The matplotlib figure on the canvas."""
        return self.canvas.figure()

    def set_palette(self, colours: Palette) -> None:
        """Switch the theme and redraw."""
        if colours is not self._colours:
            self._colours = colours
            self._draw()

    def _draw(self) -> None:
        spec = self._spec
        self.save_button.setEnabled(spec is not None)
        self.caption.setText("" if spec is None else spec.title)
        self.canvas.set_figure(None if spec is None else render(spec, self._colours.figures))

    def _ask_path(self, suggested: str) -> str:
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("Save the figure"),
            suggested,
            tr("PNG image (*.png);;SVG image (*.svg);;PDF document (*.pdf)"),
        )
        return path

    def _save(self) -> None:
        spec = self._spec
        if spec is None:
            return
        path = self.save_dialog(f"{spec.key}.png")
        if path:
            # a fresh figure at the export size: what is saved does not depend on the window
            save_figure(render(spec, self._colours.figures), Path(path))
