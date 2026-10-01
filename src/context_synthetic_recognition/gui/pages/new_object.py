"""New object page: classify an object and see why — its neighbours, its Ψ(r), B1 and B2 per step.

The object is represented relative to the training sample without its class (Theorem of the
article): the form has no class input. Taking a training object and leaving it out of its own
context repeats the workbook's check that the new-object path reproduces the training row.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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

from context_synthetic_recognition.data import Dataset
from context_synthetic_recognition.errors import CSRError
from context_synthetic_recognition.export import build_view
from context_synthetic_recognition.export.figures import object_context_figure
from context_synthetic_recognition.export.tables import Cell, Table, plain, run_tables
from context_synthetic_recognition.export.text import format_cell
from context_synthetic_recognition.export.view import RunView
from context_synthetic_recognition.export.wording import decision_text, theorem_rows
from context_synthetic_recognition.gui.canvas import Canvas
from context_synthetic_recognition.gui.i18n import tr
from context_synthetic_recognition.gui.models import TableStyle, feature_rows, style_for
from context_synthetic_recognition.gui.pages.base import Page
from context_synthetic_recognition.gui.settings import Settings
from context_synthetic_recognition.gui.state import AppState
from context_synthetic_recognition.gui.theme import CellRole, Palette
from context_synthetic_recognition.gui.widgets import TablePanel, hint, section_title
from context_synthetic_recognition.services.datasets import parse_object

VALUE_COLUMN = 2
NEW_OBJECT = "S"
TABLE_KEYS = ("new-object", "new-object-features", "new-object-steps")


def neighbour_table(view: RunView, operator: int) -> Table:
    """The neighbours of the view's new object under one operator, nearest first."""
    trace = view.trace
    representation = view.new_object.classification.representation
    assert representation is not None
    context = representation.context
    distances, order = context.distances[operator][0], context.orders[operator][0]
    ks = set(trace.permitted_k.ks)
    rows: list[tuple[Cell, ...]] = []
    for rank, index in enumerate(order.tolist(), start=1):
        rows.append(
            (
                rank,
                trace.object_ids[index],
                float(distances[index]),
                trace.classes[int(trace.class_index[index])],
                f"k = {rank}" if rank in ks else "",
            )
        )
    label = trace.operators[operator].label
    return Table(
        f"new-object-neighbours-{operator}",
        tr("Neighbours of S under {label}, nearest first").format(label=label),
        ("Rank", "Neighbour", "Distance", "Class", "Permitted k"),
        tuple(rows),
        "Step 11 · Preparation",
        tr("Equal distances go to the smaller original index; χ₁ counts the K1 rows up to rank k."),
        labels=2,
    )


class NewObjectPage(Page):
    """Classify an object typed by the user."""

    title = "New object"
    needs_result = True

    def __init__(self, state: AppState, settings: Settings, parent: QWidget | None = None) -> None:
        """Build the page."""
        super().__init__(
            state,
            settings,
            tr(
                "Type the feature values of an object (no class: the model does not use it), or "
                "take a training object and leave it out of its own context."
            ),
            parent,
        )
        self._dataset: Dataset | None = None

        # ---- the form
        self.source = QComboBox()
        self.source.setToolTip(tr("Fill the form with the values of a training object"))
        self.source.activated.connect(self._take_training_object)
        self.exclude = QCheckBox(tr("Leave it out of its own context"))
        self.exclude.setChecked(True)
        # only a training object can be left out of its own context
        self.source.currentIndexChanged.connect(lambda row: self.exclude.setEnabled(row > 0))
        self.exclude.setToolTip(
            tr("The object is not its own neighbour: the Theorem check of the workbook")
        )
        self.paste = QLineEdit()
        self.paste.setPlaceholderText(
            tr("Paste a row: n values separated by commas, semicolons, spaces or tabs")
        )
        self.paste.returnPressed.connect(self.fill_from_text)
        self.paste_button = QPushButton(tr("Paste from clipboard"))
        self.paste_button.clicked.connect(self.paste_clipboard)
        self.form = QTableWidget(0, 4)
        self.form.setHorizontalHeaderLabels(
            [tr("Feature"), tr("Type"), tr("Value"), tr("Training range")]
        )
        self.form.verticalHeader().setVisible(False)
        header = self.form.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(VALUE_COLUMN, QHeaderView.ResizeMode.Stretch)
        self.classify_button = QPushButton(tr("Classify"))
        self.classify_button.setObjectName("primary")
        self.classify_button.clicked.connect(self.classify)
        self.error = QLabel()
        self.error.setObjectName("errorText")
        self.error.setWordWrap(True)
        self.error.setVisible(False)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Training object")))
        row.addWidget(self.source, 1)
        left_layout.addLayout(row)
        left_layout.addWidget(self.exclude)
        paste_row = QHBoxLayout()
        paste_row.addWidget(self.paste, 1)
        paste_row.addWidget(self.paste_button)
        left_layout.addLayout(paste_row)
        left_layout.addWidget(self.form, 1)
        left_layout.addWidget(self.error)
        left_layout.addWidget(self.classify_button)

        # ---- the explanation
        self.decision = QLabel("—")
        self.decision.setObjectName("decision")
        self.decision.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scores = hint("", "note")
        self.theorem = hint("", "note")
        self.steps = TablePanel(compact=True)
        self.features = TablePanel(compact=True)
        self.values = TablePanel(compact=True)
        self.neighbours = TablePanel(compact=True)
        self.operator = QComboBox()
        self.operator.currentIndexChanged.connect(self._show_neighbours)
        neighbours_tab = QWidget()
        neighbours_layout = QVBoxLayout(neighbours_tab)
        neighbours_layout.setContentsMargins(8, 8, 8, 8)
        operator_row = QHBoxLayout()
        operator_row.addWidget(QLabel(tr("Operator")))
        operator_row.addWidget(self.operator)
        operator_row.addStretch(1)
        neighbours_layout.addLayout(operator_row)
        self._neighbour_split = QSplitter(Qt.Orientation.Horizontal)
        self._neighbour_split.addWidget(self.neighbours)
        self.canvas = Canvas(toolbar=False)
        self._neighbour_split.addWidget(self.canvas)
        self._neighbour_split.setStretchFactor(0, 1)
        self._neighbour_split.setStretchFactor(1, 1)
        self._neighbour_split.setSizes([420, 320])
        neighbours_layout.addWidget(self._neighbour_split, 1)
        self.tabs = QTabWidget()
        for panel, title in (
            (self.steps, tr("Steps: B1 and B2")),
            (self.features, tr("Ψ(r) of the object")),
        ):
            holder = QWidget()
            holder_layout = QVBoxLayout(holder)
            holder_layout.setContentsMargins(8, 8, 8, 8)
            holder_layout.addWidget(panel)
            self.tabs.addTab(holder, title)
        self.tabs.addTab(neighbours_tab, tr("Neighbours"))
        values_tab = QWidget()
        values_layout = QVBoxLayout(values_tab)
        values_layout.setContentsMargins(8, 8, 8, 8)
        values_layout.addWidget(self.values)
        self.tabs.addTab(values_tab, tr("Unified values"))

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.addWidget(section_title(tr("Decision of the meta-algorithm")))
        right_layout.addWidget(self.decision)
        right_layout.addWidget(self.scores)
        right_layout.addWidget(self.theorem)
        right_layout.addWidget(self.tabs, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([390, 780])
        self.body.addWidget(splitter, 1)

        state.resultChanged.connect(self.refresh)
        self.apply_palette(self.colours)
        self.refresh()

    # ------------------------------------------------------------------ the form

    def refresh(self) -> None:
        """Show the state's run: rebuild the form for another dataset, then the explanation."""
        view = self.state.view
        self.setEnabled(view is not None)
        if view is None:
            self._dataset = None
            self.form.setRowCount(0)
            self.decision.setText("—")
            self.scores.setText("")
            self.theorem.setText("")
            return
        if view.dataset is not self._dataset:
            self._dataset = view.dataset
            self._build_form(view.dataset)
            demo = view.new_object
            self.set_values(demo.values)
            self.exclude.setChecked(demo.exclude is not None)
            self.source.setCurrentIndex(0 if demo.exclude is None else demo.exclude + 1)
        self._explain(view)

    def _build_form(self, dataset: Dataset) -> None:
        self.source.clear()
        self.source.addItem(tr("— a new object —"), -1)
        for index, name in enumerate(dataset.object_ids):
            self.source.addItem(name, index)
        rows = feature_rows(dataset)
        self.form.setRowCount(len(rows))
        colours = self.colours
        for r, (name, kind, low, high, distinct) in enumerate(rows):
            role = CellRole.QUANTITATIVE if kind == "quantitative" else CellRole.NOMINAL
            if kind == "quantitative":
                info = f"{format_cell(plain(low))} … {format_cell(plain(high))}"
            else:
                levels = dataset.categories.get(name)
                codes = sorted({plain(v) for v in dataset.X[:, r]}, key=float)  # type: ignore[arg-type]
                shown = ", ".join(str(c) for c in codes[:8]) + (" …" if len(codes) > 8 else "")
                info = (
                    ", ".join(levels)
                    if levels
                    else tr("{n} codes: {codes}").format(n=distinct, codes=shown)
                )
            for column, text in ((0, name), (1, "I" if kind == "quantitative" else "J"), (3, info)):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if column < 2:
                    item.setBackground(QColor(colours.cells[role][0]))
                    item.setForeground(QColor(colours.cells[role][1]))
                self.form.setItem(r, column, item)
            self.form.setItem(r, VALUE_COLUMN, QTableWidgetItem(""))

    def set_values(self, values: np.ndarray | list[float]) -> None:
        """Put feature values into the form."""
        for row, value in enumerate(np.asarray(values, dtype=np.float64).tolist()):
            item = self.form.item(row, VALUE_COLUMN)
            if item is not None:
                item.setText(str(plain(value)))

    def values_text(self) -> str:
        """The values of the form as one row of text."""
        cells = []
        for row in range(self.form.rowCount()):
            item = self.form.item(row, VALUE_COLUMN)
            cells.append("" if item is None else item.text().strip())
        return ";".join(cells)

    def _take_training_object(self, row: int) -> None:
        dataset, index = self._dataset, int(self.source.itemData(row))
        if dataset is not None and index >= 0:
            self.set_values(dataset.X[index])

    def fill_from_text(self) -> bool:
        """Fill the form from the "paste a row" field; returns whether the text was a row."""
        return self._fill(self.paste.text())

    def paste_clipboard(self) -> None:
        """Fill the form from the clipboard."""
        self.paste.setText(QGuiApplication.clipboard().text().strip())
        self.fill_from_text()

    def _fill(self, text: str) -> bool:
        dataset = self._dataset
        if dataset is None:
            return False
        try:
            values = parse_object(dataset, text)
        except CSRError as error:
            self._show_error(str(error))
            return False
        self._show_error("")
        self.set_values(values)
        self.source.setCurrentIndex(0)
        return True

    def _show_error(self, message: str) -> None:
        self.error.setText(message)
        self.error.setVisible(bool(message))

    # ------------------------------------------------------------------ classification

    def classify(self) -> bool:
        """Classify the object of the form; returns whether the values could be read."""
        view, dataset = self.state.view, self._dataset
        if view is None or dataset is None:
            return False
        if any(not part for part in self.values_text().split(";")):
            self._show_error(tr("every feature needs a value"))
            return False
        try:
            values = parse_object(dataset, self.values_text())
            index = int(self.source.currentData())
            exclude = index if index >= 0 and self.exclude.isChecked() else None
            new_view = build_view(
                view.result,
                new_object=values,
                exclude=exclude,
                sensitivity=view.sensitivity,
                run_id=view.run_id,
            )
        except CSRError as error:
            self._show_error(str(error))
            return False
        self._show_error("")
        # the run now carries this object: the Results tables and the exports show it too
        self.state.set_result(new_view, self.state.run_folder)
        return True

    def _explain(self, view: RunView) -> None:
        demo = view.new_object
        classes = view.trace.classes
        self.decision.setText(decision_text(demo.decision, classes))
        meta = demo.classification.meta
        self.scores.setText(
            tr("score₁ = |B1|/|K1| = {s1:.4f},  score₂ = |B2|/|K2| = {s2:.4f}").format(
                s1=float(meta.scores1[0]), s2=float(meta.scores2[0])
            )
        )
        checks = theorem_rows(view)
        if demo.exclude is None:
            self.theorem.setText(tr("The class of the object is not an input of Ψ, D or R."))
        else:
            name = view.trace.object_ids[demo.exclude]
            self.theorem.setText(
                tr("Theorem check for {name}: {first}; {second}.").format(
                    name=name, first=checks[0][2], second=checks[1][2]
                )
            )
        tables = run_tables(view, keys=TABLE_KEYS)
        for panel, key in (
            (self.steps, "new-object-steps"),
            (self.features, "new-object-features"),
            (self.values, "new-object"),
        ):
            table = tables[key]
            panel.set_table(table, style_for(table, view))
        current = max(self.operator.currentIndex(), 0)
        self.operator.blockSignals(True)
        self.operator.clear()
        self.operator.addItems([o.label for o in view.trace.operators])
        self.operator.setCurrentIndex(min(current, self.operator.count() - 1))
        self.operator.blockSignals(False)
        self._show_neighbours()

    def _show_neighbours(self, *_: object) -> None:
        view = self.state.view
        o = self.operator.currentIndex()
        if view is None or o < 0:
            self.canvas.set_figure(None)
            return
        table = neighbour_table(view, o)
        base = style_for(table, view)
        ks = set(view.trace.permitted_k.ks)

        def cell(row: int, column: int, value: Cell) -> CellRole | None:
            if column == 4 and table.rows[row][0] in ks:
                return CellRole.SELECTED
            return base.cell(row, column, value) if base.cell else None

        self.neighbours.set_table(table, TableStyle(base.header_roles, cell, base.object_column))
        demo = view.new_object
        representation = demo.classification.representation
        assert representation is not None
        own = None if demo.exclude is None else int(view.trace.class_index[demo.exclude])
        name = NEW_OBJECT if demo.exclude is None else view.trace.object_ids[demo.exclude]
        figure = object_context_figure(
            view,
            o,
            name,
            representation.context.distances[o][0],
            representation.context.orders[o][0],
            self.colours.figures,
            own_class=own,
        )
        self.canvas.set_figure(figure)

    def apply_palette(self, colours: Palette) -> None:
        """Re-colour the tables and redraw the neighbourhood."""
        for panel in (self.steps, self.features, self.values, self.neighbours):
            panel.set_palette(colours)
        if self._dataset is not None:
            values = self.values_text()
            self._build_form(self._dataset)
            for row, text in enumerate(values.split(";")):
                item = self.form.item(row, VALUE_COLUMN)
                if item is not None:
                    item.setText(text)
        self._show_neighbours()
