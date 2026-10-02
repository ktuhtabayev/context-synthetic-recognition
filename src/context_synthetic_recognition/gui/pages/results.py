"""Results explorer: the steps of the workbook as a tree of tables and figures.

The tree follows the pipeline — the stage names are the workbook's. Tables are the exporters'
tables (built when first opened), coloured by what their cells mean; figures are the exporters'
figures on an interactive canvas. Selecting an object — in any table or in the selector — marks
it in every table and redraws its neighbourhood.
"""

from __future__ import annotations

from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from context_synthetic_recognition.export import figures as fig
from context_synthetic_recognition.export import tables as tab
from context_synthetic_recognition.export.figures import FigureSpec, figure_specs
from context_synthetic_recognition.export.tables import Table, TableSpec, slug, table_specs
from context_synthetic_recognition.export.text import DEFAULT_DECIMALS
from context_synthetic_recognition.export.view import RunView
from context_synthetic_recognition.gui.canvas import Canvas
from context_synthetic_recognition.gui.i18n import mark, tr
from context_synthetic_recognition.gui.models import style_for
from context_synthetic_recognition.gui.pages.base import Page
from context_synthetic_recognition.gui.settings import Settings
from context_synthetic_recognition.gui.state import AppState
from context_synthetic_recognition.gui.theme import Palette
from context_synthetic_recognition.gui.widgets import FigurePanel, TablePanel, WarningBox, hint

KIND_ROLE = Qt.ItemDataRole.UserRole
KEY_ROLE = Qt.ItemDataRole.UserRole + 1
TABLE, FIGURE, CONTEXT = "table", "figure", "context"

TABLE_NAMES = {
    "summary": "Key results",
    "dataset": "Dataset",
    "features": "Features",
    "normalized": "Normalized data",
    "same-class-counts": "Same-class counts μ",
    "chi1": "χ₁ counts",
    "psi": "Ψ(r)",
    "membership": "Membership f, stability g",
    "bit-masks": "Bit masks (Task 2)",
    "synthetic-features": "g, G, ω, η per feature",
    "object-membership": "g(S, k) per object",
    "correct-side": "Correct side of G",
    "contributions": "Contributions η",
    "hag-iterations": "Iterations",
    "hag-candidates": "Candidates: θ, γ, θ/γ",
    "hag-chosen-blocks": "Blocks of the chosen q",
    "meta-dataset": "Meta-dataset Y",
    "training-description": "Training description",
    "new-object": "New object: values",
    "new-object-features": "New object: Ψ(r)",
    "resubstitution": "All training objects",
    "new-object-steps": "New object: B1 / B2 steps",
    "metrics": "Metrics of every method",
    "class-metrics": "Precision, recall, F1",
    "margins": "Margins",
    "object-margins": "Object margins",
    "sensitivity": "⚠ Switch sensitivity",
    "properties": "Model properties",
    "operator-pairs": "Operator pairs (Definition 6)",
    "boundary-ties": "Ties at the k boundary",
}
"""Short names of the tables in the tree."""
PER_ITEM = {
    "distances": "Distances",
    "neighbours": "Sorted neighbours",
    "confusion": "Confusion",
    "roc": "ROC points",
    "predictions": "Predictions",
    "folds": "Folds",
}
FIGURE_NAMES = {
    "distances": "Distance heat maps",
    "neighbourhoods": "Neighbourhoods by rank",
    "synthetic-features": "ω and stability g",
    "hag-candidates": "θ/γ per candidate",
    "hag-criterion": "Criterion per iteration",
    "margins": "Margins of the latent features",
    "margin-widths": "Margin widths",
    "roc": "ROC curves",
    "confusion": "Confusion matrices",
    "sensitivity": "⚠ Switch sensitivity",
}
FIGURE_GROUPS = {
    "distances": tab.GROUP_DISTANCES,
    "neighbourhoods": tab.GROUP_NEIGHBOURS,
    "synthetic-features": tab.GROUP_OMEGA,
    "hag-candidates": tab.GROUP_HAG,
    "hag-criterion": tab.GROUP_HAG,
}
"""Stage of every figure; the others belong to the evaluation."""
FIGURE_MARK = "▣ "


def table_name(key: str, view: RunView) -> str:
    """The name of a table in the tree."""
    if key in TABLE_NAMES:
        return TABLE_NAMES[key]
    prefix, _, rest = key.partition("-")
    if prefix in PER_ITEM:
        label = next((o.label for o in view.trace.operators if slug(o.label) == rest), rest)
        return f"{PER_ITEM[prefix]} · {label}"
    return key


def figure_name(key: str) -> str:
    """The name of a figure in the tree."""
    if key in FIGURE_NAMES:
        return FIGURE_NAMES[key]
    for prefix, name in (("outcomes-", "Outcomes"), ("prediction-map-", "Prediction map")):
        if key.startswith(prefix):
            return f"{name} · {key[len(prefix) :]}"
    return key


class ContextPanel(QWidget):
    """The nested k-neighbourhoods of the selected object under a chosen operator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """An empty panel."""
        super().__init__(parent)
        self._view: RunView | None = None
        self._object = -1
        self._colours: Palette | None = None
        self.canvas = Canvas(coordinates=False)
        self.operator = QComboBox()
        self.operator.currentIndexChanged.connect(self.redraw)
        self.message = hint(tr("Select an object — in a table or in the selector above."), "note")
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Operator")))
        row.addWidget(self.operator)
        row.addStretch(1)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addLayout(row)
        self._layout.addWidget(self.message)
        self._layout.addWidget(self.canvas, 1)

    def set_view(self, view: RunView | None, colours: Palette) -> None:
        """Show another run."""
        self._view = view
        self._colours = colours
        self.operator.blockSignals(True)
        self.operator.clear()
        if view is not None:
            self.operator.addItems([o.label for o in view.trace.operators])
        self.operator.blockSignals(False)
        self.redraw()

    def set_object(self, index: int) -> None:
        """Show the neighbourhood of the training object ``index`` (−1 = none)."""
        self._object = index
        self.redraw()

    def set_palette(self, colours: Palette) -> None:
        """Switch the theme."""
        self._colours = colours
        self.redraw()

    def figure(self) -> Figure | None:
        """The figure on the canvas, if an object is selected."""
        return self.canvas.figure()

    def redraw(self, *_: object) -> None:
        """Draw the neighbourhood of the selected object under the chosen operator."""
        view, i, colours = self._view, self._object, self._colours
        o = self.operator.currentIndex()
        ready = view is not None and colours is not None and 0 <= i < view.trace.m and o >= 0
        self.message.setVisible(not ready)
        self.canvas.setVisible(ready)
        if not ready:
            self.canvas.set_figure(None)
            return
        assert view is not None
        assert colours is not None
        found = view.trace.operators[o]
        figure = fig.object_context_figure(
            view,
            o,
            view.trace.object_ids[i],
            found.distances[i],
            found.order[i],
            colours.figures,
            own_class=int(view.trace.class_index[i]),
        )
        self.canvas.set_figure(figure)


class ResultsPage(Page):
    """Explore every table and figure of a run."""

    title = mark("Results")
    needs_result = True

    def __init__(self, state: AppState, settings: Settings, parent: QWidget | None = None) -> None:
        """Build the page."""
        super().__init__(
            state,
            settings,
            tr(
                "Every step of the pipeline, as in the workbook. Click a column to sort; click a "
                "row to follow its object through every table."
            ),
            parent,
        )
        self._specs: dict[str, TableSpec] = {}
        self._tables: dict[str, Table] = {}
        self._figures: dict[str, FigureSpec] = {}
        self._leaves: list[QTreeWidgetItem] = []
        self._by_group: dict[str, list[QTreeWidgetItem]] = {}

        self.objects = QComboBox()
        self.objects.setMinimumWidth(110)
        self.objects.setToolTip(tr("The object marked in every table (cross-highlighting)"))
        self.objects.activated.connect(self._object_chosen)
        self.decimals = QSpinBox()
        self.decimals.setRange(0, 15)
        self.decimals.setValue(DEFAULT_DECIMALS)
        self.decimals.setToolTip(tr("Decimals shown; the values themselves are not rounded"))
        self.previous = QPushButton(tr("◀ Previous"))
        self.next = QPushButton(tr("Next ▶"))
        self.previous.setToolTip(tr("The previous table or figure (Alt+Left)"))
        self.next.setToolTip(tr("The next table or figure (Alt+Right)"))
        self.previous.clicked.connect(lambda: self.step(-1))
        self.next.clicked.connect(lambda: self.step(1))
        self.header.addWidget(QLabel(tr("Object")))
        self.header.addWidget(self.objects)
        self.header.addWidget(QLabel(tr("Decimals")))
        self.header.addWidget(self.decimals)
        self.header.addWidget(self.previous)
        self.header.addWidget(self.next)

        self.stale = WarningBox()
        self.body.addWidget(self.stale)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(240)
        self.tree.currentItemChanged.connect(self._item_changed)
        self.table_panel = TablePanel()
        self.figure_panel = FigurePanel()
        self.context_panel = ContextPanel()
        self.empty = hint(tr("Run the experiment to see its results."))
        self.stack = QStackedWidget()
        for widget in (self.empty, self.table_panel, self.figure_panel, self.context_panel):
            self.stack.addWidget(widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.tree)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 900])
        self.body.addWidget(splitter, 1)

        self.table_panel.objectClicked.connect(self._object_clicked)
        self.decimals.valueChanged.connect(self.table_panel.set_decimals)
        state.resultChanged.connect(self.rebuild)
        state.configChanged.connect(self._refresh_stale)
        state.objectSelected.connect(self._object_selected)
        self.apply_palette(self.colours)
        self.rebuild()

    # ------------------------------------------------------------------ the tree

    def rebuild(self) -> None:
        """Show the state's run (or nothing)."""
        view = self.state.view
        current = self.current_key()
        self.tree.blockSignals(True)
        self.tree.clear()
        self._leaves = []
        self._by_group = {}
        self._tables = {}
        self._specs = {} if view is None else {spec.key: spec for spec in table_specs(view)}
        self._figures = {} if view is None else {spec.key: spec for spec in figure_specs(view)}
        self.objects.clear()
        self.context_panel.set_view(view, self.colours)
        if view is not None:
            self.objects.addItem("—", -1)
            for index, name in enumerate(view.trace.object_ids):
                self.objects.addItem(name, index)
            groups: dict[str, QTreeWidgetItem] = {}
            for spec in self._specs.values():
                self._leaf(groups, spec.group, table_name(spec.key, view), TABLE, spec.key)
                if spec.key == f"neighbours-{slug(view.trace.operators[-1].label)}":
                    self._leaf(
                        groups,
                        spec.group,
                        FIGURE_MARK + tr("Neighbourhood of the selected object"),
                        CONTEXT,
                        CONTEXT,
                    )
            for key in self._figures:
                group = FIGURE_GROUPS.get(key, tab.GROUP_EVALUATION)
                self._leaf(groups, group, FIGURE_MARK + figure_name(key), FIGURE, key)
            # the order of stepping is the order on screen: stage by stage, tables then figures
            self._leaves = [item for items in self._by_group.values() for item in items]
            self.tree.expandAll()
        self.tree.blockSignals(False)
        for widget in (self.objects, self.decimals, self.previous, self.next, self.tree):
            widget.setEnabled(view is not None)
        self._refresh_stale()
        if view is None:
            self.stack.setCurrentWidget(self.empty)
            return
        self._object_selected(self.state.selected_object)
        if not (current and self.open(*current)):
            self.open(TABLE, "summary")

    def _leaf(
        self, groups: dict[str, QTreeWidgetItem], group: str, name: str, kind: str, key: str
    ) -> None:
        parent = groups.get(group)
        if parent is None:
            parent = QTreeWidgetItem(self.tree, [group])
            parent.setToolTip(0, group)
            parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            font = parent.font(0)
            font.setBold(True)
            parent.setFont(0, font)
            groups[group] = parent
        item = QTreeWidgetItem(parent, [name])
        item.setToolTip(0, name)
        item.setData(0, KIND_ROLE, kind)
        item.setData(0, KEY_ROLE, key)
        self._by_group.setdefault(group, []).append(item)

    def keys(self, kind: str) -> list[str]:
        """The keys of the tables or figures in the tree, in order."""
        return [
            str(item.data(0, KEY_ROLE)) for item in self._leaves if item.data(0, KIND_ROLE) == kind
        ]

    def current_key(self) -> tuple[str, str] | None:
        """(kind, key) of what is shown."""
        item = self.tree.currentItem()
        if item is None or item.data(0, KIND_ROLE) is None:
            return None
        return str(item.data(0, KIND_ROLE)), str(item.data(0, KEY_ROLE))

    def open(self, kind: str, key: str) -> bool:
        """Open a table or a figure by its key; returns whether the run has it."""
        for item in self._leaves:
            if item.data(0, KIND_ROLE) == kind and item.data(0, KEY_ROLE) == key:
                self.tree.setCurrentItem(item)
                return True
        return False

    def step(self, direction: int) -> None:
        """Go to the next (+1) or the previous (−1) table or figure."""
        if not self._leaves:
            return
        item = self.tree.currentItem()
        index = self._leaves.index(item) if item in self._leaves else -1
        index = min(max(index + direction, 0), len(self._leaves) - 1)
        self.tree.setCurrentItem(self._leaves[index])

    # ------------------------------------------------------------------ content

    def table(self, key: str) -> Table:
        """A table of the run (built on first use)."""
        if key not in self._tables:
            self._tables[key] = self._specs[key].build()
        return self._tables[key]

    def _item_changed(self, item: QTreeWidgetItem | None, _: object = None) -> None:
        view = self.state.view
        if item is None or view is None:
            return
        kind, key = item.data(0, KIND_ROLE), item.data(0, KEY_ROLE)
        if kind == TABLE:
            table = self.table(str(key))
            self.table_panel.set_table(table, style_for(table, view))
            self.table_panel.highlight(self._selected_name())
            self.stack.setCurrentWidget(self.table_panel)
        elif kind == FIGURE:
            self.figure_panel.set_spec(self._figures[str(key)])
            self.stack.setCurrentWidget(self.figure_panel)
        elif kind == CONTEXT:
            self.stack.setCurrentWidget(self.context_panel)
        index = self._leaves.index(item) if item in self._leaves else -1
        self.previous.setEnabled(index > 0)
        self.next.setEnabled(0 <= index < len(self._leaves) - 1)

    def _refresh_stale(self) -> None:
        self.stale.set_text(
            tr(
                "⚠ These results were computed with another configuration than the one on the "
                "Configure page. Run again to update them."
            )
            if self.state.stale
            else ""
        )

    # ------------------------------------------------------------------ the selected object

    def _selected_name(self) -> str | None:
        view, index = self.state.view, self.state.selected_object
        if view is None or not 0 <= index < view.trace.m:
            return None
        return view.trace.object_ids[index]

    def _object_chosen(self, row: int) -> None:
        self.state.select_object(int(self.objects.itemData(row)))

    def _object_clicked(self, name: object) -> None:
        view = self.state.view
        if view is not None and name in view.trace.object_ids:
            self.state.select_object(view.trace.object_ids.index(str(name)))

    def _object_selected(self, index: int) -> None:
        row = self.objects.findData(index)
        self.objects.setCurrentIndex(max(row, 0))
        self.table_panel.highlight(self._selected_name())
        self.context_panel.set_object(index)

    def apply_palette(self, colours: Palette) -> None:
        """Re-colour the table and redraw the figures."""
        self.table_panel.set_palette(colours)
        self.figure_panel.set_palette(colours)
        self.context_panel.set_palette(colours)
