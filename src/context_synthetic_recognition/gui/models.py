"""Tables on screen: a Qt model over the exporters' tables, with the workbook's colour semantics.

The pages show the same :class:`~context_synthetic_recognition.export.tables.Table` objects the
CSV, LaTeX and report exporters write. :func:`style_for` decides what every column and cell
*means* (:class:`~.theme.CellRole`): a quantitative or nominal feature, a synthetic or latent
feature, a feature of TUPLAM, a class, a correct or wrong decision, a warning. The palette turns
the meaning into colours, so both themes show the same semantics.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    QSortFilterProxyModel,
    Qt,
)
from PySide6.QtGui import QBrush, QColor, QFont

from context_synthetic_recognition.data import Dataset
from context_synthetic_recognition.export.tables import Cell, Table
from context_synthetic_recognition.export.text import DEFAULT_DECIMALS, format_cell, is_numeric
from context_synthetic_recognition.export.view import RunView
from context_synthetic_recognition.gui.theme import LIGHT, CellRole, Palette

SORT_ROLE = Qt.ItemDataRole.UserRole + 1
"""Model role holding a key that sorts numbers numerically and S₁₀ after S₉."""
RAW_ROLE = Qt.ItemDataRole.UserRole + 2
"""Model role holding the cell's value as it is in the table."""

Index = QModelIndex | QPersistentModelIndex
CellRule = Callable[[int, int, Cell], CellRole | None]
"""(row, column, value) → what the cell means."""

_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SYNTHETIC = re.compile(r"a[₀-₉]+")
_LATENT = re.compile(r"r[₀-₉]+")
_CHUNKS = re.compile(r"(\d+)")

CLASS_COLUMNS = frozenset({"Class", "True class", "True"})
"""Columns holding the class of an object."""
PREDICTION_COLUMNS = frozenset({"Predicted", "CS-model"})
"""Columns holding a decision (coloured by whether it is correct)."""
VOTE_TABLES = frozenset({"psi", "training-description"})
"""Tables whose feature columns hold aᵤ ∈ {1, 2}: the class the neighbourhood votes for."""

FORMULAS: tuple[tuple[str, str], ...] = (
    ("f_k", "Membership f_k(μ) — formula (1)"),
    ("g_k", "Stability g_k — formula (2)"),
    ("G_k", "Boundary G_k = (q₁ + q₂)/2 — formula (3)"),
    ("ω", "Informativeness ω — formula (4)"),
    ("aᵤ", "Synthetic feature aᵤ ∈ {1, 2} — formula (5)"),
    ("η", "Contribution η — formula (6)"),
    ("χ₁", "χ₁: K1 objects among the k nearest neighbours (class of S not used)"),
    ("μ", "μ: neighbours of the object's own class among the k nearest"),
    ("θ/γ", "θ/γ: within-class over between-class distance — HAG STEP 3"),
    ("θ", "θ = Σ |bₜ − M(own class)| — HAG STEP 3"),
    ("γ", "γ = Σ |bₜ − M(other class)| — HAG STEP 3"),
    ("crit", "crit = min θ/γ — HAG STEP 3; the grouping stops when crit ≤ δ"),
    ("score", "score₁ = |B1(a_p)|/|K1|, score₂ = |B2(a_p)|/|K2| — meta-algorithm Step 4"),
    ("B1", "B1(aⱼ): K1 objects with the same gradations a₀ … aⱼ — meta-algorithm Steps 1–3"),
    ("B2", "B2(aⱼ): K2 objects with the same gradations a₀ … aⱼ — meta-algorithm Steps 1–3"),
    ("AUC", "AUC — Mann–Whitney statistic of score₁ − score₂ (ties ½)"),
)
"""Header fragment → what it is, with the article's formula number (first match wins)."""


def sort_key(value: Cell) -> tuple[int, Any]:
    """A key that orders numbers numerically, text naturally (S₂ < S₁₀) and undefined last."""
    if value is None:
        return (2, ())
    if isinstance(value, int | float):
        return (0, float(value))
    text = str(value).translate(_SUBSCRIPTS)
    chunks = tuple(
        (0, float(chunk), "") if chunk.isdigit() else (1, 0.0, chunk.casefold())
        for chunk in _CHUNKS.split(text)
        if chunk
    )
    return (1, chunks)


def header_tooltip(name: str, table: Table) -> str:
    """What a column holds: the formula it comes from, then the table's note."""
    lines = [name]
    lines += [text for fragment, text in FORMULAS if fragment in name][:1]
    if table.note:
        lines.append(table.note)
    return "\n".join(lines)


@dataclass(frozen=True)
class TableStyle:
    """The meaning of a table's columns and cells."""

    header_roles: tuple[CellRole | None, ...]
    cell: CellRule | None = None
    object_column: int | None = None
    """The column that names the object of a row (S₁ …), for cross-highlighting."""

    @classmethod
    def plain(cls, table: Table) -> TableStyle:
        """No semantics: every cell in the default colours."""
        return cls((None,) * len(table.columns))


def _status(value: Cell) -> CellRole | None:
    if value is None:
        return CellRole.MUTED
    if isinstance(value, str):
        if value.startswith("✓"):
            return CellRole.GOOD
        if value.startswith("✗"):
            return CellRole.BAD
        if value.startswith("⚠"):
            return CellRole.WARNING
    return None


def _feature_types(dataset: Dataset | None) -> dict[str, CellRole]:
    if dataset is None:
        return {}
    return {
        name: CellRole.QUANTITATIVE if quantitative else CellRole.NOMINAL
        for name, quantitative in zip(dataset.feature_names, dataset.quantitative, strict=True)
    }


def _header_role(name: str, features: dict[str, CellRole], tuplam: set[str]) -> CellRole | None:
    if name in features:
        return features[name]
    if _LATENT.search(name):
        return CellRole.LATENT
    found = _SYNTHETIC.search(name)
    if found:
        return CellRole.SELECTED if found.group() in tuplam else CellRole.SYNTHETIC
    return None


def _object_column(table: Table, objects: set[str]) -> int | None:
    if not table.rows or not objects:
        return None
    for column in range(min(table.labels, len(table.columns))):
        if all(row[column] in objects for row in table.rows):
            return column
    return None


def style_for(
    table: Table, view: RunView | None = None, dataset: Dataset | None = None
) -> TableStyle:
    """Decide the colour semantics of a table.

    Args:
        table: The table.
        view: The run it belongs to (classes, TUPLAM, object names), if there is one.
        dataset: The dataset (feature types) when there is no run yet.
    """
    data = view.dataset if view is not None else dataset
    features = _feature_types(data)
    classes: tuple[Any, ...] = ()
    tuplam: set[str] = set()
    objects: set[str] = set(data.object_ids) if data is not None else set()
    if view is not None:
        classes = tuple(view.trace.classes)
        tuplam = set(view.hag.names)
    elif data is not None and len(data.classes) == 2:
        classes = tuple(data.classes)
    columns = table.columns
    headers = tuple(_header_role(name, features, tuplam) for name in columns)

    truth = next((c for c, name in enumerate(columns) if name in CLASS_COLUMNS), None)
    class_columns = {c for c, name in enumerate(columns) if name in CLASS_COLUMNS}
    predictions = {c for c, name in enumerate(columns) if name in PREDICTION_COLUMNS}
    if table.key.startswith("predictions-") and truth is not None:
        predictions |= {c for c in range(truth + 1, len(columns)) if "score" not in columns[c]}
    votes: set[int] = set()
    if table.key in VOTE_TABLES:
        votes = {
            c for c, role in enumerate(headers) if role in (CellRole.SYNTHETIC, CellRole.SELECTED)
        }
    elif table.key == "new-object-features" and "aᵤ(S)" in columns:
        votes = {columns.index("aᵤ(S)")}
    flags = (
        {c for c, role in enumerate(headers) if role is not None}
        if table.key == "correct-side"
        else set()
    )
    chosen = columns.index("Chosen q") if "Chosen q" in columns else None

    def class_role(value: Cell) -> CellRole | None:
        if classes and value == classes[0]:
            return CellRole.K1
        if len(classes) > 1 and value == classes[1]:
            return CellRole.K2
        return None

    def decision_role(value: Cell, row: int) -> CellRole | None:
        if value == 0 or (isinstance(value, str) and value.startswith("0")):
            return CellRole.REFUSAL
        if truth is None:
            return class_role(value)
        return CellRole.GOOD if value == table.rows[row][truth] else CellRole.BAD

    def cell(row: int, column: int, value: Cell) -> CellRole | None:
        if chosen is not None and table.rows[row][chosen] == 1:
            return CellRole.SELECTED
        if column in class_columns:
            return class_role(value)
        if column in predictions:
            return decision_role(value, row)
        if column in votes:
            return {1: CellRole.K1, 2: CellRole.K2}.get(value)  # type: ignore[arg-type]
        if column in flags:
            return {1: CellRole.GOOD, 0: CellRole.BAD}.get(value)  # type: ignore[arg-type]
        if chosen is None and column < table.labels and isinstance(value, str):
            # a row that is about a feature: named in the colour of the kind of feature
            # (not among HAG candidates, where only the chosen q is marked)
            if _LATENT.fullmatch(value):
                return CellRole.LATENT
            if _SYNTHETIC.fullmatch(value):
                return CellRole.SELECTED if value in tuplam else CellRole.SYNTHETIC
        return _status(value)

    return TableStyle(headers, cell, _object_column(table, objects))


class TableModel(QAbstractTableModel):
    """A table of a run as a Qt model."""

    def __init__(
        self,
        table: Table,
        style: TableStyle | None = None,
        colours: Palette = LIGHT,
        decimals: int = DEFAULT_DECIMALS,
        parent: QObject | None = None,
    ) -> None:
        """Show ``table`` with the semantics of ``style`` in the colours of ``colours``."""
        super().__init__(parent)
        self.table = table
        self.style = style or TableStyle.plain(table)
        self._colours = colours
        self._decimals = decimals
        self._highlight: Cell = None
        self._numeric = [
            c >= table.labels and is_numeric(table, c) for c in range(len(table.columns))
        ]
        self._bold = QFont()
        self._bold.setBold(True)

    # ------------------------------------------------------------------ settings

    def set_palette(self, colours: Palette) -> None:
        """Switch the theme."""
        self._colours = colours
        self._refresh()

    def set_decimals(self, decimals: int) -> None:
        """Change the number of decimals shown (the values themselves are not rounded)."""
        self._decimals = decimals
        self._refresh()

    def set_highlight(self, name: Cell) -> None:
        """Highlight the rows of the object called ``name`` (``None`` clears it)."""
        if name != self._highlight:
            self._highlight = name
            self._refresh()

    def _refresh(self) -> None:
        if self.table.rows:
            last = self.index(self.rowCount() - 1, self.columnCount() - 1)
            self.dataChanged.emit(self.index(0, 0), last)
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, self.columnCount() - 1)

    def object_of(self, row: int) -> Cell:
        """The object named in a row, or ``None`` if the table has no object column."""
        column = self.style.object_column
        return None if column is None else self.table.rows[row][column]

    def rows_of(self, name: Cell) -> list[int]:
        """The rows that name the object ``name``."""
        column = self.style.object_column
        if column is None or name is None:
            return []
        return [r for r, row in enumerate(self.table.rows) if row[column] == name]

    def header_role(self, column: int) -> CellRole | None:
        """What a column means."""
        return self.style.header_roles[column]

    # ------------------------------------------------------------------ Qt model

    def rowCount(self, parent: Index = QModelIndex()) -> int:  # noqa: B008
        """Number of rows."""
        return 0 if parent.isValid() else len(self.table.rows)

    def columnCount(self, parent: Index = QModelIndex()) -> int:  # noqa: B008
        """Number of columns."""
        return 0 if parent.isValid() else len(self.table.columns)

    def data(self, index: Index, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """A cell's text, colours, alignment, tooltip or sort key."""
        if not index.isValid():
            return None
        row, column = index.row(), index.column()
        value = self.table.rows[row][column]
        if role == Qt.ItemDataRole.DisplayRole:
            return format_cell(value, self._decimals)
        if role == SORT_ROLE:
            return sort_key(value)
        if role == RAW_ROLE:
            return value
        if role == Qt.ItemDataRole.TextAlignmentRole:
            horizontal = (
                Qt.AlignmentFlag.AlignRight if self._numeric[column] else Qt.AlignmentFlag.AlignLeft
            )
            return int(horizontal | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole:
            return repr(value) if isinstance(value, float) else None
        if role == Qt.ItemDataRole.FontRole:
            return self._bold if column < self.table.labels else None
        if role in (Qt.ItemDataRole.BackgroundRole, Qt.ItemDataRole.ForegroundRole):
            return self._brush(row, column, value, role == Qt.ItemDataRole.BackgroundRole)
        return None

    def _brush(self, row: int, column: int, value: Cell, background: bool) -> QBrush | None:
        meaning = self.style.cell(row, column, value) if self.style.cell else None
        if meaning is not None:
            return QBrush(QColor(self._colours.cells[meaning][0 if background else 1]))
        if self._highlight is not None and self.object_of(row) == self._highlight:
            return QBrush(QColor(self._colours.highlight if background else self._colours.text))
        return None

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        """A column's name and tooltip; rows are numbered."""
        if orientation == Qt.Orientation.Vertical:
            return section + 1 if role == Qt.ItemDataRole.DisplayRole else None
        if role == Qt.ItemDataRole.DisplayRole:
            return self.table.columns[section]
        if role == Qt.ItemDataRole.ToolTipRole:
            return header_tooltip(self.table.columns[section], self.table)
        return None


class SortProxy(QSortFilterProxyModel):
    """Sorting by :data:`SORT_ROLE` and filtering on any column, case-blind."""

    def __init__(self, parent: QObject | None = None) -> None:
        """Filter every column."""
        super().__init__(parent)
        self.setFilterKeyColumn(-1)
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

    def lessThan(self, left: Index, right: Index) -> bool:
        """Compare two cells by their sort keys."""
        source = self.sourceModel()
        a, b = source.data(left, SORT_ROLE), source.data(right, SORT_ROLE)
        return bool(a < b)


def dataset_table(dataset: Dataset, max_rows: int | None = None) -> Table:
    """The preview grid: objects × features with the class label."""
    rows: list[tuple[Cell, ...]] = []
    shown = dataset.m if max_rows is None else min(dataset.m, max_rows)
    # a column is shown as integers only if every value of it is integral
    integral = [bool(np.all(np.mod(dataset.X[:, j], 1) == 0)) for j in range(dataset.n)]
    for i in range(shown):
        values = [
            int(v) if whole else float(v) for v, whole in zip(dataset.X[i], integral, strict=True)
        ]
        label = dataset.y[i].item()
        rows.append((dataset.object_ids[i], *values, label))
    note = "" if shown == dataset.m else f"The first {shown} of {dataset.m} objects."
    return Table(
        "dataset",
        dataset.name,
        ("№", *dataset.feature_names, "Class"),
        tuple(rows),
        "Dataset",
        note,
    )


def feature_rows(dataset: Dataset) -> list[tuple[str, str, Cell, Cell, int]]:
    """Per feature: name, type, minimum and maximum (quantitative only), distinct values."""
    rows: list[tuple[str, str, Cell, Cell, int]] = []
    for j, (name, kind) in enumerate(
        zip(dataset.feature_names, dataset.feature_types, strict=True)
    ):
        column = dataset.X[:, j]
        quantitative = bool(dataset.quantitative[j])
        low = float(column.min()) if quantitative else None
        high = float(column.max()) if quantitative else None
        rows.append((name, kind.value, low, high, len(set(column.tolist()))))
    return rows


def labels_text(values: Sequence[object]) -> str:
    """``3, 5`` — a short list for a label."""
    return ", ".join(str(v) for v in values)
