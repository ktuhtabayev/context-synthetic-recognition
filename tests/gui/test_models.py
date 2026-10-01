"""Tables on screen: colour semantics, the Qt model, sorting and filtering."""

from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush

from context_synthetic_recognition.data import Dataset
from context_synthetic_recognition.export import RunView
from context_synthetic_recognition.export.tables import Table, TableSet, run_tables
from context_synthetic_recognition.gui.models import (
    RAW_ROLE,
    SORT_ROLE,
    SortProxy,
    TableModel,
    TableStyle,
    dataset_table,
    feature_rows,
    header_tooltip,
    labels_text,
    sort_key,
    style_for,
)
from context_synthetic_recognition.gui.theme import DARK, LIGHT, CellRole

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def tables(experiment_view: RunView) -> TableSet:
    return run_tables(experiment_view)


def roles(table: Table, view: RunView) -> list[list[CellRole | None]]:
    style = style_for(table, view)
    assert style.cell is not None
    return [[style.cell(r, c, v) for c, v in enumerate(row)] for r, row in enumerate(table.rows)]


def colour(brush: Any) -> str:
    assert isinstance(brush, QBrush)
    return brush.color().name()


# ---------------------------------------------------------------- keys and tooltips


def test_sort_keys() -> None:
    names = ["S₁₀", "S₂", "S₁", "a₁₂", "a₃"]
    assert sorted(names, key=sort_key) == ["a₃", "a₁₂", "S₁", "S₂", "S₁₀"]
    values = [3, None, "text", 1.5, -2, "Alpha"]
    assert sorted(values, key=sort_key) == [-2, 1.5, 3, "Alpha", "text", None]
    assert sort_key(2) == sort_key(2.0)
    assert sort_key("k = 10") > sort_key("k = 9")


def test_header_tooltips(tables: TableSet) -> None:
    features = tables["synthetic-features"]
    assert header_tooltip("ω (4)", features).splitlines()[:2] == [
        "ω (4)",
        "Informativeness ω — formula (4)",
    ]
    assert "formula (2)" in header_tooltip("g_k (2)", features)
    assert "formula (3)" in header_tooltip("G_k (3)", features)
    candidates = tables["hag-candidates"]
    assert "HAG STEP 3" in header_tooltip("θ/γ", candidates)
    assert header_tooltip("θ/γ", candidates).endswith(candidates.note)
    bare = Table("t", "T", ("a", "b"), ((1, 2),), "g")
    assert header_tooltip("plain", bare) == "plain"
    assert labels_text((3, 5)) == "3, 5"


# ---------------------------------------------------------------- colour semantics


def test_feature_columns_show_their_type(experiment: Dataset) -> None:
    table = dataset_table(experiment)
    style = style_for(table, dataset=experiment)
    assert style.header_roles[0] is None  # №
    assert style.header_roles[1] is CellRole.QUANTITATIVE  # x₁
    assert style.header_roles[2] is CellRole.NOMINAL  # x₂
    assert style.header_roles[-1] is None  # Class
    assert style.object_column == 0
    assert style.cell is not None
    assert style.cell(0, len(table.columns) - 1, table.rows[0][-1]) is CellRole.K2  # S₁ ∈ K2
    assert style.cell(1, len(table.columns) - 1, table.rows[1][-1]) is CellRole.K1
    assert style.cell(0, 1, table.rows[0][1]) is None


def test_synthetic_features_and_tuplam(experiment_view: RunView, tables: TableSet) -> None:
    psi = tables["psi"]
    style = style_for(psi, experiment_view)
    # TUPLAM = {a₆, a₃, a₁, a₂, a₄}: a₅ is the one synthetic feature that was not selected
    assert style.header_roles[1:7] == (
        CellRole.SELECTED,
        CellRole.SELECTED,
        CellRole.SELECTED,
        CellRole.SELECTED,
        CellRole.SYNTHETIC,
        CellRole.SELECTED,
    )
    found = roles(psi, experiment_view)
    # aᵤ = 1 votes for K1, aᵤ = 2 for K2; the class column is coloured by the class
    assert found[0] == [None, *[CellRole.K2] * 2, CellRole.K1, *[CellRole.K2] * 3, CellRole.K2]
    assert found[1][-1] is CellRole.K1
    by_feature = tables["synthetic-features"]
    names = [row[0] for row in by_feature.rows]
    marked = [found_row[0] for found_row in roles(by_feature, experiment_view)]
    assert [n for n, m in zip(names, marked, strict=True) if m is CellRole.SELECTED] == [
        "a₁",
        "a₂",
        "a₃",
        "a₄",
        "a₆",
    ]


def test_latent_features_and_the_chosen_candidate(
    experiment_view: RunView, tables: TableSet
) -> None:
    meta = tables["meta-dataset"]
    style = style_for(meta, experiment_view)
    assert style.header_roles[1] is CellRole.SELECTED  # y₀ (a₆)
    assert style.header_roles[6] is CellRole.LATENT  # r₁
    margins = roles(tables["margins"], experiment_view)
    assert margins[0][0] is CellRole.LATENT  # the row of r₁
    assert margins[0][1] is None
    candidates = tables["hag-candidates"]
    chosen = candidates.columns.index("Chosen q")
    for row, found in zip(candidates.rows, roles(candidates, experiment_view), strict=True):
        expected = CellRole.SELECTED if row[chosen] == 1 else None
        assert set(found) == {expected}, row
    assert sum(row[chosen] == 1 for row in candidates.rows) == 4  # one q per iteration


def test_decisions_are_coloured_by_being_correct(
    experiment_view: RunView, tables: TableSet
) -> None:
    predictions = tables["predictions-leave-one-out"]
    found = roles(predictions, experiment_view)
    columns = predictions.columns
    truth, own = columns.index("True class"), columns.index("CS-model")
    assert [row[truth] for row in found[:2]] == [CellRole.K2, CellRole.K1]
    # S₁ (class 2) is predicted 1: wrong; S₂ is refused; S₆ (class 1) is predicted 1: correct
    assert [found[i][own] for i in (0, 1, 5)] == [CellRole.BAD, CellRole.REFUSAL, CellRole.GOOD]
    assert found[0][columns.index("score₁ − score₂")] is None
    assert found[0][columns.index("k-NN vote ρ, k = 3")] is CellRole.GOOD
    resubstitution = tables["resubstitution"]
    predicted = resubstitution.columns.index("Predicted")
    assert [r[predicted] for r in roles(resubstitution, experiment_view)[:3]] == [
        CellRole.BAD,
        CellRole.BAD,
        CellRole.GOOD,
    ]


def test_status_flags_and_undefined_values(experiment_view: RunView, tables: TableSet) -> None:
    properties = tables["properties"]
    statuses = {
        row[3]: found[3]
        for row, found in zip(properties.rows, roles(properties, experiment_view), strict=True)
    }
    assert statuses["✓ defined"] is CellRole.GOOD
    assert statuses["✗ insufficient: 10 conflicting pair(s)"] is CellRole.BAD
    side = tables["correct-side"]
    found = roles(side, experiment_view)
    assert found[0][1:7] == [
        CellRole.GOOD,
        CellRole.GOOD,
        CellRole.BAD,
        CellRole.GOOD,
        CellRole.GOOD,
        CellRole.GOOD,
    ]
    features = tables["features"]
    assert roles(features, experiment_view)[1][3] is CellRole.MUTED  # min of a nominal feature
    warned = Table("w", "W", ("a",), (("⚠ careful",), ("plain",)), "g")
    assert roles(warned, experiment_view) == [[CellRole.WARNING], [None]]
    assert TableStyle.plain(warned) == TableStyle((None,))


def test_the_object_column(experiment_view: RunView, tables: TableSet) -> None:
    assert style_for(tables["psi"], experiment_view).object_column == 0
    assert style_for(tables["neighbours-rho"], experiment_view).object_column == 0
    assert style_for(tables["summary"], experiment_view).object_column is None
    assert style_for(tables["hag-candidates"], experiment_view).object_column is None
    empty = Table("e", "E", ("№",), (), "g")
    assert style_for(empty, experiment_view).object_column is None
    assert style_for(empty).header_roles == (None,)


# ---------------------------------------------------------------- the Qt model


def test_the_table_model(experiment_view: RunView, tables: TableSet) -> None:
    table = tables["margins"]
    model = TableModel(table, style_for(table, experiment_view))
    assert (model.rowCount(), model.columnCount()) == (len(table.rows), len(table.columns))
    assert model.rowCount(model.index(0, 0)) == 0  # a flat table: cells have no children
    assert model.columnCount(model.index(0, 0)) == 0
    width = table.columns.index("Margin width")
    index = model.index(0, width)
    assert index.data() == "0.1482"
    assert index.data(RAW_ROLE) == table.rows[0][width]
    assert index.data(Qt.ItemDataRole.ToolTipRole) == repr(table.rows[0][width])
    assert index.data(SORT_ROLE) == sort_key(table.rows[0][width])
    right = int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    assert index.data(Qt.ItemDataRole.TextAlignmentRole) == right
    label = model.index(0, 0)
    assert label.data() == "r₁"
    assert label.data(Qt.ItemDataRole.FontRole).bold()
    assert label.data(Qt.ItemDataRole.ToolTipRole) is None
    assert index.data(Qt.ItemDataRole.FontRole) is None
    assert model.data(model.index(-1, -1)) is None
    assert index.data(Qt.ItemDataRole.DecorationRole) is None
    model.set_decimals(2)
    assert index.data() == "0.15"
    horizontal = Qt.Orientation.Horizontal
    assert model.headerData(width, horizontal) == "Margin width"
    assert "Margin width" in model.headerData(width, horizontal, Qt.ItemDataRole.ToolTipRole)
    assert model.headerData(0, Qt.Orientation.Vertical) == 1
    assert model.headerData(0, Qt.Orientation.Vertical, Qt.ItemDataRole.ToolTipRole) is None
    assert model.headerData(0, horizontal, Qt.ItemDataRole.DecorationRole) is None
    assert model.header_role(0) is None
    assert model.header_role(width) is None


def test_colours_follow_the_palette(experiment_view: RunView, tables: TableSet) -> None:
    table = tables["psi"]
    model = TableModel(table, style_for(table, experiment_view))
    cell = model.index(0, 1)  # a₁(S₁) = 2: votes for K2
    background, text = Qt.ItemDataRole.BackgroundRole, Qt.ItemDataRole.ForegroundRole
    assert colour(cell.data(background)) == LIGHT.cells[CellRole.K2][0]
    assert colour(cell.data(text)) == LIGHT.cells[CellRole.K2][1]
    assert model.index(0, 0).data(background) is None
    model.set_palette(DARK)
    assert colour(cell.data(background)) == DARK.cells[CellRole.K2][0]


def test_cross_highlighting(experiment_view: RunView, tables: TableSet, qtbot: Any) -> None:
    table = tables["neighbours-rho"]
    model = TableModel(table, style_for(table, experiment_view))
    assert model.object_of(0) == "S₁"
    assert model.rows_of("S₃") == list(range(12, 18))  # six ranks per object
    assert model.rows_of(None) == []
    background = Qt.ItemDataRole.BackgroundRole
    assert model.index(12, 0).data(background) is None
    with qtbot.waitSignal(model.dataChanged):
        model.set_highlight("S₃")
    assert colour(model.index(12, 0).data(background)) == LIGHT.highlight
    assert colour(model.index(12, 0).data(Qt.ItemDataRole.ForegroundRole)) == LIGHT.text
    assert model.index(0, 0).data(background) is None
    with qtbot.assertNotEmitted(model.dataChanged):
        model.set_highlight("S₃")  # nothing changes
    summary = TableModel(tables["summary"])
    assert summary.object_of(0) is None
    assert summary.rows_of("S₁") == []
    TableModel(Table("e", "E", ("a",), (), "g")).set_highlight("S₁")  # an empty table is fine


def test_sorting_and_filtering(experiment_view: RunView, tables: TableSet) -> None:
    table = tables["psi"]
    model = TableModel(table, style_for(table, experiment_view))
    proxy = SortProxy()
    proxy.setSourceModel(model)
    first = [proxy.index(r, 0).data() for r in range(proxy.rowCount())]
    assert first == [f"S{s}" for s in ("₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉", "₁₀")]
    proxy.sort(0, Qt.SortOrder.DescendingOrder)
    assert proxy.index(0, 0).data() == "S₁₀"  # natural order: S₁₀ after S₉, not after S₁
    assert proxy.index(1, 0).data() == "S₉"
    proxy.sort(1, Qt.SortOrder.AscendingOrder)  # a₁: the 1s first
    assert proxy.index(0, 1).data() == "1"
    assert proxy.index(proxy.rowCount() - 1, 1).data() == "2"
    proxy.setFilterFixedString("s₁₀")  # any column, case-blind
    assert [proxy.index(r, 0).data() for r in range(proxy.rowCount())] == ["S₁₀"]
    proxy.setFilterFixedString("no such text")
    assert proxy.rowCount() == 0


# ---------------------------------------------------------------- dataset tables


def test_the_dataset_preview(experiment: Dataset) -> None:
    table = dataset_table(experiment)
    assert table.columns[:3] == ("№", "x₁", "x₂")
    assert table.rows[0][:4] == ("S₁", 70, 1, 4)
    assert table.note == ""
    tenth = table.columns.index("x₁₀")  # a column with fractions is shown as real numbers
    assert [type(row[tenth]) for row in table.rows] == [float] * experiment.m
    assert table.rows[9][tenth] == 4.0
    assert {type(row[1]) for row in table.rows} == {int}
    short = dataset_table(experiment, max_rows=3)
    assert len(short.rows) == 3
    assert short.note == "The first 3 of 10 objects."


def test_the_feature_rows(experiment: Dataset) -> None:
    rows = feature_rows(experiment)
    assert rows[0] == ("x₁", "quantitative", 56.0, 74.0, 10)
    assert rows[1] == ("x₂", "nominal", None, None, 2)
    assert len(rows) == experiment.n
