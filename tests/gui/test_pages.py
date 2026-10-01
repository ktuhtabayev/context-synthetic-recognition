"""The pages of the workflow, driven as a user drives them."""

import json
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QLineEdit
from pytestqt.qtbot import QtBot

from context_synthetic_recognition.config import plugin, preset
from context_synthetic_recognition.config.models import CentreMode
from context_synthetic_recognition.data.schema import FeatureType
from context_synthetic_recognition.export.report import ExportError
from context_synthetic_recognition.gui.pages.results import (
    CONTEXT,
    FIGURE,
    TABLE,
    figure_name,
    table_name,
)
from context_synthetic_recognition.gui.theme import DARK, LIGHT, CellRole
from context_synthetic_recognition.gui.window import MainWindow
from context_synthetic_recognition.services.runner import save_run

from ..test_export_figures import EXPERIMENT_FIGURES
from ..test_export_tables import EXPERIMENT_TABLES
from .conftest import RUN_TIMEOUT, Run

pytestmark = pytest.mark.gui

CSV = (
    "age,sex,pressure,label\n63,m,145,sick\n41,f,130,well\n57,f,120,well\n"
    "44,m,150,sick\n52,m,172,sick\n"
)


def facts(window: MainWindow) -> dict[str, str]:
    return {key: label.text() for key, label in window.dataset_page._fact_labels.items()}


# ---------------------------------------------------------------- Dataset


def test_the_dataset_page_before_and_after_loading(window: MainWindow) -> None:
    page = window.dataset_page
    assert facts(window)["name"] == "—"
    assert not page.tabs.isEnabled()
    assert page.source.itemText(0) == "Choose a dataset…"
    assert page.load("heart-disease-10")
    state = window.state
    assert state.dataset is not None
    assert state.dataset.m == 10
    assert state.config.dataset.path == "heart-disease-10"
    shown = facts(window)
    assert shown["objects"] == "10"
    assert shown["features"] == "13  (6 quantitative, 7 nominal)"
    assert shown["k1"] == "4 objects of class 1  (40 %)"
    assert shown["k2"] == "6 objects of class 2  (60 %)"
    assert shown["k"] == "3, 5   (formula, 2 value(s))"
    assert shown["r"] == "6"
    assert shown["operators"] == "ρ (13), ρ_I (6), ρ_J (7)"
    assert page.balance.sizes == (4, 6)
    assert page.problems.text() == ""
    assert page.preview.table().rows[0][:3] == ("S₁", 70, 1)  # type: ignore[union-attr]
    assert page.features.rowCount() == 13
    assert page.source.currentData() == "heart-disease-10"
    assert window.windowTitle().startswith("Heart-Disease (10, 13, 2) — ")


def test_the_summary_follows_the_configuration(loaded: MainWindow) -> None:
    state = loaded.state
    state.set_config(state.config.model_copy(update={"k": plugin("explicit", values=[3])}))
    assert facts(loaded)["k"] == "3   (explicit, 1 value(s))"
    assert facts(loaded)["r"] == "3"
    state.set_config(state.config.model_copy(update={"k": plugin("explicit", values=[11])}))
    assert facts(loaded)["k"] == "—"
    assert loaded.dataset_page.problems.text().startswith("⚠ ")
    state.undo_stack.undo()
    state.undo_stack.undo()
    assert facts(loaded)["k"] == "3, 5   (formula, 2 value(s))"


def test_changing_a_feature_type(loaded: MainWindow, run: Run) -> None:
    run(loaded)
    page = loaded.dataset_page
    combo = page.type_combo(1)  # x₂ is nominal
    assert combo.currentData() == "nominal"
    combo.setCurrentIndex(combo.findData("quantitative"))
    combo.activated.emit(combo.currentIndex())
    dataset = loaded.state.dataset
    assert dataset is not None
    assert dataset.feature_types[1] is FeatureType.QUANTITATIVE
    assert loaded.state.config.dataset.feature_types == {"x₂": "quantitative"}
    assert facts(loaded)["features"] == "13  (7 quantitative, 6 nominal)"
    assert loaded.state.view is None  # other data: the result is dropped
    assert page.type_combo(1).currentData() == "quantitative"


def test_opening_a_table_file(window: MainWindow, tmp_path: Path) -> None:
    file = tmp_path / "patients.csv"
    file.write_text(CSV, encoding="utf-8")
    page = window.dataset_page
    page.file_dialog = lambda: str(file)
    page.open_file()
    dataset = window.state.dataset
    assert dataset is not None
    assert (dataset.m, dataset.n) == (5, 3)
    assert dataset.feature_names == ("age", "sex", "pressure")
    assert dataset.classes == ("sick", "well")
    assert page.location.text() == str(file)
    assert window.settings.recent("recent/datasets") == [str(file.resolve())]
    # the reading options: another class column makes "label" a feature
    page.options.setChecked(True)
    page.class_column.setText("sex")
    page.reload()
    dataset = window.state.dataset
    assert dataset is not None
    assert dataset.feature_names == ("age", "pressure", "label")
    assert window.state.config.dataset.class_column == "sex"
    page.class_column.setText("no such column")
    page.reload()
    assert "no class column 'no such column'" in page.error.text()
    assert page.error.isVisibleTo(page)
    assert window.state.dataset is dataset  # a failed read changes nothing
    page.file_dialog = lambda: ""
    page.open_file()  # the dialog was cancelled
    assert window.state.dataset is dataset


def test_dataset_errors_are_shown(window: MainWindow, tmp_path: Path) -> None:
    page = window.dataset_page
    assert not page.load(str(tmp_path / "missing.csv"))
    assert "dataset file not found" in page.error.text()
    assert not page.load("no-such-dataset")
    assert "no-such-dataset" in page.error.text()
    page.reload()  # nothing is loaded: nothing to read again
    assert window.state.dataset is None
    assert page.load("heart-disease-270")
    assert not page.error.isVisibleTo(page)
    assert facts(window)["k"].startswith("3, 5, …, 237")


def test_the_dataset_page_in_the_dark_theme(loaded: MainWindow) -> None:
    loaded.set_theme("dark")
    page = loaded.dataset_page
    assert page.balance.colours == (DARK.figures.k1, DARK.figures.k2)
    item = page.features.item(0, 0)  # x₁ is quantitative
    assert item.background().color().name() == DARK.cells[CellRole.QUANTITATIVE][0]
    assert page.features.item(1, 0).background().color().name() == DARK.cells[CellRole.NOMINAL][0]


# ---------------------------------------------------------------- Configure


def test_the_configure_page_shows_the_template_preset(loaded: MainWindow) -> None:
    page = loaded.configure_page
    assert page.preset_badge.text() == "template"
    text = page.deviations.text()
    assert text.startswith("⚠ Template calculations that differ from the article are active:")
    assert "hag.centres, ADR-002" in text
    assert "hag.step4_passes, ADR-003" in text
    assert page.centres_badge.isVisibleTo(page)
    assert page.passes_badge.isVisibleTo(page)
    assert page.problems.text() == ""
    assert page.k_preview.text() == (
        "Heart-Disease (10, 13, 2): k = 3, 5  →  r = 6 synthetic features (3 operator(s) × 2 k)"
    )
    assert [row.label.text() for row in page.operators.rows] == ["ρ", "ρ_I", "ρ_J"]
    assert (page.alpha.value(), page.delta.value(), page.kappa.value()) == (0.3, 0.1, 5)
    assert page.widgets_config() == loaded.state.config


def test_presets_and_undo(loaded: MainWindow) -> None:
    page, state = loaded.configure_page, loaded.state
    page.article_button.click()
    assert state.config.hag.centres is CentreMode.FINAL
    assert state.config.hag.step4_passes == 1
    assert page.preset_badge.text() == "article"
    assert "follows the article for both switches" in page.deviations.text()
    assert not page.centres_badge.isVisibleTo(page)
    assert state.config.dataset.path == "heart-disease-10"  # the data stay
    assert not loaded.deviation_button.isVisibleTo(loaded)
    loaded.undo_action.trigger()
    assert page.preset_badge.text() == "template"
    assert page.centres.currentData() == CentreMode.RUNNING.value
    loaded.redo_action.trigger()
    page.template_button.click()
    assert state.config.hag == preset("template").hag
    assert loaded.deviation_button.text() == "⚠ 2 template calculation(s)"


def test_editing_the_method(loaded: MainWindow) -> None:
    page, state = loaded.configure_page, loaded.state
    page.alpha.setValue(0.45)
    assert state.config.hag.alpha == 0.45
    assert page.preset_badge.text() == "custom"
    page.kappa.setValue(3)
    page.passes.setCurrentIndex(page.passes.findData(1))
    page.passes.activated.emit(page.passes.currentIndex())
    assert (state.config.hag.kappa, state.config.hag.step4_passes) == (3, 1)
    assert not page.passes_badge.isVisibleTo(page)
    assert page.centres_badge.isVisibleTo(page)
    assert "hag.step4_passes" not in page.deviations.text()
    page.skip_constant.setChecked(True)
    page.seed.setValue(7)
    page.name.setText("heart study")
    page.name.editingFinished.emit()
    assert state.config.synthetic.skip_constant
    assert (state.config.seed, state.config.name) == (7, "heart study")
    page.majorizer.combo.setCurrentIndex(page.majorizer.combo.findData("tanh"))
    assert state.config.hag.majorizer == plugin("tanh")
    count = state.undo_stack.count()
    page.alpha.setValue(0.45)  # the same value: nothing to undo
    assert state.undo_stack.count() == count
    for _ in range(count):
        loaded.undo_action.trigger()
    assert state.config.hag == preset("template").hag
    assert page.alpha.value() == 0.3
    assert page.name.text() == "cs-model"


def test_the_k_rule_and_its_preview(loaded: MainWindow) -> None:
    page, state = loaded.configure_page, loaded.state
    page.k.combo.setCurrentIndex(page.k.combo.findData("explicit"))
    assert state.config.k.name == "explicit"
    assert any("explicit" in problem for problem in page.config_problems())
    assert page.problems.text().startswith("✗ ")
    values = page.k.form.widget("values")
    assert isinstance(values, QLineEdit)
    values.setText("3")
    values.editingFinished.emit()
    assert state.config.k == plugin("explicit", values=[3])
    assert page.problems.text() == ""
    assert "k = 3  →  r = 3 synthetic features" in page.k_preview.text()
    values.setText("21")
    values.editingFinished.emit()
    assert page.k_preview.text() == "No permitted k for Heart-Disease (10, 13, 2)."
    assert page.config_problems()
    loaded.state.set_dataset(None)
    assert page.k_preview.text() == "Load a dataset to see the permitted k."


def test_editing_the_operators(loaded: MainWindow) -> None:
    page, state = loaded.configure_page, loaded.state
    editor = page.operators
    editor.add_button.click()
    assert [o.label for o in state.config.context.operators] == ["ρ", "ρ_I", "ρ_J", "ρ4"]
    assert "r = 8 synthetic features (4 operator(s) × 2 k)" in page.k_preview.text()
    row = page.operators.rows[3]
    row.features.setCurrentText("x₁, x₄")
    row.features.lineEdit().editingFinished.emit()  # type: ignore[union-attr]
    assert state.config.context.operators[3].features == ("x₁", "x₄")
    row = page.operators.rows[3]
    row.label.setText("ρ")  # a label that is taken
    row.label.editingFinished.emit()
    assert "operator labels must be unique" in page.operator_error.text()
    assert state.config.context.operators[3].label == "ρ4"  # not accepted
    row.label.setText("")
    row.label.editingFinished.emit()
    assert page.operator_error.isVisibleTo(page)
    assert row.label.property("invalid") is True
    row.label.setText("ρ_x")
    row.label.editingFinished.emit()
    assert not page.operator_error.isVisibleTo(page)
    assert state.config.context.operators[3].label == "ρ_x"
    page.operators.rows[3].remove.click()
    assert len(state.config.context.operators) == 3
    for _ in range(4):  # the last operator cannot be removed
        page.operators.rows[0].remove.click()
    assert [o.label for o in state.config.context.operators] == ["ρ_J"]
    assert "r = 2 synthetic features (1 operator(s) × 2 k)" in page.k_preview.text()


def test_the_evaluation_settings(loaded: MainWindow) -> None:
    page, state = loaded.configure_page, loaded.state
    page.protocols.boxes["hold-out"].setChecked(True)
    assert [s.name for s in state.config.evaluation.protocols] == [
        "resubstitution",
        "leave-one-out",
        "hold-out",
    ]
    page.baselines.boxes["knn-vote"].setChecked(False)
    assert state.config.evaluation.baselines == ()
    assert [page.positive.itemText(i) for i in range(page.positive.count())] == ["1", "2"]
    page.positive.setCurrentIndex(1)
    page.positive.activated.emit(1)
    assert state.config.evaluation.positive_class == 2
    page.score_decimals.setValue(6)
    page.distance_decimals.setValue(8)
    page.skip_empty.setChecked(False)
    assert state.config.evaluation.score_decimals == 6
    assert state.config.context.distance_decimals == 8
    assert not state.config.context.skip_empty_operators
    for name in ("resubstitution", "leave-one-out", "hold-out"):
        page.protocols.boxes[name].setChecked(False)
    assert "evaluation: choose at least one protocol" in page.config_problems()
    assert not loaded.run_page.run_button.isEnabled()


def test_the_positive_class_without_a_dataset(window: MainWindow) -> None:
    page = window.configure_page
    assert [page.positive.itemText(i) for i in range(page.positive.count())] == ["1"]
    page.positive.setEditText("present")
    page.positive.lineEdit().editingFinished.emit()  # type: ignore[union-attr]
    assert window.state.config.evaluation.positive_class == "present"
    page.positive.setEditText("2")
    page.positive.lineEdit().editingFinished.emit()  # type: ignore[union-attr]
    assert window.state.config.evaluation.positive_class == 2


# ---------------------------------------------------------------- Run


def test_the_run_page_needs_a_dataset(window: MainWindow) -> None:
    page = window.run_page
    assert page.blockers_text() == ["no dataset is loaded (page Dataset)"]
    assert not page.run_button.isEnabled()
    assert page.plan.text() == "Nothing to run yet."
    page.start()  # nothing happens
    assert not page.is_running()
    assert page.wait()
    assert not page.summary.isVisibleTo(page)


def test_running_the_experiment(loaded: MainWindow, run: Run, qtbot: QtBot) -> None:
    page = loaded.run_page
    assert page.plan.text() == (
        "Heart-Disease (10, 13, 2) · m = 10, n = 13 · protocols: resubstitution, leave-one-out · "
        "⚠ template calculation for hag.centres, hag.step4_passes"
    )
    assert page.run_button.isEnabled()
    with qtbot.waitSignal(page.finished, timeout=RUN_TIMEOUT):
        run(loaded)
    view = loaded.state.view
    assert view is not None
    assert view.hag.label == "{a₆, a₃, a₁, a₂, a₄}"
    assert view.sensitivity is not None
    assert loaded.state.run_folder is None
    assert page.stage.text() == "Finished: TUPLAM = {a₆, a₃, a₁, a₂, a₄}, p = 4."
    assert page.progress.value() == page.progress.maximum()
    assert page.summary_note.text() == "Not saved."
    assert page.summary.table().key == "summary"  # type: ignore[union-attr]
    log = page.console.toPlainText()
    assert "— run of Heart-Disease (10, 13, 2) started —" in log
    assert "model fitted" in log  # the package's log reaches the console
    assert log.rstrip().endswith("— finished —")
    assert page.explore_button.isEnabled()
    assert not page.cancel_button.isEnabled()
    page.explore_button.click()
    assert loaded.current_page() is loaded.results_page


def test_a_saved_run_and_staleness(loaded: MainWindow, run: Run, tmp_path: Path) -> None:
    page = loaded.run_page
    page.folder_dialog = lambda: str(tmp_path / "my runs")
    page.browse.click()
    assert page.runs_dir.text() == str(tmp_path / "my runs")
    page.sensitivity.setCurrentIndex(page.sensitivity.findData("no"))
    run(loaded, save=True)  # type: ignore[call-arg]
    folder = loaded.state.run_folder
    assert folder is not None
    assert folder.parent == tmp_path / "my runs"
    assert (folder / "dataset.json").is_file()
    assert loaded.state.view.sensitivity is None  # type: ignore[union-attr]
    assert loaded.settings.runs_dir == tmp_path / "my runs"
    assert loaded.settings.recent("recent/runs") == [str(folder)]
    assert page.summary_note.text() == f"Run folder: {folder}"
    assert page.stale.text() == ""
    loaded.configure_page.article_button.click()
    assert "computed with another configuration" in page.stale.text()
    assert "article calculation for both switches" in page.plan.text()
    assert "⚠ computed with another configuration" in loaded.result_label.text()
    page.folder_dialog = lambda: ""
    page.browse.click()  # cancelled
    assert page.runs_dir.text() == str(tmp_path / "my runs")


def test_cancelling_a_run(window: MainWindow, qtbot: QtBot) -> None:
    assert window.dataset_page.load("heart-disease-270")  # 270 leave-one-out folds: long enough
    page = window.run_page
    page.save.setChecked(False)
    page.start()
    assert page.is_running()
    assert page.cancel_button.isEnabled()
    page.start()  # a second start while running is ignored
    # the page's own signal comes after it has handled the end of the thread; checking the
    # buttons as soon as the thread stops would race with that
    with qtbot.waitSignal(page.finished, timeout=RUN_TIMEOUT):
        page.cancel()
        assert page.stage.text() == "Cancelling…"
    assert not page.is_running()
    assert page.stage.text() == "Cancelled."
    assert window.state.view is None
    assert "— cancelled —" in page.console.toPlainText()
    assert page.progress.value() == 0
    assert page.run_button.isEnabled()
    page.cancel()  # nothing runs: nothing to cancel


def test_a_failed_run_is_reported(
    loaded: MainWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    from context_synthetic_recognition.errors import ModelUndefinedError
    from context_synthetic_recognition.gui.pages import run as run_module

    def fail(*args: Any, **kwargs: Any) -> None:
        raise ModelUndefinedError("every synthetic feature is constant")

    monkeypatch.setattr(run_module, "run_job", fail)
    page = loaded.run_page
    with qtbot.waitSignal(page.finished, timeout=RUN_TIMEOUT):
        page.start()
    assert page.stage.text() == "Failed: every synthetic feature is constant"
    assert "✗ every synthetic feature is constant" in page.console.toPlainText()
    assert loaded.state.view is None
    assert not page.is_running()
    assert page.run_button.isEnabled()  # the experiment can be started again


# ---------------------------------------------------------------- Results


def test_the_results_tree(evaluated: MainWindow) -> None:
    page = evaluated.results_page
    assert page.keys(TABLE) == EXPERIMENT_TABLES
    assert page.keys(FIGURE) == EXPERIMENT_FIGURES
    assert page.keys(CONTEXT) == [CONTEXT]
    assert page.current_key() == (TABLE, "summary")
    assert page.stack.currentWidget() is page.table_panel
    groups = [page.tree.topLevelItem(i).text(0) for i in range(page.tree.topLevelItemCount())]
    assert groups[:4] == [
        "Summary",
        "1 · Input data",
        "Step 1 · Scale unification",
        "Step 2 · Distances",
    ]
    assert groups[-2:] == ["Evaluation", "Checks"]
    view = evaluated.state.view
    assert view is not None
    assert table_name("distances-rho-i", view) == "Distances · ρ_I"
    assert table_name("confusion-leave-one-out", view) == "Confusion · leave-one-out"
    assert table_name("psi", view) == "Ψ(r)"
    assert table_name("something-else", view) == "something-else"
    assert figure_name("outcomes-leave-one-out") == "Outcomes · leave-one-out"
    assert figure_name("margins") == "Margins of the latent features"
    assert figure_name("unknown") == "unknown"


def test_opening_tables_and_figures(evaluated: MainWindow) -> None:
    page = evaluated.results_page
    assert page.open(TABLE, "psi")
    panel = page.table_panel
    assert panel.title.text() == "Ψ(r): synthetic features aᵤ(Sⱼ) ∈ {1, 2} by formula (5)"
    assert panel.note.text().startswith("aᵤ = 1 if χ₁ > [k/2]")
    assert panel.count.text() == "10 rows"
    assert panel.view.model().index(0, 1).data() == "2"
    assert page.open(FIGURE, "margins")
    assert page.stack.currentWidget() is page.figure_panel
    figure = page.figure_panel.figure()
    assert figure is not None
    assert len(figure.get_axes()) == 8  # four latent features, with and without the majorizer
    assert page.figure_panel.caption.text().startswith("The latent features r₁ … r_p on a line")
    assert not page.open(TABLE, "confusion-hold-out")  # this run has no hold-out
    assert page.current_key() == (FIGURE, "margins")
    assert page.open(TABLE, "margins")
    assert page.table("margins") is page.table("margins")  # built once


def test_stepping_through_the_results(evaluated: MainWindow) -> None:
    page = evaluated.results_page
    assert not page.previous.isEnabled()
    page.next.click()
    assert page.current_key() == (TABLE, "dataset")
    page.step(1)
    assert page.current_key() == (TABLE, "features")
    page.previous.click()
    page.previous.click()
    page.previous.click()  # the first leaf: stays
    assert page.current_key() == (TABLE, "summary")
    page.step(10_000)
    assert not page.next.isEnabled()
    assert page.current_key() == (TABLE, "boundary-ties")
    evaluated.go_to(page)
    evaluated._step(-1)  # Alt+Left
    assert page.current_key() == (TABLE, "operator-pairs")
    evaluated.go_to(evaluated.run_page)
    evaluated._step(-1)  # only on the Results page
    assert page.current_key() == (TABLE, "operator-pairs")


def test_following_an_object_through_the_tables(evaluated: MainWindow, qtbot: QtBot) -> None:
    page, state = evaluated.results_page, evaluated.state
    page.open(TABLE, "psi")
    view = page.table_panel.view
    with qtbot.waitSignal(state.objectSelected) as selected:
        view.clicked.emit(view.model().index(2, 3))  # a click anywhere in the row of S₃
    assert selected.args == [2]
    assert page.objects.currentText() == "S₃"
    model = page.table_panel.model()
    assert model is not None
    assert model.index(2, 0).data(Qt.ItemDataRole.BackgroundRole).color().name() == LIGHT.highlight
    page.open(TABLE, "neighbours-rho")  # the selection follows into the next table
    model = page.table_panel.model()
    assert model is not None
    assert model.index(12, 0).data(Qt.ItemDataRole.BackgroundRole).color().name() == LIGHT.highlight
    page.objects.setCurrentIndex(page.objects.findText("S₇"))
    page.objects.activated.emit(page.objects.currentIndex())
    assert state.selected_object == 6
    page.objects.setCurrentIndex(0)
    page.objects.activated.emit(0)
    assert state.selected_object == -1
    assert model.index(12, 0).data(Qt.ItemDataRole.BackgroundRole) is None
    page.open(TABLE, "summary")
    summary = page.table_panel.view
    summary.clicked.emit(summary.model().index(0, 0))  # a table without objects: no selection
    assert state.selected_object == -1


def test_the_neighbourhood_of_the_selected_object(evaluated: MainWindow) -> None:
    page = evaluated.results_page
    assert page.open(CONTEXT, CONTEXT)
    panel = page.context_panel
    assert page.stack.currentWidget() is panel
    assert panel.figure() is None
    assert panel.message.isVisibleTo(panel)
    evaluated.state.select_object(2)
    figure = panel.figure()
    assert figure is not None
    assert figure.get_axes()[0].get_title() == "ρ: neighbourhoods of S₃  (radius = distance)"
    assert not panel.message.isVisibleTo(panel)
    panel.operator.setCurrentIndex(2)
    assert panel.figure().get_axes()[0].get_title().startswith("ρ_J: neighbourhoods of S₃")  # type: ignore[union-attr]
    evaluated.state.select_object(-1)
    assert panel.figure() is None


def test_decimals_filter_copy_and_save(evaluated: MainWindow, tmp_path: Path) -> None:
    page = evaluated.results_page
    page.open(TABLE, "margins")
    panel = page.table_panel
    model = panel.view.model()
    width = panel.table().columns.index("Margin width")  # type: ignore[union-attr]
    assert model.index(0, width).data() == "0.1482"
    page.decimals.setValue(2)
    assert model.index(0, width).data() == "0.15"
    panel.filter.setText("without")
    assert panel.visible_rows() == 4
    assert panel.count.text() == "4 of 8 rows"
    panel.copy_button.click()
    copied = QGuiApplication.clipboard().text().splitlines()
    assert copied[0].split("\t")[:2] == ["Latent feature", "Variant"]
    assert len(copied) == 5
    assert copied[1].split("\t")[:2] == ["r₁", "without majorizer"]
    panel.filter.clear()
    panel.view.selectRow(0)
    assert len(panel.view.selection_text().splitlines()) == 2  # the header and the selected row
    target = tmp_path / "margins.csv"
    panel.save_dialog = lambda suggested: str(target) if suggested == "margins.csv" else ""
    panel.save_button.click()
    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("Latent feature,Variant,")
    assert len(lines) == 9
    assert "0.14817726709828" in lines[1]  # the file keeps every digit
    panel.save_dialog = lambda suggested: ""
    panel.save_button.click()  # cancelled


def test_sorting_by_a_column(evaluated: MainWindow) -> None:
    page = evaluated.results_page
    page.open(TABLE, "synthetic-features")
    view = page.table_panel.view
    model = view.model()
    names = lambda: [model.index(r, 0).data() for r in range(model.rowCount())]  # noqa: E731
    assert names() == ["a₁", "a₂", "a₃", "a₄", "a₅", "a₆"]
    omega = page.table_panel.table().columns.index("ω (4)")  # type: ignore[union-attr]
    view.sortByColumn(omega, Qt.SortOrder.AscendingOrder)
    assert names()[0] == "a₃"  # ω = 0.6, the least informative
    view.sortByColumn(0, Qt.SortOrder.DescendingOrder)
    assert names()[0] == "a₆"
    page.open(TABLE, "psi")  # another table opens in pipeline order again
    assert page.table_panel.view.model().index(0, 0).data() == "S₁"


def test_saving_a_figure(evaluated: MainWindow, tmp_path: Path) -> None:
    page = evaluated.results_page
    page.open(FIGURE, "roc")
    panel = page.figure_panel
    target = tmp_path / "roc.svg"
    panel.save_dialog = lambda suggested: str(target) if suggested == "roc.png" else ""
    panel.save_button.click()
    assert "<svg" in target.read_text(encoding="utf-8")
    panel.save_dialog = lambda suggested: ""
    panel.save_button.click()  # cancelled
    panel.set_spec(None)
    assert panel.figure() is None
    assert not panel.save_button.isEnabled()
    panel.save_button.click()


def test_the_results_follow_the_theme_and_a_new_run(evaluated: MainWindow, run: Run) -> None:
    page = evaluated.results_page
    page.open(FIGURE, "roc")
    evaluated.set_theme("dark")
    figure = page.figure_panel.figure()
    assert figure is not None
    assert figure.get_facecolor()[:3] == pytest.approx((0x1A / 255, 0x1A / 255, 0x19 / 255))
    page.open(TABLE, "psi")
    model = page.table_panel.model()
    assert model is not None
    assert (
        model.index(0, 1).data(Qt.ItemDataRole.BackgroundRole).color().name()
        == DARK.cells[
            model.style.cell(0, 1, 2)  # type: ignore[misc,index]
        ][0]
    )
    evaluated.configure_page.article_button.click()
    assert "computed with another configuration" in page.stale.text()
    run(evaluated)
    assert page.stale.text() == ""
    assert page.current_key() == (TABLE, "psi")  # the table that was open stays open
    assert evaluated.state.view.hag.settings.step4_passes == 1  # type: ignore[union-attr]


# ---------------------------------------------------------------- New object


def test_the_new_object_page_shows_the_default_demonstration(evaluated: MainWindow) -> None:
    page = evaluated.new_object_page
    assert page.isEnabled()
    assert page.form.rowCount() == 13
    assert page.source.currentText() == "S₁"
    assert page.exclude.isChecked()
    assert page.values_text() == "70;1;4;130;322;0;2;109;0;2.4;2;3;3"
    assert page.decision.text() == "Class 1"
    assert page.scores.text() == "score₁ = |B1|/|K1| = 0.5000,  score₂ = |B2|/|K2| = 0.0000"
    assert page.theorem.text() == (
        "Theorem check for S₁: ✓ same representation as in training; ✓ same decision."
    )
    assert page.form.item(0, 3).text() == "56 … 74"
    assert page.form.item(1, 3).text() == "2 codes: 0, 1"
    assert page.steps.table().key == "new-object-steps"  # type: ignore[union-attr]
    assert page.features.table().rows[0][:3] == ("a₁", "ρ", 3)  # type: ignore[union-attr]
    neighbours = page.neighbours.table()
    assert neighbours is not None
    assert neighbours.rows[0][:2] == (1, "S₉")
    assert neighbours.rows[2][4] == "k = 3"
    assert len(neighbours.rows) == 9  # S₁ is left out of its own context
    assert page.canvas.figure() is not None


def test_classifying_a_training_object(evaluated: MainWindow) -> None:
    page, state = evaluated.new_object_page, evaluated.state
    page.source.setCurrentIndex(page.source.findText("S₄"))
    page.source.activated.emit(page.source.currentIndex())
    assert page.values_text().startswith("64;1;4;128;263")
    assert page.classify()
    view = state.view
    assert view is not None
    assert view.new_object.exclude == 3
    assert page.theorem.text().startswith("Theorem check for S₄: ✓ same representation")
    assert (
        page.decision.text() == f"Class {view.trace.classes[int(view.training.decisions[3]) - 1]}"
    )
    # the Results tables and the exports now show this object
    evaluated.results_page.open(TABLE, "new-object")
    assert "S₄ left out of its own context" in evaluated.results_page.table_panel.title.text()
    assert "S₄ left out of its own context" in evaluated.export_page.demo.text()
    page.exclude.setChecked(False)  # S₄ among its own neighbours
    assert page.classify()
    assert state.view.new_object.exclude is None  # type: ignore[union-attr]
    assert page.neighbours.table().rows[0][:3] == (1, "S₄", 0.0)  # type: ignore[union-attr]
    assert page.theorem.text() == "The class of the object is not an input of Ψ, D or R."


def test_classifying_a_typed_object(evaluated: MainWindow) -> None:
    page, state = evaluated.new_object_page, evaluated.state
    page.paste.setText("58, 1, 3, 125, 250, 0, 2, 150, 0, 1, 2, 1, 7")
    assert page.fill_from_text()
    assert page.source.currentIndex() == 0  # "a new object"
    assert page.values_text() == "58;1;3;125;250;0;2;150;0;1;2;1;7"
    page.classify_button.click()
    view = state.view
    assert view is not None
    assert view.new_object.exclude is None
    assert view.new_object.values.tolist()[:3] == [58.0, 1.0, 3.0]
    assert page.decision.text() in {"Class 1", "Class 2", "0 — refusal"}
    assert len(page.neighbours.table().rows) == 10  # type: ignore[union-attr]
    assert page.canvas.figure().get_axes()[0].get_title().startswith("ρ: neighbourhoods of S ")  # type: ignore[union-attr]
    page.operator.setCurrentIndex(1)
    assert "ρ_I" in page.neighbours.title.text() or "ρ_I" in page.neighbours.table().title  # type: ignore[union-attr]
    assert page.canvas.figure().get_axes()[0].get_title().startswith("ρ_I:")  # type: ignore[union-attr]
    QGuiApplication.clipboard().setText("60 0 2 120 200 0 0 140 0 0.5 1 0 3")
    page.paste_button.click()
    assert page.values_text().startswith("60;0;2;120")


def test_values_that_cannot_be_read(evaluated: MainWindow) -> None:
    page = evaluated.new_object_page
    before = evaluated.state.view
    page.paste.setText("1 2 3")
    assert not page.fill_from_text()
    assert page.error.text() == "expected 13 values (one per feature), got 3"
    assert page.error.isVisibleTo(page)
    page.form.item(0, 2).setText("")
    assert not page.classify()
    assert page.error.text() == "every feature needs a value"
    page.form.item(0, 2).setText("old")
    assert not page.classify()
    assert "'old' is not a number" in page.error.text()
    assert evaluated.state.view is before
    page.form.item(0, 2).setText("70")
    assert page.classify()
    assert not page.error.isVisibleTo(page)


def test_the_new_object_page_without_a_run(window: MainWindow) -> None:
    page = window.new_object_page
    assert not page.isEnabled()
    assert page.form.rowCount() == 0
    assert page.decision.text() == "—"
    assert not page.classify()
    assert not page.fill_from_text()
    page.apply_palette(DARK)


# ---------------------------------------------------------------- Compare


@pytest.fixture
def two_runs(evaluated: MainWindow, run: Run, tmp_path: Path) -> MainWindow:
    """A window whose runs folder holds the template run and the article run."""
    view = evaluated.state.view
    assert view is not None
    runs = tmp_path / "runs"
    save_run(view.result, runs)
    evaluated.configure_page.article_button.click()
    evaluated.run_page.runs_dir.setText(str(runs))
    run(evaluated, save=True)  # type: ignore[call-arg]
    return evaluated


def test_comparing_two_runs(two_runs: MainWindow, qtbot: QtBot) -> None:
    page = two_runs.compare_page
    assert page.runs_dir.text() == str(two_runs.settings.runs_dir)
    assert len(page.records) == 2
    assert page.runs.rowCount() == 2
    assert page.runs.item(0, 4).text() in {"template", "article"}
    assert page.runs.item(0, 6).text().endswith("(leave-one-out)")
    assert page.message.text() == "Tick at least two runs."
    assert not page.facts.isVisibleTo(page)
    page.set_checked(0)
    assert page.checked() == [page.records[0]]
    assert not page.facts.isVisibleTo(page)
    page.set_checked(1)
    facts_table = page.facts.table()
    assert facts_table is not None
    assert facts_table.columns == ("Fact", page.records[0].run_id, page.records[1].run_id)
    rows = {row[0]: row[1:] for row in facts_table.rows}
    assert set(rows["Preset"]) == {"template", "article"}
    model = page.facts.model()
    assert model is not None
    preset_row = [row[0] for row in facts_table.rows].index("Preset")
    dataset_row = [row[0] for row in facts_table.rows].index("Dataset")
    background = Qt.ItemDataRole.BackgroundRole
    assert model.index(preset_row, 1).data(background) is not None  # they differ: marked
    assert model.index(dataset_row, 1).data(background) is None
    settings = page.settings_table.table()
    assert settings is not None
    assert [row[0] for row in settings.rows] == ["hag.centres", "hag.step4_passes"]
    page.set_checked(0, False)
    assert page.message.isVisibleTo(page)
    page.runs.selectRow(1)
    assert page.open_button.isEnabled()
    # the request repeats the run in the worker thread; wait until the page has handled its end
    with (
        qtbot.waitSignal(two_runs.run_page.finished, timeout=RUN_TIMEOUT),
        qtbot.waitSignal(page.openRequested, timeout=5000),
    ):
        page.open_button.click()
    assert two_runs.state.run_folder == page.records[1].folder


def test_an_unsaved_result_can_be_compared(evaluated: MainWindow, tmp_path: Path) -> None:
    page = evaluated.compare_page
    assert [record.run_id for record in page.records] == ["current"]  # not saved: listed first
    assert page.records[0].folder is None
    page.runs.selectRow(0)
    assert not page.open_button.isEnabled()  # there is no folder to open
    page.open_button.click()
    page.folder_dialog = lambda: str(tmp_path / "elsewhere")
    page.browse.click()
    assert page.runs_dir.text() == str(tmp_path / "elsewhere")
    assert evaluated.settings.runs_dir == tmp_path / "elsewhere"
    page.folder_dialog = lambda: ""
    page.browse.click()
    assert page.runs_dir.text() == str(tmp_path / "elsewhere")


def test_run_folders_that_cannot_be_read_are_reported(two_runs: MainWindow) -> None:
    page = two_runs.compare_page
    broken = two_runs.settings.runs_dir / "20200101_000000_ffffffff"
    broken.mkdir()
    page.set_checked(0)
    page.set_checked(1)
    page.refresh_button.click()
    assert "cannot read the manifest" in page.problems.text()
    assert len(page.records) == 2
    assert len(page.checked()) == 2  # the ticks survive a refresh
    page.apply_palette(DARK)


def test_many_runs_are_cut_to_a_screen(
    two_runs: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    from context_synthetic_recognition.gui.pages import compare

    monkeypatch.setattr(compare, "MAX_COMPARED", 1)
    page = two_runs.compare_page
    page.set_checked(0)
    page.set_checked(1)
    assert page.message.text() == "The first 1 of the 2 ticked runs are shown."
    assert len(page.facts.table().columns) == 2  # type: ignore[union-attr]


# ---------------------------------------------------------------- Export


def test_the_export_page(evaluated: MainWindow, qtbot: QtBot, tmp_path: Path) -> None:
    page = evaluated.export_page
    assert page.formats() == ["excel", "csv", "json", "figures", "html"]
    assert page.folder.text() == str(Path("exports") / "Heart-Disease (10, 13, 2)")
    assert page.demo.text() == (
        "Meta-algorithm sheets show S₁ left out of its own context; "
        "with the four ⚠ switch settings."
    )
    page.none_button.click()
    assert page.formats() == []
    assert not page.export_button.isEnabled()
    page.start()  # nothing ticked: nothing happens
    page.all_button.click()
    assert len(page.formats()) == 8
    page.set_formats(["csv", "json"])
    page.folder_dialog = lambda: str(tmp_path / "out")
    page.browse.click()
    page.decimals.setValue(2)
    page.dpi.setValue(80)
    page.figure_theme.setCurrentText("dark")
    options = page.options()
    assert (options.decimals, options.dpi, options.theme) == (2, 80, "dark")
    with qtbot.waitSignal(page.finished, timeout=RUN_TIMEOUT):
        page.export_button.click()
    summary = page.summary
    assert summary is not None
    assert summary.folder == tmp_path / "out"
    assert list(summary.files) == ["csv", "json"]
    assert (tmp_path / "out" / "trace.json").is_file()
    assert page.status.text() == f"{summary.count} files written to {tmp_path / 'out'}"
    report = page.report.toPlainText().splitlines()
    assert report[0] == f"csv: tables/  ({len(EXPERIMENT_TABLES) + 1} file(s))"
    assert report[1] == "json: trace.json  (1 file(s))"
    assert page.progress.value() == page.progress.maximum() == 2
    opened: list[Path] = []
    page.open_folder = opened.append
    page.open_button.click()
    assert opened == [tmp_path / "out"]
    page.folder_dialog = lambda: ""
    page.browse.click()
    assert page.folder.text() == str(tmp_path / "out")
    trace = json.loads((tmp_path / "out" / "trace.json").read_text(encoding="utf-8"))
    assert trace["run"] is None  # the run was not saved


def test_an_export_that_fails_is_reported(
    evaluated: MainWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from context_synthetic_recognition.gui.pages import export as export_module

    def fail(*args: Any, **kwargs: Any) -> None:
        raise ExportError("the PDF report needs reportlab")

    monkeypatch.setattr(export_module, "export_job", fail)
    page = evaluated.export_page
    page.folder.setText(str(tmp_path))
    page.set_formats(["pdf"])
    with qtbot.waitSignal(page.finished, timeout=RUN_TIMEOUT):
        page.start()
    assert page.status.text() == "Export failed: the PDF report needs reportlab"
    assert page.summary is None
    assert page.export_button.isEnabled()
    page.open_button.click()  # nothing was written: nothing to open


def test_the_export_page_follows_the_run(two_runs: MainWindow) -> None:
    page = two_runs.export_page
    assert page.folder.text() == str(
        two_runs.state.run_folder
    )  # a saved run exports into its folder
    two_runs.configure_page.template_button.click()
    assert "the export shows the run, not the edited configuration" in page.stale.text()
    two_runs.new_object_page.paste.setText("58 1 3 125 250 0 2 150 0 1 2 1 7")
    two_runs.new_object_page.fill_from_text()
    two_runs.new_object_page.classify()
    assert page.demo.text().startswith("Meta-algorithm sheets show a new object; ")


def test_the_export_page_without_a_run(window: MainWindow) -> None:
    page = window.export_page
    assert page.demo.text() == "Run the experiment first."
    assert not page.export_button.isEnabled()
    page.start()
    assert page.wait()
