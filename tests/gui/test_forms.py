"""Forms generated from the plug-ins' parameter types."""

from typing import Any

import pytest
from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QLineEdit, QSpinBox
from pytestqt.qtbot import QtBot

from context_synthetic_recognition.config import PluginSpec, plugin
from context_synthetic_recognition.core.k_strategies import K_STRATEGIES
from context_synthetic_recognition.core.majorizers import MAJORIZERS
from context_synthetic_recognition.evaluation.baselines import BASELINES
from context_synthetic_recognition.evaluation.protocols import PROTOCOLS
from context_synthetic_recognition.gui.forms import (
    NONE_TEXT,
    ParamForm,
    PluginChecklist,
    PluginEditor,
    label_for,
    schema_properties,
)

pytestmark = pytest.mark.gui


def params_type(registry: Any, name: str) -> type:
    found = registry.info(name).params_type
    assert found is not None
    return found  # type: ignore[no-any-return]


def test_schema_properties() -> None:
    assert schema_properties(None) == {}
    fields = schema_properties(params_type(K_STRATEGIES, "formula"))
    assert list(fields) == ["k_max_cap", "step"]
    assert fields["step"]["default"] == 2
    assert fields["step"]["required"] is False
    assert schema_properties(params_type(K_STRATEGIES, "explicit"))["values"]["required"] is True
    assert label_for("k_max_cap") == "k max cap"


def test_a_form_reports_only_what_differs_from_the_defaults(qtbot: QtBot) -> None:
    form = ParamForm(params_type(K_STRATEGIES, "formula"))
    qtbot.addWidget(form)
    assert form.fields() == ["k_max_cap", "step"]
    assert form.values() == {}
    cap, step = form.widget("k_max_cap"), form.widget("step")
    assert isinstance(cap, QSpinBox)
    assert isinstance(step, QSpinBox)
    assert cap.text() == NONE_TEXT  # an optional number that is not set
    assert (cap.minimum(), step.minimum(), step.value()) == (2, 2, 2)
    with qtbot.waitSignal(form.changed):
        cap.setValue(21)
    assert form.values() == {"k_max_cap": 21}
    step.setValue(4)
    assert form.values() == {"k_max_cap": 21, "step": 4}
    form.set_values({"step": 6})
    assert form.values() == {"step": 6}
    assert cap.text() == NONE_TEXT
    form.set_values({})
    assert form.values() == {}


def test_every_kind_of_field(qtbot: QtBot) -> None:
    kfold = ParamForm(params_type(PROTOCOLS, "stratified-k-fold"))
    hold_out = ParamForm(params_type(PROTOCOLS, "hold-out"))
    knn = ParamForm(params_type(BASELINES, "knn-vote"))
    sklearn = ParamForm(params_type(BASELINES, "naive-bayes"))
    for form in (kfold, hold_out, knn, sklearn):
        qtbot.addWidget(form)
    shuffle = kfold.widget("shuffle")
    assert isinstance(shuffle, QCheckBox)
    assert shuffle.isChecked()
    shuffle.setChecked(False)
    assert kfold.values() == {"shuffle": False}
    size = hold_out.widget("test_size")
    assert isinstance(size, QDoubleSpinBox)
    assert size.value() == pytest.approx(0.3)
    assert 0.0 < size.minimum() < size.maximum() < 1.0  # the bounds are exclusive
    size.setValue(0.25)
    assert hold_out.values() == {"test_size": 0.25}
    ks, operators = knn.widget("ks"), knn.widget("operators")
    assert isinstance(ks, QLineEdit)
    assert isinstance(operators, QLineEdit)
    assert (ks.text(), operators.text()) == ("3, 5", "")
    ks.setText("1; 3 ,7")
    operators.setText("ρ, ρ_I")
    assert knn.values() == {"ks": [1, 3, 7], "operators": ["ρ", "ρ_I"]}
    ks.setText("")  # nothing typed: the default
    operators.setText("")
    assert knn.values() == {}
    options = sklearn.widget("options")
    assert isinstance(options, QLineEdit)
    options.setText('{"var_smoothing": 1e-8}')
    assert sklearn.values() == {"options": {"var_smoothing": 1e-8}}
    sklearn.set_values({"options": {"priors": [0.4, 0.6]}})
    assert options.text() == '{"priors": [0.4, 0.6]}'


def test_unreadable_input_is_marked(qtbot: QtBot) -> None:
    form = ParamForm(params_type(K_STRATEGIES, "explicit"))
    qtbot.addWidget(form)
    values = form.widget("values")
    assert isinstance(values, QLineEdit)
    assert form.values() == {}  # required and empty
    assert form.error == "values: a value is needed"
    assert values.property("invalid") is True
    values.setText("3, five")
    assert form.values() == {}
    assert "invalid literal" in form.error
    with qtbot.waitSignal(form.changed):
        values.setText("3, 5")
        values.editingFinished.emit()
    assert form.values() == {"values": [3, 5]}
    assert form.error == ""
    assert values.property("invalid") is False
    empty = ParamForm()
    qtbot.addWidget(empty)
    assert empty.fields() == []
    assert not empty.isVisibleTo(empty)


def test_the_plugin_editor(qtbot: QtBot) -> None:
    editor = PluginEditor(K_STRATEGIES)
    qtbot.addWidget(editor)
    assert [editor.combo.itemText(i) for i in range(editor.combo.count())] == K_STRATEGIES.names()
    assert editor.spec() == plugin("formula")
    assert editor.summary.text() == K_STRATEGIES.info("formula").summary
    assert editor.combo.itemData(0, 3) == editor.summary.text()  # the tooltip of the item
    with qtbot.assertNotEmitted(editor.changed):
        editor.set_spec(plugin("range", stop=9))
    assert editor.spec() == plugin("range", stop=9)
    assert editor.form.fields() == ["stop", "start", "step"]
    assert editor.validate() == ""
    with qtbot.waitSignal(editor.changed):
        editor.combo.setCurrentIndex(editor.combo.findData("explicit"))
    assert editor.form.fields() == ["values"]
    assert "a value is needed" in editor.problem.text()
    values = editor.form.widget("values")
    assert isinstance(values, QLineEdit)
    values.setText("4")
    with qtbot.waitSignal(editor.changed):
        values.editingFinished.emit()
    assert editor.spec() == plugin("explicit", values=[4])
    assert "odd" in editor.problem.text()  # the plug-in's own check: every k must be odd
    assert editor.problem.isVisibleTo(editor)
    values.setText("3")
    values.editingFinished.emit()
    assert editor.validate() == ""
    assert not editor.problem.isVisibleTo(editor)


def test_the_editor_keeps_an_unknown_plugin_visible(qtbot: QtBot) -> None:
    editor = PluginEditor(MAJORIZERS)
    qtbot.addWidget(editor)
    editor.set_spec(PluginSpec(name="logistic"))  # an alias: shown under its canonical name
    assert editor.spec() == plugin("sigmoid")
    editor.set_spec(PluginSpec(name="my-majorizer", params={"k": 2}))
    assert editor.combo.currentText() == "my-majorizer (unknown)"
    assert editor.spec().name == "my-majorizer"
    assert "Unknown majorizers 'my-majorizer'" in editor.problem.text()
    assert editor.form.fields() == []


def test_the_plugin_checklist(qtbot: QtBot) -> None:
    checklist = PluginChecklist(PROTOCOLS)
    qtbot.addWidget(checklist)
    assert list(checklist.boxes) == PROTOCOLS.names()
    assert checklist.specs() == ()
    with qtbot.assertNotEmitted(checklist.changed):
        checklist.set_specs((plugin("loo"), plugin("hold-out", test_size=0.2)))
    assert checklist.specs() == (plugin("leave-one-out"), plugin("hold-out", test_size=0.2))
    assert checklist.forms["hold-out"].isEnabled()
    assert not checklist.forms["stratified-k-fold"].isEnabled()
    with qtbot.waitSignal(checklist.changed):
        checklist.boxes["resubstitution"].setChecked(True)
    # the registry's order, whatever the order of ticking
    assert [spec.name for spec in checklist.specs()] == [
        "resubstitution",
        "leave-one-out",
        "hold-out",
    ]
    with qtbot.waitSignal(checklist.changed):
        checklist.boxes["hold-out"].setChecked(False)
    assert not checklist.forms["hold-out"].isEnabled()
    assert checklist.validate() == ""
    # an unknown plug-in of a loaded configuration is not listed; the page reports it
    checklist.set_specs((plugin("bootstrap"), plugin("resubstitution")))
    assert checklist.specs() == (plugin("resubstitution"),)


def test_the_checklist_reports_invalid_parameters(qtbot: QtBot) -> None:
    checklist = PluginChecklist(BASELINES)
    qtbot.addWidget(checklist)
    checklist.set_specs((plugin("knn-vote", ks=[2]),))
    assert "knn-vote" in checklist.problem.text()
    ks = checklist.forms["knn-vote"].widget("ks")
    assert isinstance(ks, QLineEdit)
    ks.setText("three")
    with qtbot.waitSignal(checklist.changed):
        ks.editingFinished.emit()
    assert "invalid literal" in checklist.problem.text()
    ks.setText("3")
    ks.editingFinished.emit()
    assert checklist.problem.text() == ""
    assert checklist.specs() == (plugin("knn-vote", ks=[3]),)
