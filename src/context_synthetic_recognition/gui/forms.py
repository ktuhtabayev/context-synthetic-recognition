"""Forms generated from the plug-ins' parameter types.

Every plug-in declares its parameters as a pydantic dataclass (ADR-019); its JSON schema is all a
form needs: a check box for a boolean, a spin box with the declared bounds for a number, a text
field for a list. A plug-in added through the registry therefore gets its form for free.

A form reports only the values that differ from the defaults, so a configuration edited in the
GUI stays as short as a hand-written one and keeps matching its preset.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import JsonValue, TypeAdapter
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from context_synthetic_recognition.config import PluginSpec
from context_synthetic_recognition.core.registry import Registry, normalize_name
from context_synthetic_recognition.errors import CSRError
from context_synthetic_recognition.gui.i18n import tr

INT_LIMIT = 1_000_000
NONE_TEXT = "—"
"""Shown by a spin box whose value is "not set"."""


def schema_properties(params_type: type | None) -> dict[str, dict[str, Any]]:
    """The fields of a parameter type: name → JSON schema of the field."""
    if params_type is None:
        return {}
    schema = TypeAdapter(params_type).json_schema()
    properties = schema.get("properties", {})
    required = set(schema.get("required", ()))
    return {name: {**field, "required": name in required} for name, field in properties.items()}


def _variants(field: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """A field's schema without the ``null`` alternative, and whether ``null`` is allowed."""
    options = field.get("anyOf")
    if not options:
        return field, False
    real = [option for option in options if option.get("type") != "null"]
    nullable = len(real) < len(options)
    return ({**field, **real[0]} if len(real) == 1 else field), nullable


def label_for(name: str) -> str:
    """``k_max_cap`` → ``k max cap``."""
    return name.replace("_", " ")


def set_invalid(widget: QWidget, invalid: bool) -> None:
    """Mark an input as invalid (red border) or valid again."""
    if bool(widget.property("invalid")) != invalid:
        widget.setProperty("invalid", invalid)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


class _Field:
    """One input of a parameter form."""

    def __init__(self, name: str, schema: dict[str, Any], form: ParamForm) -> None:
        self.name = name
        self.required = bool(schema.get("required"))
        self.default = schema.get("default")
        variant, self.nullable = _variants(schema)
        self.kind = str(variant.get("type", "json"))
        self.items = str(variant.get("items", {}).get("type", "")) if self.kind == "array" else ""
        self.widget: QWidget
        if self.kind == "boolean":
            box = QCheckBox()
            box.toggled.connect(form.edited)
            self.widget = box
        elif self.kind == "integer":
            spin = QSpinBox()
            low = int(variant.get("minimum", variant.get("exclusiveMinimum", -INT_LIMIT - 1) + 1))
            high = int(variant.get("maximum", INT_LIMIT))
            if self.nullable:
                low -= 1  # the step below the minimum means "not set"
                spin.setSpecialValueText(NONE_TEXT)
            spin.setRange(low, high)
            spin.valueChanged.connect(form.edited)
            self.widget = spin
        elif self.kind == "number":
            real = QDoubleSpinBox()
            real.setDecimals(4)
            real.setSingleStep(0.05)
            step = 10.0 ** -real.decimals()
            low = variant.get("minimum", variant.get("exclusiveMinimum", -1e9 - step) + step)
            high = variant.get("maximum", variant.get("exclusiveMaximum", 1e9 + step) - step)
            real.setRange(float(low), float(high))
            real.valueChanged.connect(form.edited)
            self.widget = real
        else:
            line = QLineEdit()
            if self.kind == "array":
                line.setPlaceholderText(tr("values separated by commas"))
            else:
                line.setPlaceholderText(tr("JSON"))
            line.editingFinished.connect(form.edited)
            self.widget = line
        self.set(self.default)

    def set(self, value: Any) -> None:
        """Show a value (``None`` = not set)."""
        widget = self.widget
        widget.blockSignals(True)
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QSpinBox):
            widget.setValue(widget.minimum() if value is None else int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(widget.minimum() if value is None else float(value))
        elif isinstance(widget, QLineEdit):
            if value is None:
                widget.setText("")
            elif self.kind == "array":
                widget.setText(", ".join(str(item) for item in value))
            else:
                widget.setText(json.dumps(value, ensure_ascii=False))
            set_invalid(widget, False)
        widget.blockSignals(False)

    def get(self) -> Any:
        """The value typed.

        Raises:
            ValueError: The text is not a list of the declared kind or not JSON.
        """
        widget = self.widget
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QSpinBox):
            if self.nullable and widget.value() == widget.minimum():
                return None
            return widget.value()
        if isinstance(widget, QDoubleSpinBox):
            return widget.value()
        assert isinstance(widget, QLineEdit)
        text = widget.text().strip()
        if not text:
            if self.nullable or not self.required:
                return self.default if not self.nullable else None
            raise ValueError(tr("a value is needed"))
        if self.kind == "array":
            parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
            if self.items == "integer":
                return [int(part) for part in parts]
            if self.items == "number":
                return [float(part) for part in parts]
            return parts
        return json.loads(text)


class ParamForm(QWidget):
    """The parameters of one plug-in."""

    changed = Signal()

    def __init__(self, params_type: type | None = None, parent: QWidget | None = None) -> None:
        """Build the inputs of ``params_type`` (no inputs for ``None``)."""
        super().__init__(parent)
        self._layout = QFormLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._fields: dict[str, _Field] = {}
        self.error = ""
        """Why the form cannot be read (empty if it can)."""
        self._type: type | None = None
        self._built = False
        self.set_type(params_type)

    def set_type(self, params_type: type | None) -> None:
        """Build the form of a parameter type (kept as it is if the type is the same)."""
        if self._built and params_type is self._type:
            return
        self._type = params_type
        self._built = True
        while self._layout.rowCount():
            self._layout.removeRow(0)
        self._fields = {}
        for name, schema in schema_properties(params_type).items():
            field = _Field(name, schema, self)
            self._fields[name] = field
            self._layout.addRow(label_for(name), field.widget)
        self.setVisible(bool(self._fields))

    def fields(self) -> list[str]:
        """Names of the parameters."""
        return list(self._fields)

    def widget(self, name: str) -> QWidget:
        """The input of a parameter."""
        return self._fields[name].widget

    def set_values(self, params: dict[str, JsonValue]) -> None:
        """Show ``params``; parameters not given show their defaults."""
        for name, field in self._fields.items():
            field.set(params.get(name, field.default))
        self.error = ""

    def values(self) -> dict[str, JsonValue]:
        """The parameters that differ from their defaults (required ones always).

        Unreadable inputs are marked and skipped; :attr:`error` then says what is wrong.
        """
        params: dict[str, JsonValue] = {}
        problems = []
        for name, field in self._fields.items():
            try:
                value = field.get()
            except ValueError as error:
                set_invalid(field.widget, True)
                problems.append(f"{label_for(name)}: {error}")
                continue
            set_invalid(field.widget, False)
            if field.required or value != field.default:
                params[name] = value
        self.error = "; ".join(problems)
        return params

    def edited(self, *_: object) -> None:
        """An input changed."""
        self.changed.emit()


class PluginEditor(QWidget):
    """Choose a plug-in of a registry and set its parameters."""

    changed = Signal()

    def __init__(self, registry: Registry[Any], parent: QWidget | None = None) -> None:
        """List the plug-ins of ``registry``."""
        super().__init__(parent)
        self.registry = registry
        self.combo = QComboBox()
        for info in registry:
            self.combo.addItem(info.name, info.name)
            self.combo.setItemData(
                self.combo.count() - 1, info.summary, Qt.ItemDataRole.ToolTipRole
            )
        self.summary = QLabel()
        self.summary.setObjectName("note")
        self.summary.setWordWrap(True)
        self.form = ParamForm()
        self.problem = QLabel()
        self.problem.setObjectName("errorText")
        self.problem.setWordWrap(True)
        self.problem.setVisible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.combo)
        layout.addWidget(self.summary)
        layout.addWidget(self.form)
        layout.addWidget(self.problem)
        self.combo.currentIndexChanged.connect(self._chosen)
        self.form.changed.connect(self._edited)
        self._chosen()

    def _info(self) -> Any:
        name = self.combo.currentData()
        return self.registry.info(name) if name in self.registry else None

    def _chosen(self, *_: object) -> None:
        info = self._info()
        self.form.set_type(None if info is None else info.params_type)
        self.summary.setText("" if info is None else info.summary)
        self._edited()

    def _edited(self) -> None:
        self.validate()
        self.changed.emit()

    def validate(self) -> str:
        """Check the parameters; returns the problem (empty if there is none) and shows it."""
        spec = self.spec()
        message = self.form.error
        if not message:
            try:
                self.registry.make_params(spec.name, spec.params)
            except CSRError as error:
                message = str(error)
        self.problem.setText(message)
        self.problem.setVisible(bool(message))
        return message

    def spec(self) -> PluginSpec:
        """The plug-in chosen with its parameters."""
        return PluginSpec(name=str(self.combo.currentData()), params=self.form.values())

    def set_spec(self, spec: PluginSpec) -> None:
        """Show a plug-in and its parameters without reporting a change."""
        self.blockSignals(True)
        name = spec.name
        if name in self.registry:
            name = self.registry.info(name).name
        index = self.combo.findData(name)
        if index < 0:  # a plug-in this installation does not have: keep it visible
            self.combo.addItem(tr("{name} (unknown)").format(name=name), normalize_name(name))
            index = self.combo.count() - 1
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(index)
        self.combo.blockSignals(False)
        info = self._info()
        self.form.set_type(None if info is None else info.params_type)
        self.summary.setText("" if info is None else info.summary)
        self.form.set_values(dict(spec.params))
        self.validate()
        self.blockSignals(False)


class PluginChecklist(QWidget):
    """Choose any number of a registry's plug-ins, each with its parameters."""

    changed = Signal()

    def __init__(self, registry: Registry[Any], parent: QWidget | None = None) -> None:
        """One check box and one parameter form per plug-in."""
        super().__init__(parent)
        self.registry = registry
        self.boxes: dict[str, QCheckBox] = {}
        self.forms: dict[str, ParamForm] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        for info in registry:
            box = QCheckBox(info.name)
            box.setToolTip(info.summary)
            form = ParamForm(info.params_type)
            form.setContentsMargins(22, 0, 0, 6)
            form.setEnabled(False)
            box.toggled.connect(form.setEnabled)
            box.toggled.connect(self._edited)
            form.changed.connect(self._edited)
            self.boxes[info.name] = box
            self.forms[info.name] = form
            layout.addWidget(box)
            layout.addWidget(form)
        self.problem = QLabel()
        self.problem.setObjectName("errorText")
        self.problem.setWordWrap(True)
        self.problem.setVisible(False)
        layout.addWidget(self.problem)

    def _edited(self, *_: object) -> None:
        self.validate()
        self.changed.emit()

    def validate(self) -> str:
        """Check every chosen plug-in's parameters; returns and shows the problems."""
        problems = []
        for spec in self.specs():
            error = self.forms[spec.name].error
            if not error:
                try:
                    self.registry.make_params(spec.name, spec.params)
                except CSRError as failure:
                    error = str(failure)
            if error:
                problems.append(error)
        message = "\n".join(problems)
        self.problem.setText(message)
        self.problem.setVisible(bool(message))
        return message

    def specs(self) -> tuple[PluginSpec, ...]:
        """The chosen plug-ins in the registry's order."""
        return tuple(
            PluginSpec(name=name, params=self.forms[name].values())
            for name, box in self.boxes.items()
            if box.isChecked()
        )

    def set_specs(self, specs: tuple[PluginSpec, ...]) -> None:
        """Show a selection without reporting a change."""
        self.blockSignals(True)
        chosen: dict[str, PluginSpec] = {}
        for spec in specs:
            if spec.name in self.registry:
                chosen.setdefault(self.registry.info(spec.name).name, spec)
        for name, box in self.boxes.items():
            box.blockSignals(True)
            box.setChecked(name in chosen)
            box.blockSignals(False)
            self.forms[name].setEnabled(name in chosen)
            self.forms[name].set_values(dict(chosen[name].params) if name in chosen else {})
        self.validate()
        self.blockSignals(False)
