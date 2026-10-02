"""Configure page: every setting of the method, the two switches, the presets.

Each edit builds a new (immutable) configuration from the widgets and hands it to the state's undo
stack; the widgets are then refreshed from the state, so undo, redo, the presets and an opened file
all take the same path. Plug-ins are chosen from their registries and their parameter forms are
generated (:mod:`..forms`), so a plug-in added later appears here without new code.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from context_synthetic_recognition.config import (
    ContextConfig,
    ExperimentConfig,
    HAGConfig,
    MetaConfig,
    OperatorConfig,
    PreprocessingConfig,
    SyntheticConfig,
    preset,
)
from context_synthetic_recognition.config.models import CentreMode
from context_synthetic_recognition.config.presets import (
    DEVIATIONS,
    active_deviations,
    matching_preset,
)
from context_synthetic_recognition.core.contributions import WEIGHTS
from context_synthetic_recognition.core.encoders import ENCODERS
from context_synthetic_recognition.core.k_strategies import K_STRATEGIES
from context_synthetic_recognition.core.majorizers import MAJORIZERS
from context_synthetic_recognition.core.meta import DECISION_RULES
from context_synthetic_recognition.core.metrics import METRICS
from context_synthetic_recognition.core.normalizers import NORMALIZERS
from context_synthetic_recognition.evaluation.baselines import BASELINES
from context_synthetic_recognition.evaluation.protocols import PROTOCOLS
from context_synthetic_recognition.gui.forms import PluginChecklist, PluginEditor, set_invalid
from context_synthetic_recognition.gui.i18n import mark, tr
from context_synthetic_recognition.gui.pages.base import Page
from context_synthetic_recognition.gui.settings import Settings
from context_synthetic_recognition.gui.state import AppState
from context_synthetic_recognition.gui.widgets import WarningBox, badge, hint
from context_synthetic_recognition.services.configs import config_problems
from context_synthetic_recognition.services.datasets import summarize

FEATURE_SETS = ("all", "quantitative", "nominal")
SAME = 1e-9
"""A spin box value this close to the configured one is the configured one."""


def _real(spin: QDoubleSpinBox, current: float) -> float:
    """The spin box's value, or the configured value if the box only rounded it."""
    value = spin.value()
    return current if abs(value - current) < SAME else value


def _problems(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or 'value'}: {item['msg']}"
        for item in error.errors()
    )


class OperatorRow(QWidget):
    """One base operator: its label, its metric and its feature subset."""

    changed = Signal()
    removed = Signal(object)

    def __init__(self, operator: OperatorConfig, parent: QWidget | None = None) -> None:
        """Show ``operator``."""
        super().__init__(parent)
        self.label = QLineEdit()
        self.label.setMaximumWidth(90)
        self.label.setToolTip(tr("Display name, used in the names of the synthetic features"))
        self.metric_editor = PluginEditor(METRICS)
        self.features = QComboBox()
        self.features.setEditable(True)
        self.features.addItems(FEATURE_SETS)
        self.features.setToolTip(
            tr("all (I ∪ J), quantitative (I), nominal (J), or feature names separated by commas")
        )
        self.set_operator(operator)
        self.remove = QPushButton("✕")
        self.remove.setToolTip(tr("Remove this operator"))
        self.remove.setMaximumWidth(34)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.metric_editor, 2)
        layout.addWidget(self.features, 2, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.remove, 0, Qt.AlignmentFlag.AlignTop)
        self.label.editingFinished.connect(self.changed)
        self.metric_editor.changed.connect(self.changed)
        self.features.activated.connect(self.changed)
        line = self.features.lineEdit()
        assert line is not None
        line.editingFinished.connect(self.changed)
        self.remove.clicked.connect(lambda: self.removed.emit(self))

    def set_operator(self, operator: OperatorConfig) -> None:
        """Show ``operator`` without reporting a change."""
        self.label.setText(operator.label)
        set_invalid(self.label, False)
        self.metric_editor.set_spec(operator.metric)
        text = (
            operator.features
            if isinstance(operator.features, str)
            else ", ".join(operator.features)
        )
        self.features.blockSignals(True)
        self.features.setCurrentText(text)
        self.features.blockSignals(False)

    def operator(self) -> OperatorConfig:
        """The operator as typed.

        Raises:
            ValidationError: Empty label, or an explicit subset that is empty or repeats a name.
        """
        text = self.features.currentText().strip()
        features: Any = text
        if text not in FEATURE_SETS:
            features = tuple(part.strip() for part in text.split(",") if part.strip())
        return OperatorConfig(
            label=self.label.text().strip(), metric=self.metric_editor.spec(), features=features
        )


class OperatorsEditor(QWidget):
    """The list of base operators."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """An empty list with an "Add operator" button."""
        super().__init__(parent)
        self.rows: list[OperatorRow] = []
        self._rows = QVBoxLayout()
        self._rows.setSpacing(6)
        self.add_button = QPushButton(tr("Add operator"))
        self.add_button.clicked.connect(self.add)
        header = QHBoxLayout()
        for text, stretch in ((tr("Label"), 0), (tr("Metric"), 2), (tr("Features"), 2)):
            label = hint(text, "note")
            if not stretch:
                label.setFixedWidth(90)
            header.addWidget(label, stretch)
        header.addSpacing(40)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(header)
        layout.addLayout(self._rows)
        layout.addWidget(self.add_button, 0, Qt.AlignmentFlag.AlignLeft)

    def set_operators(self, operators: tuple[OperatorConfig, ...]) -> None:
        """Show ``operators`` without reporting a change."""
        if len(operators) == len(self.rows):
            # the same rows with new content: whatever is being edited keeps the focus
            for row, operator in zip(self.rows, operators, strict=True):
                row.set_operator(operator)
            return
        for row in self.rows:
            self._rows.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self.rows = []
        for operator in operators:
            self._append(operator)

    def _append(self, operator: OperatorConfig) -> OperatorRow:
        row = OperatorRow(operator)
        row.changed.connect(self.changed)
        row.removed.connect(self._remove)
        self.rows.append(row)
        self._rows.addWidget(row)
        return row

    def add(self) -> None:
        """Add an operator on all features with a label that is not taken."""
        taken = {row.label.text().strip() for row in self.rows}
        number = len(self.rows) + 1
        while f"ρ{number}" in taken:
            number += 1
        self._append(OperatorConfig(label=f"ρ{number}"))
        self.changed.emit()

    def _remove(self, row: OperatorRow) -> None:
        if len(self.rows) > 1:  # Ψ needs at least one operator
            self.rows.remove(row)
            self._rows.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
            self.changed.emit()

    def operators(self) -> tuple[OperatorConfig, ...]:
        """The operators as typed (see :meth:`OperatorRow.operator`)."""
        return tuple(row.operator() for row in self.rows)


class ConfigurePage(Page):
    """Edit the configuration."""

    title = mark("Configure")

    def __init__(self, state: AppState, settings: Settings, parent: QWidget | None = None) -> None:
        """Build the page."""
        super().__init__(
            state,
            settings,
            tr(
                "Every setting of the method. The defaults are the template preset, which "
                "reproduces the Excel experiment; Edit → Undo takes back any change."
            ),
            parent,
        )
        self._loading = False

        # ---- presets and warnings
        self.preset_badge = badge("")
        self.template_button = QPushButton(tr("Template preset"))
        self.template_button.setToolTip(tr("Running class centres, two majorizer passes (default)"))
        self.article_button = QPushButton(tr("Article preset"))
        self.article_button.setToolTip(tr("Final class centres, one majorizer pass"))
        self.template_button.clicked.connect(lambda: self.apply_preset("template"))
        self.article_button.clicked.connect(lambda: self.apply_preset("article"))
        self.header.addWidget(QLabel(tr("Matches preset:")))
        self.header.addWidget(self.preset_badge)
        self.header.addWidget(self.template_button)
        self.header.addWidget(self.article_button)
        self.deviations = WarningBox()
        self.problems = WarningBox()
        self.body.addWidget(self.deviations)
        self.body.addWidget(self.problems)

        # ---- the groups
        self.normalizer = PluginEditor(NORMALIZERS)
        self.operators = OperatorsEditor()
        self.distance_decimals = QSpinBox()
        self.distance_decimals.setRange(0, 15)
        self.distance_decimals.setToolTip(
            tr("Distances are rounded to this many decimals so that equal distances tie exactly")
        )
        self.skip_empty = QCheckBox(tr("Skip operators whose feature subset is empty"))
        self.k = PluginEditor(K_STRATEGIES)
        self.k_preview = hint("", "note")
        self.encoder = PluginEditor(ENCODERS)
        self.weights = PluginEditor(WEIGHTS)
        self.skip_constant = QCheckBox(tr("Skip constant synthetic features"))
        self.skip_constant.setToolTip(
            tr("Drop features that take one value on every training object (off: as the workbook)")
        )
        self.alpha = self._spin(0.0001, 0.9999, 0.05, tr("α — regularisation, 0 < α < 1"))
        self.delta = self._spin(0.0001, 0.4999, 0.01, tr("δ — the grouping stops when θ/γ ≤ δ"))
        self.kappa = QSpinBox()
        self.kappa.setRange(2, 9999)
        self.kappa.setToolTip(tr("ϰ — maximum |TUPLAM|"))
        self.cr1 = self._spin(0.0001, 1e6, 1.0, tr("cr1 — a candidate is chosen only if θ/γ < cr1"))
        self.majorizer = PluginEditor(MAJORIZERS)
        self.centres = QComboBox()
        self.centres.addItem(tr("running partial means (template)"), CentreMode.RUNNING.value)
        self.centres.addItem(tr("final class means (article)"), CentreMode.FINAL.value)
        self.passes = QComboBox()
        self.passes.addItem(tr("2 passes (template)"), 2)
        self.passes.addItem(tr("1 pass (article)"), 1)
        for combo in (self.centres, self.passes):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(20)
        self.centres_badge = badge(tr("⚠ differs from the article"), warning=True)
        self.passes_badge = badge(tr("⚠ differs from the article"), warning=True)
        self.decision_rule = PluginEditor(DECISION_RULES)
        self.protocols = PluginChecklist(PROTOCOLS)
        self.baselines = PluginChecklist(BASELINES)
        self.positive = QComboBox()
        self.positive.setEditable(True)
        self.positive.setToolTip(tr("The positive class of precision, recall and ROC"))
        self.score_decimals = QSpinBox()
        self.score_decimals.setRange(0, 15)
        self.name = QLineEdit()
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setToolTip(tr("Seed of the splits of the randomised protocols"))
        self.operator_error = QLabel()
        self.operator_error.setObjectName("errorText")
        self.operator_error.setWordWrap(True)
        self.operator_error.setVisible(False)

        left = QVBoxLayout()
        right = QVBoxLayout()
        left.addWidget(
            self._group(tr("Step 1 · Scale unification"), (tr("Normalizer"), self.normalizer))
        )
        operators_group = self._group(
            tr("Steps 2–3 · Base operators (metric × feature subset)"),
            (None, self.operators),
            (None, self.operator_error),
            (tr("Distance decimals"), self.distance_decimals),
            (None, self.skip_empty),
        )
        left.addWidget(operators_group)
        left.addWidget(self._group(tr("Permitted k"), (tr("Rule"), self.k), (None, self.k_preview)))
        left.addWidget(
            self._group(
                tr("Steps 4–8 · Synthetic features Ψ(r)"),
                (tr("Encoder"), self.encoder),
                (tr("Weights"), self.weights),
                (None, self.skip_constant),
            )
        )
        left.addStretch(1)

        self.hag_group = self._group(
            tr("Step 9 · Hierarchical agglomerative grouping"),
            ("α", self.alpha),
            ("δ", self.delta),
            ("ϰ", self.kappa),
            ("cr1", self.cr1),
            (tr("Majorizer ϕ"), self.majorizer),
        )
        switches = QFormLayout()
        switches.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        switches.addRow(hint(tr("⚠ The two calculations that differ from the article:"), "note"))
        for deviation, combo, differs in (
            (DEVIATIONS[0], self.centres, self.centres_badge),
            (DEVIATIONS[1], self.passes, self.passes_badge),
        ):
            row = QHBoxLayout()
            row.addWidget(combo, 1)
            row.addWidget(differs)
            title = tr("Class centres in θ, γ") if combo is self.centres else tr("STEP 4 passes")
            switches.addRow(title, row)
            switches.addRow(hint(f"{deviation.adr}: {deviation.statement}.", "note"))
        hag_layout = self.hag_group.layout()
        assert isinstance(hag_layout, QFormLayout)
        hag_layout.addRow(switches)
        right.addWidget(self.hag_group)
        right.addWidget(
            self._group(
                tr("Steps 10–12 · Meta-algorithm"), (tr("Decision rule"), self.decision_rule)
            )
        )
        right.addWidget(
            self._group(
                tr("Evaluation"),
                (tr("Protocols"), self.protocols),
                (tr("Baselines"), self.baselines),
                (tr("Positive class"), self.positive),
                (tr("Score decimals"), self.score_decimals),
            )
        )
        right.addWidget(
            self._group(tr("Experiment"), (tr("Name"), self.name), (tr("Seed"), self.seed))
        )
        right.addStretch(1)

        grid = QGridLayout()
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.addLayout(left, 0, 0)
        grid.addLayout(right, 0, 1)
        content = QWidget()
        content.setLayout(grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self.scroll_area = scroll
        self.body.addWidget(scroll, 1)

        # ---- edits
        for editor in (
            self.normalizer,
            self.k,
            self.encoder,
            self.weights,
            self.majorizer,
            self.decision_rule,
            self.protocols,
            self.baselines,
            self.operators,
        ):
            editor.changed.connect(self.commit)
        for spin in (self.distance_decimals, self.kappa, self.score_decimals, self.seed):
            spin.valueChanged.connect(self.commit)
        for real in (self.alpha, self.delta, self.cr1):
            real.valueChanged.connect(self.commit)
        for box in (self.skip_empty, self.skip_constant):
            box.toggled.connect(self.commit)
        for combo in (self.centres, self.passes, self.positive):
            combo.activated.connect(self.commit)
        self.name.editingFinished.connect(self.commit)
        positive_line = self.positive.lineEdit()
        assert positive_line is not None
        positive_line.editingFinished.connect(self.commit)

        state.configChanged.connect(self.refresh)
        state.datasetChanged.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------ building blocks

    @staticmethod
    def _spin(low: float, high: float, step: float, tip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(low, high)
        spin.setSingleStep(step)
        spin.setToolTip(tip)
        return spin

    @staticmethod
    def _group(title: str, *rows: tuple[str | None, QWidget]) -> QGroupBox:
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for label, widget in rows:
            if label is None:
                form.addRow(widget)
            else:
                form.addRow(label, widget)
        return group

    # ------------------------------------------------------------------ state → widgets

    def refresh(self) -> None:
        """Show the state's configuration."""
        config = self.state.config
        self._loading = True
        try:
            self.normalizer.set_spec(config.preprocessing.normalizer)
            self.operators.set_operators(config.context.operators)
            self.distance_decimals.setValue(config.context.distance_decimals)
            self.skip_empty.setChecked(config.context.skip_empty_operators)
            self.k.set_spec(config.k)
            self.encoder.set_spec(config.synthetic.encoder)
            self.weights.set_spec(config.synthetic.weights)
            self.skip_constant.setChecked(config.synthetic.skip_constant)
            self.alpha.setValue(config.hag.alpha)
            self.delta.setValue(config.hag.delta)
            self.kappa.setValue(config.hag.kappa)
            self.cr1.setValue(config.hag.cr1)
            self.majorizer.set_spec(config.hag.majorizer)
            self.centres.setCurrentIndex(self.centres.findData(config.hag.centres.value))
            self.passes.setCurrentIndex(self.passes.findData(config.hag.step4_passes))
            self.decision_rule.set_spec(config.meta.decision_rule)
            self.protocols.set_specs(config.evaluation.protocols)
            self.baselines.set_specs(config.evaluation.baselines)
            self._fill_positive(config)
            self.score_decimals.setValue(config.evaluation.score_decimals)
            self.name.setText(config.name)
            self.seed.setValue(config.seed)
        finally:
            self._loading = False
        self.operator_error.setVisible(False)
        self._refresh_status()

    def _fill_positive(self, config: ExperimentConfig) -> None:
        self.positive.clear()
        dataset = self.state.dataset
        labels = list(dataset.classes) if dataset is not None else []
        current = config.evaluation.positive_class
        if current not in labels and str(current) not in map(str, labels):
            labels.append(current)
        for label in labels:
            self.positive.addItem(str(label), label)
        index = next((i for i, label in enumerate(labels) if str(label) == str(current)), 0)
        self.positive.setCurrentIndex(index)

    def _refresh_status(self) -> None:
        config = self.state.config
        matched = matching_preset(config)
        self.preset_badge.setText(matched.value if matched else tr("custom"))
        active = active_deviations(config)
        self.centres_badge.setVisible(any(d.key == "centres" for d in active))
        self.passes_badge.setVisible(any(d.key == "step4_passes" for d in active))
        if active:
            lines = [tr("⚠ Template calculations that differ from the article are active:")]
            lines += [f"• {d.statement} ({d.field}, {d.adr})" for d in active]
            lines.append(
                tr(
                    "They reproduce the template workbooks exactly; the Article preset follows "
                    "the article."
                )
            )
            self.deviations.set_text("\n".join(lines))
        else:
            self.deviations.set_text(
                tr(
                    "⚠ This configuration follows the article for both switches (final class "
                    "centres, one majorizer pass); it does not reproduce the template workbooks."
                )
            )
        problems = self.config_problems()
        self.problems.set_text("\n".join(f"✗ {problem}" for problem in problems))
        self.k_preview.setText(self._k_text())

    def _k_text(self) -> str:
        dataset = self.state.dataset
        if dataset is None:
            return tr("Load a dataset to see the permitted k.")
        summary = summarize(dataset, self.state.config)
        ks = summary.permitted_k
        if ks is None or summary.r is None:
            return tr("No permitted k for {name}.").format(name=dataset.name)
        shown = (
            ks.label if ks.count <= 14 else f"{ks.ks[0]}, {ks.ks[1]}, {ks.ks[2]}, …, {ks.ks[-1]}"
        )
        return tr(
            "{name}: k = {ks}  →  r = {r} synthetic features ({operators} operator(s) × {count} k)"
        ).format(
            name=dataset.name,
            ks=shown,
            r=summary.r,
            operators=len(summary.operators),
            count=ks.count,
        )

    def config_problems(self) -> list[str]:
        """What keeps the configuration from running: plug-in parameters, the data, the forms."""
        config = self.state.config
        problems = config_problems(config)
        dataset = self.state.dataset
        if dataset is not None:
            problems += list(summarize(dataset, config).problems)
        if not config.evaluation.protocols:
            problems.append(tr("evaluation: choose at least one protocol"))
        return list(dict.fromkeys(problems))

    # ------------------------------------------------------------------ widgets → state

    def _positive_value(self) -> int | str:
        text = self.positive.currentText().strip()
        for index in range(self.positive.count()):
            if self.positive.itemText(index) == text:
                value = self.positive.itemData(index)
                return value if isinstance(value, int | str) else text
        return int(text) if text.lstrip("-").isdigit() else text

    def widgets_config(self) -> ExperimentConfig:
        """The configuration the widgets describe.

        Raises:
            ValidationError: A value the configuration does not accept (e.g. two operators with
                the same label).
        """
        base = self.state.config
        hag = base.hag
        return base.model_copy(
            update={
                "name": self.name.text().strip() or base.name,
                "seed": self.seed.value(),
                "preprocessing": PreprocessingConfig(normalizer=self.normalizer.spec()),
                "context": ContextConfig(
                    operators=self.operators.operators(),
                    distance_decimals=self.distance_decimals.value(),
                    tie_break=base.context.tie_break,
                    skip_empty_operators=self.skip_empty.isChecked(),
                ),
                "k": self.k.spec(),
                "synthetic": SyntheticConfig(
                    encoder=self.encoder.spec(),
                    weights=self.weights.spec(),
                    skip_constant=self.skip_constant.isChecked(),
                ),
                "hag": HAGConfig(
                    alpha=_real(self.alpha, hag.alpha),
                    delta=_real(self.delta, hag.delta),
                    kappa=self.kappa.value(),
                    cr1=_real(self.cr1, hag.cr1),
                    majorizer=self.majorizer.spec(),
                    centres=CentreMode(self.centres.currentData()),
                    step4_passes=self.passes.currentData(),
                ),
                "meta": MetaConfig(decision_rule=self.decision_rule.spec()),
                "evaluation": base.evaluation.model_copy(
                    update={
                        "protocols": self.protocols.specs(),
                        "baselines": self.baselines.specs(),
                        "positive_class": self._positive_value(),
                        "score_decimals": self.score_decimals.value(),
                    }
                ),
            }
        )

    def commit(self, *_: object) -> None:
        """An input changed: hand the new configuration to the state (undoable)."""
        if self._loading:
            return
        try:
            config = self.widgets_config()
        except ValidationError as error:
            self.operator_error.setText(_problems(error))
            self.operator_error.setVisible(True)
            for row in self.operators.rows:
                set_invalid(row.label, not row.label.text().strip())
            return
        self.operator_error.setVisible(False)
        if config == self.state.config:
            self._refresh_status()
            return
        self.state.set_config(config, tr("Edit the configuration"))

    def apply_preset(self, name: str) -> None:
        """Set the method settings of a preset; data, name, seed and output stay."""
        current = self.state.config
        chosen = preset(name).model_copy(
            update={"name": current.name, "seed": current.seed, "output": current.output}
        )
        self.state.set_config(chosen, tr("Preset: {name}").format(name=name))
