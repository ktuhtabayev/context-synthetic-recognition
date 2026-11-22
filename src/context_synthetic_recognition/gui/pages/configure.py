"""Configure page: every setting of the method in the order of its stages, the presets.

Each edit builds a new (immutable) configuration from the widgets and hands it to the state's undo
stack; the widgets are then refreshed from the state, so undo, redo, the presets and an opened file
all take the same path. Plug-ins are chosen from their registries and their parameter forms are
generated (:mod:`..forms`), so a plug-in added later appears here without new code. Every setting
that can deviate from the article (:data:`~context_synthetic_recognition.config.presets.DEVIATIONS`)
is a switch with a "⚠ deviates from the article" badge.
"""

from __future__ import annotations

import re
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
    AdditionalFeaturesConfig,
    ExperimentConfig,
    LocalContextConfig,
    MetaAlgorithmConfig,
    OperatorConfig,
    SyntheticFeaturesConfig,
    preset,
)
from context_synthetic_recognition.config.models import ClassMeans, RIPSConfig
from context_synthetic_recognition.config.presets import (
    DEVIATIONS,
    PresetName,
    active_deviations,
    deviation,
    matching_preset,
)
from context_synthetic_recognition.core.additional_features.majorizing_functions import (
    MAJORIZING_FUNCTIONS,
)
from context_synthetic_recognition.core.local_context.admissible_k import K_STRATEGIES
from context_synthetic_recognition.core.local_context.metrics import METRICS
from context_synthetic_recognition.core.local_context.normalizers import NORMALIZERS
from context_synthetic_recognition.core.meta_algorithm.algorithm import DECISION_RULES
from context_synthetic_recognition.core.synthetic_features.contributions import INFORMATIVENESS
from context_synthetic_recognition.core.synthetic_features.encoders import ENCODERS
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
ORIGINAL_FEATURES = ("", "nominal")
"""The choices offered for the original features of Ψ(r): none, or all of set J."""
SAME = 1e-9
"""A spin box value this close to the configured one is the configured one."""
_SEPARATORS = re.compile(r"[,;]")
_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _real(spin: QDoubleSpinBox, current: float) -> float:
    """The spin box's value, or the configured value if the box only rounded it."""
    value = spin.value()
    return current if abs(value - current) < SAME else value


def _problems(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or 'value'}: {item['msg']}"
        for item in error.errors()
    )


def names(text: str) -> tuple[str, ...]:
    """The names in a text, separated by commas or semicolons (blanks dropped)."""
    return tuple(part.strip() for part in _SEPARATORS.split(text) if part.strip())


def feature_subset(text: str) -> str | tuple[str, ...]:
    """``all``, ``quantitative``, ``nominal`` or the feature names typed."""
    text = text.strip()
    return text if text in FEATURE_SETS else names(text)


def subset_text(value: str | tuple[str, ...]) -> str:
    """How a feature subset is shown: its keyword or its names separated by commas."""
    return value if isinstance(value, str) else ", ".join(value)


def object_ids(text: str, known: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The object ids typed; ``S1`` is read as ``S₁`` when that is how ``known`` names it."""
    found = []
    for name in names(text):
        written = name.translate(_SUBSCRIPT_DIGITS)
        found.append(written if name not in known and written in known else name)
    return tuple(found)


def _subset_combo(items: tuple[str, ...], tip: str, placeholder: str = "") -> QComboBox:
    """An editable combo box: a keyword of ``items`` or names typed with commas."""
    combo = QComboBox()
    combo.setEditable(True)
    combo.addItems(items)
    combo.setToolTip(tip)
    line = combo.lineEdit()
    assert line is not None
    line.setPlaceholderText(placeholder)
    return combo


def _show_text(combo: QComboBox, text: str) -> None:
    """Show ``text`` in an editable combo box without reporting a change."""
    combo.blockSignals(True)
    combo.setCurrentText(text)
    combo.blockSignals(False)


class OperatorRow(QWidget):
    """One base operator: its label, its metric, its feature subset and its normalization."""

    changed = Signal()
    removed = Signal(object)

    def __init__(self, operator: OperatorConfig, parent: QWidget | None = None) -> None:
        """Show ``operator``."""
        super().__init__(parent)
        self.label = QLineEdit()
        self.label.setMaximumWidth(90)
        self.label.setToolTip(tr("Display name, used in the names of the synthetic features"))
        self.metric_editor = PluginEditor(METRICS)
        self.features = _subset_combo(
            FEATURE_SETS,
            tr("all (I ∪ J), quantitative (I), nominal (J), or feature names separated by commas"),
        )
        self.normalizer_editor = PluginEditor(NORMALIZERS, unset=tr("as the local contexts"))
        self.normalizer_editor.setToolTip(
            tr("The operator's own normalization; by default that of the local contexts")
        )
        self.set_operator(operator)
        self.remove = QPushButton("✕")
        self.remove.setToolTip(tr("Remove this operator"))
        self.remove.setMaximumWidth(34)
        # the normalization on a line of its own, under the metric: the row stays as narrow
        caption = QLabel(tr("Normalizer"))
        caption.setObjectName("note")
        normalization = QHBoxLayout()
        normalization.addWidget(caption, 0, Qt.AlignmentFlag.AlignTop)
        normalization.addWidget(self.normalizer_editor, 1)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setVerticalSpacing(2)
        layout.addWidget(self.label, 0, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.metric_editor, 0, 1)
        layout.addWidget(self.features, 0, 2, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.remove, 0, 3, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(normalization, 1, 1, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        self.label.editingFinished.connect(self.changed)
        self.metric_editor.changed.connect(self.changed)
        self.normalizer_editor.changed.connect(self.changed)
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
        _show_text(self.features, subset_text(operator.features))
        self.normalizer_editor.set_optional_spec(operator.normalizer)

    def operator(self) -> OperatorConfig:
        """The operator as typed.

        Raises:
            ValidationError: Empty label, or an explicit subset that is empty or repeats a name.
        """
        features: Any = feature_subset(self.features.currentText())
        return OperatorConfig(
            label=self.label.text().strip(),
            metric=self.metric_editor.spec(),
            features=features,
            normalizer=self.normalizer_editor.optional_spec(),
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
        for text, stretch in ((tr("Label"), 0), (tr("Metric"), 1), (tr("Features"), 1)):
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
                "Every setting of the method, in the order of its stages. The default preset "
                "follows the article; Edit → Undo takes back any change."
            ),
            parent,
        )
        self._loading = False

        # ---- presets and warnings
        self.preset_badge = badge("")
        self.preset_choice = QComboBox()
        self.preset_choice.setToolTip(tr("A named set of settings of the method"))
        preset_tips = {
            PresetName.DEFAULT: tr("The article's method (the default)"),
            PresetName.ALPHA_01: tr("The default with α = 0.1, as in the article's experiments"),
            PresetName.ALPHA_02: tr("The default with α = 0.2, as in the article's experiments"),
            PresetName.EXCEL_EXPERIMENT: tr(
                "The settings of the Excel experiment: odd k = 3 … 2·min|Kᵢ| − 3, running class "
                "means, two passes of the majorizing function (⚠ deviates from the article)"
            ),
        }
        for name in PresetName:
            self.preset_choice.addItem(name.value, name.value)
            self.preset_choice.setItemData(
                self.preset_choice.count() - 1,
                preset_tips.get(name, ""),
                Qt.ItemDataRole.ToolTipRole,
            )
        self.apply_button = QPushButton(tr("Apply"))
        self.apply_button.setToolTip(
            tr("Set the method settings of the chosen preset; Edit → Undo takes them back")
        )
        self.apply_button.clicked.connect(self.apply_chosen_preset)
        self.header.addWidget(QLabel(tr("Matches preset:")))
        self.header.addWidget(self.preset_badge)
        self.header.addSpacing(16)
        self.header.addWidget(QLabel(tr("Preset")))
        self.header.addWidget(self.preset_choice)
        self.header.addWidget(self.apply_button)
        self.deviations = WarningBox()
        self.problems = WarningBox()
        self.body.addWidget(self.deviations)
        self.body.addWidget(self.problems)

        # ---- the switches: every setting that can deviate from the article
        self.rips_training = QComboBox()
        self.rips_training.addItem(tr("the objects that remain in E₁ (the article)"), "remaining")
        self.rips_training.addItem(tr("the objects moved to D"), "moved")
        self.keep_blocked_cover = QCheckBox(tr("keep covering other objects with |r(S)|"))
        self.class_means = QComboBox()
        self.class_means.addItem(
            tr("final class means M₁, M₂ (the article)"), ClassMeans.FINAL.value
        )
        self.class_means.addItem(tr("running partial class means"), ClassMeans.RUNNING.value)
        self.passes = QComboBox()
        self.passes.addItem(tr("1 pass (the article)"), 1)
        self.passes.addItem(tr("2 passes"), 2)
        for combo in (self.rips_training, self.class_means, self.passes):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(20)
        self.switches: dict[str, QWidget] = {
            "keep_blocked_cover": self.keep_blocked_cover,
            "training": self.rips_training,
            "class_means": self.class_means,
            "majorizing_passes": self.passes,
        }
        """The input of every deviation (:data:`DEVIATIONS`), by its key."""
        self.deviation_badges: dict[str, QLabel] = {
            found.key: badge(tr("⚠ deviates from the article"), warning=True)
            for found in DEVIATIONS
        }
        """The badge of every deviation, shown while its switch deviates from the article."""

        # ---- R-IPS
        self.rips_metric = PluginEditor(METRICS)
        self.rips_normalizer = PluginEditor(NORMALIZERS)
        self.rips_features = _subset_combo(
            FEATURE_SETS,
            tr(
                "The features ρ is applied to: all (I ∪ J), quantitative (I), nominal (J), or "
                "feature names separated by commas"
            ),
        )
        self.test_sample = QLineEdit()
        self.test_sample.setPlaceholderText(tr("none"))
        self.test_sample.setToolTip(
            tr("T: the ids of the objects held out before R-IPS, separated by commas (E₁ = E₀ ∖ T)")
        )
        self.rips_error = self._error_label()

        # ---- the local contexts
        self.normalizer = PluginEditor(NORMALIZERS)
        self.operators = OperatorsEditor()
        self.distance_decimals = QSpinBox()
        self.distance_decimals.setRange(0, 15)
        self.distance_decimals.setToolTip(
            tr("Distances are rounded to this many decimals so that equal distances tie exactly")
        )
        self.skip_empty = QCheckBox(tr("Skip operators whose feature subset is empty"))
        self.operator_error = self._error_label()
        self.admissible_k = PluginEditor(K_STRATEGIES)
        self.k_preview = hint("", "note")

        # ---- synthetic and additional features
        self.encoder = PluginEditor(ENCODERS)
        self.informativeness = PluginEditor(INFORMATIVENESS)
        self.skip_constant = QCheckBox(tr("Skip constant synthetic features"))
        self.skip_constant.setToolTip(
            tr("Drop features that take one value on every training object (off by default)")
        )
        self.zero_value = QCheckBox(
            tr("Zero value: a = 0 where the classes are indistinguishable (f_k(μ) = 0.5)")
        )
        self.zero_value.setToolTip(
            tr(
                "The values 0, 1, 2 of the article's experiments: every feature that takes 0 on a "
                "training object is left out. Off: formula (5) alone."
            )
        )
        self.original_features = _subset_combo(
            ORIGINAL_FEATURES,
            tr(
                "Original nominal features combined with the synthetic ones: nominal (all of J) "
                "or feature names separated by commas; empty: none"
            ),
            tr("none"),
        )
        self.alpha = self._spin(
            0.0001, 0.9999, 0.05, tr("α — margin regularization parameter, 0 < α < 1")
        )
        self.delta = self._spin(0.0001, 0.4999, 0.01, tr("δ — the grouping stops when θ/γ ≤ δ"))
        self.kappa = QSpinBox()
        self.kappa.setRange(2, 9999)
        self.kappa.setToolTip(tr("ϰ — maximum |TUPLAM|"))
        self.cr1 = self._spin(0.0001, 1e6, 1.0, tr("cr1 — a candidate is chosen only if θ/γ < cr1"))
        self.majorizing_function = PluginEditor(MAJORIZING_FUNCTIONS)

        # ---- meta-algorithm, evaluation, experiment
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

        # ---- the groups, in the order of the stages
        left = QVBoxLayout()
        right = QVBoxLayout()
        self.rips_group = self._group(
            tr("R-IPS · Precedent base for training and testing"),
            (
                None,
                hint(
                    tr(
                        "The training and the test sample of the protocols rips and "
                        "nested-cross-validation (Definition 3)."
                    ),
                    "note",
                ),
            ),
            (tr("Metric ρ"), self.rips_metric),
            (tr("Normalizer"), self.rips_normalizer),
            (tr("Features"), self.rips_features),
            (tr("Test sample T"), self.test_sample),
            (None, self.rips_error),
        )
        self._add_switches(
            self.rips_group,
            ("training", tr("Training sample")),
            ("keep_blocked_cover", tr("Blocked objects")),
        )
        left.addWidget(self.rips_group)
        left.addWidget(
            self._group(tr("Step 1 · Normalization"), (tr("Normalizer"), self.normalizer))
        )
        left.addWidget(
            self._group(
                tr("Steps 2–3 · Base operators (metric × feature subset)"),
                (None, self.operators),
                (None, self.operator_error),
                (tr("Distance decimals"), self.distance_decimals),
                (None, self.skip_empty),
            )
        )
        left.addWidget(
            self._group(tr("Admissible k"), (tr("Rule"), self.admissible_k), (None, self.k_preview))
        )
        left.addStretch(1)

        right.addWidget(
            self._group(
                tr("Steps 4–8 · Synthetic features Ψ(r)"),
                (tr("Encoder"), self.encoder),
                (tr("Informativeness ω"), self.informativeness),
                (None, self.skip_constant),
                (None, self.zero_value),
                (tr("Original features"), self.original_features),
            )
        )
        self.additional_group = self._group(
            tr("Step 9 · Additional features: hierarchical agglomerative algorithm"),
            ("α", self.alpha),
            ("δ", self.delta),
            ("ϰ", self.kappa),
            ("cr1", self.cr1),
            (tr("Majorizing function φ"), self.majorizing_function),
        )
        self._add_switches(
            self.additional_group,
            ("class_means", tr("Class means in θ, γ")),
            ("majorizing_passes", tr("Passes of the majorizing function")),
        )
        right.addWidget(self.additional_group)
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
            self.rips_metric,
            self.rips_normalizer,
            self.normalizer,
            self.admissible_k,
            self.encoder,
            self.informativeness,
            self.majorizing_function,
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
        for box in (self.keep_blocked_cover, self.skip_empty, self.skip_constant, self.zero_value):
            box.toggled.connect(self.commit)
        for combo in (
            self.rips_training,
            self.rips_features,
            self.original_features,
            self.class_means,
            self.passes,
            self.positive,
        ):
            combo.activated.connect(self.commit)
        for line in (
            self.name,
            self.test_sample,
            self.rips_features.lineEdit(),
            self.original_features.lineEdit(),
            self.positive.lineEdit(),
        ):
            assert line is not None
            line.editingFinished.connect(self.commit)

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
    def _error_label() -> QLabel:
        label = QLabel()
        label.setObjectName("errorText")
        label.setWordWrap(True)
        label.setVisible(False)
        return label

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

    def _add_switches(self, group: QGroupBox, *switches: tuple[str, str]) -> None:
        """Add switches (deviation key, title) to a group, each with its badge and statement."""
        form = group.layout()
        assert isinstance(form, QFormLayout)
        form.addRow(
            hint(
                tr("⚠ Switches: one value follows the article, the other deviates from it."), "note"
            )
        )
        for key, title in switches:
            found = deviation(key)
            row = QHBoxLayout()
            row.addWidget(self.switches[key], 1)
            row.addWidget(self.deviation_badges[key])
            form.addRow(title, row)
            form.addRow(hint(f"{found.adr}: {found.statement}.", "note"))

    # ------------------------------------------------------------------ state → widgets

    def refresh(self) -> None:
        """Show the state's configuration."""
        config = self.state.config
        self._loading = True
        try:
            rips = config.rips
            self.rips_metric.set_spec(rips.metric)
            self.rips_normalizer.set_spec(rips.normalizer)
            _show_text(self.rips_features, subset_text(rips.features))
            self.test_sample.setText(", ".join(rips.test_sample))
            self.rips_training.setCurrentIndex(self.rips_training.findData(rips.training))
            self.keep_blocked_cover.setChecked(rips.keep_blocked_cover)
            context = config.local_context
            self.normalizer.set_spec(context.normalizer)
            self.operators.set_operators(context.operators)
            self.distance_decimals.setValue(context.distance_decimals)
            self.skip_empty.setChecked(context.skip_empty_operators)
            self.admissible_k.set_spec(config.admissible_k)
            synthetic = config.synthetic_features
            self.encoder.set_spec(synthetic.encoder)
            self.informativeness.set_spec(synthetic.informativeness)
            self.skip_constant.setChecked(synthetic.skip_constant)
            self.zero_value.setChecked(synthetic.zero_value)
            _show_text(self.original_features, subset_text(synthetic.original_features))
            additional = config.additional_features
            self.alpha.setValue(additional.alpha)
            self.delta.setValue(additional.delta)
            self.kappa.setValue(additional.kappa)
            self.cr1.setValue(additional.cr1)
            self.majorizing_function.set_spec(additional.majorizing_function)
            self.class_means.setCurrentIndex(
                self.class_means.findData(ClassMeans(additional.class_means).value)
            )
            self.passes.setCurrentIndex(self.passes.findData(additional.majorizing_passes))
            self.decision_rule.set_spec(config.meta_algorithm.decision_rule)
            self.protocols.set_specs(config.evaluation.protocols)
            self.baselines.set_specs(config.evaluation.baselines)
            self._fill_positive(config)
            self.score_decimals.setValue(config.evaluation.score_decimals)
            self.name.setText(config.name)
            self.seed.setValue(config.seed)
        finally:
            self._loading = False
        self._hide_errors()
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
        if matched is not None:  # the choice follows the settings (an edit keeps the last one)
            self.preset_choice.setCurrentIndex(self.preset_choice.findData(matched.value))
        active = active_deviations(config)
        keys = {found.key for found in active}
        for key, label in self.deviation_badges.items():
            label.setVisible(key in keys)
        if active:
            lines = [tr("⚠ Settings that deviate from the article are active:")]
            lines += [f"• {d.statement} ({d.field}, {d.adr})" for d in active]
            lines.append(tr("The default preset follows the article."))
            self.deviations.set_text("\n".join(lines))
        else:
            self.deviations.set_text("")
        problems = self.config_problems()
        self.problems.set_text("\n".join(f"✗ {problem}" for problem in problems))
        self.k_preview.setText(self._k_text())

    def _k_text(self) -> str:
        dataset = self.state.dataset
        if dataset is None:
            return tr("Load a dataset to see the admissible k.")
        summary = summarize(dataset, self.state.config)
        ks = summary.admissible_k
        if ks is None or summary.r is None:
            return tr("No admissible k for {name}.").format(name=dataset.name)
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

    def _original_features(self) -> Any:
        text = self.original_features.currentText().strip()
        return text if text == "nominal" else names(text)

    def widgets_config(self) -> ExperimentConfig:
        """The configuration the widgets describe.

        Every section is built anew from its inputs, so it is validated (``model_copy`` alone
        would not check the values); the keys are the fields of :class:`ExperimentConfig`.

        Raises:
            ValidationError: A value the configuration does not accept (e.g. two operators with
                the same label, an object twice in the test sample).
        """
        base = self.state.config
        additional = base.additional_features
        dataset = self.state.dataset
        known = tuple(dataset.object_ids) if dataset is not None else ()
        rips_features: Any = feature_subset(self.rips_features.currentText()) or "all"
        return base.model_copy(
            update={
                "name": self.name.text().strip() or base.name,
                "seed": self.seed.value(),
                "rips": RIPSConfig(
                    metric=self.rips_metric.spec(),
                    normalizer=self.rips_normalizer.spec(),
                    features=rips_features,
                    training=self.rips_training.currentData(),
                    keep_blocked_cover=self.keep_blocked_cover.isChecked(),
                    test_sample=object_ids(self.test_sample.text(), known),
                    distance_decimals=base.rips.distance_decimals,
                ),
                "local_context": LocalContextConfig(
                    normalizer=self.normalizer.spec(),
                    operators=self.operators.operators(),
                    distance_decimals=self.distance_decimals.value(),
                    tie_break=base.local_context.tie_break,
                    skip_empty_operators=self.skip_empty.isChecked(),
                ),
                "admissible_k": self.admissible_k.spec(),
                "synthetic_features": SyntheticFeaturesConfig(
                    encoder=self.encoder.spec(),
                    informativeness=self.informativeness.spec(),
                    skip_constant=self.skip_constant.isChecked(),
                    original_features=self._original_features(),
                    zero_value=self.zero_value.isChecked(),
                ),
                "additional_features": AdditionalFeaturesConfig(
                    alpha=_real(self.alpha, additional.alpha),
                    delta=_real(self.delta, additional.delta),
                    kappa=self.kappa.value(),
                    cr1=_real(self.cr1, additional.cr1),
                    majorizing_function=self.majorizing_function.spec(),
                    class_means=ClassMeans(self.class_means.currentData()),
                    majorizing_passes=self.passes.currentData(),
                ),
                "meta_algorithm": MetaAlgorithmConfig(decision_rule=self.decision_rule.spec()),
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

    def _hide_errors(self) -> None:
        self.rips_error.setVisible(False)
        self.operator_error.setVisible(False)
        set_invalid(self.test_sample, False)

    def _show_error(self, error: ValidationError) -> None:
        """Show why the inputs do not make a configuration, next to the section at fault."""
        self._hide_errors()
        rips = error.title == RIPSConfig.__name__
        shown = self.rips_error if rips else self.operator_error
        shown.setText(_problems(error))
        shown.setVisible(True)
        set_invalid(self.test_sample, rips)
        for row in self.operators.rows:
            set_invalid(row.label, not rips and not row.label.text().strip())

    def commit(self, *_: object) -> None:
        """An input changed: hand the new configuration to the state (undoable)."""
        if self._loading:
            return
        try:
            config = self.widgets_config()
        except ValidationError as error:
            self._show_error(error)
            return
        self._hide_errors()
        if config == self.state.config:
            self._refresh_status()
            return
        self.state.set_config(config, tr("Edit the configuration"))

    def apply_preset(self, name: str) -> None:
        """Set the method settings of a preset (undoable); data, name, seed and output stay."""
        current = self.state.config
        chosen = preset(name).model_copy(
            update={"name": current.name, "seed": current.seed, "output": current.output}
        )
        self.state.set_config(chosen, tr("Preset: {name}").format(name=name))

    def apply_chosen_preset(self) -> None:
        """Apply the preset chosen in the header."""
        self.apply_preset(str(self.preset_choice.currentData()))
