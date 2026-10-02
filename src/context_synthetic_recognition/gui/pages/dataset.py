"""Dataset page: choose or open a dataset, look at it, set the feature types.

The summary on the right is live: |K1|, |K2|, the permitted k and r = |Ψ(r)| follow the dataset
and the configuration (:func:`~context_synthetic_recognition.services.datasets.summarize`).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
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

from context_synthetic_recognition.config import DatasetConfig
from context_synthetic_recognition.data import DATASETS, LOADERS, Dataset, load_dataset
from context_synthetic_recognition.data.catalog import project_catalog, resolve_dataset
from context_synthetic_recognition.data.schema import FeatureType
from context_synthetic_recognition.errors import CSRError
from context_synthetic_recognition.export.text import format_cell
from context_synthetic_recognition.gui.i18n import mark, tr
from context_synthetic_recognition.gui.models import dataset_table, feature_rows, style_for
from context_synthetic_recognition.gui.pages.base import Page
from context_synthetic_recognition.gui.settings import RECENT_DATASETS, Settings
from context_synthetic_recognition.gui.state import AppState
from context_synthetic_recognition.gui.theme import CellRole, Palette
from context_synthetic_recognition.gui.widgets import TablePanel, WarningBox, card, hint
from context_synthetic_recognition.services.datasets import summarize

PREVIEW_ROWS = 2000
"""Objects shown in the preview grid (the statistics always cover every object)."""
DATASET_FILES = mark("Datasets (*.dat *.csv *.xlsx *.xlsm *.txt *.parquet);;All files (*)")
TYPE_COLUMN = 1


class ClassBalance(QWidget):
    """The two classes as one bar: K1 and K2 in their colours, sized by their share."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """An empty bar."""
        super().__init__(parent)
        self.sizes: tuple[int, ...] = ()
        self.colours: tuple[str, str] = ("#2a78d6", "#eb6834")
        self.surface = "#ffffff"
        self.setMinimumHeight(14)
        self.setMaximumHeight(14)

    def sizeHint(self) -> QSize:
        """A thin bar as wide as its container."""
        return QSize(200, 14)

    def set_sizes(self, sizes: tuple[int, ...]) -> None:
        """Show the class sizes (|K1|, |K2|)."""
        self.sizes = sizes
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Draw the two segments with a small gap between them."""
        total = sum(self.sizes)
        if len(self.sizes) != 2 or total == 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        width, height, gap = float(self.width()), float(self.height()), 2.0
        first = (width - gap) * self.sizes[0] / total
        painter.setBrush(QColor(self.colours[0]))
        painter.drawRoundedRect(QRectF(0, 0, first, height), 3, 3)
        painter.setBrush(QColor(self.colours[1]))
        painter.drawRoundedRect(QRectF(first + gap, 0, width - first - gap, height), 3, 3)
        painter.end()


class DatasetPage(Page):
    """Choose a dataset, preview it, edit the feature types."""

    title = mark("Dataset")

    def __init__(self, state: AppState, settings: Settings, parent: QWidget | None = None) -> None:
        """Build the page."""
        super().__init__(
            state,
            settings,
            tr(
                "Choose a dataset of the project, open a file, or drop one onto the window. "
                "Objects are rows, features are columns of type I (quantitative) or J (nominal)."
            ),
            parent,
        )
        self.file_dialog = self._ask_file

        # ---- source
        self.source = QComboBox()
        self.source.setMinimumWidth(320)
        self._fill_sources()
        self.source.activated.connect(self._source_chosen)
        self.open_button = QPushButton(tr("Open file…"))
        self.open_button.clicked.connect(self.open_file)
        self.location = hint("", "note")
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("Dataset")))
        row.addWidget(self.source)
        row.addWidget(self.open_button)
        row.addWidget(self.location, 1)
        self.body.addLayout(row)

        # ---- reading options (tabular files)
        self.options = QGroupBox(tr("Reading options for tables (CSV, Excel, Parquet)"))
        self.options.setCheckable(True)
        self.options.setChecked(False)
        form = QGridLayout(self.options)
        self.format = QComboBox()
        self.format.addItems(["auto", *LOADERS.names()])
        self.sheet = QLineEdit()
        self.sheet.setPlaceholderText(tr("first sheet"))
        self.class_column = QLineEdit()
        self.class_column.setPlaceholderText(tr("last column"))
        self.id_column = QLineEdit()
        self.id_column.setPlaceholderText(tr("none"))
        self.reload_button = QPushButton(tr("Read again"))
        self.reload_button.clicked.connect(self.reload)
        for position, (label, widget) in enumerate(
            (
                (tr("Format"), self.format),
                (tr("Sheet"), self.sheet),
                (tr("Class column"), self.class_column),
                (tr("Object-id column"), self.id_column),
            )
        ):
            line, side = divmod(position, 2)
            form.addWidget(QLabel(label), line, 2 * side)
            form.addWidget(widget, line, 2 * side + 1)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)
        form.addWidget(self.reload_button, 1, 4)
        self._option_widgets = (self.format, self.sheet, self.class_column, self.id_column)
        self.options.toggled.connect(self._options_toggled)
        self._options_toggled(False)
        self.body.addWidget(self.options)

        self.error = QLabel()
        self.error.setObjectName("errorText")
        self.error.setWordWrap(True)
        self.error.setVisible(False)
        self.body.addWidget(self.error)

        # ---- preview and features | summary
        self.preview = TablePanel()
        self.features = QTableWidget(0, 5)
        self.features.setHorizontalHeaderLabels(
            [tr("Feature"), tr("Type"), tr("min"), tr("max"), tr("Distinct values")]
        )
        self.features.verticalHeader().setVisible(False)
        self.features.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.features.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        features_tab = QWidget()
        features_layout = QVBoxLayout(features_tab)
        features_layout.setContentsMargins(8, 8, 8, 8)
        features_layout.addWidget(
            hint(
                tr(
                    "The type decides how a feature enters the distance: |x′ − y′| on the unified "
                    "scale for I, equal / not equal for J. Changing a type re-reads the dataset "
                    "and drops the current result."
                ),
                "note",
            )
        )
        features_layout.addWidget(self.features, 1)
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.addWidget(self.preview)
        self.tabs = QTabWidget()
        self.tabs.addTab(preview_tab, tr("Preview"))
        self.tabs.addTab(features_tab, tr("Features and types"))

        summary, summary_layout = card()
        summary.setMinimumWidth(290)
        summary.setMaximumWidth(420)
        self.facts = QFormLayout()
        self.facts.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._fact_labels: dict[str, QLabel] = {}
        for key, label in (
            ("name", tr("Name")),
            ("objects", tr("Objects m")),
            ("features", tr("Features n")),
            ("k1", "|K1|"),
            ("k2", "|K2|"),
            ("missing", tr("Missing values")),
            ("k", tr("Permitted k")),
            ("r", tr("Synthetic features r")),
            ("operators", tr("Base operators")),
        ):
            value = QLabel("—")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._fact_labels[key] = value
            self.facts.addRow(label, value)
        self.balance = ClassBalance()
        self.problems = WarningBox()
        summary_layout.addWidget(hint(tr("Summary — live for the current configuration"), "note"))
        summary_layout.addLayout(self.facts)
        summary_layout.addWidget(hint(tr("Class balance"), "note"))
        summary_layout.addWidget(self.balance)
        summary_layout.addWidget(self.problems)
        summary_layout.addStretch(1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.tabs)
        splitter.addWidget(summary)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        self.body.addWidget(splitter, 1)

        state.datasetChanged.connect(self.refresh)
        state.configChanged.connect(self.refresh_summary)
        self.apply_palette(self.colours)
        self.refresh()

    # ------------------------------------------------------------------ sources

    def _fill_sources(self) -> None:
        self.source.clear()
        self.source.addItem(tr("Choose a dataset…"), None)
        catalog = project_catalog()
        if catalog is not None:
            for entry in catalog:
                shape = "" if entry.shape is None else f"  ·  {entry.shape[0]} × {entry.shape[1]}"
                self.source.addItem(f"{entry.name}{shape}", entry.id)
                self.source.setItemData(
                    self.source.count() - 1, str(entry.path), Qt.ItemDataRole.ToolTipRole
                )
        for info in DATASETS:
            self.source.addItem(tr("{name}  ·  built in").format(name=info.summary), info.name)

    def _source_chosen(self, index: int) -> None:
        key = self.source.itemData(index)
        if key is not None:
            self.load(str(key))

    def _options_toggled(self, on: bool) -> None:
        for widget in self._option_widgets:
            widget.setEnabled(on)
        self.reload_button.setEnabled(on)

    def _ask_file(self) -> str:
        start = self.settings.recent(RECENT_DATASETS)
        folder = str(Path(start[0]).parent) if start else ""
        path, _ = QFileDialog.getOpenFileName(self, tr("Open a dataset"), folder, tr(DATASET_FILES))
        return path

    def open_file(self) -> None:
        """Ask for a dataset file and load it."""
        path = self.file_dialog()
        if path:
            self.load(path)

    # ------------------------------------------------------------------ loading

    def _section(self, source: str, types: dict[str, str] | None = None) -> DatasetConfig:
        """How the dataset is read, as the configuration's ``dataset`` section."""
        custom = self.options.isChecked()
        return DatasetConfig(
            path=source,
            format=self.format.currentText() if custom else "auto",  # type: ignore[arg-type]
            sheet=(self.sheet.text().strip() or None) if custom else None,
            class_column=(self.class_column.text().strip() or None) if custom else None,
            id_column=(self.id_column.text().strip() or None) if custom else None,
            feature_types=types or None,  # type: ignore[arg-type]
        )

    def _read(self, section: DatasetConfig) -> Dataset:
        source = section.path or ""
        if Path(source).is_file():
            return load_dataset(
                source,
                section.format,
                sheet=section.sheet,
                class_column=section.class_column,
                id_column=section.id_column,
                feature_types=section.feature_types,
            )
        dataset = resolve_dataset(source)
        return (
            dataset.with_feature_types(section.feature_types) if section.feature_types else dataset
        )

    def load(self, source: str, types: dict[str, str] | None = None) -> bool:
        """Load a dataset by file path or by its name in the project; returns whether it worked."""
        section = self._section(source, types)
        try:
            dataset = self._read(section)
        except CSRError as error:
            self.error.setText(str(error))
            self.error.setVisible(True)
            return False
        self.error.setVisible(False)
        if Path(source).is_file():
            self.settings.add_recent(RECENT_DATASETS, str(Path(source).resolve()))
        self.state.set_dataset(dataset, source, section)
        return True

    def reload(self) -> None:
        """Read the current source again with the options of the page."""
        if self.state.dataset_source:
            self.load(self.state.dataset_source)

    def _type_changed(self, feature: str, kind: str) -> None:
        dataset, source = self.state.dataset, self.state.dataset_source
        if dataset is None or source is None:
            return
        types: dict[str, str] = dict(self.state.config.dataset.feature_types or {})
        types[feature] = kind
        if not self.load(source, types):
            self.refresh()  # put the combo boxes back

    # ------------------------------------------------------------------ showing

    def refresh(self) -> None:
        """Show the current dataset."""
        dataset = self.state.dataset
        index = self.source.findData(self.state.dataset_source)
        self.source.setCurrentIndex(max(index, 0))
        # the path of a file the user opened; project and built-in datasets are named in the list
        source = self.state.dataset_source or ""
        self.location.setText(source if dataset is not None and Path(source).is_file() else "")
        self.tabs.setEnabled(dataset is not None)
        if dataset is not None:
            table = dataset_table(dataset, PREVIEW_ROWS)
            self.preview.set_table(table, style_for(table, dataset=dataset))
        self._fill_features(dataset)
        self.refresh_summary()

    def _fill_features(self, dataset: Dataset | None) -> None:
        table = self.features
        rows = [] if dataset is None else feature_rows(dataset)
        table.setRowCount(len(rows))
        colours = self.colours
        for r, (name, kind, low, high, distinct) in enumerate(rows):
            role = CellRole.QUANTITATIVE if kind == "quantitative" else CellRole.NOMINAL
            item = QTableWidgetItem(name)
            item.setBackground(QColor(colours.cells[role][0]))
            item.setForeground(QColor(colours.cells[role][1]))
            table.setItem(r, 0, item)
            combo = QComboBox()
            combo.addItem(tr("quantitative (I)"), FeatureType.QUANTITATIVE.value)
            combo.addItem(tr("nominal (J)"), FeatureType.NOMINAL.value)
            combo.setCurrentIndex(combo.findData(kind))
            combo.activated.connect(
                lambda _, c=combo, n=name: self._type_changed(n, str(c.currentData()))
            )
            table.setCellWidget(r, TYPE_COLUMN, combo)
            for column, value in ((2, low), (3, high), (4, distinct)):
                cell = QTableWidgetItem(format_cell(value))
                cell.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(r, column, cell)

    def type_combo(self, row: int) -> QComboBox:
        """The type selector of the feature in ``row`` of the feature table."""
        widget = self.features.cellWidget(row, TYPE_COLUMN)
        assert isinstance(widget, QComboBox)
        return widget

    def refresh_summary(self) -> None:
        """Recompute |Kᵢ|, the permitted k and r for the current dataset and configuration."""
        dataset = self.state.dataset
        facts = self._fact_labels
        if dataset is None:
            for label in facts.values():
                label.setText("—")
            self.balance.set_sizes(())
            self.problems.set_text("")
            return
        summary = summarize(dataset, self.state.config)
        quantitative = int(dataset.quantitative.sum())
        facts["name"].setText(dataset.name)
        facts["objects"].setText(str(dataset.m))
        facts["features"].setText(
            tr("{n}  ({i} quantitative, {j} nominal)").format(
                n=dataset.n, i=quantitative, j=dataset.n - quantitative
            )
        )
        sizes = dataset.class_sizes
        for key, position in (("k1", 0), ("k2", 1)):
            if position < len(sizes):
                share = 100 * sizes[position] / dataset.m
                facts[key].setText(
                    tr("{size} objects of class {label}  ({share:.0f} %)").format(
                        size=sizes[position], label=dataset.classes[position], share=share
                    )
                )
            else:
                facts[key].setText("—")
        facts["missing"].setText(tr("none"))
        facts["missing"].setToolTip(
            tr("A dataset with missing values is not loaded; the message names the cells.")
        )
        ks = summary.permitted_k
        facts["k"].setText(
            "—"
            if ks is None
            else tr("{ks}   ({rule}, {count} value(s))").format(
                ks=ks.label if ks.count <= 12 else f"{ks.ks[0]}, {ks.ks[1]}, …, {ks.ks[-1]}",
                rule=ks.rule,
                count=ks.count,
            )
        )
        facts["r"].setText("—" if summary.r is None else str(summary.r))
        operators = [f"{o.label} ({len(o.features)})" for o in summary.operators]
        skipped = [f"{s.label} — {tr('skipped')}" for s in summary.skipped_operators]
        facts["operators"].setText(", ".join([*operators, *skipped]) or "—")
        self.balance.set_sizes(sizes if len(sizes) == 2 else ())
        self.problems.set_text("\n".join(f"⚠ {problem}" for problem in summary.problems))

    def apply_palette(self, colours: Palette) -> None:
        """Re-colour the preview, the feature table and the class bar."""
        self.preview.set_palette(colours)
        self.balance.colours = (colours.figures.k1, colours.figures.k2)
        self.balance.update()
        self._fill_features(self.state.dataset)
