"""The main window: the workflow sidebar, the pages, menus, shortcuts and what is remembered."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QWidget,
)

from context_synthetic_recognition import __version__
from context_synthetic_recognition.config import ExperimentConfig, load_config, save_config
from context_synthetic_recognition.config.presets import active_deviations, matching_preset
from context_synthetic_recognition.data import load_from_config
from context_synthetic_recognition.errors import CSRError
from context_synthetic_recognition.gui.i18n import tr
from context_synthetic_recognition.gui.pages import (
    ComparePage,
    ConfigurePage,
    DatasetPage,
    ExportPage,
    NewObjectPage,
    Page,
    ResultsPage,
    RunPage,
)
from context_synthetic_recognition.gui.settings import (
    RECENT_CONFIGS,
    RECENT_DATASETS,
    RECENT_RUNS,
    Settings,
)
from context_synthetic_recognition.gui.state import AppState
from context_synthetic_recognition.gui.theme import (
    FONT_SCALES,
    font_points,
    palette,
    qt_palette,
    stylesheet,
)
from context_synthetic_recognition.services.manifest import MANIFEST_NAME

APP_TITLE = "Context-synthetic recognition"
CONFIG_SUFFIXES = {".yaml", ".yml", ".toml", ".json"}
CONFIG_FILES = "Configurations (*.yaml *.yml *.toml *.json);;All files (*)"

SHORTCUTS: tuple[tuple[str, str], ...] = (
    ("Ctrl+O", "Open a dataset"),
    ("Ctrl+Shift+O", "Open a configuration"),
    ("Ctrl+S", "Save the configuration"),
    ("Ctrl+Shift+R", "Open a run folder"),
    ("F5", "Run the experiment"),
    ("Shift+F5", "Cancel the run"),
    ("Ctrl+E", "Export"),
    ("Ctrl+Z / Ctrl+Y", "Undo / redo a change of the configuration"),
    ("Ctrl+1 … Ctrl+7", "Go to a page"),
    ("Alt+Left / Alt+Right", "Previous / next table or figure of the results"),
    ("Ctrl+C", "Copy the selected table rows"),
    ("Ctrl+T", "Switch between the light and the dark theme"),
    ("Ctrl++ / Ctrl+− / Ctrl+0", "Zoom in / out / reset"),
    ("F1", "Keyboard shortcuts"),
    ("Ctrl+Q", "Quit"),
)
"""The keyboard shortcuts, as Help → Keyboard shortcuts lists them."""


class MainWindow(QMainWindow):
    """The application window."""

    def __init__(self, settings: Settings | None = None, parent: QWidget | None = None) -> None:
        """Build the window and restore what the last session left."""
        super().__init__(parent)
        self.settings = settings or Settings()
        self.state = AppState(self)
        self.state.theme = self.settings.theme
        self.state.font_scale = self.settings.font_scale
        self.setAcceptDrops(True)
        self.resize(1360, 860)
        # dialogs are attributes, so tests can replace them
        self.config_open_dialog: Callable[[], str] = self._ask_config_to_open
        self.config_save_dialog: Callable[[], str] = self._ask_config_to_save
        self.run_folder_dialog: Callable[[], str] = self._ask_run_folder
        self.show_message: Callable[[str, str], None] = self._message_box

        self.dataset_page = DatasetPage(self.state, self.settings)
        self.configure_page = ConfigurePage(self.state, self.settings)
        self.run_page = RunPage(self.state, self.settings)
        self.results_page = ResultsPage(self.state, self.settings)
        self.new_object_page = NewObjectPage(self.state, self.settings)
        self.compare_page = ComparePage(self.state, self.settings)
        self.export_page = ExportPage(self.state, self.settings)
        self.pages: tuple[Page, ...] = (
            self.dataset_page,
            self.configure_page,
            self.run_page,
            self.results_page,
            self.new_object_page,
            self.compare_page,
            self.export_page,
        )

        self.sidebar = QListWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(184)
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.stack = QStackedWidget()
        for number, page in enumerate(self.pages, start=1):
            item = QListWidgetItem(f"{number}   {tr(page.title)}")
            item.setToolTip(f"{tr(page.title)}  (Ctrl+{number})")
            self.sidebar.addItem(item)
            self.stack.addWidget(page)
        self.sidebar.currentRowChanged.connect(self._page_changed)
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.sidebar)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        # ---- status bar
        self.dataset_label = QLabel()
        self.preset_label = QLabel()
        self.preset_label.setObjectName("badge")
        self.deviation_button = QPushButton()
        self.deviation_button.setObjectName("warningBadge")
        self.deviation_button.setFlat(True)
        self.deviation_button.clicked.connect(lambda: self.go_to(self.configure_page))
        self.result_label = QLabel()
        bar = self.statusBar()
        bar.addWidget(self.dataset_label)
        bar.addWidget(self.result_label)
        bar.addPermanentWidget(self.preset_label)
        bar.addPermanentWidget(self.deviation_button)

        self._build_menus()
        self.run_page.explore_button.clicked.connect(lambda: self.go_to(self.results_page))
        self.compare_page.openRequested.connect(self.open_run)
        self.state.datasetChanged.connect(self._refresh)
        self.state.configChanged.connect(self._refresh)
        self.state.resultChanged.connect(self._refresh)
        self.state.appearanceChanged.connect(self.apply_appearance)

        geometry = self.settings.geometry()
        if geometry is not None:
            self.restoreGeometry(geometry)
        self.apply_appearance()
        self._refresh()
        self.sidebar.setCurrentRow(0)

    # ------------------------------------------------------------------ menus

    def _action(
        self,
        menu: QMenu,
        text: str,
        slot: Callable[[], object],
        shortcut: str | QKeySequence.StandardKey | None = None,
    ) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(lambda _checked=False: slot())
        menu.addAction(action)
        return action

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu(tr("&File"))
        self._action(file_menu, tr("Open dataset…"), self.dataset_page.open_file, "Ctrl+O")
        self._action(file_menu, tr("Open configuration…"), self.open_config_dialog, "Ctrl+Shift+O")
        self._action(file_menu, tr("Save configuration as…"), self.save_config_dialog, "Ctrl+S")
        self._action(file_menu, tr("Open run folder…"), self.open_run_dialog, "Ctrl+Shift+R")
        file_menu.addSeparator()
        self.recent_menus = {
            RECENT_DATASETS: file_menu.addMenu(tr("Recent datasets")),
            RECENT_CONFIGS: file_menu.addMenu(tr("Recent configurations")),
            RECENT_RUNS: file_menu.addMenu(tr("Recent runs")),
        }
        for key, menu in self.recent_menus.items():
            menu.aboutToShow.connect(partial(self._fill_recent, key))
        file_menu.addSeparator()
        self._action(file_menu, tr("Export…"), lambda: self.go_to(self.export_page), "Ctrl+E")
        file_menu.addSeparator()
        self._action(file_menu, tr("Quit"), self.close, "Ctrl+Q")

        edit_menu = bar.addMenu(tr("&Edit"))
        self.undo_action = self.state.undo_stack.createUndoAction(self, tr("Undo"))
        self.undo_action.setShortcut(QKeySequence(QKeySequence.StandardKey.Undo))
        self.redo_action = self.state.undo_stack.createRedoAction(self, tr("Redo"))
        self.redo_action.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()
        self._action(
            edit_menu, tr("Template preset"), lambda: self.configure_page.apply_preset("template")
        )
        self._action(
            edit_menu, tr("Article preset"), lambda: self.configure_page.apply_preset("article")
        )

        run_menu = bar.addMenu(tr("&Run"))
        self.run_action = self._action(run_menu, tr("Run the experiment"), self.run, "F5")
        self.cancel_action = self._action(run_menu, tr("Cancel"), self.run_page.cancel, "Shift+F5")

        view_menu = bar.addMenu(tr("&View"))
        for number, page in enumerate(self.pages, start=1):
            self._action(view_menu, tr(page.title), partial(self.go_to, page), f"Ctrl+{number}")
        view_menu.addSeparator()
        self._action(view_menu, tr("Previous table or figure"), lambda: self._step(-1), "Alt+Left")
        self._action(view_menu, tr("Next table or figure"), lambda: self._step(1), "Alt+Right")
        view_menu.addSeparator()
        themes = QActionGroup(self)
        self.theme_actions: dict[str, QAction] = {}
        for name, text in (("light", tr("Light theme")), ("dark", tr("Dark theme"))):
            action = QAction(text, self)
            action.setCheckable(True)
            action.triggered.connect(lambda _checked=False, n=name: self.set_theme(n))
            themes.addAction(action)
            view_menu.addAction(action)
            self.theme_actions[name] = action
        self._action(view_menu, tr("Switch theme"), self.toggle_theme, "Ctrl+T")
        view_menu.addSeparator()
        zoom_in = self._action(view_menu, tr("Zoom in"), lambda: self.zoom(1), "Ctrl++")
        zoom_in.setShortcuts([QKeySequence("Ctrl++"), QKeySequence("Ctrl+=")])
        self._action(view_menu, tr("Zoom out"), lambda: self.zoom(-1), "Ctrl+-")
        self._action(view_menu, tr("Actual size"), lambda: self.zoom(0), "Ctrl+0")

        help_menu = bar.addMenu(tr("&Help"))
        self._action(help_menu, tr("Keyboard shortcuts"), self.show_shortcuts, "F1")
        self._action(help_menu, tr("About"), self.show_about)

    def _fill_recent(self, key: str) -> None:
        menu = self.recent_menus[key]
        menu.clear()
        entries = self.settings.recent(key)
        for entry in entries:
            self._action(menu, entry, partial(self.open_path, Path(entry)))
        if entries:
            menu.addSeparator()
            self._action(menu, tr("Clear this list"), lambda: self.settings.clear_recent(key))
        else:
            empty = self._action(menu, tr("(empty)"), lambda: None)
            empty.setEnabled(False)

    # ------------------------------------------------------------------ pages

    def go_to(self, page: Page) -> bool:
        """Show a page; returns ``False`` if it needs a run and there is none."""
        index = self.pages.index(page)
        item = self.sidebar.item(index)
        if not item.flags() & Qt.ItemFlag.ItemIsEnabled:
            return False
        self.sidebar.setCurrentRow(index)
        return True

    def current_page(self) -> Page:
        """The page shown."""
        return self.pages[self.stack.currentIndex()]

    def _page_changed(self, row: int) -> None:
        if 0 <= row < len(self.pages):
            self.stack.setCurrentIndex(row)

    def _step(self, direction: int) -> None:
        if self.current_page() is self.results_page:
            self.results_page.step(direction)

    def _refresh(self) -> None:
        """Enable what is possible now and update the title and the status bar."""
        state = self.state
        has_result = state.view is not None
        for index, page in enumerate(self.pages):
            item = self.sidebar.item(index)
            enabled = has_result or not page.needs_result
            flags = item.flags()
            item.setFlags(
                flags | Qt.ItemFlag.ItemIsEnabled if enabled else flags & ~Qt.ItemFlag.ItemIsEnabled
            )
        if not has_result and self.current_page().needs_result:
            self.sidebar.setCurrentRow(self.pages.index(self.run_page))
        dataset = state.dataset
        name = tr("no dataset") if dataset is None else dataset.name
        self.setWindowTitle(f"{name} — {APP_TITLE}")
        self.dataset_label.setText(
            tr("No dataset loaded")
            if dataset is None
            else tr("{name} · {m} objects · {n} features").format(
                name=dataset.name, m=dataset.m, n=dataset.n
            )
        )
        matched = matching_preset(state.config)
        self.preset_label.setText(
            tr("preset: {name}").format(name=matched.value if matched else tr("custom"))
        )
        active = active_deviations(state.config)
        self.deviation_button.setVisible(bool(active))
        self.deviation_button.setText(tr("⚠ {n} template calculation(s)").format(n=len(active)))
        self.deviation_button.setToolTip(
            "\n".join(f"{d.adr}: {d.statement}" for d in active)
            + "\n"
            + tr("Click to see the switches on the Configure page.")
        )
        if state.view is None:
            self.result_label.setText("")
        else:
            text = tr("TUPLAM = {tuplam}").format(tuplam=state.view.hag.label)
            if state.stale:
                text += "  ·  " + tr("⚠ computed with another configuration")
            self.result_label.setText(text)

    # ------------------------------------------------------------------ appearance

    def apply_appearance(self) -> None:
        """Apply the theme and the zoom to the whole application."""
        state = self.state
        app = QApplication.instance()
        if isinstance(app, QApplication):
            font = QFont(app.font())
            font.setPointSizeF(font_points(state.font_scale))
            app.setFont(font)
            app.setPalette(qt_palette(palette(state.theme)))
            app.setStyleSheet(stylesheet(palette(state.theme), state.font_scale))
        for name, action in self.theme_actions.items():
            action.setChecked(name == state.theme)
        self.settings.theme = state.theme
        self.settings.font_scale = state.font_scale

    def set_theme(self, name: str) -> None:
        """Switch to the light or the dark theme."""
        self.state.set_appearance(theme=name)

    def toggle_theme(self) -> None:
        """Switch between the light and the dark theme."""
        self.set_theme("dark" if self.state.theme == "light" else "light")

    def zoom(self, direction: int) -> None:
        """Make the interface larger (+1), smaller (−1) or its actual size (0)."""
        if direction == 0:
            scale = 1.0
        else:
            steps = list(FONT_SCALES)
            current = min(range(len(steps)), key=lambda i: abs(steps[i] - self.state.font_scale))
            scale = steps[min(max(current + direction, 0), len(steps) - 1)]
        self.state.set_appearance(font_scale=scale)

    # ------------------------------------------------------------------ files

    def _ask_config_to_open(self) -> str:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Open a configuration"), "", tr(CONFIG_FILES)
        )
        return path

    def _ask_config_to_save(self) -> str:
        suggested = f"{self.state.config.name}.yaml"
        path, _ = QFileDialog.getSaveFileName(
            self, tr("Save the configuration"), suggested, tr(CONFIG_FILES)
        )
        return path

    def _ask_run_folder(self) -> str:
        return QFileDialog.getExistingDirectory(
            self, tr("Open a run folder"), str(self.settings.runs_dir)
        )

    def _message_box(self, title: str, text: str) -> None:
        QMessageBox.information(self, title, text)

    def open_config_dialog(self) -> None:
        """Ask for a configuration file and open it."""
        path = self.config_open_dialog()
        if path:
            self.open_config(Path(path))

    def save_config_dialog(self) -> None:
        """Ask where to save the configuration and save it."""
        path = self.config_save_dialog()
        if path:
            self.save_config(Path(path))

    def open_run_dialog(self) -> None:
        """Ask for a run folder and open it."""
        folder = self.run_folder_dialog()
        if folder:
            self.open_run(Path(folder))

    def open_config(self, path: Path) -> bool:
        """Open a configuration file: its settings and, if it names one, its dataset."""
        try:
            config = load_config(path)
        except CSRError as error:
            self.report(str(error), failed=True)
            return False
        self.settings.add_recent(RECENT_CONFIGS, str(path.resolve()))
        self.state.reset_config(config)
        problem = self._load_dataset_of(config, path.parent)
        if problem:  # the settings are open; the dataset the file names is not
            self.report(problem, failed=True)
        else:
            self.report(tr("Configuration opened: {path}").format(path=path))
        return True

    def _load_dataset_of(self, config: ExperimentConfig, base: Path) -> str:
        """Load the dataset a configuration names; returns the problem if it cannot be read."""
        section = config.dataset
        if section.path is None and self.state.dataset is not None:
            return ""  # the configuration names no dataset: keep the one that is open
        try:
            dataset = load_from_config(section, base)
        except CSRError as error:
            return str(error)
        source = section.path or "default"
        candidate = base / source
        self.state.set_dataset(dataset, str(candidate) if candidate.is_file() else source, section)
        return ""

    def save_config(self, path: Path) -> bool:
        """Save the configuration (YAML, TOML or JSON by the file's suffix)."""
        try:
            save_config(self.state.config, path, overwrite=True)
        except (CSRError, OSError) as error:
            self.report(str(error), failed=True)
            return False
        self.settings.add_recent(RECENT_CONFIGS, str(path.resolve()))
        self.report(tr("Configuration saved: {path}").format(path=path))
        return True

    def open_run(self, folder: Path) -> None:
        """Repeat a saved run from its folder and show it."""
        self.go_to(self.run_page)
        self.run_page.open_run(Path(folder))

    def open_path(self, path: Path) -> None:
        """Open whatever a path is: a run folder, a configuration file or a dataset file."""
        if path.is_dir():
            if (path / MANIFEST_NAME).is_file():
                self.open_run(path)
            else:
                self.report(tr("{path} is not a run folder").format(path=path), failed=True)
        elif path.suffix.lower() in CONFIG_SUFFIXES:
            self.open_config(path)
        else:
            self.go_to(self.dataset_page)
            self.dataset_page.load(str(path))

    def run(self) -> None:
        """Run the experiment (F5)."""
        self.go_to(self.run_page)
        self.run_page.start()

    def report(self, message: str, *, failed: bool = False) -> None:
        """Say something in the status bar (longer when it is an error)."""
        self.statusBar().showMessage(("✗ " if failed else "") + message, 15000 if failed else 6000)

    # ------------------------------------------------------------------ help

    def shortcuts_text(self) -> str:
        """The keyboard shortcuts as text."""
        return "\n".join(f"{keys:28s}{tr(what)}" for keys, what in SHORTCUTS)

    def about_text(self) -> str:
        """What the application is."""
        return tr(
            "Context-synthetic recognition {version}\n\n"
            "The context-synthetic model of recognition algorithms (CS-model): local "
            "k-nearest-neighbour contexts → synthetic features Ψ(r) → hierarchical agglomerative "
            "grouping → meta-algorithm.\n\n"
            "⚠ Two template calculations differ from the article (running class centres in θ/γ; "
            "the STEP 4 majorizer applied twice). Both are switches on the Configure page; the "
            "default reproduces the template workbooks."
        ).format(version=__version__)

    def show_shortcuts(self) -> None:
        """Help → Keyboard shortcuts."""
        self.show_message(tr("Keyboard shortcuts"), self.shortcuts_text())

    def show_about(self) -> None:
        """Help → About."""
        self.show_message(tr("About"), self.about_text())

    # ------------------------------------------------------------------ Qt events

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept dropped files and folders."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        """Open the first dropped file or folder."""
        urls = [url for url in event.mimeData().urls() if url.isLocalFile()]
        if urls:
            event.acceptProposedAction()
            self.open_path(Path(urls[0].toLocalFile()))

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop background work and remember the window."""
        self.run_page.cancel()
        self.run_page.wait()
        self.export_page.wait()
        self.run_page.log_bridge.detach()
        self.settings.save_geometry(self.saveGeometry())
        self.settings.page = self.stack.currentIndex()
        self.settings.store.sync()
        super().closeEvent(event)
