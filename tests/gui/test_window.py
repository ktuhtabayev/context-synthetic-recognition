"""The main window: navigation, menus, appearance, files, what is remembered, starting it."""

import sys
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QMimeData, QPoint, QSettings, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot
from typer.testing import CliRunner

from context_synthetic_recognition import __version__
from context_synthetic_recognition.cli.app import EXIT_FAILED, app
from context_synthetic_recognition.config import (
    DatasetConfig,
    ExperimentConfig,
    load_config,
    preset,
    save_config,
)
from context_synthetic_recognition.gui import app as gui_app
from context_synthetic_recognition.gui.settings import (
    RECENT_CONFIGS,
    RECENT_DATASETS,
    RECENT_RUNS,
    Settings,
)
from context_synthetic_recognition.gui.theme import DARK, LIGHT
from context_synthetic_recognition.gui.window import SHORTCUTS, MainWindow

from .conftest import RUN_TIMEOUT, Run

pytestmark = pytest.mark.gui


def menu_texts(window: MainWindow, title: str) -> list[str]:
    menu = next(a.menu() for a in window.menuBar().actions() if a.text() == title)
    return [action.text() for action in menu.actions() if not action.isSeparator()]


def action(window: MainWindow, title: str, text: str) -> Any:
    menu = next(a.menu() for a in window.menuBar().actions() if a.text() == title)
    return next(a for a in menu.actions() if a.text() == text)


# ---------------------------------------------------------------- navigation


def test_the_window_before_anything_is_loaded(window: MainWindow) -> None:
    assert window.windowTitle() == "no dataset — Context-synthetic recognition"
    assert window.dataset_label.text() == "No dataset loaded"
    assert window.preset_label.text() == "preset: template"
    assert window.deviation_button.text() == "⚠ 2 template calculation(s)"
    assert "ADR-002" in window.deviation_button.toolTip()
    assert window.result_label.text() == ""
    assert [window.sidebar.item(i).text() for i in range(window.sidebar.count())] == [
        "1   Dataset",
        "2   Configure",
        "3   Run",
        "4   Results",
        "5   New object",
        "6   Compare",
        "7   Export",
    ]
    assert window.current_page() is window.dataset_page
    enabled = [
        bool(window.sidebar.item(i).flags() & Qt.ItemFlag.ItemIsEnabled)
        for i in range(window.sidebar.count())
    ]
    assert enabled == [True, True, True, False, False, True, False]
    assert not window.go_to(window.results_page)  # there is no run to show
    assert window.go_to(window.configure_page)
    assert window.current_page() is window.configure_page
    window.deviation_button.click()
    assert window.current_page() is window.configure_page


def test_a_run_opens_the_result_pages(loaded: MainWindow, run: Run) -> None:
    assert loaded.dataset_label.text() == "Heart-Disease (10, 13, 2) · 10 objects · 13 features"
    action(loaded, "&Run", "Run the experiment").trigger()  # F5
    assert loaded.current_page() is loaded.run_page
    assert loaded.run_page.is_running()
    loaded.run_page.wait()
    run_page = loaded.run_page
    assert run_page.task is not None
    QApplication.processEvents()
    assert loaded.state.view is not None
    assert loaded.result_label.text() == "TUPLAM = {a₆, a₃, a₁, a₂, a₄}"
    for page in (loaded.results_page, loaded.new_object_page, loaded.export_page):
        assert loaded.go_to(page)
    loaded.state.set_dataset(None)  # the data are gone: so are the result pages
    assert loaded.current_page() is loaded.run_page
    assert not loaded.go_to(loaded.export_page)


def test_the_menus(window: MainWindow) -> None:
    assert [a.text() for a in window.menuBar().actions()] == [
        "&File",
        "&Edit",
        "&Run",
        "&View",
        "&Help",
    ]
    assert menu_texts(window, "&File") == [
        "Open dataset…",
        "Open configuration…",
        "Save configuration as…",
        "Open run folder…",
        "Recent datasets",
        "Recent configurations",
        "Recent runs",
        "Export…",
        "Quit",
    ]
    assert menu_texts(window, "&Edit") == ["Undo", "Redo", "Template preset", "Article preset"]
    assert action(window, "&Run", "Run the experiment").shortcut() == QKeySequence("F5")
    assert action(window, "&File", "Open dataset…").shortcut() == QKeySequence("Ctrl+O")
    assert action(window, "&View", "Results").shortcut() == QKeySequence("Ctrl+4")
    assert not window.undo_action.isEnabled()
    action(window, "&Edit", "Article preset").trigger()
    assert window.preset_label.text() == "preset: article"
    assert window.undo_action.isEnabled()
    window.undo_action.trigger()
    assert window.redo_action.isEnabled()
    action(window, "&View", "Compare").trigger()
    assert window.current_page() is window.compare_page
    action(window, "&File", "Export…").trigger()  # needs a run: stays
    assert window.current_page() is window.compare_page
    action(window, "&Run", "Cancel").trigger()  # nothing runs


def test_help(window: MainWindow) -> None:
    shown: list[tuple[str, str]] = []
    window.show_message = lambda title, text: shown.append((title, text))
    action(window, "&Help", "Keyboard shortcuts").trigger()
    action(window, "&Help", "About").trigger()
    assert [title for title, _ in shown] == ["Keyboard shortcuts", "About"]
    assert shown[0][1].count("\n") == len(SHORTCUTS) - 1
    assert "F5" in shown[0][1]
    assert f"Context-synthetic recognition {__version__}" in shown[1][1]
    assert "⚠ Two template calculations differ from the article" in shown[1][1]


# ---------------------------------------------------------------- appearance


def test_themes_and_zoom(window: MainWindow, qtbot: QtBot) -> None:
    application = QApplication.instance()
    assert isinstance(application, QApplication)
    assert window.state.theme == "light"
    assert window.theme_actions["light"].isChecked()
    assert LIGHT.window in application.styleSheet()
    action(window, "&View", "Switch theme").trigger()  # Ctrl+T
    assert window.state.theme == "dark"
    assert DARK.window in application.styleSheet()
    assert window.theme_actions["dark"].isChecked()
    assert window.settings.theme == "dark"
    assert application.palette().color(application.palette().ColorRole.Window).name() == DARK.window
    window.theme_actions["light"].trigger()
    assert window.state.theme == "light"
    action(window, "&View", "Zoom in").trigger()
    action(window, "&View", "Zoom in").trigger()
    assert window.state.font_scale == 1.3
    assert "font-size: 13.0pt" in application.styleSheet()
    assert application.font().pointSizeF() == 13.0
    assert window.settings.font_scale == 1.3
    for _ in range(10):
        window.zoom(1)
    assert window.state.font_scale == 1.75  # the largest step
    for _ in range(10):
        window.zoom(-1)
    assert window.state.font_scale == 0.85
    action(window, "&View", "Actual size").trigger()
    assert window.state.font_scale == 1.0


def test_the_window_remembers_its_state(
    qtbot: QtBot, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    first = MainWindow(settings)
    qtbot.addWidget(first)
    first.set_theme("dark")
    first.zoom(1)
    first.resize(640, 480)  # within the 800 × 800 screen of the offscreen platform
    first.go_to(first.compare_page)
    first.close()
    assert settings.page == first.pages.index(first.compare_page)
    assert settings.geometry() is not None
    second = MainWindow(settings)
    qtbot.addWidget(second)
    assert second.state.theme == "dark"
    assert second.state.font_scale == 1.15
    assert (second.width(), second.height()) == (640, 480)
    second.close()
    second.set_theme("light")
    second.zoom(0)


# ---------------------------------------------------------------- files


def test_saving_and_opening_a_configuration(loaded: MainWindow, tmp_path: Path) -> None:
    loaded.configure_page.article_button.click()
    target = tmp_path / "experiment.yaml"
    loaded.config_save_dialog = lambda: str(target)
    action(loaded, "&File", "Save configuration as…").trigger()
    saved = load_config(target)
    assert saved.hag == preset("article").hag
    assert saved.dataset.path == "heart-disease-10"
    assert loaded.settings.recent(RECENT_CONFIGS) == [str(target.resolve())]
    assert "Configuration saved" in loaded.statusBar().currentMessage()
    loaded.configure_page.template_button.click()
    loaded.config_open_dialog = lambda: str(target)
    action(loaded, "&File", "Open configuration…").trigger()
    assert loaded.state.config.hag == preset("article").hag
    assert not loaded.undo_action.isEnabled()  # a file was opened: the history starts again
    assert loaded.state.dataset is not None
    for dialog in ("config_open_dialog", "config_save_dialog", "run_folder_dialog"):
        setattr(loaded, dialog, lambda: "")  # cancelled dialogs change nothing
    loaded.open_config_dialog()
    loaded.save_config_dialog()
    loaded.open_run_dialog()
    assert loaded.state.config.hag == preset("article").hag


def test_a_configuration_brings_its_dataset(window: MainWindow, tmp_path: Path) -> None:
    data = tmp_path / "patients.csv"
    data.write_text(
        "age,pressure,label\n63,145,sick\n41,130,well\n57,120,well\n44,150,sick\n52,172,sick\n39,118,well\n",
        encoding="utf-8",
    )
    path = tmp_path / "experiment.toml"
    save_config(
        ExperimentConfig(dataset=DatasetConfig(path="patients.csv", class_column="label"), seed=5),
        path,
    )
    assert window.open_config(path)
    dataset = window.state.dataset
    assert dataset is not None
    assert dataset.m == 6  # the relative path is resolved against the configuration's folder
    assert window.state.config.seed == 5
    assert window.state.dataset_source == str(tmp_path / "patients.csv")
    # a configuration that names no dataset keeps the one that is open
    bare = tmp_path / "bare.yaml"
    save_config(preset("article"), bare)
    assert window.open_config(bare)
    assert window.state.dataset is dataset
    assert window.state.config.hag.step4_passes == 1
    # … and one whose dataset is missing says so and keeps the settings
    lost = tmp_path / "lost.yaml"
    save_config(ExperimentConfig(dataset=DatasetConfig(path="gone.csv"), seed=9), lost)
    assert window.open_config(lost)
    assert window.state.config.seed == 9
    assert "gone.csv" in window.statusBar().currentMessage()
    assert window.state.dataset is dataset


def test_files_that_cannot_be_opened_or_saved(window: MainWindow, tmp_path: Path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text("hag: {alpha: 7}\n", encoding="utf-8")
    assert not window.open_config(broken)
    assert window.statusBar().currentMessage().startswith("✗ ")
    assert window.settings.recent(RECENT_CONFIGS) == []
    assert not window.save_config(tmp_path / "experiment.docx")  # not a configuration format
    assert "experiment.docx" in window.statusBar().currentMessage()
    folder = tmp_path / "folder.yaml"
    folder.mkdir()
    assert not window.save_config(folder)  # the file cannot be written
    assert window.statusBar().currentMessage().startswith("✗ ")


def test_opening_a_run_folder(
    evaluated: MainWindow, run: Run, qtbot: QtBot, tmp_path: Path
) -> None:
    evaluated.run_page.runs_dir.setText(str(tmp_path / "runs"))
    evaluated.configure_page.article_button.click()
    run(evaluated, save=True)  # type: ignore[call-arg]
    folder = evaluated.state.run_folder
    assert folder is not None
    evaluated.state.set_dataset(None)
    evaluated.configure_page.template_button.click()
    evaluated.run_folder_dialog = lambda: str(folder)
    with qtbot.waitSignal(evaluated.run_page.finished, timeout=RUN_TIMEOUT):
        action(evaluated, "&File", "Open run folder…").trigger()
        assert evaluated.current_page() is evaluated.run_page
    state = evaluated.state
    assert state.run_folder == folder
    assert state.view.run_id == folder.name  # type: ignore[union-attr]
    assert state.config.hag.step4_passes == 1  # the run's configuration is current again
    assert state.dataset is not None
    assert not state.stale
    assert evaluated.settings.recent(RECENT_RUNS)[0] == str(folder)
    assert f"— repeating the run {folder.name} from its folder —" in (
        evaluated.run_page.console.toPlainText()
    )
    assert evaluated.export_page.folder.text() == str(folder)


def test_opening_whatever_a_path_is(window: MainWindow, tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    data.write_text("a,b,label\n1,2,x\n3,4,y\n5,6,x\n7,8,y\n", encoding="utf-8")
    window.go_to(window.compare_page)
    window.open_path(data)
    assert window.current_page() is window.dataset_page
    assert window.state.dataset is not None
    assert window.state.dataset.m == 4
    config = tmp_path / "experiment.json"
    save_config(preset("article"), config)
    window.open_path(config)
    assert window.preset_label.text() == "preset: article"
    window.open_path(tmp_path)  # a folder without a manifest
    assert "is not a run folder" in window.statusBar().currentMessage()
    window.open_path(tmp_path / "missing.dat")
    assert "dataset file not found" in window.dataset_page.error.text()


def test_recent_files(window: MainWindow, tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    data.write_text("a,b,label\n1,2,x\n3,4,y\n5,6,x\n7,8,y\n", encoding="utf-8")
    window._fill_recent(RECENT_DATASETS)
    menu = window.recent_menus[RECENT_DATASETS]
    assert [a.text() for a in menu.actions()] == ["(empty)"]
    assert not menu.actions()[0].isEnabled()
    window.open_path(data)
    window.state.set_dataset(None)
    menu.aboutToShow.emit()
    entries = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert entries == [str(data.resolve()), "Clear this list"]
    menu.actions()[0].trigger()  # the recent file opens again
    assert window.state.dataset is not None
    menu.actions()[-1].trigger()
    assert window.settings.recent(RECENT_DATASETS) == []


def test_dropping_a_file_onto_the_window(window: MainWindow, tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    data.write_text("a,b,label\n1,2,x\n3,4,y\n5,6,x\n7,8,y\n", encoding="utf-8")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(data))])
    copy, left, none = (
        Qt.DropAction.CopyAction,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    enter = QDragEnterEvent(QPoint(5, 5), copy, mime, left, none)
    window.dragEnterEvent(enter)
    assert enter.isAccepted()
    window.dropEvent(QDropEvent(QPoint(5, 5), copy, mime, left, none))
    assert window.state.dataset is not None
    assert window.state.dataset.name == "data"
    text = QMimeData()
    text.setText("not a file")
    ignored = QDragEnterEvent(QPoint(5, 5), copy, text, left, none)
    ignored.ignore()
    window.dragEnterEvent(ignored)
    assert not ignored.isAccepted()
    window.dropEvent(QDropEvent(QPoint(5, 5), copy, text, left, none))  # nothing to open


# ---------------------------------------------------------------- painting


def test_every_page_paints_in_both_themes(evaluated: MainWindow) -> None:
    # nothing is shown on the offscreen platform, so painting is asked for: the coloured table
    # header, the class bar and the canvases draw themselves without an error
    evaluated.resize(1000, 700)
    results = evaluated.results_page
    for theme in ("light", "dark"):
        evaluated.set_theme(theme)
        for page in evaluated.pages:
            assert evaluated.go_to(page)
            image = evaluated.grab().toImage()
            assert not image.isNull()
            assert (image.width(), image.height()) == (evaluated.width(), evaluated.height())
        evaluated.go_to(results)
        results.open("table", "psi")
        results.table_panel.view.sortByColumn(1, Qt.SortOrder.DescendingOrder)  # the sort arrow
        evaluated.state.select_object(2)
        assert not evaluated.grab().isNull()
        results.open("figure", "margins")
        assert not evaluated.grab().isNull()
        results.open("context", "context")
        assert not evaluated.grab().isNull()
    surface = evaluated.grab().toImage().pixelColor(400, 5).name()
    assert surface == DARK.window  # the window is painted in the theme's colour


# ---------------------------------------------------------------- starting the application


def store(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "app.ini"), QSettings.Format.IniFormat)


def test_creating_the_window(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert gui_app.create_application() is QApplication.instance()
    plain = gui_app.create_window(store=store(tmp_path))
    qtbot.addWidget(plain)
    assert plain.state.dataset is not None  # the default dataset (ADR-035)
    assert plain.state.dataset.m == 10
    named = gui_app.create_window(dataset="heart-disease-270", store=store(tmp_path))
    qtbot.addWidget(named)
    assert named.state.dataset is not None
    assert named.state.dataset.m == 270
    config = tmp_path / "experiment.yaml"
    save_config(preset("article"), config)
    configured = gui_app.create_window(config=config, store=store(tmp_path))
    qtbot.addWidget(configured)
    assert configured.state.config.hag.step4_passes == 1
    assert configured.state.dataset is not None
    for created in (plain, named, configured):
        created.close()


def test_starting_with_a_run_folder(
    evaluated: MainWindow, run: Run, qtbot: QtBot, tmp_path: Path
) -> None:
    evaluated.run_page.runs_dir.setText(str(tmp_path / "runs"))
    run(evaluated, save=True)  # type: ignore[call-arg]
    folder = evaluated.state.run_folder
    assert folder is not None
    opened = gui_app.create_window(run=folder, store=store(tmp_path))
    qtbot.addWidget(opened)
    # the run is being repeated in the worker thread; its signals arrive once events are processed
    with qtbot.waitSignal(opened.run_page.finished, timeout=RUN_TIMEOUT):
        assert opened.run_page.is_running()
    assert opened.state.view is not None
    assert opened.state.run_folder == folder
    opened.close()


def test_the_entry_points(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def launch(**arguments: Any) -> int:
        calls.append(arguments)
        return 0

    monkeypatch.setattr(gui_app, "launch", launch)
    with pytest.raises(SystemExit) as exit_info:
        gui_app.main(["heart-disease-10", "--config", "experiment.yaml", "-r", "runs/x"])
    assert exit_info.value.code == 0
    assert calls[-1] == {
        "dataset": "heart-disease-10",
        "config": Path("experiment.yaml"),
        "run": Path("runs/x"),
    }
    runner = CliRunner()
    result = runner.invoke(app, ["gui", "heart-disease-270", "--run", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert calls[-1] == {"dataset": "heart-disease-270", "config": None, "run": tmp_path}
    assert runner.invoke(app, ["gui"]).exit_code == 0
    assert calls[-1] == {"dataset": None, "config": None, "run": None}


def test_the_application_is_launched(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    application = QApplication.instance()
    assert isinstance(application, QApplication)
    shown: list[MainWindow] = []
    monkeypatch.setattr(MainWindow, "show", lambda self: shown.append(self))
    monkeypatch.setattr(QApplication, "exec", lambda self=None: 3)
    monkeypatch.setattr(gui_app, "Settings", lambda _=None: Settings(store(tmp_path)))
    assert gui_app.launch(dataset="heart-disease-10") == 3
    assert len(shown) == 1
    assert shown[0].state.dataset is not None
    shown[0].close()


def test_csr_gui_without_pyside6(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "context_synthetic_recognition.gui.app", None)
    result = CliRunner().invoke(app, ["gui"])
    assert result.exit_code == EXIT_FAILED
    assert 'pip install "context-synthetic-recognition[gui]"' in result.output


def test_the_self_test(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, settings: Settings
) -> None:
    monkeypatch.chdir(tmp_path)  # no project folder here: the built-in default dataset is used
    report = tmp_path / "self-test.txt"
    assert gui_app.self_test(report) == 0
    lines = report.read_text(encoding="utf-8").splitlines()
    assert lines[0].endswith("— self-test")
    assert lines[1:4] == [
        "ok    language English: Run the experiment",
        "ok    language Русский: Запустить эксперимент",
        "ok    language Oʻzbekcha: Tajribani ishga tushirish",
    ]
    assert "ok    default dataset: Heart-Disease (10, 13, 2)" in lines
    assert "ok    experiment: TUPLAM = {a₆, a₃, a₁, a₂, a₄}" in lines
    assert "ok    results: a table and a figure" in lines
    assert any(line.startswith("ok    export: excel, csv, json") for line in lines)
    assert lines[-1] == "passed"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["self-test.txt"]  # nothing else
    assert settings.language == "system"  # the user's settings are not touched

    # the entry point: csr-gui --self-test REPORT
    again = tmp_path / "again.txt"
    with pytest.raises(SystemExit) as exit_info:
        gui_app.main(["--self-test", str(again)])
    assert exit_info.value.code == 0
    assert again.read_text(encoding="utf-8").splitlines()[-1] == "passed"


def test_the_self_test_reports_a_failure_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken(*_: Any, **__: Any) -> MainWindow:
        raise RuntimeError("no window today")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gui_app, "create_window", broken)
    report = tmp_path / "self-test.txt"
    assert gui_app.self_test(report) == 1
    text = report.read_text(encoding="utf-8")
    assert "RuntimeError: no window today" in text
    assert text.splitlines()[-1] == "FAILED"
    assert gui_app.self_test() == 1  # without a report: the exit code alone
