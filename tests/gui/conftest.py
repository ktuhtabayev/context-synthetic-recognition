"""Fixtures of the GUI tests: isolated settings, a main window in a temporary working folder."""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QSettings
from pytestqt.qtbot import QtBot

from context_synthetic_recognition.gui.i18n import SOURCE_LANGUAGE, set_language
from context_synthetic_recognition.gui.settings import Settings
from context_synthetic_recognition.gui.window import MainWindow

RUN_TIMEOUT = 120_000


@pytest.fixture(autouse=True)
def english_interface(qapp: QCoreApplication) -> Iterator[None]:
    """Every test starts and ends in English, whatever the language of the machine."""
    set_language(qapp, SOURCE_LANGUAGE)
    yield
    set_language(qapp, SOURCE_LANGUAGE)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings in an ini file of the test's folder: the user's settings are never touched."""
    return Settings(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))


@pytest.fixture
def window(
    qtbot: QtBot, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MainWindow]:
    """The main window with nothing loaded; relative paths (runs, exports) lead into tmp_path."""
    monkeypatch.chdir(tmp_path)
    created = MainWindow(settings)
    created.show_message = lambda title, text: None
    qtbot.addWidget(created)
    yield created
    created.run_page.cancel()
    created.run_page.wait()
    created.export_page.wait()
    created.run_page.log_bridge.detach()


Run = Callable[[MainWindow], None]


@pytest.fixture
def run(qtbot: QtBot) -> Run:
    """Run the experiment of a window (in its worker thread) and wait for the result."""

    def start(window: MainWindow, *, save: bool = False) -> None:
        window.run_page.save.setChecked(save)
        # the page's own signal: it comes after the result has reached the state (waiting for the
        # thread alone would race with the delivery of its result)
        with qtbot.waitSignal(window.run_page.finished, timeout=RUN_TIMEOUT):
            window.run_page.start()
            assert window.run_page.is_running()
        assert window.state.view is not None

    return start


@pytest.fixture
def loaded(window: MainWindow) -> MainWindow:
    """A window with Heart-Disease (10, 13, 2) loaded."""
    assert window.dataset_page.load("heart-disease-10")
    return window


@pytest.fixture
def evaluated(loaded: MainWindow, run: Run) -> MainWindow:
    """A window with Heart-Disease (10, 13, 2) evaluated (template preset, not saved)."""
    run(loaded)
    return loaded
