"""The design system, the persistent settings, the shared state, the translation hook."""

from pathlib import Path

import pytest
from PySide6.QtCore import QByteArray, QCoreApplication
from PySide6.QtGui import QPalette
from pytestqt.qtbot import QtBot

from context_synthetic_recognition.config import DatasetConfig, ExperimentConfig, HAGConfig, preset
from context_synthetic_recognition.data import Dataset
from context_synthetic_recognition.export import RunView
from context_synthetic_recognition.gui import GUI_EXTRA_HINT
from context_synthetic_recognition.gui.i18n import TRANSLATIONS, install_translator, tr
from context_synthetic_recognition.gui.settings import (
    MAX_RECENT,
    RECENT_CONFIGS,
    RECENT_DATASETS,
    Settings,
)
from context_synthetic_recognition.gui.state import AppState
from context_synthetic_recognition.gui.theme import (
    ASSETS,
    DARK,
    FONT_SCALES,
    LIGHT,
    CellRole,
    font_points,
    palette,
    qt_palette,
    stylesheet,
)

pytestmark = pytest.mark.gui


# ---------------------------------------------------------------- theme


def test_the_palettes() -> None:
    assert palette("light") is LIGHT
    assert palette("dark") is DARK
    assert palette("neon") is LIGHT
    for colours in (LIGHT, DARK):
        assert set(colours.cells) == set(CellRole)  # every meaning has its colours in both themes
        assert colours.figures.name == colours.name
        for background, text in colours.cells.values():
            assert background.startswith("#")
            assert text.startswith("#")
    # the classes have the hues of the figures, as tints that keep the text readable
    assert LIGHT.cells[CellRole.K1] != LIGHT.cells[CellRole.K2]
    assert LIGHT.cells[CellRole.GOOD][1] == "#006100"  # the workbook's green
    assert LIGHT.cells[CellRole.QUANTITATIVE][0] == "#ddebf7"  # … and its blue for type I


def test_the_style_sheet() -> None:
    light = stylesheet(LIGHT)
    assert LIGHT.window in light
    assert LIGHT.accent in light
    assert "font-size: 10.0pt" in light
    assert "QLabel#warningBadge" in light
    assert (ASSETS / "check.svg").as_posix() in light
    assert (ASSETS / "check.svg").is_file()
    large = stylesheet(DARK, 1.5)
    assert DARK.window in large
    assert LIGHT.window not in large
    assert "font-size: 15.0pt" in large
    assert font_points(1.15) == 11.5
    assert 1.0 in FONT_SCALES


def test_the_qt_palette(qtbot: QtBot) -> None:
    found = qt_palette(DARK)
    roles = QPalette.ColorRole
    assert found.color(roles.Window).name() == DARK.window
    assert found.color(roles.Text).name() == DARK.text
    assert found.color(roles.Highlight).name() == DARK.accent
    assert found.color(QPalette.ColorGroup.Disabled, roles.Text).name() == DARK.disabled_text


# ---------------------------------------------------------------- settings


def test_settings_are_remembered(settings: Settings, tmp_path: Path) -> None:
    assert (settings.theme, settings.font_scale, settings.page) == ("light", 1.0, 0)
    assert settings.runs_dir == Path("runs")
    assert settings.geometry() is None
    settings.theme = "dark"
    settings.font_scale = 1.3
    settings.page = 3
    settings.runs_dir = tmp_path / "my runs"
    settings.save_geometry(QByteArray(b"geometry"))
    settings.store.sync()
    again = Settings(type(settings.store)(settings.store.fileName(), settings.store.format()))
    assert (again.theme, again.font_scale, again.page) == ("dark", 1.3, 3)
    assert again.runs_dir == tmp_path / "my runs"
    assert again.geometry() == QByteArray(b"geometry")


def test_odd_settings_fall_back(settings: Settings) -> None:
    settings.store.setValue("appearance/theme", "neon")
    settings.store.setValue("appearance/font_scale", "huge")
    settings.store.setValue("window/page", "last")
    assert (settings.theme, settings.font_scale, settings.page) == ("light", 1.0, 0)
    settings.store.setValue("appearance/font_scale", 1.2)  # between two zoom steps
    assert settings.font_scale == 1.15


def test_recent_lists(settings: Settings) -> None:
    assert settings.recent(RECENT_DATASETS) == []
    assert settings.add_recent(RECENT_DATASETS, "a.csv") == ["a.csv"]
    settings.add_recent(RECENT_DATASETS, "b.csv")
    assert settings.add_recent(RECENT_DATASETS, "a.csv") == ["a.csv", "b.csv"]  # moved to the front
    for index in range(MAX_RECENT + 3):
        settings.add_recent(RECENT_DATASETS, f"{index}.csv")
    recent = settings.recent(RECENT_DATASETS)
    assert len(recent) == MAX_RECENT
    assert recent[0] == f"{MAX_RECENT + 2}.csv"
    assert settings.recent(RECENT_CONFIGS) == []
    settings.clear_recent(RECENT_DATASETS)
    assert settings.recent(RECENT_DATASETS) == []
    settings.store.setValue(RECENT_CONFIGS, "only.yaml")  # an ini file stores one entry as text
    assert settings.recent(RECENT_CONFIGS) == ["only.yaml"]
    settings.store.setValue(RECENT_CONFIGS, 7)
    assert settings.recent(RECENT_CONFIGS) == []


def test_default_settings_belong_to_the_application(qtbot: QtBot) -> None:
    store = Settings().store
    assert store.organizationName() == "context-synthetic-recognition"
    assert store.applicationName() == "csr-gui"


# ---------------------------------------------------------------- translation


def test_the_translation_hook(qtbot: QtBot) -> None:
    assert tr("Dataset") == "Dataset"  # English is the source language
    app = QCoreApplication.instance()
    assert app is not None
    assert install_translator(app, "en_GB") is None
    assert install_translator(app) is None or TRANSLATIONS.is_dir()
    assert "PySide6" in GUI_EXTRA_HINT
    assert "[gui]" in GUI_EXTRA_HINT


# ---------------------------------------------------------------- the shared state


def test_the_dataset_and_its_section(
    qtbot: QtBot, experiment: Dataset, experiment_view: RunView
) -> None:
    state = AppState()
    assert state.dataset is None
    assert state.view is None
    assert not state.stale
    section = DatasetConfig(path="builtin:heart-disease-10")
    with qtbot.waitSignals([state.datasetChanged, state.configChanged]):
        state.set_dataset(experiment, "heart-disease-10", section)
    assert state.dataset is experiment
    assert state.dataset_source == "heart-disease-10"
    assert state.config.dataset == section
    with qtbot.assertNotEmitted(state.configChanged):
        state.set_dataset(experiment, "heart-disease-10", section)  # the same section
    state.set_result(experiment_view, Path("runs/x"))
    state.select_object(4)
    with qtbot.waitSignal(state.resultChanged):
        state.set_dataset(experiment, "again")  # other data: the result is dropped
    assert state.view is None
    assert state.run_folder is None
    assert state.selected_object == -1
    assert state.config.dataset == section  # no section given: kept


def test_configuration_changes_can_be_undone(qtbot: QtBot) -> None:
    state = AppState()
    template = state.config
    article = preset("article")
    with qtbot.waitSignal(state.configChanged):
        state.set_config(article, "Preset: article")
    assert state.config.hag == article.hag
    assert state.undo_stack.undoText() == "Preset: article"
    with qtbot.assertNotEmitted(state.configChanged):
        state.set_config(state.config)  # nothing changed: nothing to undo
    assert state.undo_stack.count() == 1
    state.set_config(state.config.model_copy(update={"seed": 7}))
    state.undo_stack.undo()
    assert state.config.seed == 42
    assert state.config.hag == article.hag
    state.undo_stack.undo()
    assert state.config == template
    state.undo_stack.redo()
    assert state.config.hag == article.hag
    with qtbot.waitSignal(state.configChanged):
        state.reset_config(ExperimentConfig(hag=HAGConfig(kappa=3)))
    assert state.config.hag.kappa == 3
    assert not state.undo_stack.canUndo()  # a file was opened: the history starts again


def test_undo_never_changes_the_dataset_section(qtbot: QtBot, experiment: Dataset) -> None:
    state = AppState()
    state.set_config(state.config.model_copy(update={"seed": 1}))
    section = DatasetConfig(path="heart-disease-10")
    state.set_dataset(experiment, "heart-disease-10", section)
    state.undo_stack.undo()
    assert state.config.seed == 42
    assert state.config.dataset == section
    # an edited configuration that carries another dataset section does not replace it either
    state.set_config(preset("article").model_copy(update={"dataset": DatasetConfig(path="x.csv")}))
    assert state.config.dataset == section


def test_results_and_staleness(qtbot: QtBot, experiment_view: RunView, tmp_path: Path) -> None:
    state = AppState()
    with qtbot.waitSignals([state.datasetChanged, state.configChanged, state.resultChanged]):
        state.open_run(experiment_view, tmp_path)
    assert state.view is experiment_view
    assert state.dataset is experiment_view.dataset
    assert state.config == experiment_view.config
    assert state.run_folder == tmp_path
    assert not state.stale
    state.set_config(state.config.model_copy(update={"seed": 3}))
    assert state.stale  # the result on screen belongs to another configuration
    state.undo_stack.undo()
    assert not state.stale
    with qtbot.waitSignal(state.resultChanged):
        state.set_result(experiment_view)
    assert state.run_folder is None


def test_selection_and_appearance(qtbot: QtBot) -> None:
    state = AppState()
    with qtbot.waitSignal(state.objectSelected) as selected:
        state.select_object(3)
    assert selected.args == [3]
    with qtbot.assertNotEmitted(state.objectSelected):
        state.select_object(3)
    with qtbot.waitSignal(state.appearanceChanged):
        state.set_appearance(theme="dark")
    with qtbot.waitSignal(state.appearanceChanged):
        state.set_appearance(font_scale=1.3)
    assert (state.theme, state.font_scale) == ("dark", 1.3)
    with qtbot.assertNotEmitted(state.appearanceChanged):
        state.set_appearance(theme="dark", font_scale=1.3)
        state.set_appearance()
