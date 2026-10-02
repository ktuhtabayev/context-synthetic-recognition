"""The languages of the interface: the translation files, the language switch (ADR-054)."""

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest
from PySide6.QtCore import QCoreApplication, QLocale
from PySide6.QtWidgets import QLabel
from pytestqt.qtbot import QtBot

from context_synthetic_recognition.gui import app as gui_app
from context_synthetic_recognition.gui import i18n
from context_synthetic_recognition.gui.i18n import (
    LANGUAGES,
    SYSTEM,
    TRANSLATIONS,
    current_language,
    mark,
    resolve_language,
    set_language,
    tr,
)
from context_synthetic_recognition.gui.pages import Page
from context_synthetic_recognition.gui.settings import Settings
from context_synthetic_recognition.gui.window import SHORTCUTS, MainWindow

from ..conftest import ROOT
from .conftest import Run

pytestmark = pytest.mark.gui

TRANSLATED = ("ru", "uz")
PLACEHOLDER = re.compile(r"\{[^{}]*\}")


def load_script() -> ModuleType:
    """``scripts/update_translations.py`` (the scripts are not a package)."""
    spec = importlib.util.spec_from_file_location(
        "update_translations", ROOT / "scripts" / "update_translations.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script()
TEXTS: dict[str, list[str]] = script.source_texts()


# ---------------------------------------------------------------- the translation files


def test_the_interface_texts_are_collected_from_tr_and_mark() -> None:
    assert len(TEXTS) > 250
    assert "Dataset" in TEXTS  # a page title: marked where it is defined
    assert "Open a dataset" in TEXTS  # a shortcut of Help → Keyboard shortcuts
    assert all(what in TEXTS for _keys, what in SHORTCUTS)
    assert all(page.title in TEXTS for page in Page.__subclasses__())
    assert mark("anything") == "anything"
    # a tr() call with a variable shows a text that was marked elsewhere
    variables = {line.split(": tr(")[1].rstrip(")") for line in script.unmarked_calls()}
    assert variables == {
        "self.title",
        "page.title",
        "name",
        "what",
        "DATASET_FILES",
        "CONFIG_FILES",
    }


@pytest.mark.parametrize("code", TRANSLATED)
def test_every_text_is_translated(code: str) -> None:
    assert script.problems(code, TEXTS) == []
    translations = script.read_ts(TRANSLATIONS / f"csr_{code}.ts")["csr"]
    assert list(translations) == list(TEXTS)  # the order of the sources, so diffs stay small
    for source, translation in translations.items():
        # what the program fills in or relies on is kept: placeholders, the menu accelerator,
        # the parts of a file filter, the paragraphs
        assert sorted(PLACEHOLDER.findall(translation)) == sorted(PLACEHOLDER.findall(source))
        assert translation.count("&") == source.count("&"), source
        assert translation.count(";;") == source.count(";;"), source
        assert translation.count("\n") == source.count("\n"), source
        assert translation == translation.strip(), source
        for pattern in re.findall(r"\(\*[^)]*\)", source):
            assert pattern in translation, source  # the file patterns of a filter


def test_the_translations_differ_from_english_and_from_each_other() -> None:
    russian = script.read_ts(TRANSLATIONS / "csr_ru.ts")["csr"]
    uzbek = script.read_ts(TRANSLATIONS / "csr_uz.ts")["csr"]
    same = [text for text in TEXTS if russian[text] == text]
    assert set(same) <= {
        "JSON",
        "TUPLAM = {tuplam}",
        "✗ {message}",
        "score₁ = |B1|/|K1| = {s1:.4f},  score₂ = |B2|/|K2| = {s2:.4f}",
    }
    assert sum(russian[text] != uzbek[text] for text in TEXTS) > 270
    cyrillic = re.compile("[а-яё]", re.IGNORECASE)
    assert sum(bool(cyrillic.search(russian[text])) for text in TEXTS) > 270
    assert not any(cyrillic.search(uzbek[text]) for text in TEXTS)  # Uzbek in Latin script
    assert uzbek["Open a dataset"] == "Maʼlumotlar toʻplamini ochish"


@pytest.mark.parametrize("code", TRANSLATED)
def test_the_compiled_translation_is_up_to_date(qapp: QCoreApplication, code: str) -> None:
    # the application loads the .qm; it must say what the .ts says (scripts/update_translations.py)
    translations = script.read_ts(TRANSLATIONS / f"csr_{code}.ts")["csr"]
    assert set_language(qapp, code) == code
    assert current_language() == code
    stale = [text for text in TEXTS if tr(text) != translations[text]]
    assert stale == []
    assert set_language(qapp, "en") == "en"
    assert tr("Dataset") == "Dataset"


def test_updating_keeps_translations_and_drops_unused_texts(tmp_path: Path) -> None:
    existing = {
        "csr": {"Dataset": "Набор данных", "Gone": "Удалено"},
        "QPlatformTheme": {"OK": "OK"},
    }
    merged = script.merged(existing, {"New <text> & more": ["a.py:1"], "Dataset": ["b.py:2"]})
    assert merged == {
        "csr": {"New <text> & more": "", "Dataset": "Набор данных"},
        "QPlatformTheme": {"OK": "OK"},
    }
    path = tmp_path / "csr_xx.ts"
    script.write_ts(path, "xx_XX", merged)
    assert b"\r" not in path.read_bytes()
    text = path.read_text(encoding="utf-8")
    assert "<source>New &lt;text&gt; &amp; more</source>" in text
    assert '<translation type="unfinished"></translation>' in text
    assert script.read_ts(path) == merged
    assert script.read_ts(tmp_path / "missing.ts") == {}


# ---------------------------------------------------------------- choosing the language


def test_the_language_setting(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    assert list(LANGUAGES) == ["en", "ru", "uz"]
    assert LANGUAGES["uz"] == "Oʻzbekcha"
    assert settings.language == SYSTEM
    settings.language = "uz"
    assert settings.language == "uz"
    settings.store.setValue("appearance/language", "klingon")
    assert settings.language == SYSTEM  # an unknown value: follow the system
    assert resolve_language("ru") == "ru"
    for locale, expected in (
        ("ru_RU", "ru"),
        ("uz_UZ", "uz"),
        ("uz_Cyrl_UZ", "uz"),
        ("de_DE", "en"),
    ):
        monkeypatch.setattr(QLocale, "name", lambda _self, value=locale: value)
        assert i18n.system_language() == expected
        assert resolve_language(SYSTEM) == expected
        assert resolve_language(None) == expected


def texts_of(window: MainWindow) -> dict[str, list[str]]:
    return {
        "menus": [action.text() for action in window.menuBar().actions()],
        "sidebar": [window.sidebar.item(i).text() for i in range(window.sidebar.count())],
        "titles": [
            label.text()
            for page in window.pages
            for label in page.findChildren(QLabel)
            if label.objectName() == "pageTitle"
        ],
    }


def test_a_window_built_in_russian(
    qapp: QCoreApplication, settings: Settings, qtbot: QtBot
) -> None:
    set_language(qapp, "ru")
    window = MainWindow(settings)
    qtbot.addWidget(window)
    shown = texts_of(window)
    assert shown["menus"] == ["&Файл", "&Правка", "&Запуск", "&Вид", "&Справка"]
    assert shown["sidebar"][0] == "1   Набор данных"
    assert shown["titles"] == [
        "Набор данных",
        "Настройка",
        "Запуск",
        "Результаты",
        "Новый объект",
        "Сравнение",
        "Экспорт",
    ]
    assert "Открыть набор данных" in window.shortcuts_text()
    assert "Контекстно-синтетическая модель" in window.about_text()
    # a language is listed under its own name
    assert [a.text() for a in window.language_actions.values()] == [
        "Язык системы",
        "English",
        "Русский",
        "Oʻzbekcha",
    ]
    window.close()


def test_switching_the_language_keeps_the_work(
    settings: Settings, run: Run, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    window = MainWindow(settings)  # replaced below, so it is not a fixture's to tear down
    window.show_message = lambda title, text: None
    assert window.dataset_page.load("heart-disease-10")
    run(window)
    state = window.state
    window.configure_page.apply_preset("article")  # an undoable edit; the run becomes stale
    window.state.select_object(2)
    assert window.go_to(window.results_page)
    dataset, view, config = state.dataset, state.view, state.config
    assert texts_of(window)["menus"][0] == "&File"
    assert window.language_actions[SYSTEM].isChecked()

    with qtbot.waitSignal(window.languageChanged) as signal:
        assert window.set_language("uz")
    assert signal.args == ["uz"]
    assert settings.language == "uz"
    assert "started again" in window.statusBar().currentMessage()  # nobody rebuilds the window

    replacement = gui_app.rebuild_window(window)
    replacement.show_message = lambda title, text: None
    assert current_language() == "uz"
    assert replacement is not window
    assert replacement.settings is settings
    # the same state object: nothing was computed or loaded again
    assert replacement.state is state
    assert state.parent() is replacement
    assert (state.dataset, state.view, state.config) == (dataset, view, config)
    assert state.stale
    assert state.selected_object == 2
    assert state.undo_stack.canUndo()
    assert replacement.current_page() is replacement.results_page
    shown = texts_of(replacement)
    assert shown["menus"] == ["&Fayl", "&Tahrirlash", "&Hisoblash", "&Koʻrinish", "&Yordam"]
    assert shown["titles"][3] == "Natijalar"
    assert replacement.language_actions["uz"].isChecked()
    assert "S₃" in replacement.results_page.objects.currentText()  # still selected
    assert "Sozlash" in replacement.results_page.stale.text()
    assert not window.isVisible()

    # the pages work on the handed-over state: undo, run again, export
    state.undo_stack.undo()
    assert not state.stale
    run(replacement)
    assert state.view is not view
    assert (
        replacement.dataset_label.text()
        == "Heart-Disease (10, 13, 2) · obyektlar: 10 · alomatlar: 13"
    )

    assert replacement.set_language("en")
    english = gui_app.rebuild_window(replacement)
    qtbot.addWidget(english)
    assert current_language() == "en"
    assert texts_of(english)["menus"][0] == "&File"
    assert english.state is state
    assert not english.set_language("en")  # nothing to change
    english.close()


def test_the_language_stays_while_a_run_is_in_progress(
    loaded: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loaded.run_page, "is_running", lambda: True)
    assert loaded.is_busy()
    assert not loaded.set_language("ru")
    assert loaded.settings.language == SYSTEM
    assert loaded.language_actions[SYSTEM].isChecked()
    assert "before changing the language" in loaded.statusBar().currentMessage()
    monkeypatch.setattr(loaded.run_page, "is_running", lambda: False)
    assert not loaded.is_busy()


def test_the_launched_application_follows_the_language_menu(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, settings: Settings
) -> None:
    monkeypatch.chdir(tmp_path)
    settings.language = "ru"
    shown: list[MainWindow] = []
    monkeypatch.setattr(MainWindow, "show", lambda self: shown.append(self))
    monkeypatch.setattr(MainWindow, "isVisible", lambda self: True)
    monkeypatch.setattr(gui_app.QApplication, "exec", lambda self=None: 0)
    monkeypatch.setattr(gui_app, "Settings", lambda _=None: settings)
    assert gui_app.launch(dataset="heart-disease-10") == 0
    assert current_language() == "ru"  # the remembered language, before the window is built
    first = shown[-1]
    assert first.replaced_on_language_change
    assert texts_of(first)["menus"][0] == "&Файл"
    first.language_actions["uz"].trigger()  # View → Language → Oʻzbekcha
    second = shown[-1]
    assert second is not first
    assert texts_of(second)["menus"][0] == "&Fayl"
    assert second.state.dataset is not None
    assert second.statusBar().currentMessage() == ""  # no "start again": it changed at once
    second.language_actions["en"].trigger()  # the replacement follows the menu too
    third = shown[-1]
    assert texts_of(third)["menus"][0] == "&File"
    third.close()
