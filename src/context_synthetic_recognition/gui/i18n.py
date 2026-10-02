"""Translation of the user interface.

Every string the user sees in the interface passes through :func:`tr`; a constant that is shown
later (a page title, a column name) is marked with :func:`mark` where it is defined. English is
the source language. A translation is a Qt ``.qm`` file ``csr_<code>.qm`` in this package's
``translations`` folder, compiled from the ``csr_<code>.ts`` next to it by
``scripts/update_translations.py``; :func:`set_language` installs it together with Qt's own
translation of the standard dialogs (ADR-054).

Only the interface is translated. What the exporters produce — the tables and figures of the
results, the workbook, the report — stays in English, in the application too.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QCoreApplication, QLibraryInfo, QLocale, QTranslator

CONTEXT = "csr"
"""Translation context of every string."""
TRANSLATIONS = Path(__file__).parent / "translations"
SOURCE_LANGUAGE = "en"
SYSTEM = "system"
"""The setting that follows the language of Windows."""
LANGUAGES: dict[str, str] = {"en": "English", "ru": "Русский", "uz": "Oʻzbekcha"}
"""The languages of the interface: code → the language's own name."""

_installed: list[QTranslator] = []
_current = SOURCE_LANGUAGE


def tr(text: str) -> str:
    """The text in the language of the interface (the text itself if there is no translation)."""
    return QCoreApplication.translate(CONTEXT, text)


def mark(text: str) -> str:
    """Mark a constant as translatable; it is translated with :func:`tr` where it is shown."""
    return text


def system_language() -> str:
    """The language of Windows if the interface has it, else English."""
    code = QLocale.system().name().split("_")[0].lower()
    return code if code in LANGUAGES else SOURCE_LANGUAGE


def resolve_language(choice: str | None) -> str:
    """A language code for a setting: a code of :data:`LANGUAGES`, or the system's language."""
    return choice if choice in LANGUAGES else system_language()


def current_language() -> str:
    """The code of the language the interface is shown in."""
    return _current


def set_language(app: QCoreApplication, choice: str | None = SYSTEM) -> str:
    """Show the interface in a language (texts created from now on).

    Args:
        app: The application.
        choice: A code of :data:`LANGUAGES`; :data:`SYSTEM` or ``None`` follow Windows.

    Returns:
        The code of the language now in use.
    """
    global _current
    for translator in _installed:
        app.removeTranslator(translator)
    _installed.clear()
    code = resolve_language(choice)
    if code != SOURCE_LANGUAGE:
        qt_folder = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        for name, folder in ((f"qtbase_{code}", qt_folder), (f"csr_{code}", str(TRANSLATIONS))):
            translator = QTranslator(app)
            if translator.load(name, folder):
                app.installTranslator(translator)
                _installed.append(translator)
    _current = code
    return code
