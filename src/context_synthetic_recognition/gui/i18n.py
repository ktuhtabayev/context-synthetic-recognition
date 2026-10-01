"""Translation of the user interface.

Every string the user sees passes through :func:`tr`. The interface is English; a translation is a
Qt ``.qm`` file ``csr_<locale>.qm`` (``csr_ru.qm``, ``csr_uz.qm``) in this package's
``translations`` folder, loaded by :func:`install_translator` when the system locale asks for it.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QCoreApplication, QLocale, QTranslator

CONTEXT = "csr"
"""Translation context of every string."""
TRANSLATIONS = Path(__file__).parent / "translations"


def tr(text: str) -> str:
    """The text in the language of the interface (the text itself if there is no translation)."""
    return QCoreApplication.translate(CONTEXT, text)


def install_translator(app: QCoreApplication, locale: str | None = None) -> QTranslator | None:
    """Load the translation for ``locale`` (default: the system's), if the package has one.

    Returns:
        The installed translator (keep a reference to it), or ``None`` for English.
    """
    name = locale or QLocale.system().name()
    translator = QTranslator(app)
    for candidate in (name, name.split("_")[0]):
        if translator.load(f"csr_{candidate}", str(TRANSLATIONS)):
            app.installTranslator(translator)
            return translator
    return None
