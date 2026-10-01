"""What the application remembers between sessions (QSettings).

Window geometry, theme, zoom, the recent datasets, configurations and run folders, the runs
folder. Tests pass their own ``QSettings`` (an ini file in a temporary folder), so they never
touch the user's settings.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings

from context_synthetic_recognition.gui.theme import FONT_SCALES, PALETTES

ORGANIZATION = "context-synthetic-recognition"
APPLICATION = "csr-gui"
MAX_RECENT = 8

RECENT_DATASETS = "recent/datasets"
RECENT_CONFIGS = "recent/configs"
RECENT_RUNS = "recent/runs"


class Settings:
    """Typed access to the persistent settings."""

    def __init__(self, store: QSettings | None = None) -> None:
        """Wrap ``store``, or the user's settings of this application."""
        self.store = store if store is not None else QSettings(ORGANIZATION, APPLICATION)

    # ------------------------------------------------------------------ appearance

    @property
    def theme(self) -> str:
        """``light`` or ``dark``."""
        value = str(self.store.value("appearance/theme", "light"))
        return value if value in PALETTES else "light"

    @theme.setter
    def theme(self, name: str) -> None:
        self.store.setValue("appearance/theme", name)

    @property
    def font_scale(self) -> float:
        """Zoom of the interface (one of :data:`~.theme.FONT_SCALES`)."""
        try:
            value = float(str(self.store.value("appearance/font_scale", 1.0)))
        except ValueError:
            return 1.0
        return min(FONT_SCALES, key=lambda step: abs(step - value))

    @font_scale.setter
    def font_scale(self, scale: float) -> None:
        self.store.setValue("appearance/font_scale", scale)

    # ------------------------------------------------------------------ window

    def geometry(self) -> QByteArray | None:
        """The saved window geometry, if any."""
        value = self.store.value("window/geometry")
        return value if isinstance(value, QByteArray) else None

    def save_geometry(self, geometry: QByteArray) -> None:
        """Remember the window geometry."""
        self.store.setValue("window/geometry", geometry)

    @property
    def page(self) -> int:
        """The page shown when the window was closed."""
        try:
            return int(str(self.store.value("window/page", 0)))
        except ValueError:
            return 0

    @page.setter
    def page(self, index: int) -> None:
        self.store.setValue("window/page", index)

    # ------------------------------------------------------------------ folders and files

    @property
    def runs_dir(self) -> Path:
        """Where run folders are written and looked for."""
        return Path(str(self.store.value("paths/runs_dir", "runs")))

    @runs_dir.setter
    def runs_dir(self, folder: Path) -> None:
        self.store.setValue("paths/runs_dir", str(folder))

    def recent(self, key: str) -> list[str]:
        """The recent entries of a list, newest first."""
        value = self.store.value(key, [])
        if isinstance(value, str):
            return [value] if value else []
        return [str(item) for item in value] if isinstance(value, list) else []

    def add_recent(self, key: str, entry: str) -> list[str]:
        """Put ``entry`` first in a recent list and return the list."""
        entries = [entry, *(item for item in self.recent(key) if item != entry)][:MAX_RECENT]
        self.store.setValue(key, entries)
        return entries

    def clear_recent(self, key: str) -> None:
        """Forget a recent list."""
        self.store.remove(key)
