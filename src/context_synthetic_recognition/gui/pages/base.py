"""What every page shares."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from context_synthetic_recognition.gui.settings import Settings
from context_synthetic_recognition.gui.state import AppState
from context_synthetic_recognition.gui.theme import Palette, palette
from context_synthetic_recognition.gui.widgets import hint, page_title


class Page(QWidget):
    """A page of the workflow: a title, a one-line hint and its content."""

    title = ""
    """Name in the sidebar and at the top of the page."""
    needs_result = False
    """Whether the page is available only after a run."""

    def __init__(
        self, state: AppState, settings: Settings, subtitle: str, parent: QWidget | None = None
    ) -> None:
        """Create the title row; subclasses add their content to :attr:`body`."""
        super().__init__(parent)
        self.state = state
        self.settings = settings
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 12)
        outer.setSpacing(8)
        self.header = QHBoxLayout()
        self.header.addWidget(page_title(self.title))
        self.header.addStretch(1)
        outer.addLayout(self.header)
        outer.addWidget(hint(subtitle))
        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        outer.addLayout(self.body, 1)
        state.appearanceChanged.connect(self._appearance)

    @property
    def colours(self) -> Palette:
        """The palette of the current theme."""
        return palette(self.state.theme)

    def _appearance(self) -> None:
        self.apply_palette(self.colours)

    def apply_palette(self, colours: Palette) -> None:
        """Re-colour what the style sheet does not reach (tables, figures)."""
