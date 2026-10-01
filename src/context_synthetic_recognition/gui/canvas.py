"""A matplotlib figure on screen.

The only module that touches matplotlib's Qt backend: :class:`Canvas` shows one figure, with the
backend's zoom / pan toolbar if wanted, and replaces it when another figure is set.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

MINIMUM_SIZE = (260, 200)
"""A canvas never shrinks below this: a figure with no room cannot lay itself out."""


TOOLBAR_WARNING = (
    "Enum value 'Qt::ApplicationAttribute.AA_UseHighDpiPixmaps' is marked as deprecated"
)
"""matplotlib's toolbar icons read this Qt attribute, which PySide6 marks as deprecated."""


def _backend() -> Any:
    # imported on first use: the backend binds to the Qt binding that is loaded
    # The toolbar's icons are painted in a Qt callback; if warnings are errors there (python -W
    # error), the interpreter crashes. This warning of a third party is not ours to act on.
    warnings.filterwarnings("ignore", message=TOOLBAR_WARNING, category=DeprecationWarning)
    return importlib.import_module("matplotlib.backends.backend_qtagg")


class Canvas(QWidget):
    """One matplotlib figure, redrawn by Qt; zoom, pan and the cursor's coordinates optional."""

    def __init__(
        self, *, toolbar: bool = True, coordinates: bool = True, parent: QWidget | None = None
    ) -> None:
        """An empty canvas."""
        super().__init__(parent)
        self._with_toolbar = toolbar
        self._coordinates = coordinates
        self._figure: Figure | None = None
        self._widgets: list[QWidget] = []
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

    def figure(self) -> Figure | None:
        """The figure shown."""
        return self._figure

    def set_figure(self, figure: Figure | None) -> None:
        """Show a figure (``None`` clears the canvas)."""
        for widget in self._widgets:
            self._layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
        self._widgets = []
        self._figure = figure
        if figure is None:
            return
        backend = _backend()
        canvas: Any = backend.FigureCanvasQTAgg(figure)
        canvas.setMinimumSize(*MINIMUM_SIZE)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        if self._with_toolbar:
            bar: Any = backend.NavigationToolbar2QT(canvas, self, coordinates=self._coordinates)
            self._layout.addWidget(bar)
            self._widgets.append(bar)
        self._layout.addWidget(canvas, 1)
        self._widgets.append(canvas)
        canvas.draw_idle()
