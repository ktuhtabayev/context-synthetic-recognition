"""Classifiers page: classifier parts in three steps — training, test, evaluation (ADR-084).

Apart from the CS-model's workflow: the page needs no dataset and no run. On the left, the list of
the registered parts (:data:`~context_synthetic_recognition.services.classifier_parts.
CLASSIFIER_PARTS`), collapsible like the stage tree of the *Results* page; on the right, the
:class:`~context_synthetic_recognition.gui.workbench.PartView` of the chosen part. The first part
is the comparison with standard classifiers (ADR-079, ADR-080): ``csr compare run`` and
``csr compare check`` in three steps.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from context_synthetic_recognition.gui.i18n import mark, tr
from context_synthetic_recognition.gui.pages.base import Page
from context_synthetic_recognition.gui.settings import Settings
from context_synthetic_recognition.gui.state import AppState
from context_synthetic_recognition.gui.theme import Palette
from context_synthetic_recognition.gui.widgets import section_title
from context_synthetic_recognition.gui.workbench import PartView
from context_synthetic_recognition.services.classifier_parts import parts

PARTS_WIDTH = 190
"""Width of the open list of parts."""


class ClassifiersPage(Page):
    """The parts of the *Classifiers* page, each a staged workbench."""

    title = mark("Classifiers")
    finished = Signal()
    """A worker of the current part ended (successfully or not)."""

    def __init__(self, state: AppState, settings: Settings, parent: QWidget | None = None) -> None:
        """Build the list of parts and their views."""
        super().__init__(
            state,
            settings,
            tr(
                "Classifier parts in three steps: training on the training sample, the test "
                "objects classified without their class, and the comparison with the true "
                "classes. Apart from the workflow: no dataset or run is needed."
            ),
            parent,
        )
        self.parts = parts()
        self.views = [PartView(part) for part in self.parts]
        self.list = QListWidget()
        self.list.setObjectName("parts")
        for part in self.parts:
            item = QListWidgetItem(part.title)
            item.setToolTip(part.description)
            self.list.addItem(item)
        self.list.currentRowChanged.connect(self._part_chosen)
        self.toggle = QPushButton()
        self.toggle.setObjectName("partsToggle")
        self.toggle.clicked.connect(lambda: self.set_parts_open(not self.list.isVisible()))
        self.heading = section_title(tr("Parts"))
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.heading, 1)
        top.addWidget(self.toggle, 0, Qt.AlignmentFlag.AlignTop)
        side = QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        side.addLayout(top)
        side.addWidget(self.list, 1)
        self.side = QWidget()
        self.side.setLayout(side)
        self.stack = QStackedWidget()
        for view in self.views:
            view.finished.connect(self.finished)
            self.stack.addWidget(view)
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(self.side)
        row.addWidget(self.stack, 1)
        self.body.addLayout(row, 1)
        self.list.setCurrentRow(0)
        # with one part the list starts collapsed: a narrow strip that opens it
        self.set_parts_open(len(self.parts) > 1)
        self.apply_palette(self.colours)

    # ------------------------------------------------------------------ the parts

    @property
    def view(self) -> PartView:
        """The view of the chosen part."""
        return self.views[max(self.list.currentRow(), 0)]

    def set_parts_open(self, opened: bool) -> None:
        """Open the list of parts, or collapse it to a narrow strip."""
        self.list.setVisible(opened)
        self.heading.setVisible(opened)
        self.side.setFixedWidth(PARTS_WIDTH if opened else self.toggle.sizeHint().width())
        self.toggle.setText("◀" if opened else "▶")
        self.toggle.setToolTip(tr("Hide the parts") if opened else tr("Show the parts"))

    def _part_chosen(self, row: int) -> None:
        if 0 <= row < len(self.views):
            self.stack.setCurrentIndex(row)

    def choose(self, key: str) -> bool:
        """Show the part registered as ``key``."""
        for row, part in enumerate(self.parts):
            if part.key == key:
                self.list.setCurrentRow(row)
                return True
        return False

    # ------------------------------------------------------------------ for the window

    def is_running(self) -> bool:
        """Whether a part's worker runs."""
        return any(view.is_running() for view in self.views)

    def wait(self, milliseconds: int = 300_000) -> bool:
        """Block until every part's worker ends (the tests; closing the window)."""
        return all(view.wait(milliseconds) for view in self.views)

    def step(self, direction: int) -> bool:
        """The next (+1) or previous (−1) step of the chosen part (Alt+Right / Alt+Left)."""
        return self.view.step(direction)

    def apply_palette(self, colours: Palette) -> None:
        """Re-colour the tables."""
        for view in self.views:
            view.set_palette(colours)
