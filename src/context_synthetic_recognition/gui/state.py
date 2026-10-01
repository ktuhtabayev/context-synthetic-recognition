"""The state every page shares: the dataset, the configuration, the result, the selected object.

Pages do not talk to each other; they change :class:`AppState` and listen to its signals. The
configuration is changed only through :meth:`AppState.set_config`, which records an undoable
command, so Edit → Undo/Redo covers every edit of the Configure page and the presets. The
``dataset`` section of the configuration belongs to the loaded dataset: it changes with
:meth:`AppState.set_dataset` and is never touched by undo.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from context_synthetic_recognition.config import DatasetConfig, ExperimentConfig
from context_synthetic_recognition.data import Dataset
from context_synthetic_recognition.export.view import RunView


class _SetConfig(QUndoCommand):
    """Replace the configuration (undoable)."""

    def __init__(
        self, state: AppState, before: ExperimentConfig, after: ExperimentConfig, text: str
    ) -> None:
        super().__init__(text)
        self._state = state
        self._before = before
        self._after = after

    def redo(self) -> None:
        self._state._apply_config(self._after)

    def undo(self) -> None:
        self._state._apply_config(self._before)


class AppState(QObject):
    """What the application is working on."""

    datasetChanged = Signal()
    configChanged = Signal()
    resultChanged = Signal()
    """A run finished, was opened, or was dropped because its inputs changed."""
    objectSelected = Signal(int)
    """0-based training object selected in a table or a selector; −1 = none."""
    appearanceChanged = Signal()
    """Theme or zoom changed."""

    def __init__(self, parent: QObject | None = None) -> None:
        """Start with the template preset and nothing loaded."""
        super().__init__(parent)
        self.dataset: Dataset | None = None
        self.dataset_source: str | None = None
        """The file or catalogue name the dataset came from."""
        self.config: ExperimentConfig = ExperimentConfig()
        self.view: RunView | None = None
        """The evaluated run with everything the pages show, or ``None`` before a run."""
        self.run_folder: Path | None = None
        """The run folder of :attr:`view`, if the run was saved or opened from one."""
        self.selected_object: int = -1
        self.theme: str = "light"
        self.font_scale: float = 1.0
        self.undo_stack = QUndoStack(self)

    # ------------------------------------------------------------------ dataset

    def set_dataset(
        self,
        dataset: Dataset | None,
        source: str | None = None,
        section: DatasetConfig | None = None,
    ) -> None:
        """Replace the dataset; a result computed on other data is dropped.

        ``section`` — how the dataset is read — becomes the configuration's ``dataset`` section
        (kept as it is if ``None``).
        """
        self.dataset = dataset
        self.dataset_source = source
        self.selected_object = -1
        self._drop_result()
        self.datasetChanged.emit()
        if section is not None and section != self.config.dataset:
            self.config = self.config.model_copy(update={"dataset": section})
            self.configChanged.emit()

    # ------------------------------------------------------------------ configuration

    def set_config(self, config: ExperimentConfig, text: str = "Change the configuration") -> None:
        """Replace the configuration through the undo stack (no-op if nothing changed)."""
        config = config.model_copy(update={"dataset": self.config.dataset})
        if config == self.config:
            return
        self.undo_stack.push(_SetConfig(self, self.config, config, text))

    def reset_config(self, config: ExperimentConfig) -> None:
        """Replace the method settings and forget the undo history (a file was opened)."""
        self.undo_stack.clear()
        self._apply_config(config)

    def _apply_config(self, config: ExperimentConfig) -> None:
        # the dataset section follows the loaded dataset, not the undo history
        self.config = config.model_copy(update={"dataset": self.config.dataset})
        self.configChanged.emit()

    # ------------------------------------------------------------------ result

    @property
    def stale(self) -> bool:
        """Whether the result on screen was computed with another configuration or dataset."""
        view = self.view
        if view is None:
            return False
        return view.config != self.config or view.dataset is not self.dataset

    def set_result(self, view: RunView, run_folder: Path | None = None) -> None:
        """Show an evaluated run."""
        self.view = view
        self.run_folder = run_folder
        self.resultChanged.emit()

    def open_run(self, view: RunView, run_folder: Path) -> None:
        """Show a run repeated from its folder: its dataset and configuration become current."""
        self.dataset = view.dataset
        self.dataset_source = str(run_folder)
        self.selected_object = -1
        self.undo_stack.clear()
        self.config = view.config
        self.view = view
        self.run_folder = run_folder
        self.datasetChanged.emit()
        self.configChanged.emit()
        self.resultChanged.emit()

    def _drop_result(self) -> None:
        if self.view is not None:
            self.view = None
            self.run_folder = None
            self.resultChanged.emit()

    # ------------------------------------------------------------------ selection, appearance

    def select_object(self, index: int) -> None:
        """Select a training object everywhere (cross-highlighting); −1 clears the selection."""
        if index != self.selected_object:
            self.selected_object = index
            self.objectSelected.emit(index)

    def set_appearance(self, theme: str | None = None, font_scale: float | None = None) -> None:
        """Change the theme and/or the zoom."""
        changed = False
        if theme is not None and theme != self.theme:
            self.theme = theme
            changed = True
        if font_scale is not None and font_scale != self.font_scale:
            self.font_scale = font_scale
            changed = True
        if changed:
            self.appearanceChanged.emit()
