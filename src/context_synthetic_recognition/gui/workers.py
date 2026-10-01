"""Background work: the interface never computes in its own thread.

A :class:`Task` runs one function in a worker thread and reports through signals (which Qt
delivers in the interface's thread): progress, log lines, the result or the error. The functions
themselves — :func:`run_job`, :func:`open_job`, :func:`export_job` — are plain Python on top of the
services, so they are tested without a thread.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PySide6.QtCore import QObject, QThread, Signal

from context_synthetic_recognition.config import ExperimentConfig
from context_synthetic_recognition.data import Dataset
from context_synthetic_recognition.errors import CSRError
from context_synthetic_recognition.evaluation.protocols import EvaluationCancelledError
from context_synthetic_recognition.export import (
    ExportOptions,
    ExportSummary,
    RunView,
    build_view,
    export_view,
)
from context_synthetic_recognition.export.run import AUTO_SENSITIVITY_OBJECTS
from context_synthetic_recognition.log import PACKAGE_LOGGER
from context_synthetic_recognition.services.runner import load_run, run_experiment, save_run
from context_synthetic_recognition.services.sensitivity import SwitchVariant, switch_sensitivity

Progress = Callable[[str, int, int], None]
"""(stage, done, total)."""
Cancelled = Callable[[], bool]
SensitivityChoice = Literal["auto", "yes", "no"]


@dataclass(frozen=True)
class RunOutcome:
    """What a run produced."""

    view: RunView
    folder: Path | None
    """The run folder, if the run was saved."""
    warnings: tuple[str, ...] = ()


def _wanted(choice: SensitivityChoice, dataset: Dataset) -> bool:
    return choice == "yes" or (choice == "auto" and dataset.m <= AUTO_SENSITIVITY_OBJECTS)


def run_job(
    dataset: Dataset,
    config: ExperimentConfig,
    *,
    sensitivity: SensitivityChoice = "auto",
    runs_dir: Path | None = None,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> RunOutcome:
    """Evaluate the experiment, optionally the four switch settings, and save the run.

    Args:
        dataset: The data.
        config: The configuration.
        sensitivity: Evaluate the four switch settings: always, never, or (``auto``) for samples
            of at most :data:`~context_synthetic_recognition.export.run.AUTO_SENSITIVITY_OBJECTS`
            objects.
        runs_dir: Where to write the run folder; ``None`` = do not save.
        progress: Called with (stage, done, total).
        cancelled: Polled between folds; ``True`` stops the run.
    """
    result = run_experiment(dataset, config, progress=progress, cancelled=cancelled)
    variants: tuple[SwitchVariant, ...] | None = None
    if _wanted(sensitivity, dataset):
        variants = switch_sensitivity(dataset, config, progress=progress, cancelled=cancelled)
    folder = save_run(result, runs_dir, sensitivity=variants) if runs_dir is not None else None
    view = build_view(result, sensitivity=variants, run_id=None if folder is None else folder.name)
    return RunOutcome(view, folder)


def open_job(
    folder: Path, *, progress: Progress | None = None, cancelled: Cancelled | None = None
) -> RunOutcome:
    """Repeat a saved run from its folder (:func:`~...services.runner.load_run`)."""
    loaded = load_run(folder, progress=progress, cancelled=cancelled)
    view = build_view(loaded.result, sensitivity=loaded.sensitivity, run_id=folder.name)
    return RunOutcome(view, folder, loaded.warnings)


def export_job(
    view: RunView,
    folder: Path,
    formats: Sequence[str],
    options: ExportOptions,
    *,
    progress: Progress | None = None,
) -> ExportSummary:
    """Write the chosen formats of a run into ``folder``."""
    names = list(formats)

    def report(name: str) -> None:
        if progress is not None:
            progress(name, names.index(name) if name in names else 0, len(names))

    summary = export_view(view, folder, names, options, progress=report)
    if progress is not None:
        progress("done", len(names), len(names))
    return summary


class _SignalHandler(logging.Handler):
    """A logging handler that calls a function with every formatted record."""

    def __init__(self, forward: Callable[[str], None]) -> None:
        super().__init__()
        self._forward = forward
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self._forward(self.format(record))


class LogBridge(QObject):
    """Hands every log record of the package to the interface as a signal.

    The signal is emitted in whatever thread logs; Qt delivers it in the interface's thread.
    """

    message = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        """Format records as ``HH:MM:SS LEVEL message``."""
        super().__init__(parent)
        self.handler = _SignalHandler(self.message.emit)

    def attach(self) -> None:
        """Start listening to the package logger."""
        logger = logging.getLogger(PACKAGE_LOGGER)
        logger.addHandler(self.handler)
        if logger.level == logging.NOTSET or logger.level > logging.INFO:
            logger.setLevel(logging.INFO)

    def detach(self) -> None:
        """Stop listening."""
        logging.getLogger(PACKAGE_LOGGER).removeHandler(self.handler)


class Task(QThread):
    """One function in a worker thread.

    The function receives ``progress`` and ``cancelled`` keyword arguments if it is created with
    :meth:`with_progress`; its return value arrives with :attr:`succeeded`.
    """

    progressed = Signal(str, int, int)
    succeeded = Signal(object)
    failed = Signal(str)
    """An error the user can act on (a :class:`~context_synthetic_recognition.errors.CSRError`)."""
    cancelled = Signal()

    def __init__(self, work: Callable[[Task], Any], parent: QObject | None = None) -> None:
        """Prepare ``work(task)``; :meth:`start` runs it."""
        super().__init__(parent)
        self._work = work
        self._cancel = False

    @classmethod
    def with_progress(
        cls, function: Callable[..., Any], *args: Any, parent: QObject | None = None, **kwargs: Any
    ) -> Task:
        """A task for ``function(*args, progress=…, cancelled=…, **kwargs)``."""
        return cls(
            lambda task: function(
                *args, progress=task.report, cancelled=task.is_cancelled, **kwargs
            ),
            parent,
        )

    def report(self, stage: str, done: int, total: int) -> None:
        """Progress callback handed to the work (any thread)."""
        self.progressed.emit(stage, done, total)

    def is_cancelled(self) -> bool:
        """Whether :meth:`cancel` was called."""
        return self._cancel

    def cancel(self) -> None:
        """Ask the work to stop at its next check."""
        self._cancel = True

    def run(self) -> None:
        """Thread entry point: run the work and report how it ended."""
        try:
            result = self._work(self)
        except EvaluationCancelledError:
            self.cancelled.emit()
        except (CSRError, OSError) as error:
            self.failed.emit(str(error))
        except Exception as error:  # the interface must survive a bug in a worker
            logging.getLogger(PACKAGE_LOGGER).exception("unexpected error in a background task")
            self.failed.emit(f"unexpected error: {type(error).__name__}: {error}")
        else:
            self.succeeded.emit(result)
