"""Background work: the interface never computes in its own thread.

A :class:`Task` runs one function in a worker thread and reports through signals (which Qt
delivers in the interface's thread): progress, log lines, the result or the error. The functions
themselves — :func:`run_job`, :func:`open_job`, :func:`export_job`, :func:`comparison_job`,
:func:`comparison_check_job` and the stages of a classifier part (:func:`part_train_job`,
:func:`part_classify_job`, :func:`part_evaluate_job`, :func:`part_run_all_job` …) — are plain
Python on top of the services, so they are tested without a thread.
"""

from __future__ import annotations

import gc
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PySide6.QtCore import QObject, QThread, Signal

from context_synthetic_recognition.comparison import COMPARISON_SPLITS
from context_synthetic_recognition.config import ExperimentConfig
from context_synthetic_recognition.data import AnswerKey, Dataset, UnlabelledSample
from context_synthetic_recognition.errors import ConfigError, CSRError
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
from context_synthetic_recognition.services import classifier_parts as parts
from context_synthetic_recognition.services.classifier_parts import (
    CheckablePart,
    ClassifierPart,
    Evaluation,
    PartCheck,
    Predictions,
    SampleOption,
    SampleSet,
    SampleTree,
    StageTable,
    TrainedPart,
    WritablePart,
)
from context_synthetic_recognition.services.comparison import (
    ComparisonCheck,
    ComparisonOutcome,
    check_comparison,
    run_comparison,
)
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


def _split_progress(progress: Progress | None) -> Callable[[str], None] | None:
    """The service's line of progress as (stage, done, total) — one step per split."""
    if progress is None:
        return None
    done = [0]

    def report(line: str) -> None:
        progress(line, done[0], len(COMPARISON_SPLITS))
        done[0] += 1

    return report


def comparison_job(
    splits: Path | None, *, progress: Progress | None = None, cancelled: Cancelled | None = None
) -> ComparisonOutcome:
    """The comparison tables on the split files of ``splits`` (the project's by default).

    Nothing is written.

    A run takes a few seconds and is not interrupted (``cancelled`` is not polled).
    """
    del cancelled
    outcome = run_comparison(splits, progress=_split_progress(progress))
    if progress is not None:
        progress("done", len(COMPARISON_SPLITS), len(COMPARISON_SPLITS))
    return outcome


def comparison_check_job(
    folder: Path,
    splits: Path | None,
    *,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> ComparisonCheck:
    """Check the comparison written to ``folder`` against its checksums and a new run."""
    del cancelled
    check = check_comparison(folder, splits=splits, progress=_split_progress(progress))
    if progress is not None:
        progress("done", len(COMPARISON_SPLITS), len(COMPARISON_SPLITS))
    return check


# ---------------------------------------------------------------- the Classifiers page (ADR-084)


@dataclass(frozen=True, eq=False)
class TrainingStep:
    """① Training of a classifier part, with the tabs to show."""

    samples: SampleSet
    trained: TrainedPart
    tabs: tuple[StageTable, ...]
    """The tabs of the *Training* step."""
    test_tabs: tuple[StageTable, ...]
    """The tabs of the *Test* step before the objects are classified (*Test objects*)."""


@dataclass(frozen=True, eq=False)
class TestStep:
    """② Test: the decisions for the objects classified without their class, with the tabs."""

    predictions: Predictions
    tabs: tuple[StageTable, ...]


@dataclass(frozen=True, eq=False)
class EvaluationStep:
    """③ Evaluation: the decisions against the true classes, with the tabs."""

    evaluation: Evaluation
    tabs: tuple[StageTable, ...]
    write_problem: str = ""
    """Why the evaluation cannot be written (empty if it can, or if the part writes nothing)."""


@dataclass(frozen=True, eq=False)
class StagesDone:
    """The three steps run in a row."""

    training: TrainingStep
    test: TestStep
    evaluation: EvaluationStep


@dataclass(frozen=True)
class CheckOutcome:
    """A check of written results, with its tab."""

    check: PartCheck
    tab: StageTable


def part_samples_job(
    part: ClassifierPart,
    folder: Path | None,
    *,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> SampleTree:
    """What a part can run on: its tree of samples, read from ``folder`` (ADR-091)."""
    del cancelled
    tree = part.samples(folder)
    if progress is not None:
        progress("done", 1, 1)
    return tree


def _training_step(part: ClassifierPart, samples: SampleSet, trained: TrainedPart) -> TrainingStep:
    keys = [s.key for s in samples.training]
    labels = [s.title for s in samples.training]
    return TrainingStep(
        samples,
        trained,
        parts.training_view(part, trained.option, trained),
        (parts.unlabelled_table(samples.test, keys, labels),),
    )


def _evaluation_step(part: ClassifierPart, evaluation: Evaluation) -> EvaluationStep:
    problem = part.write_problem(evaluation) if isinstance(part, WritablePart) else ""
    return EvaluationStep(evaluation, parts.evaluation_view(evaluation), problem)


def part_train_job(
    part: ClassifierPart,
    option: SampleOption,
    folder: Path | None,
    *,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> TrainingStep:
    """① Read the samples and fit the part's methods on the training samples."""
    del cancelled
    samples, trained = parts.train(part, option, folder, progress)
    return _training_step(part, samples, trained)


def part_classify_job(
    part: ClassifierPart,
    trained: TrainedPart,
    unlabelled: tuple[UnlabelledSample, ...],
    *,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> TestStep:
    """② Classify the test objects from their feature values only."""
    del cancelled
    predictions = parts.classify(part, trained, unlabelled, progress)
    return TestStep(predictions, parts.prediction_view(predictions))


def part_evaluate_job(
    part: ClassifierPart,
    predictions: Predictions,
    answers: tuple[AnswerKey, ...],
    *,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> EvaluationStep:
    """③ Compare the decisions with the true classes."""
    del cancelled
    return _evaluation_step(part, parts.evaluate(part, predictions, answers, progress))


def part_run_all_job(
    part: ClassifierPart,
    option: SampleOption,
    folder: Path | None,
    *,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> StagesDone:
    """The three steps in a row; the progress names the stage and the split."""
    del cancelled
    run = parts.run_stages(part, option, folder, progress)
    return StagesDone(
        _training_step(part, run.samples, run.trained),
        TestStep(run.predictions, parts.prediction_view(run.predictions)),
        _evaluation_step(part, run.evaluation),
    )


def part_write_job(
    part: ClassifierPart,
    evaluation: Evaluation,
    folder: Path,
    *,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> list[Path]:
    """Write a part's evaluation to a results folder."""
    del cancelled
    if not isinstance(part, WritablePart):
        raise ConfigError(f"{part.title}: this part writes no files")
    written = part.write(evaluation, folder)
    if progress is not None:
        progress("done", 1, 1)
    return written


def part_check_job(
    part: ClassifierPart,
    folder: Path,
    splits: Path | None,
    *,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> CheckOutcome:
    """Check a results folder against its checksums and a new run."""
    del cancelled
    if not isinstance(part, CheckablePart):
        raise ConfigError(f"{part.title}: this part checks no files")
    checked = part.check(folder, splits, progress)
    return CheckOutcome(checked, parts.check_tab(checked))


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


class CollectorPause:
    """Python's cyclic garbage collector is off while any worker works (ADR-089).

    A collection runs in whichever thread allocates when a threshold is reached. In a worker it
    would free the wrappers of Qt objects left for collection — the widgets of a closed window —
    and so destroy them outside the interface's thread, which crashes Qt with an access
    violation. The collector is therefore switched off while a worker runs (the last worker to
    end switches it back on), and :meth:`Task.start` collects in the interface's thread first.
    """

    def __init__(self) -> None:
        """No worker yet."""
        self._lock = threading.Lock()
        self._workers = 0
        self._was_enabled = True

    @property
    def workers(self) -> int:
        """Workers running now."""
        return self._workers

    def __enter__(self) -> None:
        """A worker starts: the first one switches the collector off."""
        with self._lock:
            if self._workers == 0:
                self._was_enabled = gc.isenabled()
                gc.disable()
            self._workers += 1

    def __exit__(self, *_: object) -> None:
        """A worker ends: the last one switches the collector back on (if it was on)."""
        with self._lock:
            self._workers -= 1
            if self._workers == 0 and self._was_enabled:
                gc.enable()


COLLECTOR = CollectorPause()
"""The pause of the garbage collector shared by every :class:`Task`."""


class Task(QThread):
    """One function in a worker thread.

    The function receives ``progress`` and ``cancelled`` keyword arguments if it is created with
    :meth:`with_progress`; its return value arrives with :attr:`succeeded`. Python's garbage
    collector does not run while the work runs (:class:`CollectorPause`).
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

    def start(self, priority: QThread.Priority = QThread.Priority.InheritPriority) -> None:
        """Collect garbage in the interface's thread, then start the worker."""
        gc.collect()
        super().start(priority)

    def run(self) -> None:
        """Thread entry point: run the work and report how it ended (no collection meanwhile)."""
        with COLLECTOR:
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
