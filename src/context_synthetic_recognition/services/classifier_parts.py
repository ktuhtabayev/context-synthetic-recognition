"""Classifier parts: units of work of the *Classifiers* page in three stages (ADR-084).

A part trains methods on a training sample (① *Training*), classifies the test objects without
their class (② *Test*) and compares the decisions with the true classes (③ *Evaluation*). The
desktop application shows every part of :data:`CLASSIFIER_PARTS` from this protocol alone; adding
a part means registering it, the page does not change ("Adding a part to the Classifiers page" in
``docs/extending.md``).

The stages receive what the article allows them to see: :meth:`ClassifierPart.train` the
training samples with their classes; :meth:`ClassifierPart.classify` the trained part and
:class:`~context_synthetic_recognition.data.samples.UnlabelledSample` objects, which have no
class; :meth:`ClassifierPart.evaluate` the decisions and the
:class:`~context_synthetic_recognition.data.samples.AnswerKey` objects (ADR-085).

The tabs every part gets are built here from the stage results — *Methods*, *Training sample*
and *Training results*; *Test objects*, *Predictions* and *Predicted classes*; *Per object* and
*Confusion* — and a part adds its own through the ``tables`` of its results. Pure computation on
the stage results; reading files is the parts' :meth:`ClassifierPart.load`.
"""

from __future__ import annotations

import importlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from context_synthetic_recognition.core.arrays import IntArray
from context_synthetic_recognition.core.registry import Registry
from context_synthetic_recognition.data.samples import (
    AnswerKey,
    Decisions,
    UnlabelledSample,
    require_unlabelled,
)
from context_synthetic_recognition.data.schema import Dataset
from context_synthetic_recognition.evaluation.metrics import (
    ClassificationMetrics,
    classification_metrics,
)
from context_synthetic_recognition.export.tables import Cell, Table
from context_synthetic_recognition.export.text import csv_rows

Progress = Callable[[str, int, int], None]
"""(what is being done, done, total) — e.g. ``("336", 0, 3)``."""

GROUP = "Classifiers"
"""Group of the tables of this module."""
REFUSAL = 0
"""The decision code of a refusal."""


# ---------------------------------------------------------------- what the page shows


class CellMeaning(StrEnum):
    """What the cells of a column mean; the page colours by meaning, never by colour alone."""

    PLAIN = "plain"
    QUANTITATIVE = "quantitative"
    """A quantitative feature (the header is coloured)."""
    NOMINAL = "nominal"
    """A nominal feature (the header is coloured)."""
    CLASS = "class"
    """A class code: 1 = K1 (blue), 2 = K2 (orange)."""
    DECISION = "decision"
    """A decision: 1 = K1, 2 = K2, 0 = refusal (grey) — not whether it is correct."""
    OUTCOME = "outcome"
    """A decision against the true class: ``✓ 1`` (green), ``✗ 2`` (red), ``0`` (grey)."""
    STATUS = "status"
    """``✓`` / ``✗`` of a check."""


@dataclass(frozen=True)
class ShownTable:
    """A table with the meaning of its columns."""

    table: Table
    cells: tuple[CellMeaning, ...]
    """One meaning per column."""
    label: str = ""
    """The variant's name in a tab with several tables (the split, e.g. ``336``)."""
    object_column: int | None = None
    """The column that names the object of a row (cross-highlighting)."""
    decimals: int | None = None
    """Decimals to show (``None``: the page's default)."""

    def __post_init__(self) -> None:
        """One meaning per column."""
        if len(self.cells) != len(self.table.columns):
            raise ValueError(
                f"{self.table.key}: {len(self.cells)} meanings for {len(self.table.columns)} "
                "columns"
            )


@dataclass(frozen=True)
class StageTable:
    """A tab of a stage: one table, one per split, or why there is none."""

    key: str
    title: str
    """The tab's name (English; the page translates the names it knows)."""
    variants: tuple[ShownTable, ...] = ()
    missing: str = ""
    """Shown instead of a table (why there is none)."""


@dataclass(frozen=True)
class SampleOption:
    """What a part can run on: one or several splits."""

    key: str
    title: str
    """E.g. ``All article splits (336, 341, 367)``."""
    splits: tuple[str, ...]
    """Keys of the splits."""
    sizes: str = ""
    """E.g. ``training 336 (196, 140) · test 664``."""
    available: bool = True
    reason: str = ""
    """Why the option is unavailable."""
    complete: bool = False
    """Whether it is the full set the part's own tables and files need."""

    @property
    def label(self) -> str:
        """The entry of the sample selector: the title and the sizes."""
        return f"{self.title} · {self.sizes}" if self.sizes else self.title


@dataclass(frozen=True)
class MethodInfo:
    """A method of a part with its fixed settings (read-only on the page)."""

    key: str
    title: str
    features: str
    """E.g. ``quantitative (7)``."""
    splits: tuple[str, ...]
    """The splits it runs on."""


@dataclass(frozen=True)
class MethodGroup:
    """Methods grouped as in the article, with their fixed settings."""

    title: str
    settings: str
    methods: tuple[MethodInfo, ...]


@dataclass(frozen=True, eq=False)
class TrainingSample:
    """The training sample of one split, with its classes."""

    key: str
    """The split's key, e.g. ``336``."""
    title: str
    """E.g. ``336 — Zhuravlyov split``."""
    data: Dataset
    """The training objects (named by their place in the file) with their classes."""
    file: str = ""
    """The file it was read from, relative to the split folder."""
    sha256: str = ""
    """SHA-256 of that file."""


@dataclass(frozen=True, eq=False)
class SampleSet:
    """What :meth:`ClassifierPart.load` reads: one entry per split, in the same order."""

    option: SampleOption
    training: tuple[TrainingSample, ...]
    test: tuple[UnlabelledSample, ...]
    """The test objects without their class (the input of *Test*)."""
    answers: tuple[AnswerKey, ...]
    """Their classes, kept apart (the input of *Evaluation* only)."""

    def __post_init__(self) -> None:
        """One training sample, test sample and answer key per split, for the same objects."""
        if not len(self.training) == len(self.test) == len(self.answers):
            raise ValueError("a sample set needs one training sample, test sample and key each")
        for sample, key in zip(self.test, self.answers, strict=True):
            if not key.matches(sample):
                raise ValueError(
                    f"{key.name}: the answer key is not for the objects of {sample.name}"
                )


@dataclass(frozen=True, eq=False)
class MethodTraining:
    """One method on one training sample: its resubstitution decisions (Definition 2)."""

    method: str
    """The method's key."""
    title: str
    decisions: IntArray | None
    """(training objects,) 1, 2 or 0; ``None`` if the method is undefined."""
    undefined: str = ""
    """Why the method is undefined on this training sample."""


@dataclass(frozen=True, eq=False)
class TrainedPart:
    """The result of ① *Training*."""

    part: str
    option: SampleOption
    samples: tuple[TrainingSample, ...]
    methods: Mapping[str, tuple[MethodTraining, ...]]
    """Split key → every method's training decisions, in the part's order."""
    fitted: Any = field(default=None, repr=False)
    """The part's fitted methods (what its :meth:`ClassifierPart.classify` uses)."""
    tables: tuple[StageTable, ...] = ()
    """The part's own tabs of the *Training* step."""


@dataclass(frozen=True, eq=False)
class Predictions:
    """The result of ② *Test*: the decisions for objects classified without their class."""

    part: str
    trained: TrainedPart
    objects: tuple[UnlabelledSample, ...]
    """One per split, in the order of the training samples."""
    decisions: Mapping[str, Mapping[str, Decisions]]
    """Split key → method key → decisions, in the part's order."""
    tables: tuple[StageTable, ...] = ()

    @property
    def option(self) -> SampleOption:
        """The samples the part was trained on."""
        return self.trained.option


@dataclass(frozen=True, eq=False)
class Evaluation:
    """The result of ③ *Evaluation*: the decisions against the true classes."""

    part: str
    predictions: Predictions
    answers: tuple[AnswerKey, ...]
    metrics: Mapping[str, Mapping[str, ClassificationMetrics | None]]
    """Split key → method key → metrics on the test objects (``None``: undefined)."""
    tables: tuple[StageTable, ...] = ()
    """The part's own tabs of the *Evaluation* step (first)."""
    result: Any = field(default=None, repr=False)
    """The part's own results (what its :meth:`WritablePart.write` writes)."""

    @property
    def option(self) -> SampleOption:
        """The samples the part was trained on."""
        return self.predictions.option


@dataclass(frozen=True)
class PartCheck:
    """Written results checked against a new run."""

    folder: Path
    passed: bool
    table: Table
    """Every file with ✓ or ✗ (the *Check* tab)."""
    problems: tuple[str, ...] = ()


# ---------------------------------------------------------------- the protocol


class ClassifierPart(Protocol):
    """A unit of work of the *Classifiers* page: train, classify blind, evaluate."""

    key: str
    """Registry name."""
    title: str
    description: str
    """One line under the title."""

    def default_folder(self) -> Path | None:
        """The folder the samples are read from by default (``None``: the part needs none)."""
        ...

    def sample_options(self, folder: Path | None) -> tuple[SampleOption, ...]:
        """What the part can run on, with sizes — and why an option is unavailable."""
        ...

    def load(self, option: SampleOption, folder: Path | None) -> SampleSet:
        """Read the samples of an option (raises :class:`~...errors.CSRError` on bad files)."""
        ...

    def method_groups(self, option: SampleOption) -> tuple[MethodGroup, ...]:
        """The methods grouped as in the article, each with its fixed settings."""
        ...

    def train(
        self,
        option: SampleOption,
        training: tuple[TrainingSample, ...],
        progress: Progress | None = None,
    ) -> TrainedPart:
        """① Fit every method on the training samples (their classes are known)."""
        ...

    def classify(
        self,
        trained: TrainedPart,
        unlabelled: tuple[UnlabelledSample, ...],
        progress: Progress | None = None,
    ) -> Predictions:
        """② Classify objects from their feature values only: they have no class."""
        ...

    def evaluate(
        self,
        predictions: Predictions,
        answers: tuple[AnswerKey, ...],
        progress: Progress | None = None,
    ) -> Evaluation:
        """③ Compare the decisions with the true classes."""
        ...


@runtime_checkable
class WritablePart(Protocol):
    """A part that writes its evaluation to a results folder."""

    def default_output(self) -> Path:
        """The results folder by default."""
        ...

    def write_problem(self, evaluation: Evaluation) -> str:
        """Why this evaluation cannot be written (empty if it can)."""
        ...

    def write(self, evaluation: Evaluation, folder: Path) -> list[Path]:
        """Write the evaluation's files; returns the paths written."""
        ...


@runtime_checkable
class CheckablePart(Protocol):
    """A part that checks written results against a new run."""

    def default_output(self) -> Path:
        """The results folder by default."""
        ...

    def check(
        self, folder: Path, splits: Path | None, progress: Progress | None = None
    ) -> PartCheck:
        """Check the results folder (checksums, a new run)."""
        ...


CLASSIFIER_PARTS: Registry[ClassifierPart] = Registry("classifier parts")
"""The parts of the *Classifiers* page, in registration order."""
BUILTIN_PARTS = ("context_synthetic_recognition.services.comparison_part",)
"""Modules that register the built-in parts when they are imported."""


def parts() -> list[ClassifierPart]:
    """The registered parts (the built-in ones are registered on first use)."""
    for module in BUILTIN_PARTS:
        importlib.import_module(module)
    return [info.obj for info in CLASSIFIER_PARTS]


if TYPE_CHECKING:  # a static check, run by mypy: a dataset cannot be classified (ADR-085)

    def _a_dataset_is_not_unlabelled(
        part: ClassifierPart, trained: TrainedPart, data: Dataset
    ) -> None:
        # mypy rejects both calls; with warn_unused_ignores (strict) the type check fails if one
        # were ever accepted
        part.classify(trained, (data,))  # type: ignore[arg-type]
        classify(part, trained, (data,))  # type: ignore[arg-type]


# ---------------------------------------------------------------- running the stages


@dataclass(frozen=True, eq=False)
class StageRun:
    """The three stages run in a row ("Run all")."""

    samples: SampleSet
    trained: TrainedPart
    predictions: Predictions
    evaluation: Evaluation


STAGES = ("training", "test", "evaluation")
"""The stages, in order; the ``stage`` prefix of the progress of :func:`run_stages`."""


def train(
    part: ClassifierPart,
    option: SampleOption,
    folder: Path | None,
    progress: Progress | None = None,
) -> tuple[SampleSet, TrainedPart]:
    """Read the samples of ``option`` and run ① *Training* (the services' entry of the stage)."""
    samples = part.load(option, folder)
    return samples, part.train(option, samples.training, progress)


def classify(
    part: ClassifierPart,
    trained: TrainedPart,
    unlabelled: tuple[UnlabelledSample, ...],
    progress: Progress | None = None,
) -> Predictions:
    """Run ② *Test*: the objects are classified without their class.

    Raises:
        TypeError: An object set is not an :class:`UnlabelledSample` (ADR-085).
    """
    return part.classify(trained, tuple(require_unlabelled(s) for s in unlabelled), progress)


def evaluate(
    part: ClassifierPart,
    predictions: Predictions,
    answers: tuple[AnswerKey, ...],
    progress: Progress | None = None,
) -> Evaluation:
    """Run ③ *Evaluation*: the decisions against the true classes."""
    return part.evaluate(predictions, answers, progress)


def staged(stage: str, progress: Progress | None) -> Progress | None:
    """``progress`` with the stage in front of what is being done: ``training · 336``."""
    if progress is None:
        return None
    report = progress

    def forward(what: str, done: int, total: int) -> None:
        report(f"{stage} · {what}", done, total)

    return forward


def run_stages(
    part: ClassifierPart,
    option: SampleOption,
    folder: Path | None,
    progress: Progress | None = None,
) -> StageRun:
    """The three stages in a row; ``progress`` gets ``"<stage> · <what>"`` (:data:`STAGES`)."""
    samples, trained = train(part, option, folder, staged(STAGES[0], progress))
    predictions = classify(part, trained, samples.test, staged(STAGES[1], progress))
    evaluation = evaluate(part, predictions, samples.answers, staged(STAGES[2], progress))
    return StageRun(samples, trained, predictions, evaluation)


# ---------------------------------------------------------------- the tables every part gets


def percent(value: float) -> float:
    """A share as a percentage with two decimals."""
    return round(100 * value, 2)


def _feature_meanings(data: Dataset | UnlabelledSample) -> tuple[CellMeaning, ...]:
    return tuple(CellMeaning.QUANTITATIVE if q else CellMeaning.NOMINAL for q in data.quantitative)


def _value(x: float) -> Cell:
    return int(x) if float(x).is_integer() else float(x)


def methods_table(groups: Sequence[MethodGroup]) -> StageTable:
    """*Methods*: every method with its group, features, splits and fixed settings."""
    rows = tuple(
        (group.title, method.title, method.features, ", ".join(method.splits), group.settings)
        for group in groups
        for method in group.methods
    )
    table = Table(
        key="methods",
        title="Methods and their fixed settings",
        columns=("Group", "Method", "Features", "Splits", "Settings"),
        rows=rows,
        group=GROUP,
        note="The settings were fixed before any run; nothing is tuned on the test samples.",
        labels=2,
    )
    return StageTable("methods", "Methods", (ShownTable(table, (CellMeaning.PLAIN,) * 5),))


def training_sample_table(samples: Sequence[TrainingSample]) -> StageTable:
    """*Training sample*: the training objects, their features and their class (per split)."""
    variants = []
    for sample in samples:
        data = sample.data
        codes = data.class_index + 1
        rows = tuple(
            (data.object_ids[i], *(_value(v) for v in data.X[i]), int(codes[i]))
            for i in range(data.m)
        )
        sizes = ", ".join(f"K{c + 1}: {n}" for c, n in enumerate(data.class_sizes))
        table = Table(
            key=f"training-sample-{sample.key}",
            title=f"{sample.title}: the training sample",
            columns=("Object", *data.feature_names, "Class"),
            rows=rows,
            group=GROUP,
            note=(
                f"{data.m} training objects ({sizes}), named by their place in the file. "
                "Class: 1 = K1, 2 = K2."
            ),
        )
        cells = (CellMeaning.PLAIN, *_feature_meanings(data), CellMeaning.CLASS)
        variants.append(ShownTable(table, cells, sample.key, object_column=0))
    return StageTable("training-sample", "Training sample", tuple(variants))


def training_results_table(trained: TrainedPart) -> StageTable:
    """*Training results*: method × split — training accuracy (Definition 2), correct, refusals."""
    rows: list[tuple[Cell, ...]] = []
    for sample in trained.samples:
        truth = sample.data.class_index + 1
        for method in trained.methods[sample.key]:
            if method.decisions is None:
                rows.append((sample.key, method.title, sample.data.m, None, None, None))
                continue
            metrics = classification_metrics(truth, method.decisions)
            rows.append(
                (
                    sample.key,
                    method.title,
                    metrics.n,
                    percent(metrics.accuracy),
                    metrics.correct,
                    metrics.refusals,
                )
            )
    table = Table(
        key="training-results",
        title="Training results: every method on its training sample",
        columns=(
            "Split",
            "Method",
            "Training objects",
            "Training accuracy, %",
            "Correct",
            "Refusals",
        ),
        rows=tuple(rows),
        group=GROUP,
        note=(
            "Training correctness (Definition 2), resubstitution: a k-NN training object is not "
            "its own neighbour, a CS-model training object is left out of its own context. "
            "Refusals count as errors; — = the method is undefined on that training sample."
        ),
        labels=2,
    )
    return StageTable(
        "training-results",
        "Training results",
        (ShownTable(table, (CellMeaning.PLAIN,) * 6, decimals=2),),
    )


def unlabelled_table(unlabelled: Sequence[UnlabelledSample], keys: Sequence[str]) -> StageTable:
    """*Test objects*: their feature values only — there is no class column."""
    variants = []
    for key, sample in zip(keys, unlabelled, strict=True):
        rows = tuple(
            (sample.object_ids[i], *(_value(v) for v in sample.X[i])) for i in range(sample.m)
        )
        table = Table(
            key=f"test-objects-{key}",
            title=f"{sample.name}: the test objects, without their class",
            columns=("Object", *sample.feature_names),
            rows=rows,
            group=GROUP,
            note=(
                f"{sample.m} objects, named by their place in the file. They are classified from "
                "these values only: their classes are not part of the test step."
            ),
        )
        cells = (CellMeaning.PLAIN, *_feature_meanings(sample))
        variants.append(ShownTable(table, cells, key, object_column=0))
    return StageTable("test-objects", "Test objects", tuple(variants))


def _most_common(decisions: Sequence[int]) -> tuple[Cell, int]:
    """The most common decision (``1 / 2`` on a tie) and how many methods give it."""
    if not decisions:
        return None, 0
    counts = Counter(decisions)
    top = max(counts.values())
    winners = sorted(d for d, n in counts.items() if n == top)
    return (winners[0] if len(winners) == 1 else " / ".join(map(str, winners))), top


def predictions_table(predictions: Predictions) -> StageTable:
    """*Predictions*: object × method → 1 / 2 / 0, the scores, and how far the methods agree."""
    variants = []
    for sample, objects in zip(predictions.trained.samples, predictions.objects, strict=True):
        by_method = predictions.decisions[sample.key]
        columns: list[str] = ["Object"]
        cells: list[CellMeaning] = [CellMeaning.PLAIN]
        parts: list[tuple[str, Decisions]] = list(by_method.items())
        titles = {m.method: m.title for m in predictions.trained.methods[sample.key]}
        for key, decided in parts:
            columns.append(titles.get(key, key))
            cells.append(CellMeaning.DECISION)
            if decided.scores is not None:
                columns.append(f"{titles.get(key, key)} · score")
                cells.append(CellMeaning.PLAIN)
        columns += ["Most common", "Agreement"]
        cells += [CellMeaning.DECISION, CellMeaning.PLAIN]
        defined = [d for _, d in parts if d.decisions is not None]
        rows = []
        for i in range(objects.m):
            row: list[Cell] = [objects.object_ids[i]]
            for _, decided in parts:
                row.append(None if decided.decisions is None else int(decided.decisions[i]))
                if decided.scores is not None:
                    row.append(float(decided.scores[i]))
            common, agreement = _most_common(
                [int(d.decisions[i]) for d in defined if d.decisions is not None]
            )
            rows.append((*row, common, agreement))
        table = Table(
            key=f"predictions-{sample.key}",
            title=f"{objects.name}: the decision of every method",
            columns=tuple(columns),
            rows=tuple(rows),
            group=GROUP,
            note=(
                "1 = K1, 2 = K2, 0 = refusal; coloured by the predicted class — whether a decision "
                "is correct is not known in this step. Score: larger for K1. Agreement: how many "
                f"of the {len(defined)} defined methods give the most common decision. — = the "
                "method is undefined on this split."
            ),
        )
        variants.append(ShownTable(table, tuple(cells), sample.key, object_column=0, decimals=4))
    return StageTable("predictions", "Predictions", tuple(variants))


def predicted_classes_table(predictions: Predictions) -> StageTable:
    """*Predicted classes*: method × split — predicted K1, K2 and refusals, counts and shares."""
    rows: list[tuple[Cell, ...]] = []
    for sample, objects in zip(predictions.trained.samples, predictions.objects, strict=True):
        titles = {m.method: m.title for m in predictions.trained.methods[sample.key]}
        for key, decided in predictions.decisions[sample.key].items():
            if decided.decisions is None:
                rows.append((sample.key, titles.get(key, key), objects.m, *(None,) * 6))
                continue
            counts = [int(np.count_nonzero(decided.decisions == code)) for code in (1, 2, 0)]
            shares = [percent(c / objects.m) for c in counts]
            rows.append(
                (
                    sample.key,
                    titles.get(key, key),
                    objects.m,
                    counts[0],
                    shares[0],
                    counts[1],
                    shares[1],
                    counts[2],
                    shares[2],
                )
            )
    table = Table(
        key="predicted-classes",
        title="Predicted classes of the test objects",
        columns=(
            "Split",
            "Method",
            "Test objects",
            "Predicted K1",
            "K1, %",
            "Predicted K2",
            "K2, %",
            "Refusals",
            "Refusals, %",
        ),
        rows=tuple(rows),
        group=GROUP,
        note="Counted from the decisions alone: the true classes are not known in this step.",
        labels=2,
    )
    return StageTable(
        "predicted-classes",
        "Predicted classes",
        (ShownTable(table, (CellMeaning.PLAIN,) * 9, decimals=2),),
    )


def _outcome(decision: int, truth: int) -> str:
    if decision == REFUSAL:
        return "0"
    return f"{'✓' if decision == truth else '✗'} {decision}"


def per_object_table(evaluation: Evaluation) -> StageTable:
    """*Per object*: every decision against the true class — ✓ correct, ✗ wrong, 0 refusal."""
    predictions = evaluation.predictions
    variants = []
    for sample, objects, key in zip(
        predictions.trained.samples, predictions.objects, evaluation.answers, strict=True
    ):
        titles = {m.method: m.title for m in predictions.trained.methods[sample.key]}
        decided = list(predictions.decisions[sample.key].items())
        rows = []
        for i in range(objects.m):
            truth = int(key.classes[i])
            outcomes: list[Cell] = [
                None if d.decisions is None else _outcome(int(d.decisions[i]), truth)
                for _, d in decided
            ]
            correct = sum(1 for o in outcomes if isinstance(o, str) and o.startswith("✓"))
            rows.append((objects.object_ids[i], truth, correct, *outcomes))
        table = Table(
            key=f"per-object-{sample.key}",
            title=f"{objects.name}: every decision against the true class",
            columns=(
                "Object",
                "True class",
                "Correct methods",
                *(titles.get(k, k) for k, _ in decided),
            ),
            rows=tuple(rows),
            group=GROUP,
            note="✓ correct, ✗ wrong (the decision follows the mark), 0 = refusal (an error).",
        )
        cells = (
            CellMeaning.PLAIN,
            CellMeaning.CLASS,
            CellMeaning.PLAIN,
            *(CellMeaning.OUTCOME,) * len(decided),
        )
        variants.append(ShownTable(table, cells, sample.key, object_column=0))
    return StageTable("per-object", "Per object", tuple(variants))


def confusion_table(evaluation: Evaluation) -> StageTable:
    """*Confusion*: per method and split, true class × decision (2 × 2) and the refusals."""
    predictions = evaluation.predictions
    rows: list[tuple[Cell, ...]] = []
    for sample in predictions.trained.samples:
        titles = {m.method: m.title for m in predictions.trained.methods[sample.key]}
        for key, metrics in evaluation.metrics[sample.key].items():
            if metrics is None:
                rows.append((sample.key, titles.get(key, key), *(None,) * 7))
                continue
            (k1k1, k1k2, k1r), (k2k1, k2k2, k2r) = metrics.matrix.tolist()
            rows.append(
                (
                    sample.key,
                    titles.get(key, key),
                    k1k1,
                    k1k2,
                    k1r,
                    k2k1,
                    k2k2,
                    k2r,
                    percent(metrics.accuracy),
                )
            )
    table = Table(
        key="confusion",
        title="Confusion: true class × decision on the test objects",
        columns=(
            "Split",
            "Method",
            "K1 → K1",
            "K1 → K2",
            "K1 refused",
            "K2 → K1",
            "K2 → K2",
            "K2 refused",
            "Accuracy, %",
        ),
        rows=tuple(rows),
        group=GROUP,
        note="True class → decision. Accuracy counts refusals as errors (Definition 3).",
        labels=2,
    )
    return StageTable(
        "confusion", "Confusion", (ShownTable(table, (CellMeaning.PLAIN,) * 9, decimals=2),)
    )


def metrics_against(
    decisions: Mapping[str, Decisions], answers: AnswerKey
) -> dict[str, ClassificationMetrics | None]:
    """The metrics of every method's decisions against an answer key."""
    return {
        key: None if d.decisions is None else classification_metrics(answers.classes, d.decisions)
        for key, d in decisions.items()
    }


def training_view(
    part: ClassifierPart, option: SampleOption, trained: TrainedPart | None
) -> tuple[StageTable, ...]:
    """The tabs of ① *Training*: *Methods* (always), *Training sample*, *Training results*."""
    tabs = [methods_table(part.method_groups(option))]
    if trained is not None:
        tabs += [
            training_sample_table(trained.samples),
            training_results_table(trained),
            *trained.tables,
        ]
    return tuple(tabs)


def prediction_view(predictions: Predictions) -> tuple[StageTable, ...]:
    """The tabs of ② *Test*: *Test objects*, *Predictions*, *Predicted classes*, the part's."""
    keys = [s.key for s in predictions.trained.samples]
    return (
        unlabelled_table(predictions.objects, keys),
        predictions_table(predictions),
        predicted_classes_table(predictions),
        *predictions.tables,
    )


def evaluation_view(evaluation: Evaluation) -> tuple[StageTable, ...]:
    """The tabs of ③ *Evaluation*: the part's tables first, then *Per object* and *Confusion*."""
    return (*evaluation.tables, per_object_table(evaluation), confusion_table(evaluation))


def check_tab(check: PartCheck) -> StageTable:
    """*Check*: every written file with ✓ or ✗ and the problem found."""
    cells = tuple(
        CellMeaning.STATUS if name == "Status" else CellMeaning.PLAIN
        for name in check.table.columns
    )
    return StageTable("check", "Check", (ShownTable(check.table, cells),))


def predictions_csv(predictions: Predictions, split: str) -> str:
    """The blind predictions of one split as CSV: the object, then one column per method.

    Decisions 1 = K1, 2 = K2, 0 = refusal; an empty cell where a method is undefined. There is no
    class column: the classes of the objects are not part of the predictions (ADR-086).

    Raises:
        KeyError: The predictions have no such split.
    """
    index = [s.key for s in predictions.trained.samples].index(split)
    objects = predictions.objects[index]
    titles = {m.method: m.title for m in predictions.trained.methods[split]}
    decided = list(predictions.decisions[split].items())
    rows = [
        [
            objects.object_ids[i],
            *("" if d.decisions is None else int(d.decisions[i]) for _, d in decided),
        ]
        for i in range(objects.m)
    ]
    return csv_rows(["Object", *(titles.get(key, key) for key, _ in decided)], rows)
