"""Running the evaluation designs for the CLI and the *Classifiers* page; their files (ADR-096).

- :func:`first_round` — the samples of a design's first round, for the steps ① *Training* and
  ② *Test* of the page (a repeated design shows its first repetition);
- :func:`classify_first_round` — ② *Test*: the objects classified without their class, or, for a
  self-check, the training objects' resubstitution decisions;
- :func:`evaluate_design` — ③ *Evaluation*: every round of the design, its tables and figures;
- :func:`run_designs`, :func:`write_design`, :func:`check_design` — ``csr compare run | check
  --design``: the files of ``experiments/designs/<design>/`` and their check.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from context_synthetic_recognition.comparison import COMPARISON_SPLITS
from context_synthetic_recognition.data.catalog import find_datasets_root
from context_synthetic_recognition.data.samples import (
    AnswerKey,
    Decisions,
    UnlabelledSample,
    require_unlabelled,
)
from context_synthetic_recognition.errors import DatasetError
from context_synthetic_recognition.evaluation.metrics import ClassificationMetrics
from context_synthetic_recognition.export.tables import Cell, Table
from context_synthetic_recognition.services.classifier_parts import (
    CellMeaning,
    ClassifierPart,
    Evaluation,
    Predictions,
    SampleOption,
    SampleSet,
    ShownTable,
    StageTable,
    TrainedPart,
    TrainingSample,
    metrics_against,
)
from context_synthetic_recognition.services.comparison import library_versions, sha256
from context_synthetic_recognition.services.design_figures import FIGURES
from context_synthetic_recognition.services.design_outputs import (
    CHECKSUMS,
    DECIMALS,
    design_files,
    design_table,
    rounds_table,
)
from context_synthetic_recognition.services.designs import (
    DESIGNS,
    DIRECT,
    CaseResult,
    DesignResult,
    DesignSplit,
    EvaluationDesign,
    RoundResult,
    SplitDesignResult,
    run_design,
)

OUTPUT_DIR = ("experiments", "designs")
"""The folder of the designs' files, relative to the project root; one sub-folder per design."""
ALL = "all"
"""Every registered design (``--design all``)."""

Progress = Callable[[str, int, int], None]
Cancelled = Callable[[], bool]


@runtime_checkable
class DesignablePart(Protocol):
    """A part whose samples are splits a design can use in other ways (ADR-095)."""

    def design_splits(self, option: SampleOption, folder: Path | None) -> tuple[DesignSplit, ...]:
        """The ticked splits with every object and both samples."""
        ...


def design_keys(choice: str) -> tuple[str, ...]:
    """The designs of ``--design``: one key, or every registered one for ``all``.

    Raises:
        RegistryError: Unknown design.
    """
    if choice == ALL:
        return tuple(info.name for info in DESIGNS)
    return (DESIGNS.info(choice).name,)


def default_output(start: Path | None = None) -> Path:
    """``experiments/designs`` beside the project's dataset folder, or the relative path."""
    root = find_datasets_root(start)
    return root.parent.joinpath(*OUTPUT_DIR) if root is not None else Path(*OUTPUT_DIR)


# ---------------------------------------------------------------- the steps of the page


def first_round(
    design: EvaluationDesign, part: DesignablePart, option: SampleOption, folder: Path | None
) -> SampleSet:
    """The samples of the design's first round on every ticked split (① and ②).

    Raises:
        DatasetError: The design cannot run on a ticked split (the reason says why).
    """
    splits = part.design_splits(option, folder)
    sources = {s.key: s for s in splits}
    training, test, answers = [], [], []
    for split in splits:
        cases = design.cases(split, [o for o in splits if o.key != split.key])
        runnable = [c for c in cases if not c.unavailable and c.count]
        if not runnable:
            reasons = "; ".join(c.unavailable for c in cases if c.unavailable)
            raise DatasetError(f"{design.title} on {split.label}: {reasons or 'nothing to run'}")
        case = runnable[0]
        planned = case.plan(0)
        source = sources[planned.source] if planned.source else split
        objects = planned.classified
        name = f"{source.label}: {'the first round' if case.count > 1 else 'classified objects'}"
        training.append(
            TrainingSample(
                split.key,
                split.label,
                split.dataset.subset(planned.training),
                split.file,
                split.sha256,
            )
        )
        test.append(
            UnlabelledSample(
                objects,
                source.dataset.X[objects],
                source.dataset.feature_types,
                source.dataset.feature_names,
                name,
            )
        )
        answers.append(
            AnswerKey(
                objects, source.dataset.class_index[objects] + 1, source.dataset.classes, name
            )
        )
    return SampleSet(option, tuple(training), tuple(test), tuple(answers))


def design_plan(design: EvaluationDesign, splits: Sequence[DesignSplit]) -> StageTable:
    """① *Design plan*: every case of the design with its rounds and their sizes.

    The sizes of a design that chooses its training samples while it runs (R-IPS) are known only
    then.
    """
    rows: list[tuple[Cell, ...]] = []
    for split in splits:
        for case in design.cases(split, [o for o in splits if o.key != split.key]):
            if case.unavailable:
                rows.append((split.label, case.title, 0, None, None, case.unavailable))
                continue
            if getattr(design, "selects", False):
                rows.append(
                    (
                        split.label,
                        case.title,
                        case.count,
                        None,
                        None,
                        "the training objects are chosen by R-IPS in every round",
                    )
                )
                continue
            planned = [case.plan(i) for i in range(case.count)]
            training = sorted({p.training.size for p in planned})
            classified = sorted({p.classified.size for p in planned})
            rows.append(
                (
                    split.label,
                    case.title,
                    case.count,
                    " / ".join(map(str, training)),
                    " / ".join(map(str, classified)),
                    "",
                )
            )
    table = Table(
        key="design-plan",
        title=f"{design.title}: the rounds",
        columns=("Split", "Case", "Rounds", "Training objects", "Classified objects", "Note"),
        rows=tuple(rows),
        group="Evaluation designs",
        note=(
            f"{design.explanation} {design.settings} The steps ① and ② show the first round; "
            "③ runs every round."
        ).replace("  ", " "),
        labels=2,
    )
    return StageTable("design-plan", "Design plan", (ShownTable(table, (CellMeaning.PLAIN,) * 6),))


def classify_first_round(
    design: EvaluationDesign,
    part: ClassifierPart,
    trained: TrainedPart,
    unlabelled: tuple[UnlabelledSample, ...],
    *,
    self_check: bool,
) -> Predictions:
    """② The decisions of the first round: classified without the class, or by resubstitution.

    A self-check's objects are the training objects themselves; their decisions are those of the
    training stage, each object left out of its own classification where the method allows it
    (Definition 2). Every other design classifies the objects from their feature values only.
    """
    samples = tuple(require_unlabelled(s) for s in unlabelled)
    if not self_check:
        return part.classify(trained, samples)
    decisions = {
        sample.key: {
            m.method: Decisions(m.decisions, None, m.undefined) for m in trained.methods[sample.key]
        }
        for sample in trained.samples
    }
    return Predictions(part.key if hasattr(part, "key") else "", trained, samples, decisions)


def _shown(key: str, title: str, table: Table) -> StageTable:
    return StageTable(
        key,
        title,
        (ShownTable(table, (CellMeaning.PLAIN,) * len(table.columns), decimals=DECIMALS),),
    )


def design_tabs(design: EvaluationDesign, result: DesignResult) -> tuple[StageTable, ...]:
    """The tabs of a design's evaluation: its table, its figure, every round, R-IPS."""
    tabs = [_shown("design", "Design", design_table(design, result))]
    figure = FIGURES.get(design.key)
    spec = figure(result) if figure is not None else None
    if spec is not None:
        tabs.append(StageTable("design-figure", "Figure", figure=spec))
    if design.repeated:
        tabs.append(_shown("design-rounds", "Every round", rounds_table(design, result)))
    return tuple(tabs)


def direct_result(evaluation: Evaluation) -> DesignResult:
    """The design as given, from an evaluation of the part's own steps (nothing is run again)."""
    predictions = evaluation.predictions
    splits = []
    for sample, objects, answers in zip(
        predictions.trained.samples, predictions.objects, evaluation.answers, strict=True
    ):
        titles = {m.method: m.title for m in predictions.trained.methods[sample.key]}
        metrics: Mapping[str, ClassificationMetrics | None] = evaluation.metrics[sample.key]
        decided = predictions.decisions[sample.key]
        sizes = answers.sizes
        data = sample.data
        round_ = RoundResult(
            "",
            (data.m, *data.class_sizes),
            tuple(sizes),
            metrics,
            "",
            titles,
            {k: d.undefined for k, d in decided.items() if d.decisions is None},
        )
        methods = tuple((k, titles.get(k, k)) for k in metrics)
        splits.append(
            SplitDesignResult(
                sample.key,
                sample.title,
                (CaseResult("as-given", "", methods, (round_,)),),
            )
        )
        del objects
    return DesignResult(DIRECT, tuple(splits))


def evaluate_design(
    design: EvaluationDesign,
    part: ClassifierPart,
    predictions: Predictions,
    answers: tuple[AnswerKey, ...],
    folder: Path | None,
    *,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> Evaluation:
    """③ Every round of the design; the first round's decisions against the true classes.

    Raises:
        TypeError: The part cannot run designs.
    """
    if not isinstance(part, DesignablePart):
        raise TypeError(f"{part.title}: this part runs the design as given only")
    option = predictions.option
    splits = part.design_splits(option, folder)
    result = run_design(design, part, option, splits, progress=progress, cancelled=cancelled)
    metrics = {
        s.key: metrics_against(predictions.decisions[s.key], a)
        for s, a in zip(predictions.trained.samples, answers, strict=True)
    }
    return Evaluation(
        getattr(part, "key", ""),
        predictions,
        answers,
        metrics,
        tables=design_tabs(design, result),
        result=result,
    )


# ---------------------------------------------------------------- the CLI's files


@dataclass(frozen=True, eq=False)
class DesignOutcome:
    """A design run for the CLI: its result and its files, not yet written."""

    design: EvaluationDesign
    result: DesignResult
    files: Mapping[str, bytes]


def run_designs(
    part: DesignablePart,
    option: SampleOption,
    folder: Path | None,
    designs: Sequence[str],
    *,
    progress: Progress | None = None,
) -> list[DesignOutcome]:
    """Run designs on the ticked splits and render their files."""
    splits = part.design_splits(option, folder)
    inputs = {s.key: (s.file, s.sha256) for s in splits}
    versions = library_versions()
    found = []
    for key in designs:
        design = DESIGNS.get(key)
        result = run_design(design, part, option, splits, progress=progress)  # type: ignore[arg-type]
        files = design_files(design, result, versions=versions, inputs=inputs)
        found.append(DesignOutcome(design, result, files))
    return found


def write_design(outcome: DesignOutcome, out: Path) -> list[Path]:
    """Write a design's files to ``out`` (created if needed)."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, data in outcome.files.items():
        path = out / name
        path.write_bytes(data)
        written.append(path)
    return written


def recorded_splits(folder: Path) -> tuple[str, ...]:
    """The split keys a design folder's ``summary.json`` names; German's three if none."""
    try:
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return COMPARISON_SPLITS
    splits = summary.get("splits") if isinstance(summary, dict) else None
    keys = tuple(str(s["key"]) for s in splits or () if isinstance(s, dict) and "key" in s)
    return keys or COMPARISON_SPLITS


def check_files(folder: Path, expected: Mapping[str, bytes]) -> list[str]:
    """The problems of a folder against its checksums and the files of a new run."""
    listing = folder / CHECKSUMS
    if not listing.is_file():
        return [f"{CHECKSUMS}: missing in {folder}"]
    listed = {}
    for line in listing.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        listed[name] = digest
    problems = []
    for name, digest in listed.items():
        path = folder / name
        if not path.is_file():
            problems.append(f"{name}: missing")
        elif sha256(path.read_bytes()) != digest:
            problems.append(f"{name}: does not match {CHECKSUMS}")
    for name in sorted(set(listed) ^ (set(expected) - {CHECKSUMS})):
        where = "a new run writes it" if name in expected else "a new run does not write it"
        problems.append(f"{name}: {'not listed' if name not in listed else 'listed'}, {where}")
    for name, data in expected.items():
        path = folder / name
        if path.is_file() and path.read_bytes() != data:
            problems.append(f"{name}: differs from a new run")
    return problems
