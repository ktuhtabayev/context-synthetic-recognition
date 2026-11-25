"""The first classifier part: the comparison with standard classifiers (ADR-084).

The comparison of ADR-079 and ADR-080 in three stages (:mod:`~context_synthetic_recognition.
comparison.stages`): SVM, Random Forest, the k-NN vote and the CS-model are fitted on the
article's training samples, classify the test objects without their class, and are compared
with the true classes. Every setting is fixed; the page shows them and never lets anyone tune
them. On German's three splits the stages give exactly what ``csr compare run`` writes, and
the part writes and checks the same files. Its samples are every split file of the folder, by
dataset (ADR-090, ADR-091).

Registered as ``classifier-comparison`` in :data:`~.classifier_parts.CLASSIFIER_PARTS` on import.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from context_synthetic_recognition.comparison import (
    COMPARISON_SPLITS,
    SplitResult,
    SplitSpec,
)
from context_synthetic_recognition.comparison.settings import TABLE_KNN, TABLE_STANDARD
from context_synthetic_recognition.comparison.stages import (
    FittedSplit,
    SplitDecisions,
    classify_split,
    evaluate_split,
    fit_split,
    method_specs,
)
from context_synthetic_recognition.comparison.tables import (
    DECIMALS,
    FEATURES_STANDARD,
    KNN_KEY,
    SETTINGS_CS_MODEL,
    SETTINGS_KNN,
    STANDARD_KEY,
    comparison_tables,
    results_table,
)
from context_synthetic_recognition.data.samples import (
    AnswerKey,
    UnlabelledSample,
    require_unlabelled,
)
from context_synthetic_recognition.data.splits import parse_split_name
from context_synthetic_recognition.errors import CSRError, DatasetError
from context_synthetic_recognition.export.tables import Table
from context_synthetic_recognition.services.classifier_parts import (
    CLASSIFIER_PARTS,
    STAGES,
    CellMeaning,
    Evaluation,
    MethodGroup,
    MethodInfo,
    MethodTraining,
    PartCheck,
    Predictions,
    Progress,
    SampleChoice,
    SampleGroup,
    SampleOption,
    SampleSet,
    SampleTree,
    ShownTable,
    StageTable,
    TrainedPart,
    TrainingSample,
    metrics_against,
    staged,
)
from context_synthetic_recognition.services.comparison import (
    OUTPUT_DIR,
    SPLITS_DIR,
    ComparisonOutcome,
    check_comparison,
    check_table,
    default_output,
    library_versions,
    recorded_keys,
    render_files,
    splits_folder,
    write_comparison,
)
from context_synthetic_recognition.services.split_discovery import (
    SplitFile,
    discover_splits,
    read_splits,
    resolve_keys,
)

KEY = "classifier-comparison"
NEEDS_COMPARISON_SET = (
    "The Standard classifiers, k-NN by metric and CS-model tables need German's three splits — "
    "the Zhuravlyov, Euclidean and Chebyshev splits, training 336, 341 and 367 — and no other: "
    "tick exactly those and run the steps again."
)
WRITES_COMPARISON_SET = (
    "Only German's three splits — and no other — are written, the files of csr compare run; "
    "csr compare run --split KEY --out DIR writes one split."
)


@dataclass(frozen=True, eq=False)
class ComparisonResult:
    """What the *Evaluation* stage of the part keeps for writing."""

    results: tuple[SplitResult, ...]
    inputs: dict[str, str]
    """SHA-256 of every split file, by its path relative to the split folder."""
    folder: Path | None


def _folder(folder: Path | None) -> Path:
    return folder if folder is not None else splits_folder()


def _dataset_title(name: str, files: tuple[SplitFile, ...]) -> str:
    """``German (1000 objects)``; the sizes of all its files when they differ."""
    sizes = sorted({f.name.m for f in files if f.name is not None})
    if not sizes:
        return name
    return f"{name} ({' / '.join(map(str, sizes))} objects)"


def _choice(file: SplitFile) -> SampleChoice:
    """A split file as a sample of the tree; the comparison needs two classes."""
    reason = file.reason
    if not reason and file.name is not None and file.name.c != 2:
        reason = (
            f"{Path(file.file).name}: the comparison needs two classes, the file has {file.name.c}"
        )
    title = f"{file.title} · {file.details}" if file.details else file.title
    detail = f"{file.file}{f', row {file.row}' if file.row else ''}"
    return SampleChoice(
        file.key,
        title,
        file.label,
        available=file.available and not reason,
        reason=reason,
        detail=detail,
    )


def spec_of(sample: TrainingSample) -> SplitSpec:
    """The split of a training sample, from the name of its file (ADR-090).

    Raises:
        DatasetError: The file's name does not follow the pattern.
    """
    name = parse_split_name(Path(sample.file).name)
    return SplitSpec.of(name.key, name.dataset, sample.file, name.metric, name.training)


def training_sample(file: SplitFile) -> TrainingSample:
    """The training sample of an available split file, named by the split's label."""
    if file.split is None:
        raise DatasetError(file.reason or f"{file.file}: the file was not read")
    return TrainingSample(file.key, file.label, file.split.training_part, file.file, file.sha256)


class ComparisonPart:
    """The comparison with standard classifiers and the CS-model on the article's samples."""

    key = KEY
    title = "Comparison with standard classifiers"

    def __init__(self) -> None:
        """No split read yet."""
        self._specs: dict[str, SplitSpec] = {}

    description = (
        "SVM, Random Forest and k-NN fitted on the article's training samples, with the "
        "CS-model on the same samples; every setting is fixed (ADR-079, ADR-080)."
    )

    # ------------------------------------------------------------------ folders

    def default_folder(self) -> Path:
        """``datasets/splits`` of the project, else the Windows build's copy, else the path."""
        try:
            return splits_folder()
        except CSRError:
            return Path("datasets") / SPLITS_DIR

    def default_output(self) -> Path:
        """``experiments/comparison`` of the project, or the relative path."""
        try:
            return default_output()
        except CSRError:
            return Path(*OUTPUT_DIR)

    # ------------------------------------------------------------------ samples

    def samples(self, folder: Path | None) -> SampleTree:
        """Every split file of the folder, by dataset (ADR-090, ADR-091).

        German's three splits are ticked by default; they are the set the comparison tables
        need. A file that fails a check, or has other than two classes, is unavailable with the
        reason.
        """
        try:
            root = _folder(folder)
        except CSRError as error:
            return SampleTree((), problem=str(error), complete=COMPARISON_SPLITS)
        catalog = discover_splits(root)
        self._specs.update({f.key: f.spec for f in catalog.files if f.name is not None})
        groups = tuple(
            SampleGroup(_dataset_title(name, files), tuple(_choice(f) for f in files))
            for name, files in catalog.datasets()
        )
        available = {c.key for g in groups for c in g.choices if c.available}
        return SampleTree(
            groups,
            default=tuple(key for key in COMPARISON_SPLITS if key in available),
            complete=COMPARISON_SPLITS,
            problem=catalog.problem,
            complete_note=NEEDS_COMPARISON_SET,
        )

    def option(self, keys: tuple[str, ...], folder: Path | None = None) -> SampleOption:
        """The option of a set of splits, by key or training size, read from ``folder``.

        Raises:
            DatasetError: A split is unknown or ambiguous.
        """
        return self.samples(folder).option(resolve_keys(_folder(folder), keys))

    def load(self, option: SampleOption, folder: Path | None) -> SampleSet:
        """Read the split files of ``option``: training samples, test objects, answer keys."""
        files = read_splits(_folder(folder), option.splits)
        self._specs.update({f.key: f.spec for f in files})
        training, test, answers = [], [], []
        for file in files:
            training.append(training_sample(file))
            name = f"{file.label}: test sample"
            if file.split is not None:
                test.append(file.split.test_objects(name))
                answers.append(file.split.test_answers(name))
        return SampleSet(option, tuple(training), tuple(test), tuple(answers))

    def _spec(self, key: str) -> SplitSpec:
        """The split of a key, from the names of the files read so far or of the default folder."""
        if key not in self._specs:
            catalog = discover_splits(self.default_folder(), read=())
            self._specs.update({f.key: f.spec for f in catalog.files if f.name is not None})
        return self._specs[key]

    # ------------------------------------------------------------------ methods

    def method_groups(self, option: SampleOption) -> tuple[MethodGroup, ...]:
        """Standard classifiers, k-NN by metric, the k-NN rows not printed, the CS-model.

        Every method carries its own settings; a group's note says what its methods share.
        """
        notes = {
            TABLE_STANDARD: FEATURES_STANDARD,
            TABLE_KNN: SETTINGS_KNN,
            "Not printed": (
                "The k-NN vote with the split's own metric on all features (German's Zhuravlyov "
                "split only); "
                f"the other settings as in {TABLE_KNN}. In neither comparison table."
            ),
            "CS-model": SETTINGS_CS_MODEL,
        }
        found: dict[str, dict[str, tuple[MethodInfo, list[str]]]] = {g: {} for g in notes}
        for key in option.splits:
            for spec in method_specs(self._spec(key)):
                entry = found[spec.group].setdefault(
                    spec.key,
                    (MethodInfo(spec.key, spec.title, spec.features, (), spec.settings), []),
                )
                entry[1].append(self._spec(key).name)
        return tuple(
            MethodGroup(
                group,
                notes[group],
                tuple(
                    MethodInfo(info.key, info.title, info.features, tuple(splits), info.settings)
                    for info, splits in found[group].values()
                ),
            )
            for group in notes
            if found[group]
        )

    # ------------------------------------------------------------------ the stages

    def train(
        self,
        option: SampleOption,
        training: tuple[TrainingSample, ...],
        progress: Progress | None = None,
    ) -> TrainedPart:
        """Fit every method on each training sample; classify the training objects."""
        fitted: list[FittedSplit] = []
        methods: dict[str, tuple[MethodTraining, ...]] = {}
        for index, sample in enumerate(training):
            if progress is not None:
                progress(sample.title, index, len(training))
            split = fit_split(spec_of(sample), sample.data)
            fitted.append(split)
            methods[sample.key] = tuple(
                MethodTraining(m.spec.key, m.spec.title, m.training_decisions, m.undefined)
                for m in split.methods
            )
        if progress is not None:
            progress("done", len(training), len(training))
        return TrainedPart(self.key, option, training, methods, fitted=tuple(fitted))

    def classify(
        self,
        trained: TrainedPart,
        unlabelled: tuple[UnlabelledSample, ...],
        progress: Progress | None = None,
    ) -> Predictions:
        """Classify the test objects of every split from their feature values only."""
        samples = tuple(require_unlabelled(s) for s in unlabelled)  # no class: ADR-085
        fitted: tuple[FittedSplit, ...] = trained.fitted
        if len(samples) != len(fitted):
            raise ValueError(f"{len(unlabelled)} object sets for {len(fitted)} fitted splits")
        decisions = {}
        titles = [s.title for s in trained.samples]
        for index, (split, sample) in enumerate(zip(fitted, samples, strict=True)):
            if progress is not None:
                progress(titles[index], index, len(fitted))
            decisions[split.spec.key] = classify_split(split, sample).decisions
        if progress is not None:
            progress("done", len(fitted), len(fitted))
        return Predictions(self.key, trained, samples, decisions)

    def evaluate(
        self,
        predictions: Predictions,
        answers: tuple[AnswerKey, ...],
        progress: Progress | None = None,
    ) -> Evaluation:
        """Compare with the true classes; the comparison tables and the CS-model's for the set."""
        trained = predictions.trained
        fitted: tuple[FittedSplit, ...] = trained.fitted
        results: list[SplitResult] = []
        metrics = {}
        for index, (split, objects, key) in enumerate(
            zip(fitted, predictions.objects, answers, strict=True)
        ):
            if progress is not None:
                progress(trained.samples[index].title, index, len(fitted))
            decided = predictions.decisions[split.spec.key]
            results.append(evaluate_split(SplitDecisions(split, objects, decided), key))
            metrics[split.spec.key] = metrics_against(decided, key)
        if progress is not None:
            progress("done", len(fitted), len(fitted))
        done = tuple(results)
        inputs = {s.file: s.sha256 for s in trained.samples}
        return Evaluation(
            self.key,
            predictions,
            answers,
            metrics,
            tables=_evaluation_tables(predictions.option, done),
            result=ComparisonResult(done, inputs, None),
        )

    # ------------------------------------------------------------------ writing and checking

    def write_problem(self, evaluation: Evaluation) -> str:
        """Only German's three splits are written (the files of ``csr compare run``)."""
        complete = evaluation.option.complete and (
            tuple(s.key for s in evaluation.predictions.trained.samples) == COMPARISON_SPLITS
        )
        return "" if complete else WRITES_COMPARISON_SET

    def outcome(self, evaluation: Evaluation) -> ComparisonOutcome:
        """The files of the evaluation, as ``csr compare run`` renders them.

        Raises:
            DatasetError: The evaluation is not of German's three splits.
        """
        problem = self.write_problem(evaluation)
        if problem:
            raise DatasetError(problem)
        result: ComparisonResult = evaluation.result
        tables = comparison_tables(result.results)
        versions = library_versions()
        files = render_files(result.results, tables, versions=versions, inputs=result.inputs)
        return ComparisonOutcome(result.results, tables, files, versions, Path())

    def write(self, evaluation: Evaluation, folder: Path) -> list[Path]:
        """Write the files of ``csr compare run`` to ``folder``."""
        return write_comparison(self.outcome(evaluation), folder)

    def check(
        self, folder: Path, splits: Path | None, progress: Progress | None = None
    ) -> PartCheck:
        """``csr compare check``: the checksums, and a new run gives every file."""
        report: Callable[[str], None] | None = None
        total = len(recorded_keys(folder))
        if progress is not None:
            done = [0]

            def report(line: str) -> None:
                progress(line, done[0], total)
                done[0] += 1

        checked = check_comparison(folder, splits=splits, progress=report)
        if progress is not None:
            progress("done", total, total)
        return PartCheck(folder, checked.passed, check_table(checked), checked.problems)


def _shown(key: str, title: str, table: Table) -> StageTable:
    cells = (CellMeaning.PLAIN,) * len(table.columns)
    return StageTable(key, title, (ShownTable(table, cells, decimals=DECIMALS),))


def _evaluation_tables(
    option: SampleOption, results: tuple[SplitResult, ...]
) -> tuple[StageTable, ...]:
    keys = tuple(r.spec.key for r in results)
    if option.complete and keys == COMPARISON_SPLITS:
        standard, knn, cs_model, everything = comparison_tables(results)
        return (
            _shown(STANDARD_KEY, TABLE_STANDARD, standard),
            _shown(KNN_KEY, TABLE_KNN, knn),
            _shown("cs-model", "CS-model", cs_model),
            _shown("all-results", "All results", everything),
        )
    return (
        StageTable(STANDARD_KEY, TABLE_STANDARD, missing=NEEDS_COMPARISON_SET),
        StageTable(KNN_KEY, TABLE_KNN, missing=NEEDS_COMPARISON_SET),
        StageTable("cs-model", "CS-model", missing=NEEDS_COMPARISON_SET),
        _shown("all-results", "All results", results_table(results)),
    )


def blind_predictions(
    key: str, folder: Path | None = None, progress: Progress | None = None
) -> Predictions:
    """Train on one split's training sample and classify its test objects without their class.

    The loader reads the whole file and checks the convention (ADR-082); only the training
    sample and the test objects' feature values are kept — no answer key is built (ADR-086).

    Raises:
        DatasetError: Unknown split, or the split file is missing or breaks the convention.
    """
    part = COMPARISON_PART
    root = _folder(folder)
    (file,) = read_splits(root, (key,))
    option = SampleOption(file.key, file.short_label, (file.key,))
    training, test = [training_sample(file)], []
    if file.split is not None:
        test.append(file.split.test_objects(f"{file.label}: test sample"))
    trained = part.train(option, tuple(training), staged(STAGES[0], progress))
    return part.classify(trained, tuple(test), staged(STAGES[1], progress))


COMPARISON_PART = ComparisonPart()
"""The registered instance."""
CLASSIFIER_PARTS.add(KEY, COMPARISON_PART, summary=COMPARISON_PART.title)
