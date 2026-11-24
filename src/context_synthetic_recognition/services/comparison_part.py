"""The first classifier part: Tables 7 and 8 — standard classifiers and the CS-model (ADR-084).

The comparison of ADR-079 and ADR-080 in three stages (:mod:`~context_synthetic_recognition.
comparison.stages`): SVM, random forest, the k-NN vote and the CS-model are fitted on the
article's training samples, classify the test objects without their class, and are compared
with the true classes. Every setting is fixed; the page shows them and never lets anyone tune
them. On the three article splits the stages give exactly what ``csr compare run`` writes, and
the part writes and checks the same files.

Registered as ``tables-7-8`` in :data:`~.classifier_parts.CLASSIFIER_PARTS` on import.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from context_synthetic_recognition.comparison import HEART_SPLIT, SPLITS, SplitResult, SplitSpec
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
    SETTINGS_CS_MODEL,
    SETTINGS_KNN,
    SETTINGS_SKLEARN,
    comparison_tables,
    results_table,
)
from context_synthetic_recognition.data.samples import (
    AnswerKey,
    UnlabelledSample,
    require_unlabelled,
)
from context_synthetic_recognition.data.splits import FixedSplit, load_split_file
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
    SampleOption,
    SampleSet,
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
    render_files,
    sha256,
    splits_folder,
    write_comparison,
)

KEY = "tables-7-8"
ARTICLE = "article"
"""Key of the option with the three splits of Tables 7 and 8."""
SPECS: dict[str, SplitSpec] = {spec.key: spec for spec in (*SPLITS, HEART_SPLIT)}
"""Every split the part can read, by key."""
NEEDS_ARTICLE = (
    "Tables 7 and 8 and the CS-model's table need the three article splits (336, 341, 367): "
    "choose “All article splits” and run the steps again."
)
WRITES_ARTICLE = (
    "Only the full article set (336, 341, 367) is written, the files of csr compare run: choose "
    "“All article splits”."
)


@dataclass(frozen=True, eq=False)
class ComparisonResult:
    """What the *Evaluation* stage of the part keeps for writing."""

    results: tuple[SplitResult, ...]
    inputs: dict[str, str]
    """SHA-256 of every split file, by its path relative to the split folder."""
    folder: Path | None


def _sizes_text(split: FixedSplit) -> str:
    size, *classes = split.training_sizes
    return f"training {size} ({', '.join(map(str, classes))}) · test {split.test_sizes[0]}"


def _folder(folder: Path | None) -> Path:
    return folder if folder is not None else splits_folder()


class TablesPart:
    """Tables 7 and 8: standard classifiers and the CS-model on the article's samples."""

    key = KEY
    title = "Tables 7 and 8 — standard classifiers and the CS-model"
    description = (
        "SVM, random forest and k-NN fitted on the article's training samples, with the "
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

    def option(self, key: str) -> SampleOption:
        """An option by key, without reading the files (no sizes).

        Raises:
            DatasetError: No such option.
        """
        if key == ARTICLE:
            return SampleOption(
                ARTICLE,
                "All article splits (336, 341, 367)",
                tuple(s.key for s in SPLITS),
                complete=True,
            )
        if key not in SPECS:
            raise DatasetError(f"unknown split {key!r}; the splits are {', '.join(SPECS)}")
        return SampleOption(key, key, (key,))

    def sample_options(self, folder: Path | None) -> tuple[SampleOption, ...]:
        """The three article splits together, each split alone, and Heart's split."""
        try:
            root = _folder(folder)
        except CSRError as error:
            reason = str(error)
            return tuple(
                SampleOption(
                    o.key, o.title, o.splits, available=False, reason=reason, complete=o.complete
                )
                for o in (self.option(k) for k in (ARTICLE, *SPECS))
            )
        loaded: dict[str, FixedSplit | str] = {}
        for key, spec in SPECS.items():
            try:
                loaded[key] = load_split_file(root / spec.file)
            except CSRError as error:
                loaded[key] = str(error)
        options = []
        article = self.option(ARTICLE)
        problems = [v for k, v in loaded.items() if k in article.splits and isinstance(v, str)]
        tests = [
            v.test_sizes[0]
            for k, v in loaded.items()
            if k in article.splits and not isinstance(v, str)
        ]
        options.append(
            SampleOption(
                article.key,
                article.title,
                article.splits,
                "test " + " / ".join(map(str, tests)) if not problems else "",
                available=not problems,
                reason=problems[0] if problems else "",
                complete=True,
            )
        )
        for key in SPECS:
            value = loaded[key]
            base = self.option(key)
            if isinstance(value, str):
                options.append(
                    SampleOption(key, base.title, base.splits, available=False, reason=value)
                )
            else:
                options.append(SampleOption(key, base.title, base.splits, _sizes_text(value)))
        return tuple(options)

    def load(self, option: SampleOption, folder: Path | None) -> SampleSet:
        """Read the split files of ``option``: training samples, test objects, answer keys."""
        root = _folder(folder)
        training, test, answers = [], [], []
        for key in option.splits:
            spec = SPECS[key]
            path = root / spec.file
            split = load_split_file(path)
            training.append(
                TrainingSample(
                    key,
                    f"{key} — {spec.title}",
                    split.training_part,
                    spec.file,
                    sha256(path.read_bytes()),
                )
            )
            name = f"{key}: test sample"
            test.append(split.test_objects(name))
            answers.append(split.test_answers(name))
        return SampleSet(option, tuple(training), tuple(test), tuple(answers))

    # ------------------------------------------------------------------ methods

    def method_groups(self, option: SampleOption) -> tuple[MethodGroup, ...]:
        """Table 7, Table 8, the k-NN rows not printed, the CS-model — with their settings."""
        settings = {
            "Table 7": SETTINGS_SKLEARN,
            "Table 8": SETTINGS_KNN,
            "Not printed": (
                "The k-NN vote with the Zhuravlyov metric on all features (the 336 split only); "
                "the other settings as in Table 8. Not printed in Tables 7 and 8."
            ),
            "CS-model": SETTINGS_CS_MODEL,
        }
        found: dict[str, dict[str, tuple[MethodInfo, list[str]]]] = {g: {} for g in settings}
        for key in option.splits:
            for spec in method_specs(SPECS[key]):
                entry = found[spec.group].setdefault(
                    spec.key, (MethodInfo(spec.key, spec.title, spec.features, ()), [])
                )
                entry[1].append(key)
        return tuple(
            MethodGroup(
                group,
                settings[group],
                tuple(
                    MethodInfo(info.key, info.title, info.features, tuple(splits))
                    for info, splits in found[group].values()
                ),
            )
            for group in settings
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
                progress(sample.key, index, len(training))
            split = fit_split(SPECS[sample.key], sample.data)
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
        for index, (split, sample) in enumerate(zip(fitted, samples, strict=True)):
            if progress is not None:
                progress(split.spec.key, index, len(fitted))
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
        """Compare with the true classes; Tables 7 and 8 and the CS-model's table for the set."""
        trained = predictions.trained
        fitted: tuple[FittedSplit, ...] = trained.fitted
        results: list[SplitResult] = []
        metrics = {}
        for index, (split, objects, key) in enumerate(
            zip(fitted, predictions.objects, answers, strict=True)
        ):
            if progress is not None:
                progress(split.spec.key, index, len(fitted))
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
        """Only the three article splits are written (the files of ``csr compare run``)."""
        complete = evaluation.option.complete and tuple(
            s.key for s in evaluation.predictions.trained.samples
        ) == tuple(s.key for s in SPLITS)
        return "" if complete else WRITES_ARTICLE

    def outcome(self, evaluation: Evaluation) -> ComparisonOutcome:
        """The files of the evaluation, as ``csr compare run`` renders them.

        Raises:
            DatasetError: The evaluation is not of the three article splits.
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
        if progress is not None:
            done = [0]

            def report(line: str) -> None:
                progress(line, done[0], len(SPLITS))
                done[0] += 1

        checked = check_comparison(folder, splits=splits, progress=report)
        if progress is not None:
            progress("done", len(SPLITS), len(SPLITS))
        return PartCheck(folder, checked.passed, check_table(checked), checked.problems)


def _shown(key: str, title: str, table: Table) -> StageTable:
    cells = (CellMeaning.PLAIN,) * len(table.columns)
    return StageTable(key, title, (ShownTable(table, cells, decimals=DECIMALS),))


def _evaluation_tables(
    option: SampleOption, results: tuple[SplitResult, ...]
) -> tuple[StageTable, ...]:
    keys = tuple(r.spec.key for r in results)
    if option.complete and keys == tuple(s.key for s in SPLITS):
        seven, eight, cs_model, everything = comparison_tables(results)
        return (
            _shown("table-7", "Table 7", seven),
            _shown("table-8", "Table 8", eight),
            _shown("cs-model", "CS-model", cs_model),
            _shown("all-results", "All results", everything),
        )
    return (
        StageTable("table-7", "Table 7", missing=NEEDS_ARTICLE),
        StageTable("table-8", "Table 8", missing=NEEDS_ARTICLE),
        StageTable("cs-model", "CS-model", missing=NEEDS_ARTICLE),
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
    part = TABLES
    option = part.option(key)
    root = _folder(folder)
    training, test = [], []
    for split_key in option.splits:
        spec = SPECS[split_key]
        path = root / spec.file
        split = load_split_file(path)
        training.append(
            TrainingSample(
                split_key,
                f"{split_key} — {spec.title}",
                split.training_part,
                spec.file,
                sha256(path.read_bytes()),
            )
        )
        test.append(split.test_objects(f"{split_key}: test sample"))
    trained = part.train(option, tuple(training), staged(STAGES[0], progress))
    return part.classify(trained, tuple(test), staged(STAGES[1], progress))


TABLES = TablesPart()
"""The registered instance."""
CLASSIFIER_PARTS.add(KEY, TABLES, summary=TABLES.title)
