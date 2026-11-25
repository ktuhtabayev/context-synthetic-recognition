"""Evaluation designs: which objects of a split train the methods, which are classified (ADR-095).

A split file gives three sets: the training sample T, the test sample Q and all its m objects. The
article's evaluation fits on T and classifies Q. An evaluation design uses the same file in another
way — swapped, on itself, against random training samples, by cross-validation, with the
training sample chosen inside each fold, on smaller samples, or across splits — so that the
comparison of the methods can be read against other points of view.

A design plans *rounds* for each split: the objects to fit on and the objects to classify (or, for
a self-check, the training objects classified by resubstitution, Definition 2). The rounds of a
design are grouped in *cases* (the sizes of the training-size curve, the targets of split to
split). :func:`run_design` runs every round through the part's own stages — ``train`` on the
round's training objects, ``classify`` on the classified objects as an
:class:`~context_synthetic_recognition.data.samples.UnlabelledSample` — and compares the decisions
with the classes of an :class:`~context_synthetic_recognition.data.samples.AnswerKey` afterwards,
so the blind rule (ADR-085) holds in every design.

The designs are registered in :data:`DESIGNS` (a :class:`~context_synthetic_recognition.core.
registry.Registry`): a later design is one registration and the *Classifiers* page offers it
without a change. Every setting of a design is fixed here (:data:`SEED` …); nothing is tuned.
Planning is pure; only the part's methods compute.
"""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from context_synthetic_recognition.config import RIPSConfig, plugin
from context_synthetic_recognition.core.arrays import IntArray, readonly
from context_synthetic_recognition.core.local_context.metrics import METRICS, traits_of
from context_synthetic_recognition.core.registry import Registry
from context_synthetic_recognition.core.rips.selection import select_precedents
from context_synthetic_recognition.data.samples import AnswerKey, Decisions, UnlabelledSample
from context_synthetic_recognition.data.schema import Dataset
from context_synthetic_recognition.data.splits import FixedSplit
from context_synthetic_recognition.errors import CSRError
from context_synthetic_recognition.evaluation.metrics import ClassificationMetrics
from context_synthetic_recognition.evaluation.protocols import (
    EvaluationCancelledError,
    KFoldParams,
    stratified_k_fold,
)
from context_synthetic_recognition.services.classifier_parts import (
    ClassifierPart,
    SampleOption,
    TrainingSample,
    metrics_against,
)

SEED = 42
"""The first seed of every random choice; repetition i uses ``SEED + i``."""
REPETITIONS = 100
"""Random training samples of the control (seeds 42 … 141)."""
FOLDS = 10
"""Folds of the cross-validation and of the selection inside folds (stratified, seed 42)."""
FRACTIONS = (0.25, 0.5, 0.75)
"""Shares of the training sample of the training-size curve (and 100 %, the design as given)."""
CURVE_REPETITIONS = 20
"""Repetitions of each share of the curve (seeds 42 … 61)."""
GROUPS = ("Swapping the subsets", "Comparison points", "Honest estimate", "Optional extras")
"""The headings the designs are grouped under, in order."""
DIRECT = "direct"
"""The key of the design of the article: fit on T, classify Q."""
BUILTIN_DESIGNS = ("context_synthetic_recognition.services.designs",)
"""Modules that register the built-in designs when they are imported."""

Cancelled = Callable[[], bool]
Progress = Callable[[str, int, int], None]


# ---------------------------------------------------------------- plans


@dataclass(frozen=True, eq=False)
class DesignSplit:
    """A split as a design reads it: every object with its class, and the two samples."""

    key: str
    """Internal key (``German-341``)."""
    label: str
    """``German · Euclidean split · training 341 (173, 168)``."""
    split: FixedSplit
    metric: str
    """The metric the split was made with (its file's name): R-IPS uses it."""
    file: str = ""
    sha256: str = ""

    @property
    def dataset(self) -> Dataset:
        """Every object of the file."""
        return self.split.dataset

    @property
    def group(self) -> str:
        """The dataset the split belongs to (the part of the label before the first ``·``)."""
        return self.label.split(" · ", 1)[0]


@dataclass(frozen=True, eq=False)
class Round:
    """One fit and one classification of a design."""

    label: str
    """``repetition 3 (seed 44)``, ``fold 7`` … (empty for a design of one round)."""
    training: IntArray
    """Indices of the objects to fit on, in the split's file."""
    classified: IntArray
    """Indices of the objects to classify, in the file of ``source``."""
    self_check: bool = False
    """The training objects themselves, classified by resubstitution (Definition 2)."""
    source: str = ""
    """The key of the split whose objects are classified (empty: this split)."""
    note: str = ""

    def __post_init__(self) -> None:
        """Store read-only index arrays."""
        object.__setattr__(self, "training", readonly(np.asarray(self.training, dtype=np.int64)))
        object.__setattr__(
            self, "classified", readonly(np.asarray(self.classified, dtype=np.int64))
        )


@dataclass(frozen=True, eq=False)
class Case:
    """The rounds of a design that belong together (one share of the curve, one target split)."""

    key: str
    title: str
    """What the case is, e.g. ``25 % of the training sample`` (empty for the design itself)."""
    count: int
    """Number of rounds."""
    plan: Callable[[int], Round] = field(repr=False)
    """The round with this index (made when it is needed: R-IPS runs here)."""
    unavailable: str = ""
    """Why the case cannot be run on this split (empty if it can)."""


@runtime_checkable
class EvaluationDesign(Protocol):
    """A way of using a split's objects for training and for classification."""

    @property
    def key(self) -> str:
        """Registry name."""
        ...

    @property
    def title(self) -> str:
        """``Training → Test (as given)`` and so on."""
        ...

    @property
    def group(self) -> str:
        """One of :data:`GROUPS`."""
        ...

    @property
    def explanation(self) -> str:
        """One plain line: what the design shows."""
        ...

    @property
    def settings(self) -> str:
        """Its fixed settings (read-only on the page), or empty."""
        ...

    @property
    def repeated(self) -> bool:
        """Whether a case has several rounds (the tables add the SD, min and max)."""
        ...

    @property
    def main_case(self) -> str:
        """The case the *All designs* table shows."""
        ...

    def cases(self, split: DesignSplit, others: Sequence[DesignSplit]) -> tuple[Case, ...]:
        """The cases of the design on ``split``; ``others`` are the other ticked splits."""
        ...

    def extras(self, split: DesignSplit, cases: Sequence[CaseResult]) -> dict[str, Any]:
        """What the design reports besides the metrics (the percentile, R-IPS …)."""
        ...


DESIGNS: Registry[EvaluationDesign] = Registry("evaluation designs")
"""The designs, in registration order (the order of the page's chooser)."""


def designs() -> list[EvaluationDesign]:
    """The registered designs (the built-in ones and those of entry points)."""
    for module in BUILTIN_DESIGNS:
        importlib.import_module(module)
    DESIGNS.load_entry_points()
    return [info.obj for info in DESIGNS]


def _nothing(index: int) -> Round:
    """The plan of a case that cannot run (never asked for)."""
    del index
    empty = np.empty(0, dtype=np.int64)
    return Round("", empty, empty)


def _always(planned: Round) -> Callable[[int], Round]:
    """The plan of a case of one round."""

    def plan(index: int) -> Round:
        del index
        return planned

    return plan


def _one(label: str, training: IntArray, classified: IntArray, **kwargs: Any) -> tuple[Case, ...]:
    single = Round(label, training, classified, **kwargs)
    return (Case("as-given", "", 1, lambda _: single),)


def stratified_sample(
    class_index: IntArray, pool: IntArray, sizes: Sequence[int], seed: int
) -> IntArray:
    """``sizes[c]`` objects of every class c drawn from ``pool`` without replacement, sorted."""
    rng = np.random.default_rng(seed)
    chosen = []
    for c, size in enumerate(sizes):
        members = pool[class_index[pool] == c]
        chosen.append(rng.choice(members, size=size, replace=False))
    return np.sort(np.concatenate(chosen)).astype(np.int64)


def share(size: int, fraction: float) -> int:
    """``fraction`` of ``size`` objects, rounded half up, at least one."""
    return max(1, math.floor(size * fraction + 0.5))


@dataclass(frozen=True)
class BaseDesign:
    """The fields every design has; a design overrides :meth:`cases` (and :meth:`extras`)."""

    key: str
    title: str
    group: str
    explanation: str
    settings: str = ""
    repeated: bool = False
    main_case: str = "as-given"
    self_check: ClassVar[bool] = False
    """The design classifies the training objects by resubstitution."""
    selects: ClassVar[bool] = False
    """Planning a round chooses a training sample (R-IPS): it costs a computation."""

    def cases(self, split: DesignSplit, others: Sequence[DesignSplit]) -> tuple[Case, ...]:
        """The cases (overridden)."""
        raise NotImplementedError

    def extras(self, split: DesignSplit, cases: Sequence[CaseResult]) -> dict[str, Any]:
        """Nothing besides the metrics."""
        del split, cases
        return {}


class Direct(BaseDesign):
    """1 — fit on T, classify Q: the article's evaluation."""

    def cases(self, split: DesignSplit, others: Sequence[DesignSplit]) -> tuple[Case, ...]:
        """One round: T → Q."""
        return _one("", split.split.training, split.split.test)


class Reversed(BaseDesign):
    """2 — fit on Q, classify T."""

    def cases(self, split: DesignSplit, others: Sequence[DesignSplit]) -> tuple[Case, ...]:
        """One round: Q → T."""
        return _one("", split.split.test, split.split.training)


@dataclass(frozen=True)
class SelfCheck(BaseDesign):
    """3, 4 — a sample on itself, by resubstitution (Definition 2)."""

    sample: str = "training"
    self_check: ClassVar[bool] = True

    def cases(self, split: DesignSplit, others: Sequence[DesignSplit]) -> tuple[Case, ...]:
        """One round: the sample classified by resubstitution."""
        objects = split.split.training if self.sample == "training" else split.split.test
        return _one("", objects, objects, self_check=True)


class RandomControl(BaseDesign):
    """5 — random training samples with T's size and class counts; and T itself, for the rank."""

    def cases(self, split: DesignSplit, others: Sequence[DesignSplit]) -> tuple[Case, ...]:
        """100 random samples (seeds 42 … 141), then the sample as given."""
        data = split.dataset
        everyone = np.arange(data.m, dtype=np.int64)
        sizes = split.split.training_sizes[1:]

        def plan(i: int) -> Round:
            seed = SEED + i
            training = stratified_sample(data.class_index, everyone, sizes, seed)
            return Round(
                f"repetition {i + 1} (seed {seed})", training, np.setdiff1d(everyone, training)
            )

        given = Round("as given", split.split.training, split.split.test)
        return (
            Case("random", f"{REPETITIONS} random training samples", REPETITIONS, plan),
            Case("as-given", "the training sample as given", 1, lambda _: given),
        )

    def extras(self, split: DesignSplit, cases: Sequence[CaseResult]) -> dict[str, Any]:
        """Where the sample as given ranks among the random ones, per method (percentile)."""
        found = {c.key: c for c in cases}
        if "random" not in found or "as-given" not in found:
            return {}
        ranks: dict[str, float | None] = {}
        for key, _ in found["random"].methods:
            given = found["as-given"].accuracies(key)
            random = found["random"].accuracies(key)
            ranks[key] = percentile_rank(random, given[0]) if given and random else None
        return {"percentile": ranks}


def percentile_rank(values: Sequence[float], value: float) -> float:
    """The share of ``values`` below ``value``, ties counted half, in % (two decimals)."""
    below = sum(1 for v in values if v < value)
    equal = sum(1 for v in values if v == value)
    return round(100 * (below + equal / 2) / len(values), 2)


def _folds(split: DesignSplit) -> list[tuple[IntArray, IntArray]]:
    folds = stratified_k_fold(split.dataset.class_index, SEED, KFoldParams(folds=FOLDS))
    return [(f.train, f.test) for f in folds]


class CrossValidation(BaseDesign):
    """6 — stratified 10-fold cross-validation on the m objects; the split markers are ignored."""

    def cases(self, split: DesignSplit, others: Sequence[DesignSplit]) -> tuple[Case, ...]:
        """Ten rounds, one per fold."""
        folds = _folds(split)

        def plan(i: int) -> Round:
            training, test = folds[i]
            return Round(f"fold {i + 1}", training, test)

        return (Case("folds", f"{len(folds)} folds", len(folds), plan),)


def rips_config(metric: str) -> RIPSConfig:
    """The article's R-IPS with the split's metric (ADR-095).

    On every feature, or on the quantitative features for a metric of quantitative features only.
    """
    domain = traits_of(METRICS.get(metric)).domain
    return RIPSConfig(
        metric=plugin(metric), features="quantitative" if domain == "quantitative" else "all"
    )


class NestedSelection(BaseDesign):
    """7 — in each fold, R-IPS picks the training sample from the fold's training objects."""

    selects: ClassVar[bool] = True

    def cases(self, split: DesignSplit, others: Sequence[DesignSplit]) -> tuple[Case, ...]:
        """Ten rounds; R-IPS runs on the fold's training objects only (``core.rips``)."""
        folds = _folds(split)
        config = rips_config(split.metric)

        def plan(i: int) -> Round:
            pool, test = folds[i]
            result = select_precedents(split.dataset.subset(pool), config)
            chosen = np.sort(pool[result.training])
            return Round(
                f"fold {i + 1}",
                chosen,
                test,
                note=f"R-IPS kept {chosen.size} of the fold's {pool.size} training objects",
            )

        return (Case("folds", f"{len(folds)} folds", len(folds), plan),)

    def extras(self, split: DesignSplit, cases: Sequence[CaseResult]) -> dict[str, Any]:
        """The sizes R-IPS selected, and whether R-IPS on all m objects gives the file's T."""
        sizes = [r.training_sizes[0] for c in cases for r in c.rounds if r.training_sizes]
        return {"selected_sizes": sizes, "rips_on_all_objects": rips_reproduction(split)}


def rips_reproduction(split: DesignSplit) -> dict[str, Any]:
    """R-IPS on every object of the file against the file's training sample (reported only)."""
    try:
        result = select_precedents(split.dataset, rips_config(split.metric))
    except CSRError as error:
        return {"error": str(error)}
    rips = {int(i) for i in result.training}
    given = {int(i) for i in split.split.training}
    return {
        "metric": split.metric,
        "rips_training": len(rips),
        "file_training": len(given),
        "in_both": len(rips & given),
        "rips_only": len(rips - given),
        "file_only": len(given - rips),
        "reproduced": rips == given,
    }


class SizeCurve(BaseDesign):
    """8 — stratified random subsamples of T (25, 50, 75 %), and T itself (100 %)."""

    def cases(self, split: DesignSplit, others: Sequence[DesignSplit]) -> tuple[Case, ...]:
        """Twenty rounds per share (seeds 42 … 61), then one round with the whole of T."""
        data = split.dataset
        training, test = split.split.training, split.split.test
        counts = split.split.training_sizes[1:]
        found = []
        for fraction in FRACTIONS:
            sizes = [share(n, fraction) for n in counts]

            def plan(i: int, sizes: Sequence[int] = sizes) -> Round:
                seed = SEED + i
                chosen = stratified_sample(data.class_index, training, sizes, seed)
                return Round(f"repetition {i + 1} (seed {seed})", chosen, test)

            percent = round(100 * fraction)
            found.append(
                Case(f"{percent}", f"{percent} % of the training sample", CURVE_REPETITIONS, plan)
            )
        whole = Round("", training, test)
        found.append(Case("100", "100 % of the training sample", 1, lambda _: whole))
        return tuple(found)


def match_objects(a: Dataset, b: Dataset) -> tuple[IntArray | None, str]:
    """For every object of ``b``, the object of ``a`` with the same row (values and class).

    Returns ``(mapping, "")`` or ``(None, reason)`` when the files do not hold the same objects
    with the same values and types. Identical rows are matched in the order of the files.
    """
    if a.feature_types != b.feature_types:
        return None, (
            f"the features differ ({a.n} and {b.n} features, or other types): the same objects "
            "need the same features"
        )
    if a.m != b.m:
        return None, f"{a.m} and {b.m} objects: not the same objects"
    pools: dict[tuple[float, ...], list[int]] = {}
    for i in range(a.m):
        pools.setdefault((*a.X[i].tolist(), float(a.class_index[i])), []).append(i)
    mapping = np.empty(b.m, dtype=np.int64)
    for j in range(b.m):
        row = (*b.X[j].tolist(), float(b.class_index[j]))
        pool = pools.get(row)
        if not pool:
            return None, (
                f"object {j + 1} of the second file has no object with the same values and class "
                "in the first (other values, e.g. scaled ones)"
            )
        mapping[j] = pool.pop(0)
    return mapping, ""


class CrossSplit(BaseDesign):
    """9 — fit on split A's T, classify split B's Q without the objects of A's T."""

    def cases(self, split: DesignSplit, others: Sequence[DesignSplit]) -> tuple[Case, ...]:
        """One case per other ticked split of the same dataset."""
        targets = [o for o in others if o.group == split.group and o.key != split.key]
        if not targets:
            return (
                Case(
                    "none",
                    "",
                    0,
                    _nothing,
                    unavailable=f"no other ticked split of {split.group}",
                ),
            )
        found = []
        for target in targets:
            title = f"→ {target.label}"
            mapping, reason = match_objects(split.dataset, target.dataset)
            if mapping is None:
                found.append(Case(target.key, title, 0, _nothing, unavailable=reason))
                continue
            in_training = np.isin(mapping, split.split.training)
            test = target.split.test
            classified = test[~in_training[test]]
            single = Round(
                "",
                split.split.training,
                classified,
                source=target.key,
                note=f"{test.size - classified.size} of its test objects are in this training "
                "sample and left out",
            )
            found.append(Case(target.key, title, 1, _always(single)))
        return tuple(found)


# ---------------------------------------------------------------- results


@dataclass(frozen=True, eq=False)
class RoundResult:
    """What one round gave: the sizes and every method's metrics."""

    label: str
    training_sizes: tuple[int, ...]
    """|training|, then per class."""
    classified_sizes: tuple[int, ...]
    """|classified|, then per class."""
    metrics: Mapping[str, ClassificationMetrics | None]
    """Method key → metrics (``None``: undefined on this round)."""
    note: str = ""
    titles: Mapping[str, str] = field(default_factory=dict)
    """Method key → its title."""
    undefined: Mapping[str, str] = field(default_factory=dict)
    """Method key → why it is undefined on this round."""

    @property
    def majority_rate(self) -> float:
        """Share of the larger class among the classified objects."""
        return max(self.classified_sizes[1:]) / self.classified_sizes[0]


@dataclass(frozen=True)
class Statistic:
    """A percentage over the rounds: mean, standard deviation (n − 1), min and max."""

    mean: float
    sd: float | None
    low: float
    high: float

    @classmethod
    def of(cls, values: Sequence[float]) -> Statistic | None:
        """The statistic of some percentages (``None`` if there are none)."""
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        sd = float(array.std(ddof=1)) if array.size > 1 else None
        return cls(
            round(float(array.mean()), 2),
            None if sd is None else round(sd, 2),
            round(float(array.min()), 2),
            round(float(array.max()), 2),
        )


@dataclass(frozen=True, eq=False)
class CaseResult:
    """Every round of a case."""

    key: str
    title: str
    methods: tuple[tuple[str, str], ...]
    """(key, title) of the methods, in the part's order."""
    rounds: tuple[RoundResult, ...] = ()
    unavailable: str = ""

    def accuracies(self, method: str) -> list[float]:
        """The method's accuracy on every round where it is defined, %."""
        return [100 * m.accuracy for r in self.rounds if (m := r.metrics.get(method)) is not None]

    def balanced(self, method: str) -> list[float]:
        """The method's balanced accuracy on every round where it is defined, %."""
        return [
            100 * m.macro_recall for r in self.rounds if (m := r.metrics.get(method)) is not None
        ]

    def refusals(self, method: str) -> list[int]:
        """The method's refusals on every round where it is defined."""
        return [m.refusals for r in self.rounds if (m := r.metrics.get(method)) is not None]

    def undefined(self, method: str) -> int:
        """Rounds where the method is undefined."""
        return sum(1 for r in self.rounds if r.metrics.get(method) is None)


@dataclass(frozen=True, eq=False)
class SplitDesignResult:
    """A design on one split."""

    key: str
    label: str
    cases: tuple[CaseResult, ...]
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class DesignResult:
    """A design on every ticked split."""

    design: str
    splits: tuple[SplitDesignResult, ...]
    seconds: float = 0.0


# ---------------------------------------------------------------- running


def _sizes(class_index: IntArray, classes: int) -> tuple[int, ...]:
    counts = np.bincount(class_index, minlength=classes)
    return (int(class_index.size), *(int(c) for c in counts))


def _round(
    part: ClassifierPart,
    option: SampleOption,
    split: DesignSplit,
    planned: Round,
    sources: Mapping[str, DesignSplit],
) -> tuple[dict[str, Decisions], dict[str, str]]:
    """The decisions of one round and the methods' titles.

    The classified objects reach the part as an
    :class:`~context_synthetic_recognition.data.samples.UnlabelledSample` (ADR-085). A self-check
    takes the decisions of the training stage instead: the training objects classified by
    resubstitution (Definition 2), each left out of its own classification where the method
    allows it.
    """
    training = split.dataset.subset(planned.training)
    sample = TrainingSample(split.key, split.label, training, split.file, split.sha256)
    trained = part.train(option, (sample,))
    titles = {m.method: m.title for m in trained.methods[split.key]}
    if planned.self_check:
        return {
            m.method: Decisions(m.decisions, None, m.undefined) for m in trained.methods[split.key]
        }, titles
    source = sources[planned.source] if planned.source else split
    objects = planned.classified
    unlabelled = UnlabelledSample(
        objects,
        source.dataset.X[objects],
        source.dataset.feature_types,
        source.dataset.feature_names,
        f"{source.label}: {planned.label or 'classified objects'}",
    )
    return dict(part.classify(trained, (unlabelled,)).decisions[split.key]), titles


def round_decisions(
    part: ClassifierPart,
    option: SampleOption,
    split: DesignSplit,
    planned: Round,
    sources: Mapping[str, DesignSplit],
) -> dict[str, Decisions]:
    """The decisions of one round: fit on its training objects, classify its objects blind.

    The classified objects reach the part as an
    :class:`~context_synthetic_recognition.data.samples.UnlabelledSample` (ADR-085). A self-check
    takes the decisions of the training stage instead: the training objects classified by
    resubstitution (Definition 2), each left out of its own classification where the method
    allows it.
    """
    return _round(part, option, split, planned, sources)[0]


def run_round(
    part: ClassifierPart,
    option: SampleOption,
    split: DesignSplit,
    planned: Round,
    sources: Mapping[str, DesignSplit],
) -> RoundResult:
    """One round: its decisions (:func:`round_decisions`), then the classes.

    The classes of the classified objects are read only now (the blind rule, ADR-085).
    """
    data = split.dataset
    classes = len(data.classes)
    decided, titles = _round(part, option, split, planned, sources)
    source = sources[planned.source] if planned.source else split
    objects = planned.classified
    truth = source.dataset.class_index[objects]
    answers = AnswerKey(objects, truth + 1, source.dataset.classes, source.label)
    return RoundResult(
        planned.label,
        _sizes(data.class_index[planned.training], classes),
        _sizes(truth, classes),
        metrics_against(decided, answers),
        planned.note,
        titles,
        {key: d.undefined for key, d in decided.items() if d.decisions is None},
    )


def run_design(
    design: EvaluationDesign,
    part: ClassifierPart,
    option: SampleOption,
    splits: Sequence[DesignSplit],
    *,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> DesignResult:
    """Every round of a design on every split, through the part's stages.

    Args:
        design: The design.
        part: The classifier part whose methods are run.
        option: The ticked samples.
        splits: The splits, in the tree's order.
        progress: Called with (what, done, total) before every round.
        cancelled: Polled before every round; ``True`` stops with
            :class:`~context_synthetic_recognition.evaluation.protocols.EvaluationCancelledError`.
    """
    started = time.perf_counter()
    sources = {s.key: s for s in splits}
    planned = [(s, design.cases(s, [o for o in splits if o.key != s.key])) for s in splits]
    total = sum(c.count for _, cases in planned for c in cases if not c.unavailable)
    done = 0
    found = []
    for split, cases in planned:
        results = []
        for case in cases:
            if case.unavailable:
                results.append(CaseResult(case.key, case.title, (), unavailable=case.unavailable))
                continue
            rounds = []
            for index in range(case.count):
                if cancelled is not None and cancelled():
                    raise EvaluationCancelledError("the design was cancelled")
                if progress is not None:
                    where = " · ".join(t for t in (case.title, f"{index + 1} / {case.count}") if t)
                    progress(f"{split.label} · {where}", done, total)
                try:
                    rounds.append(run_round(part, option, split, case.plan(index), sources))
                except (CSRError, ValueError) as error:
                    rounds.append(
                        RoundResult(f"round {index + 1}", (), (), {}, f"not run: {error}")
                    )
                done += 1
            methods = _methods(rounds)
            results.append(CaseResult(case.key, case.title, methods, tuple(rounds)))
        found.append(
            SplitDesignResult(
                split.key, split.label, tuple(results), design.extras(split, tuple(results))
            )
        )
    if progress is not None:
        progress("done", total, total)
    return DesignResult(design.key, tuple(found), time.perf_counter() - started)


def _methods(rounds: Sequence[RoundResult]) -> tuple[tuple[str, str], ...]:
    keys: dict[str, str] = {}
    for r in rounds:
        for key in r.metrics:
            keys.setdefault(key, r.titles.get(key, key))
    return tuple(keys.items())


# ---------------------------------------------------------------- the built-in designs

BUILTIN = (
    Direct(
        DIRECT,
        "Training → Test (as given)",
        GROUPS[0],
        "Fit on the training sample, classify the test sample: the article's evaluation.",
    ),
    Reversed(
        "reversed",
        "Test → Training (reversed)",
        GROUPS[0],
        "Fit on the test sample, classify the training sample: the two samples swapped.",
    ),
    SelfCheck(
        "training-self",
        "Training → Training (self-check)",
        GROUPS[0],
        "The training sample on itself: each object left out of its own classification where the "
        "method allows it (Definition 2).",
        settings="Resubstitution as in Definition 2: a k-NN object is not its own neighbour, a "
        "CS-model object is left out of its own context; SVM and Random Forest classify the "
        "objects they were fitted on.",
        sample="training",
    ),
    SelfCheck(
        "test-self",
        "Test → Test (self-check)",
        GROUPS[0],
        "The test sample on itself, the same way: how well the test objects explain each other.",
        settings="Resubstitution as in Definition 2, on the test sample.",
        sample="test",
    ),
    RandomControl(
        "random-control",
        "Random training samples (control)",
        GROUPS[1],
        "Random training samples of the same size and classes: is the given one better than "
        "chance?",
        settings=f"{REPETITIONS} samples drawn from all objects with the training sample's size "
        f"and class counts, seeds {SEED} … {SEED + REPETITIONS - 1}; each fitted and the other "
        "objects classified. The percentile ranks the given sample among them.",
        repeated=True,
        main_case="random",
    ),
    CrossValidation(
        "cross-validation",
        "Whole dataset, cross-validation",
        GROUPS[1],
        "Stratified 10-fold cross-validation on all objects; the split markers are ignored.",
        settings=f"Stratified {FOLDS}-fold, seed {SEED}; mean and SD over the folds.",
        repeated=True,
        main_case="folds",
    ),
    NestedSelection(
        "nested-selection",
        "Selection inside folds",
        GROUPS[2],
        "R-IPS picks the training sample from each fold's training objects only: an honest "
        "estimate of selection and fit together.",
        settings=f"Stratified {FOLDS}-fold, seed {SEED}; in each fold R-IPS (the article's "
        "defaults, the split's metric) on the fold's training objects; fitted on its training "
        "sample, the held-out fold classified.",
        repeated=True,
        main_case="folds",
    ),
    SizeCurve(
        "size-curve",
        "Training-size curve",
        GROUPS[3],
        "Smaller stratified parts of the training sample: how the accuracy grows with its size.",
        settings="25, 50 and 75 % of each class of the training sample (rounded half up), "
        f"{CURVE_REPETITIONS} repetitions each, seeds {SEED} … {SEED + CURVE_REPETITIONS - 1}; "
        "100 % is the design as given. The test sample is classified.",
        repeated=True,
        main_case="50",
    ),
    CrossSplit(
        "cross-split",
        "Split to split",
        GROUPS[3],
        "Fit on one split's training sample, classify another split's test objects of the same "
        "dataset.",
        settings="Every ordered pair of ticked splits of one dataset whose files hold the same "
        "objects with identical values and types (matched one to one by their rows); the test "
        "objects that are in the first split's training sample are left out.",
    ),
)
"""The nine designs of ADR-095, in the chooser's order."""

for _design in BUILTIN:
    DESIGNS.add(_design.key, _design, summary=_design.title)
