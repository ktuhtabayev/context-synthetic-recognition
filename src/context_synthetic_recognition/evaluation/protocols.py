"""Evaluation protocols — training correctness (Definition 2) and generalization (Definition 3).

Registry :data:`PROTOCOLS` of splitters:

``resubstitution``
    The model fitted on every object classifies each training object with its own training row
    (its context excludes itself; sheet *Meta-algorithm (All Objects)*, Definition 2).
``leave-one-out``
    m folds: the **whole pipeline** — scale unification, permitted k, Ψ(r), ω, η, HAG and the
    meta-algorithm — is re-fitted on the other m − 1 objects, and the held-out object is
    classified without its class (sheet *Leave-One-Out*, Definition 3, ADR-009).
``stratified-k-fold``
    ``folds`` folds (10), every class dealt evenly over them after a shuffle with the seed.
``repeated-k-fold``
    ``repeats`` (10) stratified k-folds with the seeds seed, seed + 1, …
``hold-out``
    One stratified split, ``test_size`` (0.3) of every class held out.

A fold whose training part leaves the model undefined (e.g. no permitted k when the smallest class
has one object) refuses its held-out objects (decision 0, score 0) and is flagged (ADR-005). The
baselines are evaluated on the same folds.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass as params_dataclass

from context_synthetic_recognition.config.models import ExperimentConfig, PluginSpec
from context_synthetic_recognition.core.arrays import FloatArray, IntArray, freeze_fields
from context_synthetic_recognition.core.model import CSModel, fit_model
from context_synthetic_recognition.core.registry import PARAMS_CONFIG, Registry
from context_synthetic_recognition.data.schema import Dataset
from context_synthetic_recognition.errors import CSRError, ModelUndefinedError
from context_synthetic_recognition.evaluation.baselines import (
    BASELINES,
    BaselineInput,
    refusals,
)
from context_synthetic_recognition.evaluation.metrics import (
    ClassificationMetrics,
    classification_metrics,
)
from context_synthetic_recognition.evaluation.roc import RocCurve, roc_curve

logger = logging.getLogger(__name__)

CS_MODEL = "CS-model"
"""Method name of the meta-algorithm's decisions."""


class EvaluationCancelledError(CSRError):
    """The evaluation was cancelled by the caller (GUI cancel button)."""


@dataclass(frozen=True, eq=False)
class Split:
    """One fold: which objects train the model and which are classified."""

    train: IntArray
    test: IntArray
    repeat: int = 0
    resubstitution: bool = False
    """The training objects themselves are classified with their training rows."""

    def __post_init__(self) -> None:
        """Freeze the index arrays."""
        freeze_fields(self)


class Splitter(Protocol):
    """Produces the folds of a protocol from the class of every object."""

    def __call__(self, class_index: IntArray, seed: int, params: Any, /) -> list[Split]:
        """Return the folds."""
        ...


PROTOCOLS: Registry[Splitter] = Registry("protocols")
"""Evaluation protocols by name."""


@PROTOCOLS.register(
    "resubstitution", summary="every training object with its own row (Definition 2)"
)
def resubstitution(class_index: IntArray, seed: int, params: Any = None, /) -> list[Split]:
    """One fold: train on everything, classify the training rows."""
    del seed, params
    everything = np.arange(class_index.size, dtype=np.int64)
    return [Split(everything, everything, resubstitution=True)]


@PROTOCOLS.register(
    "leave-one-out",
    aliases=("loo",),
    summary="whole pipeline re-fitted without each object (Definition 3)",
)
def leave_one_out(class_index: IntArray, seed: int, params: Any = None, /) -> list[Split]:
    """M folds, each holding out one object."""
    del seed, params
    everything = np.arange(class_index.size, dtype=np.int64)
    return [Split(np.delete(everything, i), everything[i : i + 1]) for i in range(everything.size)]


@params_dataclass(frozen=True, config=PARAMS_CONFIG)
class KFoldParams:
    """Parameters of ``stratified-k-fold``."""

    folds: int = Field(default=10, ge=2)
    shuffle: bool = True
    """Shuffle every class with the seed before dealing it over the folds."""


def _stratified_folds(
    class_index: IntArray, folds: int, rng: np.random.Generator | None
) -> list[IntArray]:
    """Deal the objects of every class over the folds (round robin), optionally shuffled."""
    if folds > class_index.size:
        raise ModelUndefinedError(f"{folds} folds need at least {folds} objects")
    buckets: list[list[int]] = [[] for _ in range(folds)]
    offset = 0
    for c in np.unique(class_index):
        members = np.flatnonzero(class_index == c)
        if rng is not None:
            members = rng.permutation(members)
        for position, i in enumerate(members):
            buckets[(offset + position) % folds].append(int(i))
        offset += members.size
    return [np.sort(np.array(b, dtype=np.int64)) for b in buckets]


def _complement(test: IntArray, m: int) -> IntArray:
    mask = np.ones(m, dtype=bool)
    mask[test] = False
    return np.flatnonzero(mask).astype(np.int64)


@PROTOCOLS.register(
    "stratified-k-fold",
    aliases=("k-fold", "kfold"),
    summary="stratified k-fold cross-validation (default 10 folds)",
    params_type=KFoldParams,
)
def stratified_k_fold(
    class_index: IntArray, seed: int, params: KFoldParams | None = None, /
) -> list[Split]:
    """Stratified k folds; every object is held out exactly once."""
    chosen = params or KFoldParams()
    rng = np.random.default_rng(seed) if chosen.shuffle else None
    tests = _stratified_folds(class_index, chosen.folds, rng)
    return [Split(_complement(t, class_index.size), t) for t in tests if t.size]


@params_dataclass(frozen=True, config=PARAMS_CONFIG)
class RepeatedParams:
    """Parameters of ``repeated-k-fold``."""

    folds: int = Field(default=10, ge=2)
    repeats: int = Field(default=10, ge=1)


@PROTOCOLS.register(
    "repeated-k-fold",
    summary="stratified k-fold repeated with different shuffles (default 10 × 10)",
    params_type=RepeatedParams,
)
def repeated_k_fold(
    class_index: IntArray, seed: int, params: RepeatedParams | None = None, /
) -> list[Split]:
    """``repeats`` stratified k-folds with the seeds seed, seed + 1, and so on."""
    chosen = params or RepeatedParams()
    splits = []
    for repeat in range(chosen.repeats):
        tests = _stratified_folds(class_index, chosen.folds, np.random.default_rng(seed + repeat))
        splits += [Split(_complement(t, class_index.size), t, repeat) for t in tests if t.size]
    return splits


@params_dataclass(frozen=True, config=PARAMS_CONFIG)
class HoldOutParams:
    """Parameters of ``hold-out``."""

    test_size: float = Field(default=0.3, gt=0.0, lt=1.0)
    """Share of every class held out (at least one object, at least one left for training)."""


@PROTOCOLS.register(
    "hold-out",
    aliases=("holdout",),
    summary="one stratified train/test split (default 30 % held out)",
    params_type=HoldOutParams,
)
def hold_out(
    class_index: IntArray, seed: int, params: HoldOutParams | None = None, /
) -> list[Split]:
    """One stratified split."""
    chosen = params or HoldOutParams()
    rng = np.random.default_rng(seed)
    test: list[int] = []
    for c in np.unique(class_index):
        members = rng.permutation(np.flatnonzero(class_index == c))
        size = min(max(1, round(chosen.test_size * members.size)), members.size - 1)
        test += members[:size].tolist()
    t = np.sort(np.array(test, dtype=np.int64))
    return [Split(_complement(t, class_index.size), t)]


# ---------------------------------------------------------------- results


@dataclass(frozen=True, eq=False)
class FoldResult:
    """The CS-model on one fold."""

    number: int
    """0-based fold number (across repeats)."""
    split: Split
    decisions: IntArray
    """(q,) decisions for the test objects."""
    scores1: FloatArray
    scores2: FloatArray
    descriptions: IntArray | None
    """(q × (p + 1)) the test objects' (a₀ … a_p); ``None`` if the fold is undefined."""
    class_sizes: tuple[int, int] | None
    """|K1|, |K2| of the training part."""
    permitted_k: tuple[int, ...] | None
    r: int | None
    tuplam: tuple[int, ...] | None
    tuplam_names: tuple[str, ...] | None
    """TUPLAM as operator·k (``ρ_J·5``) — the fold's feature numbers differ from the full fit's."""
    undefined: str | None = None
    """Why the model is undefined on this fold (it then refuses its objects)."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)

    @property
    def scores(self) -> FloatArray:
        """score₁ − score₂."""
        return self.scores1 - self.scores2


@dataclass(frozen=True, eq=False)
class MethodPredictions:
    """Decisions of one method under a protocol: one entry per classified (object, repeat)."""

    method: str
    objects: IntArray
    """0-based object of every entry."""
    repeats: IntArray
    truth: IntArray
    """True class codes (1, 2)."""
    decisions: IntArray
    scores: FloatArray

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)

    def metrics(self, positive: int = 1) -> ClassificationMetrics:
        """Confusion counts and metrics over every entry."""
        return classification_metrics(self.truth, self.decisions, positive)

    def roc(self, positive: int = 1, decimals: int = 10) -> RocCurve:
        """AUC and ROC over every entry."""
        return roc_curve(self.scores, self.truth, positive, decimals)

    def repeat_metrics(self, positive: int = 1) -> list[ClassificationMetrics]:
        """Metrics of every repetition separately (repeated k-fold)."""
        return [
            classification_metrics(
                self.truth[self.repeats == r], self.decisions[self.repeats == r], positive
            )
            for r in np.unique(self.repeats)
        ]

    def by_object(self) -> IntArray:
        """Decisions in object order (single-repeat protocols)."""
        out = np.zeros(int(self.objects.max()) + 1, dtype=np.int64)
        out[self.objects] = self.decisions
        return out


@dataclass(frozen=True, eq=False)
class ProtocolResult:
    """A protocol's folds, the CS-model's decisions and the baselines'."""

    protocol: str
    """Canonical protocol name."""
    params: Any
    folds: tuple[FoldResult, ...]
    predictions: MethodPredictions
    """The CS-model."""
    baselines: tuple[MethodPredictions, ...]
    seconds: float = field(default=0.0, compare=False)

    @property
    def undefined_folds(self) -> tuple[FoldResult, ...]:
        """Folds whose model is undefined (their objects are refused)."""
        return tuple(f for f in self.folds if f.undefined is not None)

    @property
    def methods(self) -> tuple[MethodPredictions, ...]:
        """The CS-model first, then the baselines."""
        return (self.predictions, *self.baselines)


# ---------------------------------------------------------------- running


def _feature_names(model: CSModel) -> tuple[str, ...]:
    features = model.trace.features
    return tuple(f"{features[u].operator_label}·{features[u].k}" for u in model.tuplam)


def _fold(
    number: int, split: Split, dataset: Dataset, config: ExperimentConfig
) -> tuple[FoldResult, Dataset]:
    train = dataset.subset(split.train) if not split.resubstitution else dataset
    q = split.test.size
    try:
        model = fit_model(train, config)
    except ModelUndefinedError as error:
        decisions, zeros = refusals(q)
        return FoldResult(
            number,
            split,
            decisions,
            zeros,
            zeros.copy(),
            None,
            None,
            None,
            None,
            None,
            None,
            undefined=str(error),
        ), train
    if split.resubstitution:
        result = model.classify_training()
        descriptions = model.description.gradations
    else:
        result = model.classify(dataset.X[split.test])
        assert result.representation is not None
        descriptions = result.representation.description
    meta = result.meta
    return FoldResult(
        number=number,
        split=split,
        decisions=meta.decisions,
        scores1=meta.scores1,
        scores2=meta.scores2,
        descriptions=np.asarray(descriptions),
        class_sizes=model.trace.class_sizes,
        permitted_k=model.trace.permitted_k.ks,
        r=model.trace.r,
        tuplam=model.tuplam,
        tuplam_names=_feature_names(model),
    ), train


def run_protocol(
    dataset: Dataset,
    config: ExperimentConfig | None = None,
    protocol: PluginSpec | str = "leave-one-out",
    *,
    baselines: Sequence[PluginSpec] = (),
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> ProtocolResult:
    """Evaluate the CS-model (and the baselines) under one protocol.

    Args:
        dataset: All objects (two classes).
        config: The experiment configuration (its ``seed`` shuffles the folds).
        protocol: Protocol name or spec with parameters.
        baselines: Baseline specs evaluated on the same folds.
        progress: Called with (folds done, folds) after every fold.
        cancelled: Polled before every fold; returning ``True`` stops with
            :class:`EvaluationCancelledError`.

    Raises:
        RegistryError: Unknown protocol or baseline.
        ConfigError: Invalid parameters.
        EvaluationCancelled: ``cancelled`` returned ``True``.
    """
    cfg = config or ExperimentConfig()
    spec = PluginSpec(name=protocol) if isinstance(protocol, str) else protocol
    name = PROTOCOLS.info(spec.name).name
    params = PROTOCOLS.make_params(spec.name, spec.params)
    resolved = [(BASELINES.get(b.name), BASELINES.make_params(b.name, b.params)) for b in baselines]
    started = time.perf_counter()
    splits = PROTOCOLS.get(spec.name)(dataset.class_index, cfg.seed, params)
    truth_codes = dataset.class_index + 1
    folds: list[FoldResult] = []
    per_fold: list[dict[str, tuple[IntArray, FloatArray]]] = []
    methods: list[str] = []
    for number, split in enumerate(splits):
        if cancelled is not None and cancelled():
            raise EvaluationCancelledError(
                f"{name}: cancelled after {number} of {len(splits)} folds"
            )
        fold, train = _fold(number, split, dataset, cfg)
        folds.append(fold)
        exclude = split.test if split.resubstitution else None
        outputs: dict[str, tuple[IntArray, FloatArray]] = {}
        if len(train.classes) == 2:  # one class: class numbers would not match — refuse
            for baseline, baseline_params in resolved:
                data = BaselineInput(train, dataset.X[split.test], exclude, cfg)
                for output in baseline(data, baseline_params):
                    outputs[output.method] = (output.decisions, output.scores)
                    if output.method not in methods:
                        methods.append(output.method)
        per_fold.append(outputs)
        if progress is not None:
            progress(number + 1, len(splits))
    objects = np.concatenate([f.split.test for f in folds])
    repeats = np.concatenate([np.full(f.split.test.size, f.split.repeat) for f in folds])
    predictions = MethodPredictions(
        CS_MODEL,
        objects,
        repeats.astype(np.int64),
        truth_codes[objects],
        np.concatenate([f.decisions for f in folds]),
        np.concatenate([f.scores for f in folds]),
    )
    baseline_predictions = tuple(
        MethodPredictions(
            method,
            objects,
            repeats.astype(np.int64),
            truth_codes[objects],
            np.concatenate(
                [
                    out.get(method, refusals(f.split.test.size))[0]
                    for f, out in zip(folds, per_fold, strict=True)
                ]
            ),
            np.concatenate(
                [
                    out.get(method, refusals(f.split.test.size))[1]
                    for f, out in zip(folds, per_fold, strict=True)
                ]
            ),
        )
        for method in methods
    )
    seconds = round(time.perf_counter() - started, 3)
    logger.info(
        "protocol evaluated",
        extra={"protocol": name, "folds": len(folds), "seconds": seconds},
    )
    return ProtocolResult(name, params, tuple(folds), predictions, baseline_predictions, seconds)
