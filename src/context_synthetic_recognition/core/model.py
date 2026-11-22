"""The CS-model end to end: (S, E) → Ψ_ρ,k(S, E) → D(S, E) → Y(S, E) → R(Y(S, E)).

:func:`fit_model` fits, on a training sample E,

- Steps 1–8: local contexts and the synthetic features Ψ(r) with ω and η (:func:`fit_context`);
- Step 9: the hierarchical agglomerative grouping — TUPLAM and the additional features d₁ … d_p
  (:func:`~context_synthetic_recognition.core.additional_features.agglomerative.agglomerative`);
- Step 10: the meta-dataset Y = (y₀, …, y_p, d₁, …, d_p);
- Step 11: the training description (aᵢ, dᵢ) of the meta-algorithm;

and :class:`CSModel` classifies objects with the meta-algorithm (Step 12). A new object is
represented relative to the fixed E and classified **without its class**: the methods
:meth:`CSModel.represent`, :meth:`CSModel.classify` and :meth:`CSModel.predict` have no label
argument (Theorem, ADR-009).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.config.models import ExperimentConfig
from context_synthetic_recognition.core.additional_features.agglomerative import (
    AgglomerativeResult,
    AgglomerativeSettings,
    agglomerative,
)
from context_synthetic_recognition.core.arrays import FloatArray, IntArray, freeze_fields
from context_synthetic_recognition.core.meta_algorithm.algorithm import (
    DECISION_RULES,
    REFUSAL,
    MetaDescription,
    MetaResult,
    meta_classify,
)
from context_synthetic_recognition.core.synthetic_features.psi import PsiOperator, fit_psi
from context_synthetic_recognition.core.trace import PsiRepresentation, PsiTrace
from context_synthetic_recognition.data.schema import Dataset, Label
from context_synthetic_recognition.notation import additional_name, subscript, synthetic_name

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class MetaTable:
    """Y = (y₀, …, y_p, d₁, …, d_p) — Step 10 (sheet *Dataset for Meta-algorithm*)."""

    tuplam: tuple[int, ...]
    """The TUPLAM features (0-based), in selection order."""
    initial: FloatArray
    """(m × (p + 1)) yⱼ: the contribution values ηᵤ(a_tu) of the TUPLAM features."""
    additional: FloatArray
    """(m × p) dⱼ: the additional (latent) features of the hierarchical agglomerative algorithm."""
    class_index: IntArray
    """(m,) 0 = K1, 1 = K2."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)

    @property
    def p(self) -> int:
        """Number of additional features."""
        return len(self.tuplam) - 1

    @property
    def Y(self) -> FloatArray:  # noqa: N802 - the article's symbol
        """(m × (2p + 1)) the meta-dataset (y₀ … y_p, d₁ … d_p)."""
        return np.hstack([self.initial, self.additional])

    @property
    def columns(self) -> tuple[str, ...]:
        """Column names ``y₀ (a₆)``, …, ``d₁``, … of :attr:`Y`."""
        initial = tuple(f"y{subscript(j)} ({synthetic_name(u)})" for j, u in enumerate(self.tuplam))
        return initial + tuple(additional_name(j) for j in range(self.p))


@dataclass(frozen=True, eq=False)
class Representation:
    """Objects described relative to the training sample E, without their class (Theorem)."""

    context: PsiRepresentation
    """Their local contexts and synthetic features Ψ(r) (formula (5))."""
    description: IntArray
    """(q × (p + 1)) a₀ … a_p: the gradations of the TUPLAM features — the meta-algorithm input."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)


@dataclass(frozen=True, eq=False)
class Classification:
    """Objects classified by the meta-algorithm (Steps 1–5)."""

    meta: MetaResult
    """B1/B2 sizes, scores and decisions."""
    classes: tuple[Label, ...]
    """The class labels of K1 and K2."""
    representation: Representation | None = None
    """How the objects were described (``None`` for the training rows themselves)."""

    @property
    def decisions(self) -> IntArray:
        """(q,) 1 = K1, 2 = K2, 0 = refusal."""
        return self.meta.decisions

    @property
    def labels(self) -> tuple[Label | None, ...]:
        """The predicted class labels; ``None`` for a refusal."""
        return tuple(
            None if d == REFUSAL else self.classes[int(d) - 1] for d in self.meta.decisions
        )

    @property
    def scores(self) -> FloatArray:
        """(q,) score₁ − score₂ = |B1(a_p)|/|K1| − |B2(a_p)|/|K2|."""
        return self.meta.scores

    @property
    def refusals(self) -> int:
        """Number of refusals."""
        return int(np.count_nonzero(self.meta.decisions == REFUSAL))


@dataclass(frozen=True, eq=False)
class CSModel:
    """The fitted CS-model: Steps 1–11 on the training sample E, ready to classify."""

    config: ExperimentConfig
    """The configuration it was fitted with."""
    psi: PsiOperator
    """Steps 1–8: scaling, operators, admissible k, Ψ(r) with ω and η."""
    agglomerative: AgglomerativeResult
    """Step 9: TUPLAM and the additional features."""
    meta_table: MetaTable
    """Step 10: Y = (y, d)."""
    description: MetaDescription
    """Step 11: the training description (aᵢ, dᵢ) of the meta-algorithm."""
    classes: tuple[Label, ...]
    """The class labels of K1 and K2."""

    @property
    def trace(self) -> PsiTrace:
        """The Step 1–8 trace (distances, neighbours, Ψ(r), formulas (1)–(6))."""
        return self.psi.trace

    @property
    def tuplam(self) -> tuple[int, ...]:
        """TUPLAM (0-based synthetic features) in selection order."""
        return self.agglomerative.tuplam

    @property
    def p(self) -> int:
        """Number of additional features."""
        return self.agglomerative.p

    def represent(
        self, X: npt.ArrayLike, *, exclude: npt.ArrayLike | None = None
    ) -> Representation:
        """Describe objects relative to E without their class: Ψ(r) and a₀ … a_p (Theorem).

        Args:
            X: (q × n) original feature values (or one object as a vector).
            exclude: One training index per object to leave out of its context (a training
                object passed through this path then gets exactly its training row); ``None``
                for new objects.
        """
        context = self.psi.represent(X, exclude=exclude)
        return Representation(context, context.values[:, list(self.tuplam)])

    def classify(self, X: npt.ArrayLike, *, exclude: npt.ArrayLike | None = None) -> Classification:
        """Represent objects without their class and classify them (meta-algorithm, Steps 1–5).

        Args:
            X: (q × n) original feature values (or one object as a vector).
            exclude: As in :meth:`represent`.
        """
        representation = self.represent(X, exclude=exclude)
        return Classification(self._meta(representation.description), self.classes, representation)

    def predict(self, X: npt.ArrayLike, *, exclude: npt.ArrayLike | None = None) -> IntArray:
        """Decision codes 1 (K1), 2 (K2) or 0 (refusal) for every object."""
        return self.classify(X, exclude=exclude).decisions

    def classify_training(self) -> Classification:
        """Every training object classified with its own training row (resubstitution).

        Its context excludes itself (Steps 3–5), while it stays in the training description of
        the meta-algorithm — training correctness, Definition 2 (sheet *Meta-algorithm (All
        Objects)*).
        """
        return Classification(self._meta(self.description.gradations), self.classes)

    def _meta(self, queries: npt.ArrayLike) -> MetaResult:
        rule = self.config.meta_algorithm.decision_rule
        return meta_classify(self.description, queries, rule.name, rule.params)


def fit_model(dataset: Dataset, config: ExperimentConfig | None = None) -> CSModel:
    """Fit the CS-model on the training sample ``dataset`` (Steps 1–11).

    Args:
        dataset: The training sample E (two classes).
        config: Experiment configuration; the default configuration if ``None``.

    Raises:
        ModelUndefinedError: A stage is undefined for the data (Definition 1).
        DatasetError: More than two classes (ADR-010).
        RegistryError: Unknown plug-in.
        ConfigError: Invalid plug-in parameters.
    """
    cfg = config or ExperimentConfig()
    started = time.perf_counter()
    # resolve the Step 9 and Step 12 plug-ins first: a bad name fails before the costly Steps 1–8
    AgglomerativeSettings.from_config(cfg.additional_features)
    DECISION_RULES.make_params(
        cfg.meta_algorithm.decision_rule.name, cfg.meta_algorithm.decision_rule.params
    )

    psi = fit_psi(dataset, cfg)
    trace = psi.trace
    grouping = agglomerative(
        trace.contributions, trace.informativeness, trace.class_index, cfg.additional_features
    )
    columns = list(grouping.tuplam)
    meta_table = MetaTable(
        tuplam=grouping.tuplam,
        initial=trace.contributions[:, columns],
        additional=grouping.additional,
        class_index=trace.class_index,
    )
    description = MetaDescription(trace.values[:, columns], grouping.additional, trace.class_index)
    logger.info(
        "model fitted (Steps 1–11)",
        extra={
            "dataset": dataset.name,
            "tuplam": grouping.label,
            "p": grouping.p,
            "stop": grouping.stop.value,
            "seconds": round(time.perf_counter() - started, 3),
        },
    )
    return CSModel(
        config=cfg,
        psi=psi,
        agglomerative=grouping,
        meta_table=meta_table,
        description=description,
        classes=trace.classes,
    )
