"""Trace objects: every intermediate table of the pipeline as immutable data.

The GUI tables, the exporters and the golden tests read the same trace, so the workbook, the
software and the article's tables cannot drift apart. Indices are 0-based; the display names
(S₁, x₁, a₁) come from :mod:`context_synthetic_recognition.notation`.

This module covers Steps 1–8 (up to Ψ(r), formulas (1)–(6)). The trace of the agglomerative
algorithm (``AgglomerativeResult``, ``AgglomerativeIteration``, ``CandidateScan``) is in
:mod:`.additional_features.agglomerative`, the meta-algorithm's (``MetaResult``, ``MetaSteps``) in
:mod:`.meta_algorithm.algorithm`; :mod:`.model` puts them together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from context_synthetic_recognition.core.arrays import BoolArray, FloatArray, IntArray, freeze_fields
from context_synthetic_recognition.core.local_context.admissible_k import AdmissibleK
from context_synthetic_recognition.core.local_context.neighbours import (
    cumulative_count,
    neighbour_classes,
    rank_matrix,
)
from context_synthetic_recognition.core.local_context.normalizers import Scaling
from context_synthetic_recognition.core.local_context.operators import SkippedOperator
from context_synthetic_recognition.core.synthetic_features.membership import (
    Boundary,
    MembershipTable,
)
from context_synthetic_recognition.notation import synthetic_name


@dataclass(frozen=True, eq=False)
class LocalContext:
    """Steps 2–3 for one base operator: distances and neighbour order of the training sample."""

    label: str
    """ρ, ρ_I, ρ_J, …"""
    metric: str
    """Registry name of the metric."""
    features: IntArray
    """0-based indices of the features the operator uses."""
    distances: FloatArray
    """(m × m) distances, rounded (sheet *Zhuravlev Distances*)."""
    order: IntArray
    """(m × (m − 1)) neighbours of every object by (distance, index), self excluded."""
    normalizer: str = "minmax"
    """Registry name of the normalization the operator's distances use."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)

    @property
    def ranks(self) -> IntArray:
        """(m × m) rank of Sⱼ among the neighbours of Sᵢ; 0 on the diagonal (the workbook's "—")."""
        return rank_matrix(self.order, self.distances.shape[1])

    @property
    def sorted_distances(self) -> FloatArray:
        """(m × (m − 1)) distance to the neighbour at every rank."""
        return np.take_along_axis(self.distances, self.order, axis=1)

    def neighbour_classes(self, class_index: IntArray) -> IntArray:
        """Class index of the neighbour at every rank."""
        return neighbour_classes(self.order, class_index)

    def same_class_running(self, class_index: IntArray) -> IntArray:
        """μ over ranks 1 … t for every rank t (neighbour blocks, row "Same-class count μ")."""
        return cumulative_count(self.neighbour_classes(class_index), class_index)

    def k1_running(self, class_index: IntArray) -> IntArray:
        """χ₁ over ranks 1 … t for every rank t (row "K1 count χ₁" of the neighbour blocks)."""
        return cumulative_count(self.neighbour_classes(class_index), 0)


@dataclass(frozen=True, eq=False)
class SyntheticFeature:
    """Steps 4–8 for one synthetic feature aᵤ = (operator, k).

    Or for an original nominal feature combined with the synthetic ones (``original ≥ 0``,
    ADR-065: its codes are the gradations).
    """

    index: int
    """0-based u; the feature is aᵤ₊₁."""
    operator: int
    """Index of its base operator in :attr:`ContextTrace.operators`."""
    operator_label: str
    """Label of the base operator."""
    k: int
    """Neighbourhood size."""
    mu: IntArray
    """(m,) same-class count μ (Step 4, training side)."""
    chi1: IntArray
    """(m,) number of K1 objects among the k nearest (class-free)."""
    values: IntArray
    """(m,) aᵤ ∈ {1, 2} by formula (5)."""
    membership: MembershipTable
    """Formula (1) over μ = 0 … k."""
    stability: float
    """g_k by formula (2)."""
    boundary: Boundary
    """G_k by formula (3)."""
    object_membership: FloatArray
    """(m,) g(S, k) = f_k(μ_S)."""
    correct: BoolArray
    """(m,) whether S lies on its class's side of G_k."""
    omega: float
    """Informativeness ω by formula (4)."""
    informativeness: float
    """Weight used by formula (6) and the hierarchical agglomerative algorithm (= ω by default)."""
    alpha: IntArray
    """(2 × 2) α counts: ``alpha[j − 1, c]`` = objects of class c with aᵤ = j."""
    eta: FloatArray
    """(2,) contributions η(1), η(2) by formula (6)."""
    contributions: FloatArray
    """(m,) contribution value ηᵤ(a_tu) of every object."""
    zeros: int = 0
    """Training objects whose class is indistinguishable at their gradation μ (f_k(μ) = 0.5):
    the value 0 of the article's experiments (ADR-064)."""
    gradations: tuple[int, ...] = (1, 2)
    """The values j of formula (6): {1, 2}, or the codes of an original feature."""
    original: int = -1
    """0-based index of the original nominal feature; −1 for a synthetic feature."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)

    @property
    def name(self) -> str:
        """aᵤ (1-based)."""
        return synthetic_name(self.index)

    @property
    def is_original(self) -> bool:
        """Whether this is an original nominal feature combined with the synthetic ones."""
        return self.original >= 0

    @property
    def label(self) -> str:
        """``ρ_J·5`` for a synthetic feature, the feature name (``x₃``) for an original one."""
        return self.operator_label if self.is_original else f"{self.operator_label}·{self.k}"

    def eta_at(self, value: int) -> float:
        """η(j) for the value ``value``; NaN if the feature has no such value."""
        if value not in self.gradations:
            return float("nan")
        return float(self.eta[self.gradations.index(value)])

    def alpha_at(self, value: int, class_: int) -> int:
        """α of the value ``value`` in class ``class_`` (0 = K1, 1 = K2); 0 if absent."""
        if value not in self.gradations:
            return 0
        return int(self.alpha[self.gradations.index(value), class_])

    @property
    def chi2(self) -> IntArray:
        """χ₂ = k − χ₁."""
        return self.k - self.chi1


@dataclass(frozen=True)
class SkippedFeature:
    """A synthetic feature left out of Ψ(r) on the training sample."""

    operator_label: str
    k: int
    value: int
    """The value of the first training object (the one value of a constant feature)."""
    omega: float
    """Its informativeness ω — often 1 at large k, although it carries no information."""
    reason: str = "constant"
    """``constant`` (``skip_constant``, ADR-030) or ``zero`` — a training object takes the value 0
    (f_k(μ) = 0.5) and ``zero_value`` is on (ADR-064)."""
    zeros: int = 0
    """Training objects whose class is indistinguishable at their gradation (f_k(μ) = 0.5)."""


@dataclass(frozen=True, eq=False)
class ClassSequenceMasks:
    """Task 2 for one operator: the class-sequence bit masks of section 1.2, one per k ≤ 62."""

    operator: int
    """Index of the base operator."""
    operator_label: str
    ks: tuple[int, ...]
    """The admissible k for which masks are formed."""
    masks: IntArray
    """(m × len(ks)) mask of every object for every k (bit j = 1: the j-th neighbour is in K1)."""
    tables: tuple[MembershipTable, ...]
    """Formula (1) over the masks of every k."""
    stabilities: FloatArray
    """(len(ks),) g by formula (2), β = 2ᵏ − 1."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)


@dataclass(frozen=True, eq=False)
class MajorityChains:
    """Task 2 for one operator: bits by the majority rule and their membership and stability."""

    operator: int
    """Index of the base operator."""
    operator_label: str
    ks: tuple[int, ...]
    """The admissible k, first = most significant bit."""
    bits: BoolArray
    """(m × len(ks)) bit i = [μ(kᵢ) > kᵢ/2]."""
    masks: IntArray
    """(m,) mask values."""
    membership: MembershipTable
    """Formula (1) over the masks 0 … 2^len(ks) − 1."""
    stability: float
    """g by formula (2)."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)


@dataclass(frozen=True, eq=False)
class PsiTrace:
    """Steps 1–8 on a training sample: from the data to Ψ(r), contributions, informativeness."""

    object_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    classes: tuple[int | str, ...]
    """Class labels: K1, K2."""
    class_index: IntArray
    """(m,) 0-based class index of every training object."""
    class_sizes: tuple[int, int]
    """|K1|, |K2|."""
    scaling: Scaling
    """Step 1: the fitted normalization."""
    normalized: FloatArray
    """(m × n) unified training values (sheet *Normalized Dataset*)."""
    operators: tuple[LocalContext, ...]
    """Steps 2–3 per base operator."""
    skipped_operators: tuple[SkippedOperator, ...]
    """Operators left out (empty feature subset, ADR-007)."""
    admissible_k: AdmissibleK
    """The k of this fit."""
    features: tuple[SyntheticFeature, ...]
    """Ψ(r): operators in configuration order, k increasing within an operator."""
    majority_chains: tuple[MajorityChains, ...]
    """Task 2 per operator; empty when there are more than 62 admissible k."""
    skipped_features: tuple[SkippedFeature, ...] = ()
    """Constant synthetic features left out (``synthetic.skip_constant``, ADR-030)."""
    class_sequences: tuple[ClassSequenceMasks, ...] = ()
    """Task 2: the class-sequence bit masks of every operator (section 1.2)."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)

    @property
    def m(self) -> int:
        """Number of training objects."""
        return len(self.object_ids)

    @property
    def r(self) -> int:
        """Number of synthetic features |Ψ(r)|."""
        return len(self.features)

    @property
    def values(self) -> IntArray:
        """(m × r) Ψ(r): aᵤ(S_t) ∈ {1, 2} (sheet *Ψ(r) Binary Features*)."""
        return np.column_stack([f.values for f in self.features])

    @property
    def contributions(self) -> FloatArray:
        """(m × r) Ψ(r) as contribution values ηᵤ(a_tu) — the agglomerative algorithm input."""
        return np.column_stack([f.contributions for f in self.features])

    @property
    def informativeness(self) -> FloatArray:
        """(r,) the informativeness of every synthetic feature (ω by formula (4) by default)."""
        return np.array([f.informativeness for f in self.features])

    @property
    def omegas(self) -> FloatArray:
        """(r,) informativeness ω of every synthetic feature."""
        return np.array([f.omega for f in self.features])

    @property
    def metaobject(self) -> FloatArray:
        """(r,) the metaobject: stability g of every synthetic feature."""
        return np.array([f.stability for f in self.features])


@dataclass(frozen=True, eq=False)
class PsiRepresentation:
    """Objects described relative to the fixed training sample E, without their class (Theorem).

    Built by
    :meth:`~context_synthetic_recognition.core.synthetic_features.psi.PsiOperator.represent`.
    """

    normalized: FloatArray
    """(q × n) unified values (training constants; not clipped)."""
    distances: tuple[FloatArray, ...]
    """Per operator: (q × m) distances to the training objects."""
    orders: tuple[IntArray, ...]
    """Per operator: training objects by (distance, index)."""
    chi1: IntArray
    """(q × r) number of K1 objects among the k nearest, per synthetic feature."""
    values: IntArray
    """(q × r) synthetic features aᵤ ∈ {1, 2} by formula (5)."""
    contributions: FloatArray
    """(q × r) contribution values ηᵤ(aᵤ)."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)
        for array in (*self.distances, *self.orders):
            array.setflags(write=False)
