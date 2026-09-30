"""Meta-algorithm, Steps 1–5 of the article (Step 12 of the workbook).

Every training object Sᵢ is described by the gradations aᵢ = (aᵢ₀, …, aᵢₚ) of the TUPLAM
features and by its latent values dᵢ = (dᵢ₁, …, dᵢₚ) = (r₁(Sᵢ), …, r_p(Sᵢ)). An object to
classify brings only its gradations a = (a₀, …, a_p):

Step 1   B1(a₀) = {Sᵢ ∈ K1 | aᵢ₀ = a₀},   B2(a₀) = {Sᵢ ∈ K2 | aᵢ₀ = a₀}.
Step 2   For j = 1, …, p:
         B1(aⱼ) = {Sᵢ ∈ B1(aⱼ₋₁) | aᵢⱼ = aⱼ, dᵢⱼ > 0},
         B2(aⱼ) = {Sᵢ ∈ B2(aⱼ₋₁) | aᵢⱼ = aⱼ, dᵢⱼ < 0}.
Step 3   Repeat Step 2 while j < p.
Step 4   The decision rule (registry :data:`DECISION_RULES`; ``article-step-4``: K1 if
         |B1(a_p)|/|K1| > |B2(a_p)|/|K2|, K2 if <, 0 = refusal if equal).
Step 5   End.

Gradations are compared for equality only, so they may be any values: {1, 2} of formula (5) in
the CS-model, or the nominal codes of the template's meta-algorithm. Decisions use the article's
codes 1 = K1, 2 = K2, 0 = refusal (ADR-027).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.core.arrays import BoolArray, FloatArray, IntArray, freeze_fields
from context_synthetic_recognition.core.registry import Registry
from context_synthetic_recognition.errors import ModelUndefinedError

K1_DECISION = 1
"""Decision code of class K1."""
K2_DECISION = 2
"""Decision code of class K2."""
REFUSAL = 0
"""Decision code of a refusal (equal scores)."""

BLOCK_ELEMENTS = 1 << 24
"""Queries are filtered in blocks of at most this many (query, step, object) flags."""


@dataclass(frozen=True, eq=False)
class MetaDescription:
    """The training description of the meta-algorithm: (aᵢ, dᵢ) and the classes."""

    gradations: npt.NDArray[Any]
    """(m × (p + 1)) aᵢⱼ, the gradations of the TUPLAM features in TUPLAM order."""
    latent: FloatArray
    """(m × p) dᵢⱼ = r_j(Sᵢ)."""
    class_index: IntArray
    """(m,) 0 = K1, 1 = K2."""

    def __post_init__(self) -> None:
        """Check the shapes and freeze the arrays."""
        m = self.class_index.shape[0]
        if self.gradations.ndim != 2 or self.gradations.shape[0] != m:
            raise ValueError(f"gradations must be (m × (p + 1)) with m = {m}")
        if self.gradations.shape[1] < 1:
            raise ValueError("the description needs at least the first TUPLAM feature (p ≥ 0)")
        if self.latent.shape != (m, self.gradations.shape[1] - 1):
            raise ValueError(
                f"latent values must be (m × p) = ({m} × {self.gradations.shape[1] - 1}), "
                f"got {self.latent.shape}"
            )
        if not np.isin(self.class_index, (0, 1)).all():
            raise ValueError("class indices must be 0 (K1) or 1 (K2)")
        if min(self.class_sizes) == 0:
            raise ModelUndefinedError("the meta-algorithm needs training objects of both classes")
        freeze_fields(self)

    @classmethod
    def of(
        cls, gradations: npt.ArrayLike, latent: npt.ArrayLike, class_index: npt.ArrayLike
    ) -> MetaDescription:
        """Build a description from array-likes (``latent`` may be empty when p = 0)."""
        a = np.array(gradations)
        if a.ndim == 1:
            a = a[:, None]
        d = np.array(latent, dtype=np.float64)
        if d.size == 0:
            d = d.reshape(a.shape[0], 0)
        return cls(a, d, np.array(class_index, dtype=np.int64))

    @property
    def m(self) -> int:
        """Number of training objects."""
        return int(self.class_index.shape[0])

    @property
    def p(self) -> int:
        """Number of latent features."""
        return int(self.latent.shape[1])

    @property
    def class_sizes(self) -> tuple[int, int]:
        """|K1|, |K2|."""
        return (
            int(np.count_nonzero(self.class_index == 0)),
            int(np.count_nonzero(self.class_index == 1)),
        )

    @property
    def sign_condition(self) -> BoolArray:
        """(m × p) the latent-sign condition of Step 2: dᵢⱼ > 0 in K1, dᵢⱼ < 0 in K2."""
        k1 = (self.class_index == 0)[:, None]
        return np.where(k1, self.latent > 0.0, self.latent < 0.0)


@dataclass(frozen=True, eq=False)
class MetaSteps:
    """Steps 1–3 for one object in detail: B1(aⱼ) and B2(aⱼ) for j = 0 … p."""

    query: npt.NDArray[Any]
    """(p + 1,) the gradations a₀ … a_p of the object."""
    match: BoolArray
    """((p + 1) × m) aᵢⱼ = aⱼ — the original-feature condition."""
    sign: BoolArray
    """(p × m) the latent-sign condition (dᵢⱼ > 0 in K1, < 0 in K2)."""
    in_b1: BoolArray
    """((p + 1) × m) Sᵢ ∈ B1(aⱼ)."""
    in_b2: BoolArray
    """((p + 1) × m) Sᵢ ∈ B2(aⱼ)."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)

    @property
    def b1_sizes(self) -> IntArray:
        """|B1(aⱼ)| for j = 0 … p."""
        return np.asarray(np.count_nonzero(self.in_b1, axis=1), dtype=np.int64)

    @property
    def b2_sizes(self) -> IntArray:
        """|B2(aⱼ)| for j = 0 … p."""
        return np.asarray(np.count_nonzero(self.in_b2, axis=1), dtype=np.int64)

    def b1(self, j: int) -> IntArray:
        """0-based training indices of B1(aⱼ)."""
        return np.flatnonzero(self.in_b1[j]).astype(np.int64)

    def b2(self, j: int) -> IntArray:
        """0-based training indices of B2(aⱼ)."""
        return np.flatnonzero(self.in_b2[j]).astype(np.int64)


class DecisionRule(Protocol):
    """Step 4: the decision from the final sizes |B1(a_p)|, |B2(a_p)|."""

    def __call__(
        self, b1: IntArray, b2: IntArray, class_sizes: tuple[int, int], params: Any, /
    ) -> IntArray:
        """Return 1 (K1), 2 (K2) or 0 (refusal) for every object."""
        ...


DECISION_RULES: Registry[DecisionRule] = Registry("decision rules")
"""Decision rules of Step 4 by name."""


@DECISION_RULES.register(
    "article-step-4",
    aliases=("step-4",),
    summary="K1 if |B1|/|K1| > |B2|/|K2|, K2 if <, 0 = refusal if equal (default)",
)
def article_step_4(
    b1: IntArray, b2: IntArray, class_sizes: tuple[int, int], params: Any = None, /
) -> IntArray:
    """Step 4: K1 if |B1(a_p)|/|K1| > |B2(a_p)|/|K2|, K2 if <, 0 (refusal) if equal.

    The scores are compared exactly, as |B1|·|K2| against |B2|·|K1| (ADR-027).
    """
    del params
    left = np.asarray(b1, dtype=np.int64) * class_sizes[1]
    right = np.asarray(b2, dtype=np.int64) * class_sizes[0]
    return np.where(left > right, K1_DECISION, np.where(left < right, K2_DECISION, REFUSAL)).astype(
        np.int64
    )


@dataclass(frozen=True, eq=False)
class MetaResult:
    """The meta-algorithm applied to a set of objects."""

    description: MetaDescription
    """The training description the objects were classified against."""
    queries: npt.NDArray[Any]
    """(q × (p + 1)) the gradations of the objects."""
    b1_sizes: IntArray
    """(q × (p + 1)) |B1(aⱼ)| for j = 0 … p."""
    b2_sizes: IntArray
    """(q × (p + 1)) |B2(aⱼ)| for j = 0 … p."""
    decisions: IntArray
    """(q,) 1 = K1, 2 = K2, 0 = refusal (Step 4)."""
    rule: str
    """Registry name of the decision rule."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)

    @property
    def scores1(self) -> FloatArray:
        """(q,) score₁ = |B1(a_p)|/|K1|."""
        return self.b1_sizes[:, -1] / self.description.class_sizes[0]

    @property
    def scores2(self) -> FloatArray:
        """(q,) score₂ = |B2(a_p)|/|K2|."""
        return self.b2_sizes[:, -1] / self.description.class_sizes[1]

    @property
    def scores(self) -> FloatArray:
        """(q,) score₁ − score₂ (positive → K1); rounded only by the evaluation (ADR-008)."""
        return self.scores1 - self.scores2

    def steps(self, i: int) -> MetaSteps:
        """Steps 1–3 of the ``i``-th object in detail (B1, B2 at every step)."""
        query = self.queries[i]
        in_b1, in_b2, match = _filter(self.description, query[None, :])
        return MetaSteps(
            query=query,
            match=match[0],
            sign=self.description.sign_condition.T.copy(),
            in_b1=in_b1[0],
            in_b2=in_b2[0],
        )


def _filter(
    description: MetaDescription, queries: npt.NDArray[Any]
) -> tuple[BoolArray, BoolArray, BoolArray]:
    """Steps 1–3 for a block of queries: (q × (p + 1) × m) flags of B1, B2 and the matches."""
    match = np.transpose(description.gradations[None, :, :] == queries[:, None, :], (0, 2, 1))
    keep = match.copy()
    keep[:, 1:, :] &= description.sign_condition.T[None, :, :]
    keep = np.logical_and.accumulate(keep, axis=1)
    k1 = description.class_index == 0
    return keep & k1, keep & ~k1, match


def meta_classify(
    description: MetaDescription,
    queries: npt.ArrayLike,
    rule: str = "article-step-4",
    params: dict[str, Any] | None = None,
) -> MetaResult:
    """The meta-algorithm, Steps 1–5, for every object of ``queries``.

    Args:
        description: The training description (aᵢ, dᵢ) and classes.
        queries: (q × (p + 1)) gradations a₀ … a_p of the objects (or one object as a vector).
        rule: Registry name of the Step 4 decision rule.
        params: Its parameters.

    Raises:
        RegistryError: Unknown decision rule.
        ConfigError: Invalid decision-rule parameters.
        ValueError: The queries do not have p + 1 gradations.
    """
    decide = DECISION_RULES.get(rule)
    rule_params = DECISION_RULES.make_params(rule, params)
    a = np.array(queries)
    if a.ndim == 1:
        a = a[None, :]
    width = description.p + 1
    if a.ndim != 2 or a.shape[1] != width:
        raise ValueError(f"every object needs p + 1 = {width} gradations, got shape {a.shape}")
    b1_sizes = np.empty((a.shape[0], width), dtype=np.int64)
    b2_sizes = np.empty_like(b1_sizes)
    block = max(1, BLOCK_ELEMENTS // (width * max(1, description.m)))
    for start in range(0, a.shape[0], block):
        rows = slice(start, start + block)
        in_b1, in_b2, _ = _filter(description, a[rows])
        b1_sizes[rows] = np.count_nonzero(in_b1, axis=2)
        b2_sizes[rows] = np.count_nonzero(in_b2, axis=2)
    decisions = np.asarray(
        decide(b1_sizes[:, -1], b2_sizes[:, -1], description.class_sizes, rule_params),
        dtype=np.int64,
    )
    if not np.isin(decisions, (K1_DECISION, K2_DECISION, REFUSAL)).all():
        raise ValueError("a decision rule must return 1 (K1), 2 (K2) or 0 (refusal)")
    return MetaResult(
        description=description,
        queries=a,
        b1_sizes=b1_sizes,
        b2_sizes=b2_sizes,
        decisions=decisions,
        rule=DECISION_RULES.info(rule).name,
    )
