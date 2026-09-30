"""Membership (1), stability (2), boundary (3), informativeness (4) and bit masks (Steps 6–7).

The gradations are the same-class counts μ = 0 … k of a synthetic feature (β = k), or the bit
masks of an operator (β = 2^(number of permitted k) − 1). Sheets *Membership & Stability* and
*Informativeness ω*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from context_synthetic_recognition.core.arrays import BoolArray, FloatArray, IntArray, freeze_fields

DENSE_GRADATIONS = 4096
"""Tables list every gradation 0 … β up to this many; beyond it only the gradations that occur."""
MAX_MASK_BITS = 62
"""Bit masks are formed for at most this many permitted k (they must fit an int64)."""


@dataclass(frozen=True, eq=False)
class MembershipTable:
    """Formula (1) over the gradations of one synthetic feature (or bit mask).

    Row μ: d₁ₖ(μ), d₂ₖ(μ) = number of K1 and K2 objects with gradation μ, n(μ) = d₁ + d₂,
    f_k(μ) = (d₁/|K1|) / (d₁/|K1| + d₂/|K2|) — NaN (the workbook's "—") where n(μ) = 0.
    """

    gradations: IntArray
    """Gradation values μ of the rows."""
    d1: IntArray
    """d₁ₖ(μ)."""
    d2: IntArray
    """d₂ₖ(μ)."""
    share1: FloatArray
    """d₁ₖ(μ)/|K1|."""
    share2: FloatArray
    """d₂ₖ(μ)/|K2|."""
    f: FloatArray
    """f_k(μ); NaN where no object has the gradation."""
    weighted: FloatArray
    """n(μ)·max(f, 1 − f), the terms of formula (2); 0 where f is undefined."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)

    @property
    def n(self) -> IntArray:
        """Objects per gradation, d₁ₖ + d₂ₖ (the column n of the workbook)."""
        return self.d1 + self.d2

    @property
    def defined(self) -> BoolArray:
        """Rows where f_k(μ) is defined (n(μ) > 0)."""
        return self.n > 0

    def f_of(self, values: IntArray) -> FloatArray:
        """f_k(μ_S) for the gradations μ_S of objects: g(S, k) of the sheet *Informativeness ω*."""
        rows = np.searchsorted(self.gradations, values)
        if np.any(rows >= self.gradations.size) or np.any(self.gradations[rows] != values):
            raise ValueError("a gradation is not a row of the table")
        return self.f[rows]


def membership_table(
    gradations: IntArray,
    class_index: IntArray,
    class_sizes: tuple[int, int],
    beta: int,
) -> MembershipTable:
    """Formula (1): membership f_k(μ) = (d₁ₖ(μ)/|K1|) / (d₁ₖ(μ)/|K1| + d₂ₖ(μ)/|K2|).

    Args:
        gradations: Gradation of every training object (μ or bit mask).
        class_index: Class index of every training object (0 = K1, 1 = K2).
        class_sizes: |K1|, |K2|.
        beta: Largest possible gradation β. Rows 0 … β are listed when β < DENSE_GRADATIONS,
            otherwise only the gradations that occur.
    """
    values = np.asarray(gradations, dtype=np.int64)
    rows = np.arange(beta + 1, dtype=np.int64) if beta < DENSE_GRADATIONS else np.unique(values)
    position = np.searchsorted(rows, values)
    d1 = np.bincount(position[class_index == 0], minlength=rows.size).astype(np.int64)
    d2 = np.bincount(position[class_index == 1], minlength=rows.size).astype(np.int64)
    share1 = d1 / class_sizes[0]
    share2 = d2 / class_sizes[1]
    n = d1 + d2
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(n > 0, share1 / (share1 + share2), np.nan)
    weighted = np.where(n > 0, n * np.maximum(f, 1.0 - f), 0.0)
    return MembershipTable(rows, d1, d2, share1, share2, f, weighted)


def stability(table: MembershipTable, m: int) -> float:
    """Formula (2): g_k = (1/m) Σ_μ n(μ)·f_k(μ) if f_k(μ) ≥ 0.5, else n(μ)·(1 − f_k(μ)).

    0.5 ≤ g ≤ 1: 0.5 — the gradations do not separate the classes, 1 — they separate them fully.
    The terms are added in gradation order, like the workbook's SUM (numpy's pairwise ``sum``
    would differ in the last bits).
    """
    total = np.cumsum(table.weighted)[-1] if table.weighted.size else 0.0
    return float(total) / m


@dataclass(frozen=True)
class Boundary:
    """Formula (3): the class boundary G_k of a synthetic feature."""

    q1: float | None
    """q₁ = min{f_k(μ) | f_k(μ) > 0.5}; ``None`` if no f exceeds 0.5."""
    q2: float | None
    """q₂ = max{f_k(μ) | f_k(μ) < 0.5}; ``None`` if no f is below 0.5."""
    G: float
    """G_k = (q₁ + q₂)/2; the neutral 0.5 when one side is empty."""


def boundary(table: MembershipTable) -> Boundary:
    """Formula (3): G_k = (q₁ + q₂)/2, q₂ = max{f < 0.5}, q₁ = min{f > 0.5}.

    f = 0.5 belongs to neither side; if one side is empty, G_k = 0.5 (ADR-008).
    """
    f = table.f[table.defined]
    above = f[f > 0.5]
    below = f[f < 0.5]
    q1 = float(above.min()) if above.size else None
    q2 = float(below.max()) if below.size else None
    G = (q1 + q2) / 2 if q1 is not None and q2 is not None else 0.5
    return Boundary(q1, q2, G)


def correct_side(membership: FloatArray, class_index: IntArray, G: float) -> BoolArray:
    """Whether each object lies on its class's side of G: f(μ_S) > G for K1, < G for K2."""
    return np.where(class_index == 0, membership > G, membership < G)


def informativeness(correct: BoolArray) -> float:
    """Formula (4): ω = (|{S ∈ K1 | f(μ_S) > G}| + |{S ∈ K2 | f(μ_S) < G}|) / m."""
    return int(np.count_nonzero(correct)) / correct.size


def bit_masks(mu: IntArray, ks: tuple[int, ...]) -> tuple[BoolArray, IntArray]:
    """Bit representation of an operator's neighbourhoods by the majority rule (Task 2).

    Bit i of an object is 1 if its same-class count for the i-th permitted k is a majority
    (μ > k/2); the first permitted k is the most significant bit. Section 1.4, sheet
    *Membership & Stability*.

    Args:
        mu: (objects × len(ks)) same-class counts of one operator.
        ks: The permitted k.

    Returns:
        The bits (objects × len(ks)) and the mask value of every object.

    Raises:
        ValueError: More than :data:`MAX_MASK_BITS` permitted k.
    """
    if len(ks) > MAX_MASK_BITS:
        raise ValueError(f"bit masks need at most {MAX_MASK_BITS} permitted k, got {len(ks)}")
    bits = 2 * np.asarray(mu, dtype=np.int64) > np.asarray(ks, dtype=np.int64)
    weights = np.left_shift(1, np.arange(len(ks) - 1, -1, -1, dtype=np.int64))
    return bits, (bits.astype(np.int64) @ weights).astype(np.int64)
