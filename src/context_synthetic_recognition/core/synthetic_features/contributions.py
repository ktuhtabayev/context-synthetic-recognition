"""Informativeness of the synthetic features and contributions of their gradations — formula (6).

η(j) = ω·(α¹ⱼ/|K1| − α²ⱼ/|K2|): α¹ⱼ and α²ⱼ are the numbers of K1 and K2 objects with the value j
— j ∈ {1, 2} for a synthetic feature by formula (5), the codes of an original nominal feature
combined with them (ADR-065). Positive values point to K1, negative ones to K2. Ψ(r) expressed as
contribution values ηᵤ(a_tu), together with the informativeness, is the input of the
hierarchical agglomerative algorithm (sheet *Ψ(r) Contribution & Weight*).

Registry :data:`INFORMATIVENESS`: ``formula-4`` (default) — the informativeness ω of formula (4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from context_synthetic_recognition.core.arrays import FloatArray, IntArray
from context_synthetic_recognition.core.registry import Registry

GRADATIONS = (1, 2)
"""Values of a synthetic feature: the class numbers of K1 and K2 (formula (5))."""


def gradation_counts(
    values: IntArray, class_index: IntArray, gradations: Sequence[int] = GRADATIONS
) -> IntArray:
    """α — counts of the values by class.

    ``alpha[i, c]`` = objects of class c with value ``gradations[i]``.

    For a synthetic feature ``alpha[0, 0]`` = α¹₁, ``alpha[0, 1]`` = α²₁, ``alpha[1, 0]`` = α¹₂,
    ``alpha[1, 1]`` = α²₂.
    """
    alpha = np.zeros((len(gradations), 2), dtype=np.int64)
    for i, value in enumerate(gradations):
        for c in range(2):
            alpha[i, c] = np.count_nonzero((values == value) & (class_index == c))
    return alpha


def contributions(alpha: IntArray, class_sizes: tuple[int, int], weight: float) -> FloatArray:
    """Formula (6): η(j) = ω·(α¹ⱼ/|K1| − α²ⱼ/|K2|) for every value j (the rows of ``alpha``)."""
    return np.array(
        [
            weight * (alpha[i, 0] / class_sizes[0] - alpha[i, 1] / class_sizes[1])
            for i in range(alpha.shape[0])
        ]
    )


def contribution_values(
    values: IntArray, eta: FloatArray, gradations: Sequence[int] = GRADATIONS
) -> FloatArray:
    """ηᵤ(a_tu): the contribution of every object's value; 0 for a value unseen in training."""
    known = np.asarray(gradations, dtype=np.int64)
    v = np.asarray(values, dtype=np.int64)
    rows = np.searchsorted(known, v)
    found = (rows < known.size) & (known[np.minimum(rows, known.size - 1)] == v)
    return np.where(found, np.asarray(eta)[np.minimum(rows, known.size - 1)], 0.0)


def informativeness_ranks(weights: FloatArray) -> IntArray:
    """Rank of every weight: 1 = highest; equal weights → the earlier feature ranks higher.

    The workbook's ``COUNTIF(>w) + COUNTIF(prefix, =w)``.
    """
    order = np.argsort(-np.asarray(weights), kind="stable")
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(1, order.size + 1)
    return ranks


@dataclass(frozen=True)
class InformativenessInputs:
    """What a weight plug-in may use: the statistics of one synthetic feature."""

    omega: float
    """Informativeness ω (formula (4))."""
    stability: float
    """Stability g (formula (2))."""
    boundary: float
    """Class boundary G (formula (3))."""


class InformativenessFunction(Protocol):
    """Weight of a synthetic feature (used in formula (6) and in agglomerative Step 2)."""

    def __call__(self, inputs: InformativenessInputs, params: Any, /) -> float:
        """Return the weight."""
        ...


INFORMATIVENESS: Registry[InformativenessFunction] = Registry("informativeness")
"""Informativeness functions of the synthetic features by name."""


@INFORMATIVENESS.register("formula-4", aliases=("omega",), summary="ω by formula (4) (default)")
def formula_4(inputs: InformativenessInputs, params: Any = None, /) -> float:
    """The weight is the informativeness ω of formula (4)."""
    del params
    return inputs.omega
