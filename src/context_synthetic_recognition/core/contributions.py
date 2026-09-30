"""Weights of the synthetic features and contributions of their gradations — formula (6), Step 8.

η(j) = ω·(α¹ⱼ/|K1| − α²ⱼ/|K2|), j ∈ {1, 2}: α¹ⱼ and α²ⱼ are the numbers of K1 and K2 objects
with aᵤ = j. Positive values point to K1, negative ones to K2. Ψ(r) expressed as contribution
values ηᵤ(a_tu) together with the weights is the input of the HAG (the template sheet
"Dataset (Contribution & Weight)"; here *Ψ(r) Contribution & Weight*).

Registry :data:`WEIGHTS`: ``omega`` (default) — the weight is the informativeness ω of
formula (4). The template's Criterion-1 and λ·β weights can be added as further plug-ins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from context_synthetic_recognition.core.arrays import FloatArray, IntArray
from context_synthetic_recognition.core.registry import Registry

GRADATIONS = (1, 2)
"""Values of a synthetic feature: the class numbers of K1 and K2 (formula (5))."""


def gradation_counts(values: IntArray, class_index: IntArray) -> IntArray:
    """α — counts of the gradations by class: ``alpha[j − 1, c]`` = objects of class c with aᵤ = j.

    So ``alpha[0, 0]`` = α¹₁, ``alpha[0, 1]`` = α²₁, ``alpha[1, 0]`` = α¹₂, ``alpha[1, 1]`` = α²₂.
    """
    alpha = np.zeros((len(GRADATIONS), 2), dtype=np.int64)
    for j, value in enumerate(GRADATIONS):
        for c in range(2):
            alpha[j, c] = np.count_nonzero((values == value) & (class_index == c))
    return alpha


def contributions(alpha: IntArray, class_sizes: tuple[int, int], weight: float) -> FloatArray:
    """Formula (6): η(j) = ω·(α¹ⱼ/|K1| − α²ⱼ/|K2|) for j = 1, 2."""
    return np.array(
        [weight * (alpha[j, 0] / class_sizes[0] - alpha[j, 1] / class_sizes[1]) for j in range(2)]
    )


def contribution_values(values: IntArray, eta: FloatArray) -> FloatArray:
    """ηᵤ(a_tu): the contribution of every object's gradation (aᵤ = 1 → η(1), 2 → η(2))."""
    return np.asarray(eta)[np.asarray(values, dtype=np.int64) - 1]


def weight_ranks(weights: FloatArray) -> IntArray:
    """Rank of every weight: 1 = highest; equal weights → the earlier feature ranks higher.

    The workbook's ``COUNTIF(>w) + COUNTIF(prefix, =w)``.
    """
    order = np.argsort(-np.asarray(weights), kind="stable")
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(1, order.size + 1)
    return ranks


@dataclass(frozen=True)
class WeightInputs:
    """What a weight plug-in may use: the statistics of one synthetic feature."""

    omega: float
    """Informativeness ω (formula (4))."""
    stability: float
    """Stability g (formula (2))."""
    boundary: float
    """Class boundary G (formula (3))."""


class WeightFunction(Protocol):
    """Weight of a synthetic feature (used in formula (6) and in HAG STEP 2)."""

    def __call__(self, inputs: WeightInputs, params: Any, /) -> float:
        """Return the weight."""
        ...


WEIGHTS: Registry[WeightFunction] = Registry("weights")
"""Weights of the synthetic features by name."""


@WEIGHTS.register("omega", aliases=("informativeness",), summary="ω by formula (4) (default)")
def omega(inputs: WeightInputs, params: Any = None, /) -> float:
    """The weight is the informativeness ω of formula (4)."""
    del params
    return inputs.omega
