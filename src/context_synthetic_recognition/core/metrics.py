"""Metrics of the base operators (Step 2).

A metric computes the distances between the rows of two matrices of unified values restricted to
one feature subset. It rounds its result to ``decimals`` so that mathematically equal distances
tie exactly (ADR-008).

Registry :data:`METRICS`: ``zhuravlyov`` (default); HEOM, HVDM and Gower follow in M7 (ADR-011).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from context_synthetic_recognition.core.arrays import BoolArray, FloatArray
from context_synthetic_recognition.core.registry import Registry


class Metric(Protocol):
    """Distances between the rows of ``A`` and the rows of ``B``."""

    def __call__(
        self, A: FloatArray, B: FloatArray, quantitative: BoolArray, decimals: int, params: Any, /
    ) -> FloatArray:
        """Return the (len(A), len(B)) matrix of distances, rounded to ``decimals``.

        Args:
            A: Unified values of the target objects (columns = the operator's feature subset).
            B: Unified values of the reference objects (same columns).
            quantitative: Which of the columns belong to set I.
            decimals: Rounding of the distances.
            params: The metric's validated parameters (``None`` if it has none).
        """
        ...


METRICS: Registry[Metric] = Registry("metrics")
"""Distance functions by name."""


def round_distances(values: FloatArray, decimals: int) -> FloatArray:
    """Round distances to ``decimals`` (numpy: round half to even of x·10^decimals, ADR-020)."""
    return np.round(values, decimals)


@METRICS.register(
    "zhuravlyov",
    aliases=("zhuravlev", "juravlev"),  # earlier spellings (ADR-023)
    summary="ρ(x, y) = Σ_{j∈I} |x′ⱼ − y′ⱼ| + Σ_{j∈J} [xⱼ ≠ yⱼ] (default)",
)
def zhuravlyov(
    A: FloatArray, B: FloatArray, quantitative: BoolArray, decimals: int, params: Any = None, /
) -> FloatArray:
    """Zhuravlyov metric on heterogeneous features — base operators ρ, ρ_I, ρ_J.

    ρ(x, y) = Σ_{j∈I} |x′ⱼ − y′ⱼ| + Σ_{j∈J} [xⱼ ≠ yⱼ], x′ the unified quantitative values.
    As on the sheet *Zhuravlev Distances*, the quantitative part is rounded first and then the
    sum. The terms are added feature by feature in feature order (the order of the workbook's
    SUMPRODUCT and of the reference engine), so equal inputs give bit-identical distances.
    """
    del params
    quantitative_part = np.zeros((A.shape[0], B.shape[0]))
    nominal_part = np.zeros((A.shape[0], B.shape[0]))
    for j in range(A.shape[1]):
        if quantitative[j]:
            quantitative_part += np.abs(A[:, j, None] - B[None, :, j])
        else:
            nominal_part += A[:, j, None] != B[None, :, j]
    return round_distances(round_distances(quantitative_part, decimals) + nominal_part, decimals)
