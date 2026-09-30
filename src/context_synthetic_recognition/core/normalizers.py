"""Scale unification (Step 1): normalizers fitted on the training objects only.

A normalizer maps every quantitative feature (set I) by x′ = (x − offset) / scale with constants
estimated on the training sample; nominal codes (set J) are never changed. A new object is mapped
with the training constants, so its values may fall outside the training range — they are not
clipped (ADR-009).

Registry :data:`NORMALIZERS`: ``minmax`` (default, the fractional-linear transform of the
Zhuravlyov metric definition) and ``none``; more arrive with milestone M7.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from context_synthetic_recognition.core.arrays import (
    BoolArray,
    FloatArray,
    as_float_matrix,
    freeze_fields,
)
from context_synthetic_recognition.core.registry import Registry


@dataclass(frozen=True, eq=False)
class Scaling:
    """A fitted scale unification: x′ⱼ = (xⱼ − offsetⱼ) / scaleⱼ for j ∈ I.

    Quantitative features without spread on the training data (``constant``) are mapped to 0, as
    the workbook does (``IF(MAX = MIN, 0, …)``).
    """

    normalizer: str
    """Registry name of the normalizer that produced it."""
    quantitative: BoolArray
    """Mask of set I."""
    offset: FloatArray
    """Subtracted constant per feature (0 for nominal features)."""
    scale: FloatArray
    """Divisor per feature (1 for nominal and constant features)."""
    constant: BoolArray
    """Quantitative features with no spread on the training data (mapped to 0)."""
    statistics: Mapping[str, FloatArray] = field(default_factory=dict)
    """Training statistics for display, e.g. ``min`` and ``max`` (NaN for nominal features)."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        object.__setattr__(self, "statistics", MappingProxyType(dict(self.statistics)))
        freeze_fields(self)
        for values in self.statistics.values():
            values.setflags(write=False)

    def transform(self, X: FloatArray) -> FloatArray:
        """Map objects (rows of ``X``, original values) to the unified scale."""
        Z = as_float_matrix(X)
        if Z.shape[1] != self.quantitative.shape[0]:
            raise ValueError(
                f"X has {Z.shape[1]} features, the scaling was fitted on {self.quantitative.size}"
            )
        scaled = self.quantitative & ~self.constant
        Z[:, scaled] = (Z[:, scaled] - self.offset[scaled]) / self.scale[scaled]
        Z[:, self.quantitative & self.constant] = 0.0
        return Z


class Normalizer(Protocol):
    """Fits a :class:`Scaling` on the training objects."""

    def __call__(self, X: FloatArray, quantitative: BoolArray, params: Any, /) -> Scaling:
        """Estimate the scaling constants from the rows of ``X`` (training objects)."""
        ...


NORMALIZERS: Registry[Normalizer] = Registry("normalizers")
"""Scale-unification methods by name."""


def _identity(quantitative: BoolArray, name: str) -> Scaling:
    n = quantitative.size
    return Scaling(
        normalizer=name,
        quantitative=quantitative.copy(),
        offset=np.zeros(n),
        scale=np.ones(n),
        constant=np.zeros(n, dtype=bool),
    )


@NORMALIZERS.register(
    "minmax",
    aliases=("min-max", "fractional-linear"),
    summary="x′ = (x − min) / (max − min) on I, training min and max (default)",
)
def minmax(X: FloatArray, quantitative: BoolArray, params: Any = None, /) -> Scaling:
    """Fractional-linear transform to [0, 1] — Step 1, the Zhuravlyov metric definition.

    x′ⱼ = (xⱼ − minⱼ) / (maxⱼ − minⱼ) for j ∈ I with min and max over the training objects;
    0 if maxⱼ = minⱼ. Sheet *Normalized Dataset*.
    """
    del params
    Z = as_float_matrix(X)
    mask = np.asarray(quantitative, dtype=bool)
    lo = np.full(mask.size, np.nan)
    hi = np.full(mask.size, np.nan)
    lo[mask] = Z[:, mask].min(axis=0)
    hi[mask] = Z[:, mask].max(axis=0)
    scaling = _identity(mask, "minmax")
    offset = np.where(mask, lo, 0.0)
    spread = np.where(mask, hi - lo, 1.0)
    constant = mask & (spread == 0.0)
    return Scaling(
        normalizer="minmax",
        quantitative=scaling.quantitative,
        offset=offset,
        scale=np.where(constant, 1.0, spread),
        constant=constant,
        statistics={"min": lo, "max": hi},
    )


@NORMALIZERS.register("none", aliases=("identity",), summary="values unchanged")
def no_scaling(X: FloatArray, quantitative: BoolArray, params: Any = None, /) -> Scaling:
    """No scale unification: every value is used as it is."""
    del X, params
    return _identity(np.asarray(quantitative, dtype=bool), "none")
