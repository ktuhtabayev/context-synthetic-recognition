"""Scale unification (Step 1): normalizers fitted on the training objects only.

A normalizer maps the quantitative features (set I) with constants estimated on the training
sample; nominal codes (set J) are never changed. A new object is mapped with the training
constants, so its values may fall outside the training range — the per-feature affine normalizers
do not clip them (ADR-009).

Registry :data:`NORMALIZERS`:

``minmax`` (default)
    the fractional-linear transform of the Zhuravlyov metric definition;
``z-score``, ``robust``, ``max-abs``, ``decimal-scaling``
    other per-feature maps x′ = (x − offset) / scale;
``rank``
    the mid-rank among the training values, scaled to [0, 1];
``unit-length``
    every object's quantitative part divided by its Euclidean norm;
``none``
    values unchanged.

A quantitative feature without spread on the training sample carries no information and is
mapped to 0 by every fitted normalizer, as the workbook does for min–max (ADR-052).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass as params_dataclass

from context_synthetic_recognition.core.arrays import (
    BoolArray,
    FloatArray,
    as_float_matrix,
    freeze_fields,
)
from context_synthetic_recognition.core.registry import PARAMS_CONFIG, Registry


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
        return self._unify(Z)

    def _unify(self, Z: FloatArray) -> FloatArray:
        """Map the rows of ``Z`` (a copy of the original values) in place and return it."""
        scaled = self.quantitative & ~self.constant
        Z[:, scaled] = (Z[:, scaled] - self.offset[scaled]) / self.scale[scaled]
        Z[:, self.quantitative & self.constant] = 0.0
        return Z


@dataclass(frozen=True, eq=False, kw_only=True)
class RankScaling(Scaling):
    """Rank scale unification: a value becomes its mid-rank among the training values.

    A training value with ``less`` smaller and ``equal`` equal training values (itself included)
    is mapped to (less + (equal − 1)/2) / (m − 1): 0, 1/(m − 1), …, 1 when all values differ. A
    value between two training values is interpolated linearly between their levels; a value
    outside the training range gets the level of the nearest training value.
    """

    knots: tuple[FloatArray, ...]
    """Per feature, the distinct training values in increasing order (empty for set J)."""
    levels: tuple[FloatArray, ...]
    """Per feature, the unified value of each knot."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        super().__post_init__()
        for values in (*self.knots, *self.levels):
            values.setflags(write=False)

    def _unify(self, Z: FloatArray) -> FloatArray:
        for j in np.flatnonzero(self.quantitative & ~self.constant):
            knots, levels = self.knots[j], self.levels[j]
            # the training values on either side of each value: the interpolation is kept between
            # their levels, which its rounding could otherwise overstep by one unit in the last
            # place — and so reverse the order of a new value and a training value
            below = np.clip(np.searchsorted(knots, Z[:, j], side="right") - 1, 0, knots.size - 1)
            above = np.minimum(below + 1, knots.size - 1)
            Z[:, j] = np.clip(np.interp(Z[:, j], knots, levels), levels[below], levels[above])
        Z[:, self.quantitative & self.constant] = 0.0
        return Z


@dataclass(frozen=True, eq=False)
class UnitLengthScaling(Scaling):
    """Unit-length scale unification: x′ = x_I / ‖x_I‖₂ for every object (0 stays 0)."""

    def _unify(self, Z: FloatArray) -> FloatArray:
        part = Z[:, self.quantitative]
        length = np.sqrt(np.sum(np.square(part), axis=1))
        Z[:, self.quantitative] = part / np.where(length == 0.0, 1.0, length)[:, None]
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


# ---------------------------------------------------------------- further per-feature maps


def _training(X: FloatArray, quantitative: BoolArray) -> tuple[FloatArray, BoolArray, FloatArray]:
    """The training values, the mask of set I and the values of set I (objects × |I|)."""
    Z = as_float_matrix(X)
    mask = np.asarray(quantitative, dtype=bool)
    return Z, mask, Z[:, mask]


def _spread_out(mask: BoolArray, values: FloatArray) -> FloatArray:
    """A statistic of set I as a vector over all features (NaN for nominal features)."""
    out = np.full(mask.size, np.nan)
    out[mask] = values
    return out


def _affine(
    name: str,
    mask: BoolArray,
    part: FloatArray,
    offset: FloatArray,
    scale: FloatArray,
    statistics: Mapping[str, FloatArray],
) -> Scaling:
    """x′ = (x − offset) / scale on set I; ``offset``, ``scale`` and the statistics are over I."""
    lo = _spread_out(mask, part.min(axis=0))
    hi = _spread_out(mask, part.max(axis=0))
    constant = mask & (hi == lo)
    scaled = mask & ~constant
    return Scaling(
        normalizer=name,
        quantitative=mask.copy(),
        offset=np.where(scaled, _spread_out(mask, offset), 0.0),
        scale=np.where(scaled, _spread_out(mask, scale), 1.0),
        constant=constant,
        statistics={
            "min": lo,
            "max": hi,
            **{key: _spread_out(mask, values) for key, values in statistics.items()},
        },
    )


@params_dataclass(frozen=True, config=PARAMS_CONFIG)
class ZScoreParams:
    """Parameters of the ``z-score`` normalizer."""

    ddof: int = Field(default=1, ge=0, le=1)
    """1: sample standard deviation (divisor m − 1); 0: population (divisor m)."""


@NORMALIZERS.register(
    "z-score",
    aliases=("zscore", "standard"),
    summary="x′ = (x − mean) / std on I, training mean and standard deviation",
    params_type=ZScoreParams,
)
def zscore(
    X: FloatArray, quantitative: BoolArray, params: ZScoreParams | None = None, /
) -> Scaling:
    """Standardization — Step 1 with x′ⱼ = (xⱼ − meanⱼ) / stdⱼ for j ∈ I.

    Mean and standard deviation are taken over the training objects (sample standard deviation
    by default); 0 if the feature has no spread.
    """
    ddof = (params or ZScoreParams()).ddof
    _, mask, part = _training(X, quantitative)
    mean = part.mean(axis=0)
    std = part.std(axis=0, ddof=ddof) if part.shape[0] > ddof else np.zeros(part.shape[1])
    return _affine("z-score", mask, part, mean, std, {"mean": mean, "std": std})


@NORMALIZERS.register(
    "robust",
    aliases=("median-iqr",),
    summary="x′ = (x − median) / (Q3 − Q1) on I, training quartiles",
)
def robust(X: FloatArray, quantitative: BoolArray, params: Any = None, /) -> Scaling:
    """Robust scaling — Step 1 with x′ⱼ = (xⱼ − medianⱼ) / (Q3ⱼ − Q1ⱼ) for j ∈ I.

    The quartiles of the training objects are interpolated linearly (Excel's QUARTILE.INC). A
    feature whose interquartile range is 0 although it has spread is divided by its range
    max − min instead; 0 if the feature has no spread.
    """
    del params
    _, mask, part = _training(X, quantitative)
    q1, median, q3 = np.quantile(part, (0.25, 0.5, 0.75), axis=0)
    iqr = q3 - q1
    scale = np.where(iqr > 0.0, iqr, part.max(axis=0) - part.min(axis=0))
    return _affine("robust", mask, part, median, scale, {"median": median, "Q1": q1, "Q3": q3})


@NORMALIZERS.register(
    "max-abs",
    aliases=("maxabs",),
    summary="x′ = x / max|x| on I, training maximum of the absolute values",
)
def max_abs(X: FloatArray, quantitative: BoolArray, params: Any = None, /) -> Scaling:
    """Max-abs scaling — Step 1 with x′ⱼ = xⱼ / max|xⱼ| for j ∈ I (training maximum).

    Training values fall into [−1, 1] and zero stays zero; 0 if the feature has no spread.
    """
    del params
    _, mask, part = _training(X, quantitative)
    peak = np.abs(part).max(axis=0)
    return _affine("max-abs", mask, part, np.zeros(part.shape[1]), peak, {"max |x|": peak})


def decimal_power(peak: float) -> int:
    """The smallest integer j with ``peak`` / 10ʲ < 1 (0 for ``peak`` = 0)."""
    if peak <= 0.0:
        return 0
    j = int(np.floor(np.log10(peak))) + 1
    while peak / 10.0**j >= 1.0:
        j += 1
    while peak / 10.0 ** (j - 1) < 1.0:
        j -= 1
    return j


@NORMALIZERS.register(
    "decimal-scaling",
    aliases=("decimal",),
    summary="x′ = x / 10ʲ on I, j the smallest integer with max|x′| < 1 on the training data",
)
def decimal_scaling(X: FloatArray, quantitative: BoolArray, params: Any = None, /) -> Scaling:
    """Decimal scaling — Step 1 with x′ⱼ = xⱼ / 10^jⱼ for j ∈ I.

    jⱼ is the smallest integer for which every training value satisfies |x′ⱼ| < 1 (the decimal
    point is moved); 0 if the feature has no spread.
    """
    del params
    _, mask, part = _training(X, quantitative)
    power = np.array([decimal_power(float(peak)) for peak in np.abs(part).max(axis=0)], dtype=float)
    return _affine(
        "decimal-scaling", mask, part, np.zeros(part.shape[1]), 10.0**power, {"power j": power}
    )


# ---------------------------------------------------------------- rank and unit length


@NORMALIZERS.register(
    "rank",
    aliases=("quantile",),
    summary="x′ = (mid-rank − 1) / (m − 1) among the training values on I",
)
def rank(X: FloatArray, quantitative: BoolArray, params: Any = None, /) -> RankScaling:
    """Rank transform — Step 1 with x′ⱼ = (rⱼ − 1) / (m − 1) for j ∈ I.

    rⱼ is the rank of the value among the m training values of the feature (equal values share
    their mean rank). New values are interpolated between the training values and take the level
    of the nearest training value outside their range (:class:`RankScaling`); 0 if the feature
    has no spread.
    """
    del params
    Z, mask, part = _training(X, quantitative)
    m = Z.shape[0]
    lo = _spread_out(mask, part.min(axis=0))
    hi = _spread_out(mask, part.max(axis=0))
    knots: list[FloatArray] = []
    levels: list[FloatArray] = []
    for j in range(mask.size):
        if not mask[j]:
            knots.append(np.empty(0))
            levels.append(np.empty(0))
            continue
        values, counts = np.unique(Z[:, j], return_counts=True)
        less = np.cumsum(counts) - counts
        knots.append(values)
        levels.append((less + (counts - 1) / 2.0) / max(m - 1, 1))
    return RankScaling(
        normalizer="rank",
        quantitative=mask.copy(),
        offset=np.zeros(mask.size),
        scale=np.ones(mask.size),
        constant=mask & (hi == lo),
        statistics={"min": lo, "max": hi},
        knots=tuple(knots),
        levels=tuple(levels),
    )


@NORMALIZERS.register(
    "unit-length",
    aliases=("unit-norm",),
    summary="x′ = x_I / ‖x_I‖₂ for every object (its quantitative part gets length 1)",
)
def unit_length(X: FloatArray, quantitative: BoolArray, params: Any = None, /) -> UnitLengthScaling:
    """Scaling to unit length — Step 1 with x′ = x_I / ‖x_I‖₂ for every object.

    Each object is scaled by its own Euclidean norm over the quantitative features, so nothing is
    estimated on the training sample; an object whose quantitative values are all 0 stays 0.
    """
    del X, params
    mask = np.asarray(quantitative, dtype=bool)
    return UnitLengthScaling(
        normalizer="unit-length",
        quantitative=mask.copy(),
        offset=np.zeros(mask.size),
        scale=np.ones(mask.size),
        constant=np.zeros(mask.size, dtype=bool),
    )
