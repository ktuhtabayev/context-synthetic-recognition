"""Metrics of the base operators (Step 2).

A metric computes the distances between the rows of two matrices of unified values restricted to
one feature subset. It rounds its result to ``decimals`` so that mathematically equal distances
tie exactly (ADR-008).

Registry :data:`METRICS`:

``zhuravlyov`` (default), ``weighted-zhuravlyov``, ``heom``, ``gower``
    heterogeneous metrics — any feature subset;
``manhattan``, ``euclidean``, ``chebyshev``, ``minkowski``, ``canberra``, ``cosine``,
``mahalanobis``
    defined on quantitative features (set I);
``hamming``
    defined on nominal features (set J).

What the pipeline must know about a metric besides its distance function is declared with
:func:`traits` (ADR-050): the feature types it is defined on, a *fit step* that estimates its
training statistics (ranges, a covariance) from the unified training values, and a check of its
parameters against an operator's features. A fit step gets no class labels: no metric reads a
class (ADR-051), so the distances of a new object cannot depend on its class or on anyone
else's.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar

import numpy as np
from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass as params_dataclass

from context_synthetic_recognition.core.arrays import BoolArray, FloatArray, readonly
from context_synthetic_recognition.core.registry import PARAMS_CONFIG, Registry
from context_synthetic_recognition.errors import ConfigError


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
            params: The metric's validated parameters (``None`` if it has none) — or, for a
                metric with a fit step, what that step returned.
        """
        ...


class MetricFit(Protocol):
    """Fit step of a metric: its training statistics."""

    def __call__(self, Z: FloatArray, quantitative: BoolArray, params: Any, /) -> Any:
        """Return what the metric then receives as ``params``.

        Args:
            Z: Unified values of the training objects (columns = the operator's feature subset).
            quantitative: Which of the columns belong to set I.
            params: The metric's validated parameters (``None`` if it has none).
        """
        ...


class MetricCheck(Protocol):
    """Check of a metric's parameters against the features of an operator."""

    def __call__(self, quantitative: BoolArray, params: Any, /) -> None:
        """Raise ``ConfigError`` if ``params`` do not fit an operator with these features.

        Args:
            quantitative: Which of the operator's features belong to set I (one per feature).
            params: The metric's validated parameters (``None`` if it has none).
        """
        ...


Domain = Literal["any", "quantitative", "nominal"]
"""The feature types a metric is defined on."""


@dataclass(frozen=True)
class MetricTraits:
    """What the pipeline must know about a metric besides its distance function (ADR-050)."""

    domain: Domain = "any"
    """``quantitative``: set I only; ``nominal``: set J only; ``any``: heterogeneous."""
    fit: MetricFit | None = None
    """Estimates the training statistics; ``None`` if the metric has none."""
    check: MetricCheck | None = None
    """Checks the parameters against an operator's features when the operators are resolved."""


METRICS: Registry[Metric] = Registry("metrics")
"""Distance functions by name."""

_TRAITS = "_cs_metric_traits"
_M = TypeVar("_M")


def traits(
    *, domain: Domain = "any", fit: MetricFit | None = None, check: MetricCheck | None = None
) -> Callable[[_M], _M]:
    """Declare a metric's traits; put it below ``@METRICS.register`` (ADR-050)."""

    def decorate(metric: _M) -> _M:
        setattr(metric, _TRAITS, MetricTraits(domain, fit, check))
        return metric

    return decorate


def traits_of(metric: Metric) -> MetricTraits:
    """The declared traits of ``metric`` (heterogeneous, no fit step, if it declares none)."""
    found = getattr(metric, _TRAITS, None)
    return found if isinstance(found, MetricTraits) else MetricTraits()


def round_distances(values: FloatArray, decimals: int) -> FloatArray:
    """Round distances to ``decimals`` (numpy: round half to even of x·10^decimals, ADR-020)."""
    return np.round(values, decimals)


def _pairs(A: FloatArray, B: FloatArray) -> FloatArray:
    return np.zeros((A.shape[0], B.shape[0]))


# ---------------------------------------------------------------- the Zhuravlyov metric


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


@params_dataclass(frozen=True, config=PARAMS_CONFIG)
class WeightedZhuravlyovParams:
    """Parameters of the ``weighted-zhuravlyov`` metric."""

    quantitative_weight: float = Field(default=1.0, ge=0.0)
    """Weight of every quantitative feature (set I)."""
    nominal_weight: float = Field(default=1.0, ge=0.0)
    """Weight of every nominal feature (set J)."""
    feature_weights: tuple[float, ...] = ()
    """One more weight per feature of the operator, in the operator's feature order (empty: 1)."""

    @field_validator("feature_weights")
    @classmethod
    def _not_negative(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if any(value < 0.0 for value in values):
            raise ValueError("feature weights must not be negative")
        return values


def check_feature_weights(
    quantitative: BoolArray, params: WeightedZhuravlyovParams | None = None, /
) -> None:
    """``feature_weights`` must list one weight per feature of the operator (or none)."""
    given = len(params.feature_weights) if params else 0
    if given and given != quantitative.size:
        raise ConfigError(
            f"weighted-zhuravlyov: {given} feature weights for an operator of "
            f"{quantitative.size} feature(s) — give one weight per feature, in the operator's "
            "feature order"
        )


@METRICS.register(
    "weighted-zhuravlyov",
    aliases=("weighted-zhuravlev",),
    summary="Σ_{j∈I} wⱼ·|x′ⱼ − y′ⱼ| + Σ_{j∈J} wⱼ·[xⱼ ≠ yⱼ]",
    params_type=WeightedZhuravlyovParams,
)
@traits(check=check_feature_weights)
def weighted_zhuravlyov(
    A: FloatArray,
    B: FloatArray,
    quantitative: BoolArray,
    decimals: int,
    params: WeightedZhuravlyovParams | None = None,
    /,
) -> FloatArray:
    """Zhuravlyov metric with feature weights.

    ρ_w(x, y) = Σ_{j∈I} wⱼ·|x′ⱼ − y′ⱼ| + Σ_{j∈J} wⱼ·[xⱼ ≠ yⱼ] with wⱼ = (weight of the feature's
    type) · (weight of the feature). Rounded like the Zhuravlyov metric — the quantitative part
    first, then the sum — so that with every weight 1 it is that metric bit for bit.

    Raises:
        ConfigError: ``feature_weights`` does not list one weight per feature of the operator.
    """
    chosen = params or WeightedZhuravlyovParams()
    check_feature_weights(quantitative, chosen)
    n = A.shape[1]
    own = chosen.feature_weights or (1.0,) * n
    quantitative_part = _pairs(A, B)
    nominal_part = _pairs(A, B)
    for j in range(n):
        if quantitative[j]:
            weight = chosen.quantitative_weight * own[j]
            quantitative_part += weight * np.abs(A[:, j, None] - B[None, :, j])
        else:
            weight = chosen.nominal_weight * own[j]
            nominal_part += weight * (A[:, j, None] != B[None, :, j])
    return round_distances(round_distances(quantitative_part, decimals) + nominal_part, decimals)


# ---------------------------------------------------------------- HEOM and Gower


@dataclass(frozen=True, eq=False)
class Ranges:
    """Training ranges of the unified values — the fit of ``heom`` and ``gower``."""

    spread: FloatArray
    """max − min per feature of the operator over the training objects; +∞ where the feature has
    no spread, so that it contributes 0 (and for nominal features, which do not use it)."""

    def __post_init__(self) -> None:
        """Freeze the array."""
        readonly(self.spread)


def fit_ranges(Z: FloatArray, quantitative: BoolArray, params: Any = None, /) -> Ranges:
    """Training range of every quantitative feature of the operator (fit step, ADR-050)."""
    del params
    spread = np.full(Z.shape[1], np.inf)
    for j in np.flatnonzero(quantitative):
        width = float(Z[:, j].max() - Z[:, j].min())
        if width > 0.0:
            spread[j] = width
    return Ranges(spread)


@METRICS.register(
    "heom",
    summary="√(Σ_{j∈I} (|x′ⱼ − y′ⱼ| / rangeⱼ)² + Σ_{j∈J} [xⱼ ≠ yⱼ]) — Euclidean-overlap",
)
@traits(fit=fit_ranges)
def heom(
    A: FloatArray, B: FloatArray, quantitative: BoolArray, decimals: int, params: Ranges, /
) -> FloatArray:
    """Heterogeneous Euclidean-Overlap Metric (Wilson & Martinez, 1997).

    HEOM(x, y) = √(Σⱼ dⱼ²) with dⱼ = |x′ⱼ − y′ⱼ| / rangeⱼ for j ∈ I (rangeⱼ over the training
    objects) and dⱼ = [xⱼ ≠ yⱼ] for j ∈ J. A feature without spread on the training sample
    contributes 0.
    """
    total = _pairs(A, B)
    for j in range(A.shape[1]):
        if quantitative[j]:
            total += np.square(np.abs(A[:, j, None] - B[None, :, j]) / params.spread[j])
        else:
            total += A[:, j, None] != B[None, :, j]
    return round_distances(np.sqrt(total), decimals)


@METRICS.register(
    "gower",
    summary="(Σ_{j∈I} |x′ⱼ − y′ⱼ| / rangeⱼ + Σ_{j∈J} [xⱼ ≠ yⱼ]) / n — Gower distance",
)
@traits(fit=fit_ranges)
def gower(
    A: FloatArray, B: FloatArray, quantitative: BoolArray, decimals: int, params: Ranges, /
) -> FloatArray:
    """Gower distance (1 − Gower's general similarity coefficient, 1971).

    G(x, y) = (1/n)·Σⱼ dⱼ with dⱼ = |x′ⱼ − y′ⱼ| / rangeⱼ for j ∈ I (rangeⱼ over the training
    objects) and dⱼ = [xⱼ ≠ yⱼ] for j ∈ J; n is the number of features of the operator. With
    min–max unification it is the Zhuravlyov metric divided by n.
    """
    total = _pairs(A, B)
    n = A.shape[1]
    for j in range(n):
        if quantitative[j]:
            total += np.abs(A[:, j, None] - B[None, :, j]) / params.spread[j]
        else:
            total += A[:, j, None] != B[None, :, j]
    return round_distances(total / max(n, 1), decimals)


# ---------------------------------------------------------------- metrics on set I


@METRICS.register(
    "manhattan",
    aliases=("l1", "city-block"),
    summary="Σ_{j∈I} |x′ⱼ − y′ⱼ|",
)
@traits(domain="quantitative")
def manhattan(
    A: FloatArray, B: FloatArray, quantitative: BoolArray, decimals: int, params: Any = None, /
) -> FloatArray:
    """Manhattan (city-block, L₁) distance: Σⱼ |x′ⱼ − y′ⱼ| over the quantitative features."""
    del quantitative, params
    total = _pairs(A, B)
    for j in range(A.shape[1]):
        total += np.abs(A[:, j, None] - B[None, :, j])
    return round_distances(total, decimals)


@METRICS.register(
    "euclidean",
    aliases=("l2",),
    summary="√(Σ_{j∈I} (x′ⱼ − y′ⱼ)²)",
)
@traits(domain="quantitative")
def euclidean(
    A: FloatArray, B: FloatArray, quantitative: BoolArray, decimals: int, params: Any = None, /
) -> FloatArray:
    """Euclidean (L₂) distance: √(Σⱼ (x′ⱼ − y′ⱼ)²) over the quantitative features."""
    del quantitative, params
    total = _pairs(A, B)
    for j in range(A.shape[1]):
        total += np.square(A[:, j, None] - B[None, :, j])
    return round_distances(np.sqrt(total), decimals)


@METRICS.register(
    "chebyshev",
    aliases=("maximum", "l-inf"),
    summary="max_{j∈I} |x′ⱼ − y′ⱼ|",
)
@traits(domain="quantitative")
def chebyshev(
    A: FloatArray, B: FloatArray, quantitative: BoolArray, decimals: int, params: Any = None, /
) -> FloatArray:
    """Chebyshev (maximum, L∞) distance: maxⱼ |x′ⱼ − y′ⱼ| over the quantitative features."""
    del quantitative, params
    largest = _pairs(A, B)
    for j in range(A.shape[1]):
        np.maximum(largest, np.abs(A[:, j, None] - B[None, :, j]), out=largest)
    return round_distances(largest, decimals)


@params_dataclass(frozen=True, config=PARAMS_CONFIG)
class MinkowskiParams:
    """Parameters of the ``minkowski`` metric."""

    p: float = Field(default=3.0, ge=1.0)
    """The order p ≥ 1 (1: Manhattan, 2: Euclidean)."""


@METRICS.register(
    "minkowski",
    aliases=("lp",),
    summary="(Σ_{j∈I} |x′ⱼ − y′ⱼ|ᵖ)^(1/p), p ≥ 1",
    params_type=MinkowskiParams,
)
@traits(domain="quantitative")
def minkowski(
    A: FloatArray,
    B: FloatArray,
    quantitative: BoolArray,
    decimals: int,
    params: MinkowskiParams | None = None,
    /,
) -> FloatArray:
    """Minkowski distance of order p: (Σⱼ |x′ⱼ − y′ⱼ|ᵖ)^(1/p) over the quantitative features."""
    del quantitative
    p = (params or MinkowskiParams()).p
    total = _pairs(A, B)
    for j in range(A.shape[1]):
        total += np.abs(A[:, j, None] - B[None, :, j]) ** p
    return round_distances(total ** (1.0 / p), decimals)


@METRICS.register(
    "canberra",
    summary="Σ_{j∈I} |x′ⱼ − y′ⱼ| / (|x′ⱼ| + |y′ⱼ|)",
)
@traits(domain="quantitative")
def canberra(
    A: FloatArray, B: FloatArray, quantitative: BoolArray, decimals: int, params: Any = None, /
) -> FloatArray:
    """Canberra distance: Σⱼ |x′ⱼ − y′ⱼ| / (|x′ⱼ| + |y′ⱼ|) over the quantitative features.

    A feature on which both values are 0 contributes 0.
    """
    del quantitative, params
    total = _pairs(A, B)
    for j in range(A.shape[1]):
        difference = np.abs(A[:, j, None] - B[None, :, j])
        size = np.abs(A[:, j, None]) + np.abs(B[None, :, j])
        total += np.divide(difference, size, out=np.zeros_like(difference), where=size > 0.0)
    return round_distances(total, decimals)


@METRICS.register(
    "cosine",
    summary="1 − (x′·y′) / (‖x′‖·‖y′‖) on I",
)
@traits(domain="quantitative")
def cosine(
    A: FloatArray, B: FloatArray, quantitative: BoolArray, decimals: int, params: Any = None, /
) -> FloatArray:
    """Cosine distance: 1 − (x′·y′) / (‖x′‖·‖y′‖) over the quantitative features.

    It compares directions only and does not satisfy the triangle inequality. Two zero vectors
    are at distance 0; a zero vector and a non-zero one at distance 1.
    """
    del quantitative, params
    product = _pairs(A, B)
    for j in range(A.shape[1]):
        product += A[:, j, None] * B[None, :, j]
    length_a = np.sqrt(np.sum(np.square(A), axis=1))[:, None]
    length_b = np.sqrt(np.sum(np.square(B), axis=1))[None, :]
    size = length_a * length_b
    similarity = np.divide(product, size, out=np.zeros_like(product), where=size > 0.0)
    similarity[(length_a == 0.0) & (length_b == 0.0)] = 1.0
    return round_distances(np.clip(1.0 - similarity, 0.0, 2.0), decimals)


@dataclass(frozen=True, eq=False)
class Whitening:
    """The fit of ``mahalanobis``: a matrix W with W·Wᵀ = S⁺."""

    matrix: FloatArray
    """(features × rank) — the eigenvectors of the training covariance S with a positive
    eigenvalue, each divided by the square root of its eigenvalue."""

    def __post_init__(self) -> None:
        """Freeze the array."""
        readonly(self.matrix)


def fit_whitening(Z: FloatArray, quantitative: BoolArray, params: Any = None, /) -> Whitening:
    """Whitening matrix of the training covariance (fit step, ADR-050).

    S is the sample covariance of the unified training values (divisor m − 1). Directions without
    variance are left out, which is the Moore–Penrose inverse S⁺ when S is singular.
    """
    del quantitative, params
    m, n = Z.shape
    if n == 0 or m < 2:
        return Whitening(np.zeros((n, 0)))
    centred = Z - Z.mean(axis=0)
    covariance = centred.T @ centred / (m - 1)
    values, vectors = np.linalg.eigh(covariance)
    kept = values > max(float(values.max()), 0.0) * n * np.finfo(np.float64).eps
    return Whitening(vectors[:, kept] / np.sqrt(values[kept]))


@METRICS.register(
    "mahalanobis",
    summary="√((x′ − y′)ᵀ S⁺ (x′ − y′)) on I, S the training covariance",
)
@traits(domain="quantitative", fit=fit_whitening)
def mahalanobis(
    A: FloatArray, B: FloatArray, quantitative: BoolArray, decimals: int, params: Whitening, /
) -> FloatArray:
    """Mahalanobis distance: √((x′ − y′)ᵀ S⁺ (x′ − y′)) over the quantitative features.

    S is the covariance of the training objects (S⁺ its Moore–Penrose inverse when it is
    singular). Computed as the Euclidean distance of the whitened values x′·W, W·Wᵀ = S⁺.
    """
    return euclidean(A @ params.matrix, B @ params.matrix, quantitative, decimals, None)


# ---------------------------------------------------------------- a metric on set J


@METRICS.register(
    "hamming",
    aliases=("overlap",),
    summary="Σ_{j∈J} [xⱼ ≠ yⱼ]",
)
@traits(domain="nominal")
def hamming(
    A: FloatArray, B: FloatArray, quantitative: BoolArray, decimals: int, params: Any = None, /
) -> FloatArray:
    """Hamming (overlap) distance: the number of nominal features on which x and y differ."""
    del quantitative, params
    total = _pairs(A, B)
    for j in range(A.shape[1]):
        total += A[:, j, None] != B[None, :, j]
    return round_distances(total, decimals)
