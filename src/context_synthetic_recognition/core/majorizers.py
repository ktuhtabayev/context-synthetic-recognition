"""Majorizing functions ϕ of the HAG regulariser (Steps 3–4).

The HAG moves the generalized estimate b of every object towards its class:

    b ← b + α·ϕ(−b) for Sₜ ∈ K1,        b ← b − α·ϕ(−b) for Sₜ ∈ K2.

Every built-in ϕ maps ℝ onto (0, 1), increases and has ϕ(0) = 1/2, so each step α·ϕ(−b) lies in
(0, α): it always pushes K1 up and K2 down, and it pushes harder the further b lies on the wrong
side of 0 (ADR-026). Registry :data:`MAJORIZERS`:

``sigmoid`` (default)
    The logistic sigmoid σ(x) = 1/(1 + e^(−x)) of the template and the workbook.
``tanh``
    (1 + tanh x)/2 = σ(2x), a steeper logistic curve.
``arctan``
    1/2 + arctan(x)/π, with heavy tails.
``softsign``
    (1 + x/(1 + |x|))/2, algebraic tails.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from context_synthetic_recognition.core.arrays import FloatArray
from context_synthetic_recognition.core.registry import Registry


class Majorizer(Protocol):
    """A majorizing function ϕ, applied element-wise."""

    def __call__(self, x: FloatArray, params: Any, /) -> FloatArray:
        """Return ϕ(x) for every element of ``x``."""
        ...


MAJORIZERS: Registry[Majorizer] = Registry("majorizers")
"""Majorizing functions by name."""


@MAJORIZERS.register(
    "sigmoid", aliases=("logistic",), summary="σ(x) = 1/(1 + e^(−x)) — logistic (default)"
)
def sigmoid(x: FloatArray, params: Any = None, /) -> FloatArray:
    """Logistic sigmoid σ(x) = 1/(1 + e^(−x)), the majorizer of the template and the article.

    For x = −b this is 1/(1 + e^b), the operation order of the reference engine; for b > 709 the
    exponential overflows to ∞ and σ is exactly 0, its limit in double precision.
    """
    del params
    with np.errstate(over="ignore"):
        return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


@MAJORIZERS.register("tanh", summary="(1 + tanh x)/2 = σ(2x)")
def tanh(x: FloatArray, params: Any = None, /) -> FloatArray:
    """Hyperbolic tangent scaled to (0, 1): (1 + tanh x)/2."""
    del params
    return np.asarray((1.0 + np.tanh(x)) / 2.0, dtype=np.float64)


@MAJORIZERS.register("arctan", summary="1/2 + arctan(x)/π")
def arctan(x: FloatArray, params: Any = None, /) -> FloatArray:
    """The arctangent scaled to (0, 1): ϕ(x) = 1/2 + atan(x)/π."""
    del params
    return np.asarray(0.5 + np.arctan(x) / np.pi, dtype=np.float64)


@MAJORIZERS.register("softsign", summary="(1 + x/(1 + |x|))/2")
def softsign(x: FloatArray, params: Any = None, /) -> FloatArray:
    """Softsign scaled to (0, 1): (1 + x/(1 + |x|))/2."""
    del params
    return np.asarray((1.0 + x / (1.0 + np.abs(x))) / 2.0, dtype=np.float64)


def regularize(
    b: FloatArray, sign: FloatArray, alpha: float, majorizer: Majorizer, params: Any
) -> FloatArray:
    """One majorizer pass: b + s·α·ϕ(−b) with s = +1 for K1 and −1 for K2 (HAG Steps 3–4).

    Args:
        b: Generalized estimates, objects along the first axis (a vector or objects × candidates).
        sign: (m,) +1 for the objects of K1, −1 for those of K2.
        alpha: α, 0 < α < 1.
        majorizer: ϕ.
        params: Its validated parameters.
    """
    step = sign * alpha
    if b.ndim == 2:
        step = step[:, None]
    return b + step * majorizer(-b, params)
