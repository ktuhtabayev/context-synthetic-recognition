"""Majorizing functions φ of the hierarchical agglomerative algorithm (Steps 3–4).

The agglomerative algorithm moves the generalized estimate b of every object towards its class:

    b ← b + α·φ(−b) for Sₜ ∈ K1,        b ← b − α·φ(−b) for Sₜ ∈ K2.

Every built-in φ maps ℝ onto (0, 1), increases and has φ(0) = 1/2, so each step α·φ(−b) lies in
(0, α): it always pushes K1 up and K2 down, and it pushes harder the further b lies on the wrong
side of 0 (ADR-026). Registry :data:`MAJORIZING_FUNCTIONS`:

``sigmoid`` (default)
    The logistic sigmoid σ(x) = 1/(1 + e^(−x)), as in the Excel experiment workbook (the article
    does not define φ, ADR-068).
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


class MajorizingFunction(Protocol):
    """A majorizing function φ, applied element-wise."""

    def __call__(self, x: FloatArray, params: Any, /) -> FloatArray:
        """Return φ(x) for every element of ``x``."""
        ...


MAJORIZING_FUNCTIONS: Registry[MajorizingFunction] = Registry("majorizing functions")
"""Majorizing functions by name."""


@MAJORIZING_FUNCTIONS.register(
    "sigmoid", aliases=("logistic",), summary="σ(x) = 1/(1 + e^(−x)) — logistic (default)"
)
def sigmoid(x: FloatArray, params: Any = None, /) -> FloatArray:
    """Logistic sigmoid σ(x) = 1/(1 + e^(−x)), the default majorizing function φ.

    For x = −b this is 1/(1 + e^b), the operation order of the Excel experiment; for b > 709 the
    exponential overflows to ∞ and σ is exactly 0, its limit in double precision.
    """
    del params
    with np.errstate(over="ignore"):
        return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


@MAJORIZING_FUNCTIONS.register("tanh", summary="(1 + tanh x)/2 = σ(2x)")
def tanh(x: FloatArray, params: Any = None, /) -> FloatArray:
    """Hyperbolic tangent scaled to (0, 1): (1 + tanh x)/2."""
    del params
    return np.asarray((1.0 + np.tanh(x)) / 2.0, dtype=np.float64)


@MAJORIZING_FUNCTIONS.register("arctan", summary="1/2 + arctan(x)/π")
def arctan(x: FloatArray, params: Any = None, /) -> FloatArray:
    """The arctangent scaled to (0, 1): φ(x) = 1/2 + atan(x)/π."""
    del params
    return np.asarray(0.5 + np.arctan(x) / np.pi, dtype=np.float64)


@MAJORIZING_FUNCTIONS.register("softsign", summary="(1 + x/(1 + |x|))/2")
def softsign(x: FloatArray, params: Any = None, /) -> FloatArray:
    """Softsign scaled to (0, 1): (1 + x/(1 + |x|))/2."""
    del params
    return np.asarray((1.0 + x / (1.0 + np.abs(x))) / 2.0, dtype=np.float64)


def majorize(
    b: FloatArray,
    sign: FloatArray,
    alpha: float,
    majorizing_function: MajorizingFunction,
    params: Any,
) -> FloatArray:
    """One pass of the majorizing function: b + s·α·φ(−b), s = +1 for K1 and −1 for K2.

    Steps 3 and 4 of the hierarchical agglomerative algorithm.

    Args:
        b: Generalized estimates, objects along the first axis (a vector or objects × candidates).
        sign: (m,) +1 for the objects of K1, −1 for those of K2.
        alpha: α, 0 < α < 1.
        majorizing_function: φ.
        params: Its validated parameters.
    """
    step = sign * alpha
    if b.ndim == 2:
        step = step[:, None]
    return b + step * majorizing_function(-b, params)
