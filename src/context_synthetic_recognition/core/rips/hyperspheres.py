"""The hyperspheres of R-IPS: the radius r(S), the sphere count g(S) and the queue (Steps 1–2).

For an object S of class Kₜ in the working set E₁:

* r(S) — the distance to the nearest object of the other class in E₁: the radius of the
  hypersphere of S;
* g(S) = |{S′ ∈ E₁ ∩ Kₜ | ρ(S, S′) < r(S)}| — the objects of its own class strictly inside the
  hypersphere, S itself included (ρ(S, S) = 0 < r(S)), so g(S) ≥ 1 whenever r(S) > 0;
* 1/r(S) — the priority of Step 2: the largest 1/r, the smallest r, comes first.

The queue — r ascending — is fixed: the radius of a waiting object never changes. ⚠ The article
does not order equal radii; the smaller index comes first. An object with r = 0 coincides with an
object of the other class: it has no 1/r and never enters the queue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.core.arrays import FloatArray, IntArray, freeze_fields
from context_synthetic_recognition.errors import ModelUndefinedError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class Hyperspheres:
    """r(S), g(S), 1/r(S) and the queue of Step 2 for the objects of E₁ (indices of E₀)."""

    working: IntArray
    """E₁ = E₀ ∖ T: the objects that take part (ascending)."""
    radius: FloatArray
    """r(S) for every object of E₀; NaN for the objects of T."""
    sphere_count: IntArray
    """g(S) for every object of E₀; 0 for the objects of T."""
    nearest_other: IntArray
    """The nearest object of the other class in E₁ (at distance r(S)); −1 for the objects of T."""
    priority: FloatArray
    """1/r(S); NaN for the objects of T and where r(S) = 0."""
    queue: IntArray
    """The objects of E₁ with r > 0 by r ascending (the smaller index first on equal r)."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)

    @property
    def zero_radius(self) -> IntArray:
        """Objects of E₁ with r = 0 (they coincide with an object of the other class)."""
        r = self.radius[self.working]
        return self.working[r <= 0.0].astype(np.int64)


def hyperspheres(
    distances: FloatArray,
    class_index: npt.ArrayLike,
    working: npt.ArrayLike | None = None,
    *,
    decimals: int = 10,
) -> Hyperspheres:
    """Steps 1–2 of R-IPS on E₁: r(S), g(S), 1/r(S) and the queue.

    Args:
        distances: ρ between the objects of E₀, shape (m, m), symmetric with a zero diagonal.
        class_index: 0-based class of every object of E₀.
        working: The objects of E₁ = E₀ ∖ T (``None``: all of E₀).
        decimals: Rounding of 1/r, as for the distances.

    Raises:
        ModelUndefinedError: E₁ lacks a class, so no radius exists.
    """
    y = np.asarray(class_index, dtype=np.int64)
    m = y.size
    E1 = np.arange(m, dtype=np.int64) if working is None else np.asarray(working, dtype=np.int64)
    E1 = np.unique(E1)
    if np.unique(y[E1]).size < 2:
        raise ModelUndefinedError(
            "R-IPS needs objects of both classes in E₁: without the other class no object has a "
            "radius r(S)"
        )
    radius = np.full(m, np.nan)
    count = np.zeros(m, dtype=np.int64)
    nearest = np.full(m, -1, dtype=np.int64)
    sub = distances[np.ix_(E1, E1)]
    own = y[E1][:, None] == y[E1][None, :]
    other = np.where(own, np.inf, sub)
    first = np.argmin(other, axis=1)  # the smaller index on equal distances
    r = other[np.arange(E1.size), first]
    radius[E1] = r
    nearest[E1] = E1[first]
    count[E1] = np.count_nonzero(own & (sub < r[:, None]), axis=1)
    with np.errstate(divide="ignore"):
        priority = np.where(
            radius > 0.0, np.round(1.0 / np.where(radius > 0.0, radius, 1.0), decimals), np.nan
        )
    waiting = E1[r > 0.0]
    queue = waiting[np.lexsort((waiting, radius[waiting]))].astype(np.int64)
    zero = E1.size - waiting.size
    if zero:
        logger.warning(
            "objects with r = 0 coincide with an object of the other class: they never enter the "
            "queue of R-IPS",
            extra={"objects": int(zero)},
        )
    return Hyperspheres(E1, radius, count, nearest, priority, queue)
