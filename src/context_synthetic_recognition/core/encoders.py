"""Neighbourhood counts and synthetic-feature encoders (Steps 4 and 5).

Two counts are taken over the k nearest neighbours of an object:

- μ, the **same-class count** (Step 4, sheet *Synthetic Features (k-NN)*) — it uses the class of
  the object itself, so it exists on the training side only and feeds formulas (1)–(4);
- χ₁, the number of **K1 objects** (sheet *Ψ(r) Binary Features*) — it uses only the classes of
  the neighbours, so it exists for a new object too (Theorem).

An encoder turns χ₁ into the value of a synthetic feature. Registry :data:`ENCODERS`:
``formula-5`` (default). μ and the bit masks are training-side gradations, not encoders: they
read the object's own class and could not describe a new object (ADR-021).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from context_synthetic_recognition.core.arrays import IntArray
from context_synthetic_recognition.core.neighbours import cumulative_count, neighbour_classes
from context_synthetic_recognition.core.registry import Registry

K1 = 0
"""Class index of K1."""


def _at_ranks(cumulative: IntArray, ks: Sequence[int]) -> IntArray:
    return cumulative[:, np.asarray(ks, dtype=np.int64) - 1]


def same_class_counts(
    order: IntArray, class_index: IntArray, own_class: IntArray, ks: Sequence[int]
) -> IntArray:
    """μ — number of the k nearest neighbours in the object's own class (Step 4).

    Args:
        order: Neighbour order of the training objects (self excluded).
        class_index: Class index of every training object.
        own_class: Class index of every target object (training side only).
        ks: The permitted k.

    Returns:
        (targets × len(ks)) counts.
    """
    classes = neighbour_classes(order, class_index, depth=max(ks))
    return _at_ranks(cumulative_count(classes, own_class), ks)


def k1_counts(order: IntArray, class_index: IntArray, ks: Sequence[int]) -> IntArray:
    """χ₁ — number of K1 objects among the k nearest neighbours; class-free (Theorem).

    Returns:
        (targets × len(ks)) counts; χ₂ = k − χ₁.
    """
    classes = neighbour_classes(order, class_index, depth=max(ks))
    return _at_ranks(cumulative_count(classes, K1), ks)


class Encoder(Protocol):
    """Value of a synthetic feature from the K1 count of the k-neighbourhood."""

    def __call__(self, chi1: IntArray, k: int, params: Any, /) -> IntArray:
        """Return the feature value of every object, given its χ₁ for this k."""
        ...


ENCODERS: Registry[Encoder] = Registry("encoders")
"""Synthetic-feature encoders by name."""


@ENCODERS.register(
    "formula-5",
    aliases=("formula5", "majority"),
    summary="aᵤ = 1 if χ₁ > [k/2], 2 if χ₂ > [k/2] (default)",
)
def formula_5(chi1: IntArray, k: int, params: Any = None, /) -> IntArray:
    """Formula (5): the majority class of the k-neighbourhood, aᵤ ∈ {1, 2}.

    aᵤ(S) = 1 if χ₁(S, k) > [k/2], aᵤ(S) = 2 if χ₂(S, k) = k − χ₁(S, k) > [k/2]. For odd k
    exactly one of the two holds (every k strategy returns odd k).
    """
    del params
    if k % 2 == 0:
        raise ValueError(f"formula (5) needs an odd k, got {k}")
    return np.where(np.asarray(chi1) > k // 2, 1, 2).astype(np.int64)
