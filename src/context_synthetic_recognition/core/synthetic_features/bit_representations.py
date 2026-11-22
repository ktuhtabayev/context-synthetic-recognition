"""Bit representations of the k-neighbourhoods (Task 2): integers read as nominal gradations.

Two representations, both studied on the training sample with membership (1) and stability (2):

- the **class-sequence bit mask** (section 1.2): for one k, bit j is 1 when the j-th nearest
  neighbour belongs to K1, the nearest neighbour the most significant bit — 2ᵏ gradations,
  β = 2ᵏ − 1 (the article prints "β = 2k − 1"; ADR-067). The article does not code the classes as
  bits; K1 = 1 is this project's convention. Masks are formed for k ≤ :data:`MAX_MASK_BITS`.
- the **majority bit chain** (section 1.3): bit i is 1 when the object's own-class count for the
  i-th admissible k is a majority (μ > k/2), the first admissible k the most significant bit —
  2^|K̂| gradations, β = 2^|K̂| − 1.

Neither enters Ψ(r): the chain reads the object's own class, and both describe the training sample
(its distinguishability as k grows).
"""

from __future__ import annotations

import numpy as np

from context_synthetic_recognition.core.arrays import BoolArray, IntArray
from context_synthetic_recognition.core.local_context.neighbours import neighbour_classes

MAX_CHAIN_BITS = 62
"""Majority chains are formed for at most this many admissible k (they must fit an int64)."""
MAX_MASK_BITS = 62
"""Class-sequence masks are formed for k up to this value (they must fit an int64)."""


def majority_chains(mu: IntArray, ks: tuple[int, ...]) -> tuple[BoolArray, IntArray]:
    """Majority bit chains of one operator (section 1.3).

    Bit i of an object is 1 if its same-class count for the i-th admissible k is a majority
    (μ > k/2); the first admissible k is the most significant bit.

    Args:
        mu: (objects × len(ks)) own-class counts of one operator.
        ks: The admissible k.

    Returns:
        The bits (objects × len(ks)) and the chain value of every object.

    Raises:
        ValueError: More than :data:`MAX_CHAIN_BITS` admissible k.
    """
    if len(ks) > MAX_CHAIN_BITS:
        raise ValueError(
            f"majority chains need at most {MAX_CHAIN_BITS} admissible k, got {len(ks)}"
        )
    bits = 2 * np.asarray(mu, dtype=np.int64) > np.asarray(ks, dtype=np.int64)
    weights = np.left_shift(1, np.arange(len(ks) - 1, -1, -1, dtype=np.int64))
    return bits, (bits.astype(np.int64) @ weights).astype(np.int64)


def class_sequence_masks(
    order: IntArray, class_index: IntArray, ks: tuple[int, ...]
) -> tuple[tuple[int, ...], IntArray]:
    """Class-sequence bit masks of one operator (section 1.2), one column per k ≤ 62.

    Args:
        order: Neighbour order of the training objects (self excluded).
        class_index: Class index of every training object (0 = K1).
        ks: The admissible k.

    Returns:
        The k for which masks are formed (k ≤ :data:`MAX_MASK_BITS`) and the (objects × those k)
        mask values.
    """
    usable = tuple(k for k in ks if k <= MAX_MASK_BITS)
    if not usable:
        return (), np.empty((order.shape[0], 0), dtype=np.int64)
    bits = (neighbour_classes(order, class_index, depth=max(usable)) == 0).astype(np.int64)
    columns = []
    for k in usable:
        weights = np.left_shift(1, np.arange(k - 1, -1, -1, dtype=np.int64))
        columns.append(bits[:, :k] @ weights)
    return usable, np.column_stack(columns).astype(np.int64)
