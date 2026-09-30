"""Nested neighbourhoods (Step 3): reference objects ordered by distance.

For every target object the reference (training) objects are sorted by (distance, original
index): equal distances go to the smaller original index (ADR-008). A training object is never
its own neighbour. The first k entries of an order are the k-neighbourhood; the neighbourhoods
of the permitted k are nested (k = 3 ⊂ k = 5 ⊂ …).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.core.arrays import FloatArray, IntArray


def neighbour_order(distances: FloatArray, exclude: npt.ArrayLike | None = None) -> IntArray:
    """Order the reference objects of every target object — Step 3, sheets *Sorted Neighbors*.

    Args:
        distances: (targets × references) distances of one base operator.
        exclude: One reference index per target to leave out — the target itself when it is a
            training object (its own neighbour never counts). ``None``: keep every reference.

    Returns:
        (targets × L) original indices of the references by (distance, index); L = number of
        references, minus one if ``exclude`` is given.
    """
    order = np.argsort(distances, axis=1, kind="stable").astype(np.int64)
    if exclude is None:
        return order
    excluded = np.asarray(exclude, dtype=np.int64).reshape(-1)
    targets, references = distances.shape
    if excluded.shape != (targets,):
        raise ValueError(f"exclude needs one index per target ({targets}), got {excluded.size}")
    if excluded.size and (excluded.min() < 0 or excluded.max() >= references):
        raise ValueError(f"exclude indices must lie in 0 … {references - 1}")
    kept: IntArray = order[order != excluded[:, None]]
    return kept.reshape(targets, references - 1)


def rank_matrix(order: IntArray, references: int) -> IntArray:
    """Rank of every reference in every target's order (1 = nearest; 0 = not listed, i.e. self).

    The rank matrix of the sheets *Sorted Neighbors*: row i, column j = rank of Sⱼ among the
    neighbours of Sᵢ.
    """
    ranks = np.zeros((order.shape[0], references), dtype=np.int64)
    positions = np.broadcast_to(np.arange(1, order.shape[1] + 1, dtype=np.int64), order.shape)
    np.put_along_axis(ranks, order, positions, axis=1)
    return ranks


def neighbour_classes(order: IntArray, class_index: IntArray, depth: int | None = None) -> IntArray:
    """Class index of the neighbours in order, for the first ``depth`` ranks (all by default)."""
    return class_index[order if depth is None else order[:, :depth]]


def cumulative_count(classes: IntArray, target: int | IntArray) -> IntArray:
    """Number of neighbours of class ``target`` among ranks 1 … t, for every rank t.

    ``target`` is a class index, or one class index per row. With the object's own class this
    is the same-class count μ; with class K1 (index 0) it is χ₁.
    """
    wanted = np.asarray(target, dtype=np.int64)
    if wanted.ndim == 1:
        wanted = wanted[:, None]
    return np.cumsum(classes == wanted, axis=1, dtype=np.int64)
