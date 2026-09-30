"""Array type aliases and immutability helpers shared by the numerical core."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
"""Real values (distances, normalized features, f, η, …)."""
IntArray = npt.NDArray[np.int64]
"""Counts, indices, ranks and gradations."""
BoolArray = npt.NDArray[np.bool_]
"""Masks (quantitative features, correct-side flags, bits)."""


def readonly(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Mark ``array`` read-only in place and return it (trace objects are immutable)."""
    array.setflags(write=False)
    return array


def freeze_fields(obj: object) -> None:
    """Make every numpy array attribute of a (frozen) dataclass instance read-only."""
    for value in vars(obj).values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)


def as_float_matrix(values: npt.ArrayLike, what: str = "X") -> FloatArray:
    """A 2-D float64 copy of ``values``.

    Raises:
        ValueError: ``values`` is not two-dimensional.
    """
    matrix = np.array(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{what} must be a 2-D array (objects × features), got {matrix.ndim}-D.")
    return matrix
