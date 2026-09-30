"""Margins of the latent features with and without the majorizer (sheet *Margin Analysis*).

For a latent feature d on the training objects, with yᵢ = +1 for K1 and −1 for K2:

- boundary b = (min over K1 of d + max over K2 of d)/2;
- margin width = min over K1 of d − max over K2 of d (> 0: the classes are separated);
- object margin mᵢ = yᵢ·(dᵢ − b) (> 0: on its correct side); rule ŷ = K1 if d > b, K2 otherwise.

*With majorizer*: the latent features r₁ … r_p of the HAG. *Without*: the generalized estimate
R(Sₜ) = y₀ + … + yⱼ — the contribution values of the same TUPLAM features summed, no majorizing
function. The gain (width with − width without) shows how the regularisation widens the margin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.core.arrays import FloatArray, IntArray, freeze_fields
from context_synthetic_recognition.core.meta import K1_DECISION, K2_DECISION
from context_synthetic_recognition.core.model import CSModel


@dataclass(frozen=True, eq=False)
class Margin:
    """The margin of one latent feature on the training objects."""

    values: FloatArray
    """(m,) the feature d."""
    class_index: IntArray
    """(m,) 0 = K1, 1 = K2."""
    boundary: float
    """b = (min K1 + max K2)/2."""
    left: float
    """max over K2 of d."""
    left_object: int
    """First object (0-based) with the value ``left`` (the workbook's MATCH)."""
    right: float
    """min over K1 of d."""
    right_object: int
    """First object with the value ``right``."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)

    @property
    def width(self) -> float:
        """Min K1 − max K2 (> 0: the classes are separated on this feature)."""
        return self.right - self.left

    @property
    def sign(self) -> FloatArray:
        """(m,) yᵢ = +1 for K1, −1 for K2."""
        return np.where(self.class_index == 0, 1.0, -1.0)

    @property
    def object_margins(self) -> FloatArray:
        """(m,) mᵢ = yᵢ·(dᵢ − b)."""
        return self.sign * (self.values - self.boundary)

    @property
    def predictions(self) -> IntArray:
        """(m,) ŷ = 1 (K1) if d > b, 2 (K2) otherwise."""
        return np.where(self.values > self.boundary, K1_DECISION, K2_DECISION).astype(np.int64)

    @property
    def correct(self) -> int:
        """Objects with ŷ equal to their class."""
        return int(np.count_nonzero(self.predictions == self.class_index + 1))


def margin(values: npt.ArrayLike, class_index: npt.ArrayLike) -> Margin:
    """The margin of the feature ``values`` between K1 (index 0) and K2 (index 1).

    Raises:
        ValueError: A class has no objects.
    """
    d = np.array(values, dtype=np.float64)
    y = np.array(class_index, dtype=np.int64)
    if not ((y == 0).any() and (y == 1).any()):
        raise ValueError("a margin needs objects of both classes")
    left = float(d[y == 1].max())
    right = float(d[y == 0].min())
    return Margin(
        values=d,
        class_index=y,
        boundary=(right + left) / 2,
        left=left,
        left_object=int(np.flatnonzero(d == left)[0]),
        right=right,
        right_object=int(np.flatnonzero(d == right)[0]),
    )


@dataclass(frozen=True)
class MarginAnalysis:
    """Every latent feature's margin with and without the majorizer."""

    with_majorizer: tuple[Margin, ...]
    """r₁ … r_p of the HAG."""
    without_majorizer: tuple[Margin, ...]
    """y₀ + … + yⱼ for j = 1 … p."""

    @property
    def gains(self) -> tuple[float, ...]:
        """Width with − width without, per latent feature."""
        return tuple(
            a.width - b.width
            for a, b in zip(self.with_majorizer, self.without_majorizer, strict=True)
        )


def margin_analysis(model: CSModel) -> MarginAnalysis:
    """The sheet *Margin Analysis* for a fitted model (empty when p = 0)."""
    data = model.meta_dataset
    summed = np.cumsum(data.initial, axis=1)  # y₀ + … + yⱼ, left to right as the workbook's SUM
    y = data.class_index
    return MarginAnalysis(
        with_majorizer=tuple(margin(data.latent[:, j], y) for j in range(data.p)),
        without_majorizer=tuple(margin(summed[:, j + 1], y) for j in range(data.p)),
    )
