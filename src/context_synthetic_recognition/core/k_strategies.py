"""Permitted k: the neighbourhood sizes of the synthetic features.

Registry :data:`K_STRATEGIES`:

``formula`` (default, ADR-005/006)
    Odd k = k_min, k_min + 2, …, k_max with k_min = 3 fixed and k_max = 2·min_i|Kᵢ| − 3, the
    class sizes counted from the training objects of the current fit. At k = k_max a same-class
    majority needs (k + 1)/2 = min|Kᵢ| − 1 neighbours. k = 1 when min|Kᵢ| = 2; no k when
    min|Kᵢ| ≤ 1. Optional ``k_max_cap`` and even ``step``.
``article``
    The draft article's text: odd k ≤ min(|K1|, |K2|), i.e. k = 1, 3, … (to be corrected).
``explicit``
    A given list of odd k.
``range``
    Odd k = start, start + step, …, ≤ stop.

Every strategy returns odd k (formula (5) then always has a majority) and never more than the
number of neighbours a training object has (m − 1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass as params_dataclass

from context_synthetic_recognition.core.registry import PARAMS_CONFIG, Registry
from context_synthetic_recognition.errors import ModelUndefinedError
from context_synthetic_recognition.notation import join_values

K_MIN = 3
"""k_min of the formula rule, fixed for every dataset (ADR-005)."""


@dataclass(frozen=True)
class PermittedK:
    """The permitted k of one fit."""

    ks: tuple[int, ...]
    """Odd k in increasing order."""
    rule: str
    """Registry name of the strategy."""
    class_sizes: tuple[int, ...]
    """|K1|, |K2|, … of the training sample the k were derived from."""
    k_min: int | None = None
    """k_min of the formula rule."""
    k_max: int | None = None
    """k_max = 2·min|Kᵢ| − 3 of the formula rule (before any cap)."""
    note: str = ""
    """Remark shown with the range (e.g. why k = 1)."""

    def __post_init__(self) -> None:
        """Check that the k are odd, positive and strictly increasing."""
        if not self.ks:
            raise ModelUndefinedError(f"{self.rule}: no k is permitted")
        if any(k < 1 or k % 2 == 0 for k in self.ks):
            raise ValueError(f"{self.rule}: every k must be odd and positive, got {self.ks}")
        if any(b <= a for a, b in zip(self.ks, self.ks[1:], strict=False)):
            raise ValueError(f"{self.rule}: the k must be strictly increasing, got {self.ks}")

    @property
    def min_class_size(self) -> int:
        """min_i |Kᵢ|."""
        return min(self.class_sizes)

    @property
    def count(self) -> int:
        """Number of permitted k."""
        return len(self.ks)

    @property
    def label(self) -> str:
        """``3, 5`` — as on the sheet *Parameters*."""
        return join_values(self.ks)


class KStrategy(Protocol):
    """Chooses the permitted k from the training class sizes."""

    def __call__(self, class_sizes: tuple[int, ...], params: Any, /) -> PermittedK:
        """Return the permitted k for a training sample with these class sizes."""
        ...


K_STRATEGIES: Registry[KStrategy] = Registry("k strategies")
"""k strategies by name."""


def _check_odd(values: Sequence[int]) -> None:
    if any(k < 1 or k % 2 == 0 for k in values):
        raise ValueError("every k must be odd and positive")


@params_dataclass(frozen=True, config=PARAMS_CONFIG)
class FormulaParams:
    """Parameters of the ``formula`` rule (ADR-006)."""

    k_max_cap: int | None = Field(default=None, ge=K_MIN)
    """Upper limit for k_max (``None``: the literal formula)."""
    step: int = Field(default=2, ge=2)
    """Distance between consecutive k; even, so that every k stays odd."""

    @field_validator("step")
    @classmethod
    def _step_is_even(cls, value: int) -> int:
        if value % 2:
            raise ValueError("the step must be even (every k stays odd)")
        return value


@K_STRATEGIES.register(
    "formula",
    summary="odd k = 3 … 2·min|Kᵢ| − 3 (default)",
    params_type=FormulaParams,
)
def formula(class_sizes: tuple[int, ...], params: FormulaParams | None = None, /) -> PermittedK:
    """Permitted k by the k-range formula — k ∈ [k_min, k_max], k_max = 2·min_i|Kᵢ| − 3.

    k_min = 3 is fixed for every dataset; k = 1 is permitted when min|Kᵢ| = 2 (ADR-005).

    Raises:
        ModelUndefinedError: min|Kᵢ| ≤ 1 — no k is permitted (Definition 1).
    """
    chosen = params or FormulaParams()
    smallest = min(class_sizes)
    k_max = 2 * smallest - 3
    if smallest <= 1:
        raise ModelUndefinedError(
            f"no k is permitted: the smallest class has {smallest} object(s) "
            "(k_max = 2·min|Kᵢ| − 3 < 1; Definition 1, ADR-005)"
        )
    if smallest == 2:
        return PermittedK(
            (1,), "formula", class_sizes, 1, 1, note="k = 1 because min|Kᵢ| = 2 (ADR-005)"
        )
    top = k_max if chosen.k_max_cap is None else min(k_max, chosen.k_max_cap)
    note = f"k_max capped at {chosen.k_max_cap}" if top < k_max else ""
    return PermittedK(
        tuple(range(K_MIN, top + 1, chosen.step)), "formula", class_sizes, K_MIN, k_max, note
    )


@K_STRATEGIES.register("article", summary="odd k ≤ min(|K1|, |K2|) — the draft article's text")
def article(class_sizes: tuple[int, ...], params: Any = None, /) -> PermittedK:
    """Permitted k as written in the draft article: r = |{k odd, k ≤ min(|K1|, |K2|)}|.

    Kept for comparison; the text is to be corrected to the formula rule (ADR-005).
    """
    del params
    smallest = min(class_sizes)
    return PermittedK(tuple(range(1, smallest + 1, 2)), "article", class_sizes)


@params_dataclass(frozen=True, config=PARAMS_CONFIG)
class ExplicitParams:
    """Parameters of the ``explicit`` strategy."""

    values: tuple[int, ...] = Field(min_length=1)
    """The permitted k: odd, strictly increasing."""

    @field_validator("values")
    @classmethod
    def _odd_increasing(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        _check_odd(values)
        if list(values) != sorted(set(values)):
            raise ValueError("the k must be strictly increasing")
        return values


@K_STRATEGIES.register("explicit", summary="a given list of odd k", params_type=ExplicitParams)
def explicit(class_sizes: tuple[int, ...], params: ExplicitParams, /) -> PermittedK:
    """Permitted k given explicitly."""
    return PermittedK(params.values, "explicit", class_sizes)


@params_dataclass(frozen=True, config=PARAMS_CONFIG)
class RangeParams:
    """Parameters of the ``range`` strategy."""

    stop: int = Field(ge=1)
    """Largest k (inclusive; an even stop ends at stop − 1)."""
    start: int = Field(default=K_MIN, ge=1)
    """Smallest k; odd."""
    step: int = Field(default=2, ge=2)
    """Distance between consecutive k; even."""

    @field_validator("start")
    @classmethod
    def _start_is_odd(cls, value: int) -> int:
        _check_odd([value])
        return value

    @field_validator("step")
    @classmethod
    def _step_is_even(cls, value: int) -> int:
        if value % 2:
            raise ValueError("the step must be even (every k stays odd)")
        return value


@K_STRATEGIES.register(
    "range", summary="odd k = start, start + step, … ≤ stop", params_type=RangeParams
)
def k_range(class_sizes: tuple[int, ...], params: RangeParams, /) -> PermittedK:
    """Permitted k from a range."""
    return PermittedK(
        tuple(range(params.start, params.stop + 1, params.step)), "range", class_sizes
    )


def permitted_k(
    name: str, params: dict[str, Any] | None, class_sizes: Sequence[int], neighbours: int
) -> PermittedK:
    """Resolve and apply a k strategy for a training sample.

    Args:
        name: Registry name of the strategy.
        params: Its parameters (validated here).
        class_sizes: |K1|, |K2|, … of the training sample.
        neighbours: Number of neighbours a training object has (m − 1).

    Raises:
        ModelUndefinedError: No k is permitted, or a k exceeds the number of neighbours.
        ConfigError: Invalid parameters.
    """
    sizes = tuple(int(s) for s in class_sizes)
    result = K_STRATEGIES.get(name)(sizes, K_STRATEGIES.make_params(name, params))
    too_large = [k for k in result.ks if k > neighbours]
    if too_large:
        raise ModelUndefinedError(
            f"k = {join_values(too_large)} exceeds the {neighbours} neighbours of a training object"
        )
    return result
