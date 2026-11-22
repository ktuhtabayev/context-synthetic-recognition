"""The admissible values of k (K̂): the neighbourhood sizes of the synthetic features.

Registry :data:`K_STRATEGIES`:

``min-class``
    The article: odd k = 3, 5, …, k ≤ min(|K1|, |K2|) — section 1.3 starts the odd k at 3. The
    class sizes are counted from the training objects of the current fit; no k when
    min|Kᵢ| < 3.
``twice-min-class`` (ADR-005/006)
    Odd k = 3, 5, …, k_max with k_max = 2·min_i|Kᵢ| − 3: at k = k_max a same-class majority needs
    (k + 1)/2 = min|Kᵢ| − 1 neighbours. k = 1 when min|Kᵢ| = 2; no k when min|Kᵢ| ≤ 1. Optional
    ``k_max_cap`` and even ``step``.
``range``
    Odd k = start, start + step, …, ≤ stop (the article's experiments: 3 … 41).
``explicit``
    A given list of odd k.

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
"""The smallest admissible k of ``min-class`` and ``twice-min-class`` (section 1.3, ADR-005)."""


@dataclass(frozen=True)
class AdmissibleK:
    """The admissible k of one fit."""

    ks: tuple[int, ...]
    """Odd k in increasing order."""
    rule: str
    """Registry name of the strategy."""
    class_sizes: tuple[int, ...]
    """|K1|, |K2|, … of the training sample the k were derived from."""
    k_min: int | None = None
    """The smallest k of the rule (``min-class``, ``twice-min-class``)."""
    k_max: int | None = None
    """The largest k the rule allows before any cap (``min|Kᵢ|`` or ``2·min|Kᵢ| − 3``)."""
    note: str = ""
    """Remark shown with the range (e.g. why k = 1)."""

    def __post_init__(self) -> None:
        """Check that the k are odd, positive and strictly increasing."""
        if not self.ks:
            raise ModelUndefinedError(f"{self.rule}: no k is admissible")
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
        """Number of admissible k."""
        return len(self.ks)

    @property
    def label(self) -> str:
        """``3, 5`` — as on the sheet *Parameters*."""
        return join_values(self.ks)


class KStrategy(Protocol):
    """Chooses the admissible k from the training class sizes."""

    def __call__(self, class_sizes: tuple[int, ...], params: Any, /) -> AdmissibleK:
        """Return the admissible k for a training sample with these class sizes."""
        ...


K_STRATEGIES: Registry[KStrategy] = Registry("k strategies")
"""k strategies by name."""


def _check_odd(values: Sequence[int]) -> None:
    if any(k < 1 or k % 2 == 0 for k in values):
        raise ValueError("every k must be odd and positive")


@K_STRATEGIES.register(
    "min-class",
    summary="odd k = 3 … min(|K1|, |K2|) — the article",
)
def min_class(class_sizes: tuple[int, ...], params: Any = None, /) -> AdmissibleK:
    """The article's admissible k: odd k = 3, 5, …, k ≤ min(|K1|, |K2|).

    r = |{k : k odd, k ≤ min(|K1|, |K2|)}| (the article prints the condition [k/2] − k/2 = 0,
    which selects even k; the odd k of section 1.3 are meant).

    Raises:
        ModelUndefinedError: min|Kᵢ| < 3 — no odd k ≥ 3 is admissible (Definition 1).
    """
    del params
    smallest = min(class_sizes)
    if smallest < K_MIN:
        raise ModelUndefinedError(
            f"no k is admissible: the smallest class has {smallest} object(s), and the odd k "
            f"start at {K_MIN} (k ≤ min|Kᵢ|; Definition 1)"
        )
    return AdmissibleK(
        tuple(range(K_MIN, smallest + 1, 2)), "min-class", class_sizes, K_MIN, smallest
    )


@params_dataclass(frozen=True, config=PARAMS_CONFIG)
class TwiceMinClassParams:
    """Parameters of the ``twice-min-class`` rule (ADR-006)."""

    k_max_cap: int | None = Field(default=None, ge=K_MIN)
    """Upper limit for k_max (``None``: k_max = 2·min|Kᵢ| − 3)."""
    step: int = Field(default=2, ge=2)
    """Distance between consecutive k; even, so that every k stays odd."""

    @field_validator("step")
    @classmethod
    def _step_is_even(cls, value: int) -> int:
        if value % 2:
            raise ValueError("the step must be even (every k stays odd)")
        return value


@K_STRATEGIES.register(
    "twice-min-class",
    summary="odd k = 3 … 2·min|Kᵢ| − 3",
    params_type=TwiceMinClassParams,
)
def twice_min_class(
    class_sizes: tuple[int, ...], params: TwiceMinClassParams | None = None, /
) -> AdmissibleK:
    """Admissible k = 3, 5, …, k_max with k_max = 2·min_i|Kᵢ| − 3.

    k = 1 is admissible when min|Kᵢ| = 2 (ADR-005).

    Raises:
        ModelUndefinedError: min|Kᵢ| ≤ 1 — no k is admissible (Definition 1).
    """
    chosen = params or TwiceMinClassParams()
    smallest = min(class_sizes)
    k_max = 2 * smallest - 3
    if smallest <= 1:
        raise ModelUndefinedError(
            f"no k is admissible: the smallest class has {smallest} object(s) "
            "(k_max = 2·min|Kᵢ| − 3 < 1; Definition 1, ADR-005)"
        )
    if smallest == 2:
        return AdmissibleK(
            (1,), "twice-min-class", class_sizes, 1, 1, note="k = 1 because min|Kᵢ| = 2 (ADR-005)"
        )
    top = k_max if chosen.k_max_cap is None else min(k_max, chosen.k_max_cap)
    note = f"k_max capped at {chosen.k_max_cap}" if top < k_max else ""
    return AdmissibleK(
        tuple(range(K_MIN, top + 1, chosen.step)),
        "twice-min-class",
        class_sizes,
        K_MIN,
        k_max,
        note,
    )


@params_dataclass(frozen=True, config=PARAMS_CONFIG)
class ExplicitParams:
    """Parameters of the ``explicit`` strategy."""

    values: tuple[int, ...] = Field(min_length=1)
    """The admissible k: odd, strictly increasing."""

    @field_validator("values")
    @classmethod
    def _odd_increasing(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        _check_odd(values)
        if list(values) != sorted(set(values)):
            raise ValueError("the k must be strictly increasing")
        return values


@K_STRATEGIES.register("explicit", summary="a given list of odd k", params_type=ExplicitParams)
def explicit(class_sizes: tuple[int, ...], params: ExplicitParams, /) -> AdmissibleK:
    """Admissible k given explicitly."""
    return AdmissibleK(params.values, "explicit", class_sizes)


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
def k_range(class_sizes: tuple[int, ...], params: RangeParams, /) -> AdmissibleK:
    """Admissible k from a range."""
    return AdmissibleK(
        tuple(range(params.start, params.stop + 1, params.step)), "range", class_sizes
    )


def admissible_k(
    name: str, params: dict[str, Any] | None, class_sizes: Sequence[int], neighbours: int
) -> AdmissibleK:
    """Resolve and apply a k strategy for a training sample.

    Args:
        name: Registry name of the strategy.
        params: Its parameters (validated here).
        class_sizes: |K1|, |K2|, … of the training sample.
        neighbours: Number of neighbours a training object has (m − 1).

    Raises:
        ModelUndefinedError: No k is admissible, or a k exceeds the number of neighbours.
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
