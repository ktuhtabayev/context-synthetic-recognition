"""R-IPS: forming the precedent base for training and testing (Steps 1–6).

Input: the objects E₀, an optional test sample T, the distances ρ; E₁ = E₀ ∖ T, the radii r(S)
and the sphere counts g(S) of :mod:`.hyperspheres`. Output: the result set D.

Step 2  Order E₁ by 1/r(Sᵢ) descending (r ascending); stop when every r in E₁ is negative.
Step 3  Sᵢ = the first object of the queue, Sᵢ ∈ Kₜ.
Step 4  L = {Sⱼ ∈ E₁ ∩ Kₜ | ρ(Sⱼ, Sᵢ) < r(Sⱼ) and g(Sⱼ) > 1} — Sᵢ itself counts (ρ = 0); block
        Sᵢ: r(Sᵢ) = −r(Sᵢ). If |L| = 1, Sᵢ moves from E₁ to D.
Step 5  If |L| > 1, g(Sⱼ) = g(Sⱼ) − 1 for every Sⱼ ∈ L (ascending index); Sⱼ moves from E₁ to D when
        g(Sⱼ) = 1. Back to Step 2.
Step 6  Output D.

Switches (:class:`RIPSSettings`):

- ``keep_blocked_cover`` (⚠ not the article's text when on): blocking makes r(Sⱼ) negative, so
  read literally a blocked object covers nobody in Step 4 (default). On, a blocked object keeps
  covering with |r(Sⱼ)|.
- ``training`` (⚠ the article only outputs D): ``remaining`` — the objects left in E₁ form the
  training sample and the rest of E₀ the test sample (default); ``moved`` — the other way round.

Where the article is silent the conventions are: |L| = 0 leaves Sᵢ blocked in E₁; equal radii are
queued by the smaller index; an object with r = 0 never enters the queue. Every iteration blocks one
object of the queue, so the algorithm stops after at most |E₁| iterations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.core.arrays import BoolArray, IntArray, freeze_fields
from context_synthetic_recognition.core.rips.hyperspheres import Hyperspheres, hyperspheres

logger = logging.getLogger(__name__)


class TrainingRole(StrEnum):
    """Which set of R-IPS is the training sample."""

    REMAINING = "remaining"
    """The objects left in E₁ train; D and T test (default)."""
    MOVED = "moved"
    """⚠ The objects moved to D train; the rest of E₁ and T test."""


@dataclass(frozen=True)
class RIPSSettings:
    """The switches of R-IPS."""

    training: TrainingRole = TrainingRole.REMAINING
    """Which set is the training sample."""
    keep_blocked_cover: bool = False
    """⚠ Blocked objects keep covering with |r| (off: the article's text)."""

    def __post_init__(self) -> None:
        """Accept the role as text."""
        object.__setattr__(self, "training", TrainingRole(self.training))


class Outcome(StrEnum):
    """What Steps 4–5 did with the selected object."""

    MOVED = "|L| = 1: Sᵢ moves to D"
    CORRECTED = "|L| > 1: the counters of L are corrected"
    BLOCKED = "|L| = 0: Sᵢ is only blocked"


@dataclass(frozen=True)
class Correction:
    """Step 5: g(Sⱼ) before and after one correction."""

    object: int
    before: int
    after: int

    @property
    def moved(self) -> bool:
        """Whether Sⱼ moved to D (its count fell to 1)."""
        return self.after == 1


@dataclass(frozen=True)
class RIPSIteration:
    """One pass through Steps 2–5."""

    number: int
    """1-based iteration."""
    selected: int
    """Sᵢ (0-based index of E₀)."""
    radius: float
    """r(Sᵢ) when it was selected (> 0)."""
    neighbourhood: tuple[int, ...]
    """L, ascending index."""
    outcome: Outcome
    corrections: tuple[Correction, ...] = ()
    """Step 5, ascending index."""
    moved: tuple[int, ...] = ()
    """Objects moved to D in this iteration."""


@dataclass(frozen=True, eq=False)
class RIPSResult:
    """The result of R-IPS: D, what remains of E₁, and the trace."""

    spheres: Hyperspheres
    """Steps 1–2: r, g, 1/r and the queue."""
    settings: RIPSSettings
    test_sample: IntArray
    """T (ascending)."""
    moved: IntArray
    """D — the objects moved out of E₁ (ascending)."""
    remaining: IntArray
    """E₁ after the run (ascending)."""
    iterations: tuple[RIPSIteration, ...] = field(repr=False)

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)

    @property
    def m(self) -> int:
        """|E₀|."""
        return int(self.spheres.radius.size)

    @property
    def training(self) -> IntArray:
        """The training sample (by :attr:`RIPSSettings.training`)."""
        if self.settings.training is TrainingRole.REMAINING:
            return self.remaining
        return self.moved

    @property
    def test(self) -> IntArray:
        """The test sample: E₀ without the training sample (T included)."""
        mask = np.ones(self.m, dtype=bool)
        mask[self.training] = False
        return np.flatnonzero(mask).astype(np.int64)

    @property
    def training_mask(self) -> BoolArray:
        """True for the objects of the training sample."""
        mask = np.zeros(self.m, dtype=bool)
        mask[self.training] = True
        return mask

    @property
    def order(self) -> tuple[int, ...]:
        """The objects in the order Step 3 selected them."""
        return tuple(it.selected for it in self.iterations)


def run_rips(
    distances: npt.ArrayLike,
    class_index: npt.ArrayLike,
    *,
    test_sample: npt.ArrayLike | None = None,
    settings: RIPSSettings | None = None,
    decimals: int = 10,
) -> RIPSResult:
    """R-IPS, Steps 1–6, on a distance matrix of E₀.

    Args:
        distances: ρ between the objects of E₀, shape (m, m).
        class_index: 0-based class of every object.
        test_sample: T, 0-based indices of E₀ (``None``: empty).
        settings: The switches; the article's text by default.
        decimals: Rounding of 1/r.

    Raises:
        ModelUndefinedError: E₁ lacks a class.
        ValueError: The matrix is not square or a test index is out of range.
    """
    chosen = settings or RIPSSettings()
    D = np.asarray(distances, dtype=np.float64)
    y = np.asarray(class_index, dtype=np.int64)
    m = y.size
    if D.shape != (m, m):
        raise ValueError(f"the distances must be ({m} × {m}), got {D.shape}")
    T = np.unique(np.asarray([] if test_sample is None else test_sample, dtype=np.int64))
    if T.size and (T.min() < 0 or T.max() >= m):
        raise ValueError(f"test indices must lie in 0 … {m - 1}")
    in_e1 = np.ones(m, dtype=bool)
    in_e1[T] = False
    spheres = hyperspheres(D, y, np.flatnonzero(in_e1), decimals=decimals)
    r = np.where(in_e1, spheres.radius, 0.0)
    g = spheres.sphere_count.copy()
    queue = spheres.queue
    cursor = 0
    iterations: list[RIPSIteration] = []

    while True:
        # Step 2 — the first object of the fixed order that is still in E₁ and not blocked
        while cursor < queue.size and not (in_e1[queue[cursor]] and r[queue[cursor]] > 0.0):
            cursor += 1
        if cursor >= queue.size:
            break  # Step 6: every radius left in E₁ is negative (or 0)
        i = int(queue[cursor])  # Step 3

        # Step 4 — L before blocking; the radius of Sⱼ as read literally or kept for blocked objects
        cover = np.abs(r) if chosen.keep_blocked_cover else r.copy()
        candidates = in_e1 & (y == y[i]) & (D[:, i] < cover) & (g > 1)
        L = [int(j) for j in np.flatnonzero(candidates)]
        r_selected = float(r[i])
        r[i] = -r_selected
        corrections: list[Correction] = []
        moved: list[int] = []
        if len(L) > 1:  # Step 5
            outcome = Outcome.CORRECTED
            for j in L:
                before = int(g[j])
                g[j] = before - 1
                corrections.append(Correction(j, before, int(g[j])))
                if g[j] == 1:
                    in_e1[j] = False
                    moved.append(j)
        elif len(L) == 1:
            outcome = Outcome.MOVED
            in_e1[i] = False
            moved.append(i)
        else:
            outcome = Outcome.BLOCKED
        iterations.append(
            RIPSIteration(
                number=len(iterations) + 1,
                selected=i,
                radius=r_selected,
                neighbourhood=tuple(L),
                outcome=outcome,
                corrections=tuple(corrections),
                moved=tuple(moved),
            )
        )

    moved_set = np.flatnonzero(~in_e1 & ~np.isin(np.arange(m), T)).astype(np.int64)
    remaining = np.flatnonzero(in_e1).astype(np.int64)
    logger.info(
        "R-IPS finished",
        extra={
            "iterations": len(iterations),
            "moved": int(moved_set.size),
            "remaining": int(remaining.size),
            "test_sample": int(T.size),
        },
    )
    return RIPSResult(
        spheres=spheres,
        settings=chosen,
        test_sample=T,
        moved=moved_set,
        remaining=remaining,
        iterations=tuple(iterations),
    )
