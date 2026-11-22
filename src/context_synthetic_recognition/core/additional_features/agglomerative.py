"""The hierarchical agglomerative algorithm, Steps 1–5: the additional (latent) features D(t).

Input: Ψ(r) as contribution values C (m × r, C[t, u] = ηᵤ(a_tu), formula (6)), the
informativeness ω of the synthetic features and the classes K1, K2.

Step 1  P = {a₁, …, a_r}, TUPLAM = ∅.
Step 2  u = argmax ωⱼ (the first feature on ties); TUPLAM = {u}; P = P ∖ {u}; R(Sₜ) = ηᵤ(a_tu).
Step 3  For every candidate u ∈ P, in feature order: bₜ = R(Sₜ) + ηᵤ(a_tu); the majorizing function
        bₜ ← bₜ ± α·φ(−bₜ) (+ for K1, − for K2); class means M₁, M₂;
        θ = Σₜ |bₜ − M(own class)|, γ = Σₜ |bₜ − M(other class)|.
        cr1 = min θ/γ over the candidates with θ/γ < cr1₀ (the first one on ties), q = argmin.
Step 4  TUPLAM ← TUPLAM ∪ {q}, P ← P ∖ {q}, crit = cr1; R ← R + η_q with the majorizing
        function — the additional feature dⱼ = R after iteration j.
Step 5  Back to Step 3 while |TUPLAM| < ϰ, crit > δ and P ≠ ∅; otherwise output TUPLAM.

Two switches deviate from the article when on (ADR-002, ADR-003, ADR-062); the Excel experiment
workbook uses both:

- ``class_means = running`` (⚠): θ and γ are measured from running partial class means (the sum of
  the rows up to t divided by |Kᵢ|) instead of the final class means M₁ and M₂ (``final``);
- ``majorizing_passes = 2`` (⚠): in Step 4 the majorizing function is applied again to the
  already-majorized bₜ of Step 3 (``1``: R ← R + η_q, then one pass).

Conventions where the article is silent (ADR-008, ADR-024, ADR-025): θ/γ is +∞ when γ = 0 (the
candidate is never chosen); if no candidate has θ/γ < cr1₀ the algorithm stops without adding a
feature; all sums run over the objects in their original order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.config.models import AdditionalFeaturesConfig, ClassMeans
from context_synthetic_recognition.core.additional_features.majorizing_functions import (
    MAJORIZING_FUNCTIONS,
    majorize,
)
from context_synthetic_recognition.core.arrays import (
    BoolArray,
    FloatArray,
    IntArray,
    as_float_matrix,
    freeze_fields,
)
from context_synthetic_recognition.errors import ModelUndefinedError
from context_synthetic_recognition.notation import synthetic_name

logger = logging.getLogger(__name__)

BLOCK_ELEMENTS = 1 << 22
"""Step 3 evaluates the candidates in blocks of at most this many (object, candidate) values."""


class StopReason(StrEnum):
    """Why the grouping stopped (Step 5)."""

    KAPPA = "kappa"
    """|TUPLAM| reached ϰ."""
    DELTA = "delta"
    """crit ≤ δ."""
    EXHAUSTED = "exhausted"
    """No candidate is left (P = ∅), e.g. r = 1."""
    NO_CANDIDATE = "no-candidate"
    """No candidate has θ/γ < cr1₀, so Step 3 chose nothing."""

    @property
    def text(self) -> str:
        """The stopping condition in the article's notation."""
        return _STOP_TEXT[self]


_STOP_TEXT = {
    StopReason.KAPPA: "|TUPLAM| = ϰ",
    StopReason.DELTA: "crit ≤ δ",
    StopReason.EXHAUSTED: "P = ∅",
    StopReason.NO_CANDIDATE: "no θ/γ < cr1₀",
}


@dataclass(frozen=True)
class AgglomerativeSettings:
    """The parameters of one fit, with the majorizing function resolved and validated."""

    alpha: float
    """α — the margin regularization parameter, 0 < α < 1."""
    delta: float
    """δ — threshold for crit."""
    kappa: int
    """ϰ — maximum |TUPLAM|."""
    cr1: float
    """cr1₀ — initial value of the criterion in Step 3."""
    majorizing_function: str
    """Canonical registry name of φ."""
    majorizing_function_params: Any
    """Its validated parameters."""
    class_means: ClassMeans
    """Class means of θ and γ: final (the article) or running (⚠ ADR-002)."""
    majorizing_passes: int
    """Majorizing passes in Step 4: 1 (the article) or 2 (⚠ ADR-003)."""

    @classmethod
    def from_config(cls, config: AdditionalFeaturesConfig) -> AgglomerativeSettings:
        """Resolve the majorizing function of ``config`` and validate its parameters.

        Raises:
            RegistryError: Unknown majorizing function.
            ConfigError: Invalid majorizing function parameters.
        """
        spec = config.majorizing_function
        return cls(
            alpha=config.alpha,
            delta=config.delta,
            kappa=config.kappa,
            cr1=config.cr1,
            majorizing_function=MAJORIZING_FUNCTIONS.info(spec.name).name,
            majorizing_function_params=MAJORIZING_FUNCTIONS.make_params(spec.name, spec.params),
            class_means=ClassMeans(config.class_means),
            majorizing_passes=config.majorizing_passes,
        )


@dataclass(frozen=True, eq=False)
class CandidateScan:
    """Step 3 for one candidate — the columns of a candidate block on an *Additional Feature* sheet.

    Built by :meth:`AgglomerativeResult.scan`, with the same computation as the grouping itself, so
    its θ, γ and θ/γ are bit-identical to those of the iteration.
    """

    feature: int
    """The candidate u (0-based)."""
    entering: FloatArray
    """(m,) R(Sₜ) entering the iteration (column B)."""
    eta: FloatArray
    """(m,) ηᵤ(a_tu) (column D)."""
    b: FloatArray
    """(m,) bₜ = R(Sₜ) + ηᵤ(a_tu) (column F)."""
    majorized: FloatArray
    """(m,) bₜ after one majorizing function pass (column H)."""
    sum1: FloatArray
    """(m,) running Σ of the majorized bₜ over the K1 objects up to row t (column I)."""
    sum2: FloatArray
    """(m,) the same over K2 (column J)."""
    centre1: FloatArray
    """(m,) M₁ used at row t: the final mean (the article) or the running Σ/|K1| (⚠) (column K)."""
    centre2: FloatArray
    """(m,) M₂ used at row t (column L)."""
    to_own: FloatArray
    """(m,) |bₜ − M(own class)| (column M)."""
    to_other: FloatArray
    """(m,) |bₜ − M(other class)| (column N)."""
    theta: float
    """θ = Σ |bₜ − M(own)| — intra-class compactness."""
    gamma: float
    """γ = Σ |bₜ − M(other)| — inter-class separability."""
    ratio: float
    """θ/γ; +∞ when γ = 0 (ADR-025)."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)

    @property
    def name(self) -> str:
        """aᵤ (1-based)."""
        return synthetic_name(self.feature)


@dataclass(frozen=True, eq=False)
class AgglomerativeIteration:
    """One pass of Steps 3–5 (one *Additional Feature* sheet of the Excel mirror)."""

    number: int
    """j = 1, 2, … (1-based, as the additional feature d_j it produces)."""
    entering: FloatArray
    """(m,) R(Sₜ) entering the iteration."""
    tuplam_before: tuple[int, ...]
    """TUPLAM before the iteration (0-based features, in TUPLAM order)."""
    crit_before: float
    """crit before the iteration (cr1₀ before the first one)."""
    candidates: IntArray
    """P: the candidates of Step 3, in feature order."""
    theta: FloatArray
    """θ of every candidate."""
    gamma: FloatArray
    """γ of every candidate."""
    ratio: FloatArray
    """θ/γ of every candidate (+∞ when γ = 0)."""
    cr1: float
    """min θ/γ if it is below cr1₀, otherwise cr1₀ (the workbook's cell "cr1 = min θ/γ")."""
    q: int | None
    """The feature added in Step 4, or ``None`` if no θ/γ < cr1₀."""
    crit: float
    """crit after the iteration: cr1 if a feature was added, otherwise ``crit_before``."""
    once: FloatArray | None
    """(m,) R + η_q with one majorizing function pass (= the Step 3 value of q)."""
    additional: FloatArray | None
    """(m,) the additional feature d_j = new R(Sₜ) (one or two passes, ADR-003)."""
    go_on: bool
    """Whether the grouping continues with another iteration."""
    stop: StopReason | None
    """Why it stops after this iteration (``None`` if it continues)."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)

    @property
    def tuplam_after(self) -> tuple[int, ...]:
        """TUPLAM after the iteration."""
        return self.tuplam_before if self.q is None else (*self.tuplam_before, self.q)

    @property
    def available(self) -> BoolArray:
        """(r,) whether each synthetic feature is a candidate of this iteration."""
        mask = np.zeros(len(self.tuplam_before) + self.candidates.size, dtype=bool)
        mask[self.candidates] = True
        return mask

    def ratio_of(self, feature: int) -> float | None:
        """θ/γ of ``feature`` in this iteration, or ``None`` if it is not a candidate."""
        where = np.flatnonzero(self.candidates == feature)
        return float(self.ratio[where[0]]) if where.size else None


@dataclass(frozen=True, eq=False)
class AgglomerativeResult:
    """The grouping with its full trace: TUPLAM, crit, the additional features d₁ … d_p."""

    settings: AgglomerativeSettings
    contributions: FloatArray
    """(m × r) the input: ηᵤ(a_tu) of every object and synthetic feature."""
    informativeness: FloatArray
    """(r,) the informativeness ω of Step 2."""
    class_index: IntArray
    """(m,) 0 = K1, 1 = K2."""
    class_sizes: tuple[int, int]
    """|K1|, |K2|."""
    first: int
    """u of Step 2 (0-based)."""
    tuplam: tuple[int, ...]
    """TUPLAM in selection order (0-based features)."""
    iterations: tuple[AgglomerativeIteration, ...]
    """Every executed iteration of Steps 3–5."""
    stop: StopReason
    """Why the grouping stopped."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)

    @property
    def p(self) -> int:
        """Number of additional features p = |TUPLAM| − 1."""
        return len(self.tuplam) - 1

    @property
    def r(self) -> int:
        """Number of synthetic features."""
        return int(self.contributions.shape[1])

    @property
    def m(self) -> int:
        """Number of objects."""
        return int(self.contributions.shape[0])

    @property
    def additional(self) -> FloatArray:
        """(m × p) the additional features d₁ … d_p (sheet *Dataset for Meta-algorithm*)."""
        columns = [it.additional for it in self.iterations if it.additional is not None]
        if not columns:
            return np.empty((self.m, 0), dtype=np.float64)
        return np.column_stack(columns)

    @property
    def crit(self) -> tuple[float, ...]:
        """Crit after every iteration that added a feature."""
        return tuple(it.crit for it in self.iterations if it.q is not None)

    @property
    def names(self) -> tuple[str, ...]:
        """TUPLAM as names (a₆, a₃, …)."""
        return tuple(synthetic_name(u) for u in self.tuplam)

    @property
    def label(self) -> str:
        """``{a₆, a₃, a₁}`` — as on the workbook."""
        return "{" + ", ".join(self.names) + "}"

    def scan(self, iteration: int, feature: int) -> CandidateScan:
        """Step 3 of ``feature`` in the iteration with 0-based index ``iteration``, in detail.

        ``feature`` may be any synthetic feature, also one that is not a candidate of that
        iteration (the workbook computes those blocks too, marked "in TUPLAM — skipped").
        """
        return self.scan_from(self.iterations[iteration].entering, feature)

    def scan_from(self, entering: npt.ArrayLike, feature: int) -> CandidateScan:
        """Step 3 of ``feature`` for any R(Sₜ) — the same computation as the grouping's."""
        R = np.array(entering, dtype=np.float64)
        if R.shape != (self.m,):
            raise ValueError(f"R(Sₜ) needs {self.m} values, got shape {R.shape}")
        in_k1, sign = _class_signs(self.class_index)
        scan = _step3(
            R,
            self.contributions[:, [feature]],
            in_k1,
            sign,
            self.class_sizes,
            self.settings,
        )
        return CandidateScan(
            feature=feature,
            entering=R,
            eta=self.contributions[:, feature].copy(),
            b=scan.b[:, 0],
            majorized=scan.majorized[:, 0],
            sum1=scan.sum1[:, 0],
            sum2=scan.sum2[:, 0],
            centre1=scan.centre1[:, 0],
            centre2=scan.centre2[:, 0],
            to_own=scan.to_own[:, 0],
            to_other=scan.to_other[:, 0],
            theta=float(scan.theta[0]),
            gamma=float(scan.gamma[0]),
            ratio=float(scan.ratio[0]),
        )


# ---------------------------------------------------------------- Step 3


@dataclass(frozen=True)
class _Step3:
    b: FloatArray
    majorized: FloatArray
    sum1: FloatArray
    sum2: FloatArray
    centre1: FloatArray
    centre2: FloatArray
    to_own: FloatArray
    to_other: FloatArray
    theta: FloatArray
    gamma: FloatArray
    ratio: FloatArray


def _class_signs(class_index: IntArray) -> tuple[BoolArray, FloatArray]:
    in_k1 = np.asarray(class_index) == 0
    return in_k1, np.where(in_k1, 1.0, -1.0)


def _running(terms: FloatArray) -> FloatArray:
    """Running sums down the objects, strictly left to right (the workbook's running cells)."""
    return np.cumsum(terms, axis=0)


def _step3(
    entering: FloatArray,
    eta: FloatArray,
    in_k1: BoolArray,
    sign: FloatArray,
    class_sizes: tuple[int, int],
    settings: AgglomerativeSettings,
) -> _Step3:
    """Step 3 for a block of candidates: ``eta`` is (m × candidates)."""
    majorizing_function = MAJORIZING_FUNCTIONS.get(settings.majorizing_function)
    k1 = in_k1[:, None]
    b = entering[:, None] + eta
    h = majorize(b, sign, settings.alpha, majorizing_function, settings.majorizing_function_params)
    sum1 = _running(np.where(k1, h, 0.0))
    sum2 = _running(np.where(k1, 0.0, h))
    if settings.class_means is ClassMeans.RUNNING:
        centre1, centre2 = sum1 / class_sizes[0], sum2 / class_sizes[1]
    else:
        centre1 = np.broadcast_to(sum1[-1] / class_sizes[0], h.shape).copy()
        centre2 = np.broadcast_to(sum2[-1] / class_sizes[1], h.shape).copy()
    to_own = np.abs(h - np.where(k1, centre1, centre2))
    to_other = np.abs(h - np.where(k1, centre2, centre1))
    theta = _running(to_own)[-1]
    gamma = _running(to_other)[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(gamma > 0.0, theta / gamma, np.inf)
    return _Step3(b, h, sum1, sum2, centre1, centre2, to_own, to_other, theta, gamma, ratio)


def _scan_candidates(
    entering: FloatArray,
    C: FloatArray,
    candidates: IntArray,
    in_k1: BoolArray,
    sign: FloatArray,
    class_sizes: tuple[int, int],
    settings: AgglomerativeSettings,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """θ, γ and θ/γ of every candidate, in blocks of at most :data:`BLOCK_ELEMENTS` values."""
    width = max(1, BLOCK_ELEMENTS // max(1, C.shape[0]))
    theta, gamma, ratio = (np.empty(candidates.size) for _ in range(3))
    for start in range(0, candidates.size, width):
        block = slice(start, start + width)
        scan = _step3(entering, C[:, candidates[block]], in_k1, sign, class_sizes, settings)
        theta[block], gamma[block], ratio[block] = scan.theta, scan.gamma, scan.ratio
    return theta, gamma, ratio


# ---------------------------------------------------------------- Steps 1–5


def _validated_inputs(
    contributions: npt.ArrayLike, informativeness: npt.ArrayLike, class_index: npt.ArrayLike
) -> tuple[FloatArray, FloatArray, IntArray, tuple[int, int]]:
    C = as_float_matrix(contributions, "contributions")
    w = np.array(informativeness, dtype=np.float64)
    y = np.array(class_index, dtype=np.int64)
    m, r = C.shape
    if w.shape != (r,):
        raise ValueError(
            f"expected {r} informativeness values (one per synthetic feature), got shape {w.shape}"
        )
    if y.shape != (m,):
        raise ValueError(f"expected {m} class indices, got shape {y.shape}")
    if r == 0:
        raise ModelUndefinedError(
            "the hierarchical agglomerative algorithm needs at least one synthetic feature (r = 0)"
        )
    if not (np.isfinite(C).all() and np.isfinite(w).all()):
        raise ValueError("contributions and informativeness must be finite")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("class indices must be 0 (K1) or 1 (K2)")
    sizes = (int(np.count_nonzero(y == 0)), int(np.count_nonzero(y == 1)))
    if min(sizes) == 0:
        raise ModelUndefinedError(
            "the agglomerative algorithm needs objects of both classes (the class means M₁, M₂ are "
            "undefined)"
        )
    return C, w, y, sizes


def agglomerative(
    contributions: npt.ArrayLike,
    informativeness: npt.ArrayLike,
    class_index: npt.ArrayLike,
    config: AdditionalFeaturesConfig | None = None,
) -> AgglomerativeResult:
    """Hierarchical agglomerative grouping, Steps 1–5: TUPLAM and the additional features d₁ … d_p.

    Args:
        contributions: (m × r) Ψ(r) as contribution values ηᵤ(a_tu) (formula (6)).
        informativeness: (r,) informativeness of the synthetic features (ω by formula (4) by
            default).
        class_index: (m,) 0 for K1, 1 for K2.
        config: α, δ, ϰ, cr1₀, the majorizing function and the two switches; the default
            configuration.

    Raises:
        ModelUndefinedError: No synthetic feature, or a class without objects.
        RegistryError: Unknown majorizing function.
        ConfigError: Invalid majorizing function parameters.
    """
    settings = AgglomerativeSettings.from_config(config or AdditionalFeaturesConfig())
    C, w, y, sizes = _validated_inputs(contributions, informativeness, class_index)
    in_k1, sign = _class_signs(y)
    majorizing_function = MAJORIZING_FUNCTIONS.get(settings.majorizing_function)

    # Steps 1–2 — the feature of the largest informativeness (the first one on ties)
    first = int(np.argmax(w))
    tuplam = [first]
    pool = [u for u in range(C.shape[1]) if u != first]
    R = C[:, first].copy()
    crit = settings.cr1
    iterations: list[AgglomerativeIteration] = []
    stop = StopReason.EXHAUSTED

    # Steps 3–5
    while pool:
        candidates = np.array(pool, dtype=np.int64)
        theta, gamma, ratio = _scan_candidates(R, C, candidates, in_k1, sign, sizes, settings)
        below = ratio < settings.cr1
        cr1 = float(ratio.min()) if below.any() else settings.cr1
        tuplam_before, crit_before = tuple(tuplam), crit
        q: int | None = None
        once = additional = None
        if below.any():
            q = int(candidates[np.argmin(np.where(below, ratio, np.inf))])
            chosen = _step3(R, C[:, [q]], in_k1, sign, sizes, settings)
            once = chosen.majorized[:, 0]
            if settings.majorizing_passes == 2:
                additional = majorize(
                    once,
                    sign,
                    settings.alpha,
                    majorizing_function,
                    settings.majorizing_function_params,
                )
            else:
                additional = once.copy()
            crit = cr1
            tuplam.append(q)
            pool.remove(q)
        go_on = (
            q is not None and len(tuplam) < settings.kappa and crit > settings.delta and bool(pool)
        )
        reason = None if go_on else _stop_reason(q, len(tuplam), crit, pool, settings)
        iterations.append(
            AgglomerativeIteration(
                number=len(iterations) + 1,
                entering=R,
                tuplam_before=tuplam_before,
                crit_before=crit_before,
                candidates=candidates,
                theta=theta,
                gamma=gamma,
                ratio=ratio,
                cr1=cr1,
                q=q,
                crit=crit,
                once=once,
                additional=additional,
                go_on=go_on,
                stop=reason,
            )
        )
        logger.debug(
            "hierarchical agglomerative algorithm iteration",
            extra={"j": len(iterations), "q": q, "crit": crit, "tuplam": list(tuplam)},
        )
        if reason is not None:
            stop = reason
            break
        assert additional is not None  # go_on implies that a feature was added
        R = additional

    return AgglomerativeResult(
        settings=settings,
        contributions=C,
        informativeness=w,
        class_index=y,
        class_sizes=sizes,
        first=first,
        tuplam=tuple(tuplam),
        iterations=tuple(iterations),
        stop=stop,
    )


def _stop_reason(
    q: int | None, size: int, crit: float, pool: list[int], settings: AgglomerativeSettings
) -> StopReason:
    if q is None:
        return StopReason.NO_CANDIDATE
    if size >= settings.kappa:
        return StopReason.KAPPA
    if crit <= settings.delta:
        return StopReason.DELTA
    return StopReason.EXHAUSTED
