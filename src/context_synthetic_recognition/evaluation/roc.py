"""AUC and the ROC curve of the meta-algorithm's score (sheet *ROC Curve & AUC*).

The score of an object is score₁ − score₂ = |B1(a_p)|/|K1| − |B2(a_p)|/|K2| (Step 4); a larger
score points to K1. Scores are rounded to ``evaluation.score_decimals`` (10) before they are
compared, so that mathematically equal scores tie exactly (e.g. 0.5 − 2/3 and 0 − 1/6, ADR-008).

- AUC is the Mann–Whitney statistic: the share of (positive, negative) pairs in which the positive
  object has the larger score, ties counting ½.
- The ROC table lists, for every score in decreasing order (repeated scores repeated, as the
  workbook's ``LARGE`` column), TPR = positives with score ≥ threshold / positives and FPR =
  negatives with score ≥ threshold / negatives, after the point (+∞, 0, 0).

With K2 as the positive class the score is negated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.core.arrays import FloatArray, freeze_fields
from context_synthetic_recognition.core.meta import K1_DECISION

DEFAULT_DECIMALS = 10


def oriented_scores(
    scores: npt.ArrayLike, positive: int = K1_DECISION, decimals: int = DEFAULT_DECIMALS
) -> FloatArray:
    """Scores rounded to ``decimals`` and negated when K2 is the positive class."""
    values = np.round(np.asarray(scores, dtype=np.float64), decimals)
    return values if positive == K1_DECISION else -values


@dataclass(frozen=True, eq=False)
class RocCurve:
    """AUC, the pairs each positive object wins and the ROC points."""

    auc: float
    """Mann–Whitney AUC (NaN if a class is missing)."""
    pairs_won: FloatArray
    """Per positive object: negatives with a smaller score (+½ per tie); NaN for negatives."""
    thresholds: FloatArray
    """+∞, then every score in decreasing order."""
    tpr: FloatArray
    """True-positive rate at each threshold (score ≥ threshold)."""
    fpr: FloatArray
    """False-positive rate at each threshold."""

    def __post_init__(self) -> None:
        """Freeze the arrays."""
        freeze_fields(self)


def roc_curve(
    scores: npt.ArrayLike,
    truth: npt.ArrayLike,
    positive: int = K1_DECISION,
    decimals: int = DEFAULT_DECIMALS,
) -> RocCurve:
    """AUC and ROC of ``scores`` against the true class codes ``truth`` (1 = K1, 2 = K2)."""
    s = oriented_scores(scores, positive, decimals)
    t = np.asarray(truth, dtype=np.int64)
    if s.shape != t.shape or s.ndim != 1:
        raise ValueError("scores and truth must be vectors of the same length")
    pos, neg = t == positive, t != positive
    negatives, positives = np.sort(s[neg]), np.sort(s[pos])
    below = np.searchsorted(negatives, s, side="left")  # negatives with a smaller score
    ties = np.searchsorted(negatives, s, side="right") - below
    won = np.where(pos, below + 0.5 * ties, np.nan)
    pairs = positives.size * negatives.size
    auc = float(np.nansum(won) / pairs) if pairs else float("nan")
    thresholds = np.sort(s)[::-1]
    tpr = (positives.size - np.searchsorted(positives, thresholds, side="left")) / max(
        1, positives.size
    )
    fpr = (negatives.size - np.searchsorted(negatives, thresholds, side="left")) / max(
        1, negatives.size
    )
    return RocCurve(
        auc=auc,
        pairs_won=won,
        thresholds=np.concatenate([[np.inf], thresholds]),
        tpr=np.concatenate([[0.0], tpr]),
        fpr=np.concatenate([[0.0], fpr]),
    )
