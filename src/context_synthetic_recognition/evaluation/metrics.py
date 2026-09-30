"""Classification metrics with the workbook's conventions.

Sheets *Accuracy*, *Confusion Matrix* and *Precision, Recall, F1 Score*. Decisions and true
classes use the article's codes: 1 = K1, 2 = K2, 0 = refusal (ADR-027).

- With a positive class P and the other class N: TP = predicted P and true P, TN = predicted N
  and true N, FP = predicted P and true N, FN = predicted N and true P. A refusal is none of
  them: it is counted separately and is an error in the accuracy (ADR-008).
- accuracy = (TP + TN)/n, coverage = (n − refusals)/n, accuracy on the answered objects
  = (TP + TN)/(n − refusals).
- Per class c: precision = correct c / predicted c (0 if nothing was predicted as c), recall =
  correct c / actual c, F1 = 2·P·R/(P + R) (0 if P + R = 0); the macro averages are the means over
  the two classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.core.arrays import IntArray, freeze_fields
from context_synthetic_recognition.core.meta import K1_DECISION, K2_DECISION, REFUSAL

CODES = (K1_DECISION, K2_DECISION)
"""The two class codes, in class order."""


@dataclass(frozen=True)
class ClassScores:
    """Precision, recall and F1 of one class."""

    code: int
    """1 (K1) or 2 (K2)."""
    predicted: int
    """Objects predicted as this class."""
    correct: int
    """Objects of this class predicted as this class."""
    actual: int
    """Objects of this class."""
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, eq=False)
class ClassificationMetrics:
    """Confusion counts and the derived metrics of one set of decisions."""

    positive: int
    """Code of the positive class (1 by default, ``evaluation.positive_class``)."""
    tp: int
    tn: int
    fp: int
    fn: int
    refusals: int
    matrix: IntArray
    """(2 × 3) actual class (K1, K2) × predicted (K1, K2, refusal) — sheet *Confusion Matrix*."""
    classes: tuple[ClassScores, ClassScores]
    """Precision, recall, F1 of K1 and K2."""

    def __post_init__(self) -> None:
        """Freeze the matrix."""
        freeze_fields(self)

    @property
    def n(self) -> int:
        """Number of decisions."""
        return int(self.matrix.sum())

    @property
    def correct(self) -> int:
        """TP + TN."""
        return self.tp + self.tn

    @property
    def accuracy(self) -> float:
        """(TP + TN)/n; a refusal is an error."""
        return self.correct / self.n

    @property
    def coverage(self) -> float:
        """Share of objects that received a class."""
        return (self.n - self.refusals) / self.n

    @property
    def accuracy_answered(self) -> float | None:
        """(TP + TN)/(n − refusals); ``None`` if every object was refused."""
        answered = self.n - self.refusals
        return self.correct / answered if answered else None

    @property
    def macro_precision(self) -> float:
        """Mean precision of the two classes."""
        return (self.classes[0].precision + self.classes[1].precision) / 2

    @property
    def macro_recall(self) -> float:
        """Mean recall of the two classes."""
        return (self.classes[0].recall + self.classes[1].recall) / 2

    @property
    def macro_f1(self) -> float:
        """Mean F1 of the two classes."""
        return (self.classes[0].f1 + self.classes[1].f1) / 2

    @property
    def label(self) -> str:
        """``7 / 10`` — as the workbook shows the correct decisions."""
        return f"{self.correct} / {self.n}"


def _class_scores(truth: IntArray, decisions: IntArray, code: int) -> ClassScores:
    predicted = int(np.count_nonzero(decisions == code))
    correct = int(np.count_nonzero((decisions == code) & (truth == code)))
    actual = int(np.count_nonzero(truth == code))
    precision = correct / predicted if predicted else 0.0
    recall = correct / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ClassScores(code, predicted, correct, actual, precision, recall, f1)


def classification_metrics(
    truth: npt.ArrayLike, decisions: npt.ArrayLike, positive: int = K1_DECISION
) -> ClassificationMetrics:
    """Confusion counts, accuracy, coverage, precision, recall and F1 of a set of decisions.

    Args:
        truth: True class codes (1 = K1, 2 = K2).
        decisions: Decisions (1, 2 or 0 = refusal).
        positive: Code of the positive class.

    Raises:
        ValueError: Unknown codes, different lengths or no decisions.
    """
    t = np.asarray(truth, dtype=np.int64)
    d = np.asarray(decisions, dtype=np.int64)
    if t.shape != d.shape or t.ndim != 1 or t.size == 0:
        raise ValueError("truth and decisions must be non-empty vectors of the same length")
    if not np.isin(t, CODES).all():
        raise ValueError("true classes must be the codes 1 (K1) or 2 (K2)")
    if not np.isin(d, (*CODES, REFUSAL)).all():
        raise ValueError("decisions must be 1 (K1), 2 (K2) or 0 (refusal)")
    if positive not in CODES:
        raise ValueError("the positive class must be 1 (K1) or 2 (K2)")
    negative = K2_DECISION if positive == K1_DECISION else K1_DECISION
    matrix = np.array(
        [[np.count_nonzero((t == c) & (d == p)) for p in (*CODES, REFUSAL)] for c in CODES],
        dtype=np.int64,
    )
    return ClassificationMetrics(
        positive=positive,
        tp=int(np.count_nonzero((d == positive) & (t == positive))),
        tn=int(np.count_nonzero((d == negative) & (t == negative))),
        fp=int(np.count_nonzero((d == positive) & (t == negative))),
        fn=int(np.count_nonzero((d == negative) & (t == positive))),
        refusals=int(np.count_nonzero(d == REFUSAL)),
        matrix=matrix,
        classes=(_class_scores(t, d, K1_DECISION), _class_scores(t, d, K2_DECISION)),
    )
