"""Evaluation metrics, AUC/ROC and margins — the workbook's conventions."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from context_synthetic_recognition.core.model import CSModel
from context_synthetic_recognition.evaluation.margins import margin, margin_analysis
from context_synthetic_recognition.evaluation.metrics import classification_metrics
from context_synthetic_recognition.evaluation.roc import oriented_scores, roc_curve

TRUTH = [2, 1, 2, 1, 1, 1, 2, 2, 2, 2]  # Heart-Disease (10, 13, 2)
RESUB = [1, 2, 2, 1, 1, 2, 2, 2, 2, 2]
LOO = [1, 0, 0, 0, 0, 1, 2, 1, 1, 1]


def test_resubstitution_metrics_of_the_workbook() -> None:
    m = classification_metrics(TRUTH, RESUB)
    assert (m.tp, m.tn, m.fp, m.fn, m.refusals) == (2, 5, 1, 2, 0)
    assert (m.accuracy, m.coverage, m.accuracy_answered) == (0.7, 1.0, 0.7)
    assert m.label == "7 / 10"
    assert m.matrix.tolist() == [[2, 2, 0], [1, 5, 0]]
    k1, k2 = m.classes
    assert (k1.predicted, k1.correct, k1.precision, k1.recall) == (3, 2, 2 / 3, 0.5)
    assert k1.f1 == pytest.approx(0.571428571428572)
    assert m.macro_f1 == pytest.approx(0.67032967032967)
    assert (k2.precision, k2.recall) == (5 / 7, 5 / 6)


def test_leave_one_out_metrics_of_the_workbook() -> None:
    m = classification_metrics(TRUTH, LOO)
    # refusals are neither TP, TN, FP nor FN, and count as errors in the accuracy
    assert (m.tp, m.tn, m.fp, m.fn, m.refusals) == (1, 1, 4, 0, 4)
    assert (m.accuracy, m.coverage) == (0.2, 0.6)
    assert m.accuracy_answered == pytest.approx(1 / 3)
    assert m.matrix.tolist() == [[1, 0, 3], [4, 1, 1]]
    assert m.macro_precision == pytest.approx(0.6)
    assert m.macro_recall == pytest.approx(0.208333333333333)


def test_k2_as_the_positive_class() -> None:
    m = classification_metrics(TRUTH, RESUB, positive=2)
    assert (m.tp, m.tn, m.fp, m.fn) == (5, 2, 2, 1)
    assert m.accuracy == 0.7


def test_edge_cases() -> None:
    all_refused = classification_metrics([1, 2], [0, 0])
    assert all_refused.accuracy_answered is None
    assert all_refused.classes[0].precision == 0.0
    assert all_refused.classes[0].f1 == 0.0
    with pytest.raises(ValueError, match="same length"):
        classification_metrics([1, 2], [1])
    with pytest.raises(ValueError, match="true classes"):
        classification_metrics([1, 3], [1, 1])
    with pytest.raises(ValueError, match="decisions must be"):
        classification_metrics([1, 2], [1, 5])
    with pytest.raises(ValueError, match="positive class"):
        classification_metrics([1, 2], [1, 2], positive=0)


# ---------------------------------------------------------------- AUC and ROC

RESUB_SCORES = [
    0.5,
    0.5 - 2 / 3,
    0 - 1 / 6,
    0.5,
    0.5,
    0.5 - 2 / 3,
    0.5 - 2 / 3,
    0.5 - 2 / 3,
    0.5 - 2 / 3,
    0.5 - 2 / 3,
]


def test_the_resubstitution_auc_needs_rounded_scores() -> None:
    curve = roc_curve(RESUB_SCORES, TRUTH)
    assert curve.auc == pytest.approx(0.666666666666667)
    # 0.5 − 2/3 and 0 − 1/6 tie only after rounding to 10 decimals
    assert curve.pairs_won[[1, 3, 4, 5]].tolist() == [2.5, 5.5, 5.5, 2.5]
    assert np.isnan(curve.pairs_won[0])
    unrounded = roc_curve(RESUB_SCORES, TRUTH, decimals=17)
    assert unrounded.auc != curve.auc


def test_the_roc_table() -> None:
    curve = roc_curve(RESUB_SCORES, TRUTH)
    assert curve.thresholds[0] == np.inf
    assert curve.thresholds[1:4].tolist() == [0.5, 0.5, 0.5]
    assert curve.tpr[:5].tolist() == [0.0, 0.5, 0.5, 0.5, 1.0]
    assert curve.fpr[1] == pytest.approx(1 / 6)
    assert curve.fpr[-1] == 1.0


def test_k2_positive_negates_the_score() -> None:
    k1 = roc_curve(RESUB_SCORES, TRUTH)
    k2 = roc_curve(RESUB_SCORES, TRUTH, positive=2)
    # ranking K2 above K1 by −score is the same statement as K1 above K2 by the score
    assert k2.auc == pytest.approx(k1.auc)
    assert np.isnan(k2.pairs_won[1])  # S₂ ∈ K1 is now a negative
    assert not np.isnan(k2.pairs_won[0])
    assert oriented_scores([0.25], positive=2).tolist() == [-0.25]


def test_one_class_gives_nan() -> None:
    assert np.isnan(roc_curve([0.1, 0.2], [1, 1]).auc)
    with pytest.raises(ValueError, match="same length"):
        roc_curve([0.1], [1, 2])


@given(
    st.lists(
        st.tuples(st.sampled_from([-1.0, -0.5, 0.0, 0.25, 0.5, 1.0]), st.sampled_from([1, 2])),
        min_size=2,
        max_size=30,
    )
)
def test_auc_is_the_mann_whitney_statistic(pairs: list[tuple[float, int]]) -> None:
    scores = [s for s, _ in pairs]
    truth = [t for _, t in pairs]
    positives = [s for s, t in pairs if t == 1]
    negatives = [s for s, t in pairs if t == 2]
    curve = roc_curve(scores, truth)
    if not positives or not negatives:
        assert np.isnan(curve.auc)
        return
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    assert curve.auc == pytest.approx(wins / (len(positives) * len(negatives)))
    assert np.all(np.diff(curve.tpr) >= 0)
    assert np.all(np.diff(curve.fpr) >= 0)


# ---------------------------------------------------------------- margins


def test_the_margin_analysis_of_the_workbook(experiment_cs_model: CSModel) -> None:
    analysis = margin_analysis(experiment_cs_model)
    widths = [m.width for m in analysis.with_majorizer]
    assert widths == pytest.approx(
        [0.148177267098285, 0.649596904180235, 1.12009851552738, 1.57431737159627]
    )
    assert [m.width for m in analysis.without_majorizer] == pytest.approx([-0.4] * 4)
    assert analysis.gains[0] == pytest.approx(0.548177267098285)
    r1 = analysis.with_majorizer[0]
    assert r1.boundary == pytest.approx(0.186150880520446)
    assert (r1.left_object, r1.right_object) == (0, 1)  # S₁ (max K2), S₂ (min K1)
    assert r1.correct == 10
    assert r1.object_margins[0] == pytest.approx(0.0740886335491426)
    assert r1.predictions.tolist() == TRUTH
    assert analysis.without_majorizer[0].correct == 7


def test_margin_edge_cases() -> None:
    m = margin([1.0, -1.0, 0.0], [0, 1, 1])
    assert (m.left, m.right, m.boundary, m.width) == (0.0, 1.0, 0.5, 1.0)
    assert m.sign.tolist() == [1.0, -1.0, -1.0]
    with pytest.raises(ValueError, match="both classes"):
        margin([1.0, 2.0], [0, 0])
