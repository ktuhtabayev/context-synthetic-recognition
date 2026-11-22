"""Hierarchical agglomerative grouping, Steps 1–5: stopping rules, ties, γ = 0, switches, trace."""

import dataclasses

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from context_synthetic_recognition.config import AdditionalFeaturesConfig, ClassMeans, plugin
from context_synthetic_recognition.core.additional_features import (
    agglomerative as agglomerative_module,
)
from context_synthetic_recognition.core.additional_features.agglomerative import (
    StopReason,
    agglomerative,
)
from context_synthetic_recognition.core.additional_features.majorizing_functions import (
    majorize,
    sigmoid,
)
from context_synthetic_recognition.core.synthetic_features.psi import PsiOperator
from context_synthetic_recognition.errors import ModelUndefinedError, RegistryError

from .conftest import excel_experiment, workbook_additional

WORKBOOK = excel_experiment().additional_features
"""The workbook's settings: running class means, two majorizing passes."""
CRIT = [0.4281557866657403, 0.30527675846344937, 0.27336281326964995, 0.2710256546472449]


def experiment_inputs(model: PsiOperator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = model.trace
    return t.contributions, t.informativeness, t.class_index


def test_the_experiment_grouping(experiment_psi: PsiOperator) -> None:
    result = agglomerative(*experiment_inputs(experiment_psi), WORKBOOK)
    assert result.first == 5
    assert result.tuplam == (5, 2, 0, 1, 3)
    assert result.label == "{a₆, a₃, a₁, a₂, a₄}"
    assert list(result.crit) == CRIT
    assert result.stop is StopReason.KAPPA
    assert result.stop.text == "|TUPLAM| = ϰ"
    assert (result.p, result.r, result.m) == (4, 6, 10)
    assert result.additional.shape == (10, 4)
    assert [it.number for it in result.iterations] == [1, 2, 3, 4]
    assert [it.go_on for it in result.iterations] == [True, True, True, False]
    first = result.iterations[0]
    assert first.crit_before == 10.0
    assert first.candidates.tolist() == [0, 1, 2, 3, 4]
    assert first.tuplam_before == (5,)
    assert first.tuplam_after == (5, 2)
    assert first.available.tolist() == [True, True, True, True, True, False]
    assert first.ratio_of(5) is None
    assert first.ratio_of(2) == CRIT[0]
    assert first.cr1 == CRIT[0]
    assert np.array_equal(first.entering, experiment_psi.trace.contributions[:, 5])
    # d_j enters iteration j + 1
    assert np.array_equal(result.iterations[1].entering, first.additional)


def test_the_trace_is_immutable(experiment_psi: PsiOperator) -> None:
    result = agglomerative(*experiment_inputs(experiment_psi), WORKBOOK)
    with pytest.raises(ValueError, match="read-only"):
        result.iterations[0].ratio[0] = 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.first = 0  # type: ignore[misc]


def test_scans_are_bit_identical_to_the_grouping(experiment_psi: PsiOperator) -> None:
    result = agglomerative(*experiment_inputs(experiment_psi), WORKBOOK)
    for index, it in enumerate(result.iterations):
        for c, u in enumerate(it.candidates):
            scan = result.scan(index, int(u))
            assert (scan.theta, scan.gamma, scan.ratio) == (it.theta[c], it.gamma[c], it.ratio[c])
            assert scan.name == f"a{'₁₂₃₄₅₆'[int(u)]}"
        if it.q is not None and it.once is not None:
            chosen = result.scan(index, it.q)
            assert np.array_equal(chosen.majorized, it.once)
            assert np.array_equal(chosen.b, it.entering + result.contributions[:, it.q])


def test_scan_columns(experiment_psi: PsiOperator) -> None:
    result = agglomerative(*experiment_inputs(experiment_psi), WORKBOOK)
    scan = result.scan(0, 0)  # the workbook's first candidate block
    assert scan.b[0] == pytest.approx(0.316666666666667)
    assert scan.majorized[0] == pytest.approx(0.190220170417241)
    assert scan.sum2[-1] == pytest.approx(0.460880681668964)
    assert scan.centre1[1] == pytest.approx(0.110778290729023)
    assert (scan.theta, scan.gamma) == pytest.approx((1.90725378777931, 3.69353989148707))
    # a feature already in TUPLAM can be scanned too ("in TUPLAM — skipped" blocks)
    assert result.scan(0, 5).feature == 5
    with pytest.raises(ValueError, match="10 values"):
        result.scan_from(np.zeros(3), 0)


@pytest.mark.parametrize(
    ("class_means", "passes", "tuplam", "crit"),
    [
        ("running", 2, (5, 2, 0, 1, 3), CRIT),
        ("running", 1, (5, 2, 0, 1, 3), [0.4281557866657403, 0.3420036491761922]),
        ("final", 2, (5, 0, 1, 3, 4), [0.3096009047446551, 0.19932862369894389]),
        ("final", 1, (5, 0, 1, 3, 4), [0.3096009047446551, 0.26579419425411066]),
    ],
)
def test_the_four_switch_settings(
    experiment_psi: PsiOperator,
    class_means: str,
    passes: int,
    tuplam: tuple[int, ...],
    crit: list[float],
) -> None:
    config = workbook_additional(class_means=ClassMeans(class_means), majorizing_passes=passes)  # type: ignore[arg-type]
    result = agglomerative(*experiment_inputs(experiment_psi), config)
    assert result.tuplam == tuplam
    assert list(result.crit[: len(crit)]) == pytest.approx(crit, abs=1e-15)


def test_one_pass_takes_the_step_3_value_two_passes_majorize_it_again(
    experiment_psi: PsiOperator,
) -> None:
    once = agglomerative(
        *experiment_inputs(experiment_psi), workbook_additional(majorizing_passes=1)
    )
    twice = agglomerative(*experiment_inputs(experiment_psi), WORKBOOK)
    first_once, first_twice = once.iterations[0], twice.iterations[0]
    assert first_once.once is not None
    assert first_once.additional is not None
    assert np.array_equal(first_once.additional, first_once.once)
    assert first_twice.once is not None
    sign = np.where(experiment_psi.trace.class_index == 0, 1.0, -1.0)
    again = majorize(first_twice.once, sign, 0.3, sigmoid, None)
    assert np.array_equal(first_twice.additional, again)  # type: ignore[arg-type]


# ---------------------------------------------------------------- stopping rules (Step 5)


def test_a_single_synthetic_feature_needs_no_iteration() -> None:
    result = agglomerative(
        [[0.2], [-0.1], [0.3]], [0.8], [0, 1, 1], excel_experiment().additional_features
    )
    assert result.tuplam == (0,)
    assert result.iterations == ()
    assert result.stop is StopReason.EXHAUSTED
    assert result.p == 0
    assert result.additional.shape == (3, 0)
    assert result.crit == ()


def test_kappa_2_stops_after_one_iteration(experiment_psi: PsiOperator) -> None:
    result = agglomerative(*experiment_inputs(experiment_psi), workbook_additional(kappa=2))
    assert result.tuplam == (5, 2)
    assert result.stop is StopReason.KAPPA


def test_a_large_delta_stops_early(experiment_psi: PsiOperator) -> None:
    result = agglomerative(*experiment_inputs(experiment_psi), workbook_additional(delta=0.45))
    assert result.tuplam == (5, 2)  # crit 0.428 ≤ δ
    assert result.stop is StopReason.DELTA
    assert result.stop.text == "crit ≤ δ"


def test_no_candidate_below_cr1(experiment_psi: PsiOperator) -> None:
    result = agglomerative(*experiment_inputs(experiment_psi), workbook_additional(cr1=0.4))
    assert result.tuplam == (5,)
    assert result.stop is StopReason.NO_CANDIDATE
    it = result.iterations[0]
    assert it.q is None
    assert it.additional is None
    assert it.once is None
    assert it.cr1 == 0.4  # min θ/γ = 0.428 is not below cr1₀ (the workbook's J111)
    assert it.crit == it.crit_before == 0.4
    assert result.p == 0


def test_the_pool_runs_out(experiment_psi: PsiOperator) -> None:
    C, w, y = experiment_inputs(experiment_psi)
    result = agglomerative(C[:, :3], w[:3], y, workbook_additional(kappa=10, delta=1e-9))
    assert len(result.tuplam) == 3
    assert result.stop is StopReason.EXHAUSTED


# ---------------------------------------------------------------- ties and γ = 0


def test_ties_go_to_the_first_feature() -> None:
    C = np.array([[0.1, 0.2, 0.2], [0.1, -0.3, -0.3], [-0.1, -0.2, -0.2]])
    result = agglomerative(C, [0.5, 0.5, 0.4], [0, 1, 1], workbook_additional(kappa=2))
    assert result.first == 0  # equal weights: the first feature
    it = result.iterations[0]
    assert it.ratio[0] == it.ratio[1]  # identical columns give bit-identical θ/γ
    assert result.tuplam == (0, 1)  # equal θ/γ: the first candidate


def _equal_after_majorizing() -> tuple[float, float]:
    """A K1 value x and a K2 value y whose majorized values are exactly equal (float)."""
    one, minus = np.array([1.0]), np.array([-1.0])
    for x in np.linspace(-1.0, 1.0, 2001):
        c = majorize(np.array([x]), one, 0.3, sigmoid, None)[0]
        lo, hi = -5.0, 5.0
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (
                (mid, hi)
                if majorize(np.array([mid]), minus, 0.3, sigmoid, None)[0] < c
                else (lo, mid)
            )
        y = lo
        for _ in range(8):
            if majorize(np.array([y]), minus, 0.3, sigmoid, None)[0] == c:
                return float(x), float(y)
            y = float(np.nextafter(y, np.inf))
    pytest.skip("no exactly equal pair on this platform")


def test_gamma_zero_gives_an_infinite_ratio_that_is_never_chosen() -> None:
    x, y = _equal_after_majorizing()
    # R = 0 (first column, largest weight); candidate 2 makes b = (x, y): both majorize to c,
    # so with final class means M₁ = M₂ = c and θ = γ = 0 (ADR-025)
    C = np.array([[0.0, x], [0.0, y]])
    result = agglomerative(C, [1.0, 0.5], [0, 1], workbook_additional(class_means=ClassMeans.FINAL))
    it = result.iterations[0]
    assert (it.theta[0], it.gamma[0]) == (0.0, 0.0)
    assert it.ratio[0] == np.inf
    assert it.q is None
    assert result.stop is StopReason.NO_CANDIDATE


# ---------------------------------------------------------------- inputs


def test_invalid_inputs() -> None:
    C = [[0.1, 0.2], [-0.1, 0.3]]
    with pytest.raises(ValueError, match="2 informativeness values"):
        agglomerative(C, [1.0], [0, 1], excel_experiment().additional_features)
    with pytest.raises(ValueError, match="class indices"):
        agglomerative(C, [1.0, 0.5], [0], excel_experiment().additional_features)
    with pytest.raises(ValueError, match="0 \\(K1\\) or 1"):
        agglomerative(C, [1.0, 0.5], [0, 2], excel_experiment().additional_features)
    with pytest.raises(ValueError, match="finite"):
        agglomerative(
            [[np.nan, 0.2], [0.1, 0.3]], [1.0, 0.5], [0, 1], excel_experiment().additional_features
        )
    with pytest.raises(ModelUndefinedError, match="both classes"):
        agglomerative(C, [1.0, 0.5], [0, 0], excel_experiment().additional_features)
    with pytest.raises(ModelUndefinedError, match="r = 0"):
        agglomerative(np.empty((2, 0)), [], [0, 1], excel_experiment().additional_features)
    with pytest.raises(RegistryError, match="Unknown majorizing functions 'relu'"):
        agglomerative(
            C, [1.0, 0.5], [0, 1], workbook_additional(majorizing_function=plugin("relu"))
        )


@pytest.mark.parametrize("name", ["tanh", "arctan", "softsign"])
def test_other_majorizers(experiment_psi: PsiOperator, name: str) -> None:
    result = agglomerative(
        *experiment_inputs(experiment_psi),
        workbook_additional(majorizing_function=plugin(name)),
    )
    assert result.settings.majorizing_function == name
    assert result.tuplam[0] == 5
    assert len(set(result.tuplam)) == len(result.tuplam)


def test_small_blocks_give_the_same_result(
    experiment_psi: PsiOperator, monkeypatch: pytest.MonkeyPatch
) -> None:
    whole = agglomerative(*experiment_inputs(experiment_psi), WORKBOOK)
    monkeypatch.setattr(agglomerative_module, "BLOCK_ELEMENTS", 1)  # one candidate per block
    blocks = agglomerative(*experiment_inputs(experiment_psi), WORKBOOK)
    assert blocks.tuplam == whole.tuplam
    assert np.array_equal(blocks.additional, whole.additional)
    for a, b in zip(blocks.iterations, whole.iterations, strict=True):
        assert np.array_equal(a.ratio, b.ratio)


# ---------------------------------------------------------------- properties


@st.composite
def agglomerative_inputs(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, AdditionalFeaturesConfig]:
    m = draw(st.integers(2, 12))
    r = draw(st.integers(1, 7))
    values = st.sampled_from([-0.3, -0.2, -0.1, 0.0, 0.1, 0.15, 0.2, 0.3])
    C = np.array(draw(st.lists(st.lists(values, min_size=r, max_size=r), min_size=m, max_size=m)))
    w = np.array(draw(st.lists(st.sampled_from([0.4, 0.6, 0.8, 1.0]), min_size=r, max_size=r)))
    y = np.array(
        draw(
            st.permutations(
                [0, 1, *draw(st.lists(st.integers(0, 1), min_size=m - 2, max_size=m - 2))]
            )
        )
    )
    config = workbook_additional(
        kappa=draw(st.integers(2, 6)),
        delta=draw(st.sampled_from([0.01, 0.1, 0.3])),
        class_means=draw(st.sampled_from(list(ClassMeans))),
        majorizing_passes=draw(st.sampled_from([1, 2])),
    )
    return C, w, y, config


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(agglomerative_inputs())
def test_grouping_invariants(
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray, AdditionalFeaturesConfig],
) -> None:
    C, w, y, config = inputs
    result = agglomerative(C, w, y, config)
    tuplam = result.tuplam
    assert tuplam[0] == int(np.argmax(w))
    assert len(set(tuplam)) == len(tuplam)
    assert len(tuplam) <= max(1, min(config.kappa, C.shape[1]))
    assert result.additional.shape == (C.shape[0], len(tuplam) - 1)
    for it in result.iterations:
        below = np.flatnonzero(it.ratio < config.cr1)
        if it.q is None:
            assert below.size == 0
        else:
            best = below[np.argmin(it.ratio[below])]
            assert it.q == it.candidates[best]
            assert it.crit == it.ratio.min()
    for it in result.iterations[:-1]:
        assert it.go_on
        assert it.crit > config.delta
    if result.iterations:
        assert not result.iterations[-1].go_on
        assert result.iterations[-1].stop is result.stop
