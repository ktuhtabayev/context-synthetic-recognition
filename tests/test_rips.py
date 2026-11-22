"""R-IPS: the hyperspheres (Steps 1–2) and the algorithm (Steps 3–6) on worked examples."""

import logging

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from context_synthetic_recognition.config import DatasetConfig, RIPSConfig
from context_synthetic_recognition.core.rips.algorithm import (
    Outcome,
    RIPSSettings,
    TrainingRole,
    run_rips,
)
from context_synthetic_recognition.core.rips.hyperspheres import hyperspheres
from context_synthetic_recognition.core.rips.selection import select_precedents
from context_synthetic_recognition.data import Dataset, load_from_config
from context_synthetic_recognition.errors import ConfigError, ModelUndefinedError

from .conftest import ROOT


def line(points: list[float]) -> np.ndarray:
    """Distances |xᵢ − xⱼ| of points on a line."""
    x = np.asarray(points, dtype=np.float64)
    return np.abs(x[:, None] - x[None, :])


# K1 = {0, 1, 2} at x = 0, 1, 2; K2 = {3, 4} at x = 10, 11
FIVE = line([0, 1, 2, 10, 11])
FIVE_Y = np.array([0, 0, 0, 1, 1])


def test_radius_count_priority_and_queue() -> None:
    s = hyperspheres(FIVE, FIVE_Y)
    assert s.radius.tolist() == [10, 9, 8, 8, 9]
    assert s.sphere_count.tolist() == [3, 3, 3, 2, 2]
    assert s.nearest_other.tolist() == [3, 3, 3, 2, 2]
    assert s.priority.tolist() == pytest.approx([0.1, 1 / 9, 0.125, 0.125, 1 / 9])
    # r ascending; r(S₂) = r(S₃) = 8 and r(S₁) = r(S₄) = 9: the smaller index first
    assert s.queue.tolist() == [2, 3, 1, 4, 0]
    assert s.zero_radius.size == 0


def test_the_article_reading_step_by_step() -> None:
    result = run_rips(FIVE, FIVE_Y)
    first, second, third = result.iterations
    # S₂ (r = 8): L = {S₀, S₁, S₂}, every counter 3 → 2
    assert (first.selected, first.neighbourhood, first.outcome) == (2, (0, 1, 2), Outcome.CORRECTED)
    assert [(c.object, c.before, c.after) for c in first.corrections] == [
        (0, 3, 2),
        (1, 3, 2),
        (2, 3, 2),
    ]
    assert first.moved == ()
    # S₃ (r = 8): L = {S₃, S₄}, both counters fall to 1 → both move to D
    assert (second.selected, second.neighbourhood, second.moved) == (3, (3, 4), (3, 4))
    # S₁ (r = 9): the blocked S₂ (r = −8) covers nobody → L = {S₀, S₁}, both move
    assert (third.selected, third.neighbourhood, third.moved) == (1, (0, 1), (0, 1))
    assert len(result.iterations) == 3
    assert result.moved.tolist() == [0, 1, 3, 4]
    assert result.remaining.tolist() == [2]
    assert result.order == (2, 3, 1)
    assert result.training.tolist() == [2]
    assert result.test.tolist() == [0, 1, 3, 4]


def test_blocked_objects_keep_covering() -> None:
    result = run_rips(FIVE, FIVE_Y, settings=RIPSSettings(keep_blocked_cover=True))
    # S₁: the blocked S₂ still holds it inside |r(S₂)| = 8 → L = {S₀, S₁, S₂}
    assert result.iterations[2].neighbourhood == (0, 1, 2)
    assert result.moved.tolist() == [0, 1, 2, 3, 4]
    assert result.remaining.size == 0


def test_one_member_moves_the_selected_object_and_an_empty_l_blocks_it() -> None:
    # K1 = {S₀} at x = 0; K2 = {S₁, S₂} at x = 1, 5
    D, y = line([0, 1, 5]), np.array([0, 1, 1])
    s = hyperspheres(D, y)
    assert s.radius.tolist() == [1, 1, 5]
    assert s.sphere_count.tolist() == [1, 1, 2]
    result = run_rips(D, y)
    outcomes = [(it.selected, it.neighbourhood, it.outcome, it.moved) for it in result.iterations]
    assert outcomes == [
        (0, (), Outcome.BLOCKED, ()),  # g(S₀) = 1: L is empty, S₀ stays in E₁
        (1, (2,), Outcome.MOVED, (1,)),  # |L| = 1 although its member is S₂
        (2, (2,), Outcome.MOVED, (2,)),
    ]
    assert result.remaining.tolist() == [0]
    assert result.moved.tolist() == [1, 2]


def test_the_training_role_switch() -> None:
    result = run_rips(FIVE, FIVE_Y, settings=RIPSSettings(training="moved"))
    assert result.settings.training is TrainingRole.MOVED
    assert result.training.tolist() == [0, 1, 3, 4]
    assert result.test.tolist() == [2]
    assert result.training_mask.tolist() == [True, True, False, True, True]


def test_radii_and_counts_are_taken_over_e1_without_the_test_sample() -> None:
    result = run_rips(FIVE, FIVE_Y, test_sample=[2])
    s = result.spheres
    assert s.working.tolist() == [0, 1, 3, 4]
    assert np.isnan(s.radius[2])
    assert s.sphere_count[2] == 0
    assert s.nearest_other[2] == -1
    # without S₂ the nearest K1 object of S₃ is S₁ (r = 9); g counts E₁ only
    assert s.radius[[0, 1, 3, 4]].tolist() == [10, 9, 9, 10]
    assert s.sphere_count[[0, 1, 3, 4]].tolist() == [2, 2, 2, 2]
    assert s.queue.tolist() == [1, 3, 0, 4]
    assert result.test_sample.tolist() == [2]
    assert result.moved.tolist() == [0, 1, 3, 4]
    assert result.remaining.size == 0
    assert result.test.tolist() == [0, 1, 2, 3, 4]


def test_an_object_on_an_object_of_the_other_class_never_enters_the_queue(
    caplog: pytest.LogCaptureFixture,
) -> None:
    D, y = line([0, 0, 3, 4]), np.array([0, 1, 0, 1])
    with caplog.at_level(logging.WARNING):
        s = hyperspheres(D, y)
    assert s.zero_radius.tolist() == [0, 1]
    assert 0 not in s.queue
    assert 1 not in s.queue
    assert np.isnan(s.priority[0])
    assert "never enter the queue" in caplog.text


def test_undefined_and_invalid_inputs() -> None:
    with pytest.raises(ModelUndefinedError, match="both classes"):
        run_rips(FIVE, FIVE_Y, test_sample=[3, 4])
    with pytest.raises(ValueError, match="must be"):
        run_rips(FIVE[:4], FIVE_Y)
    with pytest.raises(ValueError, match="test indices"):
        run_rips(FIVE, FIVE_Y, test_sample=[5])


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.integers(0, 6), min_size=4, max_size=14),
    st.lists(st.booleans(), min_size=4, max_size=14),
    st.booleans(),
)
def test_invariants(points: list[int], labels: list[bool], keep: bool) -> None:
    n = min(len(points), len(labels))
    y = np.array(labels[:n], dtype=np.int64)
    if y.min() == y.max():
        y[0] = 1 - y[0]
    D = line([float(p) for p in points[:n]])
    result = run_rips(D, y, settings=RIPSSettings(keep_blocked_cover=keep))
    moved, remaining = set(result.moved.tolist()), set(result.remaining.tolist())
    assert moved.isdisjoint(remaining)
    assert moved | remaining == set(range(n))
    assert len(result.iterations) <= result.spheres.queue.size
    assert len(set(result.order)) == len(result.order)  # every object is selected at most once
    for it in result.iterations:
        assert it.radius > 0
        if it.outcome is Outcome.MOVED:
            assert it.moved == (it.selected,)
            assert len(it.neighbourhood) == 1
        for c in it.corrections:
            assert c.after == c.before - 1
            assert (c.object in it.moved) == c.moved


# ---------------------------------------------------------------- the datasets of the article


def _counts(indices: np.ndarray, y: np.ndarray) -> tuple[int, int, int]:
    return int(indices.size), int((y[indices] == 0).sum()), int((y[indices] == 1).sum())


SLOPE = {"x₁₁": "quantitative"}
"""Heart-Disease with the slope (x₁₁) quantitative: the article's 7 quantitative features."""
RAW_QUANTITATIVE = {"normalizer": "none", "features": "quantitative"}


@pytest.mark.parametrize(
    ("path", "types", "settings", "moved", "remaining", "keep_covering"),
    [
        ("heart-disease-270", None, {}, (108, 60, 48), (162, 90, 72), (77, 37, 40)),
        ("heart-disease-270", SLOPE, {}, (109, 61, 48), (161, 89, 72), (73, 36, 37)),
        ("german-1000", None, {}, (476, 325, 151), (524, 375, 149), (335, 195, 140)),
        (
            "german-1000",
            None,
            {"metric": "euclidean", **RAW_QUANTITATIVE},
            (503, 373, 130),
            (497, 327, 170),
            (340, 173, 167),
        ),
        (
            "german-1000",
            None,
            {"metric": "chebyshev", **RAW_QUANTITATIVE},
            (482, 363, 119),
            (518, 337, 181),
            (368, 189, 179),
        ),
    ],
    ids=["heart", "heart-slope-quantitative", "german", "german-euclid", "german-chebyshev"],
)
def test_the_splits_of_heart_and_german(
    path: str,
    types: dict[str, str] | None,
    settings: dict[str, str],
    moved: tuple[int, int, int],
    remaining: tuple[int, int, int],
    keep_covering: tuple[int, int, int],
) -> None:
    dataset = load_from_config(DatasetConfig(path=path, feature_types=types), ROOT)  # type: ignore[arg-type]
    y = dataset.class_index
    literal = select_precedents(dataset, RIPSConfig.model_validate(settings))
    assert _counts(literal.moved, y) == moved
    assert _counts(literal.remaining, y) == remaining
    assert np.array_equal(literal.training, literal.remaining)
    kept = select_precedents(
        dataset, RIPSConfig.model_validate({**settings, "keep_blocked_cover": True})
    )
    assert _counts(kept.remaining, y) == keep_covering


def test_a_test_sample_is_held_out_by_object_id(heart270: Dataset) -> None:
    held = tuple(heart270.object_ids[:30])
    result = select_precedents(heart270, RIPSConfig(test_sample=held))
    assert result.test_sample.tolist() == list(range(30))
    assert not set(result.training.tolist()) & set(range(30))
    assert set(range(30)) <= set(result.test.tolist())
    with pytest.raises(ConfigError, match="unknown objects S₉₉₉"):
        select_precedents(heart270, RIPSConfig(test_sample=("S₉₉₉",)))
    with pytest.raises(ValidationError, match="must not repeat"):
        RIPSConfig(test_sample=("S₁", "S₁"))
