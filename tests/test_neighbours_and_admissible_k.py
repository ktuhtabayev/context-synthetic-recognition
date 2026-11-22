"""Step 3 (neighbour order, ranks) and the admissible k (ADR-005, ADR-006)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from context_synthetic_recognition.core.local_context.admissible_k import (
    K_STRATEGIES,
    AdmissibleK,
    admissible_k,
    twice_min_class,
)
from context_synthetic_recognition.core.local_context.neighbours import (
    cumulative_count,
    neighbour_classes,
    neighbour_order,
    rank_matrix,
)
from context_synthetic_recognition.errors import ConfigError, ModelUndefinedError

# ---------------------------------------------------------------- neighbours


def test_ties_go_to_the_smaller_original_index_and_self_is_excluded() -> None:
    D = np.array(
        [
            [0.0, 2.0, 1.0, 2.0],
            [2.0, 0.0, 2.0, 2.0],
            [1.0, 2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0, 0.0],
        ]
    )
    order = neighbour_order(D, np.arange(4))
    assert order.tolist() == [[2, 1, 3], [0, 2, 3], [3, 0, 1], [2, 0, 1]]
    # a new object (nothing excluded) keeps every reference, a tie at 0 included
    assert neighbour_order(D[[3]]).tolist() == [[2, 3, 0, 1]]


def test_rank_matrix_and_running_counts() -> None:
    order = np.array([[2, 1], [0, 2], [1, 0]])
    ranks = rank_matrix(order, 3)
    assert ranks.tolist() == [[0, 2, 1], [1, 0, 2], [2, 1, 0]]
    classes = neighbour_classes(order, np.array([0, 1, 1]))
    assert classes.tolist() == [[1, 1], [0, 1], [1, 0]]
    assert cumulative_count(classes, 0).tolist() == [[0, 0], [1, 1], [0, 1]]
    assert cumulative_count(classes, np.array([0, 1, 1])).tolist() == [[0, 0], [0, 1], [1, 1]]
    assert neighbour_classes(order, np.array([0, 1, 1]), depth=1).tolist() == [[1], [0], [1]]


def test_exclusion_is_validated() -> None:
    D = np.zeros((2, 3))
    with pytest.raises(ValueError, match="one index per target"):
        neighbour_order(D, [0])
    with pytest.raises(ValueError, match="must lie in 0 … 2"):
        neighbour_order(D, [0, 3])


distances = arrays(np.float64, (7, 7), elements=st.integers(0, 3).map(float))


@settings(max_examples=80, deadline=None)
@given(D=distances)
def test_order_is_a_sorted_permutation_without_self(D: np.ndarray) -> None:
    order = neighbour_order(D, np.arange(7))
    for i, row in enumerate(order):
        assert sorted(row.tolist()) == [j for j in range(7) if j != i]
        keys = [(D[i, j], j) for j in row]
        assert keys == sorted(keys)  # by (distance, original index)
    ranks = rank_matrix(order, 7)
    assert np.all(np.diag(ranks) == 0)
    for i in range(7):
        assert sorted(np.delete(ranks[i], i).tolist()) == list(range(1, 7))


# ---------------------------------------------------------------- admissible k


@pytest.mark.parametrize(
    ("sizes", "ks", "k_max"),
    [
        ((4, 6), (3, 5), 5),  # Heart-Disease (10, 13, 2)
        ((150, 120), tuple(range(3, 238, 2)), 237),  # Heart-Disease (270, 13, 2)
        ((3, 9), (3,), 3),
        ((2, 5), (1,), 1),  # the only case where k_min drops (ADR-005)
    ],
)
def test_formula_rule(sizes: tuple[int, int], ks: tuple[int, ...], k_max: int) -> None:
    result = twice_min_class(sizes)
    assert result.ks == ks
    assert result.k_max == k_max
    assert result.rule == "twice-min-class"
    assert result.min_class_size == min(sizes)


def test_formula_rule_for_a_single_object_class() -> None:
    with pytest.raises(ModelUndefinedError, match=r"no k is admissible.*Definition 1"):
        twice_min_class((1, 9))


def test_cap_and_step_are_parameters() -> None:
    capped = admissible_k("twice-min-class", {"k_max_cap": 11}, (150, 120), 269)
    assert capped.ks == (3, 5, 7, 9, 11)
    assert capped.k_max == 237
    assert "capped at 11" in capped.note
    stepped = admissible_k("twice-min-class", {"step": 4}, (10, 10), 19)
    assert stepped.ks == (3, 7, 11, 15)
    with pytest.raises(ConfigError, match="the step must be even"):
        admissible_k("twice-min-class", {"step": 3}, (10, 10), 19)
    with pytest.raises(ConfigError, match="k_max_cap"):
        admissible_k("twice-min-class", {"k_max_cap": 1}, (10, 10), 19)
    with pytest.raises(ConfigError, match="k_max_kap: Unexpected keyword argument"):
        admissible_k("twice-min-class", {"k_max_kap": 5}, (10, 10), 19)


def test_other_strategies() -> None:
    assert admissible_k("min-class", None, (4, 6), 9).ks == (3,)
    assert admissible_k("min-class", None, (36, 40), 75).ks == tuple(range(3, 36, 2))
    with pytest.raises(ModelUndefinedError, match="the odd k start at 3"):
        admissible_k("min-class", None, (2, 6), 7)
    assert admissible_k("explicit", {"values": [1, 7]}, (4, 6), 9).ks == (1, 7)
    assert admissible_k("range", {"stop": 9}, (4, 6), 9).ks == (3, 5, 7, 9)
    assert admissible_k("range", {"start": 5, "stop": 10, "step": 4}, (4, 6), 9).ks == (5, 9)
    with pytest.raises(ConfigError, match="odd"):
        admissible_k("explicit", {"values": [2]}, (4, 6), 9)
    with pytest.raises(ConfigError, match="strictly increasing"):
        admissible_k("explicit", {"values": [5, 3]}, (4, 6), 9)
    with pytest.raises(ConfigError, match="odd"):
        admissible_k("range", {"start": 4, "stop": 9}, (4, 6), 9)
    with pytest.raises(ConfigError, match="even"):
        admissible_k("range", {"stop": 9, "step": 3}, (4, 6), 9)
    with pytest.raises(ModelUndefinedError, match="k = 11 exceeds the 9 neighbours"):
        admissible_k("explicit", {"values": [3, 11]}, (4, 6), 9)
    with pytest.raises(ModelUndefinedError, match="no k is admissible"):
        admissible_k("range", {"start": 9, "stop": 3}, (4, 6), 9)
    assert K_STRATEGIES.names() == ["min-class", "twice-min-class", "explicit", "range"]


def test_admissible_k_values_are_checked() -> None:
    with pytest.raises(ValueError, match="odd and positive"):
        AdmissibleK((2,), "custom", (4, 6))
    with pytest.raises(ValueError, match="strictly increasing"):
        AdmissibleK((5, 3), "custom", (4, 6))
    assert AdmissibleK((3, 5), "custom", (4, 6)).label == "3, 5"


@settings(max_examples=100, deadline=None)
@given(sizes=st.tuples(st.integers(2, 400), st.integers(2, 400)))
def test_formula_rule_properties(sizes: tuple[int, int]) -> None:
    result = twice_min_class(sizes)
    smallest = min(sizes)
    assert all(k % 2 == 1 for k in result.ks)
    assert result.ks[-1] == max(1, 2 * smallest - 3)
    assert result.ks[0] == (1 if smallest == 2 else 3)
    assert result.count == (1 if smallest == 2 else smallest - 2)
    # at k = k_max a same-class majority needs (k + 1)/2 = min|Kᵢ| − 1 neighbours
    assert (result.ks[-1] + 1) // 2 == smallest - 1
