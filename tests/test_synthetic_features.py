"""Steps 4–8: μ, χ₁, formula (5), formulas (1)–(4), bit masks, formula (6)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from context_synthetic_recognition.core.contributions import (
    WEIGHTS,
    WeightInputs,
    contribution_values,
    contributions,
    gradation_counts,
    omega,
    weight_ranks,
)
from context_synthetic_recognition.core.encoders import (
    ENCODERS,
    formula_5,
    k1_counts,
    same_class_counts,
)
from context_synthetic_recognition.core.membership import (
    MAX_MASK_BITS,
    bit_masks,
    boundary,
    correct_side,
    informativeness,
    membership_table,
    stability,
)

# The experiment's classes (S₁ … S₁₀) as class indices, and μ of a₁ (ρ, k = 3) and a₃ (ρ_I, k = 3)
CLASSES = np.array([1, 0, 1, 0, 0, 0, 1, 1, 1, 1])
SIZES = (4, 6)
MU_A1 = np.array([3, 0, 1, 1, 1, 1, 2, 2, 3, 2])
MU_A3 = np.array([1, 1, 2, 2, 2, 0, 2, 2, 3, 3])

# ---------------------------------------------------------------- μ, χ₁ and formula (5)


def test_counts_over_nested_neighbourhoods() -> None:
    order = np.array([[1, 2, 3], [0, 2, 3], [3, 1, 0], [2, 1, 0]])
    classes = np.array([0, 1, 1, 0])
    assert k1_counts(order, classes, (1, 3)).tolist() == [[0, 1], [1, 2], [1, 2], [0, 1]]
    assert same_class_counts(order, classes, classes, (1, 3)).tolist() == [
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
    ]


def test_formula_5_is_the_neighbourhood_majority() -> None:
    chi1 = np.array([0, 1, 2, 3])
    assert formula_5(chi1, 3).tolist() == [2, 2, 1, 1]
    assert formula_5(np.array([0, 1]), 1).tolist() == [2, 1]
    with pytest.raises(ValueError, match="odd k"):
        formula_5(chi1, 4)
    assert ENCODERS.get("majority") is formula_5


# ---------------------------------------------------------------- formulas (1)–(4)


def test_membership_of_a1_as_in_the_workbook() -> None:
    table = membership_table(MU_A1, CLASSES, SIZES, beta=3)
    assert table.gradations.tolist() == [0, 1, 2, 3]
    assert table.d1.tolist() == [1, 3, 0, 0]
    assert table.d2.tolist() == [0, 1, 3, 2]
    assert table.n.tolist() == [1, 4, 3, 2]
    assert table.f.tolist() == [1.0, 0.75 / (0.75 + 1 / 6), 0.0, 0.0]
    assert stability(table, 10) == pytest.approx(0.927272727272727, abs=1e-15)
    b = boundary(table)
    assert (b.q2, b.q1) == (0.0, table.f[1])
    assert np.isclose(b.G, 0.409090909090909, rtol=0, atol=1e-15)
    object_f = table.f_of(MU_A1)
    correct = correct_side(object_f, CLASSES, b.G)
    assert informativeness(correct) == 0.9


def test_f_of_one_half_belongs_to_neither_side() -> None:
    table = membership_table(MU_A3, CLASSES, SIZES, beta=3)
    assert table.f[2] == 0.5
    b = boundary(table)
    assert (b.q1, b.q2) == (pytest.approx(0.6), 0.0)
    assert np.isclose(b.G, 0.3, rtol=0, atol=1e-15)
    assert informativeness(correct_side(table.f_of(MU_A3), CLASSES, b.G)) == 0.6


def test_undefined_gradations_and_one_sided_boundary() -> None:
    table = membership_table(np.array([0, 0, 5]), np.array([0, 0, 1]), (2, 1), beta=5)
    assert np.isnan(table.f[1:5]).all()
    assert table.weighted[1:5].tolist() == [0.0] * 4
    assert table.defined.tolist() == [True, False, False, False, False, True]
    only_high = membership_table(np.array([1, 1]), np.array([0, 1]), (1, 1), beta=1)
    assert only_high.f[1] == 0.5
    assert boundary(only_high).G == 0.5
    assert boundary(only_high).q1 is None
    with pytest.raises(ValueError, match="not a row"):
        table.f_of(np.array([7]))


def test_sparse_tables_for_many_gradations() -> None:
    table = membership_table(np.array([3, 70000]), np.array([0, 1]), (1, 1), beta=100000)
    assert table.gradations.tolist() == [3, 70000]
    assert table.f.tolist() == [1.0, 0.0]


@st.composite
def gradation_samples(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, int]:
    k = draw(st.sampled_from([1, 3, 5, 7, 9]))
    m = draw(st.integers(4, 30))
    classes = np.array(draw(st.lists(st.integers(0, 1), min_size=m, max_size=m)))
    classes[:2] = [0, 1]
    mu = np.array(draw(st.lists(st.integers(0, k), min_size=m, max_size=m)))
    return mu, classes, k


@settings(max_examples=150, deadline=None)
@given(sample=gradation_samples())
def test_formula_properties(sample: tuple[np.ndarray, np.ndarray, int]) -> None:
    mu, classes, k = sample
    sizes = (int((classes == 0).sum()), int((classes == 1).sum()))
    table = membership_table(mu, classes, sizes, beta=k)
    f = table.f[table.defined]
    assert np.all((f >= 0.0) & (f <= 1.0))
    assert table.n.sum() == mu.size
    g = stability(table, mu.size)
    assert 0.5 - 1e-12 <= g <= 1.0 + 1e-12
    b = boundary(table)
    assert 0.0 <= b.G <= 1.0
    w = informativeness(correct_side(table.f_of(mu), classes, b.G))
    assert 0.0 <= w <= 1.0
    values = np.where(mu > k // 2, 1, 2)
    eta = contributions(gradation_counts(values, classes), sizes, w)
    assert eta[0] == pytest.approx(-eta[1], abs=1e-12)  # η(2) = −η(1)
    assert abs(eta[0]) <= w + 1e-12


# ---------------------------------------------------------------- bit masks (Task 2)


def test_bit_masks_first_k_is_the_most_significant_bit() -> None:
    mu = np.array([[2, 3], [1, 3], [0, 1], [2, 2]])
    bits, masks = bit_masks(mu, (3, 5))
    assert bits.tolist() == [[True, True], [False, True], [False, False], [True, False]]
    assert masks.tolist() == [3, 1, 0, 2]
    with pytest.raises(ValueError, match=f"at most {MAX_MASK_BITS}"):
        bit_masks(np.zeros((1, 63), dtype=np.int64), tuple(range(1, 127, 2)))


# ---------------------------------------------------------------- formula (6) and the weights


def test_contributions_of_a6_as_in_the_workbook() -> None:
    a6 = np.array([2, 2, 1, 2, 2, 2, 2, 2, 2, 2])
    alpha = gradation_counts(a6, CLASSES)
    assert alpha.tolist() == [[0, 1], [4, 5]]  # α¹₁, α²₁ / α¹₂, α²₂
    eta = contributions(alpha, SIZES, 1.0)
    assert eta.tolist() == [-1 / 6, 1.0 * (1 - 5 / 6)]
    assert contribution_values(a6, eta)[:3].tolist() == [eta[1], eta[1], eta[0]]


def test_weight_ranks_prefer_the_earlier_feature_on_ties() -> None:
    assert weight_ranks(np.array([0.9, 0.9, 0.6, 0.9, 0.9, 1.0])).tolist() == [2, 3, 6, 4, 5, 1]


def test_omega_weight_plugin() -> None:
    assert omega(WeightInputs(omega=0.7, stability=0.9, boundary=0.4)) == 0.7
    assert WEIGHTS.get("informativeness") is omega
