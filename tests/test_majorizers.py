"""Majorizing functions ϕ of the HAG regulariser (ADR-026)."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from context_synthetic_recognition.core.majorizers import MAJORIZERS, regularize, sigmoid

NAMES = ["sigmoid", "tanh", "arctan", "softsign"]


def test_the_registry() -> None:
    assert MAJORIZERS.names() == NAMES
    assert MAJORIZERS.info("logistic").name == "sigmoid"
    for name in NAMES:
        assert MAJORIZERS.make_params(name, {}) is None


@pytest.mark.parametrize("name", NAMES)
def test_every_majorizer_maps_onto_0_1_and_increases(name: str) -> None:
    phi = MAJORIZERS.get(name)
    x = np.linspace(-50.0, 50.0, 2001)
    y = phi(x, None)
    assert np.all((y >= 0.0) & (y <= 1.0))
    assert np.all(np.diff(y) >= 0.0)
    assert phi(np.array([0.0]), None)[0] == 0.5
    assert np.allclose(phi(x, None) + phi(-x, None), 1.0)  # ϕ(x) + ϕ(−x) = 1


def test_the_logistic_sigmoid_is_the_reference_engine_formula() -> None:
    b = np.array([-2.0, -0.3, 0.0, 0.316666666666667, 5.0])
    assert sigmoid(-b).tolist() == [1.0 / (1.0 + np.exp(v)) for v in b]


def test_tanh_is_a_steeper_logistic() -> None:
    x = np.linspace(-5, 5, 101)
    assert np.allclose(MAJORIZERS.get("tanh")(x, None), sigmoid(2 * x))


def test_no_overflow_warning_far_out() -> None:
    # pytest turns warnings into errors: a large b must give exactly 0, not an overflow warning
    assert sigmoid(np.array([-1000.0, 1000.0])).tolist() == [0.0, 1.0]


@given(st.lists(st.floats(-30, 30), min_size=1, max_size=20), st.floats(0.01, 0.99))
def test_one_pass_moves_k1_up_and_k2_down_by_less_than_alpha(
    values: list[float], alpha: float
) -> None:
    b = np.array(values)
    sign = np.where(np.arange(b.size) % 2 == 0, 1.0, -1.0)
    moved = regularize(b, sign, alpha, sigmoid, None)
    step = (moved - b) * sign
    assert np.all(step >= 0.0)
    assert np.all(step <= alpha * (1 + 1e-12) + 1e-12)  # up to the rounding of b + α·ϕ − b


def test_regularize_works_on_blocks_of_candidates() -> None:
    b = np.array([[0.1, 0.2], [-0.4, 0.5], [0.3, -0.6]])
    sign = np.array([1.0, -1.0, 1.0])
    block = regularize(b, sign, 0.3, sigmoid, None)
    for c in range(2):
        assert block[:, c].tolist() == regularize(b[:, c], sign, 0.3, sigmoid, None).tolist()
