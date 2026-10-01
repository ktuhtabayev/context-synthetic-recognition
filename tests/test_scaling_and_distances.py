"""Step 1 (normalizers) and Step 2 (metrics, base operators)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from context_synthetic_recognition.config import ContextConfig, OperatorConfig, plugin
from context_synthetic_recognition.core.metrics import METRICS, zhuravlyov
from context_synthetic_recognition.core.normalizers import NORMALIZERS, minmax, no_scaling
from context_synthetic_recognition.core.operators import resolve_operators
from context_synthetic_recognition.errors import ConfigError, ModelUndefinedError, RegistryError

QUANT = np.array([True, False, True])

# ---------------------------------------------------------------- normalizers


def test_minmax_maps_the_training_range_to_unit_interval() -> None:
    X = np.array([[1.0, 7.0, 5.0], [3.0, 8.0, 5.0], [2.0, 7.0, 5.0]])
    scaling = minmax(X, QUANT)
    Z = scaling.transform(X)
    assert Z[:, 0].tolist() == [0.0, 1.0, 0.5]
    assert Z[:, 1].tolist() == [7.0, 8.0, 7.0]  # nominal codes unchanged
    assert Z[:, 2].tolist() == [0.0, 0.0, 0.0]  # no spread → 0 (workbook: IF(MAX = MIN, 0, …))
    assert scaling.constant.tolist() == [False, False, True]
    assert scaling.statistics["min"][0] == 1.0
    assert np.isnan(scaling.statistics["max"][1])


def test_new_objects_use_the_training_constants_without_clipping() -> None:
    scaling = minmax(np.array([[1.0, 0.0, 5.0], [3.0, 1.0, 5.0]]), QUANT)
    Z = scaling.transform(np.array([[5.0, 1.0, 9.0], [0.0, 0.0, 5.0]]))
    assert Z.tolist() == [[2.0, 1.0, 0.0], [-0.5, 0.0, 0.0]]


def test_scaling_is_immutable_and_checks_width() -> None:
    scaling = minmax(np.ones((2, 3)), QUANT)
    with pytest.raises(ValueError, match="read-only"):
        scaling.offset[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        scaling.statistics["min"][0] = 1.0
    with pytest.raises(ValueError, match="the scaling was fitted on 3"):
        scaling.transform(np.ones((1, 2)))
    with pytest.raises(ValueError, match="2-D"):
        scaling.transform(np.ones(3))


def test_no_scaling_and_registry() -> None:
    X = np.array([[1.0, 2.0, 3.0]])
    assert no_scaling(X, QUANT).transform(X).tolist() == X.tolist()
    assert NORMALIZERS.get("fractional-linear") is minmax
    assert NORMALIZERS.get("identity") is no_scaling
    with pytest.raises(ConfigError, match="takes no parameters"):
        NORMALIZERS.make_params("minmax", {"range": [0, 1]})


# ---------------------------------------------------------------- the Zhuravlyov metric


def test_zhuravlyov_rounds_the_quantitative_part_then_the_sum() -> None:
    A = np.array([[0.1, 1.0, 0.2]])
    B = np.array([[0.4, 2.0, 0.0], [0.1, 1.0, 0.2]])
    D = zhuravlyov(A, B, QUANT, 10, None)
    assert D.tolist() == [[round(0.3 + 0.2, 10) + 1.0, 0.0]]
    # 0.1 + 0.2 = 0.30000000000000004 → 0.3 after rounding, so equal distances tie exactly
    assert (
        zhuravlyov(
            np.array([[0.0, 0.0]]), np.array([[0.1, 0.2]]), np.array([True, True]), 10, None
        )[0, 0]
        == 0.3
    )


unified = arrays(np.float64, (5, 4), elements=st.floats(-2, 2, allow_nan=False, width=32))
masks = arrays(np.bool_, 4)


@settings(max_examples=60, deadline=None)
@given(X=unified, quantitative=masks)
def test_zhuravlyov_metric_axioms(X: np.ndarray, quantitative: np.ndarray) -> None:
    Z = X.copy()
    Z[:, ~quantitative] = np.round(Z[:, ~quantitative])  # nominal codes
    D = zhuravlyov(Z, Z, quantitative, 10, None)
    assert np.array_equal(D, D.T)  # symmetry, bit for bit
    assert np.all(np.diag(D) == 0.0)
    assert np.all(D >= 0.0)
    for k in range(len(Z)):  # triangle inequality up to the rounding
        assert np.all(D[:, [k]] + D[[k], :] + 1e-9 >= D)
    nominal_part = zhuravlyov(
        Z[:, ~quantitative], Z[:, ~quantitative], np.zeros((~quantitative).sum(), bool), 10, None
    )
    assert np.array_equal(nominal_part, np.round(nominal_part))  # ρ_J is a Hamming count


# ---------------------------------------------------------------- base operators


NAMES = ("x₁", "x₂", "x₃")


def test_default_operators_use_all_i_and_j() -> None:
    operators, skipped = resolve_operators(ContextConfig(), NAMES, QUANT)
    assert [(o.label, o.features.tolist()) for o in operators] == [
        ("ρ", [0, 1, 2]),
        ("ρ_I", [0, 2]),
        ("ρ_J", [1]),
    ]
    assert operators[1].quantitative.tolist() == [True, True]
    assert operators[2].quantitative.tolist() == [False]
    assert skipped == ()
    assert operators[0].metric_name == "zhuravlyov"


def test_empty_subsets_are_skipped_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    operators, skipped = resolve_operators(ContextConfig(), NAMES, np.ones(3, dtype=bool))
    assert [o.label for o in operators] == ["ρ", "ρ_I"]
    assert [(s.label, s.reason) for s in skipped] == [("ρ_J", "no nominal features in the dataset")]
    assert "ADR-007" in caplog.text


def test_empty_subsets_kept_when_skipping_is_off() -> None:
    config = ContextConfig(skip_empty_operators=False)
    operators, skipped = resolve_operators(config, NAMES, np.ones(3, dtype=bool))
    assert [o.features.size for o in operators] == [3, 3, 0]
    assert skipped == ()
    D = operators[2].distances(np.ones((2, 3)), np.zeros((3, 3)), decimals=10)
    assert D.tolist() == [[0.0] * 3] * 2


def test_explicit_subsets_and_errors() -> None:
    config = ContextConfig(operators=(OperatorConfig(label="ρ₁₃", features=("x₃", "x₁")),))
    (operator,), _ = resolve_operators(config, NAMES, QUANT)
    assert operator.features.tolist() == [2, 0]
    unknown = ContextConfig(operators=(OperatorConfig(label="ρ", features=("x₉",)),))
    with pytest.raises(ConfigError, match=r"unknown features \['x₉'\]"):
        resolve_operators(unknown, NAMES, QUANT)
    # HVDM reads the classes of the training objects and is deliberately not offered (ADR-051)
    bad_metric = ContextConfig(operators=(OperatorConfig(label="ρ", metric=plugin("hvdm")),))
    with pytest.raises(RegistryError, match="Unknown metrics 'hvdm'"):
        resolve_operators(bad_metric, NAMES, QUANT)
    only_j = ContextConfig(operators=(OperatorConfig(label="ρ_J", features="nominal"),))
    with pytest.raises(ModelUndefinedError, match="no base operator is left"):
        resolve_operators(only_j, NAMES, np.ones(3, dtype=bool))


def test_blocked_distances_equal_one_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(1)
    Z = rng.random((40, 3))
    (operator, *_), _ = resolve_operators(ContextConfig(), NAMES, QUANT)
    whole = operator.distances(Z, Z, decimals=10)
    monkeypatch.setattr("context_synthetic_recognition.core.operators.BLOCK_ELEMENTS", 50)
    assert np.array_equal(operator.distances(Z, Z, decimals=10), whole)


def test_metric_registry() -> None:
    assert METRICS.names()[0] == "zhuravlyov"  # the default; the others: tests/test_metrics.py
    # earlier spellings still resolve, so older configuration files keep working
    assert METRICS.get("zhuravlev") is zhuravlyov
    assert METRICS.get("juravlev") is zhuravlyov
