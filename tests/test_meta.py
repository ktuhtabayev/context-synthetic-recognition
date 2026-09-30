"""Meta-algorithm, Steps 1–5: B1/B2 filtering, the decision rule, refusals, any gradations."""

import dataclasses

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from context_synthetic_recognition.core import meta as meta_module
from context_synthetic_recognition.core.meta import (
    DECISION_RULES,
    K1_DECISION,
    K2_DECISION,
    REFUSAL,
    MetaDescription,
    article_step_4,
    meta_classify,
)
from context_synthetic_recognition.errors import ConfigError, ModelUndefinedError, RegistryError

# the template's meta-algorithm: SET (x₃, x₆, x₁₃, x₄, x₉), training rows S₁ … S₁₀
TEMPLATE_A = [
    [4, 0, 3, 2, 0],
    [3, 0, 7, 1, 0],
    [2, 0, 7, 2, 0],
    [4, 0, 7, 2, 1],
    [2, 0, 3, 1, 1],
    [4, 0, 7, 1, 0],
    [3, 1, 6, 2, 1],
    [4, 0, 7, 1, 1],
    [4, 0, 7, 2, 0],
    [4, 0, 7, 2, 0],
]
TEMPLATE_D = [
    [-0.37806568533891394, -0.6769974102656717, -1.3589807460522612, -1.905623924879916],
    [0.40872845598398955, 0.6922541175317597, 1.0973331755196423, 1.1955874992027573],
    [-0.14942741726637135, -0.41361540108076145, -1.0620337591921905, -1.5793807325380578],
    [0.2392391500520089, 0.5460749119812512, 0.5506615357984214, 0.809109085404975],
    [0.40872845598398955, 0.6922541175317596, 1.097333175519642, 1.2912243595415676],
    [0.2392391500520089, 0.5460749119812512, 0.9685828814261568, 1.0814390907557363],
    [-0.32058777621177315, -0.8201335059825633, -1.5186565413621942, -1.962781182532284],
    [-0.37806568533891394, -0.6769974102656716, -0.8134278796772768, -1.1814449898616624],
    [-0.37806568533891394, -0.6769974102656716, -1.3589807460522612, -1.905623924879916],
    [-0.37806568533891394, -0.6769974102656716, -1.3589807460522612, -1.905623924879916],
]
TEMPLATE_Y = [1, 0, 1, 0, 0, 0, 1, 1, 1, 1]


def template() -> MetaDescription:
    return MetaDescription.of(TEMPLATE_A, TEMPLATE_D, TEMPLATE_Y)


def test_the_template_new_object_is_class_2() -> None:
    result = meta_classify(template(), [4, 0, 7, 2, 0])
    assert result.decisions.tolist() == [K2_DECISION]
    steps = result.steps(0)
    assert steps.b1(4).tolist() == []  # B1(a₄) = ∅
    assert steps.b2(4).tolist() == [8, 9]  # B2(a₄) = {S₉, S₁₀}
    assert result.b1_sizes.tolist() == [[2, 2, 2, 1, 0]]
    assert result.b2_sizes.tolist() == [[4, 4, 3, 2, 2]]
    assert steps.b1_sizes.tolist() == [2, 2, 2, 1, 0]
    assert steps.b2_sizes.tolist() == [4, 4, 3, 2, 2]
    assert result.scores1.tolist() == [0.0]
    assert result.scores2.tolist() == [2 / 6]
    assert result.scores.tolist() == [-2 / 6]
    assert result.rule == "article-step-4"


def test_sets_are_nested_and_class_pure() -> None:
    description = template()
    steps = meta_classify(description, [4, 0, 7, 2, 0]).steps(0)
    k1 = description.class_index == 0
    for j in range(description.p + 1):
        assert not steps.in_b1[j][~k1].any()
        assert not steps.in_b2[j][k1].any()
        if j:
            assert not (steps.in_b1[j] & ~steps.in_b1[j - 1]).any()
            assert not (steps.in_b2[j] & ~steps.in_b2[j - 1]).any()
    # Step 2: a kept object matches the gradation and has the latent sign of its class
    assert steps.match.shape == (5, 10)
    assert steps.sign.shape == (4, 10)
    assert steps.sign[:, 1].all()  # S₂ ∈ K1 has d > 0 everywhere


def test_the_decision_rule_and_refusal() -> None:
    sizes = (4, 6)
    b1 = np.array([2, 0, 2, 3])
    b2 = np.array([0, 2, 3, 3])
    # 2/4 > 0/6; 0/4 < 2/6; 2/4 = 3/6 → refusal; 3/4 > 3/6
    assert article_step_4(b1, b2, sizes).tolist() == [
        K1_DECISION,
        K2_DECISION,
        REFUSAL,
        K1_DECISION,
    ]
    assert DECISION_RULES.info("step-4").name == "article-step-4"
    assert DECISION_RULES.make_params("article-step-4", {}) is None


def test_p_zero_uses_only_step_1() -> None:
    description = MetaDescription.of([[1], [2], [1], [2]], np.empty((4, 0)), [0, 0, 1, 1])
    assert description.p == 0
    result = meta_classify(description, [[1], [2]])
    assert result.b1_sizes.tolist() == [[1], [1]]
    assert result.decisions.tolist() == [REFUSAL, REFUSAL]


def test_a_zero_latent_value_keeps_the_object_in_neither_set() -> None:
    description = MetaDescription.of([[1, 1], [1, 1]], [[0.0], [0.0]], [0, 1])
    result = meta_classify(description, [1, 1])
    assert result.b1_sizes.tolist() == [[1, 0]]
    assert result.b2_sizes.tolist() == [[1, 0]]
    assert result.decisions.tolist() == [REFUSAL]


def test_many_queries_in_small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    queries = np.array(TEMPLATE_A)
    whole = meta_classify(template(), queries)
    monkeypatch.setattr(meta_module, "BLOCK_ELEMENTS", 1)
    blocks = meta_classify(template(), queries)
    assert np.array_equal(whole.b1_sizes, blocks.b1_sizes)
    assert np.array_equal(whole.decisions, blocks.decisions)


def test_the_result_is_immutable() -> None:
    result = meta_classify(template(), [4, 0, 7, 2, 0])
    with pytest.raises(ValueError, match="read-only"):
        result.decisions[0] = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.rule = "x"  # type: ignore[misc]


def test_invalid_descriptions_and_queries() -> None:
    with pytest.raises(ValueError, match=r"\(m × p\)"):
        MetaDescription.of([[1, 2], [1, 2]], [[0.1, 0.2], [0.3, 0.4]], [0, 1])
    with pytest.raises(ValueError, match="gradations must be"):
        MetaDescription(np.array([[1], [2]]), np.empty((3, 0)), np.array([0, 1, 1]))
    with pytest.raises(ValueError, match="at least the first"):
        MetaDescription(np.empty((2, 0)), np.empty((2, 0)), np.array([0, 1]))
    with pytest.raises(ValueError, match="0 \\(K1\\) or 1"):
        MetaDescription.of([[1], [2]], np.empty((2, 0)), [0, 3])
    with pytest.raises(ModelUndefinedError, match="both classes"):
        MetaDescription.of([[1], [2]], np.empty((2, 0)), [0, 0])
    with pytest.raises(ValueError, match="p \\+ 1 = 5"):
        meta_classify(template(), [4, 0, 7])
    with pytest.raises(RegistryError):
        meta_classify(template(), [4, 0, 7, 2, 0], rule="majority")
    with pytest.raises(ConfigError, match="takes no parameters"):
        meta_classify(template(), [4, 0, 7, 2, 0], params={"x": 1})


def test_a_rule_must_return_decision_codes() -> None:
    DECISION_RULES.add("broken-test-rule", lambda b1, b2, sizes, params: b1 * 0 + 7)
    try:
        with pytest.raises(ValueError, match="must return 1"):
            meta_classify(template(), [4, 0, 7, 2, 0], rule="broken-test-rule")
    finally:
        DECISION_RULES._remove("broken-test-rule")


@settings(max_examples=100, deadline=None)
@given(
    st.integers(2, 12).flatmap(
        lambda m: st.tuples(
            st.lists(st.lists(st.integers(1, 2), min_size=3, max_size=3), min_size=m, max_size=m),
            st.lists(
                st.lists(st.sampled_from([-0.5, 0.0, 0.5]), min_size=2, max_size=2),
                min_size=m,
                max_size=m,
            ),
            st.permutations([0, 1] + [i % 2 for i in range(m - 2)]),
            st.lists(st.integers(1, 2), min_size=3, max_size=3),
        )
    )
)
def test_the_decision_agrees_with_the_scores(
    data: tuple[list[list[int]], list[list[float]], list[int], list[int]],
) -> None:
    a, d, y, query = data
    description = MetaDescription.of(a, d, y)
    result = meta_classify(description, query)
    s1, s2 = result.scores1[0], result.scores2[0]
    expected = K1_DECISION if s1 > s2 else K2_DECISION if s1 < s2 else REFUSAL
    assert result.decisions[0] == expected
    assert np.all(np.diff(result.b1_sizes[0]) <= 0)
    assert np.all(np.diff(result.b2_sizes[0]) <= 0)
