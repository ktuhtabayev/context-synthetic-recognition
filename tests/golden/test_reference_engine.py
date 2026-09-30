"""Cross-check Steps 1–8 against the reference engine (docs/handoff/reference-engine), bit for bit.

The reference engine is the dependency-free Python that reproduces the workbook; it is used here
as an oracle on the experiment, on Heart-Disease (270, 13, 2) and on random tie-heavy datasets.
"""

import json
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from context_synthetic_recognition.core.context import ContextModel, fit_context
from context_synthetic_recognition.data import Dataset, FeatureType

from ..conftest import GOLDEN_VALUES, reference_engine

pytestmark = pytest.mark.golden


def reference_steps(dataset: Dataset) -> dict[str, Any]:
    """Steps 1–8 as the reference engine's ``pipeline.fit`` computes them (without its HAG)."""
    ref = reference_engine()
    X, y, types = dataset.X.tolist(), dataset.y.tolist(), list(dataset.type_flags)
    m = len(X)
    lo, hi = ref.scale_params(X, types)
    Z = [ref.unify(x, types, lo, hi) for x in X]
    ks, k_max = ref.permitted_k(y)
    ops: dict[str, dict[str, Any]] = {}
    for op, _ in ref.OPERATORS:
        D = [[ref.rho(Z[i], Z[j], X[i], X[j], types, op) for j in range(m)] for i in range(m)]
        ops[op] = {"D": D, "order": [ref.neighbours(D[i], exclude=i) for i in range(m)]}
    features = []
    for op, _ in ref.OPERATORS:
        order = ops[op]["order"]
        for k in ks:
            same = [sum(1 for j in order[i][:k] if y[j] == y[i]) for i in range(m)]
            chi1 = [ref.chi1(order[i], y, k) for i in range(m)]
            a = [ref.binary_a(c, k) for c in chi1]
            fk = ref.membership(same, y, k)
            G, q1, q2 = ref.boundary(fk)
            w = ref.informativeness(same, y, fk, G)
            eta = ref.contributions(a, y, w)
            features.append(
                {
                    "mu": same,
                    "chi1": chi1,
                    "a": a,
                    "f": [fk[g]["f"] for g in range(k + 1)],
                    "g": ref.stability(fk, m, k),
                    "G": G,
                    "q1": q1,
                    "q2": q2,
                    "w": w,
                    "eta": [eta[1]["eta"], eta[2]["eta"]],
                    "alpha": [[eta[1]["a1"], eta[1]["a2"]], [eta[2]["a1"], eta[2]["a2"]]],
                }
            )
    return {"Z": Z, "ks": ks, "k_max": k_max, "ops": ops, "features": features}


def assert_identical(model: ContextModel, expected: dict[str, Any]) -> None:
    """Every Step 1–8 value equal to the reference engine's — exactly, not to a tolerance."""
    t = model.trace
    assert t.permitted_k.ks == tuple(expected["ks"])
    assert t.permitted_k.k_max == expected["k_max"]
    assert np.array_equal(t.normalized, np.array(expected["Z"]))
    for operator, op in zip(t.operators, "ZIJ", strict=True):
        assert np.array_equal(operator.distances, np.array(expected["ops"][op]["D"]))
        assert np.array_equal(operator.order, np.array(expected["ops"][op]["order"]))
    assert len(t.features) == len(expected["features"])
    for feature, ref in zip(t.features, expected["features"], strict=True):
        assert feature.mu.tolist() == ref["mu"]
        assert feature.chi1.tolist() == ref["chi1"]
        assert feature.values.tolist() == ref["a"]
        f = [None if np.isnan(v) else float(v) for v in feature.membership.f]
        assert f == ref["f"]
        assert feature.stability == ref["g"]
        assert (feature.boundary.G, feature.boundary.q1, feature.boundary.q2) == (
            ref["G"],
            ref["q1"],
            ref["q2"],
        )
        assert feature.omega == ref["w"]
        assert feature.eta.tolist() == ref["eta"]
        assert feature.alpha.tolist() == ref["alpha"]


def test_golden_values_json(experiment_model: ContextModel) -> None:
    with GOLDEN_VALUES.open(encoding="utf-8") as handle:
        golden = json.load(handle)["ref"]
    t = experiment_model.trace
    assert t.permitted_k.label == golden["klist"]
    assert t.permitted_k.k_max == golden["kmax"]
    assert t.r == golden["r"]
    assert t.omegas.tolist() == golden["w"]
    assert [f.eta[0] for f in t.features] == golden["eta1"]
    assert [f.eta[1] for f in t.features] == golden["eta2"]


def test_the_experiment_is_identical(experiment: Dataset, experiment_model: ContextModel) -> None:
    assert_identical(experiment_model, reference_steps(experiment))


@pytest.mark.slow
def test_heart_disease_270_is_identical(heart270: Dataset) -> None:
    assert_identical(fit_context(heart270), reference_steps(heart270))


@st.composite
def tie_heavy_datasets(draw: st.DrawFn) -> Dataset:
    """Small heterogeneous datasets with few distinct values — many equal distances."""
    m = draw(st.integers(6, 16))
    n = draw(st.integers(2, 5))
    flags = draw(st.lists(st.sampled_from([0, 1]), min_size=n, max_size=n))
    flags[:2] = [1, 0]  # both I and J non-empty, as the reference engine assumes
    values = draw(
        st.lists(st.lists(st.integers(0, 3), min_size=n, max_size=n), min_size=m, max_size=m)
    )
    first = draw(st.integers(3, m - 3))
    labels = [1] * first + [2] * (m - first)
    order = draw(st.permutations(range(m)))
    return Dataset(
        X=np.array(values, dtype=float),
        y=np.array(labels)[list(order)],
        feature_types=tuple(FeatureType.from_flag(f) for f in flags),
    )


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(dataset=tie_heavy_datasets())
def test_random_tie_heavy_datasets_are_identical(dataset: Dataset) -> None:
    assert_identical(fit_context(dataset), reference_steps(dataset))
