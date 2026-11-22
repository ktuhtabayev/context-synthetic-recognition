"""Cross-check against an independent implementation (``tests/golden/independent``).

The independent implementation is dependency-free Python that reproduces the Excel experiment; it
is used here as an oracle on the experiment, on Heart-Disease (270, 13, 2) and on random tie-heavy
datasets, together with ``golden_values.json``. Steps 1–8 agree bit for bit; the agglomerative
algorithm and the meta-algorithm agree exactly in their decisions (TUPLAM, B1/B2, classes) and to
1e-12 in θ/γ and the additional features.
"""

import json
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from context_synthetic_recognition.config import (
    ClassMeans,
)
from context_synthetic_recognition.core.model import CSModel, fit_model
from context_synthetic_recognition.core.synthetic_features.psi import PsiOperator, fit_psi
from context_synthetic_recognition.data import Dataset, FeatureType

from ..conftest import (
    GOLDEN_VALUES,
    excel_experiment,
    independent_core,
    independent_pipeline,
    workbook_additional,
    workbook_config,
)

pytestmark = pytest.mark.golden


def reference_steps(dataset: Dataset) -> dict[str, Any]:
    """Steps 1–8 as the independent ``pipeline.fit`` computes them (without Steps 9–12)."""
    ref = independent_core()
    X, y, types = dataset.X.tolist(), dataset.y.tolist(), list(dataset.type_flags)
    m = len(X)
    lo, hi = ref.scale_params(X, types)
    Z = [ref.unify(x, types, lo, hi) for x in X]
    ks, k_max = ref.admissible_k(y)
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


def assert_identical(model: PsiOperator, expected: dict[str, Any]) -> None:
    """Every Step 1–8 value equal to the independent one — exactly, not to a tolerance."""
    t = model.trace
    assert t.admissible_k.ks == tuple(expected["ks"])
    assert t.admissible_k.k_max == expected["k_max"]
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


def test_golden_values_json(experiment_psi: PsiOperator) -> None:
    with GOLDEN_VALUES.open(encoding="utf-8") as handle:
        golden = json.load(handle)["ref"]
    t = experiment_psi.trace
    assert t.admissible_k.label == golden["klist"]
    assert t.admissible_k.k_max == golden["kmax"]
    assert t.r == golden["r"]
    assert t.omegas.tolist() == golden["w"]
    assert [f.eta[0] for f in t.features] == golden["eta1"]
    assert [f.eta[1] for f in t.features] == golden["eta2"]


def test_the_experiment_is_identical(experiment: Dataset, experiment_psi: PsiOperator) -> None:
    assert_identical(experiment_psi, reference_steps(experiment))


@pytest.mark.slow
def test_heart_disease_270_is_identical(heart270: Dataset) -> None:
    assert_identical(fit_psi(heart270, excel_experiment()), reference_steps(heart270))


@st.composite
def tie_heavy_datasets(draw: st.DrawFn) -> Dataset:
    """Small heterogeneous datasets with few distinct values — many equal distances."""
    m = draw(st.integers(6, 16))
    n = draw(st.integers(2, 5))
    flags = draw(st.lists(st.sampled_from([0, 1]), min_size=n, max_size=n))
    flags[:2] = [1, 0]  # both I and J non-empty, as the independent implementation assumes
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
    assert_identical(fit_psi(dataset, excel_experiment()), reference_steps(dataset))


# --------------------------------------------- Steps 9–12 (agglomerative algorithm, meta-algorithm)


def _reference_fit(dataset: Dataset, **params: Any) -> dict[str, Any]:
    model: dict[str, Any] = independent_pipeline().fit(
        dataset.X.tolist(), dataset.y.tolist(), list(dataset.type_flags), **params
    )
    return model


def assert_same_model(model: CSModel, reference: dict[str, Any], dataset: Dataset) -> None:
    """TUPLAM and the decisions equal; crit and the additional features within 1e-12.

    Floats are compared to a tolerance: numpy's exp may differ from the C library's in the last
    bit on some CPUs, and Python ≥ 3.12 sums with compensation in the reference's final means.
    """
    iterations = reference["agglomerative"]["trace"]["iterations"]
    assert model.tuplam == tuple(reference["tuplam"])
    assert model.agglomerative.crit == pytest.approx([it["crit"] for it in iterations], abs=1e-12)
    additional = np.array(reference["D"], dtype=float).reshape(dataset.m, -1)
    assert np.abs(model.agglomerative.additional - additional).max(initial=0.0) <= 1e-12
    assert model.description.gradations.tolist() == reference["A"]
    predict = independent_pipeline().predict
    training = model.classify_training()
    for i in range(min(dataset.m, 40)):
        expected = predict(reference, dataset.X[i].tolist(), exclude=i)
        assert training.decisions[i] == expected["cls"]
        assert training.meta.scores1[i] == expected["s1"]
        assert training.meta.scores2[i] == expected["s2"]


def test_golden_values_of_the_model(experiment_cs_model: CSModel) -> None:
    with GOLDEN_VALUES.open(encoding="utf-8") as handle:
        golden = json.load(handle)["ref"]
    model = experiment_cs_model
    assert model.agglomerative.label == golden["tuplam"]
    assert list(model.agglomerative.crit) == golden["crit"]
    assert np.abs(model.agglomerative.additional.T - np.array(golden["additional"])).max() <= 1e-15
    training = model.classify_training()
    assert training.meta.scores1.tolist() == golden["s1"]
    assert training.meta.scores2.tolist() == golden["s2"]
    assert training.decisions.tolist() == golden["pred_list"]


@pytest.mark.parametrize("variant", range(4))
def test_the_switch_variants(experiment: Dataset, variant: int) -> None:
    with GOLDEN_VALUES.open(encoding="utf-8") as handle:
        golden = json.load(handle)["variants"][variant]
    agglomerative = workbook_additional(
        class_means=ClassMeans(golden["cent"]), majorizing_passes=golden["pas"]
    )
    model = fit_model(experiment, workbook_config(additional_features=agglomerative))
    assert model.agglomerative.label == golden["T"], golden["label"]
    assert list(model.agglomerative.crit) == pytest.approx(golden["crit"], abs=1e-15)
    decisions = model.classify_training().decisions
    assert np.mean(decisions == experiment.y) == golden["resub"]


@pytest.mark.parametrize("held_out", range(10))
def test_the_leave_one_out_folds(experiment: Dataset, held_out: int) -> None:
    # the whole pipeline re-fitted on the other nine objects; the held-out one classified blind
    with GOLDEN_VALUES.open(encoding="utf-8") as handle:
        golden = json.load(handle)["loo"][held_out]
    training = experiment.subset([i for i in range(experiment.m) if i != held_out])
    model = fit_model(training, excel_experiment())
    assert model.trace.class_sizes == (golden["K1"], golden["K2"])
    assert list(model.trace.admissible_k.ks) == golden["ks"]
    assert model.trace.r == golden["r"]
    assert [u + 1 for u in model.tuplam] == golden["T"]
    result = model.classify(experiment.X[held_out])
    assert result.representation is not None
    assert result.representation.description[0].tolist() == golden["a"]
    assert result.meta.scores1[0] == golden["s1"]
    assert result.meta.scores2[0] == golden["s2"]
    assert result.decisions[0] == golden["cls"]


@pytest.mark.parametrize(("class_means", "passes"), [("running", 2), ("final", 1)])
def test_the_model_is_the_independent_cores(
    experiment: Dataset, class_means: str, passes: int
) -> None:
    agglomerative = workbook_additional(
        class_means=ClassMeans(class_means), majorizing_passes=passes
    )  # type: ignore[arg-type]
    model = fit_model(experiment, workbook_config(additional_features=agglomerative))
    assert_same_model(
        model,
        _reference_fit(experiment, class_means=class_means, majorizing_passes=passes),
        experiment,
    )


@pytest.mark.slow
def test_heart_disease_270_model_is_the_independent_cores(heart270: Dataset) -> None:
    assert_same_model(fit_model(heart270, excel_experiment()), _reference_fit(heart270), heart270)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(dataset=tie_heavy_datasets())
def test_random_datasets_give_the_independent_cores_model(dataset: Dataset) -> None:
    assert_same_model(fit_model(dataset, excel_experiment()), _reference_fit(dataset), dataset)
