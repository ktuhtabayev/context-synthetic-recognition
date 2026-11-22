"""Variants of Ψ: normalization per operator, original features, the zero value, the metaobject."""

import numpy as np
import pytest

from context_synthetic_recognition.config import (
    ExperimentConfig,
    LocalContextConfig,
    OperatorConfig,
    plugin,
)
from context_synthetic_recognition.core.local_context.metrics import METRICS
from context_synthetic_recognition.core.model import fit_model
from context_synthetic_recognition.core.synthetic_features.bit_representations import (
    class_sequence_masks,
)
from context_synthetic_recognition.core.synthetic_features.contributions import (
    contribution_values,
    gradation_counts,
)
from context_synthetic_recognition.core.synthetic_features.membership import (
    indistinguishable,
    membership_table,
)
from context_synthetic_recognition.core.synthetic_features.psi import fit_psi
from context_synthetic_recognition.data import Dataset
from context_synthetic_recognition.errors import ConfigError
from context_synthetic_recognition.export import build_view
from context_synthetic_recognition.export.tables import run_tables
from context_synthetic_recognition.services.runner import run_experiment

from .conftest import excel_experiment


def with_synthetic(config: ExperimentConfig, **update: object) -> ExperimentConfig:
    return config.model_copy(
        update={"synthetic_features": config.synthetic_features.model_copy(update=update)}
    )


# ---------------------------------------------------------------- normalization per operator


def test_an_operator_can_use_its_own_normalization(experiment: Dataset) -> None:
    raw = OperatorConfig(
        label="E", metric=plugin("euclidean"), features="quantitative", normalizer=plugin("none")
    )
    config = excel_experiment().model_copy(
        update={"local_context": LocalContextConfig(operators=(OperatorConfig(label="ρ"), raw))}
    )
    psi = fit_psi(experiment, config)
    zhuravlyov, euclid = psi.trace.operators
    assert (zhuravlyov.normalizer, euclid.normalizer) == ("minmax", "none")
    quantitative = np.flatnonzero(experiment.quantitative)
    X = experiment.X[:, quantitative]
    expected = METRICS.get("euclidean")(X, X, np.ones(X.shape[1], dtype=bool), 10, None)
    assert np.array_equal(euclid.distances, expected)
    # the default operator is unaffected, and a training object keeps its row through represent()
    assert psi.scalings[0] is psi.scaling
    again = psi.represent(experiment.X, exclude=np.arange(experiment.m))
    assert np.array_equal(again.values, psi.trace.values)


def test_without_own_normalizers_every_operator_shares_the_default(experiment: Dataset) -> None:
    psi = fit_psi(experiment, excel_experiment())
    assert all(s is psi.scaling for s in psi.scalings)
    assert all(np.array_equal(r, psi.reference) for r in psi.references)
    assert {o.normalizer for o in psi.trace.operators} == {"minmax"}


# ---------------------------------------------------------------- original nominal features


def test_original_nominal_features_join_psi(experiment: Dataset) -> None:
    base = excel_experiment()
    psi = fit_psi(experiment, with_synthetic(base, original_features=("x₃", "x₁₃")))
    plain = fit_psi(experiment, base)
    originals = [f for f in psi.trace.features if f.is_original]
    assert [f.label for f in originals] == ["x₃", "x₁₃"]
    assert psi.trace.r == plain.trace.r + 2
    x3 = originals[0]
    codes = experiment.X[:, 2].astype(np.int64)
    assert np.array_equal(x3.values, codes)
    assert x3.gradations == tuple(int(v) for v in np.unique(codes))
    assert np.array_equal(x3.alpha, gradation_counts(codes, experiment.class_index, x3.gradations))
    assert x3.eta_at(99) != x3.eta_at(99)  # NaN for a code it does not have
    assert x3.alpha_at(99, 0) == 0
    assert x3.k == 0
    assert x3.operator == -1
    # a training object passed through the new-object path keeps its codes and contributions
    again = psi.represent(experiment.X, exclude=np.arange(experiment.m))
    assert np.array_equal(again.values, psi.trace.values)
    assert np.array_equal(again.contributions, psi.trace.contributions)


def test_an_unseen_code_contributes_nothing() -> None:
    eta = np.array([0.2, -0.1, 0.05])
    values = contribution_values(np.array([1, 3, 7, 4]), eta, (1, 3, 4))
    assert values.tolist() == [0.2, -0.1, 0.0, 0.05]
    # the synthetic values {1, 2} keep their positional meaning
    assert contribution_values(np.array([2, 1]), np.array([0.3, -0.4])).tolist() == [-0.4, 0.3]


def test_all_nominal_features_and_the_whole_model(experiment: Dataset) -> None:
    config = with_synthetic(excel_experiment(), original_features="nominal")
    model = fit_model(experiment, config)
    names = [f.label for f in model.trace.features if f.is_original]
    assert names == [experiment.feature_names[j] for j in np.flatnonzero(~experiment.quantitative)]
    decisions = model.predict(experiment.X, exclude=np.arange(experiment.m))
    assert np.array_equal(decisions, model.classify_training().decisions)


def test_quantitative_and_unknown_features_are_refused(experiment: Dataset) -> None:
    with pytest.raises(ConfigError, match="only nominal features"):
        fit_psi(experiment, with_synthetic(excel_experiment(), original_features=("x₁",)))
    with pytest.raises(ConfigError, match="unknown features"):
        fit_psi(experiment, with_synthetic(excel_experiment(), original_features=("x99",)))
    X = experiment.X.copy()
    X[0, 2] = 1.5
    broken = Dataset(X=X, y=experiment.y, feature_types=experiment.feature_types)
    with pytest.raises(ConfigError, match="non-negative integers"):
        fit_psi(broken, with_synthetic(excel_experiment(), original_features=("x₃",)))


# ---------------------------------------------------------------- the zero value


def test_indistinguishable_gradations_are_found_in_integers() -> None:
    # |K1| = 5, |K2| = 4: gradation 0 holds 5 and 4 objects (shares 1 and 1) → f = 0.5
    y = np.array([0] * 5 + [1] * 4 + [0, 1])
    gradations = np.array([0] * 9 + [1, 2])
    table = membership_table(gradations, y, (6, 5), beta=2)
    assert indistinguishable(table, gradations, (6, 5)).tolist() == [False] * 9 + [False, False]
    table = membership_table(gradations[:9], y[:9], (5, 4), beta=0)
    assert indistinguishable(table, gradations[:9], (5, 4)).all()


def test_the_zero_value_leaves_out_features_that_take_0(heart270: Dataset) -> None:
    config = ExperimentConfig(admissible_k=plugin("range", start=3, stop=41))
    plain = fit_psi(heart270, config).trace
    with_zeros = [f for f in plain.features if f.zeros]
    assert plain.r == 60
    assert len(with_zeros) == 12
    assert (with_zeros[0].operator_label, with_zeros[0].k, with_zeros[0].zeros) == ("ρ", 25, 18)
    zero = fit_psi(heart270, with_synthetic(config, zero_value=True)).trace
    assert zero.r == 48
    assert [s.reason for s in zero.skipped_features] == ["zero"] * 12
    assert [(s.operator_label, s.k) for s in zero.skipped_features] == [
        (f.operator_label, f.k) for f in with_zeros
    ]
    assert all(f.zeros == 0 for f in zero.features)


# ---------------------------------------------------------------- the metaobject


def test_the_metaobject_is_an_output(experiment: Dataset) -> None:
    result = run_experiment(experiment, excel_experiment())
    table = run_tables(build_view(result), keys=["metaobject"])["metaobject"]
    trace = result.model.trace
    assert table.columns == ("Operator", *(f"k = {k}" for k in trace.admissible_k.ks))
    assert [row[0] for row in table.rows] == [o.label for o in trace.operators]
    stability = {(f.operator_label, f.k): f.stability for f in trace.features}
    for row in table.rows:
        for k, value in zip(trace.admissible_k.ks, row[1:], strict=True):
            assert value == stability[(row[0], k)]


# ---------------------------------------------------------------- class-sequence bit masks


def test_class_sequence_masks_read_the_neighbours_classes() -> None:
    # two objects; their neighbours' classes in order: K1, K2, K1 and K2, K1, K2
    order = np.array([[1, 2, 3], [4, 3, 2]])
    classes = np.array([0, 0, 1, 0, 1])
    ks, masks = class_sequence_masks(order, classes, (1, 3))
    assert ks == (1, 3)
    assert masks.tolist() == [[1, 0b101], [0, 0b010]]
    assert class_sequence_masks(order, classes, (63,))[0] == ()


def test_the_class_sequence_masks_of_the_experiment(experiment: Dataset) -> None:
    trace = fit_psi(experiment, excel_experiment()).trace
    assert [s.operator_label for s in trace.class_sequences] == ["ρ", "ρ_I", "ρ_J"]
    first = trace.class_sequences[0]
    assert first.ks == trace.admissible_k.ks
    assert first.masks.shape == (experiment.m, len(first.ks))
    assert np.all((first.stabilities >= 0.5) & (first.stabilities <= 1.0))
    assert first.tables[0].gradations.tolist() == list(range(2 ** first.ks[0]))
    result = run_experiment(experiment, excel_experiment())
    table = run_tables(build_view(result), keys=["class-sequence-masks"])["class-sequence-masks"]
    assert table.rows[0][1] == first.stabilities[0]
