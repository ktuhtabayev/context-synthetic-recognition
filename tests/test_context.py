"""Steps 1–8 end to end: fit_context, the trace, and represent (Theorem, no label leakage)."""

import dataclasses
import inspect

import numpy as np
import pytest

from context_synthetic_recognition.config import (
    LocalContextConfig,
    OperatorConfig,
    SyntheticFeaturesConfig,
    plugin,
)
from context_synthetic_recognition.core.plugins import plugin_problems
from context_synthetic_recognition.core.synthetic_features.bit_representations import MAX_CHAIN_BITS
from context_synthetic_recognition.core.synthetic_features.psi import PsiOperator, fit_psi
from context_synthetic_recognition.data import Dataset, FeatureType
from context_synthetic_recognition.errors import ConfigError, DatasetError, ModelUndefinedError

from .conftest import article_switches, excel_experiment, workbook_config

QUANT, NOM = FeatureType.QUANTITATIVE, FeatureType.NOMINAL


def test_the_experiment_trace(experiment_psi: PsiOperator) -> None:
    t = experiment_psi.trace
    assert t.admissible_k.ks == (3, 5)
    assert t.r == 6
    assert [(f.name, f.operator_label, f.k) for f in t.features] == [
        ("a₁", "ρ", 3),
        ("a₂", "ρ", 5),
        ("a₃", "ρ_I", 3),
        ("a₄", "ρ_I", 5),
        ("a₅", "ρ_J", 3),
        ("a₆", "ρ_J", 5),
    ]
    assert t.values.shape == (10, 6)
    assert t.contributions.shape == (10, 6)
    assert t.class_sizes == (4, 6)
    assert t.classes == (1, 2)
    assert t.features[0].chi2.tolist() == (3 - t.features[0].chi1).tolist()
    assert len(t.majority_chains) == 3
    assert t.majority_chains[0].masks.tolist() == [3, 0, 0, 0, 0, 0, 3, 3, 3, 3]
    assert t.metaobject.tolist() == [f.stability for f in t.features]
    assert t.skipped_operators == ()


def test_the_trace_is_immutable(experiment_psi: PsiOperator) -> None:
    t = experiment_psi.trace
    for array in (t.normalized, t.operators[0].distances, t.features[0].values, t.class_index):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.features[0].omega = 0.0  # type: ignore[misc]


def test_both_presets_share_steps_1_to_8(experiment: Dataset) -> None:
    workbook = fit_psi(experiment, excel_experiment()).trace
    article = fit_psi(experiment, article_switches()).trace
    assert np.array_equal(workbook.contributions, article.contributions)
    assert np.array_equal(workbook.informativeness, article.informativeness)


# ---------------------------------------------------------------- the new-object path (Theorem)


def test_training_objects_left_out_of_their_context_reproduce_their_rows(
    experiment: Dataset, experiment_psi: PsiOperator
) -> None:
    context = experiment_psi.represent(experiment.X, exclude=np.arange(experiment.m))
    t = experiment_psi.trace
    assert np.array_equal(context.values, t.values)
    assert np.array_equal(context.contributions, t.contributions)
    assert np.array_equal(context.chi1, np.column_stack([f.chi1 for f in t.features]))
    for o, operator in enumerate(t.operators):
        assert np.array_equal(context.distances[o], operator.distances)
        assert np.array_equal(context.orders[o], operator.order)


def test_represent_has_no_label_argument() -> None:
    assert list(inspect.signature(PsiOperator.represent).parameters) == ["self", "X", "exclude"]


def test_a_left_out_object_never_reads_its_own_class(
    experiment: Dataset, experiment_psi: PsiOperator
) -> None:
    for i in range(experiment.m):
        poisoned_classes = experiment_psi.reference_classes.copy()
        poisoned_classes[i] = 1 - poisoned_classes[i]
        poisoned = dataclasses.replace(experiment_psi, reference_classes=poisoned_classes)
        honest = experiment_psi.represent(experiment.X[i], exclude=[i])
        attacked = poisoned.represent(experiment.X[i], exclude=[i])
        assert np.array_equal(honest.values, attacked.values)


def test_a_new_object_uses_only_its_neighbours_classes(
    experiment: Dataset, experiment_psi: PsiOperator
) -> None:
    new = np.array([[58.0, 1, 3, 125, 250, 0, 2, 150, 0, 1.0, 2, 1, 7]])
    context = experiment_psi.represent(new)
    assert context.values.shape == (1, 6)
    assert set(np.unique(context.values)) <= {1, 2}
    assert context.orders[0].shape == (1, 10)  # all training objects, none excluded
    # the same values whatever the new object's class would be: there is no way to pass it
    assert np.array_equal(experiment_psi.represent(new[0]).values, context.values)
    order = context.orders[0][0]
    chi1 = int((experiment_psi.reference_classes[order[:3]] == 0).sum())
    assert context.chi1[0, 0] == chi1
    assert context.values[0, 0] == (1 if chi1 > 1 else 2)


def test_new_objects_keep_values_outside_the_training_range(
    experiment_psi: PsiOperator,
) -> None:
    new = np.array([[90.0, 1, 3, 200, 600, 0, 2, 200, 0, 6.0, 2, 4, 7]])
    context = experiment_psi.represent(new)
    assert context.normalized[0, 0] > 1.0  # age 90 > training max 74, not clipped


# ---------------------------------------------------------------- data the model is undefined for


def dataset(y: list[int], X: np.ndarray | None = None) -> Dataset:
    m = len(y)
    values = X if X is not None else np.column_stack([np.arange(m, dtype=float), np.arange(m) % 3])
    return Dataset(X=values, y=np.array(y), feature_types=(QUANT, NOM))


def test_one_class_three_classes_and_tiny_classes() -> None:
    with pytest.raises(ModelUndefinedError, match="only one class"):
        fit_psi(dataset([1, 1, 1, 1]), excel_experiment())
    with pytest.raises(DatasetError, match=r"3 classes .*\(ADR-010\)"):
        fit_psi(dataset([1, 2, 3, 1, 2, 3]), excel_experiment())
    with pytest.raises(ModelUndefinedError, match="no k is admissible"):
        fit_psi(dataset([1, 2, 2, 2, 2]), excel_experiment())
    two = fit_psi(
        dataset([1, 1, 2, 2, 2]), excel_experiment()
    ).trace  # min|Kᵢ| = 2 → k = 1 (ADR-005)
    assert two.admissible_k.ks == (1,)
    assert two.r == 3


def test_skipped_operator_shrinks_r() -> None:
    all_quantitative = Dataset(
        X=np.arange(16, dtype=float).reshape(8, 2),
        y=[1, 1, 1, 1, 2, 2, 2, 2],
        feature_types=(QUANT, QUANT),
    )
    t = fit_psi(all_quantitative, excel_experiment()).trace
    assert [o.label for o in t.operators] == ["ρ", "ρ_I"]
    assert [s.label for s in t.skipped_operators] == ["ρ_J"]
    assert t.r == 2 * t.admissible_k.count


def test_many_admissible_k_have_no_bit_masks(heart270: Dataset) -> None:
    t = fit_psi(heart270, excel_experiment()).trace
    assert t.admissible_k.count == 118 > MAX_CHAIN_BITS
    assert t.majority_chains == ()
    assert t.r == 354
    capped = workbook_config(admissible_k=plugin("twice-min-class", k_max_cap=9))
    small = fit_psi(heart270, capped).trace
    assert small.admissible_k.ks == (3, 5, 7, 9)
    assert len(small.majority_chains) == 3
    assert small.majority_chains[0].membership.gradations.size == 16


def test_invalid_plugins_are_reported() -> None:
    config = workbook_config(
        admissible_k=plugin("twice-min-class", step=3),
        synthetic_features=SyntheticFeaturesConfig(encoder=plugin("mu")),
        local_context=LocalContextConfig(
            operators=(OperatorConfig(label="ρ", metric=plugin("zhuravlyov", p=2)),)
        ),
    )
    problems = plugin_problems(config)
    assert len(problems) == 3
    assert problems[0].startswith(
        "local_context.operators[0] (ρ).metric: metrics 'zhuravlyov' takes"
    )
    assert problems[1].startswith(
        "admissible_k: k strategies 'twice-min-class': invalid parameters"
    )
    assert problems[2].startswith("synthetic_features.encoder: Unknown encoders 'mu'")
    assert plugin_problems(excel_experiment()) == []
    with pytest.raises(ConfigError, match="the step must be even"):
        fit_psi(
            dataset([1, 1, 1, 2, 2, 2]),
            workbook_config(admissible_k=plugin("twice-min-class", step=3)),
        )


def test_encoders_must_return_gradations(experiment: Dataset) -> None:
    from context_synthetic_recognition.core.synthetic_features.encoders import ENCODERS

    ENCODERS.add("broken", lambda chi1, k, params: chi1)
    try:
        config = workbook_config(
            synthetic_features=SyntheticFeaturesConfig(encoder=plugin("broken"))
        )
        with pytest.raises(ValueError, match="gradations 1 or 2"):
            fit_psi(experiment, config)
    finally:
        ENCODERS._remove("broken")
