"""The normalizers besides min–max: per-feature maps, rank and unit length (M7)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from context_synthetic_recognition.config import ContextConfig, ExperimentConfig
from context_synthetic_recognition.core.context import fit_context
from context_synthetic_recognition.core.metrics import METRICS, traits_of
from context_synthetic_recognition.core.model import fit_model
from context_synthetic_recognition.core.normalizers import (
    NORMALIZERS,
    RankScaling,
    UnitLengthScaling,
    decimal_power,
    decimal_scaling,
    max_abs,
    minmax,
    rank,
    robust,
    unit_length,
    zscore,
)
from context_synthetic_recognition.data import Dataset
from context_synthetic_recognition.errors import ConfigError
from context_synthetic_recognition.export import build_view
from context_synthetic_recognition.export.excel import build_workbook
from context_synthetic_recognition.export.tables import run_tables
from context_synthetic_recognition.services.runner import run_experiment

QUANT = np.array([True, False, True])
X = np.array(
    [
        [10.0, 7.0, 5.0],
        [30.0, 8.0, 5.0],
        [20.0, 7.0, 5.0],
        [60.0, 9.0, 5.0],
    ]
)
"""x₁ quantitative, x₂ nominal, x₃ quantitative without spread."""
FITTED = ("minmax", "z-score", "robust", "max-abs", "decimal-scaling", "rank")
"""The normalizers that estimate constants per feature on the training sample."""
EVERY = (*FITTED, "unit-length", "none")


def fit(name: str, values: np.ndarray = X, mask: np.ndarray = QUANT, **params: object):
    return NORMALIZERS.get(name)(values, mask, NORMALIZERS.make_params(name, params))  # type: ignore[arg-type]


# ---------------------------------------------------------------- registry


def test_the_registry_lists_every_normalizer() -> None:
    assert NORMALIZERS.names() == [
        "minmax",
        "none",
        "z-score",
        "robust",
        "max-abs",
        "decimal-scaling",
        "rank",
        "unit-length",
    ]
    assert NORMALIZERS.get("zscore") is zscore
    assert NORMALIZERS.get("standard") is zscore
    assert NORMALIZERS.get("median-iqr") is robust
    assert NORMALIZERS.get("maxabs") is max_abs
    assert NORMALIZERS.get("decimal") is decimal_scaling
    assert NORMALIZERS.get("quantile") is rank
    assert NORMALIZERS.get("unit-norm") is unit_length
    assert all(info.summary for info in NORMALIZERS)
    with pytest.raises(ConfigError, match="ddof"):
        NORMALIZERS.make_params("z-score", {"ddof": 2})
    with pytest.raises(ConfigError, match="takes no parameters"):
        NORMALIZERS.make_params("robust", {"quantiles": [0.1, 0.9]})


# ---------------------------------------------------------------- definitions


def test_z_score() -> None:
    scaling = zscore(X, QUANT)
    Z = scaling.transform(X)
    assert scaling.statistics["mean"][0] == 30.0
    assert scaling.statistics["std"][0] == pytest.approx(np.std([10, 30, 20, 60], ddof=1))
    assert Z[:, 0] == pytest.approx((X[:, 0] - 30.0) / np.std(X[:, 0], ddof=1))
    assert Z[:, 0].mean() == pytest.approx(0.0)
    assert Z[:, 0].std(ddof=1) == pytest.approx(1.0)
    population = fit("z-score", ddof=0)
    assert population.transform(X)[:, 0].std() == pytest.approx(1.0)
    assert population.statistics["std"][0] == pytest.approx(np.std([10, 30, 20, 60]))
    # one training object: no spread, nothing to estimate — and no warning from numpy
    assert zscore(X[:1], QUANT).transform(X)[:, 0].tolist() == [0.0] * 4


def test_z_score_agrees_with_scikit_learn() -> None:
    preprocessing = pytest.importorskip("sklearn.preprocessing")
    rng = np.random.default_rng(2)
    data = rng.normal(size=(15, 4)) * [1.0, 10.0, 0.1, 100.0]
    mask = np.ones(4, dtype=bool)
    ours = fit("z-score", data, mask, ddof=0).transform(data)
    assert ours == pytest.approx(preprocessing.StandardScaler().fit_transform(data))
    assert robust(data, mask).transform(data) == pytest.approx(
        preprocessing.RobustScaler().fit_transform(data)
    )
    assert max_abs(data, mask).transform(data) == pytest.approx(
        preprocessing.MaxAbsScaler().fit_transform(data)
    )
    assert minmax(data, mask).transform(data) == pytest.approx(
        preprocessing.MinMaxScaler().fit_transform(data)
    )
    lengths = np.linalg.norm(unit_length(data, mask).transform(data), axis=1)
    assert lengths == pytest.approx(np.ones(15))
    assert unit_length(data, mask).transform(data) == pytest.approx(preprocessing.normalize(data))


def test_robust() -> None:
    scaling = robust(X, QUANT)
    # quartiles of 10, 20, 30, 60 (linear interpolation): 17.5, 25, 37.5
    assert [scaling.statistics[key][0] for key in ("Q1", "median", "Q3")] == [17.5, 25.0, 37.5]
    assert scaling.transform(X)[:, 0].tolist() == [-0.75, 0.25, -0.25, 1.75]
    # more than half of the values equal: the interquartile range is 0, the range is used
    flat = np.array([[1.0], [1.0], [1.0], [1.0], [9.0]])
    one = np.ones(1, dtype=bool)
    fallback = robust(flat, one)
    assert not fallback.constant[0]
    assert fallback.scale[0] == 8.0
    assert fallback.transform(flat)[:, 0].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_max_abs_and_decimal_scaling() -> None:
    data = np.array([[-80.0, 986.0, 0.05, 100.0], [20.0, -917.0, 0.02, 7.0]])
    mask = np.ones(4, dtype=bool)
    scaling = max_abs(data, mask)
    assert scaling.statistics["max |x|"].tolist() == [80.0, 986.0, 0.05, 100.0]
    assert scaling.transform(data)[:, 0].tolist() == [-1.0, 0.25]
    decimal = decimal_scaling(data, mask)
    # the smallest j with max|x| / 10ʲ < 1: 80 → 2, 986 → 3, 0.05 → −1, 100 → 3
    assert decimal.statistics["power j"].tolist() == [2.0, 3.0, -1.0, 3.0]
    assert decimal.transform(data) == pytest.approx(
        np.array([[-0.8, 0.986, 0.5, 0.1], [0.2, -0.917, 0.2, 0.007]])
    )
    assert np.abs(decimal.transform(data)).max() < 1.0
    assert decimal_power(0.0) == 0


@settings(max_examples=200, deadline=None)
@given(peak=st.floats(1e-12, 1e12, allow_nan=False))
def test_decimal_power_is_the_smallest_power_that_brings_the_value_below_one(peak: float) -> None:
    j = decimal_power(peak)
    assert peak / 10.0**j < 1.0 <= peak / 10.0 ** (j - 1)


def test_rank() -> None:
    scaling = rank(X, QUANT)
    assert isinstance(scaling, RankScaling)
    assert scaling.transform(X)[:, 0].tolist() == [0.0, 2 / 3, 1 / 3, 1.0]
    assert scaling.knots[0].tolist() == [10.0, 20.0, 30.0, 60.0]
    assert scaling.knots[1].size == 0  # a nominal feature has no knots
    # new values: interpolated between training values, the nearest level outside the range
    new = np.array([[15.0, 7.0, 5.0], [45.0, 7.0, 5.0], [5.0, 7.0, 5.0], [99.0, 7.0, 5.0]])
    assert scaling.transform(new)[:, 0] == pytest.approx([1 / 6, 5 / 6, 0.0, 1.0])
    # equal values share their mean rank: 1, 2.5, 2.5, 4 of 4
    tied = rank(np.array([[3.0], [5.0], [5.0], [9.0]]), np.ones(1, dtype=bool))
    assert tied.transform(np.array([[3.0], [5.0], [9.0]]))[:, 0].tolist() == [0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="read-only"):
        scaling.knots[0][0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        scaling.levels[0][0] = 1.0


def test_rank_never_puts_a_new_value_past_a_training_value() -> None:
    # found by the property test below: interpolating between −49.109375 (level 0) and 0
    # (level 0.4) just below 0 rounds to 0.4000000000000001 — above the level of 0 itself
    train = np.array([[1.0], [0.0], [1.0], [0.0], [-49.109375], [0.0]])
    scaling = rank(train, np.ones(1, dtype=bool))
    assert scaling.knots[0].tolist() == [-49.109375, 0.0, 1.0]
    assert scaling.levels[0].tolist() == [0.0, 0.4, 0.9]
    just_below, just_above = -8.23475053e-32, 8.23475053e-32
    Z = scaling.transform(np.array([[just_below], [0.0], [just_above], [-60.0], [1.0], [7.0]]))
    assert Z[:, 0].tolist() == [0.4, 0.4, 0.4, 0.0, 0.9, 0.9]
    assert np.interp(just_below, scaling.knots[0], scaling.levels[0]) > 0.4  # what was returned
    # between two training values the level stays between theirs
    inside = scaling.transform(np.linspace(-49.109375, 1.0, 1001)[:, None])[:, 0]
    assert np.all(np.diff(inside) >= 0.0)
    assert (inside.min(), inside.max()) == (0.0, 0.9)


def test_unit_length() -> None:
    data = np.array([[3.0, 9.0, 4.0], [0.0, 2.0, 0.0], [5.0, 1.0, 12.0]])
    scaling = unit_length(data, QUANT)
    assert isinstance(scaling, UnitLengthScaling)
    assert scaling.transform(data).tolist() == [
        [0.6, 9.0, 0.8],
        [0.0, 2.0, 0.0],  # nothing to scale: the zero vector stays
        [5 / 13, 1.0, 12 / 13],
    ]
    assert dict(scaling.statistics) == {}  # nothing is estimated on the training sample


# ---------------------------------------------------------------- what every normalizer keeps


@pytest.mark.parametrize("name", EVERY)
def test_nominal_codes_are_never_changed(name: str) -> None:
    Z = fit(name).transform(X)
    assert Z[:, 1].tolist() == [7.0, 8.0, 7.0, 9.0]
    assert fit(name).quantitative.tolist() == QUANT.tolist()


@pytest.mark.parametrize("name", FITTED)
def test_a_feature_without_spread_is_mapped_to_zero(name: str) -> None:
    scaling = fit(name)
    assert scaling.constant.tolist() == [False, False, True]
    assert scaling.transform(X)[:, 2].tolist() == [0.0] * 4
    # as with min–max in the workbook, also for a new object with another value
    assert scaling.transform(np.array([[10.0, 7.0, 123.0]]))[0, 2] == 0.0
    # min and max of the training objects come first, for every normalizer
    assert list(scaling.statistics)[:2] == ["min", "max"]
    assert (scaling.statistics["min"][0], scaling.statistics["max"][0]) == (10.0, 60.0)
    assert np.isnan(scaling.statistics["min"][1])


@pytest.mark.parametrize("name", EVERY)
def test_scalings_are_immutable_and_check_the_width(name: str) -> None:
    scaling = fit(name)
    with pytest.raises(ValueError, match="read-only"):
        scaling.offset[0] = 1.0
    for values in scaling.statistics.values():
        with pytest.raises(ValueError, match="read-only"):
            values[0] = 1.0
    with pytest.raises(ValueError, match="the scaling was fitted on 3"):
        scaling.transform(np.ones((1, 2)))
    original = X.copy()
    scaling.transform(X)
    assert np.array_equal(X, original)  # the caller's values are not touched


@pytest.mark.parametrize("name", EVERY)
def test_no_quantitative_feature_at_all(name: str) -> None:
    codes = np.array([[1.0, 2.0], [3.0, 2.0], [1.0, 1.0]])
    scaling = fit(name, codes, np.zeros(2, dtype=bool))
    assert scaling.transform(codes).tolist() == codes.tolist()


training = arrays(np.float64, (6, 3), elements=st.floats(-50, 50, allow_nan=False, width=32))
new_objects = arrays(np.float64, (4, 3), elements=st.floats(-80, 80, allow_nan=False, width=32))


@settings(max_examples=60, deadline=None)
@given(train=training, new=new_objects)
@pytest.mark.parametrize("name", EVERY)
def test_an_object_is_mapped_the_same_way_alone_and_among_others(
    name: str, train: np.ndarray, new: np.ndarray
) -> None:
    # the new-object path maps one object at a time; it must get the training objects' values
    scaling = fit(name, train, np.array([True, True, False]))
    together = scaling.transform(np.vstack([train, new]))
    for i, row in enumerate(np.vstack([train, new])):
        assert np.array_equal(scaling.transform(row[None, :])[0], together[i])
    assert np.all(np.isfinite(together))


@settings(max_examples=60, deadline=None)
@given(train=training, new=new_objects)
@pytest.mark.parametrize("name", FITTED)
def test_per_feature_normalizers_keep_the_order_of_the_values(
    name: str, train: np.ndarray, new: np.ndarray
) -> None:
    mask = np.ones(3, dtype=bool)
    scaling = fit(name, train, mask)
    values = np.vstack([train, new])
    Z = scaling.transform(values)
    for j in range(3):
        order = np.argsort(values[:, j], kind="stable")
        assert np.all(np.diff(Z[order, j]) >= 0.0)
    # the constants come from the training objects alone
    assert np.array_equal(fit(name, train, mask).transform(new), Z[len(train) :])


@settings(max_examples=40, deadline=None)
@given(train=training)
@pytest.mark.parametrize("name", ["heom", "gower"])
def test_range_normalized_metrics_do_not_depend_on_an_affine_normalizer(
    name: str, train: np.ndarray
) -> None:
    mask = np.array([True, True, False])
    codes = train.copy()
    codes[:, 2] = np.round(codes[:, 2])
    metric, step = METRICS.get(name), traits_of(METRICS.get(name)).fit
    assert step is not None
    results = []
    for normalizer in ("minmax", "z-score", "max-abs", "decimal-scaling", "none"):
        Z = fit(normalizer, codes, mask).transform(codes)
        results.append(metric(Z, Z, mask, 10, step(Z, mask, None)))
    for other in results[1:]:
        assert other == pytest.approx(results[0], abs=1e-6)


# ---------------------------------------------------------------- the pipeline and the exports


def config_with(normalizer: str) -> ExperimentConfig:
    return ExperimentConfig.model_validate({"preprocessing": {"normalizer": normalizer}})


@pytest.mark.parametrize("name", [n for n in EVERY if n != "minmax"])
def test_every_normalizer_runs_through_the_model(experiment: Dataset, name: str) -> None:
    model = fit_model(experiment, config_with(name))
    trace = model.context.trace
    assert trace.scaling.normalizer == name
    assert np.array_equal(
        trace.normalized[:, experiment.nominal], experiment.X[:, experiment.nominal]
    )
    # ρ_J does not see the quantitative features: its synthetic features are the workbook's
    workbook = fit_context(experiment).trace
    assert np.array_equal(trace.operators[2].distances, workbook.operators[2].distances)
    # Theorem: the new-object path reproduces the training rows with the training constants
    again = model.context.represent(experiment.X, exclude=np.arange(experiment.m))
    assert np.array_equal(again.normalized, trace.normalized)
    assert np.array_equal(again.values, np.column_stack([f.values for f in trace.features]))


def test_the_default_is_still_the_workbook(experiment: Dataset) -> None:
    assert ExperimentConfig().preprocessing.normalizer.name == "minmax"
    assert [o.metric.name for o in ContextConfig().operators] == ["zhuravlyov"] * 3
    trace = fit_context(experiment).trace
    assert list(trace.scaling.statistics) == ["min", "max"]
    assert trace.normalized[:, experiment.quantitative].min() == 0.0
    assert trace.normalized[:, experiment.quantitative].max() == 1.0


def test_the_exports_describe_another_normalizer_and_metric(experiment: Dataset) -> None:
    config = ExperimentConfig.model_validate(
        {
            "preprocessing": {"normalizer": "robust"},
            "context": {
                "operators": [
                    {"label": "H", "metric": "heom"},
                    {
                        "label": "E",
                        "metric": {"name": "minkowski", "params": {"p": 4}},
                        "features": "quantitative",
                    },
                    {"label": "ρ_J", "features": "nominal"},
                ]
            },
        }
    )
    view = build_view(run_experiment(experiment, config))
    tables = run_tables(view)
    features = tables["features"]
    assert features.columns == (
        "Feature",
        "Type",
        "Flag (1 = I)",
        "min",
        "max",
        "median",
        "Q1",
        "Q3",
        "Distinct values",
    )
    assert "scale unification “robust”" in features.note
    assert features.rows[1][3:8] == (None,) * 5  # x₂ is nominal
    assert "“robust”" in tables["normalized"].title
    assert "median" in tables["normalized"].note
    assert "Metric heom on 13 feature(s)" in tables["distances-h"].note

    workbook, _ = build_workbook(view)
    sheet = workbook["Normalized Dataset"]
    labels = [sheet.cell(row, 1).value for row in range(15, 22)]
    assert labels[:6] == ["Type", "min", "max", "median", "Q1", "Q3"]
    notes = " ".join(str(sheet.cell(row, 1).value) for row in range(21, 26))
    assert "“robust”" in notes
    assert "x′ = (x − median) / (Q3 − Q1)" in notes
    assert "same training constants" in notes
    assert "Distances" in workbook.sheetnames  # not the Zhuravlyov metric alone


def test_the_default_export_is_unchanged(experiment_view) -> None:
    tables = run_tables(experiment_view)
    assert tables["features"].columns == (
        "Feature",
        "Type",
        "Flag (1 = I)",
        "min",
        "max",
        "Distinct values",
    )
    assert (
        tables["features"].title == "Features: type, training minimum and maximum, distinct values"
    )
    assert tables["normalized"].note == (
        "x′ = (x − min)/(max − min) for j ∈ I with the training min and max (Step 1)."
    )
