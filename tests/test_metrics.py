"""The metrics of the base operators besides the Zhuravlyov metric, traits and fit steps (M7)."""

import inspect

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from context_synthetic_recognition.config import (
    ContextConfig,
    ExperimentConfig,
    OperatorConfig,
    plugin,
)
from context_synthetic_recognition.core.context import fit_context
from context_synthetic_recognition.core.metrics import (
    METRICS,
    MetricTraits,
    Ranges,
    Whitening,
    fit_ranges,
    fit_whitening,
    round_distances,
    traits,
    traits_of,
    zhuravlyov,
)
from context_synthetic_recognition.core.model import fit_model
from context_synthetic_recognition.core.operators import resolve_operators
from context_synthetic_recognition.data import Dataset
from context_synthetic_recognition.errors import ConfigError
from context_synthetic_recognition.services.runner import run_experiment

HETEROGENEOUS = ("zhuravlyov", "weighted-zhuravlyov", "heom", "gower")
QUANTITATIVE = (
    "manhattan",
    "euclidean",
    "chebyshev",
    "minkowski",
    "canberra",
    "cosine",
    "mahalanobis",
)
NOMINAL = ("hamming",)
NAMES = ("x₁", "x₂", "x₃")
QUANT = np.array([True, False, True])


def distances(name: str, A: np.ndarray, B: np.ndarray, mask: np.ndarray, **params: object):
    """Distances of a registered metric, fitted on ``B`` (the training side) if it has a fit."""
    metric = METRICS.get(name)
    values = METRICS.make_params(name, params)  # type: ignore[arg-type]
    fit = traits_of(metric).fit
    return metric(A, B, mask, 10, values if fit is None else fit(B, mask, values))


def operators_of(metric: str, features: object = "all", **params: object) -> ContextConfig:
    spec = plugin(metric, **params)  # type: ignore[arg-type]
    return ContextConfig(
        operators=(OperatorConfig(label="ρ*", metric=spec, features=features),)  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------- registry and traits


def test_the_registry_lists_every_metric_with_its_domain() -> None:
    assert METRICS.names() == [
        "zhuravlyov",
        "weighted-zhuravlyov",
        "heom",
        "gower",
        "manhattan",
        "euclidean",
        "chebyshev",
        "minkowski",
        "canberra",
        "cosine",
        "mahalanobis",
        "hamming",
    ]
    domains = {info.name: traits_of(info.obj).domain for info in METRICS}
    assert {name for name, domain in domains.items() if domain == "any"} == set(HETEROGENEOUS)
    assert {n for n, domain in domains.items() if domain == "quantitative"} == set(QUANTITATIVE)
    assert {name for name, domain in domains.items() if domain == "nominal"} == set(NOMINAL)
    fitted = {info.name for info in METRICS if traits_of(info.obj).fit is not None}
    assert fitted == {"heom", "gower", "mahalanobis"}
    assert METRICS.get("l2") is METRICS.get("euclidean")
    assert METRICS.get("city-block") is METRICS.get("manhattan")
    assert METRICS.get("overlap") is METRICS.get("hamming")
    assert all(info.summary for info in METRICS)


def test_no_metric_reads_a_class() -> None:
    # HVDM, whose nominal part is built from the classes of the training objects, is not offered
    assert "hvdm" not in METRICS
    for info in METRICS:
        fit = traits_of(info.obj).fit
        for function in (info.obj, fit) if fit else (info.obj,):
            names = set(inspect.signature(function).parameters)
            assert not names & {"y", "labels", "classes", "class_index"}, info.name


def test_a_metric_without_declared_traits_is_heterogeneous_and_needs_no_fit() -> None:
    def plain(A, B, quantitative, decimals, params):  # a plug-in written before M7
        return np.zeros((len(A), len(B)))

    assert traits_of(plain) == MetricTraits()

    @traits(domain="nominal")
    def declared(A, B, quantitative, decimals, params):
        return np.zeros((len(A), len(B)))

    assert traits_of(declared) == MetricTraits("nominal", None, None)


# ---------------------------------------------------------------- definitions

A = np.array([[0.0, 0.5, 1.0], [0.25, 0.25, 0.0]])
B = np.array([[1.0, 0.5, 0.0], [0.25, 0.75, 0.5], [0.0, 0.0, 0.0]])
ALL_I = np.ones(3, dtype=bool)


def test_metrics_on_quantitative_features_by_hand() -> None:
    assert distances("manhattan", A, B, ALL_I).tolist() == [[2.0, 1.0, 1.5], [1.0, 1.0, 0.5]]
    assert distances("euclidean", A, B, ALL_I)[0].tolist() == [
        round(2**0.5, 10),
        round((0.0625 + 0.0625 + 0.25) ** 0.5, 10),
        round(1.25**0.5, 10),
    ]
    assert distances("chebyshev", A, B, ALL_I).tolist() == [[1.0, 0.5, 1.0], [0.75, 0.5, 0.25]]
    assert distances("minkowski", A, B, ALL_I, p=1).tolist() == [[2.0, 1.0, 1.5], [1.0, 1.0, 0.5]]
    assert distances("minkowski", A, B, ALL_I)[0, 0] == round(2 ** (1 / 3), 10)  # p = 3
    # |0 − 1|/1 + |0.5 − 0.5|/1 + |1 − 0|/1; a feature with 0 on both sides contributes 0
    assert distances("canberra", A, B, ALL_I)[0, 0] == 2.0
    assert distances("canberra", A, B, ALL_I)[1, 2] == 2.0  # 0.25/0.25 + 0.25/0.25 + 0/0 → 0


@pytest.mark.parametrize(
    ("name", "scipy_name", "options"),
    [
        ("manhattan", "cityblock", {}),
        ("euclidean", "euclidean", {}),
        ("chebyshev", "chebyshev", {}),
        ("minkowski", "minkowski", {"p": 3.0}),
        ("minkowski", "minkowski", {"p": 1.5}),
        ("canberra", "canberra", {}),
        ("cosine", "cosine", {}),
    ],
)
def test_metrics_agree_with_scipy(name: str, scipy_name: str, options: dict[str, float]) -> None:
    distance = pytest.importorskip("scipy.spatial.distance")
    rng = np.random.default_rng(3)
    X, Y = rng.normal(size=(7, 5)), rng.normal(size=(9, 5))
    ours = distances(name, X, Y, np.ones(5, dtype=bool), **options)
    assert ours == pytest.approx(distance.cdist(X, Y, scipy_name, **options), abs=1e-9)


def test_hamming_counts_the_differing_nominal_features() -> None:
    distance = pytest.importorskip("scipy.spatial.distance")
    rng = np.random.default_rng(3)
    X = rng.integers(0, 3, size=(8, 6)).astype(float)
    mask = np.zeros(6, dtype=bool)
    ours = distances("hamming", X, X, mask)
    assert ours.tolist() == (distance.cdist(X, X, "hamming") * 6).round().tolist()
    assert np.array_equal(ours, zhuravlyov(X, X, mask, 10, None))  # ρ_J is the same count


def test_manhattan_on_set_i_and_hamming_on_set_j_are_the_workbook_operators(
    experiment: Dataset,
) -> None:
    config = ExperimentConfig(
        context=ContextConfig(
            operators=(
                OperatorConfig(label="M", metric=plugin("manhattan"), features="quantitative"),
                OperatorConfig(label="H", metric=plugin("hamming"), features="nominal"),
            )
        )
    )
    ours = fit_context(experiment, config).trace.operators
    workbook = fit_context(experiment).trace.operators  # ρ, ρ_I, ρ_J
    assert np.array_equal(ours[0].distances, workbook[1].distances)
    assert np.array_equal(ours[1].distances, workbook[2].distances)


def test_mahalanobis_uses_the_training_covariance() -> None:
    distance = pytest.importorskip("scipy.spatial.distance")
    rng = np.random.default_rng(3)
    train = rng.normal(size=(30, 4)) @ rng.normal(size=(4, 4))
    new = rng.normal(size=(5, 4))
    mask = np.ones(4, dtype=bool)
    inverse = np.linalg.inv(np.cov(train, rowvar=False))
    assert distances("mahalanobis", new, train, mask) == pytest.approx(
        distance.cdist(new, train, "mahalanobis", VI=inverse), abs=1e-9
    )


def test_mahalanobis_with_a_singular_covariance_uses_the_pseudo_inverse() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(size=(12, 2))
    train = np.column_stack([base, base[:, 0] * 2.0, np.full(12, 7.0)])  # rank 2 of 4
    mask = np.ones(4, dtype=bool)
    state = fit_whitening(train, mask)
    assert isinstance(state, Whitening)
    assert state.matrix.shape == (4, 2)
    pseudo = np.linalg.pinv(np.cov(train, rowvar=False))
    difference = train[:, None, :] - train[None, :, :]
    expected = np.sqrt(np.einsum("abi,ij,abj->ab", difference, pseudo, difference))
    assert distances("mahalanobis", train, train, mask) == pytest.approx(expected, abs=1e-8)
    # nothing to measure: no feature, one object, or no variance at all
    assert fit_whitening(np.empty((5, 0)), np.zeros(0, dtype=bool)).matrix.shape == (0, 0)
    assert fit_whitening(np.ones((1, 3)), mask[:3]).matrix.shape == (3, 0)
    assert (
        distances("mahalanobis", np.ones((2, 3)), np.ones((4, 3)), mask[:3]).tolist()
        == [[0.0] * 4] * 2
    )


def test_cosine_of_zero_vectors() -> None:
    zero, one = np.zeros((1, 2)), np.array([[0.0, 3.0]])
    mask = np.ones(2, dtype=bool)
    assert distances("cosine", zero, zero, mask)[0, 0] == 0.0
    assert distances("cosine", zero, one, mask)[0, 0] == 1.0
    assert distances("cosine", one, 5 * one, mask)[0, 0] == 0.0  # the same direction
    assert distances("cosine", one, -one, mask)[0, 0] == 2.0


def test_heom_and_gower_by_hand() -> None:
    train = np.array([[0.0, 1.0, 10.0], [4.0, 2.0, 10.0], [2.0, 1.0, 10.0]])
    new = np.array([[1.0, 2.0, 99.0], [8.0, 1.0, 10.0]])
    ranges = fit_ranges(train, QUANT)
    assert isinstance(ranges, Ranges)
    assert ranges.spread.tolist() == [4.0, np.inf, np.inf]  # x₂ nominal, x₃ without spread
    with pytest.raises(ValueError, match="read-only"):
        ranges.spread[0] = 1.0
    # x₁: |1 − 0|/4, |1 − 4|/4, |1 − 2|/4; x₂: [2 ≠ 1], [2 ≠ 2], [2 ≠ 1]; x₃ contributes 0
    assert distances("heom", new, train, QUANT)[0].tolist() == [
        round((0.25**2 + 1) ** 0.5, 10),
        0.75,
        round((0.25**2 + 1) ** 0.5, 10),
    ]
    assert distances("gower", new, train, QUANT)[0].tolist() == [
        round(1.25 / 3, 10),
        0.25,
        round(1.25 / 3, 10),
    ]
    # a new value outside the training range is not clipped: |8 − 0|/4 = 2
    assert distances("heom", new, train, QUANT)[1, 0] == 2.0
    assert distances("gower", new, train, QUANT)[1, 0] == round(2 / 3, 10)
    assert (
        distances("gower", np.empty((2, 0)), np.empty((3, 0)), np.zeros(0, bool)).tolist()
        == [[0.0] * 3] * 2
    )


def test_gower_is_the_zhuravlyov_metric_over_n_on_minmax_values() -> None:
    rng = np.random.default_rng(5)
    Z = np.column_stack([rng.random((12, 2)), rng.integers(0, 3, size=12).astype(float)])
    Z[0, :2], Z[1, :2] = 0.0, 1.0  # every quantitative feature spans [0, 1]
    mask = np.array([True, True, False])
    assert distances("gower", Z, Z, mask) == pytest.approx(
        zhuravlyov(Z, Z, mask, 10, None) / 3, abs=1e-10
    )


def test_weighted_zhuravlyov() -> None:
    X = np.array([[0.0, 1.0, 0.5], [1.0, 2.0, 0.0]])
    assert distances("weighted-zhuravlyov", X, X, QUANT)[0, 1] == 2.5  # every weight 1: ρ
    assert distances("weighted-zhuravlyov", X, X, QUANT, nominal_weight=0.5)[0, 1] == 2.0
    assert distances("weighted-zhuravlyov", X, X, QUANT, quantitative_weight=2)[0, 1] == 4.0
    weighted = distances("weighted-zhuravlyov", X, X, QUANT, feature_weights=[1, 3, 0])
    assert weighted[0, 1] == 4.0  # 1·|0 − 1| + 3·[1 ≠ 2] + 0·|0.5 − 0|
    with pytest.raises(ConfigError, match="2 feature weights for an operator of 3 feature"):
        distances("weighted-zhuravlyov", X, X, QUANT, feature_weights=[1, 2])
    with pytest.raises(ConfigError, match="feature weights must not be negative"):
        METRICS.make_params("weighted-zhuravlyov", {"feature_weights": [1, -1, 1]})
    with pytest.raises(ConfigError, match="nominal_weight"):
        METRICS.make_params("weighted-zhuravlyov", {"nominal_weight": -1})
    with pytest.raises(ConfigError, match="p"):
        METRICS.make_params("minkowski", {"p": 0.5})


unified = arrays(np.float64, (5, 4), elements=st.floats(-2, 2, allow_nan=False, width=32))
masks = arrays(np.bool_, 4)


@settings(max_examples=60, deadline=None)
@given(X=unified, quantitative=masks)
def test_weighted_zhuravlyov_with_unit_weights_is_the_zhuravlyov_metric_bit_for_bit(
    X: np.ndarray, quantitative: np.ndarray
) -> None:
    Z = X.copy()
    Z[:, ~quantitative] = np.round(Z[:, ~quantitative])
    assert np.array_equal(
        distances("weighted-zhuravlyov", Z, Z, quantitative),
        zhuravlyov(Z, Z, quantitative, 10, None),
    )


@settings(max_examples=40, deadline=None)
@given(X=unified, quantitative=masks)
@pytest.mark.parametrize("name", [*HETEROGENEOUS, *QUANTITATIVE, *NOMINAL])
def test_metric_axioms(name: str, X: np.ndarray, quantitative: np.ndarray) -> None:
    domain = traits_of(METRICS.get(name)).domain
    mask = {"any": quantitative, "quantitative": np.ones(4, bool), "nominal": np.zeros(4, bool)}[
        domain
    ]
    Z = X.copy()
    Z[:, ~mask] = np.round(Z[:, ~mask])  # nominal codes
    D = distances(name, Z, Z, mask)
    assert np.array_equal(D, D.T)  # symmetry, bit for bit
    assert np.all(np.diag(D) == 0.0)
    assert np.all(D >= 0.0)
    assert np.all(np.isfinite(D))
    assert np.array_equal(D, round_distances(D, 10))  # rounded, so equal distances tie exactly
    if name != "cosine":  # a dissimilarity of directions, not a metric
        for k in range(len(Z)):  # triangle inequality up to the rounding
            assert np.all(D[:, [k]] + D[[k], :] + 1e-8 >= D), name


# ---------------------------------------------------------------- operators: domain and fit


def test_a_metric_on_features_of_the_wrong_type_is_a_configuration_error() -> None:
    with pytest.raises(ConfigError) as error:
        resolve_operators(operators_of("euclidean"), NAMES, QUANT)
    assert str(error.value) == (
        "operator ρ*: the metric 'euclidean' is defined on quantitative features, but the "
        "operator uses the nominal feature(s) x₂ — set its features to 'quantitative' or to a "
        "list of quantitative features"
    )
    with pytest.raises(ConfigError, match=r"'hamming' is defined on nominal features.*x₁, x₃"):
        resolve_operators(operators_of("hamming"), NAMES, QUANT)
    (operator,), _ = resolve_operators(operators_of("l2", "quantitative"), NAMES, QUANT)
    assert (operator.metric_name, operator.features.tolist()) == ("euclidean", [0, 2])
    (operator,), _ = resolve_operators(operators_of("hamming", ("x₂",)), NAMES, QUANT)
    assert operator.features.tolist() == [1]
    # parameters that do not fit the operator's features are found when it is resolved
    with pytest.raises(ConfigError) as error:
        resolve_operators(
            operators_of("weighted-zhuravlyov", "quantitative", feature_weights=[1, 2, 3]),
            NAMES,
            QUANT,
        )
    assert str(error.value).startswith(
        "operator ρ*: weighted-zhuravlyov: 3 feature weights for an operator of 2 feature(s)"
    )
    resolve_operators(
        operators_of("weighted-zhuravlyov", "quantitative", feature_weights=[1, 2]), NAMES, QUANT
    )
    # an empty subset is still skipped before the domain is looked at (ADR-007)
    config = ContextConfig(
        operators=(
            OperatorConfig(label="ρ", features="all"),
            OperatorConfig(label="h", metric=plugin("hamming"), features="nominal"),
        )
    )
    operators, skipped = resolve_operators(config, NAMES, np.ones(3, dtype=bool))
    assert [o.label for o in operators] == ["ρ"]
    assert [s.label for s in skipped] == ["h"]


def test_an_operator_with_a_fit_step_must_be_fitted_on_the_training_sample() -> None:
    (operator,), _ = resolve_operators(operators_of("heom"), NAMES, QUANT)
    assert operator.fit_step is fit_ranges
    assert not operator.fitted
    train = np.array([[0.0, 1.0, 5.0], [4.0, 2.0, 7.0]])
    with pytest.raises(RuntimeError, match="operator ρ\\*: the metric 'heom' needs its training"):
        operator.distances(train, train, decimals=10)
    fitted = operator.fit(train)
    assert fitted is not operator
    assert fitted.fitted
    assert fitted.state.spread.tolist() == [4.0, np.inf, 2.0]
    assert fitted.distances(train, train, decimals=10)[0, 1] == round(3**0.5, 10)
    # a metric without a fit step is returned as it is
    default, _ = resolve_operators(ContextConfig(), NAMES, QUANT)
    assert default[0].fit_step is None
    assert default[0].fit(train) is default[0]


def test_blocked_distances_equal_one_pass_for_a_fitted_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(1)
    Z = rng.random((40, 3))
    for name in ("mahalanobis", "heom"):
        (operator,), _ = resolve_operators(operators_of(name), NAMES, np.ones(3, dtype=bool))
        operator = operator.fit(Z)
        whole = operator.distances(Z, Z, decimals=10)
        with monkeypatch.context() as patch:
            patch.setattr("context_synthetic_recognition.core.operators.BLOCK_ELEMENTS", 50)
            assert np.array_equal(operator.distances(Z, Z, decimals=10), whole)


# ---------------------------------------------------------------- the pipeline


def config_with(metric: str) -> ExperimentConfig:
    domain = traits_of(METRICS.get(metric)).domain
    features = {"any": "all", "quantitative": "quantitative", "nominal": "nominal"}[domain]
    default = ContextConfig().operators
    return ExperimentConfig(
        context=ContextConfig(operators=(*default, *operators_of(metric, features).operators))
    )


@pytest.mark.parametrize("name", [*HETEROGENEOUS[1:], *QUANTITATIVE, *NOMINAL])
def test_every_metric_runs_through_the_model(experiment: Dataset, name: str) -> None:
    model = fit_model(experiment, config_with(name))
    trace = model.context.trace
    assert [o.label for o in trace.operators] == ["ρ", "ρ_I", "ρ_J", "ρ*"]
    assert trace.operators[3].metric == name
    assert trace.r == 8
    # the first three operators are the workbook's: a fourth one does not change them
    assert [f.omega for f in trace.features[:6]] == [
        f.omega for f in fit_context(experiment).trace.features
    ]
    # Theorem: a training object described through the new-object path, left out of its own
    # context, gets exactly its training row — with the training statistics of the metric
    again = model.context.represent(experiment.X, exclude=np.arange(experiment.m))
    assert np.array_equal(again.values, np.column_stack([f.values for f in trace.features]))
    assert np.array_equal(again.distances[3], trace.operators[3].distances)
    assert set(model.classify(experiment.X).decisions.tolist()) <= {0, 1, 2}


def test_the_training_statistics_come_from_the_fold_only(experiment: Dataset) -> None:
    # one very large value: its range is used only when the object is in the training part
    X = experiment.X.copy()
    quantitative = np.flatnonzero(experiment.quantitative)
    X[0, quantitative[0]] = 1e6
    shifted = Dataset(X=X, y=experiment.y, feature_types=experiment.feature_types)
    config = ExperimentConfig(
        context=operators_of("gower"),
        preprocessing={"normalizer": "none"},  # type: ignore[arg-type]
    )
    whole = fit_context(shifted, config)
    rest = fit_context(shifted.subset(np.arange(1, shifted.m)), config)
    spread = whole.operators[0].state.spread[quantitative[0]]
    assert spread == 1e6 - X[1:, quantitative[0]].min()
    assert rest.operators[0].state.spread[quantitative[0]] == np.ptp(X[1:, quantitative[0]])
    # the left-out object is measured with the ranges of the other nine
    held_out = rest.represent(X[:1])
    assert held_out.distances[0].max() > 1.0


def test_leave_one_out_and_the_baselines_with_fitted_metrics(experiment: Dataset) -> None:
    config = ExperimentConfig(
        context=ContextConfig(
            operators=(
                OperatorConfig(label="H", metric=plugin("heom")),
                OperatorConfig(label="M", metric=plugin("mahalanobis"), features="quantitative"),
                OperatorConfig(label="ρ_J", features="nominal"),
            )
        )
    )
    result = run_experiment(experiment, config)
    loo = result.protocol("leave-one-out")
    assert len(loo.predictions.decisions) == experiment.m
    methods = [method.method for method in loo.baselines]
    assert "k-NN vote H, k = 3" in methods
    assert "k-NN vote M, k = 5" in methods
