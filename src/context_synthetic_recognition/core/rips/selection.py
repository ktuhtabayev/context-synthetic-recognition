"""R-IPS on a dataset: the distances of its proximity measure, then Steps 1–6.

The quantitative features are normalized with constants estimated on E₁ = E₀ ∖ T — the objects
that take part — so that a given test sample T cannot shape the selection (ADR-061); with T empty
this is the whole sample E₀. The metric is applied to the configured feature subset (all features
for the Zhuravlyov metric, the quantitative ones for the Euclidean and Chebyshev metrics in the
article's experiments) and its distances are rounded like those of the local contexts.
"""

from __future__ import annotations

import numpy as np

from context_synthetic_recognition.config.models import (
    LocalContextConfig,
    OperatorConfig,
    RIPSConfig,
)
from context_synthetic_recognition.core.arrays import FloatArray, IntArray
from context_synthetic_recognition.core.local_context.normalizers import NORMALIZERS
from context_synthetic_recognition.core.local_context.operators import resolve_operators
from context_synthetic_recognition.core.rips.algorithm import (
    RIPSResult,
    RIPSSettings,
    TrainingRole,
    run_rips,
)
from context_synthetic_recognition.data.schema import Dataset
from context_synthetic_recognition.errors import ConfigError

RIPS_LABEL = "ρ"
"""Label of the proximity measure of R-IPS in messages."""


def test_sample_indices(dataset: Dataset, config: RIPSConfig) -> IntArray:
    """0-based indices of the test sample T named in ``config`` by object id.

    Raises:
        ConfigError: An id is not an object of the dataset.
    """
    position = {name: i for i, name in enumerate(dataset.object_ids)}
    unknown = [name for name in config.test_sample if name not in position]
    if unknown:
        raise ConfigError(
            f"rips.test_sample: unknown objects {', '.join(unknown)} (the dataset names its "
            f"objects {dataset.object_ids[0]} … {dataset.object_ids[-1]})"
        )
    return np.array(sorted(position[name] for name in config.test_sample), dtype=np.int64)


def rips_distances(dataset: Dataset, config: RIPSConfig, working: IntArray) -> FloatArray:
    """ρ between all objects of E₀, normalized with the constants of the objects ``working``.

    Raises:
        ConfigError: Unknown metric or normalizer, invalid parameters, or features the metric is
            not defined on.
        ModelUndefinedError: The feature subset is empty for the dataset.
    """
    spec = OperatorConfig(label=RIPS_LABEL, metric=config.metric, features=config.features)
    operators, _ = resolve_operators(
        LocalContextConfig(operators=(spec,), skip_empty_operators=False),
        dataset.feature_names,
        dataset.quantitative,
    )
    normalizer = config.normalizer
    scaling = NORMALIZERS.get(normalizer.name)(
        dataset.X[working],
        dataset.quantitative,
        NORMALIZERS.make_params(normalizer.name, normalizer.params),
    )
    Z = scaling.transform(dataset.X)
    operator = operators[0].fit(Z[working])
    D = operator.distances(Z, Z, decimals=config.distance_decimals)
    np.fill_diagonal(D, 0.0)
    return D


def select_precedents(dataset: Dataset, config: RIPSConfig | None = None) -> RIPSResult:
    """R-IPS on ``dataset``: the training and the test sample (Steps 1–6).

    Raises:
        ConfigError: Invalid settings or an unknown object in the test sample.
        ModelUndefinedError: E₁ lacks a class.
    """
    chosen = config or RIPSConfig()
    T = test_sample_indices(dataset, chosen)
    working = np.setdiff1d(np.arange(dataset.m, dtype=np.int64), T)
    D = rips_distances(dataset, chosen, working)
    settings = RIPSSettings(
        training=TrainingRole(chosen.training), keep_blocked_cover=chosen.keep_blocked_cover
    )
    return run_rips(
        D,
        dataset.class_index,
        test_sample=T,
        settings=settings,
        decimals=chosen.distance_decimals,
    )
