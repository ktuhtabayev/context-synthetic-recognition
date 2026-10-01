"""Base operators of Ψ: a metric applied to a feature subset (the article's variants of Ψ).

The default operators are the Zhuravlyov metric on all features (ρ), on the quantitative features
only (ρ_I) and on the nominal features only (ρ_J). An operator whose subset is empty for the
dataset is dropped with a warning (ADR-007).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from context_synthetic_recognition.config.models import ContextConfig, OperatorConfig
from context_synthetic_recognition.core.arrays import BoolArray, FloatArray, IntArray, readonly
from context_synthetic_recognition.core.metrics import METRICS, Metric, MetricFit, traits_of
from context_synthetic_recognition.errors import ConfigError, ModelUndefinedError

logger = logging.getLogger(__name__)

BLOCK_ELEMENTS = 1 << 22
"""Distances are computed in blocks of target rows with at most this many pairs per block."""


@dataclass(frozen=True, eq=False)
class BaseOperator:
    """A metric on a feature subset."""

    label: str
    """Display name (ρ, ρ_I, ρ_J, …); synthetic features are named after it."""
    metric_name: str
    """Registry name of the metric."""
    metric: Metric
    """The distance function."""
    params: Any
    """The metric's validated parameters (``None`` if it has none)."""
    features: IntArray
    """0-based indices of the features the operator uses."""
    quantitative: BoolArray
    """Which of these features belong to set I."""
    fit_step: MetricFit | None = None
    """The metric's fit step (``None``: it needs no training statistics, ADR-050)."""
    state: Any = None
    """What the fit step returned on the training sample; the metric gets it as its parameters."""
    fitted: bool = False
    """Whether :meth:`fit` has been called (always needed when there is a fit step)."""

    def __post_init__(self) -> None:
        """Freeze the index arrays."""
        readonly(self.features)
        readonly(self.quantitative)

    def fit(self, Z: FloatArray) -> BaseOperator:
        """The operator with its metric's training statistics (ADR-050).

        Args:
            Z: Unified values of the training objects (all features). No class labels are
                passed: the statistics of a metric never depend on a class.

        Returns:
            The operator itself if its metric has no fit step, else a fitted copy.
        """
        if self.fit_step is None:
            return self
        state = self.fit_step(Z[:, self.features], self.quantitative, self.params)
        return replace(self, state=state, fitted=True)

    def distances(self, A: FloatArray, B: FloatArray, *, decimals: int) -> FloatArray:
        """Distances between the rows of ``A`` and ``B`` (unified values of all features).

        Returns:
            Matrix of shape (len(A), len(B)), rounded to ``decimals``.

        Raises:
            RuntimeError: The metric has a fit step and :meth:`fit` was not called.
        """
        if self.fit_step is not None and not self.fitted:
            raise RuntimeError(
                f"operator {self.label}: the metric '{self.metric_name}' needs its training "
                "statistics — call fit() with the unified training values first"
            )
        params = self.params if self.fit_step is None else self.state
        a = A[:, self.features]
        b = B[:, self.features]
        out = np.empty((a.shape[0], b.shape[0]))
        rows = max(1, BLOCK_ELEMENTS // max(1, b.shape[0]))
        for start in range(0, a.shape[0], rows):
            stop = min(start + rows, a.shape[0])
            out[start:stop] = self.metric(a[start:stop], b, self.quantitative, decimals, params)
        return out


@dataclass(frozen=True)
class SkippedOperator:
    """An operator left out for this dataset."""

    label: str
    reason: str


def _subset(
    operator: OperatorConfig, feature_names: Sequence[str], quantitative: BoolArray
) -> IntArray:
    if operator.features == "all":
        return np.arange(len(feature_names), dtype=np.int64)
    if operator.features == "quantitative":
        return np.flatnonzero(quantitative).astype(np.int64)
    if operator.features == "nominal":
        return np.flatnonzero(~quantitative).astype(np.int64)
    unknown = [name for name in operator.features if name not in feature_names]
    if unknown:
        raise ConfigError(
            f"operator {operator.label}: unknown features {unknown} "
            f"(features: {', '.join(feature_names)})"
        )
    return np.array([feature_names.index(name) for name in operator.features], dtype=np.int64)


def _check_domain(
    label: str,
    metric_name: str,
    metric: Metric,
    features: IntArray,
    feature_names: Sequence[str],
    quantitative: BoolArray,
) -> None:
    """A metric defined on set I (or J) must not get features of the other set (ADR-050)."""
    domain = traits_of(metric).domain
    if domain == "any":
        return
    outside = features[quantitative[features] != (domain == "quantitative")]
    if outside.size:
        other = "nominal" if domain == "quantitative" else "quantitative"
        names = ", ".join(feature_names[int(j)] for j in outside)
        raise ConfigError(
            f"operator {label}: the metric '{metric_name}' is defined on {domain} features, but "
            f"the operator uses the {other} feature(s) {names} — set its features to "
            f"'{domain}' or to a list of {domain} features"
        )


def resolve_operators(
    config: ContextConfig, feature_names: Sequence[str], quantitative: BoolArray
) -> tuple[tuple[BaseOperator, ...], tuple[SkippedOperator, ...]]:
    """Build the base operators of the configuration for a dataset's features.

    Args:
        config: The ``context`` section.
        feature_names: The dataset's feature names.
        quantitative: The dataset's mask of set I.

    Returns:
        The operators in configuration order and the operators skipped because their feature
        subset is empty (only if ``skip_empty_operators``, ADR-007). An operator whose metric
        has a fit step still has to be fitted on the training sample (:meth:`BaseOperator.fit`).

    Raises:
        ConfigError: Unknown metric, invalid metric parameters, unknown feature names, a
            metric given features of a type it is not defined on, or parameters that do not fit
            the operator's features.
        ModelUndefinedError: No operator is left.
    """
    mask = np.asarray(quantitative, dtype=bool)
    operators: list[BaseOperator] = []
    skipped: list[SkippedOperator] = []
    for spec in config.operators:
        features = _subset(spec, feature_names, mask)
        if features.size == 0 and config.skip_empty_operators:
            reason = f"no {spec.features} features in the dataset"
            skipped.append(SkippedOperator(spec.label, reason))
            logger.warning(
                "operator skipped: empty feature subset (ADR-007)",
                extra={"operator": spec.label, "features": spec.features},
            )
            continue
        info = METRICS.info(spec.metric.name)
        declared = traits_of(info.obj)
        params = METRICS.make_params(spec.metric.name, spec.metric.params)
        _check_domain(spec.label, info.name, info.obj, features, feature_names, mask)
        if declared.check is not None:
            try:
                declared.check(mask[features], params)
            except ConfigError as error:
                raise ConfigError(f"operator {spec.label}: {error}") from error
        operators.append(
            BaseOperator(
                label=spec.label,
                metric_name=info.name,
                metric=info.obj,
                params=params,
                features=features,
                quantitative=mask[features],
                fit_step=declared.fit,
            )
        )
    if not operators:
        raise ModelUndefinedError("no base operator is left: every feature subset is empty")
    return tuple(operators), tuple(skipped)
