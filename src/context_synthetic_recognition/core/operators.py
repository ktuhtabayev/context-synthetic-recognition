"""Base operators of Ψ: a metric applied to a feature subset (the article's variants of Ψ).

The default operators are the Zhuravlyov metric on all features (ρ), on the quantitative features
only (ρ_I) and on the nominal features only (ρ_J). An operator whose subset is empty for the
dataset is dropped with a warning (ADR-007).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from context_synthetic_recognition.config.models import ContextConfig, OperatorConfig
from context_synthetic_recognition.core.arrays import BoolArray, FloatArray, IntArray, readonly
from context_synthetic_recognition.core.metrics import METRICS, Metric
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

    def __post_init__(self) -> None:
        """Freeze the index arrays."""
        readonly(self.features)
        readonly(self.quantitative)

    def distances(self, A: FloatArray, B: FloatArray, *, decimals: int) -> FloatArray:
        """Distances between the rows of ``A`` and ``B`` (unified values of all features).

        Returns:
            Matrix of shape (len(A), len(B)), rounded to ``decimals``.
        """
        a = A[:, self.features]
        b = B[:, self.features]
        out = np.empty((a.shape[0], b.shape[0]))
        rows = max(1, BLOCK_ELEMENTS // max(1, b.shape[0]))
        for start in range(0, a.shape[0], rows):
            stop = min(start + rows, a.shape[0])
            out[start:stop] = self.metric(
                a[start:stop], b, self.quantitative, decimals, self.params
            )
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
        subset is empty (only if ``skip_empty_operators``, ADR-007).

    Raises:
        ConfigError: Unknown metric, invalid metric parameters or unknown feature names.
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
        operators.append(
            BaseOperator(
                label=spec.label,
                metric_name=METRICS.info(spec.metric.name).name,
                metric=METRICS.get(spec.metric.name),
                params=METRICS.make_params(spec.metric.name, spec.metric.params),
                features=features,
                quantitative=mask[features],
            )
        )
    if not operators:
        raise ModelUndefinedError("no base operator is left: every feature subset is empty")
    return tuple(operators), tuple(skipped)
