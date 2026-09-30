"""Dataset summaries for the CLI and the GUI: sizes, types, classes, permitted k, operators.

The summary is what the GUI shows live while a dataset is being set up: |Kᵢ|, the permitted k of
the configured strategy and the synthetic features r it implies.
"""

from __future__ import annotations

from dataclasses import dataclass

from context_synthetic_recognition.config.models import ExperimentConfig
from context_synthetic_recognition.core.k_strategies import PermittedK, permitted_k
from context_synthetic_recognition.core.operators import SkippedOperator, resolve_operators
from context_synthetic_recognition.data.schema import Dataset
from context_synthetic_recognition.errors import CSRError


@dataclass(frozen=True)
class OperatorSummary:
    """A base operator as it applies to a dataset."""

    label: str
    metric: str
    features: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSummary:
    """What the pipeline will make of a dataset under a configuration."""

    dataset: Dataset
    permitted_k: PermittedK | None
    """``None`` if no k is permitted (see :attr:`problems`)."""
    operators: tuple[OperatorSummary, ...]
    skipped_operators: tuple[SkippedOperator, ...]
    problems: tuple[str, ...]
    """Why the model would be undefined for these data (empty if it is defined)."""

    @property
    def r(self) -> int | None:
        """Number of synthetic features |Ψ(r)| = operators × permitted k."""
        if self.permitted_k is None:
            return None
        return len(self.operators) * self.permitted_k.count


def summarize(dataset: Dataset, config: ExperimentConfig | None = None) -> DatasetSummary:
    """Summarize ``dataset`` under ``config`` (the template preset by default)."""
    cfg = config or ExperimentConfig()
    problems: list[str] = []
    if len(dataset.classes) != 2:
        problems.append(
            f"{len(dataset.classes)} classes — the CS-model is binary (two classes, ADR-010)"
        )
    ks: PermittedK | None = None
    try:
        ks = permitted_k(cfg.k.name, cfg.k.params, dataset.class_sizes, dataset.m - 1)
    except CSRError as error:
        problems.append(str(error))
    operators: tuple[OperatorSummary, ...] = ()
    skipped: tuple[SkippedOperator, ...] = ()
    try:
        resolved, skipped = resolve_operators(
            cfg.context, dataset.feature_names, dataset.quantitative
        )
        operators = tuple(
            OperatorSummary(
                op.label, op.metric_name, tuple(dataset.feature_names[j] for j in op.features)
            )
            for op in resolved
        )
    except CSRError as error:
        problems.append(str(error))
    return DatasetSummary(dataset, ks, operators, skipped, tuple(problems))
