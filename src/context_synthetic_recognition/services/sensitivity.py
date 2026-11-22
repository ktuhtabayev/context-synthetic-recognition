"""Sensitivity of the results to the two switches of the agglomerative algorithm.

Sheet *Switch Sensitivity*. All four combinations of ``additional_features.class_means``
(running | final) and ``additional_features.majorizing_passes`` (2 | 1) — the Excel experiment's
setting first, the article's last — with TUPLAM, crit per iteration, and the resubstitution and
leave-one-out accuracies and AUCs (ADR-002 – ADR-004).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from context_synthetic_recognition.config.models import ClassMeans, ExperimentConfig
from context_synthetic_recognition.core.model import fit_model
from context_synthetic_recognition.data.schema import Dataset
from context_synthetic_recognition.evaluation.metrics import classification_metrics
from context_synthetic_recognition.evaluation.protocols import run_protocol
from context_synthetic_recognition.evaluation.roc import roc_curve
from context_synthetic_recognition.services.runner import positive_code

SETTINGS: tuple[tuple[ClassMeans, int, str], ...] = (
    (ClassMeans.RUNNING, 2, "running means, 2 passes"),
    (ClassMeans.RUNNING, 1, "running means, 1 pass"),
    (ClassMeans.FINAL, 2, "final means, 2 passes"),
    (ClassMeans.FINAL, 1, "article"),
)
"""The four combinations in the order of the Excel experiment workbook."""


@dataclass(frozen=True)
class SwitchVariant:
    """The results under one combination of the switches."""

    class_means: ClassMeans
    majorizing_passes: int
    label: str
    tuplam: str
    """TUPLAM of the fit on every object, e.g. ``{a₆, a₃, a₁, a₂, a₄}``."""
    crit: tuple[float, ...]
    resubstitution: float
    """Resubstitution accuracy (Definition 2)."""
    leave_one_out: float
    """Leave-one-out accuracy (Definition 3)."""
    auc_resubstitution: float
    auc_leave_one_out: float


def switch_sensitivity(
    dataset: Dataset,
    config: ExperimentConfig | None = None,
    *,
    progress: Callable[[str, int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[SwitchVariant, ...]:
    """Evaluate all four switch combinations with otherwise the same configuration.

    ``progress`` is called with (setting, done, total) during every leave-one-out; ``cancelled``
    is polled before every fold and stops the evaluation with ``EvaluationCancelledError``.
    """
    base = config or ExperimentConfig()
    decimals = base.evaluation.score_decimals
    positive = positive_code(dataset.classes, base.evaluation.positive_class)
    truth = dataset.class_index + 1
    variants = []
    for class_means, passes, label in SETTINGS:
        agglomerative = base.additional_features.model_copy(
            update={"class_means": class_means, "majorizing_passes": passes}
        )
        cfg = base.model_copy(update={"additional_features": agglomerative})

        def report(done: int, total: int, stage: str = label) -> None:
            if progress is not None:
                progress(stage, done, total)

        model = fit_model(dataset, cfg)
        training = model.classify_training()
        loo = run_protocol(
            dataset, cfg, "leave-one-out", progress=report, cancelled=cancelled
        ).predictions
        variants.append(
            SwitchVariant(
                class_means=class_means,
                majorizing_passes=passes,
                label=label,
                tuplam=model.agglomerative.label,
                crit=model.agglomerative.crit,
                resubstitution=classification_metrics(truth, training.decisions, positive).accuracy,
                leave_one_out=loo.metrics(positive).accuracy,
                auc_resubstitution=roc_curve(training.scores, truth, positive, decimals).auc,
                auc_leave_one_out=loo.roc(positive, decimals).auc,
            )
        )
    return tuple(variants)
