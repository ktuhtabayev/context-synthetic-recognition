"""Sensitivity of the results to the two template/article switches (sheet *Sensitivity (Switches)*).

All four combinations of ``hag.centres`` (running | final) and ``hag.step4_passes`` (2 | 1) — the
template setting first, the article setting last — with TUPLAM, crit per iteration, and the
resubstitution and leave-one-out accuracies and AUCs (ADR-002 – ADR-004).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from context_synthetic_recognition.config.models import CentreMode, ExperimentConfig
from context_synthetic_recognition.core.model import fit_model
from context_synthetic_recognition.data.schema import Dataset
from context_synthetic_recognition.evaluation.metrics import classification_metrics
from context_synthetic_recognition.evaluation.protocols import run_protocol
from context_synthetic_recognition.evaluation.roc import roc_curve
from context_synthetic_recognition.services.runner import positive_code

SETTINGS: tuple[tuple[CentreMode, int, str], ...] = (
    (CentreMode.RUNNING, 2, "template (default)"),
    (CentreMode.RUNNING, 1, "running centres, 1 pass"),
    (CentreMode.FINAL, 2, "final centres, 2 passes"),
    (CentreMode.FINAL, 1, "article"),
)
"""The four combinations in the workbook's order."""


@dataclass(frozen=True)
class SwitchVariant:
    """The results under one combination of the switches."""

    centres: CentreMode
    step4_passes: int
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
    for centres, passes, label in SETTINGS:
        hag = base.hag.model_copy(update={"centres": centres, "step4_passes": passes})
        cfg = base.model_copy(update={"hag": hag})

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
                centres=centres,
                step4_passes=passes,
                label=label,
                tuplam=model.hag.label,
                crit=model.hag.crit,
                resubstitution=classification_metrics(truth, training.decisions, positive).accuracy,
                leave_one_out=loo.metrics(positive).accuracy,
                auc_resubstitution=roc_curve(training.scores, truth, positive, decimals).auc,
                auc_leave_one_out=loo.roc(positive, decimals).auc,
            )
        )
    return tuple(variants)
