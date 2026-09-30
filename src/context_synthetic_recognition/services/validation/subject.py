"""What the package computes for the experiment workbook — the subject of its validation map."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from context_synthetic_recognition.core.arrays import FloatArray
from context_synthetic_recognition.core.hag import HAGResult
from context_synthetic_recognition.core.model import Classification, CSModel
from context_synthetic_recognition.core.properties import ModelProperties
from context_synthetic_recognition.core.trace import ContextTrace
from context_synthetic_recognition.evaluation.margins import MarginAnalysis
from context_synthetic_recognition.evaluation.protocols import ProtocolResult
from context_synthetic_recognition.services.sensitivity import SwitchVariant
from context_synthetic_recognition.services.validation.checks import Check, Grid


@dataclass(frozen=True)
class NewObject:
    """The workbook's new object (sheet *Brace for Meta-algorithm*) and its classification."""

    values: FloatArray
    """(n,) the typed feature values (row "x (input)")."""
    exclude: int | None
    """0-based training object left out of its context (leave-self-out), or ``None``."""
    classification: Classification
    """Its representation and the meta-algorithm's decision."""


@dataclass(frozen=True, eq=False)
class ExperimentEvaluation:
    """The evaluation sheets: leave-one-out with the k-NN baselines, margins, properties."""

    leave_one_out: ProtocolResult
    """The whole pipeline re-fitted per fold, with the k-NN vote baselines (k = 3, 5)."""
    margins: MarginAnalysis
    properties: ModelProperties
    sensitivity: tuple[SwitchVariant, ...]
    """The four switch settings (sheet *Sensitivity (Switches)*)."""


@dataclass(frozen=True)
class ExperimentSubject:
    """Everything the experiment workbook shows, computed by the package."""

    model: CSModel
    """The CS-model fitted on the *Dataset* sheet with the workbook's parameters."""
    new_object: NewObject
    """The *Brace* / *Meta-algorithm* demo."""
    training: Classification
    """Every training object classified with its own row (resubstitution)."""
    score_decimals: int
    """Rounding of score₁ − score₂ (sheet *Meta-algorithm (All Objects)*)."""
    evaluation: ExperimentEvaluation
    """Leave-one-out, baselines, margins, properties and the switch sensitivity."""

    @property
    def trace(self) -> ContextTrace:
        """The Step 1–8 trace."""
        return self.model.trace

    @property
    def hag(self) -> HAGResult:
        """The HAG of the fit."""
        return self.model.hag


ExperimentCheck = Check[ExperimentSubject]


def at(
    fn: Callable[[ExperimentSubject, int], Grid], index: int
) -> Callable[[ExperimentSubject], Grid]:
    """``fn`` with its second argument (a feature, step, object or method index) fixed."""
    return lambda s: fn(s, index)
