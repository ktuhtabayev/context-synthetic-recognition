"""What the exporters show: a run together with everything derived from it.

A :class:`RunView` holds the evaluated experiment (:class:`~context_synthetic_recognition.
services.runner.ExperimentResult`) and what the workbook shows beside it: every training object
classified with its own row (sheet *Meta-algorithm (All Objects)*), the new-object demonstration
of the sheets *Brace for Meta-algorithm* and *Meta-algorithm*, and — optionally — the four switch
settings of *Sensitivity (Switches)*. The Excel mirror, the tables, the figures and the report all
read the same view, so they cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.config.models import ExperimentConfig
from context_synthetic_recognition.core.arrays import FloatArray
from context_synthetic_recognition.core.hag import HAGResult
from context_synthetic_recognition.core.model import Classification, CSModel
from context_synthetic_recognition.core.trace import ContextTrace
from context_synthetic_recognition.data.schema import Dataset, Label
from context_synthetic_recognition.errors import DatasetError
from context_synthetic_recognition.evaluation.protocols import ProtocolResult
from context_synthetic_recognition.services.runner import ExperimentResult
from context_synthetic_recognition.services.sensitivity import SwitchVariant

RESUBSTITUTION = "resubstitution"
"""Canonical name of the training-correctness protocol (Definition 2)."""
LEAVE_ONE_OUT = "leave-one-out"
"""Canonical name of the protocol of the workbook's generalization sheets (Definition 3)."""


@dataclass(frozen=True, eq=False)
class NewObjectDemo:
    """The object classified on the sheets *Brace for Meta-algorithm* and *Meta-algorithm*."""

    values: FloatArray
    """(n,) its feature values in original units."""
    exclude: int | None
    """0-based training object left out of its context (leave-self-out), or ``None``."""
    classification: Classification
    """Its representation without a class (Theorem) and the meta-algorithm's decision."""

    @property
    def decision(self) -> int:
        """1 = K1, 2 = K2, 0 = refusal."""
        return int(self.classification.decisions[0])


@dataclass(frozen=True, eq=False)
class RunView:
    """An evaluated experiment with everything the exporters show."""

    result: ExperimentResult
    training: Classification
    """Every training object classified with its own row (resubstitution, Definition 2)."""
    new_object: NewObjectDemo
    sensitivity: tuple[SwitchVariant, ...] | None = None
    """The four switch settings, or ``None`` if they were not evaluated."""
    run_id: str | None = None
    """Name of the run folder, if the run was saved."""
    created: datetime | None = None
    """When the view was built (UTC)."""

    @property
    def config(self) -> ExperimentConfig:
        """The experiment configuration."""
        return self.result.config

    @property
    def dataset(self) -> Dataset:
        """The dataset the model was fitted on."""
        return self.result.dataset

    @property
    def model(self) -> CSModel:
        """The CS-model fitted on every object."""
        return self.result.model

    @property
    def trace(self) -> ContextTrace:
        """The Step 1–8 trace."""
        return self.result.model.trace

    @property
    def hag(self) -> HAGResult:
        """The HAG of the full fit."""
        return self.result.model.hag

    @property
    def labels(self) -> tuple[Label, ...]:
        """The class label of every training object."""
        classes = self.trace.classes
        return tuple(classes[int(c)] for c in self.trace.class_index)

    @property
    def slots(self) -> int:
        """Positions y₀ … of the meta-dataset the workbook lays out: the largest possible |TUPLAM|.

        min(ϰ, r), and never fewer than the features the HAG selected; positions the grouping did
        not fill are shown as "—", as in the workbook.
        """
        return max(min(self.hag.settings.kappa, self.hag.r), len(self.hag.tuplam))

    def protocol(self, name: str) -> ProtocolResult | None:
        """The result of a protocol by its canonical name, or ``None`` if it was not run."""
        return next((p for p in self.result.protocols if p.protocol == name), None)

    @property
    def cross_validations(self) -> tuple[ProtocolResult, ...]:
        """The protocols that hold objects out (everything except resubstitution)."""
        return tuple(p for p in self.result.protocols if p.protocol != RESUBSTITUTION)


def build_view(
    result: ExperimentResult,
    *,
    new_object: npt.ArrayLike | None = None,
    exclude: int | None = None,
    sensitivity: tuple[SwitchVariant, ...] | None = None,
    run_id: str | None = None,
) -> RunView:
    """Derive everything the exporters show from an evaluated experiment.

    Args:
        result: The evaluated experiment.
        new_object: The n feature values of the object to demonstrate the meta-algorithm on.
            ``None`` (default): the first training object, left out of its own context — the
            workbook's demonstration, which must reproduce that object's training row (Theorem).
        exclude: 0-based training object to leave out of the new object's context; only with
            ``new_object`` (the default demonstration always excludes its own object).
        sensitivity: The four switch settings, if they were evaluated
            (:func:`~context_synthetic_recognition.services.sensitivity.switch_sensitivity`).
        run_id: Name of the run folder.

    Raises:
        DatasetError: ``new_object`` does not have n values, or ``exclude`` is not an object.
    """
    model, dataset = result.model, result.dataset
    if new_object is None:
        values = np.array(dataset.X[0], dtype=np.float64)
        left_out: int | None = 0
    else:
        values = np.array(new_object, dtype=np.float64).reshape(-1)
        left_out = exclude
        if values.shape != (dataset.n,):
            raise DatasetError(f"the new object needs {dataset.n} values, got {values.size}")
        if left_out is not None and not 0 <= left_out < dataset.m:
            raise DatasetError(f"there is no training object № {left_out + 1} (m = {dataset.m})")
    classification = model.classify(values, exclude=None if left_out is None else [left_out])
    return RunView(
        result=result,
        training=model.classify_training(),
        new_object=NewObjectDemo(values, left_out, classification),
        sensitivity=sensitivity,
        run_id=run_id,
        created=datetime.now(tz=UTC),
    )
