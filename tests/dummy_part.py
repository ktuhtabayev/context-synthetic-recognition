"""A small classifier part for the tests: four objects, one trivial method (ADR-084).

It shows what a part needs — nothing but the protocol of
:mod:`context_synthetic_recognition.services.classifier_parts` — and is the example of
``docs/extending.md``. Its one method gives every object the class of the nearest training object
on the first feature.
"""

from pathlib import Path

import numpy as np

from context_synthetic_recognition.data import Dataset, FeatureType
from context_synthetic_recognition.data.samples import AnswerKey, Decisions, UnlabelledSample
from context_synthetic_recognition.services.classifier_parts import (
    Evaluation,
    MethodGroup,
    MethodInfo,
    MethodTraining,
    Predictions,
    Progress,
    SampleChoice,
    SampleGroup,
    SampleOption,
    SampleSet,
    SampleTree,
    TrainedPart,
    TrainingSample,
    metrics_against,
)

TYPES = (FeatureType.QUANTITATIVE,)
NAMES = ("x₁",)


class DummyPart:
    """One sample, one method: the nearest training object on x₁."""

    key = "dummy"
    title = "Dummy part"
    description = "Four objects and one trivial method, for the tests."

    def default_folder(self) -> Path | None:
        return None

    def samples(self, folder: Path | None) -> SampleTree:
        toy = SampleChoice("toy", "Toy split · training 4 (2, 2) · test 3", "Toy · training 4")
        return SampleTree((SampleGroup("Toy (7 objects)", (toy,)),), default=("toy",))

    def load(self, option: SampleOption, folder: Path | None) -> SampleSet:
        training = Dataset(
            X=np.array([[0.0], [1.0], [10.0], [11.0]]), y=[1, 1, 2, 2], feature_types=TYPES
        )
        objects = np.array([4, 5, 6])
        test = UnlabelledSample(objects, np.array([[0.5], [10.5], [2.0]]), TYPES, NAMES, "toy")
        answers = AnswerKey(objects, np.array([1, 2, 2]), (1, 2), "toy")
        return SampleSet(option, (TrainingSample("toy", "Toy", training),), (test,), (answers,))

    def method_groups(self, option: SampleOption) -> tuple[MethodGroup, ...]:
        method = MethodInfo(
            "nearest", "Nearest object", "all (1)", ("toy",), "the nearest training object on x₁"
        )
        return (MethodGroup("Trivial", "The class of the nearest training object.", (method,)),)

    def train(
        self,
        option: SampleOption,
        training: tuple[TrainingSample, ...],
        progress: Progress | None = None,
    ) -> TrainedPart:
        (sample,) = training
        decided = self._nearest(sample.data, sample.data.X, leave_out=True)
        methods = {"toy": (MethodTraining("nearest", "Nearest object", decided),)}
        return TrainedPart(self.key, option, training, methods, fitted=sample.data)

    def classify(
        self,
        trained: TrainedPart,
        unlabelled: tuple[UnlabelledSample, ...],
        progress: Progress | None = None,
    ) -> Predictions:
        (sample,) = unlabelled
        decided = self._nearest(trained.fitted, sample.X, leave_out=False)
        decisions = {"toy": {"nearest": Decisions(decided)}}
        return Predictions(self.key, trained, unlabelled, decisions)

    def evaluate(
        self,
        predictions: Predictions,
        answers: tuple[AnswerKey, ...],
        progress: Progress | None = None,
    ) -> Evaluation:
        (key,) = answers
        metrics = {"toy": metrics_against(predictions.decisions["toy"], key)}
        return Evaluation(self.key, predictions, answers, metrics)

    @staticmethod
    def _nearest(training: Dataset, X: np.ndarray, *, leave_out: bool) -> np.ndarray:
        distances = np.abs(X[:, :1] - training.X[:, 0][None, :])
        if leave_out:
            np.fill_diagonal(distances, np.inf)
        return training.class_index[np.argmin(distances, axis=1)] + 1
