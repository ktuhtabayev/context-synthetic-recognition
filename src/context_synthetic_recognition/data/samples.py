"""Objects to classify without their class, and the classes kept apart (ADR-085).

A new object is represented and classified without its class (the Theorem). The *Test* stage of
the *Classifiers* page therefore receives an :class:`UnlabelledSample`: the objects' indices in
their file, their feature values and the feature types and names — there is no class field and
no split marker. The true classes of the same objects are an :class:`AnswerKey`, which only the
*Evaluation* stage receives.

:meth:`~context_synthetic_recognition.data.splits.FixedSplit.test_objects` and
:meth:`~context_synthetic_recognition.data.splits.FixedSplit.test_answers` build both from a
split file's test sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.core.arrays import BoolArray, FloatArray, IntArray, readonly
from context_synthetic_recognition.data.schema import FeatureType, Label
from context_synthetic_recognition.errors import DatasetError
from context_synthetic_recognition.notation import object_name


@dataclass(frozen=True, eq=False)
class UnlabelledSample:
    """Objects to classify: their feature values only, without a class (the Theorem).

    Args:
        objects: 0-based index of every object in its file (object Sᵢ has index i − 1).
        X: (q × n) feature values, in the order of ``objects``.
        feature_types: Type (I or J) of every feature.
        feature_names: Name of every feature.
        name: What the objects are, e.g. ``336: test sample``.
    """

    objects: IntArray
    X: FloatArray
    feature_types: tuple[FeatureType, ...]
    feature_names: tuple[str, ...]
    name: str = "unlabelled sample"

    def __post_init__(self) -> None:
        """Check the shapes; store read-only copies."""
        objects = np.array(self.objects, dtype=np.int64).reshape(-1)
        X = np.array(self.X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] != objects.size:
            raise DatasetError(
                f"{self.name}: {objects.size} objects need a (q × n) array with q = {objects.size}"
            )
        types = tuple(FeatureType(t) for t in self.feature_types)
        if not len(types) == len(self.feature_names) == X.shape[1]:
            raise DatasetError(
                f"{self.name}: {X.shape[1]} features, {len(types)} types, "
                f"{len(self.feature_names)} names"
            )
        object.__setattr__(self, "objects", readonly(objects))
        object.__setattr__(self, "X", readonly(X))
        object.__setattr__(self, "feature_types", types)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))

    @property
    def m(self) -> int:
        """Number of objects."""
        return int(self.objects.size)

    @property
    def n(self) -> int:
        """Number of features."""
        return int(self.X.shape[1])

    @property
    def quantitative(self) -> BoolArray:
        """Mask of the quantitative features (set I)."""
        return np.array([t is FeatureType.QUANTITATIVE for t in self.feature_types], dtype=bool)

    @property
    def object_ids(self) -> tuple[str, ...]:
        """``S₁``, ``S₂``, … — the objects' names by their place in the file."""
        return tuple(object_name(int(i)) for i in self.objects)


@dataclass(frozen=True, eq=False)
class AnswerKey:
    """The true classes of the objects of an :class:`UnlabelledSample`, kept apart from them.

    Args:
        objects: 0-based index of every object in its file (the same as the sample's).
        classes: Class code of every object: 1 = K1, 2 = K2, … (the file's class order).
        labels: The class labels of K1, K2, …
        name: What the objects are, e.g. ``336: test sample``.
    """

    objects: IntArray
    classes: IntArray
    labels: tuple[Label, ...]
    name: str = "answer key"

    def __post_init__(self) -> None:
        """Check the shapes and the codes; store read-only copies."""
        objects = np.array(self.objects, dtype=np.int64).reshape(-1)
        classes = np.array(self.classes, dtype=np.int64).reshape(-1)
        if classes.size != objects.size:
            raise DatasetError(f"{self.name}: {classes.size} classes for {objects.size} objects")
        if classes.size and (classes.min() < 1 or classes.max() > len(self.labels)):
            raise DatasetError(f"{self.name}: class codes must be 1 … {len(self.labels)}")
        object.__setattr__(self, "objects", readonly(objects))
        object.__setattr__(self, "classes", readonly(classes))
        object.__setattr__(self, "labels", tuple(self.labels))

    @property
    def m(self) -> int:
        """Number of objects."""
        return int(self.objects.size)

    @property
    def sizes(self) -> tuple[int, ...]:
        """|objects|, then the objects of each class (K1, K2, …)."""
        counts = np.bincount(self.classes - 1, minlength=len(self.labels))
        return (self.m, *(int(c) for c in counts))

    def matches(self, sample: UnlabelledSample) -> bool:
        """Whether the key is for exactly these objects, in this order."""
        return bool(np.array_equal(self.objects, sample.objects))


def unlabelled(
    objects: npt.ArrayLike,
    X: npt.ArrayLike,
    feature_types: tuple[FeatureType, ...],
    feature_names: tuple[str, ...],
    name: str = "unlabelled sample",
) -> UnlabelledSample:
    """An :class:`UnlabelledSample` from plain values (no dataset, no class)."""
    return UnlabelledSample(np.asarray(objects), np.asarray(X), feature_types, feature_names, name)


@dataclass(frozen=True, eq=False)
class Decisions:
    """What one method decided for the objects of an :class:`UnlabelledSample`.

    Args:
        decisions: (q,) 1 = K1, 2 = K2, 0 = refusal; ``None`` if the method is undefined.
        scores: (q,) larger for K1, if the method gives scores.
        undefined: Why the method is undefined (empty if it is defined).
    """

    decisions: IntArray | None
    scores: FloatArray | None = None
    undefined: str = ""

    def __post_init__(self) -> None:
        """Store read-only copies."""
        if self.decisions is not None:
            object.__setattr__(
                self, "decisions", readonly(np.array(self.decisions, dtype=np.int64))
            )
        if self.scores is not None:
            object.__setattr__(self, "scores", readonly(np.array(self.scores, dtype=np.float64)))

    def same_as(self, other: Decisions) -> bool:
        """The same decisions and the same scores, value for value."""

        def same(a: npt.NDArray[np.generic] | None, b: npt.NDArray[np.generic] | None) -> bool:
            return (a is None and b is None) or (
                a is not None and b is not None and bool(np.array_equal(a, b))
            )

        return same(self.decisions, other.decisions) and same(self.scores, other.scores)


def require_unlabelled(sample: object) -> UnlabelledSample:
    """``sample`` if it is an :class:`UnlabelledSample`.

    Raises:
        TypeError: It is not — a :class:`~.schema.Dataset` carries the objects' classes, and a
            new object is classified without its class (the Theorem, ADR-085).
    """
    if not isinstance(sample, UnlabelledSample):
        raise TypeError(
            f"objects are classified without their class: give an UnlabelledSample, not a "
            f"{type(sample).__name__}"
        )
    return sample
