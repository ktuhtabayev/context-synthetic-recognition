"""Dataset schema: objects E₀ = {S₁ … Sₘ}, heterogeneous features (types I and J), classes.

A :class:`Dataset` is immutable and has no I/O; loaders (:mod:`.loaders`) build it from files.
Nominal features hold numeric category codes — the Zhuravlyov metric only asks whether two codes
are equal; string categories are encoded by the loaders and kept in :attr:`Dataset.categories`.

Classes are ordered by their labels (numbers numerically, strings alphabetically): the first is
K1, the second K2 (ADR-018). For the author's datasets (labels 1 and 2) this is the article's
numbering.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.core.arrays import BoolArray, FloatArray, IntArray, readonly
from context_synthetic_recognition.errors import DatasetError
from context_synthetic_recognition.notation import feature_name, object_name

Label = int | str
"""A class label as stored in the data file."""


class FeatureType(StrEnum):
    """Type of an original feature."""

    QUANTITATIVE = "quantitative"
    """Set I: real values, rescaled to [0, 1] and compared by |x′ⱼ − y′ⱼ|."""
    NOMINAL = "nominal"
    """Set J: category codes, compared by [xⱼ ≠ yⱼ]."""

    @property
    def flag(self) -> int:
        """The workbook's type flag: 1 = quantitative (I), 0 = nominal (J)."""
        return 1 if self is FeatureType.QUANTITATIVE else 0

    @classmethod
    def from_flag(cls, flag: int) -> FeatureType:
        """Type for a workbook/template flag (1 → I, 0 → J)."""
        if flag == 1:
            return cls.QUANTITATIVE
        if flag == 0:
            return cls.NOMINAL
        raise DatasetError(
            f"feature-type flags must be 1 (quantitative) or 0 (nominal), got {flag}"
        )


def as_labels(labels: Sequence[Label] | npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Class labels as an array: int64 if all are integers, text if all are strings."""
    values = list(np.asarray(labels, dtype=object).ravel())
    if not values:
        return np.array([], dtype=np.int64)
    if all(isinstance(v, int | np.integer) and not isinstance(v, bool) for v in values):
        return np.array([int(v) for v in values], dtype=np.int64)
    if all(isinstance(v, str) for v in values):
        return np.array(values, dtype=np.str_)
    kinds = sorted({type(v).__name__ for v in values})
    raise DatasetError(f"class labels must be all integers or all strings, got {', '.join(kinds)}")


@dataclass(frozen=True, eq=False)
class Dataset:
    """A classified sample with heterogeneous features.

    Args:
        X: Feature values, shape (m, n); nominal columns hold category codes.
        y: Class label of every object, shape (m,) — integers or strings.
        feature_types: Type (I or J) of every feature.
        feature_names: Feature names; ``x₁ … xₙ`` by default.
        object_ids: Object names; ``S₁ … Sₘ`` by default.
        name: Short name of the dataset.
        categories: For nominal features read as text: code c → category ``categories[name][c]``.
        source: Where the data came from (file path), for display only.
    """

    X: FloatArray
    y: npt.NDArray[Any]
    feature_types: tuple[FeatureType, ...]
    feature_names: tuple[str, ...] = ()
    object_ids: tuple[str, ...] = ()
    name: str = "dataset"
    categories: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    source: str | None = None

    def __post_init__(self) -> None:
        """Validate shapes, names and values; store read-only copies."""
        X = np.array(self.X, dtype=np.float64)
        if X.ndim != 2:
            raise DatasetError(f"{self.name}: X must be 2-D (objects × features), got {X.ndim}-D")
        m, n = X.shape
        if m == 0 or n == 0:
            raise DatasetError(
                f"{self.name}: the dataset needs at least one object and one feature"
            )
        if not np.all(np.isfinite(X)):
            rows, cols = np.nonzero(~np.isfinite(X))
            where = ", ".join(
                f"({r + 1}, {c + 1})" for r, c in zip(rows[:10], cols[:10], strict=True)
            )
            raise DatasetError(
                f"{self.name}: missing or non-finite values at (object, feature) {where}"
            )
        y = as_labels(self.y)
        if y.shape != (m,):
            raise DatasetError(f"{self.name}: {len(y)} class labels for {m} objects")
        types = tuple(FeatureType(t) for t in self.feature_types)
        if len(types) != n:
            raise DatasetError(f"{self.name}: {len(types)} feature types for {n} features")
        names = self.feature_names or tuple(feature_name(j) for j in range(n))
        ids = self.object_ids or tuple(object_name(i) for i in range(m))
        _check_unique(names, n, "feature names", self.name)
        _check_unique(ids, m, "object ids", self.name)
        unknown = sorted(set(self.categories) - set(names))
        if unknown:
            raise DatasetError(f"{self.name}: categories given for unknown features {unknown}")
        object.__setattr__(self, "X", readonly(X))
        object.__setattr__(self, "y", readonly(y))
        object.__setattr__(self, "feature_types", types)
        object.__setattr__(self, "feature_names", tuple(names))
        object.__setattr__(self, "object_ids", tuple(ids))
        object.__setattr__(self, "categories", MappingProxyType(dict(self.categories)))

    # ------------------------------------------------------------------ sizes and types

    @property
    def m(self) -> int:
        """Number of objects."""
        return int(self.X.shape[0])

    @property
    def n(self) -> int:
        """Number of features."""
        return int(self.X.shape[1])

    @property
    def quantitative(self) -> BoolArray:
        """Mask of the quantitative features (set I)."""
        return np.array([t is FeatureType.QUANTITATIVE for t in self.feature_types], dtype=bool)

    @property
    def nominal(self) -> BoolArray:
        """Mask of the nominal features (set J)."""
        return ~self.quantitative

    @property
    def type_flags(self) -> tuple[int, ...]:
        """The workbook's type row: 1 = quantitative, 0 = nominal."""
        return tuple(t.flag for t in self.feature_types)

    # ------------------------------------------------------------------ classes

    @property
    def classes(self) -> tuple[Label, ...]:
        """Distinct class labels in class order (K1, K2, …)."""
        return tuple(v.item() for v in np.unique(self.y))

    @property
    def class_index(self) -> IntArray:
        """0-based class index of every object (0 = K1, 1 = K2, …)."""
        return np.searchsorted(np.unique(self.y), self.y).astype(np.int64)

    @property
    def class_sizes(self) -> tuple[int, ...]:
        """|K1|, |K2|, … counted from the objects."""
        return tuple(int(c) for c in np.bincount(self.class_index, minlength=len(self.classes)))

    # ------------------------------------------------------------------ derived datasets

    def subset(self, indices: Sequence[int] | IntArray, *, name: str | None = None) -> Dataset:
        """The objects at ``indices`` (in that order), e.g. a cross-validation training fold."""
        index = np.asarray(indices, dtype=np.int64)
        return Dataset(
            X=self.X[index],
            y=self.y[index],
            feature_types=self.feature_types,
            feature_names=self.feature_names,
            object_ids=tuple(self.object_ids[i] for i in index),
            name=name or self.name,
            categories=self.categories,
            source=self.source,
        )

    def with_feature_types(self, types: Mapping[str, FeatureType | str]) -> Dataset:
        """A copy with the types of the named features replaced.

        Raises:
            DatasetError: A name is not a feature of this dataset.
        """
        unknown = sorted(set(types) - set(self.feature_names))
        if unknown:
            raise DatasetError(f"{self.name}: feature types given for unknown features {unknown}")
        new_types = tuple(
            FeatureType(types.get(name, current))
            for name, current in zip(self.feature_names, self.feature_types, strict=True)
        )
        return Dataset(
            X=self.X,
            y=self.y,
            feature_types=new_types,
            feature_names=self.feature_names,
            object_ids=self.object_ids,
            name=self.name,
            categories=self.categories,
            source=self.source,
        )

    # ------------------------------------------------------------------ identity

    def content_hash(self) -> str:
        """SHA-256 of the content: values, labels, feature names and types, object ids.

        The name and the source path are not part of the hash, so the same data read from a
        CSV file or from the workbook have the same hash.
        """
        header = {
            "features": list(self.feature_names),
            "types": [t.value for t in self.feature_types],
            "objects": list(self.object_ids),
            "labels": [v.item() for v in self.y],
            "categories": {k: list(v) for k, v in sorted(self.categories.items())},
        }
        digest = hashlib.sha256()
        digest.update(json.dumps(header, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(np.ascontiguousarray(self.X, dtype="<f8").tobytes())
        return digest.hexdigest()

    def __repr__(self) -> str:
        """``Dataset('name', m=10, n=13, classes={1: 4, 2: 6})``."""
        sizes = dict(zip(self.classes, self.class_sizes, strict=True))
        return f"Dataset({self.name!r}, m={self.m}, n={self.n}, classes={sizes})"


def _check_unique(values: Sequence[str], expected: int, what: str, dataset: str) -> None:
    if len(values) != expected:
        raise DatasetError(f"{dataset}: {len(values)} {what} for {expected} entries")
    repeated = sorted(value for value, count in Counter(values).items() if count > 1)
    if repeated:
        raise DatasetError(f"{dataset}: {what} must be unique (repeated: {', '.join(repeated)})")
