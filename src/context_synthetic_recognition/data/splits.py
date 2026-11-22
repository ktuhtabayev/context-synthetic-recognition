"""Split files: objects with a training/test flag — fixed training and test samples.

The article's training and test samples are kept as files of this layout (``datasets/splits/``):

- line 1: ``m``, ``n + 1`` (the flag column is counted) and the number of classes ``c``;
- m lines: n feature values, the class label and the flag (0 = training, 1 = test);
- last line: n type flags (1 = quantitative, 0 = nominal); further values on it are ignored.

Values are separated by any whitespace (tabs, spaces, leading or trailing), decimal commas are
accepted, and both LF and CRLF line ends are read. The objects are the file's rows in the file's
order — not the order of the original dataset — so Sᵢ is the file's i-th row.

A :class:`FixedSplit` holds every object with its class: the classes of the test objects are
needed to score a method, never to classify (methods get the test objects' features only).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from context_synthetic_recognition.core.arrays import IntArray, readonly
from context_synthetic_recognition.data.loaders import _dat_rows, _flag, _label, _number
from context_synthetic_recognition.data.schema import Dataset, Label, as_labels
from context_synthetic_recognition.errors import DatasetError

TRAINING_FLAG = 0
"""Flag of a training object."""
TEST_FLAG = 1
"""Flag of a test object."""


@dataclass(frozen=True, eq=False)
class FixedSplit:
    """A dataset with a fixed training sample and test sample.

    Args:
        dataset: Every object of the file, in the file's order, with its class.
        training: 0-based indices of the training objects (increasing).
        test: 0-based indices of the test objects (increasing).
    """

    dataset: Dataset
    training: IntArray
    test: IntArray

    def __post_init__(self) -> None:
        """Check that the two samples partition the objects; store read-only copies."""
        training = np.array(self.training, dtype=np.int64).reshape(-1)
        test = np.array(self.test, dtype=np.int64).reshape(-1)
        both = np.concatenate([training, test])
        if not training.size or not test.size:
            raise DatasetError(
                f"{self.dataset.name}: the training and test samples must be non-empty"
            )
        if np.unique(both).size != both.size or both.size != self.dataset.m:
            raise DatasetError(
                f"{self.dataset.name}: the training and test samples must partition the "
                f"{self.dataset.m} objects"
            )
        if both.min() < 0 or both.max() >= self.dataset.m:
            raise DatasetError(
                f"{self.dataset.name}: sample indices outside 0 … {self.dataset.m - 1}"
            )
        object.__setattr__(self, "training", readonly(np.sort(training)))
        object.__setattr__(self, "test", readonly(np.sort(test)))

    @property
    def name(self) -> str:
        """The dataset's name (the file stem by default)."""
        return self.dataset.name

    @property
    def training_part(self) -> Dataset:
        """The training objects as a dataset."""
        return self.dataset.subset(self.training)

    @property
    def test_part(self) -> Dataset:
        """The test objects as a dataset (their classes are for scoring only)."""
        return self.dataset.subset(self.test)

    def sizes(self, indices: IntArray) -> tuple[int, ...]:
        """Number of objects and of each class (K1, K2, …) among ``indices``."""
        counts = np.bincount(self.dataset.class_index[indices], minlength=len(self.dataset.classes))
        return (int(indices.size), *(int(c) for c in counts))

    @property
    def training_sizes(self) -> tuple[int, ...]:
        """|training|, then the training objects of each class (K1, K2, …)."""
        return self.sizes(self.training)

    @property
    def test_sizes(self) -> tuple[int, ...]:
        """|test|, then the test objects of each class (K1, K2, …)."""
        return self.sizes(self.test)


def load_split_file(
    path: Path | str,
    *,
    name: str | None = None,
    feature_types: Mapping[str, str] | None = None,
) -> FixedSplit:
    """Read a split file (see the module docstring for the layout).

    Args:
        path: The file.
        name: Dataset name; the file stem by default.
        feature_types: Feature name → ``quantitative`` / ``nominal``; overrides the file's types.

    Raises:
        DatasetError: The file is missing or does not match the layout.
    """
    file = Path(path)
    if not file.is_file():
        raise DatasetError(f"split file not found: {file}")
    rows = list(_dat_rows(file))
    if not rows:
        raise DatasetError(f"{file.name}: the file is empty")
    header = [_number(cell) for cell in rows[0]]
    if len(header) != 3 or any(v is None or not v.is_integer() or v <= 0 for v in header):
        raise DatasetError(f"{file.name}: the first row must be 'm, n + 1, c', got {rows[0]}")
    m, n_flag, c = (int(v) for v in header if v is not None)
    n = n_flag - 1
    if n < 1:
        raise DatasetError(f"{file.name}: the header's n + 1 = {n_flag} leaves no feature")
    if len(rows) != m + 2:
        raise DatasetError(
            f"{file.name}: the header says m = {m}, so {m + 2} rows are expected "
            f"(header, {m} objects, type flags); the file has {len(rows)} non-empty rows"
        )
    X = np.empty((m, n))
    labels: list[Label] = []
    flags = np.empty(m, dtype=np.int64)
    for i, row in enumerate(rows[1 : m + 1]):
        if len(row) != n + 2:
            raise DatasetError(
                f"{file.name}: object row {i + 1} has {len(row)} values, expected {n + 2} "
                "(features, class, training/test flag)"
            )
        for j, cell in enumerate(row[:n]):
            value = _number(cell)
            if value is None or not math.isfinite(value):
                raise DatasetError(
                    f"{file.name}: object row {i + 1}, feature {j + 1}: {cell!r} is not a number"
                )
            X[i, j] = value
        labels.append(_label(row[n]))
        flag = _number(row[n + 1])
        if flag not in (TRAINING_FLAG, TEST_FLAG):
            raise DatasetError(
                f"{file.name}: object row {i + 1}: the flag must be {TRAINING_FLAG} (training) "
                f"or {TEST_FLAG} (test), got {row[n + 1]!r}"
            )
        flags[i] = int(flag)
    if len(rows[-1]) < n:
        raise DatasetError(f"{file.name}: the type row has {len(rows[-1])} flags, expected n = {n}")
    types = tuple(_flag(cell, f"{file.name} type row") for cell in rows[-1][:n])
    dataset = Dataset(
        X=X, y=as_labels(labels), feature_types=types, name=name or file.stem, source=str(file)
    )
    if len(dataset.classes) != c:
        raise DatasetError(
            f"{file.name}: the header says c = {c} classes, the data have {len(dataset.classes)}"
        )
    if feature_types:
        dataset = dataset.with_feature_types(feature_types)
    return FixedSplit(
        dataset, np.flatnonzero(flags == TRAINING_FLAG), np.flatnonzero(flags == TEST_FLAG)
    )
