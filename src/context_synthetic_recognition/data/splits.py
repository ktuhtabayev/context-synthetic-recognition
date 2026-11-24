"""Split files: objects with a split marker — fixed training and test samples.

The article's training and test samples are kept as files of this layout (``datasets/splits/``,
ADR-082):

- row 1: the object count ``m``, the feature count ``n`` and the class count ``c`` — the header
  counts the features only; empty cells may follow;
- rows 2 … m + 1, one per object: the n feature values, then the class (1 … c) in column n + 1 and
  the split marker (0 = training, 1 = test) in column n + 2;
- last row: the n feature types (0 = nominal, 1 = quantitative); columns n + 1 and n + 2 stay
  empty, because they are not features.

Cells are separated by tabs and no row starts with an empty cell. Decimal commas are accepted, LF
and CRLF line ends are both read, and empty cells at the end of a row are ignored. Every departure
from the layout is reported with the file and the row (the file's line number). The objects are
the file's rows in the file's order — not the order of the original dataset — so Sᵢ is the object
of row i + 1.

A :class:`FixedSplit` holds every object with its class: the classes of the test objects are
needed to score a method, never to classify. :meth:`FixedSplit.test_objects` gives the test
objects without their class and :meth:`FixedSplit.test_answers` their classes, kept apart
(:mod:`.samples`, ADR-085).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from context_synthetic_recognition.core.arrays import IntArray, readonly
from context_synthetic_recognition.data.loaders import _number
from context_synthetic_recognition.data.samples import AnswerKey, UnlabelledSample
from context_synthetic_recognition.data.schema import Dataset, FeatureType, as_labels
from context_synthetic_recognition.errors import DatasetError

TRAINING_FLAG = 0
"""Split marker of a training object."""
TEST_FLAG = 1
"""Split marker of a test object."""


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

    def test_objects(self, name: str | None = None) -> UnlabelledSample:
        """The test objects without their class: what the *Test* stage classifies (ADR-085)."""
        data = self.dataset
        return UnlabelledSample(
            self.test,
            data.X[self.test],
            data.feature_types,
            data.feature_names,
            name or f"{data.name}: test sample",
        )

    def test_answers(self, name: str | None = None) -> AnswerKey:
        """The true classes of the test objects, kept apart: only *Evaluation* reads them.

        The class codes follow the classes of the whole file (1 = K1, 2 = K2, …).
        """
        data = self.dataset
        return AnswerKey(
            self.test,
            data.class_index[self.test] + 1,
            data.classes,
            name or f"{data.name}: test sample",
        )

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
        DatasetError: The file is missing or does not match the layout; the message names the
            file and the row.
    """
    file = Path(path)
    if not file.is_file():
        raise DatasetError(f"split file not found: {file}")
    rows = _rows(file)
    if not rows:
        raise DatasetError(f"{file.name}: the file is empty")
    m, n, c = _header(file.name, *rows[0])
    if len(rows) != m + 2:
        raise DatasetError(
            f"{file.name}: the header says m = {m}, so {m + 2} rows are expected "
            f"(the header, {m} object rows, the type row); the file has {len(rows)} non-empty rows"
        )
    X = np.empty((m, n))
    labels: list[int] = []
    markers = np.empty(m, dtype=np.int64)
    for i, (line, cells) in enumerate(rows[1 : m + 1]):
        where = f"{file.name}: row {line} (object {i + 1})"
        _no_empty_cells(where, cells)
        if len(cells) != n + 2:
            hint = "; the header's n counts the features only" if len(cells) == n + 1 else ""
            raise DatasetError(
                f"{where} has {len(cells)} values, expected n + 2 = {n + 2} "
                f"({n} features, the class, the split marker){hint}"
            )
        for j, cell in enumerate(cells[:n]):
            value = _number(cell)
            if value is None or not math.isfinite(value):
                raise DatasetError(f"{where}, column {j + 1}: {cell!r} is not a number")
            X[i, j] = value
        labels.append(_class(f"{where}, column {n + 1}", cells[n], c))
        markers[i] = _marker(f"{where}, column {n + 2}", cells[n + 1])
    types = _types(file.name, *rows[-1], n)
    dataset = Dataset(
        X=X, y=as_labels(labels), feature_types=types, name=name or file.stem, source=str(file)
    )
    if len(dataset.classes) != c:
        raise DatasetError(
            f"{file.name}: the header says c = {c} classes, the objects have {len(dataset.classes)}"
        )
    if feature_types:
        dataset = dataset.with_feature_types(feature_types)
    return FixedSplit(
        dataset, np.flatnonzero(markers == TRAINING_FLAG), np.flatnonzero(markers == TEST_FLAG)
    )


def _rows(file: Path) -> list[tuple[int, list[str]]]:
    """(line number, cells) of every non-blank line of the file.

    Cells are split at tabs and stripped; empty cells at the end of a line are dropped.
    """
    try:
        with file.open(encoding="utf-8-sig") as handle:
            lines = handle.read().split("\n")  # universal newlines: CRLF arrives as LF
    except (OSError, UnicodeDecodeError) as error:
        raise DatasetError(f"{file.name}: cannot read the file ({error})") from error
    rows = []
    for number, line in enumerate(lines, start=1):
        cells = [cell.strip() for cell in line.split("\t")]
        while cells and not cells[-1]:
            cells.pop()
        if cells:
            rows.append((number, cells))
    return rows


def _header(file: str, line: int, cells: list[str]) -> tuple[int, int, int]:
    """(m, n, c) of the first row: exactly three positive integers, then empty cells."""
    numbers = [_number(cell) for cell in cells]
    if len(cells) != 3 or any(v is None or not v.is_integer() or v <= 0 for v in numbers):
        raise DatasetError(
            f"{file}: row {line} (the header) must hold exactly three positive integers — "
            f"m objects, n features, c classes — followed only by empty cells, got {cells}"
        )
    m, n, c = (int(v) for v in numbers if v is not None)
    return m, n, c


def _no_empty_cells(where: str, cells: list[str]) -> None:
    """Reject an empty cell before the row's last value (a row never starts with one)."""
    if not cells[0]:
        raise DatasetError(f"{where} starts with an empty cell; the values begin in column 1")
    for j, cell in enumerate(cells, start=1):
        if not cell:
            raise DatasetError(f"{where}, column {j} is empty")


def _class(where: str, cell: str, c: int) -> int:
    """The class of an object row: an integer 1 … c."""
    value = _number(cell)
    if value is None or not value.is_integer() or not 1 <= value <= c:
        raise DatasetError(
            f"{where}: the class must be one of the header's c = {c} classes (1 … {c}), "
            f"got {cell!r}"
        )
    return int(value)


def _marker(where: str, cell: str) -> int:
    """The split marker of an object row: 0 (training) or 1 (test)."""
    value = _number(cell)
    if value is None or value not in (TRAINING_FLAG, TEST_FLAG):
        raise DatasetError(
            f"{where}: the split marker must be {TRAINING_FLAG} (training) or {TEST_FLAG} (test), "
            f"got {cell!r}"
        )
    return int(value)


def _types(file: str, line: int, cells: list[str], n: int) -> tuple[FeatureType, ...]:
    """The n type flags of the last row; every cell after them must be empty."""
    where = f"{file}: row {line} (the type row)"
    if not cells[0]:
        raise DatasetError(f"{where} starts with an empty cell; the type flags begin in column 1")
    for j, cell in enumerate(cells[n:], start=n + 1):
        if cell:
            raise DatasetError(
                f"{where}, column {j}: {cell!r} after the n = {n} type flags — columns n + 1 and "
                "n + 2 hold the class and the split marker, not features, so they stay empty"
            )
    if len(cells) < n:
        raise DatasetError(f"{where} has {len(cells)} type flags, expected n = {n}")
    types = []
    for j, cell in enumerate(cells, start=1):
        flag = _number(cell)
        if flag is None or flag not in (0, 1):
            raise DatasetError(
                f"{where}, column {j}: a type flag must be 0 (nominal) or 1 (quantitative), "
                f"got {cell!r}"
            )
        types.append(FeatureType.from_flag(int(flag)))
    return tuple(types)
