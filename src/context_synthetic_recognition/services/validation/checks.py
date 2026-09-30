"""Checks: a sheet range of a workbook and the values the package computes for it.

A :class:`Check` pairs a cell range with a function that returns the expected grid from a
*subject* — whatever the package computed for that workbook (a fitted model, a HAG result, …).
:func:`run_check` compares the grid with the workbook's cached cell values: numbers to a
tolerance, text exactly, ``None`` against an empty cell.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.errors import CSRError
from context_synthetic_recognition.notation import DASH, object_name

DEFAULT_TOLERANCE = 1e-9
"""Numbers must agree to this absolute difference (the Validation sheet's acceptance level)."""

Cell = float | int | str | None
Grid = list[list[Cell]]

S = TypeVar("S")
T = TypeVar("T")


class ValidationError(CSRError, ValueError):
    """The workbook cannot be validated (unexpected layout or unreadable file)."""


@dataclass(frozen=True)
class Check(Generic[S]):
    """One sheet range and the values the package computes for it."""

    sheet: str
    ref: str
    """Cell or range, e.g. ``B5:K14``."""
    what: str
    """What the range holds (shown in reports)."""
    expected: Callable[[S], Grid]
    """Values for the range (rows × columns); ``None`` = empty cell, ``—`` = undefined."""
    when: Callable[[S], bool] | None = None
    """Run the check only if this holds (e.g. only for executed HAG iterations)."""

    def applies(self, subject: S) -> bool:
        """Whether the check is run for ``subject``."""
        return self.when is None or self.when(subject)

    def on(self, part: Callable[[T], S]) -> Check[T]:
        """The same check on a larger subject that contains this one's (``part`` extracts it)."""
        expected, when = self.expected, self.when
        return Check(
            self.sheet,
            self.ref,
            self.what,
            lambda subject: expected(part(subject)),
            None if when is None else (lambda subject: when(part(subject))),
        )


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one :class:`Check`."""

    sheet: str
    ref: str
    what: str
    cells: int
    """Number of cells compared."""
    max_difference: float
    """Largest absolute difference of the numeric cells (0 if none)."""
    mismatches: tuple[str, ...]
    """Description of every differing cell."""

    @property
    def passed(self) -> bool:
        """No cell differs."""
        return not self.mismatches


@dataclass(frozen=True)
class ValidationReport:
    """All check results for one workbook."""

    workbook: str
    tolerance: float
    results: tuple[CheckResult, ...]
    kind: str = "experiment"
    """Which layout was validated (see :class:`~.WorkbookKind`)."""
    notes: tuple[str, ...] = ()
    """Settings used and remarks, shown with the report."""

    @property
    def passed(self) -> bool:
        """Every check passed."""
        return all(result.passed for result in self.results)

    @property
    def cells(self) -> int:
        """Number of cells compared."""
        return sum(result.cells for result in self.results)

    def sheets(self) -> dict[str, list[CheckResult]]:
        """Results grouped by sheet, in workbook order."""
        grouped: dict[str, list[CheckResult]] = {}
        for result in self.results:
            grouped.setdefault(result.sheet, []).append(result)
        return grouped


# ---------------------------------------------------------------- grid helpers


def rows(values: npt.ArrayLike) -> Grid:
    """2-D array → grid (a 1-D array is one row); NaN becomes the workbook's "—"."""
    array = np.asarray(values, dtype=object)
    if array.ndim == 1:
        array = array[None, :]
    return [[cell(v) for v in row] for row in array]


def cell(value: Any) -> Cell:
    """One value as a workbook cell: booleans as 0/1, NaN as "—"."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool | np.bool_):
        return int(value)
    if isinstance(value, int | np.integer):
        return int(value)
    number = float(value)
    return DASH if math.isnan(number) else number


def column(values: Sequence[Any]) -> Grid:
    """A list of values as one column."""
    return [[cell(v)] for v in values]


def row(values: Sequence[Any]) -> Grid:
    """A list of values as one row."""
    return [[cell(v) for v in values]]


def flag(value: Any) -> int:
    """A truth value as the workbook's 1/0."""
    return int(bool(value))


def set_text(indices: Sequence[int] | npt.NDArray[Any]) -> str:
    """``{S₂, S₄}`` for 0-based indices, ``∅`` for none — the workbook's set cells."""
    names = [object_name(int(i)) for i in indices]
    return "{" + ", ".join(names) + "}" if names else "∅"


_COLUMNS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def col(index: int) -> str:
    """0-based column index → letter (A … Z)."""
    return _COLUMNS[index]


def span(first_col: int, first_row: int, cols: int, rows_: int) -> str:
    """Range of ``cols`` × ``rows_`` cells from a 0-based column and a 1-based row."""
    return f"{col(first_col)}{first_row}:{col(first_col + cols - 1)}{first_row + rows_ - 1}"


# ---------------------------------------------------------------- comparison


def _compare(workbook: Any, expected: Cell, tolerance: float) -> float | None:
    """Absolute difference for numbers, 0 for equal text/empty cells, ``None`` if different."""
    if expected is None:
        return 0.0 if workbook is None or workbook == "" else None
    if isinstance(expected, str):
        return 0.0 if isinstance(workbook, str) and workbook.strip() == expected else None
    if isinstance(workbook, bool) or not isinstance(workbook, int | float):
        return None
    difference = abs(float(workbook) - float(expected))
    return difference if difference <= tolerance else None


def _show(value: Any) -> str:
    return f"{value:.12g}" if isinstance(value, float) else repr(value)


def run_check(check: Check[S], worksheet: Any, subject: S, tolerance: float) -> CheckResult:
    """Compare one range of a worksheet with the values computed for ``subject``."""
    utils = importlib.import_module("openpyxl.utils")
    min_col, min_row, max_col, max_row = utils.range_boundaries(check.ref)
    expected = check.expected(subject)
    shape = (max_row - min_row + 1, max_col - min_col + 1)
    if (len(expected), len(expected[0]) if expected else 0) != shape:
        raise ValidationError(f"{check.sheet}!{check.ref}: the map yields a different shape")
    mismatches: list[str] = []
    largest = 0.0
    for r, line in enumerate(expected):
        for c, value in enumerate(line):
            workbook_cell = worksheet.cell(min_row + r, min_col + c)
            difference = _compare(workbook_cell.value, value, tolerance)
            if difference is None:
                mismatches.append(
                    f"{workbook_cell.coordinate}: workbook {_show(workbook_cell.value)} "
                    f"≠ package {_show(value)}"
                )
            else:
                largest = max(largest, difference)
    return CheckResult(
        check.sheet, check.ref, check.what, shape[0] * shape[1], largest, tuple(mismatches)
    )


def run_checks(
    checks: Sequence[Check[S]], workbook: Any, subject: S, tolerance: float, name: str
) -> tuple[CheckResult, ...]:
    """Run every applicable check against an open workbook.

    Raises:
        ValidationError: A sheet of the map is missing.
    """
    results = []
    for check in checks:
        if not check.applies(subject):
            continue
        if check.sheet not in workbook.sheetnames:
            raise ValidationError(f"{name}: sheet '{check.sheet}' is missing")
        results.append(run_check(check, workbook[check.sheet], subject, tolerance))
    order = {sheet: i for i, sheet in enumerate(workbook.sheetnames)}
    return tuple(sorted(results, key=lambda result: order[result.sheet]))  # workbook order


def read_range(worksheet: Any, ref: str) -> list[list[Any]]:
    """The cached values of a range, rows × columns."""
    utils = importlib.import_module("openpyxl.utils")
    min_col, min_row, max_col, max_row = utils.range_boundaries(ref)
    return [
        [worksheet.cell(r, c).value for c in range(min_col, max_col + 1)]
        for r in range(min_row, max_row + 1)
    ]


def read_numbers(worksheet: Any, ref: str, what: str) -> npt.NDArray[np.float64]:
    """The numeric values of a range as a float matrix.

    Raises:
        ValidationError: A cell is empty or not a number.
    """
    values = read_range(worksheet, ref)
    for line in values:
        for value in line:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValidationError(
                    f"{worksheet.title}!{ref} ({what}): {value!r} is not a number"
                )
    return np.array(values, dtype=np.float64)
