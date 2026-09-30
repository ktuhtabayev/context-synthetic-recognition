"""Validation against the Excel experiment: workbook cells vs. the values this package computes.

A declarative map (:func:`workbook_checks`) pairs sheet ranges of the experiment workbook with
values read from the pipeline's trace. :func:`validate_workbook` loads the data from the
workbook's own *Dataset* sheet, fits the pipeline, reads the workbook's cached cell values and
compares them — numbers to a tolerance (1e-9 by default), text exactly. The golden tests and
``csr validate`` use the same map.

Covered so far: Steps 1–8 (sheets *Parameters* … *Ψ(r) Contribution & Weight*). The HAG, the
meta-algorithm and the evaluation sheets are added with milestones M3 and M4.
"""

from __future__ import annotations

import importlib
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.config.models import ExperimentConfig
from context_synthetic_recognition.core.context import fit_context
from context_synthetic_recognition.core.contributions import weight_ranks
from context_synthetic_recognition.core.trace import ContextTrace
from context_synthetic_recognition.data.loaders import load_dataset
from context_synthetic_recognition.errors import CSRError
from context_synthetic_recognition.notation import DASH

DEFAULT_TOLERANCE = 1e-9
"""Numbers must agree to this absolute difference (the Validation sheet's acceptance level)."""

Cell = float | int | str | None
Grid = list[list[Cell]]


class ValidationError(CSRError, ValueError):
    """The workbook cannot be validated (unexpected layout or unreadable file)."""


@dataclass(frozen=True)
class Check:
    """One sheet range and the values the package computes for it."""

    sheet: str
    ref: str
    """Cell or range, e.g. ``B5:K14``."""
    what: str
    """What the range holds (shown in reports)."""
    expected: Callable[[ContextTrace], Grid]
    """Values for the range (rows × columns); ``None`` = empty cell, ``—`` = undefined."""


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


# ---------------------------------------------------------------- helpers for expected grids


def _rows(values: npt.ArrayLike) -> Grid:
    """2-D array → grid; NaN becomes the workbook's "—"."""
    array = np.asarray(values, dtype=object)
    if array.ndim == 1:
        array = array[None, :]
    return [[_cell(v) for v in row] for row in array]


def _cell(value: Any) -> Cell:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool | np.bool_):
        return int(value)
    if isinstance(value, int | np.integer):
        return int(value)
    number = float(value)
    return DASH if math.isnan(number) else number


def _column(values: Sequence[Cell]) -> Grid:
    return [[v] for v in values]


def _labels(trace: ContextTrace, index: npt.ArrayLike) -> list[Cell]:
    return [trace.classes[int(i)] for i in np.asarray(index).ravel()]


# ---------------------------------------------------------------- the map (workbook layout)

WORKBOOK_OBJECTS = 10
"""The workbook's fixed layout: 10 objects, 13 features, operators ρ, ρ_I, ρ_J, 2 permitted k."""
WORKBOOK_OPERATORS = ("ρ", "ρ_I", "ρ_J")
WORKBOOK_KS = 2
_COLUMNS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_MAX_GRADATION = 5
"""The membership tables of the workbook list μ = 0 … 5."""


def _col(index: int) -> str:
    """0-based column index → letter (A … Z)."""
    return _COLUMNS[index]


def _span(col: int, row: int, cols: int, rows: int) -> str:
    return f"{_col(col)}{row}:{_col(col + cols - 1)}{row + rows - 1}"


def _parameter_checks() -> list[Check]:
    sheet = "Parameters"
    return [
        Check(sheet, "B5", "objects m", lambda t: [[t.m]]),
        Check(sheet, "B6:B7", "|K1|, |K2|", lambda t: _column(list(t.class_sizes))),
        Check(sheet, "B8", "min |Kᵢ|", lambda t: [[min(t.class_sizes)]]),
        Check(sheet, "B9", "k_min", lambda t: [[t.permitted_k.k_min]]),
        Check(sheet, "B10", "k_max = 2·min|Kᵢ| − 3", lambda t: [[t.permitted_k.k_max]]),
        Check(sheet, "B11", "permitted k", lambda t: [[t.permitted_k.label]]),
        Check(sheet, "B12", "number of permitted k", lambda t: [[t.permitted_k.count]]),
        Check(sheet, "B13", "base operators", lambda t: [[len(t.operators)]]),
        Check(sheet, "B14", "r = |Ψ(r)|", lambda t: [[t.r]]),
    ]


def _normalized_checks() -> list[Check]:
    sheet = "Normalized Dataset"

    def statistic(t: ContextTrace, key: str) -> Grid:
        values = t.scaling.statistics[key]
        return _rows(
            [[v if q else DASH for v, q in zip(values, t.scaling.quantitative, strict=True)]]
        )

    return [
        Check(sheet, "B5:N14", "x′ = (x − min)/(max − min) on I", lambda t: _rows(t.normalized)),
        Check(sheet, "O5:O14", "class", lambda t: _column(_labels(t, t.class_index))),
        Check(sheet, "B15:N15", "type flags", lambda t: _rows(t.scaling.quantitative.astype(int))),
        Check(sheet, "B16:N16", "training min", lambda t: statistic(t, "min")),
        Check(sheet, "B17:N17", "training max", lambda t: statistic(t, "max")),
    ]


def _distances_of(o: int) -> Callable[[ContextTrace], Grid]:
    return lambda t: _rows(t.operators[o].distances)


def _membership_of(u: int) -> Callable[[ContextTrace], Grid]:
    return lambda t: _membership_grid(t, u)


def _stability_of(u: int) -> Callable[[ContextTrace], Grid]:
    return lambda t: [[t.features[u].stability]]


def _mask_membership_of(o: int) -> Callable[[ContextTrace], Grid]:
    return lambda t: _mask_grid(t, o)


def _mask_stability_of(o: int) -> Callable[[ContextTrace], Grid]:
    return lambda t: [[t.bit_masks[o].stability]]


def _distance_checks() -> list[Check]:
    return [
        Check(
            "Zhuravlev Distances",
            _span(1, 5 + 13 * o, WORKBOOK_OBJECTS, WORKBOOK_OBJECTS),
            f"{label} distances",
            _distances_of(o),
        )
        for o, label in enumerate(WORKBOOK_OPERATORS)
    ]


def _neighbour_checks() -> list[Check]:
    checks: list[Check] = []
    ranks = WORKBOOK_OBJECTS - 1
    for o, label in enumerate(WORKBOOK_OPERATORS):
        sheet = f"Sorted Neighbors ({label})"

        def rank_grid(t: ContextTrace, o: int = o) -> Grid:
            ranks_ = t.operators[o].ranks
            return _rows([[DASH if v == 0 else v for v in row] for row in ranks_])

        checks.append(Check(sheet, "B5:K14", "rank of Sⱼ among the neighbours of Sᵢ", rank_grid))
        for i in range(WORKBOOK_OBJECTS):
            base = 16 + 11 * i

            def k_labels(t: ContextTrace) -> Grid:
                return [
                    [f"k = {r}" if r in t.permitted_k.ks else None for r in range(1, ranks + 1)]
                ]

            def names(t: ContextTrace, o: int = o, i: int = i) -> Grid:
                return [[t.object_ids[j] for j in t.operators[o].order[i]]]

            def index(t: ContextTrace, o: int = o, i: int = i) -> Grid:
                return _rows(t.operators[o].order[i] + 1)

            def distance(t: ContextTrace, o: int = o, i: int = i) -> Grid:
                return _rows(t.operators[o].sorted_distances[i])

            def classes(t: ContextTrace, o: int = o, i: int = i) -> Grid:
                return [_labels(t, t.class_index[t.operators[o].order[i]])]

            def same(t: ContextTrace, o: int = o, i: int = i) -> Grid:
                neighbours = t.class_index[t.operators[o].order[i]]
                return _rows((neighbours == t.class_index[i]).astype(int))

            def mu(t: ContextTrace, o: int = o, i: int = i) -> Grid:
                return _rows(t.operators[o].same_class_running(t.class_index)[i])

            def chi1(t: ContextTrace, o: int = o, i: int = i) -> Grid:
                return _rows(t.operators[o].k1_running(t.class_index)[i])

            s = f"S{i + 1}"
            for offset, what, fn in (
                (2, f"{s}: k-neighbourhood marks", k_labels),
                (3, f"{s}: neighbours", names),
                (4, f"{s}: original index", index),
                (5, f"{s}: distance", distance),
                (6, f"{s}: class", classes),
                (7, f"{s}: same class", same),
                (8, f"{s}: same-class count μ", mu),
                (9, f"{s}: K1 count χ₁", chi1),
            ):
                checks.append(Check(sheet, _span(1, base + offset, ranks, 1), what, fn))
    return checks


def _mu_checks() -> list[Check]:
    sheet = "Synthetic Features (k-NN)"
    checks = [
        Check(
            sheet,
            "G4:G8",
            "|K1|, |K2|, min, k_min, k_max",
            lambda t: _column(
                [*t.class_sizes, min(t.class_sizes), t.permitted_k.k_min, t.permitted_k.k_max]
            ),
        ),
        Check(sheet, "G9", "permitted k", lambda t: [[t.permitted_k.label]]),
    ]
    for o, label in enumerate(WORKBOOK_OPERATORS):
        top = 4 + 14 * o

        def ks(t: ContextTrace) -> Grid:
            return [list(t.permitted_k.ks)]

        def mu(t: ContextTrace, o: int = o) -> Grid:
            return _rows(np.column_stack([f.mu for f in t.features if f.operator == o]))

        checks += [
            Check(sheet, f"B{top}:C{top}", f"{label}: permitted k", ks),
            Check(sheet, f"B{top + 1}:C{top + 10}", f"{label}: same-class count μ", mu),
            Check(
                sheet,
                f"D{top + 1}:D{top + 10}",
                f"{label}: class",
                lambda t: _column(_labels(t, t.class_index)),
            ),
        ]
    return checks


def _psi_checks() -> list[Check]:
    sheet = "Ψ(r) Binary Features"

    def per_feature(attribute: str) -> Callable[[ContextTrace], Grid]:
        return lambda t: _rows(np.column_stack([getattr(f, attribute) for f in t.features]))

    return [
        Check(sheet, "A5:A10", "feature names", lambda t: _column([f.name for f in t.features])),
        Check(
            sheet, "B5:B10", "operators", lambda t: _column([f.operator_label for f in t.features])
        ),
        Check(sheet, "C5:C10", "k", lambda t: _column([f.k for f in t.features])),
        Check(sheet, "B14:G23", "χ₁ — K1 objects among the k nearest", per_feature("chi1")),
        Check(sheet, "B27:G36", "χ₂ = k − χ₁", per_feature("chi2")),
        Check(sheet, "B40:G49", "Ψ(r): aᵤ by formula (5)", per_feature("values")),
        Check(sheet, "H40:H49", "class", lambda t: _column(_labels(t, t.class_index))),
    ]


def _membership_grid(t: ContextTrace, u: int) -> Grid:
    feature = t.features[u]
    table = feature.membership
    grid: Grid = []
    for mu in range(_MAX_GRADATION + 1):
        if mu > feature.k:
            grid.append([mu, DASH, DASH, DASH, DASH, DASH, DASH, 0])
            continue
        f = table.f[mu]
        grid.append(
            [
                mu,
                int(table.d1[mu]),
                int(table.d2[mu]),
                int(table.n[mu]),
                float(table.share1[mu]),
                float(table.share2[mu]),
                DASH if math.isnan(f) else float(f),
                float(table.weighted[mu]),
            ]
        )
    return grid


def _mask_grid(t: ContextTrace, o: int) -> Grid:
    table = t.bit_masks[o].membership
    return [
        [
            int(table.gradations[g]),
            int(table.d1[g]),
            int(table.d2[g]),
            int(table.n[g]),
            DASH if math.isnan(table.f[g]) else float(table.f[g]),
            float(table.weighted[g]),
        ]
        for g in range(table.gradations.size)
    ]


def _membership_checks() -> list[Check]:
    sheet = "Membership & Stability"
    checks: list[Check] = []
    for u in range(len(WORKBOOK_OPERATORS) * WORKBOOK_KS):
        row0 = 3 + 11 * (u // 2)
        col0 = 0 if u % 2 == 0 else 9
        name = f"a{u + 1}"
        checks += [
            Check(
                sheet,
                _span(col0, row0 + 2, 8, _MAX_GRADATION + 1),
                f"{name}: d₁, d₂, n, f_k(μ) — formula (1)",
                _membership_of(u),
            ),
            Check(
                sheet,
                f"{_col(col0 + 7)}{row0 + 8}",
                f"{name}: stability g — formula (2)",
                _stability_of(u),
            ),
        ]
    checks.append(Check(sheet, "B40:G40", "meta-object (g)", lambda t: _rows(t.meta_object)))

    def bits(t: ContextTrace) -> Grid:
        columns: list[npt.NDArray[Any]] = []
        for representation in t.bit_masks:
            columns += [representation.bits[:, 0], representation.bits[:, 1], representation.masks]
        return _rows(np.column_stack(columns).astype(int))

    checks.append(Check(sheet, "B44:J53", "bits by the majority rule and masks", bits))
    for o, label in enumerate(WORKBOOK_OPERATORS):
        col0 = 7 * o
        checks += [
            Check(
                sheet,
                _span(col0, 57, 6, 4),
                f"{label} masks: formula (1)",
                _mask_membership_of(o),
            ),
            Check(
                sheet,
                f"{_col(col0 + 5)}61",
                f"{label} masks: stability g",
                _mask_stability_of(o),
            ),
        ]
    return checks


def _informativeness_checks() -> list[Check]:
    sheet = "Informativeness ω"

    def row(fn: Callable[[Any], Cell]) -> Callable[[ContextTrace], Grid]:
        return lambda t: [[fn(f) for f in t.features]]

    def q(value: float | None) -> Cell:
        return "none" if value is None else value

    return [
        Check(sheet, "B4:G4", "features", row(lambda f: f.name)),
        Check(sheet, "B5:G5", "operators", row(lambda f: f.operator_label)),
        Check(sheet, "B6:G6", "k", row(lambda f: f.k)),
        Check(sheet, "B7:G7", "q₂ = max{f < 0.5}", row(lambda f: q(f.boundary.q2))),
        Check(sheet, "B8:G8", "q₁ = min{f > 0.5}", row(lambda f: q(f.boundary.q1))),
        Check(sheet, "B9:G9", "G_k — formula (3)", row(lambda f: f.boundary.G)),
        Check(
            sheet,
            "B10:G10",
            "correctly placed objects",
            row(lambda f: int(np.count_nonzero(f.correct))),
        ),
        Check(sheet, "B11:G11", "ω — formula (4)", row(lambda f: f.omega)),
        Check(sheet, "B12:G12", "rank of ω", lambda t: _rows(weight_ranks(t.omegas))),
        Check(
            sheet,
            "B16:G25",
            "g(S, k) = f_k(μ_S)",
            lambda t: _rows(np.column_stack([f.object_membership for f in t.features])),
        ),
        Check(
            sheet,
            "B29:G38",
            "correct side of G_k",
            lambda t: _rows(np.column_stack([f.correct for f in t.features]).astype(int)),
        ),
    ]


def _contribution_checks() -> list[Check]:
    sheet = "Ψ(r) Contribution & Weight"

    def alpha(j: int, c: int) -> Callable[[ContextTrace], Grid]:
        return lambda t: [[int(f.alpha[j, c]) for f in t.features]]

    return [
        Check(sheet, "B5:G5", "α¹₁", alpha(0, 0)),
        Check(sheet, "B6:G6", "α²₁", alpha(0, 1)),
        Check(sheet, "B7:G7", "α¹₂", alpha(1, 0)),
        Check(sheet, "B8:G8", "α²₂", alpha(1, 1)),
        Check(sheet, "B12:G12", "weight ω", lambda t: _rows(t.weights)),
        Check(
            sheet,
            "B13:G14",
            "η(1), η(2) — formula (6)",
            lambda t: _rows(np.array([f.eta for f in t.features]).T),
        ),
        Check(
            sheet,
            "B18:G27",
            "Ψ(r) as contribution values ηᵤ(a_tu)",
            lambda t: _rows(t.contributions),
        ),
        Check(sheet, "H18:H27", "class", lambda t: _column(_labels(t, t.class_index))),
        Check(sheet, "B31:G31", "weights", lambda t: _rows(t.weights)),
        Check(sheet, "B32:G32", "rank of the weights", lambda t: _rows(weight_ranks(t.weights))),
    ]


def workbook_checks() -> list[Check]:
    """The map of the experiment workbook, Steps 1–8, in sheet order."""
    return [
        *_parameter_checks(),
        *_normalized_checks(),
        *_distance_checks(),
        *_neighbour_checks(),
        *_mu_checks(),
        *_psi_checks(),
        *_membership_checks(),
        *_informativeness_checks(),
        *_contribution_checks(),
    ]


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


def run_check(check: Check, worksheet: Any, trace: ContextTrace, tolerance: float) -> CheckResult:
    """Compare one range of a worksheet with the values of ``trace``."""
    utils = importlib.import_module("openpyxl.utils")
    min_col, min_row, max_col, max_row = utils.range_boundaries(check.ref)
    expected = check.expected(trace)
    shape = (max_row - min_row + 1, max_col - min_col + 1)
    if (len(expected), len(expected[0]) if expected else 0) != shape:
        raise ValidationError(f"{check.sheet}!{check.ref}: the map yields a different shape")
    mismatches: list[str] = []
    largest = 0.0
    for r, row in enumerate(expected):
        for c, value in enumerate(row):
            cell = worksheet.cell(min_row + r, min_col + c)
            difference = _compare(cell.value, value, tolerance)
            if difference is None:
                mismatches.append(
                    f"{cell.coordinate}: workbook {_show(cell.value)} ≠ package {_show(value)}"
                )
            else:
                largest = max(largest, difference)
    return CheckResult(
        check.sheet, check.ref, check.what, shape[0] * shape[1], largest, tuple(mismatches)
    )


def _check_layout(trace: ContextTrace) -> None:
    labels = tuple(operator.label for operator in trace.operators)
    problems = []
    if trace.m != WORKBOOK_OBJECTS:
        problems.append(f"{trace.m} objects (the layout has {WORKBOOK_OBJECTS})")
    if labels != WORKBOOK_OPERATORS:
        problems.append(f"operators {labels} (the layout has {WORKBOOK_OPERATORS})")
    if trace.permitted_k.count != WORKBOOK_KS:
        problems.append(
            f"{trace.permitted_k.count} permitted k (the layout has a column for {WORKBOOK_KS})"
        )
    if problems:
        raise ValidationError("the workbook layout does not match the data: " + "; ".join(problems))


def validate_workbook(
    path: Path | str,
    config: ExperimentConfig | None = None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ValidationReport:
    """Validate the package against the experiment workbook (Steps 1–8).

    The data are read from the workbook's own *Dataset* sheet; the expected values are the
    workbook's cached cell values (as last calculated by Excel or LibreOffice).

    Args:
        path: The experiment workbook.
        config: Configuration to fit (the template preset by default; Steps 1–8 are the same
            for both presets).
        tolerance: Largest accepted absolute difference of numbers.

    Raises:
        ValidationError: The workbook cannot be read or its layout does not match the data.
        DatasetError: The *Dataset* sheet cannot be read.
    """
    file = Path(path)
    dataset = load_dataset(file, "cs-workbook")
    trace = fit_context(dataset, config).trace
    _check_layout(trace)
    openpyxl = importlib.import_module("openpyxl")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            workbook = openpyxl.load_workbook(file, data_only=True)
    except Exception as error:  # openpyxl raises many types for damaged files
        raise ValidationError(f"{file.name}: cannot read the workbook ({error})") from error
    try:
        results = []
        for check in workbook_checks():
            if check.sheet not in workbook.sheetnames:
                raise ValidationError(f"{file.name}: sheet '{check.sheet}' is missing")
            results.append(run_check(check, workbook[check.sheet], trace, tolerance))
    finally:
        workbook.close()
    return ValidationReport(str(file), tolerance, tuple(results))
