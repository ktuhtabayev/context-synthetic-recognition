"""What the sheet builders share: the build context, sheet names and size limits."""

from __future__ import annotations

from dataclasses import dataclass, field

from openpyxl.workbook import Workbook

from context_synthetic_recognition.export.excel.styles import StyleBook
from context_synthetic_recognition.export.excel.writer import SheetWriter, sheet_name
from context_synthetic_recognition.export.view import RunView

ZHURAVLYOV = "zhuravlyov"
"""Registry name of the default metric."""

OVERVIEW = "Overview"
PARAMETERS = "Parameters"
DEVIATIONS = "Template Deviations"
DATASET = "Dataset"
QUANTITATIVE = "Quantitative"
NOMINAL = "Nominal"
NORMALIZED = "Normalized Dataset"
DISTANCES = "Zhuravlyov Distances"
"""The workbook's sheet is *Zhuravlev Distances*; the mirror uses the agreed spelling (ADR-023)."""
DISTANCES_OTHER = "Distances"
"""Name of the distance sheet when an operator uses another metric."""
MU = "Synthetic Features (k-NN)"
PSI = "Ψ(r) Binary Features"
MEMBERSHIP = "Membership & Stability"
OMEGA = "Informativeness ω"
CONTRIBUTION = "Ψ(r) Contribution & Weight"
META_DATASET = "Dataset for Meta-algorithm"
BRACE = "Brace for Meta-algorithm"
META = "Meta-algorithm"
ALL_OBJECTS = "Meta-algorithm (All Objects)"
MARGINS = "Margin Analysis"
ACCURACY = "Accuracy"
CONFUSION = "Confusion Matrix"
PRF1 = "Precision, Recall, F1 Score"
ROC = "ROC Curve & AUC"
SENSITIVITY = "Sensitivity (Switches)"
PROPERTIES = "Model Properties"
VALIDATION = "Validation"

PROTOCOL_SHEETS = {
    "leave-one-out": "Leave-One-Out",
    "stratified-k-fold": "Stratified K-Fold",
    "repeated-k-fold": "Repeated K-Fold",
    "hold-out": "Hold-Out",
}
"""Sheet of every hold-out protocol's folds."""


@dataclass(frozen=True)
class ExcelOptions:
    """How much detail the mirror writes on large data."""

    max_sheet_cells: int = 150_000
    """Budget of the repeated per-object blocks of one sheet (neighbour blocks, HAG candidate
    blocks, the B1/B2 blocks of every training object). When a sheet's blocks would exceed it,
    only as many blocks as fit are written and the sheet says so."""
    max_matrix_cells: int = 250_000
    """Largest object × object table (distances, ranks) written in full — about 500 objects;
    beyond it the sheet holds a note pointing to the CSV export."""
    check_overlaps: bool = False
    """Raise if a cell is written twice (used by the tests to prove that generated layouts do
    not overlap)."""


@dataclass(frozen=True)
class SheetPlan:
    """The names of the sheets whose number depends on the run."""

    distances: str
    neighbours: tuple[str, ...]
    """One *Sorted Neighbors* sheet per base operator."""
    greedy: tuple[str, ...]
    """One *Greedy upon Weight* sheet per HAG iteration laid out."""
    protocols: dict[str, str]
    """Hold-out protocol → the sheet with its folds."""
    quantitative: bool
    nominal: bool
    sensitivity: bool


def greedy_sheets(view: RunView) -> int:
    """HAG iterations the mirror lays out: every possible one up to min(ϰ, r) − 1.

    An iteration the grouping did not reach is shown as "not executed", as in the workbook.
    """
    return max(view.slots - 1, len(view.hag.iterations))


def plan_sheets(view: RunView) -> SheetPlan:
    """Decide the names of the operator, iteration and protocol sheets."""
    trace = view.trace
    taken: list[str] = []
    neighbours = []
    for operator in trace.operators:
        name = sheet_name(f"Sorted Neighbors ({operator.label})", taken)
        taken.append(name)
        neighbours.append(name)
    greedy = tuple(f"Greedy upon Weight ({j}-Latent)" for j in range(1, greedy_sheets(view) + 1))
    protocols = {
        p.protocol: PROTOCOL_SHEETS.get(p.protocol, sheet_name(p.protocol.title()))
        for p in view.cross_validations
    }
    only_default_metric = all(o.metric == ZHURAVLYOV for o in trace.operators)
    quantitative = trace.scaling.quantitative
    return SheetPlan(
        distances=DISTANCES if only_default_metric else DISTANCES_OTHER,
        neighbours=tuple(neighbours),
        greedy=greedy,
        protocols=protocols,
        quantitative=bool(quantitative.any()),
        nominal=bool((~quantitative).any()),
        sensitivity=view.sensitivity is not None,
    )


@dataclass
class Book:
    """The workbook being built."""

    view: RunView
    workbook: Workbook
    styles: StyleBook
    options: ExcelOptions
    plan: SheetPlan
    omitted: list[str] = field(default_factory=list)
    """What was left out because of the size limits (also written on the sheets concerned)."""
    sheets: list[SheetWriter] = field(default_factory=list)
    """The sheets written so far, in creation order."""

    def sheet(self, title: str, tab: str, *, freeze: str | None = "A3") -> SheetWriter:
        """Add a worksheet."""
        writer = SheetWriter(
            self.workbook,
            self.styles,
            title,
            tab,
            freeze=freeze,
            check_overlaps=self.options.check_overlaps,
        )
        self.sheets.append(writer)
        return writer

    @property
    def cells(self) -> int:
        """Cells written so far."""
        return sum(sheet.cells for sheet in self.sheets)

    def omit(self, message: str) -> str:
        """Record something left out because of its size and return the message for the sheet."""
        self.omitted.append(message)
        return message


def blocks_that_fit(count: int, cells_per_block: int, budget: int) -> int:
    """How many of ``count`` equal blocks fit the budget (at least one)."""
    if count * cells_per_block <= budget:
        return count
    return max(1, budget // max(1, cells_per_block))
