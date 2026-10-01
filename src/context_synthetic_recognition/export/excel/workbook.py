"""The Excel mirror: a workbook with the sheet structure and style of the author's experiment.

:func:`build_workbook` lays out a run like the workbook *Context-Synthetic Model – Full
Experiment*: an overview with the workbook map, the parameters with both template/article
switches, the data, Steps 1–12 table by table, the evaluation and the model-property checks —
every cell a value computed by the package (no formulas, so the file reads the same in Excel,
LibreOffice, pandas and ``csr validate``).

The layout is generated for the data: one *Sorted Neighbors* sheet per base operator, one *Greedy
upon Weight* sheet per HAG iteration, a table column per permitted k. For Heart-Disease
(10, 13, 2) with the default configuration every computed cell sits where it sits in the author's
workbook.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from openpyxl.workbook import Workbook

from context_synthetic_recognition import __version__
from context_synthetic_recognition.export.excel import (
    sheets_checks,
    sheets_context,
    sheets_evaluation,
    sheets_hag,
    sheets_input,
    sheets_meta,
)
from context_synthetic_recognition.export.excel.common import (
    ACCURACY,
    ALL_OBJECTS,
    BRACE,
    CONFUSION,
    CONTRIBUTION,
    DATASET,
    DEVIATIONS,
    MARGINS,
    MEMBERSHIP,
    META,
    META_DATASET,
    MU,
    NOMINAL,
    NORMALIZED,
    OMEGA,
    OVERVIEW,
    PARAMETERS,
    PRF1,
    PROPERTIES,
    PSI,
    QUANTITATIVE,
    ROC,
    SENSITIVITY,
    VALIDATION,
    Book,
    ExcelOptions,
    plan_sheets,
)
from context_synthetic_recognition.export.excel.sheets_input import (
    is_default_operators,
    object_span,
    operator_kind,
)
from context_synthetic_recognition.export.excel.sheets_overview import overview
from context_synthetic_recognition.export.excel.styles import StyleBook
from context_synthetic_recognition.export.view import LEAVE_ONE_OUT, RunView
from context_synthetic_recognition.export.wording import join_and, protocol_title
from context_synthetic_recognition.notation import subscript


@dataclass(frozen=True)
class SheetEntry:
    """A sheet of the mirror: its place in the workbook map and how it is built."""

    name: str
    stage: str
    """The stage of the pipeline, e.g. ``Step 5 · Ψ(r) — formula (5)``."""
    description: str
    """What it contains (the workbook map of the *Overview*)."""
    build: Callable[[Book], None]


def _names(names: tuple[str, ...], limit: int = 14) -> str:
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown}, …"


def sheet_entries(book: Book) -> list[SheetEntry]:
    """The sheets of the mirror for this run, in workbook order (the *Overview* aside)."""
    view, plan = book.view, book.plan
    trace, data = view.trace, view.dataset
    m, n, r = trace.m, data.n, trace.r
    operators = trace.operators
    quantitative = tuple(
        nm for nm, q in zip(data.feature_names, data.quantitative, strict=True) if q
    )
    nominal = tuple(
        nm for nm, q in zip(data.feature_names, data.quantitative, strict=True) if not q
    )
    entries = [
        SheetEntry(
            PARAMETERS,
            "0 · Parameters",
            "k range (k_min = 3, k_max = 2·min|Kᵢ| − 3 from the class sizes), HAG parameters "
            "α, δ, ϰ, cr1, majorizer, the two template/article switches, base operators and rules.",
            sheets_input.parameters,
        ),
        SheetEntry(
            DEVIATIONS,
            "0 · Template deviations",
            "⚠1 running class centres in θ/γ and ⚠2 double majorizer in STEP 4: exact template "
            "cells, worked examples on the template data, fixes, and the effect of both switches.",
            sheets_checks.deviations,
        ),
        SheetEntry(
            DATASET,
            "1 · Input data",
            f"Original sample: {m} objects ({object_span(view).replace(' … ', '–')}) × {n} "
            f"features ({data.feature_names[0]}–{data.feature_names[-1]}), class labels "
            f"(K1 = {trace.class_sizes[0]}, K2 = {trace.class_sizes[1]}), feature-type flags "
            "(1 = quantitative, 0 = nominal).",
            sheets_input.dataset,
        ),
    ]
    if plan.quantitative:
        entries.append(
            SheetEntry(
                QUANTITATIVE,
                "1 · Input data",
                f"The {len(quantitative)} quantitative features ({_names(quantitative)}) with "
                "class labels.",
                sheets_input.quantitative,
            )
        )
    if plan.nominal:
        entries.append(
            SheetEntry(
                NOMINAL,
                "1 · Input data",
                f"The {len(nominal)} nominal features ({_names(nominal)}) with class labels.",
                sheets_input.nominal,
            )
        )
    labels = ", ".join(o.label for o in operators)
    entries += [
        SheetEntry(
            NORMALIZED,
            "Step 1 · Scale unification",
            "Quantitative features mapped to [0, 1] by (x − min)/(max − min); nominal codes "
            "unchanged; type, min and max rows.",
            sheets_input.normalized,
        ),
        SheetEntry(
            plan.distances,
            "Step 2 · Distances",
            "Three base operators: Zhuravlyov distance ρ, its quantitative part ρ_I and its "
            "nominal part ρ_J, for all pairs."
            if is_default_operators(view)
            else f"{len(operators)} base operator(s) — {labels}: the distances of all pairs.",
            sheets_input.distances,
        ),
    ]
    for index, operator in enumerate(operators):
        kind = operator_kind(view, operator)
        if index == 0:
            text = (
                f"Rank matrix and one block per object under {operator.label}: neighbours with "
                "original index, distance, class, same-class count μ and K1 count χ₁; permitted "
                "k marked."
            )
        elif kind == "quantitative":
            text = f"The same for the quantitative operator {operator.label}."
        elif kind == "nominal":
            text = (
                f"The same for the nominal operator {operator.label} "
                "(many ties — see Model Properties)."
            )
        else:
            text = f"The same for the operator {operator.label}."
        entries.append(
            SheetEntry(
                plan.neighbours[index],
                f"Step 3 · Sorting ({operator.label})",
                text,
                lambda b, i=index, o=operator: sheets_context.sorted_neighbours(b, i, o),  # type: ignore[misc]
            )
        )
    entries += [
        SheetEntry(
            MU,
            "Step 4 · Same-class counts μ",
            "Synthetic Features based on k-NN: μ = same-class neighbours for every permitted k "
            "and operator (training-side gradations).",
            sheets_context.same_class_counts,
        ),
        SheetEntry(
            PSI,
            "Step 5 · Ψ(r) — formula (5)",
            f"Feature map a₁ … a{subscript(r)} = (operator, k); χ₁, χ₂ and the class-free "
            "synthetic features aᵤ ∈ {1, 2}.",
            sheets_context.psi,
        ),
        SheetEntry(
            MEMBERSHIP,
            "Step 6 · Formulas (1), (2)",
            "Membership f_k(μ) and stability g_k per feature, the meta-object vector, and Task 2 "
            "— bit representations with their stability.",
            sheets_context.membership,
        ),
        SheetEntry(
            OMEGA,
            "Step 7 · Formulas (3), (4)",
            "Boundary G_k = (q₁ + q₂)/2, g(S, k) per object, correct-side flags, informativeness "
            "ω and its rank.",
            sheets_context.informativeness,
        ),
        SheetEntry(
            CONTRIBUTION,
            "Step 8 · Formula (6)",
            "Gradation counts, weights ω, contributions η(1), η(2), Ψ(r) as contribution values "
            "and ranks — the HAG input (template: Dataset (Contribution & Weight)).",
            sheets_context.contribution,
        ),
    ]
    blocks = (
        "STEP 3 candidate blocks (b, majorizer, centres, θ, γ, θ/γ), STEP 3 summary, STEP 4 "
        "update and stopping rule"
    )
    for j, name in enumerate(plan.greedy, start=1):
        start = "STEP 1–2 (first feature = max ω), " if j == 1 else ""
        entries.append(
            SheetEntry(
                name,
                f"Step 9 · Latent feature {j}",
                f"{start}{blocks} → r{subscript(j)}.",
                lambda b, j=j: sheets_hag.greedy(b, j),  # type: ignore[misc]
            )
        )
    protocols = [protocol_title(p.protocol) for p in view.result.protocols]
    methods = join_and(protocols)
    methods_and_baselines = join_and([*protocols, "the baselines"])
    entries += [
        SheetEntry(
            META_DATASET,
            "Step 10 · Meta-dataset",
            "TUPLAM order, initial features y₀ … y_p (contribution values) and latent features "
            "r₁ … r_p; Y = (y, r).",
            sheets_meta.meta_dataset,
        ),
        SheetEntry(
            BRACE,
            "Step 11 · Preparation",
            "Training description (aᵢ, dᵢ), sorted by class; a new object, rescaled, its "
            "distances, ranks, χ₁ and Ψ(r) computed without its class.",
            sheets_meta.brace,
        ),
        SheetEntry(
            META,
            "Step 12 · Classification",
            "Meta-algorithm Steps 1–5 for the new object: B1/B2 with original-feature and "
            "latent-sign conditions, Step 3 checks, Step 4 class, Step 5 end.",
            sheets_meta.meta,
        ),
        SheetEntry(
            ALL_OBJECTS,
            "Step 12 · Resubstitution",
            "The meta-algorithm for every training object (B1/B2 flags per step), scores, "
            "predictions — training correctness (Definition 2).",
            sheets_meta.all_objects,
        ),
        SheetEntry(
            MARGINS,
            "Evaluation · Margins",
            "Margins of r₁ … r_p with and without majorizer: boundary b, left/right boundaries, "
            "widths, object margins, comparison.",
            sheets_evaluation.margins,
        ),
        SheetEntry(
            ACCURACY,
            "Evaluation · Accuracy",
            f"Accuracy, coverage, TP/TN/FP/FN and refusals for {methods_and_baselines}.",
            sheets_evaluation.accuracy,
        ),
        SheetEntry(
            CONFUSION,
            "Evaluation · Confusion",
            f"Confusion matrices (actual × predicted, refusals separate) for {methods}.",
            sheets_evaluation.confusion,
        ),
        SheetEntry(
            PRF1,
            "Evaluation · P / R / F1",
            "Precision, recall and F1 per class and macro average for every method.",
            sheets_evaluation.precision_recall,
        ),
        SheetEntry(
            ROC,
            "Evaluation · ROC",
            f"Scores score₁ − score₂, Mann–Whitney AUC and ROC points (TPR, FPR) for {methods}.",
            sheets_evaluation.roc,
        ),
    ]
    for protocol in view.cross_validations:
        if protocol.protocol == LEAVE_ONE_OUT:
            text = (
                f"Leave-one-out folds (whole pipeline re-fitted on {m - 1} objects): permitted k, "
                "TUPLAM, description, scores, prediction; the baselines."
            )
        else:
            text = (
                f"The folds of the {protocol_title(protocol.protocol)} (whole pipeline re-fitted "
                "per fold): permitted k, TUPLAM, description, scores, prediction; the baselines."
            )
        entries.append(
            SheetEntry(
                plan.protocols[protocol.protocol],
                "Evaluation · Generalization",
                text,
                lambda b, p=protocol: sheets_evaluation.folds(b, p),  # type: ignore[misc]
            )
        )
    if plan.sensitivity:
        entries.append(
            SheetEntry(
                SENSITIVITY,
                "Evaluation · Sensitivity",
                "The two template-vs-article switches: results of all four combinations and the "
                "package’s replication of the templates.",
                sheets_evaluation.sensitivity,
            )
        )
    entries += [
        SheetEntry(
            PROPERTIES,
            "Checks · Definitions 1–6",
            "Determinacy, training and generalization correctness, sufficiency, contextual "
            "equivalence of operators, ties, and the Theorem checks.",
            sheets_checks.properties,
        ),
        SheetEntry(
            VALIDATION,
            "Checks · Validation",
            "The key values of the run as computed by the package; the highlighted template "
            "deviations; how to re-validate this file.",
            sheets_checks.validation,
        ),
    ]
    return entries


def build_workbook(view: RunView, options: ExcelOptions | None = None) -> tuple[Workbook, Book]:
    """Lay out a run as a workbook with the structure and style of the Excel experiment.

    Returns:
        The workbook and its build record (``Book.omitted`` lists what was left out because of
        the size limits).
    """
    workbook = Workbook()
    workbook.remove(workbook.worksheets[0])
    book = Book(view, workbook, StyleBook(workbook), options or ExcelOptions(), plan_sheets(view))
    entries = sheet_entries(book)
    for entry in entries:
        entry.build(book)
    overview(book, entries)
    workbook.move_sheet(OVERVIEW, offset=-(len(workbook.sheetnames) - 1))
    workbook.active = 0
    for sheet in workbook.worksheets:
        sheet.sheet_view.tabSelected = sheet.title == OVERVIEW
    properties = workbook.properties
    properties.title = f"Context-Synthetic Model — {view.dataset.name}"
    properties.subject = (
        "Ψ(r) by (5), ω by (4), η by (6), hierarchical agglomerative grouping, meta-algorithm "
        "and evaluation"
    )
    properties.creator = f"context-synthetic-recognition {__version__}"
    if view.created is not None:
        # openpyxl stores naive UTC timestamps
        properties.created = properties.modified = view.created.replace(tzinfo=None)
    return workbook, book


def write_workbook(view: RunView, path: Path, options: ExcelOptions | None = None) -> Book:
    """Write the Excel mirror of a run to ``path`` and return its build record."""
    workbook, book = build_workbook(view, options)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    return book


__all__ = ["ExcelOptions", "SheetEntry", "build_workbook", "sheet_entries", "write_workbook"]
