"""The *Overview* sheet.

The workbook map, the pipeline specification, the key results, the two template deviations, the
colour legend and the notes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from context_synthetic_recognition import __version__
from context_synthetic_recognition.export.excel.common import (
    ACCURACY,
    BRACE,
    CONTRIBUTION,
    DATASET,
    DEVIATIONS,
    MEMBERSHIP,
    META,
    META_DATASET,
    NORMALIZED,
    OMEGA,
    OVERVIEW,
    PARAMETERS,
    PSI,
    ROC,
    VALIDATION,
    Book,
)
from context_synthetic_recognition.export.excel.sheets_checks import switch_effect
from context_synthetic_recognition.export.excel.sheets_input import (
    is_default_operators,
    object_span,
)
from context_synthetic_recognition.export.excel.writer import SheetWriter
from context_synthetic_recognition.export.view import RunView
from context_synthetic_recognition.export.wording import decision_label, protocol_title

if TYPE_CHECKING:
    from context_synthetic_recognition.export.excel.workbook import SheetEntry

GAP_HEIGHT = 10.0
SECTION_HEIGHT = 24.0
HEAD_HEIGHT = 20.0


def _section(sheet: SheetWriter, row: int, text: str) -> None:
    sheet.put(row, 2, text, "section")
    for column in (3, 4, 5):
        sheet.put(row, column, None, "section")
    sheet.height(row, SECTION_HEIGHT)


def _head(sheet: SheetWriter, row: int, titles: Sequence[str], centred: Sequence[int] = ()) -> None:
    sheet.put(row, 2, "#", "overview_head_center")
    for offset, title in enumerate(titles):
        style = "overview_head_center" if offset in centred else "overview_head"
        sheet.put(row, 3 + offset, title, style)
    sheet.height(row, HEAD_HEIGHT)


def _specification(book: Book) -> list[tuple[str, str, str]]:
    """Article element → sheet → the function of the package that computes it."""
    plan = book.plan
    labels = ", ".join(o.label for o in book.view.trace.operators)
    return [
        (
            "Data, feature types, classes",
            DATASET,
            (
                "E₀ = {S₁ … Sₘ}, X(n) heterogeneous, K1, K2   →   data.load_dataset(path) → "
                "Dataset(X, y, feature_types)"
            ),
        ),
        (
            "Scale unification",
            NORMALIZED,
            (
                "x′ = (x − min)/(max − min) for j ∈ I, training min/max   →   "
                "core.normalizers.minmax → Scaling.transform"
            ),
        ),
        (
            f"Base operators {labels}",
            plan.distances,
            (
                "Zhuravlyov metric on I ∪ J, on I and on J (article: variants of Ψ)   →   "
                "core.metrics.zhuravlyov, core.operators.BaseOperator.distances"
            ),
        ),
        (
            "Nested neighbourhoods",
            plan.neighbours[0],
            (
                "order by (ρ, original index), self excluded; μ and χ₁ per rank   →   "
                "core.neighbours.neighbour_order, core.encoders.k1_counts"
            ),
        ),
        (
            "Permitted k",
            PARAMETERS,
            "k = 3, 5, …, 2·min|Kᵢ| − 3   →   core.k_strategies.permitted_k",
        ),
        (
            "Formula (5) — Ψ(r)",
            PSI,
            "aᵤ = 1 if χ₁ > [k/2], 2 if χ₂ > [k/2]   →   core.encoders (formula-5)",
        ),
        (
            "Formulas (1), (2)",
            MEMBERSHIP,
            (
                "f_k(μ), stability g_k (β = k), meta-object, bit masks   →   "
                "core.membership.membership_table, stability, bit_masks"
            ),
        ),
        (
            "Formulas (3), (4)",
            OMEGA,
            (
                "G_k = (q₁ + q₂)/2, ω = share of objects on the correct side   →   "
                "core.membership.boundary, informativeness"
            ),
        ),
        (
            "Formula (6)",
            CONTRIBUTION,
            (
                "η(j) = ω(α¹ⱼ/|K1| − α²ⱼ/|K2|), Ψ(r) as contribution values   →   "
                "core.contributions.contributions"
            ),
        ),
        (
            "HAG, Steps 1–5",
            plan.greedy[0] if plan.greedy else CONTRIBUTION,
            (
                "u = argmax ω, b = R + η, majorizer, θ/γ, q = argmin, stop rule   →   "
                "core.hag.hag(C, w, y, config.hag)"
            ),
        ),
        (
            "Meta-description Y",
            META_DATASET,
            (
                "Y = (y₀ … y_p, r₁ … r_p), aᵢ and dᵢ   →   core.model.fit_model → "
                "CSModel.meta_dataset, CSModel.description"
            ),
        ),
        (
            "New object (Theorem)",
            BRACE,
            "context relative to E without K(S)   →   CSModel.represent(x)",
        ),
        (
            "Meta-algorithm, Steps 1–5",
            META,
            (
                "B1, B2 filtering, score comparison, refusal   →   core.meta.meta_classify, "
                "CSModel.classify(x)"
            ),
        ),
        (
            "Evaluation (Definitions 2, 3)",
            ACCURACY,
            (
                "resubstitution, leave-one-out, baselines, AUC, margins   →   "
                "evaluation.protocols.run_protocol, services.runner.run_experiment"
            ),
        ),
        (
            "Validation",
            VALIDATION,
            (
                "workbook vs package; template replication   →   "
                "services.validation.validate_workbook (csr validate)"
            ),
        ),
    ]


def _key_results(view: RunView) -> list[tuple[str, Any, str, str]]:
    """Figure, value, sheet, value style."""
    trace, hag, result = view.trace, view.hag, view.result
    formula = trace.permitted_k.rule == "formula"
    rows: list[tuple[str, Any, str, str]] = [
        (
            "Permitted k  (k_min = 3, k_max = 2·min|Kᵢ| − 3)"
            if formula
            else f"Permitted k  (strategy “{trace.permitted_k.rule}”)",
            trace.permitted_k.label,
            PARAMETERS,
            "overview_value",
        ),
        ("Synthetic features r = |Ψ(r)|", trace.r, PARAMETERS, "overview_value"),
        (
            "TUPLAM (selected synthetic features, in order)",
            hag.label,
            META_DATASET,
            "overview_value",
        ),
        ("Latent (additional) features p", hag.p, META_DATASET, "overview_value"),
    ]
    definitions = {"resubstitution": " (Definition 2)", "leave-one-out": " (Definition 3)"}
    for protocol in result.protocols:
        name = protocol_title(protocol.protocol)
        title = name[:1].upper() + name[1:]
        rows.append(
            (
                f"{title} accuracy{definitions.get(protocol.protocol, '')}",
                result.metrics(protocol.predictions).accuracy,
                ACCURACY,
                "overview_percent",
            )
        )
    for protocol in result.protocols:
        rows.append(
            (
                f"AUC — {protocol_title(protocol.protocol)}",
                result.auc(protocol.predictions),
                ROC,
                "overview_percent",
            )
        )
    demo = view.new_object
    what = (
        "a new object"
        if demo.exclude is None
        else f"default input = {trace.object_ids[demo.exclude]}, excluded"
    )
    rows.append(
        (
            f"New object ({what}): class",
            decision_label(demo.decision, trace.classes),
            META,
            "overview_value",
        )
    )
    return rows


def overview(book: Book, entries: Sequence[SheetEntry]) -> None:
    """Sheet *Overview*: what the workbook holds and the run's key results."""
    view = book.view
    trace, data = view.trace, view.dataset
    sheet = book.sheet(OVERVIEW, "info", freeze=None)
    sheet.widths({1: 2, 2: 6, 3: 40, 4: 30, 5: 118})
    sheet.height(1, 9.0)
    sheet.put(2, 2, "Context-Synthetic Model — Full Experiment", "overview_title")
    sheet.join(2, 2, 4)
    sheet.height(2, 40.0)
    operators = (
        "Zhuravlyov metric on three feature sets"
        if is_default_operators(view)
        else f"base operators {', '.join(o.label for o in trace.operators)}"
    )
    sheet.put(
        3,
        2,
        f"{data.name}  ·  {operators}  ·  Ψ(r) by (5)  ·  ω by (4)  ·  η by (6)  ·  HAG  ·  "
        "meta-algorithm  ·  evaluation",
        "overview_subtitle",
    )
    sheet.join(3, 2, 4)
    sheet.height(3, 22.0)
    sheet.height(4, GAP_HEIGHT)

    # ---- workbook map
    row = 5
    _section(sheet, row, "Workbook map")
    _head(sheet, row + 1, ["Sheet", "Stage", "What it contains"])
    row += 2
    for number, entry in enumerate(entries, start=1):
        sheet.put(row, 2, number, "overview_number")
        sheet.link(row, 3, entry.name, entry.name)
        sheet.put(row, 4, entry.stage, "overview_text")
        sheet.put(row, 5, entry.description, "overview_text")
        sheet.height(row, 31.0)
        row += 1
    sheet.height(row, GAP_HEIGHT)

    # ---- pipeline specification
    row += 1
    _section(
        sheet, row, "Algorithm — pipeline specification (article element → sheet → Python function)"
    )
    _head(
        sheet,
        row + 1,
        [
            "Article element",
            "Sheet",
            "Definition  →  Python function (context_synthetic_recognition)",
        ],
    )
    row += 2
    for number, (element, target, text) in enumerate(_specification(book), start=1):
        sheet.put(row, 2, number, "overview_number")
        sheet.put(row, 3, element, "overview_text_bold")
        sheet.link(row, 4, target, target)
        sheet.put(row, 5, text, "overview_spec")
        sheet.height(row, 22.0)
        row += 1
    sheet.height(row, GAP_HEIGHT)

    # ---- key results
    row += 1
    _section(sheet, row, "Key results")
    _head(sheet, row + 1, ["Figure", "Value", "Shown on"], centred=(1,))
    row += 2
    for number, (figure, value, target, style) in enumerate(_key_results(view), start=1):
        sheet.put(row, 2, number, "overview_number")
        sheet.put(row, 3, figure, "overview_text_bold")
        sheet.put(row, 4, value, style)
        sheet.put(row, 5, target, "overview_ref")
        sheet.height(row, 19.0)
        row += 1
    sheet.height(row, GAP_HEIGHT)

    # ---- the two template deviations
    row += 1
    _section(
        sheet,
        row,
        "⚠ Template calculations that differ from the article (default = template; configuration "
        "switches, shown on the Parameters sheet)",
    )
    effect = switch_effect(view)
    lines = [
        (
            "•  θ and γ are measured from running partial class means instead of the final class "
            "means "
            "M₁ and M₂;"
        ),
        "•  in STEP 4 the majorizer is applied twice instead of once.",
        f"   Effect on this dataset — {effect}.",
    ]
    row += 1
    for line in lines:
        sheet.merge(row, 2, 4, line, "warn_note")
        sheet.height(row, 21.0)
        row += 1
    sheet.link(
        row,
        2,
        "→  Template Deviations: exact template cells, worked examples, fixes and effect",
        DEVIATIONS,
    )
    for column, style in ((3, "tail"), (4, "tail"), (5, "tail_end")):
        sheet.put(row, column, None, style)
    sheet.join(row, 2, 4)
    sheet.height(row, 21.0)
    row += 1
    sheet.height(row, GAP_HEIGHT)

    # ---- colour legend
    row += 1
    _section(sheet, row, "Colour legend")
    first, last = data.feature_names[0], data.feature_names[-1]
    span = object_span(view)
    legend = [
        ("legend_title", "Sheet title bar"),
        ("overview_head_center", "Table / block / STEP title bar"),
        ("legend_header", f"Column header ({first} … {last}, {span}, aᵤ, k, rank)"),
        ("legend_label", f"Object label {span} and row captions"),
        ("legend_quantitative", "Quantitative feature values"),
        ("legend_nominal", "Nominal feature values"),
        ("legend_class", f"Class label ({trace.classes[0]} / {trace.classes[1]})"),
        ("legend_synthetic", "Synthetic feature values (μ, aᵤ, f, masks)"),
        ("legend_latent_header", "Permitted-k neighbourhood (Step 3) / latent header"),
        ("legend_good", "Selected / same-class / correct"),
        ("legend_initial", "Initial features y in the meta-dataset"),
        ("legend_latent", "Additional (latent) features r"),
        ("legend_key", "Key label / parameter"),
        ("legend_input", "Input value (blue) — Parameters, new object"),
        ("legend_warn1", "Warning / ⚠1 running class centres (HAG columns I–L)"),
        ("legend_warn2", "⚠2 STEP 4 majorizer applied twice (HAG STEP 4 columns H–I)"),
    ]
    row += 1
    for style, text in legend:
        sheet.put(row, 2, "Aa", style)
        sheet.merge(row, 3, 3, text, "overview_text")
        sheet.height(row, 19.0)
        row += 1
    sheet.height(row, GAP_HEIGHT)

    # ---- notes
    row += 1
    _section(sheet, row, "Notes")
    run = f"the run {view.run_id}" if view.run_id else "a run that was not saved"
    decimals = view.config.context.distance_decimals
    notes = [
        (
            f"•  This workbook was generated by context-synthetic-recognition {__version__} from "
            f"{run}: every cell is a computed value, laid out like the author’s Excel experiment "
            "(the same sheets, tables, colours and notes)."
        ),
        (
            "•  Nothing recalculates: to change a value, a class, a type flag or a parameter, "
            "change "
            "the dataset or the configuration and run again. `csr validate --against <this file>` "
            "re-computes the experiment from the Dataset and Parameters sheets and compares every "
            "computed cell."
        ),
        (
            f"•  Distances are rounded to {decimals} decimals; ties are broken by the smaller "
            "original index — the rules of the Excel experiment and of its reference engine."
        ),
        (
            "•  The template workbooks (k-NN synthetic features, HAG, meta-algorithm, model "
            "evaluation) define the calculations this layout follows; the two template "
            "calculations "
            "that differ from the article are switches."
        ),
        *(f"•  Left out for size — {message}" for message in book.omitted),
    ]
    row += 1
    for note in notes:
        sheet.put(row, 2, note, "overview_note")
        sheet.join(row, 2, 4)
        sheet.height(row, 30.0)
        row += 1
