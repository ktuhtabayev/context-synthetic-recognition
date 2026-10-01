"""The input sheets and Steps 1–2.

*Parameters*, *Dataset*, *Quantitative*, *Nominal*, *Normalized Dataset* and the distances of the
base operators.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from context_synthetic_recognition import __version__
from context_synthetic_recognition.config.io import config_hash
from context_synthetic_recognition.config.models import CentreMode, HAGConfig
from context_synthetic_recognition.config.presets import matching_preset
from context_synthetic_recognition.core.majorizers import MAJORIZERS
from context_synthetic_recognition.core.trace import OperatorContext
from context_synthetic_recognition.export.excel.common import (
    DATASET,
    NOMINAL,
    NORMALIZED,
    PARAMETERS,
    QUANTITATIVE,
    ZHURAVLYOV,
    Book,
)
from context_synthetic_recognition.export.excel.writer import SheetWriter
from context_synthetic_recognition.export.excel.writer import column_letter as _letter
from context_synthetic_recognition.export.tables import plain
from context_synthetic_recognition.export.view import RunView
from context_synthetic_recognition.notation import DASH

FEATURE_WIDTH_RAW = 9.77734375
"""Width of a feature column on the *Dataset* sheet (the template's)."""


def operator_kind(view: RunView, operator: OperatorContext) -> str:
    """``all``, ``quantitative`` or ``nominal`` (the operators ρ, ρ_I, ρ_J), else ``custom``."""
    quantitative = view.trace.scaling.quantitative
    used = np.zeros(quantitative.size, dtype=bool)
    used[operator.features] = True
    if operator.metric != ZHURAVLYOV:
        return "custom"
    if used.all():
        return "all"
    if quantitative.any() and np.array_equal(used, quantitative):
        return "quantitative"
    if (~quantitative).any() and np.array_equal(used, ~quantitative):
        return "nominal"
    return "custom"


def is_default_operators(view: RunView) -> bool:
    """Whether the operators are exactly ρ, ρ_I, ρ_J of the workbook."""
    kinds = [operator_kind(view, o) for o in view.trace.operators]
    return kinds == ["all", "quantitative", "nominal"]


def integral(values: Any) -> bool:
    """Whether every value is a whole number (a nominal-only distance counts features)."""
    array = np.asarray(values, dtype=np.float64)
    return bool(np.all(array == np.round(array)))


def object_span(view: RunView) -> str:
    """``S₁ … S₁₀`` — the first and the last object."""
    ids = view.trace.object_ids
    return f"{ids[0]} … {ids[-1]}"


# ---------------------------------------------------------------- Parameters


def _rows(sheet: SheetWriter, row: int, rows: list[tuple[Any, ...]]) -> int:
    """Key / value / meaning rows; a row may name its own three styles after the values."""
    for key, value, meaning, *styles in rows:
        key_style, value_style, text_style = styles or ("key", "value", "text")
        sheet.put(row, 1, key, key_style)
        sheet.put(row, 2, value, value_style)
        sheet.put(row, 3, meaning, text_style)
        row += 1
    return row


def _majorizer(hag: HAGConfig) -> tuple[str, str]:
    name = MAJORIZERS.info(hag.majorizer.name).name
    if name == "sigmoid":
        return (
            "σ(x) = 1 / (1 + e^(−x))",
            (
                "b ← b + α·σ(−b) for Sₜ ∈ K1,  b ← b − α·σ(−b) for Sₜ ∈ K2  "
                "(logistic sigmoid, as in the template)"
            ),
        )
    summary = MAJORIZERS.info(name).summary
    return (
        name,
        f"b ← b + α·ϕ(−b) for Sₜ ∈ K1,  b ← b − α·ϕ(−b) for Sₜ ∈ K2  ({summary})",
    )


def _operator_row(view: RunView, operator: OperatorContext) -> tuple[str, str, str]:
    quantitative = view.trace.scaling.quantitative
    n = quantitative.size
    count = operator.features.size
    label = operator.label
    kind = operator_kind(view, operator)
    if kind == "all":
        return (
            label,
            f"all {n} (I ∪ J)",
            f"{label}(x, y) = Σ(j∈I) |x′ⱼ − y′ⱼ| + Σ(j∈J) [xⱼ ≠ yⱼ]  — Zhuravlyov metric",
        )
    if kind == "quantitative":
        return (
            label,
            f"{count} quantitative (I)",
            f"{label}(x, y) = Σ(j∈I) |x′ⱼ − y′ⱼ|  — quantitative part of ρ",
        )
    if kind == "nominal":
        return (
            label,
            f"{count} nominal (J)",
            f"{label}(x, y) = Σ(j∈J) [xⱼ ≠ yⱼ]  — nominal part of ρ (Hamming)",
        )
    names = ", ".join(view.trace.feature_names[int(j)] for j in operator.features[:12])
    more = "" if count <= 12 else f", … ({count} features)"
    return label, f"{count} feature(s)", f"metric “{operator.metric}” on {names}{more}"


def parameters(book: Book) -> None:
    """Sheet *Parameters*: the k range, the HAG parameters with both switches, the operators."""
    view = book.view
    trace, config = view.trace, view.config
    ks, hag = trace.permitted_k, config.hag
    sheet = book.sheet(PARAMETERS, "info")
    sheet.widths({1: 46, 2: 22, 3: 110})
    sheet.title_bar(
        "Experiment parameters — the settings every stage of this run was computed with",
        3,
        height=22.0,
    )

    formula = ks.rule == "formula"
    sheet.bar(
        3,
        1,
        3,
        "k range  ·  formula-based (k_min = 3 fixed, k_max = 2 · min |Kᵢ| − 3 from the class sizes)"
        if formula
        else f"k range  ·  strategy “{ks.rule}”",
    )
    sheet.row(
        4, 1, ["Parameter", "Value", "Rule / meaning"], ["header_left", "header", "header_left"]
    )
    skipped = len(trace.skipped_features)
    rows: list[tuple[Any, ...]] = [
        ("Objects m", trace.m, f"Rows {object_span(view)} of the Dataset sheet"),
        (
            f"|K1|  (objects of class {trace.classes[0]})",
            trace.class_sizes[0],
            "Counted from the class column — never typed in",
        ),
        (f"|K2|  (objects of class {trace.classes[1]})", trace.class_sizes[1], ""),
        ("min |Kᵢ|", min(trace.class_sizes), "Smallest class"),
        (
            "k_min",
            DASH if ks.k_min is None else ks.k_min,
            "Fixed for every dataset" if formula else f"Not used by the strategy “{ks.rule}”",
        ),
        (
            "k_max = 2 · min |Kᵢ| − 3",
            DASH if ks.k_max is None else ks.k_max,
            "At k = k_max a same-class majority needs (k + 1)/2 = min |Kᵢ| − 1 neighbours",
        ),
        ("Permitted k  (odd, k_min … k_max)", ks.label, ks.note),
        ("Number of permitted k", ks.count, ""),
        (
            "Base operators",
            len(trace.operators),
            ", ".join(o.label for o in trace.operators) + " — see the table below",
        ),
        (
            "r = |Ψ(r)|  (synthetic features)",
            trace.r,
            "One synthetic feature per (operator, permitted k)"
            + (f"; {skipped} constant feature(s) skipped (ADR-030)" if skipped else ""),
        ),
        (
            "Layout check",
            "✓ the workbook has a column for every permitted k",
            "",
            "key",
            "value_left",
            "text",
        ),
    ]
    _rows(sheet, 5, rows)

    template = HAGConfig()
    at_template = (hag.alpha, hag.delta, hag.kappa, hag.cr1) == (
        template.alpha,
        template.delta,
        template.kappa,
        template.cr1,
    )
    sheet.bar(
        17,
        1,
        3,
        "Hierarchical agglomerative grouping (HAG)  ·  values of the template experiment"
        if at_template
        else "Hierarchical agglomerative grouping (HAG)",
    )
    sheet.row(
        18, 1, ["Parameter", "Value", "Rule / meaning"], ["header_left", "header", "header_left"]
    )

    def origin(value: float, default: float) -> str:
        return "Template value" if value == default else f"Template value: {default:g}"

    limit = trace.r - 1
    kappa_note = (
        f"{origin(hag.kappa, template.kappa)}; ϰ ≤ r − 1 = {limit} ✓"
        if hag.kappa <= limit
        else f"⚠ ϰ exceeds r − 1 = {limit}: the grouping ends when P is empty"
    )
    function, rule = _majorizer(hag)
    centres = 1 if hag.centres is CentreMode.RUNNING else 2
    _rows(
        sheet,
        19,
        [
            (
                "α  (regularisation of the margin, 0 < α < 1)",
                hag.alpha,
                origin(hag.alpha, template.alpha),
                "key",
                "input",
                "text",
            ),
            (
                "δ  (threshold for θ / γ, 0 < δ < 0.5)",
                hag.delta,
                origin(hag.delta, template.delta),
                "key",
                "input",
                "text",
            ),
            ("ϰ  (maximum |TUPLAM|)", hag.kappa, kappa_note, "key", "input", "text"),
            (
                "cr1  (initial value of the criterion)",
                plain(hag.cr1),
                origin(hag.cr1, template.cr1),
                "key",
                "input",
                "text",
            ),
            ("Majorizing function ϕ", function, rule, "key", "value_left", "text"),
            (
                "⚠1 Class centres in θ and γ",
                centres,
                (
                    "1 = running partial class means, as computed in the template cells  ·  "
                    "2 = final class means M₁, M₂ (article and the template formula box)  ·  "
                    "see “Template Deviations”"
                ),
                "warn1_key",
                "input",
                "warn1_text",
            ),
            (
                "⚠2 Majorizer passes in STEP 4",
                hag.step4_passes,
                (
                    "2 = template (the majorizer is applied again to b of STEP 3)  ·  "
                    "1 = article (R ← R + η_q, then the majorizer once)  ·  see “Template "
                    "Deviations”"
                ),
                "warn2_key",
                "input",
                "warn2_text",
            ),
        ],
    )

    default_metric = all(o.metric == ZHURAVLYOV for o in trace.operators)
    sheet.bar(
        27,
        1,
        3,
        "Base operators  ·  the same Zhuravlyov metric on different feature sets "
        "(article: variants of Ψ)"
        if default_metric
        else "Base operators  ·  a metric on a feature set (article: variants of Ψ)",
    )
    sheet.row(
        28, 1, ["Operator", "Feature set", "Definition"], ["header_left", "header", "header_left"]
    )
    row = _rows(sheet, 29, [_operator_row(view, o) for o in trace.operators])
    for skipped_operator in trace.skipped_operators:
        row = _rows(
            sheet,
            row,
            [(skipped_operator.label, "skipped", f"{skipped_operator.reason} (ADR-007)")],
        )

    row += 1
    sheet.bar(row, 1, 3, "Other rules")
    evaluation = config.evaluation
    positive = trace.classes[view.result.positive - 1]
    decimals = config.context.distance_decimals
    row = _rows(
        sheet,
        row + 1,
        [
            (
                "Tie rule",
                "smaller index first",
                "Equal distances are ordered by the original index of the neighbour",
            ),
            (
                "Distance rounding",
                f"{decimals} decimals",
                "So that mathematically equal distances tie exactly",
            ),
            (
                "Positive class (evaluation)",
                positive,
                (
                    f'Class {positive} is "positive" in the confusion matrix, precision, recall '
                    "and ROC"
                ),
            ),
            (
                "Meta-algorithm decision",
                "Step 4",
                "K1 if |B1|/|K1| > |B2|/|K2|,  K2 if <,  0 = refusal if equal",
            ),
        ],
    )

    row += 1
    sheet.bar(row, 1, 3, "This run  ·  what identifies it (reproducibility)")
    preset = matching_preset(config)
    created = "" if view.created is None else view.created.strftime("%Y-%m-%d %H:%M UTC")
    _rows(
        sheet,
        row + 1,
        [
            ("Run", view.run_id or "not saved", created, "key", "value_left", "text"),
            (
                "Configuration",
                config.name,
                f"preset: {preset.value if preset else 'custom'}  ·  SHA-256 {config_hash(config)}",
                "key",
                "value_left",
                "text",
            ),
            (
                "Dataset",
                view.dataset.name,
                f"SHA-256 of the content {view.dataset.content_hash()}",
                "key",
                "value_left",
                "text",
            ),
            (
                "Scale unification",
                trace.scaling.normalizer,
                "Fitted on the training objects only; nominal codes are never changed",
                "key",
                "value_left",
                "text",
            ),
            (
                "Constant synthetic features",
                "skipped" if config.synthetic.skip_constant else "kept",
                "synthetic.skip_constant (ADR-030); the workbook keeps them",
                "key",
                "value_left",
                "text",
            ),
            (
                "Score rounding",
                f"{evaluation.score_decimals} decimals",
                "score₁ − score₂ is rounded before AUC and ROC so that equal scores tie exactly",
                "key",
                "value_left",
                "text",
            ),
            ("Seed", config.seed, "Seed of every random choice (splits of randomised protocols)"),
            (
                "Computed by",
                f"context-synthetic-recognition {__version__}",
                (
                    "Every cell of this workbook is a value computed by the package; "
                    "`csr validate --against <this file>` re-computes and compares them"
                ),
                "key",
                "value_left",
                "text",
            ),
        ],
    )


# ---------------------------------------------------------------- Dataset, Quantitative, Nominal


def _data_table(
    book: Book,
    sheet: SheetWriter,
    top: int,
    left: int,
    features: np.ndarray,
    *,
    class_header: str = "header_plain",
) -> None:
    """A header row, one row per object and the type flags, for the features ``features``."""
    view = book.view
    dataset = view.dataset
    quantitative = dataset.quantitative
    names = [dataset.feature_names[int(j)] for j in features]
    last = left + len(names) + 1
    sheet.put(top, left, "№", "header_plain")
    sheet.row(top, left + 1, names, "header_open")
    sheet.put(top, last, "Class", class_header)
    styles = ["quantitative" if quantitative[int(j)] else "nominal" for j in features]
    labels = view.labels
    for t in range(dataset.m):
        row = top + 1 + t
        sheet.put(row, left, dataset.object_ids[t], "label")
        sheet.row(row, left + 1, [plain(dataset.X[t, int(j)]) for j in features], styles)
        sheet.put(row, last, labels[t], "class_")
    sheet.row(top + 1 + dataset.m, left + 1, [int(quantitative[int(j)]) for j in features], "flag")


def _class_sizes(book: Book, sheet: SheetWriter, column: int, style: str) -> None:
    sizes = book.view.trace.class_sizes
    sheet.put(2, column, f"K1 = {sizes[0]}", style)
    sheet.put(3, column, f"K2 = {sizes[1]}", style)


def dataset(book: Book) -> None:
    """Sheet *Dataset*: the original sample, its type flags and the two feature sets."""
    view = book.view
    data = view.dataset
    m, n = data.m, data.n
    quantitative = np.flatnonzero(data.quantitative)
    nominal = np.flatnonzero(data.nominal)
    sheet = book.sheet(DATASET, "data", freeze=None)
    sheet.ws.sheet_view.zoomScale = None
    sheet.ws.sheet_format.defaultRowHeight = 14.4
    sheet.merge(1, 1, n + 2, data.name, "title")
    _data_table(book, sheet, 2, 1, np.arange(n))
    side = n + 4
    _class_sizes(book, sheet, side, "size_note")
    sheet.merge(5, side, 3, "Sign of Quantitative Features - ", "key_center")
    sheet.put(5, side + 3, 1, "quantitative_flag")
    sheet.merge(6, side, 3, "Sign of Nominal Features - ", "key_center")
    sheet.put(6, side + 3, 0, "nominal_flag")

    totals, top = m + 7, m + 9
    nominal_left = quantitative.size + 5 if quantitative.size else 1
    if quantitative.size:
        sheet.merge(totals, 2, 4, "Total Number of Quantitative Features - ", "key_center")
        sheet.put(totals, 6, int(quantitative.size), "quantitative_flag")
        sheet.bar(top, 1, quantitative.size + 2, "Quantitative Features")
        _data_table(book, sheet, top + 1, 1, quantitative)
    if nominal.size:
        sheet.merge(
            totals, nominal_left + 1, 4, "Total Number of Nominal Features - ", "key_center"
        )
        sheet.put(totals, nominal_left + 5, int(nominal.size), "nominal_flag")
        sheet.bar(top, nominal_left, nominal.size + 2, "Nominal Features")
        _data_table(book, sheet, top + 1, nominal_left, nominal)

    for column in range(2, n + 2):
        sheet.ws.column_dimensions[_letter(column)].width = FEATURE_WIDTH_RAW
    sheet.ws.column_dimensions["A"].width = 8.109375
    for column in range(side, side + 3):
        sheet.ws.column_dimensions[_letter(column)].width = 10.77734375
    for row in range(1, top + m + 3):
        if row not in (m + 4, m + 5, m + 6):
            sheet.height(row, 15.6)


def _subset_sheet(book: Book, title: str, tab: str, features: np.ndarray, flag: int) -> None:
    view = book.view
    kind = "Quantitative" if flag else "Nominal"
    count = int(features.size)
    sheet = book.sheet(title, tab, freeze=None)
    sheet.ws.sheet_view.zoomScale = None
    sheet.ws.sheet_format.defaultRowHeight = 14.4
    sheet.merge(1, 1, count + 2, f"{kind} Features", "title")
    _data_table(book, sheet, 2, 1, features)
    side = count + 5
    _class_sizes(book, sheet, side, "size_note_centered" if flag else "size_note")
    sheet.merge(3, side + 2, 4, f"Color and Sign of {kind} Features - ", "key_center")
    sheet.put(3, side + 6, flag, "quantitative_flag" if flag else "nominal_flag")
    for column in range(2, count + 3):
        sheet.ws.column_dimensions[_letter(column)].width = 10.77734375
    for column in range(side, side + 7):
        sheet.ws.column_dimensions[_letter(column)].width = FEATURE_WIDTH_RAW
    sheet.ws.column_dimensions["A"].width = 14.21875 if flag else 14.109375
    for row in range(1, view.dataset.m + 4):
        sheet.height(row, 15.6)


def quantitative(book: Book) -> None:
    """Sheet *Quantitative*: the features of set I with the class labels."""
    features = np.flatnonzero(book.view.dataset.quantitative)
    _subset_sheet(book, QUANTITATIVE, "quantitative", features, 1)


def nominal(book: Book) -> None:
    """Sheet *Nominal*: the features of set J with the class labels."""
    features = np.flatnonzero(book.view.dataset.nominal)
    _subset_sheet(book, NOMINAL, "nominal", features, 0)


# ---------------------------------------------------------------- Normalized Dataset


def normalized(book: Book) -> None:
    """Sheet *Normalized Dataset* (Step 1): unified values, type flags, training min and max."""
    view = book.view
    trace = view.trace
    m, n = trace.m, len(trace.feature_names)
    scaling = trace.scaling
    is_quantitative = scaling.quantitative
    sheet = book.sheet(NORMALIZED, "context")
    sheet.widths({1: 10, (2, n + 2): 11})
    minmax = scaling.normalizer == "minmax"
    sheet.title_bar(
        "Step 1 · Normalized dataset — quantitative features mapped to [0, 1], "
        "nominal features unchanged"
        if minmax
        else f"Step 1 · Normalized dataset — scale unification “{scaling.normalizer}”, "
        "nominal features unchanged",
        n + 2,
    )
    sheet.bar(
        3,
        1,
        n + 2,
        f"{view.dataset.name} after scale unification:  x′ = (x − min) / (max − min)  for j ∈ I"
        if minmax
        else f"{view.dataset.name} after scale unification ({scaling.normalizer})",
    )
    sheet.row(4, 1, ["№", *trace.feature_names, "Class"], "header")
    styles = ["quantitative_num" if q else "nominal" for q in is_quantitative]
    labels = view.labels
    for t in range(m):
        sheet.put(5 + t, 1, trace.object_ids[t], "label")
        sheet.row(5 + t, 2, [plain(v) for v in trace.normalized[t]], styles)
        sheet.put(5 + t, n + 2, labels[t], "class_")
    row = 5 + m
    sheet.row(row, 1, ["Type", *(int(q) for q in is_quantitative), "1 = I"], "flag")
    for offset, key in enumerate(("min", "max"), start=1):
        statistic = scaling.statistics.get(key)
        values = [
            plain(statistic[j]) if statistic is not None and q else DASH
            for j, q in enumerate(is_quantitative)
        ]
        sheet.put(row + offset, 1, key, "statistic_label")
        sheet.row(row + offset, 2, [*values, ""], "statistic")
    span = object_span(view)
    sheet.notes(
        row + 4,
        [
            "Type 1 = quantitative feature (set I): rescaled to [0, 1] with the fractional-linear "
            f"(min–max) transform; min and max are taken over the training objects {span}."
            if minmax
            else "Type 1 = quantitative feature (set I): mapped by the scale unification "
            f"“{scaling.normalizer}”, fitted on the training objects {span}.",
            (
                "Type 0 = nominal feature (set J): the original code is kept, because the metric "
                "only "
                "asks whether two codes are equal."
            ),
            (
                "A new object (Brace for Meta-algorithm) is rescaled with the same training min "
                "and "
                "max, so its values may fall outside [0, 1]."
            ),
        ],
    )


# ---------------------------------------------------------------- distances


def _operator_bar(view: RunView, operator: OperatorContext) -> str:
    label = operator.label
    kind = operator_kind(view, operator)
    if kind == "all":
        return (
            f"Operator {label} · Zhuravlyov distance  {label}(Sᵢ, Sⱼ) = ρ_I + ρ_J  —  "
            "row = target object Sᵢ, column = Sⱼ in original order"
        )
    if kind == "quantitative":
        return (
            f"Operator {label} · quantitative part  Σ(j∈I) |x′ⱼ − y′ⱼ|  "
            "(normalized values from Step 1)"
        )
    if kind == "nominal":
        return (
            f"Operator {label} · nominal part  Σ(j∈J) [xⱼ ≠ yⱼ]  "
            "(number of nominal features with different codes)"
        )
    return (
        f"Operator {label} · metric “{operator.metric}” on {operator.features.size} feature(s)  —  "
        "row = target object Sᵢ, column = Sⱼ in original order"
    )


def distances(book: Book) -> None:
    """The distance sheet (Step 2): one m × m block per base operator."""
    view = book.view
    trace = view.trace
    m = trace.m
    operators = trace.operators
    sheet = book.sheet(book.plan.distances, "context")
    sheet.widths({1: 10, (2, m + 1): 11})
    labels = ", ".join(o.label for o in operators)
    default = is_default_operators(view)
    sheet.title_bar(
        "Step 2 · Pairwise distances of the three base operators — ρ (Zhuravlyov), "
        "ρ_I (quantitative part), ρ_J (nominal part)"
        if default
        else f"Step 2 · Pairwise distances of the base operators — {labels}",
        m + 1,
    )
    decimals = view.config.context.distance_decimals
    if m * m * len(operators) > book.options.max_matrix_cells:
        sheet.put(
            3,
            1,
            book.omit(
                f"{book.plan.distances}: the {len(operators)} matrices of {m} × {m} distances "
                "are not written (too large for a sheet) — see the CSV tables distances-*."
            ),
            "note",
        )
        return
    ids = trace.object_ids
    row = 3
    for operator in operators:
        whole = integral(operator.distances)
        off, on = ("integer", "diagonal") if whole else ("num", "diagonal_num")
        sheet.bar(row, 1, m + 1, _operator_bar(view, operator))
        sheet.row(row + 1, 1, [operator.label, *ids], "header")
        sheet.column(row + 2, 1, ids, "label")
        sheet.block(
            row + 2,
            2,
            [[plain(v) for v in line] for line in operator.distances],
            lambda r, c, off=off, on=on: on if r == c else off,  # type: ignore[misc]
        )
        row += m + 3
    count = len(operators)
    blocks = (
        "The three blocks are the three base operators of the ensemble: the same Zhuravlyov "
        "metric on all features, on the quantitative features only and on the nominal features "
        "only."
        if default
        else f"The {count} block(s) are the base operators of the ensemble."
    )
    sheet.notes(
        row,
        [
            (
                f"Each row lists the distances from the target object Sᵢ to {object_span(view)} in "
                "their original order."
            ),
            blocks,
            (
                f"Distances are rounded to {decimals} decimals so that mathematically equal "
                "distances "
                "tie exactly. The diagonal (grey) is never used as a neighbour."
            ),
        ],
    )
