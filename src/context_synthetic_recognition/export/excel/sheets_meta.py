"""The sheets of Steps 10–12.

The meta-dataset, the new object described without its class, and the meta-algorithm — for the
new object step by step and for every training object.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from context_synthetic_recognition.core.meta import REFUSAL, MetaSteps
from context_synthetic_recognition.export.excel.common import (
    ALL_OBJECTS,
    BRACE,
    META,
    META_DATASET,
    Book,
    blocks_that_fit,
)
from context_synthetic_recognition.export.excel.sheets_input import integral
from context_synthetic_recognition.export.excel.writer import SheetWriter, reference
from context_synthetic_recognition.export.tables import plain
from context_synthetic_recognition.export.view import RunView
from context_synthetic_recognition.export.wording import (
    decision_label,
    decision_text,
    set_text,
    tuple_text,
)
from context_synthetic_recognition.notation import DASH, latent_name, subscript, synthetic_name

EXCLUDED_RANK = 999
"""Rank shown (as "—") for the training object left out of the new object's context."""
MAX_COLUMNS = 16_000
"""Tables with one column per object are left out beyond this many columns (Excel's limit)."""
YES, NO = "✅", "❌"


def slot_name(view: RunView, position: int) -> str:
    """The TUPLAM feature at ``position`` (0 = y₀), or "—" if the HAG did not fill it."""
    tuplam = view.hag.tuplam
    return synthetic_name(tuplam[position]) if position < len(tuplam) else DASH


def slot_column(values: Any, position: int, m: int, convert: Any = plain) -> list[Any]:
    """Column ``position`` of an (m × filled) table, or "—" for every object."""
    array = np.asarray(values)
    if position >= array.shape[1]:
        return [DASH] * m
    return [convert(v) for v in array[:, position]]


def _slots(view: RunView, values: Any, count: int, convert: Any = plain) -> list[list[Any]]:
    """An (m × count) grid from an (m × filled) table, unfilled positions as "—"."""
    m = view.trace.m
    columns = [slot_column(values, k, m, convert) for k in range(count)]
    return [[column[t] for column in columns] for t in range(m)]


def _y_headers(view: RunView) -> list[str]:
    return [f"y{subscript(k)} ({slot_name(view, k)})" for k in range(view.slots)]


def _r_headers(view: RunView) -> list[str]:
    return [latent_name(k) for k in range(view.slots - 1)]


# ---------------------------------------------------------------- Dataset for Meta-algorithm


def meta_dataset(book: Book) -> None:
    """Sheet *Dataset for Meta-algorithm* (Step 10): TUPLAM order, y, r and Y = (y, r)."""
    view = book.view
    trace, hag, data = view.trace, view.hag, view.model.meta_dataset
    m, slots = trace.m, view.slots
    latent_slots = slots - 1
    ids, labels = trace.object_ids, view.labels
    sheet = book.sheet(META_DATASET, "meta")
    sheet.widths({1: 32, (2, max(14, 2 * slots + 4)): 13})
    sheet.title_bar(
        "Step 10 · Dataset for the meta-algorithm — initial features y (TUPLAM order) and "
        "additional (latent) features r",
        2 * slots + 4,
    )
    sheet.bar(3, 1, 4, "TUPLAM order (from the HAG sheets)")
    sheet.row(4, 1, ["Position", "Feature index", "Feature", "Selected in"], "header")
    for k in range(slots):
        filled = k < len(hag.tuplam)
        sheet.row(
            5 + k,
            1,
            [
                f"y{subscript(k)}",
                hag.tuplam[k] + 1 if filled else DASH,
                slot_name(view, k),
                "STEP 2 · max ω" if k == 0 else f"Iteration {k}",
            ],
            ["label", "value", "value", "caption"],
        )
    row = slots + 6
    sheet.merge(row, 1, 2, "|TUPLAM| = initial features", "key")
    sheet.put(row, 3, len(hag.tuplam), "value")
    sheet.merge(row + 1, 1, 2, "p = |TUPLAM| − 1 = latent features", "key")
    sheet.put(row + 1, 3, hag.p, "value")
    sheet.merge(row + 2, 1, 2, "TUPLAM", "key")
    sheet.merge(row + 2, 3, 2, hag.label, "value_left")

    top = slots + 10
    right = slots + 4
    y = _slots(view, data.initial, slots, float)
    r = _slots(view, data.latent, latent_slots, float)
    sheet.bar(
        top,
        1,
        slots + 2,
        "Initial features y₀ … y_p  (contribution values ηᵤ(a_tu) of the TUPLAM features)",
    )
    sheet.bar(top, right, latent_slots + 2, "Additional (latent) features r₁ … r_p")
    sheet.row(top + 1, 1, ["№", *_y_headers(view), "Class"], "header")
    sheet.put(top + 1, right, "№", "header")
    sheet.row(top + 1, right + 1, _r_headers(view), "latent_header")
    sheet.put(top + 1, right + latent_slots + 1, "Class", "header")
    for t in range(m):
        line = top + 2 + t
        sheet.put(line, 1, ids[t], "label")
        sheet.row(line, 2, y[t], "initial_num")
        sheet.put(line, slots + 2, labels[t], "class_")
        sheet.put(line, right, ids[t], "label")
        sheet.row(line, right + 1, r[t], "latent_num")
        sheet.put(line, right + latent_slots + 1, labels[t], "class_")

    top += m + 3
    sheet.bar(
        top,
        1,
        2 * slots + 1,
        f"Dataset for Meta-algorithms  Y = (y₀, …, y_p, r₁, …, r_p)  [{view.dataset.name}]",
    )
    sheet.put(top + 1, 1, "№", "header")
    sheet.row(top + 1, 2, _y_headers(view), "header")
    sheet.row(top + 1, slots + 2, _r_headers(view), "latent_header")
    sheet.put(top + 1, 2 * slots + 1, "Class", "header")
    for t in range(m):
        line = top + 2 + t
        sheet.put(line, 1, ids[t], "label")
        sheet.row(line, 2, y[t], "initial_num")
        sheet.row(line, slots + 2, r[t], "latent_num")
        sheet.put(line, 2 * slots + 1, labels[t], "class_")
    sheet.notes(
        top + m + 3,
        [
            (
                "y_j is the contribution value of the j-th TUPLAM feature (Step 8); r_j is the "
                "value "
                "R(Sₜ) after the j-th HAG iteration (Step 9)."
            ),
            '"—" marks positions that the HAG did not fill (the grouping stopped earlier).',
        ],
    )


# ---------------------------------------------------------------- Brace for Meta-algorithm


def sort_keys(view: RunView) -> list[int]:
    """The workbook's key class·100 + № that sorts the training objects by class (K1 first)."""
    m = view.trace.m
    factor = max(100, 10 ** len(str(m)))
    return [(int(c) + 1) * factor + t + 1 for t, c in enumerate(view.trace.class_index)]


def _description_headers(view: RunView) -> tuple[list[str], list[str]]:
    slots = view.slots
    gradations = [f"aᵢ{subscript(k)} ({slot_name(view, k)})" for k in range(slots)]
    latent = [f"dᵢ{subscript(k + 1)} ({latent_name(k)})" for k in range(slots - 1)]
    return gradations, latent


def brace(book: Book) -> None:
    """Sheet *Brace for Meta-algorithm* (Step 11): the training description and the new object."""
    view = book.view
    trace, model, demo = view.trace, view.model, view.new_object
    m, n, r, slots = trace.m, len(trace.feature_names), trace.r, view.slots
    latent_slots = slots - 1
    operators = trace.operators
    ids, labels = trace.object_ids, view.labels
    representation = demo.classification.representation
    assert representation is not None
    context = representation.context
    sheet = book.sheet(BRACE, "meta")
    table_width = 2 * slots + 1
    wide = m + 1 <= MAX_COLUMNS
    last = max(n + 1, table_width + 2, r + 1, m + 1 if wide else 0, 14)
    sheet.widths({1: 26, (2, last): 11, (last + 1, last + 2): 9})
    sheet.title_bar(
        "Step 11 · Brace for the meta-algorithm — training description (aᵢ, dᵢ) and a new object "
        "encoded without its class",
        max(n + 1, table_width + 2),
    )

    # ---- the training description and the same table sorted by class
    gradation_headers, latent_headers = _description_headers(view)
    a = _slots(view, model.description.gradations, slots, int)
    d = _slots(view, model.description.latent, latent_slots, float)
    keys = sort_keys(view)
    positions = [sum(other < key for other in keys) + 1 for key in keys]
    sheet.bar(
        3,
        1,
        table_width,
        "Training dataset for the meta-algorithm — gradations aᵢⱼ ∈ {1, 2} of the TUPLAM features "
        "and latent features dᵢⱼ = r_j",
    )

    def header(row: int, extra: bool) -> None:
        sheet.put(row, 1, "№", "header")
        sheet.row(row, 2, gradation_headers, "header")
        sheet.row(row, slots + 2, latent_headers, "latent_header")
        sheet.put(row, table_width, "Class", "header")
        if extra:
            sheet.row(row, table_width + 1, ["sort key", "position"], "header")

    def line(row: int, t: int) -> None:
        sheet.put(row, 1, ids[t], "label")
        sheet.row(row, 2, a[t], "synthetic")
        sheet.row(row, slots + 2, d[t], "latent_num")
        sheet.put(row, table_width, labels[t], "class_")

    header(4, True)
    for t in range(m):
        line(5 + t, t)
        sheet.row(5 + t, table_width + 1, [keys[t], positions[t]], "muted")
    top = m + 6
    sheet.bar(top, 1, table_width, "Sorted training dataset based on classes (K1 first)")
    header(top + 1, False)
    for offset, t in enumerate(np.argsort(keys, kind="stable")):
        line(top + 2 + offset, int(t))

    # ---- the new object: values, type flags, unified values
    top = 2 * m + 9
    quantitative = trace.scaling.quantitative
    sheet.bar(
        top,
        1,
        n + 1,
        f"New object S — its {n} feature values (blue); its context is computed relative to the "
        "training sample, without its class",
    )
    sheet.put(top + 1, 1, "Feature", "header")
    sheet.row(top + 1, 2, trace.feature_names, "header")
    sheet.put(top + 2, 1, "x  (input)", "caption")
    sheet.row(top + 2, 2, [plain(v) for v in demo.values], "input")
    sheet.put(top + 3, 1, "Type", "caption")
    sheet.row(top + 3, 2, [int(q) for q in quantitative], "flag")
    sheet.put(top + 4, 1, "x′  (unified)", "caption")
    sheet.row(
        top + 4,
        2,
        [plain(v) for v in context.normalized[0]],
        ["quantitative_num" if q else "nominal" for q in quantitative],
    )
    sheet.put(top + 6, 1, "Exclude training object №", "key")
    sheet.put(top + 6, 2, 0 if demo.exclude is None else demo.exclude + 1, "input")
    if demo.exclude is None:
        remark = "A genuinely new object: no training object is excluded from its context (0)."
    else:
        name = ids[demo.exclude]
        remark = (
            f"The object copies {name}, so {name} is excluded from its own context "
            f"(leave-self-out): the result must reproduce the {name} row of "
            '"Meta-algorithm (All Objects)". 0 = a genuinely new object.'
        )
    sheet.merge(top + 6, 3, max(n - 1, 1), remark, "text")

    # ---- its distances and ranks
    count = len(operators)
    top += 8
    if wide:
        sheet.bar(
            top,
            1,
            m + 1,
            "Distances from S to the training objects Sᵢ (same operators as Step 2)",
        )
        sheet.row(top + 1, 1, ["Operator", *ids], "header")
        for o, operator in enumerate(operators):
            distances = context.distances[o][0]
            sheet.put(top + 2 + o, 1, operator.label, "caption")
            sheet.row(
                top + 2 + o,
                2,
                [plain(v) for v in distances],
                "integer" if integral(operator.distances) and integral(distances) else "num",
            )
        top += count + 3
        sheet.bar(
            top,
            1,
            m + 1,
            "Rank of Sᵢ among the neighbours of S  (ties → smaller index;  excluded object = —)",
        )
        sheet.row(top + 1, 1, ["Operator", *ids], "header")
        for o, operator in enumerate(operators):
            order = context.orders[o][0]
            rank = np.full(m, EXCLUDED_RANK, dtype=np.int64)
            rank[order] = np.arange(1, order.size + 1)
            sheet.put(top + 2 + o, 1, operator.label, "caption")
            sheet.row(top + 2 + o, 2, [int(v) for v in rank], "rank")
        sheet.put(top + 2 + count, 1, "Class of Sᵢ", "caption")
        sheet.row(top + 2 + count, 2, labels, "class_")
        sheet.put(top + 3 + count, 1, "Index i", "caption")
        sheet.row(top + 3 + count, 2, range(1, m + 1), "muted")
        top += count + 5
    else:
        sheet.put(
            top,
            1,
            book.omit(
                f"{BRACE}: the distances and ranks of S to the {m} training objects are not "
                "written (more columns than a sheet has)."
            ),
            "note",
        )
        top += 2

    # ---- its synthetic features and its description on TUPLAM
    sheet.bar(top, 1, r + 1, "Synthetic features of S by formula (5) — Ψ(r) of the new object")
    sheet.put(top + 1, 1, "Quantity", "header_left")
    sheet.row(top + 1, 2, [f.name for f in trace.features], "header")
    ks = [f.k for f in trace.features]
    chi1 = [int(v) for v in context.chi1[0]]
    lines: list[tuple[str, list[int], str]] = [
        ("k", ks, "header_k"),
        ("χ₁ (K1 among the k nearest)", chi1, "integer"),
        ("χ₂ = k − χ₁", [k - c for k, c in zip(ks, chi1, strict=True)], "integer"),
        ("aᵤ(S) by (5)", [int(v) for v in context.values[0]], "synthetic"),
    ]
    for offset, (caption, values, style) in enumerate(lines, start=2):
        sheet.put(top + offset, 1, caption, "caption")
        sheet.row(top + offset, 2, values, style)
    top += 7
    sheet.bar(
        top,
        1,
        slots + 1,
        "Description of S on TUPLAM:  (a₀, …, a_p)  — the input of the meta-algorithm",
    )
    sheet.put(top + 1, 1, "Position", "header_left")
    sheet.row(top + 1, 2, [f"a{subscript(k)}" for k in range(slots)], "header")
    sheet.put(top + 2, 1, "Feature", "caption")
    sheet.row(top + 2, 2, [slot_name(view, k) for k in range(slots)], "value")
    sheet.put(top + 3, 1, "Value", "caption")
    sheet.row(top + 3, 2, description_cells(view), "synthetic")
    sheet.notes(
        top + 5,
        [
            (
                "Scale unification uses the training min and max (Step 1); distances, ranks and χ₁ "
                "use exactly the rules of Steps 2–5. The class of S is not an input anywhere on "
                "this "
                "sheet."
            ),
            (
                "Formula (5) needs only the classes of the neighbours, so the description "
                "(a₀, …, a_p) exists for any new object — the procedural correctness stated by the "
                "Theorem of the article."
            ),
        ],
    )


def description_cells(view: RunView) -> list[Any]:
    """The new object's (a₀, …, a_p), padded with "—" to the laid-out positions."""
    representation = view.new_object.classification.representation
    assert representation is not None
    values = representation.description[0]
    return [int(values[k]) if k < values.size else DASH for k in range(view.slots)]


# ---------------------------------------------------------------- Meta-algorithm (new object)


def step_flags(steps: MetaSteps, j: int) -> tuple[Any, Any]:
    """Membership of B1 and B2 after step ``j`` (a step beyond p keeps the sets of step p)."""
    last = min(j, steps.in_b1.shape[0] - 1)
    return steps.in_b1[last], steps.in_b2[last]


def _sets_row(sheet: SheetWriter, view: RunView, row: int, steps: MetaSteps, j: int) -> None:
    ids = view.trace.object_ids
    b1, b2 = step_flags(steps, j)
    index = subscript(j)
    sheet.put(row, 1, f"B1(a{index}) =", "key")
    sheet.merge(row, 2, 3, set_text([ids[int(i)] for i in np.flatnonzero(b1)]), "value_left")
    sheet.put(row, 5, f"B2(a{index}) =", "key")
    sheet.merge(row, 6, 5, set_text([ids[int(i)] for i in np.flatnonzero(b2)]), "value_left")


def meta(book: Book) -> None:
    """Sheet *Meta-algorithm* (Step 12): Steps 1–5 for the new object, B1 and B2 at every step."""
    view = book.view
    trace, model, demo = view.trace, view.model, view.new_object
    m, p, slots = trace.m, model.p, view.slots
    ids, labels = trace.object_ids, view.labels
    classification = demo.classification
    steps = classification.meta.steps(0)
    a = model.description.gradations
    d = model.description.latent
    in_k1 = trace.class_index == 0
    sheet = book.sheet(META, "meta")
    sheet.widths({1: 18, (2, 10): 14, (11, 23): 8, (24, 25): 4})
    sheet.hide_columns(24, 25)
    sheet.title_bar(
        "Step 12 · Meta-algorithm — classification of the new object S (Steps 1–5 of the article)",
        10,
    )
    sheet.bar(3, 1, slots + 3, 'Object to classify: (a₀, …, a_p) from "Brace for Meta-algorithm"')
    sheet.put(4, 1, "Position", "header_left")
    sheet.row(4, 2, [*(f"a{subscript(k)}" for k in range(slots)), "p", "Class"], "header")
    sheet.put(5, 1, "Feature", "caption")
    sheet.row(5, 2, [slot_name(view, k) for k in range(slots)], "value")
    sheet.put(6, 1, "Value", "caption")
    sheet.row(6, 2, description_cells(view), "synthetic")
    sheet.row(6, slots + 2, [p, "*"], "value")

    sheet.bar(
        8,
        1,
        10,
        "Step 1 · j = 0:   B1(a₀) = {Sᵢ ∈ K1 | a₀ = aᵢ₀},   B2(a₀) = {Sᵢ ∈ K2 | a₀ = aᵢ₀}",
    )
    sheet.row(9, 1, ["№", "Class", "aᵢ₀", "a₀ = aᵢ₀ ?", "in B1(a₀)", "in B2(a₀)"], "header")
    first_styles = ["label", "class_", "synthetic", "integer", "integer", "integer"]
    for t in range(m):
        sheet.row(
            10 + t,
            1,
            [
                ids[t],
                labels[t],
                int(a[t, 0]),
                YES if steps.match[0, t] else NO,
                int(steps.in_b1[0, t]),
                int(steps.in_b2[0, t]),
            ],
            first_styles,
        )
    sheet.highlight_equal(reference(10, 5, m, 2), "1")
    _sets_row(sheet, view, 10 + m, steps, 0)

    step_styles = [
        "label",
        "class_",
        "integer",
        "synthetic",
        "integer",
        "latent_num",
        "integer",
        "integer",
        "integer",
    ]
    for j in range(1, slots):
        top = m + 13 + (m + 5) * (j - 1)
        executed = j <= p
        cur, prev = subscript(j), subscript(j - 1)
        sheet.bar(
            top,
            1,
            10,
            f"Step 2 · j = {j}:   B1(a{cur}) = {{Sᵢ ∈ B1(a{prev}) | a{cur} = aᵢ{cur}, "
            f"dᵢ{cur} > 0}},   B2(a{cur}) = {{Sᵢ ∈ B2(a{prev}) | a{cur} = aᵢ{cur}, dᵢ{cur} < 0}}",
        )
        sheet.row(
            top + 1,
            1,
            [
                "№",
                "Class",
                "in B before",
                f"aᵢ{cur}",
                f"a{cur} = aᵢ{cur} ?",
                f"dᵢ{cur}",
                "latent sign",
                "keep?",
                "in B after",
            ],
            "header",
        )
        sheet.put(top + 1, 10, int(executed), "executed")
        b1_before, b2_before = step_flags(steps, j - 1)
        b1_after, b2_after = step_flags(steps, j)
        for t in range(m):
            before = bool(b1_before[t] if in_k1[t] else b2_before[t])
            after = bool(b1_after[t] if in_k1[t] else b2_after[t])
            value: Any = DASH
            latent: Any = DASH
            match: Any = DASH
            sign: Any = DASH
            keep: Any = YES if before else DASH
            if executed:
                value, latent = int(a[t, j]), float(d[t, j - 1])
                matched, ok = bool(steps.match[j, t]), bool(steps.sign[j - 1, t])
                match = YES if matched else NO
                good, bad = ("✅ (+)", "❌ (−)") if in_k1[t] else ("✅ (−)", "❌ (+)")
                sign = good if ok else bad
                keep = (YES if matched and ok else NO) if before else DASH
            sheet.row(
                top + 2 + t,
                1,
                [ids[t], labels[t], int(before), value, match, latent, sign, keep, int(after)],
                step_styles,
            )
            sheet.row(top + 2 + t, 24, [int(b1_after[t]), int(b2_after[t])], "muted")
        sheet.highlight_equal(reference(top + 2, 9, m, 1), "1")
        _sets_row(sheet, view, top + m + 2, steps, j)
        sheet.put(top + m + 3, 1, "Step 3 · j < p ?", "key")
        if j < p:
            verdict = "Go back to Step 2"
        else:
            verdict = "Go to Step 4" if j == p else "— (j > p: not executed)"
        sheet.merge(top + m + 3, 2, 4, verdict, "decision")

    top = m + 13 + (m + 5) * (slots - 1)
    result = classification.meta
    decision = demo.decision
    sheet.bar(
        top,
        1,
        10,
        "Step 4 · class:  K1 if |B1(a_p)|/|K1| > |B2(a_p)|/|K2|,  K2 if <,  0 (refusal) if equal",
    )
    final: list[tuple[str, Any, str]] = [
        ("|B1(a_p)|", int(result.b1_sizes[0, -1]), "value"),
        ("|B2(a_p)|", int(result.b2_sizes[0, -1]), "value"),
        ("score₁ = |B1|/|K1|", float(result.scores1[0]), "value_num"),
        ("score₂ = |B2|/|K2|", float(result.scores2[0]), "value_num"),
        ("Class of S", decision_label(decision, trace.classes), "good"),
    ]
    for offset, (key, value, style) in enumerate(final, start=1):
        sheet.put(top + offset, 1, key, "key")
        sheet.put(top + offset, 2, value, style)
    sheet.merge(top + 5, 3, 3, decision_text(decision, trace.classes), "decision")
    sheet.bar(top + 7, 1, 10, "Step 5 · The End")


# ---------------------------------------------------------------- Meta-algorithm (All Objects)


def all_objects(book: Book) -> None:
    """Sheet *Meta-algorithm (All Objects)*: the resubstitution, with B1/B2 flags per step."""
    view = book.view
    trace, model = view.trace, view.model
    m, slots = trace.m, view.slots
    ids, labels, classes = trace.object_ids, view.labels, trace.classes
    result = view.training.meta
    decimals = view.config.evaluation.score_decimals
    a = _slots(view, model.description.gradations, slots, int)
    wide = 2 * m + 3 <= MAX_COLUMNS
    blocks = blocks_that_fit(m, slots * (2 * m + 3), book.options.max_sheet_cells) if wide else 0
    width = 2 * m + 3 if blocks else 10
    sheet = book.sheet(ALL_OBJECTS, "meta")
    sheet.widths({1: 10, 2: 20, (3, max(m + 1, 11)): 9})
    if blocks:
        sheet.widths({m + 2: 7, (m + 3, 2 * m + 2): 6, 2 * m + 3: 7})
    sheet.title_bar(
        "Step 12 · Meta-algorithm for every training object (resubstitution) — training "
        "correctness, Definition 2",
        width,
    )
    sheet.bar(3, 1, 10, "Summary")
    sheet.row(
        4,
        1,
        [
            "№",
            "(a₀, …, a_p)",
            "|B1(a_p)|",
            "|B2(a_p)|",
            "score₁",
            "score₂",
            "score₁ − score₂",
            "Predicted",
            "True",
            "Correct",
        ],
        "header",
    )
    styles = ["label", "value", "value", "value", "num", "num", "num", "good", "class_", "integer"]
    for t in range(m):
        decision = int(result.decisions[t])
        predicted = decision_label(decision, classes)
        s1, s2 = float(result.scores1[t]), float(result.scores2[t])
        sheet.row(
            5 + t,
            1,
            [
                ids[t],
                tuple_text(a[t]),
                int(result.b1_sizes[t, -1]),
                int(result.b2_sizes[t, -1]),
                s1,
                s2,
                round(s1 - s2, decimals),
                predicted,
                labels[t],
                YES if decision != REFUSAL and predicted == labels[t] else NO,
            ],
            styles,
        )
    top = m + 7
    for t in range(blocks):
        steps = result.steps(t)
        sheet.bar(
            top,
            1,
            width,
            f"Query {ids[t]}  —  its own description (aₜ₀, …, aₜₚ) against the training dataset "
            "(B1 flags left, B2 flags right)",
            "bar_left",
        )
        sheet.row(top + 1, 1, ["Step j", *ids, "|B1|", *ids, "|B2|"], "header")
        for j in range(slots):
            b1, b2 = step_flags(steps, j)
            line = top + 2 + j
            sheet.put(line, 1, j, "label")
            sheet.row(line, 2, [int(v) for v in b1], "integer")
            sheet.put(line, m + 2, int(np.count_nonzero(b1)), "value")
            sheet.row(line, m + 3, [int(v) for v in b2], "integer")
            sheet.put(line, 2 * m + 3, int(np.count_nonzero(b2)), "value")
        sheet.highlight_equal(reference(top + 2, 2, slots, m), "1")
        sheet.highlight_equal(reference(top + 2, m + 3, slots, m), "1")
        top += slots + 3
    notes = [
        (
            "Every training object is classified with its own description (its context excludes "
            "itself, as in Steps 3–5). A predicted class 0 is a refusal (equal scores)."
        ),
        (
            "Because the training objects themselves belong to B1/B2, this measures training "
            "correctness (Definition 2), not generalization — see the Leave-One-Out sheet for that."
        ),
    ]
    if blocks < m:
        notes.append(
            book.omit(
                f"{ALL_OBJECTS}: large sample — the B1/B2 flags per step are written for "
                f"{blocks} of {m} objects; the summary above covers every object."
            )
        )
    sheet.notes(top, notes)
