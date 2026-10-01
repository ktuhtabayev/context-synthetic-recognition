"""The sheets of Steps 3–8.

Sorted neighbours, same-class counts μ, Ψ(r), membership and stability, informativeness ω,
contributions η.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from context_synthetic_recognition.core.contributions import weight_ranks
from context_synthetic_recognition.core.trace import OperatorContext, SyntheticFeature
from context_synthetic_recognition.export.excel.common import (
    CONTRIBUTION,
    MEMBERSHIP,
    MU,
    OMEGA,
    PSI,
    Book,
    blocks_that_fit,
)
from context_synthetic_recognition.export.excel.sheets_input import object_span
from context_synthetic_recognition.export.excel.writer import SheetWriter, reference
from context_synthetic_recognition.export.tables import neighbour_depth, plain
from context_synthetic_recognition.export.view import RunView
from context_synthetic_recognition.notation import DASH

_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
MEMBERSHIP_COLUMNS = (
    "μ",
    "d₁ₖ(μ)",
    "d₂ₖ(μ)",
    "n(μ)",
    "d₁ₖ/|K1|",
    "d₂ₖ/|K2|",
    "f_k(μ)",
    "n·max(f, 1−f)",
)


def count_word(count: int) -> str:
    """``six`` for 6 — small counts in running text are written out, as in the workbook."""
    return _WORDS[count] if 0 <= count < len(_WORDS) else str(count)


def _features_of(view: RunView, operator: int) -> list[SyntheticFeature]:
    return [f for f in view.trace.features if f.operator == operator]


def _feature_names(view: RunView) -> list[str]:
    return [f.name for f in view.trace.features]


def _object_rows(
    book: Book, sheet: SheetWriter, top: int, matrix: Any, style: Any, *, convert: Any = plain
) -> None:
    """Rows ``label, values…, class`` for every object, from row ``top``."""
    view = book.view
    ids, labels = view.trace.object_ids, view.labels
    values = np.asarray(matrix)
    last = 2 + values.shape[1]
    for t in range(values.shape[0]):
        sheet.put(top + t, 1, ids[t], "label")
        sheet.row(top + t, 2, [convert(v) for v in values[t]], style)
        sheet.put(top + t, last, labels[t], "class_")


# ---------------------------------------------------------------- Sorted Neighbors


def sorted_neighbours(book: Book, index: int, operator: OperatorContext) -> None:
    """Sheet *Sorted Neighbors (operator)* (Step 3): the rank matrix and one block per object."""
    view = book.view
    trace = view.trace
    m, label = trace.m, operator.label
    ids, labels, y = trace.object_ids, view.labels, trace.class_index
    ks = set(trace.permitted_k.ks)
    sheet = book.sheet(book.plan.neighbours[index], "context")
    full = (m - 1) * 8 * m <= book.options.max_sheet_cells
    depth = m - 1 if full else neighbour_depth(view)
    blocks = blocks_that_fit(m, 8 * depth, book.options.max_sheet_cells)
    sheet.widths({1: 30, (2, max(m, depth) + 1): 11})
    sheet.title_bar(
        f"Step 3 · Operator {label} — neighbours of every object sorted by ascending {label}, "
        "original index and class kept",
        m + 1 if m * m <= book.options.max_matrix_cells else depth + 1,
    )
    row = 3
    if m * m <= book.options.max_matrix_cells:
        sheet.bar(
            row,
            1,
            m + 1,
            f"Rank of Sⱼ in the neighbour list of Sᵢ under {label}  "
            "(self excluded; equal distances → smaller original index first)",
        )
        sheet.row(row + 1, 1, ["Sᵢ \\ Sⱼ", *ids], "header")
        sheet.column(row + 2, 1, ids, "label")
        ranks = operator.ranks
        sheet.block(
            row + 2,
            2,
            [[DASH if v == 0 else int(v) for v in line] for line in ranks],
            lambda r, c: "diagonal" if r == c else "integer",
        )
        row += m + 3
    else:
        sheet.put(
            row,
            1,
            book.omit(
                f"{sheet.title}: the {m} × {m} rank matrix is not written (too large for a "
                "sheet) — see the CSV tables neighbours-*."
            ),
            "note",
        )
        row += 2

    order = operator.order[:, :depth]
    distance = operator.sorted_distances[:, :depth]
    mu = operator.same_class_running(y)[:, :depth]
    chi1 = operator.k1_running(y)[:, :depth]
    marks = [f"k = {rank}" if rank in ks else None for rank in range(1, depth + 1)]
    for i in range(blocks):
        neighbours = order[i]
        target = 5 + i
        sheet.bar(
            row,
            1,
            depth + 1,
            f"{ids[i]}  ·  class {labels[i]}  —  neighbours in ascending {label}  "
            f"(target row {target} of the rank matrix)",
            "bar_left",
        )
        sheet.row(row + 1, 1, ["Rank", *range(1, depth + 1)], "header")
        lines: list[tuple[str, list[Any], str]] = [
            ("k-neighbourhood", marks, "small"),
            ("Neighbour", [ids[int(j)] for j in neighbours], "label"),
            ("Original index", [int(j) + 1 for j in neighbours], "muted"),
            (f"{label}({ids[i]}, neighbour)", [plain(v) for v in distance[i]], "num"),
            ("Class", [labels[int(j)] for j in neighbours], "class_"),
            (f"Same class as {ids[i]}", [int(y[int(j)] == y[i]) for j in neighbours], "integer"),
            ("Same-class count μ (1 … rank)", [int(v) for v in mu[i]], "integer"),
            ("K1 count χ₁ (1 … rank)", [int(v) for v in chi1[i]], "integer"),
        ]
        for offset, (caption, values, style) in enumerate(lines, start=2):
            sheet.put(row + offset, 1, caption, "caption")
            sheet.row(row + offset, 2, values, style)
        sheet.highlight_equal(reference(row + 7, 2, 1, depth), "1")
        marked = f'B${row + 2}<>""'
        for offset in (2, 8, 9):
            sheet.highlight_formula(reference(row + offset, 2, 1, depth), marked, "neighbourhood")
        row += 11
    notes = [
        (
            "Ranks use the distances of Step 2: rank = 1 + #{objects closer than Sⱼ} + #{objects "
            "at "
            "the same distance with a smaller original index}; the target itself is excluded."
        ),
        (
            "Yellow cells mark the nested neighbourhoods of the permitted k (odd, k_min … k_max "
            "from "
            "the Parameters sheet)."
        ),
        (
            "μ (same-class count) needs the class of the target — it is used only on the training "
            "side (formulas (1)–(4))."
        ),
        (
            "χ₁ (number of K1 objects among the first k) does not use the class of the target — it "
            "gives the synthetic feature (5) and is computed the same way for a new object."
        ),
    ]
    if not full:
        notes.append(
            book.omit(
                f"{sheet.title}: large sample — the blocks list ranks 1 … {depth} "
                f"(the largest permitted k and one beyond) for {blocks} of {m} objects; "
                "the CSV tables neighbours-* hold every object."
            )
        )
    sheet.notes(row, notes)


# ---------------------------------------------------------------- Synthetic Features (k-NN)


def same_class_counts(book: Book) -> None:
    """Sheet *Synthetic Features (k-NN)* (Step 4): μ per operator and permitted k."""
    view = book.view
    trace = view.trace
    m, ks = trace.m, trace.permitted_k
    per_operator = [_features_of(view, o) for o in range(len(trace.operators))]
    width = max(1, *(len(features) for features in per_operator))
    side = width + 4
    sheet = book.sheet(MU, "synthetic")
    sheet.widths({1: 10, (2, width + 1): 13, width + 2: 11, width + 3: 4, side: 30, side + 1: 52})
    sheet.title_bar(
        "Step 4 · Synthetic Features based on k-NN — same-class neighbours μ of Sᵢ for "
        "k = k_min … k_max, per operator",
        side + 1,
    )
    labels, ids = view.labels, trace.object_ids
    for o, operator in enumerate(trace.operators):
        top = 3 + (m + 4) * o
        features = per_operator[o]
        sheet.bar(top, 1, width + 2, f"Operator {operator.label}  ·  same-class count μ")
        sheet.put(top + 1, 1, "№", "header")
        sheet.row(top + 1, 2, [f.k for f in features], "header_k")
        sheet.put(top + 1, width + 2, "Class", "header")
        for t in range(m):
            sheet.put(top + 2 + t, 1, ids[t], "label")
            sheet.row(top + 2 + t, 2, [int(f.mu[t]) for f in features], "synthetic")
            sheet.put(top + 2 + t, width + 2, labels[t], "class_")
    formula = ks.rule == "formula"
    sheet.bar(3, side, 2, "k range (Parameters sheet)")
    rows: list[tuple[str, Any, str]] = [
        ("|K1|", trace.class_sizes[0], "value"),
        ("|K2|", trace.class_sizes[1], "value"),
        ("min |Kᵢ|", min(trace.class_sizes), "value"),
        ("k_min (fixed)" if formula else "k_min", DASH if ks.k_min is None else ks.k_min, "value"),
        ("k_max = 2 · min |Kᵢ| − 3", DASH if ks.k_max is None else ks.k_max, "value"),
        ("Permitted k", ks.label, "value"),
        ("Layout check", "✓ the workbook has a column for every permitted k", "value_left"),
    ]
    for offset, (key, value, style) in enumerate(rows, start=4):
        sheet.put(offset, side, key, "key")
        sheet.put(offset, side + 1, value, style)
    sheet.notes(
        3 + (m + 4) * len(trace.operators),
        [
            (
                "Cell value μ = number of the k nearest neighbours (Step 3 order of the operator) "
                "whose class equals the class of Sᵢ."
            ),
            "Only the k permitted by k_max = 2 · min |Kᵢ| − 3 are used: odd k from k_min = 3 "
            "(fixed) up to k_max; |K1| and |K2| are counted from the Dataset, so k_max follows "
            "the data."
            if formula
            else f"The k are chosen by the strategy “{ks.rule}” (Parameters sheet).",
            (
                "μ is the training-side gradation of formulas (1)–(4); the class-free synthetic "
                "feature a ∈ {1, 2} of formula (5) is built in the next sheet."
            ),
        ],
    )


# ---------------------------------------------------------------- Ψ(r) Binary Features


def psi(book: Book) -> None:
    """Sheet *Ψ(r) Binary Features* (Step 5): the feature map, χ₁, χ₂ and Ψ(r) by formula (5)."""
    view = book.view
    trace = view.trace
    m, r = trace.m, trace.r
    features = trace.features
    width = max(r + 2, 8)
    sheet = book.sheet(PSI, "synthetic")
    sheet.widths({1: 10, (2, width): 12, 4: 24})
    sheet.title_bar(
        "Step 5 · Ψ(r) — synthetic features aᵤ ∈ {1, 2} by formula (5): the majority class of the "
        "k nearest neighbours",
        width,
    )
    sheet.bar(
        3,
        1,
        8,
        "Feature map · one synthetic feature per (base operator, permitted k)   ·   "
        f"r = |Ψ(r)| = {len(trace.operators)} × (number of permitted k)",
    )
    sheet.row(
        4,
        1,
        ["Feature", "Operator", "k", "Neighbours sheet"],
        ["header", "header", "header", "header_left"],
    )
    sheet.merge(4, 5, 4, "Rule (5)", "header_left")
    rule = "aᵤ = 1 if χ₁(Sⱼ, k) > [k/2],  aᵤ = 2 if χ₂(Sⱼ, k) > [k/2]"
    for u, feature in enumerate(features):
        row = 5 + u
        sheet.row(
            row,
            1,
            [
                feature.name,
                feature.operator_label,
                feature.k,
                book.plan.neighbours[feature.operator],
            ],
            ["label", "value", "header_k", "caption"],
        )
        sheet.merge(row, 5, 4, rule, "text")
    names = _feature_names(view)
    chi1 = np.column_stack([f.chi1 for f in features])
    chi2 = np.column_stack([f.chi2 for f in features])
    tables = [
        (
            (
                "χ₁(Sⱼ, k) — number of K1 objects among the k nearest neighbours "
                "(the class of Sⱼ itself is not used)"
            ),
            chi1,
            "integer",
        ),
        (
            "χ₂(Sⱼ, k) = k − χ₁(Sⱼ, k) — number of K2 objects among the k nearest neighbours",
            chi2,
            "integer",
        ),
        ("Ψ(r) · synthetic features aᵤ(Sⱼ) ∈ {1, 2} by formula (5)", trace.values, "synthetic"),
    ]
    top = r + 6
    for title, matrix, style in tables:
        sheet.bar(top, 1, r + 2, title)
        sheet.row(top + 1, 1, ["№", *names, "Class"], "header")
        _object_rows(book, sheet, top + 2, matrix, style)
        top += m + 3
    sheet.notes(
        top,
        [
            (
                "Formula (5): the synthetic feature takes the value 1 when class K1 has the "
                "majority "
                "among the k nearest neighbours and 2 when K2 has it (k is odd, so one of the two "
                "always holds)."
            ),
            (
                "χ₁ and χ₂ only use the classes of the neighbours, never the class of Sⱼ itself — "
                "so "
                "the same feature can be computed for a new object whose class is unknown (Theorem "
                "of "
                "the article)."
            ),
            (
                f"The {count_word(r)} features are the input set Ψ(r) of the hierarchical "
                "agglomerative grouping (Steps 8–9)."
            ),
        ],
    )


# ---------------------------------------------------------------- Membership & Stability


def _membership_table(
    sheet: SheetWriter, top: int, left: int, feature: SyntheticFeature, rows: int
) -> None:
    table = feature.membership
    sheet.bar(
        top,
        left,
        8,
        f"{feature.name}  ·  operator {feature.operator_label}  ·  k = {feature.k}  ·  "
        "gradation μ = same-class count",
    )
    sheet.row(top + 1, left, MEMBERSHIP_COLUMNS, "header")
    styles = ["label", "integer", "integer", "integer", "num", "num", "synthetic_num", "num"]
    size = int(table.gradations.size)
    last = int(table.gradations[-1]) if size else -1
    for g in range(rows):
        if g < size:
            values: list[Any] = [
                int(table.gradations[g]),
                int(table.d1[g]),
                int(table.d2[g]),
                int(table.n[g]),
                float(table.share1[g]),
                float(table.share2[g]),
                float(table.f[g]),  # NaN is written as "—"
                float(table.weighted[g]),
            ]
        else:
            values = [last + 1 + g - size, DASH, DASH, DASH, DASH, DASH, DASH, 0]
        sheet.row(top + 2 + g, left, values, styles)
    foot = top + 2 + rows
    sheet.merge(
        foot,
        left,
        7,
        f"stability g_k by (2),  β = k = {feature.k},  g = Σ / m",
        "key",
    )
    sheet.put(foot, left + 7, feature.stability, "value_num")


def membership(book: Book) -> None:
    """Sheet *Membership & Stability* (Step 6): formulas (1) and (2), the meta-object, Task 2."""
    view = book.view
    trace = view.trace
    m, r = trace.m, trace.r
    operators = trace.operators
    groups = [_features_of(view, o) for o in range(len(operators))]
    cells = sum(8 * (f.membership.gradations.size + 3) for f in trace.features)
    reduced = cells > book.options.max_sheet_cells
    if reduced:
        groups = [[trace.features[u] for u in view.hag.tuplam]]
    shown = [f for group in groups for f in group]
    rows = max(f.membership.gradations.size for f in shown)
    across = max(len(group) for group in groups)
    sheet = book.sheet(MEMBERSHIP, "synthetic")
    sheet.widths({1: 10, (2, max(22, 9 * across, r + 2)): 11})
    sheet.title_bar(
        "Step 6 · Membership function f_k(μ) — formula (1) — and stability g_k — formula (2), "
        "per synthetic feature",
        max(9 * across - 1, 8),
    )
    top = 3
    for group in groups:
        for position, feature in enumerate(group):
            _membership_table(sheet, top, 1 + 9 * position, feature, rows)
        top += rows + 5
    if reduced:
        sheet.put(
            top - 1,
            1,
            book.omit(
                f"{MEMBERSHIP}: large experiment — the tables above are those of the TUPLAM "
                f"features only ({len(shown)} of {r}); the CSV table membership holds all."
            ),
            "note",
        )

    sheet.bar(
        top,
        1,
        r + 1,
        "Meta-object · the stability vector (g) of the training sample (section 1.4)",
    )
    sheet.row(top + 1, 1, ["Feature", *(f.name for f in trace.features)], "header")
    sheet.put(top + 2, 1, "Operator", "caption")
    sheet.row(top + 2, 2, [f.operator_label for f in trace.features], "value")
    sheet.put(top + 3, 1, "k", "caption")
    sheet.row(top + 3, 2, [f.k for f in trace.features], "header_k")
    sheet.put(top + 4, 1, "g_k", "caption")
    sheet.row(top + 4, 2, [f.stability for f in trace.features], "synthetic_num")
    top += 6

    masks = trace.bit_masks
    notes = [
        (
            "Formula (1): f_k(μ) = (d₁ₖ(μ)/|K1|) / (d₁ₖ(μ)/|K1| + d₂ₖ(μ)/|K2|), where d₁ₖ(μ), "
            "d₂ₖ(μ) "
            'are the numbers of K1 and K2 objects with gradation μ; "—" = no object has this '
            "gradation."
        ),
        (
            "Formula (2): g_k = (1/m) Σ n(μ)·f_k(μ) if f_k(μ) ≥ 0.5, else n(μ)·(1 − f_k(μ)); g = "
            "0.5 "
            "means no difference between the classes, g = 1 full difference."
        ),
    ]
    bits = sum(b.bits.shape[1] + 1 for b in masks)
    mask_rows = max((b.membership.gradations.size for b in masks), default=0)
    if not masks or m * bits + 6 * len(masks) * mask_rows > book.options.max_sheet_cells:
        notes.append(
            book.omit(
                f"{MEMBERSHIP}: Task 2 (bit representations) is not written — "
                f"{trace.permitted_k.count} permitted k give too many bits for a sheet."
            )
        )
        sheet.notes(top, notes)
        return
    sheet.bar(
        top,
        1,
        bits + 2,
        "Task 2 · Bit representations by the majority rule (section 1.3): bit = 1 if μ > k/2, "
        "one bit per permitted k, read as a binary number",
    )
    headers = ["№"]
    for b in masks:
        headers += [f"{b.operator_label} · bit k = {k}" for k in b.ks]
        headers.append(f"{b.operator_label} · mask")
    sheet.row(top + 1, 1, [*headers, "Class"], "header")
    styles = []
    for b in masks:
        styles += ["integer"] * len(b.ks) + ["synthetic"]
    matrix = np.column_stack([np.column_stack([b.bits.astype(int), b.masks]) for b in masks])
    _object_rows(book, sheet, top + 2, matrix, styles, convert=int)
    top += m + 3

    beta = (1 << trace.permitted_k.count) - 1
    sheet.bar(
        top,
        1,
        7 * len(masks) - 1,
        "Membership (1) and stability (2) of the bit masks — gradations 0 … 2^r − 1 = "
        f"{beta}, β = {beta}",
    )
    mask_styles = ["label", "integer", "integer", "integer", "synthetic_num", "num"]
    for o, b in enumerate(masks):
        left = 1 + 7 * o
        table = b.membership
        sheet.row(
            top + 1,
            left,
            [f"{b.operator_label} · mask", "d₁", "d₂", "n", "f", "n·max(f,1−f)"],
            "header",
        )
        for g in range(table.gradations.size):
            sheet.row(
                top + 2 + g,
                left,
                [
                    int(table.gradations[g]),
                    int(table.d1[g]),
                    int(table.d2[g]),
                    int(table.n[g]),
                    float(table.f[g]),
                    float(table.weighted[g]),
                ],
                mask_styles,
            )
        foot = top + 2 + mask_rows
        sheet.merge(foot, left, 5, "g (2)", "key")
        sheet.put(foot, left + 5, b.stability, "value_num")
    notes.append(
        "Bit masks: the first permitted k is the most significant bit. Stability does not depend "
        "on this convention (it only relabels the gradations)."
    )
    sheet.notes(top + mask_rows + 4, notes)


# ---------------------------------------------------------------- Informativeness ω


def informativeness(book: Book) -> None:
    """Sheet *Informativeness ω* (Step 7): boundary G_k — formula (3) — and ω — formula (4)."""
    view = book.view
    trace = view.trace
    m, r = trace.m, trace.r
    features = trace.features
    names = _feature_names(view)
    width = max(r + 2, 8)
    sheet = book.sheet(OMEGA, "synthetic")
    sheet.widths({1: 26, (2, width): 12})
    sheet.title_bar(
        "Step 7 · Class boundary G_k — formula (3) — and informativeness ω — formula (4), "
        "per synthetic feature",
        width,
    )
    sheet.bar(3, 1, width, "Boundary and informativeness of every synthetic feature aᵤ ∈ Ψ(r)")
    sheet.put(4, 1, "Quantity", "header_left")
    sheet.row(4, 2, names, "header")

    def side(value: float | None) -> Any:
        return "none" if value is None else value

    lines: list[tuple[str, list[Any], str]] = [
        ("Operator", [f.operator_label for f in features], "value"),
        ("k", [f.k for f in features], "header_k"),
        ("q₂ = max{f < 0.5}", [side(f.boundary.q2) for f in features], "num"),
        ("q₁ = min{f > 0.5}", [side(f.boundary.q1) for f in features], "num"),
        ("G_k = (q₁ + q₂)/2", [f.boundary.G for f in features], "synthetic_num"),
        (
            "Correctly placed objects",
            [int(np.count_nonzero(f.correct)) for f in features],
            "integer",
        ),
        ("ω = correct / m  (4)", [f.omega for f in features], "value_num"),
        ("Rank of ω", [int(v) for v in weight_ranks(trace.omegas)], "integer"),
    ]
    for offset, (caption, values, style) in enumerate(lines, start=5):
        sheet.put(offset, 1, caption, "caption")
        sheet.row(offset, 2, values, style)

    top = 14
    sheet.bar(
        top,
        1,
        width,
        "g(S, k) = f_k(μ_S) — membership value of every object at its own gradation μ_S",
    )
    sheet.row(top + 1, 1, ["№", *names, "Class"], "header")
    membership_values = np.column_stack([f.object_membership for f in features])
    _object_rows(book, sheet, top + 2, membership_values, "num", convert=float)
    top += m + 3
    sheet.bar(
        top,
        1,
        width,
        "Correct side of the boundary: 1 if (Sₜ ∈ K1 and g > G_k) or (Sₜ ∈ K2 and g < G_k)",
    )
    sheet.row(top + 1, 1, ["№", *names, "Class"], "header")
    correct = np.column_stack([f.correct for f in features]).astype(int)
    _object_rows(book, sheet, top + 2, correct, "integer", convert=int)
    for u in range(r):
        sheet.highlight_equal(reference(top + 2, 2 + u, m, 1), "1")
    sheet.notes(
        top + m + 3,
        [
            (
                "Formula (3): G_k = (q₁ + q₂)/2 with q₂ = max{f_k(μ) | f_k(μ) < 0.5} and "
                "q₁ = min{f_k(μ) | f_k(μ) > 0.5}. If one side is empty, the neutral threshold 0.5 "
                "of "
                "section 1.3 is used."
            ),
            (
                "Formula (4): ω = (|{S ∈ K1 | g(S, k) > G_k}| + |{S ∈ K2 | g(S, k) < G_k}|) / "
                "|K1 ∪ K2| — the informativeness (weight) of the synthetic feature."
            ),
        ],
    )


# ---------------------------------------------------------------- Ψ(r) Contribution & Weight


def contribution(book: Book) -> None:
    """Sheet *Ψ(r) Contribution & Weight* (Step 8): η by formula (6) and the weights."""
    view = book.view
    trace = view.trace
    m, r = trace.m, trace.r
    features = trace.features
    names = _feature_names(view)
    width = max(r + 2, 8)
    sheet = book.sheet(CONTRIBUTION, "synthetic")
    sheet.widths({1: 24, (2, width): 12})
    sheet.title_bar(
        "Step 8 · Ψ(r) with contribution values η — formula (6) — and weights ω: the input of "
        "the hierarchical agglomerative grouping",
        width,
    )
    sheet.bar(
        3,
        1,
        width,
        "Gradations of the synthetic features  "
        "(α¹ⱼ, α²ⱼ = number of objects with aᵤ = j in K1 and K2)",
    )
    sheet.put(4, 1, "Gradations {1, 2}", "header_left")
    sheet.row(4, 2, names, "header")
    counts = [
        ("α¹₁  (aᵤ = 1 in K1)", 0, 0),
        ("α²₁  (aᵤ = 1 in K2)", 0, 1),
        ("α¹₂  (aᵤ = 2 in K1)", 1, 0),
        ("α²₂  (aᵤ = 2 in K2)", 1, 1),
    ]
    for offset, (caption, j, c) in enumerate(counts, start=5):
        sheet.put(offset, 1, caption, "caption")
        sheet.row(offset, 2, [int(f.alpha[j, c]) for f in features], "integer")

    sheet.bar(
        10,
        1,
        width,
        "Weight ω (Step 7) and contributions of gradations "
        "η(j) = ω · (α¹ⱼ/|K1| − α²ⱼ/|K2|)  — formula (6)",
    )
    sheet.put(11, 1, "Quantity", "header_left")
    sheet.row(11, 2, names, "header")
    sheet.put(12, 1, "ω  (informativeness)", "caption")
    sheet.row(12, 2, [f.weight for f in features], "value_num")
    sheet.put(13, 1, "η(1)", "caption")
    sheet.row(13, 2, [float(f.eta[0]) for f in features], "num")
    sheet.put(14, 1, "η(2)", "caption")
    sheet.row(14, 2, [float(f.eta[1]) for f in features], "num")

    top = 16
    sheet.bar(top, 1, width, f"Ψ(r) with contribution values  ηᵤ(a_tu)  [{view.dataset.name}]")
    sheet.row(top + 1, 1, ["№", *names, "Class"], "header")
    _object_rows(book, sheet, top + 2, trace.contributions, "num", convert=float)
    top += m + 3
    sheet.bar(top, 1, width, "Weight [Ψ(r)] and rank (1 = highest weight; ties → earlier feature)")
    sheet.put(top + 1, 1, "Feature", "header_left")
    sheet.row(top + 1, 2, names, "header")
    sheet.put(top + 2, 1, "Weight ω", "caption")
    sheet.row(top + 2, 2, [f.weight for f in features], "value_num")
    sheet.put(top + 3, 1, "Rank", "caption")
    sheet.row(top + 3, 2, [int(v) for v in weight_ranks(trace.weights)], "integer")
    sheet.highlight_equal(reference(top + 3, 2, 1, r), "1")
    sheet.notes(
        top + 5,
        [
            (
                "Formula (6): the contribution of gradation j of feature aᵤ is its weight ω times "
                "the "
                "difference of its relative frequencies in K1 and K2 — positive values point to "
                "K1, "
                "negative to K2."
            ),
            (
                'This sheet plays the role of the template sheet "Dataset (Contribution & '
                'Weight)": '
                "the HAG works on these contribution values and weights."
            ),
        ],
    )


__all__ = [
    "contribution",
    "count_word",
    "informativeness",
    "membership",
    "object_span",
    "psi",
    "same_class_counts",
    "sorted_neighbours",
]
