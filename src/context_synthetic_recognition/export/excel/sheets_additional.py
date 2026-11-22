"""Step 9: the sheets *Additional Feature j* — one per iteration of the agglomerative algorithm.

Every sheet shows the state before the iteration, a block per candidate with bₜ = R + η, the
majorizing function, the class means, θ, γ and θ/γ (STEP 3), the summary with cr1 and q, and STEP 4
with the stopping rule. The orange ⚠1 columns (class means) and the pink ⚠2 columns (majorizing
function passes in Step 4) are the two switches that deviate from the article when on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from context_synthetic_recognition.config.models import ClassMeans
from context_synthetic_recognition.core.additional_features.agglomerative import (
    AgglomerativeIteration,
    AgglomerativeResult,
)
from context_synthetic_recognition.core.arrays import FloatArray
from context_synthetic_recognition.export.excel.common import Book, blocks_that_fit
from context_synthetic_recognition.export.excel.writer import column_letter, reference
from context_synthetic_recognition.export.tables import plain
from context_synthetic_recognition.export.wording import tuplam_text
from context_synthetic_recognition.notation import DASH, additional_name, subscript, synthetic_name

BLOCK_COLUMNS = 14
"""Columns A–N of a candidate block."""
CENTRES_NOTE = (
    "⚠1  Class means (orange columns I–L).  The article uses the final means M = Σbₜ/|K| "
    "(Parameters!B24 = 2, the default).  With B24 = 1 M₁ and M₂ in each row are running partial "
    "means of the rows above (⚠ deviates from the article).  See sheet “Switches”."
)
PASSES_NOTE = (
    "⚠2  Step 4 (pink columns H–I).  The article applies the majorizing function once to "
    "R + η_q (column F; Parameters!B25 = 1, the default).  With B25 = 2 d is the majorizing "
    "function applied again to column H (⚠ deviates from the article).  See sheet “Switches”."
)


def iteration_of(agglomerative: AgglomerativeResult, j: int) -> AgglomerativeIteration | None:
    """The j-th iteration (1-based) if it was executed."""
    return agglomerative.iterations[j - 1] if 1 <= j <= len(agglomerative.iterations) else None


def tuplam_after(agglomerative: AgglomerativeResult, count: int) -> tuple[int, ...]:
    """TUPLAM after the first ``count`` iterations (an iteration not executed changes nothing)."""
    added = [it.q for it in agglomerative.iterations[:count] if it.q is not None]
    return (agglomerative.first, *added)


def crit_after(agglomerative: AgglomerativeResult, count: int) -> float:
    """Crit after the first ``count`` iterations (cr1₀ before the first)."""
    done = agglomerative.iterations[: min(count, len(agglomerative.iterations))]
    return done[-1].crit if done else agglomerative.settings.cr1


def pool_after(agglomerative: AgglomerativeResult, count: int) -> int:
    """|P| after iteration ``count`` (0 after an iteration that was not executed)."""
    if count == 0:
        return agglomerative.r - 1
    it = iteration_of(agglomerative, count)
    return 0 if it is None else int(it.candidates.size) - (it.q is not None)


def entering(agglomerative: AgglomerativeResult, j: int) -> FloatArray:
    """R(Sₜ) entering iteration ``j``."""
    it = iteration_of(agglomerative, j)
    if it is not None:
        return it.entering
    if j == 1:
        return agglomerative.contributions[:, agglomerative.first]
    return np.zeros(agglomerative.m)


def available(agglomerative: AgglomerativeResult, j: int, feature: int) -> bool:
    """Whether ``feature`` is a candidate of iteration ``j``."""
    it = iteration_of(agglomerative, j)
    return it is not None and bool(it.available[feature])


def _majorizing_rule(agglomerative: AgglomerativeResult) -> str:
    if agglomerative.settings.majorizing_function == "sigmoid":
        return (
            "bₜ ← bₜ + α·σ(−bₜ) if Sₜ ∈ K1,  bₜ ← bₜ − α·σ(−bₜ) if Sₜ ∈ K2,  σ(x) = 1/(1 + e^(−x))"
        )
    return (
        "bₜ ← bₜ + α·φ(−bₜ) if Sₜ ∈ K1,  bₜ ← bₜ − α·φ(−bₜ) if Sₜ ∈ K2,  "
        f"φ = {agglomerative.settings.majorizing_function}"
    )


def greedy(book: Book, j: int) -> None:
    """Sheet *Additional Feature j*: iteration ``j`` of Steps 3–5 → additional feature d_j."""
    view = book.view
    agglomerative, trace = view.agglomerative, view.trace
    m, r = agglomerative.m, agglomerative.r
    settings = agglomerative.settings
    features = trace.features
    names = [f.name for f in features]
    labels = view.labels
    ids = trace.object_ids
    it = iteration_of(agglomerative, j)
    before = j - 1
    R = entering(agglomerative, j)
    additional = additional_name(j - 1)
    sheet = book.sheet(book.plan.greedy[j - 1], "additional_features")

    enter_col = max(9, r + 3)
    info_col = enter_col + 4
    width = max(20, info_col + 6)
    sheet.widths({1: 18, 2: 13, 3: 4, 4: 13, 5: 4, 6: 13, 7: 9, (8, 17): 14, (18, width): 16})
    sheet.title_bar(
        f"Step 9.{j} · Hierarchical agglomerative algorithm — additional feature  →  "
        f"additional (latent) feature {additional}",
        width,
    )

    # ---- the state before the iteration
    sheet.bar(
        3,
        1,
        7,
        "STEP 1–2 · initialisation and the first feature"
        if j == 1
        else f"State before iteration {j}",
    )
    previous = iteration_of(agglomerative, before)
    added_before = None if previous is None else previous.q
    state: list[tuple[str, Any, str]] = [
        ("Iteration executed", int(it is not None), "value_yes_no"),
        (
            ("STEP 2 · u = argmax ωⱼ (index)", agglomerative.first + 1, "value")
            if j == 1
            else (
                "Feature added in the previous iteration",
                DASH if added_before is None else added_before + 1,
                "value",
            )
        ),
        ("TUPLAM before", tuplam_text(tuplam_after(agglomerative, before)), "value"),
        ("|TUPLAM| before", len(tuplam_after(agglomerative, before)), "value"),
        (
            "crit before",
            plain(crit_after(agglomerative, before)),
            "value" if j == 1 else "value_num",
        ),
        (
            ("max ω", float(agglomerative.informativeness.max()), "value_num")
            if j == 1
            else ("Candidates |P| before", pool_after(agglomerative, before), "value")
        ),
        (
            "Candidates available now |P|",
            sum(available(agglomerative, j, u) for u in range(r)),
            "value",
        ),
    ]
    for offset, (key, value, style) in enumerate(state, start=4):
        sheet.merge(offset, 1, 3, key, "key")
        sheet.merge(offset, 4, 4, value, style)
    sheet.put(12, 1, "Feature", "header_left")
    sheet.row(12, 2, names, "header")
    members = tuplam_after(agglomerative, before)
    sheet.put(13, 1, "In TUPLAM before", "caption")
    sheet.row(13, 2, [int(u in members) for u in range(r)], "integer")
    sheet.put(14, 1, "Available in P", "caption")
    sheet.row(14, 2, [int(available(agglomerative, j, u)) for u in range(r)], "integer")
    sheet.highlight_equal(reference(13, 2, 1, r), "1")

    # ---- R(Sₜ) entering the iteration
    sheet.bar(3, enter_col, 3, "R(Sₜ) entering this iteration")
    sheet.row(4, enter_col, ["№", "R(Sₜ)", "Class"], "header")
    for t in range(m):
        sheet.row(5 + t, enter_col, [ids[t], float(R[t]), labels[t]], ["label", "num", "class_"])

    # ---- definitions and parameters
    sheet.bar(3, info_col, 7, "Definitions  ·  parameters (Parameters sheet)")
    definitions = [
        (
            "R(Sₜ)",
            (
                "η_u(a_tu) of the first feature (STEP 2), then the additional feature of the "
                "previous iteration"
            ),
        ),
        ("bₜ", "R(Sₜ) + η_u(a_tu) for a candidate u ∈ P  (STEP 3)"),
        ("majorizing_function", _majorizing_rule(agglomerative)),
        (
            "M₁, M₂",
            (
                "class means: final means Σbₜ/|Kᵢ| (mode 2, the article) or running partial "
                "means (mode 1, ⚠ deviates)"
            ),
        ),
        ("θ", "intra-class similarity Σ |bₜ − M(own class)|"),
        ("γ", "inter-class difference Σ |bₜ − M(other class)|"),
        ("q", "argmin θ/γ over the available candidates (first one on ties);  crit = min θ/γ"),
        ("stop", "continue while |TUPLAM| < ϰ and crit > δ (and P is not empty)"),
    ]
    for offset, (key, text) in enumerate(definitions, start=4):
        sheet.put(offset, info_col, key, "key")
        sheet.merge(offset, info_col + 1, 6, text, "text")
    class_means = 1 if settings.class_means is ClassMeans.RUNNING else 2
    pairs = ["key", "value"] * 3
    sheet.row(12, info_col, ["α", settings.alpha, "δ", settings.delta, "ϰ", settings.kappa], pairs)
    sheet.row(
        13,
        info_col,
        [
            "class_means",
            class_means,
            "passes",
            settings.majorizing_passes,
            "cr1₀",
            plain(settings.cr1),
        ],
        pairs,
    )

    # ---- STEP 3: one block per candidate
    top = max(16, m + 6)
    sheet.bar(
        top,
        1,
        16,
        "STEP 3 — try adding each remaining feature:  bₜ = R(Sₜ) + ηᵤ(a_tu),  majorizing "
        "function,  θ / γ",
    )
    block = (m + 2) * BLOCK_COLUMNS
    shown = list(range(r))
    if blocks_that_fit(r, block, book.options.max_sheet_cells) < r:
        has_q = it is not None and it.q is not None
        chosen = it.q if it is not None and it.q is not None else agglomerative.first
        shown = [chosen]
        which = "the chosen feature q" if has_q else names[chosen]
        sheet.put(
            top + 1,
            1,
            book.omit(
                f"{sheet.title}: large experiment — only the block of {which} "
                f"is written ({r} candidates); the summary below and the CSV table "
                "agglomerative-candidates list θ, γ and θ/γ of every candidate."
            ),
            "note",
        )
    header = [
        "№",
        "R(Sₜ)",
        "+",
        "ηᵤ(a_tu)",
        "=",
        "bₜ",
        "Class",
        "bₜ (majorized)",
        "⚠1 Σ K1 (running)",
        "⚠1 Σ K2 (running)",
        "⚠1 M₁ used",
        "⚠1 M₂ used",
        "|b − M own|",
        "|b − M other|",
    ]
    header_styles = ["header"] * 8 + ["warn1_header"] * 4 + ["header"] * 2
    row_styles = [
        "label",
        "num",
        "operator",
        "num",
        "operator",
        "num",
        "class_",
        "num",
        "warn1_num",
        "warn1_num",
        "warn1_num",
        "warn1_num",
        "num",
        "num",
    ]
    row = top + 2
    for position, u in enumerate(shown):
        feature = features[u]
        scan = agglomerative.scan_from(R, u)
        index = subscript(u + 1)
        sheet.bar(
            row,
            1,
            BLOCK_COLUMNS,
            f"Candidate {feature.name}  ·  operator {feature.operator_label}  ·  k = {feature.k}"
            f"   —   bₜ = R(Sₜ) + η{index}(a_t{index})",
            "bar_left",
        )
        if it is None:
            status = "iteration not executed"
        else:
            status = "available" if bool(it.available[u]) else "in TUPLAM — skipped"
        sheet.merge(row, 15, 2, status, "value_left")
        sheet.row(row + 1, 1, header, header_styles)
        for t in range(m):
            sheet.row(
                row + 2 + t,
                1,
                [
                    f"b{subscript(t + 1)}",
                    float(scan.entering[t]),
                    "+",
                    float(scan.eta[t]),
                    "=",
                    float(scan.b[t]),
                    labels[t],
                    float(scan.majorized[t]),
                    float(scan.sum1[t]),
                    float(scan.sum2[t]),
                    float(scan.centre1[t]),
                    float(scan.centre2[t]),
                    float(scan.to_own[t]),
                    float(scan.to_other[t]),
                ],
                row_styles,
            )
        foot = row + m + 2
        sheet.merge(foot, 11, 2, "θ, γ  →", "key")
        sheet.put(foot, 13, scan.theta, "value_num")
        sheet.put(foot, 14, scan.gamma, "value_num")
        sheet.put(foot, 15, "θ / γ", "key")
        sheet.put(foot, 16, scan.ratio if available(agglomerative, j, u) else DASH, "good_num")
        if position == 0:
            sheet.note_box(row, 18, 3, m + 2, CENTRES_NOTE, "warn1_box")
        row += m + 4

    # ---- STEP 3 summary
    sheet.bar(row, 8, 5, "STEP 3 · summary  —  cr1 = min θ/γ,  q = argmin θ/γ")
    sheet.row(row + 1, 8, ["Feature", "Available", "θ", "γ", "θ / γ"], "header")
    summary_styles = ["label", "yes_no", "num", "num", "good_num"]
    for u in range(r):
        if it is not None and bool(it.available[u]):
            position = int(np.flatnonzero(it.candidates == u)[0])
            values: list[Any] = [
                names[u],
                1,
                float(it.theta[position]),
                float(it.gamma[position]),
                float(it.ratio[position]),
            ]
        else:
            values = [names[u], 0, DASH, DASH, DASH]
        sheet.row(row + 2 + u, 8, values, summary_styles)
    cr1_row = row + r + 3
    q_row = cr1_row + 1
    sheet.highlight_formula(
        reference(row + 2, 8, r, 5), f"ROW()-{row + 1}=${column_letter(10)}${q_row}", "good"
    )
    q = None if it is None else it.q
    sheet.merge(cr1_row, 8, 2, "cr1 = min θ/γ", "key")
    sheet.put(cr1_row, 10, DASH if it is None else it.cr1, "value_num")
    sheet.merge(q_row, 8, 2, "q = argmin θ/γ", "key")
    sheet.put(q_row, 10, DASH if q is None else q + 1, "value")
    sheet.put(q_row, 11, DASH if q is None else names[q], "value")

    # ---- STEP 4
    top4 = q_row + 3
    sheet.bar(
        top4,
        1,
        9,
        "STEP 4 — R(Sₜ) ← R(Sₜ) + η_q(a_tq), majorizing function  ⇒  additional (latent) feature "
        f"{additional}",
    )
    sheet.row(
        top4 + 1,
        1,
        [
            "№",
            "R(Sₜ)",
            "+",
            "η_q(a_tq)",
            "=",
            "bₜ = R + η_q",
            "Class",
            "⚠2 bₜ majorized once",
            f"⚠2 {additional} = R(Sₜ) new",
        ],
        ["header"] * 7 + ["warn2_header"] * 2,
    )
    eta_q = np.zeros(m) if q is None else agglomerative.contributions[:, q]
    step4_styles = [
        "label",
        "num",
        "operator",
        "num",
        "operator",
        "num",
        "class_",
        "warn2_num",
        "warn2_additional",
    ]
    for t in range(m):
        once: Any = DASH
        new: Any = DASH
        if it is not None and it.once is not None and it.additional is not None:
            once, new = float(it.once[t]), float(it.additional[t])
        sheet.row(
            top4 + 2 + t,
            1,
            [
                f"b{subscript(t + 1)}",
                float(R[t]),
                "+",
                float(eta_q[t]),
                "=",
                float(R[t] + eta_q[t]),
                labels[t],
                once,
                new,
            ],
            step4_styles,
        )
    sheet.note_box(top4 + 1, 18, 3, m + 1, PASSES_NOTE, "warn2_box")

    sheet.bar(top4, 11, 6, "STEP 4 · update and stopping rule")
    after = tuplam_after(agglomerative, j)
    update: list[tuple[str, Any, str]] = [
        ("crit = cr1", crit_after(agglomerative, j), "value_num"),
        ("Feature added q", DASH if q is None else q + 1, "value"),
        ("|TUPLAM| after", len(after), "value"),
        ("TUPLAM after", tuplam_text(after), "value_left"),
        ("|P| after", pool_after(agglomerative, j), "value"),
        ("δ", settings.delta, "value"),
        ("ϰ", settings.kappa, "value"),
        ("if (|TUPLAM| < ϰ) & (crit > δ)", int(it is not None and it.go_on), "value_yes_no"),
    ]
    for offset, (key, value, style) in enumerate(update, start=1):
        sheet.merge(top4 + offset, 11, 2, key, "key")
        sheet.merge(top4 + offset, 13, 4, value, style)
    if it is None:
        step5 = "Iteration not executed"
    else:
        step5 = "Go back to Step 3" if it.go_on else "Output TUPLAM and stop"
    sheet.merge(top4 + 9, 11, 6, step5, "decision")
    # beside the last rows of the ⚠2 note while the features fit left of it, otherwise below it
    flags = top4 + 1 + max(m, 10) + (0 if 12 + r <= 18 else 2)
    sheet.put(flags, 11, "Feature", "header_left")
    sheet.row(flags, 12, names, "header")
    sheet.put(flags + 1, 11, "In TUPLAM", "caption")
    sheet.row(flags + 1, 12, [int(u in after) for u in range(r)], "integer")
    sheet.highlight_equal(reference(flags + 1, 12, 1, r), "1")
    sheet.notes(
        flags + 2,
        [
            (
                "Each candidate block recomputes bₜ with the candidate feature; the one with the "
                "smallest θ/γ is added to TUPLAM and its values become the additional feature "
                f"{additional}."
            ),
            (
                "Parameters α, δ, ϰ, cr1 and the two switches (class means, Step 4 passes) are "
                "those of the Parameters sheet; the defaults follow the article."
            ),
        ],
    )


__all__ = [
    "available",
    "crit_after",
    "entering",
    "greedy",
    "iteration_of",
    "pool_after",
    "synthetic_name",
    "tuplam_after",
]
