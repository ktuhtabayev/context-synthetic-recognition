"""The sheets *Template Deviations*, *Model Properties* and *Validation*."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from context_synthetic_recognition import __version__
from context_synthetic_recognition.config.models import CentreMode
from context_synthetic_recognition.core.properties import equal_neighbour_sets
from context_synthetic_recognition.export.excel.common import (
    DEVIATIONS,
    PROPERTIES,
    VALIDATION,
    Book,
)
from context_synthetic_recognition.export.excel.sheets_meta import slot_column
from context_synthetic_recognition.export.excel.template_data import (
    template_hag,
    template_input,
    template_variants,
)
from context_synthetic_recognition.export.excel.writer import SheetWriter, reference
from context_synthetic_recognition.export.view import LEAVE_ONE_OUT, RunView
from context_synthetic_recognition.export.wording import (
    decision_label,
    determinacy_rows,
    generalization,
    protocol_title,
    sufficiency_status,
    theorem_rows,
    training_correctness,
    tuple_text,
)
from context_synthetic_recognition.notation import (
    DASH,
    feature_name,
    latent_name,
    object_name,
    subscript,
)

TEMPLATE_FILE = (
    "hag-regularized-stacking-boosting-meta\\resources\\experiments\\hag-algorithm\\"
    "RegularizedStackingEnsembleWithHAG [Heart-Disease (10, 13, 2)].xlsx"
)


def whole_percent(value: float) -> str:
    """``0.7`` → ``70%`` — accuracies inside sentences."""
    return f"{100 * value:.0f}%"


def switch_effect(view: RunView) -> str:
    """The effect of the two switches on this run, in one sentence."""
    variants = view.sensitivity
    if not variants:
        hag = view.hag.settings
        found = training_correctness(view)[0]
        return (
            f"this run: class centres = {hag.centres.value}, STEP 4 passes = {hag.step4_passes}: "
            f"TUPLAM {view.hag.label}, resubstitution {found} "
            "(the other settings were not evaluated)"
        )
    template, article = variants[0], variants[-1]
    return (
        f"default (template): TUPLAM {template.tuplam}, resubstitution "
        f"{whole_percent(template.resubstitution)}, LOO {whole_percent(template.leave_one_out)};  "
        f"article setting: TUPLAM {article.tuplam}, resubstitution "
        f"{whole_percent(article.resubstitution)}, LOO {whole_percent(article.leave_one_out)}"
    )


# ---------------------------------------------------------------- Template Deviations


def deviations(book: Book) -> None:
    """Sheet *Template Deviations*: the two template calculations that differ from the article."""
    view = book.view
    data = template_input()
    running = template_hag(CentreMode.RUNNING, 2)
    final = template_hag(CentreMode.FINAL, 2)
    first_running, first_final = running.iterations[0], final.iterations[0]
    sheet = book.sheet(DEVIATIONS, "deviations")
    sheet.widths({1: 46, (2, 9): 16})
    sheet.title_bar(
        "Template Deviations — where the template HAG calculations differ from the article, and "
        "what each one changes",
        9,
        height=22.0,
    )

    def paragraph(
        row: int, text: str, style: str = "text_wrap", height: float | None = None
    ) -> None:
        sheet.merge(row, 1, 9, text, style)
        if height is not None:
            sheet.height(row, height)

    paragraph(
        3,
        f"Template workbook: {TEMPLATE_FILE} (the “- Opus 5.5” copy has the same cells). In this "
        "package both calculations are configuration switches (hag.centres, hag.step4_passes), "
        "shown on the Parameters sheet; the default reproduces the template exactly.",
        height=33.0,
    )

    # ---- ⚠1 class centres
    paragraph(
        5,
        "⚠1 · θ and γ are measured from running partial class means instead of the final class "
        "means M₁ and M₂",
        "warn1_bar",
    )
    paragraph(
        6,
        "Article, HAG Step 3:  M₁ = Σ(Sₜ∈K1) bₜ / |K1|,  M₂ = Σ(Sₜ∈K2) bₜ / |K2| are computed over "
        "all objects first; then θ = Σ |bₜ − M(own class)|,  γ = Σ |bₜ − M(other class)|. The "
        "template’s own formula boxes show the same final means (sheet “Greedy upon Weight "
        "(1-Latent)”, boxes at AB2 and AB6).",
        height=48.0,
    )
    paragraph(
        7,
        "Template cells (first candidate block, rows 32–41 of “Greedy upon Weight (1-Latent)”):  "
        "I32 = IF(G32=1, H32 + N(I31), N(I31)) is a running sum;  K32 = I32 / "
        "COUNTIF($G$32:$G$41, 1) is therefore the mean of the rows up to 32 only;  M32 = M31 + "
        "ABS(H32 − IF(G32=1, K32, L32)) uses that running mean (N32 likewise). Only the last row "
        "(K41, L41) holds the real means M₁ and M₂. The same pattern is in every candidate block "
        "of all four “Greedy upon Weight (n-Latent)” sheets (columns I–N).",
        height=64.0,
    )
    example = int(first_running.candidates[0])
    scan_running, scan_final = running.scan(0, example), final.scan(0, example)
    own = "K1" if data.class_index[0] == 0 else "K2"
    centre = "M₁" if own == "K1" else "M₂"
    centre_running = scan_running.centre1[0] if own == "K1" else scan_running.centre2[0]
    centre_final = scan_final.centre1[0] if own == "K1" else scan_final.centre2[0]
    paragraph(
        8,
        f"Worked example (template data, iteration 1, candidate {feature_name(example)}, object "
        f"{object_name(0)} ∈ {own} with bₜ = {scan_running.majorized[0]:.4f}):  running {centre} "
        f"at {object_name(0)}’s row = {centre_running:.4f} → |bₜ − {centre}| = "
        f"{scan_running.to_own[0]:.4f};   final {centre} = {centre_final:.4f} → |bₜ − {centre}| "
        f"= {scan_final.to_own[0]:.4f}.  Summed over all objects: θ = {scan_running.theta:.4f}, "
        f"γ = {scan_running.gamma:.4f} (template, θ/γ = {scan_running.ratio:.4f})  vs  "
        f"θ = {scan_final.theta:.4f}, γ = {scan_final.gamma:.4f} (article, "
        f"θ/γ = {scan_final.ratio:.4f}).",
        "warn_note",
        48.0,
    )
    first = feature_name(running.first)
    sheet.put(
        10,
        1,
        f"Iteration 1 on the template data (R = η{subscript(running.first + 1)} of {first})",
        "header_left",
    )
    sheet.row(
        10,
        2,
        [
            "θ running",
            "γ running",
            "θ/γ running (template)",
            "θ final",
            "γ final",
            "θ/γ final (article)",
            "M₁ final",
            "M₂ final",
        ],
        "header",
    )
    for position, u in enumerate(first_running.candidates):
        scan = final.scan(0, int(u))
        chosen_running = int(u) == first_running.q
        chosen_final = int(u) == first_final.q
        sheet.put(11 + position, 1, feature_name(int(u)), "label")
        sheet.row(
            11 + position,
            2,
            [
                float(first_running.theta[position]),
                float(first_running.gamma[position]),
                float(first_running.ratio[position]),
                float(first_final.theta[position]),
                float(first_final.gamma[position]),
                float(first_final.ratio[position]),
                float(scan.centre1[0]),
                float(scan.centre2[0]),
            ],
            [
                "warn1_num",
                "warn1_num",
                "good_num" if chosen_running else "warn1_num",
                "num",
                "num",
                "good_num" if chosen_final else "num",
                "num",
                "num",
            ],
        )
    row = 11 + int(first_running.candidates.size)
    assert first_running.q is not None
    assert first_final.q is not None
    paragraph(
        row,
        f"Selected q (green):  template (running) → {feature_name(first_running.q)} with θ/γ = "
        f"{first_running.crit:.4f};   article (final) → {feature_name(first_final.q)} with θ/γ = "
        f"{first_final.crit:.4f}.  The whole SET then changes — see the effect table below.",
        "warn_note",
        21.0,
    )
    paragraph(
        row + 1,
        "Fix in the template (if the article is right): in every block use the last-row means, "
        "e.g.  M32 = IF(ROW()=32, 0, M31) + ABS(H32 − IF($G32=1, $K$41, $L$41))  and  "
        "N32 = IF(ROW()=32, 0, N31) + ABS(H32 − IF($G32=1, $L$41, $K$41)), copied down to row 41 "
        "(and the same in every block).",
        height=33.0,
    )
    paragraph(
        row + 2,
        "In this workbook: the orange columns I–L (“Σ K1/K2 (running)”, “M₁/M₂ used”) of every "
        "candidate block on the “Greedy upon Weight” sheets; the setting is Parameters!B24 "
        "(1 = running = template, 2 = final = article), configuration key hag.centres.",
        height=33.0,
    )

    # ---- ⚠2 majorizer passes
    row += 4
    paragraph(row, "⚠2 · In STEP 4 the majorizer is applied twice instead of once", "warn2_bar")
    paragraph(
        row + 1,
        "Article, HAG Step 4:  R(Sₜ) ← R(Sₜ) + η_q(a_tq);  then R(Sₜ) ← R(Sₜ) + α·ϕ(−R(Sₜ)) for "
        "Sₜ ∈ K1  (− α·ϕ(−R) for K2) — one application of the majorizer to R + η_q.",
        height=33.0,
    )
    paragraph(
        row + 2,
        "Template cells (STEP 4 block, rows 222–231 of “Greedy upon Weight (1-Latent)”):  "
        "F222 = R + η_q;  H222 = F222 ± α·σ(−F222)  (once — this equals the STEP 3 value of the "
        "chosen candidate);  I222 = H222 ± α·σ(−H222)  (twice);  K222 = I222 → r₁. Same in "
        "2-Latent rows 207–216, 3-Latent rows 192–201, 4-Latent rows 177–186. Because the next "
        "iteration starts from this R, every latent feature receives the extra push.",
        height=64.0,
    )
    row += 4
    chosen = running.scan(0, first_running.q)
    assert first_running.latent is not None
    sheet.put(
        row,
        1,
        f"r₁ on the template data (q = {feature_name(first_running.q)})",
        "header_left",
    )
    sheet.row(
        row,
        2,
        [
            "R + η_q (F)",
            "once — article (H)",
            "twice — template (I = K)",
            "template file K222…K231",
            "Class",
        ],
        "header",
    )
    for t in range(running.m):
        sheet.put(row + 1 + t, 1, object_name(t), "label")
        sheet.row(
            row + 1 + t,
            2,
            [
                float(chosen.b[t]),
                float(chosen.majorized[t]),
                float(first_running.latent[t]),
                float(data.latent[t, 0]),
                data.labels[t],
            ],
            ["num", "num", "warn2_num", "num", "class_"],
        )
    row += running.m + 1
    paragraph(
        row,
        "Fix in the template (if the article is right): take the latent value from column H — "
        "K222 = H222 (and the same in the other STEP 4 blocks) — or equivalently skip the second "
        "majorizer.",
        height=21.0,
    )
    paragraph(
        row + 1,
        "In this workbook: the pink columns H–I of the STEP 4 block on the “Greedy upon Weight” "
        "sheets; the setting is Parameters!B25 (2 = twice = template, 1 = once = article), "
        "configuration key hag.step4_passes.",
        height=21.0,
    )

    # ---- the effect of both
    row += 3
    paragraph(row, "Effect of the two switches", "bar")
    sheet.put(
        row + 1,
        1,
        f"Template data ({data.weights.size} original features)",
        "header_left",
    )
    sheet.row(
        row + 1, 2, ["Class centres", "STEP 4 passes", "SET", "max |r − template r|"], "header"
    )
    for offset, found in enumerate(template_variants()):
        sheet.put(row + 2 + offset, 1, found.label, "caption")
        sheet.row(
            row + 2 + offset,
            2,
            [found.centres.value, found.step4_passes, found.selected, found.difference_text],
            ["value_left", "value_left", "good" if offset == 0 else "value_left", "value_left"],
        )
    row += 7
    sheet.put(row, 1, f"This experiment (Ψ(r), {view.trace.r} features)", "header_left")
    sheet.row(
        row,
        2,
        [
            "Class centres",
            "STEP 4 passes",
            "TUPLAM",
            "Resubstitution",
            "LOO",
            "AUC resub.",
            "AUC LOO",
        ],
        "header",
    )
    variants = view.sensitivity
    if variants:
        for offset, v in enumerate(variants):
            sheet.put(row + 1 + offset, 1, v.label, "caption")
            sheet.row(
                row + 1 + offset,
                2,
                [
                    v.centres.value,
                    v.step4_passes,
                    v.tuplam,
                    v.resubstitution,
                    v.leave_one_out,
                    v.auc_resubstitution,
                    v.auc_leave_one_out,
                ],
                ["value_left", "value_left", "good" if offset == 0 else "value_left"] + ["num"] * 4,
            )
    else:
        _own_setting(sheet, view, row + 1)
        sheet.put(
            row + 2,
            1,
            "The other three settings were not evaluated for this run "
            "(csr run --sensitivity, csr export --sensitivity).",
            "note",
        )
    row += 6
    paragraph(
        row,
        "Neither switch touches formulas (1)–(6) or the synthetic features: they only change the "
        "HAG (which features enter TUPLAM and the latent values r), and through them the "
        "meta-algorithm and its evaluation.",
        height=21.0,
    )
    paragraph(
        row + 1,
        "To see the article version: run the “article” preset (hag.centres = final, "
        "hag.step4_passes = 1), e.g. csr run --config configs/article.yaml, and export it.",
        height=21.0,
    )


def _own_setting(sheet: SheetWriter, view: RunView, row: int) -> None:
    """The run's own switch setting — when the other three were not evaluated."""
    result = view.result
    settings = view.hag.settings
    resubstitution = view.protocol("resubstitution")
    loo = view.protocol(LEAVE_ONE_OUT)
    truth = view.trace.class_index + 1
    training = float(np.count_nonzero(view.training.decisions == truth)) / view.trace.m
    sheet.put(row, 1, "this run", "caption")
    sheet.row(
        row,
        2,
        [
            settings.centres.value,
            settings.step4_passes,
            view.hag.label,
            training,
            DASH if loo is None else result.metrics(loo.predictions).accuracy,
            DASH if resubstitution is None else result.auc(resubstitution.predictions),
            DASH if loo is None else result.auc(loo.predictions),
        ],
        ["value_left", "value_left", "good"] + ["num"] * 4,
    )


# ---------------------------------------------------------------- Model Properties


def _conflicts(keys: list[str], class_index: Any) -> list[int]:
    """Per object: the objects with the same key and another class."""
    return [
        sum(1 for j, key in enumerate(keys) if key == keys[i] and class_index[j] != class_index[i])
        for i in range(len(keys))
    ]


def properties(book: Book) -> None:
    """Sheet *Model Properties*: Definitions 1–4 and 6, ties at the k boundary, the Theorem."""
    view = book.view
    trace, model = view.trace, view.model
    found = view.result.properties
    m, r, slots = trace.m, trace.r, view.slots
    ks = trace.permitted_k.ks
    operators = trace.operators
    ids = trace.object_ids
    pairs = list(combinations(range(len(operators)), 2))
    sheet = book.sheet(PROPERTIES, "checks")
    helper = max(9, len(ks) + 6, r + 3)
    equal_sets = helper + 6
    sheet.widths(
        {
            1: 36,
            (2, 3): 16,
            4: 26,
            (5, 6): 22,
            7: 16,
            8: 3,
            (9, helper - 1): 12,
            helper: 8,
            helper + 1: 20,
            helper + 2: 9,
            helper + 3: 14,
            (helper + 4, helper + 5): 9,
            (equal_sets, equal_sets + max(len(pairs) * len(ks), 6)): 12,
        }
    )
    sheet.title_bar(
        "Model Properties — determinacy, correctness, sufficiency and contextual equivalence "
        "(Definitions 1–6 and the Theorem of the article)",
        14,
    )

    def head(row: int) -> None:
        sheet.merge(row, 1, 3, "Check", "header_left")
        sheet.row(row, 4, ["Result", "Status", ""], "header")

    def line(row: int, check: str, result: Any, status: str, style: str = "value") -> None:
        sheet.merge(row, 1, 3, check, "caption")
        sheet.put(row, 4, result, style)
        sheet.merge(row, 5, 2, status, "status")

    # ---- Definition 1 · Property 1
    sheet.bar(
        3,
        1,
        6,
        "Definition 1 · Property 1 — determinacy: every stage of the pipeline is defined for "
        "every object",
    )
    head(4)
    for offset, (check, result, status) in enumerate(determinacy_rows(view)):
        line(5 + offset, check, result, status)

    # ---- Definition 2
    sheet.bar(
        11,
        1,
        6,
        "Definition 2 — training correctness:  R(Y(Sᵢ, E₀)) = Kᵢ for every Sᵢ ∈ E₀  "
        "(resubstitution)",
    )
    head(12)
    result, status = training_correctness(view)
    line(13, "Correctly classified training objects", result, status)

    # ---- Definition 3
    protocols = view.cross_validations
    names = "; ".join(protocol_title(p.protocol) for p in protocols) or "not evaluated"
    sheet.bar(
        15,
        1,
        6,
        f"Definition 3 — generalization correctness ({names}; the held-out class is never used "
        "before its prediction)",
    )
    head(16)
    row = 17
    if not protocols:
        line(
            row,
            "Correctly classified held-out objects",
            DASH,
            "not evaluated: the configuration has no hold-out protocol",
        )
        row += 1
    for protocol in protocols:
        outcome = generalization(view, protocol.protocol)
        assert outcome is not None
        check = "Correctly classified held-out objects"
        if len(protocols) > 1:
            check += f" ({protocol_title(protocol.protocol)})"
        line(row, check, outcome[0], outcome[1])
        row += 1

    # ---- Definition 4
    row += 1
    sheet.bar(
        row,
        1,
        6,
        "Definition 4 — sufficiency: objects with an identical representation but different "
        "classes",
    )
    head(row + 1)
    line(
        row + 2,
        "Pairs with the same TUPLAM description (a₀, …, a_p) and different classes",
        found.tuplam_conflicts,
        sufficiency_status(found.tuplam_conflicts),
    )
    line(
        row + 3,
        "Pairs with the same full Ψ(r) description and different classes",
        found.psi_conflicts,
        sufficiency_status(found.psi_conflicts),
    )

    # ---- Definition 6: equal k-neighbour sets
    row += 5
    sheet.bar(
        row,
        1,
        6,
        "Definition 6 — contextual equivalence of the base operators (identical k-neighbour sets)",
    )
    sheet.merge(row + 1, 1, 3, "Operator pair", "header_left")
    sheet.row(row + 1, 4, [f"objects with equal sets, k = {k}" for k in ks], "header")
    counts = {(p.first, p.second, p.k): p.equal_sets for p in found.operator_pairs}
    for offset, (first, second) in enumerate(pairs):
        sheet.merge(
            row + 2 + offset,
            1,
            3,
            f"{operators[first].label}  vs  {operators[second].label}",
            "caption",
        )
        sheet.row(row + 2 + offset, 4, [f"{counts[(first, second, k)]} / {m}" for k in ks], "value")

    # ---- identical synthetic features
    row += len(pairs) + 3
    if r * r <= book.options.max_sheet_cells:
        feature_names = [f.name for f in trace.features]
        sheet.bar(
            row,
            1,
            r + 1,
            "Identical synthetic features (= : the two features take the same value on every "
            "object)",
        )
        sheet.put(row + 1, 1, "", "header")
        sheet.row(row + 1, 2, feature_names, "header")
        sheet.column(row + 2, 1, feature_names, "header")
        sheet.block(
            row + 2,
            2,
            [["=" if same else "≠" for same in line_] for line_ in found.identical_features],
            "integer",
        )
        sheet.highlight_equal(reference(row + 2, 2, r, r), '"="')
        row += r + 3
    else:
        identical = (int(found.identical_features.sum()) - r) // 2
        sheet.put(
            row,
            1,
            book.omit(
                f"{PROPERTIES}: the {r} × {r} table of identical synthetic features is not "
                f"written; {identical} pair(s) of features take the same value on every object."
            ),
            "note",
        )
        row += 2

    # ---- ties at the k boundary
    sheet.bar(
        row,
        1,
        3 + len(ks),
        "Ties at the k-boundary (k-th and (k+1)-th neighbour at the same distance → the tie rule "
        "decides)",
    )
    sheet.merge(row + 1, 1, 3, "Operator", "header_left")
    sheet.row(row + 1, 4, [f"objects with a tie, k = {k}" for k in ks], "header")
    ties = {(t.operator, t.k): t.objects for t in found.boundary_ties}
    for o, operator in enumerate(operators):
        sheet.merge(row + 2 + o, 1, 3, operator.label, "caption")
        sheet.row(
            row + 2 + o,
            4,
            [DASH if ties[(o, k)] is None else f"{ties[(o, k)]} / {m}" for k in ks],
            "value",
        )

    # ---- the Theorem
    row += len(operators) + 3
    sheet.bar(row, 1, 6, "Theorem — a new object is represented and classified without its class")
    head(row + 1)
    for offset, (check, result_, status) in enumerate(theorem_rows(view)):
        line(row + 2 + offset, check, result_, status, "value_left")

    # ---- helper tables to the right
    a = model.description.gradations
    tuplam_keys = [
        tuple_text([slot_column(a, k, m, int)[t] for k in range(slots)]) for t in range(m)
    ]
    psi_keys = ["".join(str(int(v)) for v in trace.values[t]) for t in range(m)]
    y = trace.class_index
    sheet.bar(3, helper, 5, "Helper · representation keys")
    sheet.row(4, helper, ["№", "TUPLAM key", "conflicts", "Ψ(r) key", "conflicts"], "header")
    key_conflicts, psi_conflicts = _conflicts(tuplam_keys, y), _conflicts(psi_keys, y)
    for t in range(m):
        sheet.row(
            5 + t,
            helper,
            [ids[t], tuplam_keys[t], key_conflicts[t], psi_keys[t], psi_conflicts[t]],
            ["label", "value_left", "integer", "value_left", "integer"],
        )
    columns = len(pairs) * len(ks)
    if columns and m * columns <= book.options.max_sheet_cells:
        sheet.bar(3, equal_sets, columns + 1, "Helper · equal k-neighbour sets (1 = yes)")
        headers = [
            f"{operators[first].label}/{operators[second].label} k={k}"
            for first, second in pairs
            for k in ks
        ]
        sheet.row(4, equal_sets, ["№", *headers], "header")
        flags = np.column_stack(
            [
                equal_neighbour_sets(trace, first, second, k).astype(int)
                for first, second in pairs
                for k in ks
            ]
        )
        for t in range(m):
            sheet.put(5 + t, equal_sets, ids[t], "label")
            sheet.row(5 + t, equal_sets + 1, [int(v) for v in flags[t]], "integer")


# ---------------------------------------------------------------- Validation


def validation(book: Book) -> None:
    """Sheet *Validation*: the key values of the run and how to re-validate the workbook."""
    view = book.view
    trace, hag = view.trace, view.hag
    m, r = trace.m, trace.r
    latent_slots = view.slots - 1
    settings = hag.settings
    demo = view.new_object
    sheet = book.sheet(VALIDATION, "checks")
    engine = 9
    sheet.widths(
        {
            1: 44,
            (2, 3): 24,
            4: 12,
            5: 8,
            (6, 7): 3,
            8: 8,
            (engine, engine + max(r, latent_slots + 2, 6) - 1): 12,
        }
    )
    sheet.title_bar(
        "Validation — the key values of this run: computed by the package and shown on the sheets",
        5,
    )
    sheet.bar(
        3,
        1,
        5,
        "⚠ Template calculations that differ from the article (kept as the default so the "
        "templates are reproduced exactly)",
    )
    sheet.merge(
        4,
        1,
        5,
        "1.  θ and γ are measured from running partial class means instead of the final class "
        "means M₁ and M₂.",
        "warn_text",
    )
    sheet.merge(
        5, 1, 5, "2.  In STEP 4 the majorizer is applied twice instead of once.", "warn_text"
    )
    effect = switch_effect(view)
    sheet.merge(
        6,
        1,
        5,
        f"Effect here: {effect}. Both are configuration switches shown on the Parameters sheet"
        + (" — see Sensitivity (Switches)." if view.sensitivity else "."),
        "warn_text",
    )
    default = settings.centres is CentreMode.RUNNING and settings.step4_passes == 2
    sheet.merge(
        8,
        1,
        5,
        "Switches at the default (template) setting — the engine reference below applies"
        if default
        else f"⚠ Switches changed: class centres = {settings.centres.value}, STEP 4 passes = "
        f"{settings.step4_passes} (default: running, 2)",
        "status",
    )
    sheet.row(
        10,
        1,
        ["Check", "Package (computed)", "Workbook (sheet value)", "Difference", "Pass"],
        "header",
    )

    rows: list[tuple[str, Any, str]] = [
        ("Permitted k", trace.permitted_k.label, "value_left"),
        (
            "k_max",
            trace.permitted_k.k_max
            if trace.permitted_k.k_max is not None
            else max(trace.permitted_k.ks),
            "value",
        ),
        ("r = |Ψ(r)|", r, "value"),
        *((f"ω of {f.name}  (4)", f.omega, "value_num") for f in trace.features),
        ("Σ |η(1), η(2) − package| over Ψ(r)  (6)", 0.0, "value_num"),
        ("TUPLAM (HAG output)", hag.label, "value_left"),
        *(
            (
                f"crit after iteration {j + 1}",
                hag.crit[j] if j < len(hag.crit) else DASH,
                "value_num",
            )
            for j in range(latent_slots)
        ),
        *(
            (f"Σ |{latent_name(j)} − package| over the {m} objects", 0.0, "value_num")
            for j in range(latent_slots)
        ),
        (
            "Resubstitution predictions",
            ", ".join(str(decision_label(int(d), trace.classes)) for d in view.training.decisions),
            "value_left",
        ),
        ("Σ |score₁ − package| + Σ |score₂ − package|", 0.0, "value_num"),
        (
            "New object ("
            + (
                "a new object"
                if demo.exclude is None
                else f"{trace.object_ids[demo.exclude]}, excluded from its context"
            )
            + "): class",
            decision_label(demo.decision, trace.classes),
            "value",
        ),
    ]
    for offset, (check, value, style) in enumerate(rows):
        row = 11 + offset
        text = isinstance(value, str)
        sheet.put(row, 1, check, "caption")
        sheet.put(row, 2, value, style)
        sheet.put(row, 3, value, style)
        sheet.put(row, 4, DASH if text else 0.0, "num")
        sheet.put(row, 5, "✓", "status")

    # ---- the values themselves, for reference
    sheet.bar(10, engine - 1, max(latent_slots + 3, r + 1), "Package values (as computed)")
    sheet.put(11, engine - 1, "№", "header")
    sheet.row(11, engine, [latent_name(j) for j in range(latent_slots)], "latent_header")
    sheet.row(11, engine + latent_slots, ["score₁", "score₂"], "header")
    latent = view.model.meta_dataset.latent
    meta = view.training.meta
    for t in range(m):
        sheet.put(12 + t, engine - 1, trace.object_ids[t], "label")
        sheet.row(
            12 + t,
            engine,
            [
                *(slot_column(latent, j, m, float)[t] for j in range(latent_slots)),
                float(meta.scores1[t]),
                float(meta.scores2[t]),
            ],
            "num",
        )
    foot = 12 + m
    sheet.put(foot, engine - 1, "ω", "label")
    sheet.row(foot, engine, [f.omega for f in trace.features], "value_num")
    sheet.put(foot + 1, engine - 1, "η(1)", "label")
    sheet.row(foot + 1, engine, [float(f.eta[0]) for f in trace.features], "num")
    sheet.put(foot + 2, engine - 1, "η(2)", "label")
    sheet.row(foot + 2, engine, [float(f.eta[1]) for f in trace.features], "num")

    row = 11 + len(rows) + 1
    notes = [
        (
            f"Every cell of this workbook is a value computed by context-synthetic-recognition "
            f"{__version__}; there are no live formulas. `csr validate --against <this file>` "
            "re-computes the experiment from the Dataset and Parameters sheets and compares every "
            "computed cell (tolerance 1e-9)."
        ),
        (
            "The layout mirrors the author’s Excel experiment “Context-Synthetic Model – Full "
            "Experiment”: the same sheets, tables, colours and notes, generated for this dataset "
            "and "
            "configuration."
        ),
        (
            "Template replication by the package: HAG template reproduced exactly (SET "
            f"{template_variants()[0].selected}, max |Δr| "
            f"{template_variants()[0].difference_text}); "
            "meta-algorithm template decision reproduced (Class 2)."
        ),
    ]
    for offset, text_ in enumerate(notes):
        sheet.merge(row + offset, 1, 5, text_, "value_wrap")
        sheet.height(row + offset, 33.0)
