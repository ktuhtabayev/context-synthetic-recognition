"""The sheets *Switches*, *Model Properties* and *Validation*."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

import numpy as np

from context_synthetic_recognition import __version__
from context_synthetic_recognition.config.models import ClassMeans
from context_synthetic_recognition.config.presets import active_deviations, deviation
from context_synthetic_recognition.core.additional_features.agglomerative import (
    AgglomerativeResult,
    agglomerative,
)
from context_synthetic_recognition.core.properties import equal_neighbour_sets
from context_synthetic_recognition.export.excel.common import (
    DEVIATIONS,
    PROPERTIES,
    SENSITIVITY,
    VALIDATION,
    Book,
)
from context_synthetic_recognition.export.excel.sheets_meta import slot_column
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
    additional_name,
    feature_name,
    object_name,
)


def whole_percent(value: float) -> str:
    """``0.7`` → ``70%`` — accuracies inside sentences."""
    return f"{100 * value:.0f}%"


def switch_effect(view: RunView) -> str:
    """The effect of the agglomerative algorithm's two switches on this run, in one sentence."""
    variants = view.sensitivity
    if not variants:
        settings = view.agglomerative.settings
        found = training_correctness(view)[0]
        return (
            f"this run: class means = {settings.class_means.value}, majorizing passes = "
            f"{settings.majorizing_passes}: TUPLAM {view.agglomerative.label}, resubstitution "
            f"{found} (the other settings were not evaluated)"
        )
    article = next(v for v in variants if v.label == "article")
    others = [v for v in variants if v.label != "article"]
    return (
        f"the article (final means, one pass): TUPLAM {article.tuplam}, resubstitution "
        f"{whole_percent(article.resubstitution)}, LOO {whole_percent(article.leave_one_out)}; "
        + "; ".join(
            f"{v.label}: TUPLAM {v.tuplam}, resubstitution {whole_percent(v.resubstitution)}, "
            f"LOO {whole_percent(v.leave_one_out)}"
            for v in others
        )
    )


# ---------------------------------------------------------------- Switches

EXAMPLE_ROWS = 10
"""Rows of each worked example (they stay above the effect table at row 52)."""
EFFECT_ROW = 52
"""Header row of the effect table: rows 53–56 hold the four settings, as in the workbook."""


def _with(view: RunView, class_means: ClassMeans, passes: int) -> AgglomerativeResult:
    """The run's agglomerative algorithm with the two switches set (the same inputs and α, δ, ϰ)."""
    config = view.config.additional_features.model_copy(
        update={"class_means": class_means, "majorizing_passes": passes}
    )
    trace = view.trace
    return agglomerative(trace.contributions, trace.informativeness, trace.class_index, config)


def deviations(book: Book) -> None:
    """Sheet *Switches*: the settings that deviate from the article and what each one changes."""
    view = book.view
    config = view.config
    passes = config.additional_features.majorizing_passes
    final = _with(view, ClassMeans.FINAL, passes)
    running = _with(view, ClassMeans.RUNNING, passes)
    sheet = book.sheet(DEVIATIONS, "deviations")
    sheet.widths({1: 46, (2, 9): 16})
    sheet.title_bar(
        "Switches — the settings that deviate from the article, and what each one changes",
        9,
        height=22.0,
    )

    def paragraph(
        row: int, text: str, style: str = "text_wrap", height: float | None = None
    ) -> None:
        sheet.merge(row, 1, 9, text, style)
        if height is not None:
            sheet.height(row, height)

    active = active_deviations(config)
    paragraph(
        3,
        "The article is followed by default. Every setting that deviates from it is a "
        "configuration switch, flagged with ⚠ here, on the Parameters sheet and in the "
        "configuration file. This run: "
        + (
            "; ".join(f"⚠ {d.field} = {d.alternative_value} ({d.adr})" for d in active)
            if active
            else "every switch at the article's value"
        )
        + ".",
        height=33.0,
    )

    # ---- class means
    paragraph(5, f"⚠ {deviation('class_means').statement}", "warn1_bar")
    paragraph(
        6,
        "The article, Step 3 of the agglomerative algorithm:  M₁ = Σ(Sₜ∈K1) bₜ / |K1|,  "
        "M₂ = Σ(Sₜ∈K2) bₜ / |K2| over all objects; then θ = Σ |bₜ − M(own class)| and "
        "γ = Σ |bₜ − M(other class)|. The alternative (additional_features.class_means = running) "
        "measures row t from the running class sum up to row t divided by |Kᵢ|.",
        height=48.0,
    )
    if not final.iterations or not running.iterations:  # a single synthetic feature: no Step 3
        paragraph(8, "Ψ(r) has one feature: Step 3 has no candidate, the switch changes nothing.")
        _effect_table(sheet, view, paragraph)
        return
    first_final, first_running = final.iterations[0], running.iterations[0]
    shown = first_final.candidates[:EXAMPLE_ROWS]
    sheet.put(
        8,
        1,
        f"Iteration 1 on this run's data (R = η of {feature_name(final.first)})",
        "header_left",
    )
    sheet.row(
        8,
        2,
        ["θ final", "γ final", "θ/γ final", "θ running", "γ running", "θ/γ running"],
        "header",
    )
    for position, u in enumerate(shown):
        sheet.put(9 + position, 1, feature_name(int(u)), "label")
        sheet.row(
            9 + position,
            2,
            [
                float(first_final.theta[position]),
                float(first_final.gamma[position]),
                float(first_final.ratio[position]),
                float(first_running.theta[position]),
                float(first_running.gamma[position]),
                float(first_running.ratio[position]),
            ],
            [
                "num",
                "num",
                "good_num" if int(u) == first_final.q else "num",
                "warn1_num",
                "warn1_num",
                "good_num" if int(u) == first_running.q else "warn1_num",
            ],
        )
    row = 9 + EXAMPLE_ROWS
    paragraph(
        row,
        "Selected q (green):  final means → "
        + (DASH if first_final.q is None else f"{feature_name(first_final.q)}")
        + f" with θ/γ = {first_final.crit:.4f};  running means → "
        + (DASH if first_running.q is None else f"{feature_name(first_running.q)}")
        + f" with θ/γ = {first_running.crit:.4f}"
        + (
            f"  (the first {EXAMPLE_ROWS} of {first_final.candidates.size} candidates are shown)."
            if first_final.candidates.size > EXAMPLE_ROWS
            else "."
        ),
        "warn_note",
        21.0,
    )

    # ---- majorizing passes
    row = 21
    paragraph(row, f"⚠ {deviation('majorizing_passes').statement}", "warn2_bar")
    paragraph(
        row + 1,
        "The article, Step 4:  R(Sₜ) ← R(Sₜ) + η_q(a_tq);  then R(Sₜ) ← R(Sₜ) + α·φ(−R(Sₜ)) for "
        "Sₜ ∈ K1 (− α·φ(−R) for K2) — one application. The alternative "
        "(additional_features.majorizing_passes = 2) applies φ once more to the result.",
        height=33.0,
    )
    once = _with(view, config.additional_features.class_means, 1)
    twice = _with(view, config.additional_features.class_means, 2)
    d_once = once.iterations[0].additional
    d_twice = twice.iterations[0].additional
    sheet.put(row + 3, 1, f"{additional_name(0)} on this run's data", "header_left")
    sheet.row(row + 3, 2, ["once (article)", "twice", "Class"], "header")
    if d_once is not None and d_twice is not None:
        for t in range(min(EXAMPLE_ROWS, view.trace.m)):
            sheet.put(row + 4 + t, 1, object_name(t), "label")
            sheet.row(
                row + 4 + t,
                2,
                [float(d_once[t]), float(d_twice[t]), view.labels[int(view.trace.class_index[t])]],
                ["num", "warn2_num", "class_"],
            )
    else:
        sheet.put(row + 4, 1, "no feature was added in iteration 1", "note")

    # ---- the other switches
    row = 37
    paragraph(row, "Other settings that deviate from the article or the article leaves open", "bar")
    originals = config.synthetic_features.original_features or "none"
    lines = [
        (
            f"⚠ {deviation('keep_blocked_cover').statement} (rips.keep_blocked_cover = "
            f"{config.rips.keep_blocked_cover})."
        ),
        f"⚠ {deviation('training').statement} (rips.training = {config.rips.training}).",
        (
            "Admissible k: the article's odd k = 3 … min(|K1|, |K2|) (admissible_k = "
            f"min-class); this run: {config.admissible_k.name}."
        ),
        (
            "Synthetic features {0, 1, 2}: features where a training object takes 0 "
            "(f_k(μ) = 0.5) are left out when synthetic_features.zero_value is on; this run: "
            f"{config.synthetic_features.zero_value}."
        ),
        (
            "Original nominal features combined with the synthetic ones "
            f"(synthetic_features.original_features): {originals}."
        ),
        (
            "The majorizing function φ is not defined in the article; this run: "
            f"{config.additional_features.majorizing_function.name}."
        ),
    ]
    for offset, text in enumerate(lines):
        paragraph(row + 1 + offset, text, height=21.0)

    _effect_table(sheet, view, paragraph)


def _effect_table(sheet: SheetWriter, view: RunView, paragraph: Callable[..., None]) -> None:
    """Rows 51–58: the four settings of the two switches (rows 53–56, as in the workbook)."""
    paragraph(EFFECT_ROW - 1, "Effect of the two switches of the agglomerative algorithm", "bar")
    sheet.put(EFFECT_ROW, 1, f"This run (Ψ(r), {view.trace.r} features)", "header_left")
    sheet.row(
        EFFECT_ROW,
        2,
        [
            "Class means",
            "majorizing passes",
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
            sheet.put(EFFECT_ROW + 1 + offset, 1, v.label, "caption")
            sheet.row(
                EFFECT_ROW + 1 + offset,
                2,
                [
                    v.class_means.value,
                    v.majorizing_passes,
                    v.tuplam,
                    v.resubstitution,
                    v.leave_one_out,
                    v.auc_resubstitution,
                    v.auc_leave_one_out,
                ],
                ["value_left", "value_left", "good" if v.label == "article" else "value_left"]
                + ["num"] * 4,
            )
    else:
        _own_setting(sheet, view, EFFECT_ROW + 1)
        sheet.put(
            EFFECT_ROW + 2,
            1,
            "The other three settings were not evaluated for this run "
            "(csr run --sensitivity, csr export --sensitivity).",
            "note",
        )
    paragraph(
        EFFECT_ROW + 6,
        "Neither switch touches formulas (1)–(6) or the synthetic features: they change only the "
        "agglomerative algorithm (which features enter TUPLAM and the additional features d), "
        "and through them the meta-algorithm and its evaluation.",
        height=21.0,
    )


def _own_setting(sheet: SheetWriter, view: RunView, row: int) -> None:
    """The run's own switch setting — when the other three were not evaluated."""
    result = view.result
    settings = view.agglomerative.settings
    resubstitution = view.protocol("resubstitution")
    loo = view.protocol(LEAVE_ONE_OUT)
    truth = view.trace.class_index + 1
    training = float(np.count_nonzero(view.training.decisions == truth)) / view.trace.m
    sheet.put(row, 1, "this run", "caption")
    sheet.row(
        row,
        2,
        [
            settings.class_means.value,
            settings.majorizing_passes,
            view.agglomerative.label,
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
    ks = trace.admissible_k.ks
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
    trace, agglomerative = view.trace, view.agglomerative
    m, r = trace.m, trace.r
    additional_slots = view.slots - 1
    settings = agglomerative.settings
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
            (engine, engine + max(r, additional_slots + 2, 6) - 1): 12,
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
        "⚠ Settings of the agglomerative algorithm that deviate from the article (the "
        "Excel experiment uses them; the default follows the article)",
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
        5,
        1,
        5,
        "2.  In STEP 4 the majorizing function is applied twice instead of once.",
        "warn_text",
    )
    effect = switch_effect(view)
    sheet.merge(
        6,
        1,
        5,
        f"Effect here: {effect}. Both are configuration switches shown on the Parameters sheet"
        + (f" — see {SENSITIVITY}." if view.sensitivity else "."),
        "warn_text",
    )
    article = settings.class_means is ClassMeans.FINAL and settings.majorizing_passes == 1
    sheet.merge(
        8,
        1,
        5,
        "Both switches at the article's setting (final class means, one majorizing pass)"
        if article
        else f"⚠ Switches deviate from the article: class means = {settings.class_means.value}, "
        f"majorizing passes = {settings.majorizing_passes} (the article: final, 1)",
        "status",
    )
    sheet.row(
        10,
        1,
        ["Check", "Package (computed)", "Workbook (sheet value)", "Difference", "Pass"],
        "header",
    )

    rows: list[tuple[str, Any, str]] = [
        ("Admissible k", trace.admissible_k.label, "value_left"),
        (
            "k_max",
            trace.admissible_k.k_max
            if trace.admissible_k.k_max is not None
            else max(trace.admissible_k.ks),
            "value",
        ),
        ("r = |Ψ(r)|", r, "value"),
        *((f"ω of {f.name}  (4)", f.omega, "value_num") for f in trace.features),
        ("Σ |η(1), η(2) − package| over Ψ(r)  (6)", 0.0, "value_num"),
        ("TUPLAM (hierarchical agglomerative algorithm output)", agglomerative.label, "value_left"),
        *(
            (
                f"crit after iteration {j + 1}",
                agglomerative.crit[j] if j < len(agglomerative.crit) else DASH,
                "value_num",
            )
            for j in range(additional_slots)
        ),
        *(
            (f"Σ |{additional_name(j)} − package| over the {m} objects", 0.0, "value_num")
            for j in range(additional_slots)
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
    sheet.bar(10, engine - 1, max(additional_slots + 3, r + 1), "Package values (as computed)")
    sheet.put(11, engine - 1, "№", "header")
    sheet.row(
        11, engine, [additional_name(j) for j in range(additional_slots)], "additional_header"
    )
    sheet.row(11, engine + additional_slots, ["score₁", "score₂"], "header")
    additional = view.model.meta_table.additional
    meta = view.training.meta
    for t in range(m):
        sheet.put(12 + t, engine - 1, trace.object_ids[t], "label")
        sheet.row(
            12 + t,
            engine,
            [
                *(slot_column(additional, j, m, float)[t] for j in range(additional_slots)),
                float(meta.scores1[t]),
                float(meta.scores2[t]),
            ],
            "num",
        )
    foot = 12 + m
    sheet.put(foot, engine - 1, "ω", "label")
    sheet.row(foot, engine, [f.omega for f in trace.features], "value_num")
    sheet.put(foot + 1, engine - 1, "η(1)", "label")
    sheet.row(foot + 1, engine, [f.eta_at(1) for f in trace.features], "num")
    sheet.put(foot + 2, engine - 1, "η(2)", "label")
    sheet.row(foot + 2, engine, [f.eta_at(2) for f in trace.features], "num")

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
            "The two check workbooks — the additional features and the meta-algorithm on their "
            "own data — are validated the same way (`csr validate --against`)."
        ),
    ]
    for offset, text_ in enumerate(notes):
        sheet.merge(row + offset, 1, 5, text_, "value_wrap")
        sheet.height(row + offset, 33.0)
