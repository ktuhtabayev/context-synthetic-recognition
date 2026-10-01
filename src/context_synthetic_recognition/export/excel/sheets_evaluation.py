"""The evaluation sheets.

Margins, accuracy, confusion matrices, precision/recall/F1, ROC and AUC, the folds of every
hold-out protocol and the sensitivity to the two switches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from context_synthetic_recognition.core.meta import K1_DECISION
from context_synthetic_recognition.evaluation.margins import Margin
from context_synthetic_recognition.evaluation.protocols import (
    CS_MODEL,
    MethodPredictions,
    ProtocolResult,
)
from context_synthetic_recognition.export.excel.common import (
    ACCURACY,
    CONFUSION,
    MARGINS,
    PRF1,
    ROC,
    SENSITIVITY,
    Book,
)
from context_synthetic_recognition.export.excel.template_data import template_variants
from context_synthetic_recognition.export.excel.writer import column_letter, reference
from context_synthetic_recognition.export.view import LEAVE_ONE_OUT, RunView
from context_synthetic_recognition.export.wording import (
    decision_label,
    protocol_title,
    tuple_text,
)
from context_synthetic_recognition.notation import DASH, join_values, latent_name

META_ALGORITHM = "Meta-algorithm"
"""How the workbook names the CS-model's decisions in the evaluation tables."""
KNN_PREFIX = "k-NN vote "


def method_name(method: MethodPredictions) -> str:
    """A method as the workbook names it: the CS-model is the *Meta-algorithm*."""
    return META_ALGORITHM if method.method == CS_MODEL else method.method


def evaluated(view: RunView) -> list[tuple[ProtocolResult, MethodPredictions]]:
    """The rows of the method tables.

    The CS-model under every protocol, then the baselines of the hold-out protocols (a baseline's
    resubstitution is not a result of the article).
    """
    rows = [(p, p.predictions) for p in view.result.protocols]
    rows += [(p, b) for p in view.cross_validations for b in p.baselines]
    return rows


def _opposite(view: RunView) -> tuple[Any, Any]:
    """The labels of the positive and the negative class."""
    classes = view.trace.classes
    positive = view.result.positive
    return classes[positive - 1], classes[2 - positive]


# ---------------------------------------------------------------- Margin Analysis


def margins(book: Book) -> None:
    """Sheet *Margin Analysis*: every latent feature's margin with and without the majorizer."""
    view = book.view
    trace = view.trace
    m = trace.m
    slots = view.slots - 1
    ids, labels, classes = trace.object_ids, view.labels, trace.classes
    analysis = view.result.margins
    sheet = book.sheet(MARGINS, "evaluation")
    for j in range(max(slots, 1)):
        sheet.widths({1 + 6 * j: 17, (2 + 6 * j, 6 + 6 * j): 11})
    width = max(6 * slots, 6)
    sheet.title_bar(
        "Margin Analysis — margins of the latent (additional) training features with and without "
        "majorizer functions",
        width,
    )
    k1, k2 = classes
    rules = [
        (
            f"Classification rule:  ŷ = {k1} if d_ip > b,  ŷ = {k2} if d_ip ≤ b    "
            f"(class {k1} = positive +1, class {k2} = negative −1)"
        ),
        "Decision boundary (centre of the margin):  b = (min over K1 of d + max over K2 of d) / 2",
        (
            "Margin width = min over K1 of d − max over K2 of d   "
            "(> 0: the classes are separated on this feature)"
        ),
        "Object margin:  mᵢ = yᵢ · (d_ip − b),  yᵢ ∈ {−1, +1}   (> 0: object on its correct side)",
    ]
    for offset, rule in enumerate(rules, start=3):
        sheet.merge(offset, 1, width - 1, rule, "text")

    def table(top: int, found: tuple[Margin, ...], bar: str, suffix: str) -> None:
        sheet.bar(top, 1, width - 1, bar)
        for j in range(slots):
            left = 1 + 6 * j
            sheet.row(
                top + 1,
                left,
                ["№", f"{latent_name(j)}{suffix}", "Class", "margin mᵢ", "ŷ"],
                ["header", "latent_header", "header", "header", "header"],
            )
            margin = found[j] if j < len(found) else None
            for t in range(m):
                values: list[Any] = [ids[t], DASH, labels[t], DASH, DASH]
                if margin is not None:
                    values = [
                        ids[t],
                        float(margin.values[t]),
                        labels[t],
                        float(margin.object_margins[t]),
                        decision_label(int(margin.predictions[t]), classes),
                    ]
                sheet.row(
                    top + 2 + t, left, values, ["label", "latent_num", "class_", "num", "integer"]
                )
            foot = top + m + 3
            summary: list[tuple[str, Any, Any]] = [
                ("b (boundary)", DASH if margin is None else margin.boundary, None),
                (
                    "left: max K2",
                    DASH if margin is None else margin.left,
                    None if margin is None else ids[margin.left_object],
                ),
                (
                    "right: min K1",
                    DASH if margin is None else margin.right,
                    None if margin is None else ids[margin.right_object],
                ),
                ("margin width", DASH if margin is None else margin.width, None),
                ("ŷ correct (count)", DASH if margin is None else margin.correct, None),
            ]
            for offset, (key, value, name) in enumerate(summary):
                sheet.put(foot + offset, left, key, "key")
                sheet.put(foot + offset, left + 1, value, "value" if offset == 4 else "value_num")
                if name is not None:
                    sheet.put(foot + offset, left + 2, name, "value_left")

    with_top, without_top = 8, m + 17
    table(
        with_top,
        analysis.with_majorizer,
        "Margin with majorizer functions — latent features r_j of Step 9 "
        "(Dataset for Meta-algorithm)",
        "",
    )
    table(
        without_top,
        analysis.without_majorizer,
        "Margin without majorizer functions — generalized estimate R(Sₜ) = Σ ηᵤ(a_tu) over the "
        "same TUPLAM features, no majorizing function",
        " (no ϕ)",
    )
    for top in (with_top, without_top):
        for j in range(slots):
            column = 4 + 6 * j
            first = f"{column_letter(column)}{top + 2}"
            sheet.highlight_formula(
                reference(top + 2, column, m, 1), f"AND(ISNUMBER({first}),{first}<=0)", "bad"
            )

    top = 2 * m + 26
    sheet.bar(top, 1, slots + 1, "Comparing margin widths with and without majorizer")
    sheet.put(top + 1, 1, "", "header")
    sheet.row(top + 1, 2, [latent_name(j) for j in range(slots)], "latent_header")

    def widths_of(found: tuple[Margin, ...]) -> list[Any]:
        return [found[j].width if j < len(found) else DASH for j in range(slots)]

    gains = [analysis.gains[j] if j < len(analysis.gains) else DASH for j in range(slots)]
    sheet.put(top + 2, 1, "With", "caption")
    sheet.row(top + 2, 2, widths_of(analysis.with_majorizer), "value_num")
    sheet.put(top + 3, 1, "Without", "caption")
    sheet.row(top + 3, 2, widths_of(analysis.without_majorizer), "value_num")
    sheet.put(top + 4, 1, "Gain (with − without)", "caption")
    sheet.row(top + 4, 2, gains, "good_num")
    sheet.notes(
        top + 6,
        [
            (
                "The majorizer pushes K1 objects up and K2 objects down by α·σ(−b), so a positive "
                "gain means the regularisation widened the margin of that latent feature."
            )
        ],
    )


# ---------------------------------------------------------------- Accuracy, P/R/F1


def accuracy(book: Book) -> None:
    """Sheet *Accuracy*: confusion counts, accuracy and coverage of every method."""
    view = book.view
    positive, negative = _opposite(view)
    sheet = book.sheet(ACCURACY, "evaluation")
    sheet.widths({1: 30, 2: 24, (3, 11): 12})
    sheet.title_bar(
        "Accuracy  =  (TP + TN) / (TP + TN + FP + FN)   ·   "
        f"positive class = {positive};  a refusal (0) counts as an error",
        11,
    )
    sheet.row(
        3,
        1,
        [
            "Method",
            "Evaluation",
            "TP",
            "TN",
            "FP",
            "FN",
            "Refusals",
            "Accuracy",
            "Coverage",
            "Accuracy on answered",
            "Correct / m",
        ],
        "header",
    )
    styles = ["caption", "caption"] + ["integer"] * 5 + ["good_num", "num", "num", "value"]
    row = 4
    for protocol, method in evaluated(view):
        metrics = view.result.metrics(method)
        answered = metrics.accuracy_answered
        sheet.row(
            row,
            1,
            [
                method_name(method),
                protocol_title(protocol.protocol),
                metrics.tp,
                metrics.tn,
                metrics.fp,
                metrics.fn,
                metrics.refusals,
                metrics.accuracy,
                metrics.coverage,
                DASH if answered is None else answered,
                metrics.label,
            ],
            styles,
        )
        row += 1
    sheet.notes(
        row + 1,
        [
            (
                f"TP = predicted {positive} and true {positive};  TN = predicted {negative} and "
                "true "
                f"{negative};  FP = predicted {positive} and true {negative};  FN = predicted "
                f"{negative} and true {positive};  refusal = predicted 0 (equal scores in Step 4)."
            ),
            (
                'Coverage = share of objects that received a class; "Accuracy on answered" ignores '
                "refusals."
            ),
        ],
    )


def confusion(book: Book) -> None:
    """Sheet *Confusion Matrix*: actual × predicted class of the CS-model under every protocol."""
    view = book.view
    classes = view.trace.classes
    protocols = view.result.protocols
    positive, negative = _opposite(view)
    sheet = book.sheet(CONFUSION, "evaluation")
    sheet.widths({(1, max(13, 7 * len(protocols) - 1)): 13})
    sheet.title_bar(
        "Confusion Matrix — actual class (rows) × predicted class (columns), refusals shown "
        "separately",
        max(7 * len(protocols) - 2, 6),
    )
    names = [f"Class {label}" for label in classes]
    for index, protocol in enumerate(protocols):
        left = 1 + 7 * index
        matrix = view.result.metrics(protocol.predictions).matrix
        sheet.bar(3, left, 6, f"{META_ALGORITHM} · {protocol_title(protocol.protocol)}")
        sheet.row(
            4, left, ["Actual \\ Predicted", *names, "Refusal (0)", "Total", "Recall"], "header"
        )
        for c in range(2):
            total = int(matrix[c].sum())
            cells = [int(v) for v in matrix[c]]
            sheet.put(5 + c, left, names[c], "label")
            sheet.row(
                5 + c,
                left + 1,
                [*cells, total, cells[c] / total if total else DASH],
                ["good" if p == c else "integer" for p in range(3)] + ["value", "num"],
            )
        totals = matrix.sum(axis=0)
        sheet.put(7, left, "Total", "label")
        sheet.row(7, left + 1, [*(int(v) for v in totals), int(matrix.sum())], "value")
        sheet.put(8, left, "Precision", "label")
        sheet.row(
            8,
            left + 1,
            [int(matrix[c, c]) / int(totals[c]) if totals[c] else DASH for c in range(2)],
            "num",
        )
    sheet.notes(
        10,
        [
            (
                f"Diagonal (green) = correct decisions. Class {positive} is the positive class: "
                f"TP = [Class {positive}, Class {positive}], FN = [Class {positive}, Class "
                f"{negative}], "
                f"FP = [Class {negative}, Class {positive}], TN = [Class {negative}, Class "
                f"{negative}]."
            )
        ],
    )


def precision_recall(book: Book) -> None:
    """Sheet *Precision, Recall, F1 Score*: per class and macro average for every method."""
    view = book.view
    classes = view.trace.classes
    sheet = book.sheet(PRF1, "evaluation")
    sheet.widths({1: 30, 2: 24, (3, 8): 13})
    sheet.title_bar(
        "Precision, Recall, F1 Score — per class and macro average "
        "(refusals count as not predicted)",
        9,
    )
    sheet.row(
        3,
        1,
        [
            "Method",
            "Evaluation",
            "Class",
            "Predicted as class",
            "True positives",
            "Precision",
            "Recall",
            "F1",
        ],
        "header",
    )
    row = 4
    for protocol, method in evaluated(view):
        metrics = view.result.metrics(method)
        name, how = method_name(method), protocol_title(protocol.protocol)
        for scores in metrics.classes:
            sheet.row(
                row,
                1,
                [
                    name,
                    how,
                    classes[scores.code - 1],
                    scores.predicted,
                    scores.correct,
                    scores.precision,
                    scores.recall,
                    scores.f1,
                ],
                ["caption", "caption", "class_", "integer", "integer", "num", "num", "good_num"],
            )
            row += 1
        sheet.row(row, 1, [name, how, "macro"], ["caption", "caption", "key"])
        sheet.row(
            row, 6, [metrics.macro_precision, metrics.macro_recall, metrics.macro_f1], "value_num"
        )
        row += 1
    sheet.notes(
        row + 1,
        [
            (
                "Precision = TP / predicted as the class;  Recall = TP / actual members;  "
                "F1 = 2·P·R / (P + R);  0 when undefined."
            )
        ],
    )


# ---------------------------------------------------------------- ROC Curve & AUC


def roc(book: Book) -> None:
    """Sheet *ROC Curve & AUC*: scores, pairs won, AUC and the ROC points under every protocol."""
    view = book.view
    trace = view.trace
    classes, ids = trace.classes, trace.object_ids
    positive = view.result.positive
    decimals = view.config.evaluation.score_decimals
    protocols = view.result.protocols
    first, second = ("K1", "K2") if positive == K1_DECISION else ("K2", "K1")
    sheet = book.sheet(ROC, "evaluation")
    sheet.widths({(1, max(16, 8 * len(protocols))): 12})
    sheet.title_bar(
        "ROC Curve & AUC — score = score₁ − score₂ of the meta-algorithm (Step 4), "
        f"class {classes[positive - 1]} positive",
        max(8 * len(protocols) - 1, 7),
    )
    longest = 0
    for index, protocol in enumerate(protocols):
        left = 1 + 8 * index
        predictions = protocol.predictions
        curve = predictions.roc(positive, decimals)
        order = np.lexsort((predictions.objects, predictions.repeats))
        entries = int(order.size)
        longest = max(longest, entries)
        repeated = bool(np.unique(predictions.repeats).size > 1)
        title = protocol_title(protocol.protocol)
        sheet.bar(3, left, 7, title[:1].upper() + title[1:])
        sheet.row(
            4,
            left,
            ["№", "Score", "Class", f"Pairs won ({first} object vs {second})"],
            "header",
        )
        for offset, e in enumerate(order):
            name = ids[int(predictions.objects[e])]
            if repeated:
                name = f"{name} · {int(predictions.repeats[e]) + 1}"
            sheet.row(
                5 + offset,
                left,
                [
                    name,
                    round(float(predictions.scores[e]), decimals),
                    classes[int(predictions.truth[e]) - 1],
                    float(curve.pairs_won[e]),
                ],
                ["label", "num", "class_", "num"],
            )
        foot = 5 + entries
        sheet.merge(foot, left, 3, "AUC", "key")
        sheet.put(foot, left + 3, curve.auc, "good_num")
        sheet.row(foot + 2, left, ["Threshold", "TPR (recall)", "FPR"], "header")
        sheet.row(foot + 3, left, ["+∞", 0, 0], ["value", "num", "num"])
        for i in range(1, curve.thresholds.size):
            sheet.row(
                foot + 3 + i,
                left,
                [float(curve.thresholds[i]), float(curve.tpr[i]), float(curve.fpr[i])],
                "num",
            )
    sheet.notes(
        2 * longest + 10,
        [
            (
                f"AUC = Mann–Whitney statistic: share of ({first}, {second}) pairs in which the "
                f"{first} object has the larger score (ties count ½). The ROC table lists TPR and "
                "FPR "
                "for every score threshold."
            )
        ],
    )


# ---------------------------------------------------------------- the folds of a protocol


def _baseline_header(method: str) -> str:
    """``k-NN vote ρ_I, k = 3`` → ``ρ_I · k = 3``."""
    if method.startswith(KNN_PREFIX):
        return method.removeprefix(KNN_PREFIX).replace(", ", " · ")
    return method


def folds(book: Book, protocol: ProtocolResult) -> None:
    """The sheet of a hold-out protocol (*Leave-One-Out*, …): one row per held-out object."""
    view = book.view
    trace = view.trace
    m, classes, ids = trace.m, trace.classes, trace.object_ids
    decimals = view.config.evaluation.score_decimals
    baselines = protocol.baselines
    loo = protocol.protocol == LEAVE_ONE_OUT
    name = book.plan.protocols[protocol.protocol]
    sheet = book.sheet(name, "evaluation")
    sheet.widths({(1, 6): 11, 7: 34, 8: 24, (9, 12): 12})
    if baselines:
        sheet.widths({(13, 12 + len(baselines)): 12})
    how = protocol_title(protocol.protocol)
    sheet.title_bar(
        f"{name} — generalization correctness (Definition 3): the whole pipeline re-fitted "
        + ("without the held-out object" if loo else "without the held-out objects of every fold"),
        12 + max(len(baselines), 0),
    )
    part = f"the other {m - 1} objects" if loo else "the training part of the fold"
    sheet.bar(
        3,
        1,
        12,
        "CS-model: each fold re-computes scaling, k range, Ψ(r), ω, η, HAG and the "
        f"meta-algorithm on {part} (values computed by the package)",
    )
    if baselines:
        knn = all(b.method.startswith(KNN_PREFIX) for b in baselines)
        sheet.bar(
            3,
            13,
            len(baselines),
            f"Baselines · plain k-NN majority vote (same operators), {how}"
            if knn
            else f"Baselines · the same folds, {how}",
        )
    sheet.row(
        4,
        1,
        [
            "Held out",
            "True class",
            "|K1| in fold",
            "|K2| in fold",
            "Permitted k",
            "r = |Ψ(r)|",
            "TUPLAM (operator·k)",
            "(a₀, …, a_p) of the held-out",
            "score₁",
            "score₂",
            "score₁ − score₂",
            "Predicted",
            *(_baseline_header(b.method) for b in baselines),
        ],
        "header",
    )
    styles = [
        "label",
        "class_",
        "value",
        "value",
        "value",
        "value",
        "value_left",
        "value_left",
        "num",
        "num",
        "num",
        "good",
    ]
    row = 5
    entry = 0
    for fold in protocol.folds:
        sizes: tuple[Any, Any] = fold.class_sizes if fold.class_sizes else (DASH, DASH)
        for position, i in enumerate(fold.split.test):
            description = (
                tuple_text([int(v) for v in fold.descriptions[position]])
                if fold.descriptions is not None
                else DASH
            )
            sheet.row(
                row,
                1,
                [
                    ids[int(i)],
                    classes[int(trace.class_index[int(i)])],
                    *sizes,
                    join_values(fold.permitted_k) if fold.permitted_k else DASH,
                    fold.r if fold.r is not None else DASH,
                    "{" + ", ".join(fold.tuplam_names or ()) + "}",
                    description,
                    float(fold.scores1[position]),
                    float(fold.scores2[position]),
                    round(float(fold.scores[position]), decimals),
                    decision_label(int(fold.decisions[position]), classes),
                ],
                styles,
            )
            sheet.row(
                row,
                13,
                [decision_label(int(b.decisions[entry]), classes) for b in baselines],
                "integer",
            )
            row += 1
            entry += 1
    entries = row - 5
    if entries:
        sheet.highlight_formula(reference(5, 12, entries, 1), "L5<>B5", "bad")
        if baselines:
            sheet.highlight_formula(reference(5, 13, entries, len(baselines)), "M5<>$B5", "bad")
    undefined = len(protocol.undefined_folds)
    notes = [
        (
            "Every fold is a complete fit of the package on its training part "
            f"({len(protocol.folds)} "
            "fold(s)); the held-out objects are classified without their class."
        ),
        (
            "The class sizes of a fold decide its permitted k (k_max = 2·min|Kᵢ| − 3 of the fold): "
            "the formula-based k range adapts per fold."
        ),
        (
            "Predicted 0 = refusal (equal scores). Red cells = wrong or refused. Baselines use the "
            "same scaling and tie rule; they are not bound to the permitted-k rule."
        ),
    ]
    if undefined:
        notes.append(
            f"⚠ {undefined} fold(s) left the model undefined (no permitted k or no informative "
            "feature): their objects are refused (ADR-005)."
        )
    sheet.notes(row + 1, notes)


# ---------------------------------------------------------------- Sensitivity (Switches)


def sensitivity(book: Book) -> None:
    """Sheet *Sensitivity (Switches)*: the four settings of the two template/article switches."""
    view = book.view
    variants = view.sensitivity or ()
    sheet = book.sheet(SENSITIVITY, "evaluation")
    sheet.widths({(1, 2): 14, 3: 26, 4: 30, 5: 36, (6, 9): 14})
    sheet.title_bar(
        "Sensitivity — the two calculation switches (template cells vs article) and their effect "
        "on the results",
        9,
    )
    sheet.bar(
        3,
        1,
        9,
        "⚠ Two template calculations differ from the article and from the template’s own formula "
        "boxes",
    )
    sheet.merge(
        4,
        1,
        9,
        "1.  θ and γ are measured from running partial class means instead of the final class "
        "means M₁ and M₂ (template cells: |bₜ − M| uses the partial mean up to row t).",
        "warn_text",
    )
    sheet.merge(
        5,
        1,
        9,
        "2.  In STEP 4 the majorizer is applied twice instead of once (the template applies it "
        "again to the already-majorized bₜ of STEP 3).",
        "warn_text",
    )
    sheet.put(
        6,
        1,
        "Both are switches of the configuration, shown on the Parameters sheet (class centres "
        "1/2, STEP 4 passes 2/1). The default reproduces the templates; the table shows all four "
        "combinations, each a complete fit and evaluation by the package.",
        "note",
    )
    sheet.row(
        8,
        1,
        [
            "Class centres",
            "STEP 4 passes",
            "Setting",
            "TUPLAM",
            "crit per iteration",
            "Resubstitution accuracy",
            "LOO accuracy",
            "AUC resub.",
            "AUC LOO",
        ],
        "header",
    )
    for offset, v in enumerate(variants):
        template = offset == 0
        sheet.row(
            9 + offset,
            1,
            [
                f"{1 if v.centres.value == 'running' else 2} · {v.centres.value}",
                v.step4_passes,
                v.label,
                v.tuplam,
                ", ".join(f"{c:.4f}" for c in v.crit),
                v.resubstitution,
                v.leave_one_out,
                v.auc_resubstitution,
                v.auc_leave_one_out,
            ],
            ["value", "value", "good" if template else "value_left", "value_left", "value_left"]
            + ["num"] * 4,
        )

    top = 14
    sheet.bar(
        top,
        1,
        9,
        "Replication of the templates by the package (the same data as the template workbooks)",
    )
    sheet.row(
        top + 1,
        1,
        [
            "Class centres",
            "STEP 4 passes",
            "Setting",
            "SET / TUPLAM (template features)",
            "max |r − template r|",
        ],
        "header",
    )
    for offset, t in enumerate(template_variants()):
        sheet.row(
            top + 2 + offset,
            1,
            [
                f"{1 if t.centres.value == 'running' else 2} · {t.centres.value}",
                t.step4_passes,
                t.label,
                t.selected,
                t.difference_text,
            ],
            ["value", "value", "good" if offset == 0 else "value_left", "value_left", "value_left"],
        )
    sheet.merge(
        top + 6,
        1,
        9,
        "Meta-algorithm template: the package reproduces its decision for the template's new "
        "object — B1(a₄) = ∅, B2(a₄) = {S₉, S₁₀}  ⇒  Class 2  (template: Class 2)",
        "value_left",
    )


__all__ = [
    "accuracy",
    "confusion",
    "evaluated",
    "folds",
    "margins",
    "method_name",
    "precision_recall",
    "roc",
    "sensitivity",
]
