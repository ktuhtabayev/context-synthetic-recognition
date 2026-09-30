"""The experiment workbook's map, the evaluation sheets.

*Margin Analysis*, *Accuracy*, *Confusion Matrix*, *Precision, Recall, F1 Score*, *ROC Curve & AUC*,
*Leave-One-Out*, *Sensitivity (Switches)* (the experiment's four switch settings), *Model
Properties* and *Validation*. The template-replication rows of *Sensitivity (Switches)* and the
template tables of *Template Deviations* need the template workbooks; they are checked by
validating those workbooks themselves.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from context_synthetic_recognition.core.arrays import IntArray
from context_synthetic_recognition.core.properties import equal_neighbour_sets
from context_synthetic_recognition.evaluation.margins import Margin
from context_synthetic_recognition.evaluation.metrics import (
    ClassificationMetrics,
    classification_metrics,
)
from context_synthetic_recognition.evaluation.roc import roc_curve
from context_synthetic_recognition.notation import DASH, join_values, object_name, synthetic_name
from context_synthetic_recognition.services.validation.checks import (
    Cell,
    Check,
    Grid,
    col,
    row,
)
from context_synthetic_recognition.services.validation.context_map import labels
from context_synthetic_recognition.services.validation.subject import (
    ExperimentCheck,
    ExperimentSubject,
    at,
)

LATENT_SLOTS = 4
"""The workbook shows r₁ … r₄."""
BASELINE_ROWS = 6
"""k-NN vote ρ, ρ_I, ρ_J with k = 3 and 5."""
POSITIVE = 1
"""The workbook's positive class (Parameters!B36)."""


def _classes(s: ExperimentSubject) -> list[Cell]:
    return labels(s.trace, s.trace.class_index)


def _truth(s: ExperimentSubject) -> IntArray:
    return s.trace.class_index + 1


def _decision_sets(s: ExperimentSubject) -> list[IntArray]:
    """Resubstitution, leave-one-out, then the six LOO baselines — the workbook's row order."""
    loo = s.evaluation.leave_one_out
    return [
        s.training.decisions,
        loo.predictions.by_object(),
        *(b.by_object() for b in loo.baselines[:BASELINE_ROWS]),
    ]


def _metrics(s: ExperimentSubject, index: int) -> ClassificationMetrics:
    return classification_metrics(_truth(s), _decision_sets(s)[index], POSITIVE)


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


# ---------------------------------------------------------------- Margin Analysis


def _margins(s: ExperimentSubject, with_majorizer: bool) -> tuple[Margin, ...]:
    analysis = s.evaluation.margins
    return analysis.with_majorizer if with_majorizer else analysis.without_majorizer


def _margin_rows(s: ExperimentSubject, j: int, with_majorizer: bool) -> Grid:
    classes = _classes(s)
    margins = _margins(s, with_majorizer)
    if j >= len(margins):
        return [[DASH, classes[t], DASH, DASH] for t in range(s.trace.m)]
    m = margins[j]
    return [
        [m.values[t], classes[t], m.object_margins[t], int(m.predictions[t])]
        for t in range(s.trace.m)
    ]


def _margin_summary(s: ExperimentSubject, j: int, with_majorizer: bool) -> Grid:
    margins = _margins(s, with_majorizer)
    if j >= len(margins):
        return [[DASH, None] for _ in range(5)]
    m = margins[j]
    return [
        [m.boundary, None],
        [m.left, object_name(m.left_object)],
        [m.right, object_name(m.right_object)],
        [m.width, None],
        [m.correct, None],
    ]


def _margin_comparison(s: ExperimentSubject) -> Grid:
    analysis = s.evaluation.margins

    def widths(margins: tuple[Margin, ...]) -> list[Cell]:
        return [margins[j].width if j < len(margins) else DASH for j in range(LATENT_SLOTS)]

    gains: list[Cell] = [
        analysis.gains[j] if j < len(analysis.gains) else DASH for j in range(LATENT_SLOTS)
    ]
    return [widths(analysis.with_majorizer), widths(analysis.without_majorizer), gains]


def _margin_grid(
    fn: Callable[[ExperimentSubject, int, bool], Grid], j: int, with_majorizer: bool
) -> Callable[[ExperimentSubject], Grid]:
    return lambda s: fn(s, j, with_majorizer)


def _margin_checks() -> list[ExperimentCheck]:
    sheet = "Margin Analysis"
    checks: list[ExperimentCheck] = []
    for j in range(LATENT_SLOTS):
        first = 1 + 6 * j
        a, b, name = col(first), col(first + 3), col(first + 1)
        for with_majorizer, top, summary in ((True, 10, 21), (False, 29, 40)):
            what = f"r{j + 1} {'with' if with_majorizer else 'without'} majorizer"
            checks += [
                Check(
                    sheet,
                    f"{a}{top}:{b}{top + 9}",
                    f"{what}: d, class, margin, ŷ",
                    _margin_grid(_margin_rows, j, with_majorizer),
                ),
                Check(
                    sheet,
                    f"{a}{summary}:{name}{summary + 4}",
                    f"{what}: b, max K2, min K1, width, correct",
                    _margin_grid(_margin_summary, j, with_majorizer),
                ),
            ]
    checks.append(Check(sheet, "B48:E50", "widths with, without, gain", _margin_comparison))
    return checks


# ---------------------------------------------------------------- Accuracy, confusion, P/R/F1


def _accuracy_row(s: ExperimentSubject, index: int) -> Grid:
    m = _metrics(s, index)
    answered = m.accuracy_answered
    return [
        [
            m.tp,
            m.tn,
            m.fp,
            m.fn,
            m.refusals,
            m.accuracy,
            m.coverage,
            DASH if answered is None else answered,
            m.label,
        ]
    ]


def _confusion(s: ExperimentSubject, index: int) -> Grid:
    m = _metrics(s, index)
    grid: Grid = []
    for c in range(2):
        total = int(m.matrix[c].sum())
        recall: Cell = m.matrix[c, c] / total if total else DASH
        grid.append([*(int(v) for v in m.matrix[c]), total, recall])
    totals = m.matrix.sum(axis=0)
    grid.append([*(int(v) for v in totals), int(m.matrix.sum()), None])
    precision: list[Cell] = [m.matrix[c, c] / totals[c] if totals[c] else DASH for c in range(2)]
    grid.append([*precision, None, None, None])
    return grid


def _prf1(s: ExperimentSubject, index: int) -> Grid:
    m = _metrics(s, index)
    grid: Grid = [[c.predicted, c.correct, c.precision, c.recall, c.f1] for c in m.classes]
    grid.append([None, None, m.macro_precision, m.macro_recall, m.macro_f1])
    return grid


def _metric_checks() -> list[ExperimentCheck]:
    checks: list[ExperimentCheck] = []
    methods = 2 + BASELINE_ROWS
    for index in range(methods):
        checks.append(
            Check(
                "Accuracy",
                f"C{4 + index}:K{4 + index}",
                f"method {index + 1}",
                at(_accuracy_row, index),
            )
        )
        top = 4 + 3 * index
        checks.append(
            Check(
                "Precision, Recall, F1 Score",
                f"D{top}:H{top + 2}",
                f"method {index + 1}: per class and macro",
                at(_prf1, index),
            )
        )
    checks += [
        Check("Confusion Matrix", "B5:F8", "resubstitution", at(_confusion, 0)),
        Check("Confusion Matrix", "I5:M8", "leave-one-out", at(_confusion, 1)),
    ]
    return checks


# ---------------------------------------------------------------- ROC & AUC


def _scores(s: ExperimentSubject, loo: bool) -> np.ndarray:
    if loo:
        predictions = s.evaluation.leave_one_out.predictions
        out = np.zeros(s.trace.m)
        out[predictions.objects] = predictions.scores
        return out
    return np.asarray(s.training.scores)


def _roc_table(s: ExperimentSubject, loo: bool) -> tuple[Grid, Grid, Grid]:
    curve = roc_curve(_scores(s, loo), _truth(s), POSITIVE, s.score_decimals)
    rounded = np.round(_scores(s, loo), s.score_decimals)
    classes = _classes(s)
    objects: Grid = [
        [
            float(rounded[t]),
            classes[t],
            DASH if np.isnan(curve.pairs_won[t]) else float(curve.pairs_won[t]),
        ]
        for t in range(s.trace.m)
    ]
    points: Grid = [["+∞", 0.0, 0.0]]
    points += [
        [float(curve.thresholds[i]), float(curve.tpr[i]), float(curve.fpr[i])]
        for i in range(1, curve.thresholds.size)
    ]
    return objects, [[curve.auc]], points


def _roc_part(loo: bool, part: int) -> Callable[[ExperimentSubject], Grid]:
    return lambda s: _roc_table(s, loo)[part]


def _roc_checks() -> list[ExperimentCheck]:
    sheet = "ROC Curve & AUC"
    checks: list[ExperimentCheck] = []
    for loo, (a, d, t) in ((False, ("B", "D", "A")), (True, ("J", "L", "I"))):
        what = "leave-one-out" if loo else "resubstitution"
        last = chr(ord(t) + 2)
        checks += [
            Check(sheet, f"{a}5:{d}14", f"{what}: score, class, pairs won", _roc_part(loo, 0)),
            Check(sheet, f"{d}15", f"{what}: AUC", _roc_part(loo, 1)),
            Check(sheet, f"{t}18:{last}28", f"{what}: ROC thresholds, TPR, FPR", _roc_part(loo, 2)),
        ]
    return checks


# ---------------------------------------------------------------- Leave-One-Out


def _loo_rows(s: ExperimentSubject) -> Grid:
    loo = s.evaluation.leave_one_out
    classes = _classes(s)
    baselines = [b.by_object() for b in loo.baselines[:BASELINE_ROWS]]
    grid: Grid = []
    for fold in loo.folds:
        i = int(fold.split.test[0])
        sizes: tuple[Cell, Cell] = fold.class_sizes if fold.class_sizes else (DASH, DASH)
        description = (
            "(" + ", ".join(str(int(v)) for v in fold.descriptions[0]) + ")"
            if fold.descriptions is not None
            else DASH
        )
        grid.append(
            [
                classes[i],
                *sizes,
                join_values(fold.permitted_k) if fold.permitted_k else DASH,
                fold.r if fold.r is not None else DASH,
                "{" + ", ".join(fold.tuplam_names or ()) + "}",
                description,
                float(fold.scores1[0]),
                float(fold.scores2[0]),
                float(fold.scores[0]),
                int(fold.decisions[0]),
                *(int(b[i]) for b in baselines),
            ]
        )
    return grid


# ---------------------------------------------------------------- Sensitivity (Switches)


def _deviation_effects(s: ExperimentSubject) -> Grid:
    """*Template Deviations* rows 53–56: the switches' effect on this experiment."""
    return [
        [
            v.label,
            v.centres.value,
            v.step4_passes,
            v.tuplam,
            v.resubstitution,
            v.leave_one_out,
            v.auc_resubstitution,
            v.auc_leave_one_out,
        ]
        for v in s.evaluation.sensitivity
    ]


def _sensitivity_rows(s: ExperimentSubject) -> Grid:
    return [
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
        ]
        for v in s.evaluation.sensitivity
    ]


# ---------------------------------------------------------------- Model Properties


def _determinacy(s: ExperimentSubject) -> Grid:
    d = s.evaluation.properties.determinacy
    k_max = s.trace.permitted_k.k_max if s.trace.permitted_k.k_max is not None else d.k_max
    return [
        [
            f"{d.neighbours} ≥ {k_max}",
            "✓ defined" if d.neighbours >= k_max else "✗ too few objects",
        ],
        [
            d.undefined_memberships,
            "✓ defined" if d.undefined_memberships == 0 else "✗ undefined values",
        ],
        [
            d.missing_sides,
            "✓ both sides exist"
            if d.missing_sides == 0
            else f"⚠ 0.5 used for {d.missing_sides} side(s)",
        ],
        [d.undefined_features, "✓ defined" if d.undefined_features == 0 else "✗ undefined values"],
        [f"{d.refusals} refusal(s)", "✓ always defined (0 = refusal)"],
    ]


def _conflicts(keys: list[str], classes: IntArray) -> list[int]:
    return [
        sum(1 for j, key in enumerate(keys) if key == keys[i] and classes[j] != classes[i])
        for i in range(len(keys))
    ]


def _helpers(s: ExperimentSubject) -> Grid:
    a = s.model.description.gradations
    tuplam_keys = [
        "(" + ", ".join(str(int(a[t, k])) if k < a.shape[1] else DASH for k in range(5)) + ")"
        for t in range(s.trace.m)
    ]
    psi_keys = ["".join(str(int(v)) for v in s.trace.values[t]) for t in range(s.trace.m)]
    y = s.trace.class_index
    k1, k2 = _conflicts(tuplam_keys, y), _conflicts(psi_keys, y)
    return [[tuplam_keys[t], k1[t], psi_keys[t], k2[t]] for t in range(s.trace.m)]


_PAIRS = ((0, 1), (0, 2), (1, 2))


def _equal_sets(s: ExperimentSubject) -> Grid:
    ks = s.trace.permitted_k.ks
    columns = [equal_neighbour_sets(s.trace, a, b, k).astype(int) for a, b in _PAIRS for k in ks]
    return [[int(c[t]) for c in columns] for t in range(s.trace.m)]


def _training_correctness(s: ExperimentSubject) -> Grid:
    resub = _metrics(s, 0)
    wrong = s.trace.m - resub.correct
    status = "✓ correct on E₀" if wrong == 0 else f"✗ not correct on {wrong} object(s)"
    return [[resub.label, status]]


def _generalization(s: ExperimentSubject) -> Grid:
    loo = _metrics(s, 1)
    return [[loo.label, f"accuracy {_percent(loo.accuracy)},  refusals {loo.refusals}"]]


def _sufficiency(s: ExperimentSubject) -> Grid:
    p = s.evaluation.properties

    def status(n: int) -> str:
        return "✓ sufficient on E₀" if n == 0 else f"✗ insufficient: {n} conflicting pair(s)"

    return [
        [p.tuplam_conflicts, status(p.tuplam_conflicts)],
        [p.psi_conflicts, status(p.psi_conflicts)],
    ]


def _equivalence(s: ExperimentSubject) -> Grid:
    ks = s.trace.permitted_k.ks
    m = s.trace.m
    counts = {
        (p.first, p.second, p.k): p.equal_sets for p in s.evaluation.properties.operator_pairs
    }
    grid: Grid = [[f"objects with equal sets, k = {k}" for k in ks]]
    for a, b in _PAIRS:
        grid.append([f"{counts[(a, b, k)]} / {m}" for k in ks])
    return grid


def _identical(s: ExperimentSubject) -> Grid:
    matrix = s.evaluation.properties.identical_features
    r = matrix.shape[0]
    grid: Grid = [[synthetic_name(u) for u in range(r)]]
    grid += [["=" if matrix[u, v] else "≠" for v in range(r)] for u in range(r)]
    return grid


def _ties(s: ExperimentSubject) -> Grid:
    ks = s.trace.permitted_k.ks
    m = s.trace.m
    ties = s.evaluation.properties.boundary_ties
    grid: Grid = [[f"objects with a tie, k = {k}" for k in ks]]
    for o in range(len(s.trace.operators)):
        grid.append(
            [f"{next(t.objects for t in ties if (t.operator, t.k) == (o, k))} / {m}" for k in ks]
        )
    return grid


def _theorem(s: ExperimentSubject) -> Grid:
    exclude = s.new_object.exclude
    inputs: list[Cell] = [
        "x, training E, neighbour classes",
        "✓ the class of S is not an input of Ψ, D or R",
    ]
    if exclude is None:
        return [
            ["no object excluded", "— (type a training object and its number to run this check)"],
            [DASH, DASH],
            inputs,
        ]
    representation = s.new_object.classification.representation
    assert representation is not None
    same = bool(np.array_equal(representation.context.values[0], s.trace.values[exclude]))
    new = int(s.new_object.classification.decisions[0])
    training = int(s.training.decisions[exclude])
    return [
        [
            "identical" if same else "different",
            "✓ same representation as in training" if same else "✗ representation differs",
        ],
        [f"{new} vs {training}", "✓ same decision" if new == training else "✗ different decision"],
        inputs,
    ]


def _property_checks() -> list[ExperimentCheck]:
    sheet = "Model Properties"
    return [
        Check(sheet, "D5:E9", "Definition 1 / Property 1", _determinacy),
        Check(sheet, "J5:M14", "description keys and conflicts", _helpers),
        Check(sheet, "P5:U14", "equal k-neighbour sets per operator pair", _equal_sets),
        Check(sheet, "D13:E13", "Definition 2", _training_correctness),
        Check(sheet, "D17:E17", "Definition 3", _generalization),
        Check(sheet, "D21:E22", "Definition 4", _sufficiency),
        Check(sheet, "D25:E28", "Definition 6: equal sets", _equivalence),
        Check(sheet, "B31:G37", "identical synthetic features", _identical),
        Check(sheet, "D40:E43", "ties at the k boundary", _ties),
        Check(sheet, "D47:E49", "Theorem", _theorem),
    ]


# ---------------------------------------------------------------- Validation


def _validation_rows(s: ExperimentSubject) -> Grid:
    trace, hag = s.trace, s.hag
    grid: Grid = []

    def text(value: str) -> None:
        grid.append([value, value, DASH, "✓"])

    def number(value: float) -> None:
        grid.append([value, value, 0.0, "✓"])

    text(trace.permitted_k.label)
    number(trace.permitted_k.k_max or max(trace.permitted_k.ks))
    number(trace.r)
    for f in trace.features:
        number(f.omega)
    number(0.0)
    text(hag.label)
    for j in range(LATENT_SLOTS):
        grid.append(
            [hag.crit[j], hag.crit[j], 0.0, "✓"] if j < len(hag.crit) else [DASH, DASH, DASH, "✓"]
        )
    for _ in range(LATENT_SLOTS):
        number(0.0)
    text(", ".join(str(int(d)) for d in s.training.decisions))
    number(0.0)
    number(int(s.new_object.classification.decisions[0]))
    return grid


def _engine_table(s: ExperimentSubject) -> Grid:
    latent = s.model.meta_dataset.latent
    meta = s.training.meta
    rows_: Grid = []
    for t in range(s.trace.m):
        r: list[Cell] = [
            float(latent[t, j]) if j < latent.shape[1] else DASH for j in range(LATENT_SLOTS)
        ]
        rows_.append([*r, float(meta.scores1[t]), float(meta.scores2[t])])
    return rows_


def _validation_checks() -> list[ExperimentCheck]:
    sheet = "Validation"
    return [
        Check(
            sheet,
            "A8",
            "switch state",
            lambda s: [
                ["Switches at the default (template) setting — the engine reference below applies"]
            ],
            when=lambda s: (
                s.hag.settings.centres.value == "running" and s.hag.settings.step4_passes == 2
            ),
        ),
        Check(sheet, "B11:E32", "engine value, workbook value, difference, pass", _validation_rows),
        Check(sheet, "I12:N21", "engine table: r₁ … r₄, score₁, score₂", _engine_table),
        Check(sheet, "I22:N22", "engine table: ω", lambda s: row(list(s.trace.omegas))),
        Check(
            sheet,
            "I23:N24",
            "engine table: η(1), η(2)",
            lambda s: [
                [float(f.eta[0]) for f in s.trace.features],
                [float(f.eta[1]) for f in s.trace.features],
            ],
        ),
    ]


def evaluation_checks() -> list[ExperimentCheck]:
    """The map of the experiment workbook's evaluation sheets, in sheet order."""
    return [
        *_margin_checks(),
        *_metric_checks(),
        *_roc_checks(),
        Check(
            "Leave-One-Out",
            "B5:R14",
            "folds: fold data, held-out description, scores, baselines",
            _loo_rows,
        ),
        Check("Sensitivity (Switches)", "A9:I12", "the four switch settings", _sensitivity_rows),
        Check(
            "Template Deviations",
            "A53:H56",
            "effect of the switches on this experiment",
            _deviation_effects,
        ),
        *_property_checks(),
        *_validation_checks(),
    ]
