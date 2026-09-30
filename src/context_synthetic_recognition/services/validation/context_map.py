"""The experiment workbook's map, Steps 1–8: sheets *Parameters* … *Ψ(r) Contribution & Weight*.

Each check reads the Step 1–8 trace (``ContextTrace``) of the model fitted on the workbook's own
*Dataset* sheet.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.core.contributions import weight_ranks
from context_synthetic_recognition.core.trace import ContextTrace
from context_synthetic_recognition.notation import DASH
from context_synthetic_recognition.services.validation.checks import (
    Cell,
    Check,
    Grid,
    col,
    column,
    rows,
    span,
)


def labels(trace: ContextTrace, index: npt.ArrayLike) -> list[Cell]:
    """The class labels of the objects at ``index``."""
    return [trace.classes[int(i)] for i in np.asarray(index).ravel()]


WORKBOOK_OBJECTS = 10
"""The workbook's fixed layout: 10 objects, 13 features, operators ρ, ρ_I, ρ_J, 2 permitted k."""
WORKBOOK_FEATURES = 13
WORKBOOK_OPERATORS = ("ρ", "ρ_I", "ρ_J")
WORKBOOK_KS = 2
WORKBOOK_SYNTHETIC = len(WORKBOOK_OPERATORS) * WORKBOOK_KS
"""r = 6 synthetic features a₁ … a₆."""
_MAX_GRADATION = 5
"""The membership tables of the workbook list μ = 0 … 5."""


def _parameter_checks() -> list[Check[ContextTrace]]:
    sheet = "Parameters"
    return [
        Check(sheet, "B5", "objects m", lambda t: [[t.m]]),
        Check(sheet, "B6:B7", "|K1|, |K2|", lambda t: column(list(t.class_sizes))),
        Check(sheet, "B8", "min |Kᵢ|", lambda t: [[min(t.class_sizes)]]),
        Check(sheet, "B9", "k_min", lambda t: [[t.permitted_k.k_min]]),
        Check(sheet, "B10", "k_max = 2·min|Kᵢ| − 3", lambda t: [[t.permitted_k.k_max]]),
        Check(sheet, "B11", "permitted k", lambda t: [[t.permitted_k.label]]),
        Check(sheet, "B12", "number of permitted k", lambda t: [[t.permitted_k.count]]),
        Check(sheet, "B13", "base operators", lambda t: [[len(t.operators)]]),
        Check(sheet, "B14", "r = |Ψ(r)|", lambda t: [[t.r]]),
    ]


def _normalized_checks() -> list[Check[ContextTrace]]:
    sheet = "Normalized Dataset"

    def statistic(t: ContextTrace, key: str) -> Grid:
        values = t.scaling.statistics[key]
        return rows(
            [[v if q else DASH for v, q in zip(values, t.scaling.quantitative, strict=True)]]
        )

    return [
        Check(sheet, "B5:N14", "x′ = (x − min)/(max − min) on I", lambda t: rows(t.normalized)),
        Check(sheet, "O5:O14", "class", lambda t: column(labels(t, t.class_index))),
        Check(sheet, "B15:N15", "type flags", lambda t: rows(t.scaling.quantitative.astype(int))),
        Check(sheet, "B16:N16", "training min", lambda t: statistic(t, "min")),
        Check(sheet, "B17:N17", "training max", lambda t: statistic(t, "max")),
    ]


def _distances_of(o: int) -> Callable[[ContextTrace], Grid]:
    return lambda t: rows(t.operators[o].distances)


def _membership_of(u: int) -> Callable[[ContextTrace], Grid]:
    return lambda t: _membership_grid(t, u)


def _stability_of(u: int) -> Callable[[ContextTrace], Grid]:
    return lambda t: [[t.features[u].stability]]


def _mask_membership_of(o: int) -> Callable[[ContextTrace], Grid]:
    return lambda t: _mask_grid(t, o)


def _mask_stability_of(o: int) -> Callable[[ContextTrace], Grid]:
    return lambda t: [[t.bit_masks[o].stability]]


def _distance_checks() -> list[Check[ContextTrace]]:
    return [
        Check(
            "Zhuravlev Distances",
            span(1, 5 + 13 * o, WORKBOOK_OBJECTS, WORKBOOK_OBJECTS),
            f"{label} distances",
            _distances_of(o),
        )
        for o, label in enumerate(WORKBOOK_OPERATORS)
    ]


def _neighbour_checks() -> list[Check[ContextTrace]]:
    checks: list[Check[ContextTrace]] = []
    ranks = WORKBOOK_OBJECTS - 1
    for o, label in enumerate(WORKBOOK_OPERATORS):
        sheet = f"Sorted Neighbors ({label})"

        def rank_grid(t: ContextTrace, o: int = o) -> Grid:
            ranks_ = t.operators[o].ranks
            return rows([[DASH if v == 0 else v for v in row] for row in ranks_])

        checks.append(Check(sheet, "B5:K14", "rank of Sⱼ among the neighbours of Sᵢ", rank_grid))
        for i in range(WORKBOOK_OBJECTS):
            base = 16 + 11 * i

            def k_labels(t: ContextTrace) -> Grid:
                return [
                    [f"k = {r}" if r in t.permitted_k.ks else None for r in range(1, ranks + 1)]
                ]

            def names(t: ContextTrace, o: int = o, i: int = i) -> Grid:
                return [[t.object_ids[j] for j in t.operators[o].order[i]]]

            def index(t: ContextTrace, o: int = o, i: int = i) -> Grid:
                return rows(t.operators[o].order[i] + 1)

            def distance(t: ContextTrace, o: int = o, i: int = i) -> Grid:
                return rows(t.operators[o].sorted_distances[i])

            def classes(t: ContextTrace, o: int = o, i: int = i) -> Grid:
                return [labels(t, t.class_index[t.operators[o].order[i]])]

            def same(t: ContextTrace, o: int = o, i: int = i) -> Grid:
                neighbours = t.class_index[t.operators[o].order[i]]
                return rows((neighbours == t.class_index[i]).astype(int))

            def mu(t: ContextTrace, o: int = o, i: int = i) -> Grid:
                return rows(t.operators[o].same_class_running(t.class_index)[i])

            def chi1(t: ContextTrace, o: int = o, i: int = i) -> Grid:
                return rows(t.operators[o].k1_running(t.class_index)[i])

            s = f"S{i + 1}"
            for offset, what, fn in (
                (2, f"{s}: k-neighbourhood marks", k_labels),
                (3, f"{s}: neighbours", names),
                (4, f"{s}: original index", index),
                (5, f"{s}: distance", distance),
                (6, f"{s}: class", classes),
                (7, f"{s}: same class", same),
                (8, f"{s}: same-class count μ", mu),
                (9, f"{s}: K1 count χ₁", chi1),
            ):
                checks.append(Check(sheet, span(1, base + offset, ranks, 1), what, fn))
    return checks


def _mu_checks() -> list[Check[ContextTrace]]:
    sheet = "Synthetic Features (k-NN)"
    checks: list[Check[ContextTrace]] = [
        Check(
            sheet,
            "G4:G8",
            "|K1|, |K2|, min, k_min, k_max",
            lambda t: column(
                [*t.class_sizes, min(t.class_sizes), t.permitted_k.k_min, t.permitted_k.k_max]
            ),
        ),
        Check(sheet, "G9", "permitted k", lambda t: [[t.permitted_k.label]]),
    ]
    for o, label in enumerate(WORKBOOK_OPERATORS):
        top = 4 + 14 * o

        def ks(t: ContextTrace) -> Grid:
            return [list(t.permitted_k.ks)]

        def mu(t: ContextTrace, o: int = o) -> Grid:
            return rows(np.column_stack([f.mu for f in t.features if f.operator == o]))

        checks += [
            Check(sheet, f"B{top}:C{top}", f"{label}: permitted k", ks),
            Check(sheet, f"B{top + 1}:C{top + 10}", f"{label}: same-class count μ", mu),
            Check(
                sheet,
                f"D{top + 1}:D{top + 10}",
                f"{label}: class",
                lambda t: column(labels(t, t.class_index)),
            ),
        ]
    return checks


def _psi_checks() -> list[Check[ContextTrace]]:
    sheet = "Ψ(r) Binary Features"

    def per_feature(attribute: str) -> Callable[[ContextTrace], Grid]:
        return lambda t: rows(np.column_stack([getattr(f, attribute) for f in t.features]))

    return [
        Check(sheet, "A5:A10", "feature names", lambda t: column([f.name for f in t.features])),
        Check(
            sheet, "B5:B10", "operators", lambda t: column([f.operator_label for f in t.features])
        ),
        Check(sheet, "C5:C10", "k", lambda t: column([f.k for f in t.features])),
        Check(sheet, "B14:G23", "χ₁ — K1 objects among the k nearest", per_feature("chi1")),
        Check(sheet, "B27:G36", "χ₂ = k − χ₁", per_feature("chi2")),
        Check(sheet, "B40:G49", "Ψ(r): aᵤ by formula (5)", per_feature("values")),
        Check(sheet, "H40:H49", "class", lambda t: column(labels(t, t.class_index))),
    ]


def _membership_grid(t: ContextTrace, u: int) -> Grid:
    feature = t.features[u]
    table = feature.membership
    grid: Grid = []
    for mu in range(_MAX_GRADATION + 1):
        if mu > feature.k:
            grid.append([mu, DASH, DASH, DASH, DASH, DASH, DASH, 0])
            continue
        f = table.f[mu]
        grid.append(
            [
                mu,
                int(table.d1[mu]),
                int(table.d2[mu]),
                int(table.n[mu]),
                float(table.share1[mu]),
                float(table.share2[mu]),
                DASH if math.isnan(f) else float(f),
                float(table.weighted[mu]),
            ]
        )
    return grid


def _mask_grid(t: ContextTrace, o: int) -> Grid:
    table = t.bit_masks[o].membership
    return [
        [
            int(table.gradations[g]),
            int(table.d1[g]),
            int(table.d2[g]),
            int(table.n[g]),
            DASH if math.isnan(table.f[g]) else float(table.f[g]),
            float(table.weighted[g]),
        ]
        for g in range(table.gradations.size)
    ]


def _membership_checks() -> list[Check[ContextTrace]]:
    sheet = "Membership & Stability"
    checks: list[Check[ContextTrace]] = []
    for u in range(len(WORKBOOK_OPERATORS) * WORKBOOK_KS):
        row0 = 3 + 11 * (u // 2)
        col0 = 0 if u % 2 == 0 else 9
        name = f"a{u + 1}"
        checks += [
            Check(
                sheet,
                span(col0, row0 + 2, 8, _MAX_GRADATION + 1),
                f"{name}: d₁, d₂, n, f_k(μ) — formula (1)",
                _membership_of(u),
            ),
            Check(
                sheet,
                f"{col(col0 + 7)}{row0 + 8}",
                f"{name}: stability g — formula (2)",
                _stability_of(u),
            ),
        ]
    checks.append(Check(sheet, "B40:G40", "meta-object (g)", lambda t: rows(t.meta_object)))

    def bits(t: ContextTrace) -> Grid:
        columns: list[npt.NDArray[Any]] = []
        for representation in t.bit_masks:
            columns += [representation.bits[:, 0], representation.bits[:, 1], representation.masks]
        return rows(np.column_stack(columns).astype(int))

    checks.append(Check(sheet, "B44:J53", "bits by the majority rule and masks", bits))
    for o, label in enumerate(WORKBOOK_OPERATORS):
        col0 = 7 * o
        checks += [
            Check(
                sheet,
                span(col0, 57, 6, 4),
                f"{label} masks: formula (1)",
                _mask_membership_of(o),
            ),
            Check(
                sheet,
                f"{col(col0 + 5)}61",
                f"{label} masks: stability g",
                _mask_stability_of(o),
            ),
        ]
    return checks


def _informativeness_checks() -> list[Check[ContextTrace]]:
    sheet = "Informativeness ω"

    def row(fn: Callable[[Any], Cell]) -> Callable[[ContextTrace], Grid]:
        return lambda t: [[fn(f) for f in t.features]]

    def q(value: float | None) -> Cell:
        return "none" if value is None else value

    return [
        Check(sheet, "B4:G4", "features", row(lambda f: f.name)),
        Check(sheet, "B5:G5", "operators", row(lambda f: f.operator_label)),
        Check(sheet, "B6:G6", "k", row(lambda f: f.k)),
        Check(sheet, "B7:G7", "q₂ = max{f < 0.5}", row(lambda f: q(f.boundary.q2))),
        Check(sheet, "B8:G8", "q₁ = min{f > 0.5}", row(lambda f: q(f.boundary.q1))),
        Check(sheet, "B9:G9", "G_k — formula (3)", row(lambda f: f.boundary.G)),
        Check(
            sheet,
            "B10:G10",
            "correctly placed objects",
            row(lambda f: int(np.count_nonzero(f.correct))),
        ),
        Check(sheet, "B11:G11", "ω — formula (4)", row(lambda f: f.omega)),
        Check(sheet, "B12:G12", "rank of ω", lambda t: rows(weight_ranks(t.omegas))),
        Check(
            sheet,
            "B16:G25",
            "g(S, k) = f_k(μ_S)",
            lambda t: rows(np.column_stack([f.object_membership for f in t.features])),
        ),
        Check(
            sheet,
            "B29:G38",
            "correct side of G_k",
            lambda t: rows(np.column_stack([f.correct for f in t.features]).astype(int)),
        ),
    ]


def _contribution_checks() -> list[Check[ContextTrace]]:
    sheet = "Ψ(r) Contribution & Weight"

    def alpha(j: int, c: int) -> Callable[[ContextTrace], Grid]:
        return lambda t: [[int(f.alpha[j, c]) for f in t.features]]

    return [
        Check(sheet, "B5:G5", "α¹₁", alpha(0, 0)),
        Check(sheet, "B6:G6", "α²₁", alpha(0, 1)),
        Check(sheet, "B7:G7", "α¹₂", alpha(1, 0)),
        Check(sheet, "B8:G8", "α²₂", alpha(1, 1)),
        Check(sheet, "B12:G12", "weight ω", lambda t: rows(t.weights)),
        Check(
            sheet,
            "B13:G14",
            "η(1), η(2) — formula (6)",
            lambda t: rows(np.array([f.eta for f in t.features]).T),
        ),
        Check(
            sheet,
            "B18:G27",
            "Ψ(r) as contribution values ηᵤ(a_tu)",
            lambda t: rows(t.contributions),
        ),
        Check(sheet, "H18:H27", "class", lambda t: column(labels(t, t.class_index))),
        Check(sheet, "B31:G31", "weights", lambda t: rows(t.weights)),
        Check(sheet, "B32:G32", "rank of the weights", lambda t: rows(weight_ranks(t.weights))),
    ]


def context_checks() -> list[Check[ContextTrace]]:
    """The map of the experiment workbook, Steps 1–8, in sheet order."""
    return [
        *_parameter_checks(),
        *_normalized_checks(),
        *_distance_checks(),
        *_neighbour_checks(),
        *_mu_checks(),
        *_psi_checks(),
        *_membership_checks(),
        *_informativeness_checks(),
        *_contribution_checks(),
    ]
