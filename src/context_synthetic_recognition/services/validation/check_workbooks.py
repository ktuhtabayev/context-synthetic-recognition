"""Validation of the author's two check workbooks (ADR-029, ADR-060).

The additional-features check workbook
    The hierarchical agglomerative algorithm runs on the workbook's own input — 13 original
    features as contribution values, their informativeness and the classes — and every
    candidate block of its four step sheets, θ/γ, q, crit, the additional features d₁ … d₄ and
    its meta-description sheet are compared. The workbook's settings (running class means, two
    majorizing passes) reproduce SET {x₃, x₆, x₁₃, x₄, x₉}.
The meta-algorithm check workbook
    The meta-algorithm classifies the workbook's new object with its training description; the
    result is Class 2 with B1(a₄) = ∅, B2(a₄) = {S₉, S₁₀}.

The workbooks' sheet names are quoted in :mod:`.sheets`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from context_synthetic_recognition.config.models import ExperimentConfig
from context_synthetic_recognition.config.presets import PresetName, active_deviations, preset
from context_synthetic_recognition.core.additional_features.agglomerative import (
    AgglomerativeIteration,
    AgglomerativeResult,
    agglomerative,
)
from context_synthetic_recognition.core.meta_algorithm.algorithm import (
    K1_DECISION,
    K2_DECISION,
    MetaDescription,
    MetaResult,
    meta_classify,
)
from context_synthetic_recognition.core.synthetic_features.contributions import (
    informativeness_ranks,
)
from context_synthetic_recognition.notation import DASH, feature_name, subscript
from context_synthetic_recognition.services.validation import sheets
from context_synthetic_recognition.services.validation.checks import (
    Cell,
    Check,
    Grid,
    ValidationError,
    ValidationReport,
    column,
    read_numbers,
    row,
    rows,
    run_checks,
    set_text,
)

WORKBOOK_OBJECTS = 10
"""The check workbooks' fixed layout: 10 objects, 13 features."""
WORKBOOK_FEATURES = 13
WORKBOOK_SHEETS = 4
"""Four step sheets: at most four additional features."""
WORKBOOK_SET = 5
"""Initial features in the workbook's meta-description (SET)."""
INPUT_SHEET = sheets.CONTRIBUTION_INPUT


# ----------------------------------------------------------- the additional-features check workbook


@dataclass(frozen=True)
class AdditionalFeaturesWorkbookSubject:
    """The hierarchical agglomerative algorithm on the check workbook's input."""

    result: AgglomerativeResult
    labels: tuple[Cell, ...]
    """The class labels of the objects (1 or 2)."""


def read_additional_features_workbook(workbook: Any) -> tuple[Any, Any, Any, tuple[Cell, ...]]:
    """Contributions (10 × 13), informativeness, class indices and labels of the check workbook.

    Raises:
        ValidationError: A value is missing or a label is not 1 or 2.
    """
    sheet = workbook[INPUT_SHEET]
    contributions = read_numbers(sheet, "B3:N12", "contribution values")
    weights = read_numbers(sheet, "B18:N18", "weights")[0]
    labels = read_numbers(sheet, "O3:O12", "classes")[:, 0]
    if not np.isin(labels, (1, 2)).all():
        raise ValidationError(f"{INPUT_SHEET}!O3:O12: the classes must be 1 or 2")
    return contributions, weights, labels.astype(np.int64) - 1, tuple(int(v) for v in labels)


def _additional_rows(
    subject: AdditionalFeaturesWorkbookSubject, it: AgglomerativeIteration
) -> Grid:
    """The Step 4 block: R, +, η_q, =, R + η_q, class, once, twice."""
    if it.q is None or it.once is None or it.additional is None:  # pragma: no cover - see _added
        raise ValidationError(f"iteration {it.number} added no feature")
    eta = subject.result.contributions[:, it.q]
    return [
        [
            it.entering[t],
            "+",
            eta[t],
            "=",
            it.entering[t] + eta[t],
            subject.labels[t],
            it.once[t],
            it.additional[t],
        ]
        for t in range(subject.result.m)
    ]


def _workbook_block(subject: AdditionalFeaturesWorkbookSubject, iteration: int, u: int) -> Grid:
    """A candidate block of the check workbook: M and N hold the running θ and γ."""
    scan = subject.result.scan(iteration, u)
    theta, gamma = np.cumsum(scan.to_own), np.cumsum(scan.to_other)
    return [
        [
            scan.entering[t],
            "+",
            scan.eta[t],
            "=",
            scan.b[t],
            subject.labels[t],
            scan.majorized[t],
            scan.sum1[t],
            scan.sum2[t],
            scan.centre1[t],
            scan.centre2[t],
            theta[t],
            gamma[t],
        ]
        for t in range(subject.result.m)
    ]


def _added(j: int) -> Callable[[AdditionalFeaturesWorkbookSubject], bool]:
    """Whether iteration j was executed and added a feature (its sheet then has this layout)."""

    def check(subject: AdditionalFeaturesWorkbookSubject) -> bool:
        iterations = subject.result.iterations
        return j <= len(iterations) and iterations[j - 1].q is not None

    return check


def _iteration_checks(j: int) -> list[Check[AdditionalFeaturesWorkbookSubject]]:
    sheet = sheets.agglomerative_step(j)
    n = WORKBOOK_FEATURES - j  # candidates of iteration j when every earlier one added a feature
    when = _added(j)

    def it(subject: AdditionalFeaturesWorkbookSubject) -> AgglomerativeIteration:
        return subject.result.iterations[j - 1]

    checks: list[Check[AdditionalFeaturesWorkbookSubject]] = []
    for c in range(n):
        top = 32 + 15 * c

        def block(subject: AdditionalFeaturesWorkbookSubject, c: int = c) -> Grid:
            return _workbook_block(subject, j - 1, int(it(subject).candidates[c]))

        def ratio(subject: AdditionalFeaturesWorkbookSubject, c: int = c) -> Grid:
            return [[float(it(subject).ratio[c])]]

        checks += [
            Check(sheet, f"B{top}:N{top + 9}", f"candidate {c + 1}: Step 3 columns", block, when),
            Check(sheet, f"O{top + 9}", f"candidate {c + 1}: θ/γ", ratio, when),
        ]
    step4 = 42 + 15 * n

    def summary(subject: AdditionalFeaturesWorkbookSubject) -> Grid:
        state = it(subject)
        settings = subject.result.settings
        size = len(state.tuplam_after)
        return column(
            [
                state.crit,
                settings.cr1,
                None,
                size,
                None,
                state.candidates.size - 1,
                settings.delta,
                settings.kappa,
            ]
        )

    def decision(subject: AdditionalFeaturesWorkbookSubject) -> Grid:
        state = it(subject)
        settings = subject.result.settings
        go_on = len(state.tuplam_after) < settings.kappa and state.crit > settings.delta
        return [["Go back to Step 3" if go_on else "Output SET and stop"]]

    return [
        *checks,
        Check(sheet, f"O{30 + 15 * n}", "cr1 = min θ/γ", lambda s: [[it(s).cr1]], when),
        Check(sheet, f"O{33 + 15 * n}", "q", lambda s: [[int(it(s).q or 0) + 1]], when),
        Check(
            sheet,
            f"B{step4}:I{step4 + 9}",
            "Step 4: R + η_q, majorizing function twice",
            lambda s: _additional_rows(s, it(s)),
            when,
        ),
        Check(
            sheet,
            f"K{step4}:K{step4 + 9}",
            f"additional feature d{j}",
            lambda s: rows(np.asarray(it(s).additional)[:, None]),
            when,
        ),
        Check(sheet, f"O{step4}:O{step4 + 7}", "crit, cr1 reset, |SET|, |P|, δ, ϰ", summary, when),
        Check(sheet, f"N{step4 + 12}", "Step 4 condition", decision, when),
    ]


def _slots(values: Any, width: int, m: int) -> Grid:
    """``values`` (m × k) padded with "—" to ``width`` columns."""
    grid: Grid = rows(values) if values.shape[1] else [[] for _ in range(m)]
    return [line + [DASH] * (width - len(line)) for line in grid]


def _meta_dataset_checks() -> list[Check[AdditionalFeaturesWorkbookSubject]]:
    sheet = sheets.META_DESCRIPTION

    def initial(subject: AdditionalFeaturesWorkbookSubject) -> Grid:
        result = subject.result
        return _slots(result.contributions[:, list(result.tuplam)], WORKBOOK_SET, result.m)

    def additional(subject: AdditionalFeaturesWorkbookSubject) -> Grid:
        return _slots(subject.result.additional, WORKBOOK_SET - 1, subject.result.m)

    def classes(subject: AdditionalFeaturesWorkbookSubject) -> Grid:
        return column(list(subject.labels))

    return [
        Check(sheet, "B3:F12", "initial features y (SET order)", initial),
        Check(sheet, "G3:G12", "class", classes),
        Check(sheet, "J3:M12", "additional features d₁ … d₄", additional),
        Check(sheet, "N3:N12", "class", classes),
        Check(sheet, "E14", "|SET|", lambda s: [[len(s.result.tuplam)]]),
        Check(sheet, "M14", "p", lambda s: [[s.result.p]]),
        Check(
            sheet,
            "B23:J32",
            "Y = (y, d)",
            lambda s: [a + b for a, b in zip(initial(s), additional(s), strict=True)],
        ),
        Check(sheet, "K23:K32", "class", classes),
    ]


def additional_features_workbook_checks() -> list[Check[AdditionalFeaturesWorkbookSubject]]:
    """The map of the additional-features check workbook."""
    return [
        Check(
            INPUT_SHEET,
            "B19:N19",
            "rank of the weights",
            lambda s: rows(informativeness_ranks(s.result.informativeness)),
        ),
        *[check for j in range(1, WORKBOOK_SHEETS + 1) for check in _iteration_checks(j)],
        *_meta_dataset_checks(),
    ]


def validate_additional_features_workbook(
    file: Path, workbook: Any, config: ExperimentConfig | None, tolerance: float
) -> ValidationReport:
    """Replicate the additional-features check workbook: SET, θ/γ, crit and d₁ … d₄."""
    chosen = config or preset(PresetName.EXCEL_EXPERIMENT)
    contributions, weights, class_index, labels = read_additional_features_workbook(workbook)
    result = agglomerative(contributions, weights, class_index, chosen.additional_features)
    notes = [
        f"SET = {{{', '.join(feature_name(u) for u in result.tuplam)}}}; stop: {result.stop.text}",
        *[
            f"⚠ deviates from the article ({d.adr}): {d.statement}"
            for d in active_deviations(chosen)
        ],
    ]
    subject = AdditionalFeaturesWorkbookSubject(result, labels)
    results = run_checks(
        additional_features_workbook_checks(), workbook, subject, tolerance, file.name
    )
    return ValidationReport(str(file), tolerance, results, "additional-features", tuple(notes))


# ---------------------------------------------------------------- the meta-algorithm check workbook


@dataclass(frozen=True)
class MetaAlgorithmWorkbookSubject:
    """The meta-algorithm on the check workbook's training description and new object."""

    result: MetaResult


def read_meta_algorithm_workbook(workbook: Any) -> tuple[MetaDescription, Any]:
    """The training description (rows S₁ … S₁₀ in their original order) and the new object.

    Raises:
        ValidationError: A value is missing or a label is not 1 or 2.
    """
    sheet = workbook[sheets.NEW_OBJECT]
    gradations = read_numbers(sheet, "B88:F97", "gradations aᵢ")
    additional = read_numbers(sheet, "G88:J97", "additional values dᵢ")
    labels = read_numbers(sheet, "K88:K97", "classes")[:, 0]
    if not np.isin(labels, (1, 2)).all():
        raise ValidationError("Brace for Meta-algorithm!K88:K97: the classes must be 1 or 2")
    query = read_numbers(sheet, "B77:F77", "new object")[0]
    description = MetaDescription(gradations, additional, labels.astype(np.int64) - 1)
    return description, query


def _decision_cell(decision: int) -> Cell:
    if decision == K1_DECISION:
        return "Class 1"
    return "Class 2" if decision == K2_DECISION else 0


def meta_algorithm_workbook_checks() -> list[Check[MetaAlgorithmWorkbookSubject]]:
    """The map of the meta-algorithm check workbook."""
    sheet = "Meta-algorithm"

    def sizes(subject: MetaAlgorithmWorkbookSubject) -> Grid:
        description = subject.result.description
        return column([description.p + 1, description.p])

    def classes(subject: MetaAlgorithmWorkbookSubject) -> Grid:
        return column(list(subject.result.description.class_sizes))

    return [
        Check(sheet, "B24:F24", "the new object", lambda s: row(list(s.result.queries[0]))),
        Check(sheet, "Q3:Q4", "|SET|, p", sizes),
        Check(sheet, "Q6:Q7", "|K1|, |K2|", classes),
        Check(sheet, "J206:J207", "|K1|, |K2| (Step 4)", classes),
        Check(
            sheet,
            "F216:F217",
            "|B1(a_p)|, |B2(a_p)|",
            lambda s: column([int(s.result.b1_sizes[0, -1]), int(s.result.b2_sizes[0, -1])]),
        ),
        Check(
            sheet,
            "C223",
            "class of the new object (Step 4)",
            lambda s: [[_decision_cell(int(s.result.decisions[0]))]],
        ),
    ]


def validate_meta_algorithm_workbook(
    file: Path, workbook: Any, config: ExperimentConfig | None, tolerance: float
) -> ValidationReport:
    """Replicate the meta-algorithm check workbook: the new object's class and the final B1, B2."""
    chosen = config or preset(PresetName.EXCEL_EXPERIMENT)
    description, query = read_meta_algorithm_workbook(workbook)
    rule = chosen.meta_algorithm.decision_rule
    result = meta_classify(description, query, rule.name, rule.params)
    steps = result.steps(0)
    p = description.p
    notes = [
        (
            f"B1(a{subscript(p)}) = {set_text(steps.b1(p))}, "
            f"B2(a{subscript(p)}) = {set_text(steps.b2(p))}; "
            f"class: {_decision_cell(int(result.decisions[0]))}"
        )
    ]
    subject = MetaAlgorithmWorkbookSubject(result)
    results = run_checks(meta_algorithm_workbook_checks(), workbook, subject, tolerance, file.name)
    return ValidationReport(str(file), tolerance, results, "meta-algorithm", tuple(notes))
