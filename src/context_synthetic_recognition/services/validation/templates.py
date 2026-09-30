"""Validation of the author's template workbooks — the replication targets of ADR-001.

``RegularizedStackingEnsembleWithHAG [Heart-Disease (10, 13, 2)]`` (HAG template)
    The HAG runs on the template's own input — 13 original features as contribution values, their
    weights and the classes (sheet *Dataset (Contribution & Weight)*) — and every candidate block
    of the four *Greedy upon Weight* sheets, θ/γ, q, crit, the latent features and the sheet
    *Dataset for Meta-algorithm* are compared. The template setting reproduces
    SET {x₃, x₆, x₁₃, x₄, x₉} and r₁ … r₄.
``Meta-algorithm [Heart-Disease (10, 13, 2)]`` (meta-algorithm template)
    The meta-algorithm classifies the template's new object with the template's training
    description (sheet *Brace for Meta-algorithm*); the result is Class 2 with B1(a₄) = ∅,
    B2(a₄) = {S₉, S₁₀}.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from context_synthetic_recognition.config.models import ExperimentConfig
from context_synthetic_recognition.config.presets import active_deviations
from context_synthetic_recognition.core.contributions import weight_ranks
from context_synthetic_recognition.core.hag import HAGIteration, HAGResult, hag
from context_synthetic_recognition.core.meta import (
    K1_DECISION,
    K2_DECISION,
    MetaDescription,
    MetaResult,
    meta_classify,
)
from context_synthetic_recognition.notation import DASH, feature_name, subscript
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

TEMPLATE_OBJECTS = 10
"""The templates' fixed layout: 10 objects, 13 features."""
TEMPLATE_FEATURES = 13
TEMPLATE_SHEETS = 4
"""*Greedy upon Weight (1–4-Latent)*: at most four latent features."""
TEMPLATE_SET = 5
"""Initial features in the template's meta-dataset (SET)."""
INPUT_SHEET = "Dataset (Contribution & Weight)"


# ---------------------------------------------------------------- HAG template


@dataclass(frozen=True)
class HAGTemplateSubject:
    """The HAG on the template's input."""

    result: HAGResult
    labels: tuple[Cell, ...]
    """The class labels of the objects (1 or 2)."""


def read_hag_template(workbook: Any) -> tuple[Any, Any, Any, tuple[Cell, ...]]:
    """Contributions (10 × 13), weights, class indices and labels of the HAG template.

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


def _latent_rows(subject: HAGTemplateSubject, it: HAGIteration) -> Grid:
    """The STEP 4 block: R, +, η_q, =, R + η_q, class, once, twice."""
    if it.q is None or it.once is None or it.latent is None:  # pragma: no cover - see _added
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
            it.latent[t],
        ]
        for t in range(subject.result.m)
    ]


def _template_block(subject: HAGTemplateSubject, iteration: int, u: int) -> Grid:
    """A candidate block of the template: M and N hold the running θ and γ."""
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


def _added(j: int) -> Callable[[HAGTemplateSubject], bool]:
    """Whether iteration j was executed and added a feature (its sheet then has this layout)."""

    def check(subject: HAGTemplateSubject) -> bool:
        iterations = subject.result.iterations
        return j <= len(iterations) and iterations[j - 1].q is not None

    return check


def _iteration_checks(j: int) -> list[Check[HAGTemplateSubject]]:
    sheet = f"Greedy upon Weight ({j}-Latent)"
    n = TEMPLATE_FEATURES - j  # candidates of iteration j when every earlier one added a feature
    when = _added(j)

    def it(subject: HAGTemplateSubject) -> HAGIteration:
        return subject.result.iterations[j - 1]

    checks: list[Check[HAGTemplateSubject]] = []
    for c in range(n):
        top = 32 + 15 * c

        def block(subject: HAGTemplateSubject, c: int = c) -> Grid:
            return _template_block(subject, j - 1, int(it(subject).candidates[c]))

        def ratio(subject: HAGTemplateSubject, c: int = c) -> Grid:
            return [[float(it(subject).ratio[c])]]

        checks += [
            Check(sheet, f"B{top}:N{top + 9}", f"candidate {c + 1}: STEP 3 columns", block, when),
            Check(sheet, f"O{top + 9}", f"candidate {c + 1}: θ/γ", ratio, when),
        ]
    step4 = 42 + 15 * n

    def summary(subject: HAGTemplateSubject) -> Grid:
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

    def decision(subject: HAGTemplateSubject) -> Grid:
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
            "STEP 4: R + η_q, majorizer twice",
            lambda s: _latent_rows(s, it(s)),
            when,
        ),
        Check(
            sheet,
            f"K{step4}:K{step4 + 9}",
            f"latent feature r{j}",
            lambda s: rows(np.asarray(it(s).latent)[:, None]),
            when,
        ),
        Check(sheet, f"O{step4}:O{step4 + 7}", "crit, cr1 reset, |SET|, |P|, δ, ϰ", summary, when),
        Check(sheet, f"N{step4 + 12}", "STEP 4 condition", decision, when),
    ]


def _slots(values: Any, width: int, m: int) -> Grid:
    """``values`` (m × k) padded with "—" to ``width`` columns."""
    grid: Grid = rows(values) if values.shape[1] else [[] for _ in range(m)]
    return [line + [DASH] * (width - len(line)) for line in grid]


def _meta_dataset_checks() -> list[Check[HAGTemplateSubject]]:
    sheet = "Dataset for Meta-algorithm"

    def initial(subject: HAGTemplateSubject) -> Grid:
        result = subject.result
        return _slots(result.contributions[:, list(result.tuplam)], TEMPLATE_SET, result.m)

    def latent(subject: HAGTemplateSubject) -> Grid:
        return _slots(subject.result.latent, TEMPLATE_SET - 1, subject.result.m)

    def classes(subject: HAGTemplateSubject) -> Grid:
        return column(list(subject.labels))

    return [
        Check(sheet, "B3:F12", "initial features y (SET order)", initial),
        Check(sheet, "G3:G12", "class", classes),
        Check(sheet, "J3:M12", "latent features r₁ … r₄", latent),
        Check(sheet, "N3:N12", "class", classes),
        Check(sheet, "E14", "|SET|", lambda s: [[len(s.result.tuplam)]]),
        Check(sheet, "M14", "p", lambda s: [[s.result.p]]),
        Check(
            sheet,
            "B23:J32",
            "Y = (y, r)",
            lambda s: [a + b for a, b in zip(initial(s), latent(s), strict=True)],
        ),
        Check(sheet, "K23:K32", "class", classes),
    ]


def hag_template_checks() -> list[Check[HAGTemplateSubject]]:
    """The map of the HAG template workbook."""
    return [
        Check(
            INPUT_SHEET,
            "B19:N19",
            "rank of the weights",
            lambda s: rows(weight_ranks(s.result.weights)),
        ),
        *[check for j in range(1, TEMPLATE_SHEETS + 1) for check in _iteration_checks(j)],
        *_meta_dataset_checks(),
    ]


def validate_hag_template(
    file: Path, workbook: Any, config: ExperimentConfig | None, tolerance: float
) -> ValidationReport:
    """Replicate the HAG template: SET, θ/γ of every candidate, crit and r₁ … r₄."""
    chosen = config or ExperimentConfig()
    contributions, weights, class_index, labels = read_hag_template(workbook)
    result = hag(contributions, weights, class_index, chosen.hag)
    notes = [
        f"SET = {{{', '.join(feature_name(u) for u in result.tuplam)}}}; stop: {result.stop.text}",
        *[f"⚠ template calculation ({d.adr}): {d.statement}" for d in active_deviations(chosen)],
    ]
    subject = HAGTemplateSubject(result, labels)
    results = run_checks(hag_template_checks(), workbook, subject, tolerance, file.name)
    return ValidationReport(str(file), tolerance, results, "hag-template", tuple(notes))


# ---------------------------------------------------------------- meta-algorithm template


@dataclass(frozen=True)
class MetaTemplateSubject:
    """The meta-algorithm on the template's training description and new object."""

    result: MetaResult


def read_meta_template(workbook: Any) -> tuple[MetaDescription, Any]:
    """The training description (rows S₁ … S₁₀ in their original order) and the new object.

    Raises:
        ValidationError: A value is missing or a label is not 1 or 2.
    """
    sheet = workbook["Brace for Meta-algorithm"]
    gradations = read_numbers(sheet, "B88:F97", "gradations aᵢ")
    latent = read_numbers(sheet, "G88:J97", "latent values dᵢ")
    labels = read_numbers(sheet, "K88:K97", "classes")[:, 0]
    if not np.isin(labels, (1, 2)).all():
        raise ValidationError("Brace for Meta-algorithm!K88:K97: the classes must be 1 or 2")
    query = read_numbers(sheet, "B77:F77", "new object")[0]
    description = MetaDescription(gradations, latent, labels.astype(np.int64) - 1)
    return description, query


def _decision_cell(decision: int) -> Cell:
    if decision == K1_DECISION:
        return "Class 1"
    return "Class 2" if decision == K2_DECISION else 0


def meta_template_checks() -> list[Check[MetaTemplateSubject]]:
    """The map of the meta-algorithm template workbook."""
    sheet = "Meta-algorithm"

    def sizes(subject: MetaTemplateSubject) -> Grid:
        description = subject.result.description
        return column([description.p + 1, description.p])

    def classes(subject: MetaTemplateSubject) -> Grid:
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


def validate_meta_template(
    file: Path, workbook: Any, config: ExperimentConfig | None, tolerance: float
) -> ValidationReport:
    """Replicate the meta-algorithm template: the new object's class and the final B1, B2."""
    chosen = config or ExperimentConfig()
    description, query = read_meta_template(workbook)
    rule = chosen.meta.decision_rule
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
    subject = MetaTemplateSubject(result)
    results = run_checks(meta_template_checks(), workbook, subject, tolerance, file.name)
    return ValidationReport(str(file), tolerance, results, "meta-template", tuple(notes))
