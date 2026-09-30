"""The experiment workbook's map, Steps 9–12: HAG, meta-dataset, new object, meta-algorithm.

Sheets *Greedy upon Weight (1–4-Latent)*, *Dataset for Meta-algorithm*, *Brace for Meta-algorithm*,
*Meta-algorithm* and *Meta-algorithm (All Objects)*. Every cell the workbook computes is covered,
including the states of an iteration that was not executed and the positions of TUPLAM that the
HAG did not fill ("—"), so shorter groupings validate too.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from context_synthetic_recognition.config.models import CentreMode
from context_synthetic_recognition.core.arrays import FloatArray
from context_synthetic_recognition.core.hag import HAGIteration, HAGResult
from context_synthetic_recognition.core.meta import K1_DECISION, K2_DECISION, MetaSteps
from context_synthetic_recognition.core.model import Classification, CSModel, Representation
from context_synthetic_recognition.core.trace import ContextTrace
from context_synthetic_recognition.notation import DASH, object_name, subscript, synthetic_name
from context_synthetic_recognition.services.validation.checks import (
    Cell,
    Check,
    Grid,
    column,
    flag,
    row,
    rows,
    set_text,
)
from context_synthetic_recognition.services.validation.context_map import (
    WORKBOOK_FEATURES,
    WORKBOOK_OBJECTS,
    WORKBOOK_SYNTHETIC,
    labels,
)

HAG_SHEETS = 4
"""The workbook has four *Greedy upon Weight* sheets: at most p = 4 latent features."""
TUPLAM_SLOTS = HAG_SHEETS + 1
"""Positions y₀ … y₄ (a₀ … a₄) of the meta-dataset and the meta-algorithm."""
EXCLUDED_RANK = 999
"""The rank the *Brace* sheet shows for the training object left out of the new object's context."""


@dataclass(frozen=True)
class NewObject:
    """The workbook's new object (sheet *Brace for Meta-algorithm*) and its classification."""

    values: FloatArray
    """(n,) the typed feature values (row "x (input)")."""
    exclude: int | None
    """0-based training object left out of its context (leave-self-out), or ``None``."""
    classification: Classification
    """Its representation and the meta-algorithm's decision."""


@dataclass(frozen=True)
class ExperimentSubject:
    """Everything the experiment workbook shows, computed by the package."""

    model: CSModel
    """The CS-model fitted on the *Dataset* sheet with the workbook's parameters."""
    new_object: NewObject
    """The *Brace* / *Meta-algorithm* demo."""
    training: Classification
    """Every training object classified with its own row (resubstitution)."""
    score_decimals: int
    """Rounding of score₁ − score₂ (sheet *Meta-algorithm (All Objects)*)."""

    @property
    def trace(self) -> ContextTrace:
        """The Step 1–8 trace."""
        return self.model.trace

    @property
    def hag(self) -> HAGResult:
        """The HAG of the fit."""
        return self.model.hag


ExperimentCheck = Check[ExperimentSubject]


def _at(
    fn: Callable[[ExperimentSubject, int], Grid], index: int
) -> Callable[[ExperimentSubject], Grid]:
    """``fn`` with its second argument (a feature, step or object index) fixed."""
    return lambda s: fn(s, index)


# ---------------------------------------------------------------- shared helpers


def _classes(s: ExperimentSubject) -> list[Cell]:
    return labels(s.trace, s.trace.class_index)


def _slot_feature(s: ExperimentSubject, position: int) -> int | None:
    """The TUPLAM feature at ``position`` (0 = y₀), or ``None`` if the HAG did not fill it."""
    tuplam = s.hag.tuplam
    return tuplam[position] if position < len(tuplam) else None


def _slot_name(s: ExperimentSubject, position: int) -> str:
    feature = _slot_feature(s, position)
    return DASH if feature is None else synthetic_name(feature)


def _slot_values(s: ExperimentSubject, values: np.ndarray, position: int) -> list[Cell]:
    """Column ``position`` of a (m × slots) table, or "—" for every object if it is not filled."""
    if position >= values.shape[1]:
        return [DASH] * s.trace.m
    return [v.item() if hasattr(v, "item") else v for v in values[:, position]]


def _tuplam_label(tuplam: tuple[int, ...]) -> str:
    return "{" + ", ".join(synthetic_name(u) for u in tuplam) + "}"


# ---------------------------------------------------------------- Greedy upon Weight (j-Latent)


def _iteration(s: ExperimentSubject, j: int) -> HAGIteration | None:
    """The j-th iteration (1-based) if it was executed."""
    iterations = s.hag.iterations
    return iterations[j - 1] if j <= len(iterations) else None


def _tuplam_after(s: ExperimentSubject, i: int) -> tuple[int, ...]:
    """TUPLAM after the first ``i`` iterations (an unexecuted iteration changes nothing)."""
    added = [it.q for it in s.hag.iterations[:i] if it.q is not None]
    return (s.hag.first, *added)


def _crit_after(s: ExperimentSubject, i: int) -> float:
    done = s.hag.iterations[: min(i, len(s.hag.iterations))]
    return done[-1].crit if done else s.hag.settings.cr1


def _pool_after(s: ExperimentSubject, i: int) -> int:
    """|P| after iteration ``i`` (0 after an unexecuted one, as the workbook's cell shows)."""
    if i == 0:
        return s.hag.r - 1
    it = _iteration(s, i)
    return 0 if it is None else it.candidates.size - (it.q is not None)


def _entering(s: ExperimentSubject, j: int) -> FloatArray:
    it = _iteration(s, j)
    if it is not None:
        return it.entering
    if j == 1:
        return s.hag.contributions[:, s.hag.first]
    return np.zeros(s.trace.m)


def _added(s: ExperimentSubject, j: int) -> int | None:
    it = _iteration(s, j)
    return None if it is None else it.q


def _status(s: ExperimentSubject, j: int, u: int) -> str:
    it = _iteration(s, j)
    if it is None:
        return "iteration not executed"
    return "available" if bool(it.available[u]) else "in TUPLAM — skipped"


def _step5_text(s: ExperimentSubject, j: int) -> str:
    it = _iteration(s, j)
    if it is None:
        return "Iteration not executed"
    return "Go back to Step 3" if it.go_on else "Output TUPLAM and stop"


def _available(s: ExperimentSubject, j: int, u: int) -> bool:
    it = _iteration(s, j)
    return it is not None and bool(it.available[u])


def _candidate_block(s: ExperimentSubject, j: int, u: int) -> Grid:
    scan = s.hag.scan_from(_entering(s, j), u)
    classes = _classes(s)
    return [
        [
            scan.entering[t],
            "+",
            scan.eta[t],
            "=",
            scan.b[t],
            classes[t],
            scan.majorized[t],
            scan.sum1[t],
            scan.sum2[t],
            scan.centre1[t],
            scan.centre2[t],
            scan.to_own[t],
            scan.to_other[t],
        ]
        for t in range(s.trace.m)
    ]


def _candidate_totals(s: ExperimentSubject, j: int, u: int) -> Grid:
    scan = s.hag.scan_from(_entering(s, j), u)
    ratio: Cell = scan.ratio if _available(s, j, u) else DASH
    return [[scan.theta, scan.gamma, "θ / γ", ratio]]


def _summary(s: ExperimentSubject, j: int) -> Grid:
    grid: Grid = []
    for u in range(WORKBOOK_SYNTHETIC):
        if _available(s, j, u):
            scan = s.hag.scan_from(_entering(s, j), u)
            grid.append([synthetic_name(u), 1, scan.theta, scan.gamma, scan.ratio])
        else:
            grid.append([synthetic_name(u), 0, DASH, DASH, DASH])
    return grid


def _cr1_cell(s: ExperimentSubject, j: int) -> Grid:
    it = _iteration(s, j)
    return [[DASH if it is None else it.cr1]]


def _q_cells(s: ExperimentSubject, j: int) -> Grid:
    q = _added(s, j)
    return [[DASH, DASH]] if q is None else [[q + 1, synthetic_name(q)]]


def _step4_block(s: ExperimentSubject, j: int) -> Grid:
    it = _iteration(s, j)
    entering = _entering(s, j)
    classes = _classes(s)
    q = _added(s, j)
    eta = np.zeros(s.trace.m) if q is None else s.hag.contributions[:, q]
    grid: Grid = []
    for t in range(s.trace.m):
        once: Cell = DASH
        latent: Cell = DASH
        if it is not None and it.once is not None and it.latent is not None:
            once, latent = float(it.once[t]), float(it.latent[t])
        grid.append([entering[t], "+", eta[t], "=", entering[t] + eta[t], classes[t], once, latent])
    return grid


def _step4_summary(s: ExperimentSubject, j: int) -> Grid:
    it = _iteration(s, j)
    q = _added(s, j)
    after = _tuplam_after(s, j)
    settings = s.hag.settings
    return column(
        [
            _crit_after(s, j),
            DASH if q is None else q + 1,
            len(after),
            _tuplam_label(after),
            _pool_after(s, j),
            settings.delta,
            settings.kappa,
            flag(it is not None and it.go_on),
        ]
    )


def _greedy_checks(j: int) -> list[ExperimentCheck]:
    sheet = f"Greedy upon Weight ({j}-Latent)"
    before = j - 1

    def first_or_added(s: ExperimentSubject) -> Grid:
        if j == 1:
            return [[s.hag.first + 1]]
        q = _added(s, before)
        return [[DASH if q is None else q + 1]]

    def ninth(s: ExperimentSubject) -> Grid:
        return [[float(s.hag.weights.max()) if j == 1 else _pool_after(s, before)]]

    def flags(members: Callable[[ExperimentSubject], tuple[int, ...]]) -> Callable[..., Grid]:
        return lambda s: row([flag(u in members(s)) for u in range(WORKBOOK_SYNTHETIC)])

    def parameters(s: ExperimentSubject) -> Grid:
        h = s.hag.settings
        centres = 1 if h.centres is CentreMode.RUNNING else 2
        return [
            [h.alpha, "δ", h.delta, "ϰ", h.kappa],
            [centres, "passes", h.step4_passes, "cr1₀", h.cr1],
        ]

    checks: list[ExperimentCheck] = [
        Check(sheet, "D4", "iteration executed", lambda s: [[flag(_iteration(s, j))]]),
        Check(sheet, "D5", "feature added before (STEP 2: u = argmax ω)", first_or_added),
        Check(
            sheet,
            "D6:D8",
            "TUPLAM, |TUPLAM|, crit before",
            lambda s: column(
                [
                    _tuplam_label(_tuplam_after(s, before)),
                    len(_tuplam_after(s, before)),
                    _crit_after(s, before),
                ]
            ),
        ),
        Check(sheet, "D9", "max ω / |P| before", ninth),
        Check(
            sheet,
            "D10",
            "candidates |P|",
            lambda s: [[sum(_available(s, j, u) for u in range(WORKBOOK_SYNTHETIC))]],
        ),
        Check(sheet, "J5:J14", "R(Sₜ) entering", lambda s: rows(_entering(s, j)[:, None])),
        Check(sheet, "K5:K14", "class", lambda s: column(_classes(s))),
        Check(sheet, "N12:R13", "α, δ, ϰ, switches, cr1₀", parameters),
        Check(sheet, "B13:G13", "in TUPLAM before", flags(lambda s: _tuplam_after(s, before))),
        Check(
            sheet,
            "B14:G14",
            "available in P",
            lambda s: row([flag(_available(s, j, u)) for u in range(WORKBOOK_SYNTHETIC)]),
        ),
    ]
    for u in range(WORKBOOK_SYNTHETIC):
        top = 18 + 14 * u
        name = synthetic_name(u)
        checks += [
            Check(sheet, f"O{top}", f"{name}: status", _at(lambda s, u: [[_status(s, j, u)]], u)),
            Check(
                sheet,
                f"B{top + 2}:N{top + 11}",
                f"{name}: b = R + η, majorizer, centres, |b − M| (STEP 3)",
                _at(lambda s, u: _candidate_block(s, j, u), u),
            ),
            Check(
                sheet,
                f"M{top + 12}:P{top + 12}",
                f"{name}: θ, γ, θ/γ",
                _at(lambda s, u: _candidate_totals(s, j, u), u),
            ),
        ]
    checks += [
        Check(sheet, "H104:L109", "STEP 3 summary: θ, γ, θ/γ", lambda s: _summary(s, j)),
        Check(sheet, "J111", "cr1 = min θ/γ", lambda s: _cr1_cell(s, j)),
        Check(sheet, "J112:K112", "q = argmin θ/γ", lambda s: _q_cells(s, j)),
        Check(
            sheet,
            "B117:I126",
            f"STEP 4: R + η_q, majorizer ⇒ r{subscript(j)}",
            lambda s: _step4_block(s, j),
        ),
        Check(
            sheet,
            "M116:M123",
            "crit, q, |TUPLAM|, TUPLAM, |P|, δ, ϰ, continue",
            lambda s: _step4_summary(s, j),
        ),
        Check(sheet, "K124", "STEP 5: continue or stop", lambda s: [[_step5_text(s, j)]]),
        Check(sheet, "L127:Q127", "in TUPLAM after", flags(lambda s: _tuplam_after(s, j))),
    ]
    return checks


# ---------------------------------------------------------------- Dataset for Meta-algorithm


def _meta_dataset_checks() -> list[ExperimentCheck]:
    sheet = "Dataset for Meta-algorithm"

    def indices(s: ExperimentSubject) -> Grid:
        features = [_slot_feature(s, k) for k in range(TUPLAM_SLOTS)]
        return column([DASH if u is None else u + 1 for u in features])

    def headers(s: ExperimentSubject) -> Grid:
        return row([f"y{subscript(k)} ({_slot_name(s, k)})" for k in range(TUPLAM_SLOTS)])

    def initial(s: ExperimentSubject) -> Grid:
        y = s.model.meta_dataset.initial
        return rows(np.array([_slot_values(s, y, k) for k in range(TUPLAM_SLOTS)], dtype=object).T)

    def latent(s: ExperimentSubject) -> Grid:
        r = s.model.meta_dataset.latent
        return rows(np.array([_slot_values(s, r, k) for k in range(HAG_SHEETS)], dtype=object).T)

    def meta_y(s: ExperimentSubject) -> Grid:
        return [a + b for a, b in zip(initial(s), latent(s), strict=True)]

    return [
        Check(sheet, "B5:B9", "TUPLAM feature indices", indices),
        Check(
            sheet,
            "C5:C9",
            "TUPLAM features",
            lambda s: column([_slot_name(s, k) for k in range(TUPLAM_SLOTS)]),
        ),
        Check(
            sheet,
            "C11:C13",
            "|TUPLAM|, p, TUPLAM",
            lambda s: column([len(s.hag.tuplam), s.hag.p, s.hag.label]),
        ),
        Check(sheet, "B16:F16", "headers y₀ … y₄", headers),
        Check(sheet, "B17:F26", "initial features y (contribution values)", initial),
        Check(sheet, "G17:G26", "class", lambda s: column(_classes(s))),
        Check(sheet, "J17:M26", "latent features r₁ … r₄", latent),
        Check(sheet, "N17:N26", "class", lambda s: column(_classes(s))),
        Check(sheet, "B29:F29", "headers of Y", headers),
        Check(sheet, "B30:J39", "Y = (y₀ … y₄, r₁ … r₄)", meta_y),
        Check(sheet, "K30:K39", "class", lambda s: column(_classes(s))),
    ]


# ---------------------------------------------------------------- Brace for Meta-algorithm


def _gradation_headers(s: ExperimentSubject) -> Grid:
    return row([f"aᵢ{subscript(k)} ({_slot_name(s, k)})" for k in range(TUPLAM_SLOTS)])


def _training_rows(s: ExperimentSubject) -> list[list[Cell]]:
    """A (5 slots), d (4 slots), class of every training object — the *Brace* table."""
    a = s.model.description.gradations
    d = s.model.description.latent
    grads = np.array([_slot_values(s, a, k) for k in range(TUPLAM_SLOTS)], dtype=object).T
    lats = np.array([_slot_values(s, d, k) for k in range(HAG_SHEETS)], dtype=object).T
    classes = _classes(s)
    return [[*rows(grads[t])[0], *rows(lats[t])[0], classes[t]] for t in range(s.trace.m)]


def _sort_keys(s: ExperimentSubject) -> list[int]:
    """The workbook's key class·100 + № (its class labels are the numbers 1 and 2)."""
    return [int(str(label)) * 100 + t + 1 for t, label in enumerate(_classes(s))]


def _positions(s: ExperimentSubject) -> list[int]:
    keys = _sort_keys(s)
    return [sum(other < key for other in keys) + 1 for key in keys]


def _sorted_training(s: ExperimentSubject) -> Grid:
    table = _training_rows(s)
    order = np.argsort(_sort_keys(s), kind="stable")
    return [[object_name(int(t)), *table[int(t)]] for t in order]


def _ranks(s: ExperimentSubject) -> Grid:
    grid: Grid = []
    for order in _new_context(s).context.orders:
        rank = np.full(s.trace.m, EXCLUDED_RANK, dtype=np.int64)
        rank[order[0]] = np.arange(1, order.shape[1] + 1)
        grid.append([int(v) for v in rank])
    return grid


def _new_context(s: ExperimentSubject) -> Representation:
    representation = s.new_object.classification.representation
    assert representation is not None
    return representation


def _brace_checks() -> list[ExperimentCheck]:
    sheet = "Brace for Meta-algorithm"

    def description(s: ExperimentSubject) -> list[Cell]:
        values = _new_context(s).description[0]
        return [int(values[k]) if k < values.size else DASH for k in range(TUPLAM_SLOTS)]

    def synthetic(s: ExperimentSubject) -> Grid:
        context = _new_context(s).context
        ks = [f.k for f in s.trace.features]
        chi1 = context.chi1[0]
        return [
            list[Cell](ks),
            [int(v) for v in chi1],
            [k - int(c) for k, c in zip(ks, chi1, strict=True)],
            [int(v) for v in context.values[0]],
        ]

    return [
        Check(sheet, "B4:F4", "headers aᵢ₀ … aᵢ₄", _gradation_headers),
        Check(
            sheet,
            "B5:K14",
            "training description: aᵢ (gradations), dᵢ (latent), class",
            lambda s: list(_training_rows(s)),
        ),
        Check(sheet, "L5:L14", "sort key (class·100 + №)", lambda s: column(_sort_keys(s))),
        Check(sheet, "M5:M14", "position in the sorted table", lambda s: column(_positions(s))),
        Check(
            sheet, "A18:K27", "training description sorted by class (K1 first)", _sorted_training
        ),
        Check(
            sheet,
            "B32:N32",
            "feature types",
            lambda s: row([flag(q) for q in s.trace.scaling.quantitative]),
        ),
        Check(
            sheet,
            "B33:N33",
            "new object: x′ (training min/max)",
            lambda s: rows(_new_context(s).context.normalized[0]),
        ),
        Check(
            sheet,
            "B39:K41",
            "new object: distances ρ, ρ_I, ρ_J to Sᵢ",
            lambda s: rows(np.vstack([d[0] for d in _new_context(s).context.distances])),
        ),
        Check(sheet, "B45:K47", "new object: rank of Sᵢ (excluded = 999)", _ranks),
        Check(sheet, "B48:K48", "class of Sᵢ", lambda s: row(_classes(s))),
        Check(sheet, "B49:K49", "index i", lambda s: row(list(range(1, s.trace.m + 1)))),
        Check(sheet, "B53:G56", "new object: k, χ₁, χ₂, aᵤ(S) by (5)", synthetic),
        Check(
            sheet,
            "B60:F60",
            "TUPLAM positions",
            lambda s: row([_slot_name(s, k) for k in range(TUPLAM_SLOTS)]),
        ),
        Check(sheet, "B61:F61", "new object: (a₀, …, a_p)", lambda s: row(description(s))),
    ]


# ---------------------------------------------------------------- Meta-algorithm (new object)


def _step_flags(steps: MetaSteps, j: int) -> tuple[np.ndarray, np.ndarray]:
    """B1, B2 membership after step j (a step beyond p keeps the sets of step p)."""
    last = min(j, steps.in_b1.shape[0] - 1)
    return steps.in_b1[last], steps.in_b2[last]


def _new_steps(s: ExperimentSubject) -> MetaSteps:
    return s.new_object.classification.meta.steps(0)


def _meta_step1(s: ExperimentSubject) -> Grid:
    steps = _new_steps(s)
    a = s.model.description.gradations
    classes = _classes(s)
    return [
        [
            classes[t],
            int(a[t, 0]),
            "✅" if steps.match[0, t] else "❌",
            flag(steps.in_b1[0, t]),
            flag(steps.in_b2[0, t]),
        ]
        for t in range(s.trace.m)
    ]


def _meta_step(s: ExperimentSubject, j: int) -> Grid:
    """Rows of step j ≥ 1: class, in B before, aᵢⱼ, match, dᵢⱼ, sign, keep, in B after."""
    steps = _new_steps(s)
    p = s.model.p
    executed = j <= p
    b1_before, b2_before = _step_flags(steps, j - 1)
    b1_after, b2_after = _step_flags(steps, j)
    a = s.model.description.gradations
    d = s.model.description.latent
    classes = _classes(s)
    in_k1 = s.trace.class_index == 0
    grid: Grid = []
    for t in range(s.trace.m):
        before = bool(b1_before[t] if in_k1[t] else b2_before[t])
        after = bool(b1_after[t] if in_k1[t] else b2_after[t])
        if executed:
            value: Cell = int(a[t, j])
            latent: Cell = float(d[t, j - 1])
            match = "✅" if steps.match[j, t] else "❌"
            ok = bool(steps.sign[j - 1, t])
            positive, negative = ("✅ (+)", "❌ (−)") if in_k1[t] else ("✅ (−)", "❌ (+)")
            sign = positive if ok else negative
            keep = ("✅" if steps.match[j, t] and ok else "❌") if before else DASH
        else:
            value = latent = match = sign = DASH
            keep = "✅" if before else DASH
        grid.append([classes[t], flag(before), value, match, latent, sign, keep, flag(after)])
    return grid


def _meta_step_xy(s: ExperimentSubject, j: int) -> Grid:
    b1, b2 = _step_flags(_new_steps(s), j)
    return [[flag(b1[t]), flag(b2[t])] for t in range(s.trace.m)]


def _step3_text(s: ExperimentSubject, j: int) -> str:
    p = s.model.p
    if j < p:
        return "Go back to Step 2"
    return "Go to Step 4" if j == p else "— (j > p: not executed)"


def _decision_text(decision: int) -> str:
    if decision == K1_DECISION:
        return "Class 1"
    return "Class 2" if decision == K2_DECISION else "0 — refusal"


def _meta_checks() -> list[ExperimentCheck]:
    sheet = "Meta-algorithm"

    def query(s: ExperimentSubject) -> Grid:
        values = _new_context(s).description[0]
        cells: list[Cell] = [
            int(values[k]) if k < values.size else DASH for k in range(TUPLAM_SLOTS)
        ]
        return [[*cells, s.model.p]]

    def sets(j: int) -> Callable[[ExperimentSubject], Grid]:
        def grid(s: ExperimentSubject) -> Grid:
            b1, b2 = _step_flags(_new_steps(s), j)
            b1_text, b2_text = set_text(np.flatnonzero(b1)), set_text(np.flatnonzero(b2))
            return [[b1_text, None, None, f"B2(a{subscript(j)}) =", b2_text]]

        return grid

    def decision(s: ExperimentSubject) -> Grid:
        meta = s.new_object.classification.meta
        d = int(meta.decisions[0])
        return [
            [int(meta.b1_sizes[0, -1]), None],
            [int(meta.b2_sizes[0, -1]), None],
            [float(meta.scores1[0]), None],
            [float(meta.scores2[0]), None],
            [d, _decision_text(d)],
        ]

    checks: list[ExperimentCheck] = [
        Check(
            sheet,
            "B5:F5",
            "TUPLAM positions",
            lambda s: row([_slot_name(s, k) for k in range(TUPLAM_SLOTS)]),
        ),
        Check(sheet, "B6:G6", "the object's (a₀, …, a_p) and p", query),
        Check(sheet, "B10:F19", "Step 1: B1(a₀), B2(a₀)", _meta_step1),
        Check(sheet, "B20:F20", "B1(a₀), B2(a₀)", sets(0)),
    ]
    for j in range(1, HAG_SHEETS + 1):
        base = 25 + 15 * (j - 1)
        checks += [
            Check(
                sheet,
                f"J{base - 1}",
                f"step j = {j} executed (j ≤ p)",
                _at(lambda s, j: [[flag(j <= s.model.p)]], j),
            ),
            Check(
                sheet,
                f"B{base}:I{base + 9}",
                f"Step 2, j = {j}: original-feature and latent-sign conditions",
                _at(_meta_step, j),
            ),
            Check(
                sheet,
                f"X{base}:Y{base + 9}",
                f"j = {j}: in B1, in B2 after",
                _at(_meta_step_xy, j),
            ),
            Check(sheet, f"B{base + 10}:F{base + 10}", f"B1(a{subscript(j)}), B2", sets(j)),
            Check(
                sheet,
                f"B{base + 11}",
                f"Step 3 after j = {j}",
                _at(lambda s, j: [[_step3_text(s, j)]], j),
            ),
        ]
    checks.append(Check(sheet, "B84:C88", "Step 4: |B1|, |B2|, scores, class", decision))
    return checks


# ---------------------------------------------------------------- Meta-algorithm (All Objects)


def _all_objects_summary(s: ExperimentSubject) -> Grid:
    meta = s.training.meta
    table = _training_rows(s)
    classes = _classes(s)
    grid: Grid = []
    for t in range(s.trace.m):
        text = "(" + ", ".join(str(v) for v in table[t][:TUPLAM_SLOTS]) + ")"
        s1, s2 = float(meta.scores1[t]), float(meta.scores2[t])
        d = int(meta.decisions[t])
        grid.append(
            [
                text,
                int(meta.b1_sizes[t, -1]),
                int(meta.b2_sizes[t, -1]),
                s1,
                s2,
                round(s1 - s2, s.score_decimals),
                d,
                classes[t],
                "✅" if d == classes[t] else "❌",
            ]
        )
    return grid


def _all_objects_block(s: ExperimentSubject, t: int) -> Grid:
    steps = s.training.meta.steps(t)
    grid: Grid = []
    for j in range(TUPLAM_SLOTS):
        b1, b2 = _step_flags(steps, j)
        grid.append(
            [
                *[flag(v) for v in b1],
                int(np.count_nonzero(b1)),
                *[flag(v) for v in b2],
                int(np.count_nonzero(b2)),
            ]
        )
    return grid


def _all_objects_checks() -> list[ExperimentCheck]:
    sheet = "Meta-algorithm (All Objects)"
    checks: list[ExperimentCheck] = [
        Check(
            sheet,
            "B5:J14",
            "resubstitution: description, |B1|, |B2|, scores, predicted, true",
            _all_objects_summary,
        )
    ]
    for t in range(WORKBOOK_OBJECTS):
        top = 19 + 8 * t
        checks.append(
            Check(
                sheet,
                f"B{top}:W{top + TUPLAM_SLOTS - 1}",
                f"query {object_name(t)}: B1, B2 flags per step",
                _at(_all_objects_block, t),
            )
        )
    return checks


def model_checks() -> list[ExperimentCheck]:
    """The map of the experiment workbook, Steps 9–12, in sheet order."""
    greedy = [check for j in range(1, HAG_SHEETS + 1) for check in _greedy_checks(j)]
    return [
        *greedy,
        *_meta_dataset_checks(),
        *_brace_checks(),
        *_meta_checks(),
        *_all_objects_checks(),
    ]


def model_layout_problems(model: CSModel) -> list[str]:
    """Why the fitted model does not fit the workbook's Step 9–12 layout (empty if it does)."""
    problems = []
    n = len(model.trace.feature_names)
    if n != WORKBOOK_FEATURES:
        problems.append(f"{n} features (the new-object row has {WORKBOOK_FEATURES})")
    iterations = model.hag.iterations
    if len(iterations) > HAG_SHEETS or (len(iterations) == HAG_SHEETS and iterations[-1].go_on):
        problems.append(
            f"the HAG needs more than {HAG_SHEETS} iterations (the workbook has {HAG_SHEETS} "
            "Greedy upon Weight sheets)"
        )
    return problems
