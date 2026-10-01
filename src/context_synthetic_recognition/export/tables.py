"""Every table of a run as plain data — one source for CSV, JSON, Markdown, LaTeX and the report.

:func:`table_specs` lists the tables of a :class:`~.view.RunView` in pipeline order (Steps 1–12,
then the evaluation and the checks), each with its size, so that large tables can be left out
before they are built; :func:`run_tables` builds them. A :class:`Table` holds a title, column
names and rows of plain values (``None`` = undefined, the workbook's "—"); the renderers of
:mod:`.text` decide how numbers are written.

Names follow the article and the workbook: objects S₁ …, features x₁ …, synthetic features a₁ …,
latent features r₁ … (1-based, ADR-017).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from context_synthetic_recognition.config.presets import active_deviations
from context_synthetic_recognition.core.contributions import weight_ranks
from context_synthetic_recognition.core.meta import REFUSAL
from context_synthetic_recognition.core.trace import OperatorContext
from context_synthetic_recognition.evaluation.protocols import MethodPredictions, ProtocolResult
from context_synthetic_recognition.export.view import RunView
from context_synthetic_recognition.export.wording import (
    decision_label,
    decision_text,
    property_rows,
    protocol_title,
    set_text,
    tuplam_text,
    tuple_text,
)
from context_synthetic_recognition.notation import latent_name, subscript, synthetic_name

Cell = str | int | float | None
"""A table value: text, a number, or ``None`` for an undefined value (the workbook's "—")."""

DEFAULT_MAX_CELLS = 2_000_000
"""Tables with more cells are left out (and reported) unless the limit is raised."""

GROUP_INPUT = "1 · Input data"
GROUP_SCALE = "Step 1 · Scale unification"
GROUP_DISTANCES = "Step 2 · Distances"
GROUP_NEIGHBOURS = "Step 3 · Sorting"
GROUP_MU = "Step 4 · Same-class counts μ"
GROUP_PSI = "Step 5 · Ψ(r) — formula (5)"
GROUP_MEMBERSHIP = "Step 6 · Formulas (1), (2)"
GROUP_OMEGA = "Step 7 · Formulas (3), (4)"
GROUP_ETA = "Step 8 · Formula (6)"
GROUP_HAG = "Step 9 · Hierarchical agglomerative grouping"
GROUP_META_DATASET = "Step 10 · Meta-dataset"
GROUP_BRACE = "Step 11 · Preparation"
GROUP_META = "Step 12 · Classification"
GROUP_EVALUATION = "Evaluation"
GROUP_CHECKS = "Checks"
GROUP_SUMMARY = "Summary"


@dataclass(frozen=True)
class Table:
    """A table of a run as plain data."""

    key: str
    """File-system friendly name, e.g. ``psi`` or ``distances-rho-i``."""
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Cell, ...], ...]
    group: str
    """The stage of the pipeline it belongs to (the workbook's stage names)."""
    note: str = ""
    """What the values mean, with the article's formula number where there is one."""
    labels: int = 1
    """Number of leading label columns (object, feature, method … names)."""

    def __post_init__(self) -> None:
        """Check that every row has one value per column."""
        width = len(self.columns)
        for row in self.rows:
            if len(row) != width:
                raise ValueError(f"table {self.key}: a row has {len(row)} values, expected {width}")

    @property
    def cells(self) -> int:
        """Number of body cells."""
        return len(self.rows) * len(self.columns)


@dataclass(frozen=True)
class TableSpec:
    """A table that can be built, with its size known beforehand."""

    key: str
    group: str
    cells: int
    """Number of body cells the table will have."""
    build: Callable[[], Table] = field(repr=False)


@dataclass(frozen=True)
class TableSet:
    """The tables of a run and the ones left out because of their size."""

    tables: tuple[Table, ...]
    skipped: tuple[TableSpec, ...] = ()

    def __iter__(self) -> Iterator[Table]:
        """Iterate over the tables in pipeline order."""
        return iter(self.tables)

    def get(self, key: str) -> Table | None:
        """The table with this key, or ``None`` (not part of the run, or left out)."""
        return next((t for t in self.tables if t.key == key), None)

    def __getitem__(self, key: str) -> Table:
        """The table with this key.

        Raises:
            KeyError: No such table.
        """
        table = self.get(key)
        if table is None:
            raise KeyError(key)
        return table


# ---------------------------------------------------------------- helpers


def slug(text: str) -> str:
    """A file-system friendly form of a label: ``ρ_I`` → ``rho-i``."""
    names = {"ρ": "rho", "ω": "omega", "η": "eta", "Ψ": "psi", "μ": "mu", "χ": "chi", "θ": "theta"}
    out: list[str] = []
    for char in text:
        if char in names:
            out.append(names[char])
        elif char.isascii() and char.isalnum():
            out.append(char.lower())
        else:
            out.append("-")
    return "-".join(part for part in "".join(out).split("-") if part) or "x"


def number(value: Any) -> Cell:
    """A numpy or Python number as a plain cell: NaN and ±∞ become ``None`` / ``±∞``."""
    if isinstance(value, bool | np.bool_):
        return int(value)
    if isinstance(value, int | np.integer):
        return int(value)
    x = float(value)
    if math.isnan(x):
        return None
    if math.isinf(x):
        return "+∞" if x > 0 else "−∞"
    return x


def plain(value: Any) -> Cell:
    """A data value: integral values as integers (a nominal code 3, not 3.0)."""
    x = float(value)
    return int(x) if x.is_integer() else x


def _feature_headers(view: RunView) -> tuple[str, ...]:
    return tuple(f.name for f in view.trace.features)


def _object_table(
    view: RunView,
    key: str,
    title: str,
    group: str,
    headers: Sequence[str],
    matrix: Any,
    note: str = "",
    convert: Callable[[Any], Cell] = number,
) -> Table:
    """One row per training object: №, the columns of ``matrix``, the class."""
    ids, labels = view.trace.object_ids, view.labels
    values = np.asarray(matrix)
    rows = tuple(
        (ids[t], *(convert(v) for v in values[t]), labels[t]) for t in range(values.shape[0])
    )
    return Table(key, title, ("№", *headers, "Class"), rows, group, note)


def _per_feature(view: RunView, attribute: str) -> Any:
    return np.column_stack([getattr(f, attribute) for f in view.trace.features])


# ---------------------------------------------------------------- Steps 1–3


def _dataset(view: RunView) -> Table:
    d = view.dataset
    return _object_table(
        view,
        "dataset",
        f"{d.name}: original sample",
        GROUP_INPUT,
        d.feature_names,
        d.X,
        "Objects × features with the class label; nominal features hold category codes.",
        plain,
    )


def _features(view: RunView) -> Table:
    d, scaling = view.dataset, view.trace.scaling
    low = scaling.statistics.get("min")
    high = scaling.statistics.get("max")
    rows = tuple(
        (
            name,
            "quantitative (I)" if q else "nominal (J)",
            int(q),
            plain(low[j]) if low is not None and q else None,
            plain(high[j]) if high is not None and q else None,
            int(np.unique(d.X[:, j]).size),
        )
        for j, (name, q) in enumerate(zip(d.feature_names, d.quantitative, strict=True))
    )
    return Table(
        "features",
        "Features: type, training minimum and maximum, distinct values",
        ("Feature", "Type", "Flag (1 = I)", "min", "max", "Distinct values"),
        rows,
        GROUP_INPUT,
        "min and max are taken over the training objects (quantitative features only).",
    )


def _normalized(view: RunView) -> Table:
    return _object_table(
        view,
        "normalized",
        "Normalized dataset: quantitative features mapped to [0, 1], nominal features unchanged",
        GROUP_SCALE,
        view.trace.feature_names,
        view.trace.normalized,
        "x′ = (x − min)/(max − min) for j ∈ I with the training min and max (Step 1).",
        plain,
    )


def _distances(view: RunView, operator: OperatorContext) -> Table:
    ids = view.trace.object_ids
    rows = tuple((ids[i], *(plain(v) for v in operator.distances[i])) for i in range(view.trace.m))
    return Table(
        f"distances-{slug(operator.label)}",
        f"Distances of the operator {operator.label} (row = target object Sᵢ, column = Sⱼ)",
        (f"{operator.label}", *ids),
        rows,
        GROUP_DISTANCES,
        f"Metric {operator.metric} on {operator.features.size} feature(s); rounded so that equal "
        "distances tie exactly.",
    )


def neighbour_depth(view: RunView) -> int:
    """Ranks listed in the neighbour tables: up to the largest permitted k and one beyond."""
    return int(min(view.trace.m - 1, max(view.trace.permitted_k.ks) + 1))


def _neighbours(view: RunView, operator: OperatorContext) -> Table:
    trace = view.trace
    depth = neighbour_depth(view)
    ids, labels, y = trace.object_ids, view.labels, trace.class_index
    order = operator.order[:, :depth]
    distance = operator.sorted_distances[:, :depth]
    mu = operator.same_class_running(y)[:, :depth]
    chi1 = operator.k1_running(y)[:, :depth]
    rows = tuple(
        (
            ids[i],
            rank + 1,
            ids[int(order[i, rank])],
            int(order[i, rank]) + 1,
            plain(distance[i, rank]),
            labels[int(order[i, rank])],
            int(y[int(order[i, rank])] == y[i]),
            int(mu[i, rank]),
            int(chi1[i, rank]),
        )
        for i in range(trace.m)
        for rank in range(depth)
    )
    return Table(
        f"neighbours-{slug(operator.label)}",
        f"Neighbours of every object in ascending {operator.label} (self excluded)",
        (
            "Object",
            "Rank",
            "Neighbour",
            "Original index",
            f"{operator.label}(object, neighbour)",
            "Class",
            "Same class",
            "Same-class count μ (1 … rank)",
            "K1 count χ₁ (1 … rank)",
        ),
        rows,
        GROUP_NEIGHBOURS,
        "Order by (distance, original index): equal distances go to the smaller index. Ranks up "
        f"to {depth} (the largest permitted k and one beyond).",
        labels=2,
    )


# ---------------------------------------------------------------- Steps 4–8


def _named(view: RunView) -> tuple[str, ...]:
    return tuple(f"{f.name} ({f.operator_label}, k = {f.k})" for f in view.trace.features)


def _mu(view: RunView) -> Table:
    return _object_table(
        view,
        "same-class-counts",
        "Same-class counts μ: neighbours of the object's own class among the k nearest",
        GROUP_MU,
        _named(view),
        _per_feature(view, "mu"),
        "Training-side gradation of formulas (1)–(4); it uses the class of the object.",
    )


def _chi1(view: RunView) -> Table:
    return _object_table(
        view,
        "chi1",
        "χ₁(Sⱼ, k): K1 objects among the k nearest neighbours",
        GROUP_PSI,
        _named(view),
        _per_feature(view, "chi1"),
        "The class of Sⱼ itself is not used; χ₂ = k − χ₁.",
    )


def _psi(view: RunView) -> Table:
    return _object_table(
        view,
        "psi",
        "Ψ(r): synthetic features aᵤ(Sⱼ) ∈ {1, 2} by formula (5)",
        GROUP_PSI,
        _named(view),
        view.trace.values,
        "aᵤ = 1 if χ₁ > [k/2], aᵤ = 2 if χ₂ > [k/2] — the majority class of the k nearest.",
    )


def _membership(view: RunView) -> Table:
    rows: list[tuple[Cell, ...]] = []
    for f in view.trace.features:
        table = f.membership
        for g in range(table.gradations.size):
            rows.append(
                (
                    f.name,
                    f.operator_label,
                    f.k,
                    int(table.gradations[g]),
                    int(table.d1[g]),
                    int(table.d2[g]),
                    int(table.n[g]),
                    number(table.share1[g]),
                    number(table.share2[g]),
                    number(table.f[g]),
                    number(table.weighted[g]),
                )
            )
    return Table(
        "membership",
        "Membership f_k(μ) — formula (1) — and the terms of the stability g_k — formula (2)",
        (
            "Feature",
            "Operator",
            "k",
            "μ",
            "d₁ₖ(μ)",
            "d₂ₖ(μ)",
            "n(μ)",
            "d₁ₖ/|K1|",
            "d₂ₖ/|K2|",
            "f_k(μ)",
            "n·max(f, 1−f)",
        ),
        tuple(rows),
        GROUP_MEMBERSHIP,
        "f_k(μ) = (d₁ₖ/|K1|)/(d₁ₖ/|K1| + d₂ₖ/|K2|); undefined where no object has the gradation.",
        labels=4,
    )


def _synthetic_features(view: RunView) -> Table:
    trace = view.trace
    ranks = weight_ranks(trace.weights)
    rows = tuple(
        (
            f.name,
            f.operator_label,
            f.k,
            number(f.stability),
            None if f.boundary.q2 is None else f.boundary.q2,
            None if f.boundary.q1 is None else f.boundary.q1,
            number(f.boundary.G),
            int(np.count_nonzero(f.correct)),
            number(f.omega),
            number(f.weight),
            int(ranks[u]),
            int(f.alpha[0, 0]),
            int(f.alpha[0, 1]),
            int(f.alpha[1, 0]),
            int(f.alpha[1, 1]),
            number(f.eta[0]),
            number(f.eta[1]),
        )
        for u, f in enumerate(trace.features)
    )
    return Table(
        "synthetic-features",
        "Synthetic features: stability g, boundary G, informativeness ω and contributions η",
        (
            "Feature",
            "Operator",
            "k",
            "g_k (2)",
            "q₂ = max{f < 0.5}",
            "q₁ = min{f > 0.5}",
            "G_k (3)",
            "Correctly placed",
            "ω (4)",
            "Weight",
            "Rank",
            "α¹₁",
            "α²₁",
            "α¹₂",
            "α²₂",
            "η(1) (6)",
            "η(2) (6)",
        ),
        rows,
        GROUP_OMEGA,
        "G_k = (q₁ + q₂)/2 (0.5 if a side is empty); ω = correctly placed objects / m; "
        "η(j) = weight·(α¹ⱼ/|K1| − α²ⱼ/|K2|).",
        labels=3,
    )


def _object_membership(view: RunView) -> Table:
    return _object_table(
        view,
        "object-membership",
        "g(S, k) = f_k(μ_S): membership of every object at its own gradation",
        GROUP_OMEGA,
        _feature_headers(view),
        _per_feature(view, "object_membership"),
    )


def _correct_side(view: RunView) -> Table:
    return _object_table(
        view,
        "correct-side",
        "Correct side of the boundary: 1 if (S ∈ K1 and g > G_k) or (S ∈ K2 and g < G_k)",
        GROUP_OMEGA,
        _feature_headers(view),
        _per_feature(view, "correct"),
    )


def _contributions(view: RunView) -> Table:
    return _object_table(
        view,
        "contributions",
        "Ψ(r) with contribution values ηᵤ(a_tu) — the input of the HAG",
        GROUP_ETA,
        _feature_headers(view),
        view.trace.contributions,
        "Formula (6): positive values point to K1, negative ones to K2.",
    )


def _bit_masks(view: RunView) -> Table:
    masks = view.trace.bit_masks
    headers = tuple(f"{b.operator_label} · mask" for b in masks)
    matrix = np.column_stack([b.masks for b in masks])
    return _object_table(
        view,
        "bit-masks",
        "Task 2: bit representations by the majority rule, read as a binary number",
        GROUP_MEMBERSHIP,
        headers,
        matrix,
        "Bit = 1 if μ > k/2, one bit per permitted k, the first permitted k most significant.",
    )


# ---------------------------------------------------------------- Step 9


def _hag_iterations(view: RunView) -> Table:
    hag = view.hag
    rows: list[tuple[Cell, ...]] = [
        (
            0,
            "STEP 2 · u = argmax ω",
            synthetic_name(hag.first),
            None,
            None,
            1,
            tuplam_text((hag.first,)),
            "",
        )
    ]
    for it in hag.iterations:
        rows.append(
            (
                it.number,
                f"STEP 3–4 · iteration {it.number}",
                None if it.q is None else synthetic_name(it.q),
                number(it.cr1),
                number(it.crit),
                len(it.tuplam_after),
                tuplam_text(it.tuplam_after),
                "continue" if it.go_on else f"stop: {it.stop.text if it.stop else ''}",
            )
        )
    return Table(
        "hag-iterations",
        "Hierarchical agglomerative grouping: the feature added in every iteration",
        (
            "Iteration",
            "Step",
            "Feature added",
            "cr1 = min θ/γ",
            "crit",
            "|TUPLAM|",
            "TUPLAM",
            "STEP 5",
        ),
        tuple(rows),
        GROUP_HAG,
        "STEP 5 continues while |TUPLAM| < ϰ, crit > δ and P ≠ ∅; the latent feature r_j is R "
        "after iteration j.",
        labels=2,
    )


def _hag_candidates(view: RunView) -> Table:
    features = view.trace.features
    rows: list[tuple[Cell, ...]] = []
    for it in view.hag.iterations:
        for position, u in enumerate(it.candidates):
            f = features[int(u)]
            rows.append(
                (
                    it.number,
                    f.name,
                    f.operator_label,
                    f.k,
                    number(it.theta[position]),
                    number(it.gamma[position]),
                    number(it.ratio[position]),
                    int(it.q is not None and int(u) == it.q),
                )
            )
    return Table(
        "hag-candidates",
        "HAG STEP 3: θ, γ and θ/γ of every candidate in every iteration",
        ("Iteration", "Candidate", "Operator", "k", "θ", "γ", "θ/γ", "Chosen q"),
        tuple(rows),
        GROUP_HAG,
        "θ = Σ |bₜ − M(own class)|, γ = Σ |bₜ − M(other class)|; q = argmin θ/γ (first on ties).",
        labels=4,
    )


def _hag_chosen_blocks(view: RunView) -> Table:
    hag, ids, labels = view.hag, view.trace.object_ids, view.labels
    rows: list[tuple[Cell, ...]] = []
    for index, it in enumerate(hag.iterations):
        if it.q is None or it.latent is None:
            continue
        scan = hag.scan(index, it.q)
        for t in range(hag.m):
            rows.append(
                (
                    it.number,
                    synthetic_name(it.q),
                    ids[t],
                    labels[t],
                    number(scan.entering[t]),
                    number(scan.eta[t]),
                    number(scan.b[t]),
                    number(scan.majorized[t]),
                    number(scan.sum1[t]),
                    number(scan.sum2[t]),
                    number(scan.centre1[t]),
                    number(scan.centre2[t]),
                    number(scan.to_own[t]),
                    number(scan.to_other[t]),
                    number(it.latent[t]),
                )
            )
    return Table(
        "hag-chosen-blocks",
        "HAG STEP 3–4: the block of the chosen feature q in every iteration",
        (
            "Iteration",
            "q",
            "Object",
            "Class",
            "R(Sₜ)",
            "η_q(a_tq)",
            "bₜ = R + η_q",
            "bₜ (majorized)",
            "Σ K1 (running)",
            "Σ K2 (running)",
            "M₁ used",
            "M₂ used",
            "|b − M own|",
            "|b − M other|",
            "r_j = R(Sₜ) new",
        ),
        tuple(rows),
        GROUP_HAG,
        "⚠ The class centres (running or final) and the number of majorizer passes in STEP 4 are "
        "the two template/article switches.",
        labels=4,
    )


# ---------------------------------------------------------------- Steps 10–12


def _meta_dataset(view: RunView) -> Table:
    data = view.model.meta_dataset
    return _object_table(
        view,
        "meta-dataset",
        "Dataset for the meta-algorithm: Y = (y₀, …, y_p, r₁, …, r_p)",
        GROUP_META_DATASET,
        data.columns,
        data.Y,
        "y_j: contribution values of the TUPLAM features in TUPLAM order; r_j: latent features.",
    )


def _description(view: RunView) -> Table:
    model = view.model
    description = model.description
    headers = (
        *(f"aᵢ{subscript(j)} ({synthetic_name(u)})" for j, u in enumerate(model.tuplam)),
        *(f"dᵢ{subscript(j + 1)} ({latent_name(j)})" for j in range(model.p)),
    )
    matrix = np.hstack([description.gradations.astype(np.float64), description.latent])
    return _object_table(
        view,
        "training-description",
        "Training description of the meta-algorithm: gradations aᵢⱼ and latent features dᵢⱼ",
        GROUP_BRACE,
        headers,
        matrix,
        convert=plain,
    )


def _resubstitution(view: RunView) -> Table:
    meta, trace = view.training.meta, view.trace
    decimals = view.config.evaluation.score_decimals
    classes, labels = trace.classes, view.labels
    rows = []
    for t in range(trace.m):
        decision = int(meta.decisions[t])
        predicted = decision_label(decision, classes)
        rows.append(
            (
                trace.object_ids[t],
                tuple_text([int(v) for v in meta.queries[t]]),
                int(meta.b1_sizes[t, -1]),
                int(meta.b2_sizes[t, -1]),
                number(meta.scores1[t]),
                number(meta.scores2[t]),
                number(round(float(meta.scores[t]), decimals)),
                predicted,
                labels[t],
                int(decision != REFUSAL and predicted == labels[t]),
            )
        )
    return Table(
        "resubstitution",
        "Meta-algorithm for every training object (resubstitution, Definition 2)",
        (
            "№",
            "(a₀, …, a_p)",
            "|B1(a_p)|",
            "|B2(a_p)|",
            "score₁",
            "score₂",
            "score₁ − score₂",
            "Predicted",
            "True",
            "Correct",
        ),
        tuple(rows),
        GROUP_META,
        "score₁ = |B1(a_p)|/|K1|, score₂ = |B2(a_p)|/|K2|; predicted 0 = refusal (equal scores).",
    )


def _new_object(view: RunView) -> Table:
    demo, trace = view.new_object, view.trace
    representation = demo.classification.representation
    assert representation is not None
    context = representation.context
    rows = tuple(
        (
            name,
            int(q),
            plain(demo.values[j]),
            plain(context.normalized[0, j]),
        )
        for j, (name, q) in enumerate(
            zip(trace.feature_names, trace.scaling.quantitative, strict=True)
        )
    )
    excluded = (
        "a new object"
        if demo.exclude is None
        else f"{trace.object_ids[demo.exclude]} left out of its own context"
    )
    return Table(
        "new-object",
        f"New object S ({excluded}): original and unified values",
        ("Feature", "Type (1 = I)", "x", "x′"),
        rows,
        GROUP_BRACE,
        "Scale unification uses the training min and max; the class of S is not an input.",
    )


def _new_object_features(view: RunView) -> Table:
    demo, trace = view.new_object, view.trace
    representation = demo.classification.representation
    assert representation is not None
    context = representation.context
    position = {u: j for j, u in enumerate(view.model.tuplam)}
    rows = tuple(
        (
            f.name,
            f.operator_label,
            f.k,
            int(context.chi1[0, u]),
            f.k - int(context.chi1[0, u]),
            int(context.values[0, u]),
            None if u not in position else f"a{subscript(position[u])}",
        )
        for u, f in enumerate(trace.features)
    )
    return Table(
        "new-object-features",
        "New object S: χ₁, χ₂ and its synthetic features by formula (5)",
        ("Feature", "Operator", "k", "χ₁", "χ₂ = k − χ₁", "aᵤ(S)", "TUPLAM position"),
        rows,
        GROUP_BRACE,
        "Formula (5) needs only the classes of the neighbours (Theorem of the article).",
        labels=3,
    )


def _new_object_steps(view: RunView) -> Table:
    demo, trace = view.new_object, view.trace
    meta = demo.classification.meta
    steps = meta.steps(0)
    ids = trace.object_ids
    rows: list[tuple[Cell, ...]] = []
    for j, u in enumerate(view.model.tuplam):
        rows.append(
            (
                f"Step {1 if j == 0 else 2} · j = {j}",
                synthetic_name(u),
                int(steps.query[j]),
                int(steps.b1_sizes[j]),
                int(steps.b2_sizes[j]),
                set_text([ids[int(i)] for i in steps.b1(j)]),
                set_text([ids[int(i)] for i in steps.b2(j)]),
            )
        )
    decision = decision_text(demo.decision, trace.classes)
    return Table(
        "new-object-steps",
        f"Meta-algorithm for the new object S: B1 and B2 at every step — {decision}",
        ("Step", "TUPLAM feature", "aⱼ", "|B1(aⱼ)|", "|B2(aⱼ)|", "B1(aⱼ)", "B2(aⱼ)"),
        tuple(rows),
        GROUP_META,
        f"Step 4: score₁ = {float(meta.scores1[0]):.6g}, score₂ = {float(meta.scores2[0]):.6g} "
        f"⇒ {decision}.",
    )


# ---------------------------------------------------------------- evaluation


def _methods(view: RunView) -> Iterator[tuple[ProtocolResult, MethodPredictions]]:
    for protocol in view.result.protocols:
        for method in protocol.methods:
            yield protocol, method


def _metrics(view: RunView) -> Table:
    rows = []
    for protocol, method in _methods(view):
        m = view.result.metrics(method)
        auc = view.result.auc(method)
        rows.append(
            (
                protocol_title(protocol.protocol),
                method.method,
                m.tp,
                m.tn,
                m.fp,
                m.fn,
                m.refusals,
                m.accuracy,
                m.coverage,
                m.accuracy_answered,
                m.macro_precision,
                m.macro_recall,
                m.macro_f1,
                number(auc),
                m.label,
            )
        )
    positive = view.trace.classes[view.result.positive - 1]
    return Table(
        "metrics",
        "Accuracy, coverage, macro precision / recall / F1 and AUC of every method",
        (
            "Evaluation",
            "Method",
            "TP",
            "TN",
            "FP",
            "FN",
            "Refusals",
            "Accuracy",
            "Coverage",
            "Accuracy on answered",
            "Macro precision",
            "Macro recall",
            "Macro F1",
            "AUC",
            "Correct / n",
        ),
        tuple(rows),
        GROUP_EVALUATION,
        f"Positive class = {positive}; a refusal (0) counts as an error in the accuracy. AUC = "
        "Mann–Whitney statistic of score₁ − score₂ (rounded scores, ties ½).",
        labels=2,
    )


def _class_metrics(view: RunView) -> Table:
    classes = view.trace.classes
    rows = []
    for protocol, method in _methods(view):
        m = view.result.metrics(method)
        for scores in m.classes:
            rows.append(
                (
                    protocol_title(protocol.protocol),
                    method.method,
                    classes[scores.code - 1],
                    scores.predicted,
                    scores.correct,
                    scores.actual,
                    scores.precision,
                    scores.recall,
                    scores.f1,
                )
            )
    return Table(
        "class-metrics",
        "Precision, recall and F1 per class",
        (
            "Evaluation",
            "Method",
            "Class",
            "Predicted as class",
            "True positives",
            "Actual members",
            "Precision",
            "Recall",
            "F1",
        ),
        tuple(rows),
        GROUP_EVALUATION,
        "Precision = TP / predicted as the class; recall = TP / actual members; "
        "F1 = 2·P·R/(P + R); 0 when undefined.",
        labels=3,
    )


def _confusion(view: RunView, protocol: ProtocolResult) -> Table:
    classes = view.trace.classes
    m = view.result.metrics(protocol.predictions)
    matrix = m.matrix
    rows: list[tuple[Cell, ...]] = []
    for c in range(2):
        total = int(matrix[c].sum())
        rows.append(
            (
                f"Class {classes[c]}",
                *(int(v) for v in matrix[c]),
                total,
                int(matrix[c, c]) / total if total else None,
            )
        )
    totals = matrix.sum(axis=0)
    rows.append(("Total", *(int(v) for v in totals), int(matrix.sum()), None))
    rows.append(
        (
            "Precision",
            *(int(matrix[c, c]) / int(totals[c]) if totals[c] else None for c in range(2)),
            None,
            None,
            None,
        )
    )
    name = protocol_title(protocol.protocol)
    return Table(
        f"confusion-{protocol.protocol}",
        f"Confusion matrix of the CS-model · {name} (actual class × predicted class)",
        (
            "Actual \\ Predicted",
            f"Class {classes[0]}",
            f"Class {classes[1]}",
            "Refusal (0)",
            "Total",
            "Recall",
        ),
        tuple(rows),
        GROUP_EVALUATION,
        "Diagonal = correct decisions; refusals are shown separately.",
    )


def _roc(view: RunView, protocol: ProtocolResult) -> Table:
    curve = protocol.predictions.roc(view.result.positive, view.config.evaluation.score_decimals)
    rows = tuple(
        (number(curve.thresholds[i]), number(curve.tpr[i]), number(curve.fpr[i]))
        for i in range(curve.thresholds.size)
    )
    name = protocol_title(protocol.protocol)
    return Table(
        f"roc-{protocol.protocol}",
        f"ROC points of the CS-model · {name} (AUC = {curve.auc:.4f})",
        ("Threshold", "TPR (recall)", "FPR"),
        rows,
        GROUP_EVALUATION,
        "TPR and FPR at score ≥ threshold, for every score in decreasing order.",
    )


def _predictions(view: RunView, protocol: ProtocolResult) -> Table:
    trace = view.trace
    classes, ids = trace.classes, trace.object_ids
    decimals = view.config.evaluation.score_decimals
    cs = protocol.predictions
    rows = []
    for e in range(cs.objects.size):
        rows.append(
            (
                ids[int(cs.objects[e])],
                int(cs.repeats[e]),
                classes[int(cs.truth[e]) - 1],
                decision_label(int(cs.decisions[e]), classes),
                number(round(float(cs.scores[e]), decimals)),
                *(decision_label(int(b.decisions[e]), classes) for b in protocol.baselines),
            )
        )
    name = protocol_title(protocol.protocol)
    return Table(
        f"predictions-{protocol.protocol}",
        f"Predictions · {name}",
        (
            "Object",
            "Repeat",
            "True class",
            "CS-model",
            "score₁ − score₂",
            *(b.method for b in protocol.baselines),
        ),
        tuple(rows),
        GROUP_EVALUATION,
        "Predicted class per method; 0 = refusal.",
        labels=2,
    )


def _folds(view: RunView, protocol: ProtocolResult) -> Table:
    ids = view.trace.object_ids
    rows = []
    for fold in protocol.folds:
        sizes: tuple[Cell, Cell] = fold.class_sizes if fold.class_sizes else (None, None)
        rows.append(
            (
                fold.number + 1,
                fold.split.repeat,
                " ".join(ids[int(i)] for i in fold.split.test),
                *sizes,
                ", ".join(map(str, fold.permitted_k)) if fold.permitted_k else None,
                fold.r,
                set_text(list(fold.tuplam_names or ())) if fold.tuplam_names else None,
                fold.undefined or "",
            )
        )
    name = protocol_title(protocol.protocol)
    return Table(
        f"folds-{protocol.protocol}",
        f"Folds · {name}: the whole pipeline re-fitted on the training part of every fold",
        (
            "Fold",
            "Repeat",
            "Held out",
            "|K1| in fold",
            "|K2| in fold",
            "Permitted k",
            "r = |Ψ(r)|",
            "TUPLAM (operator·k)",
            "Undefined",
        ),
        tuple(rows),
        GROUP_EVALUATION,
        "The k range adapts to the class sizes of every fold; an undefined fold refuses its "
        "objects.",
    )


def _margins(view: RunView) -> Table:
    analysis = view.result.margins
    ids = view.trace.object_ids
    rows: list[tuple[Cell, ...]] = []
    for j, (a, b) in enumerate(
        zip(analysis.with_majorizer, analysis.without_majorizer, strict=True)
    ):
        for what, m in (("with majorizer", a), ("without majorizer", b)):
            rows.append(
                (
                    latent_name(j),
                    what,
                    m.boundary,
                    m.left,
                    ids[m.left_object],
                    m.right,
                    ids[m.right_object],
                    m.width,
                    m.correct,
                    a.width - b.width if m is a else None,
                )
            )
    return Table(
        "margins",
        "Margins of the latent features with and without the majorizer",
        (
            "Latent feature",
            "Variant",
            "b (boundary)",
            "left: max K2",
            "object",
            "right: min K1",
            "object",
            "Margin width",
            "ŷ correct",
            "Gain (with − without)",
        ),
        tuple(rows),
        GROUP_EVALUATION,
        "b = (min K1 + max K2)/2; width = min K1 − max K2 (> 0: the classes are separated); "
        "ŷ = K1 if d > b.",
        labels=2,
    )


def _object_margins(view: RunView) -> Table:
    analysis = view.result.margins
    headers: list[str] = []
    columns: list[Any] = []
    for j, (a, b) in enumerate(
        zip(analysis.with_majorizer, analysis.without_majorizer, strict=True)
    ):
        name = latent_name(j)
        headers += [name, f"margin {name}", f"{name} (no ϕ)", f"margin {name} (no ϕ)"]
        columns += [a.values, a.object_margins, b.values, b.object_margins]
    return _object_table(
        view,
        "object-margins",
        "Object margins mᵢ = yᵢ·(dᵢ − b) of every latent feature, with and without the majorizer",
        GROUP_EVALUATION,
        headers,
        np.column_stack(columns),
        "yᵢ = +1 for K1 and −1 for K2; mᵢ > 0: the object is on its correct side.",
    )


def _properties(view: RunView) -> Table:
    rows = tuple((r.section, r.check, r.result, r.status) for r in property_rows(view))
    return Table(
        "properties",
        "Model properties: Definitions 1–4 and the Theorem",
        ("Definition", "Check", "Result", "Status"),
        rows,
        GROUP_CHECKS,
        labels=2,
    )


def _operator_pairs(view: RunView) -> Table:
    trace = view.trace
    labels = [o.label for o in trace.operators]
    rows = tuple(
        (
            f"{labels[p.first]} vs {labels[p.second]}",
            p.k,
            p.equal_sets,
            trace.m,
        )
        for p in view.result.properties.operator_pairs
    )
    return Table(
        "operator-pairs",
        "Definition 6: objects whose k-neighbour sets coincide under two operators",
        ("Operator pair", "k", "Objects with equal sets", "m"),
        rows,
        GROUP_CHECKS,
        labels=2,
    )


def _boundary_ties(view: RunView) -> Table:
    trace = view.trace
    rows = tuple(
        (trace.operators[t.operator].label, t.k, t.objects, trace.m)
        for t in view.result.properties.boundary_ties
    )
    return Table(
        "boundary-ties",
        "Ties at the k boundary: the k-th and (k + 1)-th neighbours at the same distance",
        ("Operator", "k", "Objects with a tie", "m"),
        rows,
        GROUP_CHECKS,
        'With a tie the rule "smaller original index first" decides the neighbourhood.',
        labels=2,
    )


def _sensitivity(view: RunView) -> Table:
    variants = view.sensitivity or ()
    rows = tuple(
        (
            v.centres.value,
            v.step4_passes,
            v.label,
            v.tuplam,
            ", ".join(f"{c:.4f}" for c in v.crit),
            v.resubstitution,
            v.leave_one_out,
            number(v.auc_resubstitution),
            number(v.auc_leave_one_out),
        )
        for v in variants
    )
    return Table(
        "sensitivity",
        "Sensitivity to the two template/article switches",
        (
            "Class centres",
            "STEP 4 passes",
            "Setting",
            "TUPLAM",
            "crit per iteration",
            "Resubstitution accuracy",
            "LOO accuracy",
            "AUC resub.",
            "AUC LOO",
        ),
        rows,
        GROUP_EVALUATION,
        "⚠ Template: θ and γ from running partial class means, the majorizer applied twice in "
        "STEP 4. Article: final class means, one pass.",
        labels=3,
    )


def summary_rows(view: RunView) -> list[tuple[str, Cell]]:
    """The key figures of a run, in the order of the workbook's *Overview*."""
    trace, hag, result = view.trace, view.hag, view.result
    d = view.dataset
    quantitative = int(np.count_nonzero(d.quantitative))
    rows: list[tuple[str, Cell]] = [
        ("Dataset", d.name),
        ("Objects m", d.m),
        ("Features n", f"{d.n} ({quantitative} quantitative, {d.n - quantitative} nominal)"),
        (f"|K1| (class {trace.classes[0]})", trace.class_sizes[0]),
        (f"|K2| (class {trace.classes[1]})", trace.class_sizes[1]),
        ("Base operators", ", ".join(o.label for o in trace.operators)),
        (f"Permitted k ({trace.permitted_k.rule})", _k_text(trace.permitted_k.ks)),
        ("Synthetic features r = |Ψ(r)|", trace.r),
        ("TUPLAM (selected synthetic features, in order)", hag.label),
        ("Latent (additional) features p", hag.p),
        ("crit per iteration", ", ".join(f"{c:.4f}" for c in hag.crit) or None),
        ("HAG stopped because", hag.stop.text),
    ]
    for protocol in result.protocols:
        name = protocol_title(protocol.protocol)
        metrics = result.metrics(protocol.predictions)
        rows.append((f"Accuracy — {name}", metrics.accuracy))
        rows.append((f"Coverage — {name}", metrics.coverage))
        rows.append((f"AUC — {name}", number(result.auc(protocol.predictions))))
    demo = view.new_object
    what = (
        "new object"
        if demo.exclude is None
        else f"{trace.object_ids[demo.exclude]} left out of its context"
    )
    rows.append((f"New object ({what}): class", decision_text(demo.decision, trace.classes)))
    for deviation in active_deviations(view.config):
        rows.append((f"⚠ Template calculation ({deviation.adr})", deviation.statement))
    return rows


def _k_text(ks: tuple[int, ...]) -> str:
    if len(ks) <= 12:
        return ", ".join(map(str, ks))
    return f"{ks[0]}, {ks[1]}, {ks[2]}, …, {ks[-1]} ({len(ks)} values)"


def _summary(view: RunView) -> Table:
    return Table(
        "summary",
        "Key results",
        ("Figure", "Value"),
        tuple(summary_rows(view)),
        GROUP_SUMMARY,
    )


# ---------------------------------------------------------------- the list of tables


def table_specs(view: RunView) -> list[TableSpec]:
    """The tables of a run in pipeline order, with their sizes, not yet built."""
    trace, hag, result = view.trace, view.hag, view.result
    m, n, r, p = trace.m, len(trace.feature_names), trace.r, hag.p
    depth = neighbour_depth(view)
    specs: list[TableSpec] = []

    def add(key: str, group: str, cells: int, build: Callable[[], Table]) -> None:
        specs.append(TableSpec(key, group, cells, build))

    add("summary", GROUP_SUMMARY, 80, lambda: _summary(view))
    add("dataset", GROUP_INPUT, m * (n + 2), lambda: _dataset(view))
    add("features", GROUP_INPUT, 6 * n, lambda: _features(view))
    add("normalized", GROUP_SCALE, m * (n + 2), lambda: _normalized(view))
    for operator in trace.operators:
        add(
            f"distances-{slug(operator.label)}",
            GROUP_DISTANCES,
            m * (m + 1),
            lambda o=operator: _distances(view, o),  # type: ignore[misc]
        )
    for operator in trace.operators:
        add(
            f"neighbours-{slug(operator.label)}",
            GROUP_NEIGHBOURS,
            9 * m * depth,
            lambda o=operator: _neighbours(view, o),  # type: ignore[misc]
        )
    wide = m * (r + 2)
    add("same-class-counts", GROUP_MU, wide, lambda: _mu(view))
    add("chi1", GROUP_PSI, wide, lambda: _chi1(view))
    add("psi", GROUP_PSI, wide, lambda: _psi(view))
    gradations = sum(f.membership.gradations.size for f in trace.features)
    add("membership", GROUP_MEMBERSHIP, 11 * gradations, lambda: _membership(view))
    if trace.bit_masks:
        add(
            "bit-masks",
            GROUP_MEMBERSHIP,
            m * (len(trace.bit_masks) + 2),
            lambda: _bit_masks(view),
        )
    add("synthetic-features", GROUP_OMEGA, 17 * r, lambda: _synthetic_features(view))
    add("object-membership", GROUP_OMEGA, wide, lambda: _object_membership(view))
    add("correct-side", GROUP_OMEGA, wide, lambda: _correct_side(view))
    add("contributions", GROUP_ETA, wide, lambda: _contributions(view))
    iterations = len(hag.iterations)
    add("hag-iterations", GROUP_HAG, 8 * (iterations + 1), lambda: _hag_iterations(view))
    candidates = sum(it.candidates.size for it in hag.iterations)
    add("hag-candidates", GROUP_HAG, 8 * candidates, lambda: _hag_candidates(view))
    add("hag-chosen-blocks", GROUP_HAG, 15 * m * p, lambda: _hag_chosen_blocks(view))
    add("meta-dataset", GROUP_META_DATASET, m * (2 * p + 3), lambda: _meta_dataset(view))
    add("training-description", GROUP_BRACE, m * (2 * p + 3), lambda: _description(view))
    add("new-object", GROUP_BRACE, 4 * n, lambda: _new_object(view))
    add("new-object-features", GROUP_BRACE, 7 * r, lambda: _new_object_features(view))
    add("resubstitution", GROUP_META, 10 * m, lambda: _resubstitution(view))
    add("new-object-steps", GROUP_META, 7 * (p + 1), lambda: _new_object_steps(view))
    methods = sum(len(protocol.methods) for protocol in result.protocols)
    add("metrics", GROUP_EVALUATION, 15 * methods, lambda: _metrics(view))
    add("class-metrics", GROUP_EVALUATION, 18 * methods, lambda: _class_metrics(view))
    for protocol in result.protocols:
        name = protocol.protocol
        entries = int(protocol.predictions.objects.size)
        add(
            f"confusion-{name}",
            GROUP_EVALUATION,
            24,
            lambda q=protocol: _confusion(view, q),  # type: ignore[misc]
        )
        add(
            f"roc-{name}",
            GROUP_EVALUATION,
            3 * (entries + 1),
            lambda q=protocol: _roc(view, q),  # type: ignore[misc]
        )
        add(
            f"predictions-{name}",
            GROUP_EVALUATION,
            entries * (5 + len(protocol.baselines)),
            lambda q=protocol: _predictions(view, q),  # type: ignore[misc]
        )
        add(
            f"folds-{name}",
            GROUP_EVALUATION,
            9 * len(protocol.folds),
            lambda q=protocol: _folds(view, q),  # type: ignore[misc]
        )
    if p:
        add("margins", GROUP_EVALUATION, 20 * p, lambda: _margins(view))
        add("object-margins", GROUP_EVALUATION, m * (4 * p + 2), lambda: _object_margins(view))
    if view.sensitivity:
        add("sensitivity", GROUP_EVALUATION, 36, lambda: _sensitivity(view))
    add("properties", GROUP_CHECKS, 60, lambda: _properties(view))
    pairs = len(result.properties.operator_pairs)
    if pairs:
        add("operator-pairs", GROUP_CHECKS, 4 * pairs, lambda: _operator_pairs(view))
    ties = len(result.properties.boundary_ties)
    add("boundary-ties", GROUP_CHECKS, 4 * ties, lambda: _boundary_ties(view))
    return specs


def run_tables(
    view: RunView, *, max_cells: int | None = DEFAULT_MAX_CELLS, keys: Sequence[str] | None = None
) -> TableSet:
    """Build the tables of a run.

    Args:
        view: The run.
        max_cells: Tables with more body cells are left out and listed in ``skipped``
            (``None``: no limit).
        keys: Only these tables (by key); every table by default.
    """
    tables: list[Table] = []
    skipped: list[TableSpec] = []
    for spec in table_specs(view):
        if keys is not None and spec.key not in keys:
            continue
        if max_cells is not None and spec.cells > max_cells:
            skipped.append(spec)
            continue
        tables.append(spec.build())
    return TableSet(tuple(tables), tuple(skipped))
