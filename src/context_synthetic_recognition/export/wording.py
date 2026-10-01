"""Words shared by the exporters: how results, sets and checks are written.

The texts follow the Excel experiment (sheets *Model Properties*, *Meta-algorithm*, *Accuracy*), so
the Excel mirror, the tables and the report describe a result with the same words.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.core.meta import K1_DECISION, K2_DECISION, REFUSAL
from context_synthetic_recognition.data.schema import Label
from context_synthetic_recognition.export.view import LEAVE_ONE_OUT, RunView
from context_synthetic_recognition.notation import DASH, synthetic_name

MAX_SET_MEMBERS = 60
"""Sets with more members are abbreviated (``{S₁, …, S₆₀, … (+210)}``)."""

PROTOCOL_TITLES = {
    "resubstitution": "resubstitution",
    "leave-one-out": "leave-one-out",
    "stratified-k-fold": "stratified k-fold",
    "repeated-k-fold": "repeated k-fold",
    "hold-out": "hold-out",
}
"""How a protocol is named in running text."""


def protocol_title(name: str) -> str:
    """A protocol's name in running text."""
    return PROTOCOL_TITLES.get(name, name)


def join_and(items: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def percent(value: float) -> str:
    """``0.2`` → ``20.0%`` — as the workbook writes accuracies in text cells."""
    return f"{100 * value:.1f}%"


def set_text(names: Sequence[str], limit: int = MAX_SET_MEMBERS) -> str:
    """``{S₂, S₄}`` for the members, ``∅`` for none — the workbook's set cells."""
    if not names:
        return "∅"
    shown = list(names[:limit])
    if len(names) > limit:
        shown.append(f"… (+{len(names) - limit})")
    return "{" + ", ".join(shown) + "}"


def tuple_text(values: Sequence[object]) -> str:
    """``(2, 1, 2)`` — a description (a₀, …, a_p)."""
    return "(" + ", ".join(str(v) for v in values) + ")"


def tuplam_text(features: Sequence[int]) -> str:
    """``{a₆, a₃}`` for 0-based synthetic features."""
    return "{" + ", ".join(synthetic_name(u) for u in features) + "}"


def class_text(label: Label) -> str:
    """``Class 1`` — a class in headers and decisions."""
    return f"Class {label}"


def decision_text(decision: int, classes: Sequence[Label]) -> str:
    """The decision of Step 4 in words: ``Class 1``, ``Class 2`` or ``0 — refusal``."""
    if decision == K1_DECISION:
        return class_text(classes[0])
    if decision == K2_DECISION:
        return class_text(classes[1])
    return "0 — refusal"


def decision_label(decision: int, classes: Sequence[Label]) -> Label:
    """The predicted class label, or 0 for a refusal (the workbook's "Predicted" cells)."""
    return REFUSAL if decision == REFUSAL else classes[decision - 1]


def labels_of(decisions: npt.ArrayLike, classes: Sequence[Label]) -> list[Label]:
    """Decision codes as class labels (0 = refusal)."""
    return [decision_label(int(d), classes) for d in np.asarray(decisions).ravel()]


@dataclass(frozen=True)
class PropertyRow:
    """One line of the model-property checks (sheet *Model Properties*)."""

    section: str
    """The definition it belongs to, e.g. ``Definition 1 · Property 1``."""
    check: str
    result: str
    status: str


def _ok(condition: bool, good: str, bad: str) -> str:
    return good if condition else bad


def determinacy_rows(view: RunView) -> list[tuple[str, str | int, str]]:
    """Definition 1 / Property 1: (check, result, status) — rows 5–9 of *Model Properties*."""
    d = view.result.properties.determinacy
    declared = view.trace.permitted_k.k_max
    k_max = declared if declared is not None else d.k_max
    return [
        (
            "Enough neighbours for every permitted k  (m − 1 ≥ k_max)",
            f"{d.neighbours} ≥ {k_max}",
            _ok(d.neighbours >= k_max, "✓ defined", "✗ too few objects"),
        ),
        (
            "f_k(μ) defined at the gradation of every object (formula (1))",
            d.undefined_memberships,
            _ok(d.undefined_memberships == 0, "✓ defined", "✗ undefined values"),
        ),
        (
            "q₁ and q₂ exist for every feature (formula (3))",
            d.missing_sides,
            _ok(
                d.missing_sides == 0,
                "✓ both sides exist",
                f"⚠ 0.5 used for {d.missing_sides} side(s)",
            ),
        ),
        (
            "Synthetic feature (5) defined for every object",
            d.undefined_features,
            _ok(d.undefined_features == 0, "✓ defined", "✗ undefined values"),
        ),
        (
            "Meta-algorithm gives a class or an explicit refusal",
            f"{d.refusals} refusal(s)",
            "✓ always defined (0 = refusal)",
        ),
    ]


def training_correctness(view: RunView) -> tuple[str, str]:
    """Definition 2: (result, status) of the resubstitution."""
    truth = view.trace.class_index + 1
    correct = int(np.count_nonzero(view.training.decisions == truth))
    wrong = view.trace.m - correct
    status = "✓ correct on E₀" if wrong == 0 else f"✗ not correct on {wrong} object(s)"
    return f"{correct} / {view.trace.m}", status


def generalization(view: RunView, protocol: str = LEAVE_ONE_OUT) -> tuple[str, str] | None:
    """Definition 3: (result, status) of a hold-out protocol, or ``None`` if it was not run."""
    found = view.protocol(protocol)
    if found is None:
        return None
    metrics = view.result.metrics(found.predictions)
    return metrics.label, f"accuracy {percent(metrics.accuracy)},  refusals {metrics.refusals}"


def sufficiency_status(conflicts: int) -> str:
    """Definition 4 for a number of conflicting pairs."""
    if conflicts == 0:
        return "✓ sufficient on E₀"
    return f"✗ insufficient: {conflicts} conflicting pair(s)"


def theorem_rows(view: RunView) -> list[tuple[str, str, str]]:
    """The Theorem checks: (check, result, status) — rows 47–49 of *Model Properties*."""
    demo = view.new_object
    inputs = (
        "Inputs of the new-object path",
        "x, training E, neighbour classes",
        "✓ the class of S is not an input of Ψ, D or R",
    )
    first = (
        "Brace sheet: Ψ(r) of the typed object vs the training description of the excluded object"
    )
    second = (
        "Meta-algorithm sheet: class of the typed object vs resubstitution class of that object"
    )
    if demo.exclude is None:
        return [
            # the workbook's wording, so that `csr validate` also accepts a mirror of a new object
            (
                first,
                "no object excluded",
                "— (type a training object and its number to run this check)",
            ),
            (second, DASH, DASH),
            inputs,
        ]
    representation = demo.classification.representation
    assert representation is not None
    same = bool(np.array_equal(representation.context.values[0], view.trace.values[demo.exclude]))
    new, training = demo.decision, int(view.training.decisions[demo.exclude])
    return [
        (
            first,
            "identical" if same else "different",
            _ok(same, "✓ same representation as in training", "✗ representation differs"),
        ),
        (
            second,
            f"{new} vs {training}",
            _ok(new == training, "✓ same decision", "✗ different decision"),
        ),
        inputs,
    ]


def property_rows(view: RunView) -> list[PropertyRow]:
    """Every model-property check of the run, in the order of the sheet *Model Properties*."""
    properties = view.result.properties
    rows = [
        PropertyRow("Definition 1 · Property 1 — determinacy", check, str(result), status)
        for check, result, status in determinacy_rows(view)
    ]
    result, status = training_correctness(view)
    rows.append(
        PropertyRow(
            "Definition 2 — training correctness",
            "Correctly classified training objects",
            result,
            status,
        )
    )
    for protocol in view.cross_validations:
        found = generalization(view, protocol.protocol)
        assert found is not None
        rows.append(
            PropertyRow(
                "Definition 3 — generalization correctness",
                f"Correctly classified held-out objects ({protocol_title(protocol.protocol)})",
                found[0],
                found[1],
            )
        )
    rows += [
        PropertyRow(
            "Definition 4 — sufficiency",
            "Pairs with the same TUPLAM description (a₀, …, a_p) and different classes",
            str(properties.tuplam_conflicts),
            sufficiency_status(properties.tuplam_conflicts),
        ),
        PropertyRow(
            "Definition 4 — sufficiency",
            "Pairs with the same full Ψ(r) description and different classes",
            str(properties.psi_conflicts),
            sufficiency_status(properties.psi_conflicts),
        ),
    ]
    rows += [
        PropertyRow("Theorem — classification without the class", check, result, status)
        for check, result, status in theorem_rows(view)
    ]
    return rows
