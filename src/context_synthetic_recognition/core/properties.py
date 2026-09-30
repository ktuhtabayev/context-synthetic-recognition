"""Model-property checks: Definitions 1, 4, 6, Property 1, the Theorem (*Model Properties*).

- **Definition 1 / Property 1 (determinacy):** every stage is defined for every object — enough
  neighbours for every permitted k, f_k(μ) defined at every object's gradation (formula (1)), q₁
  and q₂ exist for every synthetic feature (formula (3); otherwise G = 0.5), formula (5) defined
  for every object, and the meta-algorithm gives a class or an explicit refusal.
- **Definition 4 (sufficiency):** no two objects share a description but differ in class; the
  conflicting pairs are counted for the TUPLAM description (a₀ … a_p) and for the full Ψ(r).
- **Definition 6 (contextual equivalence of the base operators):** objects whose k-neighbour
  sets coincide for two operators; synthetic features that take the same value on every object;
  ties at the k boundary (the k-th and (k + 1)-th neighbours at the same distance, so the tie rule
  decides the neighbourhood).
- **Theorem:** a training object passed through the new-object path (left out of its own context)
  gets its training description and its training decision.

Definitions 2 and 3 (training and generalization correctness) are the resubstitution and
leave-one-out accuracies of :mod:`context_synthetic_recognition.evaluation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt

from context_synthetic_recognition.core.arrays import BoolArray, freeze_fields
from context_synthetic_recognition.core.model import CSModel
from context_synthetic_recognition.core.trace import ContextTrace


@dataclass(frozen=True)
class Determinacy:
    """Definition 1 / Property 1."""

    neighbours: int
    """m − 1: the neighbours of a training object."""
    k_max: int
    """The largest permitted k."""
    undefined_memberships: int
    """Objects × synthetic features whose f_k(μ) is undefined."""
    missing_sides: int
    """q₁ or q₂ that do not exist (G = 0.5 is used), over all synthetic features."""
    undefined_features: int
    """Objects × synthetic features without a value by formula (5)."""
    refusals: int
    """Training objects the meta-algorithm refuses (a refusal is a defined answer)."""

    @property
    def enough_neighbours(self) -> bool:
        """M − 1 ≥ k_max."""
        return self.neighbours >= self.k_max

    @property
    def defined(self) -> bool:
        """Every stage is defined for every object."""
        return (
            self.enough_neighbours
            and self.undefined_memberships == 0
            and self.undefined_features == 0
        )


@dataclass(frozen=True)
class OperatorPair:
    """Definition 6 for two base operators and one k."""

    first: int
    """Index of the first operator."""
    second: int
    """Index of the second operator."""
    k: int
    equal_sets: int
    """Objects whose k nearest neighbours are the same set under both operators."""


@dataclass(frozen=True)
class BoundaryTies:
    """Ties at the k boundary for one operator and one k."""

    operator: int
    k: int
    objects: int | None
    """Objects whose k-th and (k + 1)-th neighbours are equally far; ``None`` if k = m − 1."""


@dataclass(frozen=True, eq=False)
class ModelProperties:
    """Definitions 1, 4 and 6 on the training sample."""

    determinacy: Determinacy
    tuplam_conflicts: int
    """Definition 4: pairs with the same (a₀ … a_p) and different classes."""
    psi_conflicts: int
    """Definition 4: pairs with the same Ψ(r) and different classes."""
    operator_pairs: tuple[OperatorPair, ...]
    """Definition 6: equal k-neighbour sets, every operator pair and permitted k."""
    identical_features: BoolArray
    """(r × r) whether two synthetic features take the same value on every object."""
    boundary_ties: tuple[BoundaryTies, ...]
    """Ties at the k boundary, every operator and permitted k."""

    def __post_init__(self) -> None:
        """Freeze the matrix."""
        freeze_fields(self)

    @property
    def sufficient(self) -> bool:
        """Definition 4 holds for the TUPLAM description."""
        return self.tuplam_conflicts == 0


def conflicting_pairs(descriptions: npt.ArrayLike, class_index: npt.ArrayLike) -> int:
    """Unordered pairs of objects with identical descriptions (rows) and different classes."""
    rows = np.asarray(descriptions)
    y = np.asarray(class_index)
    _, key = np.unique(rows, axis=0, return_inverse=True)
    key = key.ravel()
    pairs = 0
    for group in np.unique(key):
        members = y[key == group]
        pairs += int(np.count_nonzero(members == 0)) * int(np.count_nonzero(members == 1))
    return pairs


def equal_neighbour_sets(trace: ContextTrace, first: int, second: int, k: int) -> BoolArray:
    """(m,) whether the k nearest neighbours of each object are the same set under two operators."""
    a = np.sort(trace.operators[first].order[:, :k], axis=1)
    b = np.sort(trace.operators[second].order[:, :k], axis=1)
    return np.asarray((a == b).all(axis=1))


def identical_features(trace: ContextTrace) -> BoolArray:
    """(r × r) synthetic features that take the same value on every object."""
    values = trace.values
    return np.asarray((values[:, :, None] == values[:, None, :]).all(axis=0))


def boundary_ties(trace: ContextTrace, operator: int, k: int) -> int | None:
    """Objects whose k-th and (k + 1)-th neighbours are at the same distance (None if k = m − 1)."""
    distances = trace.operators[operator].sorted_distances
    if k >= distances.shape[1]:
        return None
    return int(np.count_nonzero(distances[:, k - 1] == distances[:, k]))


def determinacy(model: CSModel) -> Determinacy:
    """Definition 1 / Property 1 on the training sample."""
    trace = model.trace
    features = trace.features
    return Determinacy(
        neighbours=trace.m - 1,
        k_max=max(trace.permitted_k.ks),
        undefined_memberships=sum(int(np.isnan(f.object_membership).sum()) for f in features),
        missing_sides=sum((f.boundary.q1 is None) + (f.boundary.q2 is None) for f in features),
        undefined_features=int(np.count_nonzero(~np.isin(trace.values, (1, 2)))),
        refusals=model.classify_training().refusals,
    )


def model_properties(model: CSModel) -> ModelProperties:
    """Definitions 1, 4 and 6 for a fitted model (sheet *Model Properties*)."""
    trace = model.trace
    ks = trace.permitted_k.ks
    operators = range(len(trace.operators))
    return ModelProperties(
        determinacy=determinacy(model),
        tuplam_conflicts=conflicting_pairs(model.description.gradations, trace.class_index),
        psi_conflicts=conflicting_pairs(trace.values, trace.class_index),
        operator_pairs=tuple(
            OperatorPair(a, b, k, int(np.count_nonzero(equal_neighbour_sets(trace, a, b, k))))
            for a, b in combinations(operators, 2)
            for k in ks
        ),
        identical_features=identical_features(trace),
        boundary_ties=tuple(
            BoundaryTies(o, k, boundary_ties(trace, o, k)) for o in operators for k in ks
        ),
    )


@dataclass(frozen=True)
class TheoremCheck:
    """A training object through the new-object path, left out of its own context."""

    index: int
    same_description: bool
    """Its Ψ(r) equals its training row."""
    same_decision: bool
    """Its decision equals its resubstitution decision."""

    @property
    def holds(self) -> bool:
        """Both agree."""
        return self.same_description and self.same_decision


def theorem_check(model: CSModel, x: npt.ArrayLike, index: int) -> TheoremCheck:
    """The Theorem for the training object ``index``: classified without its class.

    The object (values ``x``) is left out of its own context; its Ψ(r) and decision are compared
    with its training row and its resubstitution decision.
    """
    result = model.classify(x, exclude=[index])
    assert result.representation is not None
    same_description = bool(
        np.array_equal(result.representation.context.values[0], model.trace.values[index])
    )
    training = model.classify_training().decisions[index]
    return TheoremCheck(index, same_description, bool(result.decisions[0] == training))
