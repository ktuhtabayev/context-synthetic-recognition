"""Model properties: Definitions 1, 4, 6, Property 1 and the Theorem (sheet Model Properties)."""

import numpy as np

from context_synthetic_recognition.core.model import CSModel
from context_synthetic_recognition.core.properties import (
    conflicting_pairs,
    equal_neighbour_sets,
    model_properties,
    theorem_check,
)
from context_synthetic_recognition.data import Dataset


def test_the_properties_of_the_experiment(experiment_cs_model: CSModel) -> None:
    p = model_properties(experiment_cs_model)
    d = p.determinacy
    assert (d.neighbours, d.k_max, d.enough_neighbours, d.defined) == (9, 5, True, True)
    assert (d.undefined_memberships, d.missing_sides, d.undefined_features, d.refusals) == (
        0,
        0,
        0,
        0,
    )
    # Definition 4: the representation is insufficient on E₀
    assert (p.tuplam_conflicts, p.psi_conflicts, p.sufficient) == (10, 10, False)
    # Definition 6: ρ vs ρ_J have identical k-sets for 5 (k = 3) and 2 (k = 5) objects
    counts = {(q.first, q.second, q.k): q.equal_sets for q in p.operator_pairs}
    assert counts == {
        (0, 1, 3): 0,
        (0, 1, 5): 0,
        (0, 2, 3): 5,
        (0, 2, 5): 2,
        (1, 2, 3): 0,
        (1, 2, 5): 0,
    }
    ties = {(t.operator, t.k): t.objects for t in p.boundary_ties}
    assert ties == {(0, 3): 0, (0, 5): 0, (1, 3): 0, (1, 5): 0, (2, 3): 5, (2, 5): 8}
    # 5 of 6 synthetic features are identical; only a₃ differs
    identical = p.identical_features
    assert identical[0].tolist() == [True, True, False, True, True, True]
    assert not identical[2, [0, 1, 3, 4, 5]].any()


def test_the_theorem_holds_for_every_training_object(
    experiment: Dataset, experiment_cs_model: CSModel
) -> None:
    for i in range(experiment.m):
        check = theorem_check(experiment_cs_model, experiment.X[i], i)
        assert check.holds, i


def test_conflicting_pairs() -> None:
    rows = [[1, 2], [1, 2], [1, 2], [2, 2]]
    assert conflicting_pairs(rows, [0, 1, 1, 0]) == 2
    assert conflicting_pairs(rows, [0, 0, 0, 1]) == 0


def test_equal_sets_of_an_operator_with_itself(experiment_cs_model: CSModel) -> None:
    assert equal_neighbour_sets(experiment_cs_model.trace, 1, 1, 3).all()
    assert np.count_nonzero(equal_neighbour_sets(experiment_cs_model.trace, 0, 2, 3)) == 5
