"""Evaluation designs: the plans, the blind rule, the article's numbers, the files (ADR-095)."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from context_synthetic_recognition.comparison import COMPARISON_SPLITS
from context_synthetic_recognition.data.samples import AnswerKey, UnlabelledSample
from context_synthetic_recognition.data.schema import Dataset
from context_synthetic_recognition.data.splits import FixedSplit
from context_synthetic_recognition.evaluation.protocols import EvaluationCancelledError
from context_synthetic_recognition.services.classifier_parts import CLASSIFIER_PARTS
from context_synthetic_recognition.services.comparison_part import ComparisonPart
from context_synthetic_recognition.services.design_outputs import (
    all_designs_table,
    design_files,
    design_table,
)
from context_synthetic_recognition.services.design_runs import (
    DesignablePart,
    classify_first_round,
    design_keys,
    first_round,
)
from context_synthetic_recognition.services.designs import (
    DESIGNS,
    BaseDesign,
    Case,
    DesignSplit,
    Round,
    designs,
    match_objects,
    percentile_rank,
    round_decisions,
    run_design,
    share,
    stratified_sample,
)

from .conftest import ROOT
from .test_comparison import CS_MODEL, PINNED
from .test_split_discovery import write_split

SPLITS_FOLDER = ROOT / "datasets" / "splits"
BLIND = (
    "direct",
    "reversed",
    "random-control",
    "cross-validation",
    "nested-selection",
    "size-curve",
    "cross-split",
)
"""The designs whose classified objects are not their training objects."""


def comparison_part() -> ComparisonPart:
    part = CLASSIFIER_PARTS.get("classifier-comparison")
    assert isinstance(part, ComparisonPart)
    return part


def toy_folder(folder: Path, *, m: int = 40, training: int = 16, n: int = 3) -> Path:
    """A small two-class split file in ``folder``, and a second one of the same objects."""
    rng = np.random.default_rng(3)
    classes = [1 + (i % 2) for i in range(m)]
    X = [[round(float(rng.normal(c, 1.0)), 3) for _ in range(n)] for c in classes]
    markers = [0 if i < training else 1 for i in range(m)]
    write_split(
        folder / "Toy" / f"Toy ({m}, {n}, 2) — training {training} (euclidean).dat",
        X,
        classes,
        markers,
        [1] * n,
    )
    other = [0 if 8 <= i < 8 + training + 2 else 1 for i in range(m)]
    write_split(
        folder / "Toy" / f"Toy ({m}, {n}, 2) — training {training + 2} (chebyshev).dat",
        X,
        classes,
        other,
        [1] * n,
    )
    return folder


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return toy_folder(tmp_path_factory.mktemp("splits"))


def design_splits(folder: Path, keys: tuple[str, ...]) -> tuple[DesignSplit, ...]:
    part = comparison_part()
    return part.design_splits(part.option(keys, folder), folder)


# ---------------------------------------------------------------- the registry and the plans


def test_the_nine_designs_are_registered() -> None:
    found = designs()
    assert [d.key for d in found][:9] == [
        "direct",
        "reversed",
        "training-self",
        "test-self",
        "random-control",
        "cross-validation",
        "nested-selection",
        "size-curve",
        "cross-split",
    ]
    assert [d.group for d in found][:9] == [
        "Swapping the subsets",
        "Swapping the subsets",
        "Swapping the subsets",
        "Swapping the subsets",
        "Comparison points",
        "Comparison points",
        "Honest estimate",
        "Optional extras",
        "Optional extras",
    ]
    assert found[0].title == "Training → Test (as given)"
    assert design_keys("all")[:9] == tuple(d.key for d in found[:9])
    assert design_keys("reversed") == ("reversed",)
    assert isinstance(comparison_part(), DesignablePart)


def test_the_plans_on_a_small_split(toy: Path) -> None:
    (split,) = design_splits(toy, ("Toy-16",))
    T, Q = split.split.training, split.split.test
    plans = {d.key: d.cases(split, ()) for d in designs()[:9]}
    (direct,) = plans["direct"]
    first = direct.plan(0)
    assert (first.training.tolist(), first.classified.tolist()) == (T.tolist(), Q.tolist())
    reversed_ = plans["reversed"][0].plan(0)
    assert (reversed_.training.tolist(), reversed_.classified.tolist()) == (Q.tolist(), T.tolist())
    for key, sample in (("training-self", T), ("test-self", Q)):
        planned = plans[key][0].plan(0)
        assert planned.self_check
        assert planned.training.tolist() == planned.classified.tolist() == sample.tolist()
    random, given = plans["random-control"]
    assert random.count == 100
    drawn = random.plan(5)
    assert drawn.label == "repetition 6 (seed 47)"
    classes = split.dataset.class_index
    assert np.bincount(classes[drawn.training]).tolist() == list(split.split.training_sizes[1:])
    assert sorted([*drawn.training, *drawn.classified]) == list(range(split.dataset.m))
    assert given.plan(0).training.tolist() == T.tolist()
    (folds,) = plans["cross-validation"]
    rounds = [folds.plan(i) for i in range(folds.count)]
    assert folds.count == 10
    assert sorted(i for r in rounds for i in r.classified) == list(range(split.dataset.m))
    (nested,) = plans["nested-selection"]
    chosen = nested.plan(0)
    assert set(chosen.training) <= set(rounds[0].training)  # R-IPS on the fold's objects only
    assert not set(chosen.training) & set(chosen.classified)
    assert chosen.note.startswith("R-IPS kept")
    curve = plans["size-curve"]
    assert [c.key for c in curve] == ["25", "50", "75", "100"]
    assert [c.count for c in curve] == [20, 20, 20, 1]
    quarter = curve[0].plan(0)
    assert np.bincount(classes[quarter.training]).tolist() == [2, 2]  # 8 × 0.25 per class
    assert set(quarter.training) <= set(T)
    assert curve[3].plan(0).training.tolist() == T.tolist()
    (alone,) = plans["cross-split"]
    assert alone.unavailable == "no other ticked split of Toy"


def test_seeds_give_the_same_samples_again() -> None:
    classes = np.array([0, 1] * 20)
    pool = np.arange(40)
    first = stratified_sample(classes, pool, (5, 3), 42)
    assert first.tolist() == stratified_sample(classes, pool, (5, 3), 42).tolist()
    assert first.tolist() != stratified_sample(classes, pool, (5, 3), 43).tolist()
    assert (share(196, 0.25), share(173, 0.25), share(1, 0.25)) == (49, 43, 1)
    assert percentile_rank([1.0, 2.0, 3.0, 4.0], 3.0) == 62.5


def test_split_to_split_needs_the_same_objects(toy: Path) -> None:
    pair = design_splits(toy, ("Toy-16", "Toy-18"))
    cross = DESIGNS.get("cross-split")
    (case,) = cross.cases(pair[0], pair[1:])
    assert not case.unavailable
    planned = case.plan(0)
    assert planned.source == "Toy-18"
    target = pair[1].split
    assert not set(planned.classified) - set(target.test)  # B's test objects …
    assert not set(planned.classified) & set(range(16))  # … without A's training objects
    german = design_splits(SPLITS_FOLDER, ("German-336", "German-341", "German-367"))
    reasons = {c.key: c.unavailable for c in cross.cases(german[0], german[1:])}
    assert "the features differ" in reasons["German-341"]
    matched = {c.key: c.unavailable for c in cross.cases(german[1], (german[0], german[2]))}
    assert matched == {"German-336": matched["German-336"], "German-367": ""}
    spambase = design_splits(SPLITS_FOLDER, ("Spambase-601", "Spambase-722"))
    (scaled,) = cross.cases(spambase[0], spambase[1:])
    assert "other values" in scaled.unavailable
    mapping, reason = match_objects(german[1].dataset, german[2].dataset)
    assert reason == ""
    assert mapping is not None
    assert sorted(mapping.tolist()) == list(range(1000))


# ---------------------------------------------------------------- the blind rule


def _flip(split: DesignSplit, objects: np.ndarray, change: str) -> DesignSplit:
    data = split.dataset
    labels = data.class_index.copy()
    if change == "flipped":
        labels[objects] = 1 - labels[objects]
    else:
        labels[objects] = np.random.default_rng(1).permutation(labels[objects])
    y = np.asarray(data.classes)[labels]
    changed = Dataset(X=data.X, y=y, feature_types=data.feature_types, name=data.name)
    return DesignSplit(
        split.key,
        split.label,
        FixedSplit(changed, split.split.training, split.split.test),
        split.metric,
        split.file,
        split.sha256,
    )


@pytest.mark.parametrize("key", BLIND)
@pytest.mark.parametrize("change", ["shuffled", "flipped"])
def test_the_classes_of_the_classified_objects_change_no_decision(
    toy: Path, key: str, change: str
) -> None:
    part = comparison_part()
    splits = design_splits(toy, ("Toy-16", "Toy-18"))
    option = part.option(("Toy-16", "Toy-18"), toy)
    design = DESIGNS.get(key)
    split = splits[0]
    case = next(c for c in design.cases(split, splits[1:]) if not c.unavailable)
    planned = case.plan(0)
    sources = {s.key: s for s in splits}
    before = round_decisions(part, option, split, planned, sources)
    source = sources[planned.source] if planned.source else split
    changed = _flip(source, planned.classified, change)
    sources_after = {**sources, changed.key: changed}
    after = round_decisions(
        part, option, changed if changed.key == split.key else split, planned, sources_after
    )
    assert list(before) == list(after)
    for method, decided in before.items():
        assert decided.same_as(after[method]), (key, method)


@pytest.mark.parametrize("key", BLIND)
def test_deleting_the_classes_changes_no_decision(toy: Path, key: str) -> None:
    # objects that never had a class: their feature values alone reach classify
    part = comparison_part()
    keys = ("Toy-16", "Toy-18")
    option = part.option(keys, toy)
    design = DESIGNS.get(key)
    samples = first_round(design, part, option, toy)
    trained = part.train(option, samples.training)
    bare = tuple(
        UnlabelledSample(s.objects, s.X, s.feature_types, s.feature_names, "bare")
        for s in samples.test
    )
    decided = classify_first_round(design, part, trained, bare, self_check=False)
    again = classify_first_round(design, part, trained, samples.test, self_check=False)
    for split in keys:
        for method, d in decided.decisions[split].items():
            assert d.same_as(again.decisions[split][method]), (key, method)


@pytest.mark.parametrize("key", ["training-self", "test-self"])
def test_a_self_check_reads_no_answer_key(toy: Path, key: str) -> None:
    part = comparison_part()
    option = part.option(("Toy-16",), toy)
    design = DESIGNS.get(key)
    samples = first_round(design, part, option, toy)
    trained = part.train(option, samples.training)
    decided = classify_first_round(design, part, trained, samples.test, self_check=True)
    (answers,) = samples.answers
    flipped = AnswerKey(answers.objects, 3 - answers.classes, answers.labels, answers.name)
    assert flipped.classes.tolist() != answers.classes.tolist()
    # the decisions are the training stage's resubstitution: the key is read afterwards only
    for method in trained.methods["Toy-16"]:
        d = decided.decisions["Toy-16"][method.method]
        if method.decisions is None:
            assert d.decisions is None
        else:
            assert d.decisions is not None
            assert d.decisions.tolist() == method.decisions.tolist()


# ---------------------------------------------------------------- the article's numbers


@pytest.fixture(scope="module")
def german() -> tuple[DesignSplit, ...]:
    return design_splits(SPLITS_FOLDER, COMPARISON_SPLITS)


def _accuracy(result, split: str, method: str) -> tuple[float, float]:
    found = next(s for s in result.splits if s.key == split)
    (case,) = found.cases
    (value,) = case.accuracies(method)
    (balanced,) = case.balanced(method)
    return round(value, 2), round(balanced, 2)


def test_the_design_as_given_is_todays_evaluation(german: tuple[DesignSplit, ...]) -> None:
    part = comparison_part()
    option = part.option(COMPARISON_SPLITS, SPLITS_FOLDER)
    result = run_design(DESIGNS.get("direct"), part, option, german)
    for key, methods in PINNED.items():
        for method, (test, balanced, _) in methods.items():
            assert _accuracy(result, key, method) == (test, balanced), (key, method)
    for key, models in CS_MODEL.items():
        for method, (test, balanced, _, _) in models.items():
            assert _accuracy(result, key, method) == (test, balanced), (key, method)


def test_the_training_self_check_is_todays_training_accuracy(
    german: tuple[DesignSplit, ...],
) -> None:
    part = comparison_part()
    option = part.option(COMPARISON_SPLITS, SPLITS_FOLDER)
    result = run_design(DESIGNS.get("training-self"), part, option, german)
    for key, methods in PINNED.items():
        for method, (_, _, training) in methods.items():
            assert _accuracy(result, key, method)[0] == training, (key, method)
    for key, models in CS_MODEL.items():
        for method, (_, _, training, _) in models.items():
            assert _accuracy(result, key, method)[0] == training, (key, method)


# ---------------------------------------------------------------- results, unavailable, cancel


def test_a_design_on_a_small_split_and_its_files(toy: Path) -> None:
    part = comparison_part()
    option = part.option(("Toy-16", "Toy-18"), toy)
    splits = part.design_splits(option, toy)
    design = DESIGNS.get("size-curve")
    result = run_design(design, part, option, splits)
    again = run_design(design, part, option, splits)
    files = design_files(design, result, versions={}, inputs={})
    assert files == design_files(design, again, versions={}, inputs={})  # the seeds
    assert list(files) == [
        "size-curve.csv",
        "size-curve.md",
        "size-curve.tex",
        "rounds.csv",
        "summary.json",
        "checksums.sha256",
    ]
    table = design_table(design, result)
    assert table.columns[:4] == ("Split", "Case", "Method", "Rounds")
    assert "SD" in table.columns
    # the CS-model cannot run on 4 training objects of a class: unavailable, with the reason
    notes = {row[0:3]: row[-1] for row in table.rows}
    assert any("undefined" in str(note) for note in notes.values())
    everything = all_designs_table({"size-curve": result})
    assert everything.columns[2] == "Training-size curve · accuracy, %"


def test_a_long_design_can_be_cancelled(toy: Path) -> None:
    part = comparison_part()
    option = part.option(("Toy-16",), toy)
    splits = part.design_splits(option, toy)
    seen: list[str] = []
    with pytest.raises(EvaluationCancelledError):
        run_design(
            DESIGNS.get("random-control"),
            part,
            option,
            splits,
            progress=lambda what, done, total: seen.append(what),
            cancelled=lambda: len(seen) >= 3,
        )
    assert seen[0].startswith("Toy · Euclidean split · training 16 (8, 8) · 100 random")


@dataclass(frozen=True)
class EveryOther(BaseDesign):
    """A design for the tests: fit on every other pair of objects, classify the rest."""

    def cases(self, split, others):
        everyone = np.arange(split.dataset.m)
        planned = Round("", everyone[everyone % 4 < 2], everyone[everyone % 4 >= 2])
        return (Case("as-given", "", 1, lambda _: planned),)


@pytest.fixture
def every_other() -> Iterator[EveryOther]:
    design = EveryOther("every-other", "Every other object", "Optional extras", "For the tests.")
    DESIGNS.add(design.key, design)
    yield design
    DESIGNS.remove(design.key)


def test_a_design_is_one_registration(every_other: EveryOther, toy: Path) -> None:
    assert every_other in designs()
    part = comparison_part()
    option = part.option(("Toy-16",), toy)
    result = run_design(every_other, part, option, part.design_splits(option, toy))
    (split,) = result.splits
    (case,) = split.cases
    assert case.rounds[0].training_sizes == (20, 10, 10)
