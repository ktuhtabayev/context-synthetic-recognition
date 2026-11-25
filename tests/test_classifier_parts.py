"""Classifier parts: the three stages, the blind test and "Run all" (ADR-084, ADR-085, ADR-086)."""

import csv
import json
import sys
from collections.abc import Iterator
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from context_synthetic_recognition.cli.app import EXIT_USAGE, app
from context_synthetic_recognition.comparison import COMPARISON_SPLITS
from context_synthetic_recognition.data import load_split_file
from context_synthetic_recognition.data.samples import (
    AnswerKey,
    UnlabelledSample,
    require_unlabelled,
    unlabelled,
)
from context_synthetic_recognition.errors import DatasetError
from context_synthetic_recognition.services import classifier_parts as parts
from context_synthetic_recognition.services.classifier_parts import (
    CLASSIFIER_PARTS,
    CellMeaning,
    CheckablePart,
    StageRun,
    WritablePart,
    run_stages,
)
from context_synthetic_recognition.services.comparison import (
    CHECKSUMS,
    library_versions,
    run_comparison,
)
from context_synthetic_recognition.services.comparison_part import (
    NEEDS_COMPARISON_SET,
    WRITES_COMPARISON_SET,
    ComparisonPart,
    blind_predictions,
)
from context_synthetic_recognition.services.split_discovery import discover_splits

from .conftest import ROOT
from .dummy_part import DummyPart
from .test_split_discovery import toy, write_split

SPLITS_FOLDER = ROOT / "datasets" / "splits"
SPLITS = tuple(discover_splits(SPLITS_FOLDER, read=()).resolve(k).spec for k in COMPARISON_SPLITS)
"""German's three splits, from the names of their files (ADR-090)."""
LABELS = {
    "German-336": "German · Zhuravlyov split · training 336 (196, 140)",
    "German-341": "German · Euclidean split · training 341 (173, 168)",
    "German-367": "German · Chebyshev split · training 367 (190, 177)",
}
"""Their names on the page and in every table (ADR-091)."""
OUT = ROOT / "experiments" / "comparison"
runner = CliRunner()


def comparison_part() -> ComparisonPart:
    part = CLASSIFIER_PARTS.get("classifier-comparison")
    assert isinstance(part, ComparisonPart)
    return part


@pytest.fixture(scope="module")
def article() -> StageRun:
    """Training, test and evaluation on the three article splits (one "Run all")."""
    part = comparison_part()
    return run_stages(part, part.option(COMPARISON_SPLITS, SPLITS_FOLDER), SPLITS_FOLDER)


# ---------------------------------------------------------------- the blind test (ADR-085)


def test_an_unlabelled_sample_has_no_class() -> None:
    split = load_split_file(SPLITS_FOLDER / SPLITS[1].file)
    sample = split.test_objects("341: test sample")
    assert {f.name for f in fields(sample)} == {
        "objects",
        "X",
        "feature_types",
        "feature_names",
        "name",
    }
    assert not any(hasattr(sample, name) for name in ("y", "classes", "class_index", "markers"))
    assert sample.objects.tolist() == split.test.tolist()
    assert np.array_equal(sample.X, split.dataset.X[split.test])
    assert (
        sample.object_ids[0] == f"S{''.join('₀₁₂₃₄₅₆₇₈₉'[int(d)] for d in str(split.test[0] + 1))}"
    )
    key = split.test_answers("341: test sample")
    assert key.matches(sample)
    assert key.sizes == (659, 527, 132)
    assert key.classes.tolist() == (split.dataset.class_index[split.test] + 1).tolist()
    with pytest.raises(TypeError, match="classified without their class"):
        require_unlabelled(split.test_part)  # a dataset carries the classes
    with pytest.raises(ValueError, match="read-only"):
        sample.X[0, 0] = 1.0


def _with_test_classes(folder: Path, change: str) -> Path:
    """A copy of the split files whose test objects have other classes."""
    rng = np.random.default_rng(7)
    target = folder / change
    for spec in SPLITS:
        source = SPLITS_FOLDER / spec.file
        lines = source.read_bytes().decode("utf-8").split("\n")
        m, n = (int(v) for v in lines[0].split("\t")[:2])
        rows = [line.split("\t") for line in lines[1 : m + 1]]
        test = [i for i, cells in enumerate(rows) if cells[n + 1] == "1"]
        classes = [int(rows[i][n]) for i in test]
        if change == "flipped":
            classes = [3 - c for c in classes]
        else:
            classes = rng.permutation(classes).tolist()
        for i, c in zip(test, classes, strict=True):
            rows[i][n] = str(c)
        lines[1 : m + 1] = ["\t".join(cells) for cells in rows]
        path = target / spec.file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes("\n".join(lines).encode("utf-8"))
    return target


@pytest.mark.parametrize("change", ["shuffled", "flipped", "deleted"])
def test_the_test_classes_change_no_prediction(
    article: StageRun, tmp_path: Path, change: str
) -> None:
    part = comparison_part()
    option = part.option(COMPARISON_SPLITS, SPLITS_FOLDER)
    if change == "deleted":
        # objects that never had a class: their feature values alone
        samples = article.samples
        objects = tuple(
            unlabelled(s.objects, s.X, s.feature_types, s.feature_names, s.name)
            for s in samples.test
        )
        trained = part.train(option, samples.training)
    else:
        samples = part.load(option, _with_test_classes(tmp_path, change))
        objects = samples.test
        changed = [
            int(np.count_nonzero(new.classes != old.classes))
            for new, old in zip(samples.answers, article.samples.answers, strict=True)
        ]
        assert all(changed), changed  # the classes did change
        trained = part.train(option, samples.training)
    predictions = parts.classify(part, trained, objects)
    reference = article.predictions
    kinds = set()
    for split in COMPARISON_SPLITS:
        assert list(predictions.decisions[split]) == list(reference.decisions[split])
        for method, decided in predictions.decisions[split].items():
            assert decided.same_as(reference.decisions[split][method]), (split, method)
            kinds.add(method.rsplit("-", 2)[0] if method.startswith("cs-") else method)
    # SVM, Random Forest, every k-NN row and the CS-model
    assert {"svm", "random-forest", "knn-zhuravlyov-5", "cs-model"} <= kinds
    assert len(predictions.decisions["German-336"]) == 18


def test_a_dataset_cannot_be_classified(article: StageRun) -> None:
    part = comparison_part()
    split = load_split_file(SPLITS_FOLDER / SPLITS[0].file)
    with pytest.raises(TypeError, match="not a Dataset"):
        parts.classify(part, article.trained, (split.test_part,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="not a Dataset"):
        part.classify(article.trained, (split.test_part,))  # type: ignore[arg-type]
    fitted = article.trained.fitted[0]
    with pytest.raises(TypeError, match="not a Dataset"):
        fitted.methods[0].classify(split.test_part)  # type: ignore[arg-type]


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="numpy's stubs use Python 3.12 syntax: mypy runs on 3.12 (pyproject.toml)",
)
def test_mypy_rejects_a_dataset(tmp_path: Path) -> None:
    # the same check runs in the type check itself: classifier_parts holds the call with a
    # "type: ignore[arg-type]" that fails mypy --strict (unused) if a dataset were accepted
    api = pytest.importorskip("mypy.api")
    snippet = tmp_path / "blind.py"
    snippet.write_text(
        "from context_synthetic_recognition.data import Dataset\n"
        "from context_synthetic_recognition.services.classifier_parts import (\n"
        "    ClassifierPart, TrainedPart,\n"
        ")\n\n\n"
        "def blind(part: ClassifierPart, trained: TrainedPart, data: Dataset) -> None:\n"
        "    part.classify(trained, (data,))\n",
        encoding="utf-8",
    )
    out, _, status = api.run(
        [
            str(snippet),
            "--strict",
            "--no-incremental",
            "--follow-imports=silent",
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert status == 1, out
    assert 'Argument 2 to "classify" of "ClassifierPart" has incompatible type' in out
    assert '"tuple[Dataset]"' in out


# ---------------------------------------------------------------- Run all = today's results


def test_run_all_gives_todays_results_byte_for_byte(article: StageRun) -> None:
    part = comparison_part()
    assert part.write_problem(article.evaluation) == ""
    staged = part.outcome(article.evaluation)
    direct = run_comparison(SPLITS_FOLDER)
    assert dict(staged.files) == dict(direct.files)
    assert [t.rows for t in staged.tables] == [t.rows for t in direct.tables]
    recorded = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))["versions"]
    if recorded != library_versions():
        pytest.skip(f"the results were written with {recorded}, installed: {library_versions()}")
    for name, data in staged.files.items():
        assert data == (OUT / name).read_bytes(), name


def test_the_evaluation_tabs(article: StageRun) -> None:
    tabs = {t.key: t for t in parts.evaluation_view(article.evaluation)}
    assert list(tabs) == [
        "standard-classifiers",
        "knn-by-metric",
        "cs-model",
        "all-results",
        "per-object",
        "confusion",
    ]
    (standard,) = tabs["standard-classifiers"].variants
    assert standard.table.rows == (
        ("SVM", 74.85, 50.68, 61.14),
        ("Random Forest", 74.25, 56.45, 59.87),
    )
    assert len(tabs["all-results"].variants[0].table.rows) == 46
    per_object = {v.label: v for v in tabs["per-object"].variants}
    assert list(per_object) == list(LABELS.values())
    table = per_object[LABELS["German-336"]].table
    assert table.columns[:3] == ("Object", "True class", "Correct methods")
    assert len(table.columns) == 3 + 18
    marks = {str(v)[:1] for row in table.rows for v in row[3:]}
    assert marks <= {"✓", "✗", "0"}
    svm = article.evaluation.metrics["German-336"]["svm"]
    assert svm is not None
    assert sum(1 for row in table.rows if str(row[3]).startswith("✓")) == svm.correct == 497
    assert per_object[LABELS["German-336"]].cells[1] is CellMeaning.CLASS
    assert set(per_object[LABELS["German-336"]].cells[3:]) == {CellMeaning.OUTCOME}
    confusion = tabs["confusion"].variants[0].table
    first = confusion.rows[0]
    assert first[:2] == (LABELS["German-336"], "SVM")
    assert sum(first[2:8]) == 664  # type: ignore[arg-type]
    assert first[8] == 74.85


def test_a_single_split_has_no_comparison_tables(tmp_path: Path) -> None:
    part = comparison_part()
    run = run_stages(part, part.option(("341",), SPLITS_FOLDER), SPLITS_FOLDER)
    tabs = {t.key: t for t in run.evaluation.tables}
    assert tabs["standard-classifiers"].missing == NEEDS_COMPARISON_SET
    assert not tabs["standard-classifiers"].variants
    (everything,) = tabs["all-results"].variants
    assert len(everything.table.rows) == 14  # 8 methods, 6 CS-models
    assert part.write_problem(run.evaluation) == WRITES_COMPARISON_SET
    with pytest.raises(DatasetError, match=r"Only German.s three splits"):
        part.write(run.evaluation, tmp_path)
    assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------- the stages' tabs


def test_the_training_and_test_tabs(article: StageRun) -> None:
    part = comparison_part()
    training = {
        t.key: t for t in parts.training_view(part, article.trained.option, article.trained)
    }
    assert list(training) == ["methods", "training-sample", "training-results"]
    methods = training["methods"].variants[0].table
    assert {row[0] for row in methods.rows} == {
        "Standard classifiers",
        "k-NN by metric",
        "Not printed",
        "CS-model",
    }
    assert methods.rows[8][:4] == (
        "Not printed",
        "k-NN Zhuravlyov, k = 3",
        "all",
        "German · Zhuravlyov split · training 336",
    )
    sample = {v.label: v for v in training["training-sample"].variants}[LABELS["German-341"]]
    assert sample.table.columns[-1] == "Class"
    assert len(sample.table.rows) == 341
    assert sample.cells[-1] is CellMeaning.CLASS
    results = training["training-results"].variants[0].table
    svm = next(r for r in results.rows if r[:2] == (LABELS["German-336"], "SVM"))
    assert svm[2:] == (336, 83.63, 281, 0)

    test = {t.key: t for t in parts.prediction_view(article.predictions)}
    assert list(test) == ["test-objects", "predictions", "predicted-classes"]
    objects = {v.label: v for v in test["test-objects"].variants}[LABELS["German-367"]]
    assert objects.table.columns == ("Object", *article.samples.test[2].feature_names)
    assert "Class" not in objects.table.columns
    assert len(objects.table.rows) == 633
    predictions = {v.label: v for v in test["predictions"].variants}[LABELS["German-336"]]
    columns = predictions.table.columns
    assert columns[:3] == ("Object", "SVM", "SVM · score")
    assert columns[-2:] == ("Most common", "Agreement")
    assert not any("class" in c.lower() and c != "Most common" for c in columns)
    assert CellMeaning.OUTCOME not in predictions.cells  # correctness is unknown here
    first = predictions.table.rows[0]
    assert first[1] in (1, 2)
    assert isinstance(first[2], float)
    assert 1 <= first[-1] <= 18  # type: ignore[operator]
    counts = test["predicted-classes"].variants[0].table
    for row in counts.rows:
        if row[3] is not None:
            assert row[3] + row[5] + row[7] == row[2]  # type: ignore[operator]


def test_the_tree_of_samples() -> None:
    part = comparison_part()
    tree = part.samples(SPLITS_FOLDER)
    assert [g.title for g in tree.groups][:3] == [
        "German (1000 objects)",
        "Heart-Disease (270 objects)",
        "Spambase (2102 objects)",
    ]
    german, heart, spambase = tree.groups[:3]
    assert [c.title for c in german.choices] == [
        "Zhuravlyov split · 20 features · training 336 (196, 140) · test 664",
        "Euclidean split · 7 features · training 341 (173, 168) · test 659",
        "Chebyshev split · 7 features · training 367 (190, 177) · test 633",
    ]
    assert [c.title for c in heart.choices] == [
        "Zhuravlyov split · 13 features · training 76 (36, 40) · test 194"
    ]
    assert [c.title for c in spambase.choices] == [
        "Euclidean split · 57 features · training 601 (361, 240) · test 1501",
        "Euclidean split · 57 features · training 722 (467, 255) · test 1380",
    ]
    assert [c.label for c in spambase.choices] == [
        "Spambase · Euclidean split · training 601 (361, 240)",
        "Spambase · Euclidean split · training 722 (467, 255)",
    ]
    # the keys are internal: no title or label shows one
    for choice in tree.choices:
        assert choice.available, choice.reason
        assert choice.key not in choice.title
        assert choice.key not in choice.label
    assert tree.default == COMPARISON_SPLITS == tree.complete
    # any set can be ticked; the option keeps the tree's order
    shuffled = tree.option(["German-367", "German-336", "German-341"])
    assert shuffled.splits == COMPARISON_SPLITS
    assert shuffled.complete
    assert shuffled.available
    assert not tree.option(["German-336"]).complete
    assert not tree.option([*COMPARISON_SPLITS, "Heart-Disease-76"]).complete
    mixed = tree.option(["Spambase-601", "German-341"])
    assert mixed.splits == ("German-341", "Spambase-601")
    assert mixed.title == (
        "German · Euclidean split · training 341 (173, 168); "
        "Spambase · Euclidean split · training 601 (361, 240)"
    )
    nothing = tree.option([])
    assert not nothing.available
    assert nothing.reason == "Tick at least one sample."
    assert not tree.option(["336"]).available  # the old keys are not keys of the tree
    assert tree.known(["tables-7-8", "article", "336", "German-341"]) == ("German-341",)
    assert part.option(("336", "341", "367"), SPLITS_FOLDER).complete  # the CLI's old keys


def test_a_spambase_split_runs_through_the_three_stages() -> None:
    # ADR-088: Spambase's splits are samples of the part, not columns of the comparison tables
    part = comparison_part()
    groups = part.method_groups(part.option(("601",), SPLITS_FOLDER))
    assert [g.title for g in groups] == [
        "Standard classifiers",
        "k-NN by metric",
        "CS-model",
    ]  # no Zhuravlyov row
    run = run_stages(part, part.option(("601",), SPLITS_FOLDER), SPLITS_FOLDER)
    (sample,) = run.trained.samples
    assert sample.data.class_sizes == (361, 240)
    assert run.predictions.objects[0].m == 1501
    assert len(run.predictions.decisions["Spambase-601"]) == 14  # 8 standard methods, 6 CS-models
    tabs = {t.key: t for t in run.evaluation.tables}
    assert tabs["standard-classifiers"].missing == NEEDS_COMPARISON_SET
    (everything,) = tabs["all-results"].variants
    assert {row[0] for row in everything.table.rows} == {
        "Spambase · Euclidean split · training 601 (361, 240)"
    }
    assert len(everything.table.rows) == 14
    assert part.write_problem(run.evaluation) == WRITES_COMPARISON_SET


def test_unavailable_samples(tmp_path: Path) -> None:
    part = comparison_part()
    (tmp_path / "German").mkdir()
    for spec in SPLITS:
        (tmp_path / spec.file).write_bytes((SPLITS_FOLDER / spec.file).read_bytes())
    broken = tmp_path / SPLITS[0].file
    broken.write_bytes(broken.read_bytes().replace(b"1000\t20\t2\t", b"1000\t21\t2\t", 1))
    write_split(
        tmp_path / "Three" / "Three (6, 1, 3) — training 3 (euclidean).dat",
        [[0.0], [1.0], [2.0], [0.5], [1.5], [2.5]],
        [1, 2, 3, 1, 2, 3],
        [0, 0, 0, 1, 1, 1],
        [1],
    )
    toy(tmp_path / "German", "notes.dat")
    tree = part.samples(tmp_path)
    assert [g.title for g in tree.groups] == ["German (1000 objects)", "Three (6 objects)"]
    german = tree.choice("German-336")
    assert german is not None
    assert not german.available
    assert "row 2 (object 1) has 22 values" in german.reason
    assert german.detail == "German/German (1000, 20, 2) — training 336 (zhuravlyov).dat, row 2"
    three = tree.choice("Three-3")
    assert three is not None
    assert "the comparison needs two classes, the file has 3" in three.reason
    notes = tree.choice("German/notes.dat")
    assert notes is not None
    assert notes.title == "notes.dat"
    assert "does not follow" in notes.reason
    assert tree.default == ("German-341", "German-367")  # German's three, as far as available
    option = tree.option(COMPARISON_SPLITS)
    assert not option.available
    assert option.reason == german.reason
    with pytest.raises(DatasetError, match="unknown split 'x'"):
        part.option(("x",), tmp_path)
    missing = part.samples(tmp_path / "none" / "nested")
    assert missing.groups == ()
    assert "no folder of split files" in missing.problem
    assert missing.option([]).reason == missing.problem


def test_the_registry_and_the_capabilities() -> None:
    assert "classifier-comparison" in [p.key for p in parts.parts()]
    part = comparison_part()
    assert isinstance(part, WritablePart)
    assert isinstance(part, CheckablePart)
    dummy = DummyPart()
    assert not isinstance(dummy, WritablePart)
    assert not isinstance(dummy, CheckablePart)


@pytest.fixture
def dummy() -> Iterator[DummyPart]:
    part = DummyPart()
    CLASSIFIER_PARTS.add(part.key, part)
    yield part
    CLASSIFIER_PARTS.remove(part.key)


def test_a_part_without_tabs_of_its_own(dummy: DummyPart) -> None:
    assert dummy in parts.parts()
    option = dummy.samples(None).option(("toy",))
    run = run_stages(dummy, option, None)
    assert [t.key for t in parts.training_view(dummy, option, run.trained)] == [
        "methods",
        "training-sample",
        "training-results",
    ]
    assert [t.key for t in parts.prediction_view(run.predictions)] == [
        "test-objects",
        "predictions",
        "predicted-classes",
    ]
    tabs = parts.evaluation_view(run.evaluation)
    assert [t.key for t in tabs] == ["per-object", "confusion"]
    per_object = tabs[0].variants[0].table
    assert [row[3] for row in per_object.rows] == ["✓ 1", "✓ 2", "✗ 1"]
    metrics = run.evaluation.metrics["toy"]["nearest"]
    assert metrics is not None
    assert metrics.correct == 2


# ---------------------------------------------------------------- csr compare predict (ADR-086)


def test_csr_compare_predict(tmp_path: Path, article: StageRun) -> None:
    out = tmp_path / "blind.csv"
    result = runner.invoke(
        app,
        ["compare", "predict", "--split", "341", "--out", str(out), "--splits", str(SPLITS_FOLDER)],
    )
    assert result.exit_code == 0, result.output
    head = b'\xef\xbb\xbfObject,SVM,Random Forest,"k-NN Euclidean, k = 3"'
    assert out.read_bytes().startswith(head)  # a byte-order mark (ADR-083)
    with out.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    assert header[0] == "Object"
    assert len(header) == 1 + 14  # 8 standard methods, 6 CS-models
    assert not any("class" in name.lower() for name in header)
    assert len(rows) == 1 + 659
    decided = article.predictions.decisions["German-341"]
    for column, key in enumerate(decided, start=1):
        values = decided[key].decisions
        assert values is not None
        assert [int(row[column]) for row in rows[1:]] == values.tolist(), key
    assert [row[0] for row in rows[1:]] == list(article.samples.test[1].object_ids)
    assert "659 test objects classified without their class" in result.output


def test_csr_compare_predict_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unknown = runner.invoke(app, ["compare", "predict", "--split", "999"])
    assert unknown.exit_code == EXIT_USAGE
    assert "unknown split '999'" in unknown.output
    missing = runner.invoke(
        app,
        [
            "compare",
            "predict",
            "--splits",
            str(tmp_path / "none"),
            "--out",
            str(tmp_path / "x.csv"),
        ],
    )
    assert missing.exit_code == EXIT_USAGE
    assert "no folder of split files" in missing.output
    assert not (tmp_path / "x.csv").exists()
    monkeypatch.chdir(tmp_path)
    default = runner.invoke(
        app, ["compare", "predict", "--split", "76", "--splits", str(SPLITS_FOLDER)]
    )
    assert default.exit_code == 0, default.output
    assert (tmp_path / "predictions-Heart-Disease-76.csv").is_file()


def test_blind_predictions_build_no_answer_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from context_synthetic_recognition.data import splits

    def no_answers(*args: object, **kwargs: object) -> AnswerKey:
        raise AssertionError("the answer key was built")

    monkeypatch.setattr(splits.FixedSplit, "test_answers", no_answers)
    predictions = blind_predictions("76", SPLITS_FOLDER)
    assert predictions.objects[0].m == 194
    assert all(isinstance(o, UnlabelledSample) for o in predictions.objects)


def test_the_check_of_the_part(tmp_path: Path) -> None:
    part = comparison_part()
    check = part.check(tmp_path, SPLITS_FOLDER)
    assert not check.passed
    assert check.table.rows == ((CHECKSUMS, "✗", f"missing in {tmp_path}"),)
    tab = parts.check_tab(check)
    assert tab.key == "check"
    assert tab.variants[0].cells[1] is CellMeaning.STATUS
