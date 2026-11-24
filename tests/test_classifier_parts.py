"""Classifier parts: the three stages, the blind test and "Run all" (ADR-084, ADR-085, ADR-086)."""

import csv
import json
from collections.abc import Iterator
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from context_synthetic_recognition.cli.app import EXIT_USAGE, app
from context_synthetic_recognition.comparison import SPLITS
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
    ARTICLE,
    NEEDS_ARTICLE,
    WRITES_ARTICLE,
    TablesPart,
    blind_predictions,
)

from .conftest import ROOT
from .dummy_part import DummyPart

SPLITS_FOLDER = ROOT / "datasets" / "splits"
OUT = ROOT / "experiments" / "comparison"
runner = CliRunner()


def tables_part() -> TablesPart:
    part = CLASSIFIER_PARTS.get("tables-7-8")
    assert isinstance(part, TablesPart)
    return part


@pytest.fixture(scope="module")
def article() -> StageRun:
    """Training, test and evaluation on the three article splits (one "Run all")."""
    part = tables_part()
    return run_stages(part, part.option(ARTICLE), SPLITS_FOLDER)


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
    part = tables_part()
    option = part.option(ARTICLE)
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
    for split in ("336", "341", "367"):
        assert list(predictions.decisions[split]) == list(reference.decisions[split])
        for method, decided in predictions.decisions[split].items():
            assert decided.same_as(reference.decisions[split][method]), (split, method)
            kinds.add(method.rsplit("-", 2)[0] if method.startswith("cs-") else method)
    # SVM, random forest, every k-NN row and the CS-model
    assert {"svm", "random-forest", "knn-zhuravlyov-5", "cs-model"} <= kinds
    assert len(predictions.decisions["336"]) == 18


def test_a_dataset_cannot_be_classified(article: StageRun) -> None:
    part = tables_part()
    split = load_split_file(SPLITS_FOLDER / SPLITS[0].file)
    with pytest.raises(TypeError, match="not a Dataset"):
        parts.classify(part, article.trained, (split.test_part,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="not a Dataset"):
        part.classify(article.trained, (split.test_part,))  # type: ignore[arg-type]
    fitted = article.trained.fitted[0]
    with pytest.raises(TypeError, match="not a Dataset"):
        fitted.methods[0].classify(split.test_part)  # type: ignore[arg-type]


def test_mypy_rejects_a_dataset(tmp_path: Path) -> None:
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
    part = tables_part()
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
        "table-7",
        "table-8",
        "cs-model",
        "all-results",
        "per-object",
        "confusion",
    ]
    (seven,) = tabs["table-7"].variants
    assert seven.table.rows == (
        ("SVM", 74.85, 50.68, 61.14),
        ("Random Forest", 74.25, 56.45, 59.87),
    )
    assert len(tabs["all-results"].variants[0].table.rows) == 46
    per_object = {v.label: v for v in tabs["per-object"].variants}
    assert list(per_object) == ["336", "341", "367"]
    table = per_object["336"].table
    assert table.columns[:3] == ("Object", "True class", "Correct methods")
    assert len(table.columns) == 3 + 18
    marks = {str(v)[:1] for row in table.rows for v in row[3:]}
    assert marks <= {"✓", "✗", "0"}
    svm = article.evaluation.metrics["336"]["svm"]
    assert svm is not None
    assert sum(1 for row in table.rows if str(row[3]).startswith("✓")) == svm.correct == 497
    assert per_object["336"].cells[1] is CellMeaning.CLASS
    assert set(per_object["336"].cells[3:]) == {CellMeaning.OUTCOME}
    confusion = tabs["confusion"].variants[0].table
    first = confusion.rows[0]
    assert first[:2] == ("336", "SVM")
    assert sum(first[2:8]) == 664  # type: ignore[arg-type]
    assert first[8] == 74.85


def test_a_single_split_has_no_tables_7_and_8(tmp_path: Path) -> None:
    part = tables_part()
    run = run_stages(part, part.option("341"), SPLITS_FOLDER)
    tabs = {t.key: t for t in run.evaluation.tables}
    assert tabs["table-7"].missing == NEEDS_ARTICLE
    assert not tabs["table-7"].variants
    (everything,) = tabs["all-results"].variants
    assert len(everything.table.rows) == 14  # 8 methods, 6 CS-models
    assert part.write_problem(run.evaluation) == WRITES_ARTICLE
    with pytest.raises(DatasetError, match="Only the full article set"):
        part.write(run.evaluation, tmp_path)
    assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------- the stages' tabs


def test_the_training_and_test_tabs(article: StageRun) -> None:
    part = tables_part()
    training = {
        t.key: t for t in parts.training_view(part, article.trained.option, article.trained)
    }
    assert list(training) == ["methods", "training-sample", "training-results"]
    methods = training["methods"].variants[0].table
    assert {row[0] for row in methods.rows} == {"Table 7", "Table 8", "Not printed", "CS-model"}
    assert methods.rows[8][:4] == ("Not printed", "k-NN Zhuravlyov, k = 3", "all", "336")
    sample = {v.label: v for v in training["training-sample"].variants}["341"]
    assert sample.table.columns[-1] == "Class"
    assert len(sample.table.rows) == 341
    assert sample.cells[-1] is CellMeaning.CLASS
    results = training["training-results"].variants[0].table
    svm = next(r for r in results.rows if r[:2] == ("336", "SVM"))
    assert svm[2:] == (336, 83.63, 281, 0)

    test = {t.key: t for t in parts.prediction_view(article.predictions)}
    assert list(test) == ["test-objects", "predictions", "predicted-classes"]
    objects = {v.label: v for v in test["test-objects"].variants}["367"]
    assert objects.table.columns == ("Object", *article.samples.test[2].feature_names)
    assert "Class" not in objects.table.columns
    assert len(objects.table.rows) == 633
    predictions = {v.label: v for v in test["predictions"].variants}["336"]
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


def test_the_sample_options() -> None:
    part = tables_part()
    options = {o.key: o for o in part.sample_options(SPLITS_FOLDER)}
    assert list(options) == ["article", "336", "341", "367", "76"]
    assert options["article"].label == "All article splits (336, 341, 367) · test 664 / 659 / 633"
    assert options["article"].complete
    assert options["336"].label == "336 · training 336 (196, 140) · test 664"
    assert options["76"].label == "76 · training 76 (36, 40) · test 194"
    assert all(o.available for o in options.values())


def test_unavailable_sample_options(tmp_path: Path) -> None:
    part = tables_part()
    (tmp_path / "German").mkdir()
    for spec in SPLITS[1:]:
        (tmp_path / spec.file).write_bytes((SPLITS_FOLDER / spec.file).read_bytes())
    options = {o.key: o for o in part.sample_options(tmp_path)}
    assert not options["article"].available
    assert "split file not found" in options["article"].reason
    assert not options["336"].available
    assert options["341"].available
    with pytest.raises(DatasetError, match="unknown split 'x'"):
        part.option("x")
    missing = part.sample_options(tmp_path / "none" / "nested")
    assert all(not o.available for o in missing)


def test_the_registry_and_the_capabilities() -> None:
    assert "tables-7-8" in [p.key for p in parts.parts()]
    part = tables_part()
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
    (option,) = dummy.sample_options(None)
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
    decided = article.predictions.decisions["341"]
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
    assert "split file not found" in missing.output
    assert not (tmp_path / "x.csv").exists()
    monkeypatch.chdir(tmp_path)
    default = runner.invoke(
        app, ["compare", "predict", "--split", "76", "--splits", str(SPLITS_FOLDER)]
    )
    assert default.exit_code == 0, default.output
    assert (tmp_path / "predictions-76.csv").is_file()


def test_blind_predictions_build_no_answer_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from context_synthetic_recognition.data import splits

    def no_answers(*args: object, **kwargs: object) -> AnswerKey:
        raise AssertionError("the answer key was built")

    monkeypatch.setattr(splits.FixedSplit, "test_answers", no_answers)
    predictions = blind_predictions("76", SPLITS_FOLDER)
    assert predictions.objects[0].m == 194
    assert all(isinstance(o, UnlabelledSample) for o in predictions.objects)


def test_the_check_of_the_part(tmp_path: Path) -> None:
    part = tables_part()
    check = part.check(tmp_path, SPLITS_FOLDER)
    assert not check.passed
    assert check.table.rows == ((CHECKSUMS, "✗", f"missing in {tmp_path}"),)
    tab = parts.check_tab(check)
    assert tab.key == "check"
    assert tab.variants[0].cells[1] is CellMeaning.STATUS
