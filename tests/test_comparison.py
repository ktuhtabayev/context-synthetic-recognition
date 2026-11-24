"""Tables 7 and 8: standard classifiers on the article's training and test samples (ADR-079),
and the CS-model on the same samples (ADR-080).

The pinned numbers are those of the project's own run (scikit-learn 1.9.1, numpy 2.5.3); they
agree with an independent run of the same settings on the same files.
"""

import hashlib
import importlib.util
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from typer.testing import CliRunner

from context_synthetic_recognition.cli.app import EXIT_FAILED, EXIT_USAGE, app
from context_synthetic_recognition.comparison import (
    ALPHAS,
    HEART_SPLIT,
    SPLITS,
    SplitResult,
    SplitSpec,
    compare_split,
    comparison_config,
    cs_model_config,
    cs_model_metrics,
    run_cs_model,
)
from context_synthetic_recognition.comparison.tables import (
    cs_model_table,
    latex_caption,
    results_table,
    summary_data,
    table_7,
    table_8,
)
from context_synthetic_recognition.config import DatasetConfig
from context_synthetic_recognition.config.presets import active_deviations
from context_synthetic_recognition.data import (
    Dataset,
    FixedSplit,
    load_builtin,
    load_from_config,
    load_split_file,
)
from context_synthetic_recognition.data.catalog import ENVIRONMENT_VARIABLE
from context_synthetic_recognition.errors import DatasetError, ModelUndefinedError
from context_synthetic_recognition.services import comparison as service
from context_synthetic_recognition.services.comparison import (
    CHECKSUMS,
    ComparisonOutcome,
    check_comparison,
    check_table,
    default_output,
    library_versions,
    run_comparison,
    splits_folder,
)

from .conftest import ROOT

SPLITS_FOLDER = ROOT / "datasets" / "splits"
OUT = ROOT / "experiments" / "comparison"
runner = CliRunner()

FILES = {
    "German/German (1000, 20, 2) — training 336 (zhuravlyov).dat": (
        "de91b117361122bdf22430a56ea4b2f06582739e037845d2e799a8da46b56135"
    ),
    "German/German (1000, 7, 2) — training 341 (euclidean).dat": (
        "2a11746d5b41579140337ee6bc1b53879d06f80c65d3fe32eff00bd857db3052"
    ),
    "German/German (1000, 7, 2) — training 367 (chebyshev).dat": (
        "1c818e6772cae85394d0134e3e29a0d6116d58acee0bb9964d1327f00948f772"
    ),
    "Heart-Disease/Heart-Disease (270, 13, 2) — training 76 (zhuravlyov).dat": (
        "ef9e5c65815b5f7c8e35dc0ad3d462c8394d10e05d299ca723540e202c153833"
    ),
}
"""The article's split files, in the split-file convention (ADR-082)."""

COUNTS = {
    "336": ((336, 196, 140), (664, 504, 160), 20),
    "341": ((341, 173, 168), (659, 527, 132), 7),
    "367": ((367, 190, 177), (633, 510, 123), 7),
    "76": ((76, 36, 40), (194, 114, 80), 13),
}
"""Training (n, K1, K2), test (n, K1, K2) and features of every split (7 quantitative each)."""

MAJORITY = {"336": 75.90, "341": 79.97, "367": 80.57}
"""Majority-class rate of the test sample, %."""

PINNED = {
    "336": {
        "svm": (74.85, 70.63, 83.63),
        "random-forest": (74.25, 69.81, 100.0),
        "knn-euclidean-3": (57.38, 50.60, 58.93),
        "knn-euclidean-5": (60.09, 51.74, 55.95),
        "knn-chebyshev-3": (54.37, 46.48, 54.17),
        "knn-chebyshev-5": (56.78, 48.71, 55.36),
        "knn-canberra-3": (57.08, 51.68, 61.01),
        "knn-canberra-5": (60.09, 53.02, 58.93),
        "knn-zhuravlyov-3": (68.83, 61.55, 50.30),
        "knn-zhuravlyov-5": (73.49, 64.20, 54.17),
    },
    "341": {
        "svm": (50.68, 52.41, 59.53),
        "random-forest": (56.45, 52.61, 100.0),
        "knn-euclidean-3": (58.42, 54.98, 48.97),
        "knn-euclidean-5": (54.78, 49.58, 53.08),
        "knn-chebyshev-3": (58.88, 54.13, 48.09),
        "knn-chebyshev-5": (59.64, 53.75, 54.25),
        "knn-canberra-3": (54.63, 52.89, 46.04),
        "knn-canberra-5": (56.75, 55.07, 49.56),
    },
    "367": {
        "svm": (61.14, 57.99, 62.67),
        "random-forest": (59.87, 55.97, 100.0),
        "knn-euclidean-3": (57.50, 54.81, 50.68),
        "knn-euclidean-5": (59.87, 56.59, 50.68),
        "knn-chebyshev-3": (59.08, 51.47, 50.68),
        "knn-chebyshev-5": (60.19, 53.39, 50.14),
        "knn-canberra-3": (58.61, 53.96, 47.96),
        "knn-canberra-5": (60.03, 56.69, 48.50),
    },
}
"""Test accuracy, balanced accuracy (test) and training accuracy (Definition 2), %."""

CS_MODEL = {
    "336": {
        "cs-model-euclidean-0.1": (62.95, 51.92, 58.93, 2),
        "cs-model-euclidean-0.2": (69.13, 50.87, 59.23, 0),
        "cs-model-chebyshev-0.1": (30.87, 48.07, 44.94, 0),
        "cs-model-chebyshev-0.2": (30.87, 48.07, 44.94, 0),
        "cs-model-canberra-0.1": (70.18, 53.06, 61.01, 18),
        "cs-model-canberra-0.2": (69.88, 53.50, 61.31, 20),
        "cs-model-zhuravlyov-0.1": (71.54, 53.31, 62.20, 21),
        "cs-model-zhuravlyov-0.2": (68.07, 56.36, 63.69, 0),
    },
    "341": {
        "cs-model-euclidean-0.1": (56.90, 48.07, 53.08, 0),
        "cs-model-euclidean-0.2": (56.15, 52.42, 52.20, 0),
        "cs-model-chebyshev-0.1": (39.15, 49.18, 53.37, 0),
        "cs-model-chebyshev-0.2": (35.51, 50.59, 52.20, 0),
        "cs-model-canberra-0.1": (77.39, 51.23, 51.32, 0),
        "cs-model-canberra-0.2": (77.39, 51.23, 51.32, 0),
    },
    "367": {
        "cs-model-euclidean-0.1": (77.41, 48.66, 53.95, 0),
        "cs-model-euclidean-0.2": (77.41, 48.66, 53.95, 0),
        "cs-model-chebyshev-0.1": (78.36, 54.18, 54.77, 0),
        "cs-model-chebyshev-0.2": (29.07, 51.04, 51.23, 0),
        "cs-model-canberra-0.1": (21.33, 49.33, 49.05, 0),
        "cs-model-canberra-0.2": (21.33, 49.33, 49.05, 0),
    },
}
"""The CS-model (ADR-080): test accuracy (refusals as errors), balanced accuracy, training
accuracy (Definition 2), %, and the refusals on the test sample."""


def percent(value: float) -> float:
    return round(100 * value, 2)


@pytest.fixture(scope="module")
def outcome() -> ComparisonOutcome:
    """One run of the comparison on the project's split files."""
    return run_comparison(SPLITS_FOLDER)


def script() -> ModuleType:
    """``scripts/run_comparison.py`` as a module."""
    spec = importlib.util.spec_from_file_location(
        "run_comparison", ROOT / "scripts" / "run_comparison.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- the split files


def test_the_split_files_keep_their_bytes() -> None:
    found = {p.relative_to(SPLITS_FOLDER).as_posix() for p in SPLITS_FOLDER.rglob("*.dat")}
    assert found == set(FILES)
    for name, digest in FILES.items():
        assert hashlib.sha256((SPLITS_FOLDER / name).read_bytes()).hexdigest() == digest, name
    assert {spec.file for spec in (*SPLITS, HEART_SPLIT)} == set(FILES)


@pytest.mark.parametrize("spec", [*SPLITS, HEART_SPLIT], ids=lambda spec: spec.key)
def test_the_split_files_give_the_articles_samples(spec: SplitSpec) -> None:
    split = load_split_file(SPLITS_FOLDER / spec.file)
    training, test, n = COUNTS[spec.key]
    assert split.training_sizes == training
    assert split.test_sizes == test
    assert split.dataset.n == n
    assert int(split.dataset.quantitative.sum()) == 7
    # the training objects come first, then the test objects
    assert split.training.tolist() == list(range(training[0]))


def test_the_split_files_hold_the_projects_datasets() -> None:
    german = load_from_config(DatasetConfig(path="german-1000"), ROOT)
    quantitative = np.flatnonzero(german.quantitative)

    def objects(X: np.ndarray, y: np.ndarray) -> Counter[tuple[float, ...]]:
        return Counter(map(tuple, np.column_stack([np.round(X, 9), y]).tolist()))

    def minmax(X: np.ndarray, columns: np.ndarray) -> np.ndarray:
        Z = X.copy()
        low, high = Z[:, columns].min(axis=0), Z[:, columns].max(axis=0)
        Z[:, columns] = (Z[:, columns] - low) / (high - low)
        return Z

    for spec in SPLITS[1:]:  # the 7 quantitative features, raw values
        data = load_split_file(SPLITS_FOLDER / spec.file).dataset
        assert objects(data.X, data.y) == objects(german.X[:, quantitative], german.y)
    # all 20 features, the quantitative ones min–max scaled over all 1000 objects
    data = load_split_file(SPLITS_FOLDER / SPLITS[0].file).dataset
    assert objects(data.X, data.y) == objects(minmax(german.X, quantitative), german.y)
    heart = load_builtin("heart-disease-270").with_feature_types({"x₁₁": "quantitative"})
    data = load_split_file(SPLITS_FOLDER / HEART_SPLIT.file).dataset
    assert data.quantitative.tolist() == heart.quantitative.tolist()  # the slope too (ADR-072)
    columns = np.flatnonzero(heart.quantitative)
    assert objects(data.X, data.y) == objects(minmax(heart.X, columns), heart.y)


# ---------------------------------------------------------------- the numbers


def test_the_pinned_numbers(outcome: ComparisonOutcome) -> None:
    assert [r.spec.key for r in outcome.results] == ["336", "341", "367"]
    for result in outcome.results:
        training, test, n = COUNTS[result.spec.key]
        assert (result.training_sizes, result.test_sizes, result.n_features) == (
            training,
            test,
            n,
        )
        assert result.n_quantitative == 7
        assert percent(result.majority_rate) == MAJORITY[result.spec.key]
        found = {
            m.key: (
                percent(m.test.accuracy),
                percent(m.test.macro_recall),
                percent(m.training.accuracy),
            )
            for m in result.methods
        }
        assert found == PINNED[result.spec.key], result.spec.key
        for method in result.methods:
            assert method.test.refusals == 0
            assert method.test.n == test[0]
            assert method.training.n == training[0]


def test_no_method_reaches_the_majority_class_rate(outcome: ComparisonOutcome) -> None:
    for result in outcome.results:
        best = max(m.test.accuracy for m in result.methods)
        assert best < result.majority_rate


def test_the_methods_and_their_tables(outcome: ComparisonOutcome) -> None:
    german_336 = outcome.results[0]
    assert [m.title for m in german_336.methods[:4]] == [
        "SVM",
        "Random Forest",
        "k-NN Euclidean, k = 3",
        "k-NN Euclidean, k = 5",
    ]
    assert {m.key: m.table for m in german_336.methods} == {
        "svm": "Table 7",
        "random-forest": "Table 7",
        **{
            f"knn-{m}-{k}": "Table 8"
            for m in ("euclidean", "chebyshev", "canberra")
            for k in (3, 5)
        },
        "knn-zhuravlyov-3": None,
        "knn-zhuravlyov-5": None,
    }
    features = {m.key: (m.features, m.n_features) for m in german_336.methods}
    assert features["svm"] == ("all", 20)
    assert features["knn-euclidean-3"] == ("quantitative", 7)
    assert features["knn-zhuravlyov-5"] == ("all", 20)
    assert german_336.column == "336 (196, 140)"
    with pytest.raises(KeyError):
        german_336.method("logistic-regression")
    config = comparison_config(SPLITS[0])
    assert config.seed == 42
    assert config.local_context.distance_decimals == 10
    assert [o.label for o in config.local_context.operators] == [
        "Euclidean",
        "Chebyshev",
        "Canberra",
        "Zhuravlyov",
    ]
    assert len(comparison_config(SPLITS[1]).local_context.operators) == 3


def test_tables_7_and_8(outcome: ComparisonOutcome) -> None:
    seven, eight, _, results = outcome.tables
    columns = (
        "336 (196, 140) — Zhuravlyov split",
        "341 (173, 168) — Euclidean split",
        "367 (190, 177) — Chebyshev split",
    )
    assert seven.columns == ("Method", *columns)
    assert seven.rows == (
        ("SVM", 74.85, 50.68, 61.14),
        ("Random Forest", 74.25, 56.45, 59.87),
    )
    assert eight.columns == ("Metric", *columns)
    assert eight.rows == (
        ("Euclidean", "57.38 / 60.09", "58.42 / 54.78", "57.50 / 59.87"),
        ("Chebyshev", "54.37 / 56.78", "58.88 / 59.64", "59.08 / 60.19"),
        ("Canberra", "57.08 / 60.09", "54.63 / 56.75", "58.61 / 60.03"),
    )
    majority = (
        "Majority-class rate of the test sample: 75.90 % (336), 79.97 % (341), 80.57 % (367)."
    )
    assert majority in seven.note
    assert majority in eight.note
    assert "random_state = 42" in seven.note
    assert "10 decimals" in eight.note
    assert len(results.rows) == 46  # 26 methods, 20 CS-models
    assert results.rows[0][:5] == ("336", "Zhuravlyov", "SVM", "all (20)", "Table 7")
    assert results.rows[8][2:5] == ("k-NN Zhuravlyov, k = 3", "all (20)", None)
    assert table_7([]).rows == (("SVM",), ("Random Forest",))
    assert table_8(outcome.results).rows == eight.rows
    assert results_table(outcome.results[:1]).rows == results.rows[:18]
    assert latex_caption(seven).title == "Accuracy of SVM and random forest on the test samples, %"
    assert latex_caption(results) is results


def test_the_summary(outcome: ComparisonOutcome) -> None:
    summary = json.loads(outcome.files["summary.json"])
    assert summary["versions"] == library_versions()
    assert summary["settings"]["svm"] == {"kernel": "rbf", "C": 1.0, "gamma": "scale"}
    assert summary["settings"]["random-forest"] == {"n_estimators": 200, "random_state": 42}
    assert summary["settings"]["knn"]["ks"] == [3, 5]
    split = summary["splits"][0]
    assert split["file_sha256"] == FILES[split["file"]]
    assert split["training"] == {"n": 336, "K1": 196, "K2": 140}
    assert split["majority_rate"] == 75.9
    svm = split["methods"][0]
    assert svm["test"] == {
        "accuracy": 74.85,
        "balanced_accuracy": 70.63,
        "correct": 497,
        "n": 664,
        "refusals": 0,
    }
    assert svm["training"]["correct"] == 281
    assert summary["settings"]["cs_model"]["alphas"] == [0.1, 0.2]
    assert summary["settings"]["cs_model"]["k"] == "odd k = 3 … 41"
    zhuravlyov = split["cs_model"][6]
    assert zhuravlyov["key"] == "cs-model-zhuravlyov-0.1"
    assert (zhuravlyov["r"], zhuravlyov["left_out_zero"]) == (20, 0)
    assert zhuravlyov["test"]["refusals"] == 21
    assert zhuravlyov["test"]["accuracy_answered"] == 73.87


def test_two_classes_are_needed(outcome: ComparisonOutcome) -> None:
    split = load_split_file(SPLITS_FOLDER / SPLITS[1].file)
    data = split.dataset
    labels = np.where(np.arange(data.m) % 7 == 0, 3, data.y)
    three = Dataset(X=data.X, y=labels, feature_types=data.feature_types)
    with pytest.raises(ValueError, match="two classes"):
        compare_split(SPLITS[1], FixedSplit(three, split.training, split.test))
    assert isinstance(outcome.results[0], SplitResult)


# ---------------------------------------------------------------- the committed results


def test_the_checksums_of_the_committed_results() -> None:
    lines = (OUT / CHECKSUMS).read_text(encoding="utf-8").splitlines()
    assert [line.split("  ")[1] for line in lines] == [
        "table-7.csv",
        "table-7.md",
        "table-7.tex",
        "table-8.csv",
        "table-8.md",
        "table-8.tex",
        "cs-model.csv",
        "cs-model.md",
        "cs-model.tex",
        "results.csv",
        "results.md",
        "summary.json",
    ]
    for line in lines:
        digest, name = line.split("  ")
        assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest, name
    assert sorted(p.name for p in OUT.iterdir()) == sorted([*committed_names(), CHECKSUMS])


def test_the_committed_tables_hold_the_pinned_numbers() -> None:
    seven = (OUT / "table-7.csv").read_text(encoding="utf-8-sig").splitlines()
    assert seven[1:] == ["SVM,74.85,50.68,61.14", "Random Forest,74.25,56.45,59.87"]
    # the CSV files carry a byte-order mark, so that Excel reads them as UTF-8 (ADR-083)
    for name in ("table-7.csv", "table-8.csv", "cs-model.csv", "results.csv"):
        data = (OUT / name).read_bytes()
        assert data.startswith(b"\xef\xbb\xbf"), name
        assert data.count(b"\xef\xbb\xbf") == 1, name
    for name in ("table-7.md", "table-7.tex", "summary.json", CHECKSUMS):
        assert not (OUT / name).read_bytes().startswith(b"\xef\xbb\xbf"), name
    eight = (OUT / "table-8.md").read_text(encoding="utf-8")
    assert "| Chebyshev | 54.37 / 56.78 | 58.88 / 59.64 | 59.08 / 60.19 |" in eight
    tex = (OUT / "table-7.tex").read_text(encoding="utf-8")
    assert r"\label{tab:table-7}" in tex
    assert "SVM & 74.85 & 50.68 & 61.14" in tex


def test_a_new_run_reproduces_every_result_byte_for_byte(tmp_path: Path) -> None:
    recorded = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))["versions"]
    if recorded != library_versions():
        pytest.skip(f"the results were written with {recorded}, installed: {library_versions()}")
    files = script().run(tmp_path)
    assert [f.name for f in files] == [*committed_names(), CHECKSUMS]
    for file in files:
        assert file.read_bytes() == (OUT / file.name).read_bytes(), file.name
    check = check_comparison(OUT, splits=SPLITS_FOLDER)
    assert check.passed, check.problems
    assert check.same_versions


def committed_names() -> list[str]:
    lines = (OUT / CHECKSUMS).read_text(encoding="utf-8").splitlines()
    return [line.split("  ")[1] for line in lines]


# ---------------------------------------------------------------- the service


def test_the_project_folders(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(ENVIRONMENT_VARIABLE, raising=False)
    assert splits_folder(ROOT) == SPLITS_FOLDER
    assert default_output(ROOT) == OUT
    with pytest.raises(DatasetError, match="no datasets folder"):
        splits_folder(tmp_path)
    with pytest.raises(DatasetError, match="no datasets folder"):
        default_output(tmp_path)
    (tmp_path / "datasets").mkdir()
    with pytest.raises(DatasetError, match="no 'splits' folder"):
        splits_folder(tmp_path)
    assert default_output(tmp_path) == tmp_path / "experiments" / "comparison"
    # an installation from source carries no copy; the Windows build does (ADR-081)
    assert service.bundled_splits() is None
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    monkeypatch.setattr(service, "bundled_splits", lambda: bundle)
    assert splits_folder(tmp_path) == bundle
    (tmp_path / "datasets" / "splits").mkdir()
    assert splits_folder(tmp_path) == tmp_path / "datasets" / "splits"  # the project's first


def test_a_missing_library_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def version(name: str) -> str:
        raise service.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(service.metadata, "version", version)
    assert library_versions() == {"numpy": "missing", "scikit-learn": "missing"}


def test_checking_a_folder_without_results(tmp_path: Path) -> None:
    check = check_comparison(tmp_path)
    assert not check.passed
    assert check.problems == (f"{CHECKSUMS}: missing in {tmp_path}",)
    assert check.checked == ()
    table = check_table(check)
    assert table.rows == ((CHECKSUMS, "✗", f"missing in {tmp_path}"),)
    assert check.recorded_versions == {}  # nothing was re-run or read
    assert check.same_versions
    assert table.note.endswith("written with —.")


def test_a_damaged_summary_records_no_versions(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text("[1, 2]", encoding="utf-8")
    assert service._recorded_versions(tmp_path) == {}
    (tmp_path / "summary.json").write_text("{", encoding="utf-8")
    assert service._recorded_versions(tmp_path) == {}


# ---------------------------------------------------------------- csr compare


def test_csr_compare_run_and_check(tmp_path: Path) -> None:
    out = tmp_path / "comparison"
    ran = runner.invoke(app, ["compare", "run", "-o", str(out), "--splits", str(SPLITS_FOLDER)])
    assert ran.exit_code == 0, ran.output
    assert "| SVM | 74.85 | 50.68 | 61.14 |" in ran.output
    assert "| Canberra | 57.08 / 60.09 | 54.63 / 56.75 | 58.61 / 60.03 |" in ran.output
    assert "| Zhuravlyov | 71.54 / 68.07 | — | — |" in ran.output
    assert sorted(p.name for p in out.iterdir()) == sorted([*committed_names(), CHECKSUMS])
    checked = runner.invoke(
        app, ["compare", "check", "-o", str(out), "--splits", str(SPLITS_FOLDER)]
    )
    assert checked.exit_code == 0, checked.output
    assert "✓ table-7.csv" in checked.output
    assert "comparison check passed" in checked.output

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    summary["versions"]["scikit-learn"] = "0.0"
    (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    failed = runner.invoke(
        app, ["compare", "check", "-o", str(out), "--splits", str(SPLITS_FOLDER)]
    )
    assert failed.exit_code == EXIT_FAILED
    assert "✗ summary.json" in failed.output
    assert "summary.json: does not match checksums.sha256" in failed.output
    assert "written with numpy" in failed.output
    assert "scikit-learn 0.0; installed:" in failed.output
    assert "comparison check FAILED" in failed.output


def test_csr_compare_reports_missing_split_files(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["compare", "run", "-o", str(tmp_path), "--splits", str(tmp_path / "none")]
    )
    assert result.exit_code == EXIT_USAGE
    assert "split file not found" in result.output
    result = runner.invoke(
        app, ["compare", "check", "-o", str(tmp_path), "--splits", str(tmp_path / "none")]
    )
    assert result.exit_code == EXIT_FAILED  # no checksums file: nothing is re-run
    assert f"{CHECKSUMS}: missing" in result.output
    (tmp_path / CHECKSUMS).write_text("", encoding="utf-8")
    result = runner.invoke(
        app, ["compare", "check", "-o", str(tmp_path), "--splits", str(tmp_path / "none")]
    )
    assert result.exit_code == EXIT_USAGE  # the new run cannot read the split files
    assert "split file not found" in result.output


def test_the_documentation_shows_the_committed_tables() -> None:
    page = (ROOT / "docs" / "comparison.md").read_text(encoding="utf-8")
    for name in ("table-7.md", "table-8.md", "cs-model.md"):
        lines = (OUT / name).read_text(encoding="utf-8").splitlines()
        rows = [line for line in lines if line.startswith("|")]
        assert len(rows) >= 4
        assert "\n".join(rows) in page, name


# ---------------------------------------------------------------- the CS-model (ADR-080)


def test_the_cs_model_pinned_numbers(outcome: ComparisonOutcome) -> None:
    for result in outcome.results:
        found = {
            c.key: (
                percent(c.test.accuracy),
                percent(c.test.macro_recall),
                percent(c.training.accuracy),
                c.test.refusals,
            )
            for c in result.cs_model
            if c.test is not None and c.training is not None
        }
        assert found == CS_MODEL[result.spec.key], result.spec.key
        for model in result.cs_model:
            # k = 3 … 41: 20 synthetic features, none takes 0 on a training object (ADR-070)
            assert (model.r, model.left_out_zero, model.undefined) == (20, 0, "")
            assert model.tuplam
            assert model.training is not None
            assert model.training.n == result.training_sizes[0]
    zhuravlyov = outcome.results[0].cs_model[6]
    assert zhuravlyov.title == "CS-model Zhuravlyov, α = 0.1"
    assert (zhuravlyov.features, zhuravlyov.n_features) == ("all", 20)
    assert zhuravlyov.tuplam == ("ρ·41", "ρ·3", "ρ·5", "ρ·15", "ρ·13")
    assert outcome.results[1].cs_model[0].n_features == 7


def test_the_cs_model_settings() -> None:
    assert ALPHAS == (0.1, 0.2)
    assert cs_model_metrics(SPLITS[0]) == ("euclidean", "chebyshev", "canberra", "zhuravlyov")
    assert cs_model_metrics(SPLITS[2]) == ("euclidean", "chebyshev", "canberra")
    config = cs_model_config("canberra", 0.2, all_features=False)
    (operator,) = config.local_context.operators
    assert (operator.metric.name, operator.features) == ("canberra", "quantitative")
    assert operator.normalizer is not None
    assert operator.normalizer.name == "minmax"
    assert config.admissible_k.name == "range"
    assert config.synthetic_features.zero_value
    assert config.additional_features.alpha == 0.2
    assert active_deviations(config) == []  # no ⚠ switch is on
    assert cs_model_config("zhuravlyov", 0.1, all_features=True).local_context.operators[
        0
    ].features == ("all")


def test_the_cs_model_table(outcome: ComparisonOutcome) -> None:
    table = outcome.tables[2]
    assert table.key == "cs-model"
    assert table.title.startswith("CS-model on the same samples")
    assert table.rows == (
        ("Euclidean", "62.95 / 69.13", "56.90 / 56.15", "77.41 / 77.41"),
        ("Chebyshev", "30.87 / 30.87", "39.15 / 35.51", "78.36 / 29.07"),
        ("Canberra", "70.18 / 69.88", "77.39 / 77.39", "21.33 / 21.33"),
        ("Zhuravlyov", "71.54 / 68.07", None, None),
    )
    assert "Refusals count as errors (Definition 3)" in table.note
    assert latex_caption(table) is table


def test_no_cs_model_reaches_the_majority_class_rate(outcome: ComparisonOutcome) -> None:
    for result in outcome.results:
        best = max(c.test.accuracy for c in result.cs_model if c.test is not None)
        assert best < result.majority_rate
        # near chance on both classes: the balanced accuracy stays below 57 %
        assert max(c.test.macro_recall for c in result.cs_model if c.test is not None) < 0.57


def test_an_undefined_cs_model_is_reported(
    outcome: ComparisonOutcome, monkeypatch: pytest.MonkeyPatch
) -> None:
    from context_synthetic_recognition.comparison import cs_model as module

    def undefined(*args: object) -> None:
        raise ModelUndefinedError("no admissible k")

    monkeypatch.setattr(module, "fit_model", undefined)
    split = load_split_file(SPLITS_FOLDER / SPLITS[1].file)
    models = run_cs_model(SPLITS[1], split)
    assert len(models) == 6
    assert {m.undefined for m in models} == {"no admissible k"}
    assert all(m.test is None and m.r is None for m in models)
    result = replace(outcome.results[1], cs_model=models)
    assert cs_model_table([result]).rows[0] == ("Euclidean", "— / —")
    row = results_table([result]).rows[8]
    assert row[2] == "CS-model Euclidean, α = 0.1"
    assert row[12:] == (None,) * 7
    data = summary_data([result], versions={}, inputs={SPLITS[1].file: "x"})
    assert data["splits"][0]["cs_model"][0] == {
        "key": "cs-model-euclidean-0.1",
        "title": "CS-model Euclidean, α = 0.1",
        "undefined": "no admissible k",
    }
