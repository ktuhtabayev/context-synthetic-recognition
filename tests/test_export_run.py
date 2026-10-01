"""Exporting a run: the formats, the run folder, ``csr run --export`` and ``csr export``."""

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from context_synthetic_recognition.cli.app import EXIT_USAGE, app
from context_synthetic_recognition.config import DatasetConfig, ExperimentConfig
from context_synthetic_recognition.data import Dataset, load_builtin
from context_synthetic_recognition.data.snapshot import (
    SNAPSHOT_NAME,
    dataset_data,
    dataset_from_data,
    load_snapshot,
    save_snapshot,
)
from context_synthetic_recognition.errors import ConfigError, DatasetError, RegistryError
from context_synthetic_recognition.export import (
    EXPORTERS,
    ExportOptions,
    RunView,
    export_result,
    export_view,
    resolve_formats,
)
from context_synthetic_recognition.export import run as export_module
from context_synthetic_recognition.export.excel import ExcelOptions
from context_synthetic_recognition.export.text import INDEX_COLUMNS
from context_synthetic_recognition.services.runner import (
    ExperimentResult,
    RunError,
    latest_run,
    load_run,
    results_dict,
    run_experiment,
    save_run,
)
from context_synthetic_recognition.services.validation import validate_workbook

from .conftest import EXPERIMENT_RUN_ID
from .shapes import shape
from .test_export_figures import EXPERIMENT_FIGURES
from .test_export_tables import EXPERIMENT_TABLES

runner = CliRunner()
FORMATS = ["excel", "csv", "json", "markdown", "latex", "figures", "html", "pdf"]
RUN_FILES = ["dataset.json", "folds.csv", "manifest.yaml", "predictions.csv", "results.json"]


def trace_tables(folder: Path) -> dict[str, Any]:
    data = json.loads((folder / "trace.json").read_text(encoding="utf-8"))
    return {table["key"]: table for table in data["tables"]}


# ---------------------------------------------------------------- dataset snapshots


def test_a_snapshot_keeps_the_dataset(experiment: Dataset, tmp_path: Path) -> None:
    file = save_snapshot(experiment, tmp_path)
    assert file == tmp_path / SNAPSHOT_NAME
    data = file.read_bytes()
    assert data.endswith(b"}\n")
    assert b"\r" not in data
    for loaded in (load_snapshot(tmp_path), load_snapshot(file)):
        assert loaded.content_hash() == experiment.content_hash()
        assert loaded.name == experiment.name
        assert loaded.feature_names == experiment.feature_names
        assert loaded.feature_types == experiment.feature_types
        assert loaded.object_ids == experiment.object_ids
        assert loaded.X.tolist() == experiment.X.tolist()
        assert loaded.y.tolist() == experiment.y.tolist()


@pytest.mark.parametrize("name", ["numeric", "heart270"])
def test_snapshots_of_other_datasets(name: str, tmp_path: Path) -> None:
    dataset = shape(name).dataset  # text labels; category names
    loaded = load_snapshot(save_snapshot(dataset, tmp_path / "data.json"))
    assert loaded.content_hash() == dataset.content_hash()
    assert loaded.y.tolist() == dataset.y.tolist()
    assert loaded.categories == dataset.categories
    assert loaded.source == dataset.source


def test_snapshot_errors(experiment: Dataset, tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="cannot read the dataset snapshot"):
        load_snapshot(tmp_path)
    file = tmp_path / "broken.json"
    file.write_text("{", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid JSON"):
        load_snapshot(file)
    with pytest.raises(DatasetError, match="not a dataset snapshot of this package"):
        dataset_from_data({"format": "something else"})
    with pytest.raises(DatasetError, match="not a dataset snapshot of this package"):
        dataset_from_data([1, 2])
    incomplete = dataset_data(experiment)
    del incomplete["X"]
    with pytest.raises(DatasetError, match=r"here: the dataset snapshot is incomplete \('X'\)"):
        dataset_from_data(incomplete, source="here")
    changed = dataset_data(experiment)
    changed["X"][0][0] += 1.0
    with pytest.raises(DatasetError, match="does not match its recorded hash"):
        dataset_from_data(changed)
    del changed["content_hash"]  # a snapshot written by hand has no hash to check
    assert dataset_from_data(changed).X[0, 0] == experiment.X[0, 0] + 1.0


# ---------------------------------------------------------------- run folders


@pytest.fixture
def run_folder(
    experiment_result: ExperimentResult, experiment_view: RunView, tmp_path: Path
) -> Path:
    return save_run(experiment_result, tmp_path / "runs", sensitivity=experiment_view.sensitivity)


def test_a_run_folder_is_self_contained(
    run_folder: Path, experiment_result: ExperimentResult, experiment_view: RunView
) -> None:
    assert sorted(p.name for p in run_folder.iterdir()) == RUN_FILES
    stored = json.loads((run_folder / "results.json").read_text(encoding="utf-8"))
    assert [v["label"] for v in stored["sensitivity"]] == [
        "template (default)",
        "running centres, 1 pass",
        "final centres, 2 passes",
        "article",
    ]
    assert stored["sensitivity"][0]["centres"] == "running"
    loaded = load_run(run_folder)
    assert loaded.folder == run_folder
    assert loaded.manifest.run_id == run_folder.name
    assert loaded.warnings == ()
    assert loaded.sensitivity == experiment_view.sensitivity
    assert loaded.result.dataset.content_hash() == experiment_result.dataset.content_hash()
    repeated, original = results_dict(loaded.result), results_dict(experiment_result)
    for key in ("dataset", "config_hash", "model", "protocols", "margins", "properties"):
        assert repeated[key] == original[key], key


def test_the_latest_run(run_folder: Path, tmp_path: Path) -> None:
    runs = run_folder.parent
    assert latest_run(runs) == run_folder
    (runs / "notes").mkdir()
    (runs / "99999999_999999_zzzzzzzz").mkdir()  # not a run id (the hash is hexadecimal)
    (runs / "20991231_235959_0a1b2c3d.txt").write_text("a file", encoding="utf-8")
    assert latest_run(runs) == run_folder
    newer = runs / "20991231_235959_0a1b2c3d"
    newer.mkdir()
    assert latest_run(runs) == newer
    assert latest_run(tmp_path / "nowhere") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert latest_run(empty) is None


def test_a_run_without_stored_results(run_folder: Path) -> None:
    (run_folder / "results.json").unlink()
    loaded = load_run(run_folder)
    assert loaded.sensitivity is None
    assert loaded.warnings == ()


def test_differences_from_the_stored_results_are_reported(run_folder: Path) -> None:
    file = run_folder / "results.json"
    stored = json.loads(file.read_text(encoding="utf-8"))
    stored["model"]["r"] = 99
    stored["protocols"]["leave-one-out"]["methods"]["CS-model"]["accuracy"] = 0.9
    stored["sensitivity"][0]["centres"] = "sideways"  # not a setting: ignored, evaluated anew
    file.write_text(json.dumps(stored), encoding="utf-8")
    loaded = load_run(run_folder)
    assert loaded.warnings == (
        "model.r: stored 99, repeated 6",
        "leave-one-out: accuracy stored 0.9, repeated 0.2",
    )
    assert loaded.sensitivity is None


def test_run_folders_that_cannot_be_read(
    run_folder: Path, experiment_result: ExperimentResult, tmp_path: Path
) -> None:
    with pytest.raises(RunError, match="not a run folder"):
        load_run(tmp_path / "nowhere")
    with pytest.raises(ConfigError, match="cannot read the manifest"):
        load_run(tmp_path)
    (run_folder / "results.json").write_text("{", encoding="utf-8")
    with pytest.raises(RunError, match="cannot read the results"):
        load_run(run_folder)
    (run_folder / "results.json").unlink()
    # another dataset in the folder: the manifest's hash gives it away
    save_snapshot(shape("tiny").dataset, run_folder)
    with pytest.raises(RunError, match="the dataset is not the one the run was computed on"):
        load_run(run_folder)
    # a folder of version 0.4 has no snapshot, and the file its configuration names is gone
    config = ExperimentConfig(dataset=DatasetConfig(path=str(tmp_path / "gone.csv")))
    old = save_run(run_experiment(experiment_result.dataset, config), tmp_path / "old")
    (old / SNAPSHOT_NAME).unlink()
    with pytest.raises(RunError, match=r"has no dataset\.json and its dataset cannot be found"):
        load_run(old)


def test_a_run_folder_without_snapshot_falls_back_to_the_configuration(tmp_path: Path) -> None:
    config = ExperimentConfig(dataset=DatasetConfig(path="builtin:heart-disease-10"))
    result = run_experiment(load_builtin("heart-disease-10"), config)
    folder = save_run(result, tmp_path)
    (folder / SNAPSHOT_NAME).unlink()
    loaded = load_run(folder)
    assert loaded.result.dataset.content_hash() == result.dataset.content_hash()
    assert loaded.sensitivity is None  # not evaluated when the run was saved
    assert "sensitivity" not in json.loads((folder / "results.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- the exporters


def test_the_formats() -> None:
    assert EXPORTERS.names() == FORMATS
    assert resolve_formats(["all"]) == FORMATS
    assert resolve_formats(["ALL"]) == FORMATS
    assert resolve_formats(["xlsx, md", "tex", "md"]) == ["excel", "markdown", "latex"]
    assert resolve_formats(["report", "plots", "tables", "trace", "workbook"]) == [
        "html",
        "figures",
        "csv",
        "json",
        "excel",
    ]
    assert resolve_formats([" , "]) == []
    with pytest.raises(RegistryError, match="docx"):
        resolve_formats(["csv", "docx"])


def test_exporting_every_format(experiment_view: RunView, tmp_path: Path) -> None:
    pytest.importorskip("reportlab")
    folder = tmp_path / "out"
    seen: list[str] = []
    summary = export_view(
        experiment_view, folder, options=ExportOptions(dpi=50), progress=seen.append
    )
    assert seen == FORMATS
    assert list(summary.files) == FORMATS
    assert summary.folder == folder
    assert summary.notes == ()
    tables = len(EXPERIMENT_TABLES)
    assert {name: len(files) for name, files in summary.files.items()} == {
        "excel": 1,
        "csv": tables + 1,
        "json": 1,
        "markdown": tables + 1,
        "latex": tables + 1,
        "figures": 2 * len(EXPERIMENT_FIGURES),
        "html": 1,
        "pdf": 1,
    }
    assert summary.count == 3 * (tables + 1) + 2 * len(EXPERIMENT_FIGURES) + 4
    for files in summary.files.values():
        for path in files:
            assert path.stat().st_size > 0, path
    assert sorted(p.name for p in folder.iterdir()) == [
        "figures",
        "latex",
        "markdown",
        "report.html",
        "report.pdf",
        "tables",
        "trace.json",
        "workbook.xlsx",
    ]
    # the workbook is the mirror: it passes the validation of the experiment
    report = validate_workbook(folder / "workbook.xlsx")
    assert report.passed
    assert report.cells == 11631
    # tables/: one CSV per table and an index
    with (folder / "tables" / "index.csv").open(encoding="utf-8", newline="") as handle:
        index = list(csv.reader(handle))
    assert tuple(index[0]) == INDEX_COLUMNS
    assert [row[1] for row in index[1:]] == EXPERIMENT_TABLES
    assert index[1][0] == "summary.csv"
    psi = (folder / "tables" / "psi.csv").read_text(encoding="utf-8").splitlines()
    assert psi[1] == "S₁,2,2,1,2,2,2,2"
    # trace.json: the same tables with what identifies the run
    trace = json.loads((folder / "trace.json").read_text(encoding="utf-8"))
    assert trace["format"] == "context-synthetic-recognition/trace"
    assert trace["run"] == EXPERIMENT_RUN_ID
    assert trace["dataset"]["hash"] == experiment_view.dataset.content_hash()
    assert trace["config"]["hag"]["centres"] == "running"
    assert [t["key"] for t in trace["tables"]] == EXPERIMENT_TABLES
    # markdown/ and latex/: one file per table and one that holds or inputs them all
    together = (folder / "markdown" / "tables.md").read_text(encoding="utf-8")
    assert together.startswith(f"# {experiment_view.dataset.name}: tables of the run\n")
    assert "## Step 9 · Hierarchical agglomerative grouping\n" in together
    assert "### Ψ(r): synthetic features aᵤ(Sⱼ) ∈ {1, 2} by formula (5)\n" in together
    assert (folder / "markdown" / "psi.md").read_text(encoding="utf-8").startswith("# Ψ(r)")
    document = (folder / "latex" / "all-tables.tex").read_text(encoding="utf-8")
    assert document.count(r"\input{") == tables
    assert (
        (folder / "latex" / "margins.tex").read_text(encoding="utf-8").startswith(r"\begin{table}")
    )
    assert (folder / "report.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    assert (folder / "report.pdf").read_bytes().startswith(b"%PDF-")
    for path in folder.rglob("*"):
        if path.suffix in {".csv", ".json", ".md", ".tex", ".html", ".svg"}:
            assert b"\r\n" not in path.read_bytes(), path  # the same bytes on every platform


def test_the_export_options(experiment_view: RunView, tmp_path: Path) -> None:
    options = ExportOptions(
        decimals=2,
        max_table_cells=100,
        article_max_rows=5,
        article_max_columns=6,
        figure_formats=("svg",),
        theme="dark",
        dpi=40,
    )
    summary = export_view(experiment_view, tmp_path, ["csv", "markdown", "figures"], options)
    assert "table neighbours-rho (540 cells) is larger than max_table_cells and was left out" in (
        summary.notes
    )
    assert not (tmp_path / "tables" / "neighbours-rho.csv").exists()
    assert (tmp_path / "tables" / "summary.csv").exists()
    written = {p.stem for p in (tmp_path / "markdown").glob("*.md")} - {"tables"}
    assert "confusion-resubstitution" in written
    assert "psi" not in written  # 10 rows: left to the CSV export
    assert "| Class 1 | 2 | 2 | 0 | 4 | 0.50 |" in (
        tmp_path / "markdown" / "confusion-resubstitution.md"
    ).read_text(encoding="utf-8")
    assert {p.suffix for p in (tmp_path / "figures").iterdir()} == {".svg"}
    assert "#1a1a19" in (tmp_path / "figures" / "roc.svg").read_text(encoding="utf-8")
    with pytest.raises(ConfigError, match=r"unknown theme 'neon' \(use light, dark\)"):
        export_view(experiment_view, tmp_path, ["figures"], ExportOptions(theme="neon"))
    with pytest.raises(ConfigError, match="unknown theme"):
        export_view(experiment_view, tmp_path, ["html"], ExportOptions(theme="neon"))


def test_what_the_mirror_leaves_out_is_reported(tmp_path: Path) -> None:
    options = ExportOptions(excel=ExcelOptions(max_sheet_cells=4000, max_matrix_cells=8000))
    summary = export_view(shape("heart270"), tmp_path, ["excel"], options)
    assert any("270 × 270 distances are not written" in note for note in summary.notes)
    assert len(summary.notes) == len(set(summary.notes))


def test_the_sensitivity_rule(
    experiment_result: ExperimentResult,
    experiment_view: RunView,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants = experiment_view.sensitivity
    assert variants is not None
    calls: list[int] = []

    def evaluate(dataset: Dataset, config: ExperimentConfig) -> Any:
        calls.append(dataset.m)
        return variants

    monkeypatch.setattr(export_module, "switch_sensitivity", evaluate)
    seen: list[str] = []
    # "auto": a small sample is evaluated
    export_result(experiment_result, tmp_path / "auto", ["json"], progress=seen.append)
    assert calls == [10]
    assert seen == ["sensitivity", "json"]
    assert "sensitivity" in trace_tables(tmp_path / "auto")
    # … a larger one is not
    monkeypatch.setattr(export_module, "AUTO_SENSITIVITY_OBJECTS", 9)
    export_result(experiment_result, tmp_path / "large", ["json"])
    assert calls == [10]
    assert "sensitivity" not in trace_tables(tmp_path / "large")
    # on request, never, and given
    export_result(experiment_result, tmp_path / "asked", ["json"], sensitivity=True)
    assert calls == [10, 10]
    export_result(experiment_result, tmp_path / "never", ["json"], sensitivity=False)
    export_result(experiment_result, tmp_path / "given", ["json"], sensitivity=variants[:2])
    assert calls == [10, 10]
    assert "sensitivity" not in trace_tables(tmp_path / "never")
    assert len(trace_tables(tmp_path / "given")["sensitivity"]["rows"]) == 2


def test_exporting_another_new_object(experiment_result: ExperimentResult, tmp_path: Path) -> None:
    values = experiment_result.dataset.X[3].tolist()
    export_result(
        experiment_result,
        tmp_path,
        ["json"],
        sensitivity=False,
        new_object=values,
        exclude=3,
        run_id="a-run",
    )
    trace = json.loads((tmp_path / "trace.json").read_text(encoding="utf-8"))
    assert trace["run"] == "a-run"
    tables = {t["key"]: t for t in trace["tables"]}
    assert tables["new-object"]["title"].startswith("New object S (S₄ left out of its own context)")


# ---------------------------------------------------------------- csr run --export, csr export


def test_csr_run_with_export(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "heart-disease-10",
            "--runs-dir",
            str(tmp_path),
            "--export",
            "csv,json",
            "-e",
            "md",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "  exporting csv …" in result.output
    assert f"  csv       tables/ ({len(EXPERIMENT_TABLES) + 1} files)" in result.output
    assert "  json      trace.json" in result.output
    assert f"  markdown  markdown/ ({len(EXPERIMENT_TABLES) + 1} files)" in result.output
    assert "switches " not in result.output  # evaluated for the export, printed only on request
    (folder,) = tmp_path.iterdir()
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        [*RUN_FILES, "markdown", "tables", "trace.json"]
    )
    stored = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    assert len(stored["sensitivity"]) == 4
    assert "sensitivity" in trace_tables(folder)
    assert json.loads((folder / "trace.json").read_text(encoding="utf-8"))["run"] == folder.name


def test_csr_run_export_errors(tmp_path: Path) -> None:
    unsaved = runner.invoke(app, ["run", "heart-disease-10", "--no-save", "--export", "csv"])
    assert unsaved.exit_code == EXIT_USAGE
    assert "--export writes into the run folder; remove --no-save" in unsaved.output
    unknown = runner.invoke(
        app, ["run", "heart-disease-10", "--runs-dir", str(tmp_path), "--export", "docx"]
    )
    assert unknown.exit_code == EXIT_USAGE
    assert "Formats: all, excel, csv, json, markdown, latex, figures, html, pdf." in unknown.output
    assert list(tmp_path.iterdir()) == []  # checked before anything is computed


@pytest.fixture
def saved_run(tmp_path: Path) -> Path:
    """A run of Heart-Disease (10, 13, 2) saved without exports and without sensitivity."""
    runs = tmp_path / "runs"
    result = runner.invoke(app, ["run", "heart-disease-10", "--runs-dir", str(runs)])
    assert result.exit_code == 0, result.output
    (folder,) = runs.iterdir()
    return folder


def test_csr_export_of_the_latest_run(saved_run: Path) -> None:
    runs = saved_run.parent
    result = runner.invoke(
        app, ["export", "--runs-dir", str(runs), "-f", "json,md", "--decimals", "2"]
    )
    assert result.exit_code == 0, result.output
    assert f"{saved_run.name}: repeating the run from its manifest" in result.output
    assert "evaluating the four switch settings" in result.output
    assert "⚠ the repeated run differs" not in result.output
    assert f"Heart-Disease (10, 13, 2) · run {saved_run.name}" in result.output
    assert f"  folder: {saved_run}" in result.output
    assert "sensitivity" in trace_tables(saved_run)
    assert "| r₁ | with majorizer | 0.19 |" in (saved_run / "markdown" / "margins.md").read_text(
        encoding="utf-8"
    )


def test_csr_export_by_name_into_another_folder(saved_run: Path, tmp_path: Path) -> None:
    out = tmp_path / "elsewhere"
    result = runner.invoke(
        app,
        [
            "export",
            saved_run.name,
            "--runs-dir",
            str(saved_run.parent),
            "--out",
            str(out),
            "-f",
            "json",
            "--object",
            "4",
            "--no-sensitivity",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "evaluating the four switch settings" not in result.output
    assert sorted(p.name for p in saved_run.iterdir()) == RUN_FILES  # the run folder is untouched
    tables = trace_tables(out)
    assert "sensitivity" not in tables
    assert tables["new-object"]["title"].startswith("New object S (S₄ left out of its own context)")


def test_csr_export_of_a_typed_object_and_every_format(saved_run: Path) -> None:
    pytest.importorskip("reportlab")
    values = "70 1 4 130 322 0 2 109 0 2.4 2 3 3"
    result = runner.invoke(
        app, ["export", str(saved_run), "--values", values, "--dpi", "50", "--theme", "dark"]
    )
    assert result.exit_code == 0, result.output
    for name in ("workbook.xlsx", "trace.json", "report.html", "report.pdf"):
        assert (saved_run / name).is_file(), name
    assert f"  figures   figures/ ({2 * len(EXPERIMENT_FIGURES)} files)" in result.output
    assert trace_tables(saved_run)["new-object"]["title"].startswith(
        "New object S (a new object): "
    )
    # a typed object is not a training object: the Theorem check of the mirror has nothing to
    # compare, and the validation of the experiment still accepts the workbook
    assert validate_workbook(saved_run / "workbook.xlsx").passed


def test_csr_export_reports_a_changed_result(saved_run: Path) -> None:
    file = saved_run / "results.json"
    stored = json.loads(file.read_text(encoding="utf-8"))
    stored["model"]["p"] = 1
    file.write_text(json.dumps(stored), encoding="utf-8")
    result = runner.invoke(app, ["export", str(saved_run), "-f", "json", "--no-sensitivity"])
    assert result.exit_code == 0, result.output
    assert "⚠ the repeated run differs from the stored results — model.p: stored 1, repeated 4" in (
        result.output
    )


def test_csr_export_errors(saved_run: Path, tmp_path: Path) -> None:
    runs = str(saved_run.parent)
    both = runner.invoke(app, ["export", str(saved_run), "--values", "1 2", "--object", "1"])
    assert both.exit_code == EXIT_USAGE
    assert "give at most one of --values and --object" in both.output
    cases = {
        ("export", "--runs-dir", str(tmp_path / "none")): "no run folder in",
        ("export", "20200101_000000_ffffffff", "--runs-dir", runs): "run folder not found",
        ("export", str(saved_run), "-f", "docx"): "Formats: all, excel",
        ("export", str(saved_run), "--object", "11"): "there is no training object № 11 (m = 10)",
        ("export", str(saved_run), "--values", "1 2 3"): "expected 13 values",
        ("export", str(saved_run), "-f", "figures", "--theme", "neon"): "unknown theme 'neon'",
    }
    for arguments, message in cases.items():
        result = runner.invoke(app, list(arguments))
        assert result.exit_code == EXIT_USAGE, arguments
        assert message in result.output, arguments
    (saved_run / SNAPSHOT_NAME).write_text("{}", encoding="utf-8")
    broken = runner.invoke(app, ["export", str(saved_run), "-f", "json"])
    assert broken.exit_code == EXIT_USAGE
    assert "not a dataset snapshot of this package" in broken.output
