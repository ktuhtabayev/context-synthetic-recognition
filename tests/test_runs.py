"""Saved runs side by side (services.runs)."""

import json
from pathlib import Path

import pytest

from context_synthetic_recognition.data import Dataset
from context_synthetic_recognition.errors import ConfigError
from context_synthetic_recognition.services.runner import (
    ExperimentResult,
    RunError,
    run_experiment,
    save_run,
)
from context_synthetic_recognition.services.runs import (
    CURRENT,
    comparison,
    config_differences,
    differing_rows,
    list_runs,
    read_run,
    record_of,
    run_facts,
)

from .conftest import article_switches


@pytest.fixture(scope="module")
def article_result(experiment: Dataset) -> ExperimentResult:
    return run_experiment(experiment, article_switches())


@pytest.fixture
def runs(
    experiment_result: ExperimentResult, article_result: ExperimentResult, tmp_path: Path
) -> Path:
    """A runs folder with the workbook's run and the run with the article's switches."""
    folder = tmp_path / "runs"
    save_run(experiment_result, folder)
    save_run(article_result, folder)
    return folder


def test_reading_a_run(runs: Path, experiment_result: ExperimentResult) -> None:
    records, problems = list_runs(runs)
    assert problems == []
    assert len(records) == 2
    assert [r.run_id for r in records] == sorted((r.run_id for r in records), reverse=True)
    workbook_run = next(
        r for r in records if r.manifest.config.additional_features.majorizing_passes == 2
    )
    assert workbook_run.folder is not None
    assert read_run(workbook_run.folder).run_id == workbook_run.run_id
    assert workbook_run.dataset == experiment_result.dataset.name
    assert workbook_run.protocols == ("resubstitution", "leave-one-out")
    assert workbook_run.accuracy("leave-one-out") == 0.2
    assert workbook_run.accuracy("hold-out") is None
    assert workbook_run.created == workbook_run.manifest.created


def test_folders_that_are_not_runs(runs: Path, tmp_path: Path) -> None:
    assert list_runs(tmp_path / "nowhere") == ([], [])
    (runs / "notes").mkdir()  # not a run id: ignored
    broken = runs / "20200101_000000_ffffffff"
    broken.mkdir()  # a run id without a manifest
    bad = runs / "20200102_000000_ffffffff"
    bad.mkdir()
    first = next(f for f in runs.iterdir() if (f / "manifest.yaml").is_file())
    (bad / "manifest.yaml").write_bytes((first / "manifest.yaml").read_bytes())
    (bad / "results.json").write_text("{", encoding="utf-8")
    records, problems = list_runs(runs)
    assert len(records) == 2
    assert len(problems) == 2
    assert any("cannot read the manifest" in problem for problem in problems)
    assert any("cannot read the results" in problem for problem in problems)
    with pytest.raises(RunError, match="not a run folder"):
        read_run(runs / "missing")
    with pytest.raises(ConfigError):
        read_run(broken)
    (bad / "results.json").unlink()  # a run without results is still a run
    without = read_run(bad)
    assert without.results == {}
    assert without.dataset == "—"
    assert without.protocols == ()


def test_the_facts_of_a_run(experiment_result: ExperimentResult) -> None:
    record = record_of(experiment_result)
    assert record.folder is None
    assert record.run_id == CURRENT
    facts = dict(run_facts(record))
    assert facts["Dataset"] == experiment_result.dataset.name
    assert facts["Objects m"] == 10
    assert facts["Preset"] == "excel-experiment"
    assert facts["⚠ Class means in θ, γ"] == "running"
    assert facts["⚠ majorizing passes"] == 2
    assert facts["Admissible k"] == "3, 5"
    assert facts["TUPLAM"] == "{a₆, a₃, a₁, a₂, a₄}"
    assert facts["Additional features p"] == 4
    assert facts["Accuracy · leave-one-out"] == 0.2
    assert facts["Refusals · leave-one-out"] == 4
    assert facts["AUC · leave-one-out"] == pytest.approx(0.2708333333333333)
    assert str(facts["Best baseline · leave-one-out"]).startswith("k-NN vote ")
    assert facts["Definition 1: defined"] == 1
    assert facts["Definition 4: conflicts on (a₀ … a_p)"] == 10
    assert facts["Seed"] == 42
    assert len(str(facts["Configuration SHA-256"])) == 12
    assert record_of(experiment_result, "mine").run_id == "mine"


def test_runs_side_by_side(runs: Path) -> None:
    records, _ = list_runs(runs)
    table = comparison(records)
    assert table.columns == ("Fact", *(r.run_id for r in records))
    rows = {row[0]: row[1:] for row in table.rows}
    assert set(rows["Preset"]) == {"excel-experiment", "custom"}
    assert set(rows["⚠ majorizing passes"]) == {1, 2}
    assert len(set(rows["Dataset"])) == 1
    different = {table.rows[r][0] for r in differing_rows(table)}
    assert {
        "Preset",
        "⚠ Class means in θ, γ",
        "⚠ majorizing passes",
        "Configuration SHA-256",
    } <= different
    assert "Dataset" not in different
    assert "Objects m" not in different
    settings = config_differences(records)
    names = [row[0] for row in settings.rows]
    assert names == [
        "additional_features.class_means",
        "additional_features.majorizing_passes",
    ]
    assert set(settings.rows[0][1:]) == {"running", "final"}
    same = config_differences([records[0], records[0]])
    assert same.rows == ()
    assert differing_rows(comparison([records[0], records[0]])) == set()


def test_runs_with_different_protocols(experiment_result: ExperimentResult, runs: Path) -> None:
    record = record_of(experiment_result)
    partial = dict(record.results)
    partial["protocols"] = {"hold-out": {"methods": {"CS-model": {"accuracy": 0.5}}}}
    other = type(record)(None, record.manifest, json.loads(json.dumps(partial)))
    table = comparison([record, other])
    rows = {row[0]: row[1:] for row in table.rows}
    assert rows["Accuracy · leave-one-out"] == (0.2, None)  # not run: undefined, not zero
    assert rows["Accuracy · hold-out"] == (None, 0.5)
    assert "Best baseline · hold-out" not in rows
