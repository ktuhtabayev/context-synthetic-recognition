"""The experiment runner, run folders, switch sensitivity, skip_constant, built-in dataset paths."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from context_synthetic_recognition.cli.app import EXIT_USAGE, app
from context_synthetic_recognition.config import (
    DatasetConfig,
    ExperimentConfig,
    SyntheticConfig,
    plugin,
    preset,
    save_config,
)
from context_synthetic_recognition.core.context import fit_context
from context_synthetic_recognition.data import Dataset, FeatureType, load_from_config
from context_synthetic_recognition.errors import ConfigError, ModelUndefinedError
from context_synthetic_recognition.evaluation.protocols import run_protocol
from context_synthetic_recognition.services.manifest import read_manifest
from context_synthetic_recognition.services.runner import (
    positive_code,
    results_dict,
    run_experiment,
    save_run,
)
from context_synthetic_recognition.services.sensitivity import switch_sensitivity

from .conftest import CONFIGS, GOLDEN_VALUES

runner = CliRunner()


def test_run_experiment_on_the_experiment(experiment: Dataset) -> None:
    stages: list[tuple[str, int, int]] = []
    result = run_experiment(experiment, progress=lambda *a: stages.append(a))
    assert [p.protocol for p in result.protocols] == ["resubstitution", "leave-one-out"]
    assert result.positive == 1
    loo = result.protocol("leave-one-out")
    assert result.metrics(loo.predictions).accuracy == 0.2
    assert result.auc(loo.predictions) == pytest.approx(0.2708333333333333)
    assert len(loo.baselines) == 6
    assert result.margins.with_majorizer[0].width == pytest.approx(0.148177267098285)
    assert result.properties.tuplam_conflicts == 10
    assert stages[-1] == ("leave-one-out", 10, 10)
    assert set(result.timings) >= {"fit", "resubstitution", "leave-one-out"}
    with pytest.raises(KeyError):
        result.protocol("hold-out")


def test_positive_class() -> None:
    assert positive_code((1, 2), 1) == 1
    assert positive_code(("absent", "present"), "present") == 2
    assert positive_code((1, 2), "2") == 2
    with pytest.raises(ConfigError, match="not a class"):
        positive_code(("absent", "present"), 1)


def test_save_run(experiment: Dataset, tmp_path: Path) -> None:
    result = run_experiment(experiment)
    folder = save_run(result, tmp_path / "runs")
    manifest = read_manifest(folder)
    assert manifest.dataset_hash == experiment.content_hash()
    assert "leave-one-out" in manifest.timings
    results = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    assert results["model"]["tuplam"] == ["a₆", "a₃", "a₁", "a₂", "a₄"]
    assert results["protocols"]["leave-one-out"]["methods"]["CS-model"]["accuracy"] == 0.2
    assert results["template_deviations"] == ["ADR-002", "ADR-003"]
    assert results == results_dict(result)
    with (folder / "predictions.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 7 * experiment.m  # two protocols × (CS-model + 6 baselines)
    refused = [r for r in rows if r["protocol"] == "leave-one-out" and r["method"] == "CS-model"]
    assert [r["predicted"] for r in refused][:3] == ["1", "", ""]
    with (folder / "folds.csv").open(encoding="utf-8") as handle:
        folds = list(csv.DictReader(handle))
    assert len(folds) == 1 + experiment.m
    assert folds[2]["k"] == "3"


def test_switch_sensitivity_reproduces_the_golden_variants(experiment: Dataset) -> None:
    with GOLDEN_VALUES.open(encoding="utf-8") as handle:
        golden = json.load(handle)["variants"]
    variants = switch_sensitivity(experiment)
    for variant, expected in zip(variants, golden, strict=True):
        assert variant.label == expected["label"]
        assert variant.tuplam == expected["T"]
        assert list(variant.crit) == pytest.approx(expected["crit"], abs=1e-15)
        assert variant.resubstitution == expected["resub"]
        assert variant.leave_one_out == expected["loo"]
        assert variant.auc_resubstitution == pytest.approx(expected["auc_r"])
        assert variant.auc_leave_one_out == pytest.approx(expected["auc_l"])


# ---------------------------------------------------------------- skip_constant (ADR-030)


def test_skipping_constant_features(experiment: Dataset) -> None:
    skip = ExperimentConfig(synthetic=SyntheticConfig(skip_constant=True))
    fold3 = experiment.subset([i for i in range(experiment.m) if i != 2])
    kept = fit_context(fold3, skip).trace
    assert [(s.operator_label, s.k, s.value) for s in kept.skipped_features] == [
        ("ρ", 3, 2),
        ("ρ_J", 3, 2),
        ("ρ_J", 5, 2),
    ]
    assert [f.name for f in kept.features] == ["a₁", "a₂", "a₃"]  # renumbered
    assert [(f.operator_label, f.k) for f in kept.features] == [("ρ", 5), ("ρ_I", 3), ("ρ_I", 5)]
    assert fit_context(fold3).trace.skipped_features == ()
    # the full fit has no constant feature; leave-one-out changes in folds 5 and 6
    loo = run_protocol(experiment, skip, "leave-one-out")
    assert loo.predictions.by_object().tolist() == [1, 0, 0, 0, 1, 1, 2, 1, 1, 1]
    assert [f.number for f in loo.undefined_folds] == [3]  # every feature constant in fold 4


def test_every_feature_constant_is_undefined() -> None:
    # min|Kᵢ| = 2 → k = 1; the nearest neighbour of every object is in K1, so a ≡ 1
    X = np.array([[0.0], [0.1], [0.2], [-5.0], [5.0]])
    y = np.array([1, 1, 1, 2, 2])
    dataset = Dataset(X=X, y=y, feature_types=(FeatureType.QUANTITATIVE,))
    config = ExperimentConfig(synthetic=SyntheticConfig(skip_constant=True))
    with pytest.raises(ModelUndefinedError, match="every synthetic feature is constant"):
        fit_context(dataset, config)


# ---------------------------------------------------------------- built-in datasets in configs


def test_builtin_dataset_path() -> None:
    dataset = load_from_config(DatasetConfig(path="builtin:heart-disease-10"))
    assert dataset.m == 10
    retyped = load_from_config(
        DatasetConfig(path="builtin: heart-disease-10", feature_types={"x₂": "quantitative"})
    )
    assert retyped.quantitative[1]


def test_the_large_data_example_config() -> None:
    from context_synthetic_recognition.config import load_config

    config = load_config(CONFIGS / "heart-disease-270-large-data.yaml")
    assert config.synthetic.skip_constant
    assert config.k.params == {"k_max_cap": 21}
    assert config.dataset.path == "heart-disease-270"  # the datasets folder, by id
    assert preset("template").synthetic.skip_constant is False


# ---------------------------------------------------------------- csr run


def test_csr_run(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "heart-disease-10", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "leave-one-out (10 folds):" in result.output
    assert "resubstitution (1 fold):" in result.output
    assert "CS-model                              20.0 %" in result.output
    assert "Definition 4: 10 conflicting pair(s)" in result.output
    assert "run folder:" in result.output
    assert len(list(tmp_path.iterdir())) == 1


def test_csr_run_with_a_config_and_sensitivity(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    save_config(
        ExperimentConfig(
            dataset=DatasetConfig(path="builtin:heart-disease-10"),
            evaluation=preset("template").evaluation.model_copy(
                update={"protocols": (plugin("leave-one-out"),), "baselines": ()}
            ),
        ),
        path,
    )
    result = runner.invoke(app, ["run", "-c", str(path), "--no-save", "--sensitivity"])
    assert result.exit_code == 0, result.output
    assert "switches article" in result.output
    assert "run folder" not in result.output


def test_csr_run_errors() -> None:
    missing = runner.invoke(app, ["run", "cancer-589", "--no-save"])
    assert missing.exit_code == EXIT_USAGE
    assert "unknown dataset 'cancer-589'" in missing.output


def test_config_problems_cover_the_evaluation_plugins() -> None:
    from context_synthetic_recognition.services.configs import config_problems

    assert config_problems(ExperimentConfig()) == []
    config = ExperimentConfig(
        evaluation=preset("template").evaluation.model_copy(
            update={
                "protocols": (plugin("bootstrap"), plugin("stratified-k-fold", folds=1)),
                "baselines": (plugin("knn-vote", ks=[2]),),
            }
        )
    )
    problems = config_problems(config)
    assert len(problems) == 3
    assert problems[0].startswith("evaluation.protocols[0]: Unknown protocols 'bootstrap'")
    assert problems[2].startswith("evaluation.baselines[0]")
