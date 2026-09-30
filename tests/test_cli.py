"""The ``csr`` command line (configuration commands)."""

from pathlib import Path

from typer.testing import CliRunner

from context_synthetic_recognition import __version__
from context_synthetic_recognition.cli.app import EXIT_FAILED, EXIT_USAGE, app
from context_synthetic_recognition.config import config_hash, load_config, preset

from .conftest import WORKBOOK

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"context-synthetic-recognition {__version__}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "config" in result.output


def test_show_prints_the_preset_with_deviation_warnings() -> None:
    template = runner.invoke(app, ["config", "show"])
    assert template.exit_code == 0
    assert "centres: running" in template.output
    assert template.output.count("⚠ template calculation") == 2
    article = runner.invoke(app, ["-v", "config", "show", "--preset", "article", "-f", "json"])
    assert article.exit_code == 0
    assert '"centres": "final"' in article.output
    assert "⚠" not in article.output


def test_init_then_check(tmp_path: Path) -> None:
    path = tmp_path / "experiment.toml"
    created = runner.invoke(app, ["config", "init", str(path), "--preset", "article"])
    assert created.exit_code == 0
    assert load_config(path) == preset("article")

    checked = runner.invoke(app, ["config", "check", str(path)])
    assert checked.exit_code == 0
    assert "preset: article" in checked.output
    assert config_hash(preset("article")) in checked.output
    assert "⚠" not in checked.output


def test_init_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    assert runner.invoke(app, ["config", "init", str(path)]).exit_code == 0
    again = runner.invoke(app, ["config", "init", str(path), "-p", "article"])
    assert again.exit_code == EXIT_USAGE
    assert "already exists" in again.output
    forced = runner.invoke(app, ["config", "init", str(path), "-p", "article", "--force"])
    assert forced.exit_code == 0
    assert load_config(path) == preset("article")


def test_check_reports_template_deviations(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    runner.invoke(app, ["config", "init", str(path)])
    result = runner.invoke(app, ["-vv", "config", "check", str(path)])
    assert result.exit_code == 0
    assert "preset: template" in result.output
    assert "running partial class means" in result.output
    assert "majorizer is applied twice" in result.output


def test_check_invalid_file_exits_with_usage_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("hag:\n  alpha: 1.5\n  centre: final\n", encoding="utf-8")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == EXIT_USAGE
    assert "hag.alpha: Input should be less than 1" in result.output
    assert "hag.centre: Extra inputs are not permitted" in result.output


def test_check_reports_unknown_plugins_and_bad_parameters(tmp_path: Path) -> None:
    path = tmp_path / "plugins.yaml"
    path.write_text("k:\n  name: formula\n  params: {step: 3}\n", encoding="utf-8")
    result = runner.invoke(app, ["config", "check", str(path)])
    assert result.exit_code == EXIT_USAGE
    assert "k: k strategies 'formula': invalid parameters" in result.output


def test_an_old_spelling_of_the_metric_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "old.yaml"
    operator = "  - label: ρ\n    metric: {name: zhuravlev}\n"
    path.write_text(f"context:\n  operators:\n{operator}", encoding="utf-8")
    assert runner.invoke(app, ["config", "check", str(path)]).exit_code == 0


# ---------------------------------------------------------------- csr data, csr validate


def test_data_list() -> None:
    result = runner.invoke(app, ["data", "list"])
    assert result.exit_code == 0
    assert "heart-disease-270" in result.output
    assert "template-extended" in result.output


def test_data_info_of_a_builtin_dataset() -> None:
    result = runner.invoke(app, ["data", "info", "heart-disease-270"])
    assert result.exit_code == 0
    assert "classes:   K1 = 1: 150, K2 = 2: 120" in result.output
    assert "k (formula): 3, 5, 7, …, 237  (118 values; k_max = 2·120 − 3 = 237)" in result.output
    assert "r = |Ψ(r)| = 354 synthetic features" in result.output


def test_data_info_of_a_file_with_a_config(tmp_path: Path) -> None:
    data = tmp_path / "d.csv"
    data.write_text("x,colour,class\n1,r,1\n2,g,1\n3,r,2\n4,g,2\n5,r,2\n", encoding="utf-8")
    config = tmp_path / "c.yaml"
    config.write_text("k: {name: explicit, params: {values: [1, 3]}}\n", encoding="utf-8")
    result = runner.invoke(app, ["data", "info", str(data), "-c", str(config)])
    assert result.exit_code == 0
    assert "(csv)" in result.output
    assert "I (quantitative) 1: x" in result.output
    assert "J (nominal)      1: colour" in result.output
    assert "k (explicit): 1, 3  (2 values)" in result.output
    assert "r = |Ψ(r)| = 6" in result.output


def test_data_info_reports_undefined_models_and_errors(tmp_path: Path) -> None:
    data = tmp_path / "d.csv"
    data.write_text("x,class\n1,1\n2,2\n3,2\n", encoding="utf-8")
    result = runner.invoke(app, ["data", "info", str(data)])
    assert result.exit_code == 0
    assert "⚠ no k is permitted" in result.output
    assert "skipped ρ_J" in result.output
    missing = runner.invoke(app, ["data", "info", str(tmp_path / "none.csv")])
    assert missing.exit_code == EXIT_USAGE
    assert "dataset file not found" in missing.output


def test_validate_against_the_workbook() -> None:
    result = runner.invoke(app, ["validate", "--against", str(WORKBOOK)])
    assert result.exit_code == 0, result.output
    assert "4022 cells in 319 checks" in result.output
    assert result.output.count("✓") == 11
    assert "validation passed" in result.output
    detailed = runner.invoke(app, ["validate", "-a", str(WORKBOOK), "--details"])
    assert "B5:K14" in detailed.output


def test_validate_reports_failures_and_bad_files(tmp_path: Path) -> None:
    failing = runner.invoke(app, ["validate", "-a", str(WORKBOOK), "--tolerance", "-1"])
    assert failing.exit_code == EXIT_FAILED
    assert "✗" in failing.output
    assert "validation FAILED" in failing.output
    missing = runner.invoke(app, ["validate", "-a", str(tmp_path / "none.xlsx")])
    assert missing.exit_code == EXIT_USAGE
