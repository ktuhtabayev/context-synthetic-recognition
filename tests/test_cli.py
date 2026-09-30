"""The ``csr`` command line (configuration commands)."""

from pathlib import Path

from typer.testing import CliRunner

from context_synthetic_recognition import __version__
from context_synthetic_recognition.cli.app import EXIT_USAGE, app
from context_synthetic_recognition.config import config_hash, load_config, preset

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
