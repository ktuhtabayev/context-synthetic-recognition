"""The ``csr`` command.

M1 provides configuration commands; ``run``, ``validate``, ``classify`` and ``export`` follow with
the milestones that implement them.
"""

import sys
from pathlib import Path
from typing import Annotated

import typer

from context_synthetic_recognition import __version__
from context_synthetic_recognition.config import (
    ConfigFormat,
    PresetName,
    active_deviations,
    config_hash,
    dumps,
    load_config,
    matching_preset,
    preset,
    save_config,
)
from context_synthetic_recognition.errors import CSRError
from context_synthetic_recognition.log import configure_logging

EXIT_USAGE = 2

app = typer.Typer(
    name="csr",
    help="Context-synthetic model of recognition algorithms (CS-model).",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(
    help="Create, show and check experiment configurations.", no_args_is_help=True
)
app.add_typer(config_app, name="config")


def _fail(error: CSRError) -> typer.Exit:
    typer.echo(f"error: {error}", err=True)
    return typer.Exit(EXIT_USAGE)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"context-synthetic-recognition {__version__}")
        raise typer.Exit()


@app.callback()
def main_options(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", count=True, help="More log output (-v, -vv).")
    ] = 0,
) -> None:
    """Context-synthetic model of recognition algorithms (CS-model)."""
    levels = {0: "WARNING", 1: "INFO"}
    configure_logging(levels.get(verbose, "DEBUG"))


@config_app.command("show")
def config_show(
    name: Annotated[
        PresetName, typer.Option("--preset", "-p", help="Preset to print.")
    ] = PresetName.TEMPLATE,
    fmt: Annotated[
        ConfigFormat, typer.Option("--format", "-f", help="Output format.")
    ] = ConfigFormat.YAML,
) -> None:
    """Print a preset configuration."""
    typer.echo(dumps(preset(name), fmt), nl=False)


@config_app.command("init")
def config_init(
    path: Annotated[Path, typer.Argument(help="File to create (.yaml, .yml, .toml or .json).")],
    name: Annotated[
        PresetName, typer.Option("--preset", "-p", help="Preset to start from.")
    ] = PresetName.TEMPLATE,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Write a new configuration file from a preset."""
    try:
        written = save_config(preset(name), path, overwrite=force)
    except CSRError as error:
        raise _fail(error) from error
    typer.echo(f"wrote {written} (preset: {PresetName(name).value})")


@config_app.command("check")
def config_check(
    path: Annotated[Path, typer.Argument(help="Configuration file to validate.")],
) -> None:
    """Validate a configuration file and report its hash, preset and template deviations."""
    try:
        config = load_config(path)
    except CSRError as error:
        raise _fail(error) from error
    matched = matching_preset(config)
    typer.echo(f"{path}: valid")
    typer.echo(f"  name:   {config.name}")
    typer.echo(f"  hash:   {config_hash(config)}")
    typer.echo(f"  preset: {matched.value if matched else 'custom'}")
    for deviation in active_deviations(config):
        typer.echo(f"  ⚠ template calculation ({deviation.adr}): {deviation.statement}")


def _utf8_console() -> None:
    """Formulas use Unicode (ρ, θ, ⚠); Windows code pages cannot encode them."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    """Entry point of the ``csr`` console script."""
    _utf8_console()
    app()
