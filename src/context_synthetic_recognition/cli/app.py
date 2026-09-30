"""The ``csr`` command.

Commands: ``config`` (show, init, check), ``data`` (list, info) and ``validate`` (the package
against the Excel experiment, Steps 1–8 so far); ``run``, ``classify`` and ``export`` follow with
the milestones that implement them.
"""

import sys
from pathlib import Path
from typing import Annotated

import typer

from context_synthetic_recognition import __version__
from context_synthetic_recognition.config import (
    ConfigFormat,
    ExperimentConfig,
    PresetName,
    active_deviations,
    config_hash,
    dumps,
    load_config,
    matching_preset,
    preset,
    save_config,
)
from context_synthetic_recognition.core.k_strategies import PermittedK
from context_synthetic_recognition.core.plugins import plugin_problems
from context_synthetic_recognition.data import (
    DATASETS,
    LOADERS,
    Dataset,
    detect_format,
    load_builtin,
    load_dataset,
)
from context_synthetic_recognition.errors import CSRError
from context_synthetic_recognition.log import configure_logging
from context_synthetic_recognition.notation import class_name
from context_synthetic_recognition.services.datasets import summarize
from context_synthetic_recognition.services.validation import DEFAULT_TOLERANCE, validate_workbook

EXIT_FAILED = 1
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
data_app = typer.Typer(
    help="Inspect datasets: sizes, feature types, classes, permitted k.", no_args_is_help=True
)
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")


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


# ---------------------------------------------------------------- csr config


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
    """Validate a configuration file: structure, plug-in names and their parameters."""
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
    problems = plugin_problems(config)
    for problem in problems:
        typer.echo(f"  error: {problem}", err=True)
    if problems:
        raise typer.Exit(EXIT_USAGE)


def _config_or_default(path: Path | None) -> ExperimentConfig:
    return load_config(path) if path is not None else ExperimentConfig()


# ---------------------------------------------------------------- csr data


def _k_text(ks: PermittedK) -> str:
    shown = ks.label if ks.count <= 8 else f"{ks.ks[0]}, {ks.ks[1]}, {ks.ks[2]}, …, {ks.ks[-1]}"
    details = [f"{ks.count} value" + ("s" if ks.count > 1 else "")]
    if ks.k_max is not None:
        details.append(f"k_max = 2·{ks.min_class_size} − 3 = {ks.k_max}")
    if ks.note:
        details.append(ks.note)
    return f"{shown}  ({'; '.join(details)})"


def _load_source(
    source: str, fmt: str, sheet: str | None, class_column: str | None, id_column: str | None
) -> tuple[Dataset, str]:
    path = Path(source)
    if not path.is_file() and source in DATASETS:
        return load_builtin(source), f"built-in dataset '{DATASETS.info(source).name}'"
    chosen = detect_format(path, sheet) if fmt == "auto" and path.is_file() else fmt
    dataset = load_dataset(
        path, chosen, sheet=sheet, class_column=class_column, id_column=id_column
    )
    return dataset, f"{path} ({chosen})"


@data_app.command("list")
def data_list() -> None:
    """List the built-in datasets and the supported file formats."""
    typer.echo("built-in datasets:")
    for info in DATASETS:
        typer.echo(f"  {info.name:20s} {info.obj.description}")
    typer.echo("formats:")
    for loader in LOADERS:
        typer.echo(f"  {loader.name:20s} {loader.summary}")


@data_app.command("info")
def data_info(
    source: Annotated[
        str, typer.Argument(help="Dataset file, or a built-in name (csr data list).")
    ],
    fmt: Annotated[
        str, typer.Option("--format", "-f", help="File format (auto: from extension and content).")
    ] = "auto",
    sheet: Annotated[str | None, typer.Option(help="Excel sheet.")] = None,
    class_column: Annotated[str | None, typer.Option(help="Class column of a table.")] = None,
    id_column: Annotated[str | None, typer.Option(help="Object-id column of a table.")] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Configuration (k strategy, operators); template."),
    ] = None,
) -> None:
    """Show a dataset: objects, feature types I/J, classes, permitted k and synthetic features."""
    try:
        config = _config_or_default(config_path)
        dataset, where = _load_source(source, fmt, sheet, class_column, id_column)
    except CSRError as error:
        raise _fail(error) from error
    summary = summarize(dataset, config)
    flags = list(zip(dataset.feature_names, dataset.quantitative, strict=True))
    quantitative = [name for name, q in flags if q]
    nominal = [name for name, q in flags if not q]
    classes = ", ".join(
        f"{class_name(i)} = {label}: {size}"
        for i, (label, size) in enumerate(zip(dataset.classes, dataset.class_sizes, strict=True))
    )
    typer.echo(dataset.name)
    typer.echo(f"  source:    {where}")
    typer.echo(f"  objects:   m = {dataset.m}")
    typer.echo(f"  features:  n = {dataset.n}")
    typer.echo(f"    I (quantitative) {len(quantitative)}: {', '.join(quantitative) or '—'}")
    typer.echo(f"    J (nominal)      {len(nominal)}: {', '.join(nominal) or '—'}")
    typer.echo(f"  classes:   {classes}")
    typer.echo(f"  hash:      {dataset.content_hash()}")
    if summary.permitted_k is not None:
        typer.echo(f"  k ({config.k.name}): {_k_text(summary.permitted_k)}")
    operators = ", ".join(f"{op.label} ({len(op.features)} features)" for op in summary.operators)
    typer.echo(f"  operators: {operators or '—'}")
    for skipped in summary.skipped_operators:
        typer.echo(f"    skipped {skipped.label}: {skipped.reason} (ADR-007)")
    if summary.r is not None:
        typer.echo(f"  r = |Ψ(r)| = {summary.r} synthetic features")
    for problem in summary.problems:
        typer.echo(f"  ⚠ {problem}")


# ---------------------------------------------------------------- csr validate


@app.command("validate")
def validate(
    against: Annotated[
        Path, typer.Option("--against", "-a", help="The Excel experiment workbook.")
    ],
    tolerance: Annotated[
        float, typer.Option(help="Largest accepted absolute difference of numbers.")
    ] = DEFAULT_TOLERANCE,
    config_path: Annotated[
        Path | None, typer.Option("--config", "-c", help="Configuration; the template preset.")
    ] = None,
    details: Annotated[
        bool, typer.Option("--details", help="List every check, not only failed ones.")
    ] = False,
) -> None:
    """Compare the package with the cached cell values of the Excel experiment (Steps 1–8)."""
    try:
        report = validate_workbook(against, _config_or_default(config_path), tolerance=tolerance)
    except CSRError as error:
        raise _fail(error) from error
    typer.echo(against.name)
    typer.echo(f"  tolerance {tolerance:g}: {report.cells} cells in {len(report.results)} checks")
    for sheet, results in report.sheets().items():
        failed = [result for result in results if not result.passed]
        cells = sum(result.cells for result in results)
        largest = max(result.max_difference for result in results)
        mark = "✓" if not failed else f"✗ {len(failed)} of {len(results)} checks failed"
        typer.echo(f"  {sheet:28s} {cells:5d} cells  max |Δ| {largest:.1e}  {mark}")
        for result in results if details else failed:
            typer.echo(f"      {'✓' if result.passed else '✗'} {result.ref:9s} {result.what}")
            for mismatch in result.mismatches[:5]:
                typer.echo(f"          {mismatch}")
    typer.echo("  (Steps 9–12 and the evaluation sheets follow with milestones M3 and M4.)")
    if not report.passed:
        typer.echo("validation FAILED", err=True)
        raise typer.Exit(EXIT_FAILED)
    typer.echo("validation passed")


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
