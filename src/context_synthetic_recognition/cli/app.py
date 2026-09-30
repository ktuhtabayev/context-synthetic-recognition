"""The ``csr`` command.

Commands: ``config`` (show, init, check), ``data`` (list, info), ``fit`` (the CS-model: Ψ(r), the
HAG and the meta-dataset), ``classify`` (an object through the meta-algorithm, step by step),
``run`` (fit and evaluate under every configured protocol, write a run folder) and ``validate``
(the package against the Excel experiment and the template workbooks); ``export`` follows with
milestone M5.
"""

import sys
from collections.abc import Sequence
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
from context_synthetic_recognition.core.arrays import IntArray
from context_synthetic_recognition.core.k_strategies import PermittedK
from context_synthetic_recognition.core.meta import REFUSAL
from context_synthetic_recognition.core.model import Classification, CSModel, fit_model
from context_synthetic_recognition.data import (
    DATASETS,
    LOADERS,
    Dataset,
    detect_format,
    load_builtin,
    load_dataset,
    load_from_config,
)
from context_synthetic_recognition.errors import CSRError, DatasetError
from context_synthetic_recognition.log import configure_logging
from context_synthetic_recognition.notation import class_name, subscript, synthetic_name
from context_synthetic_recognition.services.configs import config_problems
from context_synthetic_recognition.services.datasets import parse_object, summarize
from context_synthetic_recognition.services.runner import run_experiment, save_run
from context_synthetic_recognition.services.sensitivity import switch_sensitivity
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
    problems = config_problems(config)
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
        at_most = " at most (constant ones are skipped)" if config.synthetic.skip_constant else ""
        typer.echo(f"  r = |Ψ(r)| = {summary.r} synthetic features{at_most}")
    for problem in summary.problems:
        typer.echo(f"  ⚠ {problem}")


# ---------------------------------------------------------------- csr fit / csr classify


SourceArgument = Annotated[
    str, typer.Argument(help="Dataset file, or a built-in name (csr data list).")
]
FormatOption = Annotated[
    str, typer.Option("--format", "-f", help="File format (auto: from extension and content).")
]
SheetOption = Annotated[str | None, typer.Option(help="Excel sheet.")]
ClassOption = Annotated[str | None, typer.Option(help="Class column of a table.")]
IdOption = Annotated[str | None, typer.Option(help="Object-id column of a table.")]
ConfigOption = Annotated[
    Path | None, typer.Option("--config", "-c", help="Configuration; the template preset.")
]


def _fitted(
    source: str,
    fmt: str,
    sheet: str | None,
    class_column: str | None,
    id_column: str | None,
    config_path: Path | None,
) -> tuple[Dataset, ExperimentConfig, CSModel]:
    config = _config_or_default(config_path)
    dataset, _ = _load_source(source, fmt, sheet, class_column, id_column)
    return dataset, config, fit_model(dataset, config)


def _names(dataset: Dataset, indices: Sequence[int] | IntArray) -> str:
    chosen = [dataset.object_ids[int(i)] for i in indices]
    return "{" + ", ".join(chosen) + "}" if chosen else "∅"


@app.command("fit")
def fit(
    source: SourceArgument,
    fmt: FormatOption = "auto",
    sheet: SheetOption = None,
    class_column: ClassOption = None,
    id_column: IdOption = None,
    config_path: ConfigOption = None,
) -> None:
    """Fit the CS-model: permitted k, Ψ(r), the HAG (TUPLAM, crit, latent features)."""
    try:
        dataset, config, model = _fitted(source, fmt, sheet, class_column, id_column, config_path)
    except CSRError as error:
        raise _fail(error) from error
    trace, grouping = model.trace, model.hag
    settings = grouping.settings
    typer.echo(dataset.name)
    typer.echo(
        f"  objects m = {dataset.m}; |K1| = {trace.class_sizes[0]}, |K2| = {trace.class_sizes[1]}"
    )
    typer.echo(f"  k ({config.k.name}): {_k_text(trace.permitted_k)}")
    typer.echo(f"  r = |Ψ(r)| = {trace.r} synthetic features")
    typer.echo(
        f"  HAG: α = {settings.alpha:g}, δ = {settings.delta:g}, ϰ = {settings.kappa}, "
        f"cr1₀ = {settings.cr1:g}, ϕ = {settings.majorizer}, centres = {settings.centres.value}, "
        f"STEP 4 passes = {settings.step4_passes}"
    )
    for deviation in active_deviations(config):
        typer.echo(f"    ⚠ template calculation ({deviation.adr}): {deviation.statement}")
    first = grouping.first
    typer.echo(f"    STEP 2: u = {synthetic_name(first)} (weight {grouping.weights[first]:.6g})")
    for it in grouping.iterations:
        if it.q is None:
            typer.echo(f"    iteration {it.number}: no θ/γ < cr1₀ among {it.candidates.size}")
        else:
            typer.echo(
                f"    iteration {it.number}: q = {synthetic_name(it.q)}, "
                f"crit = θ/γ = {it.crit:.12g} (of {it.candidates.size} candidates)"
            )
    typer.echo(f"    stop: {grouping.stop.text}")
    typer.echo(f"  TUPLAM = {grouping.label}; p = {grouping.p} latent features")
    training = model.classify_training()
    truth = dataset.y.tolist()
    correct = sum(label == true for label, true in zip(training.labels, truth, strict=True))
    typer.echo(
        f"  training objects (Definition 2): {correct} of {dataset.m} classified correctly, "
        f"{training.refusals} refusals"
    )


def _explain(dataset: Dataset, model: CSModel, result: Classification) -> None:
    steps = result.meta.steps(0)
    description = result.meta.queries[0]
    typer.echo(f"  TUPLAM = {model.hag.label} → positions a₀ … a{subscript(model.p)}")
    typer.echo(f"  description (a₀, …, a_p) = ({', '.join(str(int(v)) for v in description)})")
    for j in range(model.p + 1):
        step = "Step 1" if j == 0 else "Step 2"
        typer.echo(
            f"  {step}  j = {j}: B1(a{subscript(j)}) = {_names(dataset, steps.b1(j))}   "
            f"B2(a{subscript(j)}) = {_names(dataset, steps.b2(j))}"
        )
    meta = result.meta
    k1, k2 = model.description.class_sizes
    b1, b2 = int(meta.b1_sizes[0, -1]), int(meta.b2_sizes[0, -1])
    decision = int(meta.decisions[0])
    if decision == REFUSAL:
        verdict = "0 — refusal (equal scores)"
    else:
        verdict = f"K{decision} (class {result.labels[0]})"
    typer.echo(
        f"  Step 4: |B1|/|K1| = {b1}/{k1} = {b1 / k1:.6g}, |B2|/|K2| = {b2}/{k2} = {b2 / k2:.6g}"
        f"  →  {verdict}"
    )


@app.command("classify")
def classify(
    source: SourceArgument,
    values: Annotated[
        str | None,
        typer.Option("--values", help="A new object: n values separated by commas or spaces."),
    ] = None,
    obj: Annotated[
        int | None,
        typer.Option(
            "--object",
            help="Training object № (1-based), left out of its own context (leave-self-out).",
        ),
    ] = None,
    fmt: FormatOption = "auto",
    sheet: SheetOption = None,
    class_column: ClassOption = None,
    id_column: IdOption = None,
    config_path: ConfigOption = None,
) -> None:
    """Classify one object without its class: Ψ(r), TUPLAM description, B1/B2 per step, class."""
    if (values is None) == (obj is None):
        typer.echo("error: give exactly one of --values and --object", err=True)
        raise typer.Exit(EXIT_USAGE)
    try:
        dataset, _, model = _fitted(source, fmt, sheet, class_column, id_column, config_path)
        if obj is not None:
            if not 1 <= obj <= dataset.m:
                raise DatasetError(f"there is no training object № {obj} (m = {dataset.m})")
            x, exclude = dataset.X[obj - 1], [obj - 1]
            what = f"{dataset.object_ids[obj - 1]} (left out of its own context)"
        else:
            x, exclude = parse_object(dataset, values or ""), None
            what = "new object"
        result = model.classify(x, exclude=exclude)
    except CSRError as error:
        raise _fail(error) from error
    typer.echo(f"{dataset.name}: {what}")
    if result.representation is not None:
        values_row = result.representation.context.values[0]
        features = ", ".join(
            f"{f.name} = {int(v)}" for f, v in zip(model.trace.features, values_row, strict=True)
        )
        typer.echo(f"  Ψ(r) by formula (5): {features}")
    _explain(dataset, model, result)


# ---------------------------------------------------------------- csr run


def _percent(value: float | None) -> str:
    return "—" if value is None or value != value else f"{100 * value:5.1f} %"


def _dataset_for_run(
    source: str | None, config_path: Path | None, config: ExperimentConfig
) -> Dataset:
    if source is not None:
        return _load_source(source, "auto", None, None, None)[0]
    base = config_path.parent if config_path is not None else None
    return load_from_config(config.dataset, base)


@app.command("run")
def run(
    source: Annotated[
        str | None,
        typer.Argument(help="Dataset file or built-in name; default: dataset.path of the config."),
    ] = None,
    config_path: ConfigOption = None,
    save: Annotated[
        bool, typer.Option("--save/--no-save", help="Write the run folder (manifest, results).")
    ] = True,
    runs_dir: Annotated[
        Path | None, typer.Option(help="Folder for run folders; default: output.runs_dir.")
    ] = None,
    sensitivity: Annotated[
        bool, typer.Option("--sensitivity", help="Also evaluate all four switch settings.")
    ] = False,
) -> None:
    """Fit and evaluate the CS-model: every configured protocol, baselines, margins, properties."""
    try:
        config = _config_or_default(config_path)
        dataset = _dataset_for_run(source, config_path, config)
        typer.echo(f"{dataset.name}: m = {dataset.m}, n = {dataset.n}", err=True)
        result = run_experiment(
            dataset,
            config,
            progress=lambda stage, done, total: (
                typer.echo(f"  {stage}: {done}/{total}", err=True) if done == total else None
            ),
        )
        variants = switch_sensitivity(dataset, config) if sensitivity else ()
    except CSRError as error:
        raise _fail(error) from error
    model = result.model
    typer.echo(dataset.name)
    typer.echo(f"  k ({config.k.name}): {_k_text(model.trace.permitted_k)}; r = {model.trace.r}")
    typer.echo(f"  TUPLAM = {model.hag.label}; p = {model.p}; stop: {model.hag.stop.text}")
    for deviation in active_deviations(config):
        typer.echo(f"  ⚠ template calculation ({deviation.adr}): {deviation.statement}")
    positive = model.classes[result.positive - 1]
    typer.echo(f"  positive class: {positive}")
    for protocol in result.protocols:
        undefined = len(protocol.undefined_folds)
        note = f"; {undefined} undefined fold(s) refused" if undefined else ""
        count = len(protocol.folds)
        typer.echo(f"  {protocol.protocol} ({count} fold{'s' if count != 1 else ''}{note}):")
        typer.echo(
            f"    {'method':34s} {'accuracy':>9s} {'coverage':>9s} {'refusals':>8s}"
            f" {'macro F1':>9s} {'AUC':>6s}"
        )
        for method in protocol.methods:
            metrics = result.metrics(method)
            auc = result.auc(method)
            typer.echo(
                f"    {method.method:34s} {_percent(metrics.accuracy):>9s}"
                f" {_percent(metrics.coverage):>9s} {metrics.refusals:8d}"
                f" {_percent(metrics.macro_f1):>9s} {'—' if auc != auc else f'{auc:.3f}':>6s}"
            )
    if result.margins.with_majorizer:
        widths = ", ".join(
            f"r{j + 1} {a.width:.4f} ({b.width:.4f})"
            for j, (a, b) in enumerate(
                zip(result.margins.with_majorizer, result.margins.without_majorizer, strict=True)
            )
        )
        typer.echo(f"  margin widths with majorizer (without): {widths}")
    properties = result.properties
    typer.echo(
        f"  Definition 1: {'defined' if properties.determinacy.defined else 'NOT defined'};"
        f" Definition 4: {properties.tuplam_conflicts} conflicting pair(s) on (a₀ … a_p)"
    )
    for variant in variants:
        typer.echo(
            f"  switches {variant.label:26s} {variant.tuplam}: resubstitution"
            f" {_percent(variant.resubstitution)}, LOO {_percent(variant.leave_one_out)},"
            f" AUC {variant.auc_resubstitution:.3f} / {variant.auc_leave_one_out:.3f}"
        )
    if save:
        folder = save_run(result, runs_dir)
        typer.echo(f"  run folder: {folder}")


# ---------------------------------------------------------------- csr validate


@app.command("validate")
def validate(
    against: Annotated[
        Path,
        typer.Option(
            "--against", "-a", help="The Excel experiment workbook or a template workbook."
        ),
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
    """Compare the package with the cached cells of the experiment or a template workbook."""
    try:
        report = validate_workbook(against, _config_or_default(config_path), tolerance=tolerance)
    except CSRError as error:
        raise _fail(error) from error
    typer.echo(f"{against.name}  [{report.kind}]")
    for note in report.notes:
        typer.echo(f"  {note}")
    typer.echo(f"  tolerance {tolerance:g}: {report.cells} cells in {len(report.results)} checks")
    for sheet, results in report.sheets().items():
        failed = [result for result in results if not result.passed]
        cells = sum(result.cells for result in results)
        largest = max(result.max_difference for result in results)
        mark = "✓" if not failed else f"✗ {len(failed)} of {len(results)} checks failed"
        typer.echo(f"  {sheet:31s} {cells:5d} cells  max |Δ| {largest:.1e}  {mark}")
        for result in results if details else failed:
            typer.echo(f"      {'✓' if result.passed else '✗'} {result.ref:9s} {result.what}")
            for mismatch in result.mismatches[:5]:
                typer.echo(f"          {mismatch}")
    if report.kind == "experiment":
        typer.echo(
            "  (not compared: Overview, the input sheets Dataset, Quantitative, Nominal, and the "
            "template-data tables — validate the template workbooks for those)"
        )
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
