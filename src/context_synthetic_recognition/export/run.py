"""Exporting a run: every format into one folder.

Formats (registry :data:`EXPORTERS`; ``all`` selects every one):

``excel``
    ``workbook.xlsx`` — the mirror of the author's Excel experiment (:mod:`.excel`).
``csv``
    ``tables/<key>.csv`` and ``tables/index.csv`` — every table at full precision.
``json``
    ``trace.json`` — the same tables in one file, with the run's identification.
``markdown``, ``latex``
    ``markdown/<key>.md`` (+ ``tables.md``), ``latex/<key>.tex`` (+ ``all-tables.tex``) — the
    tables for the article, rounded; tables too large for a page are left to the CSV export.
``figures``
    ``figures/<key>.png`` and ``.svg``.
``html``, ``pdf``
    ``report.html`` (self-contained) and ``report.pdf`` (needs the optional extra ``[pdf]``).

The CLI (``csr run --export``, ``csr export``) and the GUI call :func:`export_result`.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from context_synthetic_recognition import __version__
from context_synthetic_recognition.config.io import config_hash, to_data
from context_synthetic_recognition.core.registry import Registry
from context_synthetic_recognition.errors import ConfigError
from context_synthetic_recognition.export.excel import ExcelOptions, write_workbook
from context_synthetic_recognition.export.figures import DPI, save_figures
from context_synthetic_recognition.export.report import build_report, html_text, write_pdf
from context_synthetic_recognition.export.tables import (
    DEFAULT_MAX_CELLS,
    Table,
    TableSet,
    run_tables,
)
from context_synthetic_recognition.export.text import (
    DEFAULT_DECIMALS,
    INDEX_COLUMNS,
    csv_text,
    index_rows,
    json_text,
    latex_document,
    latex_table,
    markdown_text,
    table_data,
    write_text,
)
from context_synthetic_recognition.export.theme import THEMES, Theme
from context_synthetic_recognition.export.view import RunView, build_view
from context_synthetic_recognition.services.runner import ExperimentResult
from context_synthetic_recognition.services.sensitivity import SwitchVariant, switch_sensitivity

WORKBOOK_NAME = "workbook.xlsx"
TRACE_NAME = "trace.json"
REPORT_HTML = "report.html"
REPORT_PDF = "report.pdf"
ALL = "all"
AUTO_SENSITIVITY_OBJECTS = 60
"""With ``sensitivity="auto"`` the four switch settings are evaluated up to this many objects
(each needs a leave-one-out of its own); beyond it only on request."""

Progress = Callable[[str], None]
"""Called with the name of the format about to be written."""


@dataclass(frozen=True)
class ExportOptions:
    """Settings of the exporters."""

    decimals: int = DEFAULT_DECIMALS
    """Decimals of real numbers in Markdown, LaTeX and the report."""
    max_table_cells: int = DEFAULT_MAX_CELLS
    """Tables with more cells are left out of the CSV and JSON exports (and reported)."""
    article_max_rows: int = 200
    article_max_columns: int = 20
    """Larger tables are not written as Markdown or LaTeX (they do not fit a page)."""
    figure_formats: tuple[str, ...] = ("png", "svg")
    dpi: int = DPI
    theme: str = "light"
    """``light`` (print, the article) or ``dark``."""
    excel: ExcelOptions = field(default_factory=ExcelOptions)


@dataclass
class ExportContext:
    """What an exporter gets: the run, the target folder and the tables (built once)."""

    view: RunView
    folder: Path
    options: ExportOptions
    notes: list[str] = field(default_factory=list)
    _tables: TableSet | None = None

    @property
    def tables(self) -> TableSet:
        """The tables of the run (built on first use, shared by the exporters)."""
        if self._tables is None:
            self._tables = run_tables(self.view, max_cells=self.options.max_table_cells)
            for spec in self._tables.skipped:
                self.notes.append(
                    f"table {spec.key} ({spec.cells:,} cells) is larger than max_table_cells "
                    "and was left out"
                )
        return self._tables

    def article_tables(self) -> list[Table]:
        """The tables small enough for a page (Markdown, LaTeX)."""
        options = self.options
        return [
            t
            for t in self.tables
            if len(t.rows) <= options.article_max_rows
            and len(t.columns) <= options.article_max_columns
        ]


Exporter = Callable[[ExportContext], list[Path]]
"""Writes one format and returns the files written."""

EXPORTERS: Registry[Exporter] = Registry("exporters")
"""Export formats by name."""


@dataclass(frozen=True)
class ExportSummary:
    """What an export wrote."""

    folder: Path
    files: dict[str, tuple[Path, ...]]
    """Format → the files written."""
    notes: tuple[str, ...] = ()
    """What was left out or shortened because of its size."""

    @property
    def count(self) -> int:
        """Number of files written."""
        return sum(len(paths) for paths in self.files.values())


# ---------------------------------------------------------------- the exporters


@EXPORTERS.register(
    "excel",
    aliases=("xlsx", "workbook"),
    summary="workbook.xlsx — the mirror of the Excel experiment",
)
def export_excel(context: ExportContext) -> list[Path]:
    """The Excel mirror."""
    path = context.folder / WORKBOOK_NAME
    book = write_workbook(context.view, path, context.options.excel)
    context.notes += book.omitted
    return [path]


def _write_index(folder: Path, tables: Sequence[Table], extension: str) -> Path:
    path = folder / "index.csv"
    folder.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(INDEX_COLUMNS)
        writer.writerows(index_rows(tables, extension))
    return path


@EXPORTERS.register(
    "csv", aliases=("tables",), summary="tables/*.csv — every table at full precision"
)
def export_csv(context: ExportContext) -> list[Path]:
    """Every table as a CSV file, with an index."""
    folder = context.folder / "tables"
    tables = list(context.tables)
    written = [write_text(folder / f"{t.key}.csv", csv_text(t)) for t in tables]
    return [*written, _write_index(folder, tables, ".csv")]


@EXPORTERS.register("json", aliases=("trace",), summary="trace.json — every table in one file")
def export_json(context: ExportContext) -> list[Path]:
    """Every table in one JSON file, with what identifies the run."""
    view = context.view
    data = {
        "format": "context-synthetic-recognition/trace",
        "version": 1,
        "package": __version__,
        "run": view.run_id,
        "created": None if view.created is None else view.created.isoformat(),
        "dataset": {
            "name": view.dataset.name,
            "hash": view.dataset.content_hash(),
            "m": view.dataset.m,
            "n": view.dataset.n,
        },
        "config_hash": config_hash(view.config),
        "config": to_data(view.config),
        "tables": [table_data(t) for t in context.tables],
    }
    return [write_text(context.folder / TRACE_NAME, json_text(data))]


@EXPORTERS.register("markdown", aliases=("md",), summary="markdown/*.md — tables for the article")
def export_markdown(context: ExportContext) -> list[Path]:
    """The tables as Markdown: one file each and all of them in ``tables.md``."""
    folder = context.folder / "markdown"
    decimals = context.options.decimals
    tables = context.article_tables()
    written = [
        write_text(folder / f"{t.key}.md", markdown_text(t, decimals, level=1)) for t in tables
    ]
    parts = [f"# {context.view.dataset.name}: tables of the run\n"]
    group = None
    for t in tables:
        if t.group != group:
            group = t.group
            parts.append(f"## {group}\n")
        parts.append(markdown_text(t, decimals, level=3))
    written.append(write_text(folder / "tables.md", "\n".join(parts)))
    return written


@EXPORTERS.register("latex", aliases=("tex",), summary="latex/*.tex — tables for the article")
def export_latex(context: ExportContext) -> list[Path]:
    """The tables as LaTeX (booktabs), and a document that previews them all."""
    folder = context.folder / "latex"
    decimals = context.options.decimals
    tables = context.article_tables()
    written = [write_text(folder / f"{t.key}.tex", latex_table(t, decimals)) for t in tables]
    title = f"{context.view.dataset.name}: tables of the run"
    written.append(
        write_text(folder / "all-tables.tex", latex_document([t.key for t in tables], title))
    )
    return written


def _theme(context: ExportContext) -> Theme:
    try:
        return THEMES[context.options.theme]
    except KeyError:
        known = ", ".join(THEMES)
        raise ConfigError(f"unknown theme {context.options.theme!r} (use {known})") from None


@EXPORTERS.register(
    "figures", aliases=("plots",), summary="figures/*.png, *.svg — the figures of the run"
)
def export_figures(context: ExportContext) -> list[Path]:
    """The figures as PNG and SVG."""
    options = context.options
    return save_figures(
        context.view,
        context.folder / "figures",
        options.figure_formats,
        theme=_theme(context),
        dpi=options.dpi,
    )


@EXPORTERS.register("html", aliases=("report",), summary="report.html — the run report")
def export_html(context: ExportContext) -> list[Path]:
    """The run report as one self-contained HTML file."""
    report = build_report(context.view, context.tables)
    text = html_text(report, theme=_theme(context), decimals=context.options.decimals)
    return [write_text(context.folder / REPORT_HTML, text)]


@EXPORTERS.register("pdf", summary="report.pdf — the run report (optional extra [pdf])")
def export_pdf(context: ExportContext) -> list[Path]:
    """The run report as a PDF."""
    report = build_report(context.view, context.tables)
    path = context.folder / REPORT_PDF
    return [write_pdf(report, path, theme=_theme(context), decimals=context.options.decimals)]


# ---------------------------------------------------------------- running them


def resolve_formats(formats: Sequence[str]) -> list[str]:
    """Canonical format names for what a user typed (``all``, names, aliases, comma lists).

    Raises:
        RegistryError: Unknown format.
    """
    names: list[str] = []
    for item in formats:
        for part in item.split(","):
            key = part.strip()
            if not key:
                continue
            chosen = EXPORTERS.names() if key.lower() == ALL else [EXPORTERS.info(key).name]
            names += [name for name in chosen if name not in names]
    return names


def export_view(
    view: RunView,
    folder: Path,
    formats: Sequence[str] = (ALL,),
    options: ExportOptions | None = None,
    *,
    progress: Progress | None = None,
) -> ExportSummary:
    """Write the chosen formats of a run into ``folder``.

    Raises:
        RegistryError: Unknown format.
        ExportError: A format cannot be written (e.g. the PDF report without reportlab).
    """
    context = ExportContext(view, folder, options or ExportOptions())
    folder.mkdir(parents=True, exist_ok=True)
    files: dict[str, tuple[Path, ...]] = {}
    for name in resolve_formats(formats):
        if progress is not None:
            progress(name)
        files[name] = tuple(EXPORTERS.get(name)(context))
    return ExportSummary(folder, files, tuple(dict.fromkeys(context.notes)))


def export_result(
    result: ExperimentResult,
    folder: Path,
    formats: Sequence[str] = (ALL,),
    options: ExportOptions | None = None,
    *,
    sensitivity: Sequence[SwitchVariant] | bool | Literal["auto"] = "auto",
    new_object: Sequence[float] | None = None,
    exclude: int | None = None,
    run_id: str | None = None,
    progress: Progress | None = None,
) -> ExportSummary:
    """Export an evaluated experiment.

    Args:
        result: The evaluated experiment.
        folder: Where to write (usually the run folder).
        formats: Format names, aliases or ``all``.
        options: Settings of the exporters.
        sensitivity: The four switch settings — given, evaluated now (``True``), left out
            (``False``) or, with ``"auto"``, evaluated for samples of at most
            :data:`AUTO_SENSITIVITY_OBJECTS` objects.
        new_object: The object to demonstrate the meta-algorithm on (default: the first
            training object, left out of its own context).
        exclude: 0-based training object to leave out of ``new_object``'s context.
        run_id: Name of the run folder, shown in the exports.
        progress: Called with the name of every format before it is written.
    """
    variants: tuple[SwitchVariant, ...] | None
    if isinstance(sensitivity, bool | str):
        wanted = sensitivity is True or (
            sensitivity == "auto" and result.dataset.m <= AUTO_SENSITIVITY_OBJECTS
        )
        if wanted and progress is not None:
            progress("sensitivity")
        variants = switch_sensitivity(result.dataset, result.config) if wanted else None
    else:
        variants = tuple(sensitivity)
    view = build_view(
        result, new_object=new_object, exclude=exclude, sensitivity=variants, run_id=run_id
    )
    return export_view(view, folder, formats, options, progress=progress)
