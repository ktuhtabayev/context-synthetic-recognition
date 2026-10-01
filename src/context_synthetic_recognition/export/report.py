"""The run report: one document that tells a run from the data to the evaluation.

:func:`build_report` arranges the tables and figures of a run into sections — summary, data,
local contexts, synthetic features, HAG, meta-algorithm, evaluation, model properties,
reproducibility — with the two template deviations highlighted at the top. The document is plain
data (:class:`Report`); :func:`html_text` renders it as one self-contained HTML file (figures
embedded as SVG) and :func:`write_pdf` as a PDF (reportlab, optional extra ``[pdf]``; figures
embedded as PNG).

Long tables are shortened in the report — it is for reading; the CSV tables hold every row.
"""

from __future__ import annotations

import base64
import html
import importlib
import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from context_synthetic_recognition import __version__
from context_synthetic_recognition.config.io import config_hash, to_data
from context_synthetic_recognition.config.presets import (
    DEVIATIONS,
    active_deviations,
    matching_preset,
)
from context_synthetic_recognition.errors import CSRError
from context_synthetic_recognition.export.figures import (
    FigureSpec,
    figure_specs,
    render,
    save_figure,
)
from context_synthetic_recognition.export.tables import Cell, Table, TableSet, run_tables
from context_synthetic_recognition.export.text import DEFAULT_DECIMALS, format_cell, is_numeric
from context_synthetic_recognition.export.theme import LIGHT, Theme
from context_synthetic_recognition.export.view import RunView
from context_synthetic_recognition.export.wording import join_and, protocol_title

MAX_ROWS = 24
"""Rows of a table shown in the report before it is shortened."""
MAX_COLUMNS = 16
"""Columns of a table shown in the report before the middle ones are left out."""


class ExportError(CSRError, RuntimeError):
    """An export cannot be written (e.g. an optional dependency is missing)."""


# ---------------------------------------------------------------- the document


@dataclass(frozen=True)
class Heading:
    """A section heading."""

    text: str
    level: int = 2


@dataclass(frozen=True)
class Paragraph:
    """Running text; ``warning`` marks the ⚠ boxes, ``note`` small print."""

    text: str
    kind: Literal["text", "note", "warning"] = "text"


@dataclass(frozen=True)
class TableBlock:
    """A table, possibly shortened for reading."""

    table: Table
    omitted_rows: int = 0
    omitted_columns: int = 0


@dataclass(frozen=True)
class FigureBlock:
    """A figure with its caption."""

    spec: FigureSpec


@dataclass(frozen=True)
class Code:
    """Preformatted text (the configuration)."""

    text: str


Block = Heading | Paragraph | TableBlock | FigureBlock | Code


@dataclass(frozen=True)
class Report:
    """A run report as data."""

    title: str
    subtitle: str
    blocks: tuple[Block, ...] = field(default_factory=tuple)


def shorten(table: Table, max_rows: int = MAX_ROWS, max_columns: int = MAX_COLUMNS) -> TableBlock:
    """A table cut to a readable size: the first rows, and the first and last columns."""
    columns = list(range(len(table.columns)))
    omitted_columns = 0
    if len(columns) > max_columns:
        tail = max(1, max_columns // 4)
        head = max_columns - tail - 1
        omitted_columns = len(columns) - head - tail
        keep = [*columns[:head], None, *columns[-tail:]]
    else:
        keep = list(columns)
    rows = table.rows[:max_rows]
    omitted_rows = len(table.rows) - len(rows)
    if not omitted_rows and not omitted_columns:
        return TableBlock(table)

    def pick(row: Sequence[Cell]) -> tuple[Cell, ...]:
        return tuple("…" if c is None else row[c] for c in keep)

    cut = Table(
        table.key,
        table.title,
        tuple("…" if c is None else table.columns[c] for c in keep),
        tuple(pick(row) for row in rows),
        table.group,
        table.note,
        table.labels,
    )
    return TableBlock(cut, omitted_rows, omitted_columns)


def build_report(view: RunView, tables: TableSet | None = None) -> Report:
    """Arrange the tables and figures of a run into the report."""
    found = tables if tables is not None else run_tables(view)
    figures = {spec.key: spec for spec in figure_specs(view)}
    trace, hag, config, data = view.trace, view.hag, view.config, view.dataset
    blocks: list[Block] = []

    def table(key: str, **limits: int) -> None:
        item = found.get(key)
        if item is not None and item.rows:
            blocks.append(shorten(item, **limits))

    def figure(key: str) -> None:
        if key in figures:
            blocks.append(FigureBlock(figures[key]))

    # ---- summary
    blocks.append(Heading("Summary"))
    active = active_deviations(config)
    statements = "; ".join(d.statement for d in DEVIATIONS)
    if active:
        blocks.append(
            Paragraph(
                "⚠ Two template calculations differ from the article: "
                f"{statements}. Both are configuration switches. This run uses the template "
                f"calculation for {join_and([f'{d.field} ({d.adr})' for d in active])}"
                + (
                    " — the default, which reproduces the template workbooks exactly."
                    if len(active) == len(DEVIATIONS)
                    else "."
                ),
                "warning",
            )
        )
    else:
        blocks.append(
            Paragraph(
                "⚠ Two template calculations differ from the article: "
                f"{statements}. This run follows the article for both (hag.centres = final, "
                "hag.step4_passes = 1), so it does not reproduce the template workbooks.",
                "warning",
            )
        )
    table("summary", max_rows=60)

    # ---- data
    quantitative = int(data.quantitative.sum())
    blocks.append(Heading("Data"))
    blocks.append(
        Paragraph(
            f"{data.name}: {data.m} objects, {data.n} features ({quantitative} quantitative, "
            f"{data.n - quantitative} nominal), classes K1 = {trace.classes[0]} "
            f"({trace.class_sizes[0]} objects) and K2 = {trace.classes[1]} "
            f"({trace.class_sizes[1]} objects)."
        )
    )
    table("features")
    table("dataset", max_rows=12)

    # ---- local contexts
    blocks.append(Heading("Local contexts Ψ_ρ,k (Steps 1–3)"))
    operators = "; ".join(
        f"{o.label} — metric {o.metric} on {o.features.size} feature(s)" for o in trace.operators
    )
    blocks.append(
        Paragraph(
            f"Scale unification: {trace.scaling.normalizer}, fitted on the training objects. "
            f"Base operators: {operators}. Distances are rounded to "
            f"{config.context.distance_decimals} decimals; neighbours are ordered by (distance, "
            "original index), the object itself excluded."
        )
    )
    figure("distances")
    figure("neighbourhoods")

    # ---- synthetic features
    blocks.append(Heading("Synthetic features Ψ(r) (Steps 4–8)"))
    ks = trace.permitted_k
    blocks.append(
        Paragraph(
            f"Permitted k ({ks.rule}): {ks.count} value(s), k = {ks.ks[0]} … {ks.ks[-1]}"
            + (f" ({ks.note})" if ks.note else "")
            + f"; r = |Ψ(r)| = {trace.r} synthetic features aᵤ ∈ {{1, 2}} by formula (5), each "
            "with its stability g (2), boundary G (3), informativeness ω (4) and contributions "
            "η (6)."
            + (
                f" {len(trace.skipped_features)} constant feature(s) were skipped (ADR-030)."
                if trace.skipped_features
                else ""
            )
        )
    )
    table("synthetic-features")
    figure("synthetic-features")
    table("psi", max_rows=12)

    # ---- HAG
    settings = hag.settings
    blocks.append(Heading("Hierarchical agglomerative grouping (Step 9)"))
    blocks.append(
        Paragraph(
            f"α = {settings.alpha:g}, δ = {settings.delta:g}, ϰ = {settings.kappa}, "
            f"cr1₀ = {settings.cr1:g}, majorizer ϕ = {settings.majorizer}; ⚠ class centres = "
            f"{settings.centres.value}, ⚠ STEP 4 passes = {settings.step4_passes}. "
            f"TUPLAM = {hag.label}, p = {hag.p} latent feature(s); the grouping stopped because "
            f"{hag.stop.text}."
        )
    )
    table("hag-iterations")
    figure("hag-candidates")
    figure("hag-criterion")

    # ---- meta-algorithm
    blocks.append(Heading("Meta-algorithm (Steps 10–12)"))
    blocks.append(
        Paragraph(
            "The meta-dataset Y = (y₀, …, y_p, r₁, …, r_p) holds the contribution values of the "
            "TUPLAM features and the latent features. An object is classified from its "
            "gradations (a₀, …, a_p) alone: B1 and B2 are filtered step by step and the class "
            "follows from |B1|/|K1| against |B2|/|K2| (0 = refusal if equal)."
        )
    )
    table("meta-dataset", max_rows=12)
    table("new-object-steps")
    table("resubstitution", max_rows=12)

    # ---- evaluation
    blocks.append(Heading("Evaluation"))
    names = join_and([protocol_title(p.protocol) for p in view.result.protocols])
    blocks.append(
        Paragraph(
            f"Protocols: {names}. Hold-out protocols re-fit the whole pipeline — scaling, k "
            "range, Ψ(r), ω, η, HAG, meta-algorithm — on the training part of every fold; the "
            "baselines run on the same folds."
        )
    )
    table("metrics", max_rows=40)
    for protocol in view.result.protocols:
        figure(f"outcomes-{protocol.protocol}")
        figure(f"prediction-map-{protocol.protocol}")
    figure("roc")
    figure("confusion")
    if hag.p:
        blocks.append(Heading("Margins of the latent features", 3))
        table("margins")
        figure("margins")
        figure("margin-widths")
    if view.sensitivity:
        blocks.append(Heading("Sensitivity to the two switches", 3))
        table("sensitivity")
        figure("sensitivity")

    # ---- model properties
    blocks.append(Heading("Model properties (Definitions 1–6, Theorem)"))
    table("properties", max_rows=40)
    table("operator-pairs")
    table("boundary-ties")

    # ---- reproducibility
    blocks.append(Heading("Reproducibility"))
    preset = matching_preset(config)
    blocks.append(
        Paragraph(
            f"Run {view.run_id or '(not saved)'} · context-synthetic-recognition {__version__} · "
            f"preset {preset.value if preset else 'custom'} · configuration SHA-256 "
            f"{config_hash(config)} · dataset SHA-256 {data.content_hash()} · seed {config.seed}."
        )
    )
    if found.skipped:
        blocks.append(
            Paragraph(
                "Tables left out because of their size: "
                + ", ".join(f"{s.key} ({s.cells:,} cells)" for s in found.skipped)
                + ".",
                "note",
            )
        )
    blocks.append(Code(yaml.safe_dump(to_data(config), sort_keys=False, allow_unicode=True)))
    created = "" if view.created is None else view.created.strftime("%Y-%m-%d %H:%M UTC")
    subtitle = "  ·  ".join(part for part in (data.name, view.run_id or "", created) if part)
    return Report("Context-synthetic model — run report", subtitle, tuple(blocks))


# ---------------------------------------------------------------- HTML

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f6f6f4; color: #0b0b0b;
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 32px 28px 64px; background: #ffffff; }
header h1 { font-size: 26px; margin: 0 0 4px; }
header p { margin: 0; color: #52514e; }
h2 { font-size: 19px; margin: 40px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #e1e0d9; }
h3 { font-size: 16px; margin: 28px 0 8px; }
p { margin: 10px 0; }
p.note { color: #52514e; font-size: 13px; }
p.warning { background: #fce4d6; border-left: 4px solid #c55a11; color: #5b2a08;
  padding: 10px 14px; border-radius: 4px; font-weight: 600; }
figure { margin: 18px 0; }
figure svg { max-width: 100%; height: auto; display: block; }
figcaption, caption { color: #52514e; font-size: 13px; text-align: left; padding: 6px 0; }
.scroll { overflow-x: auto; margin: 12px 0 4px; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
caption { caption-side: top; font-weight: 600; color: #0b0b0b; font-size: 14px; }
th, td { padding: 4px 10px; border-bottom: 1px solid #e1e0d9; text-align: left;
  white-space: nowrap; }
thead th { background: #f0efec; border-bottom: 1px solid #c3c2b7; font-weight: 600; }
td.num, th.num { text-align: right; }
tbody th { font-weight: 600; }
pre { background: #f6f6f4; border: 1px solid #e1e0d9; border-radius: 4px; padding: 12px 14px;
  font: 12.5px/1.45 ui-monospace, "Cascadia Mono", Consolas, monospace; overflow-x: auto; }
footer { margin-top: 40px; color: #898781; font-size: 12.5px; }
@media print { body { background: #ffffff; } main { max-width: none; padding: 0; }
  h2 { break-after: avoid; } figure, table { break-inside: avoid; } }
"""


def _svg(spec: FigureSpec, theme: Theme) -> str:
    buffer = io.BytesIO()
    figure = render(spec, theme)
    figure.savefig(buffer, format="svg", metadata={"Date": None})
    text = buffer.getvalue().decode("utf-8")
    return text[text.index("<svg") :]


def _html_table(block: TableBlock, decimals: int) -> str:
    table = block.table
    numeric = [c >= table.labels and is_numeric(table, c) for c in range(len(table.columns))]

    def cls(c: int) -> str:
        return ' class="num"' if numeric[c] else ""

    head = "".join(f"<th{cls(c)}>{html.escape(name)}</th>" for c, name in enumerate(table.columns))
    lines = []
    for row in table.rows:
        cells = []
        for c, value in enumerate(row):
            tag = "th" if c < table.labels else "td"
            scope = ' scope="row"' if tag == "th" else ""
            cells.append(
                f"<{tag}{scope}{cls(c)}>{html.escape(format_cell(value, decimals))}</{tag}>"
            )
        lines.append("<tr>" + "".join(cells) + "</tr>")
    remark = _omitted(block)
    note = " ".join(part for part in (table.note, remark) if part)
    foot = f'<p class="note">{html.escape(note)}</p>' if note else ""
    return (
        f'<div class="scroll"><table><caption>{html.escape(table.title)}</caption>'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(lines)}</tbody></table></div>{foot}"
    )


def _omitted(block: TableBlock) -> str:
    parts = []
    if block.omitted_rows:
        parts.append(f"{block.omitted_rows} more row(s)")
    if block.omitted_columns:
        parts.append(f"{block.omitted_columns} more column(s)")
    if not parts:
        return ""
    return f"Shortened: {join_and(parts)} in the table {block.table.key} of the CSV export."


def html_text(report: Report, *, theme: Theme = LIGHT, decimals: int = DEFAULT_DECIMALS) -> str:
    """The report as one self-contained HTML document (figures embedded as SVG)."""
    body: list[str] = []
    for block in report.blocks:
        if isinstance(block, Heading):
            body.append(f"<h{block.level}>{html.escape(block.text)}</h{block.level}>")
        elif isinstance(block, Paragraph):
            kind = "" if block.kind == "text" else f' class="{block.kind}"'
            body.append(f"<p{kind}>{html.escape(block.text)}</p>")
        elif isinstance(block, TableBlock):
            body.append(_html_table(block, decimals))
        elif isinstance(block, FigureBlock):
            body.append(
                f'<figure role="img" aria-label="{html.escape(block.spec.title, quote=True)}">'
                f"{_svg(block.spec, theme)}"
                f"<figcaption>{html.escape(block.spec.title)}</figcaption></figure>"
            )
        else:
            body.append(f"<pre>{html.escape(block.text)}</pre>")
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(report.title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        f"<main>\n<header><h1>{html.escape(report.title)}</h1>"
        f"<p>{html.escape(report.subtitle)}</p></header>\n"
        + "\n".join(body)
        + f"\n<footer>Generated by context-synthetic-recognition {html.escape(__version__)}."
        "</footer>\n</main>\n</body>\n</html>\n"
    )


# ---------------------------------------------------------------- PDF

_PDF_FONT = "CSR-Sans"
_PDF_BOLD = "CSR-Sans-Bold"
_PDF_MONO = "CSR-Mono"


def _reportlab(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ExportError(
            'the PDF report needs reportlab: pip install "context-synthetic-recognition[pdf]"'
        ) from error


def _register_fonts() -> None:
    """Register DejaVu Sans (shipped with matplotlib): it has the Greek and the subscripts."""
    matplotlib = importlib.import_module("matplotlib")
    pdfmetrics = _reportlab("reportlab.pdfbase.pdfmetrics")
    ttfonts = _reportlab("reportlab.pdfbase.ttfonts")
    folder = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, file in (
        (_PDF_FONT, "DejaVuSans.ttf"),
        (_PDF_BOLD, "DejaVuSans-Bold.ttf"),
        (_PDF_MONO, "DejaVuSansMono.ttf"),
    ):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(ttfonts.TTFont(name, str(folder / file)))


def _markup(text: str) -> str:
    """Text for a reportlab paragraph: XML-escaped, line breaks kept."""
    return html.escape(text, quote=False).replace("\n", "<br/>")


def write_pdf(
    report: Report,
    path: Path,
    *,
    theme: Theme = LIGHT,
    decimals: int = DEFAULT_DECIMALS,
    dpi: int = 160,
) -> Path:
    """Write the report as a PDF (A4; figures embedded as PNG).

    Raises:
        ExportError: reportlab is not installed (optional extra ``[pdf]``).
    """
    platypus = _reportlab("reportlab.platypus")
    styles_module = _reportlab("reportlab.lib.styles")
    colors = _reportlab("reportlab.lib.colors")
    pagesizes = _reportlab("reportlab.lib.pagesizes")
    units = _reportlab("reportlab.lib.units")
    _register_fonts()
    mm = units.mm
    page_width, page_height = pagesizes.A4
    margin = 16 * mm
    usable = page_width - 2 * margin
    style = styles_module.ParagraphStyle
    ink, secondary = colors.HexColor(theme.ink), colors.HexColor(theme.ink_secondary)
    base = style("base", fontName=_PDF_FONT, fontSize=9.5, leading=13.5, textColor=ink)
    styles = {
        "title": style("title", parent=base, fontName=_PDF_BOLD, fontSize=18, leading=23),
        "subtitle": style("subtitle", parent=base, textColor=secondary, spaceAfter=8),
        2: style(
            "h2",
            parent=base,
            fontName=_PDF_BOLD,
            fontSize=13,
            leading=17,
            spaceBefore=14,
            spaceAfter=5,
        ),
        3: style(
            "h3",
            parent=base,
            fontName=_PDF_BOLD,
            fontSize=11,
            leading=15,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "text": style("text", parent=base, spaceAfter=5),
        "note": style(
            "note", parent=base, fontSize=8, leading=11, textColor=secondary, spaceAfter=6
        ),
        "warning": style(
            "warning",
            parent=base,
            fontName=_PDF_BOLD,
            textColor=colors.HexColor("#5b2a08"),
            backColor=colors.HexColor("#fce4d6"),
            borderPadding=(6, 8, 6, 8),
            spaceBefore=6,
            spaceAfter=12,
        ),
        "caption": style(
            "caption", parent=base, fontName=_PDF_BOLD, fontSize=9, spaceBefore=8, spaceAfter=3
        ),
        "code": style("code", parent=base, fontName=_PDF_MONO, fontSize=7.5, leading=10),
    }
    story: list[Any] = [
        platypus.Paragraph(_markup(report.title), styles["title"]),
        platypus.Paragraph(_markup(report.subtitle), styles["subtitle"]),
    ]
    grid = colors.HexColor(theme.grid)
    for block in report.blocks:
        if isinstance(block, Heading):
            story.append(platypus.Paragraph(_markup(block.text), styles[min(block.level, 3)]))
        elif isinstance(block, Paragraph):
            story.append(platypus.Paragraph(_markup(block.text), styles[block.kind]))
        elif isinstance(block, TableBlock):
            story += _pdf_table(block, platypus, colors, styles, usable, decimals, grid)
        elif isinstance(block, FigureBlock):
            buffer = io.BytesIO()
            save_figure_to(block.spec, buffer, theme, dpi)
            width_in, height_in = block.spec.size
            width = min(usable, width_in * 72)
            image = platypus.Image(buffer, width=width, height=width * height_in / width_in)
            story.append(
                platypus.KeepTogether(
                    [image, platypus.Paragraph(_markup(block.spec.title), styles["note"])]
                )
            )
        else:
            story.append(platypus.Preformatted(block.text, styles["code"]))

    def footer(canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setFont(_PDF_FONT, 7.5)
        canvas.setFillColor(secondary)
        canvas.drawString(margin, 9 * mm, f"context-synthetic-recognition {__version__}")
        canvas.drawRightString(page_width - margin, 9 * mm, f"Page {document.page}")
        canvas.restoreState()

    path.parent.mkdir(parents=True, exist_ok=True)
    document = platypus.SimpleDocTemplate(
        str(path),
        pagesize=(page_width, page_height),
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title=report.title,
        subject=report.subtitle,
        author=f"context-synthetic-recognition {__version__}",
        invariant=1,
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return path


def save_figure_to(spec: FigureSpec, buffer: io.BytesIO, theme: Theme, dpi: int) -> None:
    """Render a figure as PNG into ``buffer`` (rewound for reading)."""
    figure = render(spec, theme)
    figure.savefig(buffer, format="png", dpi=dpi, facecolor=figure.get_facecolor())
    buffer.seek(0)


def _pdf_table(
    block: TableBlock,
    platypus: Any,
    colors: Any,
    styles: dict[Any, Any],
    usable: float,
    decimals: int,
    grid: Any,
) -> list[Any]:
    table = block.table
    columns = len(table.columns)
    text = [list(table.columns)] + [[format_cell(v, decimals) for v in row] for row in table.rows]
    measure = _reportlab("reportlab.pdfbase.pdfmetrics").stringWidth
    numeric = [c >= table.labels and is_numeric(table, c) for c in range(columns)]

    def natural(size: float) -> list[float]:
        """Width every column needs at a font size: its widest cell, long text capped."""
        widths = []
        for c in range(columns):
            # a header may wrap between its words; a number never wraps
            words = text[0][c].split() or [""]
            head = max(measure(word, _PDF_BOLD, size) for word in words)
            body = max((measure(line[c], _PDF_FONT, size) for line in text[1:]), default=0.0)
            widths.append(max(head, body if numeric[c] else min(body, 220.0)) + 7.0)
        return widths

    # the largest font at which the table fits the page; below 5.5 pt the text columns wrap
    size = 8.0
    widths = natural(size)
    while sum(widths) > usable and size > 5.5:
        size -= 0.5
        widths = natural(size)
    excess = sum(widths) - usable
    flexible = [c for c in range(columns) if not numeric[c]]
    if excess > 0 and flexible:
        room = sum(widths[c] for c in flexible)
        shrink = max(0.25, 1.0 - excess / room)
        for c in flexible:
            widths[c] *= shrink
    if sum(widths) > usable:
        widths = [w * usable / sum(widths) for w in widths]
    cell_style = styles["text"].clone("cell", fontSize=size, leading=size * 1.25, spaceAfter=0)
    head_style = cell_style.clone("head", fontName=_PDF_BOLD)
    data = [
        [platypus.Paragraph(_markup(value), head_style if r == 0 else cell_style) for value in line]
        for r, line in enumerate(text)
    ]
    flowable = platypus.Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    flowable.setStyle(
        platypus.TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efec")),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, grid),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    out = [platypus.Paragraph(_markup(table.title), styles["caption"]), flowable]
    note = " ".join(part for part in (table.note, _omitted(block)) if part)
    if note:
        out.append(platypus.Paragraph(_markup(note), styles["note"]))
    return out


def figure_data_uri(spec: FigureSpec, theme: Theme = LIGHT, dpi: int = 160) -> str:
    """A figure as a PNG ``data:`` URI (for viewers that cannot show inline SVG)."""
    buffer = io.BytesIO()
    save_figure_to(spec, buffer, theme, dpi)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


__all__ = [
    "Block",
    "Code",
    "ExportError",
    "FigureBlock",
    "Heading",
    "Paragraph",
    "Report",
    "TableBlock",
    "build_report",
    "figure_data_uri",
    "html_text",
    "save_figure",
    "shorten",
    "write_pdf",
]
