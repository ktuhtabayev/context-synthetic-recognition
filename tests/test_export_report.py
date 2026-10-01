"""The run report: the document, its HTML form and the PDF."""

import importlib
import re
from pathlib import Path
from typing import Any

import pytest

from context_synthetic_recognition import __version__
from context_synthetic_recognition.export import RunView
from context_synthetic_recognition.export.figures import figure_specs
from context_synthetic_recognition.export.report import (
    Code,
    ExportError,
    FigureBlock,
    Heading,
    Paragraph,
    Report,
    TableBlock,
    build_report,
    figure_data_uri,
    html_text,
    shorten,
    write_pdf,
)
from context_synthetic_recognition.export.tables import Table, run_tables
from context_synthetic_recognition.export.theme import DARK

from .conftest import EXPERIMENT_RUN_ID
from .shapes import SHAPES, shape

SECTIONS = [
    "Summary",
    "Data",
    "Local contexts Ψ_ρ,k (Steps 1–3)",
    "Synthetic features Ψ(r) (Steps 4–8)",
    "Hierarchical agglomerative grouping (Step 9)",
    "Meta-algorithm (Steps 10–12)",
    "Evaluation",
    "Model properties (Definitions 1–6, Theorem)",
    "Reproducibility",
]


@pytest.fixture(scope="module")
def report(experiment_view: RunView) -> Report:
    return build_report(experiment_view)


def wide_table(rows: int, columns: int) -> Table:
    return Table(
        "wide",
        "Wide",
        tuple(f"c{c}" for c in range(columns)),
        tuple(tuple(r * columns + c for c in range(columns)) for r in range(rows)),
        "g",
    )


# ---------------------------------------------------------------- the document


def test_shortening_tables() -> None:
    small = wide_table(3, 4)
    assert shorten(small) == TableBlock(small)
    cut = shorten(wide_table(30, 20))
    assert (cut.omitted_rows, cut.omitted_columns) == (6, 5)
    assert len(cut.table.rows) == 24
    # the first columns, a gap, the last columns
    assert cut.table.columns == (*(f"c{c}" for c in range(11)), "…", "c16", "c17", "c18", "c19")
    assert cut.table.rows[0][10:13] == (10, "…", 16)
    assert cut.table.key == "wide"
    rows_only = shorten(wide_table(30, 4), max_rows=10)
    assert (rows_only.omitted_rows, rows_only.omitted_columns) == (20, 0)
    assert rows_only.table.columns == small.columns


def test_the_report_of_the_experiment(report: Report, experiment_view: RunView) -> None:
    assert report.title == "Context-synthetic model — run report"
    assert report.subtitle.startswith(
        f"{experiment_view.dataset.name}  ·  {EXPERIMENT_RUN_ID}  ·  "
    )
    assert report.subtitle.endswith(" UTC")
    headings = [b for b in report.blocks if isinstance(b, Heading)]
    assert [h.text for h in headings if h.level == 2] == SECTIONS
    assert [h.text for h in headings if h.level == 3] == [
        "Margins of the latent features",
        "Sensitivity to the two switches",
    ]
    tables = [b.table.key for b in report.blocks if isinstance(b, TableBlock)]
    assert tables == [
        "summary",
        "features",
        "dataset",
        "synthetic-features",
        "psi",
        "hag-iterations",
        "meta-dataset",
        "new-object-steps",
        "resubstitution",
        "metrics",
        "margins",
        "sensitivity",
        "properties",
        "operator-pairs",
        "boundary-ties",
    ]
    figures = [b.spec.key for b in report.blocks if isinstance(b, FigureBlock)]
    assert sorted(figures) == sorted(spec.key for spec in figure_specs(experiment_view))
    assert isinstance(report.blocks[-1], Code)
    assert "centres: running" in report.blocks[-1].text


def test_the_report_highlights_the_template_deviations(report: Report) -> None:
    warnings = [b for b in report.blocks if isinstance(b, Paragraph) and b.kind == "warning"]
    assert len(warnings) == 1
    assert report.blocks[1] is warnings[0]  # the first thing after the "Summary" heading
    text = warnings[0].text
    assert text.startswith("⚠ Two template calculations differ from the article: ")
    assert "hag.centres (ADR-002) and hag.step4_passes (ADR-003)" in text
    assert text.endswith("the default, which reproduces the template workbooks exactly.")
    hag = next(
        b.text for b in report.blocks if isinstance(b, Paragraph) and b.text.startswith("α = ")
    )
    assert "⚠ class centres = running, ⚠ STEP 4 passes = 2" in hag
    assert "TUPLAM = {a₆, a₃, a₁, a₂, a₄}, p = 4 latent feature(s)" in hag
    # a run with the article's switches says so as prominently
    article = build_report(shape("numeric"))
    warning = article.blocks[1]
    assert isinstance(warning, Paragraph)
    assert warning.kind == "warning"
    assert "This run follows the article for both" in warning.text


def test_the_reproducibility_section(report: Report, experiment_view: RunView) -> None:
    text = next(
        b.text for b in report.blocks if isinstance(b, Paragraph) and b.text.startswith("Run ")
    )
    assert text.startswith(f"Run {EXPERIMENT_RUN_ID} · context-synthetic-recognition {__version__}")
    assert "preset template" in text
    assert experiment_view.dataset.content_hash() in text
    unsaved = build_report(shape("tiny"))
    text = next(
        b.text for b in unsaved.blocks if isinstance(b, Paragraph) and b.text.startswith("Run ")
    )
    assert text.startswith("Run (not saved) · ")
    assert "preset custom" in text


def test_tables_left_out_are_named(experiment_view: RunView) -> None:
    limited = build_report(experiment_view, run_tables(experiment_view, max_cells=100))
    notes = [b.text for b in limited.blocks if isinstance(b, Paragraph) and b.kind == "note"]
    assert len(notes) == 1
    assert notes[0].startswith("Tables left out because of their size: ")
    assert "neighbours-rho (540 cells)" in notes[0]
    assert "dataset" not in [b.table.key for b in limited.blocks if isinstance(b, TableBlock)]


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_reports_of_other_shapes(name: str) -> None:
    view = shape(name)
    found = build_report(view)
    sections = [b.text for b in found.blocks if isinstance(b, Heading) and b.level == 2]
    assert sections == SECTIONS
    sub = [b.text for b in found.blocks if isinstance(b, Heading) and b.level == 3]
    assert sub == (["Margins of the latent features"] if view.hag.p else [])
    for block in found.blocks:
        if isinstance(block, TableBlock):
            assert len(block.table.rows) <= 60
            assert len(block.table.columns) <= 16


# ---------------------------------------------------------------- HTML


def test_the_html_report(report: Report, experiment_view: RunView) -> None:
    text = html_text(report)
    assert text.startswith('<!doctype html>\n<html lang="en">')
    assert text.endswith("</html>\n")
    assert "<title>Context-synthetic model — run report</title>" in text
    assert text.count("<h2>") == len(SECTIONS)
    assert text.count('<p class="warning">') == 1
    assert text.count("<figure ") == len(figure_specs(experiment_view))
    assert text.count("<svg") == len(figure_specs(experiment_view))
    assert text.count("<table>") == 15
    # self-contained: no script, nothing loaded from elsewhere
    assert "<script" not in text
    assert not re.findall(r'(?:src|href)="(?!#|data:)', text)
    assert '<td class="num">0.1482</td>' in text  # 4 decimals by default
    assert '<th scope="row">S₁</th>' in text
    # 17 columns: the widest table is cut and says where the rest is
    assert text.count("Shortened: ") == 1
    assert "Shortened: 2 more column(s) in the table synthetic-features of the CSV export." in text
    assert f"Generated by context-synthetic-recognition {__version__}." in text
    assert "0.14818" in html_text(report, decimals=5)


def test_the_html_escapes_and_says_what_was_shortened() -> None:
    table = Table(
        "odd",
        "a < b & c",
        ("Name", "Value"),
        tuple((f"<{i}>", float(i)) for i in range(30)),
        "g",
        note="x > y",
    )
    report = Report("T <1>", "sub & title", (Heading("H & M", 3), shorten(table), Code("a < b")))
    text = html_text(report, theme=DARK)
    assert "<title>T &lt;1&gt;</title>" in text
    assert "<h3>H &amp; M</h3>" in text
    assert "<caption>a &lt; b &amp; c</caption>" in text
    assert '<th scope="row">&lt;0&gt;</th>' in text
    assert "<pre>a &lt; b</pre>" in text
    assert (
        '<p class="note">x &gt; y Shortened: 6 more row(s) in the table odd of the CSV export.</p>'
        in text
    )


def test_a_figure_as_a_data_uri(experiment_view: RunView) -> None:
    spec = next(s for s in figure_specs(experiment_view) if s.key == "hag-criterion")
    uri = figure_data_uri(spec, dpi=40)
    assert uri.startswith("data:image/png;base64,iVBORw0KGgo")


# ---------------------------------------------------------------- PDF


def test_the_pdf_report(report: Report, tmp_path: Path) -> None:
    pytest.importorskip("reportlab")
    path = write_pdf(report, tmp_path / "deep" / "report.pdf", dpi=50)
    data = path.read_bytes()
    assert data.startswith(b"%PDF-")
    assert data.rstrip().endswith(b"%%EOF")
    assert len(data) > 50_000
    assert b"Context-synthetic model" in data or b"/Title" in data
    again = write_pdf(report, tmp_path / "again.pdf", dpi=50)
    assert again.read_bytes() == data  # invariant: no time stamp, no random ids


@pytest.mark.parametrize("name", ["numeric", "nominal", "heart270"])
def test_pdf_reports_of_other_shapes(name: str, tmp_path: Path) -> None:
    pytest.importorskip("reportlab")
    path = write_pdf(build_report(shape(name)), tmp_path / f"{name}.pdf", theme=DARK, dpi=40)
    assert path.read_bytes().startswith(b"%PDF-")


def test_the_pdf_needs_reportlab(
    report: Report, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = importlib.import_module

    def without_reportlab(name: str, package: str | None = None) -> Any:
        if name.startswith("reportlab"):
            raise ImportError(name)
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", without_reportlab)
    with pytest.raises(ExportError, match=r"needs reportlab: pip install .*\[pdf\]"):
        write_pdf(report, tmp_path / "report.pdf")
    assert not (tmp_path / "report.pdf").exists()
