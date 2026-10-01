"""Exporters: the Excel mirror, CSV/JSON, Markdown and LaTeX tables, figures and the run report.

Everything is exported from one :class:`~.view.RunView` — an evaluated experiment with what the
workbook shows beside it — so the workbook, the tables, the figures and the report agree by
construction. :func:`~.run.export_result` writes any set of formats into a folder; the CLI
(``csr run --export``, ``csr export``) and the GUI call it.
"""

from context_synthetic_recognition.export.run import (
    ALL,
    EXPORTERS,
    ExportOptions,
    ExportSummary,
    export_result,
    export_view,
    resolve_formats,
)
from context_synthetic_recognition.export.view import RunView, build_view

__all__ = [
    "ALL",
    "EXPORTERS",
    "ExportOptions",
    "ExportSummary",
    "RunView",
    "build_view",
    "export_result",
    "export_view",
    "resolve_formats",
]
