"""Tables as text: CSV, JSON, Markdown and LaTeX.

CSV and JSON keep every digit (they are data); Markdown and LaTeX round to a number of decimals
(4 by default, like the workbook's ``0.0000`` cells) and write an undefined value as "—". The
LaTeX renderer turns the article's notation (ρ_I, a₆, S₁₀, f_k(μ), θ/γ, …) into math mode and
uses ``booktabs`` rules; long tables become ``longtable`` environments.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from context_synthetic_recognition.export.tables import Cell, Table
from context_synthetic_recognition.notation import DASH

DEFAULT_DECIMALS = 4
"""Decimals of real numbers in Markdown, LaTeX and the report (the workbook's ``0.0000``)."""
LONGTABLE_ROWS = 40
"""LaTeX tables with more rows are set as ``longtable`` (they may break across pages)."""


def format_cell(value: Cell, decimals: int = DEFAULT_DECIMALS) -> str:
    """A cell for reading: integers as they are, reals rounded, undefined as "—"."""
    if value is None:
        return DASH
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.{decimals}f}"
        return "0." + "0" * decimals if float(text) == 0 else text  # no "-0.0000"
    return value


def is_numeric(table: Table, column: int) -> bool:
    """Whether a column holds numbers only (undefined cells aside) — it is then right-aligned."""
    values = [row[column] for row in table.rows if row[column] is not None]
    return bool(values) and all(
        isinstance(v, int | float) and not isinstance(v, bool) for v in values
    )


# ---------------------------------------------------------------- CSV and JSON


def csv_text(table: Table) -> str:
    """The table as CSV: a header row, full precision, an empty cell for an undefined value."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(table.columns)
    for row in table.rows:
        writer.writerow(["" if v is None else repr(v) if isinstance(v, float) else v for v in row])
    return buffer.getvalue()


def table_data(table: Table) -> dict[str, Any]:
    """The table as JSON-ready data."""
    return {
        "key": table.key,
        "title": table.title,
        "group": table.group,
        "note": table.note,
        "columns": list(table.columns),
        "rows": [list(row) for row in table.rows],
    }


def json_text(data: Any) -> str:
    """JSON with the notation kept as Unicode (ρ, a₆) and a final newline."""
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------- Markdown


def _markdown_cell(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def markdown_table(table: Table, decimals: int = DEFAULT_DECIMALS) -> str:
    """The table as a GitHub-flavoured Markdown table (no title)."""
    alignment = [
        "---:" if c >= table.labels and is_numeric(table, c) else ":---"
        for c in range(len(table.columns))
    ]
    lines = [
        "| " + " | ".join(_markdown_cell(c) for c in table.columns) + " |",
        "| " + " | ".join(alignment) + " |",
    ]
    for row in table.rows:
        cells = (_markdown_cell(format_cell(v, decimals)) for v in row)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def markdown_text(table: Table, decimals: int = DEFAULT_DECIMALS, level: int = 3) -> str:
    """The table as Markdown with its title as a heading and its note below."""
    parts = [f"{'#' * level} {table.title}\n", markdown_table(table, decimals)]
    if table.note:
        parts.append(f"*{table.note}*\n")
    return "\n".join(parts)


# ---------------------------------------------------------------- LaTeX

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    # with the default (OT1) font encoding these three print as other characters in text mode
    "|": r"\textbar{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "№": "No.",
    "—": "---",
    "–": "--",
    "“": "``",
    "”": "''",
    "’": "'",
    "✓": r"\checkmark{}",
    "✅": r"\checkmark{}",
    "⚠": r"\textbf{(!)}",
}
"""Text-mode replacements."""

_LATEX_MATH = {
    "ρ": r"\rho",
    "ω": r"\omega",
    "η": r"\eta",
    "θ": r"\theta",
    "γ": r"\gamma",
    "α": r"\alpha",
    "β": r"\beta",
    "δ": r"\delta",
    "λ": r"\lambda",
    "μ": r"\mu",
    "σ": r"\sigma",
    "χ": r"\chi",
    "ϰ": r"\varkappa",
    "ϕ": r"\varphi",
    "Ψ": r"\Psi",
    "Σ": r"\Sigma",
    "∈": r"\in",
    "∅": r"\emptyset",
    "≥": r"\geq",
    "≤": r"\leq",
    "≠": r"\neq",
    "≈": r"\approx",
    "−": "-",
    "±": r"\pm",
    "·": r"\cdot",
    "×": r"\times",
    "→": r"\to",
    "←": r"\leftarrow",
    "⇒": r"\Rightarrow",
    "…": r"\ldots",
    "∞": r"\infty",
    "∪": r"\cup",
    "∖": r"\setminus",
    "ŷ": r"\hat{y}",
    "½": r"\tfrac{1}{2}",
    "✗": r"\times",
    "❌": r"\times",
}
"""Characters set in math mode."""

_SUBSCRIPTS = dict(zip("₀₁₂₃₄₅₆₇₈₉ᵢⱼₖₚₜᵤₘₙ", "0123456789ijkptumn", strict=True))
_SUPERSCRIPTS = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789", strict=True))


def _is_word(char: str) -> bool:
    return char.isascii() and char.isalnum()


def latex_text(text: str) -> str:
    r"""Text with the article's notation as LaTeX: special characters escaped, symbols in math mode.

    ``a₆`` → ``$a_{6}$``, ``ρ_I`` → ``$\rho_{I}$``, ``f_k(μ)`` → ``$f_{k}$($\mu$)``,
    ``score₁`` → ``score$_{1}$``; an underscore that is not a subscript is escaped.
    """
    pieces: list[tuple[bool, str]] = []  # (math?, text)

    def pull_base() -> None:
        """Move a preceding single letter (S, a, x, r, f, …) into math mode as the base."""
        if not pieces or pieces[-1][0]:
            return
        plain = pieces[-1][1]
        if plain and plain[-1].isascii() and plain[-1].isalpha():
            if len(plain) > 1 and _is_word(plain[-2]):
                return  # the end of a longer word: score₁
            pieces[-1] = (False, plain[:-1])
            pieces.append((True, plain[-1]))

    def add(math: bool, piece: str) -> None:
        if pieces and pieces[-1][0] == math and not math:
            pieces[-1] = (False, pieces[-1][1] + piece)
            return
        if math and piece[0] in "_^'" and pieces and pieces[-1][0]:
            pieces[-1] = (True, pieces[-1][1].rstrip())  # \rho_{I}, not \rho _{I}
        pieces.append((math, piece))

    i, n = 0, len(text)
    while i < n:
        char = text[i]
        if char == "′":  # x′: a prime on the preceding letter
            pull_base()
            add(True, "'")
            i += 1
        elif char in _SUBSCRIPTS or char in _SUPERSCRIPTS:
            table, mark = (_SUBSCRIPTS, "_") if char in _SUBSCRIPTS else (_SUPERSCRIPTS, "^")
            j = i
            while j < n and text[j] in table:
                j += 1
            pull_base()
            add(True, f"{mark}{{{''.join(table[c] for c in text[i:j])}}}")
            i = j
        elif char == "_" and i + 1 < n and _is_word(text[i + 1]):
            j = i + 1
            while j < n and _is_word(text[j]):
                j += 1
            pull_base()
            if pieces and pieces[-1][0]:
                add(True, f"_{{{text[i + 1 : j]}}}")
            else:
                add(False, r"\_" + text[i + 1 : j])
            i = j
        elif char in _LATEX_MATH:
            command = _LATEX_MATH[char]
            add(True, command + (" " if command.startswith("\\") else ""))
            i += 1
        else:
            add(False, _LATEX_SPECIAL.get(char, char))
            i += 1

    out: list[str] = []
    k = 0
    while k < len(pieces):
        math, piece = pieces[k]
        if not math:
            out.append(piece)
            k += 1
            continue
        run = [piece]
        k += 1
        while k < len(pieces) and pieces[k][0]:
            run.append(pieces[k][1])
            k += 1
        out.append("$" + "".join(run).strip() + "$")
    return "".join(out)


def _latex_cell(value: Cell, decimals: int) -> str:
    """A cell as LaTeX; a negative number in math mode, so that it gets a minus, not a hyphen."""
    text = format_cell(value, decimals)
    if isinstance(value, int | float) and not isinstance(value, bool) and text.startswith("-"):
        return f"${text}$"
    return latex_text(text)


def latex_table(
    table: Table,
    decimals: int = DEFAULT_DECIMALS,
    *,
    label: str | None = None,
    long_rows: int = LONGTABLE_ROWS,
) -> str:
    r"""The table as LaTeX (``booktabs``; ``longtable`` beyond ``long_rows`` rows).

    The preamble needs ``\usepackage{booktabs, longtable, amsmath, amssymb}``.
    """
    columns = len(table.columns)
    spec = "".join(
        "r" if c >= table.labels and is_numeric(table, c) else "l" for c in range(columns)
    )
    header = " & ".join(latex_text(c) for c in table.columns) + r" \\"
    body = [" & ".join(_latex_cell(v, decimals) for v in row) + r" \\" for row in table.rows]
    caption = latex_text(table.title)
    if table.note:
        caption += ". " + latex_text(table.note)
    name = label or f"tab:{table.key}"
    if len(table.rows) > long_rows:
        lines = [
            rf"\begin{{longtable}}{{{spec}}}",
            rf"  \caption{{{caption}}}\label{{{name}}} \\",
            r"  \toprule",
            f"  {header}",
            r"  \midrule",
            r"  \endfirsthead",
            r"  \toprule",
            f"  {header}",
            r"  \midrule",
            r"  \endhead",
            r"  \bottomrule",
            r"  \endfoot",
            *(f"  {line}" for line in body),
            r"\end{longtable}",
        ]
    else:
        lines = [
            r"\begin{table}[htbp]",
            r"  \centering",
            rf"  \caption{{{caption}}}",
            rf"  \label{{{name}}}",
            rf"  \begin{{tabular}}{{{spec}}}",
            r"    \toprule",
            f"    {header}",
            r"    \midrule",
            *(f"    {line}" for line in body),
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    return "\n".join(lines) + "\n"


def latex_document(keys: Iterable[str], title: str) -> str:
    """A small document that inputs the table files — to preview them all at once."""
    inputs = "\n".join(rf"\input{{{key}}}" for key in keys)
    return (
        "% Preview of the exported tables: pdflatex or xelatex all-tables.tex\n"
        "\\documentclass[a4paper,10pt]{article}\n"
        "\\usepackage[margin=18mm,landscape]{geometry}\n"
        "\\usepackage{booktabs, longtable, amsmath, amssymb}\n"
        f"\\title{{{latex_text(title)}}}\n"
        "\\date{}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\small\n"
        f"{inputs}\n"
        "\\end{document}\n"
    )


# ---------------------------------------------------------------- files


def write_text(path: Path, text: str) -> Path:
    """Write a UTF-8 text file with LF line endings (the same bytes on every platform)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def index_rows(tables: Sequence[Table], extension: str) -> list[list[Any]]:
    """Rows of a folder's index: file, key, group, title, size."""
    return [
        [f"{t.key}{extension}", t.key, t.group, t.title, len(t.rows), len(t.columns), t.note]
        for t in tables
    ]


INDEX_COLUMNS = ("file", "key", "group", "title", "rows", "columns", "note")
"""Columns of ``index.csv``."""
