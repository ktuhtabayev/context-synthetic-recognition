# Exporting

A run can be written out as an Excel workbook that mirrors the author's experiment, as tables
(CSV, JSON, Markdown, LaTeX), as figures (PNG, SVG) and as a report (HTML, PDF). Every format
reads the same run, so they cannot disagree.

!!! warning "Two template calculations differ from the article"
    Every export highlights them: a warning block first in the report, the ⚠ rows of the
    *Parameters* sheet, the sheets *Template Deviations* and *Sensitivity (Switches)*, the
    `summary` table and the `sensitivity` figure. See [DECISIONS.md](DECISIONS.md) (ADR-002 –
    ADR-004).

## Commands

```powershell
.\.venv\Scripts\csr.exe run heart-disease-10 --export all          # run, save, export every format
.\.venv\Scripts\csr.exe run heart-disease-10 -e excel -e csv,json  # only some formats
.\.venv\Scripts\csr.exe export                                     # the latest run in runs\
.\.venv\Scripts\csr.exe export 20261001_120000_5a0c7b92 -f html,pdf
.\.venv\Scripts\csr.exe export runs\20261001_120000_5a0c7b92 --out article\tables -f latex --decimals 3
.\.venv\Scripts\csr.exe export --object 4                          # demonstrate S₄ left out of its context
.\.venv\Scripts\csr.exe export --values "58 1 3 125 250 0 2 150 0 1 2 1 7"   # … or a new object
```

`csr run --export` writes into the new run folder. `csr export` takes a run folder (a path, a
name under `--runs-dir`, or nothing for the latest run) and **repeats the run from it**: the
manifest's configuration on the folder's `dataset.json`. The computation is deterministic, so the
export shows the complete trace although the folder stores only the results; if the repeated
results differ from the stored ones (another package version), the command says so.

| Option | Meaning |
|---|---|
| `-f`, `--format` | `all` (default) or any of the formats below; repeat it or separate by commas |
| `--out`, `-o` | where to write (default: the run folder) |
| `--object N` | the training object to demonstrate the meta-algorithm on, left out of its own context (default 1) |
| `--values "…"` | a new object (n values) for the same sheets and tables |
| `--sensitivity` / `--no-sensitivity` | evaluate / leave out the four switch settings (default: as stored with the run, otherwise only for at most 60 objects) |
| `--decimals` | decimals in Markdown, LaTeX and the report (default 4) |
| `--dpi` | resolution of the PNG figures (default 200) |
| `--theme` | `light` (default; print) or `dark` |

## Formats

| Format (aliases) | Files | Content |
|---|---|---|
| `excel` (`xlsx`, `workbook`) | `workbook.xlsx` | the mirror of the Excel experiment |
| `csv` (`tables`) | `tables/<key>.csv`, `tables/index.csv` | every table at full precision |
| `json` (`trace`) | `trace.json` | the same tables in one file, with the configuration and the hashes |
| `markdown` (`md`) | `markdown/<key>.md`, `markdown/tables.md` | tables that fit a page, rounded |
| `latex` (`tex`) | `latex/<key>.tex`, `latex/all-tables.tex` | the same as `booktabs` tables |
| `figures` (`plots`) | `figures/<key>.png`, `.svg` | the figures |
| `html` (`report`) | `report.html` | the run report, one self-contained file |
| `pdf` | `report.pdf` | the run report (needs the extra `[pdf]`: `pip install -e ".[pdf]"`) |

Text files are UTF-8 with LF line endings; the SVG and PDF files carry no time stamp. The same run
exports to the same bytes on every platform.

## The Excel mirror

`workbook.xlsx` has the sheets, the layout and the style of *Context-Synthetic Model – Full
Experiment*: the overview with the workbook map, *Parameters* with both switches, the data,
Steps 1–12 table by table, the evaluation sheets and the model-property checks.

- **Values, no formulas.** Every cell is a number or a text computed by the package, so the file
  reads the same in Excel, LibreOffice and pandas.
- **Generated for the data.** One *Sorted Neighbors* sheet per base operator, one *Greedy upon
  Weight* sheet per possible HAG iteration, a column per permitted k and per synthetic feature, a
  fold sheet per hold-out protocol (*Leave-One-Out*, *Stratified K-Fold*, *Repeated K-Fold*,
  *Hold-Out*).
- **The experiment is reproduced cell for cell.** For Heart-Disease (10, 13, 2) with the default
  configuration every computed cell sits at its original address:

    ```powershell
    .\.venv\Scripts\csr.exe run heart-disease-10 --export excel
    .\.venv\Scripts\csr.exe validate --against runs\<run>\workbook.xlsx   # 11,631 cells in 29 sheets
    ```

- **Differences from the original** (all intended): the distance sheet is named *Zhuravlyov
  Distances*; labels that the original shows as equation drawings are written as text; texts
  about live formulas are reworded; *Parameters* has one more block, "This run", with what
  identifies the run.

### Large data

A sheet per step does not scale to every table. Two limits (`ExcelOptions`) keep the workbook
usable; whatever they leave out is said on the sheet itself and listed by the command, and the
CSV tables hold everything.

| Limit | Default | Effect when exceeded |
|---|---|---|
| `max_matrix_cells` | 250,000 | an object × object table (distances, ranks) is replaced by a note naming its CSV table — beyond about 500 objects |
| `max_sheet_cells` | 150,000 | repeated blocks are written for as many objects or candidates as fit: neighbour blocks, HAG candidate blocks (the chosen feature q only), B1/B2 flags per object, membership tables (TUPLAM features only) |

## Tables

The tables follow the pipeline. `tables/index.csv` lists them with their titles and sizes.

| Stage | Keys |
|---|---|
| Summary, input | `summary`, `dataset`, `features` |
| Steps 1–3 | `normalized`, `distances-<operator>`, `neighbours-<operator>` |
| Steps 4–8 | `same-class-counts`, `chi1`, `psi`, `membership`, `bit-masks`, `synthetic-features`, `object-membership`, `correct-side`, `contributions` |
| Step 9 (HAG) | `hag-iterations`, `hag-candidates`, `hag-chosen-blocks` |
| Steps 10–12 | `meta-dataset`, `training-description`, `new-object`, `new-object-features`, `resubstitution`, `new-object-steps` |
| Evaluation | `metrics`, `class-metrics`, `confusion-<protocol>`, `roc-<protocol>`, `predictions-<protocol>`, `folds-<protocol>`, `margins`, `object-margins`, `sensitivity` |
| Checks | `properties`, `operator-pairs`, `boundary-ties` |

`normalized`, `psi` and `meta-dataset` are the datasets the model derives (the normalized data,
Ψ(r) and Y); they are written here, in the run folder, not to `datasets/`.

CSV and JSON keep every digit; an undefined value is an empty cell (`null` in JSON). Markdown and
LaTeX round and write "—". The LaTeX tables need `booktabs`, `longtable`, `amsmath` and `amssymb`;
the notation is set in math mode (a₆ → `$a_{6}$`, ρ_I → `$\rho_{I}$`), each table has the label
`tab:<key>`, and `all-tables.tex` previews them all:

```latex
\usepackage{booktabs, longtable, amsmath, amssymb}
...
\input{latex/synthetic-features}   % Table~\ref{tab:synthetic-features}
```

## Figures

| Key | Shows |
|---|---|
| `distances` | the distance matrix of every base operator, objects grouped by class |
| `neighbourhoods` | the class of the neighbour at every rank, with the permitted k |
| `synthetic-features` | informativeness ω and stability g of Ψ(r); the TUPLAM features emphasised |
| `hag-candidates`, `hag-criterion` | θ/γ of every candidate per iteration with the chosen q; crit against δ |
| `margins`, `margin-widths` | the latent features on a line with and without the majorizer; margin widths |
| `roc`, `confusion` | ROC of score₁ − score₂ and the confusion matrix per protocol |
| `outcomes-<protocol>` | correct, wrong and refused decisions of every method |
| `prediction-map-<protocol>` | the decision of every method for every held-out object (up to 40 objects) |
| `sensitivity` | ⚠ accuracy under the four settings of the two switches |

K1 and K2 have the same two colours in every figure; correct / wrong / refused are always named in
the legend, never shown by colour alone.

## From Python

```python
from pathlib import Path

from context_synthetic_recognition.data import load_builtin
from context_synthetic_recognition.export import ExportOptions, build_view, export_result
from context_synthetic_recognition.export.figures import draw_roc, figure_specs, render
from context_synthetic_recognition.export.tables import run_tables
from context_synthetic_recognition.services.runner import run_experiment

result = run_experiment(load_builtin("heart-disease-10"))
summary = export_result(result, Path("out"), ["excel", "latex"], ExportOptions(decimals=3))
summary.files["excel"]  # (out/workbook.xlsx,)
summary.notes  # what was left out because of its size

view = build_view(result)  # what every exporter reads
run_tables(view)["psi"].rows[0]  # ('S₁', 2, 2, 1, 2, 2, 2, 2)
figure = render(figure_specs(view)[0])  # a matplotlib Figure
```

The `draw_*` functions of `export.figures` draw on axes they are given (`draw_roc(ax, view,
protocol)`), so the figures can be placed in other layouts — the GUI uses them that way.

A new format is a plug-in of the `EXPORTERS` registry:

```python
from context_synthetic_recognition.export.run import EXPORTERS, ExportContext


@EXPORTERS.register("summary-txt", summary="summary.txt — the key figures")
def export_summary(context: ExportContext) -> list[Path]:
    path = context.folder / "summary.txt"
    rows = context.tables["summary"].rows
    path.write_text("\n".join(f"{name}: {value}" for name, value in rows), encoding="utf-8")
    return [path]
```
