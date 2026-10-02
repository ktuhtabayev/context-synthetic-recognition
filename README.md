# Context-synthetic recognition

Python implementation of the **context-synthetic model of recognition algorithms (CS-model)** —
library, `csr` command line and desktop GUI — built on the author's complete Excel experiment
on Heart-Disease (10, 13, 2), which serves as the executable specification.

```text
(S, E) → Ψ_ρ,k(S, E) → D(S, E) → Y(S, E) → R(Y(S, E))
```

Base operators (the Zhuravlyov metric on all, quantitative and nominal features) turn each object's
k-nearest-neighbour context into class-free synthetic features Ψ(r) (formula (5)); membership,
stability, informativeness and contributions (formulas (1)–(6)) weight them; the hierarchical
agglomerative grouping (HAG) builds latent features; the meta-algorithm classifies — a new object
without its class (Theorem of the article).

> [!WARNING]
> **Two template calculations differ from the article.**
> - θ and γ are measured from running partial class means instead of the final class means M₁ and M₂.
> - In STEP 4 the majorizer is applied twice instead of once.
>
> Both are configuration switches (`hag.centres`, `hag.step4_passes`). The default reproduces the
> template workbooks exactly; the `article` preset follows the article. See
> [docs/DECISIONS.md](docs/DECISIONS.md) (ADR-002 – ADR-004).

## Status

| Milestone | Content | State |
|---|---|---|
| M0 | Analysis and plan | ✓ |
| M1 | Skeleton: packaging, tooling, CI, configuration, logging, docs | ✓ |
| M2 | Data layer; core pipeline up to Ψ(r) and formulas (1)–(6); golden tests Steps 1–8 | ✓ |
| M3 | HAG, meta-algorithm, new-object path, template replication | ✓ |
| M4 | Evaluation (LOO re-fit, baselines, metrics, margins, properties), `csr validate` | ✓ |
| M5 | Exporters: Excel mirror, CSV/JSON, Markdown/LaTeX tables, figures, HTML/PDF report | ✓ |
| M6 | Desktop GUI (PySide6): dataset, configure, run, results explorer, new object, compare, export | ✓ |
| M7 | More metrics (HEOM, Gower, Euclidean, Mahalanobis, …) and normalizers (z-score, robust, rank, …) | ✓ |
| M8 | Reproduction guide, Russian and Uzbek interface, Windows executable, documentation site | ✓ |

Documentation: <https://ktuhtabayev.github.io/context-synthetic-recognition/>

## The desktop application without Python

Download `csr-gui-<version>-windows.zip` from the
[latest release](https://github.com/ktuhtabayev/context-synthetic-recognition/releases/latest),
unpack it anywhere and start `csr-gui.exe`. Nothing is installed; the Heart-Disease datasets are
built in. The interface is available in English, Russian and Uzbek (**View → Language**).

## Quick start (Windows PowerShell)

The project is developed, tested and supported on Windows only (ADR-049).

```powershell
git clone https://github.com/ktuhtabayev/context-synthetic-recognition.git
cd context-synthetic-recognition
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Commands:

```powershell
.\.venv\Scripts\csr.exe config show                      # the template preset (default)
.\.venv\Scripts\csr.exe config show --preset article     # final centres, one STEP 4 pass
.\.venv\Scripts\csr.exe config init my-experiment.yaml   # start a configuration file
.\.venv\Scripts\csr.exe config check my-experiment.yaml  # validate; hash, preset, deviations, plug-ins
.\.venv\Scripts\csr.exe data list                        # built-in datasets and file formats
.\.venv\Scripts\csr.exe data info heart-disease-270      # sizes, types I/J, |Kᵢ|, permitted k, r
.\.venv\Scripts\csr.exe fit heart-disease-10               # Ψ(r), the HAG step by step, TUPLAM
.\.venv\Scripts\csr.exe classify heart-disease-10 --object 1                         # S₁ left out of its context
.\.venv\Scripts\csr.exe classify heart-disease-10 --values "58 1 3 125 250 0 2 150 0 1 2 1 7"  # a new object
.\.venv\Scripts\csr.exe validate --against "resources\experiments\context-synthetic-model\Context-Synthetic Model – Full Experiment [Heart-Disease (10, 13, 2)] - Opus 5.5.xlsx"
.\.venv\Scripts\csr.exe validate --against "tests\data\templates\RegularizedStackingEnsembleWithHAG [Heart-Disease (10, 13, 2)] - Opus 5.5.xlsx"
.\.venv\Scripts\csr.exe run heart-disease-10 --sensitivity  # resubstitution, LOO, k-NN baselines, margins, switches
.\.venv\Scripts\csr.exe run -c configs\heart-disease-270-large-data.yaml   # the large-data setting (ADR-030)
.\.venv\Scripts\csr.exe run heart-disease-10 --export all  # … and export: workbook, tables, figures, report
.\.venv\Scripts\csr.exe export -f excel,html               # export the latest run again (repeated from its folder)
.\.venv\Scripts\csr.exe gui                                # the desktop application
```

`csr validate` compares the package with a workbook's cached cell values, reading the inputs (data,
HAG parameters and both switches, the new object) from the workbook itself: every computed sheet
of the experiment (11,631 cells in 29 sheets, Steps 1–12 and the evaluation), the HAG template
(SET {x₃, x₆, x₁₃, x₄, x₉}, 6,131 cells) and the meta-algorithm template (Class 2, B1(a₄) = ∅,
B2(a₄) = {S₉, S₁₀}). `csr run` evaluates the model and writes a run folder
([Evaluation](docs/evaluation.md)).

`csr export` writes a run as `workbook.xlsx` — a mirror of the Excel experiment with the same
sheets, layout and style, generated for any data; for Heart-Disease (10, 13, 2) it passes
`csr validate` cell for cell — as CSV/JSON, as Markdown and LaTeX tables, as PNG/SVG figures and
as an HTML/PDF report ([Exporting](docs/exporting.md)).

`csr gui` starts the desktop application: load a dataset, configure the method (the two ⚠
switches are always in sight), run in the background, explore every table and figure of the
workbook with cross-highlighting, classify a new object with a step-by-step explanation,
compare runs and export — in English, Russian or Uzbek ([Desktop application](docs/gui.md)).

To repeat the Excel experiment yourself — the workbook checked cell for cell, every result step
by step, any saved run again — follow [Reproducing the experiment](docs/reproduction.md).

![The results explorer](docs/images/gui/results-table.png)

The Zhuravlyov metric with the fractional-linear (min–max) transform is the default and
reproduces the workbook. Other metrics — weighted Zhuravlyov, HEOM, Gower, Manhattan, Euclidean,
Chebyshev, Minkowski, Canberra, cosine, Mahalanobis, Hamming — and other normalizers — z-score,
robust, max-abs, decimal scaling, rank, unit length — are chosen in the configuration or on the
Configure page ([Algorithm reference](docs/algorithm-reference.md#normalizers-and-metrics),
[Extending](docs/extending.md)). No metric reads a class label.

> [!NOTE]
> On large datasets the literal k range (k_max = 2·min|Kᵢ| − 3) reaches near-global
> neighbourhoods where ω is inflated; use `configs/heart-disease-270-large-data.yaml`
> (constant synthetic features skipped, k ≤ 21) as the starting point — see ADR-030.

From Python:

```python
from context_synthetic_recognition.core.model import fit_model
from context_synthetic_recognition.data import load_builtin

model = fit_model(load_builtin("heart-disease-10"))  # Steps 1–11, template preset
model.trace.permitted_k.ks  # (3, 5)
model.hag.label  # TUPLAM = {a₆, a₃, a₁, a₂, a₄}
model.meta_dataset.Y  # Y = (y₀ … y₄, r₁ … r₄)
result = model.classify(new_objects)  # no label argument (Theorem)
result.decisions  # 1 = K1, 2 = K2, 0 = refusal
result.meta.steps(0).b1(4)  # B1(a₄) of the first object

from context_synthetic_recognition.services.runner import run_experiment

evaluation = run_experiment(load_builtin("heart-disease-10"))  # resubstitution + leave-one-out
evaluation.metrics(evaluation.protocol("leave-one-out").predictions).accuracy  # 0.2
```

Tests and checks:

```powershell
.\.venv\Scripts\python.exe -m pytest --cov
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\mypy.exe
```

## Layout

```text
configs/            preset configurations (template.yaml, article.yaml)
docs/               documentation (mkdocs), DECISIONS.md; handoff/ = design material (read-only)
resources/          the Excel experiments (specification, read-only)
src/context_synthetic_recognition/
  core/             pure numerical core: Steps 1–12 (context, HAG, meta-algorithm, model),
                    model properties, trace objects, plug-in registries
  evaluation/       protocols (LOO re-fit, k-fold, hold-out), baselines, metrics, ROC, margins
  config/           typed configuration, presets, the two switches
  data/             dataset schema, loaders, built-in datasets
  services/         experiment runner, run folders, validation against the workbooks,
                    switch sensitivity, dataset summaries
  export/           Excel mirror, tables (CSV, JSON, Markdown, LaTeX), figures, report
  gui/              desktop application (PySide6): pages, shared state, worker threads
  cli/              the csr command
tests/              pytest suite; tests/data/templates = copies of the template workbooks
```

## Documentation

- [Architecture](docs/architecture.md) · [Algorithm reference](docs/algorithm-reference.md) ·
  [Reproducing the experiment](docs/reproduction.md) · [Datasets](docs/datasets.md) · [Evaluation](docs/evaluation.md) · [Exporting](docs/exporting.md) ·
  [Desktop application](docs/gui.md) · [Extending](docs/extending.md) ·
  [Decisions](docs/DECISIONS.md) · [Development](docs/development.md)
- Specification: `resources/experiments/context-synthetic-model/` and
  `docs/handoff/CONTEXT_HANDOFF.md`

The article is in preparation; its draft is not part of this repository yet.

## License

MIT — see [LICENSE](LICENSE).
