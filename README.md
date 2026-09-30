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
| M3 | HAG, meta-algorithm, new-object path, template replication | next |
| M4 | Evaluation (LOO re-fit, baselines, metrics, margins, properties), `csr validate` | |
| M5 | Exporters (Excel mirror, figures, report) | |
| M6 | Desktop GUI (PySide6) | |
| M7 | More metrics, normalizers and datasets | |
| M8 | Documentation, polish, release build | |

## Quick start (Windows PowerShell)

```powershell
git clone https://github.com/ktuhtabayev/context-synthetic-recognition.git
cd context-synthetic-recognition
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Commands available now:

```powershell
.\.venv\Scripts\csr.exe config show                      # the template preset (default)
.\.venv\Scripts\csr.exe config show --preset article     # final centres, one STEP 4 pass
.\.venv\Scripts\csr.exe config init my-experiment.yaml   # start a configuration file
.\.venv\Scripts\csr.exe config check my-experiment.yaml  # validate; hash, preset, deviations, plug-ins
.\.venv\Scripts\csr.exe data list                        # built-in datasets and file formats
.\.venv\Scripts\csr.exe data info heart-disease-270      # sizes, types I/J, |Kᵢ|, permitted k, r
.\.venv\Scripts\csr.exe validate --against "resources\experiments\context-synthetic-model\Context-Synthetic Model – Full Experiment [My Experiment on Heart-Disease (10, 13, 2)] - Opus 5.5.xlsx"
```

`csr validate` compares the package with the workbook's cached cell values (Steps 1–8 so far:
4,022 cells, all within 4.4e-15).

From Python:

```python
from context_synthetic_recognition.core.context import fit_context
from context_synthetic_recognition.data import load_builtin

model = fit_context(load_builtin("heart-disease-10"))  # Steps 1–8, template preset
model.trace.permitted_k.ks  # (3, 5)
model.trace.omegas  # ω = [0.9, 0.9, 0.6, 0.9, 0.9, 1.0]
model.represent(new_objects).values  # Ψ(r) of new objects, without their class
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
  core/             pure numerical core: Steps 1–8, trace objects, plug-in registries
  config/           typed configuration, presets, the two switches
  data/             dataset schema, loaders, built-in datasets
  services/         runs and manifests, validation against the workbook, dataset summaries
  cli/              the csr command
tests/              pytest suite
```

## Documentation

- [Architecture](docs/architecture.md) · [Algorithm reference](docs/algorithm-reference.md) ·
  [Datasets](docs/datasets.md) · [Extending](docs/extending.md) · [Decisions](docs/DECISIONS.md) ·
  [Development](docs/development.md)
- Specification: `resources/experiments/context-synthetic-model/` and
  `docs/handoff/CONTEXT_HANDOFF.md`

The article is in preparation; its draft is not part of this repository yet.

## License

MIT — see [LICENSE](LICENSE).
