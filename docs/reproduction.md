# Reproducing the experiment

The Excel experiment on Heart-Disease (10, 13, 2) is the specification of this package: every
number the workbook computes is recomputed by the software and compared with it. This page shows
how to repeat that comparison, how to obtain each result of the workbook yourself, and how to
repeat any run that was saved.

"Reproduced" means: equal to the workbook's cached cell values within an absolute tolerance of
1e-9 for numbers, and exactly for text, classes and sets. Distances are rounded to 10 decimals and
ties go to the smaller original index, as in the workbook (ADR-008), so the result does not depend
on the machine.

## 1 · Set up

```powershell
git clone https://github.com/ktuhtabayev/context-synthetic-recognition.git
cd context-synthetic-recognition
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The repository contains everything the reproduction needs: the experiment workbook in
`resources\experiments\context-synthetic-model\`, copies of the two template workbooks in
`tests\data\templates\`, and the dataset in `datasets\`.

## 2 · Check the package against the workbook

```powershell
.\.venv\Scripts\csr.exe validate --against "resources\experiments\context-synthetic-model\Context-Synthetic Model – Full Experiment [Heart-Disease (10, 13, 2)] - Opus 5.5.xlsx"
```

The command reads the data, the HAG parameters, both switches and the new object from the
workbook itself, runs the whole pipeline and compares every computed sheet:

```text
  tolerance 1e-09: 11631 cells in 579 checks
  Parameters                         10 cells  max |Δ| 0.0e+00  ✓
  Normalized Dataset                179 cells  max |Δ| 5.0e-16  ✓
  Zhuravlev Distances               300 cells  max |Δ| 0.0e+00  ✓
  …
  Greedy upon Weight (4-Latent)     987 cells  max |Δ| 2.8e-14  ✓
  Meta-algorithm (All Objects)     1190 cells  max |Δ| 3.3e-16  ✓
  Leave-One-Out                     170 cells  max |Δ| 3.3e-11  ✓
  Validation                        167 cells  max |Δ| 3.9e-16  ✓
validation passed
```

All 29 computed sheets pass — Steps 1–12, the evaluation, the switch sensitivity, the model
properties and the workbook's own *Validation* sheet. `--details` lists every one of the 579
checks; the exit code is 1 if any cell differs.

The two template workbooks the experiment follows are checked the same way:

```powershell
.\.venv\Scripts\csr.exe validate --against "tests\data\templates\RegularizedStackingEnsembleWithHAG [Heart-Disease (10, 13, 2)] - Opus 5.5.xlsx"
.\.venv\Scripts\csr.exe validate --against "tests\data\templates\Meta-algorithm [Heart-Disease (10, 13, 2)] - Opus 5.5.xlsx"
```

The first reproduces the grouping of the HAG template on its own data — SET {x₃, x₆, x₁₃, x₄, x₉}
and r₁ … r₄, 6,131 cells; the second its meta-algorithm — the new object is Class 2 with
B1(a₄) = ∅ and B2(a₄) = {S₉, S₁₀}.

## 3 · Obtain the results step by step

### The model: Steps 1–11

```powershell
.\.venv\Scripts\csr.exe fit heart-disease-10
```

```text
Heart-Disease (10, 13, 2)
  objects m = 10; |K1| = 4, |K2| = 6
  k (formula): 3, 5  (2 values; k_max = 2·4 − 3 = 5)
  r = |Ψ(r)| = 6 synthetic features
  HAG: α = 0.3, δ = 0.1, ϰ = 5, cr1₀ = 10, ϕ = sigmoid, centres = running, STEP 4 passes = 2
    ⚠ template calculation (ADR-002): θ and γ are measured from running partial class means instead of the final class means M₁ and M₂
    ⚠ template calculation (ADR-003): in STEP 4 the majorizer is applied twice instead of once
    STEP 2: u = a₆ (weight 1)
    iteration 1: q = a₃, crit = θ/γ = 0.428155786666 (of 5 candidates)
    iteration 2: q = a₁, crit = θ/γ = 0.305276758463 (of 4 candidates)
    iteration 3: q = a₂, crit = θ/γ = 0.27336281327 (of 3 candidates)
    iteration 4: q = a₄, crit = θ/γ = 0.271025654647 (of 2 candidates)
    stop: |TUPLAM| = ϰ
  TUPLAM = {a₆, a₃, a₁, a₂, a₄}; p = 4 latent features
  training objects (Definition 2): 7 of 10 classified correctly, 0 refusals
```

### A new object: Step 12 and the Theorem

A training object left out of its own context, and a new object typed as 13 values — neither
command takes a class:

```powershell
.\.venv\Scripts\csr.exe classify heart-disease-10 --object 1
.\.venv\Scripts\csr.exe classify heart-disease-10 --values "58 1 3 125 250 0 2 150 0 1 2 1 7"
```

```text
Heart-Disease (10, 13, 2): S₁ (left out of its own context)
  Ψ(r) by formula (5): a₁ = 2, a₂ = 2, a₃ = 1, a₄ = 2, a₅ = 2, a₆ = 2
  description (a₀, …, a_p) = (2, 1, 2, 2, 2)
  Step 1  j = 0: B1(a₀) = {S₂, S₄, S₅, S₆}   B2(a₀) = {S₁, S₇, S₈, S₉, S₁₀}
  …
  Step 4: |B1|/|K1| = 2/4 = 0.5, |B2|/|K2| = 0/6 = 0  →  K1 (class 1)
```

The second object ends with |B1|/|K1| = 0.5 and |B2|/|K2| = 4/6, class K2.

### The evaluation and the two switches

```powershell
.\.venv\Scripts\csr.exe run heart-disease-10 --sensitivity --export all
```

| Result | Value |
|---|---|
| Permitted k | 3, 5 (k_max = 2·4 − 3 = 5); r = 6 |
| TUPLAM | {a₆, a₃, a₁, a₂, a₄}, p = 4 |
| crit per iteration | 0.428155786666, 0.305276758463, 0.27336281327, 0.271025654647 |
| Resubstitution | 7 of 10 (70 %), no refusal, AUC 0.667 |
| Leave-one-out, pipeline re-fitted per fold | 2 of 10 (20 %), 4 refusals, AUC 0.271 |
| Best k-NN vote baseline (ρ_I, k = 3) | 70 % in both protocols, AUC 0.646 |
| Margin widths r₁ … r₄ with the majorizer | 0.1482, 0.6496, 1.1201, 1.5743 (−0.4 without it) |

The four settings of the two template-vs-article switches (sheet *Sensitivity (Switches)*):

| Class centres | STEP 4 passes | TUPLAM | Resubstitution | Leave-one-out | AUC (resub. / LOO) |
|---|---|---|---|---|---|
| running (template) | 2 (template) | {a₆, a₃, a₁, a₂, a₄} | 70 % | 20 % | 0.667 / 0.271 |
| running | 1 | {a₆, a₃, a₁, a₂, a₄} | 70 % | 10 % | 0.667 / 0.104 |
| final | 2 | {a₆, a₁, a₂, a₄, a₅} | 50 % | 10 % | 0.583 / 0.250 |
| final (article) | 1 (article) | {a₆, a₁, a₂, a₄, a₅} | 50 % | 0 % | 0.583 / 0.083 |

!!! warning "The default follows the templates, not the article"
    The first row is the default configuration: it reproduces the template workbooks cell for
    cell. The last row is the article's calculation; run it with
    `csr run heart-disease-10 -c configs\article.yaml` (ADR-002 – ADR-004).

### The workbook itself

`--export all` writes the run folder with `workbook.xlsx` — a workbook with the sheets, layout and
style of the experiment, every cell a value computed by the package. It passes the same
comparison as the original:

```powershell
.\.venv\Scripts\csr.exe validate --against "runs\<run id>\workbook.xlsx"
```

Open the two workbooks side by side to compare any table by eye; the other exports of the folder
(CSV, JSON, Markdown and LaTeX tables, figures, the HTML and PDF report) are described in
[Exporting](exporting.md).

### From Python

```python
from context_synthetic_recognition.core.model import fit_model
from context_synthetic_recognition.data import load_dataset
from context_synthetic_recognition.services.runner import run_experiment

workbook = (
    "resources/experiments/context-synthetic-model/"
    "Context-Synthetic Model – Full Experiment [Heart-Disease (10, 13, 2)] - Opus 5.5.xlsx"
)
data = load_dataset(workbook)  # the Dataset sheet
model = fit_model(data)  # Steps 1–11, template preset
model.trace.permitted_k.ks  # (3, 5)
model.hag.label  # '{a₆, a₃, a₁, a₂, a₄}'

result = run_experiment(data)
loo = result.protocol("leave-one-out")
result.metrics(loo.predictions).accuracy  # 0.2
round(result.auc(loo.predictions), 3)  # 0.271
```

Every intermediate table of the workbook is in `model.trace` (Steps 1–8), `model.hag`
(Step 9) and `model.meta_dataset` (Step 10) — see the [algorithm reference](algorithm-reference.md)
for the module behind each sheet.

### In the desktop application

`csr gui` opens the default dataset, Heart-Disease (10, 13, 2). **F5** runs the experiment with
the template preset; the *Results* page then shows the workbook's steps as tables and figures,
*New object* repeats the classification above, and *Export* writes the same run folder
([Desktop application](gui.md)).

## 4 · Repeat a saved run

A run folder is self-contained. `manifest.yaml` holds the complete configuration, its hash, the
hash of the dataset, the seed, the Python and package versions; `dataset.json` holds the data
exactly as it was used.

```powershell
.\.venv\Scripts\csr.exe export "runs\<run id>" --format json --out repeated
```

`csr export` repeats the run from those two files and writes it again. The computation is
deterministic: every table of the repeated `trace.json` is identical to the first one — only its
`created` time stamp differs. The command first checks that the dataset is unchanged (its hash)
and warns if the repeated results differ from the stored `results.json`. To repeat a run on
another machine, copy its folder; to repeat it with another version of the package, compare the
versions recorded in the manifest.

The run id ends with the first eight characters of the configuration hash
(`20261002_014445_5a0c7b92` — `5a0c7b92…` is the template preset), so runs of the same
configuration are recognised at a glance; *Compare* in the desktop application puts saved runs
side by side.

## 5 · The larger dataset

Heart-Disease (270, 13, 2) shows what the literal k range does on larger data (ADR-030):

```powershell
.\.venv\Scripts\csr.exe run heart-disease-270 --no-save                              # literal k = 3 … 237
.\.venv\Scripts\csr.exe run -c configs\heart-disease-270-large-data.yaml --no-save   # constant features skipped, k ≤ 21
```

| Heart-Disease 270, leave-one-out | accuracy | AUC |
|---|---|---|
| literal k = 3 … 237 | 7.0 % | 0.141 |
| `skip_constant` + `k_max_cap: 21` | 81.5 % | 0.862 |
| k-NN vote, best of ρ, ρ_I, ρ_J with k = 3, 5 | 81.1 % | 0.877 |

The first command is slow: leave-one-out re-fits a model with about 350 synthetic features 270
times.

## 6 · The acceptance tests

The comparison with the workbook is also part of the test suite, so it runs on every change:

```powershell
.\.venv\Scripts\python.exe -m pytest -m golden        # the workbook, the templates, the reference engine
.\.venv\Scripts\python.exe -m pytest --cov            # everything, with coverage
```

The golden tests check every sheet of the experiment, the two template workbooks, the Excel
mirror of the run, and the dependency-free reference engine in `docs\handoff\reference-engine`
with its `golden_values.json`.

## If a value differs

- **A changed setting.** `csr validate` uses the template preset with the workbook's own
  parameters. A configuration passed with `--config` that changes the method — another
  normalizer or metric, `synthetic.skip_constant`, the article preset — gives other numbers by
  design.
- **A changed workbook.** The comparison reads the *cached* values of the cells. A workbook that
  was edited and saved without recalculation, or recalculated by a program that rounds
  differently, no longer holds the values of the specification.
- **Another version.** The manifest of a run records the versions of Python, numpy and the
  package. Differences below 1e-9 between versions are within the tolerance; anything larger is a
  defect — please report it with the run folder.
