# Reproducing the Excel experiment

The author's Excel experiment on Heart-Disease (10, 13, 2) computes the method step by step in a
workbook. The article is the specification of this package; the workbook is an acceptance test:
with the workbook's own settings — the `excel-experiment` preset — every number it computes is
recomputed by the software and compared with it. This page shows how to repeat that comparison,
how to obtain each result yourself, and how to repeat any run that was saved. The article's own
experiment on the larger datasets is described in [Computational experiment](experiments.md).

"Reproduced" means: equal to the workbook's cached cell values within an absolute tolerance of
1e-9 for numbers, and exactly for text, classes and sets. Distances are rounded to 10 decimals and
ties go to the smaller original index, as in the workbook (ADR-008), so the result does not depend
on the machine.

!!! warning "The Excel experiment deviates from the article in two places"
    It measures θ and γ from running partial class means instead of the final class means M₁ and
    M₂ (ADR-002), and applies the majorizing function twice in Step 4 instead of once (ADR-003).
    It also takes odd k = 3 … 2·min|Kᵢ| − 3 instead of the article's k ≤ min|Kᵢ|. The
    `excel-experiment` preset has these settings; the default configuration follows the article.

## 1 · Set up

```powershell
git clone https://github.com/ktuhtabayev/context-synthetic-recognition.git
cd context-synthetic-recognition
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The repository contains everything the reproduction needs: the author's workbooks in
`tests\data\workbooks\` (byte-for-byte copies; a test pins their checksums) and the datasets in
`datasets\`.

| Workbook | Role |
|---|---|
| `Excel experiment [Heart-Disease (10, 13, 2)].xlsx` | the whole method and its evaluation, sheet by sheet |
| `Additional features [Heart-Disease (10, 13, 2)].xlsx` | check workbook: the agglomerative algorithm on its own data |
| `Meta-algorithm [Heart-Disease (10, 13, 2)].xlsx` | check workbook: the meta-algorithm on a new object |
| `Synthetic features by k-NN [Heart-Disease (10, 13, 2)].xlsx` | check workbook: synthetic features from the k nearest neighbours; read by the tests as a dataset in the experiment's layout, not compared by `csr validate` |

## 2 · Check the package against the workbook

```powershell
.\.venv\Scripts\csr.exe validate --against "tests\data\workbooks\Excel experiment [Heart-Disease (10, 13, 2)].xlsx"
```

The command reads the data, the parameters of the agglomerative algorithm, both switches and the
new object from the workbook itself, runs the whole pipeline with the `excel-experiment` preset and
compares every computed sheet:

```text
  agglomerative algorithm settings of the workbook (Parameters sheet): α = 0.3, δ = 0.1, ϰ = 5, cr1₀ = 10, class means = running, majorizing passes = 2
  ⚠ deviates from the article (ADR-002): θ and γ are measured from running partial class means instead of the final class means M₁ and M₂
  ⚠ deviates from the article (ADR-003): in Step 4 the majorizing function is applied twice instead of once
  tolerance 1e-09: 11622 cells in 579 checks
  Parameters                         10 cells  max |Δ| 0.0e+00  ✓
  …
  Normalized Dataset                179 cells  max |Δ| 5.0e-16  ✓
  …
  Meta-algorithm (All Objects)     1190 cells  max |Δ| 3.3e-16  ✓
  …
  Leave-One-Out                     170 cells  max |Δ| 3.3e-11  ✓
  Sensitivity (Switches)             32 cells  max |Δ| 0.0e+00  ✓
  Model Properties                  182 cells  max |Δ| 0.0e+00  ✓
  Validation                        166 cells  max |Δ| 3.9e-16  ✓
validation passed
```

All 29 computed sheets pass — the steps of the method, the evaluation, the switch sensitivity, the
model properties and the workbook's own *Validation* sheet. `--details` lists every one of the 579
checks; the exit code is 1 if any cell differs.

The two check workbooks are compared the same way:

```powershell
.\.venv\Scripts\csr.exe validate --against "tests\data\workbooks\Additional features [Heart-Disease (10, 13, 2)].xlsx"
.\.venv\Scripts\csr.exe validate --against "tests\data\workbooks\Meta-algorithm [Heart-Disease (10, 13, 2)].xlsx"
```

The first reproduces the agglomerative algorithm on its own data — SET = {x₃, x₆, x₁₃, x₄, x₉},
stop at |TUPLAM| = ϰ, and the additional features: 6,131 cells in 117 checks. The second
reproduces the meta-algorithm — the new object is Class 2 with B1(a₄) = ∅ and
B2(a₄) = {S₉, S₁₀}: 14 cells in 6 checks.

## 3 · Obtain the results step by step

### The model

```powershell
.\.venv\Scripts\csr.exe fit heart-disease-10 -c configs\excel-experiment.yaml
```

```text
Heart-Disease (10, 13, 2)
  objects m = 10; |K1| = 4, |K2| = 6
  k (twice-min-class): 3, 5  (2 values; k_max = 2·4 − 3 = 5)
  r = |Ψ(r)| = 6 synthetic features
  agglomerative algorithm: α = 0.3, δ = 0.1, ϰ = 5, cr1₀ = 10, φ = sigmoid, class means = running, majorizing passes = 2
    ⚠ deviates from the article (ADR-002): θ and γ are measured from running partial class means instead of the final class means M₁ and M₂
    ⚠ deviates from the article (ADR-003): in Step 4 the majorizing function is applied twice instead of once
    …
    iteration 1: q = a₃, crit = θ/γ = 0.428155786666 (of 5 candidates)
    iteration 2: q = a₁, crit = θ/γ = 0.305276758463 (of 4 candidates)
    iteration 3: q = a₂, crit = θ/γ = 0.27336281327 (of 3 candidates)
    iteration 4: q = a₄, crit = θ/γ = 0.271025654647 (of 2 candidates)
    stop: |TUPLAM| = ϰ
  TUPLAM = {a₆, a₃, a₁, a₂, a₄}; p = 4 additional features
  training objects (Definition 2): 7 of 10 classified correctly, 0 refusals
```

Without `-c` the command uses the default configuration, the article's: k = 3 only (k ≤ min|Kᵢ|
= 4), r = 3, TUPLAM = {a₁, a₃, a₂}, 5 of 10 training objects classified correctly.

### A new object and the Theorem

A training object left out of its own context, and a new object typed as 13 values — neither
command takes a class:

```powershell
.\.venv\Scripts\csr.exe classify heart-disease-10 --object 1 -c configs\excel-experiment.yaml
.\.venv\Scripts\csr.exe classify heart-disease-10 --values "58 1 3 125 250 0 2 150 0 1 2 1 7" -c configs\excel-experiment.yaml
```

```text
Heart-Disease (10, 13, 2): S₁ (left out of its own context)
  Ψ(r) by formula (5): a₁ = 2, a₂ = 2, a₃ = 1, a₄ = 2, a₅ = 2, a₆ = 2
  TUPLAM = {a₆, a₃, a₁, a₂, a₄} → positions a₀ … a₄
  description (a₀, …, a_p) = (2, 1, 2, 2, 2)
  Step 1  j = 0: B1(a₀) = {S₂, S₄, S₅, S₆}   B2(a₀) = {S₁, S₇, S₈, S₉, S₁₀}
  …
  Step 4: |B1|/|K1| = 2/4 = 0.5, |B2|/|K2| = 0/6 = 0  →  K1 (class 1)
```

The second object ends with |B1|/|K1| = 0.5 and |B2|/|K2| = 4/6, class K2.

### The evaluation and the two switches

```powershell
.\.venv\Scripts\csr.exe run heart-disease-10 -c configs\excel-experiment.yaml --sensitivity --export all
```

| Result | Value |
|---|---|
| Admissible k | 3, 5 (k_max = 2·4 − 3 = 5); r = 6 |
| TUPLAM | {a₆, a₃, a₁, a₂, a₄}, p = 4 |
| crit per iteration | 0.428155786666, 0.305276758463, 0.27336281327, 0.271025654647 |
| Resubstitution | 7 of 10 (70 %), no refusal, AUC 0.667 |
| Leave-one-out, pipeline re-fitted per fold | 2 of 10 (20 %), 4 refusals, AUC 0.271 |
| Best k-NN vote baseline (ρ_I, k = 3) | 70 % in both protocols, AUC 0.646 |
| Margin widths of d₁ … d₄ with the majorizing function | 0.1482, 0.6496, 1.1201, 1.5743 (−0.4 without it) |

The four settings of the two switches of the agglomerative algorithm (sheet *Sensitivity
(Switches)*), all with k = 3, 5:

| Class means | Majorizing passes | TUPLAM | Resubstitution | Leave-one-out | AUC (resub. / LOO) |
|---|---|---|---|---|---|
| ⚠ running (Excel experiment) | ⚠ 2 (Excel experiment) | {a₆, a₃, a₁, a₂, a₄} | 70 % | 20 % | 0.667 / 0.271 |
| ⚠ running | 1 | {a₆, a₃, a₁, a₂, a₄} | 70 % | 10 % | 0.667 / 0.104 |
| final | ⚠ 2 | {a₆, a₁, a₂, a₄, a₅} | 50 % | 10 % | 0.583 / 0.250 |
| final (article) | 1 (article) | {a₆, a₁, a₂, a₄, a₅} | 50 % | 0 % | 0.583 / 0.083 |

The default configuration also changes the k range: `csr run heart-disease-10` takes k = 3 only
and gives TUPLAM = {a₁, a₃, a₂}, resubstitution 50 %, leave-one-out 0 % with 5 refusals, AUC
0.583 / 0.083.

### The workbook itself

`--export all` writes the run folder with `workbook.xlsx` — a workbook in the layout and style of
the Excel experiment, every cell a value computed by the package. It passes the same comparison as
the original, 11,622 cells in 579 checks:

```powershell
.\.venv\Scripts\csr.exe validate --against "runs\<run id>\workbook.xlsx"
```

Open the two workbooks side by side to compare any table by eye; the other exports of the folder
(CSV, JSON, Markdown and LaTeX tables, figures, the HTML and PDF report) are described in
[Exporting](exporting.md).

### From Python

```python
from context_synthetic_recognition.config import preset
from context_synthetic_recognition.core.model import fit_model
from context_synthetic_recognition.data import load_dataset
from context_synthetic_recognition.services.runner import run_experiment

workbook = "tests/data/workbooks/Excel experiment [Heart-Disease (10, 13, 2)].xlsx"
data = load_dataset(workbook)  # the Dataset sheet
config = preset("excel-experiment")
model = fit_model(data, config)
model.trace.admissible_k.ks  # (3, 5)
model.agglomerative.label  # '{a₆, a₃, a₁, a₂, a₄}'

result = run_experiment(data, config)
loo = result.protocol("leave-one-out")
result.metrics(loo.predictions).accuracy  # 0.2
round(result.auc(loo.predictions), 3)  # 0.271
```

Every intermediate table of the workbook is in `model.trace` (local contexts and synthetic
features), `model.agglomerative` (the agglomerative algorithm) and `model.meta_table` (the
meta-description Y) — see the [algorithm reference](algorithm-reference.md) for the module behind
each sheet.

### In the desktop application

`csr gui` opens the default dataset, Heart-Disease (10, 13, 2). Choose the `excel-experiment`
preset on the *Configure* page (or **Edit → Apply a preset**) and press **F5**; the *Results* page
then shows the workbook's steps as tables and figures, *New object* repeats the classification
above, and *Export* writes the same run folder ([Desktop application](gui.md)).

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
(`20261122_040221_36a214d4` — `36a214d4…` is the `excel-experiment` preset, `babf263b…` the
default), so runs of the same configuration are recognised at a glance; *Compare* in the desktop
application puts saved runs side by side.

## 5 · The larger dataset

Heart-Disease (270, 13, 2) shows what the Excel experiment's k range does on larger data
(ADR-030):

```powershell
.\.venv\Scripts\csr.exe run heart-disease-270 -c configs\excel-experiment.yaml --no-save   # k = 3 … 237
.\.venv\Scripts\csr.exe run -c configs\heart-disease-270-large-data.yaml --no-save         # constant features skipped, the article's k
.\.venv\Scripts\csr.exe run heart-disease-270 --no-save                                    # the article: k = 3 … 119
```

| Heart-Disease 270, leave-one-out | accuracy | AUC |
|---|---|---|
| `excel-experiment`: k = 3 … 237 | 7.0 % | 0.141 |
| `heart-disease-270-large-data.yaml`: `skip_constant` on the article's settings | 80.7 % | 0.867 |
| default (the article): k = 3 … 119 | 80.7 % | 0.867 |
| k-NN vote, best of ρ, ρ_I, ρ_J with k = 3, 5 (ρ, k = 5) | 81.1 % | 0.877 |

Leave-one-out re-fits the model 270 times; the first command, with 354 synthetic features, is the
slowest.

## 6 · The acceptance tests

The comparison with the workbooks is also part of the test suite, so it runs on every change:

```powershell
.\.venv\Scripts\python.exe -m pytest -m golden        # the workbooks, the exported workbook, the independent re-implementation
.\.venv\Scripts\python.exe -m pytest --cov            # everything, with coverage
```

The golden tests check every sheet of the experiment workbook, the two check workbooks, the
exported workbook of a run, and an independent re-implementation used as a test oracle in
`tests\golden\independent`, together with `tests\golden\golden_values.json`.

## If a value differs

- **A changed setting.** `csr validate` uses the `excel-experiment` preset with the workbook's own
  parameters. A configuration passed with `--config` that changes the method — another normalizer
  or metric, `synthetic_features.skip_constant`, the default (article) configuration — gives other
  numbers by design.
- **A changed workbook.** The comparison reads the *cached* values of the cells. A workbook that
  was edited and saved without recalculation, or recalculated by a program that rounds
  differently, no longer holds the values of the specification.
- **Another version.** The manifest of a run records the versions of Python, numpy and the
  package. Differences below 1e-9 between versions are within the tolerance; anything larger is a
  defect — please report it with the run folder.
