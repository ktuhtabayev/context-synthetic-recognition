# Evaluation

The CS-model is evaluated by the article's two definitions: training correctness (resubstitution,
Definition 2) and generalization correctness (Definition 3) — on the test sample of the R-IPS
split, in nested cross-validation with R-IPS inside every fold, or with leave-one-out, k-fold and
hold-out, each with the **whole pipeline re-fitted per fold**. k-NN baselines run on the same
folds; the metrics follow the evaluation sheets of the Excel experiment (*Accuracy*, *Confusion
Matrix*, *Precision, Recall, F1 Score*, *ROC Curve & AUC*); the margins of the additional features
and the model-property checks complete the picture.

## Running an experiment

```powershell
.\.venv\Scripts\csr.exe run heart-disease-10                     # the article's settings: resubstitution + leave-one-out
.\.venv\Scripts\csr.exe run heart-disease-10 -c configs\excel-experiment.yaml --sensitivity   # plus the four switch settings
.\.venv\Scripts\csr.exe run -c configs\heart-disease-270-large-data.yaml
.\.venv\Scripts\csr.exe run heart-disease-10 --export all        # plus workbook, tables, figures, report
```

The protocols are chosen in the configuration. The R-IPS split and nested cross-validation on
Heart with the slope quantitative, as in the article's experiments:

```yaml
schema_version: 2
dataset:
  path: heart-disease-270
  feature_types: {x₁₁: quantitative}
admissible_k: {name: range, params: {start: 3, stop: 41}}
synthetic_features:
  zero_value: true
additional_features:
  alpha: 0.1
evaluation:
  protocols:
    - rips
    - {name: nested-cross-validation, params: {folds: 10}}
```

```powershell
.\.venv\Scripts\csr.exe run -c heart-rips.yaml
```

Every run writes `runs/<YYYYMMDD_HHMMSS>_<hash>/` (unless `--no-save`):

| File | Content |
|---|---|
| `manifest.yaml` | configuration, its hash, the dataset hash, seed, package versions, timings |
| `results.json` | the model (TUPLAM, crit), metrics per protocol and method, margins, properties, the active deviations |
| `predictions.csv` | protocol, method, repetition, object, true class, decision, predicted class, score |
| `folds.csv` | per fold: held-out objects, \|K1\|, \|K2\|, admissible k, r, TUPLAM, undefined |
| `dataset.json` | the dataset exactly as it was used (values, labels, types, names, hash) |

The folder is self-contained: `csr export` repeats the run from `manifest.yaml` and
`dataset.json` and writes the Excel workbook, the tables, the figures and the report —
see [Exporting](exporting.md). `csr run … --export all` does both at once. When the four
switch settings were evaluated, `results.json` keeps them under `sensitivity`.

From Python:

```python
from context_synthetic_recognition.config import preset
from context_synthetic_recognition.data import load_builtin
from context_synthetic_recognition.services.runner import run_experiment, save_run

result = run_experiment(load_builtin("heart-disease-10"), preset("excel-experiment"))
loo = result.protocol("leave-one-out")
result.metrics(loo.predictions).accuracy  # 0.2
result.auc(loo.predictions)  # 0.2708…
[b.method for b in loo.baselines]  # k-NN vote ρ, k = 3 …
save_run(result)
```

## Protocols

| Name | Folds | Parameters |
|---|---|---|
| `resubstitution` | the full fit classifies every training object with its own row (its context excludes itself) — Definition 2 | — |
| `rips` | one fold: R-IPS (configuration section `rips`) splits the sample; the model fitted on the training sample classifies the test sample without its classes — Definition 3 | — |
| `nested-cross-validation` | stratified outer folds; inside each, R-IPS splits the rest into a training and a test sample, every candidate of the grid is fitted on that training sample and scored on that test sample, and the best one classifies the outer fold (ADR-069) | `folds` (10), `shuffle`, `grid` (α = 0.1, 0.2) |
| `leave-one-out` | m folds; normalization, admissible k, Ψ(r), ω, η, the agglomerative algorithm and the meta-algorithm re-fitted on m − 1 objects | — |
| `stratified-k-fold` | classes shuffled with the seed, dealt round robin | `folds` (10), `shuffle` |
| `repeated-k-fold` | stratified k-fold with seeds seed, seed + 1, … | `folds` (10), `repeats` (10) |
| `hold-out` | one stratified split | `test_size` (0.3) |

The R-IPS settings are those of the configuration: the proximity measure, its normalization
(fitted on E₁ = E₀ ∖ T) and features, an optional test sample T named by object ids
(`rips.test_sample`), and the two ⚠ switches `rips.keep_blocked_cover` and `rips.training`
(ADR-061). In nested cross-validation the grid takes dotted configuration paths, e.g.
`{additional_features.alpha: [0.1, 0.2]}`; the inner accuracy counts refusals as errors, the first
candidate wins ties, and a test sample T is not used inside the folds.

A fold whose model is undefined — e.g. no admissible k because the smallest class is too small —
refuses its held-out objects and is flagged (ADR-005).

## Metrics

Decisions are 1 (K1), 2 (K2) or 0 (refusal). With the positive class P (class 1 by default,
`evaluation.positive_class`): TP, TN, FP, FN count answered objects only; a refusal is reported
separately and counts as an error in the accuracy (TP + TN)/n. Coverage is the share of answered
objects, and the accuracy on the answered objects is (TP + TN)/(n − refusals) — for the R-IPS
split these are the quantities of Definition 3 (ADR-069). Precision is 0 for a class that was
never predicted. The AUC is the Mann–Whitney statistic of score₁ − score₂ rounded to 10 decimals
(ADR-008, ADR-031).

## Baselines

`knn-vote` (default): the majority class of the k nearest training objects for every operator and
k (k = 3, 5 by default), with the fold's normalization, rounding and tie rule. With the optional
extra `[sklearn]`: `logistic-regression`, `random-forest`, `svm`, `decision-tree`, `naive-bayes`
on scaled quantitative and one-hot nominal features (ADR-032):

```yaml
evaluation:
  baselines:
    - knn-vote
    - {name: random-forest, params: {options: {n_estimators: 500}}}
```

## Large datasets

!!! warning "The Excel experiment's k range on large data"
    With k_max = 2·min|Kᵢ| − 3 (`twice-min-class`, the `excel-experiment` preset) the largest
    neighbourhoods cover almost the whole sample. There the training-side μ separates the classes
    trivially, so ω → 1, while the class-free synthetic feature (5) collapses to the majority
    class. On Heart-Disease (270, 13, 2) the agglomerative algorithm then starts from constant
    features and leave-one-out drops to 7.0 % (ADR-030).

The article's k ≤ min|Kᵢ| (the default `min-class`) halves the range.
`configs/heart-disease-270-large-data.yaml` adds the large-data correction of ADR-030 to the
article's settings:

```yaml
synthetic_features:
  skip_constant: true
```

| Heart-Disease 270, leave-one-out | accuracy | AUC |
|---|---|---|
| `excel-experiment`: k = 3 … 237 | 7.0 % | 0.141 |
| `heart-disease-270-large-data.yaml`: `skip_constant` on the article's settings | 80.7 % | 0.867 |
| default (the article): k = 3 … 119 | 80.7 % | 0.867 |
| k-NN vote, best of ρ, ρ_I, ρ_J with k = 3, 5 | 81.1 % | 0.877 |

`skip_constant` is off by default. With the article's k range no synthetic feature of
Heart-Disease (270, 13, 2) is constant, so the file gives the default's result. Until ADR-076 it
also capped k at 21 with the Excel experiment's ⚠ running class means and two majorizing passes:
81.5 %, AUC 0.862 (ADR-030).

## Margins and model properties

`evaluation.margins.margin_analysis` gives, for every additional feature, the boundary b, the width
min K1 − max K2 and the object margins — with the majorizing function (d₁ … d_p) and without (the
summed contribution values). `core.properties.model_properties` checks Definition 1/Property 1,
counts the conflicting pairs of Definition 4 and the operator equivalences and boundary ties of
Definition 6; `theorem_check` confirms that a training object left out of its own context gets its
training description and decision.
