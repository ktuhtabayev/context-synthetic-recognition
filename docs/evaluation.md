# Evaluation

The CS-model is evaluated like the workbook does: training correctness (resubstitution,
Definition 2), generalization (leave-one-out with the **whole pipeline re-fitted per fold**,
Definition 3), k-NN baselines on the same folds, the metrics of the *Accuracy*, *Confusion
Matrix*, *Precision, Recall, F1 Score* and *ROC Curve & AUC* sheets, the margins of the latent
features and the model-property checks.

## Running an experiment

```powershell
.\.venv\Scripts\csr.exe run heart-disease-10                     # template preset, resubstitution + LOO
.\.venv\Scripts\csr.exe run heart-disease-10 --sensitivity       # plus the four switch settings
.\.venv\Scripts\csr.exe run -c configs\heart-disease-270-large-data.yaml
.\.venv\Scripts\csr.exe run heart-disease-10 --export all        # plus workbook, tables, figures, report
```

Every run writes `runs/<YYYYMMDD_HHMMSS>_<hash>/` (unless `--no-save`):

| File | Content |
|---|---|
| `manifest.yaml` | configuration, its hash, the dataset hash, seed, package versions, timings |
| `results.json` | TUPLAM, crit, metrics per protocol and method, margins, properties |
| `predictions.csv` | protocol, method, repetition, object, true class, decision, predicted class, score |
| `folds.csv` | per fold: held-out objects, \|K1\|, \|K2\|, permitted k, r, TUPLAM, undefined |
| `dataset.json` | the dataset exactly as it was used (values, labels, types, names, hash) |

The folder is self-contained: `csr export` repeats the run from `manifest.yaml` and
`dataset.json` and writes the Excel mirror, the tables, the figures and the report —
see [Exporting](exporting.md). `csr run … --export all` does both at once. When the four
switch settings were evaluated, `results.json` keeps them under `sensitivity`.

From Python:

```python
from context_synthetic_recognition.data import load_builtin
from context_synthetic_recognition.services.runner import run_experiment, save_run

result = run_experiment(load_builtin("heart-disease-10"))
loo = result.protocol("leave-one-out")
result.metrics(loo.predictions).accuracy  # 0.2
result.auc(loo.predictions)  # 0.2708…
[b.method for b in loo.baselines]  # k-NN vote ρ, k = 3 …
save_run(result)
```

## Protocols

| Name | Folds | Parameters |
|---|---|---|
| `resubstitution` | the full fit classifies every training row (its context excludes itself) | — |
| `leave-one-out` | m folds; scaling, k range, Ψ(r), ω, η, HAG and meta-algorithm re-fitted on m − 1 objects | — |
| `stratified-k-fold` | classes shuffled with the seed, dealt round robin | `folds` (10), `shuffle` |
| `repeated-k-fold` | stratified k-fold with seeds seed, seed + 1, … | `folds` (10), `repeats` (10) |
| `hold-out` | one stratified split | `test_size` (0.3) |

A fold whose model is undefined — e.g. no permitted k because the smallest class has one object —
refuses its held-out objects and is flagged (ADR-005).

## Metrics

Decisions are 1 (K1), 2 (K2) or 0 (refusal). With the positive class P (class 1 by default,
`evaluation.positive_class`): TP, TN, FP, FN count answered objects only; a refusal is reported
separately and counts as an error in the accuracy (TP + TN)/n. Coverage is the share of answered
objects. Precision is 0 for a class that was never predicted. The AUC is the Mann–Whitney
statistic of score₁ − score₂ rounded to 10 decimals (ADR-008, ADR-031).

## Baselines

`knn-vote` (default): the majority class of the k nearest training objects for every operator and
k (k = 3, 5 by default), with the fold's scaling, rounding and tie rule. With the optional extra
`[sklearn]`: `logistic-regression`, `random-forest`, `svm`, `decision-tree`, `naive-bayes` on
scaled quantitative and one-hot nominal features (ADR-032):

```yaml
evaluation:
  baselines:
    - knn-vote
    - {name: random-forest, params: {options: {n_estimators: 500}}}
```

## Large datasets

!!! warning "The literal k range on large data"
    With k_max = 2·min|Kᵢ| − 3 the largest neighbourhoods cover almost the whole sample. There the
    training-side μ separates the classes trivially, so ω → 1, while the class-free synthetic
    feature (5) collapses to the majority class. On Heart-Disease (270, 13, 2) the HAG then starts
    from constant features and leave-one-out drops to 7 % (ADR-030).

The recommended large-data setting skips constant synthetic features and caps k:

```yaml
k: {name: formula, params: {k_max_cap: 21}}
synthetic:
  skip_constant: true
```

| Heart-Disease 270, leave-one-out | accuracy | AUC |
|---|---|---|
| literal k = 3 … 237 | 7.0 % | 0.141 |
| `skip_constant` + `k_max_cap: 21` (`configs/heart-disease-270-large-data.yaml`) | 81.5 % | 0.862 |
| k-NN vote, best of ρ, ρ_I, ρ_J with k = 3, 5 | 81.1 % | 0.877 |

Both are off by default so that the defaults reproduce the Excel experiment exactly.

## Margins and model properties

`evaluation.margins.margin_analysis` gives, for every latent feature, the boundary b, the width
min K1 − max K2 and the object margins — with the majorizer (r₁ … r_p) and without (the summed
contribution values). `core.properties.model_properties` checks Definition 1/Property 1, counts
the conflicting pairs of Definition 4 and the operator equivalences and boundary ties of
Definition 6; `theorem_check` confirms that a training object left out of its own context gets its
training description and decision.
