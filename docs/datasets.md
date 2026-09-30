# Datasets

A dataset is a classified sample E₀ = {S₁ … Sₘ} with heterogeneous features: quantitative
(set I) and nominal (set J). The model is binary: exactly two classes K1 and K2 (ADR-010).

## Formats

| Format | Files | Layout |
|---|---|---|
| `cs-workbook` | `.xlsx` | The *Dataset* sheet of the Excel experiment: a header row with `Class` above the class column, one row per object, then the feature-type row (1 = quantitative, 0 = nominal). |
| `template-extended` | `.csv`, `.dat` | The template project's layout: first row `m, n, c`; m rows of n values and the class label; last row of n type flags. `.dat` is whitespace-separated and may use decimal commas. |
| `csv` | `.csv`, `.txt` | A table with a header row; the delimiter (`,` `;` tab `\|`) is detected. |
| `xlsx` | `.xlsx` | A table with a header row (first sheet by default). |
| `parquet` | `.parquet` | A table; needs `pip install "context-synthetic-recognition[parquet]"`. |

`format: auto` (the default) picks the format from the extension and the content: an Excel file
with a *Dataset* sheet in the experiment's layout is `cs-workbook`, a CSV whose first row is
`m, n, c` is `template-extended`.

For tables with a header row, the class column is the last column unless `class_column` names
another; `id_column` names an optional column of object ids. Feature types come from
`feature_types` in the configuration; features not listed there are inferred — numeric columns
as quantitative, the others as nominal — and a warning lists the inferred types. Nominal text
values become category codes (alphabetical order of the categories).

```yaml
dataset:
  path: data/patients.csv        # relative to the configuration file
  format: auto
  class_column: outcome
  id_column: patient
  feature_types:
    age: quantitative
    chest_pain: nominal
```

Missing values are rejected with an error that names their positions.

## Classes

Class labels may be integers or text. They are sorted — numbers numerically, text alphabetically
— and the first class is **K1**, the second **K2** (ADR-018). For the author's data (labels 1 and
2) this is the article's numbering. Class 1 is also the positive class of the evaluation.

## Built-in datasets

| Name | Content |
|---|---|
| `heart-disease-10` | Heart-Disease (10, 13, 2): the 10 objects of the Excel experiment (\|K1\| = 4, \|K2\| = 6) |
| `heart-disease-270` | Heart-Disease (270, 13, 2): UCI Statlog (Heart), 6 quantitative + 7 nominal features, \|K1\| = 150, \|K2\| = 120 |

```python
from context_synthetic_recognition.data import load_builtin, load_dataset

heart = load_builtin("heart-disease-270")
experiment = load_dataset(
    "resources/experiments/context-synthetic-model/Context-Synthetic Model – Full Experiment [My Experiment on Heart-Disease (10, 13, 2)] - Opus 5.5.xlsx"
)
```

## Inspecting a dataset

`csr data info` shows what the pipeline will make of a dataset under a configuration:

```text
> csr data info heart-disease-270
Heart-Disease (270, 13, 2)
  source:    built-in dataset 'heart-disease-270'
  objects:   m = 270
  features:  n = 13
    I (quantitative) 6: x₁, x₄, x₅, x₈, x₁₀, x₁₂
    J (nominal)      7: x₂, x₃, x₆, x₇, x₉, x₁₁, x₁₃
  classes:   K1 = 1: 150, K2 = 2: 120
  hash:      bf78a86ebd7478a48a0a013f28765be5145e451f84d84e92a0c2dcd49e315456
  k (formula): 3, 5, 7, …, 237  (118 values; k_max = 2·120 − 3 = 237)
  operators: ρ (13 features), ρ_I (6 features), ρ_J (7 features)
  r = |Ψ(r)| = 354 synthetic features
```

Options: `--format`, `--sheet`, `--class-column`, `--id-column` and `--config` (for another k
strategy or other operators). An operator whose feature subset is empty — ρ_J on a dataset without
nominal features — is skipped and reported (ADR-007).

The **content hash** identifies the data in run manifests; it covers values, labels, types,
feature names and object ids, not the file name, so the workbook's Dataset sheet and the built-in
CSV of the same 10 objects have the same hash.
