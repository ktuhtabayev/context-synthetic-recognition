# Datasets

A dataset is a classified sample E₀ = {S₁ … Sₘ} with heterogeneous features: quantitative
(set I) and nominal (set J). The model is binary: exactly two classes K1 and K2 (ADR-010).

## Formats

| Format | Files | Layout |
|---|---|---|
| `cs-workbook` | `.xlsx` | The *Dataset* sheet of the Excel experiment: a header row with `Class` above the class column, one row per object, then the feature-type row (1 = quantitative, 0 = nominal). |
| `template-extended` (alias `extended`) | `.csv`, `.dat` | The author's layout (see [below](#the-extended-layout)): first row `m, n, c`; m rows of n values and the class label; last row of n type flags. `.dat` is whitespace-separated and may use decimal commas. |
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

## The dataset folder

The project keeps the author's datasets in `datasets/`, in the layout of the template project
(ADR-035):

```text
datasets/
  default.dat, default.csv                  the default dataset: Heart-Disease (10, 13, 2)
  raw/
    Heart-Disease/
      Heart-Disease (270, 13, 2).dat        original datasets, one folder per family,
      Heart-Disease (270, 13, 2).csv        named <Name> (m, n, c)
  synthetic/                                reserved for pre-existing synthetic datasets — not read
```

The files are the author's, byte for byte (Git stores them unchanged, the hooks leave them alone),
so their SHA-256 checksums are the same on every machine. The synthetic-feature datasets this
project generates from raw data — normalized values, Ψ(r), the meta-dataset Y — are written to
run folders by the exporters, with the manifest that says how they were produced.

### The extended layout

Every dataset file follows the same layout, as a `.dat` (whitespace-separated) or a `.csv`
(comma-separated) file:

```text
10 13 2                                             m (objects) n (features) c (classes)
70.0 1.0 4.0 130.0 322.0 0.0 2.0 109.0 0.0 2.4 2.0 3.0 3.0 2     n values, then the class (1 … c)
…                                                   (m rows)
1 0 0 1 1 0 0 1 0 1 0 1 0                           n type flags: 1 = quantitative, 0 = nominal
```

In the `.csv` the first row is padded with empty fields to n + 1 columns and the flag row ends with
an empty field under the class column; the `.dat` writes every value with a decimal point
(`70.0`), the `.csv` without (`70`). Features are named x₁ … xₙ and objects S₁ … Sₘ. The loader
checks the header against the rows: m rows of n + 1 values, n flags, c classes.

### Names and checks

| You write | Means |
|---|---|
| nothing / `default` | the default dataset, `datasets/default.dat` |
| `heart-disease-270` | a dataset by id (`<name>-<m>`; `-<n>` is added when two datasets would share it) |
| `Heart-Disease (270, 13, 2)` | the same, by display name |
| `datasets/raw/Heart-Disease/Heart-Disease (270, 13, 2).csv` | a file, in any supported format |
| `builtin:heart-disease-270` | the package's built-in copy |

When both formats of a dataset exist, the `.dat` is read (the `.csv` of a large dataset may be a
rounded copy). `csr data check` loads every file and checks that the header's (m, n, c) matches the
file name and that the `.dat` and `.csv` hold identical data; it prints each file's SHA-256.

```powershell
.\.venv\Scripts\csr.exe data list          # the dataset folder, the built-ins and the formats
.\.venv\Scripts\csr.exe data check         # consistency and checksums
.\.venv\Scripts\csr.exe data info          # the default dataset
.\.venv\Scripts\csr.exe run heart-disease-270 --no-save
```

The folder is found from the working directory (or a configuration file's folder) upwards; the
environment variable `CSR_DATASETS` overrides it. Without a folder — e.g. an installed GUI — the
package's built-in copies are used, and `heart-disease-10` is the default.

### Adding a dataset

Put `<Name> (m, n, c).dat` (and, if you like, the `.csv`) under `datasets/raw/<Family>/`, run
`csr data check`, and name it by its id. Other formats (tables with a header row, Excel, Parquet)
are read from any path; the catalogue can learn them later through the loader registry.

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
    "resources/experiments/context-synthetic-model/Context-Synthetic Model – Full Experiment [Heart-Disease (10, 13, 2)] - Opus 5.5.xlsx"
)
```

The built-ins are copies of the dataset folder's Heart-Disease files (the same data, checked by
the tests). A configuration names a dataset of the folder by id, or a built-in with the prefix
`builtin:`; no path means the default dataset. The configured `feature_types` apply in every case:

```yaml
dataset:
  path: heart-disease-270        # datasets/raw/Heart-Disease/Heart-Disease (270, 13, 2).dat
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
