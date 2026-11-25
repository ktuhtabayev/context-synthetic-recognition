# Datasets

A dataset is a classified sample E₀ = {S₁ … Sₘ} with heterogeneous features: quantitative
(set I) and nominal (set J). The model is binary: exactly two classes K1 and K2 (ADR-010).

## Formats

| Format | Files | Layout |
|---|---|---|
| `cs-workbook` | `.xlsx` | The *Dataset* sheet of the Excel experiment: a header row with `Class` above the class column, one row per object, then the feature-type row (1 = quantitative, 0 = nominal). |
| `extended` | `.csv`, `.dat` | The author's layout (see [below](#the-extended-layout)): first row `m, n, c`; m rows of n values and the class label; last row of n type flags. `.dat` is whitespace-separated and may use decimal commas. |
| `csv` | `.csv`, `.txt` | A table with a header row; the delimiter (`,` `;` tab `\|`) is detected. |
| `xlsx` | `.xlsx` | A table with a header row (first sheet by default). |
| `parquet` | `.parquet` | A table; needs `pip install "context-synthetic-recognition[parquet]"`. |

`format: auto` (the default) picks the format from the extension and the content: an Excel file
with a *Dataset* sheet in the experiment's layout is `cs-workbook`, a CSV whose first row is
`m, n, c` is `extended`.

For tables with a header row, the class column is the last column unless `class_column` names
another; `id_column` names an optional column of object ids. Feature types come from the file
(the type row or flags) or from `feature_types` in the configuration, which overrides them;
features of a table without types are inferred — numeric columns as quantitative, the others as
nominal — and a warning lists the inferred types. Nominal text values become category codes
(alphabetical order of the categories).

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

The project keeps the author's datasets in `datasets/` (ADR-035), the original datasets grouped
by the kind of their features (ADR-087):

```text
datasets/
  default.dat, default.csv                  the default dataset: Heart-Disease (10, 13, 2)
  raw/
    Heterogeneous/                          quantitative and nominal features
      Heart-Disease/
        Heart-Disease (270, 13, 2).dat      one folder per family, named <Name> (m, n, c),
        Heart-Disease (270, 13, 2).csv      the author's ReadMe.txt beside most of them
      German/, Heart-Failure/, Ionosfera/
    Nominal/                                nominal features only
      Molecular-Biology/, Tic-Tac-Toe/
    Quantitative/                           quantitative features only
      Breast-Cancer-Wisconsin/, Spambase/
  synthetic/                                reserved for pre-existing synthetic datasets — not read
  splits/                                   training and test samples (split files with a
    German/, Heart-Disease/, Spambase/      training/test flag) — not read by the catalogue
```

The files are the author's, byte for byte (Git stores them unchanged, the hooks leave them alone),
so their SHA-256 checksums are the same on every machine, and a test pins them; German was added in
the same way (ADR-073), and so were the split files in `splits/`, which only the *Classifiers*
part reads ([Standard classifiers](comparison.md), ADR-079, ADR-088). Every file of `splits/` is a
sample, found by its name; [Adding a split file](comparison.md#adding-a-split-file) gives the
name pattern and the checks (ADR-090). The catalogue finds a dataset at any depth under `raw/`, so
its ids do not depend on the kind folder.

Four datasets stay on the author's disk and out of the repository (`.gitignore`): TCGA_LUAD
(589, 57772, 2), whose two files are larger than GitHub accepts (100 MB), and Cancer
(589, 44, 2), Hypertension (147, 29, 2) and Uzb-Kor (100, 24, 2), which are not from an open
archive. Where they are on disk the catalogue lists and reads them like any other dataset. The synthetic-feature
datasets this project generates from raw data — normalized values, Ψ(r), the meta-description Y —
are written to run folders by the exporters, with the manifest that says how they were produced.

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
| `heart-disease-270`, `german-1000` | a dataset by id (`<name>-<m>`; `-<n>` is added when two datasets would share it) |
| `Heart-Disease (270, 13, 2)` | the same, by display name |
| `datasets/raw/Heterogeneous/Heart-Disease/Heart-Disease (270, 13, 2).csv` | a file, in any supported format |
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

Put `<Name> (m, n, c).dat` (and, if you like, the `.csv`) under
`datasets/raw/<Kind>/<Family>/` — `Heterogeneous`, `Nominal` or `Quantitative` by the kind of its
features — run `csr data check`, and name it by its id. Other formats (tables with a header row, Excel, Parquet)
are read from any path; the catalogue can learn them later through the loader registry.

## Classes

Class labels may be integers or text. They are sorted — numbers numerically, text alphabetically
— and the first class is **K1**, the second **K2** (ADR-018). For the author's data (labels 1 and
2) this is the article's numbering. Class 1 is also the positive class of the evaluation.

## The author's datasets

| Dataset | Content |
|---|---|
| `heart-disease-10` (built in; `default` in the folder) | Heart-Disease (10, 13, 2): the 10 objects of the Excel experiment (\|K1\| = 4, \|K2\| = 6) |
| `heart-disease-270` (built in and in the folder) | Heart-Disease (270, 13, 2): UCI Statlog (Heart), 6 quantitative + 7 nominal features as flagged in the file, \|K1\| = 150, \|K2\| = 120 |
| `german-1000` (in the folder) | German (1000, 20, 2): UCI Statlog (German Credit Data), 7 quantitative + 13 nominal features, \|K1\| = 700, \|K2\| = 300 |
| `heart-failure-299` | Heart-Failure (299, 12, 2): UCI Heart Failure Clinical Records, 7 quantitative + 5 nominal features, \|K1\| = 203, \|K2\| = 96 |
| `ionosfera-350` | Ionosfera (350, 33, 2): Ionosphere (Kaggle), 32 quantitative + 1 nominal feature, \|K1\| = 125, \|K2\| = 225 |
| `molecular-biology-106`, `-100-50`, `-100-10` | Molecular-Biology (106, 57, 2): UCI Promoter Gene Sequences, 57 nominal features, \|K1\| = \|K2\| = 53; (100, 50, 2) and (100, 10, 2): 100 of its objects with their first 50 or 10 features, \|K1\| = \|K2\| = 50 |
| `tic-tac-toe-958` | Tic-Tac-Toe (958, 9, 2): UCI Tic-Tac-Toe Endgame, 9 nominal features, \|K1\| = 332, \|K2\| = 626 |
| `breast-cancer-wisconsin-569` | Breast-Cancer-Wisconsin (569, 30, 2): UCI Breast Cancer Wisconsin (Diagnostic), 30 quantitative features, \|K1\| = 357, \|K2\| = 212 |
| `spambase-4204` | Spambase (4204, 57, 2): UCI Spambase without its duplicate rows, 57 quantitative features, \|K1\| = 2528, \|K2\| = 1676 |

### Heart: the slope as a quantitative feature

In the article's experiments Heart has 7 quantitative and 6 nominal features: the slope x₁₁ is
quantitative, while the file flags it nominal. The file stays as it is; the configuration sets the
type (ADR-072):

```yaml
dataset:
  path: heart-disease-270
  feature_types:
    x₁₁: quantitative
```

`feature_types` applies wherever a configuration is given: every command reads the dataset of
the configuration when no dataset argument names one, and applies the configuration's
`feature_types` and reading options to a dataset named on the command line too (ADR-077); so do
`load_from_config` in Python and `scripts\run_experiments.py`.

```python
from context_synthetic_recognition.config import DatasetConfig
from context_synthetic_recognition.data import load_builtin, load_dataset, load_from_config

heart = load_builtin("heart-disease-270")
heart_7 = load_from_config(
    DatasetConfig(path="heart-disease-270", feature_types={"x₁₁": "quantitative"})
)
experiment = load_dataset("tests/data/workbooks/Excel experiment [Heart-Disease (10, 13, 2)].xlsx")
```

The built-ins are copies of the dataset folder's Heart-Disease files (the same data, checked by
the tests). A configuration names a dataset of the folder by id, or a built-in with the prefix
`builtin:`; no path means the default dataset.

## Inspecting a dataset

`csr data info` shows what the pipeline will make of a dataset under a configuration:

```text
> csr data info heart-disease-270
Heart-Disease (270, 13, 2)
  source:    …\datasets\raw\Heterogeneous\Heart-Disease\Heart-Disease (270, 13, 2).dat
  objects:   m = 270
  features:  n = 13
    I (quantitative) 6: x₁, x₄, x₅, x₈, x₁₀, x₁₂
    J (nominal)      7: x₂, x₃, x₆, x₇, x₉, x₁₁, x₁₃
  classes:   K1 = 1: 150, K2 = 2: 120
  hash:      bf78a86ebd7478a48a0a013f28765be5145e451f84d84e92a0c2dcd49e315456
  k (min-class): 3, 5, 7, …, 119  (59 values; k ≤ min|Kᵢ| = 120)
  operators: ρ (13 features), ρ_I (6 features), ρ_J (7 features)
  r = |Ψ(r)| = 177 synthetic features
```

Options: `--format`, `--sheet`, `--class-column`, `--id-column` and `--config` (for another k
strategy or other operators). An operator whose feature subset is empty — ρ_J on a dataset without
nominal features — is skipped and reported (ADR-007).

The **content hash** identifies the data in run manifests; it covers values, labels, types,
feature names and object ids, not the file name, so the workbook's Dataset sheet and the built-in
CSV of the same 10 objects have the same hash.
