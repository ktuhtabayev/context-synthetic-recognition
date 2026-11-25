# Changelog

All notable changes are recorded here, following [Keep a Changelog](https://keepachangelog.com/)
and [Semantic Versioning](https://semver.org/). Each milestone is a minor version.

## [Unreleased]

### Added

- Seven datasets from open archives, with the author's read-me files and pinned checksums
  (ADR-087): Heart-Failure (299, 12, 2), Ionosfera (350, 33, 2), Molecular-Biology (106, 57, 2),
  (100, 50, 2) and (100, 10, 2), Tic-Tac-Toe (958, 9, 2), Breast-Cancer-Wisconsin (569, 30, 2) and
  Spambase (4204, 57, 2).
- Spambase's two training and test samples (ADR-088), brought to the split-file convention and kept
  as `datasets/splits/Spambase/Spambase (2102, 57, 2) — training 601 (euclidean).dat` and
  `… — training 722 (euclidean).dat`: samples of the *Classifiers* page and of
  `csr compare predict --split 601 | 722`, not part of the comparison tables.
- Every split file of `datasets/splits` is a sample, now and for files added later (ADR-090):
  `services.split_discovery` finds them by their name,
  `<Dataset> (<m>, <n>, <c>) — training <t> (<metric>).dat`, and checks the header against the
  name, the training size against the split markers and the metric against the registry. A file
  that fails a check is listed as unavailable with its file, row and reason. Split keys are
  `<Dataset>-<t>` (`German-341`); the CLI still accepts the training size alone (`--split 601`).
  How to add one: [Adding a split file](docs/comparison.md#adding-a-split-file).
- `csr compare run --split KEY --out DIR` writes one split's evaluation (results, summary,
  checksums), never into `experiments/comparison`; `csr compare check` checks such a folder too.
- The *Classifiers* page shows the samples as a tree of datasets with checkboxes (ADR-091): one
  split, a whole dataset or splits of several datasets can be ticked, with the keyboard as well;
  German's three splits are ticked by default; unavailable files are greyed with their reason;
  *Refresh* reads the folder again. The page remembers the part and the ticks.

### Changed

- The comparison tables are named by their content (ADR-092): "Table 7" is now *Standard
  classifiers* (`standard-classifiers.*`), "Table 8" *k-NN by metric* (`knn-by-metric.*`), and the
  part "Tables 7 and 8" *Comparison with standard classifiers* (`classifier-comparison`). The old
  names are gone from the project, the earlier entries of this changelog included; there are no
  aliases. No number changed.
- "Random Forest" is written with capital letters wherever a user reads it (ADR-093).
- Every split is named by its dataset in every table, note, file, progress line and status:
  "German · Euclidean split · training 341 (173, 168)" instead of "341". "All article splits" is
  gone: the comparison tables appear when German's three splits are ticked.
- Every method shows its own settings — on the *Methods* tab, in the tables' notes and in
  `summary.json` — and the facts a group shares are said once (ADR-094); Random Forest's row no
  longer shows SVM's settings. A long text shows in full in its tooltip.
- The methods of a split follow one rule instead of a list per split; every split keeps exactly
  its earlier methods, and a method a split cannot support is shown as unavailable with the reason
  (ADR-090).

- The original datasets are grouped by the kind of their features: `datasets/raw/Heterogeneous/`,
  `Nominal/` and `Quantitative/` (ADR-087). German and Heart-Disease keep their bytes and ids.
  TCGA_LUAD (589, 57772, 2) — larger than GitHub accepts — and Cancer, Hypertension and Uzb-Kor stay
  on the author's disk (`.gitignore`); the catalogue still reads them there.

### Fixed

- The desktop application could crash with an access violation when Python's garbage collector ran
  in a worker thread and freed Qt objects there; workers now pause the collector, and it collects
  in the interface's thread (ADR-089).
- Spambase was missing from the *Classifiers* page of the Windows build: the build of 0.10.0
  predates Spambase's split files, so neither its code nor its copy of `datasets/splits` had them.
  The splits are now found from the folder the build carries, so a rebuild includes every file of
  `datasets/splits`.

## [0.10.0] — 2026-11-24 — Standard classifiers and a staged Classifiers page

Standard classifiers on the article's training and test samples (the Standard classifiers and
k-NN by metric tables), with the CS-model beside them, the split files read by the author's
convention, CSV files that open in Excel, and the *Classifiers* page as a workbench in three steps whose test step classifies the
objects without their class.

### Added

- The comparison with standard classifiers: SVM, Random Forest and k-NN on the article's training and test
  samples, apart from the CS-model's pipeline (ADR-079). The split files are kept in
  `datasets/splits/` and read by `data.splits.load_split_file`. The package `comparison`,
  the service `services.comparison`, `csr compare run | check`, `scripts/run_comparison.py`
  and the desktop application's *Classifiers* page write and check
  `experiments/comparison/` (CSV, Markdown, LaTeX, `summary.json`, checksums); a test pins
  every number. No method reaches the majority-class rate of a test sample.
- The CS-model on the same samples, with settings approved before the run (ADR-080): one operator
  per metric, min–max on the training sample, odd k = 3 … 41, the zero value, α = 0.1 and 0.2.
  It is written to a separate `cs-model` table beside the comparison tables, and its rows are added to the
  results and the summary. Its balanced accuracy stays between 48 % and 57 %.
- Classifier parts (ADR-084): units of work in three stages — training on the training sample,
  the test objects classified without their class, and the evaluation against the true classes.
  The protocol and the registry `services.classifier_parts.CLASSIFIER_PARTS` (with
  `Registry.remove`); the comparison with standard classifiers is the first part. A new part is registered in the services
  and appears on the *Classifiers* page without a change to the page (`docs/extending.md`).
- The blind test (ADR-085): `data.samples.UnlabelledSample` (the objects' places, feature values,
  types and names, and no class), `AnswerKey` (the classes, kept apart for the evaluation) and
  `Decisions`; `FixedSplit.test_objects()` and `test_answers()`. Tests show that shuffling,
  flipping or deleting the test classes changes no prediction of any method, and that a dataset
  with classes is rejected at run time and by mypy.
- `csr compare predict [--split KEY] [--out FILE]` writes the blind predictions of one split's test
  objects as CSV: the object and one column per method, no class column (ADR-086).

### Changed

- The Windows build carries scikit-learn and the article's split files, so the *Classifiers*
  page works in `csr-gui.exe`. Its self-test also computes the comparison tables (ADR-081), and checks
  that the classifier parts are registered.
- The *Classifiers* page is a staged workbench (ADR-084): a list of parts, the samples (the three
  article splits, each split alone, Heart's split, with their sizes), and the steps ① Training,
  ② Test (the classes hidden: test objects, predictions, predicted classes) and ③ Evaluation (the
  classes revealed: the Standard classifiers and k-NN by metric tables, the CS-model's table, all results, per object, confusion,
  check, writing the files). Each step shows its state by a symbol and a word (not run, running,
  done, out of date, unavailable); *Run all* runs the three in a row; Alt+← / Alt+→ move between
  the steps. Run all on the three article splits gives today's tables and files byte for byte.
- The comparison runs in three stages (`comparison.stages`); `compare_split` is the three in a row
  and gives the same numbers. Its SVM and Random Forest are fitted again on the training sample at
  the test stage, deterministically (ADR-085).

### Fixed

- The split files are read by the author's convention (ADR-082): the header gives `m n c` with n
  the number of features, each object row holds the n values, the class and the split marker, and
  the last row holds the n types and two empty cells. Three type rows that broke it are
  corrected, so only those lines and the files' checksums changed; every number of the comparison tables
  and of the CS-model stays the same. `data.splits.load_split_file` checks the convention and
  names the file and the row of every departure, and the *Classifiers* page shows the message.
- CSV files open correctly in Excel: every CSV file the project writes starts with a byte-order
  mark (ADR-083), so Excel reads it as UTF-8 and no longer shows "—" as "â€”" or α as "Î±". This
  covers the comparison's and the study's results, the CSV export, the run folders and the
  desktop application's *Save CSV…*. The committed CSV files differ from the previous ones only
  by those three bytes.

## [0.9.0] — 2026-11-22 — rebuild on the article

Its GitHub Release and its tag were removed on 2026-11-24; 0.10.0 includes these changes.

The project is rebuilt on the author's article (ADR-058): its stages, its names and its defaults.

### Added

- R-IPS, the first stage: the hyperspheres r(S), g(S), Steps 1–6, an optional test sample T, and
  the configuration section `rips` (ADR-061). Two switches, flagged ⚠: `rips.keep_blocked_cover`
  and `rips.training: moved`.
- The protocols `rips` (Definition 3 on the R-IPS split: accuracy with refusals as errors, accuracy
  on the answered objects, refusals, class split) and `nested-cross-validation` (R-IPS inside every
  outer fold, α chosen on the inner test sample) (ADR-069).
- The zero value of the article's experiments, `synthetic_features.zero_value` (ADR-064);
  original nominal features combined with the synthetic ones,
  `synthetic_features.original_features` (ADR-065); a normalizer per base operator,
  `local_context.operators[].normalizer` (ADR-066).
- Class-sequence bit masks and majority chains as training-side analysis (ADR-067); the
  metaobject in every output (ADR-074).
- The dataset German (1000, 20, 2), byte for byte (ADR-073).
- The article's computational experiment rebuilt from real runs: `scripts/run_experiments.py`
  writes `experiments/` with checksums; a test reproduces it byte for byte. The results do not
  reach the accuracies the article prints; nothing was tuned (ADR-070).
- `docs/glossary.md`: the article's terms, symbols, English names and code names.
- The admissible-k strategy `min-class` (odd k = 3 … min|Kᵢ|).
- The dataset summary reports unknown objects of the test sample T before a run.

### Changed

- **Defaults follow the article** — admissible k `min-class`, final class means, one majorizing
  pass in Step 4 (ADR-062, ADR-063). This changes the results: on Heart-Disease (10, 13, 2)
  resubstitution 70 % → 50 % and leave-one-out 20 % → 0 %; on Heart-Disease (270, 13, 2)
  resubstitution 44.81 % → 84.44 % and leave-one-out 7.04 % → 80.74 % (`tests/test_defaults.py`).
- Presets `default`, `alpha-0.1`, `alpha-0.2`, `excel-experiment`; configuration files of the same
  names in `configs/` (ADR-071). `csr validate` runs the Excel experiment workbook with
  `excel-experiment` and still agrees in every compared cell (11,622 cells, 579 checks; ADR-058).
- Every name follows the article (ADR-059): the core is ordered by stage (`core/rips`,
  `core/local_context`, `core/synthetic_features`, `core/additional_features`,
  `core/meta_algorithm`); the configuration has `schema_version: 2` with the sections `rips`,
  `local_context`, `admissible_k`, `synthetic_features`, `additional_features`,
  `meta_algorithm`, `evaluation` — schema-1 files are not read; the Excel mirror, the tables, the
  figures, the report, the CLI and the desktop application use the same names. The renaming
  changed no number.
- The Excel mirror's *Switches* sheet explains the switches on the run's own data (ADR-075).
- The desktop application: a preset list, the R-IPS group, a normalizer per operator, the zero
  value and original features, and a "⚠ deviates from the article" badge beside every switch that
  is on; Russian and Uzbek translations updated.
- `configs/heart-disease-270-large-data.yaml` follows the article's settings and keeps only the
  constant-feature skip; on Heart-Disease (270, 13, 2) it now gives the default's results
  (leave-one-out 81.5 % → 80.7 %, ADR-076).
- `csr fit`, `csr classify`, `csr data info` and `csr run` read the configuration's dataset when
  no dataset is named, and apply its feature types and reading options when one is (ADR-077).
- The results tree, the tables and the figures are grouped by the article's stages; the Excel
  mirror keeps the workbook's Steps 1–12 (ADR-078). The additional features are named d₁ … d_p
  everywhere.
- The study of the article's experiments treats the slope of Heart-Disease (x₁₁) as quantitative,
  as the article does, through `dataset.feature_types`; the data file is unchanged (ADR-072).

### Removed

- The author's source material is no longer part of the repository; the workbooks used as
  acceptance tests are copied to `tests/data/workbooks/` and the independent re-implementation
  used as a test oracle to `tests/golden/independent/` (ADR-060).

### Fixed

- Nine small windows flashed on the screen before the main window opened: the parameter forms of
  the protocols and baselines on the Configure page showed themselves before they were placed on
  the page. They now appear only within it, as before; nothing else changes.

## [0.8.0] — 2026-10-02 — M8 release build, languages, documentation

### Added

- The interface in Russian and Uzbek (Latin script), 288 texts each, and **View → Language**
  with *System language*, English, Русский and Oʻzbekcha. The window changes at once and keeps
  the dataset, the configuration with its undo history, the run and the selected object; the
  choice is remembered. Only the interface is translated — the tables, figures and exported
  files stay in English (ADR-054).
- `scripts/update_translations.py` keeps the translation files in step with the code and compiles
  them; the tests fail if a text is missing, stale or lost a placeholder.
- A Windows executable of the desktop application: `scripts/build_exe.py` (PyInstaller, optional
  extra `[release]`) builds `csr-gui-<version>-windows.zip`, and the CI job *Windows executable*
  keeps it as the artifact `csr-gui-windows` of every run (ADR-055).
- `csr-gui --self-test [REPORT]`: checks an installation without showing a window — every
  language, the default experiment, a table, a figure and an export of the run.
- *Reproducing the experiment* (`docs/reproduction.md`): the workbook checked cell for cell,
  every result step by step, repeating a saved run.
- The documentation site is published to GitHub Pages from `main` (ADR-056).
- Development guide: translations, the executable, releasing.

### Changed

- The sidebar grows with the longest page name (other languages, larger zoom).
- The package's development status is *Beta*.
- Architecture: the extension-points table lists what is built in and, separately, what is not.

### Fixed

- `csr --help` printed "Start the desktop application (needs the extra )": the help renderer
  took `[gui]` for markup. The line no longer uses square brackets.
- The title at the top of a page was not passed through the translation hook.
- Docstrings that still announced finished milestones.

## [0.7.0] — 2026-10-02 — M7 metrics and normalizers

### Added

- Metrics (ADR-051): `weighted-zhuravlyov` (type and feature weights), `heom`, `gower` on any
  features; `manhattan`, `euclidean`, `chebyshev`, `minkowski` (order p), `canberra`, `cosine`,
  `mahalanobis` on quantitative features; `hamming` on nominal features. The Zhuravlyov
  operators ρ, ρ_I, ρ_J stay the default. HVDM is not offered: it reads the classes of the
  training objects (author).
- Metric traits (ADR-050): `core.local_context.metrics.traits(domain=…, fit=…, check=…)`. A
  metric names the feature types it is defined on (a mismatch is a configuration error that names
  the features), may estimate training statistics in a fit step — ranges for HEOM and Gower, the
  covariance for Mahalanobis — re-fitted in every fold and never given a class label, and may
  check its parameters against an operator's features.
- Normalizers (ADR-052): `z-score` (parameter `ddof`), `robust` (median and interquartile range),
  `max-abs`, `decimal-scaling`, `rank` (mid-ranks, new values interpolated) and `unit-length`
  (per object). Min–max stays the default; every fitted normalizer maps a feature without
  spread to 0 and records the training minimum and maximum next to its own statistics.
- The features table, the report and the *Normalized Dataset* sheet show the statistics and the
  formula of the normalizer in use; for min–max they are unchanged.
- Documentation: definitions of every metric and normalizer in the algorithm reference, metric
  traits in *Extending*.

### Changed

- The project is Windows-only (ADR-049): every CI job runs on Windows, the package metadata
  names Windows, and `scripts/gui_screenshots.py` always uses the native Windows platform.
- No dataset was added: Heart-Disease stays the only one in the repository (ADR-053, author).

## [0.6.0] — 2026-10-01 — M6 desktop application

### Added

- Desktop application (`gui`, PySide6; optional extra `[gui]`; `csr gui` and the `csr-gui`
  script) with the workflow Dataset → Configure → Run → Results → New object → Compare → Export
  (ADR-044):
  - *Dataset*: project and built-in datasets, files, drag-and-drop; preview; feature types;
    reading options; live |K1|, |K2|, admissible k and r.
  - *Configure*: every setting with plug-in parameter forms generated from the parameter types;
    presets; the two ⚠ switches marked; validation; undo / redo (ADR-046).
  - *Run*: worker thread with progress, log and cancel; optional run folder; switch sensitivity.
  - *Results*: the workbook's steps as a tree of sortable tables and interactive figures, with
    the workbook's colour semantics, formula tooltips and cross-highlighting of the selected
    object (ADR-045).
  - *New object*: a form or a pasted row; the decision explained by B1 / B2 per step, Ψ(r) and
    the neighbours; the Theorem check for a training object left out of its context.
  - *Compare*: saved runs side by side with the differences marked (ADR-047).
  - *Export*: every export format in the background.
  - Light and dark themes, zoom, keyboard shortcuts, recent files, remembered window state;
    every string passes through a translation hook.
- `services.runs`: `list_runs`, `comparison`, `config_differences` — saved runs side by side
  without recomputation (ADR-047).
- Figure `draw_object_context`: one object's neighbours with the nested k-neighbourhoods.
- GUI user guide with screenshots (`docs/gui.md`, `scripts/gui_screenshots.py`); pytest-qt
  tests of the whole workflow on Qt's offscreen platform (ADR-048).

### Changed

- `switch_sensitivity` takes a `cancelled` callback, like `run_protocol`.

## [0.5.0] — 2026-10-01 — M5 exporters

### Added

- Excel mirror (`export.excel`, `workbook.xlsx`): the sheets, layout and style of the Excel
  experiment, generated for any data — a *Sorted Neighbors* sheet per operator, a step sheet per
  iteration of the agglomerative algorithm, a fold sheet per hold-out protocol; values, no
  formulas. The export of Heart-Disease (10, 13, 2) passes `csr validate` (11,631 cells) and
  matches the experiment workbook in structure and style (golden tests). Size limits with notes
  on the sheets for large data (ADR-037, ADR-038).
- Tables of a run as plain data (`export.tables`, 44 for the experiment) and their forms: CSV
  and JSON at full precision, Markdown and LaTeX (`booktabs`, the notation in math mode) for
  the article (ADR-040).
- Figures (`export.figures`, matplotlib): distances, neighbourhoods, ω and stability, the
  candidates and the criterion of the agglomerative algorithm, margins, ROC, confusion,
  outcomes, prediction map, sensitivity; PNG and SVG, light and dark; drawn on given axes for
  reuse by the GUI (ADR-041).
- Run report (`export.report`): self-contained `report.html` and `report.pdf` (optional extra
  `[pdf]`, reportlab), with the two switches that deviate from the article highlighted first
  (ADR-041).
- `csr run --export FORMATS` and `csr export [RUN]` (`--format`, `--out`, `--object`, `--values`,
  `--sensitivity/--no-sensitivity`, `--decimals`, `--dpi`, `--theme`); `export_result` and the
  `EXPORTERS` registry in Python (ADR-036, ADR-042).
- Run folders are self-contained: `dataset.json` (the dataset as used) and the switch settings
  in `results.json`; `load_run` repeats a run from its folder and reports differences from the
  stored results (ADR-036).
- The worked examples of the switches sheet are computed by the package from the data of the
  agglomerative algorithm's check workbook, shipped with the package as JSON (ADR-043).
- The dataset folder (ADR-035): `datasets/default.dat|.csv` (Heart-Disease (10, 13, 2), the default
  dataset) and `datasets/raw/Heart-Disease/Heart-Disease (270, 13, 2).dat|.csv`, byte for byte;
  `data.catalog` (discovery, ids, `.dat`/`.csv` identity checks, name resolution with the
  built-ins as fallback); `csr data check`; `csr data list` shows the folder; `data info`, `fit`,
  `classify` and `run` use the default dataset without an argument; `dataset.path` may be empty
  or an id; format alias `extended`.

### Changed

- matplotlib is a core dependency; reportlab is the optional extra `[pdf]` (ADR-041).
- `csr validate` accepts the sheet *Zhuravlyov Distances* of an exported workbook as well as
  the experiment's *Zhuravlev Distances*, and reports the sheet under the name it found
  (ADR-039).
- SVG figures are written with LF line endings on every platform.

## [0.4.0] — 2026-09-30 — M4 evaluation

### Added

- Evaluation layer (`evaluation`): protocols `resubstitution`, `leave-one-out` (the whole pipeline
  re-fitted per fold), `stratified-k-fold`, `repeated-k-fold`, `hold-out`; undefined folds refuse
  and are flagged; progress and cancel callbacks (ADR-032).
- Baselines on the same folds: the k-NN vote per operator and k, and scikit-learn classifiers
  (logistic regression, Random Forest, SVM, decision tree, naive Bayes; optional extra
  `[sklearn]`) (ADR-032).
- Metrics with the workbook's conventions: confusion counts with refusals separate, accuracy,
  coverage, precision/recall/F1 per class and macro, Mann–Whitney AUC on rounded scores, ROC table;
  margins of the additional features with and without the majorizing function (ADR-031).
- Model properties (`core.properties`): Definitions 1, 4 and 6, Property 1, boundary ties and the
  Theorem check (ADR-031).
- Experiment runner and run folders (`services.runner`: manifest, `results.json`,
  `predictions.csv`, `folds.csv`), switch sensitivity (`services.sensitivity`), `csr run`
  (`--sensitivity`, `--no-save`, `--runs-dir`) and `builtin:<name>` dataset paths (ADR-033).
- Switch `synthetic_features.skip_constant` (default off) and the large-data example configuration
  `configs/heart-disease-270-large-data.yaml` (constant features skipped, k ≤ 21), after the
  study of ADR-030: on Heart-Disease (270, 13, 2) the literal k range gives 7 % leave-one-out,
  the large-data setting 81.5 %.
- `csr validate` covers every computed sheet of the experiment: 11,631 cells in 29 sheets, in
  workbook order (ADR-034); a golden test checks the tables of the switches sheet that use the
  data of the agglomerative algorithm's check workbook against the copy of that workbook.
- Documentation: the Evaluation page; ADR-030 – ADR-034.

### Changed

- The preset configurations contain `synthetic_features.skip_constant: false`; their hashes change.
- `csr config check` also checks the evaluation protocols and baselines
  (`services.configs.config_problems`).

## [0.3.0] — 2026-09-30 — M3 agglomerative algorithm, meta-algorithm, check-workbook replication

### Added

- The hierarchical agglomerative algorithm, Steps 1–5 (`core.additional_features.agglomerative`),
  with both switches that deviate from the article, the stopping rules and a full trace: every
  iteration's θ, γ, θ/γ, cr1, q, crit and additional feature; the per-object columns of any
  candidate block on demand, bit-identical to the grouping (ADR-024). θ/γ = +∞ when γ = 0
  (ADR-025).
- Majorizing functions (`core.additional_features.majorizing_functions`): logistic sigmoid
  (default), tanh, arctan and softsign, scaled to (0, 1) (ADR-026).
- Meta-algorithm, Steps 1–5 (`core.meta_algorithm`): B1/B2 at every step for any gradations, the
  decision rule registry (`step-4`, exact comparison, refusal = 0) (ADR-027).
- The CS-model end to end (`core.model.fit_model`, `CSModel`): the meta-dataset Y = (y, r), the
  training description, `represent` / `classify` / `predict` without a label (Theorem) and
  `classify_training` (resubstitution) (ADR-028).
- `csr fit` and `csr classify` (`--values`, `--object`) with the B1/B2 explanation.
- `csr validate` covers Steps 9–12 of the experiment (10,272 cells in 19 sheets, all within
  3.0e-14) with its inputs read from the workbook, and recognises the check workbooks of the
  agglomerative algorithm (SET {x₃, x₆, x₁₃, x₄, x₉}, 6,131 cells) and of the meta-algorithm
  (Class 2) (ADR-029).
- Copies of the two check workbooks in `tests/data/` for CI (now `tests/data/workbooks/`); golden
  tests for the switch variants, the ten leave-one-out folds and the replication of the check
  workbooks; cross-checks with an independent re-implementation used as a test oracle on the
  experiment, Heart-Disease (270, 13, 2) and random datasets.

### Changed

- `services.validation` is a package (`checks`, `context_map`, `model_map`, `experiment`,
  `check_workbooks`); `Check` is generic over the computed subject and may be conditional.
- `csr config check` also checks the majorizing function and the decision rule.

## [0.2.0] — 2026-09-30 — M2 core pipeline to Ψ(r)

### Added

- Data layer: immutable `Dataset` schema (types I/J, class order K1, K2, categories, content
  hash); loaders for the experiment workbook's Dataset sheet, the extended CSV/DAT layout, CSV,
  XLSX and Parquet (optional `[parquet]` extra) with format detection; built-in datasets
  `heart-disease-10` and `heart-disease-270`.
- Core Steps 1–8: min–max normalization fitted on the training objects; the Zhuravlyov metric
  and base operators ρ, ρ_I, ρ_J (empty subsets skipped, ADR-007); neighbour order by
  (distance, index); admissible k (`twice-min-class`, a strategy for the draft's text,
  `explicit`, `range`); μ, χ₁ and formula (5); formulas (1)–(4) and majority bit chains;
  formula (6) and the weights ω by formula (4).
- `fit_context` with an immutable trace of every Step 1–8 table, and
  `ContextModel.represent` — new objects described without their class (Theorem).
- Plug-in parameter validation (`Registry.make_params`, frozen pydantic dataclasses);
  `csr config check` reports unknown plug-ins and invalid parameters.
- `csr data list`, `csr data info` (sizes, types, |Kᵢ|, admissible k, r) and
  `csr validate --against <workbook>` (Steps 1–8: 4,022 cells, all within 4.4e-15).
- Golden tests from the workbook's cached cells; bit-for-bit cross-checks against an independent
  re-implementation used as a test oracle on the experiment, Heart-Disease (270, 13, 2) and random
  tie-heavy datasets; leakage tests for the new-object path.
- Documentation: datasets, extending (plug-ins), ADR-018 – ADR-023.

### Changed

- The metric is spelled **Zhuravlyov** (registry name `zhuravlyov`; `zhuravlev` and `juravlev`
  remain aliases, ADR-023). The preset configurations' hashes change accordingly.
- mypy checks with Python 3.12 semantics (numpy's stubs need it); the code stays 3.11-compatible.
- Entry-point plug-ins may register themselves on import; the entry-point group of a kind is its
  normalized name (e.g. `context_synthetic_recognition.k-strategies`).

## [0.1.0] — 2026-09-30 — M1 skeleton

### Added

- Packaging (`src/` layout, setuptools, `csr` console script, `py.typed`), MIT license.
- Typed experiment configuration (pydantic v2): strict keys, immutable, YAML/TOML/JSON,
  format-independent SHA-256 hash; two presets in `configs/`: the Excel experiment's setting
  (default; now `excel-experiment`) and the article's (now `default`).
- The two switches that deviate from the article, `additional_features.class_means`
  (running | final) and `additional_features.majorizing_passes` (2 | 1), described once in
  `config.presets.DEVIATIONS` and flagged by the CLI and file headers.
- Plug-in registry with aliases, parameter types and entry points.
- Structured logging (key=value console lines, JSON lines per run) and the exception hierarchy.
- Run folders `runs/<YYYYMMDD_HHMMSS>_<hash>/` with `manifest.yaml` (configuration, hashes, seed,
  package versions, timings).
- `csr config show | init | check`.
- Tooling: ruff, mypy (strict), pytest with hypothesis and coverage ≥ 90 %, pre-commit, GitHub
  Actions (Windows, Python 3.11–3.13, strict docs build).
- Documentation: architecture, algorithm reference with notes for the article text, ADR log
  (ADR-001 – ADR-017), development guide.
