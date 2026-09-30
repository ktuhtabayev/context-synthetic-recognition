# Changelog

All notable changes are recorded here, following [Keep a Changelog](https://keepachangelog.com/)
and [Semantic Versioning](https://semver.org/). Each milestone is a minor version.

## [Unreleased]

## [0.4.0] — 2026-09-30 — M4 evaluation

### Added

- Evaluation layer (`evaluation`): protocols `resubstitution`, `leave-one-out` (the whole pipeline
  re-fitted per fold), `stratified-k-fold`, `repeated-k-fold`, `hold-out`; undefined folds refuse
  and are flagged; progress and cancel callbacks (ADR-032).
- Baselines on the same folds: the k-NN vote per operator and k, and scikit-learn classifiers
  (logistic regression, random forest, SVM, decision tree, naive Bayes; optional extra
  `[sklearn]`) (ADR-032).
- Metrics with the workbook's conventions: confusion counts with refusals separate, accuracy,
  coverage, precision/recall/F1 per class and macro, Mann–Whitney AUC on rounded scores, ROC table;
  margins of the latent features with and without the majorizer (ADR-031).
- Model properties (`core.properties`): Definitions 1, 4 and 6, Property 1, boundary ties and the
  Theorem check (ADR-031).
- Experiment runner and run folders (`services.runner`: manifest, `results.json`,
  `predictions.csv`, `folds.csv`), switch sensitivity (`services.sensitivity`), `csr run`
  (`--sensitivity`, `--no-save`, `--runs-dir`) and `builtin:<name>` dataset paths (ADR-033).
- Switch `synthetic.skip_constant` (default off) and the large-data example configuration
  `configs/heart-disease-270-large-data.yaml` (constant features skipped, k ≤ 21), after the
  study of ADR-030: on Heart-Disease (270, 13, 2) the literal k range gives 7 % leave-one-out,
  the large-data setting 81.5 %.
- `csr validate` covers every computed sheet of the experiment: 11,631 cells in 29 sheets, in
  workbook order (ADR-034); a golden test checks the template-data tables of *Template Deviations*
  against the template copy.
- Documentation: the Evaluation page; ADR-030 – ADR-034.

### Changed

- The preset configurations contain `synthetic.skip_constant: false`; their hashes change.
- `csr config check` also checks the evaluation protocols and baselines
  (`services.configs.config_problems`).

## [0.3.0] — 2026-09-30 — M3 HAG, meta-algorithm, template replication

### Added

- HAG, Steps 1–5 (`core.hag`), with both template/article switches, the stopping rules and a full
  trace: every iteration's θ, γ, θ/γ, cr1, q, crit and latent feature; the per-object columns of
  any candidate block on demand, bit-identical to the grouping (ADR-024). θ/γ = +∞ when γ = 0
  (ADR-025).
- Majorizing functions (`core.majorizers`): logistic sigmoid (default), tanh, arctan and softsign,
  scaled to (0, 1) (ADR-026).
- Meta-algorithm, Steps 1–5 (`core.meta`): B1/B2 at every step for any gradations, the decision
  rule registry (`article-step-4`, exact comparison, refusal = 0) (ADR-027).
- The CS-model end to end (`core.model.fit_model`, `CSModel`): the meta-dataset Y = (y, r), the
  training description, `represent` / `classify` / `predict` without a label (Theorem) and
  `classify_training` (resubstitution) (ADR-028).
- `csr fit` and `csr classify` (`--values`, `--object`) with the B1/B2 explanation.
- `csr validate` covers Steps 9–12 of the experiment (10,272 cells in 19 sheets, all within
  3.0e-14) with its inputs read from the workbook, and recognises the HAG template (SET
  {x₃, x₆, x₁₃, x₄, x₉}, 6,131 cells) and the meta-algorithm template (Class 2) (ADR-029).
- Copies of the two template workbooks in `tests/data/templates` for CI; golden tests for the
  switch variants, the ten leave-one-out folds and the template replication; cross-checks with
  the reference engine on the experiment, Heart-Disease (270, 13, 2) and random datasets.

### Changed

- `services.validation` is a package (`checks`, `context_map`, `model_map`, `experiment`,
  `templates`); `Check` is generic over the computed subject and may be conditional.
- `csr config check` also checks the majorizer and the decision rule.

## [0.2.0] — 2026-09-30 — M2 core pipeline to Ψ(r)

### Added

- Data layer: immutable `Dataset` schema (types I/J, class order K1, K2, categories, content
  hash); loaders for the experiment workbook's Dataset sheet, the template's CSV/DAT layout, CSV,
  XLSX and Parquet (optional `[parquet]` extra) with format detection; built-in datasets
  `heart-disease-10` and `heart-disease-270`.
- Core Steps 1–8: min–max scale unification fitted on the training objects; the Zhuravlyov metric
  and base operators ρ, ρ_I, ρ_J (empty subsets skipped, ADR-007); neighbour order by
  (distance, index); permitted k (`formula`, `article`, `explicit`, `range`); μ, χ₁ and
  formula (5); formulas (1)–(4) and bit masks; formula (6) and the `omega` weights.
- `fit_context` with an immutable trace of every Step 1–8 table, and
  `ContextModel.represent` — new objects described without their class (Theorem).
- Plug-in parameter validation (`Registry.make_params`, frozen pydantic dataclasses);
  `csr config check` reports unknown plug-ins and invalid parameters.
- `csr data list`, `csr data info` (sizes, types, |Kᵢ|, permitted k, r) and
  `csr validate --against <workbook>` (Steps 1–8: 4,022 cells, all within 4.4e-15).
- Golden tests from the workbook's cached cells; bit-for-bit cross-checks against the reference
  engine on the experiment, Heart-Disease (270, 13, 2) and random tie-heavy datasets; leakage
  tests for the new-object path.
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
  format-independent SHA-256 hash; `template` (default) and `article` presets in `configs/`.
- The two template-vs-article switches `hag.centres` (running | final) and `hag.step4_passes`
  (2 | 1), described once in `config.presets.DEVIATIONS` and flagged by the CLI and file headers.
- Plug-in registry with aliases, parameter types and entry points.
- Structured logging (key=value console lines, JSON lines per run) and the exception hierarchy.
- Run folders `runs/<YYYYMMDD_HHMMSS>_<hash>/` with `manifest.yaml` (configuration, hashes, seed,
  package versions, timings).
- `csr config show | init | check`.
- Tooling: ruff, mypy (strict), pytest with hypothesis and coverage ≥ 90 %, pre-commit, GitHub
  Actions (Windows and Linux, Python 3.11–3.13, strict docs build).
- Documentation: architecture, algorithm reference with notes for the article text, ADR log
  (ADR-001 – ADR-017), development guide.
