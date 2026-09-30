# Changelog

All notable changes are recorded here, following [Keep a Changelog](https://keepachangelog.com/)
and [Semantic Versioning](https://semver.org/). Each milestone is a minor version.

## [Unreleased]

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
