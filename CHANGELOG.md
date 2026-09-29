# Changelog

All notable changes are recorded here, following [Keep a Changelog](https://keepachangelog.com/)
and [Semantic Versioning](https://semver.org/). Each milestone is a minor version.

## [Unreleased]

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
