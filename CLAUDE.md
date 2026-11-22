# context-synthetic-recognition — project memory for Claude Code

Python implementation (library + CLI + desktop GUI) of the context-synthetic model of recognition
algorithms (CS-model) of the author's article (an unpublished draft, kept outside the repository).

Start here: `docs/glossary.md` (the article's terms, symbols and code names), `docs/DECISIONS.md`
(ADR log; ADR-058 onwards rebuild the project on the article), `docs/architecture.md`.

Non-negotiables (details in DECISIONS.md):
- Specification = the article (ADR-058). Its stages give the structure: R-IPS → local contexts →
  synthetic features Ψ(r) → additional features (hierarchical agglomerative algorithm) →
  meta-algorithm → evaluation. Names come from `docs/glossary.md`.
- The Excel experiment workbook (`tests/data/workbooks/Excel experiment [Heart-Disease (10, 13, 2)].xlsx`)
  and the three check workbooks beside it are acceptance tests (tolerance 1e-9), run with the
  preset `excel-experiment`; `csr validate` must pass.
- Admissible k: odd, k_min = 3, by default k ≤ min|Kᵢ| of the training classes — never hard-coded.
- Distances rounded to 10 decimals; ties → smaller original index; normalization fitted on the
  training objects only (for R-IPS on E₁ = E₀ ∖ T).
- A new object is represented and classified without its class (Theorem); every protocol re-fits
  the whole pipeline per fold.
- Defaults follow the article. Four switches deviate from it when on and are flagged ⚠ in docs,
  outputs and GUI: `rips.keep_blocked_cover`, `rips.training: moved`,
  `additional_features.class_means: running`, `additional_features.majorizing_passes: 2`.
- Nothing in the project may mention or point to the author's earlier projects or their code — no
  names, paths, "ported from" or legacy identifiers in code, comments, docs, commit messages or
  file names.
- The draft article and the author's source folders (`resources/`, `docs/handoff/`: untracked,
  ADR-060) stay outside the repository and are never modified; the path of the manuscripts is never
  written into the project.
- One ADR per decision in `docs/DECISIONS.md`, in the article's terms. Ask the author before
  anything that changes numerical results. Never invent results: every number comes from a real
  run and is pinned by a test.
- The core is pure (no I/O); CLI and GUI share the services; the GUI computes nothing.
- Windows only (ADR-049); Python 3.12 `.venv`; quality gates: ruff, mypy --strict, pytest with
  coverage ≥ 90 %, `csr validate`.
