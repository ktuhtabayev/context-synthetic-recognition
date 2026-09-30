# Architecture and method decisions (ADR log)

Every decision that shapes the method, the numbers or the architecture is recorded here. Decisions
marked **author** were taken by the author of the article (2026-09-30, answers to the M0 plan);
anything that changes numerical results is asked before it is changed (CLAUDE.md).

Status values: *accepted*, *superseded by ADR-n*.

!!! warning "Two template calculations differ from the article"
    - **θ and γ are measured from running partial class means instead of the final class means M₁
      and M₂** (ADR-002).
    - **In STEP 4 the majorizer is applied twice instead of once** (ADR-003).

    Both are switches. The default is the template setting, which reproduces the template
    workbooks exactly; the *article* preset flips both (ADR-004).

---

## ADR-001 — The Excel experiment is the specification

*Accepted, 2026-09-30.*

- The workbook
  `resources/experiments/context-synthetic-model/Context-Synthetic Model – Full Experiment [My Experiment on Heart-Disease (10, 13, 2)] - Opus 5.5.xlsx`
  specifies every computation; its cell formulas define each step and its Validation sheet values
  are acceptance tests (tolerance 1e-9).
- `docs/handoff/CONTEXT_HANDOFF.md` records the agreed decisions; where it and the design chat
  differ, the handoff wins.
- `docs/handoff/reference-engine` (with `golden_values.json`) is a test oracle, not the
  architecture.
- The template workbooks of `../hag-regularized-stacking-boosting-meta` (HAG, meta-algorithm) are
  replication targets: SET {x₃, x₆, x₁₃, x₄, x₉} and r₁…r₄ to 1e-12; the meta-algorithm template's
  new object is classified as Class 2 with B1(a₄) = ∅, B2(a₄) = {S₉, S₁₀}.
- `resources/experiments/`, `docs/handoff/` and the template project are never modified.

## ADR-002 — Switch: class centres in θ and γ (template deviation 1)

*Accepted, 2026-09-30.*

The template's cells compute |bₜ − M| with the **running** partial class sum up to row t divided
by the full class size |Kᵢ| (not by the number of rows seen so far), although the template's
formula box and the article use the final means M₁ = Σbₜ/|K1|, M₂ = Σbₜ/|K2|.

Config: `hag.centres = running | final` (default `running`).

## ADR-003 — Switch: majorizer passes in STEP 4 (template deviation 2)

*Accepted, 2026-09-30.*

The template applies the majorizer again to the already-majorized bₜ of STEP 3; the article
computes R ← R + η_q and applies the majorizer once.

Config: `hag.step4_passes = 2 | 1` (default `2`).

## ADR-004 — Canonical default: template; article as a preset (author)

*Accepted, 2026-09-30.*

The default configuration is the template setting (`running`, `2`). It reproduces the four template
workbooks and the full experiment: TUPLAM {a₆, a₃, a₁, a₂, a₄}, resubstitution 70 %, LOO 20 % on
Heart-Disease (10, 13, 2). The `article` preset (`final`, `1`) gives TUPLAM {a₆, a₁, a₂, a₄, a₅},
resubstitution 50 %, LOO 0 %. Both presets ship as `configs/template.yaml` and
`configs/article.yaml`; the CLI and the GUI flag every active template deviation.

## ADR-005 — Permitted k (author)

*Accepted, 2026-09-30.*

- Odd k = k_min, k_min + 2, …, k_max with **k_min = 3 fixed** for every dataset and
  **k_max = 2·min_i|K_i| − 3**, class sizes counted from the training objects of the current fit
  (each cross-validation fold has its own range). Never hard-coded.
- **k = 1 is permitted when min|Kᵢ| = 2** (then k_max = 1; this is the only case where k_min drops).
- min|Kᵢ| ≤ 1: no k is permitted. `fit` raises `ModelUndefinedError` (Definition 1). In
  cross-validation, a fold in that state predicts **0 (refusal)** for its held-out objects — counted
  as errors in accuracy — and is flagged in the report.
- The article's text r = |{k odd, k ≤ min(|K1|, |K2|)}| is to be corrected to this rule.

## ADR-006 — Literal k range by default, optional cap and step (author)

*Accepted, 2026-09-30.*

The default is the literal formula range (Heart-Disease (270, 13, 2): k = 3 … 237, r = 354). The
`formula` k strategy accepts an optional `k_max_cap` and an even `step`; both are part of the
configuration and therefore of the run manifest and its hash.

## ADR-007 — Operators with an empty feature subset are skipped (author)

*Accepted, 2026-09-30.*

An operator whose feature subset is empty for the dataset (e.g. ρ_J on Cancer, which has no nominal
features — every distance would be 0 and the tie rule would decide every neighbourhood) is dropped
with a warning; r shrinks accordingly. Config: `context.skip_empty_operators = true`.

## ADR-008 — Numerical conventions copied from the workbook

*Accepted, 2026-09-30.*

- Distances are rounded to 10 decimals (the quantitative part first, then the sum) so that equal
  distances tie exactly.
- Neighbours are ordered by (distance, original index): ties go to the smaller original index; the
  object itself is excluded.
- f = 0.5 belongs to neither q₁ = min{f > 0.5} nor q₂ = max{f < 0.5}; if one side is empty,
  G_k = 0.5. Membership is undefined ("—") for a gradation no object has.
- HAG STEP 2: u = argmax ω, first feature on ties. STEP 3: candidates in feature order; θ/γ is
  compared exactly (identical candidates give bit-identical ratios, so the implementation keeps one
  operation order for every candidate); q is the **first** minimum, and only if min θ/γ < cr1₀ = 10 —
  otherwise the grouping stops (the workbook; the template *project* falls back to the first
  candidate instead). cr1 is reset in every iteration.
- ϰ is not clamped to r − 1: when a fold has fewer synthetic features, HAG ends when P is empty.
- Scores score₁ − score₂ are rounded to 10 decimals before AUC and ROC (e.g. 0.5 − 2/3 vs 0 − 1/6).
- Class 1 is the positive class; a refusal (0) counts as an error in accuracy and is reported
  separately (coverage).

## ADR-009 — New objects are represented without their class (Theorem)

*Accepted, 2026-09-30.*

- Scale unification is fitted on the training objects only; a new object uses the training min/max
  and its values are not clipped to [0, 1].
- A new object's context is computed against the fixed training sample E with the same operators,
  rounding and tie rule; its synthetic features use only the neighbours' classes (χ₁, formula (5)).
- The model API makes this structural: `represent` and `predict` take no label argument. Leakage
  tests pass poisoned labels and check that nothing changes.
- A training object passed through the new-object path is excluded from its own context
  (leave-self-out), which reproduces its training row.
- Leave-one-out re-fits the whole pipeline (scaling, k range, Ψ(r), ω, η, HAG, meta-algorithm) on
  the other m − 1 objects. The k-NN baselines use the fold's scaling and k = 3, 5 as in the
  workbook, independent of the fold's permitted k.

## ADR-010 — Binary classification now; K classes as an extension point (author)

*Accepted, 2026-09-30.*

Formulas (1)–(6), the HAG signs (+1 for K1, −1 for K2) and the meta-algorithm are binary in the
article and stay exact. The data layer and interfaces carry arbitrary class labels; more than two
classes raise a clear error until a K-class formulation is specified.

## ADR-011 — Additional metrics: HEOM, HVDM, Gower first (author)

*Accepted, 2026-09-30.*

The Zhuravlyov operators ρ, ρ_I, ρ_J remain the default. The first additional metrics are the
heterogeneous metrics HEOM, HVDM and Gower (milestone M7); further metrics follow through the
metric registry.

## ADR-012 — First real dataset: Heart-Disease (270, 13, 2) (author)

*Accepted, 2026-09-30.*

Taken from the template project's `datasets/raw` (6 quantitative + 7 nominal features, classes
150/120). Other datasets follow through the dataset registry.

## ADR-013 — GUI toolkit: PySide6 (author)

*Accepted, 2026-09-30.*

PySide6 (official Qt binding, LGPL) instead of the template's PyQt6; the API is nearly identical, it
suits a PyInstaller release, and pytest-qt supports it. Figures use matplotlib (ADR-017).

## ADR-014 — Public repository; unpublished material stays local (author)

*Accepted, 2026-09-30.*

- Public GitHub repository `ktuhtabayev/context-synthetic-recognition`, CI on Windows and Linux.
- Committed: the experiment workbooks, `CONTEXT_HANDOFF.md`, the task prompt, the images and the
  reference engine. **Not committed** until the article is published (listed in `.gitignore`, kept
  on disk): `docs/handoff/article/` (the draft) and `docs/handoff/CHAT_TRANSCRIPT.md`.
- Protected folders are stored byte for byte (`-text` in `.gitattributes`); other text files use LF.
- No Git LFS: the binary files total about 4 MB.

## ADR-015 — Engineering conventions

*Accepted, 2026-09-30.*

- Python ≥ 3.11 (developed on 3.12, like the template), `src/` layout, setuptools, a `.venv` with
  an editable pip install (the template's workflow). Runtime dependencies are added with the
  milestone that first uses them.
- ruff (lint + format, line length 100), mypy `strict` on the whole package, pytest with
  hypothesis and coverage ≥ 90 %, pre-commit, GitHub Actions (lint, type-check, tests on Windows and
  Linux, Python 3.11–3.13, strict docs build).
- Conventional commits; one branch per milestone (`feat/m<n>-…`), merged into `main` and tagged
  `v0.<n>.0`; `CHANGELOG.md` in Keep-a-Changelog form.
- Docstrings (Google style) name the article formula or step each public function implements; the
  article's notation (ρ, α, θ, ϰ, …) is used in docstrings and messages.
- Documentation: mkdocs-material on MkDocs 1.x (`mkdocs<2`): MkDocs 2.0 removes the plugin and
  theme system Material depends on.
- Missing values: rejected with a clear error by default.
- License: MIT, as declared by the template project.
- *Amended in M2:* mypy runs with `python_version = "3.12"` because numpy ≥ 2.5 ships its stubs with
  PEP 695 `type` statements; ruff's `py311` target and the Python 3.11 test jobs keep the code
  itself 3.11-compatible.

## ADR-016 — Configuration: typed, strict, hashable

*Accepted, 2026-09-30.*

- pydantic v2 models, unknown keys are errors, instances are immutable; YAML, TOML and JSON.
- Pluggable components are `{name, params}` specs; each plug-in declares its parameter type, which
  validates `params` when the pipeline resolves it and generates the GUI form.
- The two template-vs-article switches are first-class typed fields (`hag.centres`,
  `hag.step4_passes`), described once in `config/presets.py` (`DEVIATIONS`).
- Configuration hash: SHA-256 of the canonical JSON form (sorted keys) — independent of the file
  format. Run folders: `runs/<YYYYMMDD_HHMMSS>_<hash[:8]>/` with `manifest.yaml` (configuration,
  hashes, seed, package versions, timings).
- Relative dataset paths are resolved against the configuration file's folder.

## ADR-017 — Indexing and figures

*Accepted, 2026-09-30.*

- Objects, features and synthetic features are shown 1-based (S₁, x₁, a₁) as in the workbook and
  the article; code is 0-based internally.
- matplotlib is the single figure path for the GUI and for the article exports (PNG/SVG).

## ADR-018 — Data layer: schema, formats, class order, built-in datasets

*Accepted, 2026-09-30 (M2).*

- `Dataset` is immutable and I/O-free: values X (nominal columns hold category codes), labels,
  feature types I/J, feature names (`x₁ …` by default), object ids (`S₁ …`), categories.
- **Class order:** labels are sorted (numbers numerically, text alphabetically); the first class is
  K1, the second K2. For the author's labels 1 and 2 this is the article's numbering.
- Formats (registry `LOADERS`, the names of `dataset.format`): `cs-workbook` (the experiment's
  *Dataset* sheet), `template-extended` (the template's `m, n, c` / objects / flags layout, `.csv`
  or whitespace `.dat` with decimal commas; a placeholder flag under the class column is
  accepted), `csv` (delimiter detected), `xlsx`, `parquet` (optional extra
  `context-synthetic-recognition[parquet]`, pyarrow). `auto` detects the format from the extension
  and the content. Row counts are checked strictly (header m, n, c must match the data).
- Tables with a header row: types from `dataset.feature_types`; features not listed there are
  inferred (numeric → quantitative, otherwise nominal) **with a warning**; nominal text becomes codes
  in alphabetical order of the categories.
- Content hash (run manifests): SHA-256 of names, types, object ids, labels, categories and the
  float64 values; the dataset name and the file path are not part of it (the workbook's Dataset
  sheet and the built-in CSV of the same data have the same hash).
- Built-in datasets (registry `DATASETS`, package data): `heart-disease-10` and `heart-disease-270`,
  copied from the template project's `datasets/raw` (line endings normalized to LF). Heart-Disease
  is the UCI Statlog (Heart) dataset (CC BY 4.0).

## ADR-019 — Layering and plug-in parameters

*Accepted, 2026-09-30 (M2).*

- The core imports only pure modules besides numpy: `config.models` (typed settings) and
  `data.schema` (the `Dataset` type); it performs no I/O. Loaders, validation against the workbook
  and dataset summaries live in `data.loaders` and `services`.
- Each kind of plug-in has its registry next to its implementation (`METRICS`, `NORMALIZERS`,
  `K_STRATEGIES`, `ENCODERS`, `WEIGHTS`, `LOADERS`, `DATASETS`). Parameter types are frozen pydantic
  dataclasses with `extra="forbid"` (`PARAMS_CONFIG`); `Registry.make_params` validates a spec's
  `params` and raises `ConfigError` with every problem. `csr config check` checks the plug-ins of
  the kinds implemented so far.
- `csr validate --against <workbook>` exists from M2 on and covers Steps 1–8; M3 and M4 extend its
  map (`services.validation.workbook_checks`), which the golden tests share.

## ADR-020 — Numerical implementation reproduces the workbook bit for bit

*Accepted, 2026-09-30 (M2).*

- Zhuravlyov distances add the per-feature terms in feature order (the order of the workbook's
  SUMPRODUCT and of the reference engine); the quantitative part is rounded, then the sum.
- Rounding is `numpy.round` (round half to even of x·10^d). It could differ from Excel's
  half-away-from-zero only for a value exactly halfway at the 11th decimal; the workbook's data do
  not reach that case.
- Neighbour order: stable `argsort` of every row, i.e. by (distance, original index); the target
  itself is removed from the order.
- μ and χ₁ come from cumulative sums over the ordered neighbour classes (all k in one pass).
- f, g, G, ω and η use the workbook's operation order; the terms of stability (2) are added in
  gradation order (numpy's pairwise `sum` would differ in the last bits).
- Result: Steps 1–8 are **identical** to the reference engine (not merely within 1e-9) on the
  experiment, on Heart-Disease (270, 13, 2) and on random tie-heavy datasets (property tests), and
  all 4,022 workbook cells of Steps 1–8 agree to ≤ 4.4e-15 (Excel keeps 15 significant digits).
- Distances are computed in blocks of at most 2²² pairs; the full matrices are kept in the trace.

## ADR-021 — Encoders and training-side gradations

*Accepted, 2026-09-30 (M2).*

- Formula (5) is the only synthetic-feature encoder (registry `ENCODERS`, `formula-5`). The same-class
  count μ and the bit masks read the object's own class, so they could not describe a new object
  (Theorem); they are computed as training-side gradations for formulas (1)–(4) and Task 2, not
  offered as encoders.
- Bit masks (first permitted k = most significant bit, β = 2^(number of k) − 1) are formed when at
  most 62 k are permitted (int64); with more (Heart-Disease 270: 118) the trace has none.
- Membership tables list every gradation 0 … β up to 4,096 of them, otherwise only the gradations
  that occur.
- The weight of a synthetic feature is a plug-in (`WEIGHTS`, default `omega` = ω by (4)); it
  receives ω, g and G, leaving room for the template's Criterion-1 and λ·β weights.

## ADR-022 — k strategies

*Accepted, 2026-09-30 (M2).*

`formula` (default; ADR-005/006: optional `k_max_cap` ≥ 3 and even `step`), `article` (the draft's
text, odd k = 1, 3, … ≤ min(|K1|, |K2|), for comparison only), `explicit` (a list of odd k) and
`range` (odd start, even step, inclusive stop). Every k must be odd (formula (5) then always has a
majority) and at most the number of neighbours of a training object (m − 1); otherwise the fit
raises `ModelUndefinedError`.

## ADR-023 — Spelling of the metric: Zhuravlyov (author)

*Accepted, 2026-09-30 (M2).*

The metric is written **Zhuravlyov** (Журавлёв, ё → yo) in code, configuration and documentation;
its registry name is `zhuravlyov`. The earlier spellings `zhuravlev` and `juravlev` remain aliases,
so older configuration files still load. The protected material keeps its own spelling: the
workbook's sheet *Zhuravlev Distances* (the validation map must name it exactly) and the files in
`docs/handoff/`. The preset configurations' hashes changed with the name.
