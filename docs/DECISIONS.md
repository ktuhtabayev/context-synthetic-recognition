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
  `resources/experiments/context-synthetic-model/Context-Synthetic Model – Full Experiment [Heart-Disease (10, 13, 2)] - Opus 5.5.xlsx`
  specifies every computation; its cell formulas define each step and its Validation sheet values
  are acceptance tests (tolerance 1e-9).
  *(The author renamed the workbook on 2026-10-01 and asked for one naming throughout: the file
  name was updated everywhere in the project, the handoff documents included, and the text of
  cell A3 of the sheet* Template Deviations *— which quotes the file name of the HAG template —
  was changed inside the workbook to the template's current name. Only that text changed: every
  other part of the file is identical and `csr validate` passes on all 11,631 cells.)*
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

## ADR-024 — HAG: implementation and trace

*Accepted, 2026-09-30 (M3).*

- `core.hag.hag` follows Steps 1–5 as specified in ADR-008: candidates in feature order, u = the
  first maximal weight, q = the first minimal θ/γ among those below cr1₀, no feature added if
  none is below (the grouping then stops). STEP 3 evaluates the candidates of an iteration in
  blocks of at most 2²² (object, candidate) values.
- Sums over the objects run in their original order (running sums, as the workbook's cells);
  the final means are the last running sums divided by |Kᵢ|. The majorizer step is
  b + (s·α)·ϕ(−b) with σ(−b) = 1/(1 + e^b), the reference engine's operation order (the
  workbook's `EXP(−b)/(1 + EXP(−b))` is the same number up to the last bit).
- Every iteration records R entering, the candidates, θ, γ, θ/γ, cr1, q, crit, R + η_q with one
  pass, the new R (= r_j), whether it continues and why it stops (`|TUPLAM| = ϰ`, `crit ≤ δ`,
  `P = ∅`, `no θ/γ < cr1₀`). The per-object columns of a candidate block (b, majorized b, running
  sums, the centres used, |b − M|) are recomputed on demand by `HAGResult.scan` / `scan_from`
  with the same code, so they are bit-identical to the grouping while the trace stays
  O(m·r) instead of O(ϰ·m·r·columns).
- With one STEP 4 pass the new R is the STEP 3 value of q; with two passes the majorizer is
  applied to it once more (ADR-003). r = 1 needs no iteration (p = 0).
- Against the reference engine: identical in the running mode (experiment and Heart-Disease
  (270, 13, 2)); in the final mode crit differs in the last bits, because the reference's
  `sum()` is compensated on Python ≥ 3.12. The tests compare the decisions (TUPLAM, B1/B2,
  classes) exactly and θ/γ, crit and the latent features to 1e-12, which also covers CPUs whose
  vectorised `exp` differs from the C library's in the last bit.

## ADR-025 — θ/γ = +∞ when γ = 0 (author)

*Accepted, 2026-09-30 (M3).*

HAG STEP 3 is undefined when γ = 0 (the workbook would show `#DIV/0!`, the reference engine
stops with an error). The candidate's θ/γ is set to +∞: it can never become q, as in the template
project. If no candidate is left below cr1₀ the grouping stops as usual (`no θ/γ < cr1₀`). This
does not occur on Heart-Disease; no earlier number changes.

## ADR-026 — Majorizing functions scaled to (0, 1) (author)

*Accepted, 2026-09-30 (M3).*

Registry `MAJORIZERS`: `sigmoid` (default, σ(x) = 1/(1 + e^(−x)), alias `logistic`), `tanh`
((1 + tanh x)/2 = σ(2x)), `arctan` (1/2 + arctan(x)/π) and `softsign` ((1 + x/(1 + |x|))/2).
Every built-in ϕ maps ℝ onto (0, 1), increases and has ϕ(x) + ϕ(−x) = 1, so each step α·ϕ(−b)
lies in (0, α): it pushes K1 up and K2 down, harder the further b lies on the wrong side of 0.
The raw tanh, arctan and softsign (range (−1, 1)) were rejected: their step changes sign and
would pull correctly placed objects back towards 0 — a different regularisation. Custom
majorizers register through the registry.

## ADR-027 — Meta-algorithm: gradations, decisions, refusal

*Accepted, 2026-09-30 (M3).*

- Gradations are compared for equality only, so any values work: {1, 2} of formula (5), or the
  nominal codes of the template's meta-algorithm. The latent-sign condition is strict: an object
  with dᵢⱼ = 0 falls out of both sets.
- Step 4 is a plug-in (registry `DECISION_RULES`, default `article-step-4`). It compares the
  scores exactly, as the integers |B1|·|K2| and |B2|·|K1|; for realistic sizes this equals the
  workbook's comparison of the two floating-point scores.
- Decisions use the article's codes 1 = K1, 2 = K2, 0 = refusal; the class labels (numbers or
  text) are attached separately and a refusal has no label.
- p = 0 (one TUPLAM feature) uses Step 1 only.
- The core keeps score₁ − score₂ unrounded; rounding to `evaluation.score_decimals` belongs to
  the evaluation (ADR-008).
- Queries are filtered in blocks; the sets B1(aⱼ), B2(aⱼ) of one object at every step are
  recomputed on demand (`MetaResult.steps`).

## ADR-028 — Model API: fit, represent, classify

*Accepted, 2026-09-30 (M3).*

- `core.model.fit_model(dataset, config)` fits Steps 1–11 and returns a `CSModel`: the Step 1–8
  context model, the HAG result, the meta-dataset Y = (y₀ … y_p, r₁ … r_p) and the training
  description (aᵢ, dᵢ). The HAG majorizer and the decision rule are resolved first, so a wrong
  plug-in name fails before the costly Steps 1–8.
- `represent`, `classify` and `predict` take no label (Theorem, ADR-009); `exclude` leaves a
  training object out of its own context.
- `classify_training` classifies every training object with its training row: its context
  excludes itself while it stays in the meta-algorithm's training description (sheet
  *Meta-algorithm (All Objects)*, Definition 2). It equals `classify(X, exclude=all)`.
- CLI: `csr fit` (permitted k, r, the HAG iteration by iteration, TUPLAM, training correctness)
  and `csr classify` (`--values` for a new object, `--object N` for a training object left out of
  its context) with B1/B2 at every step.

## ADR-029 — Validation of Steps 9–12 and of the template workbooks (author: copies)

*Accepted, 2026-09-30 (M3).*

- `services.validation` recognises three layouts from their sheet names: the experiment and the
  two templates. For the experiment the inputs come from the workbook itself — the *Dataset*
  sheet, α, δ, ϰ, cr1 and **both switches** from *Parameters* B19–B25, and the new object with the
  excluded training object from *Brace for Meta-algorithm* B31:N31 and B35 — so the cached values
  are always compared with the setting they were calculated with. A `--config` is used as given;
  the report notes when its HAG settings differ from the workbook's.
- The Step 9–12 map covers every computed cell of the four *Greedy upon Weight* sheets (also the
  blocks of features already in TUPLAM and the states of an iteration that was not executed),
  *Dataset for Meta-algorithm*, *Brace for Meta-algorithm*, *Meta-algorithm* and *Meta-algorithm
  (All Objects)*. Result: 10,272 cells in 19 sheets, all within 3.0e-14.
- HAG template: every candidate block, θ/γ, cr1, q, the STEP 4 blocks, r₁ … r₄ and Y — 6,131
  cells within 3.6e-15; SET {x₃, x₆, x₁₃, x₄, x₉}. Meta-algorithm template: the new object
  (4, 0, 7, 2, 0) is Class 2 with B1(a₄) = ∅, B2(a₄) = {S₉, S₁₀}.
- The two `- Opus 5.5` template workbooks (same cells as the originals) are copied byte for byte
  to `tests/data/templates/`, so CI replicates the templates (**author**); a test compares their
  SHA-256 with the originals whenever the template project is present.

## ADR-030 — Constant synthetic features and the k range on large data (author)

*Accepted, 2026-09-30 (M4).*

**Finding.** On Heart-Disease (270, 13, 2) the literal k range (k = 3 … 237, ADR-005/006) gives
synthetic features at k ≥ 233 that are constant on the training sample (aᵤ ≡ 1, so η = 0) but have
ω = 1: μ counts neighbours of the object's *own* class, and when k approaches m it separates the
classes trivially, while the class-free formula (5) collapses to the majority class. HAG STEP 2
starts from such features. Leave-one-out accuracy (whole pipeline re-fitted per fold):

| Heart-Disease 270 | template | article |
|---|---|---|
| literal k = 3 … 237 | 7.0 % | 6.7 % |
| literal, constant features skipped | 68.1 % | 62.6 % |
| `k_max_cap` 119 / 51 / 21 / 5 | 80.0 / 79.3 / 81.5 / 79.6 % | 80.7 / 80.0 / 80.7 / 80.0 % |
| the draft's rule k = 1 … 119 | 78.5 % | 81.5 % |
| k-NN vote baselines (ρ, ρ_I, ρ_J; k = 3, 5) | 76 – 81 % | |

**Decisions (author).**

- New switch `synthetic.skip_constant` (default **off**): a synthetic feature that takes one value
  on every training object is dropped with a warning and the remaining ones are renumbered
  a₁ … a_r; if every feature is constant the model is undefined (a fold then refuses). Off by
  default because the workbook keeps them — skipping would change its leave-one-out folds 3, 5
  and 6 (template 20 % → 30 %); the full fit and resubstitution are unaffected. `csr validate`
  notes when a configuration turns it on.
- The literal k range stays the default. For large data the recommended setting is
  `configs/heart-disease-270-large-data.yaml`: `skip_constant: true` and `k_max_cap: 21`
  (Heart-Disease 270: LOO 81.5 %, AUC 0.862).
- For the article: ω by formula (4) is computed from the training-side μ and is inflated at large
  k; the k range or the weight needs a statement in the text.

## ADR-031 — Evaluation conventions (the workbook's evaluation sheets)

*Accepted, 2026-09-30 (M4).*

- **Metrics** (`evaluation.metrics`): positive class P (class 1 by default,
  `evaluation.positive_class`), negative N. TP = predicted P and true P, TN = predicted N and true
  N, FP = predicted P and true N, FN = predicted N and true P; a refusal is none of them, is
  reported separately and is an error in the accuracy (TP + TN)/n. Coverage = answered/n; accuracy
  on the answered objects. Per class: precision = correct/predicted (0 if nothing was predicted),
  recall = correct/actual, F1 = 2PR/(P + R) (0 if P + R = 0); macro = mean of the two classes.
- **AUC and ROC** (`evaluation.roc`): Mann–Whitney on score₁ − score₂ rounded to
  `score_decimals` (ties ½); the ROC table lists every score in decreasing order after (+∞, 0, 0)
  with TPR and FPR at score ≥ threshold. With K2 positive the score is negated (the AUC is the
  same). Computed by sorting, O(n log n).
- **Margins** (`evaluation.margins`): b = (min K1 + max K2)/2, width = min K1 − max K2,
  mᵢ = yᵢ(dᵢ − b) with y = +1 for K1 (the HAG's sign, independent of the positive class),
  ŷ = K1 if d > b; the objects named at max K2 and min K1 are the first with that value (the
  workbook's MATCH). *Without majorizer* = y₀ + … + yⱼ, summed left to right.
- **Properties** (`core.properties`): Definition 1/Property 1 checks, Definition 4 as conflicting
  pairs (TUPLAM description and full Ψ(r)), Definition 6 as objects with equal k-neighbour sets
  per operator pair and k, identical synthetic features, ties at the k boundary (k-th and
  (k + 1)-th neighbour equally far), and the Theorem check for a training object left out of its
  own context.

## ADR-032 — Protocols and baselines

*Accepted, 2026-09-30 (M4).*

- Registry `PROTOCOLS`: `resubstitution` (the full fit classifies the training rows),
  `leave-one-out` (the whole pipeline re-fitted without each object), `stratified-k-fold`
  (`folds` = 10; every class shuffled with the seed and dealt round robin over the folds),
  `repeated-k-fold` (`folds`, `repeats`; seeds seed, seed + 1, …), `hold-out` (`test_size` = 0.3
  of every class, at least one object on each side). Nested cross-validation is left out: the
  method has no tuned hyper-parameters yet.
- A fold whose model is undefined refuses its objects (decision 0, score 0) and is flagged
  (ADR-005). A training part with one class also refuses the baselines (the class numbers of a
  one-class sample would not match). Folds report |K1|, |K2|, the permitted k, r and TUPLAM as
  operator·k (the fold's feature numbers differ from the full fit's).
- Registry `BASELINES`, evaluated on the same folds: `knn-vote` (every operator × k, k = 3 and 5
  by default as in the workbook, the fold's scaling, rounding and tie rule; a k larger than the
  available neighbours refuses; score (χ₁ − χ₂)/k) and the scikit-learn classifiers
  `logistic-regression`, `random-forest`, `svm`, `decision-tree`, `naive-bayes` (optional extra
  `[sklearn]`, also in `[dev]`): quantitative features scaled by the fold's normalizer, nominal
  ones one-hot encoded with the training part's categories, `options` passed to the estimator,
  `random_state` = the experiment seed, score P(K1) − P(K2) or the negated decision function.
- `run_protocol` takes `progress` and `cancelled` callbacks (for the GUI);
  cancelling raises `EvaluationCancelledError`.

## ADR-033 — Experiment runner and run folders

*Accepted, 2026-09-30 (M4).*

- `services.runner.run_experiment` fits the model on every object and runs every protocol of
  `evaluation.protocols` with the `evaluation.baselines`, then the margins and the properties;
  `save_run` writes `runs/<YYYYMMDD_HHMMSS>_<hash>/` with `manifest.yaml`, `results.json` (model
  summary, metrics per protocol and method, margins, properties, timings), `predictions.csv` (one
  row per protocol, method, repetition and object) and `folds.csv`.
- `services.sensitivity.switch_sensitivity` evaluates the four switch settings (sheet
  *Sensitivity (Switches)*).
- `csr run [SOURCE] [-c CONFIG] [--sensitivity] [--no-save] [--runs-dir]`; without SOURCE the
  configuration's `dataset.path` is used, which may name a built-in dataset as
  `builtin:heart-disease-270`.
- `services.configs.config_problems` checks the plug-ins of every layer (`csr config check`); the
  core's `plugin_problems` keeps to the core's kinds.

## ADR-034 — Validation of the evaluation sheets

*Accepted, 2026-09-30 (M4).*

- The experiment's map now covers every computed sheet — 29 of 33, 11,631 cells, reported in
  workbook order — adding *Margin Analysis*, *Accuracy*, *Confusion Matrix*, *Precision, Recall,
  F1 Score*, *ROC Curve & AUC*, *Leave-One-Out* (all ten folds and the six k-NN baselines),
  *Sensitivity (Switches)* (the four settings), *Model Properties*, *Validation* and the
  experiment rows of *Template Deviations*. All agree within 3.3e-11 (the Leave-One-Out sheet
  stores the engine's scores rounded to 10 decimals; everything else within 3.0e-14).
- Not compared: *Overview*, the input sheets *Dataset*, *Quantitative*, *Nominal*, and the
  template-data tables of *Template Deviations* and *Sensitivity (Switches)* — the golden tests
  check the latter against the template workbook copies (ADR-029).

## ADR-035 — The dataset folder and loading workflow (author)

*Accepted, 2026-09-30 (M5).*

**Author.** Only the raw Heart-Disease datasets for now, in the template's layout: the default
dataset Heart-Disease (10, 13, 2) as `datasets/default.dat` and `default.csv`, the raw
Heart-Disease (270, 13, 2) in `datasets/raw/Heart-Disease/` as `.dat` and `.csv`;
`datasets/synthetic/` is reserved for pre-existing synthetic datasets and not used; no "-N"
variants. Every future dataset follows the same extended layout (first row m n c; m rows of n
values and the class; last row of n type flags, 1 = quantitative, 0 = nominal).

**Design.**

- `data.catalog` discovers the folder: `default.*` at the root, `raw/**/<Name> (m, n, c).dat|.csv`
  (family sub-folders), nothing else. The formats of one stem are one dataset; the `.dat` is read
  (the `.csv` of DDos (10000, 80, 2) in the template project is a rounded copy — 24,723 values
  differ by up to 0.48 %). Ids are `<name>-<m>`, with `-<n>` when two datasets would share one.
- `csr data check` (`DatasetCatalog.check`) verifies every file: readable, header (m, n, c) equal
  to the name's, `.dat` and `.csv` identical in data (content hash), SHA-256 per file.
- Names resolve in the order: an existing file; the folder's catalogue (id, display name,
  `default`); the package's built-ins (`heart-disease-10` is the default when there is no folder).
  The folder is found from the working directory or the configuration file's folder upwards;
  `CSR_DATASETS` overrides it. `dataset.path` may be empty (the default) or an id.
- The files are stored byte for byte (`.gitattributes`: `datasets/** -text`; the pre-commit hooks
  exclude them), so checksums do not depend on the platform. The built-in package copies hold the
  same data (tested) and remain for installations without the folder.
- Synthetic-feature datasets generated by this project (normalized values, Ψ(r), Y) are written to
  run folders by the exporters (M5), not to `datasets/`.

## ADR-036 — Run folders are self-contained; `csr export` repeats the run

*Accepted, 2026-10-01 (M5).*

- `save_run` also writes `dataset.json` (`data.snapshot`): the dataset exactly as it was used —
  values, labels, feature names and types, object ids, categories, and its content hash. Floats
  are written with `repr`, so the snapshot has the hash of the dataset it was taken from.
- `results.json` holds the four switch settings under `sensitivity` when they were evaluated.
- `services.runner.load_run` repeats a run from its folder: the manifest's configuration on the
  snapshot. The computation is deterministic, so the repeated run has the complete trace the
  exporters need, and the folder stays small (it stores results, not every intermediate table).
  The dataset's hash must equal the manifest's (`RunError` otherwise); a folder of version 0.4
  (no snapshot) falls back to the configuration's `dataset.path`; differences between the stored
  and the repeated results (another package version) are returned as warnings and printed.
- CLI: `csr run … --export FORMATS` exports into the new run folder; `csr export [RUN]` exports a
  saved run (the latest by default; `--out` writes elsewhere; `--object`, `--values`,
  `--sensitivity/--no-sensitivity`, `--decimals`, `--dpi`, `--theme`).

## ADR-037 — The Excel mirror: values at the experiment's addresses, a generated layout

*Accepted, 2026-10-01 (M5).*

**Context.** The author asked for a workbook that mirrors the experiment's sheet structure and
style. The experiment is live formulas for one shape (10 objects, 13 features, 3 operators,
k ∈ {3, 5}, 4 HAG iterations); the software runs on any data.

**Decision.**

- The mirror (`export.excel`, `workbook.xlsx`) holds **values, no formulas**: every cell is a
  number or a text computed by the package. The file reads the same in Excel, LibreOffice, pandas
  and `csr validate`, and cannot disagree with the run it documents. Texts that describe live
  formulas ("type a value …", "the engine recomputes …") are reworded for a static export.
- The layout is **generated**: one *Sorted Neighbors* sheet per base operator, one *Greedy upon
  Weight* sheet per HAG iteration (min(ϰ, r) − 1 of them; an iteration the grouping did not reach
  is shown as "not executed", as in the workbook), table columns per permitted k and per
  synthetic feature, y₀ … of the meta-dataset for max(min(ϰ, r), |TUPLAM|) positions, a fold sheet
  per hold-out protocol; *Quantitative*, *Nominal* and *Sensitivity (Switches)* only when the run
  has them.
- For Heart-Disease (10, 13, 2) with the default configuration **every computed cell sits at its
  original address**. Two golden tests keep it so: the export passes `csr validate` (11,631 cells
  in 29 sheets, tolerance 1e-9), and it is compared with the experiment workbook sheet by sheet —
  the same sheets in the same order, every number at the same address, the same merged ranges,
  frozen panes, tab colours, conditional formats, column widths and cell styles. The intended
  differences are listed in `tests/golden/test_mirror.py`: text where the original shows
  Office-Math drawings, the reworded texts, the spelling Zhuravlyov, and one added block on
  *Parameters* ("This run": what identifies it).
- The method tables of *Accuracy* and *Precision, Recall, F1 Score* list the CS-model under every
  protocol and the baselines under the hold-out protocols (as the experiment does); a baseline's
  resubstitution is in the CSV tables only.
- Styles are a palette of named styles (`export.excel.styles`) taken from the experiment; sheet
  builders name a style per cell (`SheetWriter`). `ExcelOptions.check_overlaps` makes a cell
  written twice an error; the tests build every shape with it, so generated tables cannot overlap.

## ADR-038 — Size limits of the mirror and of the tables

*Accepted, 2026-10-01 (M5).*

A sheet per step works for hundreds of objects, not for every table: Heart-Disease (270, 13, 2)
with the literal k range has 118 permitted k, r = 354 and 3 × 270 × 270 distances.

- `ExcelOptions.max_matrix_cells` (250,000): a larger object × object table (distances, ranks) is
  replaced by a note that names the CSV table holding it.
- `ExcelOptions.max_sheet_cells` (150,000): the budget of a sheet's repeated blocks — neighbour
  blocks (ranks up to the largest permitted k and one beyond, for as many objects as fit), HAG
  candidate blocks (only the chosen feature q; θ, γ and θ/γ of every candidate stay in the
  summary), B1/B2 flags per training object, the membership tables (TUPLAM features only), Task 2
  bit masks.
- Whatever is left out is said **on the sheet** and reported by the export (`ExportSummary.notes`,
  printed by the CLI). Nothing is shortened silently, and the summaries above the blocks always
  cover every object.
- CSV and JSON hold every table in full up to `ExportOptions.max_table_cells` (2,000,000);
  Markdown and LaTeX only tables that fit a page (200 rows, 20 columns); the report shortens to
  24 rows and 16 columns and says so under the table.

## ADR-039 — The mirror's sheet is named *Zhuravlyov Distances*

*Accepted, 2026-10-01 (M5).*

The project spells the name Zhuravlyov (author, M2); the experiment workbook's sheet is
*Zhuravlev Distances* and is protected source material. The mirror names the sheet *Zhuravlyov
Distances* (and *Distances* when an operator uses another metric); `csr validate` accepts both
(`services.validation.checks.SHEET_ALIASES`) and reports the sheet under the name it found. The
validation map keeps the original name, so the experiment workbook validates unchanged.

## ADR-040 — One table model for CSV, JSON, Markdown, LaTeX and the report

*Accepted, 2026-10-01 (M5).*

- `export.tables` turns a run into plain tables (`Table`: key, title, columns, rows, pipeline
  stage, note) in pipeline order — 44 for the experiment: the data, Steps 1–12, the evaluation
  per protocol, margins, sensitivity, model properties. `table_specs` knows every table's size
  before it is built.
- The renderers (`export.text`) only decide how values are written: CSV and JSON keep every digit
  (`repr`, undefined = empty / `null`); Markdown and LaTeX round (4 decimals by default, like the
  workbook's `0.0000`) and write "—". LaTeX uses `booktabs` (`longtable` beyond 40 rows) and sets
  the article's notation in math mode (a₆ → `$a_{6}$`, ρ_I → `$\rho_{I}$`).
- Text files are UTF-8 with LF line endings on every platform; SVG figures carry no time stamp
  and use a fixed id salt, the PDF is written in reportlab's invariant mode — the same run gives
  the same bytes.
- The report and the figures read the same `RunView` as the Excel mirror, so the formats cannot
  disagree. `EXPORTERS` is a registry (`excel`, `csv`, `json`, `markdown`, `latex`, `figures`,
  `html`, `pdf`; `all`), so a format can be added as a plug-in.

## ADR-041 — Figures and the run report; their dependencies

*Accepted, 2026-10-01 (M5).*

- **Figures** (`export.figures`): matplotlib's object API only (no pyplot, no display, no global
  state). Every figure is drawn by a `draw_*` function on given axes, so the GUI (M6) reuses the
  drawings. PNG (200 dpi) and SVG; light theme by default (print, the article), dark available.
- **Design rules** (kept by the tests where they can be): colour by job — K1 and K2 have the same
  two colours in every figure, magnitudes one blue ramp, the series that is the point in the
  accent and its context in grey; no dual axes; a legend whenever there are two or more series;
  correct / wrong / refused always carry a label, never colour alone; objects are named along an
  axis only up to 40. The three categorical colours in use were checked for colour-vision
  deficiencies and contrast against both surfaces.
- **⚠ The two template deviations are highlighted** in every format: a warning block first in
  the report, the HAG parameters marked ⚠, the *Template Deviations* and *Sensitivity (Switches)*
  sheets, the `summary` table and the `sensitivity` figure.
- **Report** (`export.report`): a document model (headings, paragraphs, tables, figures, code)
  rendered as one self-contained `report.html` (inline SVG, no script, nothing loaded from the
  network) and as `report.pdf` (A4; DejaVu Sans, which ships with matplotlib and has the Greek
  letters and subscripts).
- **Dependencies.** matplotlib becomes a core dependency (figures are a core deliverable and the
  GUI will embed them). reportlab is the optional extra `[pdf]` (pure Python, also in `[dev]`);
  without it `pdf` raises `ExportError` with the install hint and every other format works.

## ADR-042 — What an export shows by default: the new object and the switch settings

*Accepted, 2026-10-01 (M5).*

- **New object.** The sheets *Brace for Meta-algorithm* and *Meta-algorithm* (and the tables
  `new-object*`) demonstrate the first training object left out of its own context — the
  workbook's demonstration, which must reproduce that object's training row (Theorem).
  `--object N` chooses another training object, `--values "…"` a new one.
- **Sensitivity.** The four switch settings need a leave-one-out each. They are evaluated when
  asked (`--sensitivity`), taken from the run folder when stored there, and otherwise evaluated
  automatically only for samples of at most 60 objects (`AUTO_SENSITIVITY_OBJECTS`); without them
  the sheet, the table and the figure are left out.

## ADR-043 — The worked examples of *Template Deviations* are computed, not copied

*Accepted, 2026-10-01 (M5).*

The sheet *Template Deviations* explains both deviations on the HAG template's own data (13
original features as contribution values, 10 objects). The mirror needs those numbers for every
export, and the template project is read-only and absent in CI. The template's input —
contributions, weights, classes — and its latent features r₁ … r₄ are kept in the package
(`export/excel/hag_template.json`, taken from the workbook copy of ADR-029; a test compares the
two). Everything shown about it — θ, γ and θ/γ with running and with final centres, r₁ with one
and with two majorizer passes, SET under the four switch settings — is computed by the package's
HAG at export time, so the examples cannot drift from the code.
