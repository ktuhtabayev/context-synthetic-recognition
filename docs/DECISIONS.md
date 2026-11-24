# Architecture and method decisions (ADR log)

Every decision that shapes the method, the numbers or the architecture is recorded here. Decisions
marked **author** were taken by the author of the article (2026-09-30, answers to the M0 plan);
anything that changes numerical results is asked before it is changed (CLAUDE.md).

Status values: *accepted*, *superseded by ADR-n*.

!!! warning "Switches that deviate from the article"
    - **In R-IPS a blocked object keeps covering other objects with |r(S)|**, although blocking
      makes r(S) negative — `rips.keep_blocked_cover: true` (ADR-061).
    - **The objects R-IPS moves to D form the training sample** instead of the objects that remain
      in E₁ — `rips.training: moved` (ADR-061).
    - **θ and γ are measured from running partial class means instead of the final class means M₁
      and M₂** — `additional_features.class_means: running` (ADR-002).
    - **In Step 4 of the agglomerative algorithm the majorizing function is applied twice instead
      of once** — `additional_features.majorizing_passes: 2` (ADR-003).

    The defaults follow the article: every switch is set to the article's value (ADR-062). The
    `excel-experiment` preset turns on the last two and reproduces the Excel experiment workbook
    exactly.

---

## ADR-001 — The Excel experiment is the specification

*Accepted, 2026-09-30. Superseded by ADR-058: the article is the specification; the workbook
remains an acceptance test, run with the `excel-experiment` preset. Where the source material is
kept now: ADR-060.*

- The Excel experiment workbook (now
  `tests/data/workbooks/Excel experiment [Heart-Disease (10, 13, 2)].xlsx`, ADR-060)
  specifies every computation; its cell formulas define each step and its Validation sheet values
  are acceptance tests (tolerance 1e-9).
  *(The author renamed the workbook on 2026-10-01 and asked for one naming throughout: the file
  name was updated everywhere in the project, the author's design notes included, and the text of
  cell A3 of the workbook's switches sheet — which quotes another workbook's file name, that of
  the agglomerative algorithm's check workbook — was changed inside the workbook to that file's
  current name. Only that text changed: every other part of the file is identical and
  `csr validate` passes on all 11,631 cells.)*
  *(On 2026-10-11 the author moved the experiment workbooks to another folder of the source
  material and renamed the full experiment from "Context-Synthetic Model – Full Experiment …" to
  "Context-Synthetic Recognition Model – Full Experiment …"; the paths were updated everywhere,
  the author's design notes included. Both files are byte for byte the previous ones: a copy
  saved with the article's setting of the two switches was set back to the workbook's setting.)*
- The author's design notes (kept outside the repository, ADR-060) record the agreed decisions;
  where their summary of the decisions and the transcript of the design conversation differ, the
  summary wins.
- An independent re-implementation (now `tests/golden/independent`, with `golden_values.json`)
  is a test oracle, not the architecture.
- The check workbooks of the agglomerative algorithm and of the meta-algorithm (now
  `tests/data/workbooks/Additional features [Heart-Disease (10, 13, 2)].xlsx` and
  `tests/data/workbooks/Meta-algorithm [Heart-Disease (10, 13, 2)].xlsx`) are replication
  targets: SET {x₃, x₆, x₁₃, x₄, x₉} and r₁…r₄ to 1e-12; the new object of the meta-algorithm's
  check workbook is classified as Class 2 with B1(a₄) = ∅, B2(a₄) = {S₉, S₁₀}.
- The author's source material — the experiment workbooks, the design notes and the originals of
  the check workbooks — is never modified.

## ADR-002 — Switch: class means in θ and γ

*Accepted, 2026-09-30.*

The cells of the agglomerative algorithm's check workbook compute |bₜ − M| with the **running**
partial class sum up to row t divided by the full class size |Kᵢ| (not by the number of rows seen
so far), although that workbook's formula box and the article use the final means
M₁ = Σbₜ/|K1|, M₂ = Σbₜ/|K2|.

Config: `additional_features.class_means = running | final` (default `running`; since ADR-062
the default is `final`, the article, and `running` is a switch that deviates from it — set by
the `excel-experiment` preset).

## ADR-003 — Switch: passes of the majorizing function in Step 4

*Accepted, 2026-09-30.*

The agglomerative algorithm's check workbook applies the majorizing function again to the
already-majorized bₜ of Step 3; the article computes R ← R + η_q and applies the majorizing
function once.

Config: `additional_features.majorizing_passes = 2 | 1` (default `2`; since ADR-062 the default
is `1`, the article, and `2` is a switch that deviates from it — set by the `excel-experiment`
preset).

## ADR-004 — Canonical default: the Excel experiment's setting; the article as a preset (author)

*Accepted, 2026-09-30. Superseded by ADR-062 and ADR-071: the defaults follow the article; the
presets are `default`, `alpha-0.1`, `alpha-0.2` and `excel-experiment`.*

The default configuration is the Excel experiment's setting (`running`, `2`). It reproduces the
check workbooks and the full experiment: TUPLAM {a₆, a₃, a₁, a₂, a₄}, resubstitution 70 %, LOO
20 % on Heart-Disease (10, 13, 2). The article's preset (`final`, `1`, with the same admissible k)
gives TUPLAM {a₆, a₁, a₂, a₄, a₅}, resubstitution 50 %, LOO 0 %. Both presets ship as
configuration files in `configs/` (replaced by the presets of ADR-071); the CLI and the GUI flag
every active switch that deviates from the article.

## ADR-005 — Admissible k (author)

*Accepted, 2026-09-30. Superseded by ADR-063 as the default: the article's odd k = 3 …
min(|K1|, |K2|) (strategy `min-class`) is the default; this rule remains available as the
strategy `twice-min-class`.*

- Odd k = k_min, k_min + 2, …, k_max with **k_min = 3 fixed** for every dataset and
  **k_max = 2·min_i|K_i| − 3**, class sizes counted from the training objects of the current fit
  (each cross-validation fold has its own range). Never hard-coded.
- **k = 1 is admissible when min|Kᵢ| = 2** (then k_max = 1; this is the only case where k_min
  drops).
- min|Kᵢ| ≤ 1: no k is admissible. `fit` raises `ModelUndefinedError` (Definition 1). In
  cross-validation, a fold in that state predicts **0 (refusal)** for its held-out objects — counted
  as errors in accuracy — and is flagged in the report.
- The article's text r = |{k odd, k ≤ min(|K1|, |K2|)}| is to be corrected to this rule.

## ADR-006 — Literal k range by default, optional cap and step (author)

*Accepted, 2026-09-30. Superseded by ADR-063 as the default: the article's odd k = 3 …
min(|K1|, |K2|) (strategy `min-class`) is the default; the literal range of ADR-005 remains
available as the strategy `twice-min-class`, with the cap and the step.*

The default is the literal formula range (Heart-Disease (270, 13, 2): k = 3 … 237, r = 354). The
k strategy of this rule (`twice-min-class`) accepts an optional `k_max_cap` and an even `step`;
both are part of the configuration and therefore of the run manifest and its hash.

## ADR-007 — Operators with an empty feature subset are skipped (author)

*Accepted, 2026-09-30.*

An operator whose feature subset is empty for the dataset (e.g. ρ_J on Cancer, which has no nominal
features — every distance would be 0 and the tie rule would decide every neighbourhood) is dropped
with a warning; r shrinks accordingly. Config: `local_context.skip_empty_operators = true`.

## ADR-008 — Numerical conventions copied from the workbook

*Accepted, 2026-09-30.*

- Distances are rounded to 10 decimals (the quantitative part first, then the sum) so that equal
  distances tie exactly.
- Neighbours are ordered by (distance, original index): ties go to the smaller original index; the
  object itself is excluded.
- f = 0.5 belongs to neither q₁ = min{f > 0.5} nor q₂ = max{f < 0.5}; if one side is empty,
  G_k = 0.5. Membership is undefined ("—") for a gradation no object has.
- Agglomerative algorithm, Step 2: u = argmax ω, first feature on ties. Step 3: candidates in
  feature order; θ/γ is compared exactly (identical candidates give bit-identical ratios, so the
  implementation keeps one operation order for every candidate); q is the **first** minimum, and
  only if min θ/γ < cr1₀ = 10 — otherwise the grouping stops (as in the workbook; falling back to
  the first candidate instead was not adopted). cr1 is reset in every iteration.
- ϰ is not clamped to r − 1: when a fold has fewer synthetic features, the agglomerative
  algorithm ends when P is empty.
- Scores score₁ − score₂ are rounded to 10 decimals before AUC and ROC (e.g. 0.5 − 2/3 vs 0 − 1/6).
- Class 1 is the positive class; a refusal (0) counts as an error in accuracy and is reported
  separately (coverage).

## ADR-009 — New objects are represented without their class (Theorem)

*Accepted, 2026-09-30.*

- Normalization is fitted on the training objects only; a new object uses the training min/max
  and its values are not clipped to [0, 1].
- A new object's context is computed against the fixed training sample E with the same operators,
  rounding and tie rule; its synthetic features use only the neighbours' classes (χ₁, formula (5)).
- The model API makes this structural: `represent` and `predict` take no label argument. Leakage
  tests pass poisoned labels and check that nothing changes.
- A training object passed through the new-object path is excluded from its own context
  (leave-self-out), which reproduces its training row.
- Leave-one-out re-fits the whole pipeline (scaling, k range, Ψ(r), ω, η, the agglomerative
  algorithm, the meta-algorithm) on the other m − 1 objects. The k-NN baselines use the fold's
  scaling and k = 3, 5 as in the workbook, independent of the fold's admissible k.

## ADR-010 — Binary classification now; K classes as an extension point (author)

*Accepted, 2026-09-30.*

Formulas (1)–(6), the signs of the agglomerative algorithm (+1 for K1, −1 for K2) and the
meta-algorithm are binary in the article and stay exact. The data layer and interfaces carry
arbitrary class labels; more than two classes raise a clear error until a K-class formulation is
specified.

## ADR-011 — Additional metrics: HEOM, HVDM, Gower first (author)

*Accepted, 2026-09-30.*

The Zhuravlyov operators ρ, ρ_I, ρ_J remain the default. The first additional metrics are the
heterogeneous metrics HEOM, HVDM and Gower (milestone M7); further metrics follow through the
metric registry.

*Amended in M7 (ADR-051):* HEOM, Gower and the other planned metrics were added; HVDM was left
out by the author because it reads the classes of the training objects.

## ADR-012 — First real dataset: Heart-Disease (270, 13, 2) (author)

*Accepted, 2026-09-30.*

Supplied by the author as a raw data file (6 quantitative + 7 nominal features, classes
150/120). Other datasets follow through the dataset registry.

## ADR-013 — GUI toolkit: PySide6 (author)

*Accepted, 2026-09-30.*

PySide6 (official Qt binding, LGPL) rather than PyQt6; the API is nearly identical, it suits a
PyInstaller release, and pytest-qt supports it. Figures use matplotlib (ADR-017).

## ADR-014 — Public repository; unpublished material stays local (author)

*Accepted, 2026-09-30. See ADR-060: the author's source material is no longer in the
repository.*

- Public GitHub repository `ktuhtabayev/context-synthetic-recognition`, CI on Windows.
- Committed: the experiment workbooks, the author's design notes (the summary of the decisions,
  the task description and the images) and an independent re-implementation used as a test
  oracle. **Not committed** until the article is published (listed in `.gitignore`, kept on
  disk): the article draft and the transcript of the design conversation.
- Protected folders are stored byte for byte (`-text` in `.gitattributes`); other text files use LF.
- No Git LFS: the binary files total about 4 MB.
- *Amended 2026-10-11 (author):* a copy of the article draft, the step-by-step guides, a related
  article with its program and datasets, and the workbook of another algorithm
  (Heart-Disease (7, 13, 2)) are committed to the public repository at the author's request. The
  draft and the transcript kept with the design notes stay held back. The binary and data files
  now total about 8 MB, still without Git LFS.

## ADR-015 — Engineering conventions

*Accepted, 2026-09-30.*

- Python ≥ 3.11 (developed on 3.12), `src/` layout, setuptools, a `.venv` with an editable pip
  install. Runtime dependencies are added with the milestone that first uses them.
- ruff (lint + format, line length 100), mypy `strict` on the whole package, pytest with
  hypothesis and coverage ≥ 90 %, pre-commit, GitHub Actions (lint, type-check, tests on Windows,
  Python 3.11–3.13, strict docs build).
- Conventional commits; one branch per milestone (`feat/m<n>-…`), merged into `main` and tagged
  `v0.<n>.0`; `CHANGELOG.md` in Keep-a-Changelog form.
- Docstrings (Google style) name the article formula or step each public function implements; the
  article's notation (ρ, α, θ, ϰ, …) is used in docstrings and messages.
- Documentation: mkdocs-material on MkDocs 1.x (`mkdocs<2`): MkDocs 2.0 removes the plugin and
  theme system Material depends on.
- Missing values: rejected with a clear error by default.
- License: MIT.
- *Amended in M2:* mypy runs with `python_version = "3.12"` because numpy ≥ 2.5 ships its stubs with
  PEP 695 `type` statements; ruff's `py311` target and the Python 3.11 test jobs keep the code
  itself 3.11-compatible.

## ADR-016 — Configuration: typed, strict, hashable

*Accepted, 2026-09-30.*

- pydantic v2 models, unknown keys are errors, instances are immutable; YAML, TOML and JSON.
- Pluggable components are `{name, params}` specs; each plug-in declares its parameter type, which
  validates `params` when the pipeline resolves it and generates the GUI form.
- The two switches that deviate from the article are first-class typed fields
  (`additional_features.class_means`, `additional_features.majorizing_passes`), described once in
  `config/presets.py` (`DEVIATIONS`).
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
  *Dataset* sheet), `extended` (the extended `m, n, c` / objects / flags layout, `.csv`
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
  copied from the author's raw data files (line endings normalized to LF). Heart-Disease is the
  UCI Statlog (Heart) dataset (CC BY 4.0).

## ADR-019 — Layering and plug-in parameters

*Accepted, 2026-09-30 (M2).*

- The core imports only pure modules besides numpy: `config.models` (typed settings) and
  `data.schema` (the `Dataset` type); it performs no I/O. Loaders, validation against the workbook
  and dataset summaries live in `data.loaders` and `services`.
- Each kind of plug-in has its registry next to its implementation (`METRICS`, `NORMALIZERS`,
  `K_STRATEGIES`, `ENCODERS`, `INFORMATIVENESS`, `LOADERS`, `DATASETS`). Parameter types are
  frozen pydantic dataclasses with `extra="forbid"` (`PARAMS_CONFIG`); `Registry.make_params`
  validates a spec's `params` and raises `ConfigError` with every problem. `csr config check`
  checks the plug-ins of the kinds implemented so far.
- `csr validate --against <workbook>` exists from M2 on and covers Steps 1–8; M3 and M4 extend its
  map (`services.validation.workbook_checks`), which the golden tests share.

## ADR-020 — Numerical implementation reproduces the workbook bit for bit

*Accepted, 2026-09-30 (M2).*

- Zhuravlyov distances add the per-feature terms in feature order (the order of the workbook's
  SUMPRODUCT and of the independent re-implementation used as a test oracle); the quantitative
  part is rounded, then the sum.
- Rounding is `numpy.round` (round half to even of x·10^d). It could differ from Excel's
  half-away-from-zero only for a value exactly halfway at the 11th decimal; the workbook's data do
  not reach that case.
- Neighbour order: stable `argsort` of every row, i.e. by (distance, original index); the target
  itself is removed from the order.
- μ and χ₁ come from cumulative sums over the ordered neighbour classes (all k in one pass).
- f, g, G, ω and η use the workbook's operation order; the terms of stability (2) are added in
  gradation order (numpy's pairwise `sum` would differ in the last bits).
- Result: Steps 1–8 are **identical** to the independent re-implementation (not merely within
  1e-9) on the experiment, on Heart-Disease (270, 13, 2) and on random tie-heavy datasets
  (property tests), and all 4,022 workbook cells of Steps 1–8 agree to ≤ 4.4e-15 (Excel keeps 15
  significant digits).
- Distances are computed in blocks of at most 2²² pairs; the full matrices are kept in the trace.

## ADR-021 — Encoders and training-side gradations

*Accepted, 2026-09-30 (M2).*

- Formula (5) is the only synthetic-feature encoder (registry `ENCODERS`, `formula-5`). The
  same-class count μ and the majority bit chains read the object's own class, so they could not
  describe a new object (Theorem); they are computed as training-side gradations for
  formulas (1)–(4) and Task 2, not offered as encoders.
- Majority bit chains (first admissible k = most significant bit, β = 2^(number of k) − 1) are
  formed when at most 62 k are admissible (int64); with more (Heart-Disease 270: 118) the trace
  has none.
- Membership tables list every gradation 0 … β up to 4,096 of them, otherwise only the gradations
  that occur.
- The weight of a synthetic feature is a plug-in (`INFORMATIVENESS`, default `formula-4` = ω by
  (4), alias `omega`); it receives ω, g and G, leaving room for further weights such as
  Criterion-1 and λ·β.

## ADR-022 — k strategies

*Accepted, 2026-09-30 (M2). Its default is superseded by ADR-063: `min-class`, the article's odd
k = 3 … min(|K1|, |K2|).*

`twice-min-class` (the default then; ADR-005/006: optional `k_max_cap` ≥ 3 and even `step`), a
strategy for the draft's text (odd k = 1, 3, … ≤ min(|K1|, |K2|), for comparison only),
`explicit` (a list of odd k) and `range` (odd start, even step, inclusive stop). Every k must be
odd (formula (5) then always has a majority) and at most the number of neighbours of a training
object (m − 1); otherwise the fit raises `ModelUndefinedError`.

## ADR-023 — Spelling of the metric: Zhuravlyov (author)

*Accepted, 2026-09-30 (M2). See ADR-060 for where the protected material is kept now.*

The metric is written **Zhuravlyov** (Журавлёв, ё → yo) in code, configuration and documentation;
its registry name is `zhuravlyov`. The earlier spellings `zhuravlev` and `juravlev` remain aliases,
so older configuration files still load. The protected material keeps its own spelling: the
workbook's sheet *Zhuravlev Distances* (the validation map must name it exactly) and the files of
the author's design notes. The preset configurations' hashes changed with the name.

## ADR-024 — The hierarchical agglomerative algorithm: implementation and trace

*Accepted, 2026-09-30 (M3).*

- `core.additional_features.agglomerative` follows Steps 1–5 as specified in ADR-008: candidates
  in feature order, u = the first maximal weight, q = the first minimal θ/γ among those below
  cr1₀, no feature added if none is below (the grouping then stops). Step 3 evaluates the
  candidates of an iteration in blocks of at most 2²² (object, candidate) values.
- Sums over the objects run in their original order (running sums, as the workbook's cells);
  the final means are the last running sums divided by |Kᵢ|. The step of the majorizing function
  is b + (s·α)·φ(−b) with σ(−b) = 1/(1 + e^b), the operation order of the independent
  re-implementation used as a test oracle (the workbook's `EXP(−b)/(1 + EXP(−b))` is the same
  number up to the last bit).
- Every iteration records R entering, the candidates, θ, γ, θ/γ, cr1, q, crit, R + η_q with one
  pass, the new R (= r_j), whether it continues and why it stops (`|TUPLAM| = ϰ`, `crit ≤ δ`,
  `P = ∅`, `no θ/γ < cr1₀`). The per-object columns of a candidate block (b, majorized b, running
  sums, the class means used, |b − M|) are recomputed on demand by `AgglomerativeResult.scan` /
  `scan_from` with the same code, so they are bit-identical to the grouping while the trace stays
  O(m·r) instead of O(ϰ·m·r·columns).
- With one pass in Step 4 the new R is the Step 3 value of q; with two passes the majorizing
  function is applied to it once more (ADR-003). r = 1 needs no iteration (p = 0).
- Against the independent re-implementation: identical in the running mode (experiment and
  Heart-Disease (270, 13, 2)); in the final mode crit differs in the last bits, because its
  `sum()` is compensated on Python ≥ 3.12. The tests compare the decisions (TUPLAM, B1/B2,
  classes) exactly and θ/γ, crit and the additional features to 1e-12, which also covers CPUs
  whose vectorised `exp` differs from the C library's in the last bit.

## ADR-025 — θ/γ = +∞ when γ = 0 (author)

*Accepted, 2026-09-30 (M3).*

Step 3 of the agglomerative algorithm is undefined when γ = 0 (the workbook would show
`#DIV/0!`, the independent re-implementation stops with an error). The candidate's θ/γ is set
to +∞: it can never become q. If no candidate is left below cr1₀ the grouping stops as usual
(`no θ/γ < cr1₀`). This does not occur on Heart-Disease; no earlier number changes.

## ADR-026 — Majorizing functions scaled to (0, 1) (author)

*Accepted, 2026-09-30 (M3).*

Registry `MAJORIZING_FUNCTIONS`: `sigmoid` (default, σ(x) = 1/(1 + e^(−x)), alias `logistic`),
`tanh` ((1 + tanh x)/2 = σ(2x)), `arctan` (1/2 + arctan(x)/π) and `softsign`
((1 + x/(1 + |x|))/2). Every built-in φ maps ℝ onto (0, 1), increases and has φ(x) + φ(−x) = 1,
so each step α·φ(−b) lies in (0, α): it pushes K1 up and K2 down, harder the further b lies on
the wrong side of 0. The raw tanh, arctan and softsign (range (−1, 1)) were rejected: their step
changes sign and would pull correctly placed objects back towards 0 — a different
regularisation. Custom majorizing functions register through the registry.

## ADR-027 — Meta-algorithm: gradations, decisions, refusal

*Accepted, 2026-09-30 (M3).*

- Gradations are compared for equality only, so any values work: {1, 2} of formula (5), or the
  nominal codes of the meta-algorithm's check workbook. The sign condition on the additional
  features is strict: an object with dᵢⱼ = 0 falls out of both sets.
- Step 4 is a plug-in (registry `DECISION_RULES`, default `step-4`). It compares the
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
  context model, the result of the agglomerative algorithm, the meta-dataset
  Y = (y₀ … y_p, r₁ … r_p) and the training description (aᵢ, dᵢ). The majorizing function and the
  decision rule are resolved first, so a wrong plug-in name fails before the costly Steps 1–8.
- `represent`, `classify` and `predict` take no label (Theorem, ADR-009); `exclude` leaves a
  training object out of its own context.
- `classify_training` classifies every training object with its training row: its context
  excludes itself while it stays in the meta-algorithm's training description (sheet
  *Meta-algorithm (All Objects)*, Definition 2). It equals `classify(X, exclude=all)`.
- CLI: `csr fit` (admissible k, r, the agglomerative algorithm iteration by iteration, TUPLAM,
  training correctness) and `csr classify` (`--values` for a new object, `--object N` for a
  training object left out of its context) with B1/B2 at every step.

## ADR-029 — Validation of Steps 9–12 and of the check workbooks (author: copies)

*Accepted, 2026-09-30 (M3). See ADR-060 for where the copies are kept now.*

- `services.validation` recognises three layouts from their sheet names: the experiment and the
  two check workbooks. For the experiment the inputs come from the workbook itself — the *Dataset*
  sheet, α, δ, ϰ, cr1 and **both switches** from *Parameters* B19–B25, and the new object with the
  excluded training object from *Brace for Meta-algorithm* B31:N31 and B35 — so the cached values
  are always compared with the setting they were calculated with. A `--config` is used as given;
  the report notes when its settings of the agglomerative algorithm differ from the workbook's.
- The Step 9–12 map covers every computed cell of the workbook's four step sheets of the
  agglomerative algorithm (also the blocks of features already in TUPLAM and the states of an
  iteration that was not executed), *Dataset for Meta-algorithm*, *Brace for Meta-algorithm*,
  *Meta-algorithm* and *Meta-algorithm (All Objects)*. Result: 10,272 cells in 19 sheets, all
  within 3.0e-14.
- Check workbook of the agglomerative algorithm: every candidate block, θ/γ, cr1, q, the Step 4
  blocks, r₁ … r₄ and Y — 6,131 cells within 3.6e-15; SET {x₃, x₆, x₁₃, x₄, x₉}. Check workbook
  of the meta-algorithm: the new object (4, 0, 7, 2, 0) is Class 2 with B1(a₄) = ∅,
  B2(a₄) = {S₉, S₁₀}.
- The two check workbooks (in the versions whose cells equal the originals') are copied byte for
  byte into `tests/data/` (now `tests/data/workbooks/`, ADR-060), so CI replicates them
  (**author**); a test compared their SHA-256 with the originals whenever the originals were
  present.

## ADR-030 — Constant synthetic features and the k range on large data (author)

*Accepted, 2026-09-30 (M4). Its default k range is superseded by ADR-063.*

**Finding.** On Heart-Disease (270, 13, 2) the literal k range (k = 3 … 237, ADR-005/006) gives
synthetic features at k ≥ 233 that are constant on the training sample (aᵤ ≡ 1, so η = 0) but have
ω = 1: μ counts neighbours of the object's *own* class, and when k approaches m it separates the
classes trivially, while the class-free formula (5) collapses to the majority class. Step 2 of
the agglomerative algorithm starts from such features. Leave-one-out accuracy (whole pipeline
re-fitted per fold):

| Heart-Disease 270 | Excel experiment's setting | article |
|---|---|---|
| literal k = 3 … 237 | 7.0 % | 6.7 % |
| literal, constant features skipped | 68.1 % | 62.6 % |
| `k_max_cap` 119 / 51 / 21 / 5 | 80.0 / 79.3 / 81.5 / 79.6 % | 80.7 / 80.0 / 80.7 / 80.0 % |
| the draft's rule k = 1 … 119 | 78.5 % | 81.5 % |
| k-NN vote baselines (ρ, ρ_I, ρ_J; k = 3, 5) | 76 – 81 % | |

**Decisions (author).**

- New switch `synthetic_features.skip_constant` (default **off**): a synthetic feature that takes
  one value on every training object is dropped with a warning and the remaining ones are
  renumbered a₁ … a_r; if every feature is constant the model is undefined (a fold then refuses).
  Off by default because the workbook keeps them — skipping would change its leave-one-out folds
  3, 5 and 6 (Excel experiment's setting: 20 % → 30 %); the full fit and resubstitution are
  unaffected. `csr validate` notes when a configuration turns it on.
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
  mᵢ = yᵢ(dᵢ − b) with y = +1 for K1 (the sign of the agglomerative algorithm, independent of the
  positive class), ŷ = K1 if d > b; the objects named at max K2 and min K1 are the first with
  that value (the workbook's MATCH). *Without majorizing function* = y₀ + … + yⱼ, summed left to
  right.
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
  one-class sample would not match). Folds report |K1|, |K2|, the admissible k, r and TUPLAM as
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
  experiment rows of the workbook's switches sheet. All agree within 3.3e-11 (the Leave-One-Out
  sheet stores the engine's scores rounded to 10 decimals; everything else within 3.0e-14).
- Not compared: *Overview*, the input sheets *Dataset*, *Quantitative*, *Nominal*, and the
  tables of the switches sheet and of *Sensitivity (Switches)* that use the data of the
  agglomerative algorithm's check workbook — the golden tests check the latter against the
  copies of the check workbooks (ADR-029).

## ADR-035 — The dataset folder and loading workflow (author)

*Accepted, 2026-09-30 (M5).*

**Author.** Only the raw Heart-Disease datasets for now, in the extended layout: the default
dataset Heart-Disease (10, 13, 2) as `datasets/default.dat` and `default.csv`, the raw
Heart-Disease (270, 13, 2) in `datasets/raw/Heart-Disease/` as `.dat` and `.csv`;
`datasets/synthetic/` is reserved for pre-existing synthetic datasets and not used; no "-N"
variants. Every future dataset follows the same extended layout (first row m n c; m rows of n
values and the class; last row of n type flags, 1 = quantitative, 0 = nominal).

**Design.**

- `data.catalog` discovers the folder: `default.*` at the root, `raw/**/<Name> (m, n, c).dat|.csv`
  (family sub-folders), nothing else. The formats of one stem are one dataset; the `.dat` is read
  (a `.csv` may be a rounded copy: in the author's DDos (10000, 80, 2) files 24,723 values
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
k ∈ {3, 5}, 4 iterations of the agglomerative algorithm); the software runs on any data.

**Decision.**

- The mirror (`export.excel`, `workbook.xlsx`) holds **values, no formulas**: every cell is a
  number or a text computed by the package. The file reads the same in Excel, LibreOffice, pandas
  and `csr validate`, and cannot disagree with the run it documents. Texts that describe live
  formulas ("type a value …", "the engine recomputes …") are reworded for a static export.
- The layout is **generated**: one *Sorted Neighbors* sheet per base operator, one step sheet of
  the agglomerative algorithm per iteration (min(ϰ, r) − 1 of them; an iteration the grouping did
  not reach is shown as "not executed", as in the workbook), table columns per admissible k and per
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
with the literal k range has 118 admissible k, r = 354 and 3 × 270 × 270 distances.

- `ExcelOptions.max_matrix_cells` (250,000): a larger object × object table (distances, ranks) is
  replaced by a note that names the CSV table holding it.
- `ExcelOptions.max_sheet_cells` (150,000): the budget of a sheet's repeated blocks — neighbour
  blocks (ranks up to the largest admissible k and one beyond, for as many objects as fit),
  candidate blocks of the agglomerative algorithm (only the chosen feature q; θ, γ and θ/γ of
  every candidate stay in the summary), B1/B2 flags per training object, the membership tables
  (TUPLAM features only), Task 2 majority bit chains.
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

*Accepted, 2026-10-01 (M5). Amended by ADR-083: CSV files carry a byte-order mark.*

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
- **⚠ The two switches that deviate from the article are highlighted** in every format: a
  warning block first in the report, the parameters of the agglomerative algorithm marked ⚠, the
  switches sheet and the sheet *Sensitivity (Switches)*, the `summary` table and the
  `sensitivity` figure.
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

## ADR-043 — The worked examples of the switches sheet are computed, not copied

*Accepted, 2026-10-01 (M5). Superseded by ADR-075: the examples use the run's own data.*

The workbook's switches sheet explains both switches on the own data of the agglomerative
algorithm's check workbook (13 original features as contribution values, 10 objects). The mirror
needs those numbers for every export, and the original workbook is read-only and absent in CI.
That workbook's input — contributions, weights, classes — and its additional features r₁ … r₄ are
kept in the package as a JSON data file (taken from the workbook copy of ADR-029; a test compares
the two). Everything shown about it — θ, γ and θ/γ with running and with final class means, r₁
with one and with two passes of the majorizing function, SET under the four switch settings — is
computed by the package's agglomerative algorithm at export time, so the examples cannot drift
from the code.

## ADR-044 — Desktop application: one more interface above the services

*Accepted, 2026-10-01 (M6).*

**Context.** The author asked for an excellent desktop GUI: Dataset → Configure → Run → Explore
results → Classify a new object → Compare runs → Export. ADR-013 chose PySide6; the design keeps
patterns that have proved themselves in desktop applications of this kind: one window, a theme
dictionary shared by the style sheet and matplotlib, a worker thread, `QSettings`, document-mode
tabs.

**Decision.**

- Package `gui`, installed with the optional extra `[gui]` (`PySide6-Essentials` — Widgets, Gui,
  Core and Svg are all the application uses; the full `PySide6` also satisfies it). Nothing in the
  library or the CLI imports Qt; `csr gui` and the `csr-gui` script start the application, and
  without PySide6 `csr gui` prints the install hint.
- **The GUI computes nothing.** A run is `services.runner.run_experiment`; what the pages show is
  the exporters' `RunView`, its tables (`export.tables`) and its figures (`export.figures`). The
  screen, the Excel mirror and the article's tables therefore cannot disagree, and a table or
  figure added to the exporters appears in the results explorer without GUI code.
- **Pages share one state** (`gui.state.AppState`: dataset, configuration with its undo stack, the
  run on screen, the selected object, the appearance) and never call each other.
- **Background work** (`gui.workers`): `run_job`, `open_job` and `export_job` are plain functions
  over the services; a `Task` (`QThread`) runs one and reports progress, the result, the error or
  the cancellation through signals. The run can be cancelled between folds (`switch_sensitivity`
  takes the same `cancelled` callback as `run_protocol` for this). The package's log is shown in
  the Run page through a logging handler.
- **Look.** The Fusion style with the application's own palette and style sheet — the same on
  every platform; a light and a dark theme from one set of design tokens (`gui.theme.Palette`),
  which also names the matching figure theme; zoom steps scale the whole interface.
- **Remembered** (`QSettings`): window geometry, theme, zoom, runs folder, recent datasets,
  configurations and runs.
- **Translation.** Every user-visible string passes through `gui.i18n.tr`; a translation is a Qt
  `.qm` file `gui/translations/csr_<locale>.qm`, loaded when the system locale asks for it. The
  interface is English; no translation ships yet. *(M8: Russian and Uzbek ship, with a
  language switch in the application — ADR-054.)*

## ADR-045 — Colour semantics of the tables on screen

*Accepted, 2026-10-01 (M6).*

Tables are the exporters' `Table` objects behind a Qt model; `gui.models.style_for` decides what
every column and cell *means* (`CellRole`) and the theme turns the meaning into colours, so both
themes show the same semantics.

- From the workbook: quantitative (blue) and nominal (yellow) features, synthetic features (light
  green), additional features (stronger green), correct (green) and wrong (red), warnings.
- **Classes** are tinted in the two colours they have in every figure (K1 blue, K2 orange) —
  the workbook colours the class column with one colour; two make a table readable at a glance and
  tie tables to figures. The same tints mark aᵤ = 1 / aᵤ = 2 in Ψ(r): the class a neighbourhood
  votes for.
- **Decisions** (predicted class per method) are coloured by whether they are correct, wrong or a
  refusal — the table form of the *prediction map* figure.
- **TUPLAM** features and the chosen candidate q of an iteration of the agglomerative algorithm
  are yellow ("selected").
- Colour is never the only cue: classes and decisions are written, statuses carry ✓ / ✗ / ⚠.
- Selecting an object (a row or the selector) marks it in every table (cross-highlighting) and
  draws its neighbours with the nested k-neighbourhoods (`export.figures.draw_object_context`).
- Header tooltips name the article's formula a column comes from.

## ADR-046 — Editing the configuration in the GUI

*Accepted, 2026-10-01 (M6).*

- Every edit builds a new immutable `ExperimentConfig` from the widgets and pushes it on a
  `QUndoStack`; the widgets are then refreshed from the state, so undo, redo, the presets and an
  opened file take one path.
- **The `dataset` section belongs to the loaded dataset**: it is written when a dataset is loaded
  (path, reading options, feature types) and is never changed by undo or by a preset. Presets
  also keep the name, the seed and the output folder.
- **Parameter forms are generated** from the plug-ins' parameter types (their JSON schema): check
  box, spin box with the declared bounds, list field, JSON field. A form reports only values that
  differ from the defaults, so a configuration edited in the GUI stays as short as a hand-written
  one, keeps its hash, and keeps matching its preset.
- Protocols and baselines are check lists in the registry's order; a configuration that lists
  the same plug-in twice shows its first entry.
- Problems — invalid parameters, a k rule that admits nothing, no protocol, data the model is
  undefined for — are listed on the Configure and Run pages and keep *Run* disabled.
- After an edit the run on screen is kept and marked as computed with another configuration
  ("stale") until the experiment is run again; changing the data drops it.

## ADR-047 — Runs in the GUI: saving, opening, comparing, the new object

*Accepted, 2026-10-01 (M6).*

- Saving a run is optional on the Run page; the runs folder is one setting for the whole
  application.
- **Opening a run** repeats it from its folder (`load_run`, ADR-036) in the worker thread; its
  dataset and configuration become current.
- **Comparing** reads only `manifest.yaml` and `results.json` (`services.runs`: `list_runs`,
  `comparison`, `config_differences`) — nothing is recomputed. A result that was not saved takes
  part as *current*.
- **New object.** Classifying an object on the New object page replaces the run's demonstration
  object (`RunView.new_object`), so the *New object* tables of the results and every export show
  the object the user classified. The default is the workbook's: S₁ left out of its own context.

## ADR-048 — Testing the GUI

*Accepted, 2026-10-01 (M6).*

- pytest-qt drives the real application — load, configure, run in the worker thread, explore,
  classify, compare, export — on Qt's `offscreen` platform (set in `tests/conftest.py` before Qt
  loads): no window appears and CI needs no display.
- File dialogs and message boxes are attributes of the pages and the window, replaced in tests;
  settings go to an ini file in the test's temporary folder, never to the user's.
- The jobs behind the worker thread are tested as plain functions; the thread itself with small
  tasks; a cancelled run on Heart-Disease (270, 13, 2).
- The screenshots of the user guide are made by `scripts/gui_screenshots.py`, which drives the
  same application (native fonts on Windows, the window never mapped on screen).
- The suite turns warnings into errors, with one exemption: matplotlib's Qt toolbar reads a Qt
  attribute (`AA_UseHighDpiPixmaps`) that PySide6 marks as deprecated. The warning is raised
  inside a Qt paint callback, where an error crashes the interpreter instead of failing a test;
  `gui.canvas` ignores it for the same reason. A test paints every page in both themes.

## ADR-049 — Windows only (author)

*Accepted, 2026-10-01.*

**Decision.** The project is developed, tested and released for Windows only; no other operating
system is supported.

- CI: every job (lint and type-check, tests on Python 3.11–3.13, strict docs build) runs on
  `windows-latest`.
- Package metadata names Windows (`Environment :: Win32 (MS Windows)`,
  `Operating System :: Microsoft :: Windows`).
- `scripts/gui_screenshots.py` always uses the native Windows platform; `.gitignore` lists no
  artefacts of other systems.
- Unchanged, because they do not depend on the operating system: Qt's `offscreen` platform for the
  GUI tests (ADR-048: no window appears during a test run), LF line endings in the repository and
  in every exported text file (ADR-014, ADR-040: the same bytes and checksums in every checkout),
  forward slashes in displayed relative paths.
- The CI line of the task description among the author's design notes was changed to name
  Windows alone — the one phrase, with the author's explicit permission; nothing else in the
  protected material was touched (the design notes are now kept outside the repository,
  ADR-060).
- At the author's request the earlier entries of this log (ADR-014, ADR-015, ADR-048) and of
  `CHANGELOG.md` (0.1.0, 0.6.0) were reworded to name Windows alone; until v0.6.0 the CI also ran
  on a second operating system.

No numerical result changes.

## ADR-050 — Metric traits: domain, fit step, parameter check

*Accepted, 2026-10-02 (M7).*

**Context.** Until M7 a metric was only a function `(A, B, quantitative, decimals, params) →
distances` on the normalized values. The metrics of M7 need more: HEOM and Gower divide by the
training range of a feature, Mahalanobis needs the training covariance, Euclidean and its relatives
are meaningless on nominal codes, and feature weights must match the features of the operator.

**Decision.** A metric declares what the pipeline must know about it with
`core.local_context.metrics.traits(domain=…, fit=…, check=…)`, placed below
`@METRICS.register`; a metric that declares nothing — every plug-in written before M7 — is
heterogeneous and needs no fit.

- **Domain** (`any`, `quantitative`, `nominal`). `resolve_operators` raises a `ConfigError` that
  names the features when an operator gives a metric features of the other type, so the mistake
  is reported by `csr data info`, by the GUI's problem list and before a run — not silently
  "fixed" by dropping features. An empty subset is still skipped first (ADR-007).
- **Fit step** `(Z, quantitative, params) → state`: the metric's training statistics, estimated
  on the normalized training values of the operator's features. `BaseOperator.fit(Z)` returns a
  fitted copy and the metric then receives the state in place of its parameters; calling
  `distances` on an operator that still needs its fit is an error. `fit_context` fits the
  operators right after the normalization; the k-NN baselines fit them on the fold's training
  part. Leave-one-out therefore re-estimates the statistics in every fold, like everything else
  (ADR-009).
- **A fit step gets no class labels.** Its signature has no place for them, so a metric cannot
  make the distances of an object depend on a class — its own or anyone else's (ADR-051).
- **Check** `(quantitative, params)`: parameters that depend on the operator's features (the
  number of feature weights) are verified when the operators are resolved.
- Fitting on the *normalized* values keeps one definition for every normalizer: a metric that
  divides by the training range or whitens with the training covariance gives the same distances
  for every affine normalizer (tested), and composes meaningfully with the rank transform.

No configuration field was added: the hash of every existing configuration is unchanged.

## ADR-051 — The metrics of M7; HVDM is left out (author)

*Accepted, 2026-10-02 (M7). Amends ADR-011.*

**Author.** Add every planned metric. **Leave HVDM out**: its nominal part (the value difference
metric) is built from the class frequencies of the training sample, so the class of a training
object would enter its own distances. Only metrics that never read a class label are offered;
`hvdm` is not a registered name.

**Metrics** (x′, y′ = normalized values; I, J = the quantitative and nominal features of the
operator; n = their number):

| Name | Domain | Definition |
|---|---|---|
| `zhuravlyov` (default) | any | Σ_I \|x′ⱼ − y′ⱼ\| + Σ_J [xⱼ ≠ yⱼ] |
| `weighted-zhuravlyov` | any | Σ_I wⱼ·\|x′ⱼ − y′ⱼ\| + Σ_J wⱼ·[xⱼ ≠ yⱼ], wⱼ = type weight · feature weight |
| `heom` | any | √(Σ_I (\|x′ⱼ − y′ⱼ\| / rangeⱼ)² + Σ_J [xⱼ ≠ yⱼ]) |
| `gower` | any | (Σ_I \|x′ⱼ − y′ⱼ\| / rangeⱼ + Σ_J [xⱼ ≠ yⱼ]) / n |
| `manhattan` | I | Σ \|x′ⱼ − y′ⱼ\| |
| `euclidean` | I | √(Σ (x′ⱼ − y′ⱼ)²) |
| `chebyshev` | I | max \|x′ⱼ − y′ⱼ\| |
| `minkowski` | I | (Σ \|x′ⱼ − y′ⱼ\|ᵖ)^(1/p), p ≥ 1 (default 3) |
| `canberra` | I | Σ \|x′ⱼ − y′ⱼ\| / (\|x′ⱼ\| + \|y′ⱼ\|) |
| `cosine` | I | 1 − (x′·y′) / (‖x′‖·‖y′‖) |
| `mahalanobis` | I | √((x′ − y′)ᵀ S⁺ (x′ − y′)), S the training covariance |
| `hamming` | J | Σ [xⱼ ≠ yⱼ] |

**Conventions.**

- Every metric rounds its result to `distance_decimals` and adds its terms feature by feature in
  feature order, so equal objects give bit-identical distances and D is symmetric bit for bit
  (ADR-008). `weighted-zhuravlyov` rounds like the Zhuravlyov metric (the quantitative part, then
  the sum): with every weight 1 it is that metric bit for bit (tested).
- `rangeⱼ` is the training range of the normalized values. A feature without spread on the
  training sample contributes 0 (as min–max maps it to 0); a new value outside the training range
  is not clipped, so its term may exceed 1 (ADR-009).
- With min–max normalization `gower` is the Zhuravlyov metric divided by n: up to the rounding of
  the distances the neighbours, and therefore Ψ(r), are those of ρ. It is offered for the name
  the literature uses.
- `mahalanobis`: sample covariance (divisor m − 1) of the normalized training values; when it is
  singular — a feature without spread, collinear features, fewer objects than features — the
  Moore–Penrose inverse S⁺ is used, computed as a whitening matrix from the eigenvectors with a
  positive eigenvalue.
- `canberra`: a feature on which both values are 0 contributes 0. `cosine`: two zero vectors are
  at distance 0, a zero and a non-zero vector at distance 1; it is a dissimilarity of directions
  and does not satisfy the triangle inequality (the property tests exempt it from that axiom).
- `feature_weights` of `weighted-zhuravlyov` lists one weight per feature of the operator, in the
  operator's feature order; weights are non-negative.

The default operators ρ, ρ_I, ρ_J and every golden value are unchanged.

## ADR-052 — The normalizers of M7

*Accepted, 2026-10-02 (M7).*

**Author.** Add all six: z-score, robust, max-abs, decimal scaling, rank, unit length (per object).

| Name | Map of a quantitative feature | Training statistics |
|---|---|---|
| `minmax` (default) | (x − min) / (max − min) | min, max |
| `z-score` | (x − mean) / std | min, max, mean, std |
| `robust` | (x − median) / (Q3 − Q1) | min, max, median, Q1, Q3 |
| `max-abs` | x / max\|x\| | min, max, max \|x\| |
| `decimal-scaling` | x / 10ʲ | min, max, power j |
| `rank` | (mid-rank − 1) / (m − 1) | min, max |
| `unit-length` | x_I / ‖x_I‖₂ per object | — |
| `none` | x | — |

**Conventions.**

- Nominal codes are never changed. Constants are estimated on the training objects only and a new
  object is mapped with them (ADR-009).
- **No spread → 0.** A quantitative feature whose training values are all equal is mapped to 0 by
  every fitted normalizer, for training and new objects alike — the workbook's rule for min–max
  (`IF(MAX = MIN, 0, …)`). `unit-length` and `none` estimate nothing and have no such features.
- `z-score`: sample standard deviation (divisor m − 1) by default; parameter `ddof: 0` gives the
  population one.
- `robust`: quartiles by linear interpolation (Excel's QUARTILE.INC). If the interquartile range
  is 0 although the feature has spread (more than half of the values equal), the range max − min
  is the divisor, so the feature stays on a comparable scale.
- `decimal-scaling`: j is the smallest integer with max|x| / 10ʲ < 1 on the training sample — the
  literal definition, so j is negative for a feature whose values are all below 0.1.
- `rank`: equal training values share their mean rank; a new value is interpolated linearly
  between the two training values around it and takes the level of the nearest training value
  outside the training range. This one normalizer therefore does clip new values — a rank cannot
  leave [0, 1].
- `unit-length`: each object is divided by the Euclidean norm of its own quantitative values; an
  object whose quantitative values are all 0 stays 0. Features on large scales dominate the norm,
  so this normalizer suits data whose quantitative features share a unit.
- `Scaling` stays the fitted object of every normalizer; `RankScaling` and `UnitLengthScaling`
  override the map. Every fitted normalizer records the training `min` and `max` first, then its
  own statistics; the features table, the report and the *Normalized Dataset* sheet show them all.
  For min–max the exports are unchanged cell for cell.

## ADR-053 — No new datasets in M7 (author)

*Accepted, 2026-10-02 (M7).*

The author has further datasets — Cancer (589, 44, 2), Molecular-Biology (100, 10, 2) and
(100, 50, 2), DDos (10000, 22, 2) and (10000, 80, 2). The author decided to add none of them
now: ADR-035 stands (Heart-Disease only), nothing was copied into the repository, and the work a
10,000-object sample would need (a lean trace, a cap on k, k-fold instead of leave-one-out) is not
part of M7. Any dataset in the extended layout can still be run from its path
([Datasets](datasets.md)).

## ADR-054 — The languages of the interface: English, Russian, Uzbek (author)

*Accepted, 2026-10-02 (M8).*

**Author.** Translate the interface into Russian and Uzbek; Uzbek in Latin script; translate the
interface only, not the exports; add a language switch to the application.

**Design.**

- English is the source language in the code. A translation is `gui/translations/csr_<code>.ts`
  (Qt's translation source, the file to edit) compiled into `csr_<code>.qm` (what the application
  loads); both are committed, so an installation needs no build step. Every text is in the one
  context `csr`.
- `scripts/update_translations.py` collects the literal texts of `tr(...)` and `mark(...)` from
  the `gui` package with Python's own parser — Qt's `lupdate` would file them under class names —
  merges them into the `.ts` files and runs `pyside6-lrelease`. `mark()` names a constant that is
  translated where it is shown (page titles, the shortcut list, file filters, column names).
- The tests keep the files honest: every text translated, none stale, placeholders, menu
  accelerators and file patterns kept, and the `.qm` files saying what the `.ts` files say.
- **View → Language**: *System language* (the default — the language of Windows, else English),
  English, Русский, Oʻzbekcha, each under its own name. The choice is a setting.
- **Switching rebuilds the window.** The texts of a window are set when it is built, so
  `app.rebuild_window` builds a new `MainWindow` on the *same* `AppState` — dataset,
  configuration with its undo history, run, selected object — opens it on the same page and
  closes the old one. Nothing is computed or loaded again. While a run or an export is in
  progress the language stays as it is.
- Qt's own texts (the button of a message box) come from Qt's `qtbase_ru`; Qt has no Uzbek
  translation, so `csr_uz.ts` carries those few texts itself (context `QPlatformTheme`).
- **Not translated**, by the author's decision or because they are data: the tables and figures
  of the results and everything exported (they come from the exporters), the names and summaries
  of plug-ins (they are what a configuration file says), messages of the core. Uzbek uses ʻ
  (U+02BB) in oʻ and gʻ and ʼ (U+02BC) for the tutuq belgisi.
- The sidebar grows with the longest page name, so a language (or a zoom step) cannot cut it.

The Uzbek wording is the author's to review. Terms used: *alomat* for feature (author; not
*belgi*), *sinf* for class, *oʻrgatuvchi obyekt* for training object, *hisoblash* for a run.

## ADR-055 — The Windows executable of the desktop application (author)

*Accepted, 2026-10-02 (M8).*

**Author.** Build the desktop application with PyInstaller; build it in CI as a workflow
artifact; publish it with the GitHub Release of v0.8.0.

- `scripts/build_exe.py` freezes `csr-gui` as a **folder** (`dist\csr-gui\`, packed as
  `csr-gui-<version>-windows.zip`), not as a single file: a one-file program unpacks itself on
  every start and is more often held back by antivirus software.
- Contents: Python, numpy, matplotlib, openpyxl, Qt (PySide6-Essentials), reportlab (the PDF
  report) and the package with its data — the built-in datasets, the data of the worked examples
  (ADR-043), the translations. Left out: pyarrow (Parquet) and scikit-learn (the comparison
  classifiers), which would multiply the size; the application reports the missing extra when
  one is asked for.
  Qt's software OpenGL renderer and Qt's translations of other languages are removed (27 MB).
  The result is about 150 MB unpacked, 70 MB packed.
- **`csr-gui --self-test [REPORT]`** checks an installation without showing a window: it loads
  every language, opens the default dataset, runs the experiment in the worker thread, draws a
  table and a figure and exports every format. A windowed program has no console, so the result
  is the exit code and the report file. The build script runs it on the frozen application in an
  empty folder and fails the build if it does not pass — this is what found that matplotlib's
  SVG and PDF backends and reportlab had to be named for PyInstaller.
- CI job *Windows executable*: builds, self-tests and uploads `csr-gui-windows` (the zip and the
  self-test report) on every run. A release takes the artifact of the tag's own CI run, so the
  published executable is built from exactly the tagged commit.
- No code signing: Windows SmartScreen will warn about an unknown publisher on first start.

## ADR-056 — The documentation site on GitHub Pages (author)

*Accepted, 2026-10-02 (M8). See ADR-060: the author's design notes are no longer in the
repository.*

**Author.** Publish the docs site.

- The site is the `mkdocs build --strict` of `main`, deployed by the CI jobs *Docs build* and
  *Publish the documentation* (GitHub's Pages actions) to
  <https://ktuhtabayev.github.io/context-synthetic-recognition/>. Branches and tags build the
  site but do not publish it.
- **Only what is in the repository is published, and not the author's design notes.**
  `mkdocs.yml` excludes their folder, and the CI job fails if that folder appears in the built
  site. The site is never deployed from a working copy: on the author's disk the same folder
  also holds the unpublished article draft and the design transcript (ADR-014).

## ADR-057 — M8 housekeeping

*Accepted, 2026-10-02 (M8).*

- **Reproduction guide** (`docs/reproduction.md`, author): how to check the package against the
  workbook, obtain each result, repeat a saved run. Its numbers and outputs are those of the
  commands it shows; its Python example is run as written before a release.
- **`csr gui` help line** (author): typer's help renderer reads `[gui]` as markup and printed
  "(needs the extra )". The text no longer uses square brackets, and a test keeps brackets out
  of the first line of every command's help.
- **Wording** (author): docstrings that still announced finished milestones were corrected; the
  extension-points table of the architecture page now lists what is built in and, separately,
  what is not (the Criterion-1 and λ·β weights, nested cross-validation, HVDM).
- **Development status** (author): the package classifier is *4 - Beta*.

## ADR-058 — The article is the specification (author)

*Accepted, 2026-11-22. Supersedes ADR-001.*

- The author's article on the CS-model (the draft of 19 October 2026) specifies the method:
  R-IPS, the local contexts, the synthetic features Ψ(r) by formula (5) with membership (1),
  stability (2), boundary (3), informativeness (4) and contributions (6), the hierarchical
  agglomerative algorithm (Steps 1–5) with the additional features, the meta-algorithm and the
  accuracy of Definitions 2 and 3. The draft stays outside the repository (ADR-060).
- The Excel experiment workbook stays an acceptance test. `csr validate` reads its inputs and
  runs with the preset `excel-experiment` — the workbook's own settings: odd k = 3 … 2·min|Kᵢ| − 3,
  running class means, two majorizing passes (ADR-002, ADR-003, ADR-063). 11,622 cells in 579
  checks agree to 1e-9. Nine cells fewer than before (11,631) are compared: cells whose text only
  worded the old names of the switches.
- Where the article and the workbook differ, the article is the default and the workbook's choice
  stays available as a switch or a strategy, flagged ⚠ wherever it is on.
- Where the article is silent or contradicts itself, the choice is recorded here and flagged
  (ADR-061, ADR-063, ADR-064, ADR-067, ADR-068).

## ADR-059 — The article's names; configuration schema 2 (author)

*Accepted, 2026-11-22.*

- Every name follows the article through `docs/glossary.md` (Russian term → symbol → English →
  code). The core is ordered by the stages of the method: `core/rips`, `core/local_context`,
  `core/synthetic_features`, `core/additional_features`, `core/meta_algorithm`; `core/model.py`
  puts them together. Configuration sections, CLI output, the Excel mirror, the tables, the
  figures, the report and the desktop application use the same names.
- The configuration has `schema_version: 2` with the sections `dataset`, `rips`,
  `local_context`, `admissible_k`, `synthetic_features`, `additional_features`,
  `meta_algorithm`, `evaluation`, `seed`, `output`. A file of schema 1 is rejected with a message;
  nothing converts it (*author: no reader, a clean break*).
- The workbooks' own sheet names are quoted in one module
  (`services/validation/sheets.py`); the Excel mirror names its sheets in the article's terms
  (for example *Additional Feature j*, *Ψ(r) Contributions*, *Meta-description*, *New Object*,
  *Switches*, *Switch Sensitivity*, *Zhuravlyov Distances*), and a map between the two lets one
  validation map check the workbook and an exported mirror alike.
- The renaming and the restructuring changed no number: five cases — the experiment and
  Heart-Disease (270, 13, 2) under both former settings, leave-one-out included, and the
  large-data configuration — were compared field by field before and after: no difference.

## ADR-060 — The author's source material stays outside the repository (author)

*Accepted, 2026-11-22.*

- The folders `resources/` and `docs/handoff/` are no longer tracked: they stay on the author's
  disk, ignored by `.gitignore`. The Git history is unchanged. (Later the same day, at the author's
  request, the GitHub Release v0.8.0 and its tag `v0.8.0` were deleted; the 0.8.0 code stays in the
  history. On 2026-11-24, after v0.10.0 was published, the GitHub Release v0.9.0, its download and
  its tag `v0.9.0` were deleted the same way, at the author's request: v0.10.0 is the only published
  release.)
- The four workbooks used as acceptance tests are copied byte for byte to `tests/data/workbooks/`
  under neutral names: *Excel experiment*, *Synthetic features by k-NN*, *Additional features* and
  *Meta-algorithm* (each `[Heart-Disease (10, 13, 2)].xlsx`); a test pins their SHA-256 checksums.
- The independent re-implementation used as a test oracle lives in `tests/golden/independent/`,
  with `tests/golden/golden_values.json` (*author: keep the cross-check*).
- The draft article is not copied into the repository, and the folder it is kept in is not named
  in the project.

## ADR-061 — R-IPS: forming the precedent base for training and testing (author)

*Accepted, 2026-11-22.*

R-IPS is the first stage (`core/rips`, configuration section `rips`). For the working set
E₁ = E₀ ∖ T — T an optional test sample given by object ids, empty by default (*author: over E₁,
without T*): r(S) is the distance to the nearest object of the other class in E₁, g(S) the number
of own-class objects of E₁ strictly inside that hypersphere (S included). Steps 2–6 queue E₁ by
1/r(S), take the first object Sᵢ, form L, block Sᵢ (r(Sᵢ) = −r(Sᵢ)), move Sᵢ to D when |L| = 1, and
otherwise lower g(Sⱼ) by one for every Sⱼ ∈ L and move Sⱼ to D at g(Sⱼ) = 1; the output is D.

Conventions where the article is silent (flagged ⚠ in the code and the outputs):

- equal radii are queued by the smaller index; an object with r(S) = 0 never enters the queue;
  |L| = 0 leaves Sᵢ blocked in E₁;
- **blocking** is read literally: a blocked object has a negative radius and covers nobody in
  Step 4 (default). The switch `rips.keep_blocked_cover` (⚠) lets it cover with |r(S)|;
- **the training sample** is the set of objects that remain in E₁; the rest of E₀ is the test
  sample (default, `rips.training: remaining`). The switch `rips.training: moved` (⚠) takes D as the
  training sample instead;
- the quantitative features are normalized with constants estimated on E₁, so that T cannot shape
  the selection; distances are rounded like those of the local contexts.

Reproduced (pinned in `tests/test_rips.py`; Zhuravlyov metric on all features, min–max
normalization):

| Data | D | remaining in E₁ | remaining, blocked objects keep covering |
|---|---|---|---|
| Heart-Disease (270, 13, 2) | 108 (60, 48) | 162 (90, 72) | 77 (37, 40) |
| Heart, slope x₁₁ quantitative (ADR-072) | 109 (61, 48) | 161 (89, 72) | 73 (36, 37) |
| German (1000, 20, 2) | 476 (325, 151) | 524 (375, 149) | 335 (195, 140) |

The article prints training samples of 76 (36, 40) objects for Heart and 336 (196, 140) for
German. No reading of the algorithm reproduces them exactly; the object lists of the article's
experiments (kept by the author outside the repository) are closest to the reading in which
blocked objects keep covering, but differ in single objects. The gap is reported, not tuned
(ADR-070).

## ADR-062 — The agglomerative algorithm follows the article by default (author)

*Accepted, 2026-11-22. Supersedes ADR-004 for the defaults.*

- Defaults: θ and γ measured from the final class means M₁, M₂; one majorizing pass in Step 4
  (R ← R + η_q, then φ once). The Excel experiment's calculations stay as switches:
  `additional_features.class_means: running` (⚠, ADR-002) and
  `additional_features.majorizing_passes: 2` (⚠, ADR-003); the preset `excel-experiment` turns on
  both.
- The change of the defaults (together with ADR-063) changes the results; the numbers before
  (preset `excel-experiment`) and after (default configuration) are pinned in
  `tests/test_defaults.py`:

  | Data | Setting | Admissible k | r | Resubstitution | Leave-one-out |
  |---|---|---|---|---|---|
  | Heart-Disease (10, 13, 2) | before | 3, 5 | 6 | 70 % | 20 % (4 refusals) |
  | Heart-Disease (10, 13, 2) | after | 3 | 3 | 50 % | 0 % (5 refusals) |
  | Heart-Disease (270, 13, 2) | before | 3 … 237 (118 values) | 354 | 44.81 % | 7.04 % (8 refusals) |
  | Heart-Disease (270, 13, 2) | after | 3 … 119 (59 values) | 177 | 84.44 % (4 refusals) | 80.74 % (10 refusals) |

  On Heart-Disease (10, 13, 2), min|Kᵢ| = 4 leaves the single admissible k = 3.
- The validation of the Excel experiment workbook is unaffected: it runs with the workbook's
  settings (ADR-058).

## ADR-063 — Admissible k: odd k = 3 … min|Kᵢ| (author)

*Accepted, 2026-11-22. Supersedes ADR-005 and ADR-006 for the default.*

- Default strategy `min-class`: odd k = 3, 5, …, k ≤ min(|K1|, |K2|), with the class sizes of the
  training sample of the current fit (so every fold of a protocol has its own k). The model is
  undefined when min|Kᵢ| < 3 (Definition 1).
- The article states the condition as [k/2] − k/2 = 0, which selects the even k; section 1.3 starts
  the odd k at 3 and formula (5) needs a majority, so the odd k are meant. ⚠ Recorded as a probable
  misprint.
- The strategies `twice-min-class` (odd k = 3 … 2·min|Kᵢ| − 3, the Excel experiment's rule, with
  the optional cap and step of ADR-006), `range` (the article's experiments: 3 … 41) and `explicit`
  stay available.

## ADR-064 — The zero value of the article's experiments (author)

*Accepted, 2026-11-22.*

- Formula (5) gives the values {1, 2}. The article's experiments also use 0. Reading (*author*):
  a training object takes 0 where the classes cannot be told apart at its gradation
  (f_k(μ) = 0.5, tested exactly as d₁·|K2| = d₂·|K1| in integers), and every synthetic feature
  that takes 0 on a training object is left out, so that the remaining features keep the values
  {1, 2}.
- Switch `synthetic_features.zero_value` (off by default: formula (5) alone). The features left
  out are listed with the reason "zero" in the trace, the outputs and the *Parameters* sheet.
- It is a plain switch, not one of the settings that deviate from the article: no ⚠ badge or
  warning (*author*, after the Step 8 report).

## ADR-065 — Original nominal features combined with the synthetic ones (author)

*Accepted, 2026-11-22.*

- `synthetic_features.original_features`: `nominal` (every feature of J) or feature names. Each
  acts as a feature of Ψ(r) whose gradations are its codes (non-negative integers): membership,
  stability, informativeness and contributions are computed over its codes, and an unseen code of
  a new object contributes 0. Quantitative features are rejected (*author: nominal originals as
  gradations*). Empty by default.
- A plain switch, not one of the settings that deviate from the article: no ⚠ badge or warning
  (*author*, after the Step 8 report).

## ADR-066 — Normalization per base operator

*Accepted, 2026-11-22.*

- `local_context.normalizer` is the default for every base operator; `operators[].normalizer`
  overrides it for one operator (for example `none` for the Euclidean or Chebyshev metric on the
  raw quantitative features). Each normalizer is fitted on the training objects only; operators
  that share a normalizer share one fitted scaling.

## ADR-067 — Bit representations as training-side analysis (author)

*Accepted, 2026-11-22.*

- Section 1.2 (class-sequence bit masks) and section 1.3 (majority chains) are computed for every
  operator on the training sample, with membership (1) and stability (2) over their gradations
  (*author: as training-side analysis*); neither enters Ψ(r).
- Class-sequence mask for one k: bit j is 1 when the j-th nearest neighbour belongs to K1, the
  nearest neighbour the most significant bit; 2ᵏ gradations, β = 2ᵏ − 1. The article prints
  "β = 2k − 1": ⚠ a probable misprint. The article does not say which class is the bit 1; K1 = 1 is
  this project's convention. Masks are formed for k ≤ 62 (int64).
- Majority chain: bit i is 1 when the own-class count for the i-th admissible k is a majority; the
  first admissible k is the most significant bit; β = 2^|K̂| − 1.
- The metaobject — the vector of stabilities g_k of the synthetic features (Theorem) — is part of
  the trace (ADR-074).

## ADR-068 — The majorizing function φ

*Accepted, 2026-11-22.*

- The article uses φ in Steps 3–4 of the agglomerative algorithm without defining it. The logistic
  sigmoid of the Excel experiment stays the default (`additional_features.majorizing_function:
  sigmoid`); the other built-ins of ADR-026 stay available. ⚠ Open for the author.

## ADR-069 — Definition 3: accuracy on the R-IPS split; nested cross-validation (author)

*Accepted, 2026-11-22.*

- Protocol `rips` (aliases `r-ips`, `definition-3`): R-IPS on the whole sample (ADR-061), the
  CS-model fitted on the training sample, the test sample classified without its class. Reported:
  the accuracy with refusals counted as errors (Definition 3), the accuracy on the answered
  objects, the refusals and the class split of both samples.
- Protocol `nested-cross-validation`: in every outer fold R-IPS splits the training part into an
  inner training and test sample; the candidate settings (by default α ∈ {0.1, 0.2}, the article's
  values) are compared on the inner test sample, and the best one is fitted on the inner training
  sample and scores the outer fold.
- Resubstitution (Definition 2), leave-one-out, k-fold and hold-out stay.

## ADR-070 — The article's computational experiment, rebuilt from real runs (author)

*Accepted, 2026-11-22.*

- `scripts/run_experiments.py` rebuilds Tables 1–6 and the k-NN comparison of the article from
  runs of the package and writes `experiments/results/` with `experiments/checksums.sha256`;
  `tests/test_experiments.py` checks the checksums and that a new run reproduces every file byte for
  byte. The numbers the article prints are kept in the script only for the comparison.
- Nothing is tuned. Settings the article leaves open are chosen once, before any run, and listed
  in the results (normalization of the Euclidean, Chebyshev and Canberra CS-models, the features of
  the Canberra metric, the k-NN of the comparison, both values of α for Table 6).
- The results do not reach the article's: the best test accuracy is 79.7 % (Heart, Zhuravlyov
  split with blocked objects covering, α = 0.1) against 100 % printed; no training accuracy reaches
  90 %; no synthetic feature takes the value 0 on these training samples with k = 3 … 41, so
  Table 1 has 20 binary features per metric where the article prints 21 to 30 (of 31).

## ADR-071 — Presets

*Accepted, 2026-11-22. Supersedes ADR-004 for the presets.*

- `default` (the article: ADR-062, ADR-063), `alpha-0.1` and `alpha-0.2` (the article's
  experiments), `excel-experiment` (the workbook's settings: twice-min-class k, running class
  means, two passes, α = 0.3). Configuration files of the same names are in `configs/`.

## ADR-072 — Heart-Disease: slope as a quantitative feature (author)

*Accepted, 2026-11-22.*

- The article's experiments treat 7 features of Heart-Disease (270, 13, 2) as quantitative; the
  data file marks 6. The slope of the peak exercise ST segment (x₁₁) is the seventh: the study of
  the article's experiments (ADR-070) and the R-IPS tests set it through
  `dataset.feature_types: {x₁₁: quantitative}`; the data file is unchanged, and any configuration
  can do the same.

## ADR-073 — German (1000, 20, 2) (author)

*Accepted, 2026-11-22.*

- The German credit data of the article's experiments are added byte for byte to
  `datasets/raw/German/` (`.dat`, `.csv` and the read-me); a test pins their checksums. Like every
  other dataset they are used under their own names; the article's names of its tables' files are
  not reproduced (*author*).

## ADR-074 — The metaobject in every output

*Accepted, 2026-11-22.*

- The metaobject (the stabilities g_k of the synthetic features, the Theorem) is in the trace, the
  run's JSON, the tables (`metaobject`), the Excel mirror and the report, next to the class-sequence
  masks and majority chains (ADR-067).

## ADR-075 — The *Switches* sheet of the Excel mirror works on the run's own data

*Accepted, 2026-11-22. Supersedes ADR-043.*

- The mirror's *Switches* sheet lists every setting that deviates from the article and is on in
  the run (four possible, ADR-061, ADR-002, ADR-003). The worked examples of the two switches of
  the agglomerative algorithm — θ, γ and θ/γ of the first candidates with final and with running
  class means, the first additional feature with one and with two majorizing passes — and the
  effect table of the four settings are computed on the run's own data by the package's
  agglomerative algorithm at export time.
- The data file that held the check workbook's input for these examples is gone; that workbook is
  checked by `csr validate` and the golden tests instead (ADR-029, ADR-060).

## ADR-076 — The large-data configuration on the article's settings (author)

*Accepted, 2026-11-22. Amends ADR-030.*

- `configs/heart-disease-270-large-data.yaml` keeps the correction of ADR-030
  (`synthetic_features.skip_constant: true`) and otherwise follows the article: odd k = 3 …
  min|Kᵢ|, final class means, one majorizing pass. Before, it capped k at 21 (`twice-min-class`)
  with running class means and two passes: leave-one-out 81.5 %, AUC 0.862 (ADR-030).
- With the article's k range no synthetic feature of Heart-Disease (270, 13, 2) is constant, so
  the configuration gives the default's model and results: r = 177, leave-one-out 80.7 % with 10
  refusals, AUC 0.867 (pinned in `tests/test_defaults.py`).

## ADR-077 — Every command reads the configuration's dataset (author)

*Accepted, 2026-11-22.*

- Without a dataset argument, `csr fit`, `csr classify`, `csr data info` and `csr run` load the
  configuration's `dataset.path` (relative to the configuration file; none: the default dataset).
- With a dataset argument the command line names the data; the configuration's reading options
  (format, sheet, class and id column) fill in what the command line leaves out, and its
  `feature_types` apply. Before, only `csr run` without an argument read the section, so the
  slope of Heart-Disease (ADR-072) was lost whenever a dataset was named.

## ADR-078 — Stage names in the results tree and the tables (author)

*Accepted, 2026-11-22.*

- The groups of the tables and figures — the results tree of the desktop application, the
  Markdown and LaTeX tables, the titles of the figures — are named by the article's stages:
  *Input data*, *Local contexts · …*, *Synthetic features · …*, *Additional features · …*,
  *Meta-algorithm · …*, then *Evaluation* and *Checks*.
- The Excel mirror keeps the Excel experiment's Steps 1–12, because it copies that workbook's
  layout; the groups of the *Configure* page are unchanged.
- The additional features are named d₁ … d_p wherever they are shown (CLI, run results, Excel
  mirror, validation report); the earlier r₁ … r_p are gone.

## ADR-079 — Standard classifiers on the article's training and test samples: Tables 7 and 8 (author)

*Accepted, 2026-11-22. Its split-file layout is superseded by ADR-082.*

- The article's training and test samples (split files: features, class and a flag, 0 =
  training, 1 = test) are copied byte for byte to `datasets/splits/<family>/` under neutral names:
  German split with the Zhuravlyov metric (all 20 features, 336 training objects), with the
  Euclidean metric (the 7 quantitative features, raw values, 341) and with the Chebyshev metric
  (the same 7 features, 367), and Heart-Disease with the Zhuravlyov metric (76). Heart's split can
  be read but is not used in the tables. A test pins their checksums; the dataset catalogue does
  not read the folder. `data.splits.load_split_file` reads the layout (decimal commas, any
  whitespace, LF or CRLF, extra values after the types ignored) into a `FixedSplit`.
- The comparison is kept apart from the CS-model's pipeline: the package `comparison` (pure
  computation), the service `services.comparison`, `csr compare run | check`,
  `scripts/run_comparison.py` and the *Classifiers* page of the desktop application (the eighth
  page, after *Export*, so that Ctrl+1 … Ctrl+7 keep their pages). It uses the `svm`,
  `random-forest` and `knn-vote` baselines, the metric registry and the table model with explicit
  options and changes none of them; no default was changed.
- Settings, fixed before any run (*author*); nothing is tuned or chosen on the test samples:
  - SVM `SVC(kernel="rbf", C=1.0, gamma="scale")`; random forest
    `RandomForestClassifier(n_estimators=200, random_state=42)`. Both use every feature of the
    file: the quantitative ones min–max fitted on the training sample, the nominal ones one-hot
    encoded with the training sample's categories (an unseen category encodes as all zeros).
  - k-NN majority vote with k = 3 and k = 5 on the 7 quantitative features, min–max fitted on the
    training sample, with the Euclidean, Chebyshev and Canberra metrics. Distances are rounded to
    10 decimals, and ties go to the smaller training index. An extra row, not printed in the
    tables, uses the Zhuravlyov metric on all 20 features of the 336 split.
  - Reported per split and method: accuracy on the test sample (the tables' cells), balanced
    accuracy (the mean recall of the two classes), the majority-class rate of the test sample,
    and the sample sizes. The accuracy on the training sample is resubstitution as in Definition
    2: a k-NN training object is not its own neighbour (*author*, after the first run).
- Table 7 has rows SVM and random forest; Table 8 has rows Euclidean, Chebyshev and Canberra, with
  each cell "k = 3 / k = 5". The draft prints Table 8 as a second "Table 7"; it is numbered 8
  here. Both have the three German splits as columns, and their notes give the majority-class
  rates and the settings. They are written as CSV, Markdown and LaTeX, with `results.csv|md`,
  `summary.json` and `checksums.sha256`, to `experiments/comparison/`.
- `summary.json` records the versions of numpy and scikit-learn (the random forest depends on
  them), not the package version, so a version bump changes no file. The byte-for-byte test of
  a new run is skipped when the installed versions differ from the recorded ones; the pinned
  numbers are checked in any case.
- An independent run of the same settings on the same files (scikit-learn 1.9.1, numpy 2.5.3)
  gives the same value for all 55 numbers: 26 test accuracies, 26 balanced accuracies and 3
  majority-class rates.
- No method reaches the majority-class rate of a test sample (75.90 %, 79.97 %, 80.57 %); the
  best is SVM on the Zhuravlyov split, at 74.85 %. The CS-model is run on these splits under
  ADR-080.

## ADR-080 — The CS-model on the article's training and test samples (author)

*Accepted, 2026-11-22.*

- The CS-model runs on the same three German splits as Tables 7 and 8, fitted on the training
  sample, with the test objects classified without their class (the Theorem). The author approved
  the settings before the first run; nothing is tuned or chosen on the test samples:
  - one base operator per metric, as in the k-NN rows: Euclidean, Chebyshev and Canberra on the
    quantitative features of every split, and Zhuravlyov on all 20 features of the 336 split
    only. Every operator is min–max scaled on the training sample (*author*: the same distances
    as Table 8's k-NN; raw values would keep the 336 file's scaling over all 1000 objects, test
    objects included);
  - odd k = 3 … 41 and the zero value of the article's experiments (Table 1), α = 0.1 and 0.2,
    as in ADR-070; otherwise the article's defaults (final class means, one majorizing pass, no ⚠
    switch).
- Reported: accuracy on the test sample with refusals as errors (Definition 3), accuracy on the
  answered objects, refusals, balanced accuracy, accuracy on the training sample (Definition 2),
  r, the features left out for the zero value, and TUPLAM.
- The results go into a separate table, `cs-model` (CSV, Markdown, LaTeX), beside Tables 7 and 8.
  Its rows are the metrics and each cell is "α = 0.1 / α = 0.2". The rows are also added to
  `results.csv|md` and `summary.json`, and the table is shown on the *Classifiers* page. Tables 7
  and 8 keep the article's layout.
- Result: r = 20 on every split, and no synthetic feature takes 0 on a training object (as in
  ADR-070). No setting reaches the majority-class rate of a test sample. Each model assigns most
  test objects to one class, and which one depends on the metric and α. Test accuracies therefore
  range from 21.33 % to 78.36 %, while the balanced accuracy stays between 48 % and 57 %, near
  chance; SVM reaches 70.63 % on the 336 split.

## ADR-081 — The Windows build carries scikit-learn and the article's split files (author)

*Accepted, 2026-11-22. Amends ADR-055.*

- So that the *Classifiers* page (ADR-079, ADR-080) works in `csr-gui.exe`, the build no longer
  leaves out scikit-learn and scipy. It also carries a copy of `datasets/splits` in the
  package's data (`data/split_files`). Only pyarrow (Parquet) stays out.
- The comparison reads the project's `datasets/splits` when there is one and the bundled copy
  otherwise. Without a project folder its default results folder is `experiments\comparison`
  under the folder the application runs in.
- `scripts/build_exe.py` stops when scikit-learn is not installed, and CI installs the
  `[sklearn]` extra for the executable, so a build cannot silently lack it.
- The frozen application's self-test also computes Tables 7 and 8 and fails when scikit-learn
  or the split files are missing. Run from source, it checks them only where the project's split
  files are found.

## ADR-082 — The split-file convention (author)

*Accepted, 2026-11-24. Supersedes ADR-079 for the split-file layout.*

- The files in `datasets/splits/` follow the author's convention, which is their specification:
  - row 1: the object count m, the feature count n and the class count c, e.g. `1000 7 2`. The
    header counts the features only; empty cells may follow on that row;
  - rows 2 … m + 1, one per object: the n feature values, then the class (1 or 2) in column n + 1
    and the split marker (0 = training, 1 = test) in column n + 2;
  - last row: the n feature types (0 = nominal, 1 = quantitative). Its cells in columns n + 1 and
    n + 2 are empty, because those columns are not features;
  - cells are tab-separated, lines end with LF, there is no BOM, decimal commas are allowed, and
    no line starts with an empty cell.
- The author changed the headers on purpose from `m, n + 1, c` (`1000 8 2`, `1000 21 2`,
  `270 14 2`) to `m, n, c` (`1000 7 2`, `1000 20 2`, `270 13 2`), and removed the leading tab of
  every line. Those headers stay as they are. Three type rows did not fit the convention and were
  corrected with the author's approval: the Euclidean and Chebyshev splits held `0 0` in columns 8
  and 9, and Heart's 13th cell held `0     0`. Each now holds the n types followed by two empty
  cells. The objects, classes, types and samples are unchanged, and so is every number of
  Tables 7 and 8 and of the CS-model. Only the files' SHA-256 changed, pinned in a test and
  recorded in `summary.json`.
- `data.splits.load_split_file` reads n from the header and checks the convention, naming the file
  and the row of every departure: exactly three numbers in the header; m object rows; exactly
  n + 2 values in each; the class one of the c classes (1 … c); the marker 0 or 1; exactly n type
  flags 0/1 in the last row and no other non-empty cell; no row starting with an empty cell. A
  header that still counts the marker column is reported at the first object row. Decimal commas,
  LF or CRLF line ends and empty trailing cells are still read. The reading of ADR-079 is gone: it
  took the header as `m, n + 1, c`, split cells at any whitespace and ignored values after the
  types.

## ADR-083 — CSV files carry a byte-order mark, so that Excel reads them as UTF-8 (author)

*Accepted, 2026-11-24. Amends ADR-040 for `.csv` files only.*

- Every `.csv` file the project writes starts with the UTF-8 byte-order mark (`EF BB BF`, the
  encoding `utf-8-sig`) and keeps LF line ends. Without it Excel reads a UTF-8 file as
  Windows-1252 and shows "—" as "â€”" and α as "Î±".
- One helper writes them: `export.text.write_csv` (with `csv_bytes` for the files a service builds
  in memory and `csv_rows` for rows without a `Table`). It is used by the comparison
  (`table-7.csv`, `table-8.csv`, `cs-model.csv`, `results.csv`), the study
  (`experiments/results/*.csv`), the export format `csv` (`tables/<key>.csv`, `tables/index.csv`),
  every run folder (`predictions.csv`, `folds.csv`) and the desktop application's *Save CSV…*
  button.
- Markdown, LaTeX, JSON, HTML, YAML and the checksum files stay UTF-8 without a BOM (ADR-040).
  Copying a table to the clipboard writes no file and has no BOM. The split files (ADR-082) and the
  author's datasets are inputs and are not rewritten; every reader of a CSV, the project's own
  loaders included, reads `utf-8-sig`, so a file with or without the mark gives the same data.
- No number changes: each new CSV equals the previous one with the three bytes in front. Only the
  checksums that list the CSV files change (`experiments/checksums.sha256`,
  `experiments/comparison/checksums.sha256`).

## ADR-084 — The *Classifiers* page as a staged workbench of classifier parts

*Accepted, 2026-11-24. Amends ADR-079 for the desktop application.*

- The *Classifiers* page stays one entry of the main sidebar. Inside it, a **classifier part** is
  a unit of work in three stages: ① *Training* on the training sample, ② *Test*, where the test
  objects are classified without their class, and ③ *Evaluation*, where the decisions are
  compared with the true classes. The comparison of ADR-079 and ADR-080 is the first part,
  "Tables 7 and 8" (standard classifiers and the CS-model). Further parts are registered in the
  services; the page does not change.
- Modules:
  - `data.samples`: `UnlabelledSample` and `AnswerKey` (ADR-085).
  - `comparison.stages` (pure): the comparison in three stages, `fit_split`, `classify_split` and
    `evaluate_split`. `compare_split` is these three in a row, so `csr compare run`, the pinned
    numbers and the byte-for-byte test exercise the staged code.
  - `services.classifier_parts`: the protocol `ClassifierPart`, the registry `CLASSIFIER_PARTS`
    (a `core.registry.Registry`), the stage types and the tables every part gets.
  - `services.comparison_part`: the part "Tables 7 and 8", registered under `tables-7-8`.
  - `gui.workbench`: `PartView`, the stepper and the three step views; `gui.pages.classifiers`:
    the page with the list of parts.
- The protocol (types of `services.classifier_parts`):
  - `key`, `title`, `description`, and `default_folder()`, the folder its samples are read from
    by default;
  - `sample_options(folder) → (SampleOption, …)`: what the part can run on, each with its sizes,
    and with the reason when it is unavailable; `load(option, folder) → SampleSet`: the training
    samples with their classes, the test objects as `UnlabelledSample`s and, apart, their
    `AnswerKey`s;
  - `method_groups(option) → (MethodGroup, …)`: the methods grouped as in the article, each with
    its fixed settings as text;
  - `train(training) → TrainedPart`: the fitted methods and the training report (resubstitution,
    Definition 2);
  - `classify(trained, unlabelled) → Predictions`: per split, method and object the decision 1, 2
    or 0 and the score where the method gives one;
  - `evaluate(predictions, answers) → Evaluation`: the metrics per method, the part's own tables
    and the correctness of every object;
  - optional capabilities, found by `isinstance` on runtime-checkable protocols:
    `write(evaluation, folder)` with `write_problem(evaluation)` and `default_output()`, and
    `check(folder, splits)`. "Tables 7 and 8" has both: the files and the check of
    `csr compare run | check`.
  - The registry gains `Registry.remove`, so that a test can register a part and take it away;
    built-in parts are the modules of `BUILTIN_PARTS`, imported on first use, and a part of
    another package registers through the entry-point group
    `context_synthetic_recognition.classifier-parts`.
- Every stage result carries the part's extra tabs (`StageTable`). The tabs every part gets are
  built in the services from the stage results: *Methods*, *Training sample* and *Training
  results*; *Test objects*, *Predictions* and *Predicted classes*; *Per object* and *Confusion*.
  A per-split table carries one variant per split, chosen in the tab. A table states the meaning
  of its columns (class, decision, outcome, status), and the page colours by meaning: K1 blue, K2
  orange, refusal grey, correct green ✓, wrong red ✗. A part without extra tabs still works; a
  test registers a dummy part and runs it through the page.
- How the comparison maps onto the stages: *Training* fits the k-NN vote and the CS-model on the
  training sample and classifies the training objects (resubstitution, Definition 2). SVM and
  random forest are scikit-learn estimators inside the `svm` and `random-forest` baselines, which
  fit and predict in one call; their fitted state is the training sample with the fixed options,
  and *Test* fits them again on it, deterministically (ADR-085). *Evaluation* builds the same
  `SplitResult`s as before, so Tables 7 and 8, the CS-model's table, the results table and the
  files of `experiments/comparison` are the same byte for byte. Tables 7 and 8 and the CS-model's
  table need the three article splits; with fewer splits these tabs say so, and writing is allowed
  only for the full set.
- The page (wireframe at 1360 × 860; the parts list collapses to a narrow strip):

  ```text
  ┌────────────┬─────────────────────────────────────────────────────────────────────────────┐
  │ PARTS   ◀  │ Tables 7 and 8 — standard classifiers and the CS-model                      │
  │ ● Tables   │ one-line description                                                        │
  │   7 and 8  │ Samples [All article splits (336, 341, 367) ▾] Split files [datasets\…] [Browse…] │
  │            │ [① Training ✓ done] ─ [② Test ○ not run] ─ [③ Evaluation ○ not run]  [Run all] ▓▓░ │
  │            │ ┌─ step view ─────────────────────────────────────────────────────────────┐ │
  │            │ │ banner · action button                                                  │ │
  │            │ │ [tab] [tab] [tab]                                                       │ │
  │            │ │ (Split [336 ▾] for per-split tables) table with filter, copy, Save CSV  │ │
  │            │ └─────────────────────────────────────────────────────────────────────────┘ │
  │            │ status line · ⚠ warning box                                                 │
  └────────────┴─────────────────────────────────────────────────────────────────────────────┘
  ```

  - ① *Training*: the button *Fit on the training sample*; tabs *Methods* (read-only fixed
    settings, shown before any run), *Training sample* (the objects with their features and
    class), *Training results* (method × split: training accuracy, correct, refusals).
  - ② *Test*: the banner "Classes hidden: the test objects are classified from their feature
    values only", the button *Classify the test sample*; tabs *Test objects* (feature values
    only — the class column does not exist), *Predictions* (object × method: 1 / 2 / 0 coloured by
    the predicted class, never green or red; the score where there is one; *Agreement* and *Most
    common*), *Predicted classes* (method × split: predicted K1, K2 and refusals, counts and
    shares).
  - ③ *Evaluation*: the banner "True classes revealed", the button *Compare with the true
    classes*; tabs *Table 7*, *Table 8*, *CS-model*, *All results*, *Per object* (✓ / ✗ / 0
    against the true class), *Confusion* (2 × 2 and refusals per method and split) and *Check*;
    the results folder, *Write the files*, *Check the written files*, *Open the folder*.
  - Step states, each shown by an icon and a word, never by colour alone: ○ not run, ◔ running,
    ✓ done, ⟳ out of date (an earlier step was run again, or the samples or the split folder
    changed), ⊘ unavailable (the tooltip says why). A step can be opened only when the step before
    it is done. Alt+← / Alt+→ move between the steps, as on the *Results* page.
  - *Run all* runs the three stages in one worker; the progress bar names the stage and the split.
  - Every stage, the reading of the sample options and the check run in a worker thread
    (`gui.workers.Task`); the window waits for them on closing. The page computes nothing and
    never offers to change a setting (ADR-079, ADR-080).

## ADR-085 — The blind test: `UnlabelledSample` and `AnswerKey`

*Accepted, 2026-11-24.*

- The input of the *Test* stage is `data.samples.UnlabelledSample`: the indices of the objects in
  their file, their feature values, the feature types and the feature names. It has no class
  field and no split marker. `FixedSplit.test_objects()` builds it from the test sample by
  dropping the class column; `FixedSplit.test_answers()` builds the `AnswerKey` (the same indices
  and the true classes), which only the *Evaluation* stage receives.
- Every fitted method of a part classifies through `classify(unlabelled)`, and a part's
  `classify(trained, unlabelled)` takes nothing else. A `Dataset` (which carries classes) is
  rejected: by mypy (a static assertion in `services.classifier_parts`, which fails the type check
  if a `Dataset` were ever accepted) and at run time (`TypeError`).
- SVM and random forest are fitted again on the training sample at the *Test* stage: the project's
  baselines fit and predict in one call, and splitting them would change the plug-in signature of
  every baseline. The fit is deterministic (fixed options, `random_state = 42`), so the decisions
  are the same as those of a single fit. The k-NN vote keeps the training sample as its model;
  the CS-model keeps the fitted `CSModel` and classifies with `CSModel.classify` (the Theorem).
- Tests prove the blindness: shuffling, flipping or deleting the classes of the test objects
  changes no decision and no score of any method — SVM, random forest, every k-NN row and every
  CS-model.

## ADR-086 — `csr compare predict`: the blind predictions as a file

*Accepted, 2026-11-24.*

- `csr compare predict [--split KEY] [--out FILE] [--splits FOLDER]` trains the methods of
  "Tables 7 and 8" on the training sample of one split (336 by default; 341, 367 or 76) and
  classifies its test objects without their class. It writes one CSV file through the project's
  CSV writer (ADR-083): the object, then one column per method with the decision 1, 2 or 0. There
  is no class column.
- The loader reads the whole split file and checks the convention (ADR-082), the classes
  included; the command drops the test classes right after (the `AnswerKey` is never built) and
  never writes them. Without `--out` the file is `predictions-<KEY>.csv` in the working folder, so
  `experiments/comparison` keeps the files of `csr compare run`.

## ADR-087 — The datasets by the kind of their features; four kept on the author's disk (author)

*Accepted, 2026-11-24. Amends ADR-035 for the layout of `datasets/raw/`.*

- The author grouped the original datasets by the kind of their features:
  `datasets/raw/Heterogeneous/` (quantitative and nominal), `Nominal/` and `Quantitative/`, each with
  one folder per family as before. German and Heart-Disease moved to `Heterogeneous/` byte for byte;
  their checksums (ADR-073) and ids are unchanged. The catalogue already finds datasets at any
  depth under `raw/`, so no code changes; a test checks that every dataset sits in the folder of its
  kind.
- Seven datasets from open archives join the repository byte for byte, with the author's
  `ReadMe.txt` and a test that pins their SHA-256: Heart-Failure (299, 12, 2) and Ionosfera
  (350, 33, 2) (heterogeneous); Molecular-Biology (106, 57, 2), (100, 50, 2) and (100, 10, 2), and
  Tic-Tac-Toe (958, 9, 2) (nominal); Breast-Cancer-Wisconsin (569, 30, 2) and Spambase (4204, 57, 2)
  (quantitative).
- Four stay on the author's disk only, listed in `.gitignore` (*author*): TCGA_LUAD (589, 57772, 2),
  whose files (391 MB and 421 MB) are larger than GitHub accepts — the project uses no Git LFS
  (ADR-014), and its read-me stays with them — and Cancer
  (589, 44, 2), Hypertension (147, 29, 2) and Uzb-Kor (100, 24, 2), which are not from an open
  archive. Where they are on disk the catalogue reads them; the tests of the dataset folder leave
  them out, so they pass on the author's machine and in a checkout alike.

## ADR-088 — Spambase's training and test samples (author)

*Accepted, 2026-11-24.*

- Two split files of Spambase, added by the author as `spam1_601e.dat` and `spam1_722en.dat`, are
  kept under the neutral names of the other split files (ADR-079):
  `datasets/splits/Spambase/Spambase (2102, 57, 2) — training 601 (euclidean).dat` and
  `… — training 722 (euclidean).dat`. Each holds the same 2102 objects — half of each class of
  Spambase (4204, 57, 2) — with 57 quantitative features: raw values in the 601 file, min–max scaled
  over the 2102 objects in the 722 file; a test checks both relations. Training 601 (361, 240), test
  1501 (903, 598); training 722 (467, 255), test 1380 (797, 583).
- They came in the layout ADR-082 replaced and were brought to the convention as the other files
  were (*author*): the header counts the features only (`2102 57 2` instead of `2102 58 2`), no
  line starts with a tab, and the type row holds the 57 types followed by two empty cells — the
  601 file held `0 0` there, the 722 file a merged `0       0`. Every object row is unchanged.
- The part "Tables 7 and 8" offers them as samples, like Heart's split: the *Classifiers* page runs
  the three stages on them and `csr compare predict --split 601 | 722` writes their blind
  predictions. They are not columns of Tables 7 and 8, and no result of theirs is written to
  `experiments/`. The Windows build carries them with the other split files (ADR-081).

## ADR-089 — No garbage collection in a worker thread

*Accepted, 2026-11-24.*

- Python's cyclic garbage collector runs in whichever thread allocates when a threshold is reached.
  In a worker thread (`gui.workers.Task`) it can free the Python wrappers of Qt objects that wait
  for collection — the widgets of a closed window — and so destroy them outside the interface's
  thread. Qt then crashes with an access violation. It happened in CI on main (Python 3.11 and
  3.13) after the *Classifiers* page started more workers, during the test suite.
- A `Task` therefore collects garbage in the interface's thread when it starts, and the collector
  is switched off while any worker works (`gui.workers.CollectorPause`, counted, so overlapping
  workers are safe); the last worker to end switches it back on, unless it was off before.
  Reference counting still frees everything without a cycle while a worker runs.

## Open questions for the author

*Open, 2026-11-22.* Each is flagged where it is used; none is decided silently.

- α, δ, ϰ and cr1₀ are not given in the article (defaults 0.3, 0.1, 5, 10); ϰ ≤ n − 1 is not
  enforced.
- φ is not defined (ADR-068).
- The admissible-k condition and β are probably misprinted (ADR-063, ADR-067).
- The training samples of R-IPS differ from the printed ones by one to three objects (ADR-061).
- Tables 1–6: 31 synthetic features are printed for k = 3 … 41, which has 20 odd values; Table 6
  gives one pair of accuracies for two values of α; the normalization of the Euclidean and
  Chebyshev CS-models and the features of the Canberra metric are not stated; the k-NN settings of
  the comparison are not stated (ADR-070).
