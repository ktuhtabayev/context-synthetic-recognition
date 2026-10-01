# Prompt for Claude Code CLI — build `context-synthetic-recognition`

Run Claude Code from `D:\PhD\CODING\context-synthetic-recognition` and paste everything below the line
(or type: “Read docs/handoff/PROMPT_FOR_CLAUDE_CODE.md and follow it.”).

---

You are a senior research-software engineer and Python architect. Build **context-synthetic-recognition**: a production-quality, research-grade Python project (library + CLI + excellent desktop GUI, under Git) that implements the **context-synthetic model of recognition algorithms (CS-model)** from my draft article, using my completed Excel experiment as the executable specification. The current folder is the (empty) project root; the template project `..\hag-regularized-stacking-boosting-meta` sits next to it.

## 1. Read everything first — in this order, in depth

1. `CLAUDE.md` and `docs/handoff/README.md`.
2. `docs/handoff/CONTEXT_HANDOFF.md` — **authoritative** list of decisions, formulas, conventions, golden values and open questions.
3. `docs/handoff/CHAT_TRANSCRIPT.md` — the full design conversation with Claude that produced the experiments (origin: https://claude.ai/code/session_01Drh9SjE6L3BXHRB9Hw8LdM). You cannot open that link from the CLI (it needs a claude.ai login), and `claude --teleport` only works for web sessions attached to this same Git repository — this one was not — so the exported transcript is your copy of the chat. Read it all; where it conflicts with CONTEXT_HANDOFF.md, the handoff wins.
4. The Excel experiment `resources/experiments/context-synthetic-model/Context-Synthetic Model – Full Experiment [Heart-Disease (10, 13, 2)] - Opus 5.5.xlsx`: open it with openpyxl twice (formulas, and `data_only=True` for cached values) and study **every one of the 33 sheets** — the Overview (traceability map: article element → sheet → planned function), Parameters, Template Deviations (the two template calculations that differ from the article, with exact template cells, worked examples and fixes), Steps 1–12, the four HAG iteration sheets, Meta-algorithm (+ All Objects), Margin Analysis, Accuracy, Confusion Matrix, Precision/Recall/F1, ROC & AUC, Leave-One-Out, Sensitivity (Switches), Model Properties and Validation. Treat its formulas as the specification of each computation and its Validation sheet as acceptance tests. Also read the earlier k-NN workbook in the same folder.
5. The article draft `docs/handoff/article/article-draft-cs-model.docx` (Russian; formulas (1)–(6) are images — the handoff and the workbook spell them out), and the images in `docs/handoff/images/` (hand-drawn algorithm order, k-range formula, Zhuravlev metric).
6. `docs/handoff/reference-engine/` — a small dependency-free Python engine that reproduces the workbook exactly (including the template-replication mode) with `golden_values.json`. It is an executable spec and a test oracle, **not** the architecture to copy.
7. The template project `..\hag-regularized-stacking-boosting-meta` (**read-only — never modify it**): inspect its full structure, packaging, module layout, naming, code style, configuration, logging, tests, GUI (framework, look, flows), docs, scripts, Git history/conventions and its `resources\experiments\{hag-algorithm,meta-algorithm,model-evaluation}` workbooks (the HAG, meta-algorithm and model-evaluation templates my experiment follows). Note what it does well (reuse those conventions for consistency across my projects) and where it falls short (do better here). It was built for the same problem statement and solution structure, so align terminology and method boundaries with it.

Then reply with: (a) what you understood — the problem, the pipeline and the two known template deviations; (b) a comparison of the template project's structure/style with what you propose; (c) the architecture and milestone plan below; (d) your questions (see §8). **Do not write project code until I approve the plan.**

## 2. What the software must do (the method — keep it exact)

Follow CONTEXT_HANDOFF.md §3 literally. In short: dataset with heterogeneous features (types I/J) and classes → normalization fitted on training data → base operators (metric × feature subset; default: Zhuravlev on all, on quantitative only, on nominal only; distances rounded to 10 decimals) → neighbour ordering (ties → smaller original index, self excluded) → **formula-based permitted k** (odd, k_min = 3 fixed, k_max = 2·min_i|K_i| − 3, computed from the training classes, never hard-coded) → same-class counts μ (training side) and class-free synthetic features Ψ(r) by (5) → membership (1), stability (2), bit masks and meta-object → boundary (3) and informativeness ω (4) → contributions η (6) → hierarchical agglomerative grouping (template parameters α = 0.3, δ = 0.1, ϰ = 5, cr1 = 10, sigmoid majorizer) → meta-dataset Y = (y, r) → meta-algorithm (B1/B2 filtering, refusal = 0) → classification of a new object **without its class** (the article's Theorem) → evaluation (resubstitution, leave-one-out with the whole pipeline re-fitted per fold, baselines, confusion matrix, precision/recall/F1, AUC with rounded scores, ROC, margins with/without majorizer) → model-property checks (Definitions 1–6).

**Highlight these two template deviations in code, docs and the GUI, and implement both as switches (default = template, so the templates are reproduced exactly):**
- θ and γ are measured from running partial class means instead of the final class means M₁ and M₂ (`centres = "running" | "final"`);
- in STEP 4 the majorizer is applied twice instead of once (`step4_passes = 2 | 1`).

## 3. Senior-level architecture — built for future expansion

- **Layering**: pure, vectorized computational core (numpy; no I/O, no GUI) → application/service layer (experiment runner, config, persistence, export) → interfaces (Python API, `typer` CLI, GUI). The CLI and GUI call the same services.
- **Everything pluggable through registries/protocols** (entry points or a simple registry decorator), each with docs, tests and a GUI form generated from its parameter schema:
  - *metrics*: Zhuravlev (default) plus weighted Zhuravlev, Euclidean, Manhattan, Chebyshev, Minkowski-p, Canberra, cosine, Mahalanobis, Hamming, HEOM, HVDM, Gower — and user-defined ones; per-feature-subset application (all / I / J / custom subsets);
  - *normalizers*: min–max (default, fitted on training data only), z-score, robust (median/IQR), max-abs, decimal scaling, quantile/rank, unit-length, none;
  - *k strategies*: the formula rule (default), the article's “odd k ≤ min(|K1|,|K2|)” rule, explicit lists, ranges;
  - *synthetic-feature encoders*: formula (5) majority, same-class count μ, bit masks by the majority rule;
  - *weights*: ω by (4) (default); leave room for the template's Criterion-1 / λ·β weights;
  - *majorizing functions*: logistic sigmoid (default), tanh, arctan, softsign, custom; *class-centre modes* and *STEP 4 passes* as above;
  - *tie-breaking policies*, *meta-algorithm decision rules*, *evaluation protocols* (resubstitution, leave-one-out, stratified k-fold, repeated/nested CV, hold-out) and *baselines* (k-NN vote, and scikit-learn classifiers for comparison).
- **Data layer**: loaders for my Excel “Dataset sheet” layout and for CSV/XLSX/Parquet with a dataset schema (feature types, class column, ids, missing-value policy, category encoding); schema validation with clear errors; a dataset registry; support many real heterogeneous datasets, not only Heart-Disease (10, 13, 2). Design interfaces so multi-class support can be added (the article is binary — keep binary exact, make K-class a documented extension point).
- **Typed configuration** (pydantic v2 or dataclasses) saved as YAML/TOML; every run writes a manifest (config, config hash, dataset hash, seed, package versions, timings) to `runs/<timestamp>_<hash>/` for reproducibility; experiments can be re-run from the manifest.
- **Full trace objects**: every intermediate table the workbook shows (normalized data, distance matrices, rank matrices, neighbour blocks with μ and χ₁, Ψ(r), f/g, ω, η, each HAG iteration with every candidate's b, centres, θ, γ, θ/γ, the meta-algorithm's B1/B2 per step, margins, metrics) is captured as data, so the GUI, the exporters and the tests use one source of truth. This is what makes the model interpretable (“trace every decision from local relations to the final class”).
- **Exporters**: an Excel workbook that mirrors my experiment's sheet structure and style (openpyxl), CSV/JSON, LaTeX/Markdown tables and PNG/SVG figures for the article, and an HTML/PDF run report.
- **Engineering quality**: Python ≥ 3.11, `src/` layout, pyproject (uv or the template's tool), full type hints (mypy strict on the core), ruff + formatter, pre-commit, structured logging, clear exceptions, docstrings with formula numbers from the article, performance for thousands of objects (vectorized distances, caching per operator), deterministic results.

## 4. The GUI must be excellent

Desktop GUI (PySide6/Qt recommended — confirm against the template's choice and keep what works), professional and pleasant to use for research:
- **Workflow**: Project/Dataset → Configure → Run → Explore results → Classify new object → Compare runs → Export.
- **Dataset**: load by file dialog or drag-and-drop, preview grid, edit feature types and class column, statistics, missing values, class balance; live display of |K_i| and the permitted k.
- **Configure**: operators (metric × feature subset) with parameter forms, normalization, k strategy with a live preview, HAG parameters α, δ, ϰ, cr1, majorizer, the two template/article switches (clearly labelled with a warning badge), evaluation protocol, seed; presets (“Template”, “Article”); validation with inline messages.
- **Run**: background worker threads with progress, log console, cancel; the UI never freezes.
- **Results explorer mirroring the workbook steps**: tabbed/stepper views for every stage with sortable tables and the same colour semantics as the workbook (class colours, synthetic-feature and latent-feature colours, selected feature highlight, warnings), formula tooltips (“formula (4)”), and cross-highlighting (select an object → see it in every table).
- **Visualizations** (interactive, exportable): distance heatmaps per operator, neighbour graph with nested k-neighbourhoods, g_k and ω per feature, θ/γ per candidate per HAG iteration (with the chosen q), latent-feature margins with and without majorizer, ROC curve, confusion matrix, LOO fold overview, sensitivity of the switches.
- **New object**: a form (or paste a row) to classify an unseen object and an explanation panel showing its neighbours, its Ψ(r), and the step-by-step B1/B2 filtering of the meta-algorithm.
- **Compare runs** side by side; **Export** everything (Excel mirror, CSV/JSON, figures, LaTeX tables, report).
- **UX quality bar**: consistent design system, light and dark themes, high-DPI, keyboard shortcuts, recent projects, persistent window/settings state, undo/redo in configuration, accessible colour-blind-safe palettes and font scaling, English UI now and i18n-ready (Russian/Uzbek later). Include screenshots in the docs and pytest-qt tests for the main flows.

## 5. Tests and validation (acceptance criteria)

- **Golden tests against the workbook**: with the default configuration on Heart-Disease (10, 13, 2) reproduce every intermediate table and the Validation sheet values (tolerance 1e-9): permitted k = 3, 5; ω; η; TUPLAM {a₆, a₃, a₁, a₂, a₄}; crit per iteration; r₁…r₄; resubstitution and LOO predictions and scores; AUC; margins; Definition 4/6 checks; and all four switch variants. Provide a command such as `csr validate --against "<workbook>.xlsx"` that reads the workbook and reports pass/fail per sheet.
- **Template replication tests**: on the HAG template's own data (from `..\hag-regularized-stacking-boosting-meta\resources\experiments\hag-algorithm\*.xlsx`) the default switches must reproduce SET {x₃, x₆, x₁₃, x₄, x₉} and r₁…r₄ (to 1e-12); the meta-algorithm template's new object must be classified as Class 2 with B1(a₄) = ∅, B2(a₄) = {S₉, S₁₀}.
- Cross-check against `docs/handoff/reference-engine` and `golden_values.json`.
- Unit tests for every plug-in; property-based tests (hypothesis) — metric axioms where applicable, ranks form a permutation, 0 ≤ f ≤ 1, 0.5 ≤ g ≤ 1, k range rules; **leakage tests** proving the new-object path never reads its label and LOO never sees the held-out object; GUI smoke tests; coverage ≥ 90 % on the core.

## 6. Git, docs and delivery

- Initialize Git (main branch), `.gitignore`, `.gitattributes` (keep `.xlsx`/`.docx` binary; consider Git LFS for experiment files), conventional commits, one feature branch per milestone, a tag per milestone, CI (GitHub Actions: lint, type-check, tests on Windows and Linux). Commit only after tests pass. Keep `CHANGELOG.md`.
- Docs (mkdocs-material or the template's choice): README with quick start, architecture overview (diagrams), algorithm reference mapping every article formula/step/definition to code, how-to add a metric/normalizer/dataset, GUI user guide, experiment reproduction guide, and `docs/DECISIONS.md` (ADR log — record every decision, including the two switches).
- Keep `resources/experiments/` and `docs/handoff/` unchanged; never modify the template project or the original workbooks.

## 7. Milestones (propose refinements)

M0 analysis & plan → M1 skeleton (packaging, tooling, CI, config, logging) → M2 core pipeline to Ψ(r) + golden tests for Steps 1–8 → M3 HAG + meta-algorithm + template-replication tests → M4 evaluation (LOO re-fit, baselines, metrics, margins, properties) + `validate` command → M5 exporters (Excel mirror, figures, report) → M6 GUI → M7 more metrics/normalizers/datasets → M8 docs, polish, release build (optional PyInstaller for the GUI). After each milestone: summary, test results, what's next.

## 8. Working rules

- Ask me before deciding anything that changes numerical results or the method (list: canonical HAG calculation — template or article; behaviour when min|K_i| < 3; multi-class generalization; which additional metrics/datasets first — see CONTEXT_HANDOFF.md §7). Everything else: choose sensibly, record it in DECISIONS.md.
- Windows paths and PowerShell-friendly commands; no absolute paths inside the code.
- Prefer clarity and correctness over cleverness; every public function documents which article formula/step it implements.
