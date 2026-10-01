# Context handoff — Context-Synthetic Model (CS-model) experiment

This file distils the design conversation (see `CHAT_TRANSCRIPT.md`) into the decisions the Python project must follow.
Where this file and the transcript differ, **this file wins** (it records the final state of every decision).

Session of origin: https://claude.ai/code/session_01Drh9SjE6L3BXHRB9Hw8LdM (web session — not readable from the CLI; the transcript export is in this folder).

---

## 1. Sources

| Source | Where | Role |
|---|---|---|
| Full Excel experiment | `resources/experiments/context-synthetic-model/Context-Synthetic Model – Full Experiment [Heart-Disease (10, 13, 2)] - Opus 5.5.xlsx` | **Executable specification.** 33 sheets (incl. “Template Deviations”), 10 615 live formulas, every intermediate table of the pipeline, a Validation sheet with golden values. |
| k-NN synthetic-features workbook | `resources/experiments/context-synthetic-model/Forming Synthetic Features based on k-NN (Heart-Disease 10,13,2) - Opus 5.5.xlsx` | First part of the experiment (Steps 1–4), the user's hand-checked version. |
| Draft article (Russian) | `docs/handoff/article/article-draft-cs-model.docx` | «Контекстно-синтетическая модель алгоритмов распознавания на основе локальных метрических отношений» (Ensembles of algorithms and Zhuravlev's algebra – 6). Formulas (1)–(6), HAG Steps 1–5, meta-algorithm Steps 1–5, Definitions 1–6, Theorem. The “Вычислительный эксперимент” section is still empty — this project fills it. |
| Hand-drawn algorithm sketches | `docs/handoff/images/algorithm-sketch-*.jpg` | Order of the pipeline: dataset → S₁ → S₂…S₁₀ distances → sorted neighbours with classes and nested k=3,5,7 → k-NN table (№, k=3, k=5, …, Class). |
| k-range formula | `docs/handoff/images/k-range-formula.png` | k ∈ [k_min, k_max], k_max = 2·min_i |K_i| − 3 (the 6 and 4 in the image are only an example). |
| Zhuravlev metric | `docs/handoff/images/zhuravlev-metric-definition.png` | ρ(x, y) = Σ_{j∈I} |x_j − y_j| + Σ_{j∈J} [x_j ≠ y_j]; quantitative values mapped to [0, 1] by the fractional-linear (min–max) transform. |
| Reference engine | `docs/handoff/reference-engine/csmodel/` + `golden_values.json` | Dependency-free Python that reproduces the workbook exactly. An executable spec for tests — not the target architecture. |
| Template project (read-only) | `../hag-regularized-stacking-boosting-meta/` | Earlier project of the same author: structure, style, conventions; its `resources/experiments/{hag-algorithm,meta-algorithm,model-evaluation}/*.xlsx` are the HAG, meta-algorithm and model-evaluation templates the full experiment follows. |

## 2. Dataset format (Heart-Disease (10, 13, 2))

Sheet `Dataset`: rows 3–12 = objects S₁…S₁₀; columns B–N = features x₁…x₁₃; column O = class (1 or 2); row 13 = feature-type flags
(1 = quantitative, set I; 0 = nominal, set J). Here I = {x₁, x₄, x₅, x₈, x₁₀, x₁₂}, J = {x₂, x₃, x₆, x₇, x₉, x₁₁, x₁₃}; |K1| = 4, |K2| = 6.
Symbols on Dataset/Quantitative/Nominal are Office-Math drawings over the cells (ignore them when parsing).
The project must accept other heterogeneous datasets (CSV/XLSX) with a schema: feature types, class column, optional object ids.

## 3. Pipeline (final decisions) — (S, E) → Ψ_ρ,k(S, E) → D(S, E) → Y(S, E) → R(Y(S, E))

1. **Scale unification**: for j ∈ I, x′ = (x − min)/(max − min), min/max from the **training** objects only (0 if max = min); nominal codes unchanged. A new object uses the training min/max (values may fall outside [0, 1]; not clipped).
2. **Base operators** (same Zhuravlev metric on different feature sets — the article's "variants of Ψ"):
   ρ = ρ_I + ρ_J (all features), ρ_I = Σ_{j∈I}|x′_j − y′_j| (quantitative only), ρ_J = Σ_{j∈J}[x_j ≠ y_j] (nominal only, Hamming).
   Distances **rounded to 10 decimals** (quantitative part first, then the sum) so equal distances tie exactly.
3. **Neighbours**: for each object, all other objects sorted by (distance, original index) — **ties → smaller original index**; the object itself is excluded. Each neighbour keeps its original index and class.
4. **Permitted k**: odd k = k_min, k_min+2, …, k_max with **k_min = 3 fixed for every dataset** and **k_max = 2·min_i|K_i| − 3**, class sizes counted from the (training) data — never hard-coded. Here k ∈ {3, 5}. (At k = k_max a same-class majority needs (k+1)/2 = min|K_i| − 1 neighbours.) If min|K_i| < 3 no k is permitted. This replaces the article text r = |{k odd, k ≤ min(|K1|,|K2|)}| — flag that text for correction.
5. **Same-class count μ** (Step 4, training side only — uses the object's own class): number of the k nearest neighbours in the class of S_i.
6. **Synthetic features Ψ(r)** — formula (5), class-free: χ₁ = number of K1 objects among the k nearest; a = 1 if χ₁ > ⌊k/2⌋, a = 2 if χ₂ = k − χ₁ > ⌊k/2⌋. One feature per (operator, permitted k): a₁=(ρ,3), a₂=(ρ,5), a₃=(ρ_I,3), a₄=(ρ_I,5), a₅=(ρ_J,3), a₆=(ρ_J,5); r = 3 × |permitted k| = 6.
7. **Membership (1)**: f_k(μ) = (d₁ₖ(μ)/|K1|)/(d₁ₖ(μ)/|K1| + d₂ₖ(μ)/|K2|) for μ = 0…k over the μ gradations; undefined (“—”) when no object has μ.
   **Stability (2)**: g_k = (1/m) Σ n(μ)·max(f, 1 − f), β = k for counts; bit masks (Task 2, bit = 1 if μ > k/2, first permitted k = most significant bit) use β = 2^r − 1. Meta-object = vector (g) of the sample.
8. **Boundary (3)**: G_k = (q₁ + q₂)/2, q₂ = max{f < 0.5}, q₁ = min{f > 0.5}; if one side is empty → G_k = 0.5.
   **Informativeness (4)**: ω = (|{S∈K1: f(μ_S) > G}| + |{S∈K2: f(μ_S) < G}|)/m — the weight of the synthetic feature.
9. **Contributions (6)**: η(j) = ω·(α¹_j/|K1| − α²_j/|K2|), j ∈ {1, 2}, α counts of a = j in K1/K2. Ψ(r) expressed as contribution values η_u(a_tu) + weights + ranks = HAG input (plays the role of the template sheet “Dataset (Contribution & Weight)”).
10. **HAG (Steps 1–5)**, template parameters α = 0.3, δ = 0.1, ϰ = 5 (ϰ ≤ r − 1), cr1 = 10, majorizer ϕ = logistic sigmoid σ(x) = 1/(1+e^(−x)), b ← b + α·σ(−b) for K1, b ← b − α·σ(−b) for K2.
    STEP 2: u = argmax ω (first on ties), R(S_t) = η_u(a_tu). STEP 3: for every remaining candidate (feature order) b = R + η_cand, majorize, class centres, θ = Σ|b − M(own)|, γ = Σ|b − M(other)|, cr1 = min θ/γ (strict <, first on ties), q = argmin. STEP 4: add q, crit = cr1, update R, continue while |TUPLAM| < ϰ and crit > δ and P ≠ ∅. Latent feature r_j = R after iteration j; p = |TUPLAM| − 1.
11. **Meta-dataset** Y = (y₀…y_p, r₁…r_p): y_j = contribution values of the TUPLAM features in TUPLAM order; meta-algorithm training description a_i (gradations {1,2} of the TUPLAM features) and d_i = r values.
12. **Meta-algorithm (Steps 1–5)**: B1(a₀) = {S_i∈K1 | a_i0 = a₀}, B2(a₀) = {S_i∈K2 | a_i0 = a₀}; for j = 1…p: B1(a_j) = {S_i∈B1(a_{j−1}) | a_ij = a_j, d_ij > 0}, B2(a_j) = {S_i∈B2(a_{j−1}) | a_ij = a_j, d_ij < 0}; Step 4: K1 if |B1|/|K1| > |B2|/|K2|, K2 if <, **0 = refusal** if equal.
13. **New object (Theorem)**: its context is computed relative to the fixed training sample E with the same operators, rescaling and tie rule; its a-values use only the neighbours' classes. The class of S is never an input of Ψ, D or R. (Workbook demo: typing a training object and excluding it from its own context reproduces its training row.)
14. **Evaluation**: resubstitution (training correctness, Definition 2), **leave-one-out with the whole pipeline re-fitted on the other m − 1 objects** (generalization, Definition 3 — the k range adapts per fold), baselines = plain k-NN majority vote per operator and k (LOO). Metrics: TP/TN/FP/FN with class 1 positive, refusals separate (count as errors in accuracy), coverage, precision/recall/F1 per class + macro, AUC = Mann–Whitney on score = score₁ − score₂ (**round scores to 10 decimals** before comparing, e.g. 0.5 − 2/3 vs 0 − 1/6), ROC points. Margin analysis of every latent feature with and without majorizer (b = (min_K1 + max_K2)/2, width = min_K1 − max_K2, m_i = y_i(d − b), ŷ = 1 if d > b).
15. **Model-property checks**: Definition 1/Property 1 (every stage defined), 2, 3, 4 (sufficiency: pairs with identical description and different classes), 6 (contextual equivalence of operators: identical k-neighbour sets; identical synthetic features), ties at the k boundary.

## 4. ⚠ Two template calculations differ from the article — highlight them everywhere

1. **θ and γ are measured from running partial class means instead of the final class means M₁ and M₂** (template cells: |b_t − M| uses the partial mean up to row t, although the template's formula box shows M = Σb/|K|).
2. **In STEP 4 the majorizer is applied twice instead of once** (the template majorizes the already-majorized b of STEP 3 again; the article: R ← R + η_q, then one majorizer).

Both are configuration switches: `centres = running | final`, `step4_passes = 2 | 1`. **Default = template** (reproduces the templates exactly); the article setting is `final` + `1`. Effect on Heart-Disease: default TUPLAM {a₆, a₃, a₁, a₂, a₄} (resubstitution 70 %, LOO 20 %); article {a₆, a₁, a₂, a₄, a₅} (50 %, 0 %). On the HAG template's own data the default reproduces SET {x₃, x₆, x₁₃, x₄, x₉} and r₁…r₄ to 4.4e-16; the article setting gives {x₃, x₁, x₅, x₉, x₆}. The user must decide which is canonical — keep both.

## 5. Golden values (Heart-Disease, default settings) — acceptance tests

- permitted k = 3, 5; k_max = 5; r = 6
- ω = [0.9, 0.9, 0.6, 0.9, 0.9, 1.0]; η(1) = [−0.15, −0.15, 0.2, −0.15, −0.15, −1/6]; η(2) = [0.15, 0.15, −0.2, 0.15, 0.15, 1/6]
- TUPLAM = {a₆, a₃, a₁, a₂, a₄}; crit = 0.428155786666, 0.305276758463, 0.273362813270, 0.271025654647; iteration 4 stops (|TUPLAM| = ϰ)
- r₁ = [0.112062247, 0.2602395141, −0.7336758046, 0.6034761627, 0.6034761627, 0.2602395141, −0.3497306129 ×4]
- r₄ = [−0.3086791131, 1.2656382585, −2.6615557218, 1.5033926584, 1.5033926584, 1.2656382585, −1.0143821466 ×4] (r₂, r₃ in `golden_values.json`)
- resubstitution predictions = 1, 2, 2, 1, 1, 2, 2, 2, 2, 2 (7/10); AUC 0.6667
- LOO predictions = 1, 0, 0, 0, 0, 1, 2, 1, 1, 1 (2/10, 4 refusals); AUC 0.2708
- LOO k-NN baselines: ρ k=3 50 %, ρ k=5 50 %, ρ_I k=3 70 %, ρ_I k=5 40 %, ρ_J k=3 50 %, ρ_J k=5 50 %
- margin widths with majorizer r₁…r₄ = 0.1482, 0.6496, 1.1201, 1.5743; without = −0.4000 each
- Definition 6: ρ vs ρ_J identical k-sets 5/10 (k=3), 2/10 (k=5); ρ_J ties at the k boundary 5/10 (k=3), 8/10 (k=5); 5 of 6 synthetic features identical (only a₃ differs); Definition 4: 10 conflicting pairs
- switch variants: see `golden_values.json` → `variants`; template replication → `tpl`

## 6. Findings to carry into the article / research backlog

- Very little diversity in Ψ(r) on this sample; ρ is dominated by its nominal part; ρ_J has many ties (tie rule decides).
- The representation is insufficient on E₀ (Definition 4) → training correctness fails for 3 objects.
- LOO is extremely noisy with m = 10; real datasets are needed (the full UCI Heart-Disease and others).
- The majorizer widens every latent margin (the regularisation works as intended on the training side).

## 7. Open questions (ask the user; do not guess silently)

1. Canonical HAG calculation: template (running centres, 2 passes) or article (final centres, 1 pass)?
2. k_min = 3 fixed — what should happen when min|K_i| < 3 (no permitted k)?
3. Multi-class generalization of (1)–(6), HAG and the meta-algorithm (article is binary).
4. Other metrics to include in Ψ first (e.g. HEOM/HVDM/Gower, Euclidean on I) and whether the Zhuravlev metric should be weighted.
5. Real datasets to support first, and their feature-type schemas.

## 8. Excel sheet → code mapping (from the workbook Overview)

Parameters → config; Dataset → io.load_dataset; Normalized Dataset → preprocessing (normalizers); Zhuravlev Distances → metrics; Sorted Neighbors (ρ, ρ_I, ρ_J) → neighbours; Synthetic Features (k-NN) → μ counts; Ψ(r) Binary Features → encoders (5); Membership & Stability → (1), (2), bit masks; Informativeness ω → (3), (4); Ψ(r) Contribution & Weight → (6); Greedy upon Weight (1–4-Latent) → hag; Dataset for Meta-algorithm / Brace → meta-dataset, new-object representation; Meta-algorithm (+ All Objects) → meta.classify; Margin Analysis, Accuracy, Confusion Matrix, Precision/Recall/F1, ROC & AUC, Leave-One-Out, Sensitivity → evaluation; Model Properties → property checks; Validation → golden tests.
