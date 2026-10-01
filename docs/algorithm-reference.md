# Algorithm reference

Every element of the article, the workbook sheet that specifies it and the module that implements
it. Modules marked *planned* arrive with the milestone shown.

!!! warning "Two template calculations differ from the article"
    - θ and γ are measured from running partial class means instead of the final class means M₁
      and M₂ — switch `hag.centres` (ADR-002).
    - In STEP 4 the majorizer is applied twice instead of once — switch `hag.step4_passes`
      (ADR-003).

## Pipeline

(S, E) → Ψ_ρ,k(S, E) → D(S, E) → Y(S, E) → R(Y(S, E))

| # | Article element | Formula / step | Workbook sheet | Module | Milestone |
|---|---|---|---|---|---|
| 0 | Parameters | — | Parameters | `config.models` | M1 ✓ |
| 1 | Data E₀, types I/J, classes K1, K2 | problem statement | Dataset, Quantitative, Nominal | `data.schema.Dataset`, `data.loaders.load_dataset` | M2 ✓ |
| 2 | Scale unification (fractional-linear) | Zhuravlyov metric definition | Normalized Dataset | `core.normalizers.minmax`; [other normalizers](#normalizers-and-metrics) | M2 ✓, M7 ✓ |
| 3 | Base operators ρ, ρ_I, ρ_J | variants of Ψ | Zhuravlev Distances | `core.metrics.zhuravlyov`, `core.operators.resolve_operators`; [other metrics](#normalizers-and-metrics) | M2 ✓, M7 ✓ |
| 4 | Nested neighbourhoods | local context Ψ_ρ,k | Sorted Neighbors (ρ, ρ_I, ρ_J) | `core.neighbours.neighbour_order`, `rank_matrix` | M2 ✓ |
| 5 | Permitted k | k_max = 2·min\|Kᵢ\| − 3 | Parameters, Synthetic Features (k-NN) | `core.k_strategies.formula` (ADR-022) | M2 ✓ |
| 6 | Same-class count μ | section 1.3 | Synthetic Features (k-NN) | `core.encoders.same_class_counts` | M2 ✓ |
| 7 | Synthetic features Ψ(r) | formula (5) | Ψ(r) Binary Features | `core.encoders.k1_counts`, `formula_5` | M2 ✓ |
| 8 | Membership, stability, meta-object, bit masks | formulas (1), (2); section 1.4 | Membership & Stability | `core.membership.membership_table`, `stability`, `bit_masks` | M2 ✓ |
| 9 | Boundary, informativeness | formulas (3), (4) | Informativeness ω | `core.membership.boundary`, `informativeness` | M2 ✓ |
| 10 | Contributions | formula (6) | Ψ(r) Contribution & Weight | `core.contributions.contributions`; Steps 1–8 together: `core.context.fit_context` | M2 ✓ |
| 11 | Hierarchical agglomerative grouping | Steps 1–5 | Greedy upon Weight (1–4-Latent) | `core.hag.hag` (trace: `HAGResult`, `scan`), `core.majorizers` | M3 ✓ |
| 12 | Meta-description Y = (y, r) | Y(2p − 1) | Dataset for Meta-algorithm | `core.model.MetaDataset`, `fit_model` | M3 ✓ |
| 13 | New object without its class | Theorem, Corollary | Brace for Meta-algorithm | `core.model.CSModel.represent` / `classify` / `predict` (no label argument) | M2 / M3 ✓ |
| 14 | Meta-algorithm | Steps 1–5 | Meta-algorithm (+ All Objects) | `core.meta.meta_classify`, `DECISION_RULES`; `CSModel.classify_training` | M3 ✓ |
| 15 | Margins with / without majorizer | — | Margin Analysis | `evaluation.margins.margin_analysis` | M4 ✓ |
| 16 | Training correctness | Definition 2 | Accuracy, Confusion Matrix, Precision/Recall/F1, ROC & AUC | `evaluation.protocols` (`resubstitution`), `evaluation.metrics`, `evaluation.roc` | M4 ✓ |
| 17 | Generalization correctness | Definition 3 | Leave-One-Out | `evaluation.protocols.run_protocol` (`leave-one-out`, k-fold, hold-out), `evaluation.baselines` | M4 ✓ |
| 18 | Template-vs-article switches | — | Template Deviations, Sensitivity (Switches) | `config.presets`, `core.hag`, `services.sensitivity.switch_sensitivity` | M1 / M3 / M4 ✓ |
| 19 | Determinacy, sufficiency, contextual equivalence | Definitions 1, 4, 6; Property 1; Theorem | Model Properties | `core.properties.model_properties`, `theorem_check` | M4 ✓ |
| 20 | Acceptance tests | — | Validation — every computed sheet; the HAG and meta-algorithm templates | `services.validation.validate_workbook`, `csr validate` | M2 → M4 ✓ |

## Normalizers and metrics

The article defines the model for *a* metric ρ on heterogeneous features; the experiment uses the
Zhuravlyov metric with the fractional-linear transform. Both are plug-ins, and the default
reproduces the workbook exactly. The alternatives below are chosen in the configuration
([Extending](extending.md)); with any of them the rest of the pipeline — neighbour order, ties,
permitted k, formulas (1)–(6), HAG, meta-algorithm, the Theorem — is unchanged.

**Step 1 · normalizers** (`core.normalizers`, ADR-052). Constants are estimated on the training
objects; nominal codes are never changed; a quantitative feature without spread becomes 0.

| Name | x′ for j ∈ I | Notes |
|---|---|---|
| `minmax` (default) | (x − min) / (max − min) | the workbook's Step 1 |
| `z-score` | (x − mean) / std | sample standard deviation; `ddof: 0` for the population one |
| `robust` | (x − median) / (Q3 − Q1) | quartiles by linear interpolation; the range replaces an interquartile range of 0 |
| `max-abs` | x / max\|x\| | zero stays zero |
| `decimal-scaling` | x / 10ʲ | j the smallest integer with max\|x′\| < 1 |
| `rank` | (mid-rank − 1) / (m − 1) | new values are interpolated between training values and stay in [0, 1] |
| `unit-length` | x_I / ‖x_I‖₂ | per object; nothing is estimated |
| `none` | x | |

**Step 2 · metrics** (`core.metrics`, ADR-050, ADR-051). x′, y′ are unified values; every result
is rounded to 10 decimals and ties go to the smaller original index, as for ρ.

| Name | Features | Distance |
|---|---|---|
| `zhuravlyov` (default) | any | Σ_I \|x′ⱼ − y′ⱼ\| + Σ_J [xⱼ ≠ yⱼ] |
| `weighted-zhuravlyov` | any | Σ_I wⱼ·\|x′ⱼ − y′ⱼ\| + Σ_J wⱼ·[xⱼ ≠ yⱼ] |
| `heom` | any | √(Σ_I (\|x′ⱼ − y′ⱼ\| / rangeⱼ)² + Σ_J [xⱼ ≠ yⱼ]) |
| `gower` | any | (Σ_I \|x′ⱼ − y′ⱼ\| / rangeⱼ + Σ_J [xⱼ ≠ yⱼ]) / n |
| `manhattan` | I | Σ \|x′ⱼ − y′ⱼ\| |
| `euclidean` | I | √(Σ (x′ⱼ − y′ⱼ)²) |
| `chebyshev` | I | max \|x′ⱼ − y′ⱼ\| |
| `minkowski` | I | (Σ \|x′ⱼ − y′ⱼ\|ᵖ)^(1/p), p ≥ 1 |
| `canberra` | I | Σ \|x′ⱼ − y′ⱼ\| / (\|x′ⱼ\| + \|y′ⱼ\|) |
| `cosine` | I | 1 − (x′·y′) / (‖x′‖·‖y′‖) |
| `mahalanobis` | I | √((x′ − y′)ᵀ S⁺ (x′ − y′)) |
| `hamming` | J | Σ [xⱼ ≠ yⱼ] |

rangeⱼ and the covariance S are *training statistics*: they are estimated on the training sample
(in every fold of a cross-validation) from the feature values alone. No metric reads a class
label; HVDM, whose nominal part is built from the class frequencies of the training objects, is
therefore not offered (ADR-051). With min–max unification `gower` is ρ divided by the number of
features — the same neighbours up to the rounding of the distances — and `hamming` on J and
`manhattan` on I are ρ_J and ρ_I.

## Notes for the article text

Found while building the experiment; for the author to decide.

- **Permitted k.** The text r = |{k odd, k ≤ min(|K1|, |K2|)}| should become k = 3, 5, …,
  2·min|Kᵢ| − 3 (with k = 1 when min|Kᵢ| = 2), see ADR-005.
- **ϰ.** The text says ϰ ≤ n − 1; the bound that matters is ϰ ≤ r − 1 (number of synthetic
  features).
- **β in (2).** "β = 2k − 1" is the number of gradations of a bit mask minus one, 2^(number of bits)
  − 1; for the counts μ, β = k.
- **Dimension of Y.** Y = (v₀, …, v_p, z₁, …, z_p) has 2p + 1 components, not 2p − 1.
- **Meta-algorithm Step 4** is empty in the draft; the rule used is K1 if |B1|/|K1| > |B2|/|K2|,
  K2 if <, 0 (refusal) if equal.
- **HAG STEP 3** in the article uses the final means M₁, M₂ and one majorizer pass in STEP 4; the
  template cells do not (see the warning above).
- **HAG STEP 3, γ = 0.** θ/γ is undefined when γ = 0; the implementation treats it as +∞ (the
  candidate is never chosen, ADR-025). The text could state this.
- **HAG STEP 3, no θ/γ < cr1.** If no candidate improves on the initial cr1 the grouping stops
  without adding a feature (ADR-008); the article's steps do not mention this case.
- **Meta-algorithm with p = 0.** When TUPLAM has a single feature only Step 1 applies.
- **ω at large k.** ω (formula (4)) is computed from μ, the neighbours of the object's *own*
  class. When k approaches m, μ separates the classes trivially and ω → 1, while the class-free
  feature (5) becomes constant. With the literal k range this dominates the HAG on large data
  (Heart-Disease 270: leave-one-out 7 %; 81.5 % with constant features skipped and k ≤ 21,
  ADR-030). The text should bound k or say how ω treats such features.
- **Which metrics ρ are admissible.** The Theorem needs the distances of a new object to be
  computable without any class. A metric whose definition uses class labels of the training
  sample (HVDM's value difference part) keeps that for a *new* object but lets the class of a
  *training* object enter its own distances; the text could require ρ to be defined on the
  feature values alone (ADR-051).
