# Algorithm reference

Every element of the article, the module that implements it and — where the Excel experiment
computes it — the workbook sheet that checks it. The terms and symbols are those of the
[glossary](glossary.md). A dash in the sheet column means that the Excel experiment does not
compute the element; the article is then the only specification.

!!! warning "Four switches deviate from the article"
    - `rips.keep_blocked_cover` — a blocked object keeps covering other objects with |r(S)|
      (ADR-061).
    - `rips.training: moved` — the moved set D is the training sample (ADR-061).
    - `additional_features.class_means: running` — θ and γ are measured from running partial
      class means instead of the final class means M₁ and M₂ (ADR-002).
    - `additional_features.majorizing_passes: 2` — in Step 4 the majorizing function is applied
      twice instead of once (ADR-003).

    All four are off by default; the `excel-experiment` preset turns on the last two (ADR-062).

## Pipeline

E₀ → R-IPS → (S, E) → Ψ_ρ,k(S, E) → D(S, E) → Y(S, E) → R(Y(S, E))

### R-IPS — forming the precedent base

| Article element | Formula / step | Module | Workbook sheet |
|---|---|---|---|
| Initial set E₀, test sample T, working set E₁ = E₀ ∖ T | — | `config.models.RIPSConfig` (`test_sample`) | — |
| Proximity measure ρ, normalization on E₁ | — | `core.rips.selection.rips_distances` | — |
| Radius r(S), sphere count g(S), priority 1/r(S), queue | Steps 1–2 | `core.rips.hyperspheres.hyperspheres` | — |
| Neighbourhood L, blocking r(Sᵢ) = −r(Sᵢ), counter correction g(Sⱼ) − 1, result set D | Steps 3–6 | `core.rips.algorithm.run_rips` (`RIPSResult`, ⚠ `keep_blocked_cover`) | — |
| Training and test sample | — | `core.rips.selection.select_precedents` (⚠ `training`) | — |

Where the article is silent the conventions are: |L| = 0 leaves Sᵢ blocked in E₁; equal radii are
queued by the smaller index; an object with r = 0 never enters the queue; the objects that remain
in E₁ form the training sample (the article only outputs D). Blocking is taken literally: a
blocked object, whose radius is negative, covers nobody (ADR-061).

### Local contexts Ψ_ρ,k

| Article element | Formula / step | Module | Workbook sheet |
|---|---|---|---|
| Data E, types I/J, classes K1, K2 | problem statement | `data.schema.Dataset`, `data.loaders.load_dataset` | Dataset, Quantitative, Nominal |
| Normalization to [0; 1] | Zhuravlyov metric definition | `core.local_context.normalizers.minmax`; per operator: `local_context.operators[].normalizer` (ADR-066); [other normalizers](#normalizers-and-metrics) | Normalized Dataset |
| Base operators ρ, ρ_I, ρ_J | variants of Ψ | `core.local_context.metrics.zhuravlyov`, `core.local_context.operators.resolve_operators`; [other metrics](#normalizers-and-metrics) | the distance sheet |
| Nested neighbourhoods | local context Ψ_ρ,k | `core.local_context.neighbours.neighbour_order`, `rank_matrix` | Sorted Neighbors (ρ, ρ_I, ρ_J) |
| Admissible values of k, K̂ | odd k ≤ min(\|K1\|, \|K2\|) | `core.local_context.admissible_k.min_class` (ADR-063); the Excel experiment: `twice-min-class`, k ≤ 2·min\|Kᵢ\| − 3 (ADR-005) | Parameters, Synthetic Features (k-NN) |

### Synthetic features Ψ(r)

| Article element | Formula / step | Module | Workbook sheet |
|---|---|---|---|
| Own-class count μ (training side) | section 1.3 | `core.synthetic_features.encoders.own_class_counts` | Synthetic Features (k-NN) |
| Binary synthetic feature a ∈ {1, 2} from χ₁ | formula (5) | `core.synthetic_features.encoders.k1_counts`, `formula_5` | Ψ(r) Binary Features |
| Zero value: a = 0 where f_k(μ) = 0.5, such features left out | the article's experiments | `synthetic_features.zero_value` (ADR-064) | — |
| Original nominal features as features of Ψ(r) | — | `synthetic_features.original_features` (ADR-065) | — |
| Membership, stability, metaobject | formulas (1), (2); section 1.4 | `core.synthetic_features.membership.membership_table`, `stability` | Membership & Stability |
| Class-sequence bit masks (β = 2ᵏ − 1), majority bit chains | sections 1.2, 1.3 | `core.synthetic_features.bit_representations.class_sequence_masks`, `majority_chains` (ADR-067) | Membership & Stability (majority chains) |
| Class boundary, informativeness ω | formulas (3), (4) | `core.synthetic_features.membership.boundary`, `informativeness` | Informativeness ω |
| Contributions η, generalized estimate R(S) | formula (6) | `core.synthetic_features.contributions.contributions` | Ψ(r) Contribution & Weight |
| The fitted operator Ψ | Steps 1–8 together | `core.synthetic_features.psi.fit_psi`, `PsiOperator` | — |

### Additional (latent) features and the meta-algorithm

| Article element | Formula / step | Module | Workbook sheet |
|---|---|---|---|
| Hierarchical agglomerative algorithm, TUPLAM, d₁ … d_p | Steps 1–5 | `core.additional_features.agglomerative.agglomerative` (`AgglomerativeResult`, `scan`) | the four iteration sheets |
| Majorizing function φ, margin regularization parameter α | Steps 3–4 | `core.additional_features.majorizing_functions` (sigmoid; the article does not define φ, ADR-068) | the four iteration sheets |
| Meta-description Y = (y₀, …, y_p, d₁, …, d_p) | — | `core.model.MetaTable`, `fit_model` | Dataset for Meta-algorithm |
| New object without its class | Theorem, Corollary | `core.model.CSModel.represent` / `classify` / `predict` (no label argument) | Brace for Meta-algorithm |
| Meta-algorithm, B1/B2, refusal | Steps 1–5 | `core.meta_algorithm.algorithm.meta_classify`, `DECISION_RULES`; `CSModel.classify_training` | Meta-algorithm (+ All Objects) |

### Evaluation and properties

| Article element | Formula / step | Module | Workbook sheet |
|---|---|---|---|
| Margins with / without the majorizing function | — | `evaluation.margins.margin_analysis` | Margin Analysis |
| Training correctness | Definition 2 | `evaluation.protocols` (`resubstitution`), `evaluation.metrics`, `evaluation.roc` | Accuracy, Confusion Matrix, Precision, Recall, F1 Score, ROC Curve & AUC |
| Generalization correctness | Definition 3 | `evaluation.protocols`: `rips`, `nested-cross-validation` (ADR-069), `leave-one-out`, k-fold, hold-out; `evaluation.baselines` | Leave-One-Out |
| The switches | — | `config.presets.DEVIATIONS`, `services.sensitivity.switch_sensitivity` | the switches sheet, Sensitivity (Switches) |
| Determinacy, sufficiency, contextual equivalence | Definitions 1, 4, 6; Property 1; Theorem | `core.properties.model_properties`, `theorem_check` | Model Properties |
| Acceptance tests | — | `services.validation.validate_workbook`, `csr validate` | Validation — and every sheet above |

## Normalizers and metrics

The article defines the model for *a* metric ρ on heterogeneous features; its experiments use the
Zhuravlyov, Euclidean, Chebyshev and Canberra metrics. All are plug-ins, chosen per base operator
and for R-IPS ([Extending](extending.md)); with any of them the rest of the pipeline — neighbour
order, ties, admissible k, formulas (1)–(6), the agglomerative algorithm, the meta-algorithm, the
Theorem — is unchanged.

**Normalizers** (`core.local_context.normalizers`, ADR-052). Constants are estimated on the
training objects (in R-IPS on E₁); nominal codes are never changed; a quantitative feature without
spread becomes 0. An operator can have its own normalizer, e.g. `none` for the Euclidean metric on
the raw quantitative features (ADR-066).

| Name | x′ for j ∈ I | Notes |
|---|---|---|
| `minmax` (default) | (x − min) / (max − min) | normalization to [0; 1] |
| `z-score` | (x − mean) / std | sample standard deviation; `ddof: 0` for the population one |
| `robust` | (x − median) / (Q3 − Q1) | quartiles by linear interpolation; the range replaces an interquartile range of 0 |
| `max-abs` | x / max\|x\| | zero stays zero |
| `decimal-scaling` | x / 10ʲ | j the smallest integer with max\|x′\| < 1 |
| `rank` | (mid-rank − 1) / (m − 1) | new values are interpolated between training values and stay in [0, 1] |
| `unit-length` | x_I / ‖x_I‖₂ | per object; nothing is estimated |
| `none` | x | |

**Metrics** (`core.local_context.metrics`, ADR-050, ADR-051). x′, y′ are normalized values; every
result is rounded to 10 decimals and ties go to the smaller original index, as for ρ.

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
therefore not offered (ADR-051). With min–max normalization `gower` is ρ divided by the number of
features — the same neighbours up to the rounding of the distances — and `hamming` on J and
`manhattan` on I are ρ_J and ρ_I.

## Notes for the article text

Found while implementing the article; for the author to decide.

- **Admissible k.** The text prints the condition [k/2] − k/2 = 0, which selects even k; the odd
  k of section 1.3 (k = 3, 5, …, k ≤ min(|K1|, |K2|)) are implemented (ADR-063).
- **φ, α, δ, ϰ, cr1₀.** The majorizing function is not defined; the logistic sigmoid of the Excel
  experiment is used (ADR-068). α, δ, ϰ and cr1₀ are not given either; the defaults are 0.3, 0.1,
  5 and 10, and the article's experiments use α = 0.1 and 0.2.
- **R-IPS.** Blocking makes r(S) negative, so a blocked object covers nobody in Step 4; the text
  does not say whether that is meant. Equal radii, |L| = 0 and r = 0 are not covered, and the
  algorithm outputs only D — which set is the training sample is not stated (ADR-061).
- **ϰ.** The text says ϰ ≤ n − 1; the bound that matters is ϰ ≤ r − 1 (number of synthetic
  features).
- **β in (2).** "β = 2k − 1" fits the class-sequence bit masks as β = 2ᵏ − 1; for the counts μ,
  β = k (ADR-067).
- **Dimension of Y.** Y = (v₀, …, v_p, z₁, …, z_p) has 2p + 1 components, not 2p − 1.
- **Agglomerative Step 3, γ = 0.** θ/γ is undefined when γ = 0; the implementation treats it as +∞
  (the candidate is never chosen, ADR-025).
- **Agglomerative Step 3, no θ/γ < cr1.** If no candidate improves on the initial cr1 the
  algorithm stops without adding a feature (ADR-008); the steps do not mention this case.
- **Meta-algorithm with p = 0.** When TUPLAM has a single feature only Step 1 applies.
- **The experiments.** Table 1 counts 31 synthetic features for k ∈ {3, …, 41}, which has 20
  values; on Heart the slope x₁₁ is treated as quantitative (7 quantitative features), while the
  dataset file flags it nominal (ADR-072). See [Computational experiment](experiments.md).
- **ω at large k.** ω (formula (4)) is computed from μ, the neighbours of the object's *own*
  class. When k approaches m, μ separates the classes trivially and ω → 1, while the class-free
  feature (5) becomes constant. With the Excel experiment's k range (k ≤ 2·min|Kᵢ| − 3) this
  dominates the agglomerative algorithm on Heart-Disease (270, 13, 2): leave-one-out 7.0 %, against
  81.5 % with constant features skipped and k ≤ 21 (ADR-030) and 80.7 % with the article's
  k ≤ min|Kᵢ|, where no synthetic feature is constant (ADR-076).
- **Which metrics ρ are admissible.** The Theorem needs the distances of a new object to be
  computable without any class. A metric whose definition uses class labels of the training
  sample (HVDM's value difference part) lets the class of a *training* object enter its own
  distances; the text could require ρ to be defined on the feature values alone (ADR-051).
