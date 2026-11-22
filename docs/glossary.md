# Glossary

The terms of the article, in the order of its method, with the English name used in the
documentation and the name used in the code. Symbols are those of the article.

Spelling: Oxford English, as in the rest of the project (*normalization*, *generalized*,
*neighbour*, *colour*); the metric is the *Zhuravlyov* metric.

## The model

| Article (Russian) | Symbol | Documentation | Code |
|---|---|---|---|
| контекстно-синтетическая модель алгоритмов распознавания (КС-модель) | M_CS = (X, E, ρ, K̂, K, Ψ, D, R) | context-synthetic model of recognition algorithms (CS-model) | `CSModel`, `fit_model` |
| алгоритм модели | A(S, E) = R(Y(S, E)) | algorithm of the model | `CSModel.classify` |
| пространство исходных признаков | X, X(n) = (x₁, …, xₙ) | space of original features | `Dataset.X` |
| количественные / номинальные признаки | I / J | quantitative / nominal features | `quantitative` |
| обучающая выборка | E₀, E | training sample | `training` |
| множество классов | K = {K₁, K₂} | classes | `classes`, `class_index` |
| допустимый объект | S ∈ Ŝ | admissible object | — |
| распознающий оператор | | recognition operator | — |
| базовый оператор (базовый алгоритм) | Bⱼ | base operator | `BaseOperator` |
| семейство операторов формирования синтетических признаков | Ψ, Ψ_ρ,k(S, E) | family Ψ of synthetic-feature operators | `PsiOperator`, `fit_psi` |
| семейство операторов формирования дополнительных признаков | D | operator D of the additional features | `fit_additional_features` |
| решающее правило метаалгоритма | R | decision rule of the meta-algorithm | `decision_rule` |
| метаописание | Y(S, E) | meta-description | `MetaDescription` |
| метапризнаки | | meta-features | — |

## R-IPS — forming the precedent base for training and testing

| Article (Russian) | Symbol | Documentation | Code |
|---|---|---|---|
| формирование базы прецедентов для обучения и тестирования | | forming the precedent base | stage `rips` |
| алгоритм R-IPS | | R-IPS | `run_rips`, `RIPSResult`, configuration section `rips` |
| множество объектов | E₀ | initial set | `objects` |
| тестовая выборка | T | test sample | `test_sample` |
| E₁ = E₀ \ T | E₁ | working set; after the run, the remaining objects | `remaining` |
| результирующее множество | D | result set (the moved objects) | `moved` |
| система гипершаров | | hyperspheres | `hyperspheres` |
| расстояние до ближайшего объекта противоположного класса | r(S) | radius | `radius` |
| объекты своего класса с ρ < r(S) | g(S) | sphere count | `sphere_count` |
| приоритет | 1/r(Sᵢ) | priority | `priority` |
| упорядоченный список | | queue | `queue` |
| окрестность | L | neighbourhood L | `neighbourhood` |
| «заблокировать» объект | r(Sᵢ) = −r(Sᵢ) | blocking | `blocked` |
| коррекция характеристик соседей | g(Sⱼ) = g(Sⱼ) − 1 | counter correction | `corrections` |

## Local contexts

| Article (Russian) | Symbol | Documentation | Code |
|---|---|---|---|
| мера близости, метрика | ρ | proximity measure, metric | `metric`, `METRICS` |
| метрика Журавлёва / Евклида / Чебышева / Канберра | | Zhuravlyov / Euclidean / Chebyshev / Canberra metric | `zhuravlyov`, `euclidean`, `chebyshev`, `canberra` |
| нормирование значений в [0; 1] | | normalization to [0; 1] | `normalizer: minmax` |
| локальный контекст | Ψ_ρ,k(S, E) | local context | `LocalContext` |
| k ближайших соседей; последовательность вложенных окрестностей | k | k nearest neighbours; nested neighbourhoods | `neighbour_order` |
| множество допустимых значений параметра k | K̂ | admissible values of k | `AdmissibleK`, `admissible_k` |

## Bit representations

| Article (Russian) | Symbol | Documentation | Code |
|---|---|---|---|
| битовая последовательность (маска) по последовательности классов соседей | β = 2ᵏ − 1 | class-sequence bit mask | `class_sequence_masks` |
| битовая цепочка по мажоритарному правилу | | majority bit chain | `majority_chains` |
| мажоритарное правило | dᵢ > k/2 | majority rule | — |
| градация признака в номинальной шкале | μ | gradation | `gradations` |
| количество соседей того же класса | μ, dᵢ | own-class count | `own_class_counts` |
| представители классов K₁, K₂ среди k ближайших | χ₁(S, k), χ₂(S, k) | class counts | `k1_counts`, `chi1` |

## Synthetic features Ψ(r)

| Article (Russian) | Symbol | Documentation | Code |
|---|---|---|---|
| синтетический признак | a[k/2] ∈ Ψ(r) | synthetic feature | `SyntheticFeature` |
| набор синтетических признаков | Ψ(r), r | set of synthetic features, its size r | `features`, `r` |
| функция принадлежности | f_k(μ), formula (1) | membership function | `membership_table` |
| число объектов с градацией μ в K₁, K₂ | d₁ₖ(μ), d₂ₖ(μ), n(μ) | gradation counts | `d1`, `d2`, `n` |
| устойчивость признака | g_k, formula (2), β | stability | `stability` |
| различимость | | distinguishability | — |
| метаобъект | (g_k) | metaobject | `metaobject` |
| граница между классами | G_k, q₁, q₂, formula (3) | class boundary | `boundary` |
| информативность признака | ω[k/2], formula (4) | informativeness | `informativeness` |
| значение функции принадлежности объекта | g(S, k) | object membership | `object_membership` |
| синтетический признак в множестве {1, 2} | a, formula (5) | binary synthetic feature | `encoder: formula-5` |
| значение 0 (классы неразличимы по функции принадлежности) | a = 0 | zero value | `zero_value` |
| вклад градации | η\[k/2\](j), α¹ⱼ, α²ⱼ, formula (6) | contribution | `contributions` |
| обобщённая оценка | R(Sₜ) = Σ ηᵢ(aₜᵢ) | generalized estimate | `estimates` |
| набор Ψ(d) ⊂ Ψ(r) | Ψ(d) | selected synthetic features | — |

## Additional (latent) features D(t)

| Article (Russian) | Symbol | Documentation | Code |
|---|---|---|---|
| дополнительные (латентные) признаки | D(t); dᵢⱼ; zⱼ | additional (latent) features d₁, …, d_p | `additional`, notation `d₁` |
| число дополнительных признаков | p = \|TUPLAM\| − 1 | p | `p` |
| иерархический агломеративный алгоритм (группировка) | Steps 1–5 | hierarchical agglomerative algorithm | `agglomerative`, `AgglomerativeResult` |
| жадная стратегия | | greedy strategy | — |
| множества индексов признаков | P, TUPLAM | candidates P, TUPLAM | `candidates`, `tuplam` |
| мажорирующая функция | φ(·) | majorizing function | `majorizing_function`, `MAJORIZING_FUNCTIONS` |
| параметр для регуляризации отступа | α, 0 < α < 1 | margin regularization parameter | `alpha` |
| отступ от границы между классами | | margin | `margins` |
| внутриклассовое сходство | θ | intra-class similarity | `theta` |
| межклассовое различие | γ | inter-class difference | `gamma` |
| порог для отношения θ/γ | δ, 0 < δ < 0.5 | threshold | `delta` |
| максимальное число дополнительных признаков | ϰ | maximum number of additional features | `kappa` |
| критерий | cr1, crit | criterion | `cr1`, `crit` |
| средние значения классов | M₁, M₂ | class means | `class_means` |
| регуляризатор | | regularizer | — |

## Meta-algorithm

| Article (Russian) | Symbol | Documentation | Code |
|---|---|---|---|
| метаалгоритм | Steps 1–5 | meta-algorithm | `meta_algorithm`, `meta_classify` |
| набор синтетических и дополнительных признаков | Y = (v₀, …, v_p, z₁, …, z_p) | meta-description columns | `MetaDescription` |
| описание объекта обучающей выборки | Sᵢ = (aᵢ₀, …, aᵢₚ, dᵢ₁, …, dᵢₚ) | training row | `MetaDescription.values`, `.additional` |
| множества объектов | B1(aⱼ), B2(aⱼ) | B1, B2 | `in_b1`, `in_b2` |
| отказ | 0 | refusal | `REFUSAL` |

## Properties and evaluation

| Article (Russian) | Symbol | Documentation | Code |
|---|---|---|---|
| Определение 1. Полнота КС-модели | | Definition 1 — completeness of the CS-model | `properties` |
| Свойство 1. Определённость представления | | Property 1 — determinacy of the representation | `Determinacy` |
| Определение 2. Обучающая корректность | | Definition 2 — training correctness | protocol `resubstitution` |
| Определение 3. Обобщающая корректность | E_v | Definition 3 — generalization correctness | protocols `rips`, `leave-one-out`, `stratified-k-fold`, … |
| независимая выборка | E_v | independent sample | — |
| вложенная кросс-валидация | | nested cross-validation | protocol `nested-cross-validation` |
| Определение 4. Достаточность контекстно-синтетического представления | | Definition 4 — sufficiency of the representation | `sufficiency` |
| Определение 5. Полнота семейства базовых операторов | B = {B₁, …, B_q} | Definition 5 — completeness of the family of base operators | `properties` |
| Определение 6. Контекстная эквивалентность метрик | ρ₁ ∼_Ψ,k ρ₂ | Definition 6 — contextual equivalence of metrics | `properties` |
| Утверждение, Теорема, Следствие, Замечание | | Statement, Theorem, Corollary, Remark | `theorem_check` |
| вычислительный эксперимент | | computational experiment | `experiments/` |
| точность: обучение / тест | | training / test accuracy | `accuracy` |
