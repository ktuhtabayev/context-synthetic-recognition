# Extending the pipeline

Every exchangeable part of the method is a plug-in chosen by name from a registry. A plug-in is
a function with the signature of its kind; it may declare a parameter type, which validates the
`params` of the configuration and later generates the GUI form (ADR-019).

| Kind | Registry | Signature | Default |
|---|---|---|---|
| normalizer | `core.normalizers.NORMALIZERS` | `(X, quantitative, params) → Scaling` | `minmax` |
| metric | `core.metrics.METRICS` | `(A, B, quantitative, decimals, params) → distances`, with optional [traits](#metric-traits) | `zhuravlyov` |
| k strategy | `core.k_strategies.K_STRATEGIES` | `(class_sizes, params) → PermittedK` | `formula` |
| encoder | `core.encoders.ENCODERS` | `(chi1, k, params) → values in {1, 2}` | `formula-5` |
| weights | `core.contributions.WEIGHTS` | `(WeightInputs, params) → float` | `omega` |
| majorizer | `core.majorizers.MAJORIZERS` | `(x, params) → ϕ(x)` element-wise | `sigmoid` |
| decision rule | `core.meta.DECISION_RULES` | `(b1, b2, class_sizes, params) → 1, 2 or 0` | `article-step-4` |
| protocol | `evaluation.protocols.PROTOCOLS` | `(class_index, seed, params) → [Split]` | `resubstitution`, `leave-one-out` |
| baseline | `evaluation.baselines.BASELINES` | `(BaselineInput, params) → [BaselineOutput]` | `knn-vote` |
| dataset format | `data.loaders.LOADERS` | `(path, LoadOptions) → Dataset` | detected |
| export format | `export.run.EXPORTERS` | `(ExportContext) → [Path]` | `all` ([Exporting](exporting.md)) |

## Built-in metrics and normalizers

| Metric | Features | Parameters |
|---|---|---|
| `zhuravlyov` (default), `heom`, `gower` | any | — |
| `weighted-zhuravlyov` | any | `quantitative_weight`, `nominal_weight`, `feature_weights` |
| `manhattan`, `euclidean`, `chebyshev`, `canberra`, `cosine`, `mahalanobis` | quantitative | — |
| `minkowski` | quantitative | `p` ≥ 1 (default 3) |
| `hamming` | nominal | — |

Normalizers: `minmax` (default), `z-score` (parameter `ddof`), `robust`, `max-abs`,
`decimal-scaling`, `rank`, `unit-length`, `none`. Their definitions are in the
[algorithm reference](algorithm-reference.md#normalizers-and-metrics) (ADR-051, ADR-052).

```yaml
preprocessing:
  normalizer: z-score
context:
  operators:
    - {label: ρ, features: all}                                  # the Zhuravlyov metric
    - {label: H, metric: heom, features: all}
    - {label: E, metric: euclidean, features: quantitative}
    - label: ρ_w
      metric: {name: weighted-zhuravlyov, params: {nominal_weight: 0.5}}
      features: [x₁, x₂, x₃]
```

A metric defined on quantitative features must be given `features: quantitative` (or a list of
quantitative features), one defined on nominal features `features: nominal`; anything else is
reported as a configuration error that names the features (ADR-050).

## Example: a metric with a parameter

```python
import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from context_synthetic_recognition.core.metrics import METRICS, round_distances, traits
from context_synthetic_recognition.core.registry import PARAMS_CONFIG


@dataclass(frozen=True, config=PARAMS_CONFIG)
class LorentzianParams:
    scale: float = Field(default=1.0, gt=0)


@METRICS.register("lorentzian", params_type=LorentzianParams, summary="Σ_I ln(1 + |x′ − y′| / s)")
@traits(domain="quantitative")
def lorentzian(A, B, quantitative, decimals, params):
    total = np.zeros((len(A), len(B)))
    for j in range(A.shape[1]):
        total += np.log1p(np.abs(A[:, j, None] - B[None, :, j]) / params.scale)
    return round_distances(total, decimals)
```

It can then be used as an operator:

```yaml
context:
  operators:
    - label: L
      metric: {name: lorentzian, params: {scale: 0.5}}
      features: quantitative
```

`csr config check` reports an unknown plug-in name or invalid parameters before a run.

### Metric traits

`traits(domain=…, fit=…, check=…)`, placed below `@METRICS.register`, tells the pipeline what it
must know about a metric besides its distance function (ADR-050). A metric that declares nothing
is heterogeneous and needs no fit.

- `domain` — `"any"` (default), `"quantitative"` or `"nominal"`: the feature types the metric
  is defined on.
- `fit` — a function `(Z, quantitative, params) → state` that estimates training statistics on
  the unified training values of the operator's features. The metric then receives `state` in
  place of `params`. It is called once per training sample — in every fold of a cross-validation —
  and gets no class labels.
- `check` — a function `(quantitative, params)` that raises `ConfigError` when the parameters
  do not fit the operator's features.

```python
from dataclasses import dataclass as state


@state(frozen=True)
class Spread:
    scale: np.ndarray


def fit_spread(Z, quantitative, params):
    deviation = Z.std(axis=0)
    return Spread(np.where(deviation > 0, deviation, np.inf))  # no spread: the feature adds 0


@METRICS.register("standardized-manhattan", summary="Σ_I |x′ − y′| / stdⱼ")
@traits(domain="quantitative", fit=fit_spread)
def standardized_manhattan(A, B, quantitative, decimals, params):
    total = np.zeros((len(A), len(B)))
    for j in range(A.shape[1]):
        total += np.abs(A[:, j, None] - B[None, :, j]) / params.scale[j]
    return round_distances(total, decimals)
```

## Rules a plug-in must keep

- **Metrics** return distances rounded to `decimals` (use `round_distances`), so that equal
  distances tie exactly and the tie rule (smaller original index) decides (ADR-008). A metric
  never reads a class label: its fit step is not given any (ADR-050, ADR-051).
- **Normalizers** estimate their constants on the training objects only, never change nominal
  codes and map a quantitative feature without spread to 0 (ADR-052).
- **k strategies** return odd k only; the pipeline also checks k ≤ m − 1.
- **Encoders** use only χ₁, the classes of the neighbours — never the class of the object itself,
  which a new object does not have (Theorem, ADR-009, ADR-021).
- **Majorizers** should map ℝ onto (0, 1) and increase, like the built-in ones, so that every step
  α·ϕ(−b) stays in (0, α) (ADR-026).
- **Decision rules** receive the final sizes |B1(a_p)|, |B2(a_p)| and the class sizes and return
  1 (K1), 2 (K2) or 0 (refusal) for every object (ADR-027).
- **Protocols** return the folds; the runner re-fits the whole pipeline on every training part
  and never shows it the held-out objects' classes (ADR-032).
- **Baselines** get the fold's training part only and return decisions 1, 2, 0 with a score that
  is larger for K1.
- Everything is deterministic; the configuration hash covers the plug-in names and parameters.

## Plug-ins from other packages

A package can register plug-ins through entry points; `Registry.load_entry_points()` adds them:

```toml
[project.entry-points."context_synthetic_recognition.metrics"]
lorentzian = "my_package.metrics:lorentzian"
```
