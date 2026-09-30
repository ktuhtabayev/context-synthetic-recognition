# Extending the pipeline

Every exchangeable part of the method is a plug-in chosen by name from a registry. A plug-in is
a function with the signature of its kind; it may declare a parameter type, which validates the
`params` of the configuration and later generates the GUI form (ADR-019).

| Kind | Registry | Signature | Default |
|---|---|---|---|
| normalizer | `core.normalizers.NORMALIZERS` | `(X, quantitative, params) → Scaling` | `minmax` |
| metric | `core.metrics.METRICS` | `(A, B, quantitative, decimals, params) → distances` | `zhuravlyov` |
| k strategy | `core.k_strategies.K_STRATEGIES` | `(class_sizes, params) → PermittedK` | `formula` |
| encoder | `core.encoders.ENCODERS` | `(chi1, k, params) → values in {1, 2}` | `formula-5` |
| weights | `core.contributions.WEIGHTS` | `(WeightInputs, params) → float` | `omega` |
| dataset format | `data.loaders.LOADERS` | `(path, LoadOptions) → Dataset` | detected |

## Example: a metric with a parameter

```python
import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from context_synthetic_recognition.core.metrics import METRICS, round_distances
from context_synthetic_recognition.core.registry import PARAMS_CONFIG


@dataclass(frozen=True, config=PARAMS_CONFIG)
class WeightedParams:
    nominal_weight: float = Field(default=1.0, ge=0)


@METRICS.register(
    "weighted-zhuravlyov", params_type=WeightedParams, summary="Σ_I |x′ − y′| + w·Σ_J [x ≠ y]"
)
def weighted_zhuravlyov(A, B, quantitative, decimals, params):
    quantitative_part = np.zeros((len(A), len(B)))
    nominal_part = np.zeros((len(A), len(B)))
    for j in range(A.shape[1]):
        if quantitative[j]:
            quantitative_part += np.abs(A[:, j, None] - B[None, :, j])
        else:
            nominal_part += A[:, j, None] != B[None, :, j]
    return round_distances(quantitative_part + params.nominal_weight * nominal_part, decimals)
```

It can then be used as an operator:

```yaml
context:
  operators:
    - label: ρ_w
      metric: {name: weighted-zhuravlyov, params: {nominal_weight: 0.5}}
      features: all
```

`csr config check` reports an unknown plug-in name or invalid parameters before a run.

## Rules a plug-in must keep

- **Metrics** return distances rounded to `decimals` (use `round_distances`), so that equal
  distances tie exactly and the tie rule (smaller original index) decides (ADR-008).
- **k strategies** return odd k only; the pipeline also checks k ≤ m − 1.
- **Encoders** use only χ₁, the classes of the neighbours — never the class of the object itself,
  which a new object does not have (Theorem, ADR-009, ADR-021).
- Everything is deterministic; the configuration hash covers the plug-in names and parameters.

## Plug-ins from other packages

A package can register plug-ins through entry points; `Registry.load_entry_points()` adds them:

```toml
[project.entry-points."context_synthetic_recognition.metrics"]
weighted-zhuravlyov = "my_package.metrics:weighted_zhuravlyov"
```
