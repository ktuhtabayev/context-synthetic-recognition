# Extending the pipeline

Every exchangeable part of the method is a plug-in chosen by name from a registry. A plug-in is
a function with the signature of its kind; it may declare a parameter type, which validates the
`params` of the configuration and later generates the GUI form (ADR-019).

| Kind | Registry | Signature | Default |
|---|---|---|---|
| normalizer | `core.local_context.normalizers.NORMALIZERS` | `(X, quantitative, params) → Scaling` | `minmax` |
| metric | `core.local_context.metrics.METRICS` | `(A, B, quantitative, decimals, params) → distances`, with optional [traits](#metric-traits) | `zhuravlyov` |
| k strategy | `core.local_context.admissible_k.K_STRATEGIES` | `(class_sizes, params) → AdmissibleK` | `min-class` |
| encoder | `core.synthetic_features.encoders.ENCODERS` | `(chi1, k, params) → values in {1, 2}` | `formula-5` |
| informativeness | `core.synthetic_features.contributions.INFORMATIVENESS` | `(InformativenessInputs, params) → float` | `formula-4` |
| majorizing function | `core.additional_features.majorizing_functions.MAJORIZING_FUNCTIONS` | `(x, params) → φ(x)` element-wise | `sigmoid` |
| decision rule | `core.meta_algorithm.algorithm.DECISION_RULES` | `(b1, b2, class_sizes, params) → 1, 2 or 0` | `step-4` |
| protocol | `evaluation.protocols.PROTOCOLS` | `(class_index, seed, params) → [Split]`; marked with `uses_dataset`: `(dataset, config, params) → [Split]` | `resubstitution`, `leave-one-out` |
| baseline | `evaluation.baselines.BASELINES` | `(BaselineInput, params) → [BaselineOutput]` | `knn-vote` |
| dataset format | `data.loaders.LOADERS` | `(path, LoadOptions) → Dataset` | detected |
| export format | `export.run.EXPORTERS` | `(ExportContext) → [Path]` | `all` ([Exporting](exporting.md)) |
| classifier part | `services.classifier_parts.CLASSIFIER_PARTS` | the protocol `ClassifierPart`: train, classify without the class, evaluate ([below](#adding-a-part-to-the-classifiers-page)) | `classifier-comparison` |

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

The local contexts have one normalizer; an operator can override it with its own (ADR-066). R-IPS
has its own metric, normalizer and features in the section `rips`:

```yaml
local_context:
  normalizer: z-score
  operators:
    - {label: ρ, features: all}                                  # the Zhuravlyov metric
    - {label: H, metric: heom, features: all}
    - {label: E, metric: euclidean, features: quantitative, normalizer: none}
    - label: ρ_w
      metric: {name: weighted-zhuravlyov, params: {nominal_weight: 0.5}}
      features: [x₁, x₂, x₃]
rips:
  metric: chebyshev
  normalizer: none
  features: quantitative
```

A metric defined on quantitative features must be given `features: quantitative` (or a list of
quantitative features), one defined on nominal features `features: nominal`; anything else is
reported as a configuration error that names the features (ADR-050).

## Example: a metric with a parameter

```python
import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from context_synthetic_recognition.core.local_context.metrics import (
    METRICS,
    round_distances,
    traits,
)
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
local_context:
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
- **k strategies** take the class sizes of the training sample and return odd k only; the
  pipeline also checks k ≤ m − 1.
- **Encoders** use only χ₁, the classes of the neighbours — never the class of the object itself,
  which a new object does not have (Theorem, ADR-009, ADR-021).
- **Informativeness** functions weigh a synthetic feature from its ω, stability and boundary; the
  weight enters the contributions (6) and Step 2 of the agglomerative algorithm.
- **Majorizing functions** should map ℝ onto (0, 1) and increase, like the built-in ones, so that
  every step α·φ(−b) stays in (0, α) (ADR-026).
- **Decision rules** receive the final sizes |B1(a_p)|, |B2(a_p)| and the class sizes and return
  1 (K1), 2 (K2) or 0 (refusal) for every object (ADR-027).
- **Protocols** return the folds; the runner re-fits the whole pipeline on every training part
  and never shows it the held-out objects' classes (ADR-032). A protocol that needs the features —
  R-IPS does — is marked with `uses_dataset` and receives the dataset and the configuration; a
  fold may carry its own configuration, as nested cross-validation does (ADR-069).
- **Baselines** get the fold's training part only and return decisions 1, 2, 0 with a score that
  is larger for K1.
- Everything is deterministic; the configuration hash covers the plug-in names and parameters.

## Plug-ins from other packages

A package can register plug-ins through entry points; `Registry.load_entry_points()` adds them:

```toml
[project.entry-points."context_synthetic_recognition.metrics"]
lorentzian = "my_package.metrics:lorentzian"
```

## Adding a part to the Classifiers page

The *Classifiers* page of the desktop application shows **classifier parts**: units of work in
three stages — ① *Training* on the training sample, ② *Test*, where the test objects are
classified without their class, and ③ *Evaluation* against the true classes (ADR-084). The page
renders any part of the registry `services.classifier_parts.CLASSIFIER_PARTS` from its protocol
alone, so a new part is registered in the services and the page does not change.

A part is an object with `key`, `title`, `description` and these methods (types of
`services.classifier_parts` and `data.samples`):

| Method | Returns | What it gets |
|---|---|---|
| `default_folder()` | the folder its samples are read from by default, or `None` | — |
| `sample_options(folder)` | `SampleOption`s: what it runs on, with sizes, or why one is unavailable | the folder |
| `load(option, folder)` | a `SampleSet`: training samples with their classes, the test objects as `UnlabelledSample`s and, apart, their `AnswerKey`s | the option |
| `method_groups(option)` | `MethodGroup`s: the methods with their fixed settings (the *Methods* tab) | the option |
| `train(option, training)` | a `TrainedPart`: per split the methods' training decisions, and the fitted state in `fitted` | the training samples only |
| `classify(trained, unlabelled)` | `Predictions`: per split and method a `Decisions` (1, 2 or 0 per object, the scores if any) | the trained part and objects **without a class** |
| `evaluate(predictions, answers)` | an `Evaluation`: per split and method the `ClassificationMetrics` | the decisions and the answer keys |

`train`, `classify` and `evaluate` also take an optional `progress(what, done, total)`. A stage
result may carry the part's own tabs in `tables` (`StageTable`s); without them the part still gets
*Training sample*, *Training results*, *Test objects*, *Predictions*, *Predicted classes*, *Per
object* and *Confusion*. A part that also has `write_problem(evaluation)`, `write(evaluation,
folder)` and `default_output()` gets *Write the files*; one with `check(folder, splits)` gets
*Check the written files* and the *Check* tab.

The smallest part — one sample of seven objects, one method that takes the class of the nearest
training object (`tests/dummy_part.py`, which the tests register and run through the page):

```python
from pathlib import Path

import numpy as np

from context_synthetic_recognition.data import Dataset, FeatureType
from context_synthetic_recognition.data.samples import AnswerKey, Decisions, UnlabelledSample
from context_synthetic_recognition.services.classifier_parts import (
    CLASSIFIER_PARTS,
    Evaluation,
    MethodGroup,
    MethodInfo,
    MethodTraining,
    Predictions,
    SampleOption,
    SampleSet,
    TrainedPart,
    TrainingSample,
    metrics_against,
)

TYPES = (FeatureType.QUANTITATIVE,)


def nearest(training: Dataset, X: np.ndarray, *, leave_out: bool) -> np.ndarray:
    """The class code (1 or 2) of the nearest training object on x₁."""
    distances = np.abs(X[:, :1] - training.X[:, 0][None, :])
    if leave_out:  # Definition 2: a training object is not its own neighbour
        np.fill_diagonal(distances, np.inf)
    return training.class_index[np.argmin(distances, axis=1)] + 1


class NearestPart:
    key = "nearest"
    title = "Nearest object"
    description = "Four training objects, three test objects, one trivial method."

    def default_folder(self) -> Path | None:
        return None

    def sample_options(self, folder):
        return (SampleOption("toy", "Toy", ("toy",), "training 4 (2, 2) · test 3"),)

    def load(self, option, folder):
        training = Dataset(
            X=np.array([[0.0], [1.0], [10.0], [11.0]]), y=[1, 1, 2, 2], feature_types=TYPES
        )
        objects = np.array([4, 5, 6])
        test = UnlabelledSample(objects, np.array([[0.5], [10.5], [2.0]]), TYPES, ("x₁",), "toy")
        answers = AnswerKey(objects, np.array([1, 2, 2]), (1, 2), "toy")
        return SampleSet(option, (TrainingSample("toy", "Toy", training),), (test,), (answers,))

    def method_groups(self, option):
        method = MethodInfo("nearest", "Nearest object", "all (1)", ("toy",))
        return (MethodGroup("Trivial", "The class of the nearest training object.", (method,)),)

    def train(self, option, training, progress=None):
        (sample,) = training
        decided = nearest(sample.data, sample.data.X, leave_out=True)
        methods = {"toy": (MethodTraining("nearest", "Nearest object", decided),)}
        return TrainedPart(self.key, option, training, methods, fitted=sample.data)

    def classify(self, trained, unlabelled, progress=None):
        (sample,) = unlabelled  # feature values only: there is no class to read
        decisions = {
            "toy": {"nearest": Decisions(nearest(trained.fitted, sample.X, leave_out=False))}
        }
        return Predictions(self.key, trained, unlabelled, decisions)

    def evaluate(self, predictions, answers, progress=None):
        (key,) = answers
        metrics = {"toy": metrics_against(predictions.decisions["toy"], key)}
        return Evaluation(self.key, predictions, answers, metrics)


CLASSIFIER_PARTS.add(NearestPart.key, NearestPart())
```

Rules a part keeps:

- **Blind test.** `classify` reads feature values only. `UnlabelledSample` has no class field,
  `data.samples.require_unlabelled` rejects anything else at run time, and mypy rejects a
  `Dataset` (ADR-085). The classes of the test objects reach `evaluate` only, as `AnswerKey`s.
- **Training on the training sample only**: scaling, encodings and neighbours are fitted there.
- **No computation in the interface**: the page calls the part in a worker thread; whatever a tab
  shows is a `StageTable` the part or the services build.
- **Registration**: a module of the package adds itself to `BUILTIN_PARTS`; a part of another
  package registers through the entry-point group `context_synthetic_recognition.classifier-parts`
  (see [Plug-ins from other packages](#plug-ins-from-other-packages)).
