"""Typed experiment configuration (pydantic v2).

One :class:`ExperimentConfig` describes a run completely. Unknown keys are errors and instances
are immutable. The defaults are the **template** setting, which reproduces the author's template
workbooks and the full Excel experiment exactly (ADR-004); :mod:`.presets` holds the named presets
and the two template-vs-article switches.

Pluggable components are :class:`PluginSpec` values (registry name + parameters). Their
parameters are validated by the component's own parameter type when the pipeline resolves them.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class ConfigModel(BaseModel):
    """Base of all configuration models: strict keys, immutable, field docs as descriptions."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        use_attribute_docstrings=True,
    )


class PluginSpec(ConfigModel):
    """A pluggable component chosen by registry name, with its parameters.

    In YAML a bare name is accepted as shorthand: ``normalizer: minmax``.
    """

    name: str = Field(min_length=1)
    """Registry name (case-insensitive; ``_`` and spaces count as ``-``)."""
    params: dict[str, JsonValue] = Field(default_factory=dict)
    """Keyword parameters, validated by the component's parameter type when it is resolved."""

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_name(cls, value: Any) -> Any:
        return {"name": value} if isinstance(value, str) else value


def plugin(name: str, **params: JsonValue) -> PluginSpec:
    """Shorthand constructor: ``plugin("sigmoid")``."""
    return PluginSpec(name=name, params=params)


class CentreMode(StrEnum):
    """Class centres used by θ and γ in HAG STEP 3 (template deviation 1, ADR-002)."""

    RUNNING = "running"
    """Template cells: |bₜ − M| uses the running partial class sum up to row t divided by |Kᵢ|."""
    FINAL = "final"
    """Article (and the template's formula box): final class means M₁ = Σbₜ/|K1|, M₂ = Σbₜ/|K2|."""


class DatasetConfig(ConfigModel):
    """Where the data comes from and how to read it (loaders arrive in M2)."""

    path: str | None = None
    """Dataset file; relative paths are resolved against the configuration file's folder."""
    format: Literal["auto", "cs-workbook", "template-extended", "csv", "xlsx", "parquet"] = "auto"
    """``auto`` picks the loader from the file extension and content."""
    sheet: str | None = None
    """Sheet name for Excel files (the experiment workbook uses ``Dataset``)."""
    class_column: str | None = None
    """Column holding the class label (tabular formats)."""
    id_column: str | None = None
    """Optional column of object ids."""
    feature_types: dict[str, Literal["quantitative", "nominal"]] | None = None
    """Feature name → type (set I or J); overrides types stored in the file."""
    missing_values: Literal["error"] = "error"
    """Policy for missing values; only ``error`` for now (ADR-015)."""


class PreprocessingConfig(ConfigModel):
    """Scale unification (Step 1), fitted on the training objects only."""

    normalizer: PluginSpec = Field(default_factory=lambda: plugin("minmax"))
    """Fractional-linear (min–max) transform of the quantitative features by default."""


class OperatorConfig(ConfigModel):
    """A base operator of Ψ: a metric applied to a feature subset (article: variants of Ψ)."""

    label: str = Field(min_length=1)
    """Display name, e.g. ``ρ``, ``ρ_I``, ``ρ_J``; used in synthetic-feature names."""
    metric: PluginSpec = Field(default_factory=lambda: plugin("zhuravlyov"))
    """Distance on the selected features (Zhuravlyov metric by default)."""
    features: Literal["all", "quantitative", "nominal"] | tuple[str, ...] = "all"
    """``all`` (I ∪ J), ``quantitative`` (I), ``nominal`` (J) or explicit feature names."""

    @field_validator("features")
    @classmethod
    def _explicit_subset_is_a_set(cls, value: str | tuple[str, ...]) -> str | tuple[str, ...]:
        if isinstance(value, tuple):
            if not value:
                raise ValueError("an explicit feature subset must not be empty")
            if len(set(value)) != len(value):
                raise ValueError("an explicit feature subset must not repeat a feature")
        return value


def _default_operators() -> tuple[OperatorConfig, ...]:
    return (
        OperatorConfig(label="ρ", features="all"),
        OperatorConfig(label="ρ_I", features="quantitative"),
        OperatorConfig(label="ρ_J", features="nominal"),
    )


class ContextConfig(ConfigModel):
    """Local contexts Ψ_ρ,k: base operators, distances and neighbour ordering (Steps 2–3)."""

    operators: tuple[OperatorConfig, ...] = Field(default_factory=_default_operators, min_length=1)
    """Base operators; one synthetic feature per (operator, permitted k)."""
    distance_decimals: int = Field(10, ge=0, le=15)
    """Distances are rounded to this many decimals so that equal distances tie exactly."""
    tie_break: Literal["smaller-index"] = "smaller-index"
    """Equal distances: the neighbour with the smaller original index comes first."""
    skip_empty_operators: bool = True
    """Drop (with a warning) operators whose feature subset is empty for the dataset (ADR-007)."""

    @field_validator("operators")
    @classmethod
    def _labels_are_unique(cls, value: tuple[OperatorConfig, ...]) -> tuple[OperatorConfig, ...]:
        labels = [operator.label for operator in value]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f"operator labels must be unique (repeated: {', '.join(duplicates)})")
        return value


class SyntheticConfig(ConfigModel):
    """Synthetic features Ψ(r) and their weights (formulas (1)–(6))."""

    encoder: PluginSpec = Field(default_factory=lambda: plugin("formula-5"))
    """Class-free majority feature aᵤ ∈ {1, 2} by formula (5)."""
    weights: PluginSpec = Field(default_factory=lambda: plugin("omega"))
    """Informativeness ω by formula (4), used by the contributions (6) and by HAG STEP 2."""


class HAGConfig(ConfigModel):
    """Hierarchical agglomerative grouping, Steps 1–5 (template parameters by default)."""

    alpha: float = Field(0.3, gt=0.0, lt=1.0)
    """α — regularisation of the margin, 0 < α < 1."""
    delta: float = Field(0.1, gt=0.0, lt=0.5)
    """δ — threshold for θ/γ, 0 < δ < 0.5."""
    kappa: int = Field(5, ge=2)
    """ϰ — maximum |TUPLAM| (the article asks ϰ ≤ r − 1; not clamped, see ADR-008)."""
    cr1: float = Field(10.0, gt=0.0)
    """Initial value of the criterion in STEP 3; a candidate is chosen only if θ/γ < cr1."""
    majorizer: PluginSpec = Field(default_factory=lambda: plugin("sigmoid"))
    """Majorizing function ϕ; b ← b ± α·ϕ(−b) (+ for K1, − for K2)."""
    centres: CentreMode = CentreMode.RUNNING
    """⚠ Template deviation 1 (ADR-002): ``running`` = template cells, ``final`` = article."""
    step4_passes: Literal[1, 2] = 2
    """⚠ Template deviation 2 (ADR-003): majorizer passes in STEP 4; 2 = template, 1 = article."""


class MetaConfig(ConfigModel):
    """Meta-algorithm, Steps 1–5."""

    decision_rule: PluginSpec = Field(default_factory=lambda: plugin("article-step-4"))
    """K1 if |B1|/|K1| > |B2|/|K2|, K2 if <, 0 (refusal) if equal."""


class EvaluationConfig(ConfigModel):
    """Evaluation protocols, baselines and metric conventions."""

    protocols: tuple[PluginSpec, ...] = Field(
        default_factory=lambda: (plugin("resubstitution"), plugin("leave-one-out"))
    )
    """Resubstitution (Definition 2) and leave-one-out with the whole pipeline re-fitted."""
    baselines: tuple[PluginSpec, ...] = Field(default_factory=lambda: (plugin("knn-vote"),))
    """Plain k-NN majority vote per operator and k."""
    positive_class: int | str = 1
    """Label of the positive class for the confusion matrix, precision/recall and ROC."""
    score_decimals: int = Field(10, ge=0, le=15)
    """Scores score₁ − score₂ are rounded before AUC/ROC so that equal scores tie exactly."""


class OutputConfig(ConfigModel):
    """Where runs are written."""

    runs_dir: str = "runs"
    """Each run gets ``<runs_dir>/<YYYYMMDD_HHMMSS>_<config hash>/`` with its manifest."""


class ExperimentConfig(ConfigModel):
    """Complete description of one experiment."""

    schema_version: Literal[1] = 1
    """Version of this configuration format."""
    name: str = "cs-model"
    """Short name of the experiment."""
    description: str = ""
    """Free text."""
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    k: PluginSpec = Field(default_factory=lambda: plugin("formula"))
    """Permitted k: odd k = 3 … 2·min|Kᵢ| − 3 from the training classes (ADR-005, ADR-006)."""
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)
    hag: HAGConfig = Field(default_factory=HAGConfig)
    meta: MetaConfig = Field(default_factory=MetaConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    seed: int = Field(42, ge=0)
    """Seed of every random choice (splits of randomised protocols)."""
    output: OutputConfig = Field(default_factory=OutputConfig)
