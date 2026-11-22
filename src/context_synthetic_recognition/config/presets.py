"""Named presets and the settings that deviate from the article.

Every switch whose non-default value departs from the article is described once in
:data:`DEVIATIONS`; the CLI, the GUI badges, the exporters and the docs read it from there.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from context_synthetic_recognition.config.models import (
    ClassMeans,
    ExperimentConfig,
    PluginSpec,
)


@dataclass(frozen=True)
class Deviation:
    """A setting that deviates from the article, exposed as a switch."""

    key: str
    """Short identifier."""
    field: str
    """Dotted path of the switch in :class:`ExperimentConfig`."""
    statement: str
    """The deviation in one sentence (shown verbatim in CLI, GUI and docs)."""
    alternative_value: Any
    """The value that deviates from the article."""
    article_value: Any
    """The value that follows the article."""
    adr: str
    """Decision record in docs/DECISIONS.md."""


DEVIATIONS: tuple[Deviation, ...] = (
    Deviation(
        key="keep_blocked_cover",
        field="rips.keep_blocked_cover",
        statement=(
            "in R-IPS a blocked object keeps covering other objects with |r(S)| although "
            "blocking makes r(S) negative"
        ),
        alternative_value=True,
        article_value=False,
        adr="ADR-061",
    ),
    Deviation(
        key="training",
        field="rips.training",
        statement=(
            "the objects R-IPS moves to D form the training sample instead of the objects that "
            "remain in E₁"
        ),
        alternative_value="moved",
        article_value="remaining",
        adr="ADR-061",
    ),
    Deviation(
        key="class_means",
        field="additional_features.class_means",
        statement=(
            "θ and γ are measured from running partial class means "
            "instead of the final class means M₁ and M₂"
        ),
        alternative_value=ClassMeans.RUNNING,
        article_value=ClassMeans.FINAL,
        adr="ADR-002",
    ),
    Deviation(
        key="majorizing_passes",
        field="additional_features.majorizing_passes",
        statement="in Step 4 the majorizing function is applied twice instead of once",
        alternative_value=2,
        article_value=1,
        adr="ADR-003",
    ),
)


def deviation(key: str) -> Deviation:
    """The deviation with this key.

    Raises:
        KeyError: No deviation has the key.
    """
    for found in DEVIATIONS:
        if found.key == key:
            return found
    raise KeyError(key)


class PresetName(StrEnum):
    """Names of the built-in presets."""

    DEFAULT = "default"
    """The default configuration."""
    ALPHA_01 = "alpha-0.1"
    """The default with α = 0.1, as in the article's experiments."""
    ALPHA_02 = "alpha-0.2"
    """The default with α = 0.2, as in the article's experiments."""
    EXCEL_EXPERIMENT = "excel-experiment"
    """The settings of the Excel experiment workbook: running class means, two majorizing
    passes, α = 0.3 and odd k = 3 … 2·min|Kᵢ| − 3; it reproduces the workbook exactly."""


def _field_value(config: ExperimentConfig, dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        value = getattr(value, part)
    return value


def preset(name: PresetName | str) -> ExperimentConfig:
    """Return a preset configuration (see :class:`PresetName`)."""
    chosen = PresetName(name)
    default = ExperimentConfig()
    additional = default.additional_features
    if chosen is PresetName.DEFAULT:
        return default
    if chosen in (PresetName.ALPHA_01, PresetName.ALPHA_02):
        alpha = 0.1 if chosen is PresetName.ALPHA_01 else 0.2
        return default.model_copy(
            update={
                "name": f"cs-model-{chosen.value}",
                "additional_features": additional.model_copy(update={"alpha": alpha}),
            }
        )
    workbook = additional.model_copy(
        update={"alpha": 0.3, "class_means": ClassMeans.RUNNING, "majorizing_passes": 2}
    )
    return default.model_copy(
        update={
            "name": "cs-model-excel-experiment",
            "admissible_k": PluginSpec(name="twice-min-class"),
            "additional_features": workbook,
        }
    )


def active_deviations(config: ExperimentConfig) -> list[Deviation]:
    """Deviations whose switch is at its deviating value in ``config``."""
    return [d for d in DEVIATIONS if _field_value(config, d.field) == d.alternative_value]


# Fields that do not define the method: a configuration with only these changed still matches.
_NON_METHOD_FIELDS = {"name", "description", "dataset", "seed", "output"}


def matching_preset(config: ExperimentConfig) -> PresetName | None:
    """The preset whose method settings equal those of ``config``, if any (the first one)."""
    method = config.model_dump(mode="json", exclude=_NON_METHOD_FIELDS)
    for name in PresetName:
        if preset(name).model_dump(mode="json", exclude=_NON_METHOD_FIELDS) == method:
            return name
    return None
