"""Named presets and the two template calculations that differ from the article.

The deviations are described once here; the CLI, the GUI badges, the exporters and the docs read
them from :data:`DEVIATIONS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from context_synthetic_recognition.config.models import CentreMode, ExperimentConfig


@dataclass(frozen=True)
class Deviation:
    """A template calculation that differs from the article, exposed as a switch."""

    key: str
    """Short identifier."""
    field: str
    """Dotted path of the switch in :class:`ExperimentConfig`."""
    statement: str
    """The deviation in one sentence (shown verbatim in CLI, GUI and docs)."""
    template_value: Any
    """Value that reproduces the template workbooks (default)."""
    article_value: Any
    """Value that follows the article."""
    adr: str
    """Decision record in docs/DECISIONS.md."""


DEVIATIONS: tuple[Deviation, ...] = (
    Deviation(
        key="centres",
        field="hag.centres",
        statement=(
            "θ and γ are measured from running partial class means "
            "instead of the final class means M₁ and M₂"
        ),
        template_value=CentreMode.RUNNING,
        article_value=CentreMode.FINAL,
        adr="ADR-002",
    ),
    Deviation(
        key="step4_passes",
        field="hag.step4_passes",
        statement="in STEP 4 the majorizer is applied twice instead of once",
        template_value=2,
        article_value=1,
        adr="ADR-003",
    ),
)


class PresetName(StrEnum):
    """Names of the built-in presets."""

    TEMPLATE = "template"
    """Template calculations (default): reproduces the template workbooks exactly."""
    ARTICLE = "article"
    """Article calculations: final class centres and one majorizer pass in STEP 4."""


def _field_value(config: ExperimentConfig, dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        value = getattr(value, part)
    return value


def preset(name: PresetName | str) -> ExperimentConfig:
    """Return the preset configuration ``template`` or ``article``."""
    chosen = PresetName(name)
    template = ExperimentConfig()
    if chosen is PresetName.TEMPLATE:
        return template
    hag = template.hag.model_copy(update={"centres": CentreMode.FINAL, "step4_passes": 1})
    return template.model_copy(update={"name": "cs-model-article", "hag": hag})


def active_deviations(config: ExperimentConfig) -> list[Deviation]:
    """Deviations whose switch is at the template value in ``config``."""
    return [d for d in DEVIATIONS if _field_value(config, d.field) == d.template_value]


# Fields that do not define the method: a configuration with only these changed still matches.
_NON_METHOD_FIELDS = {"name", "description", "dataset", "seed", "output"}


def matching_preset(config: ExperimentConfig) -> PresetName | None:
    """The preset whose method settings equal those of ``config``, if any."""
    method = config.model_dump(mode="json", exclude=_NON_METHOD_FIELDS)
    for name in PresetName:
        if preset(name).model_dump(mode="json", exclude=_NON_METHOD_FIELDS) == method:
            return name
    return None
