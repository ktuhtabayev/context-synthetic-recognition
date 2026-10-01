"""The HAG template's own data — the worked examples of the sheet *Template Deviations*.

The two template calculations that differ from the article (ADR-002, ADR-003) are explained on
the template workbook's data: 13 original features as contribution values, their weights and the
classes of 10 objects. That input (and the template's latent features r₁ … r₄, for comparison) is
kept in ``hag_template.json``; everything shown about it — θ, γ and θ/γ with running and with
final class centres, r₁ with one and with two majorizer passes, the SET of every switch setting —
is computed by the package's HAG, so the examples cannot drift from the code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import numpy as np

from context_synthetic_recognition.config.models import CentreMode, HAGConfig
from context_synthetic_recognition.core.arrays import FloatArray, IntArray
from context_synthetic_recognition.core.hag import HAGResult, hag
from context_synthetic_recognition.notation import feature_name

RESOURCE = "hag_template.json"
IDENTICAL_BELOW = 1e-9
"""Latent features closer than this to the template's count as identical."""

SETTINGS: tuple[tuple[CentreMode, int, str], ...] = (
    (CentreMode.RUNNING, 2, "template cells"),
    (CentreMode.RUNNING, 1, "running centres, 1 pass"),
    (CentreMode.FINAL, 2, "final centres, 2 passes"),
    (CentreMode.FINAL, 1, "article"),
)
"""The four switch settings in the workbook's order."""


@dataclass(frozen=True, eq=False)
class TemplateInput:
    """The input of the HAG template workbook."""

    contributions: FloatArray
    """(10 × 13) contribution values of the template's original features."""
    weights: FloatArray
    class_index: IntArray
    labels: tuple[int, ...]
    """The class labels (1 or 2)."""
    latent: FloatArray
    """(10 × 4) the template workbook's own latent features r₁ … r₄."""


@dataclass(frozen=True)
class TemplateVariant:
    """The template data under one setting of the two switches."""

    centres: CentreMode
    step4_passes: int
    label: str
    selected: str
    """SET, e.g. ``{x₃, x₆, x₁₃, x₄, x₉}``."""
    difference: float
    """max |r − template r| over the latent features both have."""

    @property
    def difference_text(self) -> str:
        """The difference as the sheets show it."""
        if self.difference < IDENTICAL_BELOW:
            return "< 1e-9  (identical)"
        return f"{self.difference:.1e}"


@lru_cache(maxsize=1)
def template_input() -> TemplateInput:
    """Read the template's input from the package data."""
    resource = files("context_synthetic_recognition.export.excel") / RESOURCE
    data = json.loads(resource.read_text(encoding="utf-8"))
    labels = tuple(int(v) for v in data["classes"])
    return TemplateInput(
        contributions=np.array(data["contributions"], dtype=np.float64),
        weights=np.array(data["weights"], dtype=np.float64),
        class_index=np.array(labels, dtype=np.int64) - 1,
        labels=labels,
        latent=np.array(data["latent"], dtype=np.float64),
    )


def template_hag(centres: CentreMode = CentreMode.RUNNING, step4_passes: int = 2) -> HAGResult:
    """The HAG on the template data with the template's parameters and the given switches."""
    data = template_input()
    config = HAGConfig(centres=centres, step4_passes=step4_passes)  # type: ignore[arg-type]
    return hag(data.contributions, data.weights, data.class_index, config)


def feature_set(result: HAGResult) -> str:
    """``{x₃, x₆, …}`` — the template's features are original ones."""
    return "{" + ", ".join(feature_name(u) for u in result.tuplam) + "}"


@lru_cache(maxsize=1)
def template_variants() -> tuple[TemplateVariant, ...]:
    """The four switch settings on the template data."""
    reference = template_input().latent
    variants = []
    for centres, passes, label in SETTINGS:
        result = template_hag(centres, passes)
        shared = min(result.latent.shape[1], reference.shape[1])
        difference = float(np.abs(result.latent[:, :shared] - reference[:, :shared]).max())
        variants.append(TemplateVariant(centres, passes, label, feature_set(result), difference))
    return tuple(variants)
