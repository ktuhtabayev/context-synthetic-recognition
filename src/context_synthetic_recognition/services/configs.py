"""Checks of a whole configuration before a run: every plug-in of every layer.

:func:`~context_synthetic_recognition.core.plugins.plugin_problems` covers the core's plug-ins
(normalizer, metrics, k strategy, encoder, weights, majorizer, decision rule);
:func:`config_problems` adds the evaluation's protocols and baselines, which live above the core.
"""

from __future__ import annotations

from typing import Any

from context_synthetic_recognition.config.models import ExperimentConfig, PluginSpec
from context_synthetic_recognition.core.plugins import plugin_problems
from context_synthetic_recognition.core.registry import Registry
from context_synthetic_recognition.errors import ConfigError, RegistryError
from context_synthetic_recognition.evaluation.baselines import BASELINES
from context_synthetic_recognition.evaluation.protocols import PROTOCOLS


def config_problems(config: ExperimentConfig) -> list[str]:
    """Unknown plug-ins and invalid parameters anywhere in ``config`` (empty if none)."""
    problems = plugin_problems(config)

    def check(registry: Registry[Any], spec: PluginSpec, where: str) -> None:
        try:
            registry.make_params(spec.name, spec.params)
        except (RegistryError, ConfigError) as error:
            problems.append(f"{where}: {error}")

    for i, spec in enumerate(config.evaluation.protocols):
        check(PROTOCOLS, spec, f"evaluation.protocols[{i}]")
    for i, spec in enumerate(config.evaluation.baselines):
        check(BASELINES, spec, f"evaluation.baselines[{i}]")
    return problems
