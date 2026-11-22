"""Check the plug-ins a configuration names against the registries.

The core's kinds are checked here (normalizer, metrics, k strategy, encoder, informativeness,
majorizing function, decision rule); ``services.configs.config_problems`` adds the evaluation
protocols and baselines.
"""

from __future__ import annotations

from typing import Any

from context_synthetic_recognition.config.models import ExperimentConfig, PluginSpec
from context_synthetic_recognition.core.additional_features.majorizing_functions import (
    MAJORIZING_FUNCTIONS,
)
from context_synthetic_recognition.core.local_context.admissible_k import K_STRATEGIES
from context_synthetic_recognition.core.local_context.metrics import METRICS
from context_synthetic_recognition.core.local_context.normalizers import NORMALIZERS
from context_synthetic_recognition.core.meta_algorithm.algorithm import DECISION_RULES
from context_synthetic_recognition.core.registry import Registry
from context_synthetic_recognition.core.synthetic_features.contributions import INFORMATIVENESS
from context_synthetic_recognition.core.synthetic_features.encoders import ENCODERS
from context_synthetic_recognition.errors import ConfigError, RegistryError


def plugin_problems(config: ExperimentConfig) -> list[str]:
    """Unknown plug-in names and invalid plug-in parameters in ``config`` (empty if none)."""
    problems: list[str] = []

    def check(registry: Registry[Any], spec: PluginSpec, where: str) -> None:
        try:
            registry.make_params(spec.name, spec.params)
        except (RegistryError, ConfigError) as error:
            problems.append(f"{where}: {error}")

    check(NORMALIZERS, config.local_context.normalizer, "local_context.normalizer")
    for i, operator in enumerate(config.local_context.operators):
        check(METRICS, operator.metric, f"local_context.operators[{i}] ({operator.label}).metric")
    check(K_STRATEGIES, config.admissible_k, "admissible_k")
    check(ENCODERS, config.synthetic_features.encoder, "synthetic_features.encoder")
    check(
        INFORMATIVENESS,
        config.synthetic_features.informativeness,
        "synthetic_features.informativeness",
    )
    check(
        MAJORIZING_FUNCTIONS,
        config.additional_features.majorizing_function,
        "additional_features.majorizing_function",
    )
    check(DECISION_RULES, config.meta_algorithm.decision_rule, "meta_algorithm.decision_rule")
    return problems
