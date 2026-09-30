"""Check the plug-ins a configuration names against the registries.

The core's kinds are checked here (normalizer, metrics, k strategy, encoder, weights, majorizer,
decision rule); ``services.configs.config_problems`` adds the evaluation protocols and baselines.
"""

from __future__ import annotations

from typing import Any

from context_synthetic_recognition.config.models import ExperimentConfig, PluginSpec
from context_synthetic_recognition.core.contributions import WEIGHTS
from context_synthetic_recognition.core.encoders import ENCODERS
from context_synthetic_recognition.core.k_strategies import K_STRATEGIES
from context_synthetic_recognition.core.majorizers import MAJORIZERS
from context_synthetic_recognition.core.meta import DECISION_RULES
from context_synthetic_recognition.core.metrics import METRICS
from context_synthetic_recognition.core.normalizers import NORMALIZERS
from context_synthetic_recognition.core.registry import Registry
from context_synthetic_recognition.errors import ConfigError, RegistryError


def plugin_problems(config: ExperimentConfig) -> list[str]:
    """Unknown plug-in names and invalid plug-in parameters in ``config`` (empty if none)."""
    problems: list[str] = []

    def check(registry: Registry[Any], spec: PluginSpec, where: str) -> None:
        try:
            registry.make_params(spec.name, spec.params)
        except (RegistryError, ConfigError) as error:
            problems.append(f"{where}: {error}")

    check(NORMALIZERS, config.preprocessing.normalizer, "preprocessing.normalizer")
    for i, operator in enumerate(config.context.operators):
        check(METRICS, operator.metric, f"context.operators[{i}] ({operator.label}).metric")
    check(K_STRATEGIES, config.k, "k")
    check(ENCODERS, config.synthetic.encoder, "synthetic.encoder")
    check(WEIGHTS, config.synthetic.weights, "synthetic.weights")
    check(MAJORIZERS, config.hag.majorizer, "hag.majorizer")
    check(DECISION_RULES, config.meta.decision_rule, "meta.decision_rule")
    return problems
