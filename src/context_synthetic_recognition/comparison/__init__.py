"""Standard classifiers against the CS-model on the article's training and test samples.

Kept apart from the model's pipeline: it reads the split files through the data layer, classifies
with the project's baselines and metrics, and builds the comparison tables with the table model; it
changes none of them. Pure computation — the service (:mod:`~context_synthetic_recognition.
services.comparison`) reads the files and writes the results.
"""

from context_synthetic_recognition.comparison.cs_model import (
    ALPHAS,
    CSModelResult,
    cs_model_config,
    cs_model_metrics,
    run_cs_model,
)
from context_synthetic_recognition.comparison.methods import (
    MethodResult,
    SplitResult,
    comparison_config,
    knn_key,
)
from context_synthetic_recognition.comparison.settings import (
    COMPARISON_SPLITS,
    KNN_KS,
    KNN_METRICS,
    SKLEARN_OPTIONS,
    SplitSpec,
    metric_title,
    own_metrics,
    percent,
)
from context_synthetic_recognition.comparison.stages import compare_split

__all__ = [
    "ALPHAS",
    "COMPARISON_SPLITS",
    "KNN_KS",
    "KNN_METRICS",
    "SKLEARN_OPTIONS",
    "CSModelResult",
    "MethodResult",
    "SplitResult",
    "SplitSpec",
    "compare_split",
    "comparison_config",
    "cs_model_config",
    "cs_model_metrics",
    "knn_key",
    "metric_title",
    "own_metrics",
    "percent",
    "run_cs_model",
]
