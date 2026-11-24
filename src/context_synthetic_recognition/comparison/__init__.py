"""Standard classifiers against the CS-model on the article's training and test samples.

Kept apart from the model's pipeline: it reads the split files through the data layer, classifies
with the project's baselines and metrics, and builds Tables 7 and 8 with the table model; it
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
    HEART_SPLIT,
    KNN_KS,
    KNN_METRICS,
    SKLEARN_OPTIONS,
    SPAMBASE_SPLITS,
    SPLITS,
    SplitSpec,
    percent,
)
from context_synthetic_recognition.comparison.stages import compare_split

__all__ = [
    "ALPHAS",
    "HEART_SPLIT",
    "KNN_KS",
    "KNN_METRICS",
    "SKLEARN_OPTIONS",
    "SPAMBASE_SPLITS",
    "SPLITS",
    "CSModelResult",
    "MethodResult",
    "SplitResult",
    "SplitSpec",
    "compare_split",
    "comparison_config",
    "cs_model_config",
    "cs_model_metrics",
    "knn_key",
    "percent",
    "run_cs_model",
]
