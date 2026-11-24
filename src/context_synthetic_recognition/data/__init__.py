"""Data layer: datasets, their loaders, built-in datasets, split files and blind tests.

A blind test is an unlabelled sample, its answer key and the decisions (ADR-085).
"""

from context_synthetic_recognition.data.builtin import DATASETS, BuiltinDataset, load_builtin
from context_synthetic_recognition.data.loaders import (
    LOADERS,
    LoadOptions,
    detect_format,
    load_dataset,
    load_from_config,
    table_to_dataset,
)
from context_synthetic_recognition.data.samples import AnswerKey, Decisions, UnlabelledSample
from context_synthetic_recognition.data.schema import Dataset, FeatureType, Label
from context_synthetic_recognition.data.splits import FixedSplit, load_split_file

__all__ = [
    "DATASETS",
    "LOADERS",
    "AnswerKey",
    "BuiltinDataset",
    "Dataset",
    "Decisions",
    "FeatureType",
    "FixedSplit",
    "Label",
    "LoadOptions",
    "UnlabelledSample",
    "detect_format",
    "load_builtin",
    "load_dataset",
    "load_from_config",
    "load_split_file",
    "table_to_dataset",
]
