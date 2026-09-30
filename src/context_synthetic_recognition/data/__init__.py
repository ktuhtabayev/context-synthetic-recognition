"""Data layer: dataset schema, loaders for every supported format, built-in datasets."""

from context_synthetic_recognition.data.builtin import DATASETS, BuiltinDataset, load_builtin
from context_synthetic_recognition.data.loaders import (
    LOADERS,
    LoadOptions,
    detect_format,
    load_dataset,
    load_from_config,
    table_to_dataset,
)
from context_synthetic_recognition.data.schema import Dataset, FeatureType, Label

__all__ = [
    "DATASETS",
    "LOADERS",
    "BuiltinDataset",
    "Dataset",
    "FeatureType",
    "Label",
    "LoadOptions",
    "detect_format",
    "load_builtin",
    "load_dataset",
    "load_from_config",
    "table_to_dataset",
]
