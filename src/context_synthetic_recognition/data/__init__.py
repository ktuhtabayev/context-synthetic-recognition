"""Data layer: dataset schema, loaders of every supported format, built-in datasets, split files."""

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
from context_synthetic_recognition.data.splits import FixedSplit, load_split_file

__all__ = [
    "DATASETS",
    "LOADERS",
    "BuiltinDataset",
    "Dataset",
    "FeatureType",
    "FixedSplit",
    "Label",
    "LoadOptions",
    "detect_format",
    "load_builtin",
    "load_dataset",
    "load_from_config",
    "load_split_file",
    "table_to_dataset",
]
