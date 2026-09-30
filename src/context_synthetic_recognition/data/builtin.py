"""Built-in datasets shipped with the package (registry :data:`DATASETS`).

The files are the template project's Heart-Disease files in its extended layout (``m, n, c`` /
objects / type flags), line endings normalized to LF. Heart-Disease is the Statlog (Heart)
dataset of the UCI Machine Learning Repository (CC BY 4.0): 6 quantitative and 7 nominal
features; class 1 = absence, class 2 = presence of heart disease.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files

from context_synthetic_recognition.core.registry import Registry
from context_synthetic_recognition.data.loaders import LoadOptions, load_template_extended
from context_synthetic_recognition.data.schema import Dataset


@dataclass(frozen=True)
class BuiltinDataset:
    """A dataset file inside the package."""

    filename: str
    """File in ``context_synthetic_recognition/data/datasets``."""
    description: str
    """One line for listings."""

    def load(self) -> Dataset:
        """Read the dataset."""
        resource = files("context_synthetic_recognition.data") / "datasets" / self.filename
        with as_file(resource) as path:
            return load_template_extended(path, LoadOptions(name=path.stem))


DATASETS: Registry[BuiltinDataset] = Registry("datasets")
"""Built-in datasets by name."""

DATASETS.add(
    "heart-disease-10",
    BuiltinDataset(
        "Heart-Disease (10, 13, 2).csv",
        "the 10 objects of the Excel experiment (|K1| = 4, |K2| = 6)",
    ),
    summary="Heart-Disease (10, 13, 2) — the Excel experiment",
)
DATASETS.add(
    "heart-disease-270",
    BuiltinDataset(
        "Heart-Disease (270, 13, 2).csv",
        "Statlog (Heart), UCI: 270 objects (|K1| = 150, |K2| = 120)",
    ),
    summary="Heart-Disease (270, 13, 2) — Statlog (Heart), UCI",
)


def load_builtin(name: str) -> Dataset:
    """Load the built-in dataset ``name`` (see :data:`DATASETS`)."""
    return DATASETS.get(name).load()
