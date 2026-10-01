"""Dataset snapshots: a dataset exactly as it was used, kept in the run folder.

A run folder holds ``dataset.json`` — values, labels, feature names and types, object ids and
categories — so that the run can be repeated and exported later without the original file
(reproducibility, ADR-036). Floats are written with ``repr``, so the snapshot has the same
content hash as the dataset it was taken from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from context_synthetic_recognition.data.schema import Dataset, FeatureType, as_labels
from context_synthetic_recognition.errors import DatasetError

SNAPSHOT_NAME = "dataset.json"
SNAPSHOT_FORMAT = "context-synthetic-recognition/dataset"
SNAPSHOT_VERSION = 1


def dataset_data(dataset: Dataset) -> dict[str, Any]:
    """A dataset as JSON-ready data."""
    return {
        "format": SNAPSHOT_FORMAT,
        "version": SNAPSHOT_VERSION,
        "name": dataset.name,
        "source": dataset.source,
        "content_hash": dataset.content_hash(),
        "feature_names": list(dataset.feature_names),
        "feature_types": [t.value for t in dataset.feature_types],
        "object_ids": list(dataset.object_ids),
        "labels": [v.item() for v in dataset.y],
        "categories": {name: list(levels) for name, levels in dataset.categories.items()},
        "X": [[float(v) for v in row] for row in dataset.X],
    }


def dataset_from_data(data: Any, *, source: str = "snapshot") -> Dataset:
    """A dataset from the data written by :func:`dataset_data`.

    Raises:
        DatasetError: Not a dataset snapshot, or its content does not match its recorded hash.
    """
    if not isinstance(data, dict) or data.get("format") != SNAPSHOT_FORMAT:
        raise DatasetError(f"{source}: not a dataset snapshot of this package")
    try:
        dataset = Dataset(
            X=np.array(data["X"], dtype=np.float64),
            y=as_labels(data["labels"]),
            feature_types=tuple(FeatureType(t) for t in data["feature_types"]),
            feature_names=tuple(data["feature_names"]),
            object_ids=tuple(data["object_ids"]),
            name=str(data["name"]),
            categories={k: tuple(v) for k, v in data.get("categories", {}).items()},
            source=data.get("source"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DatasetError(f"{source}: the dataset snapshot is incomplete ({error})") from error
    recorded = data.get("content_hash")
    if recorded is not None and recorded != dataset.content_hash():
        raise DatasetError(f"{source}: the dataset snapshot does not match its recorded hash")
    return dataset


def save_snapshot(dataset: Dataset, path: Path) -> Path:
    """Write a dataset snapshot (``path`` may be a folder: the file is then ``dataset.json``)."""
    file = path / SNAPSHOT_NAME if path.is_dir() else path
    text = json.dumps(dataset_data(dataset), ensure_ascii=False, separators=(",", ":"))
    file.write_text(text + "\n", encoding="utf-8", newline="\n")
    return file


def load_snapshot(path: Path) -> Dataset:
    """Read a dataset snapshot (``path`` may be the file or the folder that holds it).

    Raises:
        DatasetError: The file is missing, unreadable or not a snapshot.
    """
    file = path / SNAPSHOT_NAME if path.is_dir() else path
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except OSError as error:
        raise DatasetError(
            f"{file}: cannot read the dataset snapshot ({error.strerror})"
        ) from error
    except json.JSONDecodeError as error:
        raise DatasetError(f"{file}: not valid JSON ({error})") from error
    return dataset_from_data(data, source=str(file))
