"""The project's dataset folder: discovery, identity checks and name resolution.

Layout (the author's, as in the template project; ADR-035)::

    datasets/
      default.dat, default.csv         the default dataset (Heart-Disease (10, 13, 2))
      raw/<Family>/<Name (m, n, c)>.dat|.csv
                                       original datasets,
                                       e.g. raw/Heart-Disease/Heart-Disease (270, 13, 2).dat
      synthetic/                       reserved for pre-existing synthetic datasets — not read

Every file is in the extended layout (:func:`~context_synthetic_recognition.data.loaders.
load_template_extended`): a first row ``m n c``; m rows of n feature values and the class label; a
last row of n type flags, 1 = quantitative (set I), 0 = nominal (set J). ``.dat`` is whitespace-
separated, ``.csv`` comma-separated; the files are kept byte for byte.

The ``.dat`` and ``.csv`` files of one stem are the same dataset in two formats: the ``.dat`` is
read (the ``.csv`` of a large dataset may be a rounded copy), and :meth:`DatasetCatalog.check`
verifies that both hold identical data and that the header's (m, n, c) matches the file name.

Datasets are named by id (``heart-disease-270``), by display name (``Heart-Disease (270, 13, 2)``),
by ``default`` or by a file path. Without a ``datasets`` folder the package's built-in copies are
used, so an installed GUI still has its default dataset.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from context_synthetic_recognition.data.loaders import LoadOptions, load_template_extended
from context_synthetic_recognition.data.schema import Dataset
from context_synthetic_recognition.errors import DatasetError

logger = logging.getLogger(__name__)

DATASETS_DIR = "datasets"
"""Name of the project's dataset folder."""
DEFAULT_STEM = "default"
"""Stem of the default dataset's files at the root of the folder."""
RAW_DIR = "raw"
"""Sub-folder of the original datasets."""
RESERVED_DIRS = ("synthetic",)
"""Sub-folders that are not read (pre-existing synthetic datasets, ADR-035)."""
EXTENSIONS = (".dat", ".csv")
"""File formats of the catalogue, in order of preference."""
DEFAULT_ID = "default"
ENVIRONMENT_VARIABLE = "CSR_DATASETS"
"""Overrides the dataset folder."""

_SHAPE = re.compile(r"^(?P<name>.*?)\s*\((?P<m>\d+),\s*(?P<n>\d+),\s*(?P<c>\d+)\)$")


def _slug(text: str) -> str:
    return "-".join(re.findall(r"[a-z0-9]+", text.lower()))


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes (the files are stored byte for byte)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class CatalogEntry:
    """One dataset of the folder: the same data in one or more formats."""

    id: str
    """``default`` or ``<name>-<m>`` (``-<n>`` added when two datasets would share an id)."""
    name: str
    """Display name: the file stem, e.g. ``Heart-Disease (270, 13, 2)``."""
    files: tuple[Path, ...]
    """The files, ``.dat`` first."""
    role: str
    """``default`` or ``raw``."""
    shape: tuple[int, int, int] | None
    """(m, n, c) from the file name (the default dataset: from its first row)."""

    @property
    def path(self) -> Path:
        """The file that is read (the ``.dat`` if there is one)."""
        return self.files[0]

    @property
    def formats(self) -> tuple[str, ...]:
        """The extensions present, e.g. ``(".dat", ".csv")``."""
        return tuple(f.suffix.lower() for f in self.files)

    def load(self) -> Dataset:
        """Read the dataset from its preferred file."""
        return load_extended(self.path, self.name)


@dataclass(frozen=True)
class EntryCheck:
    """The result of :meth:`DatasetCatalog.check` for one entry."""

    entry: CatalogEntry
    shape: tuple[int, int, int] | None
    """(m, n, c) as loaded."""
    content_hash: str | None
    """Hash of the data (values, types, labels) of the preferred file."""
    sha256: dict[str, str] = field(default_factory=dict)
    """SHA-256 of every file, by extension."""
    problems: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        """No problem found."""
        return not self.problems


def read_shape(path: Path) -> tuple[int, int, int] | None:
    """(m, n, c) from the first row of an extended-layout file, or ``None`` if it has none."""
    try:
        with path.open(encoding="utf-8-sig") as handle:
            first = handle.readline()
    except (OSError, UnicodeDecodeError):
        return None
    cells = [cell for cell in re.split(r"[,;\s]+", first.strip()) if cell]
    if len(cells) != 3 or not all(cell.isdigit() for cell in cells):
        return None
    m, n, c = (int(cell) for cell in cells)
    return m, n, c


def load_extended(path: Path, name: str | None = None) -> Dataset:
    """Read a file of the extended layout (``.dat`` or ``.csv``)."""
    if not path.is_file():
        raise DatasetError(f"dataset file not found: {path}")
    return load_template_extended(path, LoadOptions(name=name or path.stem))


@dataclass(frozen=True)
class DatasetCatalog:
    """The datasets of one ``datasets`` folder."""

    root: Path
    entries: tuple[CatalogEntry, ...]

    @property
    def default(self) -> CatalogEntry | None:
        """The default dataset, if the folder has one."""
        return next((e for e in self.entries if e.role == DEFAULT_ID), None)

    def __iter__(self) -> Iterator[CatalogEntry]:
        """Iterate over the entries (the default first, then the raw datasets by name)."""
        return iter(self.entries)

    def find(self, key: str) -> CatalogEntry | None:
        """The entry with this id or display name (case-insensitive), or ``None``."""
        wanted = key.strip().lower()
        for entry in self.entries:
            if wanted in (entry.id, entry.name.lower(), _slug(entry.name)):
                return entry
        return None

    def check(self) -> list[EntryCheck]:
        """Load every file of every entry; compare the header with the name and the formats.

        Returns:
            One result per entry: the shape and data hash as loaded, the SHA-256 of every file
            and the problems found (a header that contradicts the file name, formats that hold
            different data, unreadable files).
        """
        results = []
        for entry in self.entries:
            problems: list[str] = []
            sha = {f.suffix.lower(): file_sha256(f) for f in entry.files}
            loaded: dict[str, Dataset] = {}
            for file in entry.files:
                try:
                    loaded[file.suffix.lower()] = load_extended(file, entry.name)
                except DatasetError as error:
                    problems.append(str(error))
            first = next(iter(loaded.values()), None)
            shape = (first.m, first.n, len(first.classes)) if first is not None else None
            if first is not None and entry.shape is not None and shape != entry.shape:
                problems.append(
                    f"{entry.path.name}: the file holds {shape}, its name says {entry.shape}"
                )
            hashes = {ext: d.content_hash() for ext, d in loaded.items()}
            if len(set(hashes.values())) > 1:
                problems.append(
                    f"{entry.name}: the {' and '.join(hashes)} files hold different data "
                    f"({entry.path.suffix} is used)"
                )
            results.append(
                EntryCheck(
                    entry,
                    shape,
                    hashes.get(entry.path.suffix.lower()),
                    sha,
                    tuple(problems),
                )
            )
        return results


def find_datasets_root(start: Path | None = None) -> Path | None:
    """The dataset folder, or ``None`` if there is none.

    ``$CSR_DATASETS`` if set, otherwise the nearest ``datasets`` folder from ``start`` (the working
    directory by default) upwards.
    """
    override = os.environ.get(ENVIRONMENT_VARIABLE)
    if override:
        root = Path(override)
        if not root.is_dir():
            raise DatasetError(f"{ENVIRONMENT_VARIABLE} = {override}: not a folder")
        return root
    folder = (start or Path.cwd()).resolve()
    for candidate in (folder, *folder.parents):
        if (candidate / DATASETS_DIR).is_dir():
            return candidate / DATASETS_DIR
    return None


def _grouped(files: list[Path]) -> dict[Path, list[Path]]:
    groups: dict[Path, list[Path]] = {}
    for file in files:
        groups.setdefault(file.with_suffix(""), []).append(file)
    for group in groups.values():
        group.sort(key=lambda f: EXTENSIONS.index(f.suffix.lower()))
    return groups


def discover(root: Path) -> DatasetCatalog:
    """Scan a ``datasets`` folder (see the module docstring for the layout)."""
    entries: list[CatalogEntry] = []
    defaults = [root / f"{DEFAULT_STEM}{ext}" for ext in EXTENSIONS]
    present = [f for f in defaults if f.is_file()]
    if present:
        shape = read_shape(present[0])
        entries.append(CatalogEntry(DEFAULT_ID, DEFAULT_STEM, tuple(present), DEFAULT_ID, shape))
    raw_dir = root / RAW_DIR
    raw_files = (
        [f for f in raw_dir.rglob("*") if f.is_file() and f.suffix.lower() in EXTENSIONS]
        if raw_dir.is_dir()
        else []
    )
    raw: list[tuple[str, tuple[Path, ...], tuple[int, int, int] | None]] = []
    for stem, group in sorted(_grouped(raw_files).items(), key=lambda item: item[0].name.lower()):
        match = _SHAPE.match(stem.name)
        if match:
            m, n, c = int(match["m"]), int(match["n"]), int(match["c"])
            raw.append((f"{_slug(match['name'])}-{m}", tuple(group), (m, n, c)))
        else:
            raw.append((_slug(stem.name), tuple(group), None))
    bases = [base for base, _, _ in raw]
    for base, files, shape in raw:
        shared = bases.count(base) > 1 and shape is not None  # e.g. DDos (10000, 80 | 22, 2)
        identifier = f"{base}-{shape[1]}" if shared and shape else base
        entries.append(CatalogEntry(identifier, files[0].stem, files, RAW_DIR, shape))
    ids = [e.id for e in entries]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise DatasetError(f"{root}: several datasets share the id(s) {', '.join(duplicates)}")
    return DatasetCatalog(root, tuple(entries))


def project_catalog(start: Path | None = None) -> DatasetCatalog | None:
    """The catalogue of the project's dataset folder, or ``None`` if there is none."""
    root = find_datasets_root(start)
    return discover(root) if root is not None else None


def resolve_dataset(source: str | Path | None = None, start: Path | None = None) -> Dataset:
    """Load a dataset by file path, id, display name or ``default`` (``None`` = the default).

    Order: an existing file; the project catalogue (``datasets`` folder); the package's built-in
    datasets (``heart-disease-10`` is also the default when there is no ``datasets`` folder).

    Raises:
        DatasetError: Nothing matches; the message lists what is available.
    """
    # data.builtin imports the loaders, which this module imports — look it up here
    from context_synthetic_recognition.data.builtin import DATASETS, load_builtin
    from context_synthetic_recognition.data.loaders import load_dataset

    key = DEFAULT_ID if source is None else str(source)
    path = Path(key)
    if path.is_file():
        return load_dataset(path)
    if path.suffix.lower() in {*EXTENSIONS, ".xlsx", ".xlsm", ".txt", ".parquet", ".pq"}:
        raise DatasetError(f"dataset file not found: {path}")
    catalog = project_catalog(start)
    if catalog is not None:
        entry = catalog.find(key)
        if entry is not None:
            return entry.load()
    if key == DEFAULT_ID and (catalog is None or catalog.default is None):
        return load_builtin("heart-disease-10")
    if key in DATASETS:
        return load_builtin(key)
    available = [e.id for e in catalog] if catalog is not None else []
    available += [f"builtin: {name}" for name in DATASETS.names()]
    raise DatasetError(f"unknown dataset {key!r}; available: {', '.join(available)}")
