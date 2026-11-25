"""Discovery of the split files: every file of ``datasets/splits`` is a sample (ADR-090).

Every ``*.dat`` file below the split folder, in any sub-folder, is read by its name and its
content; nothing lists the files in code, so a file added later is found without a change:

- the name follows ``<Dataset> (<m>, <n>, <c>) — training <t> (<metric>).dat``
  (:func:`~context_synthetic_recognition.data.splits.parse_split_name`);
- the file follows the split-file convention (ADR-082) and its header holds the m, n and c of the
  name; the split markers give t training objects;
- the metric is a registered metric (``METRICS``); its canonical name is kept.

A file that fails a check is listed as unavailable, with its file, its row (where the problem has
one) and the reason; it is never used and never stops the others. The key of a split is
``<Dataset>-<t>`` (``German-341``): internal, never shown in the desktop application. Two files
with one key are both unavailable. The CLI also accepts the training size alone (``341``) when
exactly one split has it — the keys of the earlier versions.

Reading the folder is the only I/O; the checks are pure functions of the names and the loaded
splits.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from context_synthetic_recognition.comparison.settings import SplitSpec, metric_title
from context_synthetic_recognition.core.local_context.metrics import METRICS
from context_synthetic_recognition.data.splits import (
    SPLIT_SUFFIX,
    FixedSplit,
    SplitName,
    load_split_file,
    parse_split_name,
)
from context_synthetic_recognition.errors import CSRError, DatasetError

_ROW = re.compile(r"\brow (\d+)\b")


@dataclass(frozen=True, eq=False)
class SplitFile:
    """One file of the split folder: its name, its samples, or why it cannot be used."""

    file: str
    """The file, relative to the split folder (``German/German (1000, 7, 2) — … .dat``)."""
    name: SplitName | None
    """What the name says (``None``: the name does not follow the pattern)."""
    split: FixedSplit | None = None
    """The objects and the two samples (``None``: not read, or unavailable)."""
    sha256: str = ""
    """SHA-256 of the file's bytes (empty if it was not read)."""
    reason: str = ""
    """Why the file cannot be used (empty if it can)."""
    row: int | None = None
    """The file's line the problem is on, if it has one."""

    @property
    def available(self) -> bool:
        """Read, and every check passed."""
        return self.split is not None and not self.reason

    @property
    def key(self) -> str:
        """``German-341``; the file itself when the name does not follow the pattern."""
        return self.name.key if self.name is not None else self.file

    @property
    def dataset(self) -> str:
        """The dataset of the name; else the sub-folder the file is in (or its stem)."""
        if self.name is not None:
            return self.name.dataset
        parts = Path(self.file).parts
        return parts[0] if len(parts) > 1 else Path(self.file).stem

    @property
    def title(self) -> str:
        """``Euclidean split``; the file name when the name does not follow the pattern."""
        return f"{metric_title(self.name.metric)} split" if self.name else Path(self.file).name

    @property
    def short_label(self) -> str:
        """``German · Euclidean split · training 341``."""
        if self.name is None:
            return f"{self.dataset} · {self.title}"
        return f"{self.dataset} · {self.title} · training {self.name.training}"

    @property
    def label(self) -> str:
        """``German · Euclidean split · training 341 (173, 168)`` — the name everywhere."""
        if self.split is None:
            return self.short_label
        size, *classes = self.split.training_sizes
        return f"{self.dataset} · {self.title} · training {size} ({', '.join(map(str, classes))})"

    @property
    def details(self) -> str:
        """``7 features · training 341 (173, 168) · test 659`` (from the name when not read)."""
        if self.split is None:
            if self.name is None:
                return ""
            return f"{self.name.n} features · training {self.name.training}"
        size, *classes = self.split.training_sizes
        return (
            f"{self.split.dataset.n} features · training {size} "
            f"({', '.join(map(str, classes))}) · test {self.split.test_sizes[0]}"
        )

    @property
    def spec(self) -> SplitSpec:
        """The split as the comparison reads it.

        Raises:
            DatasetError: The name does not follow the pattern.
        """
        if self.name is None:
            raise DatasetError(self.reason or f"{self.file}: not a split file")
        return SplitSpec.of(self.key, self.dataset, self.file, self.name.metric, self.name.training)


@dataclass(frozen=True, eq=False)
class SplitCatalog:
    """The split files of a folder, by dataset (alphabetical) and training size."""

    folder: Path | None
    files: tuple[SplitFile, ...]
    problem: str = ""
    """Why the folder could not be read (empty if it could)."""

    def find(self, key: str) -> SplitFile | None:
        """The file with this key (``None`` if there is none)."""
        return next((f for f in self.files if f.key == key), None)

    def resolve(self, key: str) -> SplitFile:
        """The file of a key, or of a training size that exactly one split has (``341``).

        Raises:
            DatasetError: No split has the key, or several have that training size.
        """
        found = self.find(key)
        if found is None:
            found = next((f for f in self.files if f.key.casefold() == key.casefold()), None)
        if found is not None:
            return found
        if key.isdigit():
            sized = [f for f in self.files if f.name is not None and f.name.training == int(key)]
            if len(sized) == 1:
                return sized[0]
            if sized:
                raise DatasetError(
                    f"split {key!r} is ambiguous: {', '.join(f.key for f in sized)}; give the key"
                )
        if self.problem:
            raise DatasetError(self.problem)
        known = ", ".join(f.key for f in self.files if f.name is not None) or "none"
        raise DatasetError(f"unknown split {key!r}; the splits are {known}")

    def require(self, key: str) -> SplitFile:
        """An available file by key or training size.

        Raises:
            DatasetError: Unknown key, or the file is unavailable (with the reason).
        """
        found = self.resolve(key)
        if not found.available or found.split is None:
            raise DatasetError(found.reason or f"{found.file}: the file was not read")
        return found

    def datasets(self) -> tuple[tuple[str, tuple[SplitFile, ...]], ...]:
        """(dataset, its files) in the catalogue's order."""
        grouped: dict[str, list[SplitFile]] = {}
        for file in self.files:
            grouped.setdefault(file.dataset, []).append(file)
        return tuple((name, tuple(files)) for name, files in grouped.items())

    @property
    def keys(self) -> tuple[str, ...]:
        """Keys of the available files."""
        return tuple(f.key for f in self.files if f.available)


def _order(file: SplitFile) -> tuple[str, int, str]:
    return file.dataset.casefold(), file.name.training if file.name else 0, file.file.casefold()


def _row(message: str) -> int | None:
    found = _ROW.search(message)
    return int(found[1]) if found else None


def check_split(file: str, name: SplitName, split: FixedSplit) -> tuple[str, int | None]:
    """The checks of a read file against its name: ``(reason, row)``, empty if it passes."""
    data = split.dataset
    header = (data.m, data.n, len(data.classes))
    if header != (name.m, name.n, name.c):
        said = " ".join(map(str, header))
        return (
            (
                f"{Path(file).name}: row 1 (the header) says m n c = {said}, "
                f"the name says ({name.m}, {name.n}, {name.c})"
            ),
            1,
        )
    if split.training.size != name.training:
        return (
            (
                f"{Path(file).name}: the split markers give {split.training.size} training "
                f"objects, the name says training {name.training}"
            ),
            None,
        )
    return "", None


def _metric(name: SplitName) -> tuple[SplitName, str]:
    """The name with the metric's canonical name, or why the metric is unknown."""
    try:
        canonical = METRICS.info(name.metric).name
    except CSRError:
        return name, (
            f"unknown metric {name.metric!r} in the name; the metrics are "
            f"{', '.join(METRICS.names())}"
        )
    return replace(name, metric=canonical), ""


def _read(folder: Path, relative: str, name: SplitName) -> SplitFile:
    path = folder / relative
    try:
        content = path.read_bytes()
        split = load_split_file(path)
    except (OSError, CSRError) as error:
        message = str(error)
        return SplitFile(relative, name, reason=message, row=_row(message))
    reason, row = check_split(relative, name, split)
    digest = hashlib.sha256(content).hexdigest()
    if reason:
        return SplitFile(relative, name, sha256=digest, reason=reason, row=row)
    return SplitFile(relative, name, split, digest)


def discover_splits(folder: Path, *, read: Collection[str] | None = None) -> SplitCatalog:
    """Every split file below ``folder``, checked (see the module docstring).

    Args:
        folder: The split folder.
        read: Keys of the files to read; ``None`` reads every file. The names of all files are
            checked either way (for the keys), the others' content is not read.
    """
    if not folder.is_dir():
        return SplitCatalog(folder, (), f"{folder}: no folder of split files")
    paths = sorted(p for p in folder.rglob(f"*{SPLIT_SUFFIX}") if p.is_file())
    named: list[tuple[str, SplitName | None, str]] = []
    for path in paths:
        relative = path.relative_to(folder).as_posix()
        try:
            parsed, problem = _metric(parse_split_name(path.name))
        except DatasetError as error:
            named.append((relative, None, str(error)))
            continue
        named.append((relative, parsed, f"{path.name}: {problem}" if problem else ""))
    owners: dict[str, list[str]] = {}
    for relative, found, _ in named:
        if found is not None:
            owners.setdefault(found.key, []).append(relative)
    files = []
    for relative, found, problem in named:
        if found is None or problem:
            files.append(SplitFile(relative, found, reason=problem))
            continue
        name = found
        others = [o for o in owners[name.key] if o != relative]
        if others:
            reason = (
                f"{Path(relative).name}: the key {name.key} is also the key of "
                f"{', '.join(others)}; rename one of the files"
            )
            files.append(SplitFile(relative, name, reason=reason))
        elif read is None or name.key in read:
            files.append(_read(folder, relative, name))
        else:
            files.append(SplitFile(relative, name))
    return SplitCatalog(folder, tuple(sorted(files, key=_order)))


def split_files(catalog: SplitCatalog, keys: Iterable[str]) -> tuple[SplitFile, ...]:
    """The available files of ``keys`` (keys or training sizes), in the order given.

    Raises:
        DatasetError: A key is unknown or its file is unavailable.
    """
    return tuple(catalog.require(key) for key in keys)


def resolve_keys(folder: Path, keys: Iterable[str]) -> tuple[str, ...]:
    """The keys of ``keys`` (keys or training sizes), from the names of the files alone.

    Raises:
        DatasetError: A key is unknown or ambiguous, or the folder does not exist.
    """
    catalog = discover_splits(folder, read=())
    return tuple(catalog.resolve(key).key for key in keys)


def read_splits(folder: Path, keys: Iterable[str]) -> tuple[SplitFile, ...]:
    """Read the files of ``keys`` (keys or training sizes) — and only those — in the order given.

    Raises:
        DatasetError: A key is unknown or ambiguous, or its file is unavailable (with the reason).
    """
    wanted = resolve_keys(folder, keys)
    return split_files(discover_splits(folder, read=set(wanted)), wanted)
