"""The comparison with standard classifiers on the article's training and test samples (ADR-079).

The CS-model's table beside them shows the model on the same samples (ADR-080).

Shared by ``csr compare``, ``scripts/run_comparison.py`` and the desktop application: read the
split files of ``datasets/splits/``, run :mod:`~context_synthetic_recognition.comparison`, render
the tables and write them with their checksums; check written results against a new run.

Files of the output folder (``experiments/comparison/`` of the project by default)::

    standard-classifiers.csv .md .tex   Standard classifiers (SVM, Random Forest)
    knn-by-metric.csv .md .tex          k-NN by metric (k = 3 / k = 5)
    cs-model.csv .md .tex               the CS-model, α = 0.1 / α = 0.2
    results.csv .md                     every split × method
    summary.json                        settings, library versions, every number
    checksums.sha256                    SHA-256 of the files above
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from importlib.resources import files
from pathlib import Path

from context_synthetic_recognition.comparison import SPLITS, SplitResult, compare_split
from context_synthetic_recognition.comparison.tables import (
    DECIMALS,
    GROUP,
    comparison_tables,
    summary_data,
)
from context_synthetic_recognition.data.catalog import find_datasets_root
from context_synthetic_recognition.data.splits import load_split_file
from context_synthetic_recognition.errors import DatasetError
from context_synthetic_recognition.export.tables import Cell, Table
from context_synthetic_recognition.export.text import (
    csv_bytes,
    csv_text,
    json_text,
    latex_table,
    markdown_text,
)

SPLITS_DIR = "splits"
"""Sub-folder of the dataset folder with the split files."""
BUNDLED_DIR = "split_files"
"""Folder of the package's data with the copy of the split files the Windows build carries
(ADR-081); an installation from source has none."""
OUTPUT_DIR = ("experiments", "comparison")
"""The output folder, relative to the project root (the dataset folder's parent)."""
CHECKSUMS = "checksums.sha256"
LIBRARIES = ("numpy", "scikit-learn")
"""Libraries the numbers depend on; their versions are recorded in ``summary.json``."""

Progress = Callable[[str], None]
"""Receives a line of progress text (the GUI's status, the CLI's output)."""


@dataclass(frozen=True, eq=False)
class ComparisonOutcome:
    """A run of the comparison: the results, the tables and the files, not yet written."""

    results: tuple[SplitResult, ...]
    tables: tuple[Table, Table, Table, Table]
    """Standard classifiers, k-NN by metric, the CS-model's table and the full results table."""
    files: Mapping[str, bytes]
    """File name → content, in writing order; ``checksums.sha256`` last."""
    versions: Mapping[str, str]
    """Versions of numpy and scikit-learn used."""
    splits_folder: Path


@dataclass(frozen=True)
class ComparisonCheck:
    """Written results compared with their checksums and with a new run."""

    folder: Path
    checked: tuple[str, ...]
    """Files listed in ``checksums.sha256``."""
    problems: tuple[str, ...]
    recorded_versions: Mapping[str, str] = field(default_factory=dict)
    installed_versions: Mapping[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """No problem found."""
        return not self.problems

    @property
    def same_versions(self) -> bool:
        """The installed numpy and scikit-learn are the ones the results were written with."""
        return dict(self.recorded_versions) == dict(self.installed_versions)


def bundled_splits() -> Path | None:
    """The copy of the split files the Windows build carries; ``None`` from source."""
    folder = Path(str(files("context_synthetic_recognition.data") / BUNDLED_DIR))
    return folder if folder.is_dir() else None


def splits_folder(start: Path | None = None) -> Path:
    """``datasets/splits`` of the project, else the copy the Windows build carries.

    The project's dataset folder is searched from ``start`` (the working directory).

    Raises:
        DatasetError: The project has no split files and the installation carries none.
    """
    root = find_datasets_root(start)
    if root is not None and (root / SPLITS_DIR).is_dir():
        return root / SPLITS_DIR
    bundled = bundled_splits()
    if bundled is not None:
        return bundled
    if root is None:
        raise DatasetError("no datasets folder found (the split files are in datasets/splits)")
    raise DatasetError(f"{root}: no '{SPLITS_DIR}' folder with the split files")


def default_output(start: Path | None = None) -> Path:
    """``experiments/comparison`` beside the project's dataset folder.

    Raises:
        DatasetError: There is no dataset folder (give the results folder).
    """
    root = find_datasets_root(start)
    if root is None:
        raise DatasetError("no datasets folder found: give the results folder")
    return root.parent.joinpath(*OUTPUT_DIR)


def library_versions() -> dict[str, str]:
    """Installed versions of numpy and scikit-learn (``missing`` if not installed)."""
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def sha256(content: bytes) -> str:
    """SHA-256 of some bytes."""
    return hashlib.sha256(content).hexdigest()


def render_files(
    results: tuple[SplitResult, ...],
    tables: tuple[Table, Table, Table, Table],
    *,
    versions: Mapping[str, str],
    inputs: Mapping[str, str],
) -> dict[str, bytes]:
    """The output files' contents (UTF-8, LF line ends), the checksums last.

    The CSV files start with a byte-order mark, so that Excel reads them as UTF-8 (ADR-083).
    """
    standard, knn, cs_model, results_table = tables
    texts = {
        "standard-classifiers.csv": csv_text(standard),
        "standard-classifiers.md": markdown_text(standard, DECIMALS, level=2),
        "standard-classifiers.tex": latex_table(standard, DECIMALS),
        "knn-by-metric.csv": csv_text(knn),
        "knn-by-metric.md": markdown_text(knn, DECIMALS, level=2),
        "knn-by-metric.tex": latex_table(knn, DECIMALS),
        "cs-model.csv": csv_text(cs_model),
        "cs-model.md": markdown_text(cs_model, DECIMALS, level=2),
        "cs-model.tex": latex_table(cs_model, DECIMALS),
        "results.csv": csv_text(results_table),
        "results.md": markdown_text(results_table, DECIMALS, level=2),
        "summary.json": json_text(summary_data(results, versions=versions, inputs=inputs)),
    }
    files = {
        name: csv_bytes(text) if name.endswith(".csv") else text.encode("utf-8")
        for name, text in texts.items()
    }
    files[CHECKSUMS] = "".join(f"{sha256(data)}  {name}\n" for name, data in files.items()).encode()
    return files


def run_comparison(
    splits: Path | None = None, *, progress: Progress | None = None
) -> ComparisonOutcome:
    """Read the split files, run every method and render the output files (nothing is written).

    Args:
        splits: The folder of the split files; ``datasets/splits`` of the project by default.
        progress: Called with a line of text before each split.

    Raises:
        DatasetError: A split file is missing or malformed.
        ConfigError: scikit-learn is not installed.
    """
    folder = splits or splits_folder()
    results = []
    inputs = {}
    for spec in SPLITS:
        path = folder / spec.file
        if progress is not None:
            progress(f"{spec.key} — {spec.title}: {path.name}")
        split = load_split_file(path)
        inputs[spec.file] = sha256(path.read_bytes())
        results.append(compare_split(spec, split))
    done = tuple(results)
    tables = comparison_tables(done)
    versions = library_versions()
    files = render_files(done, tables, versions=versions, inputs=inputs)
    return ComparisonOutcome(done, tables, files, versions, folder)


def write_comparison(outcome: ComparisonOutcome, out: Path) -> list[Path]:
    """Write the files of a run to ``out`` (created if needed); returns the paths written."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, data in outcome.files.items():
        path = out / name
        path.write_bytes(data)
        written.append(path)
    return written


def _recorded_versions(folder: Path) -> dict[str, str]:
    try:
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    versions = summary.get("versions", {}) if isinstance(summary, dict) else {}
    return {str(k): str(v) for k, v in versions.items()} if isinstance(versions, dict) else {}


def check_comparison(
    out: Path, *, splits: Path | None = None, progress: Progress | None = None
) -> ComparisonCheck:
    """Check written results: every file matches ``checksums.sha256`` and a new run.

    Args:
        out: The folder the results were written to.
        splits: The folder of the split files; ``datasets/splits`` of the project by default.
        progress: Called with a line of text before each split of the new run.
    """
    problems: list[str] = []
    listing = out / CHECKSUMS
    if not listing.is_file():
        return ComparisonCheck(out, (), (f"{CHECKSUMS}: missing in {out}",))
    listed: dict[str, str] = {}
    for line in listing.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        listed[name] = digest
    for name, digest in listed.items():
        path = out / name
        if not path.is_file():
            problems.append(f"{name}: missing")
        elif sha256(path.read_bytes()) != digest:
            problems.append(f"{name}: does not match {CHECKSUMS}")
    outcome = run_comparison(splits, progress=progress)
    expected = outcome.files
    for name in sorted(set(listed) ^ (set(expected) - {CHECKSUMS})):
        where = "a new run writes it" if name in expected else "a new run does not write it"
        problems.append(f"{name}: {'not listed' if name not in listed else 'listed'}, {where}")
    for name, data in expected.items():
        path = out / name
        if path.is_file() and path.read_bytes() != data:
            problems.append(f"{name}: differs from a new run")
    return ComparisonCheck(
        out,
        tuple(listed),
        tuple(problems),
        _recorded_versions(out),
        dict(outcome.versions),
    )


def _versions_text(versions: Mapping[str, str]) -> str:
    return ", ".join(f"{name} {version}" for name, version in versions.items()) or "—"


def check_table(check: ComparisonCheck) -> Table:
    """The result of a check: every listed file with ✓ or ✗, then the other problems."""
    rows: list[tuple[Cell, ...]] = []
    for name in check.checked:
        found = [p.removeprefix(f"{name}: ") for p in check.problems if p.startswith(f"{name}: ")]
        rows.append((name, "✗" if found else "✓", "; ".join(found)))
    for problem in check.problems:
        if not any(problem.startswith(f"{name}: ") for name in check.checked):
            name, _, text = problem.partition(": ")
            rows.append((name, "✗", text))
    versions = (
        f"written with {_versions_text(check.recorded_versions)}"
        if check.same_versions
        else (
            f"written with {_versions_text(check.recorded_versions)}, installed: "
            f"{_versions_text(check.installed_versions)}"
        )
    )
    return Table(
        key="check",
        title=f"Check of {check.folder}",
        columns=("File", "Status", "Problem"),
        rows=tuple(rows),
        group=GROUP,
        note=f"Checksums and a new run compared byte for byte; {versions}.",
    )
