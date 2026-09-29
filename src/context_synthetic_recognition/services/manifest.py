"""Run folders and run manifests (reproducibility).

Every run writes ``<runs_dir>/<YYYYMMDD_HHMMSS>_<config hash[:8]>/manifest.yaml`` with the full
configuration, its hash, the dataset hash, the seed, package versions and timings, so that the run
can be repeated from the manifest alone.
"""

from __future__ import annotations

import platform
import re
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from context_synthetic_recognition import __version__
from context_synthetic_recognition.config.io import config_hash
from context_synthetic_recognition.config.models import ExperimentConfig
from context_synthetic_recognition.errors import ConfigError

MANIFEST_NAME = "manifest.yaml"
RUN_ID_PATTERN = re.compile(r"^(\d{8}_\d{6})_([0-9a-f]{8})(?:-(\d+))?$")

# Distributions whose versions are recorded when installed (numerical results depend on them).
TRACKED_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "openpyxl",
    "pydantic",
    "PyYAML",
    "typer",
    "PySide6",
    "matplotlib",
    "scikit-learn",
)


def make_run_id(hash_hex: str, now: datetime) -> str:
    """Run id ``YYYYMMDD_HHMMSS_<first 8 hex digits of the configuration hash>``."""
    return f"{now:%Y%m%d_%H%M%S}_{hash_hex[:8]}"


def create_run_dir(runs_root: Path, hash_hex: str, now: datetime | None = None) -> Path:
    """Create a new, empty run folder; a clash in the same second gets a ``-2``, ``-3`` suffix."""
    stamp = now if now is not None else datetime.now()
    base = make_run_id(hash_hex, stamp)
    runs_root.mkdir(parents=True, exist_ok=True)
    candidate, counter = runs_root / base, 1
    while True:
        try:
            candidate.mkdir()
        except FileExistsError:
            counter += 1
            candidate = runs_root / f"{base}-{counter}"
        else:
            return candidate


def package_versions() -> dict[str, str]:
    """Installed versions of this package and of the tracked dependencies."""
    versions = {"context-synthetic-recognition": __version__}
    for name in TRACKED_DISTRIBUTIONS:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


class RunManifest(BaseModel):
    """Everything needed to identify and repeat a run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    created: datetime
    """Creation time (UTC)."""
    config: ExperimentConfig
    config_hash: str
    dataset_hash: str | None = None
    """SHA-256 of the dataset content as loaded (filled once the data layer exists)."""
    seed: int
    python: str = Field(default_factory=lambda: sys.version.split()[0])
    platform: str = Field(default_factory=platform.platform)
    packages: dict[str, str] = Field(default_factory=package_versions)
    timings: dict[str, float] = Field(default_factory=dict)
    """Seconds per stage."""


def build_manifest(
    config: ExperimentConfig,
    *,
    run_id: str,
    dataset_hash: str | None = None,
    timings: dict[str, float] | None = None,
) -> RunManifest:
    """Manifest of a run with the current environment."""
    return RunManifest(
        run_id=run_id,
        created=datetime.now(tz=UTC),
        config=config,
        config_hash=config_hash(config),
        dataset_hash=dataset_hash,
        seed=config.seed,
        timings=dict(timings or {}),
    )


def write_manifest(manifest: RunManifest, run_dir: Path) -> Path:
    """Write ``manifest.yaml`` into the run folder."""
    path = run_dir / MANIFEST_NAME
    text = yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def read_manifest(path: Path) -> RunManifest:
    """Read a manifest; ``path`` may be the file or its run folder.

    Raises:
        ConfigError: The file is missing, unreadable or not a valid manifest.
    """
    file = path / MANIFEST_NAME if path.is_dir() else path
    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8"))
        return RunManifest.model_validate(data)
    except OSError as error:
        raise ConfigError(f"{file}: cannot read the manifest ({error.strerror}).") from error
    except (yaml.YAMLError, ValidationError) as error:
        raise ConfigError(f"{file}: not a valid run manifest.\n{error}") from error
