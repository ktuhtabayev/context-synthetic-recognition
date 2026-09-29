"""Run folders and manifests."""

from datetime import datetime
from pathlib import Path

import pytest

from context_synthetic_recognition import __version__
from context_synthetic_recognition.config import config_hash, preset
from context_synthetic_recognition.errors import ConfigError
from context_synthetic_recognition.services.manifest import (
    MANIFEST_NAME,
    RUN_ID_PATTERN,
    build_manifest,
    create_run_dir,
    make_run_id,
    package_versions,
    read_manifest,
    write_manifest,
)

NOON = datetime(2026, 9, 30, 12, 0, 1)


def test_run_id_is_timestamp_and_config_hash() -> None:
    run_id = make_run_id("e1b225e2e083dac1", NOON)
    assert run_id == "20260930_120001_e1b225e2"
    assert RUN_ID_PATTERN.match(run_id)


def test_run_dirs_in_the_same_second_get_a_suffix(tmp_path: Path) -> None:
    first = create_run_dir(tmp_path / "runs", "ab" * 32, NOON)
    second = create_run_dir(tmp_path / "runs", "ab" * 32, NOON)
    third = create_run_dir(tmp_path / "runs", "ab" * 32, NOON)
    assert [p.name for p in (first, second, third)] == [
        "20260930_120001_abababab",
        "20260930_120001_abababab-2",
        "20260930_120001_abababab-3",
    ]
    assert all(RUN_ID_PATTERN.match(p.name) for p in (first, second, third))
    assert all(p.is_dir() and not any(p.iterdir()) for p in (first, second, third))


def test_manifest_round_trip(tmp_path: Path) -> None:
    config = preset("article")
    run_dir = create_run_dir(tmp_path, config_hash(config), NOON)
    manifest = build_manifest(
        config, run_id=run_dir.name, dataset_hash="0" * 64, timings={"fit": 0.25}
    )
    path = write_manifest(manifest, run_dir)
    assert path.name == MANIFEST_NAME
    restored = read_manifest(run_dir)
    assert restored == manifest
    assert restored.config == config
    assert restored.config_hash == config_hash(config)
    assert restored.seed == config.seed
    assert restored.created.tzinfo is not None


def test_manifest_records_the_environment() -> None:
    manifest = build_manifest(preset("template"), run_id="x")
    assert manifest.packages["context-synthetic-recognition"] == __version__
    assert "pydantic" in manifest.packages
    assert manifest.python.count(".") == 2
    assert manifest.platform
    assert set(package_versions()) <= set(manifest.packages) | {"context-synthetic-recognition"}


def test_invalid_or_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read the manifest"):
        read_manifest(tmp_path / "missing.yaml")
    bad = tmp_path / MANIFEST_NAME
    bad.write_text("run_id: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not a valid run manifest"):
        read_manifest(tmp_path)
