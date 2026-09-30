"""Reading, writing and hashing experiment configurations (YAML, TOML, JSON)."""

from __future__ import annotations

import hashlib
import json
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any

import tomli_w
import yaml
from pydantic import ValidationError

from context_synthetic_recognition.config.models import ExperimentConfig
from context_synthetic_recognition.config.presets import active_deviations, matching_preset
from context_synthetic_recognition.errors import ConfigError


class ConfigFormat(StrEnum):
    """Serialisation formats of a configuration."""

    YAML = "yaml"
    TOML = "toml"
    JSON = "json"


_SUFFIXES = {
    ".yaml": ConfigFormat.YAML,
    ".yml": ConfigFormat.YAML,
    ".toml": ConfigFormat.TOML,
    ".json": ConfigFormat.JSON,
}


def format_for(path: Path) -> ConfigFormat:
    """Configuration format implied by the file extension."""
    try:
        return _SUFFIXES[path.suffix.lower()]
    except KeyError:
        known = ", ".join(_SUFFIXES)
        raise ConfigError(f"{path.name}: unknown configuration extension (use {known}).") from None


def to_data(config: ExperimentConfig) -> dict[str, Any]:
    """Plain JSON-compatible data of a configuration."""
    return config.model_dump(mode="json")


def config_hash(config: ExperimentConfig) -> str:
    """SHA-256 of the canonical JSON form (sorted keys); independent of the file format."""
    canonical = json.dumps(
        to_data(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _without_none(value: Any) -> Any:
    """TOML has no null: drop ``None`` entries (they all mean "use the default")."""
    if isinstance(value, dict):
        return {key: _without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_none(item) for item in value if item is not None]
    return value


def _header(config: ExperimentConfig) -> list[str]:
    name = matching_preset(config)
    lines = [
        "context-synthetic-recognition experiment configuration",
        f"preset: {name.value if name else 'custom'}",
    ]
    for deviation in active_deviations(config):
        lines.append(f"⚠ template calculation ({deviation.adr}): {deviation.statement}")
    return lines


def dumps(config: ExperimentConfig, fmt: ConfigFormat | str = ConfigFormat.YAML) -> str:
    """Serialise a configuration; YAML and TOML start with a comment header."""
    chosen = ConfigFormat(fmt)
    data = to_data(config)
    if chosen is ConfigFormat.JSON:
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    header = "".join(f"# {line}\n" for line in _header(config)) + "\n"
    if chosen is ConfigFormat.TOML:
        return header + tomli_w.dumps(_without_none(data))
    return header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _describe(error: ValidationError, source: str) -> str:
    lines = [f"{source}: invalid configuration"]
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "(root)"
        lines.append(f"  {location}: {item['msg']}")
    return "\n".join(lines)


def from_data(data: object, *, source: str = "configuration") -> ExperimentConfig:
    """Validate plain data (e.g. parsed YAML) into a configuration.

    Raises:
        ConfigError: The data do not describe a valid configuration.
    """
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: expected a mapping at the top level.")
    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as error:
        raise ConfigError(_describe(error, source)) from error


def loads(text: str, fmt: ConfigFormat | str, *, source: str = "configuration") -> ExperimentConfig:
    """Parse and validate a serialised configuration."""
    chosen = ConfigFormat(fmt)
    try:
        if chosen is ConfigFormat.YAML:
            data: object = yaml.safe_load(text)
        elif chosen is ConfigFormat.TOML:
            data = tomllib.loads(text)
        else:
            data = json.loads(text)
    except (yaml.YAMLError, tomllib.TOMLDecodeError, json.JSONDecodeError) as error:
        raise ConfigError(f"{source}: not valid {chosen.value.upper()}: {error}") from error
    return from_data(data, source=source)


def load_config(path: Path) -> ExperimentConfig:
    """Read a configuration file (format from the extension)."""
    fmt = format_for(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"{path}: cannot read the file ({error.strerror}).") from error
    return loads(text, fmt, source=path.name)


def save_config(config: ExperimentConfig, path: Path, *, overwrite: bool = False) -> Path:
    """Write a configuration file (format from the extension).

    Raises:
        ConfigError: The file exists and ``overwrite`` is false.
    """
    fmt = format_for(path)
    if path.exists() and not overwrite:
        raise ConfigError(f"{path} already exists (overwrite it explicitly).")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(config, fmt), encoding="utf-8", newline="\n")
    return path
