"""Packaging: version, typing marker, module entry point."""

import subprocess
import sys
import tomllib

import context_synthetic_recognition as csr

from .conftest import ROOT


def test_version_matches_pyproject() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert csr.__version__ == project["version"]


def test_package_is_typed() -> None:
    assert (ROOT / "src" / "context_synthetic_recognition" / "py.typed").is_file()


def test_python_dash_m_runs_the_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "context_synthetic_recognition", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == f"context-synthetic-recognition {csr.__version__}"


def test_importing_the_package_configures_no_output_handlers() -> None:
    import logging

    handlers = logging.getLogger("context_synthetic_recognition").handlers
    assert all(isinstance(handler, logging.NullHandler) for handler in handlers[:1])
