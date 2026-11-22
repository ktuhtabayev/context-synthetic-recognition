r"""Build the Windows executable of the desktop application with PyInstaller.

Run from the repository root::

    .\.venv\Scripts\python.exe -m pip install -e ".[gui,pdf,release,sklearn]"
    .\.venv\Scripts\python.exe scripts\build_exe.py

The result is ``dist\csr-gui\`` — a folder with ``csr-gui.exe`` and everything it needs, to be
copied as a whole — and ``dist\csr-gui-<version>-windows.zip``, the same folder packed. The
script then runs ``csr-gui.exe --self-test``: the frozen application loads every language, opens
the default dataset, runs the experiment, draws a table and a figure, exports the run and
computes Tables 7 and 8; the report is ``dist\self-test.txt`` (``--no-test`` skips this).

The executable contains Python, numpy, matplotlib, Qt, scikit-learn and the package with its
datasets, translations and a copy of the article's split files (``datasets/splits``), so the
*Classifiers* page works there (ADR-081). The optional extra for Parquet files is left out; the
application says so when it is asked for.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
WORK = ROOT / "build" / "pyinstaller"
NAME = "csr-gui"
PACKAGE = "context_synthetic_recognition"
DISTRIBUTION = "context-synthetic-recognition"
EXCLUDED = (
    # an optional extra that would multiply the size, and tools of the development environment
    "pyarrow",
    "pytest",
    "hypothesis",
    "IPython",
    "tkinter",
    "mypy",
)
HIDDEN = (
    # figures are saved through backends matplotlib picks by file type at run time
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_svg",
    "matplotlib.backends.backend_pdf",
    # the scikit-learn baselines import their estimators by name at run time (ADR-081)
    "sklearn.linear_model",
    "sklearn.ensemble",
    "sklearn.svm",
    "sklearn.tree",
    "sklearn.naive_bayes",
)
COLLECTED = ("reportlab",)
"""Packages imported by name at run time (the PDF report): taken whole, with their data."""
SPLITS = ROOT / "datasets" / "splits"
"""The article's split files, carried in the package's data (ADR-081)."""
BUNDLED_SPLITS = f"{PACKAGE}/data/split_files"
"""Where the build puts them (``services.comparison.BUNDLED_DIR``)."""
REQUIRED = ("sklearn",)
"""Optional extras the build must contain (the Classifiers page, ADR-081)."""
LANGUAGES = ("en", "ru", "uz")
"""The languages of the interface: Qt's own translations are kept for these."""


def say(text: str) -> None:
    """Print a line."""
    print(text, flush=True)  # noqa: T201 - a script


def build() -> Path:
    """Freeze the application; returns the folder with the executable."""
    import PyInstaller.__main__

    missing = [module for module in REQUIRED if find_spec(module) is None]
    if missing or not SPLITS.is_dir():
        raise SystemExit(
            f"the build needs {', '.join(missing) or SPLITS}: "
            'pip install -e ".[gui,pdf,release,sklearn]" in the project folder'
        )
    arguments = [
        str(ROOT / "scripts" / "csr_gui_launcher.py"),
        "--name",
        NAME,
        "--windowed",
        "--noconfirm",
        "--clean",
        "--paths",
        str(ROOT / "src"),
        "--collect-data",
        PACKAGE,
        "--copy-metadata",
        DISTRIBUTION,
        "--add-data",
        f"{SPLITS}{os.pathsep}{BUNDLED_SPLITS}",
        "--distpath",
        str(DIST),
        "--workpath",
        str(WORK),
        "--specpath",
        str(WORK),
    ]
    for module in EXCLUDED:
        arguments += ["--exclude-module", module]
    for module in HIDDEN:
        arguments += ["--hidden-import", module]
    for package in COLLECTED:
        arguments += ["--collect-submodules", package, "--collect-data", package]
    PyInstaller.__main__.run(arguments)
    return DIST / NAME


def prune(folder: Path) -> int:
    """Remove what the application never loads; returns the bytes freed.

    Qt's software OpenGL renderer (the interface is drawn without OpenGL) and Qt's translations
    of languages the interface does not have.
    """
    qt = folder / "_internal" / "PySide6"
    kept = tuple(f"_{code}.qm" for code in LANGUAGES)
    unused = [qt / "opengl32sw.dll"]
    unused += [f for f in (qt / "translations").glob("*.qm") if not f.name.endswith(kept)]
    freed = 0
    for path in unused:
        if path.is_file():
            freed += path.stat().st_size
            path.unlink()
    return freed


def self_test(folder: Path) -> bool:
    """Run the frozen application's self-test; the report is ``dist/self-test.txt``.

    It runs in an empty folder, away from the project's ``datasets``, as it will on a machine
    that has only the executable.
    """
    report = DIST / "self-test.txt"
    report.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as elsewhere:
        result = subprocess.run(
            [str(folder / f"{NAME}.exe"), "--self-test", str(report)],
            check=False,
            timeout=900,
            cwd=elsewhere,
        )
    if report.is_file():
        say(report.read_text(encoding="utf-8").rstrip())
    return result.returncode == 0 and report.is_file()


def pack(folder: Path) -> Path:
    """Pack the folder as ``csr-gui-<version>-windows.zip`` next to it."""
    archive = shutil.make_archive(
        str(DIST / f"{NAME}-{version(DISTRIBUTION)}-windows"), "zip", root_dir=DIST, base_dir=NAME
    )
    return Path(archive)


def main(argv: list[str] | None = None) -> int:
    """Build, test and pack the executable."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--no-test", action="store_true", help="do not run the self-test")
    arguments = parser.parse_args(argv)
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:  # the report holds characters a Windows code page cannot encode
        reconfigure(encoding="utf-8", errors="replace")
    folder = build()
    say(f"removed {prune(folder) / 1e6:.0f} MB the application never loads")
    if not arguments.no_test and not self_test(folder):
        say("the self-test of the frozen application FAILED")
        return 1
    archive = pack(folder)
    size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e6
    say(f"{folder}  ({size:.0f} MB)")
    say(f"{archive}  ({archive.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
