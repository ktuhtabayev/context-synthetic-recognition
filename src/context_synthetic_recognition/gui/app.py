"""Starting the desktop application: ``csr gui`` and the ``csr-gui`` script."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from context_synthetic_recognition.gui.i18n import install_translator
from context_synthetic_recognition.gui.settings import APPLICATION, ORGANIZATION, Settings
from context_synthetic_recognition.gui.window import APP_TITLE, MainWindow


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """The Qt application (the running one, if there is one)."""
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    app = QApplication(list(argv) if argv is not None else sys.argv[:1])
    app.setOrganizationName(ORGANIZATION)
    app.setApplicationName(APPLICATION)
    app.setApplicationDisplayName(APP_TITLE)
    app.setStyle("Fusion")  # the same look on every platform, coloured by the theme
    return app


def create_window(
    *,
    dataset: str | None = None,
    config: Path | None = None,
    run: Path | None = None,
    store: QSettings | None = None,
) -> MainWindow:
    """Create the main window and open what the command line named.

    Args:
        dataset: A dataset file, or the id or name of a dataset of the project.
        config: A configuration file (its dataset is loaded too, if it names one).
        run: A run folder to repeat and show.
        store: Settings store (default: the user's).
    """
    window = MainWindow(Settings(store))
    if config is not None:
        window.open_config(config)
    if dataset is not None:
        window.dataset_page.load(dataset)
    if run is not None:
        window.open_run(run)
    elif window.state.dataset is None and config is None:
        window.dataset_page.load("default")  # the project's default dataset (ADR-035)
    return window


def launch(
    *, dataset: str | None = None, config: Path | None = None, run: Path | None = None
) -> int:
    """Run the application until its window is closed; returns the exit code."""
    app = create_application()
    translator = install_translator(app)
    window = create_window(dataset=dataset, config=config, run=run)
    window.show()
    code = app.exec()
    del translator
    return int(code)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of the ``csr-gui`` script."""
    parser = argparse.ArgumentParser(
        prog="csr-gui", description="Desktop application of the context-synthetic model."
    )
    parser.add_argument("dataset", nargs="?", help="dataset file, or a dataset of the project")
    parser.add_argument("-c", "--config", type=Path, help="configuration file")
    parser.add_argument("-r", "--run", type=Path, help="run folder to open")
    arguments = parser.parse_args(argv)
    sys.exit(launch(dataset=arguments.dataset, config=arguments.config, run=arguments.run))
