"""Starting the desktop application: ``csr gui`` and the ``csr-gui`` script."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
from collections.abc import Sequence
from importlib.util import find_spec
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

from context_synthetic_recognition import __version__
from context_synthetic_recognition.export.run import EXPORTERS
from context_synthetic_recognition.gui.i18n import LANGUAGES, SOURCE_LANGUAGE, set_language, tr
from context_synthetic_recognition.gui.settings import APPLICATION, ORGANIZATION, Settings
from context_synthetic_recognition.gui.window import APP_TITLE, MainWindow
from context_synthetic_recognition.gui.workers import export_job


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


def rebuild_window(window: MainWindow) -> MainWindow:
    """Replace ``window`` by one in the language of the settings, working on the same state.

    The texts of a window are set when it is built, so a change of language builds the window
    again. Its dataset, configuration, undo history, run and selected object are handed over; the
    new window opens on the same page.
    """
    app = create_application()
    set_language(app, window.settings.language)
    page = window.stack.currentIndex()
    shown = window.isVisible()
    replacement = MainWindow(window.settings, state=window.release_state())
    replacement.go_to(replacement.pages[page])
    unseen = Qt.WidgetAttribute.WA_DontShowOnScreen  # screenshots, the self-test
    replacement.setAttribute(unseen, window.testAttribute(unseen))
    if shown:
        replacement.show()
    window.close()
    window.deleteLater()
    return replacement


def launch(
    *, dataset: str | None = None, config: Path | None = None, run: Path | None = None
) -> int:
    """Run the application until its window is closed; returns the exit code."""
    app = create_application()
    set_language(app, Settings().language)
    windows = [create_window(dataset=dataset, config=config, run=run)]

    def follow(window: MainWindow) -> None:
        def switch(_code: str) -> None:
            windows[0] = rebuild_window(window)
            follow(windows[0])

        window.replaced_on_language_change = True
        window.languageChanged.connect(switch)

    follow(windows[0])
    windows[0].show()
    return int(app.exec())


SELF_TEST_SECONDS = 300.0
"""How long the self-test waits for the experiment."""


def _self_test(app: QApplication, lines: list[str]) -> bool:
    """The checks of :func:`self_test`; appends what it found to ``lines``."""
    passed = True

    def check(what: str, ok: bool) -> None:
        nonlocal passed
        passed = passed and ok
        lines.append(f"{'ok  ' if ok else 'FAIL'}  {what}")

    for code, name in LANGUAGES.items():
        set_language(app, code)
        text = tr("Run the experiment")
        check(
            f"language {name}: {text}", (text == "Run the experiment") == (code == SOURCE_LANGUAGE)
        )
    set_language(app, SOURCE_LANGUAGE)
    with tempfile.TemporaryDirectory() as scratch:
        store = QSettings(str(Path(scratch) / "settings.ini"), QSettings.Format.IniFormat)
        window = create_window(store=store)
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        window.show()  # laid out by the platform, never mapped on screen
        dataset = window.state.dataset
        check(f"default dataset: {dataset.name if dataset else 'none'}", dataset is not None)
        ended: list[bool] = []
        window.run_page.finished.connect(lambda *_: ended.append(True))
        window.run_page.save.setChecked(False)
        window.run_page.start()
        deadline = time.monotonic() + SELF_TEST_SECONDS
        while not ended and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        view = window.state.view
        check(f"experiment: TUPLAM = {view.hag.label if view else 'no result'}", view is not None)
        if view is not None:
            results = window.results_page
            window.go_to(results)
            opened = results.open("table", "psi") and results.open("figure", "margins")
            app.processEvents()
            check("results: a table and a figure", opened)
            # the PDF report needs an optional package; the other formats are always there
            formats = [f for f in EXPORTERS.names() if f != "pdf" or find_spec("reportlab")]
            written = export_job(
                view, Path(scratch) / "export", formats, window.export_page.options()
            )
            missing = [f for f in formats if not written.files.get(f)]
            check(f"export: {', '.join(formats)} ({written.count} files)", not missing)
        window.close()
        app.processEvents()
    return passed


def self_test(report: Path | None = None) -> int:
    """Check an installation of the application without showing it.

    The window is built by the platform but never mapped on screen: every language is loaded,
    the default dataset is opened, the experiment is run in the worker thread and a table and a
    figure of the results are drawn. The user's settings are not touched and nothing is written
    besides the report.

    Args:
        report: A file to write what was checked to (a windowed program has no console).

    Returns:
        0 if everything works, 1 otherwise.
    """
    lines = [f"context-synthetic-recognition {__version__} — self-test"]
    try:
        passed = _self_test(create_application(), lines)
    except Exception:  # a report instead of a crash dialog
        lines.append(traceback.format_exc())
        passed = False
    lines.append("passed" if passed else "FAILED")
    if report is not None:
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if passed else 1


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of the ``csr-gui`` script."""
    parser = argparse.ArgumentParser(
        prog="csr-gui", description="Desktop application of the context-synthetic model."
    )
    parser.add_argument("dataset", nargs="?", help="dataset file, or a dataset of the project")
    parser.add_argument("-c", "--config", type=Path, help="configuration file")
    parser.add_argument("-r", "--run", type=Path, help="run folder to open")
    parser.add_argument(
        "--self-test",
        nargs="?",
        const="",
        metavar="REPORT",
        help="check the installation without showing a window (exit code 0 = it works); "
        "REPORT: a file to write the result to",
    )
    arguments = parser.parse_args(argv)
    if arguments.self_test is not None:
        sys.exit(self_test(Path(arguments.self_test) if arguments.self_test else None))
    sys.exit(launch(dataset=arguments.dataset, config=arguments.config, run=arguments.run))
