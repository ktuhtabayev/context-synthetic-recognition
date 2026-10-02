r"""Make the screenshots of the GUI user guide (docs/images/gui).

Run from the repository root::

    .\.venv\Scripts\python.exe scripts\gui_screenshots.py
    .\.venv\Scripts\python.exe scripts\gui_screenshots.py --target somewhere-else

The application runs the experiment on Heart-Disease (10, 13, 2) for real — the template preset,
then the article preset — and every page is grabbed at 1360 × 860; the window is then rebuilt in
Russian and in Uzbek for the two language images. Nothing appears on screen: the window is laid
out by the native Windows platform (for the system fonts) but never mapped.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"  # the same pixels on every screen

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

from context_synthetic_recognition.gui.app import (
    create_application,
    create_window,
    rebuild_window,
)
from context_synthetic_recognition.gui.window import MainWindow

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs" / "images" / "gui"
SIZE = (1360, 860)
NEW_OBJECT = "58 1 3 125 250 0 2 150 0 1 2 1 7"


def pump(app: QApplication, seconds: float = 0.25) -> None:
    """Let the event loop work for a moment (layouts, queued signals, repaints)."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def run(app: QApplication, window: MainWindow) -> None:
    """Run the experiment and wait for the result."""
    window.run_page.start()
    while window.run_page.is_running():
        pump(app, 0.05)
    pump(app)


def main() -> None:
    """Drive the application through the workflow and save one image per page."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--target", type=Path, default=TARGET, help="folder for the images")
    target = parser.parse_args().target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    app = create_application([])
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        # the relative folders "runs" and "exports" stay out of the repository — and out of the
        # images: no path of this machine is shown
        os.chdir(work)
        store = QSettings(str(work / "settings.ini"), QSettings.Format.IniFormat)
        window = create_window(dataset="heart-disease-10", store=store)
        window.resize(*SIZE)
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        window.show()

        def shot(name: str) -> None:
            pump(app)
            window.grab().save(str(target / f"{name}.png"))
            print("saved", name)  # noqa: T201 - a script

        shot("dataset")
        window.go_to(window.configure_page)
        shot("configure")
        window.go_to(window.run_page)
        run(app, window)
        shot("run")
        results = window.results_page
        window.go_to(results)
        window.state.select_object(2)
        results.open("table", "psi")
        shot("results-table")
        results.open("table", "predictions-leave-one-out")
        shot("results-predictions")
        results.open("figure", "margins")
        shot("results-figure")
        results.open("context", "context")
        shot("results-neighbourhood")
        new_object = window.new_object_page
        window.go_to(new_object)
        new_object.paste.setText(NEW_OBJECT)
        new_object.fill_from_text()
        new_object.classify()
        new_object.tabs.setCurrentIndex(2)
        shot("new-object")
        # a second run with the article's switches, to compare
        window.configure_page.apply_preset("article")
        window.go_to(window.run_page)
        run(app, window)
        compare = window.compare_page
        window.go_to(compare)
        compare.set_checked(0)
        compare.set_checked(1)
        shot("compare")
        window.go_to(window.export_page)
        shot("export")
        window.set_theme("dark")
        window.go_to(results)
        results.open("table", "hag-candidates")
        shot("dark-results")
        # the same work in the other languages of the interface (View → Language)
        window.set_theme("light")
        for code, page in (("ru", 1), ("uz", 0)):
            window.settings.language = code
            window = rebuild_window(window)
            window.go_to(window.pages[page])
            shot(f"language-{code}")
        window.close()
        pump(app)
        os.chdir(ROOT)


if __name__ == "__main__":
    main()
