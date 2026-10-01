"""Desktop application (PySide6).

Workflow: Dataset → Configure → Run → Results → New object → Compare → Export. The GUI is one
more interface above the services — it computes nothing itself. A run is
:func:`~context_synthetic_recognition.services.runner.run_experiment` in a worker thread; what the
pages show is the :class:`~context_synthetic_recognition.export.view.RunView` the exporters read,
its tables (:mod:`context_synthetic_recognition.export.tables`) and its figures
(:mod:`context_synthetic_recognition.export.figures`), so the screen, the Excel mirror and the
article's tables cannot disagree.

PySide6 is the optional extra ``[gui]``; nothing here is imported by the library or the CLI unless
the application is started (``csr gui`` or ``csr-gui``).
"""

from __future__ import annotations

GUI_EXTRA_HINT = (
    'the desktop application needs PySide6: pip install "context-synthetic-recognition[gui]"'
)
"""Shown when the application is started without PySide6."""

__all__ = ["GUI_EXTRA_HINT"]
