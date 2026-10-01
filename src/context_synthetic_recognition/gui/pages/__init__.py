"""The pages of the workflow, in the order of the sidebar."""

from __future__ import annotations

from context_synthetic_recognition.gui.pages.base import Page
from context_synthetic_recognition.gui.pages.compare import ComparePage
from context_synthetic_recognition.gui.pages.configure import ConfigurePage
from context_synthetic_recognition.gui.pages.dataset import DatasetPage
from context_synthetic_recognition.gui.pages.export import ExportPage
from context_synthetic_recognition.gui.pages.new_object import NewObjectPage
from context_synthetic_recognition.gui.pages.results import ResultsPage
from context_synthetic_recognition.gui.pages.run import RunPage

__all__ = [
    "ComparePage",
    "ConfigurePage",
    "DatasetPage",
    "ExportPage",
    "NewObjectPage",
    "Page",
    "ResultsPage",
    "RunPage",
]
