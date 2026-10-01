"""The Excel mirror: a workbook with the sheet structure and style of the author's experiment."""

from context_synthetic_recognition.export.excel.common import ExcelOptions
from context_synthetic_recognition.export.excel.workbook import build_workbook, write_workbook

__all__ = ["ExcelOptions", "build_workbook", "write_workbook"]
