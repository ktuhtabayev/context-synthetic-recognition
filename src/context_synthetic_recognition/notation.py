"""The article's notation for display: S₁, x₁, a₁, r₁ (1-based, subscript digits).

Code is 0-based internally; everything shown to a user (tables, messages, exports, the GUI) uses
these 1-based names, as the workbook and the article do (ADR-017).
"""

from __future__ import annotations

_SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

DASH = "—"
"""Shown for an undefined value (e.g. f_k(μ) of a gradation no object has), as in the workbook."""


def subscript(number: int) -> str:
    """``12`` → ``₁₂``."""
    return str(number).translate(_SUBSCRIPT_DIGITS)


def object_name(index: int) -> str:
    """Name of the object with 0-based ``index``: ``S₁`` for 0."""
    return f"S{subscript(index + 1)}"


def feature_name(index: int) -> str:
    """Name of the original feature with 0-based ``index``: ``x₁`` for 0."""
    return f"x{subscript(index + 1)}"


def synthetic_name(index: int) -> str:
    """Name of the synthetic feature with 0-based ``index``: ``a₁`` for 0 (formula (5))."""
    return f"a{subscript(index + 1)}"


def latent_name(index: int) -> str:
    """Name of the latent (additional) feature with 0-based ``index``: ``r₁`` for 0 (HAG)."""
    return f"r{subscript(index + 1)}"


def class_name(index: int) -> str:
    """Name of the class with 0-based ``index``: ``K1`` for 0 (the article writes K1, K2)."""
    return f"K{index + 1}"


def join_values(values: tuple[int, ...] | list[int]) -> str:
    """``(3, 5)`` → ``3, 5`` — the workbook's way of listing the permitted k."""
    return ", ".join(str(value) for value in values)
