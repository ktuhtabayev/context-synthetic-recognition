"""The design system: one set of tokens per theme for the style sheet, the tables and the figures.

A :class:`Palette` holds the colours of the application chrome and of the table cells; the cell
colours carry the workbook's meaning (:class:`CellRole`) — quantitative and nominal features,
synthetic and latent features, the features of TUPLAM, correct and wrong decisions, warnings.
The classes K1 and K2 have the two colours they have in every figure
(:mod:`context_synthetic_recognition.export.theme`), in a tint that keeps the text readable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from PySide6.QtGui import QColor, QPalette

from context_synthetic_recognition.export.theme import DARK as FIGURES_DARK
from context_synthetic_recognition.export.theme import LIGHT as FIGURES_LIGHT
from context_synthetic_recognition.export.theme import Theme as FigureTheme

ASSETS = Path(__file__).parent / "assets"
"""Images the style sheet refers to."""
BASE_FONT_PT = 10.0
"""Font size at scale 1.0 (points)."""
FONT_SCALES = (0.85, 1.0, 1.15, 1.3, 1.5, 1.75)
"""The steps of View → Zoom."""


class CellRole(StrEnum):
    """What a table cell means; decides its colours (the workbook's colour semantics)."""

    QUANTITATIVE = "quantitative"
    """A quantitative feature (set I) — the workbook's blue."""
    NOMINAL = "nominal"
    """A nominal feature (set J) — the workbook's yellow."""
    K1 = "k1"
    K2 = "k2"
    REFUSAL = "refusal"
    """The decision 0."""
    SYNTHETIC = "synthetic"
    """A synthetic feature aᵤ — the workbook's light green."""
    LATENT = "latent"
    """A latent feature r_j — the workbook's stronger green."""
    SELECTED = "selected"
    """A feature of TUPLAM, or the chosen candidate q."""
    GOOD = "good"
    BAD = "bad"
    WARNING = "warning"
    MUTED = "muted"
    """An undefined value."""


@dataclass(frozen=True)
class Palette:
    """The colours of one theme."""

    name: str
    window: str
    surface: str
    surface_alt: str
    """Alternating table rows, the sidebar."""
    border: str
    border_strong: str
    text: str
    text_muted: str
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_soft: str
    accent_text: str
    """Text on the accent colour."""
    header: str
    """Table headers."""
    hover: str
    disabled: str
    disabled_text: str
    highlight: str
    """Rows of the object selected in another table (cross-highlighting)."""
    warning_bg: str
    warning_border: str
    warning_text: str
    error_text: str
    cells: dict[CellRole, tuple[str, str]]
    """Cell role → (background, text)."""
    figures: FigureTheme
    """The matching theme of the matplotlib figures."""


LIGHT = Palette(
    name="light",
    window="#f5f6f8",
    surface="#ffffff",
    surface_alt="#f3f5f8",
    border="#d0d4da",
    border_strong="#b6bcc6",
    text="#1f2430",
    text_muted="#5b6472",
    accent="#2563eb",
    accent_hover="#1d4ed8",
    accent_pressed="#1e40af",
    accent_soft="#dbeafe",
    accent_text="#ffffff",
    header="#e8ebf0",
    hover="#eef2f7",
    disabled="#e9ecf0",
    disabled_text="#9aa2ae",
    highlight="#fff3bf",
    warning_bg="#fff4e5",
    warning_border="#f0a030",
    warning_text="#7a4100",
    error_text="#b3261e",
    cells={
        CellRole.QUANTITATIVE: ("#ddebf7", "#1f2430"),
        CellRole.NOMINAL: ("#fff2cc", "#1f2430"),
        CellRole.K1: ("#d6e6fa", "#123f7a"),
        CellRole.K2: ("#fbe0d3", "#8a3410"),
        CellRole.REFUSAL: ("#ececec", "#5b6472"),
        CellRole.SYNTHETIC: ("#e2efda", "#1f2430"),
        CellRole.LATENT: ("#b7dea8", "#1f2430"),
        CellRole.SELECTED: ("#ffe699", "#1f2430"),
        CellRole.GOOD: ("#c6efce", "#006100"),
        CellRole.BAD: ("#ffc7ce", "#9c0006"),
        CellRole.WARNING: ("#fce4d6", "#7a4100"),
        CellRole.MUTED: ("#ffffff", "#9aa2ae"),
    },
    figures=FIGURES_LIGHT,
)

DARK = Palette(
    name="dark",
    window="#1a1a19",
    surface="#232322",
    surface_alt="#2a2a28",
    border="#3a3a37",
    border_strong="#52514e",
    text="#f1f0ec",
    text_muted="#b0aea6",
    accent="#3987e5",
    accent_hover="#5a9cec",
    accent_pressed="#256abf",
    accent_soft="#1d3557",
    accent_text="#ffffff",
    header="#2f2f2d",
    hover="#323230",
    disabled="#2a2a28",
    disabled_text="#75736c",
    highlight="#5c4a00",
    warning_bg="#3d2a10",
    warning_border="#c98500",
    warning_text="#ffd08a",
    error_text="#ff8a80",
    cells={
        CellRole.QUANTITATIVE: ("#1f3a57", "#f1f0ec"),
        CellRole.NOMINAL: ("#4d4320", "#f1f0ec"),
        CellRole.K1: ("#1d3557", "#a9cdf7"),
        CellRole.K2: ("#55291a", "#f7b99d"),
        CellRole.REFUSAL: ("#3a3a37", "#b0aea6"),
        CellRole.SYNTHETIC: ("#26402a", "#f1f0ec"),
        CellRole.LATENT: ("#2f5c33", "#f1f0ec"),
        CellRole.SELECTED: ("#6b5300", "#fff3c4"),
        CellRole.GOOD: ("#1e4d2b", "#a8e6b4"),
        CellRole.BAD: ("#5c1f24", "#ffb3ba"),
        CellRole.WARNING: ("#4d2f14", "#ffd08a"),
        CellRole.MUTED: ("#232322", "#75736c"),
    },
    figures=FIGURES_DARK,
)

PALETTES = {"light": LIGHT, "dark": DARK}


def palette(name: str) -> Palette:
    """The palette of a theme; an unknown name gives the light one."""
    return PALETTES.get(name, LIGHT)


def qt_palette(colours: Palette) -> QPalette:
    """The palette for Qt's own drawing (check boxes, spin buttons, scroll bars of the style)."""
    roles = QPalette.ColorRole
    found = QPalette()
    for role, colour in (
        (roles.Window, colours.window),
        (roles.WindowText, colours.text),
        (roles.Base, colours.surface),
        (roles.AlternateBase, colours.surface_alt),
        (roles.Text, colours.text),
        (roles.Button, colours.surface),
        (roles.ButtonText, colours.text),
        (roles.ToolTipBase, colours.surface),
        (roles.ToolTipText, colours.text),
        (roles.PlaceholderText, colours.disabled_text),
        (roles.Highlight, colours.accent),
        (roles.HighlightedText, colours.accent_text),
        (roles.Link, colours.accent),
        (roles.Mid, colours.border),
        (roles.Dark, colours.border_strong),
        (roles.Light, colours.surface),
    ):
        found.setColor(role, QColor(colour))
    for role in (roles.Text, roles.WindowText, roles.ButtonText):
        found.setColor(QPalette.ColorGroup.Disabled, role, QColor(colours.disabled_text))
    return found


def font_points(scale: float) -> float:
    """The base font size at a zoom step."""
    return round(BASE_FONT_PT * scale, 2)


def stylesheet(colours: Palette, scale: float = 1.0) -> str:
    """The application style sheet for a palette and a font scale."""
    c = colours
    size = font_points(scale)
    small = round(size * 0.9, 2)
    large = round(size * 1.35, 2)
    check = (ASSETS / "check.svg").as_posix()
    return f"""
QWidget {{
    background-color: {c.window};
    color: {c.text};
    font-size: {size}pt;
}}
QLabel, QCheckBox, QRadioButton {{ background: transparent; }}
QToolTip {{
    background-color: {c.surface};
    color: {c.text};
    border: 1px solid {c.border_strong};
    padding: 4px 6px;
}}

/* ---- navigation */
QListWidget#sidebar {{
    background-color: {c.surface_alt};
    border: none;
    border-right: 1px solid {c.border};
    outline: 0;
    padding: 8px 0;
}}
QListWidget#sidebar::item {{
    padding: 9px 16px;
    border-left: 3px solid transparent;
}}
QListWidget#sidebar::item:hover {{ background-color: {c.hover}; }}
QListWidget#sidebar::item:selected {{
    background-color: {c.accent_soft};
    color: {c.text};
    border-left: 3px solid {c.accent};
    font-weight: 600;
}}
QListWidget#sidebar::item:disabled {{ color: {c.disabled_text}; }}

/* ---- text */
QLabel#pageTitle {{ font-size: {large}pt; font-weight: 600; }}
QLabel#pageHint, QLabel#note {{ color: {c.text_muted}; font-size: {small}pt; }}
QLabel#sectionTitle {{ font-weight: 600; }}
QLabel#errorText {{ color: {c.error_text}; }}
QLabel#badge {{
    background-color: {c.accent_soft};
    border: 1px solid {c.accent};
    border-radius: 9px;
    padding: 1px 9px;
    font-size: {small}pt;
}}
QLabel#warningBadge, QPushButton#warningBadge {{
    background-color: {c.warning_bg};
    color: {c.warning_text};
    border: 1px solid {c.warning_border};
    border-radius: 9px;
    padding: 1px 9px;
    font-size: {small}pt;
    font-weight: 600;
}}
QFrame#warningBox {{
    background-color: {c.warning_bg};
    border: 1px solid {c.warning_border};
    border-radius: 6px;
}}
QFrame#warningBox QLabel {{ color: {c.warning_text}; }}
QLabel#decision {{
    background-color: {c.accent_soft};
    border: 1px solid {c.accent};
    border-radius: 6px;
    padding: 8px 14px;
    font-size: {large}pt;
    font-weight: 600;
}}

/* ---- cards */
QFrame#card, QGroupBox {{
    background-color: {c.surface};
    border: 1px solid {c.border};
    border-radius: 8px;
}}
QGroupBox {{ margin-top: 12px; padding: 12px 10px 10px 10px; font-weight: 600; }}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    background-color: {c.window};
}}
QFrame#card QWidget, QGroupBox QWidget {{ background-color: transparent; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ border: none; }}

/* ---- inputs */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit,
QFrame#card QLineEdit, QFrame#card QComboBox, QFrame#card QSpinBox, QFrame#card QDoubleSpinBox,
QGroupBox QLineEdit, QGroupBox QComboBox, QGroupBox QSpinBox, QGroupBox QDoubleSpinBox,
QGroupBox QPlainTextEdit {{
    background-color: {c.surface};
    border: 1px solid {c.border};
    border-radius: 5px;
    padding: 3px 6px;
    selection-background-color: {c.accent};
    selection-color: {c.accent_text};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {{
    border: 1px solid {c.accent};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background-color: {c.disabled};
    color: {c.disabled_text};
}}
QLineEdit[invalid="true"], QPlainTextEdit[invalid="true"] {{ border: 1px solid {c.error_text}; }}
QComboBox QAbstractItemView {{
    background-color: {c.surface};
    border: 1px solid {c.border_strong};
    selection-background-color: {c.accent_soft};
    selection-color: {c.text};
}}

QCheckBox::indicator, QGroupBox::indicator, QTableView::indicator, QTableWidget::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {c.border_strong};
    border-radius: 3px;
    background-color: {c.surface};
}}
QCheckBox::indicator:hover, QGroupBox::indicator:hover {{ border: 1px solid {c.accent}; }}
QCheckBox::indicator:checked, QGroupBox::indicator:checked, QTableView::indicator:checked,
QTableWidget::indicator:checked {{
    background-color: {c.accent};
    border: 1px solid {c.accent};
    image: url({check});
}}
QCheckBox::indicator:disabled, QGroupBox::indicator:disabled {{
    background-color: {c.disabled};
    border: 1px solid {c.border};
}}
QCheckBox {{ spacing: 7px; }}

/* ---- buttons */
QPushButton, QFrame#card QPushButton, QGroupBox QPushButton {{
    background-color: {c.surface};
    border: 1px solid {c.border_strong};
    border-radius: 5px;
    padding: 5px 14px;
}}
QPushButton:hover {{ background-color: {c.hover}; }}
QPushButton:pressed {{ background-color: {c.header}; }}
QPushButton:disabled {{ background-color: {c.disabled}; color: {c.disabled_text}; }}
QPushButton#primary, QFrame#card QPushButton#primary, QGroupBox QPushButton#primary {{
    background-color: {c.accent};
    color: {c.accent_text};
    border: 1px solid {c.accent};
    font-weight: 600;
}}
QPushButton#primary:hover {{ background-color: {c.accent_hover}; }}
QPushButton#primary:pressed {{ background-color: {c.accent_pressed}; }}
QPushButton#primary:disabled {{
    background-color: {c.disabled};
    color: {c.disabled_text};
    border: 1px solid {c.border};
}}

/* ---- tables and trees */
QTableView, QTreeWidget, QListWidget, QFrame#card QTableView, QGroupBox QTableView,
QGroupBox QListWidget, QFrame#card QListWidget {{
    background-color: {c.surface};
    alternate-background-color: {c.surface_alt};
    border: 1px solid {c.border};
    gridline-color: {c.border};
    selection-background-color: {c.accent_soft};
    selection-color: {c.text};
}}
QHeaderView::section {{
    background-color: {c.header};
    color: {c.text};
    border: none;
    border-right: 1px solid {c.border};
    border-bottom: 1px solid {c.border};
    padding: 4px 8px;
    font-weight: 600;
}}
QTableCornerButton::section {{ background-color: {c.header}; border: none; }}
QTreeWidget::item {{ padding: 3px 2px; }}
QTreeWidget::item:selected, QListWidget::item:selected {{
    background-color: {c.accent_soft};
    color: {c.text};
}}

/* ---- tabs */
QTabWidget::pane {{ border: 1px solid {c.border}; border-radius: 4px; top: -1px; }}
QTabBar::tab {{
    background-color: transparent;
    color: {c.text_muted};
    padding: 6px 14px;
    border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:hover {{ color: {c.text}; }}
QTabBar::tab:selected {{ color: {c.text}; border-bottom: 2px solid {c.accent}; font-weight: 600; }}

/* ---- the rest */
QProgressBar {{
    background-color: {c.surface_alt};
    border: 1px solid {c.border};
    border-radius: 5px;
    text-align: center;
    min-height: 16px;
}}
QProgressBar::chunk {{ background-color: {c.accent}; border-radius: 4px; }}
QPlainTextEdit#console {{ font-family: Consolas, "DejaVu Sans Mono", monospace; }}
QMenuBar {{ background-color: {c.window}; }}
QMenuBar::item:selected {{ background-color: {c.hover}; }}
QMenu {{ background-color: {c.surface}; border: 1px solid {c.border_strong}; padding: 4px; }}
QMenu::item {{ padding: 5px 22px; }}
QMenu::item:selected {{ background-color: {c.accent_soft}; color: {c.text}; }}
QMenu::item:disabled {{ color: {c.disabled_text}; }}
QMenu::separator {{ height: 1px; background: {c.border}; margin: 4px 8px; }}
QStatusBar {{ background-color: {c.surface_alt}; border-top: 1px solid {c.border}; }}
QStatusBar QLabel {{ padding: 0 6px; }}
QSplitter::handle {{ background-color: {c.border}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QScrollBar:vertical {{ background: {c.window}; width: 12px; margin: 0; }}
QScrollBar:horizontal {{ background: {c.window}; height: 12px; margin: 0; }}
QScrollBar::handle {{ background: {c.border_strong}; border-radius: 5px; min-height: 24px;
    min-width: 24px; margin: 2px; }}
QScrollBar::handle:hover {{ background: {c.text_muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
"""
