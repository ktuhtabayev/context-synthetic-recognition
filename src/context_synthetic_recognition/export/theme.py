"""Colours of the figures and the report — one source for the exporters and, later, the GUI.

The palette is a validated colour-blind-safe set (adjacent and all-pairs CVD separation checked
for the first three categorical slots in both modes): the classes K1 and K2 take the first two
slots in every figure, so a class always has the same colour; base operators take the slots in
order. Magnitudes use one blue ramp, light to dark. Outcomes (correct, wrong, refused) use the
reserved status colours and are always paired with a label or a symbol, never colour alone.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """The colours of one mode (light or dark)."""

    name: str
    surface: str
    """Figure and axes background."""
    ink: str
    """Titles and values."""
    ink_secondary: str
    """Axis labels, legends, notes."""
    ink_muted: str
    """Tick labels."""
    grid: str
    """Hairline gridlines."""
    axis: str
    """Baselines and axis lines."""
    context: str
    """De-emphasised marks: what is shown for comparison, not as the point."""
    series: tuple[str, ...]
    """Categorical slots in their fixed order; classes K1, K2 are slots 1 and 2."""
    ramp: tuple[str, ...]
    """One-hue sequential ramp, from near-surface to the strongest step."""
    good: str
    critical: str
    warning: str
    band: str
    """The margin band between the classes."""

    @property
    def k1(self) -> str:
        """Colour of class K1."""
        return self.series[0]

    @property
    def k2(self) -> str:
        """Colour of class K2."""
        return self.series[1]

    @property
    def accent(self) -> str:
        """The emphasised series of a one-series or emphasis chart."""
        return self.series[0]

    def slot(self, index: int) -> str:
        """Categorical colour of series ``index``; series beyond the palette fall back to grey."""
        return self.series[index] if index < len(self.series) else self.context


LIGHT = Theme(
    name="light",
    surface="#ffffff",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    context="#c3c2b7",
    series=(
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#4a3aa7",
        "#e34948",
    ),
    ramp=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
    good="#0ca30c",
    critical="#d03b3b",
    warning="#fab219",
    band="#f0efec",
)
"""Light mode — the default of the exported figures (print, the article)."""

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    context="#52514e",
    series=(
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ),
    ramp=("#104281", "#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"),
    good="#0ca30c",
    critical="#d03b3b",
    warning="#fab219",
    band="#2c2c2a",
)
"""Dark mode — the same hues stepped for a dark surface."""

THEMES = {"light": LIGHT, "dark": DARK}
