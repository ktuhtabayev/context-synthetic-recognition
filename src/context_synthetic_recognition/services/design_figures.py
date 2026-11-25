"""The figures of the evaluation designs: the random control and the training-size curve (ADR-096).

- *Random training samples*: one panel per split, one row per method; the accuracies of the 100
  random training samples as grey dots, the training sample as given as a diamond in the accent
  colour — where it falls in the distribution is the point.
- *Training-size curve*: one row of panels per split — the standard classifiers with the k-NN rows
  of k = 5, and the CS-models with α = 0.1 — the mean accuracy at 25, 50, 75 and 100 % of the
  training sample with ± one standard deviation. The tables give every method.

The colours are the project's validated palette (``export.theme``): series in their fixed order,
never more than six to a panel, each with a marker and a legend entry, so colour is never the only
cue.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from context_synthetic_recognition.export.figures import FigureSpec, style_axes
from context_synthetic_recognition.export.theme import Theme
from context_synthetic_recognition.services.designs import (
    CaseResult,
    DesignResult,
    SplitDesignResult,
    Statistic,
)

MARKERS = ("o", "s", "^", "D", "v", "P")
"""One marker per series of a panel (with its colour)."""
PANEL_SERIES = 6
"""At most this many series in one panel."""


def _case(split: SplitDesignResult, key: str) -> CaseResult | None:
    return next((c for c in split.cases if c.key == key and not c.unavailable), None)


def _offsets(count: int) -> np.ndarray:
    """Fixed vertical offsets of ``count`` dots within a row (no randomness: the same image)."""
    return (((np.arange(count) * 0.6180339887) % 1.0) - 0.5) * 0.5


def random_control_figure(result: DesignResult) -> FigureSpec | None:
    """The accuracies of the random training samples, and the sample as given (a diamond)."""
    splits = [s for s in result.splits if _case(s, "random") and _case(s, "as-given")]
    if not splits:
        return None
    rows = max(len(c.methods) for s in splits if (c := _case(s, "random")) is not None)

    def draw(figure: Figure, theme: Theme) -> None:
        axes = figure.subplots(1, len(splits), squeeze=False)[0]
        for index, (ax, split) in enumerate(zip(axes, splits, strict=True)):
            random, given = _case(split, "random"), _case(split, "as-given")
            assert random is not None
            assert given is not None
            for row, (key, _) in enumerate(random.methods):
                values = random.accuracies(key)
                ax.scatter(
                    values,
                    row + _offsets(len(values)),
                    s=7,
                    color=theme.context,
                    linewidths=0,
                    zorder=2,
                )
                mark = given.accuracies(key)
                if mark:
                    ax.scatter(
                        mark,
                        [row],
                        marker="D",
                        s=34,
                        color=theme.accent,
                        edgecolors=theme.surface,
                        linewidths=1.2,
                        zorder=3,
                    )
            ax.set_yticks(range(len(random.methods)))
            ax.set_yticklabels([t for _, t in random.methods] if index == 0 else [])
            ax.set_ylim(len(random.methods) - 0.5, -0.5)
            ax.set_xlabel("Accuracy of the classified objects, %")
            ax.set_title(split.label, fontsize=8, loc="left")
            style_axes(ax, theme, grid="x")
        handles = [
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                ms=4,
                color=theme.context,
                label="random training samples of the same size and classes",
            ),
            Line2D(
                [],
                [],
                ls="",
                marker="D",
                ms=6,
                color=theme.accent,
                mec=theme.surface,
                label="the training sample as given",
            ),
        ]
        legend = figure.legend(handles=handles, loc="outside upper left", ncols=2, frameon=False)
        for text in legend.get_texts():
            text.set_color(theme.ink_secondary)

    return FigureSpec(
        "random-control",
        "Random training samples: the accuracy of every method, and of the sample as given",
        (min(4.2 * len(splits), 12.0), 0.24 * rows + 1.6),
        draw,
    )


def _drawn(
    methods: Sequence[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """The methods of the curve: the standard ones with k = 5, and the CS-models with α = 0.1."""
    first = [
        (k, t)
        for k, t in methods
        if k in ("svm", "random-forest") or (k.startswith("knn-") and k.endswith("-5"))
    ]
    second = [(k, t) for k, t in methods if k.startswith("cs-model-") and k.endswith("-0.1")]
    if not first and not second:  # another part: its first methods
        first = list(methods[:PANEL_SERIES])
    return first[:PANEL_SERIES], second[:PANEL_SERIES]


def size_curve_figure(result: DesignResult) -> FigureSpec | None:
    """The mean accuracy at 25, 50, 75 and 100 % of the training sample, ± one SD."""
    splits = [s for s in result.splits if any(not c.unavailable for c in s.cases)]
    if not splits:
        return None

    def draw(figure: Figure, theme: Theme) -> None:
        axes = figure.subplots(len(splits), 2, squeeze=False, sharey="row")
        for row, split in enumerate(splits):
            cases = [c for c in split.cases if not c.unavailable]
            shares = [int(c.key) for c in cases]
            methods = cases[0].methods
            for column, chosen in enumerate(_drawn(methods)):
                ax = axes[row][column]
                for slot, (key, title) in enumerate(chosen):
                    stats = [Statistic.of(c.accuracies(key)) for c in cases]
                    points = [
                        (s, st) for s, st in zip(shares, stats, strict=True) if st is not None
                    ]
                    if not points:
                        continue
                    ax.errorbar(
                        [p for p, _ in points],
                        [st.mean for _, st in points],
                        yerr=[st.sd or 0.0 for _, st in points],
                        color=theme.slot(slot),
                        marker=MARKERS[slot],
                        ms=4,
                        lw=1.6,
                        capsize=2,
                        label=title,
                    )
                ax.set_xticks([25, 50, 75, 100])
                ax.set_xlabel("Share of the training sample, %")
                if column == 0:
                    ax.set_ylabel("Accuracy on the test sample, %")
                ax.set_title(
                    f"{split.label} · {'standard classifiers' if column == 0 else 'CS-model'}",
                    fontsize=8,
                    loc="left",
                )
                style_axes(ax, theme)
                if chosen:
                    legend = ax.legend(fontsize=7, frameon=False, loc="best")
                    for text in legend.get_texts():
                        text.set_color(theme.ink_secondary)

    return FigureSpec(
        "size-curve",
        "Training-size curve: the mean accuracy on the test sample, ± one standard deviation",
        (10.0, 3.2 * len(splits) + 0.6),
        draw,
    )


FIGURES = {"random-control": random_control_figure, "size-curve": size_curve_figure}
"""The design whose results have a figure → the figure."""
