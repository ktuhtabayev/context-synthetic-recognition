"""Figures of a run (matplotlib): what the tables show, as pictures for the article and the report.

Every figure is drawn by a ``draw_*`` function on axes it is given, so the same drawings serve
the exported files (PNG, SVG), the HTML/PDF report and — later — the GUI. :func:`figure_specs`
lists the figures of a run; :func:`save_figures` writes them.

Design: one job per chart and colour by job — the classes K1 and K2 keep the same two colours in
every figure, base operators take the next categorical slots, magnitudes use one blue ramp, the
one series that is the point is drawn in the accent and its context in grey; marks are thin, the
grid is a hairline, a legend is present whenever there are two or more series, and outcomes
(correct, wrong, refused) always carry a label beside their colour. Only :class:`matplotlib.
figure.Figure` is used — no pyplot, no global state, no display needed.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

from context_synthetic_recognition.core.meta import REFUSAL
from context_synthetic_recognition.evaluation.protocols import ProtocolResult
from context_synthetic_recognition.export.theme import LIGHT, Theme
from context_synthetic_recognition.export.view import RunView
from context_synthetic_recognition.export.wording import protocol_title
from context_synthetic_recognition.notation import latent_name

WIDTH = 7.2
"""Figure width in inches (a two-column page)."""
DPI = 200
"""Resolution of the PNG files."""
LINE = 2.0
MARKER = 7.0
RING = 1.6
"""Surface-coloured ring around markers, so they stay legible where they overlap."""
MAX_TICK_LABELS = 40
"""Objects are named along an axis only up to this many."""
MAX_PANELS = 6
_GOLDEN = 0.6180339887498949

RC: Any = {
    "font.family": "DejaVu Sans",  # ships with matplotlib; has ρ, ω, ϰ and the subscripts
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "svg.hashsalt": "context-synthetic-recognition",  # reproducible SVG ids
    "figure.dpi": 100,
}
"""Text sizes and the settings that make the files reproducible."""


@dataclass(frozen=True)
class FigureSpec:
    """A figure that can be rendered."""

    key: str
    """File name stem, e.g. ``roc`` or ``distances``."""
    title: str
    """What it shows (the caption in the report)."""
    size: tuple[float, float]
    """Width and height in inches."""
    draw: Callable[[Figure, Theme], None]


# ---------------------------------------------------------------- chrome


def style_axes(ax: Axes, theme: Theme, *, grid: str | None = "y") -> None:
    """Recessive chrome: a hairline grid, no top and right spines, muted ticks."""
    ax.set_facecolor(theme.surface)
    ax.grid(False)
    if grid is not None:
        ax.grid(True, axis=cast(Any, grid), color=theme.grid, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.axis)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=theme.ink_muted, length=3, width=0.8)
    ax.title.set_color(theme.ink)
    ax.xaxis.label.set_color(theme.ink_secondary)
    ax.yaxis.label.set_color(theme.ink_secondary)


def _bare(ax: Axes, theme: Theme) -> None:
    """Axes for a grid of cells: no spines, no grid, no tick marks."""
    style_axes(ax, theme, grid=None)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def _legend(ax: Axes, theme: Theme, handles: Sequence[Any], **kwargs: Any) -> None:
    legend = ax.legend(handles=list(handles), frameon=False, **kwargs)
    for text in legend.get_texts():
        text.set_color(theme.ink_secondary)


def _message(ax: Axes, theme: Theme, text: str) -> None:
    _bare(ax, theme)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color=theme.ink_muted)


def _ink_on(colour: Any) -> str:
    """Text colour that reads on a filled cell: white on dark fills, near-black on light ones."""
    red, green, blue = to_rgb(colour)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#ffffff" if luminance < 0.5 else "#0b0b0b"


def _ramp(theme: Theme) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("magnitude", list(theme.ramp))


def _marker(colour: str, theme: Theme, label: str, *, hollow: bool = False) -> Line2D:
    return Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=MARKER,
        markerfacecolor="none" if hollow else colour,
        markeredgecolor=colour if hollow else theme.surface,
        markeredgewidth=1.6 if hollow else RING,
        label=label,
    )


def _class_handles(view: RunView, theme: Theme) -> list[Patch]:
    classes = view.trace.classes
    return [
        Patch(facecolor=theme.k1, label=f"K1 (class {classes[0]})"),
        Patch(facecolor=theme.k2, label=f"K2 (class {classes[1]})"),
    ]


def _cell_gaps(ax: Axes, theme: Theme, rows: int, columns: int) -> None:
    """A thin surface gap between the cells of a grid (while the cells are large enough)."""
    if max(rows, columns) > MAX_TICK_LABELS:
        return
    ax.set_xticks(np.arange(-0.5, columns), minor=True)
    ax.set_yticks(np.arange(-0.5, rows), minor=True)
    ax.grid(True, which="minor", color=theme.surface, linewidth=1.4)
    ax.tick_params(which="minor", length=0)


# ---------------------------------------------------------------- Steps 2–3


def draw_distances(ax: Axes, view: RunView, operator: int, theme: Theme = LIGHT) -> None:
    """Heat map of one operator's distance matrix, the objects grouped by class (Step 2)."""
    trace = view.trace
    found = trace.operators[operator]
    order = np.argsort(trace.class_index, kind="stable")
    matrix = found.distances[np.ix_(order, order)]
    m = trace.m
    _bare(ax, theme)
    image = ax.imshow(matrix, cmap=_ramp(theme), interpolation="nearest", aspect="equal")
    _cell_gaps(ax, theme, m, m)
    if m <= MAX_TICK_LABELS:
        names = [trace.object_ids[int(i)] for i in order]
        ax.set_xticks(range(m), names, rotation=90)
        ax.set_yticks(range(m), names)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    boundary = trace.class_sizes[0] - 0.5
    for line in (ax.axhline, ax.axvline):
        line(boundary, color=theme.surface, linewidth=3.0)
        line(boundary, color=theme.ink, linewidth=0.9)
    ax.set_xlabel(
        f"K1 ({trace.class_sizes[0]} objects)  |  K2 ({trace.class_sizes[1]} objects)", labelpad=4
    )
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.outline.set_visible(False)
    bar.ax.set_facecolor(theme.surface)
    bar.ax.tick_params(colors=theme.ink_muted, length=0)
    ax.set_title(f"{found.label}(Sᵢ, Sⱼ)")


def draw_neighbourhoods(ax: Axes, view: RunView, operator: int, theme: Theme = LIGHT) -> None:
    """The class of every object's neighbours by rank, with the permitted k marked (Step 3)."""
    trace = view.trace
    found = trace.operators[operator]
    m, ks = trace.m, trace.permitted_k.ks
    depth = int(min(m - 1, max(ks) + 2))
    classes = trace.class_index[found.order[:, :depth]].astype(float)
    own = trace.class_index.astype(float)[:, None]
    gap = np.full((m, 1), np.nan)
    grid = np.hstack([own, gap, classes])
    _bare(ax, theme)
    ax.imshow(
        np.ma.masked_invalid(grid),
        cmap=ListedColormap([theme.k1, theme.k2]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
    )
    _cell_gaps(ax, theme, m, depth + 2)
    # every permitted k while they are few, otherwise the smallest and the largest
    shown = ks if len(ks) <= 6 else (ks[0], ks[-1])
    for k in shown:
        if k > depth:
            continue
        x = k + 1.5
        ax.axvline(x, color=theme.surface, linewidth=3.0)
        ax.axvline(x, color=theme.ink, linewidth=0.9)
        ax.text(
            x,
            -0.5 - 0.03 * m,
            f"k = {k}",
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=theme.ink_secondary,
        )
    step = max(1, -(-depth // 8))
    ranks = list(range(1, depth + 1, step))
    ax.set_xticks([0, *(r + 1 for r in ranks)], ["own", *map(str, ranks)])
    if m <= MAX_TICK_LABELS:
        ax.set_yticks(range(m), trace.object_ids)
    else:
        ax.set_yticks([])
    ax.set_xlabel("rank of the neighbour")
    ax.set_title(f"neighbours under {found.label}", pad=14)


def draw_object_context(
    ax: Axes,
    view: RunView,
    operator: int,
    name: str,
    distances: Any,
    order: Any,
    theme: Theme = LIGHT,
    *,
    own_class: int | None = None,
) -> None:
    """One object's local context Ψ_ρ,k: its neighbours with the nested k-neighbourhoods (Step 3).

    The object is at the centre; every neighbour sits at its distance from it (the angle only
    spreads the points), coloured by its class; a circle marks the k-th neighbour of every
    permitted k, so the neighbourhoods are nested as in the article.

    Args:
        ax: The axes to draw on.
        view: The run (operators, classes, permitted k, object names).
        operator: Index of the base operator.
        name: Name of the object at the centre (a training object or ``S``).
        distances: (m,) its distances to the training objects under the operator.
        order: Its neighbours by (distance, index) — training-object indices, itself excluded.
        theme: Colours.
        own_class: 0-based class of the object, if it is a training object (colours the centre).
    """
    trace = view.trace
    found = trace.operators[operator]
    ks = trace.permitted_k.ks
    ranked = np.asarray(order, dtype=np.int64)
    radii = np.asarray(distances, dtype=np.float64)[ranked]
    depth = int(min(ranked.size, max(ks) + 3, MAX_TICK_LABELS))
    _bare(ax, theme)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    angles = 2.0 * np.pi * _GOLDEN * np.arange(depth) + 0.5 * np.pi
    x, y = radii[:depth] * np.cos(angles), radii[:depth] * np.sin(angles)
    limit = float(radii[:depth].max()) if depth else 1.0
    limit = limit if limit > 0 else 1.0
    # the neighbourhoods among the neighbours shown — at most six circles
    for k in tuple(k for k in ks if k <= depth)[:MAX_PANELS]:
        radius = float(radii[k - 1])
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                radius,
                fill=False,
                linestyle=(0, (4, 3)),
                linewidth=0.9,
                edgecolor=theme.ink_muted,
            )
        )
        ax.text(
            0.0,
            -radius,
            f"k = {k}",
            ha="center",
            va="top",
            fontsize=7.5,
            color=theme.ink_secondary,
        )
        limit = max(limit, radius)
    classes = trace.class_index[ranked[:depth]]
    colours = [theme.k1 if c == 0 else theme.k2 for c in classes]
    ax.scatter(x, y, s=MARKER**2, c=colours, edgecolors=theme.surface, linewidths=RING, zorder=3)
    if depth <= 25:
        for t in range(depth):
            ax.annotate(
                trace.object_ids[int(ranked[t])],
                (float(x[t]), float(y[t])),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=7.5,
                color=theme.ink_secondary,
            )
    centre = theme.ink if own_class is None else (theme.k1 if own_class == 0 else theme.k2)
    ax.scatter(
        [0.0],
        [0.0],
        s=(MARKER * 1.5) ** 2,
        c=[centre],
        marker="D",
        edgecolors=theme.surface,
        linewidths=RING,
        zorder=4,
    )
    ax.annotate(
        name,
        (0.0, 0.0),
        textcoords="offset points",
        xytext=(8, 7),
        fontsize=8.5,
        fontweight="bold",
        color=theme.ink,
    )
    pad = 1.18 * limit
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    handles = [
        _marker(theme.k1, theme, f"K1 (class {trace.classes[0]})"),
        _marker(theme.k2, theme, f"K2 (class {trace.classes[1]})"),
    ]
    _legend(ax, theme, handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncols=2)
    ax.set_title(f"{found.label}: neighbourhoods of {name}  (radius = distance)")


def object_context_figure(
    view: RunView,
    operator: int,
    name: str,
    distances: Any,
    order: Any,
    theme: Theme = LIGHT,
    *,
    own_class: int | None = None,
    size: tuple[float, float] = (5.6, 5.0),
) -> Figure:
    """:func:`draw_object_context` as a figure of its own, with the exporters' text sizes."""
    with matplotlib.rc_context(RC):
        figure = Figure(figsize=size, layout="constrained", facecolor=theme.surface)
        draw_object_context(
            figure.subplots(), view, operator, name, distances, order, theme, own_class=own_class
        )
    return figure


# ---------------------------------------------------------------- Steps 6–8


def _lollipops(
    ax: Axes,
    theme: Theme,
    positions: Sequence[float],
    values: Sequence[float],
    emphasised: Sequence[bool],
    baseline: float,
) -> None:
    for x, value, strong in zip(positions, values, emphasised, strict=True):
        colour = theme.accent if strong else theme.context
        ax.plot([x, x], [baseline, value], color=colour, linewidth=LINE, solid_capstyle="round")
        ax.plot(
            [x],
            [value],
            marker="o",
            markersize=MARKER,
            markerfacecolor=colour,
            markeredgecolor=theme.surface,
            markeredgewidth=RING,
            zorder=3,
        )


def draw_feature_quality(
    ax: Axes, view: RunView, quantity: str = "omega", theme: Theme = LIGHT
) -> None:
    """ω (formula (4)) or g_k (formula (2)) of every synthetic feature; TUPLAM is emphasised.

    Few features are drawn one by one; many (a long k range) as a line over k per operator.
    """
    trace = view.trace
    features = trace.features
    values = [f.omega if quantity == "omega" else f.stability for f in features]
    baseline = 0.0 if quantity == "omega" else 0.5
    title = (
        "informativeness ω — formula (4)" if quantity == "omega" else "stability g — formula (2)"
    )
    selected = set(view.hag.tuplam)
    style_axes(ax, theme)
    ax.set_title(title)
    per_operator = max(
        sum(1 for f in features if f.operator == o) for o in range(len(trace.operators))
    )
    if per_operator <= 8:
        positions: list[float] = []
        labels: list[str] = []
        x = 0.0
        previous = features[0].operator
        for f in features:
            if f.operator != previous:
                x += 0.6
                previous = f.operator
            positions.append(x)
            labels.append(f"{f.name}\n{f.operator_label}·{f.k}")
            x += 1.0
        _lollipops(ax, theme, positions, values, [f.index in selected for f in features], baseline)
        ax.set_xticks(positions, labels)
        ax.set_xlim(-0.6, positions[-1] + 0.6)
        ax.grid(False, axis="x")
        handles = [
            _marker(theme.accent, theme, "in TUPLAM"),
            _marker(theme.context, theme, "not selected"),
        ]
        _legend(
            ax,
            theme,
            handles,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            ncols=2,
            borderaxespad=0.1,
        )
    else:
        for o, operator in enumerate(trace.operators):
            own = [f for f in features if f.operator == o]
            xs = [f.k for f in own]
            ys = [f.omega if quantity == "omega" else f.stability for f in own]
            colour = theme.slot(o)
            ax.plot(
                xs, ys, color=colour, linewidth=LINE, solid_capstyle="round", label=operator.label
            )
            chosen = [f for f in own if f.index in selected]
            ax.plot(
                [f.k for f in chosen],
                [f.omega if quantity == "omega" else f.stability for f in chosen],
                linestyle="None",
                marker="o",
                markersize=MARKER,
                markerfacecolor=colour,
                markeredgecolor=theme.surface,
                markeredgewidth=RING,
                zorder=3,
            )
        ks = trace.permitted_k.ks
        ax.set_xticks(ks[:: max(1, -(-len(ks) // 12))])
        ax.set_xlabel("k (dots: the features in TUPLAM)")
        _legend(
            ax,
            theme,
            ax.get_legend_handles_labels()[0],
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            ncols=len(trace.operators),
            borderaxespad=0.1,
        )
    ax.set_ylim(baseline - 0.02, 1.06)


# ---------------------------------------------------------------- Step 9


def draw_candidates(ax: Axes, view: RunView, iteration: int, theme: Theme = LIGHT) -> None:
    """θ/γ of every candidate of one HAG iteration; the chosen q is emphasised (STEP 3)."""
    it = view.hag.iterations[iteration]
    features = view.trace.features
    style_axes(ax, theme)
    finite = np.isfinite(it.ratio)
    candidates = it.candidates[finite]
    ratios = it.ratio[finite]
    positions = np.arange(candidates.size)
    chosen = candidates == it.q if it.q is not None else np.zeros(candidates.size, dtype=bool)
    ax.plot(
        positions[~chosen],
        ratios[~chosen],
        linestyle="None",
        marker="o",
        markersize=MARKER if candidates.size <= 40 else 4.0,
        markerfacecolor=theme.context,
        markeredgecolor=theme.surface,
        markeredgewidth=RING if candidates.size <= 40 else 0.6,
    )
    if it.q is not None and chosen.any():
        x, y = float(positions[chosen][0]), float(ratios[chosen][0])
        ax.plot(
            [x],
            [y],
            linestyle="None",
            marker="o",
            markersize=MARKER + 1.5,
            markerfacecolor=theme.accent,
            markeredgecolor=theme.surface,
            markeredgewidth=RING,
            zorder=3,
        )
        # beside the dot, on the side that has room
        on_right = x <= (candidates.size - 1) / 2
        ax.annotate(
            f"q = {features[it.q].name}\n{y:.4f}",
            (x, y),
            xytext=(9 if on_right else -9, 12),
            textcoords="offset points",
            ha="left" if on_right else "right",
            va="bottom",
            fontsize=7.5,
            color=theme.ink,
        )
    if candidates.size <= 14:
        ax.set_xticks(positions, [features[int(u)].name for u in candidates])
    else:
        ax.set_xticks([])
        ax.set_xlabel(f"{candidates.size} candidates")
    ax.grid(False, axis="x")
    ax.margins(x=0.12, y=0.28)
    ax.set_title(f"iteration {it.number}")


def draw_criterion(ax: Axes, view: RunView, theme: Theme = LIGHT) -> None:
    """Crit = min θ/γ after every HAG iteration against the threshold δ (STEP 5)."""
    hag = view.hag
    style_axes(ax, theme)
    added = [it for it in hag.iterations if it.q is not None]
    if not added:
        _message(ax, theme, "The HAG added no feature (p = 0).")
        return
    steps = [it.number for it in added]
    crit = [it.crit for it in added]
    ax.plot(
        steps,
        crit,
        color=theme.accent,
        linewidth=LINE,
        solid_capstyle="round",
        solid_joinstyle="round",
        marker="o",
        markersize=MARKER,
        markeredgecolor=theme.surface,
        markeredgewidth=RING,
    )
    delta = hag.settings.delta
    ax.axhline(delta, color=theme.ink_muted, linewidth=1.0)
    ax.text(
        steps[-1],
        delta,
        f"δ = {delta:g}  ",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=theme.ink_secondary,
    )
    for index in {0, len(steps) - 1}:
        ax.annotate(
            f"{crit[index]:.4f}",
            (steps[index], crit[index]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=theme.ink,
        )
    names = view.trace.features
    ax.set_xticks(steps, [f"{it.number}\n{names[it.q].name}" for it in added if it.q is not None])
    ax.set_xlim(steps[0] - 0.4, steps[-1] + 0.4)
    low, high = min(*crit, delta), max(*crit, delta)
    pad = max((high - low) * 0.22, 0.02)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("iteration · feature q added")
    ax.set_ylabel("crit = min θ/γ")
    ax.set_title(f"HAG criterion (stops: {hag.stop.text})")


# ---------------------------------------------------------------- margins


def _jitter(count: int) -> Any:
    """A deterministic vertical spread, so that equal values stay visible."""
    return ((np.arange(count) * _GOLDEN) % 1.0 - 0.5) * 0.44


def draw_margin(
    ax: Axes, view: RunView, latent: int, with_majorizer: bool = True, theme: Theme = LIGHT
) -> None:
    """One latent feature on a line: both classes, the boundary b and the margin between them."""
    analysis = view.result.margins
    margins = analysis.with_majorizer if with_majorizer else analysis.without_majorizer
    style_axes(ax, theme, grid="x")
    margin = margins[latent]
    d, y = margin.values, margin.class_index
    left, right = margin.left, margin.right
    overlap = margin.width < 0
    ax.axvspan(
        min(left, right),
        max(left, right),
        color=theme.critical if overlap else theme.band,
        alpha=0.14 if overlap else 1.0,
        linewidth=0,
    )
    ax.axvline(margin.boundary, color=theme.ink, linewidth=1.0)
    wrong = margin.predictions != y + 1
    spread = _jitter(d.size)
    for index, colour in ((0, theme.k1), (1, theme.k2)):
        members = y == index
        row = 1.0 - index
        for mask, hollow in ((members & ~wrong, False), (members & wrong, True)):
            if mask.any():
                ax.plot(
                    d[mask],
                    row + spread[mask],
                    linestyle="None",
                    marker="o",
                    markersize=MARKER if d.size <= 60 else 4.0,
                    markerfacecolor="none" if hollow else colour,
                    markeredgecolor=colour if hollow else theme.surface,
                    markeredgewidth=1.6 if hollow else (RING if d.size <= 60 else 0.5),
                    zorder=3,
                )
    span = max(float(d.max() - d.min()), 1e-9)
    ax.set_xlim(float(d.min()) - 0.06 * span, float(d.max()) + 0.06 * span)
    ax.set_ylim(-0.7, 1.7)
    ax.set_yticks([0.0, 1.0], ["K2", "K1"])
    name = latent_name(latent) + ("" if with_majorizer else ", no ϕ")
    ax.set_title(f"{name}\nwidth {margin.width:+.4f}", fontsize=8.5)


def draw_margin_widths(ax: Axes, view: RunView, theme: Theme = LIGHT) -> None:
    """Margin width of every latent feature, with the majorizer and without it."""
    analysis = view.result.margins
    style_axes(ax, theme)
    with_, without = analysis.with_majorizer, analysis.without_majorizer
    steps = list(range(1, len(with_) + 1))
    ax.axhline(0.0, color=theme.axis, linewidth=1.0)
    series = (
        ("without majorizer", [m.width for m in without], theme.context),
        ("with majorizer ϕ", [m.width for m in with_], theme.accent),
    )
    for label, widths, colour in series:
        ax.plot(
            steps,
            widths,
            color=colour,
            linewidth=LINE,
            solid_capstyle="round",
            marker="o",
            markersize=MARKER,
            markeredgecolor=theme.surface,
            markeredgewidth=RING,
            label=label,
        )
        ax.annotate(
            f"{widths[-1]:+.4f}",
            (steps[-1], widths[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=theme.ink,
        )
    ax.set_xticks(steps, [latent_name(j) for j in range(len(steps))])
    ax.set_xlim(0.6, len(steps) + 0.75)
    ax.margins(y=0.2)
    ax.set_ylabel("min K1 − max K2  (> 0: separated)")
    ax.set_title("margin width of the latent features")
    _legend(ax, theme, ax.get_legend_handles_labels()[0], loc="best")


# ---------------------------------------------------------------- evaluation


def draw_roc(ax: Axes, view: RunView, protocol: ProtocolResult, theme: Theme = LIGHT) -> None:
    """ROC curve of the CS-model's score; the baselines are drawn as grey context."""
    result = view.result
    decimals = view.config.evaluation.score_decimals
    style_axes(ax, theme, grid="both")
    # the chance diagonal is a reference threshold, hence dashed (gridlines stay solid)
    ax.plot([0, 1], [0, 1], color=theme.ink_muted, linewidth=1.0, linestyle=(0, (4, 3)))
    aucs = []
    for baseline in protocol.baselines:
        curve = baseline.roc(result.positive, decimals)
        aucs.append(curve.auc)
        ax.plot(curve.fpr, curve.tpr, color=theme.context, linewidth=1.2, solid_capstyle="round")
    curve = protocol.predictions.roc(result.positive, decimals)
    ax.plot(
        curve.fpr,
        curve.tpr,
        color=theme.accent,
        linewidth=LINE,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )
    handles = [
        Line2D(
            [0], [0], color=theme.accent, linewidth=LINE, label=f"CS-model · AUC {curve.auc:.3f}"
        )
    ]
    finite = [a for a in aucs if np.isfinite(a)]
    if finite:
        span = f"{min(finite):.3f}" if len(finite) == 1 else f"{min(finite):.3f}–{max(finite):.3f}"
        handles.append(
            Line2D([0], [0], color=theme.context, linewidth=1.2, label=f"baselines · AUC {span}")
        )
    handles.append(
        Line2D(
            [0], [0], color=theme.ink_muted, linewidth=1.0, linestyle=(0, (4, 3)), label="chance"
        )
    )
    _legend(ax, theme, handles, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR (recall)")
    ax.set_title(protocol_title(protocol.protocol))


def draw_confusion(ax: Axes, view: RunView, protocol: ProtocolResult, theme: Theme = LIGHT) -> None:
    """Confusion matrix of the CS-model: actual class × predicted class, refusals separate."""
    classes = view.trace.classes
    matrix = view.result.metrics(protocol.predictions).matrix
    _bare(ax, theme)
    ramp = _ramp(theme)
    top = max(int(matrix.max()), 1)
    ax.imshow(matrix, cmap=ramp, vmin=0, vmax=top, interpolation="nearest", aspect="auto")
    _cell_gaps(ax, theme, 2, 3)
    for row in range(2):
        for column in range(3):
            count = int(matrix[row, column])
            ax.text(
                column,
                row,
                str(count),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold" if row == column else "normal",
                color=_ink_on(ramp(count / top)),
            )
    names = [f"class {classes[0]}", f"class {classes[1]}"]
    ax.set_xticks(range(3), [*names, "refusal"])
    ax.set_yticks(range(2), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(protocol_title(protocol.protocol))


def draw_outcomes(ax: Axes, view: RunView, protocol: ProtocolResult, theme: Theme = LIGHT) -> None:
    """Correct, wrong and refused decisions of every method under one protocol (stacked bars)."""
    result = view.result
    methods = protocol.methods
    style_axes(ax, theme, grid="x")
    parts = (
        ("correct", theme.good),
        ("wrong", theme.critical),
        ("refused (0)", theme.context),
    )
    total = int(methods[0].decisions.size)
    for row, method in enumerate(methods):
        metrics = result.metrics(method)
        counts = (metrics.correct, metrics.n - metrics.correct - metrics.refusals, metrics.refusals)
        left = 0.0
        for (_, colour), count in zip(parts, counts, strict=True):
            if count == 0:
                continue
            ax.barh(
                row,
                count,
                left=left,
                height=0.56,
                color=colour,
                edgecolor=theme.surface,
                linewidth=1.6,
            )
            if count / total >= 0.07:
                ax.text(
                    left + count / 2,
                    row,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=_ink_on(colour),
                )
            left += count
    ax.set_yticks(range(len(methods)), [m.method for m in methods])
    ax.invert_yaxis()
    ax.set_xlim(0, total)
    ax.set_xlabel("decisions")
    ax.grid(False, axis="y")
    handles = [Patch(facecolor=colour, label=label) for label, colour in parts]
    _legend(ax, theme, handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=3)
    ax.set_title(protocol_title(protocol.protocol), pad=26)


def draw_sensitivity(ax: Axes, view: RunView, theme: Theme = LIGHT) -> None:
    """Accuracy under the four settings of the two template/article switches."""
    variants = view.sensitivity or ()
    style_axes(ax, theme, grid="x")
    if not variants:
        _message(ax, theme, "The switch settings were not evaluated.")
        return
    series = (
        ("resubstitution", [v.resubstitution for v in variants], theme.series[0]),
        ("leave-one-out", [v.leave_one_out for v in variants], theme.series[1]),
    )
    rows = range(len(variants))
    for row, variant in zip(rows, variants, strict=True):
        ends = sorted((variant.resubstitution, variant.leave_one_out))
        ax.plot(ends, [row, row], color=theme.axis, linewidth=1.2, zorder=1)
    for label, values, colour in series:
        ax.plot(
            values,
            list(rows),
            linestyle="None",
            marker="o",
            markersize=MARKER + 1,
            markerfacecolor=colour,
            markeredgecolor=theme.surface,
            markeredgewidth=RING,
            label=label,
            zorder=3,
        )
    for row, variant in zip(rows, variants, strict=True):
        low, high = sorted((variant.resubstitution, variant.leave_one_out))
        ax.annotate(
            f"{100 * high:.0f} %",
            (high, row),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=theme.ink,
        )
        if high - low > 0.08:
            ax.annotate(
                f"{100 * low:.0f} %",
                (low, row),
                xytext=(-8, 0),
                textcoords="offset points",
                va="center",
                ha="right",
                fontsize=7.5,
                color=theme.ink,
            )
    ax.set_yticks(list(rows), [f"{v.label}\n{v.tuplam}" for v in variants])
    ax.invert_yaxis()
    ax.set_ylim(len(variants) - 0.5, -0.5)
    ax.set_xlim(-0.08, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100 %"])
    ax.set_xlabel("accuracy")
    ax.grid(False, axis="y")
    _legend(
        ax,
        theme,
        ax.get_legend_handles_labels()[0],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=2,
    )
    ax.set_title("the two switches: class centres × STEP 4 passes", pad=26)


def draw_prediction_map(
    ax: Axes, view: RunView, protocol: ProtocolResult, theme: Theme = LIGHT
) -> None:
    """Every method's decision for every held-out object: correct, wrong (×) or refused (0)."""
    trace = view.trace
    methods = protocol.methods
    predictions = protocol.predictions
    order = np.argsort(predictions.objects, kind="stable")
    entries = int(order.size)
    _bare(ax, theme)
    truth = predictions.truth[order]
    for column in range(entries):
        colour = theme.k1 if truth[column] == 1 else theme.k2
        ax.plot([column], [0], marker="s", markersize=9, color=colour, linestyle="None")
    for row, method in enumerate(methods, start=1):
        decisions = method.decisions[order]
        for column in range(entries):
            decision = int(decisions[column])
            if decision == REFUSAL:
                colour, symbol = theme.context, "0"
            elif decision == truth[column]:
                colour, symbol = theme.good, ""
            else:
                colour, symbol = theme.critical, "×"
            ax.plot([column], [row], marker="s", markersize=9, color=colour, linestyle="None")
            if symbol:
                ax.text(
                    column,
                    row,
                    symbol,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=_ink_on(colour),
                )
    ax.set_yticks(range(len(methods) + 1), ["true class", *(m.method for m in methods)])
    ax.set_xticks(range(entries), [trace.object_ids[int(predictions.objects[e])] for e in order])
    ax.set_xlim(-0.7, entries - 0.3)
    ax.set_ylim(len(methods) + 0.6, -0.6)
    handles = [
        *_class_handles(view, theme),
        Patch(facecolor=theme.good, label="correct"),
        Patch(facecolor=theme.critical, label="wrong (×)"),
        Patch(facecolor=theme.context, label="refused (0)"),
    ]
    _legend(ax, theme, handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=5)
    ax.set_title(protocol_title(protocol.protocol), pad=26)


# ---------------------------------------------------------------- the figures of a run


def _panels(figure: Figure, rows: int, columns: int, **kwargs: Any) -> list[Axes]:
    axes = figure.subplots(rows, columns, squeeze=False, **kwargs)
    return [ax for line in axes for ax in line]


def figure_specs(view: RunView) -> list[FigureSpec]:
    """The figures of a run, in pipeline order."""
    trace, hag, result = view.trace, view.hag, view.result
    operators = len(trace.operators)
    specs: list[FigureSpec] = []
    columns = min(operators, 3)
    rows = -(-operators // columns)

    def distances(figure: Figure, theme: Theme) -> None:
        axes = _panels(figure, rows, columns)
        for o in range(operators):
            draw_distances(axes[o], view, o, theme)
        for ax in axes[operators:]:
            ax.set_visible(False)
        figure.suptitle("Step 2 · distances of the base operators", color=theme.ink, fontsize=10.5)

    def neighbourhoods(figure: Figure, theme: Theme) -> None:
        axes = _panels(figure, rows, columns)
        for o in range(operators):
            draw_neighbourhoods(axes[o], view, o, theme)
        for ax in axes[operators:]:
            ax.set_visible(False)
        handles = _class_handles(view, theme)
        legend = figure.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False)
        for text in legend.get_texts():
            text.set_color(theme.ink_secondary)
        figure.suptitle(
            "Step 3 · nested neighbourhoods: the class of the neighbour at every rank",
            color=theme.ink,
            fontsize=10.5,
        )

    specs.append(
        FigureSpec(
            "distances",
            "Distance matrices of the base operators (Step 2); objects grouped by class.",
            (WIDTH, 0.6 + 2.6 * rows),
            distances,
        )
    )
    height = min(1.1 + 0.17 * trace.m, 8.5) if trace.m <= MAX_TICK_LABELS else 4.2
    specs.append(
        FigureSpec(
            "neighbourhoods",
            "Class of the neighbours by rank under every operator; the lines mark the permitted "
            "k (Step 3).",
            (WIDTH, 0.9 + height * rows),
            neighbourhoods,
        )
    )

    def quality(figure: Figure, theme: Theme) -> None:
        axes = _panels(figure, 2, 1)
        draw_feature_quality(axes[0], view, "omega", theme)
        draw_feature_quality(axes[1], view, "stability", theme)
        figure.suptitle("Steps 6–7 · the synthetic features Ψ(r)", color=theme.ink, fontsize=10.5)

    specs.append(
        FigureSpec(
            "synthetic-features",
            "Informativeness ω (formula (4)) and stability g (formula (2)) of the synthetic "
            "features; the features of TUPLAM are emphasised.",
            (WIDTH, 5.0),
            quality,
        )
    )

    executed = [index for index, it in enumerate(hag.iterations) if it.candidates.size]
    if executed:
        shown = executed[:MAX_PANELS]
        per_row = min(len(shown), 4)
        lines = -(-len(shown) // per_row)

        def candidates(figure: Figure, theme: Theme) -> None:
            axes = _panels(figure, lines, per_row, sharey=True)
            for position, index in enumerate(shown):
                draw_candidates(axes[position], view, index, theme)
            for ax in axes[len(shown) :]:
                ax.set_visible(False)
            for line in range(lines):
                axes[line * per_row].set_ylabel("θ/γ")
            figure.suptitle(
                "Step 9 · HAG STEP 3: θ/γ of the candidates, q = argmin",
                color=theme.ink,
                fontsize=10.5,
            )

        specs.append(
            FigureSpec(
                "hag-candidates",
                "θ/γ of every candidate in every HAG iteration; the chosen feature q is "
                "emphasised.",
                (WIDTH, 0.5 + 2.5 * lines),
                candidates,
            )
        )
        specs.append(
            FigureSpec(
                "hag-criterion",
                "The criterion crit = min θ/γ after every HAG iteration, against δ.",
                (WIDTH * 0.72, 3.0),
                lambda figure, theme: draw_criterion(figure.subplots(), view, theme),
            )
        )

    p = hag.p
    if p:
        shown_latent = min(p, MAX_PANELS)

        def margins(figure: Figure, theme: Theme) -> None:
            axes = _panels(figure, 2, shown_latent)
            for j in range(shown_latent):
                draw_margin(axes[j], view, j, True, theme)
                draw_margin(axes[shown_latent + j], view, j, False, theme)
            handles = [
                _marker(theme.k1, theme, f"K1 (class {trace.classes[0]})"),
                _marker(theme.k2, theme, f"K2 (class {trace.classes[1]})"),
                _marker(theme.ink_muted, theme, "ŷ ≠ class", hollow=True),
                Line2D([0], [0], color=theme.ink, linewidth=1.0, label="boundary b"),
            ]
            legend = figure.legend(
                handles=handles, loc="outside lower center", ncols=4, frameon=False
            )
            for text in legend.get_texts():
                text.set_color(theme.ink_secondary)
            figure.suptitle(
                "Margins of the latent features: with the majorizer (top) and without (bottom)",
                color=theme.ink,
                fontsize=10.5,
            )

        specs.append(
            FigureSpec(
                "margins",
                "The latent features r₁ … r_p on a line, with the majorizer and without: both "
                "classes, the boundary b and the margin (red band: the classes overlap).",
                (WIDTH, 4.3),
                margins,
            )
        )
        specs.append(
            FigureSpec(
                "margin-widths",
                "Margin width min K1 − max K2 of the latent features with and without the "
                "majorizer.",
                (WIDTH * 0.72, 3.0),
                lambda figure, theme: draw_margin_widths(figure.subplots(), view, theme),
            )
        )

    protocols = result.protocols
    count = len(protocols)

    def roc(figure: Figure, theme: Theme) -> None:
        axes = _panels(figure, 1, count)
        for ax, protocol in zip(axes, protocols, strict=True):
            draw_roc(ax, view, protocol, theme)
        figure.suptitle("ROC of score₁ − score₂ (Step 4)", color=theme.ink, fontsize=10.5)

    def confusion(figure: Figure, theme: Theme) -> None:
        axes = _panels(figure, 1, count)
        for ax, protocol in zip(axes, protocols, strict=True):
            draw_confusion(ax, view, protocol, theme)
        figure.suptitle("Confusion matrices of the CS-model", color=theme.ink, fontsize=10.5)

    specs.append(
        FigureSpec(
            "roc",
            "ROC curves of the meta-algorithm's score under every protocol; baselines in grey.",
            (min(WIDTH, 0.4 + 3.4 * count), 3.9),
            roc,
        )
    )
    specs.append(
        FigureSpec(
            "confusion",
            "Confusion matrices of the CS-model (actual × predicted; refusals separate).",
            (min(WIDTH, 0.6 + 3.3 * count), 2.6),
            confusion,
        )
    )
    for protocol in protocols:
        methods = len(protocol.methods)
        name = protocol.protocol
        specs.append(
            FigureSpec(
                f"outcomes-{name}",
                f"Correct, wrong and refused decisions of every method · {protocol_title(name)}.",
                (WIDTH, 1.25 + 0.3 * methods),
                lambda figure, theme, q=protocol: draw_outcomes(  # type: ignore[misc]
                    figure.subplots(), view, q, theme
                ),
            )
        )
        entries = int(protocol.predictions.objects.size)
        single = bool(np.unique(protocol.predictions.repeats).size == 1)
        if entries <= MAX_TICK_LABELS and single and name != "resubstitution":
            specs.append(
                FigureSpec(
                    f"prediction-map-{name}",
                    f"The decision of every method for every held-out object · "
                    f"{protocol_title(name)}.",
                    (WIDTH, 1.3 + 0.26 * (methods + 1)),
                    lambda figure, theme, q=protocol: draw_prediction_map(  # type: ignore[misc]
                        figure.subplots(), view, q, theme
                    ),
                )
            )
    if view.sensitivity:
        specs.append(
            FigureSpec(
                "sensitivity",
                "⚠ Accuracy under the four settings of the two template/article switches.",
                (WIDTH, 3.1),
                lambda figure, theme: draw_sensitivity(figure.subplots(), view, theme),
            )
        )
    return specs


def render(spec: FigureSpec, theme: Theme = LIGHT) -> Figure:
    """Draw a figure (the caller saves or shows it)."""
    with matplotlib.rc_context(RC):
        figure = Figure(figsize=spec.size, layout="constrained", facecolor=theme.surface)
        spec.draw(figure, theme)
    return figure


def save_figure(figure: Figure, path: Path, dpi: int = DPI) -> Path:
    """Write a figure; the format follows the extension (png, svg, pdf).

    An SVG has no time stamp and LF line endings: the same bytes on every platform.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower().lstrip(".")
    metadata: dict[str, Any] = {"Date": None} if extension == "svg" else {}
    buffer = io.BytesIO()
    with matplotlib.rc_context(RC):
        figure.savefig(
            buffer,
            format=extension,
            dpi=dpi,
            facecolor=figure.get_facecolor(),
            metadata=metadata,
        )
    path.write_bytes(buffer.getvalue())
    return path


def save_figures(
    view: RunView,
    folder: Path,
    formats: Sequence[str] = ("png", "svg"),
    *,
    theme: Theme = LIGHT,
    dpi: int = DPI,
) -> list[Path]:
    """Render every figure of a run into ``folder``, one file per figure and format."""
    written = []
    for spec in figure_specs(view):
        figure = render(spec, theme)
        for extension in formats:
            written.append(save_figure(figure, folder / f"{spec.key}.{extension}", dpi))
    return written
