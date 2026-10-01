"""The figures of a run: what is drawn for every shape, the files, the themes."""

from pathlib import Path
from typing import Any

import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from context_synthetic_recognition.export import RunView
from context_synthetic_recognition.export.figures import (
    MAX_PANELS,
    draw_candidates,
    draw_confusion,
    draw_criterion,
    draw_distances,
    draw_feature_quality,
    draw_margin,
    draw_margin_widths,
    draw_neighbourhoods,
    draw_outcomes,
    draw_prediction_map,
    draw_roc,
    draw_sensitivity,
    figure_specs,
    object_context_figure,
    render,
    save_figure,
    save_figures,
)
from context_synthetic_recognition.export.theme import DARK, LIGHT, THEMES

from .shapes import SHAPES, shape

EXPERIMENT_FIGURES = [
    "distances",
    "neighbourhoods",
    "synthetic-features",
    "hag-candidates",
    "hag-criterion",
    "margins",
    "margin-widths",
    "roc",
    "confusion",
    "outcomes-resubstitution",
    "outcomes-leave-one-out",
    "prediction-map-leave-one-out",
    "sensitivity",
]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def axes() -> tuple[Figure, Any]:
    figure = Figure(figsize=(4, 3))
    return figure, figure.subplots()


def legend(ax: Any) -> list[str]:
    return [text.get_text() for text in ax.get_legend().get_texts()]


def test_the_themes() -> None:
    assert THEMES == {"light": LIGHT, "dark": DARK}
    assert LIGHT.k1 == LIGHT.accent == LIGHT.series[0] == "#2a78d6"
    assert LIGHT.k2 == "#eb6834"
    assert LIGHT.slot(2) == "#1baf7a"
    assert LIGHT.slot(len(LIGHT.series)) == LIGHT.context  # beyond the palette: grey, not a new hue
    assert DARK.surface == "#1a1a19"
    assert len(DARK.series) == len(LIGHT.series)
    assert len(DARK.ramp) == len(LIGHT.ramp)


def test_the_figures_of_the_experiment(experiment_view: RunView) -> None:
    specs = figure_specs(experiment_view)
    assert [spec.key for spec in specs] == EXPERIMENT_FIGURES
    for spec in specs:
        assert spec.title
        assert spec.size[0] <= 7.2  # the width of a page
        figure = render(spec)
        assert figure.get_axes(), spec.key
        assert to_hex(figure.get_facecolor()) == LIGHT.surface
        assert figure.get_size_inches().tolist() == pytest.approx(list(spec.size))


def test_the_dark_theme(experiment_view: RunView) -> None:
    for spec in figure_specs(experiment_view):
        figure = render(spec, DARK)
        assert to_hex(figure.get_facecolor()) == DARK.surface
        for ax in figure.get_axes():
            if ax.get_visible() and ax.axison:
                assert to_hex(ax.get_facecolor()) == DARK.surface, spec.key


def test_the_files(experiment_view: RunView, tmp_path: Path) -> None:
    written = save_figures(experiment_view, tmp_path / "figures", dpi=60)
    assert [p.name for p in written] == [
        f"{key}.{extension}" for key in EXPERIMENT_FIGURES for extension in ("png", "svg")
    ]
    for path in written:
        data = path.read_bytes()
        if path.suffix == ".png":
            assert data.startswith(PNG_SIGNATURE), path.name
        else:
            text = data.decode("utf-8")
            assert "<svg" in text, path.name
            assert "<dc:date>" not in text, path.name  # no time stamp: the same bytes every time
    only_png = save_figures(experiment_view, tmp_path / "png", ("png",), theme=DARK, dpi=40)
    assert {p.suffix for p in only_png} == {".png"}


def test_the_svg_is_reproducible(experiment_view: RunView, tmp_path: Path) -> None:
    spec = next(s for s in figure_specs(experiment_view) if s.key == "margins")
    first = save_figure(render(spec), tmp_path / "a" / "margins.svg")
    second = save_figure(render(spec), tmp_path / "b" / "margins.svg")
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_figures_of_other_shapes(name: str) -> None:
    view = shape(name)
    specs = figure_specs(view)
    keys = [spec.key for spec in specs]
    assert len(set(keys)) == len(keys)
    for spec in specs:
        figure = render(spec)
        assert figure.get_axes(), spec.key
    assert ("margins" in keys) == ("margin-widths" in keys) == (view.hag.p > 0)
    assert ("hag-candidates" in keys) == any(it.candidates.size for it in view.hag.iterations)
    assert "sensitivity" not in keys
    for protocol in view.result.protocols:
        assert f"outcomes-{protocol.protocol}" in keys


def test_which_figures_a_run_has() -> None:
    numeric = [spec.key for spec in figure_specs(shape("numeric"))]
    assert "prediction-map-leave-one-out" in numeric
    assert "prediction-map-hold-out" in numeric
    assert "prediction-map-resubstitution" not in numeric
    nominal = [spec.key for spec in figure_specs(shape("nominal"))]
    assert "margins" not in nominal  # p = 0: no latent feature
    assert "hag-candidates" not in nominal
    assert "prediction-map-repeated-k-fold" not in nominal  # several decisions per object
    heart = [spec.key for spec in figure_specs(shape("heart270"))]
    assert "prediction-map-stratified-k-fold" not in heart  # 270 objects do not fit an axis


def test_drawing_on_given_axes(experiment_view: RunView) -> None:
    # the draw functions take axes, so the GUI can place them on its own canvases
    view = experiment_view
    figure, ax = axes()
    draw_distances(ax, view, 1)
    assert ax.get_title() == "ρ_I(Sᵢ, Sⱼ)"
    assert len(figure.get_axes()) == 2  # the matrix and its colour bar
    _, ax = axes()
    draw_neighbourhoods(ax, view, 0, DARK)
    assert [t.get_text() for t in ax.get_yticklabels()][:2] == ["S₁", "S₂"]
    _, ax = axes()
    draw_feature_quality(ax, view, "omega")
    assert next(t.get_text() for t in ax.get_xticklabels()).startswith("a₁")
    _, ax = axes()
    draw_candidates(ax, view, 0)
    assert ax.get_title() == "iteration 1"
    assert [t.get_text() for t in ax.get_xticklabels()][:3] == ["a₁", "a₂", "a₃"]
    _, ax = axes()
    draw_criterion(ax, view)
    assert ax.get_title() == "HAG criterion (stops: |TUPLAM| = ϰ)"
    assert ax.get_ylabel() == "crit = min θ/γ"
    _, ax = axes()
    draw_margin(ax, view, 0, True)
    assert ax.get_title().split("\n") == ["r₁", "width +0.1482"]
    _, ax = axes()
    draw_margin_widths(ax, view)
    assert legend(ax) == ["without majorizer", "with majorizer ϕ"]
    loo = view.result.protocol("leave-one-out")
    _, ax = axes()
    draw_roc(ax, view, loo)
    assert legend(ax) == ["CS-model · AUC 0.271", "baselines · AUC 0.000–0.646", "chance"]
    _, ax = axes()
    draw_confusion(ax, view, loo)
    assert ax.get_title() == "leave-one-out"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["class 1", "class 2", "refusal"]
    _, ax = axes()
    draw_outcomes(ax, view, loo)
    assert next(t.get_text() for t in ax.get_yticklabels()) == "CS-model"
    # status colours never stand alone: the legend names them
    assert legend(ax) == ["correct", "wrong", "refused (0)"]
    _, ax = axes()
    draw_prediction_map(ax, view, loo)
    assert [t.get_text() for t in ax.get_yticklabels()][:2] == ["true class", "CS-model"]
    assert legend(ax)[:2] == ["K1 (class 1)", "K2 (class 2)"]
    _, ax = axes()
    draw_sensitivity(ax, view)
    assert ax.get_legend() is not None


def test_many_panels_are_capped() -> None:
    literal = shape("literal")
    assert len(literal.hag.iterations) <= MAX_PANELS
    spec = next(s for s in figure_specs(literal) if s.key == "synthetic-features")
    figure = render(spec)
    # 354 features are not named one by one along the axis
    assert all(len(ax.get_xticklabels()) < 60 for ax in figure.get_axes())


def test_the_context_of_one_object(experiment_view: RunView) -> None:
    view = experiment_view
    found = view.trace.operators[0]
    figure = object_context_figure(
        view, 0, "S₃", found.distances[2], found.order[2], own_class=int(view.trace.class_index[2])
    )
    ax = figure.get_axes()[0]
    assert ax.get_title() == "ρ: neighbourhoods of S₃  (radius = distance)"
    circles = [patch for patch in ax.patches if isinstance(patch, Circle)]
    assert len(circles) == 2  # the permitted k: 3 and 5
    # the circle of k passes through the k-th neighbour: the neighbourhoods are nested
    radii = [float(found.distances[2][found.order[2][k - 1]]) for k in (3, 5)]
    assert [circle.get_radius() for circle in circles] == radii
    assert radii[0] <= radii[1]
    texts = [text.get_text() for text in ax.texts]
    assert [text for text in texts if text.startswith("k = ")] == ["k = 3", "k = 5"]
    assert "S₃" in texts  # the object at the centre
    assert {"S₁", "S₂", "S₉"} <= set(texts)  # its neighbours are named
    assert "S₁₀" not in texts  # the ninth neighbour: beyond the largest k and three more
    assert legend(ax) == ["K1 (class 1)", "K2 (class 2)"]
    assert to_hex(figure.get_facecolor()) == LIGHT.surface


def test_the_context_on_large_data() -> None:
    view = shape("literal")  # 270 objects, 118 permitted k
    found = view.trace.operators[1]
    figure = object_context_figure(view, 1, "S", found.distances[0], found.order[0], DARK)
    ax = figure.get_axes()[0]
    assert ax.get_title().startswith("ρ_I: neighbourhoods of S ")
    circles = [patch for patch in ax.patches if isinstance(patch, Circle)]
    assert len(circles) == 6  # the first six of the neighbourhoods among the 40 neighbours shown
    texts = [text.get_text() for text in ax.texts]
    assert [text for text in texts if text.startswith("k = ")] == [
        f"k = {k}" for k in (3, 5, 7, 9, 11, 13)
    ]
    assert [text for text in texts if not text.startswith("k = ")] == ["S"]  # too many to name
    assert to_hex(figure.get_facecolor()) == DARK.surface
