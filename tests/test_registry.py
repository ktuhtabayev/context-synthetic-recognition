"""Plug-in registry: names, aliases, errors, entry points."""

from dataclasses import dataclass
from importlib.metadata import EntryPoint

import pytest

from context_synthetic_recognition.core import registry as registry_module
from context_synthetic_recognition.core.registry import Registry, normalize_name
from context_synthetic_recognition.errors import RegistryError


def euclidean() -> str:
    return "euclidean"


def manhattan() -> str:
    return "manhattan"


@dataclass(frozen=True)
class MinkowskiParams:
    p: float = 2.0


def test_register_and_get_with_normalised_names() -> None:
    metrics: Registry[object] = Registry("metrics")

    @metrics.register("Zhuravlev", aliases=("juravlev",), summary="ρ(x, y) on I ∪ J")
    def zhuravlev() -> str:
        return "zhuravlev"

    assert metrics.get("zhuravlev") is zhuravlev
    assert metrics.get(" JURAVLEV ") is zhuravlev
    assert metrics.info("Zhuravlev").summary == "ρ(x, y) on I ∪ J"
    assert "juravlev" in metrics
    assert 3 not in metrics
    assert metrics.names() == ["zhuravlev"]
    assert len(metrics) == 1


@pytest.mark.parametrize(
    ("raw", "key"),
    [("Min Max", "min-max"), ("min_max", "min-max"), ("  z-score ", "z-score")],
)
def test_normalize_name(raw: str, key: str) -> None:
    assert normalize_name(raw) == key


def test_empty_name_is_rejected() -> None:
    with pytest.raises(RegistryError, match="must not be empty"):
        normalize_name("  ")


def test_unknown_name_lists_the_available_ones() -> None:
    metrics: Registry[object] = Registry("metrics")
    metrics.add("euclidean", euclidean)
    metrics.add("manhattan", manhattan)
    with pytest.raises(
        RegistryError, match=r"Unknown metrics 'cosine'\. Available: euclidean, manhattan"
    ):
        metrics.get("cosine")
    with pytest.raises(RegistryError, match="Available: none"):
        Registry[object]("normalizers").get("minmax")


def test_duplicates_and_alias_clashes_are_errors() -> None:
    metrics: Registry[object] = Registry("metrics")
    metrics.add("euclidean", euclidean, aliases=("l2",))
    with pytest.raises(RegistryError, match="'euclidean' is already registered"):
        metrics.add("Euclidean", manhattan)
    with pytest.raises(RegistryError, match="'l2' is already registered"):
        metrics.add("manhattan", manhattan, aliases=("l2",))


def test_replace_swaps_the_plugin_and_its_aliases() -> None:
    metrics: Registry[object] = Registry("metrics")
    metrics.add("minkowski", euclidean, aliases=("lp",))
    metrics.add("minkowski", manhattan, replace=True, params_type=MinkowskiParams)
    assert metrics.get("minkowski") is manhattan
    assert metrics.info("minkowski").params_type is MinkowskiParams
    assert "lp" not in metrics


def test_iteration_and_repr_keep_registration_order() -> None:
    metrics: Registry[object] = Registry("metrics")
    metrics.add("manhattan", manhattan)
    metrics.add("euclidean", euclidean)
    assert [info.name for info in metrics] == ["manhattan", "euclidean"]
    assert repr(metrics) == "Registry('metrics', ['manhattan', 'euclidean'])"


def test_entry_points_are_loaded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    group = "context_synthetic_recognition.metrics"
    points = [
        EntryPoint("chebyshev", f"{__name__}:euclidean", group),
        EntryPoint("manhattan", f"{__name__}:manhattan", group),
    ]

    def fake_entry_points(*, group: str) -> list[EntryPoint]:
        return [point for point in points if point.group == group]

    monkeypatch.setattr(registry_module, "entry_points", fake_entry_points)
    metrics: Registry[object] = Registry("metrics")
    metrics.add("manhattan", manhattan)
    assert metrics.load_entry_points() == 1
    assert metrics.get("chebyshev") is euclidean
    assert metrics.load_entry_points() == 0
