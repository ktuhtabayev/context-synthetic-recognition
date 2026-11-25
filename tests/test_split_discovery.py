"""Discovery of the split files: every file of the split folder is a sample (ADR-090)."""

import hashlib
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from context_synthetic_recognition.comparison import COMPARISON_SPLITS, SplitSpec
from context_synthetic_recognition.comparison.stages import compare_split, method_specs
from context_synthetic_recognition.data.splits import NAME_PATTERN, parse_split_name
from context_synthetic_recognition.errors import DatasetError
from context_synthetic_recognition.services.split_discovery import (
    SplitCatalog,
    discover_splits,
    split_files,
)

from .conftest import ROOT

SPLITS_FOLDER = ROOT / "datasets" / "splits"

CHECKSUMS = {
    "German/German (1000, 20, 2) — training 336 (zhuravlyov).dat": (
        "de91b117361122bdf22430a56ea4b2f06582739e037845d2e799a8da46b56135"
    ),
    "German/German (1000, 7, 2) — training 341 (euclidean).dat": (
        "2a11746d5b41579140337ee6bc1b53879d06f80c65d3fe32eff00bd857db3052"
    ),
    "German/German (1000, 7, 2) — training 367 (chebyshev).dat": (
        "1c818e6772cae85394d0134e3e29a0d6116d58acee0bb9964d1327f00948f772"
    ),
    "Heart-Disease/Heart-Disease (270, 13, 2) — training 76 (zhuravlyov).dat": (
        "ef9e5c65815b5f7c8e35dc0ad3d462c8394d10e05d299ca723540e202c153833"
    ),
    "Spambase/Spambase (2102, 57, 2) — training 601 (euclidean).dat": (
        "83f702042e2257832d966676e7287645b83fa4c8710650ca19098678676875af"
    ),
    "Spambase/Spambase (2102, 57, 2) — training 722 (euclidean).dat": (
        "e7aee964e8622ce380847b9d8abd371eecb2d027c8a0a22035e3432b7ceed0b8"
    ),
}
"""The six split files of today, pinned; a file added later is checked, not pinned."""


def write_split(
    path: Path,
    X: Sequence[Sequence[float]],
    classes: Sequence[int],
    markers: Sequence[int],
    types: Sequence[int],
    *,
    header: str | None = None,
    type_row: str | None = None,
) -> Path:
    """A split file in the convention (ADR-082): header, object rows, type row."""
    m, n = len(X), len(types)
    lines = [header or f"{m}\t{n}\t{len(set(classes))}"]
    lines += [
        "\t".join([*(f"{v:g}" for v in row), str(c), str(s)])
        for row, c, s in zip(X, classes, markers, strict=True)
    ]
    lines.append(type_row or "\t".join(map(str, types)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


def toy(
    folder: Path,
    name: str,
    *,
    types: Sequence[int] = (1, 1, 0),
    training: int = 6,
    m: int = 12,
    **kwargs: str,
) -> Path:
    """A small two-class split file: ``training`` objects first, then the test objects."""
    rng = np.random.default_rng(len(name))
    classes = [1 + (i % 2) for i in range(m)]
    X = [
        [
            float(rng.integers(0, 3)) if t == 0 else round(float(rng.normal(c, 1.0)), 3)
            for t in types
        ]
        for c in classes
    ]
    markers = [0 if i < training else 1 for i in range(m)]
    return write_split(folder / name, X, classes, markers, types, **kwargs)


@pytest.fixture(scope="module")
def catalog() -> SplitCatalog:
    return discover_splits(SPLITS_FOLDER)


# ---------------------------------------------------------------- today's six files


def test_the_six_files_keep_their_bytes() -> None:
    found = {p.relative_to(SPLITS_FOLDER).as_posix() for p in SPLITS_FOLDER.rglob("*.dat")}
    assert set(CHECKSUMS) <= found
    for name, digest in CHECKSUMS.items():
        assert hashlib.sha256((SPLITS_FOLDER / name).read_bytes()).hexdigest() == digest, name


def test_every_file_of_the_folder_is_found(catalog: SplitCatalog) -> None:
    assert catalog.problem == ""
    keys = [f.key for f in catalog.files]
    assert keys[:6] == [
        "German-336",
        "German-341",
        "German-367",
        "Heart-Disease-76",
        "Spambase-601",
        "Spambase-722",
    ]
    for file in catalog.files:  # a file added later is checked, not pinned
        assert file.available, file.reason
        assert file.name is not None
        assert file.sha256 == hashlib.sha256((SPLITS_FOLDER / file.file).read_bytes()).hexdigest()
        assert file.split is not None
        assert file.split.training.size == file.name.training
    assert [name for name, _ in catalog.datasets()][:3] == ["German", "Heart-Disease", "Spambase"]
    german = catalog.require("German-341")
    assert german.label == "German · Euclidean split · training 341 (173, 168)"
    assert german.short_label == "German · Euclidean split · training 341"
    assert german.details == "7 features · training 341 (173, 168) · test 659"
    assert catalog.require("Spambase-722").label == (
        "Spambase · Euclidean split · training 722 (467, 255)"
    )
    assert catalog.keys[:6] == tuple(keys[:6])


def test_the_keys_of_the_earlier_versions(catalog: SplitCatalog) -> None:
    for old, key in {
        "336": "German-336",
        "341": "German-341",
        "367": "German-367",
        "76": "Heart-Disease-76",
        "601": "Spambase-601",
        "722": "Spambase-722",
    }.items():
        assert catalog.resolve(old).key == key
    assert catalog.resolve("german-341").key == "German-341"
    with pytest.raises(DatasetError, match="unknown split '999'"):
        catalog.resolve("999")
    with pytest.raises(DatasetError, match="unknown split 'x'"):
        split_files(catalog, ["x"])


def test_the_name_pattern() -> None:
    name = parse_split_name("German (1000, 7, 2) — training 341 (euclidean).dat")
    assert (name.dataset, name.m, name.n, name.c, name.training, name.metric) == (
        "German",
        1000,
        7,
        2,
        341,
        "euclidean",
    )
    assert name.key == "German-341"
    assert parse_split_name("My data (10, 2, 2) — training 4 (Chebyshev).DAT").key == "My data-4"
    for bad in (
        "German (1000, 7, 2) - training 341 (euclidean).dat",  # a hyphen, not a dash
        "German (1000, 7) — training 341 (euclidean).dat",
        "German (1000, 7, 2) — training 341.dat",
        "notes.dat",
        "German (0, 7, 2) — training 341 (euclidean).dat",
    ):
        with pytest.raises(DatasetError, match="name"):
            parse_split_name(bad)
    assert NAME_PATTERN == "<Dataset> (<m>, <n>, <c>) — training <t> (<metric>).dat"


# ---------------------------------------------------------------- the methods of a split


def test_the_method_rule(catalog: SplitCatalog) -> None:
    assert COMPARISON_SPLITS == ("German-336", "German-341", "German-367")
    keys = {f.key: [m.key for m in method_specs(f.spec)] for f in catalog.files[:6]}
    standard = ["svm", "random-forest"]
    knn = [f"knn-{m}-{k}" for m in ("euclidean", "chebyshev", "canberra") for k in (3, 5)]
    cs_model = [
        f"cs-model-{m}-{a}" for m in ("euclidean", "chebyshev", "canberra") for a in (0.1, 0.2)
    ]
    zhuravlyov_knn = ["knn-zhuravlyov-3", "knn-zhuravlyov-5"]
    zhuravlyov_cs = ["cs-model-zhuravlyov-0.1", "cs-model-zhuravlyov-0.2"]
    # 336 keeps its Zhuravlyov rows; the other five keep theirs (none)
    assert keys["German-336"] == standard + knn + zhuravlyov_knn + cs_model + zhuravlyov_cs
    for key in ("German-341", "German-367", "Heart-Disease-76", "Spambase-601", "Spambase-722"):
        assert keys[key] == standard + knn + cs_model, key
    spec = catalog.require("Heart-Disease-76").spec
    assert spec.extra_knn == ()
    assert SplitSpec.of("German-336", "German", "x.dat", "zhuravlyov", 336).extra_knn == (
        "zhuravlyov",
    )


# ---------------------------------------------------------------- a folder with new files


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """The six files and a set of new ones, good and bad."""
    root = tmp_path / "splits"
    shutil.copytree(SPLITS_FOLDER, root)
    toy(root / "Toy", "Toy (12, 3, 2) — training 6 (euclidean).dat")
    toy(root / "Nominal", "Nominal (12, 3, 2) — training 6 (zhuravlyov).dat", types=(0, 0, 0))
    toy(root / "Toy", "notes.dat")  # not the name pattern
    toy(root / "Toy", "Wrong (13, 3, 2) — training 6 (euclidean).dat")  # m of the name
    toy(root / "Toy", "Count (12, 3, 2) — training 5 (euclidean).dat")  # the markers give 6
    toy(root / "Toy", "Metric (12, 3, 2) — training 6 (nearness).dat")
    toy(root / "Toy", "Types (12, 3, 2) — training 6 (euclidean).dat", type_row="1\t2\t0")
    toy(root / "Head", "Head (12, 3, 2) — training 6 (euclidean).dat", header="12\t4\t2")
    toy(root / "Dup", "Dup (12, 3, 2) — training 6 (euclidean).dat")
    toy(root / "Dup", "Dup (12, 3, 2) — training 6 (chebyshev).dat")
    return root


def test_a_new_dataset_appears_without_a_change(folder: Path) -> None:
    catalog = discover_splits(folder)
    toy_file = catalog.require("Toy-6")
    assert toy_file.label == "Toy · Euclidean split · training 6 (3, 3)"
    assert toy_file.details == "3 features · training 6 (3, 3) · test 6"
    assert [name for name, _ in catalog.datasets()] == [
        "Count",
        "Dup",
        "German",
        "Head",
        "Heart-Disease",
        "Metric",
        "Nominal",
        "Spambase",
        "Toy",
        "Types",
        "Wrong",
    ]
    split = toy_file.split
    assert split is not None
    result = compare_split(toy_file.spec, split)
    assert result.training_sizes == (6, 3, 3)
    assert result.methods[0].test is not None
    assert all(f.available for f in catalog.files if f.dataset in ("German", "Spambase"))


def reasons(catalog: SplitCatalog) -> dict[str, tuple[str, int | None]]:
    return {Path(f.file).name: (f.reason, f.row) for f in catalog.files if not f.available}


def test_bad_files_are_unavailable_with_their_reason(folder: Path) -> None:
    found = reasons(discover_splits(folder))
    assert set(found) == {
        "notes.dat",
        "Wrong (13, 3, 2) — training 6 (euclidean).dat",
        "Count (12, 3, 2) — training 5 (euclidean).dat",
        "Metric (12, 3, 2) — training 6 (nearness).dat",
        "Types (12, 3, 2) — training 6 (euclidean).dat",
        "Head (12, 3, 2) — training 6 (euclidean).dat",
        "Dup (12, 3, 2) — training 6 (euclidean).dat",
        "Dup (12, 3, 2) — training 6 (chebyshev).dat",
    }
    reason, row = found["notes.dat"]
    assert "does not follow" in reason
    assert NAME_PATTERN in reason
    assert row is None
    reason, row = found["Wrong (13, 3, 2) — training 6 (euclidean).dat"]
    assert "row 1 (the header) says m n c = 12 3 2, the name says (13, 3, 2)" in reason
    assert row == 1
    reason, _ = found["Count (12, 3, 2) — training 5 (euclidean).dat"]
    assert "the split markers give 6 training objects, the name says training 5" in reason
    reason, _ = found["Metric (12, 3, 2) — training 6 (nearness).dat"]
    assert "unknown metric 'nearness'" in reason
    reason, row = found["Types (12, 3, 2) — training 6 (euclidean).dat"]
    assert "row 14 (the type row), column 2: a type flag must be 0 (nominal) or 1" in reason
    assert row == 14
    reason, row = found["Head (12, 3, 2) — training 6 (euclidean).dat"]
    assert "row 2 (object 1) has 5 values, expected n + 2 = 6" in reason
    assert row == 2
    reason, _ = found["Dup (12, 3, 2) — training 6 (euclidean).dat"]
    assert "the key Dup-6 is also the key of Dup/Dup (12, 3, 2) — training 6 (chebyshev).dat" in (
        reason
    )
    catalog = discover_splits(folder)
    with pytest.raises(DatasetError, match="also the key of"):
        catalog.require("Dup-6")
    with pytest.raises(DatasetError, match="unknown split 'x'"):
        catalog.resolve("x")


def test_an_ambiguous_training_size(folder: Path) -> None:
    catalog = discover_splits(folder)
    with pytest.raises(DatasetError, match="split '6' is ambiguous: "):
        catalog.resolve("6")
    assert catalog.resolve("336").key == "German-336"


def test_a_nominal_only_dataset(folder: Path) -> None:
    catalog = discover_splits(folder)
    nominal = catalog.require("Nominal-6")
    assert nominal.available
    split = nominal.split
    assert split is not None
    specs = {m.key: m for m in method_specs(nominal.spec, split.dataset.n, 0, 6)}
    for key in ("knn-euclidean-3", "knn-canberra-5", "cs-model-chebyshev-0.1"):
        assert specs[key].unavailable == "no quantitative features", key
    assert specs["svm"].unavailable == ""
    result = compare_split(nominal.spec, split)
    knn = result.method("knn-euclidean-3")
    assert knn.test is None
    assert knn.undefined == "no quantitative features"
    assert result.method("svm").test is not None
    assert {c.undefined for c in result.cs_model} == {"no quantitative features"}


def test_reading_only_some_files(folder: Path) -> None:
    catalog = discover_splits(folder, read={"Toy-6"})
    assert catalog.require("Toy-6").available
    german = catalog.find("German-336")
    assert german is not None
    assert german.split is None
    assert not german.available
    assert german.reason == ""
    with pytest.raises(DatasetError, match="was not read"):
        catalog.require("German-336")


def test_a_missing_folder(tmp_path: Path) -> None:
    catalog = discover_splits(tmp_path / "none")
    assert catalog.files == ()
    assert "no folder of split files" in catalog.problem
    with pytest.raises(DatasetError, match="no folder of split files"):
        catalog.resolve("336")
