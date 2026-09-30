"""Data layer: schema, loaders of every format, format detection, built-in datasets."""

import logging
from pathlib import Path

import numpy as np
import openpyxl
import pytest

from context_synthetic_recognition.config import DatasetConfig
from context_synthetic_recognition.data import (
    DATASETS,
    LOADERS,
    Dataset,
    FeatureType,
    detect_format,
    load_builtin,
    load_dataset,
    load_from_config,
)
from context_synthetic_recognition.data.schema import as_labels
from context_synthetic_recognition.errors import DatasetError

from .conftest import KNN_WORKBOOK, WORKBOOK

QUANT, NOM = FeatureType.QUANTITATIVE, FeatureType.NOMINAL

# ---------------------------------------------------------------- schema


def small(**changes: object) -> Dataset:
    fields: dict[str, object] = {
        "X": [[1.0, 0.0], [2.0, 1.0], [3.0, 1.0]],
        "y": [1, 2, 2],
        "feature_types": (QUANT, NOM),
    }
    fields.update(changes)
    return Dataset(**fields)  # type: ignore[arg-type]


def test_defaults_names_classes_and_masks() -> None:
    d = small()
    assert (d.m, d.n) == (3, 2)
    assert d.feature_names == ("x₁", "x₂")
    assert d.object_ids == ("S₁", "S₂", "S₃")
    assert d.classes == (1, 2)
    assert d.class_sizes == (1, 2)
    assert d.class_index.tolist() == [0, 1, 1]
    assert d.quantitative.tolist() == [True, False]
    assert d.nominal.tolist() == [False, True]
    assert d.type_flags == (1, 0)
    assert repr(d) == "Dataset('dataset', m=3, n=2, classes={1: 1, 2: 2})"


def test_arrays_are_read_only_copies() -> None:
    X = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 1.0]])
    d = small(X=X)
    X[0, 0] = 99.0
    assert d.X[0, 0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        d.X[0, 0] = 5.0
    with pytest.raises(ValueError, match="read-only"):
        d.y[0] = 5


def test_string_labels_are_ordered_alphabetically() -> None:
    d = small(y=["yes", "no", "yes"])
    assert d.classes == ("no", "yes")
    assert d.class_index.tolist() == [1, 0, 1]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"X": [1.0, 2.0]}, "must be 2-D"),
        ({"X": np.empty((0, 2)), "y": []}, "at least one object"),
        (
            {"X": [[1.0, np.nan], [2.0, 1.0], [3.0, 1.0]]},
            r"missing or non-finite values .* \(1, 2\)",
        ),
        ({"y": [1, 2]}, "2 class labels for 3 objects"),
        ({"feature_types": (QUANT,)}, "1 feature types for 2 features"),
        ({"feature_names": ("a", "a")}, r"feature names must be unique \(repeated: a\)"),
        ({"object_ids": ("S", "T")}, "2 object ids for 3 entries"),
        ({"categories": {"z": ("a",)}}, "categories given for unknown features"),
        ({"y": [1, "a", 2]}, "all integers or all strings"),
    ],
)
def test_schema_errors(changes: dict[str, object], message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        small(**changes)


def test_feature_type_flags() -> None:
    assert FeatureType.from_flag(1) is QUANT
    assert FeatureType.from_flag(0) is NOM
    assert (QUANT.flag, NOM.flag) == (1, 0)
    with pytest.raises(DatasetError, match="must be 1"):
        FeatureType.from_flag(2)


def test_as_labels_types() -> None:
    assert as_labels([1, 2]).dtype == np.int64
    assert as_labels(np.array([1, 2], dtype=np.int32)).tolist() == [1, 2]
    assert as_labels(["a"]).dtype.kind == "U"
    assert as_labels([]).size == 0


def test_subset_and_type_override() -> None:
    d = small(feature_names=("age", "sex"), categories={"sex": ("f", "m")})
    fold = d.subset([2, 0], name="fold")
    assert fold.object_ids == ("S₃", "S₁")
    assert fold.X[:, 0].tolist() == [3.0, 1.0]
    assert fold.name == "fold"
    assert fold.categories == {"sex": ("f", "m")}
    retyped = d.with_feature_types({"sex": "quantitative"})
    assert retyped.feature_types == (QUANT, QUANT)
    with pytest.raises(DatasetError, match="unknown features"):
        d.with_feature_types({"height": "nominal"})


def test_content_hash_ignores_name_and_source_only() -> None:
    d = small()
    assert d.content_hash() == small(name="other", source="x.csv").content_hash()
    assert d.content_hash() != small(y=[2, 1, 2]).content_hash()
    assert d.content_hash() != small(feature_types=(QUANT, QUANT)).content_hash()
    assert d.content_hash() != small(X=[[1.0, 0.0], [2.0, 1.0], [3.5, 1.0]]).content_hash()


# ---------------------------------------------------------------- workbook and template layout


def test_workbook_dataset_sheet(experiment: Dataset) -> None:
    assert experiment.name == "Heart + Disease (10, 13, 2)"
    assert (experiment.m, experiment.n) == (10, 13)
    assert experiment.type_flags == (1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0)
    assert experiment.y.tolist() == [2, 1, 2, 1, 1, 1, 2, 2, 2, 2]
    assert experiment.class_sizes == (4, 6)
    assert experiment.X[0].tolist() == [70, 1, 4, 130, 322, 0, 2, 109, 0, 2.4, 2, 3, 3]
    assert experiment.object_ids[9] == "S₁₀"


def test_builtin_datasets_match_the_workbook(experiment: Dataset) -> None:
    ten = load_builtin("heart-disease-10")
    assert ten.content_hash() == experiment.content_hash()
    big = load_builtin("heart-disease-270")
    assert (big.m, big.n, big.class_sizes) == (270, 13, (150, 120))
    assert big.type_flags == experiment.type_flags
    assert DATASETS.names() == ["heart-disease-10", "heart-disease-270"]


def test_detect_format(tmp_path: Path) -> None:
    assert detect_format(WORKBOOK) == "cs-workbook"
    assert detect_format(KNN_WORKBOOK) == "cs-workbook"
    extended = tmp_path / "e.csv"
    extended.write_text("2,1,2,,\n1,1\n2,2\n1,\n", encoding="utf-8")
    assert detect_format(extended) == "template-extended"
    table = tmp_path / "t.csv"
    table.write_text("a,class\n1,1\n", encoding="utf-8")
    assert detect_format(table) == "csv"
    assert detect_format(tmp_path / "x.dat") == "template-extended"
    assert detect_format(tmp_path / "x.parquet") == "parquet"
    with pytest.raises(DatasetError, match=r"unknown dataset extension '\.json'"):
        detect_format(tmp_path / "x.json")


def test_plain_excel_table_is_not_a_cs_workbook(tmp_path: Path) -> None:
    path = tmp_path / "table.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Data"
    for row in (["age", "sex", "class"], [50, "m", "yes"], [60, "f", "no"], [55, "m", "no"]):
        sheet.append(row)
    workbook.save(path)
    assert detect_format(path) == "xlsx"
    d = load_dataset(path, feature_types={"sex": "nominal"})
    assert d.feature_names == ("age", "sex")
    assert d.classes == ("no", "yes")
    assert d.categories == {"sex": ("f", "m")}
    assert d.X[:, 1].tolist() == [1.0, 0.0, 1.0]


def write_extended(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_template_csv_and_dat_agree(tmp_path: Path) -> None:
    csv_file = write_extended(tmp_path / "d.csv", "3,2,2,\n1.5,0,1\n2.5,1,2\n3,1,2\n1,0,\n")
    dat_file = write_extended(tmp_path / "d.dat", "3 2 2\n1,5\t0 1\n2,5 1 2\n3 1 2\n1 0 0\n")
    a, b = load_dataset(csv_file), load_dataset(dat_file)
    assert a.content_hash() == b.content_hash()
    assert a.X[:, 0].tolist() == [1.5, 2.5, 3.0]
    assert a.feature_types == (QUANT, NOM)
    assert a.name == "d"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "the file is empty"),
        ("a,b\n1,2\n", "first row must be 'm, n, c'"),
        ("3,2,2\n1,0,1\n2,1,2\n1,0\n", "3 objects, type flags"),
        ("2,2,2\n1,0\n2,1,2\n1,0\n", "object row 1 has 2 values, expected 3"),
        ("2,2,2\n1,x,1\n2,1,2\n1,0\n", r"'x' is not a number"),
        ("2,2,2\n1,0,1\n2,1,2\n1\n", "the type row has 1 flags"),
        ("2,2,2\n1,0,1\n2,1,2\n1,2\n", "flags must be 1 or 0"),
        ("2,2,3\n1,0,1\n2,1,2\n1,0\n", "the header says c = 3 classes, the data have 2"),
    ],
)
def test_template_layout_errors(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        load_dataset(write_extended(tmp_path / "bad.csv", text), "template-extended")


def test_feature_type_override_for_typed_formats(tmp_path: Path) -> None:
    path = write_extended(tmp_path / "d.csv", "2,2,2\n1,0,1\n2,1,2\n1,0\n")
    d = load_dataset(path, feature_types={"x₂": "quantitative"})
    assert d.feature_types == (QUANT, QUANT)


# ---------------------------------------------------------------- tables with a header row


def test_csv_table_with_types_ids_and_inference(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "patients.csv"
    path.write_text(
        "id;age;chol;sex;outcome\nP1;50;1,5;m;sick\nP2;60;2,5;f;healthy\nP3;70;3;m;sick\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        d = load_dataset(
            path, class_column="outcome", id_column="id", feature_types={"age": "quantitative"}
        )
    assert d.feature_names == ("age", "chol", "sex")
    assert d.feature_types == (QUANT, QUANT, NOM)
    assert d.object_ids == ("P1", "P2", "P3")
    assert d.X[:, 1].tolist() == [1.5, 2.5, 3.0]
    assert d.classes == ("healthy", "sick")
    assert "feature types inferred" in caplog.text


@pytest.mark.parametrize(
    ("text", "options", "message"),
    [
        ("a,class\n", {}, "the table has no rows"),
        ("a,a\n1,2\n", {}, "column names must be non-empty and unique"),
        ("a,class\n1,1\n", {"class_column": "label"}, "no class column 'label'"),
        ("a,class\n1,1\n", {"id_column": "id"}, "no id column 'id'"),
        ("a,class\n,1\n", {}, r"missing values at row 1 'a'"),
        ("class\n1\n", {}, "no feature columns"),
        ("a,class\nx,1\n", {"feature_types": {"a": "quantitative"}}, "quantitative feature 'a'"),
        ("a,class\n1,1\n", {"feature_types": {"b": "nominal"}}, "unknown columns"),
    ],
)
def test_csv_table_errors(
    tmp_path: Path, text: str, options: dict[str, object], message: str
) -> None:
    path = tmp_path / "t.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(DatasetError, match=message):
        load_dataset(path, "csv", **options)  # type: ignore[arg-type]


def test_parquet_table(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "d.parquet"
    table = pa.table({"x": [1.0, 2.0, 4.0], "colour": ["red", "blue", "red"], "y": [1, 2, 2]})
    parquet.write_table(table, path)
    d = load_dataset(path)
    assert d.feature_types == (QUANT, NOM)
    assert d.categories == {"colour": ("blue", "red")}
    assert d.classes == (1, 2)


def test_missing_file_and_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="dataset file not found"):
        load_dataset(tmp_path / "none.csv")
    path = tmp_path / "t.csv"
    path.write_text("a,class\n1,1\n", encoding="utf-8")
    with pytest.raises(Exception, match="Unknown loaders 'json'"):
        load_dataset(path, "json")


def test_workbook_errors(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="no sheet 'Data'"):
        load_dataset(WORKBOOK, "cs-workbook", sheet="Data")
    broken = tmp_path / "broken.xlsx"
    broken.write_bytes(b"not a zip file")
    with pytest.raises(DatasetError, match="cannot read the Excel file"):
        load_dataset(broken, "cs-workbook")
    path = tmp_path / "wb.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Dataset"
    sheet.append(["title"])
    sheet.append(["№", None, "Class"])
    sheet.append([None, 1.0, 1])
    workbook.save(path)
    with pytest.raises(DatasetError, match="feature-type row after the objects is missing"):
        load_dataset(path, "cs-workbook")


def test_load_from_config_resolves_relative_paths(tmp_path: Path) -> None:
    write_extended(tmp_path / "d.csv", "2,1,2\n1,1\n2,2\n1\n")
    d = load_from_config(DatasetConfig(path="d.csv"), base_dir=tmp_path)
    assert d.m == 2
    default = load_from_config(DatasetConfig())  # no path: the default dataset (ADR-035)
    assert (default.name, default.m) == ("default", 10)


def test_loader_registry_names_match_the_configuration() -> None:
    formats = set(DatasetConfig.model_fields["format"].annotation.__args__)  # type: ignore[union-attr]
    names = {name for info in LOADERS for name in (info.name, *info.aliases)}
    assert formats - {"auto"} == names
