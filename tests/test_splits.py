"""Split files: features, class and split marker (data/splits.py, ADR-082)."""

from pathlib import Path

import numpy as np
import pytest

from context_synthetic_recognition.data import (
    AnswerKey,
    Decisions,
    FeatureType,
    FixedSplit,
    UnlabelledSample,
    load_split_file,
)
from context_synthetic_recognition.errors import DatasetError

# the convention: header "m n c" (features only) padded with empty cells, no leading empty cell,
# tabs, LF, decimal commas, test rows between training rows, the n type flags and two empty cells
LAYOUT = "".join(
    [
        "5\t2\t2\t\n",
        "0,5\t1\t1\t0\n",
        "1,25\t2\t2\t1\n",
        "0\t1\t2\t0\n",
        "2\t3\t1\t1\n",
        "3,5\t2\t1\t0\n",
        "1\t0\t\t\n",
    ]
)
TYPE_ROW = "1\t0\t\t\n"
FIRST_OBJECT = "0,5\t1\t1\t0\n"


def write(tmp_path: Path, text: str, name: str = "split.dat") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8"))
    return path


def test_the_layout_is_read(tmp_path: Path) -> None:
    split = load_split_file(write(tmp_path, LAYOUT))
    data = split.dataset
    assert data.name == "split"
    assert data.X.tolist() == [[0.5, 1.0], [1.25, 2.0], [0.0, 1.0], [2.0, 3.0], [3.5, 2.0]]
    assert data.y.tolist() == [1, 2, 2, 1, 1]
    assert data.feature_types == (FeatureType.QUANTITATIVE, FeatureType.NOMINAL)
    assert split.training.tolist() == [0, 2, 4]
    assert split.test.tolist() == [1, 3]
    assert split.training_sizes == (3, 2, 1)
    assert split.test_sizes == (2, 1, 1)
    assert split.training_part.object_ids == ("S₁", "S₃", "S₅")
    assert split.test_part.y.tolist() == [2, 1]


def test_the_test_objects_and_their_answer_key(tmp_path: Path) -> None:
    # ADR-085: the test objects without their class, the classes kept apart
    split = load_split_file(write(tmp_path, LAYOUT))
    objects = split.test_objects()
    assert objects.name == "split: test sample"
    assert objects.objects.tolist() == [1, 3]
    assert objects.object_ids == ("S₂", "S₄")
    assert objects.X.tolist() == [[1.25, 2.0], [2.0, 3.0]]
    assert objects.feature_names == ("x₁", "x₂")
    assert objects.quantitative.tolist() == [True, False]
    assert (objects.m, objects.n) == (2, 2)
    assert not hasattr(objects, "y")
    answers = split.test_answers("key")
    assert answers.classes.tolist() == [2, 1]
    assert answers.labels == (1, 2)
    assert answers.sizes == (2, 1, 1)
    assert answers.matches(objects)
    with pytest.raises(DatasetError, match="class codes must be 1 … 2"):
        AnswerKey(answers.objects, np.array([1, 3]), (1, 2))
    with pytest.raises(DatasetError, match="2 classes for 1 objects"):
        AnswerKey(np.array([1]), np.array([1, 2]), (1, 2))
    with pytest.raises(DatasetError, match="need a \\(q × n\\) array with q = 3"):
        UnlabelledSample(np.array([0, 1, 2]), objects.X, objects.feature_types, ("a", "b"))
    with pytest.raises(DatasetError, match="2 features, 2 types, 1 names"):
        UnlabelledSample(objects.objects, objects.X, objects.feature_types, ("a",))
    assert Decisions(None).same_as(Decisions(None))
    assert not Decisions(np.array([1])).same_as(Decisions(np.array([2])))
    assert not Decisions(np.array([1]), np.array([0.5])).same_as(Decisions(np.array([1])))


@pytest.mark.parametrize(
    "variant",
    [
        LAYOUT.replace("\n", "\r\n"),  # CRLF
        LAYOUT.replace("\t\n", "\n").replace("\t\n", "\n"),  # no empty trailing cells
        LAYOUT.replace("\t\t\n", "\t\t\t\t\n"),  # more empty trailing cells
        "﻿" + LAYOUT,  # a BOM
        LAYOUT.replace("0,5", "0.5").replace("1,25", "1.25").replace("3,5", "3.5"),  # points
        LAYOUT + "\n\n",  # blank lines at the end
    ],
)
def test_variants_give_the_same_split(tmp_path: Path, variant: str) -> None:
    base = load_split_file(write(tmp_path, LAYOUT, "a.dat"))
    other = load_split_file(write(tmp_path, variant, "b.dat"))
    assert other.dataset.content_hash() == base.dataset.content_hash()
    assert other.training.tolist() == base.training.tolist()
    assert other.test.tolist() == base.test.tolist()


def test_a_name_and_feature_types_can_be_given(tmp_path: Path) -> None:
    split = load_split_file(
        write(tmp_path, LAYOUT), name="toy", feature_types={"x₂": "quantitative"}
    )
    assert split.name == "toy"
    assert split.dataset.quantitative.tolist() == [True, True]


def test_the_samples_are_read_only(tmp_path: Path) -> None:
    split = load_split_file(write(tmp_path, LAYOUT))
    with pytest.raises(ValueError, match="read-only"):
        split.training[0] = 1


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "the file is empty"),
        # the header: exactly three positive integers m, n, c
        ("5\t2\n", r"row 1 \(the header\) must hold exactly three positive integers"),
        ("5\t2\t2\t1\n", r"row 1 \(the header\) must hold exactly three"),
        ("5\t2,5\t2\n", r"row 1 \(the header\) must hold exactly three"),
        ("\t5\t2\t2\n", r"row 1 \(the header\) must hold exactly three"),
        (LAYOUT.replace("\t", " "), r"row 1 \(the header\)"),  # spaces do not separate cells
        # the old header "m, n + 1, c" counted the marker column
        (
            LAYOUT.replace("5\t2\t2\t\n", "5\t3\t2\t\n"),
            (
                r"row 2 \(object 1\) has 4 values, expected n \+ 2 = 5 \(3 features, the class, "
                r"the split marker\); the header's n counts the features only"
            ),
        ),
        # m object rows
        (LAYOUT.replace("5\t2\t2\t\n", "6\t2\t2\t\n"), "m = 6, so 8 rows are expected"),
        (LAYOUT.replace("5\t2\t2\t\n", "4\t2\t2\t\n"), "m = 4, so 6 rows are expected"),
        # n + 2 values in every object row
        (
            LAYOUT.replace(FIRST_OBJECT, "0,5\t1\t1\n"),
            r"row 2 \(object 1\) has 3 values, expected n \+ 2 = 4",
        ),
        (
            LAYOUT.replace(FIRST_OBJECT, "0,5\t1\t1\t0\t7\n"),
            r"row 2 \(object 1\) has 5 values, expected n \+ 2 = 4",
        ),
        (
            LAYOUT.replace(FIRST_OBJECT, "\t0,5\t1\t1\t0\n"),
            r"row 2 \(object 1\) starts with an empty cell",
        ),
        (LAYOUT.replace(FIRST_OBJECT, "0,5\t\t1\t0\n"), r"row 2 \(object 1\), column 2 is empty"),
        (
            LAYOUT.replace(FIRST_OBJECT, "x\t1\t1\t0\n"),
            r"row 2 \(object 1\), column 1: 'x' is not a number",
        ),
        (
            LAYOUT.replace(FIRST_OBJECT, "0     5\t1\t1\t0\n"),
            r"row 2 \(object 1\), column 1: '0     5' is not a number",
        ),
        # the class is one of the c classes
        (
            LAYOUT.replace(FIRST_OBJECT, "0,5\t1\t3\t0\n"),
            (
                r"row 2 \(object 1\), column 3: the class must be one of the header's c = 2 "
                r"classes \(1 … 2\), got '3'"
            ),
        ),
        (LAYOUT.replace(FIRST_OBJECT, "0,5\t1\t1,5\t0\n"), "the class must be one of"),
        (LAYOUT.replace(FIRST_OBJECT, "0,5\t1\ta\t0\n"), "the class must be one of"),
        (LAYOUT.replace("5\t2\t2\t\n", "5\t2\t3\t\n"), "c = 3 classes, the objects have 2"),
        # the split marker is 0 or 1
        (
            LAYOUT.replace(FIRST_OBJECT, "0,5\t1\t1\t2\n"),
            (
                r"row 2 \(object 1\), column 4: the split marker must be 0 \(training\) or 1 "
                r"\(test\), got '2'"
            ),
        ),
        (LAYOUT.replace(FIRST_OBJECT, "0,5\t1\t1\tx\n"), "the split marker must be 0"),
        # the type row: exactly n flags 0/1, then only empty cells
        (
            LAYOUT.replace(TYPE_ROW, "1\t0\t0\t0\n"),  # the old defect: values after the types
            r"row 7 \(the type row\), column 3: '0' after the n = 2 type flags",
        ),
        (LAYOUT.replace(TYPE_ROW, "1\t0\t\t1\n"), r"row 7 \(the type row\), column 4: '1' after"),
        (
            LAYOUT.replace(TYPE_ROW, "1\t0     0\t\t\n"),  # the old defect: a merged cell
            (
                r"row 7 \(the type row\), column 2: a type flag must be 0 \(nominal\) or 1 "
                r"\(quantitative\), got '0     0'"
            ),
        ),
        (LAYOUT.replace(TYPE_ROW, "1\t\t\t\n"), r"row 7 \(the type row\) has 1 type flags"),
        (LAYOUT.replace(TYPE_ROW, "1\t5\t\t\n"), "column 2: a type flag must be 0"),
        (
            LAYOUT.replace(TYPE_ROW, "\t1\t0\t\n"),
            r"row 7 \(the type row\) starts with an empty cell",
        ),
        # both samples non-empty
        ("3\t1\t2\t\n1\t1\t0\n2\t2\t0\n3\t1\t0\n1\t\t\n", "non-empty"),
    ],
)
def test_layout_errors_name_the_file_and_row(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(DatasetError, match=message) as error:
        load_split_file(write(tmp_path, text))
    assert str(error.value).startswith("split.dat: ") or "non-empty" in message


def test_rows_keep_their_line_numbers_across_blank_lines(tmp_path: Path) -> None:
    text = LAYOUT.replace(FIRST_OBJECT, "\n" + "0,5\t1\t1\t9\n")
    with pytest.raises(DatasetError, match=r"row 3 \(object 1\), column 4"):
        load_split_file(write(tmp_path, text))


def test_a_file_that_is_not_utf8_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "split.dat"
    path.write_bytes(b"5\t2\t2\n\xff\xfe\n")
    with pytest.raises(DatasetError, match=r"split\.dat: cannot read the file"):
        load_split_file(path)


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_split_file(tmp_path / "missing.dat")


def test_the_samples_must_partition_the_objects(tmp_path: Path) -> None:
    data = load_split_file(write(tmp_path, LAYOUT)).dataset
    with pytest.raises(DatasetError, match="partition"):
        FixedSplit(data, np.array([0, 1, 2]), np.array([2, 3, 4]))
    with pytest.raises(DatasetError, match="partition"):
        FixedSplit(data, np.array([0, 1]), np.array([2, 3]))
    with pytest.raises(DatasetError, match="outside"):
        FixedSplit(data, np.array([0, 1, 2]), np.array([3, 5]))
