"""Split files: a dataset with a training/test flag column (data/splits.py)."""

from pathlib import Path

import numpy as np
import pytest

from context_synthetic_recognition.data import FeatureType, FixedSplit, load_split_file
from context_synthetic_recognition.errors import DatasetError

# leading tabs, decimal commas, CRLF, test rows between training rows, extra trailing type values
LAYOUT = (
    "\t5\t3\t2\t\t\t\r\n"
    "\t0,5\t1\t1\t0\r\n"
    "\t1,25\t2\t2\t1\r\n"
    "\t0\t1\t2\t0\r\n"
    "\t2\t3\t1\t1\r\n"
    "\t3,5\t2\t1\t0\r\n"
    "\t1\t0\t0     0\t\t\r\n"
)


def write(tmp_path: Path, text: str, name: str = "split.dat") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode("ascii"))
    return path


def test_the_layout_is_read_with_its_variations(tmp_path: Path) -> None:
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


def test_lf_and_spaces_give_the_same_split(tmp_path: Path) -> None:
    crlf = load_split_file(write(tmp_path, LAYOUT, "a.dat"))
    lf = load_split_file(write(tmp_path, LAYOUT.replace("\r\n", "\n").replace("\t", "  "), "b.dat"))
    assert lf.dataset.content_hash() == crlf.dataset.content_hash()
    assert lf.training.tolist() == crlf.training.tolist()


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
        ("", "empty"),
        ("5 3\n", "first row"),
        ("5 1 2\n" + "1 1 0\n" * 5 + "1\n", "leaves no feature"),
        (LAYOUT.replace("\t5\t3\t2", "\t6\t3\t2"), "8 rows are expected"),
        (LAYOUT.replace("\t0,5\t1\t1\t0", "\t0,5\t1\t1"), "object row 1 has 3 values"),
        (LAYOUT.replace("\t0,5\t1\t1\t0", "\tx\t1\t1\t0"), "'x' is not a number"),
        (LAYOUT.replace("\t0,5\t1\t1\t0", "\t0,5\t1\t1\t2"), "the flag must be 0"),
        (LAYOUT.replace("\t1\t0\t0     0\t\t", "\t1"), "type row has 1 flags"),
        (LAYOUT.replace("\t1\t0\t0     0\t\t", "\t1\t5"), "flags must be 1 or 0"),
        (LAYOUT.replace("\t5\t3\t2", "\t5\t3\t3"), "c = 3 classes"),
        ("3 2 2\n1 1 0\n2 2 0\n3 1 0\n1 0\n", "non-empty"),  # no test object
    ],
)
def test_layout_errors_are_reported(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        load_split_file(write(tmp_path, text))


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
