"""The project's dataset folder: layout, discovery, identity checks, resolution (ADR-035)."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from context_synthetic_recognition.cli.app import EXIT_FAILED, EXIT_USAGE, app
from context_synthetic_recognition.config import DatasetConfig
from context_synthetic_recognition.data import load_builtin, load_from_config
from context_synthetic_recognition.data.catalog import (
    ENVIRONMENT_VARIABLE,
    discover,
    find_datasets_root,
    project_catalog,
    read_shape,
    resolve_dataset,
)
from context_synthetic_recognition.errors import DatasetError

from .conftest import ROOT

runner = CliRunner()

DATASETS = ROOT / "datasets"
SMALL_DAT = "3 2 2\n1.0 0.0 1\n2.0 1.0 2\n3.0 1.0 2\n1 0\n"
SMALL_CSV = "3,2,2,\r\n1,0,1\r\n2,1,2\r\n3,1,2\r\n1,0,\r\n"


def folder(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "datasets"
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


@pytest.fixture
def no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENVIRONMENT_VARIABLE, raising=False)


# ---------------------------------------------------------------- the author's folder


def test_the_project_folder(no_override: None) -> None:
    catalog = project_catalog(ROOT)
    assert catalog is not None
    assert catalog.root == DATASETS
    assert [(e.id, e.name, e.formats, e.shape) for e in catalog] == [
        ("default", "default", (".dat", ".csv"), (10, 13, 2)),
        ("heart-disease-270", "Heart-Disease (270, 13, 2)", (".dat", ".csv"), (270, 13, 2)),
    ]
    assert catalog.default is not None
    assert catalog.default.path == DATASETS / "default.dat"
    assert catalog.find("Heart-Disease (270, 13, 2)") is catalog.find("heart-disease-270")


def test_the_project_datasets_are_consistent(no_override: None) -> None:
    catalog = project_catalog(ROOT)
    assert catalog is not None
    checks = catalog.check()
    assert all(c.passed for c in checks), [c.problems for c in checks]
    # the .dat and .csv hold the same data; the package's built-in copies too
    assert checks[0].content_hash == load_builtin("heart-disease-10").content_hash()
    assert checks[1].content_hash == load_builtin("heart-disease-270").content_hash()
    assert set(checks[1].sha256) == {".dat", ".csv"}


def test_the_files_keep_the_authors_bytes() -> None:
    # CRLF in the .csv, LF in the .dat — stored byte for byte (.gitattributes: datasets/** -text)
    assert b"\r\n" in (DATASETS / "default.csv").read_bytes()
    assert b"\r\n" not in (DATASETS / "default.dat").read_bytes()
    assert read_shape(DATASETS / "default.csv") == (10, 13, 2)


# ---------------------------------------------------------------- discovery rules


def test_layout_rules(tmp_path: Path) -> None:
    root = folder(
        tmp_path,
        {
            "default.csv": SMALL_CSV,
            "raw/Toy/Toy (3, 2, 2).dat": SMALL_DAT,
            "raw/Toy/Toy (3, 2, 2).csv": SMALL_CSV,
            "raw/Flat (3, 2, 2).dat": SMALL_DAT,
            "raw/Notes.txt": "not a dataset",
            "raw/Unnamed.dat": SMALL_DAT,
            "synthetic/Psi (3, 2, 2).dat": SMALL_DAT,  # reserved: never read
        },
    )
    catalog = discover(root)
    assert [(e.id, e.role, e.formats) for e in catalog] == [
        ("default", "default", (".csv",)),
        ("flat-3", "raw", (".dat",)),
        ("toy-3", "raw", (".dat", ".csv")),
        ("unnamed", "raw", (".dat",)),
    ]
    assert catalog.find("unnamed") is not None
    assert catalog.find("unnamed").shape is None  # type: ignore[union-attr]
    assert all(c.passed for c in catalog.check())


def test_ids_of_datasets_with_the_same_name_and_size(tmp_path: Path) -> None:
    other = "3 1 2\n1.0 1\n2.0 2\n3.0 2\n1\n"
    root = folder(tmp_path, {"raw/D/D (3, 2, 2).dat": SMALL_DAT, "raw/D/D (3, 1, 2).dat": other})
    assert [e.id for e in discover(root)] == ["d-3-1", "d-3-2"]
    clash = folder(
        tmp_path / "b", {"raw/x/A (3, 2, 2).dat": SMALL_DAT, "raw/y/A (3, 2, 2).dat": SMALL_DAT}
    )
    with pytest.raises(DatasetError, match="share the id"):
        discover(clash)


def test_check_reports_problems(tmp_path: Path) -> None:
    rounded = SMALL_CSV.replace("2,1,2", "2.5,1,2")
    root = folder(
        tmp_path,
        {
            "raw/A (4, 2, 2).dat": SMALL_DAT,  # the name says 4 objects, the file has 3
            "raw/B (3, 2, 2).dat": SMALL_DAT,
            "raw/B (3, 2, 2).csv": rounded,  # the formats disagree
            "raw/C (3, 2, 2).dat": "garbage\n",
        },
    )
    problems = {c.entry.id: c.problems for c in discover(root).check()}
    assert "holds (3, 2, 2), its name says (4, 2, 2)" in problems["a-4"][0]
    assert "hold different data (.dat is used)" in problems["b-3"][0]
    assert "the first row must be 'm, n, c'" in problems["c-3"][0]


# ---------------------------------------------------------------- finding and resolving


def test_finding_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_override: None
) -> None:
    root = folder(tmp_path, {"default.dat": SMALL_DAT})
    nested = tmp_path / "configs" / "deeper"
    nested.mkdir(parents=True)
    assert find_datasets_root(nested) == root
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, str(root))
    assert find_datasets_root(Path("/")) == root
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, str(tmp_path / "missing"))
    with pytest.raises(DatasetError, match="not a folder"):
        find_datasets_root()


def test_resolving_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_override: None
) -> None:
    root = folder(tmp_path, {"default.dat": SMALL_DAT, "raw/T/Toy (3, 2, 2).csv": SMALL_CSV})
    monkeypatch.chdir(tmp_path)
    assert resolve_dataset().name == "default"
    assert resolve_dataset("toy-3").name == "Toy (3, 2, 2)"
    assert resolve_dataset(root / "default.dat").m == 3
    assert resolve_dataset("heart-disease-270").m == 270  # built-in fallback
    with pytest.raises(DatasetError, match="dataset file not found"):
        resolve_dataset("missing.dat")
    with pytest.raises(DatasetError, match="available: default, toy-3, builtin: heart-disease-10"):
        resolve_dataset("cancer-589")


def test_without_a_folder_the_builtins_are_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_override: None
) -> None:
    root = folder(tmp_path / "project", {"raw/T/Toy (3, 2, 2).dat": SMALL_DAT})
    monkeypatch.chdir(root.parent)
    assert resolve_dataset().m == 10  # no default.*: the built-in Heart-Disease (10, 13, 2)


def test_configurations_name_datasets_by_id(tmp_path: Path, no_override: None) -> None:
    root = folder(tmp_path, {"default.dat": SMALL_DAT, "raw/T/Toy (3, 2, 2).dat": SMALL_DAT})
    configs = tmp_path / "configs"
    configs.mkdir()
    assert load_from_config(DatasetConfig(path="toy-3"), configs).name == "Toy (3, 2, 2)"
    assert load_from_config(DatasetConfig(), configs).name == "default"
    typed = load_from_config(DatasetConfig(path="toy-3", feature_types={"x₁": "nominal"}), configs)
    assert not typed.quantitative[0]
    assert root.exists()


# ---------------------------------------------------------------- csr data


def test_csr_data_list_and_check(no_override: None) -> None:
    listing = runner.invoke(app, ["data", "list"])
    assert listing.exit_code == 0, listing.output
    assert "heart-disease-270    Heart-Disease (270, 13, 2)   (270, 13, 2)" in listing.output
    assert "raw/Heart-Disease" in listing.output
    check = runner.invoke(app, ["data", "check"])
    assert check.exit_code == 0, check.output
    assert "all datasets consistent" in check.output
    assert "sha256 0a3b879f08c6e8e93f399da47595d4c20e1e5f461937291f144e132d7812d650" in check.output
    info = runner.invoke(app, ["data", "info"])
    assert "(the default dataset)" in info.output


def test_csr_data_check_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_override: None
) -> None:
    root = folder(tmp_path, {"raw/A (4, 2, 2).dat": SMALL_DAT})
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, str(root))
    failed = runner.invoke(app, ["data", "check"])
    assert failed.exit_code == EXIT_FAILED
    assert "1 dataset(s) with problems" in failed.output
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, str(tmp_path / "nothing"))
    broken = runner.invoke(app, ["data", "list"])
    assert broken.exit_code == EXIT_USAGE
