"""The project's dataset folder: layout, discovery, identity checks, resolution (ADR-035)."""

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from context_synthetic_recognition.cli.app import EXIT_FAILED, EXIT_USAGE, app
from context_synthetic_recognition.config import DatasetConfig
from context_synthetic_recognition.data import load_builtin, load_from_config
from context_synthetic_recognition.data.catalog import (
    ENVIRONMENT_VARIABLE,
    DatasetCatalog,
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
LOCAL_ONLY = frozenset({"cancer-589", "hypertension-147", "tcga-luad-589", "uzb-kor-100"})
"""Datasets the author keeps on disk only (ADR-087, .gitignore): absent from a checkout, so the
tests ignore them where they are present."""
LOCAL_ONLY_FILES = ("Cancer", "Hypertension", "Uzb-Kor", "TCGA_LUAD")
"""Folders of :data:`LOCAL_ONLY`."""
TYPES = ("Heterogeneous", "Nominal", "Quantitative")
"""The folders of ``raw/``: the kind of features of the datasets in them (ADR-087)."""
COMMITTED = [
    ("default", "default", (".dat", ".csv"), (10, 13, 2)),
    (
        "breast-cancer-wisconsin-569",
        "Breast-Cancer-Wisconsin (569, 30, 2)",
        (".dat", ".csv"),
        (569, 30, 2),
    ),
    ("german-1000", "German (1000, 20, 2)", (".dat", ".csv"), (1000, 20, 2)),
    ("heart-disease-270", "Heart-Disease (270, 13, 2)", (".dat", ".csv"), (270, 13, 2)),
    ("heart-failure-299", "Heart-Failure (299, 12, 2)", (".dat", ".csv"), (299, 12, 2)),
    ("ionosfera-350", "Ionosfera (350, 33, 2)", (".dat", ".csv"), (350, 33, 2)),
    (
        "molecular-biology-100-10",
        "Molecular-Biology (100, 10, 2)",
        (".dat", ".csv"),
        (100, 10, 2),
    ),
    (
        "molecular-biology-100-50",
        "Molecular-Biology (100, 50, 2)",
        (".dat", ".csv"),
        (100, 50, 2),
    ),
    ("molecular-biology-106", "Molecular-Biology (106, 57, 2)", (".dat", ".csv"), (106, 57, 2)),
    ("spambase-4204", "Spambase (4204, 57, 2)", (".dat", ".csv"), (4204, 57, 2)),
    ("tic-tac-toe-958", "Tic-Tac-Toe (958, 9, 2)", (".dat", ".csv"), (958, 9, 2)),
]
"""The datasets of the repository: id, name, formats, (m, n, c)."""
CHECKSUMS = {
    "breast-cancer-wisconsin-569": (
        "e0a3cdad637adf2dc105a0da11312b19b26e0210a8360d8d7ad0dad85c431a85",
        "55a6a761c3ea0e8aa43b80799dfe0ce13e44c0f0a04a1bcc23d3fab592a7ebc6",
    ),
    "heart-failure-299": (
        "f386c5c233d0438039a83c4c28d223497dfec0c1ab9f694a89c321a028064f7c",
        "04c3453dcf52a024a7a29a39514ec62b37d1ce70129e5e843bb605930e045f8e",
    ),
    "ionosfera-350": (
        "0df0572d977dce7de1951d43ce44f637895041bb88fc73c6a4a9a8b3b60fb4af",
        "cde8376bbfd2e61deb59831db75be2aea35e16fb02398463b28b3a279af4ce71",
    ),
    "molecular-biology-100-10": (
        "8ca8af7776957ccaa738f60584e3636d5af88d73fe11229411d45c2906c98dee",
        "eac6dfa14f52d2ab19d5a5fb92036f38e1937b6643e2f3d03f493286bcc7bae0",
    ),
    "molecular-biology-100-50": (
        "7af05a00c947369c49e1906db4b95b0e13f20855879b0d8955ee454375cedd66",
        "4587fbbdcba0db05defe485dd3db1694a231fb0631889ac5260a8d9ecb88ae03",
    ),
    "molecular-biology-106": (
        "60afe78d9791edf4d1a99aef04010a2486f3c985cb9c42988844a198137f6409",
        "b486c90c764dbb20f786f0015f9d1058428ca17aee9a19164dd9da6ea4acf17b",
    ),
    "spambase-4204": (
        "dfcfd0b60428c5b8bb86961c5109cae3853467a73e7e0a33df2e3d000547b502",
        "ed4bdda282e5b24aded44a1dcc8f848878e374108ca7ac2998cc567b8d431f62",
    ),
    "tic-tac-toe-958": (
        "36fff8f813c0d24d0580840b16823235abd84e42bb9f5890f027444c90be41d0",
        "30e4be64303f5c418f65d3e382adb8d29c50ddaca3f5ea321ef4645d47474471",
    ),
}
"""SHA-256 of the .dat and the .csv of the datasets added with ADR-087 (byte for byte)."""


def committed(catalog: DatasetCatalog) -> DatasetCatalog:
    """The catalogue without the datasets kept on the author's disk only."""
    return DatasetCatalog(catalog.root, tuple(e for e in catalog if e.id not in LOCAL_ONLY))


def committed_copy(tmp_path: Path) -> Path:
    """A copy of the dataset folder as a checkout has it (without the local-only datasets)."""

    def ignored(folder: str, names: list[str]) -> list[str]:
        return [n for n in names if n == "splits" or any(n.startswith(s) for s in LOCAL_ONLY_FILES)]

    return Path(shutil.copytree(DATASETS, tmp_path / "datasets", ignore=ignored))


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
    found = [(e.id, e.name, e.formats, e.shape) for e in committed(catalog)]
    assert found == COMMITTED
    assert {e.id for e in catalog} - {entry[0] for entry in COMMITTED} <= LOCAL_ONLY
    heart = catalog.find("heart-disease-270")
    assert heart is not None
    assert heart.path == DATASETS / "raw" / "Heterogeneous" / "Heart-Disease" / f"{heart.name}.dat"
    assert catalog.default is not None
    assert catalog.default.path == DATASETS / "default.dat"
    assert catalog.find("Heart-Disease (270, 13, 2)") is catalog.find("heart-disease-270")
    assert catalog.find("German (1000, 20, 2)") is catalog.find("german-1000")


def test_the_project_datasets_are_consistent(no_override: None) -> None:
    catalog = project_catalog(ROOT)
    assert catalog is not None
    checks = {c.entry.id: c for c in committed(catalog).check()}
    assert all(c.passed for c in checks.values()), [c.problems for c in checks.values()]
    # the .dat and .csv hold the same data; the package's built-in copies too
    assert checks["default"].content_hash == load_builtin("heart-disease-10").content_hash()
    heart = checks["heart-disease-270"]
    assert heart.content_hash == load_builtin("heart-disease-270").content_hash()
    assert set(heart.sha256) == {".dat", ".csv"}


def test_the_draft_datasets_keep_their_bytes(no_override: None) -> None:
    # Heart (270, 13, 2) and German (1000, 20, 2), the two datasets of the experiments
    catalog = project_catalog(ROOT)
    assert catalog is not None
    pair = tuple(e for e in catalog if e.id in ("german-1000", "heart-disease-270"))
    checks = {c.entry.id: c for c in DatasetCatalog(catalog.root, pair).check()}
    german = checks["german-1000"]
    assert german.shape == (1000, 20, 2)
    assert german.sha256 == {
        ".dat": "ee6c75bcc239c7792eddb627a68b8040777587ef086853843ff1ed758e0a48b4",
        ".csv": "6d9cdf0af7d85eb20d49f5a362cde0067e12d01607dec4fc6143783f514d75aa",
    }
    assert checks["heart-disease-270"].sha256 == {
        ".dat": "309be5bc53d641781c212cd2e2da9b9e2e4728785b8438061aab4a27c7c02e94",
        ".csv": "984cd77f05791472f55e77be279914941b0cf3fd598e36afaef9ba937bcbfb42",
    }
    data = load_from_config(DatasetConfig(path="german-1000"), ROOT)
    assert data.class_sizes == (700, 300)
    assert int(data.quantitative.sum()) == 7  # 7 quantitative, 13 nominal features


def test_the_datasets_by_kind_of_features(no_override: None) -> None:
    # ADR-087: raw/<kind>/<family>/ — the kind says which features the dataset has
    catalog = project_catalog(ROOT)
    assert catalog is not None
    checks = {c.entry.id: c for c in committed(catalog).check() if c.entry.id in CHECKSUMS}
    assert set(checks) == set(CHECKSUMS)
    for key, (dat, csv) in CHECKSUMS.items():
        assert checks[key].sha256 == {".dat": dat, ".csv": csv}, key
    for entry in committed(catalog):
        if entry.role == "default":
            continue
        kind = entry.path.relative_to(DATASETS / "raw").parts[0]
        assert kind in TYPES, entry.path
        data = entry.load()
        quantitative = int(data.quantitative.sum())
        expected = {"Quantitative": data.n, "Nominal": 0}.get(kind)
        if expected is None:
            assert 0 < quantitative < data.n, entry.name
        else:
            assert quantitative == expected, entry.name


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


def test_csr_data_list_and_check(
    no_override: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the folder as a checkout has it: the local-only datasets would only make the check slow
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, str(committed_copy(tmp_path)))
    listing = runner.invoke(app, ["data", "list"])
    assert listing.exit_code == 0, listing.output
    assert "heart-disease-270    Heart-Disease (270, 13, 2)   (270, 13, 2)" in listing.output
    assert "raw/Heterogeneous/Heart-Disease" in listing.output
    assert "raw/Quantitative/Spambase" in listing.output
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
