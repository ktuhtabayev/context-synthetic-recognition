"""``csr compare run | check | predict --design`` (ADR-096)."""

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from context_synthetic_recognition.cli.app import EXIT_FAILED, EXIT_USAGE, app

from .conftest import ROOT
from .test_designs import toy_folder

SPLITS_FOLDER = ROOT / "datasets" / "splits"
runner = CliRunner()


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return toy_folder(tmp_path_factory.mktemp("splits"))


def test_every_design_of_one_split(toy: Path, tmp_path: Path) -> None:
    out = tmp_path / "designs"
    ran = runner.invoke(
        app,
        [
            "compare",
            "run",
            "--design",
            "all",
            "--split",
            "Toy-16",
            "-o",
            str(out),
            "--splits",
            str(toy),
        ],
    )
    assert ran.exit_code == 0, ran.output
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [
            "direct",
            "reversed",
            "training-self",
            "test-self",
            "random-control",
            "cross-validation",
            "nested-selection",
            "size-curve",
            "cross-split",
            "all-designs",
        ]
    )
    assert sorted(p.name for p in (out / "random-control").iterdir()) == [
        "checksums.sha256",
        "random-control.csv",
        "random-control.md",
        "random-control.tex",
        "rounds.csv",
        "summary.json",
    ]
    summary = json.loads((out / "cross-split" / "summary.json").read_text(encoding="utf-8"))
    (case,) = summary["splits"][0]["cases"]
    assert case["unavailable"] == "no other ticked split of Toy"
    checked = runner.invoke(
        app, ["compare", "check", "--design", "all", "-o", str(out), "--splits", str(toy)]
    )
    assert checked.exit_code == 0, checked.output
    assert "design check passed" in checked.output
    assert "method × design" in checked.output
    table = (out / "all-designs" / "all-designs.md").read_text(encoding="utf-8")
    assert "| Toy · Euclidean split · training 16 (8, 8) | SVM |" in table
    (out / "reversed" / "reversed.md").write_bytes(b"changed\n")
    failed = runner.invoke(
        app,
        [
            "compare",
            "check",
            "--design",
            "reversed",
            "-o",
            str(out / "reversed"),
            "--splits",
            str(toy),
        ],
    )
    assert failed.exit_code == EXIT_FAILED
    assert "reversed.md: does not match checksums.sha256" in failed.output


def test_the_design_options_are_checked(toy: Path, tmp_path: Path) -> None:
    unknown = runner.invoke(app, ["compare", "run", "--design", "nope", "-o", str(tmp_path)])
    assert unknown.exit_code == EXIT_USAGE
    assert "nope" in unknown.output
    no_out = runner.invoke(
        app, ["compare", "run", "--design", "reversed", "--split", "Toy-16", "--splits", str(toy)]
    )
    assert no_out.exit_code == EXIT_USAGE
    assert "--split needs --out" in no_out.output


def test_the_blind_predictions_of_the_reversed_design(tmp_path: Path) -> None:
    out = tmp_path / "reversed.csv"
    result = runner.invoke(
        app,
        [
            "compare",
            "predict",
            "--split",
            "76",
            "--design",
            "reversed",
            "--out",
            str(out),
            "--splits",
            str(SPLITS_FOLDER),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        "trained on the 194 test objects (114, 80); 76 training objects classified" in result.output
    )
    with out.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 1 + 76  # the training objects, predicted without their class
    assert not any("class" in name.lower() for name in rows[0])
    wrong = runner.invoke(
        app,
        [
            "compare",
            "predict",
            "--split",
            "76",
            "--design",
            "size-curve",
            "--splits",
            str(SPLITS_FOLDER),
        ],
    )
    assert wrong.exit_code == EXIT_USAGE
    assert "direct or reversed" in wrong.output
