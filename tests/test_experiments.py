"""The computational experiment of the article: the committed results are those of a real run."""

import csv
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from .conftest import ROOT

EXPERIMENTS = ROOT / "experiments"


def study() -> ModuleType:
    """``scripts/run_experiments.py`` as a module."""
    spec = importlib.util.spec_from_file_location(
        "run_experiments", ROOT / "scripts" / "run_experiments.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_checksums_of_the_committed_results() -> None:
    lines = (EXPERIMENTS / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for line in lines:
        digest, name = line.split("  ")
        assert hashlib.sha256((EXPERIMENTS / name).read_bytes()).hexdigest() == digest, name
        # the CSV files carry a byte-order mark, so that Excel reads them as UTF-8 (ADR-083)
        bom = (EXPERIMENTS / name).read_bytes().startswith(b"\xef\xbb\xbf")
        assert bom == name.endswith(".csv"), name


@pytest.mark.slow
def test_a_new_run_reproduces_every_result_byte_for_byte(tmp_path: Path) -> None:
    files = study().run(tmp_path)
    assert [f.name for f in files] == [
        "rips_samples.csv",
        "table1_binary_features.csv",
        "tables2_6_accuracy.csv",
        "knn_comparison.csv",
        "summary.json",
        "checksums.sha256",
    ]
    for file in files:
        committed = EXPERIMENTS / file.relative_to(tmp_path)
        assert file.read_bytes() == committed.read_bytes(), file.name


def rows(name: str) -> list[dict[str, str]]:
    with (EXPERIMENTS / "results" / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_the_rips_samples_against_the_article() -> None:
    samples = {(r["dataset"], r["split_metric"], r["reading"]): r for r in rows("rips_samples.csv")}
    heart = samples[("Heart", "zhuravlyov", "keep-covering")]
    assert (heart["training"], heart["training_K1"], heart["training_K2"]) == ("73", "36", "37")
    assert (heart["article_training"], heart["article_K1"], heart["article_K2"]) == (
        "76",
        "36",
        "40",
    )
    german = samples[("German", "zhuravlyov", "keep-covering")]
    assert (german["training"], german["article_training"]) == ("335", "336")
    literal = samples[("Heart", "zhuravlyov", "literal")]
    assert (literal["training"], literal["test"]) == ("161", "109")


def test_the_model_does_not_reach_the_articles_accuracy() -> None:
    # Tables 2–5 report 96–100 % on the training and 100 % on the test samples
    results = rows("tables2_6_accuracy.csv")
    assert len(results) == 56
    best = max(float(r["test_accuracy"]) for r in results if r["test_accuracy"])
    assert best == 79.7  # Heart, Zhuravlyov split (blocked objects keep covering), α = 0.1
    assert max(float(r["training_accuracy"]) for r in results if r["training_accuracy"]) < 90
    binary = rows("table1_binary_features.csv")
    assert {r["binary_features"] for r in binary} == {"20"}  # no feature takes 0 here
