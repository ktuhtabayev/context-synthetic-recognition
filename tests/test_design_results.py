"""The committed results of the evaluation designs: ``experiments/designs`` (ADR-096).

Every number comes from ``csr compare run --design all`` on German's three splits and is pinned
here; the checksums of the committed files are checked. A new run of every design must give its
files byte for byte. The random control's new run (300 rounds) takes longer than CI should wait,
so it is opt-in: ``$env:CSR_SLOW_DESIGNS = "1"`` before
``pytest -m slow tests/test_design_results.py``.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from context_synthetic_recognition.comparison import COMPARISON_SPLITS
from context_synthetic_recognition.services.comparison import library_versions
from context_synthetic_recognition.services.comparison_part import COMPARISON_PART
from context_synthetic_recognition.services.design_runs import check_files, run_designs
from context_synthetic_recognition.services.designs import DESIGNS

from .conftest import ROOT

DESIGNS_DIR = ROOT / "experiments" / "designs"
SPLITS_FOLDER = ROOT / "datasets" / "splits"
SHORT = (
    "direct",
    "reversed",
    "training-self",
    "test-self",
    "cross-validation",
    "nested-selection",
    "size-curve",
    "cross-split",
)
"""Designs a new run checks in every test run (each takes at most a few minutes)."""
LONG = ("random-control",)
"""The design whose new run takes about six minutes here and longer in CI: opt-in."""

PINNED: dict[str, dict[tuple[str, str], dict[str, tuple[float, float | None, float]]]] = {
    "direct": {
        ("German-336", "as-given"): {
            "svm": (74.85, None, 70.63),
            "random-forest": (74.25, None, 69.81),
            "knn-euclidean-5": (60.09, None, 51.74),
            "cs-model-euclidean-0.1": (62.95, None, 51.92),
        },
        ("German-341", "as-given"): {
            "svm": (50.68, None, 52.41),
            "random-forest": (56.45, None, 52.61),
            "knn-euclidean-5": (54.78, None, 49.58),
            "cs-model-euclidean-0.1": (56.9, None, 48.07),
        },
        ("German-367", "as-given"): {
            "svm": (61.14, None, 57.99),
            "random-forest": (59.87, None, 55.97),
            "knn-euclidean-5": (59.87, None, 56.59),
            "cs-model-euclidean-0.1": (77.41, None, 48.66),
        },
    },
    "reversed": {
        ("German-336", "as-given"): {
            "svm": (61.61, None, 55.77),
            "random-forest": (63.1, None, 56.94),
            "knn-euclidean-5": (58.04, None, 51.07),
            "cs-model-euclidean-0.1": (58.33, None, 50.1),
        },
        ("German-341", "as-given"): {
            "svm": (49.56, None, 48.85),
            "random-forest": (51.91, None, 51.31),
            "knn-euclidean-5": (49.27, None, 48.67),
            "cs-model-euclidean-0.1": (0.0, None, 0.0),
        },
        ("German-367", "as-given"): {
            "svm": (52.04, None, 50.36),
            "random-forest": (53.68, None, 52.21),
            "knn-euclidean-5": (51.5, None, 50.07),
            "cs-model-euclidean-0.1": (0.0, None, 0.0),
        },
    },
    "training-self": {
        ("German-336", "as-given"): {
            "svm": (83.63, None, 81.89),
            "random-forest": (100.0, None, 100.0),
            "knn-euclidean-5": (55.95, None, 53.67),
            "cs-model-euclidean-0.1": (58.93, None, 55.51),
        },
        ("German-341", "as-given"): {
            "svm": (59.53, None, 59.65),
            "random-forest": (100.0, None, 100.0),
            "knn-euclidean-5": (53.08, None, 53.02),
            "cs-model-euclidean-0.1": (53.08, None, 52.97),
        },
        ("German-367", "as-given"): {
            "svm": (62.67, None, 62.48),
            "random-forest": (100.0, None, 100.0),
            "knn-euclidean-5": (50.68, None, 50.49),
            "cs-model-euclidean-0.1": (53.95, None, 52.34),
        },
    },
    "test-self": {
        ("German-336", "as-given"): {
            "svm": (90.51, None, 82.66),
            "random-forest": (100.0, None, 100.0),
            "knn-euclidean-5": (73.04, None, 53.66),
            "cs-model-euclidean-0.1": (75.9, None, 50.43),
        },
        ("German-341", "as-given"): {
            "svm": (81.03, None, 52.94),
            "random-forest": (100.0, None, 100.0),
            "knn-euclidean-5": (77.24, None, 52.83),
            "cs-model-euclidean-0.1": (0.0, None, 0.0),
        },
        ("German-367", "as-given"): {
            "svm": (81.83, None, 53.56),
            "random-forest": (100.0, None, 100.0),
            "knn-euclidean-5": (78.2, None, 52.85),
            "cs-model-euclidean-0.1": (0.0, None, 0.0),
        },
    },
    "random-control": {
        ("German-336", "random"): {
            "svm": (73.47, 1.8, 69.02),
            "random-forest": (74.62, 1.59, 69.03),
            "knn-euclidean-5": (62.51, 1.88, 54.22),
            "cs-model-euclidean-0.1": (64.21, 15.3, 50.74),
        },
        ("German-341", "random"): {
            "svm": (62.39, 3.87, 58.9),
            "random-forest": (60.72, 2.08, 59.31),
            "knn-euclidean-5": (57.62, 2.37, 54.68),
            "cs-model-euclidean-0.1": (58.02, 9.61, 53.05),
        },
        ("German-367", "random"): {
            "svm": (64.36, 3.78, 59.17),
            "random-forest": (62.05, 2.34, 60.11),
            "knn-euclidean-5": (58.52, 2.32, 54.78),
            "cs-model-euclidean-0.1": (60.74, 8.49, 54.01),
        },
    },
    "cross-validation": {
        ("German-336", "folds"): {
            "svm": (76.2, 4.13, 65.86),
            "random-forest": (76.1, 2.92, 65.88),
            "knn-euclidean-5": (64.7, 2.58, 51.17),
            "cs-model-euclidean-0.1": (63.2, 21.9, 46.19),
        },
        ("German-341", "folds"): {
            "svm": (69.9, 0.99, 50.5),
            "random-forest": (69.6, 3.6, 57.43),
            "knn-euclidean-5": (64.8, 3.43, 51.9),
            "cs-model-euclidean-0.1": (65.6, 12.88, 50.38),
        },
        ("German-367", "folds"): {
            "svm": (69.6, 0.97, 50.1),
            "random-forest": (70.9, 2.18, 59.02),
            "knn-euclidean-5": (65.0, 3.53, 51.86),
            "cs-model-euclidean-0.1": (57.5, 19.71, 49.55),
        },
    },
    "nested-selection": {
        ("German-336", "folds"): {
            "svm": (73.4, 2.95, 62.43),
            "random-forest": (73.3, 3.2, 63.5),
            "knn-euclidean-5": (65.9, 3.11, 52.88),
            "cs-model-euclidean-0.1": (24.0, 12.65, 40.0),
        },
        ("German-341", "folds"): {
            "svm": (70.0, 1.89, 51.62),
            "random-forest": (66.1, 3.54, 57.21),
            "knn-euclidean-5": (61.7, 3.06, 52.55),
            "cs-model-euclidean-0.1": (57.4, 19.87, 49.67),
        },
        ("German-367", "folds"): {
            "svm": (70.3, 2.54, 53.45),
            "random-forest": (68.1, 2.96, 59.4),
            "knn-euclidean-5": (61.8, 3.39, 53.57),
            "cs-model-euclidean-0.1": (58.1, 17.26, 51.31),
        },
    },
    "size-curve": {
        ("German-336", "50"): {
            "svm": (73.73, 1.88, 63.75),
            "random-forest": (73.43, 1.89, 66.28),
            "knn-euclidean-5": (60.59, 2.64, 52.32),
            "cs-model-euclidean-0.1": (53.76, 23.78, 47.68),
        },
        ("German-341", "50"): {
            "svm": (53.42, 6.26, 53.01),
            "random-forest": (55.58, 2.39, 52.71),
            "knn-euclidean-5": (54.44, 2.69, 51.38),
            "cs-model-euclidean-0.1": (50.27, 18.68, 47.97),
        },
        ("German-367", "50"): {
            "svm": (61.39, 7.02, 56.58),
            "random-forest": (60.01, 2.8, 54.7),
            "knn-euclidean-5": (57.35, 3.19, 53.36),
            "cs-model-euclidean-0.1": (45.79, 14.17, 49.64),
        },
    },
    "cross-split": {
        ("German-341", "German-367"): {
            "svm": (50.26, None, 51.51),
            "random-forest": (56.72, None, 52.15),
            "knn-euclidean-5": (54.8, None, 48.93),
            "cs-model-euclidean-0.1": (57.42, None, 47.51),
        },
        ("German-367", "German-341"): {
            "svm": (61.78, None, 58.33),
            "random-forest": (60.21, None, 56.01),
            "knn-euclidean-5": (60.38, None, 56.79),
            "cs-model-euclidean-0.1": (77.66, None, 48.94),
        },
    },
}
"""Design → (split, case) → method → (accuracy, its SD, balanced accuracy), %, from the run of
``csr compare run --design all`` (scikit-learn 1.9.1, numpy 2.5.3)."""
RIPS_ON_ALL = {
    "German-336": (524, 336, 325),
    "German-341": (506, 341, 192),
    "German-367": (523, 367, 226),
}
"""R-IPS on all 1000 objects (the article's defaults): its training size, the file's, both."""


def summary(design: str) -> dict[str, Any]:
    return json.loads((DESIGNS_DIR / design / "summary.json").read_text(encoding="utf-8"))


def mean(design: str, split: str, case: str, method: str) -> tuple[float, float]:
    data = summary(design)
    found = next(s for s in data["splits"] if s["key"] == split)
    chosen = next(c for c in found["cases"] if c["key"] == case)
    entry = next(m for m in chosen["methods"] if m["key"] == method)
    return entry["accuracy"]["mean"], entry["balanced_accuracy"]["mean"]


def test_every_design_has_its_folder() -> None:
    assert sorted(p.name for p in DESIGNS_DIR.iterdir()) == sorted(
        [*(info.name for info in DESIGNS if info.name in (*SHORT, *LONG)), "all-designs"]
    )
    for folder in DESIGNS_DIR.iterdir():
        if folder.name == "all-designs":
            continue
        lines = (folder / "checksums.sha256").read_text(encoding="utf-8").splitlines()
        names = [line.split("  ")[1] for line in lines]
        assert sorted(p.name for p in folder.iterdir()) == sorted([*names, "checksums.sha256"])
        for line in lines:
            digest, name = line.split("  ")
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name
        data = summary(folder.name)
        assert [s["key"] for s in data["splits"]] == list(COMPARISON_SPLITS)


def _new_run(design: str, tmp: Path) -> None:
    recorded = summary(design)["versions"]
    if recorded != library_versions():
        pytest.skip(f"written with {recorded}, installed: {library_versions()}")
    option = COMPARISON_PART.option(COMPARISON_SPLITS, SPLITS_FOLDER)
    (outcome,) = run_designs(COMPARISON_PART, option, SPLITS_FOLDER, (design,))
    assert check_files(DESIGNS_DIR / design, outcome.files) == []
    del tmp


@pytest.mark.parametrize("design", SHORT)
def test_a_new_run_of_a_short_design_gives_its_files(design: str, tmp_path: Path) -> None:
    _new_run(design, tmp_path)


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("CSR_SLOW_DESIGNS"), reason="set CSR_SLOW_DESIGNS=1")
@pytest.mark.parametrize("design", LONG)
def test_a_new_run_of_a_long_design_gives_its_files(design: str, tmp_path: Path) -> None:
    _new_run(design, tmp_path)


def test_the_pinned_numbers() -> None:
    for design, splits in PINNED.items():
        data = summary(design)
        for (split, case), methods in splits.items():
            found = next(s for s in data["splits"] if s["key"] == split)
            chosen = next(c for c in found["cases"] if c["key"] == case)
            values = {
                m["key"]: (
                    m["accuracy"]["mean"],
                    m["accuracy"]["sd"],
                    m["balanced_accuracy"]["mean"],
                )
                for m in chosen["methods"]
                if m["key"] in methods
            }
            assert values == methods, (design, split)


def test_the_two_checks_of_the_designs() -> None:
    # the design as given is the comparison, and the training self-check its training accuracy
    from .test_comparison import PINNED as TODAY

    for split, methods in TODAY.items():
        for method, (test, balanced, training) in methods.items():
            assert mean("direct", split, "as-given", method) == (test, balanced)
            assert mean("training-self", split, "as-given", method)[0] == training
    # R-IPS on all objects does not reproduce the files' training samples (reported only)
    for split in summary("nested-selection")["splits"]:
        rips = split["extras"]["rips_on_all_objects"]
        assert (rips["rips_training"], rips["file_training"], rips["in_both"]) == RIPS_ON_ALL[
            split["key"]
        ]
        assert not rips["reproduced"]
        assert len(split["extras"]["selected_sizes"]) == 10


def test_the_random_control_ranks_the_given_sample() -> None:
    data = summary("random-control")
    for split in data["splits"]:
        ranks = split["extras"]["percentile"]
        assert set(ranks) >= {"svm", "random-forest"}
        assert all(r is None or 0 <= r <= 100 for r in ranks.values())
        (random,) = [c for c in split["cases"] if c["key"] == "random"]
        assert random["rounds"] == 100
        assert set(random["training"]) == {next(iter(random["training"]))}  # T's size, every round


def test_the_table_of_every_design() -> None:
    folder = DESIGNS_DIR / "all-designs"
    lines = (folder / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert [line.split("  ")[1] for line in lines] == ["all-designs.csv", "all-designs.md"]
    for line in lines:
        digest, name = line.split("  ")
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name
    table = (folder / "all-designs.md").read_text(encoding="utf-8")
    svm, rf = (
        PINNED["direct"][("German-336", "as-given")]["svm"],
        PINNED["reversed"][("German-336", "as-given")]["random-forest"],
    )
    assert f"| German · Zhuravlyov split · training 336 (196, 140) | SVM | {svm[0]:.2f} |" in table
    assert f"{rf[0]:.2f}" in table
