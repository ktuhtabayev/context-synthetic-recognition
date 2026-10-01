"""The tables of a run and their text forms: CSV, JSON, Markdown, LaTeX."""

import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from context_synthetic_recognition.data import Dataset
from context_synthetic_recognition.errors import DatasetError
from context_synthetic_recognition.export import RunView, build_view
from context_synthetic_recognition.export.tables import (
    Table,
    TableSet,
    number,
    plain,
    run_tables,
    slug,
    table_specs,
)
from context_synthetic_recognition.export.text import (
    csv_text,
    format_cell,
    index_rows,
    is_numeric,
    json_text,
    latex_document,
    latex_table,
    latex_text,
    markdown_table,
    markdown_text,
    table_data,
    write_text,
)
from context_synthetic_recognition.export.wording import (
    decision_label,
    decision_text,
    join_and,
    labels_of,
    percent,
    property_rows,
    protocol_title,
    set_text,
    sufficiency_status,
    theorem_rows,
    tuplam_text,
    tuple_text,
)
from context_synthetic_recognition.services.runner import ExperimentResult

from .shapes import SHAPES, shape

EXPERIMENT_TABLES = [
    "summary",
    "dataset",
    "features",
    "normalized",
    "distances-rho",
    "distances-rho-i",
    "distances-rho-j",
    "neighbours-rho",
    "neighbours-rho-i",
    "neighbours-rho-j",
    "same-class-counts",
    "chi1",
    "psi",
    "membership",
    "bit-masks",
    "synthetic-features",
    "object-membership",
    "correct-side",
    "contributions",
    "hag-iterations",
    "hag-candidates",
    "hag-chosen-blocks",
    "meta-dataset",
    "training-description",
    "new-object",
    "new-object-features",
    "resubstitution",
    "new-object-steps",
    "metrics",
    "class-metrics",
    "confusion-resubstitution",
    "roc-resubstitution",
    "predictions-resubstitution",
    "folds-resubstitution",
    "confusion-leave-one-out",
    "roc-leave-one-out",
    "predictions-leave-one-out",
    "folds-leave-one-out",
    "margins",
    "object-margins",
    "sensitivity",
    "properties",
    "operator-pairs",
    "boundary-ties",
]


@pytest.fixture(scope="module")
def tables(experiment_view: RunView) -> TableSet:
    return run_tables(experiment_view)


# ---------------------------------------------------------------- the view


def test_the_view_of_the_experiment(experiment_view: RunView, experiment: Dataset) -> None:
    view = experiment_view
    assert view.dataset is experiment
    assert view.labels == (2, 1, 2, 1, 1, 1, 2, 2, 2, 2)
    assert view.slots == 5  # min(ϰ, r) = min(5, 6)
    assert view.hag.label == "{a₆, a₃, a₁, a₂, a₄}"
    # the default demonstration: S₁ left out of its own context reproduces its training row
    demo = view.new_object
    assert demo.exclude == 0
    assert demo.values.tolist() == experiment.X[0].tolist()
    assert demo.decision == int(view.training.decisions[0]) == 1
    assert view.protocol("leave-one-out") is not None
    assert view.protocol("hold-out") is None
    assert [p.protocol for p in view.cross_validations] == ["leave-one-out"]
    assert view.sensitivity is not None
    assert len(view.sensitivity) == 4
    assert view.created is not None


def test_a_typed_new_object(experiment_result: ExperimentResult, experiment: Dataset) -> None:
    typed = build_view(experiment_result, new_object=experiment.X[3], exclude=3)
    assert typed.new_object.exclude == 3
    assert typed.new_object.decision == int(typed.training.decisions[3])
    free = build_view(experiment_result, new_object=experiment.X[3])
    assert free.new_object.exclude is None
    assert free.sensitivity is None
    assert free.run_id is None
    rows = theorem_rows(free)
    assert rows[0][1] == "no object excluded"
    assert theorem_rows(typed)[0][1:] == ("identical", "✓ same representation as in training")
    with pytest.raises(DatasetError, match="needs 13 values, got 2"):
        build_view(experiment_result, new_object=[1.0, 2.0])
    with pytest.raises(DatasetError, match="no training object № 11"):
        build_view(experiment_result, new_object=experiment.X[0], exclude=10)


# ---------------------------------------------------------------- wording


def test_wording() -> None:
    assert join_and([]) == ""
    assert join_and(["a"]) == "a"
    assert join_and(["a", "b"]) == "a and b"
    assert join_and(["a", "b", "c"]) == "a, b and c"
    assert percent(0.2) == "20.0%"
    assert set_text([]) == "∅"
    assert set_text(["S₂", "S₄"]) == "{S₂, S₄}"
    assert set_text(["S₁", "S₂", "S₃"], limit=2) == "{S₁, S₂, … (+1)}"
    assert tuple_text([2, 1, 2]) == "(2, 1, 2)"
    assert tuplam_text([5, 2]) == "{a₆, a₃}"
    assert decision_text(1, ("absent", "present")) == "Class absent"
    assert decision_text(2, ("absent", "present")) == "Class present"
    assert decision_text(0, (1, 2)) == "0 — refusal"
    assert decision_label(0, (1, 2)) == 0
    assert labels_of(np.array([1, 2, 0]), ("a", "b")) == ["a", "b", 0]
    assert protocol_title("stratified-k-fold") == "stratified k-fold"
    assert protocol_title("bootstrap") == "bootstrap"
    assert sufficiency_status(0) == "✓ sufficient on E₀"
    assert sufficiency_status(10) == "✗ insufficient: 10 conflicting pair(s)"


def test_the_property_rows_of_the_experiment(experiment_view: RunView) -> None:
    rows = property_rows(experiment_view)
    assert [r.section.split(" ")[0] + " " + r.section.split(" ")[1] for r in rows] == (
        ["Definition 1"] * 5
        + ["Definition 2", "Definition 3", "Definition 4", "Definition 4"]
        + ["Theorem —"] * 3
    )
    by_check = {r.check: r for r in rows}
    assert by_check["Correctly classified training objects"].result == "7 / 10"
    assert by_check["Correctly classified training objects"].status == (
        "✗ not correct on 3 object(s)"
    )
    held_out = by_check["Correctly classified held-out objects (leave-one-out)"]
    assert held_out.status == "accuracy 20.0%,  refusals 4"
    assert rows[7].status == "✗ insufficient: 10 conflicting pair(s)"
    assert rows[-2].result == "1 vs 1"
    assert rows[-2].status == "✓ same decision"


# ---------------------------------------------------------------- the tables


def test_the_tables_of_the_experiment(experiment_view: RunView, tables: TableSet) -> None:
    assert [t.key for t in tables] == EXPERIMENT_TABLES
    assert tables.skipped == ()
    assert tables.get("hold-out") is None
    with pytest.raises(KeyError):
        tables["hold-out"]
    summary = dict(tables["summary"].rows)
    assert summary["Dataset"] == experiment_view.dataset.name
    assert summary["Features n"] == "13 (6 quantitative, 7 nominal)"
    assert summary["Permitted k (formula)"] == "3, 5"
    assert summary["TUPLAM (selected synthetic features, in order)"] == "{a₆, a₃, a₁, a₂, a₄}"
    assert summary["crit per iteration"] == "0.4282, 0.3053, 0.2734, 0.2710"
    assert summary["Accuracy — leave-one-out"] == 0.2
    assert summary["AUC — leave-one-out"] == pytest.approx(0.2708333333333333)
    assert summary["New object (S₁ left out of its context): class"] == "Class 1"
    assert [k for k in summary if k.startswith("⚠")] == [
        "⚠ Template calculation (ADR-002)",
        "⚠ Template calculation (ADR-003)",
    ]
    psi = tables["psi"]
    assert psi.columns[1] == "a₁ (ρ, k = 3)"
    assert psi.rows[0] == ("S₁", 2, 2, 1, 2, 2, 2, 2)
    assert tables["features"].rows[1] == ("x₂", "nominal (J)", 0, None, None, 2)
    assert tables["distances-rho-j"].rows[0][:4] == ("S₁", 0, 3, 4)
    assert tables["neighbours-rho"].rows[0][:4] == ("S₁", 1, "S₉", 9)
    assert tables["hag-iterations"].rows[0][:3] == (0, "STEP 2 · u = argmax ω", "a₆")
    assert tables["hag-iterations"].rows[0][3] is None
    assert tables["resubstitution"].rows[0][:4] == ("S₁", "(2, 1, 2, 2, 2)", 2, 0)
    assert tables["new-object-steps"].title.endswith("— Class 1")
    assert tables["roc-leave-one-out"].rows[0] == ("+∞", 0.0, 0.0)
    assert tables["margins"].rows[0][7] == pytest.approx(0.148177267098285)
    assert tables["sensitivity"].rows[3][2] == "article"
    assert [row[0] for row in tables["metrics"].rows[:2]] == ["resubstitution"] * 2
    # every value is plain data
    for table in tables:
        for row in table.rows:
            assert all(v is None or type(v) in (str, int, float) for v in row), table.key


def test_the_table_sizes_are_known_beforehand(experiment_view: RunView, tables: TableSet) -> None:
    specs = {spec.key: spec for spec in table_specs(experiment_view)}
    assert list(specs) == EXPERIMENT_TABLES
    for table in tables:
        assert table.cells <= specs[table.key].cells, table.key


def test_large_tables_are_left_out_and_keys_select(experiment_view: RunView) -> None:
    limited = run_tables(experiment_view, max_cells=100)
    assert "neighbours-rho" in [s.key for s in limited.skipped]
    assert limited.get("neighbours-rho") is None
    assert limited.get("summary") is not None
    chosen = run_tables(experiment_view, max_cells=None, keys=["psi", "margins"])
    assert [t.key for t in chosen] == ["psi", "margins"]


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_tables_of_other_shapes(name: str) -> None:
    view = shape(name)
    found = run_tables(view)
    keys = [t.key for t in found]
    assert len(set(keys)) == len(keys)
    specs = {spec.key: spec for spec in table_specs(view)}
    for table in found:
        assert table.cells <= specs[table.key].cells, table.key
        assert table.title
        assert csv_text(table).count("\n") == len(table.rows) + 1
        assert markdown_text(table)
        assert latex_table(table)
    assert ("margins" in keys) == (view.hag.p > 0)
    assert "sensitivity" not in keys
    for protocol in view.result.protocols:
        assert f"folds-{protocol.protocol}" in keys


def test_tables_of_the_other_shapes_in_detail() -> None:
    numeric = run_tables(shape("numeric"))
    assert [t.key for t in numeric if t.key.startswith("distances")] == [
        "distances-rho",
        "distances-rho-i",
    ]
    assert numeric["dataset"].columns[:3] == ("№", "age", "pressure")
    assert numeric["confusion-hold-out"].columns[1:3] == ("Class healthy", "Class sick")
    summary = dict(numeric["summary"].rows)
    assert "New object (new object): class" in summary
    assert not [k for k in summary if k.startswith("⚠")]  # the article's switches
    nominal = run_tables(shape("nominal"))
    assert nominal.get("margins") is None
    assert nominal["hag-candidates"].rows == ()
    assert dict(nominal["summary"].rows)["Latent (additional) features p"] == 0
    assert {row[1] for row in nominal["predictions-repeated-k-fold"].rows} == {0, 1}
    literal = dict(run_tables(shape("literal"), keys=["summary"])["summary"].rows)
    assert literal["Permitted k (formula)"] == "3, 5, 7, …, 237 (118 values)"


def test_a_table_checks_its_rows() -> None:
    with pytest.raises(ValueError, match="a row has 1 values, expected 2"):
        Table("t", "T", ("a", "b"), ((1,),), "g")
    table = Table("t", "T", ("a", "b"), ((1, 2), (3, 4)), "g")
    assert table.cells == 4


def test_cell_helpers() -> None:
    assert slug("ρ_I") == "rho-i"
    assert slug("ρ") == "rho"
    assert slug("Ψ(r) & ω") == "psi-r-omega"
    assert slug("—") == "x"
    assert number(np.float64(0.5)) == 0.5
    assert number(np.int64(3)) == 3
    assert type(number(np.int64(3))) is int
    assert number(np.bool_(True)) == 1
    assert number(float("nan")) is None
    assert number(float("inf")) == "+∞"
    assert number(float("-inf")) == "−∞"
    assert plain(3.0) == 3
    assert type(plain(3.0)) is int
    assert plain(0.25) == 0.25


# ---------------------------------------------------------------- text forms

SMALL = Table(
    "small",
    "ω of ρ_I | k",
    ("Feature", "ω", "Class", "Note"),
    (("a₁", 0.9, 1, "x|y"), ("a₂", None, 2, "5 % of S₁₀"), ("a₃", -0.00001, 1, "")),
    "Step 7",
    note="ω by formula (4).",
)


def test_format_cell() -> None:
    assert format_cell(None) == "—"
    assert format_cell(True) == "1"
    assert format_cell(7) == "7"
    assert format_cell(0.123456) == "0.1235"
    assert format_cell(0.123456, 2) == "0.12"
    assert format_cell(-0.00001) == "0.0000"  # no "-0.0000"
    assert format_cell("S₁") == "S₁"
    assert is_numeric(SMALL, 1)
    assert is_numeric(SMALL, 2)
    assert not is_numeric(SMALL, 0)
    assert not is_numeric(Table("e", "E", ("a",), ((None,),), "g"), 0)


def test_csv_keeps_every_digit(tables: TableSet) -> None:
    text = csv_text(SMALL)
    assert text.splitlines()[0] == "Feature,ω,Class,Note"
    assert text.splitlines()[2] == "a₂,,2,5 % of S₁₀"
    assert text.splitlines()[3].startswith("a₃,-1e-05,1,")
    rows = list(csv.reader(io.StringIO(csv_text(tables["margins"]))))
    assert float(rows[1][7]) == tables["margins"].rows[0][7]
    assert "\r" not in text


def test_json_form(tables: TableSet) -> None:
    data = table_data(tables["psi"])
    assert data["key"] == "psi"
    assert data["rows"][0] == ["S₁", 2, 2, 1, 2, 2, 2, 2]
    text = json_text({"tables": [data]})
    assert "a₁ (ρ, k = 3)" in text  # the notation is kept as Unicode
    assert text.endswith("}\n")
    assert json.loads(text)["tables"][0]["columns"][0] == "№"


def test_markdown() -> None:
    text = markdown_table(SMALL, decimals=2)
    lines = text.splitlines()
    assert lines[0] == "| Feature | ω | Class | Note |"
    assert lines[1] == "| :--- | ---: | ---: | :--- |"
    assert lines[2] == "| a₁ | 0.90 | 1 | x\\|y |"
    assert lines[3] == "| a₂ | — | 2 | 5 % of S₁₀ |"
    full = markdown_text(SMALL, level=2)
    assert full.startswith("## ω of ρ_I | k\n")
    assert full.endswith("*ω by formula (4).*\n")
    assert "*" not in markdown_text(Table("t", "T", ("a",), ((1,),), "g"))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a₆", "$a_{6}$"),
        ("ρ_I", r"$\rho_{I}$"),
        ("S₁₀", "$S_{10}$"),
        ("f_k(μ)", r"$f_{k}$($\mu$)"),
        ("score₁", "score$_{1}$"),
        ("x′", "$x'$"),
        ("θ/γ", r"$\theta$/$\gamma$"),
        ("k = 3", "k = 3"),
        ("run_id", r"run\_id"),
        ("5 % of A & B", r"5 \% of A \& B"),
        ("a|b < c", r"a\textbar{}b \textless{} c"),
        ("№ 3 — ok", "No. 3 --- ok"),
        ("m²", "$m^{2}$"),
        ("ω ≥ 0.5", r"$\omega$ $\geq$ 0.5"),
        ("✓ defined", r"\checkmark{} defined"),
        ("{a₆, a₃}", r"\{$a_{6}$, $a_{3}$\}"),
    ],
)
def test_latex_notation(text: str, expected: str) -> None:
    assert latex_text(text) == expected


def test_latex_table() -> None:
    text = latex_table(SMALL, decimals=2)
    lines = text.splitlines()
    assert lines[0] == r"\begin{table}[htbp]"
    assert r"  \caption{$\omega$ of $\rho_{I}$ \textbar{} k. $\omega$ by formula (4).}" in lines
    assert r"  \label{tab:small}" in lines
    assert r"  \begin{tabular}{lrrl}" in lines
    assert r"    Feature & $\omega$ & Class & Note \\" in lines
    assert r"    $a_{1}$ & 0.90 & 1 & x\textbar{}y \\" in lines
    assert r"    $a_{2}$ & --- & 2 & 5 \% of $S_{10}$ \\" in lines
    assert lines[-1] == r"\end{table}"
    # a negative number is set in math mode: a minus sign, not a hyphen
    signed = Table("s", "S", ("a", "b"), (("x", -0.15), ("y-z", -3)), "g")
    assert r"    x & $-0.1500$ \\" in latex_table(signed)
    assert r"    y-z & $-3$ \\" in latex_table(signed)
    long = latex_table(SMALL, label="tab:x", long_rows=2)
    assert long.startswith("\\begin{longtable}{lrrl}\n")
    assert (
        r"\caption{$\omega$ of $\rho_{I}$ \textbar{} k. $\omega$ by formula (4).}\label{tab:x}"
        in (long)
    )
    assert long.count(r"\toprule") == 2
    assert long.endswith("\\end{longtable}\n")


def test_latex_document_and_index(tmp_path: Path) -> None:
    document = latex_document(["psi", "margins"], "Heart-Disease: tables of a₆")
    assert "\\input{psi}\n\\input{margins}\n" in document
    assert r"\title{Heart-Disease: tables of $a_{6}$}" in document
    assert r"\usepackage{booktabs, longtable, amsmath, amssymb}" in document
    assert index_rows([SMALL], ".csv") == [
        ["small.csv", "small", "Step 7", "ω of ρ_I | k", 3, 4, "ω by formula (4)."]
    ]
    path = write_text(tmp_path / "deep" / "folder" / "small.md", "a\nb\n")
    assert path.read_bytes() == b"a\nb\n"  # LF on every platform
