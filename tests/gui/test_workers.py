"""Background work: the jobs (plain Python) and the thread that runs them."""

import json
import logging
import time
from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from context_synthetic_recognition.config import ExperimentConfig
from context_synthetic_recognition.data import Dataset
from context_synthetic_recognition.errors import DatasetError
from context_synthetic_recognition.evaluation.protocols import EvaluationCancelledError
from context_synthetic_recognition.export import ExportOptions, RunView
from context_synthetic_recognition.gui.workers import (
    LogBridge,
    Task,
    export_job,
    open_job,
    run_job,
)
from context_synthetic_recognition.log import PACKAGE_LOGGER

pytestmark = pytest.mark.gui


# ---------------------------------------------------------------- the jobs


def test_the_run_job(experiment: Dataset, tmp_path: Path) -> None:
    stages: list[tuple[str, int, int]] = []
    outcome = run_job(
        experiment,
        ExperimentConfig(),
        runs_dir=tmp_path / "runs",
        progress=lambda *report: stages.append(report),
    )
    view = outcome.view
    assert view.hag.label == "{a₆, a₃, a₁, a₂, a₄}"
    assert view.sensitivity is not None  # "auto": ten objects are few enough
    assert len(view.sensitivity) == 4
    assert outcome.folder is not None
    assert outcome.folder.parent == tmp_path / "runs"
    assert view.run_id == outcome.folder.name
    assert outcome.warnings == ()
    stored = json.loads((outcome.folder / "results.json").read_text(encoding="utf-8"))
    assert len(stored["sensitivity"]) == 4
    names = [stage for stage, _, _ in stages]
    assert names.index("leave-one-out") < names.index("article")  # the run, then the settings
    assert stages[-1] == ("article", 10, 10)


def test_the_sensitivity_choice(experiment: Dataset) -> None:
    config = ExperimentConfig()
    never = run_job(experiment, config, sensitivity="no")
    assert never.view.sensitivity is None
    assert never.folder is None  # no runs folder: not saved
    assert never.view.run_id is None
    assert run_job(experiment, config, sensitivity="yes").view.sensitivity is not None


def test_a_cancelled_run(experiment: Dataset, tmp_path: Path) -> None:
    with pytest.raises(EvaluationCancelledError):
        run_job(experiment, ExperimentConfig(), runs_dir=tmp_path, cancelled=lambda: True)
    assert list(tmp_path.iterdir()) == []  # nothing is saved
    # cancelling during the switch settings stops them too
    calls = {"count": 0}

    def after_the_run() -> bool:
        calls["count"] += 1
        return calls["count"] > 12  # resubstitution and ten leave-one-out folds pass

    with pytest.raises(EvaluationCancelledError):
        run_job(experiment, ExperimentConfig(), runs_dir=tmp_path, cancelled=after_the_run)
    assert list(tmp_path.iterdir()) == []


def test_the_open_job(experiment: Dataset, tmp_path: Path) -> None:
    saved = run_job(experiment, ExperimentConfig(), runs_dir=tmp_path)
    assert saved.folder is not None
    opened = open_job(saved.folder)
    assert opened.folder == saved.folder
    assert opened.view.run_id == saved.folder.name
    assert opened.view.sensitivity == saved.view.sensitivity  # taken from the folder
    assert opened.view.hag.label == saved.view.hag.label
    assert opened.warnings == ()
    file = saved.folder / "results.json"
    stored = json.loads(file.read_text(encoding="utf-8"))
    stored["model"]["r"] = 99
    file.write_text(json.dumps(stored), encoding="utf-8")
    assert open_job(saved.folder).warnings == ("model.r: stored 99, repeated 6",)


def test_the_export_job(experiment_view: RunView, tmp_path: Path) -> None:
    reports: list[tuple[str, int, int]] = []
    summary = export_job(
        experiment_view,
        tmp_path,
        ["csv", "json"],
        ExportOptions(),
        progress=lambda *report: reports.append(report),
    )
    assert list(summary.files) == ["csv", "json"]
    assert (tmp_path / "trace.json").is_file()
    assert reports == [("csv", 0, 2), ("json", 1, 2), ("done", 2, 2)]
    assert export_job(experiment_view, tmp_path, ["json"], ExportOptions()).count == 1


# ---------------------------------------------------------------- the thread


def test_a_task_reports_its_result(qtbot: QtBot) -> None:
    progress: list[tuple[str, int, int]] = []

    def work(*, progress: object, cancelled: object, factor: int) -> int:
        assert callable(progress)
        assert callable(cancelled)
        progress("half", 1, 2)
        assert cancelled() is False
        return 21 * factor

    task = Task.with_progress(work, factor=2)
    task.progressed.connect(lambda *report: progress.append(report))
    with qtbot.waitSignal(task.succeeded, timeout=10_000) as done:
        task.start()
    assert done.args == [42]
    assert task.wait(10_000)
    qtbot.waitUntil(lambda: progress == [("half", 1, 2)])


def test_a_task_reports_errors(qtbot: QtBot, caplog: pytest.LogCaptureFixture) -> None:
    def known(task: Task) -> None:
        raise DatasetError("no such dataset")

    def missing(task: Task) -> None:
        raise FileNotFoundError("gone.csv")

    def bug(task: Task) -> None:
        raise ZeroDivisionError("a bug")

    for work, message in (
        (known, "no such dataset"),
        (missing, "gone.csv"),
        (bug, "unexpected error: ZeroDivisionError: a bug"),
    ):
        task = Task(work)
        with (
            qtbot.waitSignal(task.failed, timeout=10_000) as failed,
            caplog.at_level(logging.CRITICAL),
        ):
            task.start()
        assert failed.args == [message]
        assert task.wait(10_000)


def test_a_task_can_be_cancelled(qtbot: QtBot) -> None:
    def work(task: Task) -> None:
        deadline = time.monotonic() + 10
        while not task.is_cancelled() and time.monotonic() < deadline:
            time.sleep(0.005)
        raise EvaluationCancelledError("stopped")

    task = Task(work)
    assert not task.is_cancelled()
    with qtbot.waitSignal(task.cancelled, timeout=10_000):
        task.start()
        task.cancel()
    assert task.is_cancelled()
    assert task.wait(10_000)


def test_the_log_reaches_the_interface(qtbot: QtBot) -> None:
    bridge = LogBridge()
    lines: list[str] = []
    bridge.message.connect(lines.append)
    logger = logging.getLogger(f"{PACKAGE_LOGGER}.core.test")
    package = logging.getLogger(PACKAGE_LOGGER)
    level = package.level
    try:
        package.setLevel(logging.WARNING)
        bridge.attach()
        assert package.level == logging.INFO  # the interface shows what the pipeline reports
        logger.info("context fitted (Steps 1–8)")
        assert len(lines) == 1
        assert lines[0].endswith("INFO    context fitted (Steps 1–8)")
        bridge.detach()
        logger.info("not shown")
        assert len(lines) == 1
        package.setLevel(logging.DEBUG)
        bridge.attach()
        assert package.level == logging.DEBUG  # a more talkative level is left alone
    finally:
        bridge.detach()
        package.setLevel(level)
