"""Structured logging: key=value console lines, JSON lines, idempotent configuration."""

import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from context_synthetic_recognition.log import (
    PACKAGE_LOGGER,
    JsonFormatter,
    configure_logging,
    record_fields,
)


@pytest.fixture(autouse=True)
def _restore_package_logger() -> Iterator[None]:
    logger = logging.getLogger(PACKAGE_LOGGER)
    saved = (list(logger.handlers), logger.level, logger.propagate)
    yield
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        if handler not in saved[0]:
            handler.close()
    for handler in saved[0]:
        logger.addHandler(handler)
    logger.setLevel(saved[1])
    logger.propagate = saved[2]


def test_console_line_carries_the_context_fields() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    logging.getLogger(f"{PACKAGE_LOGGER}.core").info(
        "permitted k computed", extra={"k": [3, 5], "k_max": 5, "operator": "ρ_J"}
    )
    line = stream.getvalue().strip()
    assert line.endswith(
        "context_synthetic_recognition.core: permitted k computed k=[3, 5] k_max=5 operator='ρ_J'"
    )
    assert " INFO " in line


def test_level_filters_records() -> None:
    stream = io.StringIO()
    configure_logging("WARNING", stream=stream)
    logging.getLogger(PACKAGE_LOGGER).info("hidden")
    assert stream.getvalue() == ""


def test_json_lines_file(tmp_path: Path) -> None:
    path = tmp_path / "run" / "log.jsonl"
    configure_logging("DEBUG", stream=io.StringIO(), json_path=path)
    logger = logging.getLogger(f"{PACKAGE_LOGGER}.hag")
    logger.debug("iteration done", extra={"q": "a₃", "crit": 0.4281557866657403})
    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("failed")
    configure_logging("DEBUG", stream=io.StringIO())  # closes the file handler
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["message"] == "iteration done"
    assert records[0]["q"] == "a₃"
    assert records[0]["crit"] == 0.4281557866657403
    assert records[0]["level"] == "DEBUG"
    assert records[0]["time"].endswith("+00:00")
    assert "ValueError: boom" in records[1]["exception"]


def test_reconfiguring_replaces_only_own_handlers() -> None:
    logger = logging.getLogger(PACKAGE_LOGGER)
    foreign = logging.NullHandler()
    logger.addHandler(foreign)
    configure_logging("INFO", stream=io.StringIO())
    configure_logging("INFO", stream=io.StringIO())
    own = [h for h in logger.handlers if getattr(h, "_csr_handler", False)]
    assert len(own) == 1
    assert foreign in logger.handlers
    assert logger.propagate is False


def test_record_fields_ignore_standard_attributes() -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "msg", None, None)
    record.custom = 1
    assert record_fields(record) == {"custom": 1}
    assert json.loads(JsonFormatter().format(record))["custom"] == 1
