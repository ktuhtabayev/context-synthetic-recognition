"""Structured logging.

Records carry key–value context through ``extra``::

    logger.info("permitted k computed", extra={"k": [3, 5], "k_max": 5})

The console shows ``12:00:01 INFO    context_synthetic_recognition.core: permitted k computed
k=[3, 5] k_max=5``; a run folder receives the same records as JSON lines.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

PACKAGE_LOGGER = "context_synthetic_recognition"
CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

_STANDARD_ATTRIBUTES = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {
    "message",
    "asctime",
    "taskName",
}
_OWN_HANDLER = "_csr_handler"


def record_fields(record: logging.LogRecord) -> dict[str, object]:
    """Return the key–value context a record received through ``extra``."""
    return {
        key: value
        for key, value in vars(record).items()
        if key not in _STANDARD_ATTRIBUTES and not key.startswith("_")
    }


def _format_value(value: object) -> str:
    return repr(value) if isinstance(value, str) else str(value)


class KeyValueFormatter(logging.Formatter):
    """Human-readable line followed by ``key=value`` pairs of the record's context."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the record and append its context fields."""
        line = super().format(record)
        fields = record_fields(record)
        if fields:
            line += " " + " ".join(f"{key}={_format_value(value)}" for key, value in fields.items())
        return line


class JsonFormatter(logging.Formatter):
    """One JSON object per record: time (UTC), level, logger, message and context fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise the record as a single JSON line."""
        payload: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **record_fields(record),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
    json_path: Path | None = None,
) -> logging.Logger:
    """Configure the package logger for an application (CLI, GUI, scripts).

    Calling it again replaces the handlers it added before, so it is safe to call once per run.

    Args:
        level: Minimum level of the package logger.
        stream: Console stream; ``sys.stderr`` by default.
        json_path: Optional file that receives every record as a JSON line.

    Returns:
        The package logger.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    for handler in list(logger.handlers):
        if getattr(handler, _OWN_HANDLER, False):
            logger.removeHandler(handler)
            handler.close()

    console = logging.StreamHandler(stream if stream is not None else sys.stderr)
    console.setFormatter(KeyValueFormatter(CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    handlers: list[logging.Handler] = [console]
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_file = logging.FileHandler(json_path, encoding="utf-8")
        json_file.setFormatter(JsonFormatter())
        handlers.append(json_file)

    for handler in handlers:
        setattr(handler, _OWN_HANDLER, True)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
