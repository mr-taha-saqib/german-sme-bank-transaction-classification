"""
logging_utils.py
================
Structured logging shared by the pipeline, the retraining workflow and the API.

``LOG_FORMAT=json`` (the API default inside Docker) emits one JSON object per line
with timestamp, level, logger, message and any ``extra={...}`` fields; ``text``
emits the familiar human-readable format.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(level: str = "INFO", log_file: str | Path | None = None,
                  fmt: str | None = None) -> None:
    """Configure the root logger (idempotent: existing handlers are replaced)."""
    fmt = (fmt or os.environ.get("LOG_FORMAT", "text")).lower()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    formatter: logging.Formatter
    if fmt == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                                      datefmt="%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)
    # Quieten chatty third-party loggers.
    for noisy in ("matplotlib", "PIL", "urllib3", "alembic", "mlflow", "httpx", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
