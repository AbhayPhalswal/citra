"""
=============================================================================
CITRA — ONE LOGGING SETUP FOR EVERY PROCESS
=============================================================================

WHY THIS FILE EXISTS
---------------------
Nine modules each called logging.basicConfig() at import time with their
own copy of the same format string. basicConfig is first-caller-wins, so
which format a process actually used depended on import order, the
module name was never in the line, and nothing could parse the output:
"why didn't my lamp turn on" meant reading two interleaved text logs by
eye.

Every module now does:

    import citra_logging
    citra_logging.configure()
    logger = logging.getLogger("jarvis_router")

configure() is idempotent - the first call in a process wins, later ones
are no-ops - so importing modules in any order gives one consistent
setup. Each line carries a timestamp, the level and the MODULE NAME, and
by default is one JSON object, so `jarvis_voice_assistant.log` can be
grepped for '"module": "jarvis_hardware"' or loaded by any log tool.

ENVIRONMENT
------------
    CITRA_LOG_FORMAT=json   (default) one JSON object per line
    CITRA_LOG_FORMAT=text   the old human format, for a terminal you are
                            watching live:  2026-.. [INFO] module: msg
    CITRA_LOG_LEVEL=INFO    any logging level name
=============================================================================
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import IO

FORMAT_ENV = "CITRA_LOG_FORMAT"
LEVEL_ENV = "CITRA_LOG_LEVEL"

TEXT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# The LogRecord attributes that are logging's own bookkeeping. Anything
# a caller passes via `extra={...}` is whatever is NOT in here, and gets
# carried into the JSON line as its own field.
_STANDARD_RECORD_FIELDS = frozenset(vars(logging.makeLogRecord({}))) | {
    "message", "asctime", "taskName",
}


class JsonFormatter(logging.Formatter):
    """
    One JSON object per line: ts, level, module, msg - plus exc when an
    exception is attached, and any extra={} fields the caller added.

    ensure_ascii=False so the em-dashes and the odd emoji in this
    codebase's messages stay readable rather than becoming \\u escapes.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "module": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        # ISO-8601 with milliseconds and the local UTC offset, e.g.
        # 2026-09-13T13:05:07.123+0530 - sortable, and unambiguous when a
        # log is read from a different timezone than it was written in.
        base = time.strftime("%Y-%m-%dT%H:%M:%S", self.converter(record.created))
        offset = time.strftime("%z", self.converter(record.created))
        return f"{base}.{int(record.msecs):03d}{offset}"


def make_formatter(fmt: str | None = None) -> logging.Formatter:
    """The formatter configure() installs. `fmt` is 'json' or 'text'."""
    chosen = (fmt or os.environ.get(FORMAT_ENV, "json")).strip().lower()
    if chosen == "text":
        return logging.Formatter(TEXT_FORMAT)
    return JsonFormatter()


def configure(
    level: int | str | None = None,
    fmt: str | None = None,
    stream: IO[str] | None = None,
    force: bool = False,
) -> logging.Logger:
    """
    Install one handler on the root logger, once per process.

    Returns the root logger. Safe to call from every module at import
    time: the first call sets things up and every later call returns
    immediately, unless force=True (tests use that to re-point the
    output at a buffer).
    """
    root = logging.getLogger()
    if getattr(root, "_citra_configured", False) and not force:
        return root

    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(make_formatter(fmt))
    root.addHandler(handler)

    chosen_level = level if level is not None else os.environ.get(LEVEL_ENV, "INFO")
    if isinstance(chosen_level, str):
        chosen_level = logging.getLevelName(chosen_level.upper())
        if not isinstance(chosen_level, int):
            chosen_level = logging.INFO
    root.setLevel(chosen_level)

    root._citra_configured = True  # type: ignore[attr-defined]
    return root
