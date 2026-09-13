"""
citra_logging replaces nine competing logging.basicConfig calls. What
matters: every line is parseable JSON with a timestamp, level and the
MODULE NAME, exceptions and extra fields come through, configure() is
first-call-wins so import order cannot change the format, and the text
format is still there for a terminal someone is watching live.
"""
import io
import json
import logging

import pytest

import citra_logging


@pytest.fixture
def captured():
    """Root logger re-pointed at a buffer for the duration of one test."""
    buffer = io.StringIO()
    citra_logging.configure(level="DEBUG", fmt="json", stream=buffer, force=True)
    yield buffer
    citra_logging.configure(level="INFO", fmt="json", stream=io.StringIO(), force=True)


def _lines(buffer: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]


def test_each_line_is_json_with_ts_level_module_and_msg(captured):
    logging.getLogger("jarvis_hardware").info("OK %s -> %s", "http://relay/relay1/on", {"relay": 1})

    (line,) = _lines(captured)
    assert line["level"] == "INFO"
    assert line["module"] == "jarvis_hardware"
    assert line["msg"] == "OK http://relay/relay1/on -> {'relay': 1}"
    # 2026-09-13T13:05:07.123+0530 - date, T, time, millis, offset
    assert len(line["ts"]) == len("2026-09-13T13:05:07.123+0530")
    assert line["ts"][10] == "T"
    assert line["ts"][19] == "."


def test_exceptions_are_attached(captured):
    try:
        raise RuntimeError("board caught fire")
    except RuntimeError:
        logging.getLogger("t").exception("relay call failed")

    (line,) = _lines(captured)
    assert line["level"] == "ERROR"
    assert "RuntimeError: board caught fire" in line["exc"]
    assert "Traceback" in line["exc"]


def test_extra_fields_become_json_fields(captured):
    logging.getLogger("t").warning("slow", extra={"latency_ms": 812.5, "board": "hall-fans"})

    (line,) = _lines(captured)
    assert line["latency_ms"] == 812.5
    assert line["board"] == "hall-fans"
    # And none of logging's own bookkeeping leaks into the line.
    assert "args" not in line and "levelno" not in line and "pathname" not in line


def test_non_json_values_in_extra_are_stringified(captured):
    logging.getLogger("t").info("x", extra={"path": object()})
    (line,) = _lines(captured)
    assert isinstance(line["path"], str)


def test_unicode_survives(captured):
    logging.getLogger("t").info("Timed out — board may be offline ✗")
    assert "—" in captured.getvalue() and "✗" in captured.getvalue()
    assert _lines(captured)[0]["msg"].endswith("✗")


def test_debug_is_dropped_at_info_level():
    buffer = io.StringIO()
    citra_logging.configure(level="INFO", fmt="json", stream=buffer, force=True)
    try:
        logging.getLogger("t").debug("hidden")
        logging.getLogger("t").info("shown")
    finally:
        citra_logging.configure(level="INFO", fmt="json", stream=io.StringIO(), force=True)
    assert [line["msg"] for line in _lines(buffer)] == ["shown"]


def test_configure_is_first_call_wins():
    first = io.StringIO()
    second = io.StringIO()
    citra_logging.configure(level="INFO", fmt="json", stream=first, force=True)
    try:
        citra_logging.configure(level="DEBUG", fmt="text", stream=second)  # a later module importing
        logging.getLogger("t").info("where does this go")
    finally:
        citra_logging.configure(level="INFO", fmt="json", stream=io.StringIO(), force=True)

    assert second.getvalue() == ""
    assert _lines(first)[0]["msg"] == "where does this go"


def test_configure_installs_exactly_one_handler():
    citra_logging.configure(stream=io.StringIO(), force=True)
    citra_logging.configure(stream=io.StringIO())
    citra_logging.configure(stream=io.StringIO())
    try:
        assert len(logging.getLogger().handlers) == 1
    finally:
        citra_logging.configure(level="INFO", fmt="json", stream=io.StringIO(), force=True)


def test_text_format_is_the_old_human_line():
    buffer = io.StringIO()
    citra_logging.configure(level="INFO", fmt="text", stream=buffer, force=True)
    try:
        logging.getLogger("jarvis_router").info("Tool call dispatched")
    finally:
        citra_logging.configure(level="INFO", fmt="json", stream=io.StringIO(), force=True)

    line = buffer.getvalue().strip()
    assert line.endswith("[INFO] jarvis_router: Tool call dispatched")
    with pytest.raises(json.JSONDecodeError):
        json.loads(line)


def test_format_and_level_come_from_the_environment(monkeypatch):
    monkeypatch.setenv(citra_logging.FORMAT_ENV, "text")
    monkeypatch.setenv(citra_logging.LEVEL_ENV, "warning")
    buffer = io.StringIO()
    citra_logging.configure(stream=buffer, force=True)
    try:
        logging.getLogger("t").info("dropped")
        logging.getLogger("t").warning("kept")
    finally:
        citra_logging.configure(level="INFO", fmt="json", stream=io.StringIO(), force=True)

    assert buffer.getvalue().strip().endswith("[WARNING] t: kept")


def test_a_bad_level_name_falls_back_to_info(monkeypatch):
    monkeypatch.setenv(citra_logging.LEVEL_ENV, "loud")
    citra_logging.configure(stream=io.StringIO(), force=True)
    try:
        assert logging.getLogger().level == logging.INFO
    finally:
        citra_logging.configure(level="INFO", fmt="json", stream=io.StringIO(), force=True)
