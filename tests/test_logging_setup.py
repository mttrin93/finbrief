"""Logging: the JSON-lines format Phase 6/7 analyses read back."""

import io
import json
import logging

import pytest

from finbrief.observability.logging_setup import PACKAGE_LOGGER, configure_logging, log_event


@pytest.fixture
def captured():
    """Configure package logging into a buffer, and reset the logger afterwards."""
    stream = io.StringIO()
    logger = configure_logging(logging.DEBUG, stream=stream)
    yield stream, logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)


def lines(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_an_event_is_one_json_object_per_line(captured):
    stream, logger = captured
    log_event(logger, "chat_turn", latency_ms=812, model="openai/gpt-4o-mini")
    log_event(logger, "chat_turn", latency_ms=44, model="openai/gpt-4o-mini")

    first, second = lines(stream)
    assert first["event"] == "chat_turn"
    assert first["level"] == "INFO"
    assert first["logger"] == PACKAGE_LOGGER
    assert first["fields"] == {"latency_ms": 812, "model": "openai/gpt-4o-mini"}
    assert first["ts"].endswith("+00:00")
    assert second["fields"]["latency_ms"] == 44


def test_fields_cannot_collide_with_the_envelope(captured):
    stream, logger = captured
    # `logger` and `event` are field names here, not the call's own arguments.
    log_event(logger, "chat_turn", ts="not-a-timestamp", logger="not-a-logger", event="nested")

    (record,) = lines(stream)
    assert record["ts"].endswith("+00:00")
    assert record["logger"] == PACKAGE_LOGGER
    assert record["event"] == "chat_turn"
    assert record["fields"] == {
        "ts": "not-a-timestamp",
        "logger": "not-a-logger",
        "event": "nested",
    }


def test_configuring_twice_does_not_duplicate_lines(captured):
    stream, _ = captured
    logger = configure_logging(logging.DEBUG, stream=stream)
    log_event(logger, "chat_turn")
    assert len(lines(stream)) == 1


def test_child_loggers_are_captured(captured):
    stream, _ = captured
    log_event(logging.getLogger("finbrief.agent.agent"), "chat_turn")

    (record,) = lines(stream)
    assert record["logger"] == "finbrief.agent.agent"


def test_a_non_serialisable_field_does_not_break_the_line(captured):
    stream, logger = captured
    log_event(logger, "chat_turn", strategy=object())
    (record,) = lines(stream)
    assert "object object" in record["fields"]["strategy"]


def test_an_exception_is_recorded(captured):
    stream, logger = captured
    try:
        raise ValueError("upstream refused")
    except ValueError:
        logger.exception("chat_turn_failed", extra={"event": "chat_turn_failed"})

    (record,) = lines(stream)
    assert record["event"] == "chat_turn_failed"
    assert "upstream refused" in record["error"]
