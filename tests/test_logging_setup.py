"""Logging: the JSON-lines format Phase 6/7 analyses read back."""

import io
import json
import logging

import pytest

from finbrief.config import ConfigError
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


def test_the_configured_level_is_a_threshold_not_an_inherited_default(captured):
    stream, logger = captured
    configure_logging(logging.WARNING, stream=stream)

    log_event(logger, "chat_turn")  # INFO — below the threshold
    log_event(logger, "chat_turn_failed", level=logging.ERROR)

    (record,) = lines(stream)
    assert record["event"] == "chat_turn_failed"
    assert record["level"] == "ERROR"


def test_a_bad_log_level_in_the_environment_raises_before_anything_is_installed(
    monkeypatch,
):
    monkeypatch.setenv("LOG_LEVEL", "chatty")
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.handlers.clear()

    with pytest.raises(ConfigError, match="LOG_LEVEL"):
        configure_logging()

    assert not logger.handlers, "a config failure must not leave a half-configured logger"


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


# --- The file sink (T8, #10) ----------------------------------------------------------


def test_no_log_file_means_no_file_is_opened(captured, tmp_path):
    # The default the hermetic suite runs under. Asserted rather than assumed:
    # `configure_logging` is called by four entry points with no arguments, and a default-on
    # sink would have every test in this repo writing into the working directory.
    stream, logger = captured
    log_event(logger, "chat_turn")

    assert not list(tmp_path.iterdir())
    # `FileHandler` specifically, not a handler count: pytest installs its own capture handler
    # on every logger, so counting would assert something about pytest rather than about us.
    assert not [h for h in logger.handlers if isinstance(h, logging.FileHandler)]


def test_the_file_sink_gets_the_same_bytes_as_the_stream(tmp_path):
    stream = io.StringIO()
    path = tmp_path / "events.jsonl"
    logger = configure_logging(logging.DEBUG, stream=stream, path=path)

    log_event(logger, "retrieval", strategy="hybrid", latency_ms=812)

    # The *same bytes*, not merely the same event: one formatter instance per handler would
    # let the console and the artifact T10 reads disagree about a line, and the disagreeing
    # copy is the one nobody looks at.
    assert path.read_text(encoding="utf-8") == stream.getvalue()


def test_the_sink_appends_across_processes(tmp_path):
    # A Streamlit rerun re-enters `configure_logging`; a run tomorrow re-enters it in a new
    # process. Truncating on open would silently discard the samples ADR-0005's p50 needs.
    path = tmp_path / "events.jsonl"
    first = configure_logging(logging.DEBUG, path=path, stream=io.StringIO())
    log_event(first, "retrieval", k=5)
    second = configure_logging(logging.DEBUG, path=path, stream=io.StringIO())
    log_event(second, "retrieval", k=5)

    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_reconfiguring_closes_the_file_it_replaces(tmp_path):
    # `handlers.clear()` drops a handler without closing it. That is free for a stderr stream
    # and an open file descriptor per Streamlit rerun for a file — and the app reconfigures on
    # *every* rerun, deliberately (`app/Home.py`). Asserted on the handler rather than by
    # counting descriptors, because the descriptor count is platform trivia and the closed
    # handler is the invariant.
    path = tmp_path / "events.jsonl"
    configure_logging(logging.DEBUG, path=path, stream=io.StringIO())
    installed = logging.getLogger(PACKAGE_LOGGER).handlers
    (replaced,) = [h for h in installed if isinstance(h, logging.FileHandler)]

    configure_logging(logging.DEBUG, path=path, stream=io.StringIO())

    assert replaced.stream is None or replaced.stream.closed


def test_a_missing_parent_directory_is_created_rather_than_crashing(tmp_path):
    # `data/` does not exist on a fresh clone, and the recommended path lives under it.
    path = tmp_path / "fresh" / "clone" / "events.jsonl"
    logger = configure_logging(logging.DEBUG, path=path, stream=io.StringIO())
    log_event(logger, "retrieval", k=5)

    assert path.exists()


def test_an_unusable_log_path_is_a_config_error(tmp_path):
    # It lands in `app/Home.py`'s configuration banner with the bad `LOG_LEVEL` and the missing
    # key, rather than as a raw traceback from inside the logging machinery.
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    logging.getLogger(PACKAGE_LOGGER).handlers.clear()

    with pytest.raises(ConfigError, match="FINBRIEF_LOG_FILE"):
        configure_logging(logging.DEBUG, path=blocker / "events.jsonl", stream=io.StringIO())

    assert not logging.getLogger(PACKAGE_LOGGER).handlers, (
        "a config failure must not leave a half-configured logger"
    )
