"""Structured JSON-lines logging.

Phase 6 (Tier-1) is *log capture*: per-query strategy config, retrieval hits, latency,
token counts, gate-trigger metadata, and agent-vs-original query divergence. Phase 7's
A/B analysis and the security-gate catch analysis read those lines back, so the line
format is a contract, not a convenience: **one JSON object per line**.

    configure_logging()
    log_event(logging.getLogger(__name__), "chat_turn", latency_ms=812)

    {"ts": "...", "level": "INFO", "logger": "finbrief.agent.agent",
     "event": "chat_turn", "fields": {"latency_ms": 812}}

Never put a secret or an API key in `fields` — these lines are meant to be kept.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, TextIO

from finbrief.config import resolve_log_level

#: The package logger everything under `finbrief.*` propagates to.
PACKAGE_LOGGER = "finbrief"

_HANDLER_TAG = "_finbrief_json_handler"


class JsonLinesFormatter(logging.Formatter):
    """Render a log record as a single JSON object.

    Structured payloads travel in `extra={"fields": {...}}` and are nested under
    `fields` rather than flattened, so a caller can never collide with (or overwrite)
    an envelope key such as `level` or `ts`.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            # UTC with millisecond precision: comparable across machines, and fine
            # enough to order the latency measurements Phase 7 reads back.
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload["fields"] = fields
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        # `default=str` keeps a stray non-serialisable value (a Path, an enum, a
        # timestamp) from turning an observability line into a crash.
        return json.dumps(payload, default=str)


def configure_logging(
    level: int | None = None, *, stream: TextIO | None = None
) -> logging.Logger:
    """Install the JSON-lines handler on the `finbrief` logger. Idempotent.

    Scoped to the package logger rather than the root so Streamlit's own logging is left
    alone, and `propagate=False` so records are not also printed by the root handler.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(resolve_log_level() if level is None else level)
    logger.propagate = False

    for existing in [h for h in logger.handlers if getattr(h, _HANDLER_TAG, False)]:
        logger.removeHandler(existing)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonLinesFormatter())
    setattr(handler, _HANDLER_TAG, True)
    logger.addHandler(handler)
    return logger


def log_event(
    logger: logging.Logger, event: str, /, *, level: int = logging.INFO, **fields: Any
) -> None:
    """Emit one structured event. `event` is the stable key analyses filter on.

    `logger` and `event` are positional-only so that almost any field name is usable.
    `level` is the one reserved word: it selects the log level, so a payload cannot use it
    as a field key — give the datum a different name. The envelope's own `level` is
    unreachable from `fields` either way, since fields are nested rather than flattened.
    """
    logger.log(level, event, extra={"event": event, "fields": fields})
