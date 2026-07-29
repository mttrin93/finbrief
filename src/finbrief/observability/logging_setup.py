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
import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from finbrief.config import ConfigError, load_env, resolve_log_file, resolve_log_level

#: The package logger everything under `finbrief.*` propagates to.
PACKAGE_LOGGER = "finbrief"

#: Serialises the handler swap in `configure_logging`. Streamlit runs one script thread per
#: session and `app/Home.py` calls `configure_logging` on every rerun, so two sessions can
#: reach the swap at once; interleaved, both could observe an empty handler list and both
#: install, permanently doubling every JSON line Phase 7 reads back as a latency sample.
#: A precaution, deliberately untested: `logging`'s own lock makes the window between the
#: clear and the add too narrow to reproduce in-process, so a test asserting the invariant
#: would pass without this lock and prove nothing.
_CONFIGURE_LOCK = threading.Lock()


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
        # Omitted rather than written as `null` when there is no turn in scope: an ingest run
        # has no turns at all, and a key that is present-and-empty on every one of ~5,800
        # ingest lines says nothing a missing key does not.
        turn_id = getattr(record, "turn_id", None)
        if turn_id is not None:
            payload["turn_id"] = turn_id
        fields = getattr(record, "fields", None)
        if fields:
            payload["fields"] = fields
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        # `default=str` keeps a stray non-serialisable value (a Path, an enum, a
        # timestamp) from turning an observability line into a crash.
        return json.dumps(payload, default=str)


def _file_handler(path: Path) -> logging.FileHandler:
    """Open the append-only sink at `path`, creating its parent directory.

    The parent is created because the recommended path lives under `data/`, which does not
    exist in a fresh clone. Every failure becomes a `ConfigError` so an unwritable path lands
    in `app/Home.py`'s configuration banner beside the missing key and the bad `LOG_LEVEL`,
    rather than as a traceback out of the logging machinery on the first event of a run.

    **Append, never truncate.** `app/Home.py` reconfigures on every rerun and tomorrow's run
    reconfigures in a new process; truncating on open would discard the latency samples
    ADR-0005's p50 is measured from, and discard them silently.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Opened eagerly (`delay` left at its default) so a bad path fails *here*, inside this
        # `try` — a deferred open fails on the first event instead, in a different frame, with
        # no configuration banner in front of it.
        return logging.FileHandler(path, mode="a", encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"FINBRIEF_LOG_FILE={str(path)!r} cannot be opened for appending: {exc}"
        ) from exc


def configure_logging(
    level: int | None = None, *, stream: TextIO | None = None, path: Path | None = None
) -> logging.Logger:
    """Install the JSON-lines handlers on the `finbrief` logger. Idempotent.

    Scoped to the package logger rather than the root so Streamlit's own logging is left
    alone, and `propagate=False` so records are not also printed by the root handler.

    `path` is the persistence half T8 (#10) added, and it is **off unless named** — see
    `config.resolve_log_file` for why the default is nothing rather than a path. When it is
    named the sink is installed *beside* the stream rather than instead of it, and both
    handlers share **one** `JsonLinesFormatter` instance: the console and the file T10 reads
    back cannot then disagree about a line, and the copy that would have disagreed is the one
    nobody is looking at.
    """
    if level is None or path is None:
        # Resolved before anything is mutated, so a bad `LOG_LEVEL` raises `ConfigError`
        # for the caller's config banner rather than half-configuring the logger.
        load_env()
        level = resolve_log_level(os.environ) if level is None else level
        path = resolve_log_file(os.environ) if path is None else path

    logger = logging.getLogger(PACKAGE_LOGGER)
    formatter = JsonLinesFormatter()
    handlers: list[logging.Handler] = [
        logging.StreamHandler(stream if stream is not None else sys.stderr)
    ]
    # Opened before the swap, for the reason in `_file_handler`: a `ConfigError` here must
    # leave the caller's existing logger intact rather than a half-configured one.
    if path is not None:
        handlers.append(_file_handler(path))
    for handler in handlers:
        handler.setFormatter(formatter)

    with _CONFIGURE_LOCK:
        logger.setLevel(level)
        logger.propagate = False
        # `propagate=False` already claims sole ownership of this logger, and nothing else in
        # the package installs a handler on it, so clearing is the whole swap — but a cleared
        # handler is not a closed one. `StreamHandler.close` leaves its stream alone (which is
        # why a buffer a test configured into stays readable afterwards) while
        # `FileHandler.close` closes the file: without this, naming the sink leaked one file
        # descriptor per Streamlit rerun, and the app reconfigures on every rerun by design.
        for replaced in logger.handlers:
            replaced.close()
        logger.handlers.clear()
        for handler in handlers:
            logger.addHandler(handler)
    return logger


#: The turn every event on this thread belongs to, or `None` outside one. A `ContextVar`
#: rather than a parameter threaded through `screen` → `answer` → the tool → `retrieve` →
#: `fuse`, because that chain is eight signatures long and the value is the same at every
#: step. `log_event` reads it; `turn` is the only writer.
_TURN_ID: ContextVar[str | None] = ContextVar("finbrief_turn_id", default=None)


@contextmanager
def turn(turn_id: str) -> Iterator[str]:
    """Tag every event emitted inside this block with `turn_id`.

    **The join T10 (#11) needs, and the reason it is not line order.** A `retrieval` line
    carries per-chunk provenance and — deliberately, since a variant is user-derived — no
    question. Without an identifier the provenance can be aggregated but never *attributed*,
    and the only other correlation available is position in the file: correct for a serial
    harness, wrong the moment the agent issues two searches in one step, and asserted by
    nothing either way. That is the check-that-cannot-fail class CLAUDE.md names.

    The identifier is the caller's, not generated here, because a meaningful one is worth
    more than a unique one: an evaluation harness names the golden-set row and the arm it is
    running, so a surprising per-bucket number leads back to its own retrieval lines.
    It must be **derived from nothing the analyst typed** — it lands in a kept log.

    Nesting is the `ContextVar` semantics: an inner block wins and the outer value is
    restored on exit.
    """
    token = _TURN_ID.set(turn_id)
    try:
        yield turn_id
    finally:
        _TURN_ID.reset(token)


def log_event(
    logger: logging.Logger,
    event: str,
    /,
    *,
    level: int = logging.INFO,
    exc_info: bool = False,
    **fields: Any,
) -> None:
    """Emit one structured event. `event` is the stable key analyses filter on.

    `logger` and `event` are positional-only so that almost any field name is usable.
    `level` and `exc_info` are the two reserved words: they select the log level and attach
    the current traceback, so a payload cannot use either as a field key — give the datum a
    different name. The envelope's own `level` is unreachable from `fields` either way, since
    fields are nested rather than flattened — and so is `turn_id`, which is read from the
    context here rather than taken as a field.

    `exc_info` exists so that a *failure* is an event like everything else. `app/Home.py` used
    `logger.exception` directly, which was the only `log_event` bypass in the codebase and
    therefore the only line that carried no `turn_id` — a failed turn being precisely the one
    whose provenance a reader wants (issue #10 review). `JsonLinesFormatter` already renders
    `record.exc_info` into the envelope's `error`; this just stops the caller having to reach
    past the emitter to get it.

    The turn is read **at the call site**, not in the handler: the value belongs to the
    context that emitted the event, and a handler is free to run somewhere else.
    """
    logger.log(
        level,
        event,
        exc_info=exc_info,
        extra={"event": event, "fields": fields, "turn_id": _TURN_ID.get()},
    )
