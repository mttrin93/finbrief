"""Reading the JSON-lines events back — the consumer half of `logging_setup`.

`log_event` is the only emitter and this is the only reader, and that pairing is the whole
reason this module exists rather than a `json.loads` in the harness. T10 (#11) reads four
things out of a run that nothing else records: the latency samples ADR-0005's ≤1.5s p50 is
judged against, token counts, the agent-vs-original query divergence rate, and gate-trigger
metadata. Hand-rolled digging gets every one of them wrong the same way — a renamed field
reads back as `None` and `None` averages as nothing, so the number shrinks its own
denominator and says so to nobody.

**What this module will not do: statistics.** It returns samples. ADR-0005's ≤1.5s budget
and ADR-0006's ≤1s budget are medians over these samples and belong to the report that
quotes them; `security/report.py` already owns its own median over in-process `Screening`
objects, and a second one here would be a second definition of the same word. `Samples`
carries the count it *could not* provide precisely so that a caller cannot compute a mean
over a narrowed denominator without seeing it.

**Absence is not zero, and a missing file is not an empty run.** Both halves of the
honest-absence rule (CLAUDE.md) apply here for the same reason they apply to a checkpointed
payload: a log file written before a field existed is read by the code that added it.
`Event.field` returns `None` for a field the line never carried and the caller supplies any
default; a *missing* sink raises, because "nobody enabled the log" read as "this run emitted
nothing" is how a p50 over zero samples gets reported as a budget met.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Event:
    """One line of the log, parsed. The envelope, plus the payload as it was written.

    `fields` is the emitter's payload verbatim — not flattened into this object, for the
    reason `JsonLinesFormatter` nests it: a field named `level` or `ts` would otherwise
    collide with the envelope, and the collision would be silent.
    """

    event: str
    ts: datetime
    level: str
    logger: str
    fields: Mapping[str, Any] = dataclass_field(default_factory=dict)
    #: The turn this line belongs to, or `None` for a line emitted outside one (an ingest
    #: run, a script, a live turn from before the field existed). See `logging_setup.turn`.
    turn_id: str | None = None
    #: A formatted traceback, when the line recorded an exception.
    error: str | None = None

    def field(self, name: str, default: Any = None) -> Any:
        """The payload field `name`, or `default` when the line does not carry it.

        The default is the *caller's* choice and defaults to `None`: this module cannot know
        whether a missing `input_tokens` should read as nothing or as zero, and only one of
        those is honest for a cost table.
        """
        return self.fields.get(name, default)


@dataclass(frozen=True, slots=True)
class Samples:
    """The values a field took, and the count of lines that did not carry it.

    `absent` is not diagnostic detail. A mean over `present` alone is a mean over a
    denominator the caller never chose, which is the "no silent caps" failure this repo
    keeps hitting: `total` is what a rate must be reported against.
    """

    present: tuple[Any, ...]
    absent: int

    @property
    def total(self) -> int:
        return len(self.present) + self.absent


@dataclass(frozen=True, slots=True)
class EventLog:
    """Every event a sink recorded, plus the count of lines that were not events.

    Read eagerly rather than streamed. The volume is bounded by a human typing questions
    and by an evaluation run over a golden set of ~24-28, so the whole file fits in memory
    comfortably — and an eager read is what lets `malformed` be *returned* rather than
    logged into the same stream nobody is reading.
    """

    events: tuple[Event, ...]
    #: Lines that could not be parsed as an event, counted rather than raised or dropped.
    #: A run killed mid-write leaves a truncated final line, and losing a whole run's
    #: samples to it would be worse than skipping it — but skipping in silence would let a
    #: half-unreadable file present as a complete one.
    malformed: int
    path: Path

    def of(self, *names: str) -> tuple[Event, ...]:
        """Every event whose name is one of `names`, in the order it was written."""
        wanted = frozenset(names)
        return tuple(event for event in self.events if event.event in wanted)

    def samples(self, name: str, field: str) -> Samples:
        """`field` across every `name` line — the values present, and how many were not.

        The unit T10 measures a budget from: `samples("retrieval", "latency_ms")` for
        ADR-0005's translation budget, `samples("input_gate", "latency_ms")` for ADR-0006's,
        `samples("agent_query", "verbatim")` for the divergence rate.
        """
        selected = self.of(name)
        present = tuple(event.fields[field] for event in selected if field in event.fields)
        return Samples(present=present, absent=len(selected) - len(present))

    def by_turn(self) -> Mapping[str | None, tuple[Event, ...]]:
        """Events grouped by `turn_id`, so a question's lines can be read together.

        The join AC-4's "per-chunk provenance, machine-readable" needs: without it a
        `retrieval` line's provenance can be aggregated but never attributed, because the
        only other correlation available is line order — which holds for a serial harness,
        breaks under the agent's parallel tool calls, and fails no assertion either way.
        """
        grouped: dict[str | None, list[Event]] = {}
        for event in self.events:
            grouped.setdefault(event.turn_id, []).append(event)
        return {turn_id: tuple(events) for turn_id, events in grouped.items()}


def read_events(path: Path | str) -> EventLog:
    """Parse the sink at `path`. Raises if it does not exist; never raises on its contents.

    A line is an event when it is a JSON object carrying an `event` name and a parsable
    `ts`; anything else is counted in `malformed`. The envelope keys are validated because
    they are the emitter's, so a line missing one was not written by `log_event` — while the
    *payload* is validated not at all, because a field the emitter has not added yet is the
    normal state of a file that outlived a deploy.
    """
    path = Path(path)
    events: list[Event] = []
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parsed = _event(line)
            if parsed is None:
                malformed += 1
            else:
                events.append(parsed)
    return EventLog(events=tuple(events), malformed=malformed, path=path)


def _event(line: str) -> Event | None:
    """One line as an `Event`, or `None` if it is not one. The only place parsing decides."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    name, raw_ts = payload.get("event"), payload.get("ts")
    if not isinstance(name, str) or not isinstance(raw_ts, str):
        return None
    try:
        ts = datetime.fromisoformat(raw_ts)
    except ValueError:
        return None
    fields = payload.get("fields")
    return Event(
        event=name,
        ts=ts,
        level=str(payload.get("level", "")),
        logger=str(payload.get("logger", "")),
        # `fields` is absent from the envelope entirely when the payload was empty
        # (`JsonLinesFormatter` omits it), which is an empty payload and not a malformed line.
        fields=fields if isinstance(fields, dict) else {},
        turn_id=payload.get("turn_id"),
        error=payload.get("error"),
    )
