"""ADR-0005's latency budget, read from the log — and the refusals that keep it honest.

The clause is "dominance is judged within **≤1.5s p50 added by translation**, measured from
the Phase-6 structured logs". Three things about measuring it here rather than with a
stopwatch, all of which #11's non-negotiable 2 turns on:

**The sink is off unless named, so absence is the likely state.** `.env.example` ships
`FINBRIEF_LOG_FILE` commented out and `configure_logging()` takes no arguments, so a fresh
checkout's evaluation run records nothing. A p50 over zero samples that renders as "budget
met" is the exact failure this repo enforces against, so every function here raises rather
than returning a number it could not compute. `scripts/evaluate.py` also checks the switch
*before* spending anything, because discovering it afterwards costs the whole run.

**The planner's round and the retrieval rounds are separate lines, and only one of them is
what the clause is about.** `query_translation.latency_ms` is the chat round translation adds;
`retrieval.latency_ms` is the whole retrieval, timed from before the translate branch.
ADR-0011 calls the split the interesting half, and `events.Samples`' own docstring says the
budget takes two calls and not one.

**A replayed arm's planner latency is not the shipped path's**, and this is the subtlety the
harness introduces. ADR-0004 §9's replay means the scored `+translation` arms emit
`query_translation` lines with a stub in place of a model — near-zero, and meaningless as a
cost. So the planner's real cost is measured from the **resolve** pass, where the planner
actually runs, and `planner_added_ms` reads only lines that carry token counts: a line with no
`input_tokens` reported no spend because no model was called. That filter is the difference
between reporting translation's cost and reporting the stub's.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from finbrief.observability.events import EventLog, read_events


class NoSamples(RuntimeError):
    """A statistic was asked for over a log that carries none of the lines it needs."""


class SinkMissing(RuntimeError):
    """The sink was never enabled, so there is no log to read."""


def load_log(path: Path | str | None) -> EventLog:
    """The events at `path`, refusing both an unnamed sink and a missing file.

    `read_events` already raises on a missing file — "nobody enabled the log" read as "this
    run emitted nothing" is how a p50 over zero samples gets reported as a budget met
    (`observability/events.py`). This adds the `None` case, which is the one a fresh checkout
    hits.
    """
    if path is None:
        raise SinkMissing(
            "FINBRIEF_LOG_FILE was never set, so this run emitted no persisted events. "
            "ADR-0005's '<=1.5s p50 added by translation' cannot be measured from nothing, and "
            "reporting it as met would be an absence presented as a measurement."
        )
    return read_events(path)


def p50(values: Sequence[float], *, what: str) -> float:
    """The median of `values`, or `NoSamples` naming what was missing.

    Never `0.0` for an empty sequence, and never `None` either: a caller that got a float
    knows it is a measurement, which is the whole point of raising here.
    """
    if not values:
        raise NoSamples(
            f"no samples for {what}: the log carries none of those lines. A median over zero "
            f"samples is not a budget met — enable FINBRIEF_LOG_FILE and re-run the stage that "
            f"emits them."
        )
    return float(statistics.median(values))


@dataclass(frozen=True, slots=True)
class TranslationCost:
    """What translation added, split the way ADR-0005's clause needs it.

    `planner_p50_ms` is the chat round; `retrieval_delta_p50_ms` is the difference between the
    `+translation` and `−translation` retrieval medians, which is the retrieval rounds the
    extra variants cost. The clause is judged on their sum, and both halves are printed so a
    reader can see which one spent the budget — ADR-0004's amendment predicts the normalised
    variant costs a retrieval round and never a chat round, and this is where that shows.
    """

    planner_p50_ms: float
    planner_samples: int
    retrieval_translated_p50_ms: float
    retrieval_untranslated_p50_ms: float
    translated_samples: int
    untranslated_samples: int
    budget_ms: float

    @property
    def retrieval_delta_p50_ms(self) -> float:
        return self.retrieval_translated_p50_ms - self.retrieval_untranslated_p50_ms

    @property
    def added_p50_ms(self) -> float:
        """Total p50 translation adds: the planner's round plus the extra retrieval rounds.

        The planner's round is *inside* `retrieval.latency_ms` on a live turn, but not on this
        harness's scored arms — they replay it through a stub. So adding the resolve pass's
        planner median to the retrieval delta is the honest reconstruction of what the shipped
        path pays, and it is stated as a reconstruction rather than printed as one
        measurement.
        """
        return self.planner_p50_ms + self.retrieval_delta_p50_ms

    @property
    def within_budget(self) -> bool:
        return self.added_p50_ms <= self.budget_ms


def translation_cost(log: EventLog, *, budget_ms: float = 1500.0) -> TranslationCost:
    """ADR-0005's added-latency p50, from the two kinds of line that carry it.

    Raises `NoSamples` if either half is missing, rather than substituting a zero for the half
    it could not measure — a budget met because one of its two terms was silently absent is
    not met.
    """
    planner = [
        event.field("latency_ms")
        for event in log.of("query_translation")
        # Only a line that reported spend, so the stub-replayed arms cannot flatter the number.
        # A `query_translation` line with no `input_tokens` called no model (see the module
        # docstring, and `query_translation.translate`'s own honest-absence comment).
        if event.field("input_tokens") is not None and event.field("latency_ms") is not None
    ]
    translated: list[float] = []
    untranslated: list[float] = []
    for event in log.of("retrieval"):
        latency = event.field("latency_ms")
        if latency is None:
            continue
        (translated if event.field("translation") else untranslated).append(latency)
    return TranslationCost(
        planner_p50_ms=p50(planner, what="query_translation lines with token counts"),
        planner_samples=len(planner),
        retrieval_translated_p50_ms=p50(translated, what="retrieval lines with translation on"),
        retrieval_untranslated_p50_ms=p50(
            untranslated, what="retrieval lines with translation off"
        ),
        translated_samples=len(translated),
        untranslated_samples=len(untranslated),
        budget_ms=budget_ms,
    )


@dataclass(frozen=True, slots=True)
class TokenSpend:
    """What the run's metered calls reported, per event, with the unmetered half named.

    Absence is per field, not per record (`observability/tokens.py`): a provider that returned
    half a pair must not put a fabricated zero on the line, so each count carries its own call
    count and a mean is always over the calls that reported *that* field.
    """

    event: str
    input_tokens: int
    output_tokens: int
    input_calls: int
    output_calls: int
    lines: int

    @property
    def unmetered_lines(self) -> int:
        """Lines that reported no input count at all — a call whose cost is unknown, not "
        "free."""
        return self.lines - self.input_calls


def token_spend(log: EventLog, event: str) -> TokenSpend:
    """Token counts across `event`'s lines, counting each field's denominator separately."""
    lines = log.of(event)
    inputs = [e.field("input_tokens") for e in lines if e.field("input_tokens") is not None]
    outputs = [e.field("output_tokens") for e in lines if e.field("output_tokens") is not None]
    return TokenSpend(
        event=event,
        input_tokens=sum(inputs),
        output_tokens=sum(outputs),
        input_calls=len(inputs),
        output_calls=len(outputs),
        lines=len(lines),
    )
