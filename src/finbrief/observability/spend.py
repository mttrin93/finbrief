"""What this conversation cost — read back out of the log T8 already writes (T12 item 5, #13).

**This module parses nothing.** `observability/events.py` is the one reader of the sink,
paired with the one emitter, and adding a second parser here is exactly the drift that pairing
exists to prevent: a renamed field would read back as `None`, `None` sums as nothing, and the
total would narrow its own denominator silently. So this takes an `EventLog` and does
arithmetic on it.

**Two events, because the spend is in two places.** `agent_turn` carries the answering loop's
own calls summed over the turn, and `query_translation` carries the planner's — kept apart in
the log because ADR-0005 judges translation on the cost *it* adds (ADR-0011). A reader wanting
the whole
per-turn spend joins them on `turn_id`, and that is what this does. The gate's classifier is the
one paid call in the system that is **not** metered, on purpose and recorded in ADR-0011: it
returns a bare `Verdict`, so metering it means changing that return type for the cheapest call
there is. The meter therefore reports what it can measure and this module says which call it
cannot, because a total presented as "the conversation" while missing a known call is the
narrowing failure this repo keeps finding.

**Scoped to the conversation by `turn_id`, and that is why no offset is needed here.** The
sink is append-only across every run and app session that names it, so a statistic over the
whole file is a statistic over all of them — the defect ADR-0011's T10 amendment records, where
a planner p50 was reported over 13 appended runs. T10's fix is `sink_offset`, because a *harness
run* has no key of its own on a line. A conversation does: `app/Home.py` opens every turn as
`f"{thread_id}:{suffix}"` and `thread_id` is a `uuid4` minted per session, so the prefix selects
this conversation's lines and nothing else — a sharper filter than a byte offset, which would
also admit another tab writing concurrently. The app passes an offset anyway, but for read cost
rather than for correctness.

**Absence is per field, never zero.** That rule is `observability/tokens.py`'s and it is
repeated here because this is the module that would break it: a provider can report
`input_tokens` and not `output_tokens`, so each field carries its own denominator and a field
nothing reported is `None` rather than `0`. A zero here is a claim that calls were free.
"""

from __future__ import annotations

from dataclasses import dataclass

from finbrief.observability.events import Event, EventLog

#: The two events that carry a token count and belong to a chat round.
TOKEN_EVENTS = ("agent_turn", "query_translation")

#: The token fields `tokens.usage_fields` writes. `total_tokens` is deliberately not among them
#: — it is `input + output` and a third number is a third thing that can disagree.
TOKEN_FIELDS = ("input_tokens", "output_tokens")

#: The `max_sub_queries` value at which the planner makes **no chat round at all** — ADR-0004
#: §6: a cap of 0 removes the `model.invoke`, it does not truncate its output. So a
#: `query_translation` line at this cap is a line behind zero calls, and counting it as a call
#: that failed to report usage would report a *complete* total as partial.
#:
#: `evaluation/latency.py` asserts the identical fact for the identical reason, under
#: `PLANNER_DISABLED_CAP`, and the two are bound to each other by
#: `test_spend.py::test_the_planner_cap_agrees_with_the_one_the_latency_pool_uses` rather than
#: left to drift. It is duplicated rather than imported because `evaluation/` is the harness and
#: the app must not depend on it.
PLANNER_SILENT_CAP = 0


@dataclass(frozen=True, slots=True)
class Tokens:
    """One token field's total, and how many of the calls behind it reported one.

    `total` is `None` when **no** call reported this field, which is not the same as zero: a
    provider that returns no `usage` block did not perform a free call. `reported_calls` against
    `calls` is what makes a partial total visible instead of merely wrong — the `usage_total`
    defect was a single denominator counting a reply as metered if it reported *any* usage, so a
    provider returning half a pair made the other half read as summed-and-complete.
    """

    total: int | None
    reported_calls: int
    calls: int

    @property
    def measured(self) -> bool:
        """Whether anything reported this field at all."""
        return self.total is not None

    @property
    def partial(self) -> bool:
        """Whether some of the calls behind this total reported no count for it.

        Guarded on `measured`, because "partial" is a claim *about a total*: with nothing
        reported there is no total to be incomplete, and the caller's honest sentence is "not
        measured" rather than "partial". Without the guard every unmeasured field read as
        partial, which is an absence reported as a measurement of a different kind.
        """
        return self.measured and self.reported_calls < self.calls


@dataclass(frozen=True, slots=True)
class Spend:
    """A conversation's measured token spend, and the denominators behind it."""

    turns: int
    calls: int
    input: Tokens
    output: Tokens

    @property
    def measured(self) -> bool:
        """Whether any call in this conversation reported any count."""
        return self.input.measured or self.output.measured

    @property
    def partial(self) -> bool:
        """Whether a total is shown here that is not the whole story.

        Three ways for that to be true, and the third is easy to miss: a field's own calls did
        not all report it, or **one field was reported and the other was not at all** — an
        input-only pair is a partial picture of a call's cost even though `input.partial` is
        false. Nothing measured is not partial; it is unmeasured, and the caller says that
        instead.
        """
        if not self.measured:
            return False
        return (
            self.input.partial
            or self.output.partial
            or not self.input.measured
            or not self.output.measured
        )

    def dollars(
        self, *, input_per_mtok: float | None, output_per_mtok: float | None
    ) -> float | None:
        """The cost of this spend at the configured prices, or `None` when it cannot be priced.

        `None` for either of two reasons, and the caller says which: no price is configured (the
        default — see `config.Settings.input_cost_per_mtok`), or nothing reported the tokens a
        price would multiply. Both are absences and neither is `$0.00`, which on a spend panel
        would read as "this conversation was free".

        A **partial** total is still priced, because the tokens in it were really spent; what
        the caller owes beside the figure is the fact that it is a floor rather than a total,
        which is `partial` above.
        """
        prices = (
            (self.input.total, input_per_mtok),
            (self.output.total, output_per_mtok),
        )
        priced = [
            tokens * price / 1_000_000
            for tokens, price in prices
            if tokens is not None and price is not None
        ]
        return sum(priced) if priced else None


def conversation_spend(log: EventLog, *, thread_id: str) -> Spend:
    """The spend on `thread_id`'s lines in `log` — see the module docstring for the scoping.

    A line whose `turn_id` is absent, or belongs to another conversation, is not this
    conversation's: an evaluation run and a second browser tab both write to the same file.
    """
    prefix = f"{thread_id}:"
    mine = [
        event
        for event in log.of(*TOKEN_EVENTS)
        if event.turn_id is not None and event.turn_id.startswith(prefix)
    ]
    totals: dict[str, int | None] = dict.fromkeys(TOKEN_FIELDS)
    reported: dict[str, int] = dict.fromkeys(TOKEN_FIELDS, 0)
    calls = 0
    for event in mine:
        made = _calls_behind(event)
        calls += made
        for name in TOKEN_FIELDS:
            count = event.field(name)
            if count is None:
                continue
            totals[name] = count + (totals[name] or 0)
            # How many of *this line's* calls carried the field. `agent_turn` sums a whole turn
            # and says so per field; `query_translation` is one reply, so the field's presence
            # is its own denominator.
            reported[name] += event.field(f"{name}_calls", made)
    return Spend(
        # Turns, not lines: a turn writes one `agent_turn` line and one `query_translation` line
        # per search, so counting lines would report a two-search turn as three turns.
        turns=len({event.turn_id for event in mine}),
        calls=calls,
        input=Tokens(totals["input_tokens"], reported["input_tokens"], calls),
        output=Tokens(totals["output_tokens"], reported["output_tokens"], calls),
    )


def _calls_behind(event: Event) -> int:
    """How many paid chat calls one line represents.

    `agent_turn` says so itself (`calls`, written only once something was reported — so its
    absence means the turn reported no usage at all, not that it made none, and the fallback is
    the honest floor of one). A `query_translation` line is one call, or **none** at
    `PLANNER_SILENT_CAP`: at a cap of 0 no `model.invoke` happens, so charging that line a call
    would inflate the denominator and report a complete total as partial.
    """
    if event.event == "query_translation":
        cap = event.field("max_sub_queries")
        return 0 if cap == PLANNER_SILENT_CAP else 1
    return int(event.field("calls", 1))
