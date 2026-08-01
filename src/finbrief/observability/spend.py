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
there is. The meter therefore reports what it can measure and this module owns the sentence
saying which call it cannot (`UNMETERED_CLASSIFIER_NOTE`, rendered by the analytics page),
because a total presented as "the conversation" while missing a known call is the narrowing
failure this repo keeps finding.

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

**A turn with no total at all is a third kind of absence**, and it is counted rather than folded
into the arithmetic. `agent_turn` is written *last*, so a turn still being answered — or one
that ended without reporting — leaves answering lines behind with no total beside them;
`unfinished` is how many, and a caller printing a figure owes that count. Folding it in would be
the same narrowing in a new place: the answering calls of such a turn were really made.
"""

from __future__ import annotations

from dataclasses import dataclass

from finbrief.observability.events import Event, EventLog

#: The two events that carry a token count and belong to a chat round.
TOKEN_EVENTS = ("agent_turn", "query_translation")

#: The caveat every surface rendering these totals is describing — **one string, because it was
#: two.** `app/Home.py`'s sidebar meter and `app/pages/1_Analytics.py` carried the same sentence
#: typed out twice, which is the drift this repo keeps catching: the copies disagree on the turn
#: one of them is edited, and the one on screen is whichever page the reader opened.
#:
#: The claim itself is ADR-0011 §4's — the gate's classifier is a paid call that never enters
#: `calls`, so no value of `Spend.partial` is evidence about it and a *complete* total is still
#: missing one call per turn. It is rendered on the analytics page, which is the surface that
#: totals a whole log; the sidebar shows one conversation's figures and does not repeat it
#: (#14 copy pass).
UNMETERED_CLASSIFIER_NOTE = "One call per turn isn't metered, so these figures are incomplete."

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

#: The lines the answering path writes *while* a turn is being answered — one per search
#: (`retrieval/retrieve.py`) and one per planner round (`retrieval/query_translation.py`). Their
#: presence under a turn id is the log's only sign that the answering path **started**.
#:
#: `input_gate` is deliberately not among them, and the omission is the whole point of the list
#: being narrow: a question the gate refuses writes a gate line, no answering line and no
#: `agent_turn`, and it is a turn that *finished* — it never reached the agent. Counting it as
#: unfinished would report a refusal as a measurement in progress.
ANSWERING_EVENTS = ("retrieval", "query_translation")

#: The line `agent/agent.py` writes **last**, once the turn has an answer. It is therefore the
#: completion marker, and its absence beside an answering line is not "this turn was free": it
#: is a turn with no total yet — one being answered now, or one that ended without reporting.
COMPLETED_EVENT = "agent_turn"


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
    #: How many of the lines behind `calls` did not report a call count of their own, so that
    #: `calls` counted each of them as **one**.
    #:
    #: `agent_turn` writes `calls` only once something reported usage (`tokens.usage_total`), so
    #: a turn that made five calls and metered none of them arrives here as a line with no count
    #: on it. Counting it as one is the honest floor — it certainly made a call — but `calls` is
    #: then a floor and not a count, and a caller printing it owes the reader that word. Without
    #: this field it could not: a floor rendered as a total is the narrowing this whole path is
    #: built against, and `partial` cannot carry it because `partial` is about *token* reporting
    #: and this is about the denominator itself (code review of #13).
    floored: int
    #: How many of this conversation's turns started answering and never wrote a total —
    #: `ANSWERING_EVENTS` under a turn id with no `COMPLETED_EVENT` beside it.
    #:
    #: **An absence with two causes and one honest sentence**, which is why it is a count here
    #: and not a boolean called `in_progress`: a turn being answered at this moment and a turn
    #: that raised halfway through leave the log in the same shape, and the reader is owed both
    #: readings rather than the flattering one. What it is *not* is a spend of zero — the
    #: answering loop's calls for such a turn were really made and are really missing from the
    #: figures beside this, and a total that omits them without saying so is the silent
    #: narrowing this module exists against.
    #:
    #: Separate from `partial` on the rule this file already states twice: `partial` is about
    #: *reported versus counted* calls on lines that exist, and this is about a line that never
    #: arrived. Two different claims get two different fields and two different sentences.
    unfinished: int
    input: Tokens
    output: Tokens
    #: The models this conversation's **answering** turns ran on — one `agent_turn`'s `model`
    #: field per turn, and `None` among them for a turn that recorded none (T14, #15).
    #:
    #: A set rather than a single value, because one conversation can hold several: the
    #: checkpointer is keyed on `thread_id` and not on the model, so switching the picker
    #: mid-conversation keeps the history and adds a second model to the same thread.
    #:
    #: **`query_translation` lines are deliberately not in here.** The planner builds its own
    #: model with `build_chat_model(settings)` and takes no override
    #: (`retrieval/retrieve.py`), so its line is always on the configured model and carries no
    #: `model` field at all — reading those absences in would make every translated turn
    #: unattributed for a reason that does not exist.
    models: frozenset[str | None] = frozenset()

    @property
    def measured(self) -> bool:
        """Whether any call in this conversation reported any count."""
        return self.input.measured or self.output.measured

    def all_answered_on(self, model: str) -> bool:
        """Whether every answering turn here ran on `model`, and something said so.

        An **equality** against a one-element set, which is what makes each of the three ways to
        fail fail: a second model in the conversation, a turn that recorded no model at all, and
        a conversation with no answering line yet. The last two are absences and this returns
        `False` for them, because "nothing attributed this" is not "this ran on the priced
        model" — the distinction the whole module is built around.
        """
        return self.models == frozenset({model})

    @property
    def calls_are_a_floor(self) -> bool:
        """Whether `calls` counts every call, or only every call it could see."""
        return self.floored > 0

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
        self,
        *,
        input_per_mtok: float | None,
        output_per_mtok: float | None,
        priced_model: str,
    ) -> float | None:
        """The cost of this spend at the configured prices, or `None` when it cannot be priced.

        `None` for **three** reasons now, and the caller says which: no price is configured (the
        default — see `config.Settings.input_cost_per_mtok`), nothing reported the tokens a
        price would multiply, or the turns did not all run on `priced_model`. All three are
        absences and none is `$0.00`, which on a spend panel would read as "this conversation
        was free".

        **The third is T14's (#15), and it is the same shape as `Tokens.partial`.** The two
        price knobs are a single pair configured for one model, and no per-model rate card
        ships — ADR-0011 refused one because OpenRouter fronts many upstreams and routes by
        availability, so a price in this repo is a figure nobody measured going stale in the one
        panel about spend. Four selectable models make that stronger, not weaker. What they also
        make reachable is a turn on Haiku multiplied by the gpt-4o-mini rate, and **a wrong
        dollar figure is worse than no dollar figure** — especially here, where the wrongness is
        invisible because the tokens behind it are real.

        `priced_model` is a **required** keyword rather than an optional check. An optional one
        defaults to not checking, which is how the wrong figure would survive in every caller
        that had not been updated — and the two callers here are the two surfaces that render
        spend.

        A **partial** total is still priced, because the tokens in it were really spent; what
        the caller owes beside the figure is the fact that it is a floor rather than a total,
        which is `partial` above.
        """
        if not self.all_answered_on(priced_model):
            return None
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

    def ours(*names: str) -> list[Event]:
        return [
            event
            for event in log.of(*names)
            if event.turn_id is not None and event.turn_id.startswith(prefix)
        ]

    mine = ours(*TOKEN_EVENTS)
    # Turn *ids*, differenced. A turn writes one answering line per search and one completion
    # line, so counting lines would call a two-search turn two unfinished ones; and the
    # difference is taken over sets so a turn whose answering line and completion line are both
    # present cancels out however many of the first it wrote.
    started = {event.turn_id for event in ours(*ANSWERING_EVENTS)}
    completed = {event.turn_id for event in ours(COMPLETED_EVENT)}
    totals: dict[str, int | None] = dict.fromkeys(TOKEN_FIELDS)
    reported: dict[str, int] = dict.fromkeys(TOKEN_FIELDS, 0)
    calls = 0
    floored = 0
    for event in mine:
        made, is_floor = calls_behind(event)
        calls += made
        floored += is_floor
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
        floored=floored,
        unfinished=len(started - completed),
        input=Tokens(totals["input_tokens"], reported["input_tokens"], calls),
        output=Tokens(totals["output_tokens"], reported["output_tokens"], calls),
        # Answering lines only — `COMPLETED_EVENT` and not `TOKEN_EVENTS`. See `Spend.models`
        # for why the planner's line is excluded rather than read as an absence.
        models=frozenset(event.field("model") for event in ours(COMPLETED_EVENT)),
    )


def calls_behind(event: Event) -> tuple[int, bool]:
    """How many paid chat calls one line represents, and whether that number is a floor.

    **Public because `observability/analytics.py` shares it rather than copying it** (T13, #14).
    The dashboard totals the same two events over a whole sink where this panel totals them over
    one conversation, and "how many paid calls does this line stand behind" is one question: two
    implementations of it are two answers the day one of them learns about a new cap.

    `agent_turn` says so itself (`calls`, written only once something was reported — so its
    absence means the turn reported no usage at all, not that it made none, and the fallback is
    the honest floor of one). **The floor is returned as a floor**, because a caller printing
    the total owes the reader the difference: a five-call turn that metered nothing arrives as a
    line worth `1` here, and `Spend.floored` is what stops that being displayed as a count
    (code review of #13).

    A `query_translation` line is one call, or **none** at `PLANNER_SILENT_CAP`: at a cap of 0
    no `model.invoke` happens, so charging that line a call would inflate the denominator and
    report a complete total as partial.
    """
    if event.event == "query_translation":
        cap = event.field("max_sub_queries")
        return (0 if cap == PLANNER_SILENT_CAP else 1), False
    reported = event.field("calls")
    return (1, True) if reported is None else (int(reported), False)
