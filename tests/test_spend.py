"""The token and cost meter, over the log T8 writes (T12 item 5, #13).

Driven through the **real emitter and the real reader** — `log_event` writes the lines and
`events.read_events` parses them — for the reason `test_event_log.py` does it: the pair is what
keeps a renamed field from reading back as `None` and summing as nothing, and a test that
hand-built `Event` objects would prove the arithmetic while missing exactly that.
"""

from __future__ import annotations

import logging

import pytest

from finbrief.evaluation.latency import PLANNER_DISABLED_CAP
from finbrief.observability.events import read_events
from finbrief.observability.logging_setup import log_event, turn
from finbrief.observability.spend import PLANNER_SILENT_CAP, conversation_spend
from finbrief.observability.tokens import usage_total

THREAD = "3294dcff-0f78-4e82-a07c-47e8552e378f"
OTHER_THREAD = "aaaaaaaa-0f78-4e82-a07c-47e8552e378f"


class Reply:
    """A model reply that reports the usage a provider chose to report, and no more."""

    def __init__(self, **usage):
        self.usage_metadata = dict(usage) if usage else None


@pytest.fixture
def sink(tmp_path, monkeypatch):
    """A real JSON-lines sink, written by the real handler."""
    from finbrief.observability import logging_setup

    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("FINBRIEF_LOG_FILE", str(path))
    logging_setup.configure_logging()
    return path


@pytest.fixture
def emitter():
    return logging.getLogger("finbrief.test_spend")


def an_agent_turn(emitter, *replies, thread_id=THREAD, suffix="aaaa"):
    """One `agent_turn` line, metered exactly as `agent/agent.py` meters one."""
    with turn(f"{thread_id}:{suffix}"):
        log_event(emitter, "agent_turn", thread_id=thread_id, **usage_total(replies))


def a_planner_call(emitter, reply=None, *, cap=3, thread_id=THREAD, suffix="aaaa"):
    """One `query_translation` line, metered exactly as `query_translation.py` meters one."""
    from finbrief.observability.tokens import usage_fields

    with turn(f"{thread_id}:{suffix}"):
        log_event(
            emitter,
            "query_translation",
            max_sub_queries=cap,
            sub_queries=0,
            **(usage_fields(reply) if reply is not None else {}),
        )


def spend_from(path, thread_id=THREAD):
    return conversation_spend(read_events(path), thread_id=thread_id)


def test_the_meter_sums_the_turns_of_one_conversation(sink, emitter):
    an_agent_turn(emitter, Reply(input_tokens=120, output_tokens=40), suffix="one")
    an_agent_turn(emitter, Reply(input_tokens=80, output_tokens=20), suffix="two")

    spend = spend_from(sink)

    assert spend.turns == 2
    assert spend.calls == 2
    assert spend.input.total == 200
    assert spend.output.total == 60
    assert not spend.partial, "both calls reported both fields"


def test_the_planners_own_line_is_part_of_the_conversations_spend(sink, emitter):
    # Kept apart in the log because ADR-0005 judges translation on the cost *it* adds
    # (ADR-0011), so a reader wanting the whole per-turn spend has to join the two. A meter that
    # read only `agent_turn` would under-report every translated turn and never say so.
    an_agent_turn(emitter, Reply(input_tokens=100, output_tokens=30))
    a_planner_call(emitter, Reply(input_tokens=25, output_tokens=8))

    spend = spend_from(sink)

    assert spend.calls == 2
    assert spend.input.total == 125
    assert spend.output.total == 38


def test_another_conversation_in_the_same_file_is_not_this_ones_spend(sink, emitter):
    """**The defect ADR-0011's T10 amendment records, in the app rather than the harness.**

    The sink is append-only across every run and session that names it, so a total over the
    whole file is a total over all of them — which is how a planner p50 came to be reported over
    13 appended runs while the artifact claimed it was one run's. Here the scoping key is on the
    line already: every turn id is `f"{thread_id}:{suffix}"`, so the prefix selects this
    conversation.
    """
    an_agent_turn(emitter, Reply(input_tokens=100, output_tokens=30))
    an_agent_turn(emitter, Reply(input_tokens=9999, output_tokens=9999), thread_id=OTHER_THREAD)

    spend = spend_from(sink)

    assert spend.turns == 1
    assert spend.input.total == 100, "the other tab's spend is not this conversation's"


def test_a_line_written_outside_a_turn_belongs_to_no_conversation(sink, emitter):
    # An evaluation run drives `retrieve()` directly and opens no turn scope, so its
    # `query_translation` lines carry no `turn_id`. Attributing them to whichever conversation
    # happens to be reading would put a harness's spend on an analyst's meter.
    log_event(emitter, "query_translation", max_sub_queries=3, sub_queries=1, input_tokens=5000)
    an_agent_turn(emitter, Reply(input_tokens=100, output_tokens=30))

    spend = spend_from(sink)

    assert spend.input.total == 100


# --------------------------------------------------------------------------------------
# Absence, per field — the `usage_total` defect this meter would otherwise repeat
# --------------------------------------------------------------------------------------


def test_a_field_no_call_reported_is_absent_and_not_zero(sink, emitter):
    # The rule `observability/tokens.py` states and this is the module that would break it: a
    # provider that returns no `usage` block did not perform a free call. `0` on a spend panel
    # is a claim, and it is the wrong one.
    an_agent_turn(emitter, Reply())

    spend = spend_from(sink)

    assert spend.input.total is None
    assert spend.output.total is None
    assert not spend.measured, "nothing reported anything, so there is no total"
    assert not spend.partial, "and nothing to call partial either — it is unmeasured"


def test_half_a_reported_pair_is_partial_and_says_so(sink, emitter):
    """The exact shape of the `usage_total` defect: `{"input_tokens": 120}` and nothing else.

    A single denominator counted a reply as metered if it reported *any* usage, so the missing
    half read as summed-and-complete. Here `output` is absent, `input` is a real total, and the
    pair is flagged — because an input-only figure is a partial picture of what a call cost even
    though the input side is complete.
    """
    an_agent_turn(emitter, Reply(input_tokens=120))

    spend = spend_from(sink)

    assert spend.input.total == 120
    assert spend.input.reported_calls == 1
    assert not spend.input.partial, "every call there was reported this field"
    assert spend.output.total is None
    assert spend.partial, "one field measured and the other absent is still a partial total"


def test_one_reporting_call_among_several_narrows_only_its_own_field(sink, emitter):
    # Two calls in one agent turn, one of which reported nothing. The total is real and it is a
    # floor, and the denominator is what makes that visible rather than merely true.
    an_agent_turn(emitter, Reply(input_tokens=100, output_tokens=30), Reply())

    spend = spend_from(sink)

    assert spend.calls == 2
    assert spend.input.total == 100
    assert spend.input.reported_calls == 1
    assert spend.input.partial, "2 calls, 1 reported — the total is a floor"
    assert spend.partial


def test_a_planner_that_never_ran_is_not_a_call_that_failed_to_report(sink, emitter):
    """**A complete total must not be reported as partial** — the mirror of the usual bug.

    At `max_sub_queries=0` translation reduces to the deterministic ticker form and asks a model
    nothing (ADR-0004 §6: the cap removes the `model.invoke`), so its line stands behind
    **zero**
    calls. Charging it one would put an unreportable call in the denominator and make a fully
    reported conversation read as partial — an absence invented rather than preserved.
    """
    an_agent_turn(emitter, Reply(input_tokens=100, output_tokens=30))
    a_planner_call(emitter, cap=PLANNER_SILENT_CAP)

    spend = spend_from(sink)

    assert spend.calls == 1, "the planner made no chat round at this cap"
    assert not spend.partial, "so nothing is missing from the total"


def test_a_planner_that_ran_and_reported_nothing_is_a_call_in_the_denominator(sink, emitter):
    # The contrast that makes the test above a measurement rather than a coincidence: same line,
    # a cap that *does* invoke the model, no usage reported. That call is real and unreported.
    an_agent_turn(emitter, Reply(input_tokens=100, output_tokens=30))
    a_planner_call(emitter, cap=3)

    spend = spend_from(sink)

    assert spend.calls == 2
    assert spend.input.partial, "the planner's round is in the denominator and reported nothing"


def test_a_turn_that_metered_nothing_makes_the_call_count_a_floor(sink, emitter):
    """`calls` is a floor, not a count, and the shape has to say which.

    `tokens.usage_total` writes `calls` only once something reported usage, so a turn that made
    five calls and metered none of them arrives as a line with **no** count on it. One is the
    honest floor — it certainly made a call — but a floor rendered as a count is the same
    fabrication in the denominator that this module refuses in the numerator, and the surface
    could not say so because nothing carried the distinction (code review of #13).

    Paired with the reported case below rather than asserted alone: a `floored` that was always
    truthy would pass a negative-only test.
    """
    an_agent_turn(emitter)  # no replies at all, so `usage_total` writes no `calls`
    a_planner_call(emitter, Reply(input_tokens=40, output_tokens=20), cap=3)

    spend = spend_from(sink)

    assert spend.calls == 2, "the floor of one for the turn, plus the planner's real round"
    assert spend.floored == 1
    assert spend.calls_are_a_floor, "so the count may not be displayed as a total"


def test_a_turn_that_reported_its_calls_is_a_count_and_not_a_floor(sink, emitter):
    # The other half. Same two lines, and the only difference is that the turn metered itself,
    # which is what makes `calls_are_a_floor` a measurement of the line and not of the shape.
    an_agent_turn(emitter, Reply(input_tokens=100, output_tokens=30))
    a_planner_call(emitter, Reply(input_tokens=40, output_tokens=20), cap=3)

    spend = spend_from(sink)

    assert spend.calls == 2
    assert spend.floored == 0
    assert not spend.calls_are_a_floor


def test_the_planner_cap_agrees_with_the_one_the_latency_pool_uses():
    """Two modules asserting the same fact about ADR-0004 §6, bound rather than left to drift.

    `evaluation/latency.py` needs it to keep the ablation arms out of a planner latency pool and
    this module needs it to keep them out of a spend denominator. It is duplicated rather than
    imported because `evaluation/` is the harness and the app must not depend on it — so the
    duplication is checked here instead, which is the mechanism this repo uses for a fact that
    has
    to live in two places.
    """
    assert PLANNER_SILENT_CAP == PLANNER_DISABLED_CAP


# --------------------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------------------


def test_tokens_are_priced_only_when_a_price_is_configured(sink, emitter):
    # Unpriced is the default and it is an absence, not `$0.00`: this project reaches every
    # model through OpenRouter, which routes by availability, so a price in the repo would be a
    # figure nobody measured going stale in the one panel about spend.
    an_agent_turn(emitter, Reply(input_tokens=1_000_000, output_tokens=1_000_000))
    spend = spend_from(sink)

    assert spend.dollars(input_per_mtok=None, output_per_mtok=None) is None
    assert spend.dollars(input_per_mtok=0.15, output_per_mtok=0.60) == pytest.approx(0.75)


def test_an_unmeasured_conversation_cannot_be_priced_however_the_prices_are_set(sink, emitter):
    an_agent_turn(emitter, Reply())
    spend = spend_from(sink)

    assert spend.dollars(input_per_mtok=0.15, output_per_mtok=0.60) is None


def test_a_partial_total_is_still_priced_because_those_tokens_were_really_spent(sink, emitter):
    # Priced, and the caller owes the reader `partial` beside it: the figure is a floor.
    # Refusing to price it would be the opposite error — withholding a real cost because it is
    # incomplete.
    an_agent_turn(emitter, Reply(input_tokens=1_000_000))
    spend = spend_from(sink)

    assert spend.dollars(input_per_mtok=0.15, output_per_mtok=0.60) == pytest.approx(0.15)
    assert spend.partial


def test_an_empty_log_is_an_unmeasured_conversation_and_not_a_free_one(sink, emitter):
    spend = spend_from(sink)

    assert spend.turns == 0
    assert spend.calls == 0
    assert not spend.measured
    assert spend.unfinished == 0, "nothing has started, so nothing is unfinished"
    assert spend.dollars(input_per_mtok=0.15, output_per_mtok=0.60) is None


# --------------------------------------------------------------------------------------
# A turn that started answering and wrote no total (#13, manual testing)
# --------------------------------------------------------------------------------------


def a_retrieval(emitter, *, thread_id=THREAD, suffix="aaaa"):
    """One `retrieval` line, as `retrieval/retrieve.py` writes one per search.

    Only the event *name* and the turn id are load-bearing here — this is the sign that the
    answering path started, not a measurement of it — but the fields are the shape the emitter
    really writes, so a rename shows up as a failure here rather than as a silent `0`.
    """
    with turn(f"{thread_id}:{suffix}"):
        log_event(emitter, "retrieval", strategy="hybrid", k=6, hits=6, latency_ms=210)


def test_a_turn_that_started_answering_and_wrote_no_total_is_unfinished(sink, emitter):
    """The shape a turn leaves the log in while it is still being answered.

    `agent_turn` is written last, so between the first search and the answer the log holds
    answering lines and no total. The same shape outlives a turn that raised inside
    `app/Home.py`'s `except`, which is why the count is permanent rather than a live flag: those
    calls were really made and are really missing from the figures.
    """
    a_planner_call(emitter, Reply(input_tokens=25, output_tokens=8))
    a_retrieval(emitter)

    spend = spend_from(sink)

    assert spend.unfinished == 1
    # The planner's own round *did* report, so this is not the unmeasured case — which is
    # exactly why the count has to be said out loud: the panel shows a real total that is
    # missing the answering loop's calls entirely, and nothing else on it would hint at that.
    assert spend.measured and spend.input.total == 25
    assert not spend.partial, "the lines that exist reported both fields; the absent one is not"


def test_a_finished_turn_is_not_unfinished_however_many_searches_it_made(sink, emitter):
    # Two answering lines and one completion line, differenced over turn *ids* — counting lines
    # would call a two-search turn two unfinished ones, and the honest answer is zero.
    a_retrieval(emitter)
    a_retrieval(emitter)
    an_agent_turn(emitter, Reply(input_tokens=120, output_tokens=40))

    spend = spend_from(sink)

    assert spend.unfinished == 0
    assert spend.turns == 1


def test_two_turns_of_which_one_never_finished_are_counted_as_one(sink, emitter):
    a_retrieval(emitter, suffix="one")
    an_agent_turn(emitter, Reply(input_tokens=120, output_tokens=40), suffix="one")
    a_retrieval(emitter, suffix="two")

    spend = spend_from(sink)

    assert spend.unfinished == 1, "the second turn has no total; the first has one"


def test_a_question_the_gate_refused_is_a_finished_turn(sink, emitter):
    """**Why `ANSWERING_EVENTS` is a narrow list and not "any line under a turn id".**

    A refused question writes an `input_gate` line, no answering line and no `agent_turn` — and
    it is a turn that *finished*: it never reached the agent and there is no total to wait for.
    Counting it would put a permanent "still being answered" caption on the panel of anyone
    whose question tripped the denylist once — an absence reported as the wrong absence.
    """
    with turn(f"{THREAD}:aaaa"):
        log_event(emitter, "input_gate", verdict="blocked", layer=2, latency_ms=3)

    spend = spend_from(sink)

    assert spend.unfinished == 0
    assert not spend.measured, "and the gate's classifier is not metered either (ADR-0011)"


def test_another_conversations_unfinished_turn_is_not_this_ones(sink, emitter):
    # The scoping that applies to every other figure here applies to this one: a second tab
    # mid-turn, or an evaluation run, must not put an in-progress caption on this conversation.
    a_retrieval(emitter, thread_id=OTHER_THREAD)

    assert spend_from(sink).unfinished == 0
    assert spend_from(sink, thread_id=OTHER_THREAD).unfinished == 1
