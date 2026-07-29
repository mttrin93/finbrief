"""Token capture: what a reply reported, and what an unreported cost reads back as (T8, #10)."""

import io
import json
import logging

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from finbrief.observability.logging_setup import configure_logging
from finbrief.observability.tokens import usage_fields, usage_total
from finbrief.rag import answer_question
from finbrief.retrieval.query_translation import translate


def a_reply(text: str = "ok", *, input_tokens: int | None = None, output_tokens: int = 0):
    """An `AIMessage` with or without the `usage_metadata` a provider may or may not return."""
    if input_tokens is None:
        return AIMessage(text)
    return AIMessage(
        text,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def test_a_reported_usage_becomes_two_fields():
    assert usage_fields(a_reply(input_tokens=1_204, output_tokens=317)) == {
        "input_tokens": 1_204,
        "output_tokens": 317,
    }


def test_total_tokens_is_not_logged():
    # It is `input + output`, and a third number is a third thing that can disagree.
    assert "total_tokens" not in usage_fields(a_reply(input_tokens=10, output_tokens=5))


def test_an_unreported_usage_adds_no_fields_at_all():
    # Not `{"input_tokens": 0}`. OpenRouter fronts many upstreams and a missing `usage` block
    # means the spend was not reported, which is a different fact from a call that was free.
    assert usage_fields(a_reply()) == {}
    assert usage_fields(None) == {}
    assert usage_fields("not a message") == {}


class Duck:
    """Something with a `usage_metadata` attribute that is not an `AIMessage`.

    Needed because `AIMessage` **cannot** carry a malformed block — pydantic validates
    `usage_metadata` against its TypedDict and rejects a missing `output_tokens` outright, so
    the shapes below are unreachable through the real class. That is worth knowing rather than
    asserting around: the guards in `usage_fields` are there because it reads an attribute off
    an arbitrary reply and a log call must not be able to raise, not because a real provider
    reply is expected to look like this.
    """

    def __init__(self, usage_metadata):
        self.usage_metadata = usage_metadata


def test_a_partial_or_malformed_usage_block_yields_only_what_it_reported():
    assert usage_fields(Duck({"input_tokens": 10})) == {"input_tokens": 10}
    # A string where a count belongs is not a count. Left out rather than coerced: a coerced
    # value is a number in a cost table that no provider ever sent.
    assert usage_fields(Duck({"input_tokens": "many"})) == {}
    assert usage_fields(Duck(None)) == {}
    # `True` is an `int` in Python, and it is not a token count.
    assert usage_fields(Duck({"input_tokens": True, "output_tokens": 5})) == {
        "output_tokens": 5
    }


def test_a_turn_sums_its_calls_and_says_how_many_reported():
    replies = [
        a_reply(input_tokens=900, output_tokens=40),
        a_reply(),  # a call whose provider reported nothing
        a_reply(input_tokens=1_100, output_tokens=210),
    ]

    assert usage_total(replies) == {
        # The denominators, and the reason they are on the line: this total covers 2 of 3 calls,
        # and a partial total presented as a turn's spend understates it silently.
        "calls": 3,
        "input_tokens": 2_000,
        "input_tokens_calls": 2,
        "output_tokens": 250,
        "output_tokens_calls": 2,
    }


def test_a_turn_where_nothing_reported_is_absent_rather_than_zeroed():
    assert usage_total([a_reply(), a_reply()]) == {}
    assert usage_total([]) == {}


def test_a_half_reported_reply_leaves_the_other_half_off_the_total():
    """The fabricated zero this function used to write (issue #10 review).

    `usage_fields` has always handled a provider that reports one half of the pair —
    `test_a_partial_or_malformed_usage_block_yields_only_what_it_reported` is that test. The
    *aggregate* did not: it summed with `.get(name, 0)` and counted the reply as metered, so
    `output_tokens: 0` went onto the `agent_turn` line, indistinguishable from a real zero,
    against this module's own "absent, never zero".
    """
    half = Duck({"input_tokens": 120, "output_tokens": None})

    # Asserted first, because it is the premise: the single-reply path already omits the half
    # nobody reported, and the aggregate has to agree with it rather than fill the gap in.
    assert usage_fields(half) == {"input_tokens": 120}
    assert usage_total([half]) == {"calls": 1, "input_tokens": 120, "input_tokens_calls": 1}
    # The whole equality, not `"output_tokens" not in ...`: a zero *and* a wrong denominator
    # were both wrong before, and only an equality pins both.
    assert usage_total([half, a_reply(input_tokens=80, output_tokens=40)]) == {
        "calls": 2,
        "input_tokens": 200,
        "input_tokens_calls": 2,
        # 1, not 2. One reply reported this field, so one is what it is summed over — the
        # per-reply denominator claimed two and made the half-total read as complete.
        "output_tokens": 40,
        "output_tokens_calls": 1,
    }


def test_a_turn_where_only_the_unreported_half_is_missing_omits_only_that_half():
    # The mirror case, so the fix is not accidentally one-sided.
    assert usage_total([Duck({"output_tokens": 55})]) == {
        "calls": 1,
        "output_tokens": 55,
        "output_tokens_calls": 1,
    }


# --- Where the counts land: the three metered call sites (T8, #10) ---------------------
#
# The classifier's call is deliberately not one of them. `classify()` returns a bare `Verdict`,
# so metering it means either changing that return type or adding a per-turn event duplicating
# `input_gate` — and it is the cheapest model call in the system, one word out. **The gate's
# token spend is therefore unmeasured, and this comment is the record of that** rather than a
# gap a reader has to infer: an unscored criterion reads as a passed one.


class MeteredFakeChatModel(GenericFakeChatModel):
    """A fake whose replies carry `usage_metadata`, the way a real provider's do.

    Needed because `GenericFakeChatModel` reports none — which is exactly the *absent* case,
    and is why every existing test in this repo exercises the no-usage half for free.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        result.generations[0].message.usage_metadata = {
            "input_tokens": 1_204,
            "output_tokens": 317,
            "total_tokens": 1_521,
        }
        return result


@pytest.fixture
def emitted():
    """Every event a test emits, as parsed JSON lines."""
    stream = io.StringIO()
    configure_logging(logging.DEBUG, stream=stream)
    yield lambda: [json.loads(line) for line in stream.getvalue().splitlines()]


def only(events, name):
    (event,) = [e for e in events if e["event"] == name]
    return event["fields"]


def test_the_rag_answer_line_carries_the_generations_tokens(filings_store, emitted):
    answer_question(
        "What are the risks to Apple's supply chain?",
        k=2,
        store=filings_store,
        model=MeteredFakeChatModel(messages=iter(["Apple flags concentration [1]."])),
    )

    fields = only(emitted(), "rag_answer")
    assert fields["input_tokens"] == 1_204
    assert fields["output_tokens"] == 317


def test_a_provider_that_reports_no_usage_leaves_the_keys_off_the_line(filings_store, emitted):
    answer_question(
        "What are the risks to Apple's supply chain?",
        k=2,
        store=filings_store,
        model=GenericFakeChatModel(messages=iter(["Apple flags concentration [1]."])),
    )

    fields = only(emitted(), "rag_answer")
    assert "input_tokens" not in fields and "output_tokens" not in fields


def test_an_ungrounded_turn_reports_no_tokens_because_no_model_ran(
    empty_filings_store, emitted
):
    # The fallback path calls nothing, so its spend is *nothing to measure* rather than zero.
    answer_question("What are Nestle's dividends?", k=2, store=empty_filings_store)

    fields = only(emitted(), "rag_answer")
    assert fields["grounded"] is False
    assert "input_tokens" not in fields


def test_the_planners_tokens_land_on_the_translation_line_not_the_answer_line(emitted):
    # ADR-0005 judges translation on the cost *it* adds, so the planner's spend has to be
    # separable from generation's. Two lines, joinable on `turn_id`.
    translate(
        "Is Tesla in trouble?",
        model=MeteredFakeChatModel(messages=iter(["Tesla liquidity\nTesla risk factors"])),
        max_sub_queries=3,
    )

    fields = only(emitted(), "query_translation")
    assert fields["input_tokens"] == 1_204
    assert fields["output_tokens"] == 317


def test_a_planner_that_never_ran_reports_no_tokens(emitted):
    # `max_sub_queries=0` removes the `model.invoke` call rather than truncating its output
    # (ADR-0004 §6), so there is no spend — not a spend of zero.
    translate(
        "Is Tesla in trouble?", model=MeteredFakeChatModel(messages=iter([])), max_sub_queries=0
    )

    fields = only(emitted(), "query_translation")
    assert fields["sub_queries"] == 0
    assert "input_tokens" not in fields
