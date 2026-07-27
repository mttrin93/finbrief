"""Query translation: the original question, plus up to three sub-queries (ADR-0004).

Below seam 1, like `test_hybrid.py`. What is asserted here is the one invariant translation
lives or dies by — **it only ever adds** — and the parsing that stands between a chat model's
free text and a list of queries that will be embedded and paid for.

The model is scripted throughout. What a real model decomposes a question *into* is not
something a test can assert without becoming a test of that model; what it can assert is that
whatever comes back, the analyst's own words survive it, the cap holds, and nothing malformed
reaches the retrievers.
"""

from __future__ import annotations

import pytest
from fakes import ScriptedChatModel
from langchain_core.messages import AIMessage, SystemMessage

from finbrief.prompts import query_translation_prompt
from finbrief.retrieval.query_translation import sub_queries, translate

QUESTION = "Is Tesla in trouble?"


def a_model(*replies: str) -> ScriptedChatModel:
    return ScriptedChatModel(messages=iter([AIMessage(reply) for reply in replies]))


def test_the_analysts_own_question_is_always_the_first_variant():
    # ADR-0004's central invariant, and the reason it is written as "translation only ever
    # *adds*": a rewrite that replaced the question could strip the literal identifiers — a
    # ticker, `Item 1A`, a ratio name — that BM25 is in the pipeline to match, degrading the
    # exact-identifier bucket hybrid search exists to serve.
    variants = translate(
        QUESTION, model=a_model("Tesla risk factors\nTesla debt levels"), max_sub_queries=3
    )

    assert variants[0] == QUESTION
    assert variants == (QUESTION, "Tesla risk factors", "Tesla debt levels")


def test_translation_never_returns_fewer_variants_than_it_was_given():
    # A model that answers with nothing usable is a translation that added nothing — which is a
    # legal outcome, not an error, precisely because the original is retained. The retrieval
    # then runs exactly as it would have with translation off.
    assert translate(QUESTION, model=a_model(""), max_sub_queries=3) == (QUESTION,)


def test_the_sub_query_cap_is_enforced_not_merely_requested():
    # The cap is a latency budget, not a preference: ADR-0005 judges dominance within <=1.5s p50
    # added by translation, and every extra variant is two more candidate lists to retrieve. A
    # model that returns six sub-queries must not be able to spend that budget for us.
    variants = translate(
        QUESTION,
        model=a_model("one\ntwo\nthree\nfour\nfive\nsix"),
        max_sub_queries=3,
    )

    assert variants == (QUESTION, "one", "two", "three")


def test_a_cap_of_zero_does_not_call_the_model_at_all():
    # `FINBRIEF_MAX_SUB_QUERIES=0` is a configuration `config.py` permits. Paying for a
    # generation whose every line is then discarded is the one behaviour it cannot mean.
    model = a_model("one\ntwo")

    assert translate(QUESTION, model=model, max_sub_queries=0) == (QUESTION,)
    assert model.prompts == [], "no generation was requested"


def test_the_model_is_given_the_translation_prompt_and_the_question():
    # `prompts.py` owns everything the model reads (CLAUDE.md). If this prompt were assembled
    # here, the one module that guarantees the scope sentences agree would not own it.
    model = a_model("Tesla liquidity")

    translate(QUESTION, model=model, max_sub_queries=3)

    (prompt,) = model.prompts
    assert isinstance(prompt[0], SystemMessage)
    assert prompt[0].text == query_translation_prompt(3)
    assert prompt[-1].text == QUESTION


def test_the_prompt_asks_for_the_cap_that_will_actually_be_enforced():
    # A prompt hardcoding "three" against `FINBRIEF_MAX_SUB_QUERIES=1` asks for two lines we pay
    # to generate and then discard. The number the model reads is the number the code keeps.
    model = a_model("Tesla liquidity")

    translate(QUESTION, model=model, max_sub_queries=1)

    assert "At most 1," in model.prompts[0][0].text


def test_a_failed_translation_is_raised_rather_than_quietly_skipped():
    # Deliberate, and the opposite of what a UX instinct suggests. `±translation` is one axis of
    # the measured A/B (ADR-0002), so a run that silently fell back to no-translation would
    # report a translation-enabled number for a retrieval where translation never happened —
    # the same failure ADR-0005 refuses when it makes `hybrid` raise rather than serve vector
    # results under a hybrid label. Phase 5 owns turning this into a graceful UI failure.
    class Broken(ScriptedChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise RuntimeError("upstream refused")

    with pytest.raises(RuntimeError, match="upstream refused"):
        translate(QUESTION, model=Broken(messages=iter([])), max_sub_queries=3)


def test_translation_is_logged_by_count_and_never_by_text(caplog):
    # A sub-query is derived from the analyst's question, so it is user content — and these
    # lines are kept. Same rule the `agent_query` line follows (ADR-0003 §2).
    with caplog.at_level("INFO", logger="finbrief.retrieval.query_translation"):
        translate(
            QUESTION, model=a_model("Tesla liquidity\nTesla deliveries"), max_sub_queries=3
        )

    (record,) = [r for r in caplog.records if getattr(r, "event", None) == "query_translation"]
    assert record.fields["sub_queries"] == 2
    assert record.fields["max_sub_queries"] == 3
    assert record.fields["latency_ms"] >= 0
    assert "Tesla" not in str(record.fields)


# --------------------------------------------------------------------------------------
# Parsing a chat model's free text into queries that will be embedded
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "- Tesla risk factors\n- Tesla debt levels",
        "1. Tesla risk factors\n2. Tesla debt levels",
        "1) Tesla risk factors\n2) Tesla debt levels",
        "* Tesla risk factors\n\n* Tesla debt levels",
        "  Tesla risk factors  \n\tTesla debt levels\n",
    ],
)
def test_list_furniture_is_stripped_however_the_model_chose_to_format_it(reply):
    # Every one of these is a plausible reply to "one per line, no numbering", and a leading
    # `1. ` or `- ` is a token that gets embedded and BM25-indexed. The prompt asks; this is
    # what makes it not matter.
    assert sub_queries(reply, question=QUESTION, limit=3) == (
        "Tesla risk factors",
        "Tesla debt levels",
    )


def test_a_preamble_line_is_not_mistaken_for_a_sub_query():
    # "Here are the sub-queries:" would otherwise be retrieved, embedded and fused as if the
    # analyst had asked it. A trailing colon is the narrow signal that catches it — narrow on
    # purpose, because a genuine query does not end in one.
    reply = "Here are three sub-queries:\n- Tesla liquidity\n- Tesla deliveries"

    assert sub_queries(reply, question=QUESTION, limit=3) == (
        "Tesla liquidity",
        "Tesla deliveries",
    )


def test_a_sub_query_that_merely_repeats_the_question_is_dropped():
    # It would cost a second embedding and two more candidate lists to retrieve exactly what the
    # retained original already retrieves — and RRF would then count the same list twice,
    # inflating that chunk's score for no evidence. Compared case-insensitively on stripped
    # text, because neither casing nor whitespace is a translation.
    reply = f"{QUESTION.upper()}\nTesla liquidity"

    assert sub_queries(reply, question=QUESTION, limit=3) == ("Tesla liquidity",)


def test_a_repeated_sub_query_is_kept_once():
    reply = "Tesla liquidity\ntesla liquidity\nTesla deliveries"

    assert sub_queries(reply, question=QUESTION, limit=3) == (
        "Tesla liquidity",
        "Tesla deliveries",
    )


def test_a_refusal_to_decompose_yields_no_sub_queries():
    # The prompt tells the model to return nothing when the question is already atomic, and
    # "nothing" arrives as a blank reply or a bare marker. Either way the retrieval runs on the
    # original alone, which is the correct answer for an already-specific question.
    assert sub_queries("", question=QUESTION, limit=3) == ()
    assert sub_queries("-\n*\n", question=QUESTION, limit=3) == ()
