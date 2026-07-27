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

from finbrief.config import TICKER_BY_COMPANY_NAME
from finbrief.prompts import query_translation_prompt
from finbrief.retrieval.query_translation import normalised, sub_queries, translate

QUESTION = "Is Tesla in trouble?"

#: A question naming no Universe company, so the planner's own behaviour can be asserted with
#: exact tuples. `QUESTION` names Tesla and therefore also gains a deterministic ticker-form
#: variant (see the normalisation section below), which would make every count here ambiguous
#: about which half produced it.
NO_COMPANY = "Is the outlook deteriorating?"


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
    assert variants == (
        QUESTION,
        "Is TSLA in trouble?",  # the deterministic ticker form; see the normalisation section
        "Tesla risk factors",
        "Tesla debt levels",
    )


def test_translation_never_returns_fewer_variants_than_it_was_given():
    # A model that answers with nothing usable is a translation that added nothing — which is a
    # legal outcome, not an error, precisely because the original is retained. The retrieval
    # then runs exactly as it would have with translation off.
    assert translate(NO_COMPANY, model=a_model(""), max_sub_queries=3) == (NO_COMPANY,)


def test_the_sub_query_cap_is_enforced_not_merely_requested():
    # The cap is a latency budget, not a preference: ADR-0005 judges dominance within <=1.5s p50
    # added by translation, and every extra variant is two more candidate lists to retrieve. A
    # model that returns six sub-queries must not be able to spend that budget for us.
    variants = translate(
        NO_COMPANY,
        model=a_model("one\ntwo\nthree\nfour\nfive\nsix"),
        max_sub_queries=3,
    )

    assert variants == (NO_COMPANY, "one", "two", "three")


def test_a_cap_of_zero_does_not_call_the_model_at_all():
    # `FINBRIEF_MAX_SUB_QUERIES=0` is a configuration `config.py` permits. Paying for a
    # generation whose every line is then discarded is the one behaviour it cannot mean.
    model = a_model("one\ntwo")

    assert translate(NO_COMPANY, model=model, max_sub_queries=0) == (NO_COMPANY,)
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
    assert sub_queries(reply, known=[QUESTION], limit=3) == (
        "Tesla risk factors",
        "Tesla debt levels",
    )


def test_a_preamble_line_is_not_mistaken_for_a_sub_query():
    # "Here are the sub-queries:" would otherwise be retrieved, embedded and fused as if the
    # analyst had asked it. A trailing colon is the narrow signal that catches it — narrow on
    # purpose, because a genuine query does not end in one.
    reply = "Here are three sub-queries:\n- Tesla liquidity\n- Tesla deliveries"

    assert sub_queries(reply, known=[QUESTION], limit=3) == (
        "Tesla liquidity",
        "Tesla deliveries",
    )


def test_a_sub_query_that_merely_repeats_the_question_is_dropped():
    # It would cost a second embedding and two more candidate lists to retrieve exactly what the
    # retained original already retrieves — and RRF would then count the same list twice,
    # inflating that chunk's score for no evidence. Compared case-insensitively on stripped
    # text, because neither casing nor whitespace is a translation.
    reply = f"{QUESTION.upper()}\nTesla liquidity"

    assert sub_queries(reply, known=[QUESTION], limit=3) == ("Tesla liquidity",)


def test_a_repeated_sub_query_is_kept_once():
    reply = "Tesla liquidity\ntesla liquidity\nTesla deliveries"

    assert sub_queries(reply, known=[QUESTION], limit=3) == (
        "Tesla liquidity",
        "Tesla deliveries",
    )


def test_a_refusal_to_decompose_yields_no_sub_queries():
    # The prompt tells the model to return nothing when the question is already atomic, and
    # "nothing" arrives as a blank reply or a bare marker. Either way the retrieval runs on the
    # original alone, which is the correct answer for an already-specific question.
    assert sub_queries("", known=[QUESTION], limit=3) == ()
    assert sub_queries("-\n*\n", known=[QUESTION], limit=3) == ()


# --------------------------------------------------------------------------------------
# Entity normalisation: the company name an analyst types -> the ticker the index carries
# --------------------------------------------------------------------------------------


def test_a_universe_company_name_gains_a_ticker_form_variant():
    # The T6 finding this exists for (ADR-0004 amendment): a chunk's provenance header carries
    # `TSLA`, not `Tesla`, so `tsla` is a token on all 280 of Tesla's chunks while `tesla` is a
    # token on 34 of them — body mentions only. A lexical retriever handed the name therefore
    # sees almost none of the filer, which is why hybrid alone made issue #6's case *worse*.
    assert normalised("How much debt does Tesla carry?") == "How much debt does TSLA carry?"


def test_normalisation_is_deterministic_and_needs_no_model():
    # Deliberately not the planner's job. It is a lookup in `config.TICKER_BY_COMPANY_NAME`, so
    # the `+translation` arm of the A/B gains a variant without gaining model variance — and it
    # still works with the sub-query cap set to zero.
    model = a_model("unused")

    variants = translate("Tesla debt", model=model, max_sub_queries=0)

    assert variants == ("Tesla debt", "TSLA debt")
    assert model.prompts == [], "no generation was requested"


def test_the_original_is_still_retained_ahead_of_the_normalised_form():
    # ADR-0004's invariant is not weakened by this: normalisation *adds* a surface form and
    # never replaces one. `ford` is in fact a sharper lexical token than `f`, so the original is
    # sometimes the better of the two — and it is always there.
    variants = translate(
        "Is Tesla in trouble?", model=a_model("Tesla liquidity"), max_sub_queries=3
    )

    assert variants == ("Is Tesla in trouble?", "Is TSLA in trouble?", "Tesla liquidity")


def test_every_company_named_is_normalised_in_one_variant_not_one_variant_each():
    # A comparison question names several filers. One substitution pass over the whole query
    # bounds normalisation at **one** extra variant however many companies appear — keeping the
    # added retrieval cost flat rather than linear in the question's breadth.
    assert normalised("Compare Tesla and General Motors debt") == "Compare TSLA and GM debt"


def test_a_single_character_ticker_is_never_substituted_in():
    # Ford. Measured on the ingested collection: `f` is a token in 534 of 5,842 chunks and in
    # chunks belonging to five filers, because a single letter is also a footnote marker and a
    # table label; `ford` is in 232 chunks belonging to one. Substituting the ticker would make
    # the query *less* discriminating, which is the opposite of the point.
    assert normalised("Ford debt") is None
    assert "f" not in TICKER_BY_COMPANY_NAME.values()


def test_a_query_that_already_names_the_ticker_gains_nothing():
    # There is nothing to normalise, and a duplicate variant would cost an embedding and let RRF
    # count one candidate list twice.
    assert normalised("TSLA debt") is None


def test_normalisation_fires_only_for_universe_names_not_for_any_capitalised_word():
    # The guard against this becoming an entity recogniser. Nothing is inferred: a form is
    # substituted if and only if it is in the Universe's own name map.
    assert normalised("What are Nestle's risk factors?") is None
    assert normalised("Compare Rivian and Lucid") is None
    assert normalised("What does the Company say about Apple Pay?") == (
        "What does the Company say about AAPL Pay?"
    ), "a name inside a product name is still that name — the original is retained regardless"


def test_a_name_is_matched_on_word_boundaries_not_as_a_substring():
    # `Meta` must not fire inside `Metabolism`, and `Lilly` must not fire inside `Lillybrook`.
    # A substring match would corrupt a word into a ticker and embed the wreckage.
    assert normalised("metabolism and metadata") is None
    assert normalised("Meta advertising revenue") == "META advertising revenue"


def test_the_longest_name_form_wins_so_a_legal_name_is_not_half_substituted():
    # `Goldman`, `Goldman Sachs` and `The Goldman Sachs Group, Inc.` all map to `GS`. Matching
    # the shortest first would leave `GS Sachs Group, Inc.` — a query naming a company that does
    # not exist, embedded and BM25-indexed as such.
    assert normalised("The Goldman Sachs Group, Inc. risk factors") == "GS risk factors"
    assert normalised("Goldman Sachs risk factors") == "GS risk factors"


def test_case_is_ignored_when_matching_but_the_ticker_is_written_as_the_index_has_it():
    assert normalised("tesla debt") == "TSLA debt"
    assert normalised("TESLA debt") == "TSLA debt"


def test_a_sub_query_that_only_repeats_the_normalised_form_is_dropped():
    # The dedup has to know about every variant already in play, not just the original — a
    # planner that echoes the ticker form back would otherwise be retrieved twice.
    variants = translate(
        "Tesla debt", model=a_model("TSLA debt\nTesla liquidity"), max_sub_queries=3
    )

    assert variants == ("Tesla debt", "TSLA debt", "Tesla liquidity")


def test_normalisation_is_logged_as_its_own_count_never_as_text(caplog):
    with caplog.at_level("INFO", logger="finbrief.retrieval.query_translation"):
        translate("Tesla debt", model=a_model("Tesla liquidity"), max_sub_queries=3)

    (record,) = [r for r in caplog.records if getattr(r, "event", None) == "query_translation"]
    assert record.fields["normalised"] is True
    assert record.fields["sub_queries"] == 1
    assert record.fields["variants"] == 3
    assert "Tesla" not in str(record.fields) and "TSLA" not in str(record.fields)
