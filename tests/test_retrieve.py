"""`retrieve(question, strategy, translate, k)` — the deterministic retrieval seam (ADR-0003).

Testing seam 1 (spec §Testing Decisions): all retrieval-quality behaviour is asserted
here, at the same entry point the evaluation harness and `search_filings` drive.

Runs against a real on-disk Chroma in `tmp_path` (the `filings_store` fixture), filled with
chunks of the *recorded* AAPL filing and embedded by a keyword-counting fake — real store
and real distance ranking, no network and no paid embeddings. The ingested `data/chroma`
store is deliberately not a fixture: it needs the live embedding model to be queryable at
all.

**What the fake embedding can and cannot show.** It is lexical, so it agrees with BM25 far more
than a real embedding does — which means the *interesting* half of ADR-0004's prediction (a
chunk BM25 recovers that vector search ranked outside `k`) is not something this fixture can
stage. What is asserted here is the composition: which variants ran, through which retrievers,
that the original is retained, that provenance describes what happened, and that `vector`
without translation still returns exactly what T3 returned. The quality claim is issue #6's
live before/after and ADR-0002's golden set (#4).
"""

from __future__ import annotations

import pytest
from fakes import ScriptedChatModel
from langchain_core.messages import AIMessage

from finbrief.config import RRF_K, RetrievalStrategy, Settings
from finbrief.ingestion.model import Section
from finbrief.retrieval.hybrid import Retriever
from finbrief.retrieval.retrieve import Retrieval, retrieve

SETTINGS = Settings.from_env({"OPENROUTER_API_KEY": "sk-test", "FINBRIEF_RETRIEVAL_K": "3"})

QUESTION = "What are the risks to Apple's supply chain?"

VECTOR = RetrievalStrategy.VECTOR
HYBRID = RetrievalStrategy.HYBRID


def a_translator(*sub_queries: str) -> ScriptedChatModel:
    """A scripted translation model. What it decomposes into is the script's business."""
    return ScriptedChatModel(messages=iter([AIMessage("\n".join(sub_queries))]))


def test_retrieve_returns_k_contexts_carrying_a_chunks_provenance(filings_store):
    result = retrieve(QUESTION, strategy=VECTOR, k=3, store=filings_store)

    assert isinstance(result, Retrieval)
    assert len(result.contexts) == 3
    assert [context.rank for context in result.contexts] == [1, 2, 3]
    for context in result.contexts:
        assert context.ticker == "AAPL"
        assert context.fiscal_year == 2025
        assert context.filing_type == "10-K"
        assert context.section in set(Section)
        assert context.accession == "0000320193-25-000079"
        assert context.body


def test_contexts_come_back_nearest_first(filings_store):
    result = retrieve(QUESTION, strategy=VECTOR, k=5, store=filings_store)

    distances = [context.distance for context in result.contexts]
    assert distances == sorted(distances), "rank 1 is the nearest chunk, not an arbitrary one"


def test_a_risk_question_retrieves_risk_factors_and_a_margin_question_the_mda(filings_store):
    # The one assertion that would notice a retrieval that returns *something* for every
    # question: two questions whose answers live in different Sections of the same filing.
    risks = retrieve(QUESTION, strategy=VECTOR, k=3, store=filings_store)
    margins = retrieve(
        "How did gross margin change?", strategy=VECTOR, k=3, store=filings_store
    )

    assert risks.contexts[0].section is Section.RISK_FACTORS
    assert margins.contexts[0].section is Section.MDA


def test_the_same_question_retrieves_the_same_contexts_in_the_same_order(filings_store):
    # ADR-0003 stakes the RAGAs and A/B numbers on this: a chain that reorders between
    # runs makes a strategy comparison unrepeatable.
    first = retrieve(QUESTION, strategy=VECTOR, k=4, store=filings_store)
    second = retrieve(QUESTION, strategy=VECTOR, k=4, store=filings_store)

    assert [c.chunk_id for c in first.contexts] == [c.chunk_id for c in second.contexts]
    assert [c.distance for c in first.contexts] == [c.distance for c in second.contexts]


def test_hybrid_is_deterministic_too(filings_store):
    # More moving parts, same requirement — and one more way to fail it: fusion accumulates into
    # a dict, so a tie broken by insertion order would be stable within a process and not
    # across a refactor. `hybrid.fuse` breaks ties on chunk id for exactly this.
    first = retrieve(QUESTION, strategy=HYBRID, k=4, store=filings_store)
    second = retrieve(QUESTION, strategy=HYBRID, k=4, store=filings_store)

    assert [c.chunk_id for c in first.contexts] == [c.chunk_id for c in second.contexts]
    assert [c.fused_score for c in first.contexts] == [c.fused_score for c in second.contexts]


def test_a_context_body_omits_the_provenance_header_the_index_carries(filings_store):
    # `Chunk.text` prepends `AAPL | FY2025 10-K | Item 1A. Risk Factors` for BM25
    # (ADR-0004). Handing it to the LLM again — inside a block already labelled with the
    # same metadata — only invites it to be cited as the filer's own words.
    result = retrieve(QUESTION, strategy=VECTOR, k=1, store=filings_store)

    assert not result.contexts[0].body.startswith("AAPL | FY2025 10-K |")


def test_a_bm25_hit_also_arrives_without_the_header_it_was_matched_on(filings_store):
    # The header is what BM25 matched, which makes it the one retriever with a reason to keep it
    # — and the reason this is asserted separately from the vector path. Stripping it in one
    # branch and not the other would put machine furniture in a citation only under `hybrid`.
    result = retrieve("AAPL Item 1A", strategy=HYBRID, k=3, store=filings_store)

    assert result.contexts
    for context in result.contexts:
        assert not context.body.startswith("AAPL | FY2025 10-K |")


def test_a_context_states_the_citation_a_marker_resolves_to(filings_store):
    result = retrieve(QUESTION, strategy=VECTOR, k=1, store=filings_store)

    assert result.contexts[0].citation == "AAPL 10-K FY2025, Item 1A"


def test_retrieving_from_an_empty_collection_returns_nothing(empty_filings_store):
    # The tiered-error-handling contract (spec §Tools): retrieval level, empty result ->
    # the caller's fallback message. Not an exception, and never an unguarded `[0]`.
    result = retrieve("anything", strategy=VECTOR, k=5, store=empty_filings_store)

    assert result.contexts == ()


def test_hybrid_over_an_empty_collection_returns_nothing_rather_than_raising(
    empty_filings_store,
):
    # BM25 is the half that could raise instead: `rank_bm25` divides by the corpus's average
    # document length, so an un-ingested collection is a `ZeroDivisionError` on the first query
    # unless the index refuses to be built at all.
    result = retrieve("anything", strategy=HYBRID, k=5, store=empty_filings_store)

    assert result.contexts == ()


# --------------------------------------------------------------------------------------
# The four configurations (ADR-0004, ADR-0002's A/B axes)
# --------------------------------------------------------------------------------------


def test_vector_without_translation_still_returns_exactly_what_it_did_before_fusion(
    filings_store,
):
    # The baseline the whole A/B is measured against. `vector` + no translation is now one
    # candidate list through RRF, and RRF over a single list is strictly decreasing in rank — so
    # it must be a no-op on order and on distance. If Phase 4 moved this, every comparison it
    # produces is against a baseline nobody measured.
    result = retrieve(QUESTION, strategy=VECTOR, k=5, store=filings_store)

    from finbrief.retrieval.vectorstore import nearest_chunks

    unfused = nearest_chunks(filings_store, QUESTION, 5)
    assert [c.chunk_id for c in result.contexts] == [d.id for d, _ in unfused]
    assert [c.distance for c in result.contexts] == [distance for _, distance in unfused]


def test_the_original_question_is_always_the_first_variant(filings_store):
    # ADR-0004's invariant, at the seam that has to honour it. Asserted for both switch
    # positions, because "translation only ever adds" is a claim about the translated case.
    plain = retrieve(QUESTION, strategy=HYBRID, k=3, store=filings_store)
    translated = retrieve(
        QUESTION,
        strategy=HYBRID,
        translate=True,
        k=3,
        store=filings_store,
        settings=SETTINGS,
        model=a_translator("Apple supplier concentration", "Apple manufacturing risk"),
    )

    assert plain.variants == (QUESTION,)
    assert translated.variants[0] == QUESTION
    assert translated.question == QUESTION
    assert translated.sub_queries == (
        "Apple supplier concentration",
        "Apple manufacturing risk",
    )


def test_translation_reports_that_it_ran_even_when_it_added_nothing(filings_store):
    # `translated` is a fact about the run, not about `variants`' length. "Translation was off"
    # and "translation ran and the model offered nothing usable" are different things, and only
    # the second one is worth a second look in the RAG-viz panel.
    result = retrieve(
        QUESTION,
        strategy=VECTOR,
        translate=True,
        k=3,
        store=filings_store,
        settings=SETTINGS,
        model=a_translator(""),
    )

    assert result.variants == (QUESTION,)
    assert result.translated is True
    assert retrieve(QUESTION, strategy=VECTOR, k=3, store=filings_store).translated is False


def test_every_variant_runs_through_every_retriever_the_strategy_names(filings_store):
    # The symmetric composition, read back off the provenance: two variants × two retrievers is
    # four candidate lists, and ADR-0004 wants all four. An asymmetric shortcut — sub-queries
    # through vector only, say — would leave a variant × retriever pair unrepresented here.
    result = retrieve(
        "Apple supply chain and competition",
        strategy=HYBRID,
        translate=True,
        k=5,
        store=filings_store,
        settings=SETTINGS,
        model=a_translator("Apple competition risk"),
    )

    surfaced = {
        (row.variant, row.retriever)
        for context in result.contexts
        for row in context.provenance
    }
    assert surfaced == {
        ("Apple supply chain and competition", Retriever.VECTOR),
        ("Apple supply chain and competition", Retriever.BM25),
        ("Apple competition risk", Retriever.VECTOR),
        ("Apple competition risk", Retriever.BM25),
    }


def test_vector_only_never_consults_the_lexical_retriever(filings_store):
    # The other direction: `vector` must be vector. A provenance row naming BM25 under this
    # strategy would mean the A/B's two arms differ by less than their labels claim.
    result = retrieve(QUESTION, strategy=VECTOR, k=5, store=filings_store)

    assert all(
        row.retriever is Retriever.VECTOR
        for context in result.contexts
        for row in context.provenance
    )


def test_hybrid_deduplicates_by_chunk_id_across_variants_and_retrievers(filings_store):
    # Four candidate lists over one small filing overlap heavily. Every overlap must collapse to
    # one context — a duplicate would be cited twice under two numbers, and the sources panel
    # would show the same passage as two independent pieces of evidence.
    result = retrieve(
        QUESTION,
        strategy=HYBRID,
        translate=True,
        k=5,
        store=filings_store,
        settings=SETTINGS,
        model=a_translator("Apple supplier concentration"),
    )

    chunk_ids = [context.chunk_id for context in result.contexts]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_a_chunk_ranks_by_its_fused_score_and_the_score_is_its_provenance_summed(
    filings_store,
):
    result = retrieve(QUESTION, strategy=HYBRID, k=5, store=filings_store)

    scores = [context.fused_score for context in result.contexts]
    assert scores == sorted(scores, reverse=True), "higher RRF score is a better chunk"
    for context in result.contexts:
        assert context.fused_score == sum(row.contribution for row in context.provenance)
        assert all(row.contribution == 1 / (RRF_K + row.rank) for row in context.provenance)


def test_a_chunk_only_bm25_found_reports_no_distance(filings_store):
    # A query with no vector signal at all — the fake embedding's vocabulary contains none of
    # these terms, so every chunk is equidistant — but plenty of lexical signal. Whatever BM25
    # pulls up that vector search did not return has no distance to report, and must say so
    # rather than print a number no measurement produced.
    result = retrieve("0000320193-25-000079", strategy=HYBRID, k=3, store=filings_store)

    lexical_only = [
        context for context in result.contexts if Retriever.VECTOR not in context.retrievers
    ]
    assert lexical_only, "the accession is a lexical-only signal in this fixture"
    assert all(context.distance is None for context in lexical_only)


# --------------------------------------------------------------------------------------
# Configuration, refusal, logging
# --------------------------------------------------------------------------------------


def test_a_strategy_named_as_a_string_lands_on_the_same_branch_as_the_enum(filings_store):
    # `retrieve` normalises rather than trusts: `"hybrid" is RetrievalStrategy.HYBRID` is
    # `False`, so an unnormalised string would take the `else` branch of every dispatch below it
    # and serve vector results under a hybrid label — the one outcome ADR-0005 rules out.
    as_string = retrieve("supply chain risk", strategy="hybrid", k=3, store=filings_store)
    as_enum = retrieve("supply chain risk", strategy=HYBRID, k=3, store=filings_store)

    assert [c.chunk_id for c in as_string.contexts] == [c.chunk_id for c in as_enum.contexts]
    assert any(Retriever.BM25 in c.retrievers for c in as_string.contexts)


def test_an_unknown_strategy_is_refused_by_name_rather_than_taken_as_vector(filings_store):
    with pytest.raises(ValueError, match="bm25"):
        retrieve("anything", strategy="bm25", k=1, store=filings_store)


def test_k_and_the_store_fall_back_to_the_application_settings(monkeypatch, filings_store):
    # The production path: the app passes neither, so `settings.retrieval_k` and the
    # configured `filings` collection decide. Every other test here injects both.
    #
    # The fake hands back a *populated* store, which is what makes `SETTINGS`'
    # `FINBRIEF_RETRIEVAL_K=3` observable. Pointed at an empty collection this asserted
    # `== ()`, true for every possible k, so half of what the test is named for went
    # unchecked (issue #5 review).
    import finbrief.retrieval.retrieve as retrieve_module

    opened = []

    def fake_default_filings_store(settings):
        opened.append(settings)
        return filings_store

    monkeypatch.setattr(retrieve_module, "default_filings_store", fake_default_filings_store)
    monkeypatch.setattr(retrieve_module, "get_settings", lambda: SETTINGS)

    result = retrieve(QUESTION, strategy=VECTOR)

    assert len(result.contexts) == 3, "k came from FINBRIEF_RETRIEVAL_K, not the default"
    assert opened == [SETTINGS], "the one place the filings collection is opened"


def test_translating_reads_the_sub_query_cap_from_configuration(monkeypatch, filings_store):
    # The cap is *enforced* rather than defaulted (`config.py`), because ADR-0005 judges
    # dominance within a latency budget that assumes it. A retrieval that translated without
    # consulting configuration could spend that budget on any number of variants.
    import finbrief.retrieval.retrieve as retrieve_module

    capped = Settings.from_env(
        {
            "OPENROUTER_API_KEY": "sk-test",
            "FINBRIEF_RETRIEVAL_K": "3",
            "FINBRIEF_MAX_SUB_QUERIES": "1",
        }
    )
    monkeypatch.setattr(retrieve_module, "get_settings", lambda: capped)

    result = retrieve(
        QUESTION,
        strategy=VECTOR,
        translate=True,
        k=3,
        store=filings_store,
        model=a_translator("one", "two", "three"),
    )

    assert result.sub_queries == ("one",)


def test_retrieval_logs_what_it_returned_without_logging_the_filing_text(filings_store, caplog):
    with caplog.at_level("INFO", logger="finbrief.retrieval.retrieve"):
        result = retrieve(QUESTION, strategy=VECTOR, k=2, store=filings_store)

    (record,) = [r for r in caplog.records if getattr(r, "event", None) == "retrieval"]
    assert record.fields["strategy"] == "vector"
    assert record.fields["translation"] is False
    assert record.fields["k"] == 2
    assert record.fields["hits"] == 2
    assert record.fields["chunk_ids"] == [c.chunk_id for c in result.contexts]
    assert record.fields["latency_ms"] >= 0
    assert result.contexts[0].body[:40] not in str(record.fields)


def test_the_log_line_carries_provenance_by_variant_index_never_by_query_text(
    filings_store, caplog
):
    # ADR-0004 asks for per-chunk provenance to be *logged*, and ADR-0003 §2's rule says these
    # lines may not carry a question — a sub-query is derived from one, so it is user content
    # too. Hence the one place `Surfaced`'s text representation is inverted: the panel renders
    # the variant, the log records its index.
    with caplog.at_level("INFO", logger="finbrief.retrieval.retrieve"):
        retrieve(
            QUESTION,
            strategy=HYBRID,
            translate=True,
            k=2,
            store=filings_store,
            settings=SETTINGS,
            model=a_translator("Apple supplier concentration"),
        )

    (record,) = [r for r in caplog.records if getattr(r, "event", None) == "retrieval"]
    assert record.fields["translation"] is True
    assert record.fields["variants"] == 2
    assert record.fields["candidate_lists"] == 4, "two variants × two retrievers"
    assert record.fields["rrf_k"] == RRF_K
    assert record.fields["fused_candidates"] >= record.fields["hits"]
    surfaced = record.fields["provenance"][0]["surfaced"]
    assert surfaced, "the top chunk records which list surfaced it"
    assert all(row["variant"] in (0, 1) for row in surfaced)
    assert all(row["retriever"] in ("vector", "bm25") for row in surfaced)
    assert "supply chain" not in str(record.fields)
    assert "supplier concentration" not in str(record.fields)
