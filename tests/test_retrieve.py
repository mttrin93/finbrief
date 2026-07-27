"""`retrieve(question, strategy, k)` — the deterministic retrieval seam (ADR-0003).

Testing seam 1 (spec §Testing Decisions): all retrieval-quality behaviour is asserted
here, at the same entry point the evaluation harness and `search_filings` will drive.

Runs against a real on-disk Chroma in `tmp_path` (the `filings_store` fixture), filled with
chunks of the *recorded* AAPL filing and embedded by a keyword-counting fake — real store
and real distance ranking, no network and no paid embeddings. The ingested `data/chroma`
store is deliberately not a fixture: it needs the live embedding model to be queryable at
all.
"""

from __future__ import annotations

import pytest

from finbrief.config import RetrievalStrategy, Settings
from finbrief.ingestion.model import Section
from finbrief.retrieval.retrieve import retrieve

SETTINGS = Settings.from_env({"OPENROUTER_API_KEY": "sk-test", "FINBRIEF_RETRIEVAL_K": "3"})

QUESTION = "What are the risks to Apple's supply chain?"


def test_retrieve_returns_k_contexts_carrying_a_chunks_provenance(filings_store):
    contexts = retrieve(QUESTION, strategy=RetrievalStrategy.VECTOR, k=3, store=filings_store)

    assert len(contexts) == 3
    assert [context.rank for context in contexts] == [1, 2, 3]
    for context in contexts:
        assert context.ticker == "AAPL"
        assert context.fiscal_year == 2025
        assert context.filing_type == "10-K"
        assert context.section in set(Section)
        assert context.accession == "0000320193-25-000079"
        assert context.body


def test_contexts_come_back_nearest_first(filings_store):
    contexts = retrieve(QUESTION, strategy=RetrievalStrategy.VECTOR, k=5, store=filings_store)

    distances = [context.distance for context in contexts]
    assert distances == sorted(distances), "rank 1 is the nearest chunk, not an arbitrary one"


def test_a_risk_question_retrieves_risk_factors_and_a_margin_question_the_mda(filings_store):
    # The one assertion that would notice a retrieval that returns *something* for every
    # question: two questions whose answers live in different Sections of the same filing.
    risks = retrieve(QUESTION, strategy=RetrievalStrategy.VECTOR, k=3, store=filings_store)
    margins = retrieve(
        "How did gross margin change?",
        strategy=RetrievalStrategy.VECTOR,
        k=3,
        store=filings_store,
    )

    assert risks[0].section is Section.RISK_FACTORS
    assert margins[0].section is Section.MDA


def test_the_same_question_retrieves_the_same_contexts_in_the_same_order(filings_store):
    # ADR-0003 stakes the RAGAs and A/B numbers on this: a chain that reorders between
    # runs makes a strategy comparison unrepeatable.
    first = retrieve(QUESTION, strategy=RetrievalStrategy.VECTOR, k=4, store=filings_store)
    second = retrieve(QUESTION, strategy=RetrievalStrategy.VECTOR, k=4, store=filings_store)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.distance for c in first] == [c.distance for c in second]


def test_a_context_body_omits_the_provenance_header_the_index_carries(filings_store):
    # `Chunk.text` prepends `AAPL | FY2025 10-K | Item 1A. Risk Factors` for BM25
    # (ADR-0004). Handing it to the LLM again — inside a block already labelled with the
    # same metadata — only invites it to be cited as the filer's own words.
    contexts = retrieve(QUESTION, strategy=RetrievalStrategy.VECTOR, k=1, store=filings_store)

    assert not contexts[0].body.startswith("AAPL | FY2025 10-K |")


def test_a_context_states_the_citation_a_marker_resolves_to(filings_store):
    contexts = retrieve(QUESTION, strategy=RetrievalStrategy.VECTOR, k=1, store=filings_store)

    assert contexts[0].citation == "AAPL 10-K FY2025, Item 1A"


def test_retrieving_from_an_empty_collection_returns_nothing(empty_filings_store):
    # The tiered-error-handling contract (spec §Tools): retrieval level, empty result ->
    # the caller's fallback message. Not an exception, and never an unguarded `[0]`.
    assert (
        retrieve("anything", strategy=RetrievalStrategy.VECTOR, k=5, store=empty_filings_store)
        == ()
    )


def test_hybrid_is_refused_rather_than_silently_retrieving_vector(filings_store):
    # `config.DEFAULT_STRATEGY` is already `hybrid` — pre-registered before any A/B data
    # exists (ADR-0005). Answering a hybrid request with vector results would report a
    # strategy that never ran; Phase 4 implements it here.
    with pytest.raises(NotImplementedError, match="Phase 4"):
        retrieve("anything", strategy=RetrievalStrategy.HYBRID, k=5, store=filings_store)


def test_k_and_the_store_fall_back_to_the_application_settings(monkeypatch, filings_store):
    # The production path: the app passes neither, so `settings.retrieval_k` and the
    # configured `filings` collection decide. Every other test here injects both.
    #
    # The fake hands back a *populated* store, which is what makes `SETTINGS`'
    # `FINBRIEF_RETRIEVAL_K=3` observable. Pointed at an empty collection this asserted
    # `== ()`, true for every possible k, so half of what the test is named for went
    # unchecked (issue #5 review). The empty case has its own test above.
    import finbrief.retrieval.retrieve as retrieve_module

    opened = []

    def fake_build_filings_store(settings):
        opened.append(settings)
        return filings_store

    monkeypatch.setattr(retrieve_module, "build_filings_store", fake_build_filings_store)
    monkeypatch.setattr(retrieve_module, "get_settings", lambda: SETTINGS)

    contexts = retrieve(QUESTION, strategy=RetrievalStrategy.VECTOR)

    assert len(contexts) == 3, "k came from FINBRIEF_RETRIEVAL_K, not the signature default"
    assert opened == [SETTINGS], "the one place the filings collection is opened"


def test_retrieval_logs_what_it_returned_without_logging_the_filing_text(filings_store, caplog):
    with caplog.at_level("INFO", logger="finbrief.retrieval.retrieve"):
        contexts = retrieve(
            QUESTION, strategy=RetrievalStrategy.VECTOR, k=2, store=filings_store
        )

    (record,) = [r for r in caplog.records if getattr(r, "event", None) == "retrieval"]
    assert record.fields["strategy"] == "vector"
    assert record.fields["k"] == 2
    assert record.fields["hits"] == 2
    assert record.fields["chunk_ids"] == [context.chunk_id for context in contexts]
    assert record.fields["latency_ms"] >= 0
    assert contexts[0].body[:40] not in str(record.fields)
