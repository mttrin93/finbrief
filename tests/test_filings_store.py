"""Writing chunks to the `filings` collection, and not writing them twice.

Runs against a real on-disk Chroma in `tmp_path` with a deterministic fake embedding —
real store semantics, no network. What is under test is the ingestion contract (accession
idempotency, ADR-0007), not Chroma.
"""

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import FakeEmbeddings

from finbrief.config import Settings
from finbrief.ingestion.chunking import chunk_filing
from finbrief.ingestion.model import ExtractedFiling, FilingRef, Section
from finbrief.retrieval.vectorstore import (
    build_filings_store,
    chunk_counts_by_ticker,
    delete_superseded,
    ingested_accessions,
    write_chunks,
)

BODY = (
    "The Company designs, manufactures and markets smartphones and personal computers. "
) * 20


def a_filing(*, ticker="AAPL", accession="0000320193-25-000079", fiscal_year=2025):
    return ExtractedFiling(
        ref=FilingRef(
            ticker=ticker,
            form="10-K",
            accession=accession,
            fiscal_year=fiscal_year,
            filing_date="2025-10-31",
        ),
        latest_annual_form="10-K",
        sections={s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section},
    )


@pytest.fixture
def store(tmp_path):
    return build_filings_store(
        persist_directory=str(tmp_path / "chroma"), embeddings=FakeEmbeddings(size=32)
    )


def count(store) -> int:
    return len(store.get(include=[])["ids"])


def test_writing_a_filing_stores_every_chunk_with_its_metadata(store):
    chunks = chunk_filing(a_filing())

    written = write_chunks(store, chunks)

    assert written == len(chunks)
    assert count(store) == len(chunks)
    stored = store.get(where={"section": "Item 1A"}, include=["metadatas"])
    assert stored["metadatas"]
    assert all(m["ticker"] == "AAPL" for m in stored["metadatas"])
    assert all(m["fiscal_year"] == 2025 for m in stored["metadatas"])


def test_re_ingesting_the_same_accession_does_not_duplicate_the_knowledge_base(store):
    chunks = chunk_filing(a_filing())
    write_chunks(store, chunks)
    before = count(store)

    write_chunks(store, chunk_filing(a_filing()))

    assert count(store) == before, "chunk ids are derived from the accession, so this upserts"


def test_a_second_company_adds_to_the_collection_rather_than_replacing_it(store):
    write_chunks(store, chunk_filing(a_filing(ticker="AAPL")))
    apple_chunks = count(store)

    write_chunks(store, chunk_filing(a_filing(ticker="MSFT", accession="0000789019-25-000118")))

    assert count(store) > apple_chunks
    assert ingested_accessions(store) == {"0000320193-25-000079", "0000789019-25-000118"}


def test_an_empty_collection_reports_no_ingested_accessions(store):
    assert ingested_accessions(store) == set()


def test_writing_no_chunks_is_a_no_op(store):
    # A filing whose every Section was incorporated by reference produces exactly this
    # input, and Chroma raises on an empty `add_texts` — so the guard is behavior.
    assert write_chunks(store, []) == 0
    assert count(store) == 0


def test_delete_superseded_removes_only_the_tickers_other_accessions(store):
    write_chunks(store, chunk_filing(a_filing(ticker="AAPL", accession="0-24-old")))
    write_chunks(store, chunk_filing(a_filing(ticker="AAPL", accession="0-25-new")))
    write_chunks(store, chunk_filing(a_filing(ticker="MSFT", accession="0-25-msft")))

    removed = delete_superseded(store, "AAPL", "0-25-new")

    assert removed > 0
    assert ingested_accessions(store) == {"0-25-new", "0-25-msft"}


def test_delete_superseded_is_a_no_op_when_the_ticker_holds_one_filing(store):
    write_chunks(store, chunk_filing(a_filing()))
    before = count(store)

    assert delete_superseded(store, "AAPL", "0000320193-25-000079") == 0
    assert count(store) == before


def test_chunk_counts_by_ticker_reads_back_what_each_company_holds(store):
    # The ingest report cites these counts as evidence of what the knowledge base holds,
    # so they come from the collection itself — not from what one run happened to write.
    write_chunks(store, chunk_filing(a_filing(ticker="AAPL")))
    write_chunks(store, chunk_filing(a_filing(ticker="MSFT", accession="0000789019-25-000118")))

    counts = chunk_counts_by_ticker(store)

    assert set(counts) == {"AAPL", "MSFT"}
    assert sum(counts.values()) == count(store)


def test_an_empty_collection_reports_no_chunk_counts(store):
    assert chunk_counts_by_ticker(store) == {}


def test_the_collection_is_named_so_news_and_glossary_can_live_beside_it(store, tmp_path):
    # PLAN.md keeps `filings`, `news` and `glossary` as separate collections; a chunk's
    # provenance is only unambiguous if the filings KB owns its own namespace. Asserted
    # behaviourally — a neighbour collection in the same persist directory must not see
    # the filings chunks — rather than against `store._collection.name`, which is
    # langchain-chroma internals a library refactor can rename with nothing broken.
    write_chunks(store, chunk_filing(a_filing()))
    neighbour = Chroma(
        collection_name="news",
        embedding_function=FakeEmbeddings(size=32),
        persist_directory=str(tmp_path / "chroma"),  # the same directory the fixture uses
    )

    assert count(store)
    assert len(neighbour.get(include=[])["ids"]) == 0


def test_build_filings_store_falls_back_to_settings_for_both_arguments(monkeypatch, tmp_path):
    # Production passes neither argument, so this is the only path a real ingest takes —
    # and every other test here overrides both. A renamed `chroma_dir` or an embeddings
    # constructor that stopped being consulted would ship green.
    import finbrief.retrieval.vectorstore as vectorstore

    built = []
    monkeypatch.setattr(
        vectorstore,
        "build_embeddings",
        lambda settings: built.append(settings) or FakeEmbeddings(size=32),
    )
    settings = Settings.from_env(
        {"OPENROUTER_API_KEY": "key", "FINBRIEF_CHROMA_DIR": str(tmp_path / "from-settings")}
    )

    store = build_filings_store(settings)

    assert built == [settings], "the one shared embedding model, not a second constructor"
    write_chunks(store, chunk_filing(a_filing()))
    assert (tmp_path / "from-settings").exists()
