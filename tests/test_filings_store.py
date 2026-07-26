"""Writing chunks to the `filings` collection, and not writing them twice.

Runs against a real on-disk Chroma in `tmp_path` with a deterministic fake embedding —
real store semantics, no network. What is under test is the ingestion contract (accession
idempotency, ADR-0007), not Chroma.
"""

import pytest
from langchain_core.embeddings import FakeEmbeddings

from finbrief.ingestion.chunking import chunk_filing
from finbrief.ingestion.model import ExtractedFiling, FilingRef, Section
from finbrief.retrieval.vectorstore import (
    FILINGS_COLLECTION,
    build_filings_store,
    chunk_counts_by_ticker,
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


def test_the_collection_is_named_so_news_and_glossary_can_live_beside_it(store):
    # PLAN.md keeps `filings`, `news` and `glossary` as separate collections; a chunk's
    # provenance is only unambiguous if the filings KB owns its own namespace.
    assert store._collection.name == FILINGS_COLLECTION
