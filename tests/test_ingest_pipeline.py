"""Ordering and idempotency of the ingestion pipeline, against a real on-disk Chroma.

The ordering claim — gate everything before writing anything — is load-bearing: ADR-0007
wants the gate to fail before any retrieval number exists, and a half-written index from a
run that was always going to fail is what produces numbers nobody can explain. It is a
claim about *sequence*, so only a test that lets one company pass and another fail can
check it.
"""

import pytest
from langchain_core.embeddings import FakeEmbeddings

from finbrief.ingestion.chunking import chunk_filing
from finbrief.ingestion.gate import SectionGateError
from finbrief.ingestion.model import ExtractedFiling, FilingRef, Section
from finbrief.ingestion.pipeline import ingest
from finbrief.retrieval.vectorstore import (
    build_filings_store,
    ingested_accessions,
    write_chunks,
)

#: Long enough to clear the gate's body-vs-heading ratio for *every* Section — Item 7's
#: heading is twelve words, so it needs 240 words of body where Item 1's needs 40.
BODY = "The Company designs and sells devices to customers worldwide. " * 40


def a_filing(ticker="AAPL", accession="0000320193-25-000079", *, sections=None):
    return ExtractedFiling(
        ref=FilingRef(
            ticker=ticker,
            form="10-K",
            accession=accession,
            fiscal_year=2025,
            filing_date="2025-10-31",
        ),
        latest_annual_form="10-K",
        sections=sections
        if sections is not None
        else {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section},
    )


@pytest.fixture
def store(tmp_path):
    return build_filings_store(
        persist_directory=str(tmp_path / "chroma"), embeddings=FakeEmbeddings(size=32)
    )


def count(store) -> int:
    return len(store.get(include=[])["ids"])


def test_a_single_failing_company_stops_the_whole_run_writing(store):
    # The good company is first, so a pipeline that gated per-filing instead of up front
    # would already have written it by the time the bad one raised.
    broken = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    del broken[Section.MDA]

    with pytest.raises(SectionGateError):
        ingest(
            [a_filing("AAPL"), a_filing("MSFT", "0000789019-25-000118", sections=broken)],
            store=store,
        )

    assert count(store) == 0, "no company is written when any company fails"


def test_a_clean_run_writes_every_company(store):
    report = ingest([a_filing("AAPL"), a_filing("MSFT", "0000789019-25-000118")], store=store)

    assert set(report.chunks_written) == {"AAPL", "MSFT"}
    assert report.total_chunks == count(store)
    assert report.skipped == ()


def test_a_second_run_skips_what_is_already_ingested(store):
    ingest([a_filing()], store=store)

    report = ingest([a_filing()], store=store)

    assert report.skipped == ("AAPL",)
    assert report.chunks_written == {}


def test_force_re_embeds_without_leaving_orphaned_chunks(store):
    # Ids embed a chunk index, so a chunker that produces *fewer* chunks would upsert over
    # the low indices and strand the high ones — still in the collection, still
    # retrievable, belonging to a chunking scheme that no longer exists.
    ingest([a_filing()], store=store)
    before = count(store)

    shorter = "The Company designs and sells devices to customers worldwide. " * 28
    shrunk = a_filing(sections={s: f"{s.value}. {s.heading}\n\n{shorter}" for s in Section})
    report = ingest([shrunk], store=store, force=True)

    assert report.chunks_written == {"AAPL": count(store)}
    assert count(store) < before, "the chunks the shorter text no longer needs are gone"
    assert ingested_accessions(store) == {"0000320193-25-000079"}


def test_a_new_fiscal_years_filing_supersedes_the_old_one(store):
    # The KB is the *latest* 10-K per company (spec §Knowledge base), and idempotency is
    # keyed on the accession — a fiscal-year rollover is a new accession, so without the
    # eviction the old year's chunks would sit beside the new ones and retrieval would
    # mix two fiscal years for the same ticker with no error.
    ingest([a_filing(accession="0000320193-24-000123")], store=store)

    ingest([a_filing(accession="0000320193-25-000079")], store=store)

    assert ingested_accessions(store) == {"0000320193-25-000079"}


def test_eviction_happens_even_when_the_current_filing_is_skipped(store):
    # A store built before the eviction existed can already hold two years for one
    # ticker; the idempotent skip must not preserve that state.
    write_chunks(store, chunk_filing(a_filing(accession="0000320193-24-000123")))
    write_chunks(store, chunk_filing(a_filing(accession="0000320193-25-000079")))

    report = ingest([a_filing(accession="0000320193-25-000079")], store=store)

    assert report.skipped == ("AAPL",)
    assert ingested_accessions(store) == {"0000320193-25-000079"}
