"""Ordering and idempotency of the ingestion pipeline, against a real on-disk Chroma.

The ordering claim — gate everything before writing anything — is load-bearing: ADR-0007
wants the gate to fail before any retrieval number exists, and a half-written index from a
run that was always going to fail is what produces numbers nobody can explain. It is a
claim about *sequence*, so only a test that lets one company pass and another fail can
check it.
"""

import pytest
from langchain_core.embeddings import FakeEmbeddings

from finbrief.ingestion.gate import SectionGateError
from finbrief.ingestion.model import ExtractedFiling, FilingRef, Section
from finbrief.ingestion.pipeline import fetch_filings, ingest
from finbrief.retrieval.vectorstore import build_filings_store, ingested_accessions

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


def test_fetch_filings_visits_every_ticker_in_order():
    seen = []

    def fake_fetch(ticker):
        seen.append(ticker)
        return a_filing(ticker)

    filings = fetch_filings(["AAPL", "MSFT", "NVDA"], fetch=fake_fetch)

    assert seen == ["AAPL", "MSFT", "NVDA"]
    assert [f.ref.ticker for f in filings] == ["AAPL", "MSFT", "NVDA"]
