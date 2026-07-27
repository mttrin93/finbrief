"""Ordering and idempotency of the ingestion pipeline, against a real on-disk Chroma.

The ordering claim — gate everything before writing anything — is load-bearing: ADR-0007
wants the gate to fail before any retrieval number exists, and a half-written index from a
run that was always going to fail is what produces numbers nobody can explain. It is a
claim about *sequence*, so only a test that lets one company pass and another fail can
check it.
"""

import pytest
from langchain_core.embeddings import FakeEmbeddings

from finbrief.ingestion.chunking import chunk_filing, content_hash
from finbrief.ingestion.gate import SectionGateError
from finbrief.ingestion.model import ExtractedFiling, FilingRef, Section
from finbrief.ingestion.pipeline import ingest
from finbrief.retrieval.vectorstore import (
    build_filings_store,
    content_hashes_by_accession,
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


def accessions(store) -> set[str]:
    return set(content_hashes_by_accession(store))


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


def test_a_rerun_after_an_extractor_fix_rewrites_the_same_accession(store):
    """The skip is keyed on the text as well as the accession — because a fix moved text.

    Commit `a0614de` moved eleven Sections' boundaries without a single accession changing.
    An accession-only skip therefore left the KB holding chunks of pre-repair text while
    the run reported `skipped (already ingested)` and the verification artifact attested to
    the repaired text: the knowledge base and its own evidence disagreeing, with `--force`
    the only cure and nothing to say it was needed (ADR-0007 §7).
    """
    ingest([a_filing()], store=store)
    # The JPM shape in miniature: same accession, same company, a boundary that moved.
    repaired = a_filing(
        sections={
            s: f"{s.value}. {s.heading}\n\nThe repaired opening line. {BODY}" for s in Section
        }
    )

    report = ingest([repaired], store=store)

    assert report.skipped == ()
    assert report.chunks_written == {"AAPL": count(store)}
    assert content_hashes_by_accession(store) == {
        "0000320193-25-000079": {content_hash(repaired)}
    }
    stored = store.get(where={"section": "Item 1"}, include=["documents"])
    assert any("The repaired opening line." in document for document in stored["documents"])


def test_a_rewrite_leaves_no_chunk_of_the_version_it_replaced(store):
    # Same accession, shorter text: the chunk count falls, so the ids the old text needed
    # and the new one does not must go — orphans are as retrievable as anything else.
    ingest([a_filing()], store=store)
    before = count(store)
    # Still wordy enough for the gate — Item 7's twelve-word heading needs 240 — and short
    # enough to need fewer chunks than the text already in the store.
    shorter = "The Company designs and sells devices to customers worldwide. " * 28
    shrunk = a_filing(sections={s: f"{s.value}. {s.heading}\n\n{shorter}" for s in Section})

    report = ingest([shrunk], store=store)

    assert count(store) < before
    assert report.chunks_written == {"AAPL": count(store)}
    assert content_hashes_by_accession(store) == {
        "0000320193-25-000079": {content_hash(shrunk)}
    }


def test_a_failed_rewrite_leaves_the_version_already_in_the_store(store, monkeypatch):
    # Pruning runs *after* the write for the same reason the eviction does: clearing the
    # accession first would answer a paid-API error by emptying the company outright.
    ingest([a_filing()], store=store)
    before = count(store)

    import finbrief.ingestion.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "write_chunks",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embedding API is down")),
    )
    moved = a_filing(sections={s: f"{s.value}. {s.heading}\n\nx\n\n{BODY}" for s in Section})

    with pytest.raises(RuntimeError, match="embedding API"):
        ingest([moved], store=store)

    assert count(store) == before, "the version we had is still answerable"


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
    assert accessions(store) == {"0000320193-25-000079"}


def test_a_new_fiscal_years_filing_supersedes_the_old_one(store):
    # The KB is the *latest* 10-K per company (spec §Knowledge base), and idempotency is
    # keyed on the accession — a fiscal-year rollover is a new accession, so without the
    # eviction the old year's chunks would sit beside the new ones and retrieval would
    # mix two fiscal years for the same ticker with no error.
    ingest([a_filing(accession="0000320193-24-000123")], store=store)

    ingest([a_filing(accession="0000320193-25-000079")], store=store)

    assert accessions(store) == {"0000320193-25-000079"}


def test_eviction_happens_even_when_the_current_filing_is_skipped(store):
    # A store built before the eviction existed can already hold two years for one
    # ticker; the idempotent skip must not preserve that state.
    write_chunks(store, chunk_filing(a_filing(accession="0000320193-24-000123")))
    write_chunks(store, chunk_filing(a_filing(accession="0000320193-25-000079")))

    report = ingest([a_filing(accession="0000320193-25-000079")], store=store)

    assert report.skipped == ("AAPL",)
    assert accessions(store) == {"0000320193-25-000079"}


def test_a_failed_write_leaves_the_previous_year_rather_than_nothing(store, monkeypatch):
    """Eviction runs after the write, so a paid-API failure cannot empty a company.

    `write_chunks` is where the embedding call happens. With the eviction first, an error
    on company 8 of 15 left that ticker holding neither its old accession's chunks nor its
    new ones — a company silently absent from the KB, which is the failure ADR-0007's gate
    exists to prevent in the first place. Deleting last means the worst interleaved state
    is a stale year still present, which the next run repairs.
    """
    write_chunks(store, chunk_filing(a_filing(accession="0-24-last-year")))
    before = count(store)
    assert before

    import finbrief.ingestion.pipeline as pipeline_module

    def refuses_to_embed(*args, **kwargs):
        raise RuntimeError("embedding API is down")

    monkeypatch.setattr(pipeline_module, "write_chunks", refuses_to_embed)

    with pytest.raises(RuntimeError, match="embedding API"):
        ingest([a_filing(accession="0-25-this-year")], store=store)

    assert count(store) == before, "last year's chunks are still answerable"
    assert accessions(store) == {"0-24-last-year"}
