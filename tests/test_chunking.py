"""Section-aware chunking: the metadata contract the golden set and hybrid search rely on.

ADR-0002 authors ground truth against a cited `(ticker, section, fiscal_year)`, and
ADR-0004 dedups fused candidate lists by chunk id — so those fields, and the id's
stability, are behavior rather than bookkeeping.
"""

from finbrief.config import CHUNK_SIZE_CHARS
from finbrief.ingestion.chunking import CHUNK_OVERLAP_CHARS, chunk_filing
from finbrief.ingestion.model import ExtractedFiling, FilingRef, Section

PARAGRAPH = (
    "The Company designs, manufactures and markets smartphones and personal computers, "
    "and sells a variety of related services to customers worldwide.\n\n"
)


def a_filing(*, ticker: str = "AAPL", accession: str = "0000320193-25-000079"):
    return ExtractedFiling(
        ref=FilingRef(
            ticker=ticker,
            form="10-K",
            accession=accession,
            fiscal_year=2025,
            filing_date="2025-10-31",
        ),
        latest_annual_form="10-K",
        sections={s: f"{s.value}. {s.heading}\n\n{PARAGRAPH * 40}" for s in Section},
    )


def test_every_chunk_carries_the_metadata_the_golden_set_cites():
    chunks = chunk_filing(a_filing())

    assert chunks
    for chunk in chunks:
        assert chunk.metadata == {
            "ticker": "AAPL",
            "filing_type": "10-K",
            "section": chunk.section.value,
            "fiscal_year": 2025,
            "accession": "0000320193-25-000079",
        }


def test_metadata_values_are_all_chroma_primitives():
    # Chroma rejects a metadata value that is not str/int/float/bool, and it rejects it at
    # write time — after the embedding call has been paid for. An enum member is the easy
    # mistake here, since `section` is one everywhere else in the pipeline.
    for chunk in chunk_filing(a_filing()):
        for key, value in chunk.metadata.items():
            assert type(value) in (str, int, float, bool), f"{key} is {type(value)}"


def test_chunk_ids_are_unique_and_stable_across_runs():
    # ADR-0004 dedups the fused candidate lists by chunk id, and re-ingesting upserts by
    # it — so an id that moves between runs silently duplicates the knowledge base.
    first = chunk_filing(a_filing())
    second = chunk_filing(a_filing())

    ids = [c.id for c in first]
    assert len(set(ids)) == len(ids)
    assert ids == [c.id for c in second]


def test_ids_are_scoped_to_the_accession_so_two_filings_never_collide():
    this_year = {c.id for c in chunk_filing(a_filing(accession="0000320193-25-000079"))}
    last_year = {c.id for c in chunk_filing(a_filing(accession="0000320193-24-000123"))}

    assert not (this_year & last_year)


def test_no_chunk_spans_two_sections():
    # The point of section-aware chunking: a chunk labelled `Item 1A` that trails off into
    # Item 1B's text is mislabelled grounding, and nothing downstream can detect it.
    for chunk in chunk_filing(a_filing()):
        other_headings = [s.heading for s in Section if s is not chunk.section]
        assert not any(heading in chunk.text for heading in other_headings)


def test_every_section_produces_at_least_one_chunk():
    assert {c.section for c in chunk_filing(a_filing())} == set(Section)


def test_chunk_bodies_respect_the_configured_chunk_size():
    # `retrieval/embeddings.py` runs with `check_embedding_ctx_length=False`, so nothing
    # splits an over-long text on the way out — the size ceiling is the whole guarantee.
    for chunk in chunk_filing(a_filing()):
        assert len(chunk.body) <= CHUNK_SIZE_CHARS


def test_chunks_overlap_so_a_sentence_split_across_them_is_still_retrievable():
    chunks = [c for c in chunk_filing(a_filing()) if c.section is Section.BUSINESS]

    assert len(chunks) > 1, "the fixture needs to be long enough to split"
    assert CHUNK_OVERLAP_CHARS > 0
    tail = chunks[0].body[-CHUNK_OVERLAP_CHARS:]
    assert any(word in chunks[1].body for word in tail.split()[-5:])


def test_each_chunk_names_its_company_and_section_for_bm25():
    # ADR-0004's exact-identifier bucket is BM25 finding the literal "Item 1A" or "AAPL".
    # Metadata is invisible to BM25, which scores text — so without this, only the one
    # chunk that happens to contain the heading can match a query naming the section.
    for chunk in chunk_filing(a_filing()):
        assert "AAPL" in chunk.text
        assert chunk.section.value in chunk.text
        assert chunk.body in chunk.text, "the body is carried verbatim, never rewritten"
