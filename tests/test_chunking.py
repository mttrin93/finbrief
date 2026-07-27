"""Section-aware chunking: the metadata contract the golden set and hybrid search rely on.

ADR-0002 authors ground truth against a cited `(ticker, section, fiscal_year)`, and
ADR-0004 dedups fused candidate lists by chunk id — so those fields, and the id's
stability, are behavior rather than bookkeeping.
"""

from finbrief.config import CHUNK_SIZE_CHARS
from finbrief.ingestion import chunking, model
from finbrief.ingestion.chunking import CHUNK_OVERLAP_CHARS, chunk_filing, content_hash
from finbrief.ingestion.model import ExtractedFiling, FilingRef, Section

PARAGRAPH = (
    "The Company designs, manufactures and markets smartphones and personal computers, "
    "and sells a variety of related services to customers worldwide.\n\n"
)


def a_filing(*, ticker: str = "AAPL", accession: str = "0000320193-25-000079", sections=None):
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
        else {s: f"{s.value}. {s.heading}\n\n{PARAGRAPH * 40}" for s in Section},
    )


def test_every_chunk_carries_the_metadata_the_golden_set_cites():
    filing = a_filing()

    chunks = chunk_filing(filing)

    assert chunks
    for chunk in chunks:
        assert chunk.metadata == {
            "ticker": "AAPL",
            "filing_type": "10-K",
            "section": chunk.section.value,
            "fiscal_year": 2025,
            "accession": "0000320193-25-000079",
            # Not for retrieval: it is how a re-run tells this filing from a later
            # extraction of the same filing (ADR-0007 §7).
            "content_hash": content_hash(filing),
        }


def test_the_content_hash_tracks_the_text_and_not_the_accession():
    # The whole point of it. An extractor fix moves a Section's boundary without EDGAR
    # re-publishing anything, so the accession cannot answer "is this what we stored?".
    filing = a_filing()
    moved = a_filing(
        sections={
            s: f"{s.value}. {s.heading}\n\nTable of Contents\n\n{PARAGRAPH * 40}"
            for s in Section
        }
    )

    assert content_hash(moved) != content_hash(filing)
    assert content_hash(a_filing()) == content_hash(filing), "and is stable across runs"


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
    # Every sentence is unique to its position — with the repeated-paragraph fixture any
    # word of chunk 0's tail appeared in chunk 1 whether or not overlap existed, and the
    # test stayed green with the overlap deleted outright.
    prose = " ".join(
        f"Sentence number {i:04d} of the business narrative continues here." for i in range(60)
    )
    sections = dict(a_filing().sections)
    sections[Section.BUSINESS] = f"Item 1. Business\n\n{prose}"

    chunks = [
        c
        for c in chunk_filing(a_filing(sections=sections))
        if c.section is Section.BUSINESS and "Sentence number" in c.body
    ]

    assert len(chunks) > 1, "the fixture needs to be long enough to split"
    for previous, current in zip(chunks, chunks[1:], strict=False):
        head = current.body[:40]
        assert head in previous.body, (
            f"each chunk must reopen inside its predecessor's tail "
            f"({CHUNK_OVERLAP_CHARS}-char overlap), but {head!r} appears nowhere there"
        )


def test_no_chunk_opens_with_the_separator_it_was_split_on():
    # `keep_separator=True` hands the separator to the *following* chunk, so every chunk
    # split at a sentence boundary opened with a dangling ". " — 31 of the 152 chunks of
    # the recorded AAPL filing did. That fragment is embedded, BM25-indexed and shown as
    # the first words of a citation. The overlap test above hid it behind `lstrip(". ")`.
    prose = " ".join(
        f"Sentence number {i:04d} of the business narrative continues here." for i in range(60)
    )
    sections = dict(a_filing().sections)
    sections[Section.BUSINESS] = f"Item 1. Business\n\n{prose}"

    chunks = chunk_filing(a_filing(sections=sections))

    assert len(chunks) > len(Section), "the fixture needs to be long enough to split"
    for chunk in chunks:
        assert not chunk.body.startswith((". ", ".", " ", "\n")), (
            f"chunk {chunk.id} opens with a separator fragment: {chunk.body[:40]!r}"
        )


def test_the_content_hash_covers_every_splitter_parameter(monkeypatch):
    # It is not just the sizes that decide what gets stored: the separator list picks where
    # every boundary falls and `_KEEP_SEPARATOR` picks which side of the cut keeps the
    # separator. Hashing only the sizes meant changing either rewrote every chunk while the
    # hash stayed put, so the next run reported `skipped (already ingested)` over text it
    # no longer produces — the exact hole ADR-0007 §7 exists to close.
    filing = a_filing()
    baseline = content_hash(filing)

    monkeypatch.setattr(chunking, "_KEEP_SEPARATOR", True)
    assert content_hash(filing) != baseline, "_KEEP_SEPARATOR is not in the hash"

    monkeypatch.setattr(chunking, "_KEEP_SEPARATOR", "end")
    monkeypatch.setattr(chunking, "_SEPARATORS", ("\n\n", " ", ""))
    assert content_hash(filing) != baseline, "_SEPARATORS is not in the hash"


def test_the_content_hash_covers_the_provenance_header_it_indexes(monkeypatch):
    # `Section.heading` is printed into `Chunk.text`, which is what gets embedded — so
    # rewording one changes every stored chunk of that Section. Hashing `section.value`
    # alone left that invisible.
    filing = a_filing()
    baseline = content_hash(filing)

    monkeypatch.setitem(model._HEADINGS, Section.MDA, "MD&A")

    assert Section.MDA.heading == "MD&A", "the fixture needs the reword to take effect"
    assert content_hash(filing) != baseline, "section.heading is not in the hash"


def test_a_section_incorporated_by_reference_is_not_chunked():
    # The gate deliberately *passes* a pointer Section (it is a lawful filing), so this
    # skip is the only thing keeping it out of the KB: as a chunk it would retrieve for
    # every market-risk question and ground none of them (ADR-0007 amendment). Six of
    # fifteen Universe filers answer Item 7A this way.
    sections = dict(a_filing().sections)
    sections[Section.MARKET_RISK] = (
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk\n\n"
        "Refer to the Market Risk Management section of Management's discussion and "
        "analysis on pages 133-142 for a discussion of quantitative and qualitative "
        "disclosures about market risk."
    )

    chunks = chunk_filing(a_filing(sections=sections))

    assert {c.section for c in chunks} == set(Section) - {Section.MARKET_RISK}


def test_each_chunk_names_its_company_and_section_for_bm25():
    # ADR-0004's exact-identifier bucket is BM25 finding the literal "Item 1A" or "AAPL".
    # Metadata is invisible to BM25, which scores text — so without this, only the one
    # chunk that happens to contain the heading can match a query naming the section.
    for chunk in chunk_filing(a_filing()):
        assert "AAPL" in chunk.text
        assert chunk.section.value in chunk.text
        assert chunk.body in chunk.text, "the body is carried verbatim, never rewritten"
