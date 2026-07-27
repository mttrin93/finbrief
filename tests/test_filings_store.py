"""Writing chunks to the `filings` collection, and not writing them twice.

Runs against a real on-disk Chroma in `tmp_path` with a deterministic fake embedding —
real store semantics, no network. What is under test is the ingestion contract (accession
idempotency, ADR-0007), not Chroma.
"""

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import FakeEmbeddings

from finbrief.config import Settings
from finbrief.ingestion.chunking import chunk_filing, content_hash
from finbrief.ingestion.model import ExtractedFiling, FilingRef, Section
from finbrief.retrieval.vectorstore import (
    all_chunks,
    build_filings_store,
    chunk_counts_by_ticker,
    content_hashes_by_accession,
    default_filings_store,
    delete_orphaned_chunks,
    delete_superseded,
    holds_any_chunks,
    nearest_chunks,
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


def accessions(store) -> set[str]:
    return set(content_hashes_by_accession(store))


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
    assert accessions(store) == {"0000320193-25-000079", "0000789019-25-000118"}


def test_an_empty_collection_reports_nothing_ingested(store):
    assert content_hashes_by_accession(store) == {}


def test_the_store_reports_whether_it_holds_anything_at_all(store):
    # `scripts/retrieval_smoke.py` asks this before it spends anything on embeddings, so
    # that "nobody has ingested yet" cannot be reported as five broken retrievals.
    assert holds_any_chunks(store) is False

    write_chunks(store, chunk_filing(a_filing()))

    assert holds_any_chunks(store) is True


def test_nearest_chunks_returns_the_documents_with_their_distances(store):
    # The one Chroma read `retrieve()` is built on, kept in this module because CLAUDE.md
    # makes it the only place the collection is opened, written, or read. `FakeEmbeddings`
    # is random, so what is asserted here is the *shape* of the contract — k, the pairing,
    # the metadata — while ordering and relevance belong to `test_retrieve.py`, which uses a
    # deterministic fake.
    write_chunks(store, chunk_filing(a_filing()))

    hits = nearest_chunks(store, "supply chain risk", 3)

    assert len(hits) == 3
    for document, distance in hits:
        assert isinstance(distance, float)
        assert document.metadata["ticker"] == "AAPL"
        assert document.page_content


def test_nearest_chunks_on_an_empty_collection_returns_nothing(store):
    assert nearest_chunks(store, "anything", 5) == []


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
    assert accessions(store) == {"0-25-new", "0-25-msft"}


def test_delete_superseded_is_a_no_op_when_the_ticker_holds_one_filing(store):
    write_chunks(store, chunk_filing(a_filing()))
    before = count(store)

    assert delete_superseded(store, "AAPL", "0000320193-25-000079") == 0
    assert count(store) == before


def test_the_store_reports_the_content_hash_its_chunks_were_written_with(store):
    # What makes the skip decision correct rather than merely cheap: the accession says
    # which document, the hash says which *version of the extraction* of it.
    filing = a_filing()
    write_chunks(store, chunk_filing(filing))

    assert content_hashes_by_accession(store) == {
        "0000320193-25-000079": {content_hash(filing)}
    }


def test_a_chunk_written_before_the_hash_existed_matches_no_run(store):
    # The committed KB is exactly this: chunks with no `content_hash` in their metadata.
    # They must not be mistaken for a match, or the stale rows they came from survive.
    chunks = chunk_filing(a_filing())
    store.add_texts(
        texts=[c.text for c in chunks],
        metadatas=[
            {k: v for k, v in c.metadata.items() if k != "content_hash"} for c in chunks
        ],
        ids=[c.id for c in chunks],
    )

    assert content_hashes_by_accession(store) == {"0000320193-25-000079": {""}}


def test_pruning_removes_the_chunks_a_rewrite_no_longer_covers(store):
    # Ids embed a chunk index, so text that now splits into fewer chunks upserts over the
    # low indices and strands the high ones — still retrievable, belonging to a version of
    # the Section that no longer exists.
    write_chunks(store, chunk_filing(a_filing()))
    shorter = a_filing()
    kept = chunk_filing(
        type(shorter)(
            ref=shorter.ref,
            latest_annual_form="10-K",
            sections={s: f"{s.value}. {s.heading}\n\nA short body." for s in Section},
        )
    )
    write_chunks(store, kept)

    removed = delete_orphaned_chunks(store, "0000320193-25-000079", {c.id for c in kept})

    assert removed > 0
    assert count(store) == len(kept)


def test_pruning_keeps_another_filings_chunks_and_reports_nothing_removed(store):
    write_chunks(store, chunk_filing(a_filing(ticker="MSFT", accession="0-25-msft")))
    chunks = chunk_filing(a_filing())
    write_chunks(store, chunks)
    before = count(store)

    assert delete_orphaned_chunks(store, "0000320193-25-000079", {c.id for c in chunks}) == 0
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


def test_all_chunks_reads_back_every_chunk_as_it_was_indexed(store):
    # BM25 scores *text*, so hybrid search needs the whole corpus rather than a neighbourhood
    # of it (ADR-0004) — and this is the read that supplies it, in the one module allowed to
    # open the collection. What it must preserve is the *indexed* text, provenance header
    # included: the header is the whole reason a ticker and a Section literal are matchable on
    # every chunk of a Section and not only the one the splitter left the heading in.
    written = chunk_filing(a_filing())
    write_chunks(store, written)

    held = all_chunks(store)

    assert {document.id for document in held} == {chunk.id for chunk in written}
    by_id = {document.id: document for document in held}
    for chunk in written:
        assert by_id[chunk.id].page_content == chunk.text
        assert by_id[chunk.id].metadata["ticker"] == "AAPL"
        assert by_id[chunk.id].metadata["section"] == chunk.section.value


def test_all_chunks_of_an_empty_collection_is_empty_rather_than_an_error(store):
    # The un-ingested case: `store.get` returns `None` for the lists it was asked to include,
    # which an unguarded zip would turn into a `TypeError` on the first query.
    assert all_chunks(store) == ()


def test_the_application_shares_one_handle_on_the_collection(monkeypatch, tmp_path):
    # Phase 4's BM25 index is cached against the store object that holds its corpus
    # (`retrieval/hybrid.py`). A `Chroma` opened afresh per query — which is what `retrieve()`
    # used to do — would miss that cache every time and rebuild a 5,800-chunk lexical index for
    # one question.
    import finbrief.retrieval.vectorstore as vectorstore

    monkeypatch.setattr(
        vectorstore, "build_embeddings", lambda settings: FakeEmbeddings(size=32)
    )
    settings = Settings.from_env(
        {"OPENROUTER_API_KEY": "key", "FINBRIEF_CHROMA_DIR": str(tmp_path / "shared")}
    )
    default_filings_store.cache_clear()

    assert default_filings_store(settings) is default_filings_store(settings)
