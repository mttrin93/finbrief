"""The persisted `filings` collection — the one place chunks are written and read.

PLAN.md keeps `filings`, `news` and `glossary` in separate collections so a chunk's
provenance is unambiguous; this module owns the first. It takes its embedding model from
`retrieval/embeddings.py` and never constructs one — an index is only searchable by the
model that wrote it, and a second constructor here is exactly how ingest and query drift
apart with no error to show for it.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from finbrief.config import Settings, get_settings
from finbrief.ingestion.chunking import Chunk
from finbrief.retrieval.embeddings import build_embeddings

#: The collection holding curated 10-K Sections. A plain constant so ingest, retrieval and
#: the evaluation harness cannot disagree about which collection they mean.
FILINGS_COLLECTION = "filings"


def build_filings_store(
    settings: Settings | None = None,
    *,
    persist_directory: str | None = None,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Open (or create) the persisted `filings` collection.

    `embeddings` is injectable for tests only — a fake keeps the suite hermetic while
    still exercising real Chroma semantics. Production passes nothing and gets the one
    shared model.

    Settings are resolved lazily, per argument. A test that supplies both overrides names
    everything this function needs, and reading `.env` anyway would make it fail on a
    machine that has no API key — the one thing `conftest.py` exists to prevent.
    """

    def resolved() -> Settings:
        nonlocal settings
        settings = settings or get_settings()
        return settings

    return Chroma(
        collection_name=FILINGS_COLLECTION,
        embedding_function=embeddings or build_embeddings(resolved()),
        persist_directory=persist_directory or resolved().chroma_dir,
    )


def write_chunks(store: Chroma, chunks: Sequence[Chunk]) -> int:
    """Upsert `chunks` into the collection and return how many were written.

    Idempotent by construction rather than by convention: the ids come from
    `chunking.chunk_id`, which is rooted in the accession number, so re-ingesting a filing
    overwrites its own rows instead of appending a second copy of the knowledge base.
    """
    if not chunks:
        return 0
    store.add_texts(
        texts=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        ids=[chunk.id for chunk in chunks],
    )
    return len(chunks)


def ingested_accessions(store: Chroma) -> set[str]:
    """Every accession number already in the collection.

    Lets a re-run skip a filing it already holds *before* paying for its embeddings.
    `write_chunks` would keep the collection correct either way; this keeps it cheap.
    """
    stored = store.get(include=["metadatas"])
    return {
        metadata["accession"]
        for metadata in stored["metadatas"] or ()
        if metadata.get("accession")
    }
