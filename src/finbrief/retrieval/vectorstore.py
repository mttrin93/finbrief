"""The persisted `filings` collection — the one place chunks are written and read.

PLAN.md keeps `filings`, `news` and `glossary` in separate collections so a chunk's
provenance is unambiguous; this module owns the first. It takes its embedding model from
`retrieval/embeddings.py` and never constructs one — an index is only searchable by the
model that wrote it, and a second constructor here is exactly how ingest and query drift
apart with no error to show for it.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
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

    Settings are resolved only when an argument is missing. A test that supplies both
    overrides names everything this function needs, and reading `.env` anyway would make
    it fail on a machine that has no API key — the one thing `conftest.py` exists to
    prevent.
    """
    if embeddings is None or persist_directory is None:
        settings = settings or get_settings()
    # `is None`, matching the guard above — not `or`. With truthiness, a falsy-but-present
    # argument (`persist_directory=""`) skips the guard, leaves `settings` unresolved, and
    # then reports the mistake as `AttributeError: 'NoneType' has no attribute
    # 'chroma_dir'` from inside this function.
    return Chroma(
        collection_name=FILINGS_COLLECTION,
        embedding_function=(
            embeddings if embeddings is not None else build_embeddings(settings)
        ),
        persist_directory=(
            persist_directory if persist_directory is not None else settings.chroma_dir
        ),
    )


def nearest_chunks(store: Chroma, question: str, k: int) -> list[tuple[Document, float]]:
    """The `k` chunks nearest `question`, nearest first, each with its distance.

    The read side of this module's rule (CLAUDE.md: this is the only place the collection is
    opened, written, or read). `retrieval/retrieve.py` owns *what a retrieval means* — the
    strategy, the ranking contract, the `Context` shape — and calls this for the one Chroma
    operation underneath it, so the collection's API stays crossed in a single file.

    Distances are Chroma's, in the collection's own space (L2 for `filings`): **lower is
    nearer**, and not a normalised similarity. Deliberately returned raw rather than through
    `similarity_search_with_relevance_scores`, whose 0…1 rescaling assumes normalised
    embeddings and would silently invent a similarity we have not verified.
    """
    return store.similarity_search_with_score(question, k=k)


def holds_any_chunks(store: Chroma) -> bool:
    """Whether the collection holds anything at all, without reading what.

    `limit=1` and no `include`, so this stays an existence check on a 5,800-chunk
    collection rather than a metadata scan. Exists so `scripts/retrieval_smoke.py` can tell
    "retrieval is broken" from "nobody has ingested yet" before it spends anything on
    embeddings.
    """
    return bool(store.get(limit=1, include=[])["ids"])


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


def delete_orphaned_chunks(store: Chroma, accession: str, keep_ids: Collection[str]) -> int:
    """Remove chunks of one filing that the ids just written no longer include.

    A re-embed needs this because ids carry a chunk *index*: text that now splits into
    fewer chunks upserts over `…:Item 1:0-19` and leaves `…:Item 1:20-24` behind, orphaned,
    still retrievable, belonging to a version of the Section that no longer exists.

    Pruning *after* the write rather than clearing first, for the reason `delete_superseded`
    runs last too: `write_chunks` is where the paid embedding call happens, and a run that
    deleted first would answer an API error by leaving the company with nothing at all.
    Upsert-then-prune means the worst interleaved state is the version we already had.
    """
    held = store.get(where={"accession": accession}, include=[])["ids"]
    orphans = [chunk_id for chunk_id in held if chunk_id not in keep_ids]
    if orphans:
        store.delete(ids=orphans)
    return len(orphans)


def delete_superseded(store: Chroma, ticker: str, accession: str) -> int:
    """Remove `ticker`'s chunks belonging to any filing other than `accession`.

    The KB holds the *latest* 10-K per company (spec §Knowledge base), and idempotency is
    keyed on the accession — so when a company files a new year's 10-K, the new accession
    would be written *beside* the old one, and retrieval would mix two fiscal years for
    the same ticker with no error to show for it. Returns how many chunks were evicted,
    so the caller can leave evidence of what a re-run removed.
    """
    stale = store.get(
        where={"$and": [{"ticker": ticker}, {"accession": {"$ne": accession}}]},
        include=[],
    )
    ids = stale["ids"]
    if ids:
        store.delete(ids=ids)
    return len(ids)


def chunk_counts_by_ticker(store: Chroma) -> dict[str, int]:
    """How many chunks each company holds in the collection, read from the store itself.

    The ingest report cites these as evidence of what the knowledge base contains, so
    they cannot come from what one run wrote — after an idempotent skip the write count
    is zero and the holding is not.
    """
    stored = store.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for metadata in stored["metadatas"] or ():
        ticker = metadata.get("ticker")
        if ticker:
            counts[ticker] = counts.get(ticker, 0) + 1
    return counts


def content_hashes_by_accession(store: Chroma) -> dict[str, set[str]]:
    """What the collection holds, keyed by filing: accession -> its chunks' content hashes.

    Lets a re-run skip a filing it already holds *before* paying for its embeddings — and,
    because the value is `chunking.content_hash` rather than just the accession's presence,
    tell that case apart from a filing whose extraction has changed since it was written
    (`pipeline.ingest`). A set because a run interrupted mid-write can leave two.

    A chunk from before the hash existed reports `""`, which matches nothing a run would
    write, so such a filing is re-ingested rather than trusted — the conservative answer,
    and the one the currently committed knowledge base needs.
    """
    stored = store.get(include=["metadatas"])
    holdings: dict[str, set[str]] = {}
    for metadata in stored["metadatas"] or ():
        accession = metadata.get("accession")
        if accession:
            holdings.setdefault(accession, set()).add(str(metadata.get("content_hash", "")))
    return holdings
