"""`retrieve(question, strategy, k)` — the deterministic retrieval engine (ADR-0003).

The one entry point to the knowledge base. The evaluation harness calls it directly to
produce clean `(question → contexts → answer)` triples, and Phase 3's `search_filings`
tool wraps *this* function so the measured chain and the shipped chain are one code path.
Nothing above it may open the `filings` collection itself.

**Return shape.** ADR-0003 writes the signature as `→ (contexts, scores)`. It is one
sequence of `Context` here, each carrying its own `distance` and `rank`, because two
parallel lists can desynchronise — a caller that sorts, filters or dedups one of them
(which is exactly what Phase 4's RRF fusion does) silently mispairs every score. The ADR's
"scores" is `Context.distance`.

**Vector only, for now.** Query translation and BM25 + RRF fusion arrive in Phase 4
(ADR-0004) behind the same `strategy` flag; `hybrid` raises here rather than quietly
retrieving something that was never measured.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

from finbrief.config import RetrievalStrategy, Settings, get_settings
from finbrief.ingestion.model import Section
from finbrief.observability.logging_setup import log_event
from finbrief.retrieval.vectorstore import build_filings_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Context:
    """One retrieved chunk, as it is handed to the LLM and shown in the sources panel.

    Named for the LLM's side of the boundary, per CONTEXT.md: a **Chunk** is what the
    store holds, a **Context** is what a turn was grounded in. It carries the chunk's
    provenance rather than a reference to it so a citation, a log line and the sources
    panel all read the same fields.
    """

    chunk_id: str
    body: str
    ticker: str
    filing_type: str
    section: Section
    fiscal_year: int
    accession: str
    #: Chroma's vector distance — **lower is nearer**. Deliberately not called a "score":
    #: the `filings` collection is L2, so this is a squared distance and not a normalised
    #: similarity, and calling it a score would invite a reader to expect 0…1 and
    #: higher-is-better. Phase 4's RRF consumes `rank`, which is what fusion needs anyway.
    distance: float
    #: 1-based position in this retrieval, so a citation index and the panel agree.
    rank: int

    @property
    def citation(self) -> str:
        """`AAPL 10-K FY2025, Item 1A` — what an inline `[n]` marker resolves to."""
        return f"{self.ticker} {self.filing_type} FY{self.fiscal_year}, {self.section.value}"


def retrieve(
    question: str,
    *,
    strategy: RetrievalStrategy = RetrievalStrategy.VECTOR,
    k: int | None = None,
    store: Chroma | None = None,
    settings: Settings | None = None,
) -> tuple[Context, ...]:
    """Retrieve the `k` chunks nearest `question`, nearest first.

    Deterministic: the same question, strategy and `k` against the same collection return
    the same contexts in the same order (ADR-0003 — the eval harness depends on it).

    `store` is injectable so tests can run against a fixture collection, and so the
    evaluation harness can point at a throwaway index; production passes nothing.
    """
    if strategy is not RetrievalStrategy.HYBRID and strategy is not RetrievalStrategy.VECTOR:
        raise ValueError(f"unknown retrieval strategy {strategy!r}")
    if strategy is RetrievalStrategy.HYBRID:
        raise NotImplementedError(
            "hybrid retrieval (query translation + BM25 + RRF) lands in Phase 4, "
            "ADR-0004. Until then set FINBRIEF_RETRIEVAL_STRATEGY=vector."
        )
    if k is None or store is None:
        settings = settings or get_settings()
        k = k if k is not None else settings.retrieval_k
        store = store if store is not None else build_filings_store(settings)

    started = time.perf_counter()
    hits = store.similarity_search_with_score(question, k=k)
    contexts = tuple(
        _as_context(document, distance, rank)
        for rank, (document, distance) in enumerate(hits, start=1)
    )
    log_event(
        logger,
        "retrieval",
        strategy=strategy.value,
        k=k,
        hits=len(contexts),
        latency_ms=round((time.perf_counter() - started) * 1000),
        # What was retrieved, not how good it was: enough for Phase 7 to reconstruct a
        # retrieval from the logs, and never the chunk text (these lines are kept).
        chunk_ids=[context.chunk_id for context in contexts],
        distances=[round(context.distance, 4) for context in contexts],
    )
    return contexts


def _as_context(document: Document, distance: float, rank: int) -> Context:
    """Rebuild a `Context` from a stored chunk's metadata and its indexed text.

    The body is recovered by dropping the provenance header `Chunk.text` prepends for
    BM25's benefit (ADR-0004). The header is machine furniture — repeating it inside a
    numbered context block, and again in the sources panel next to the same metadata, only
    invites the model to cite it as if the filer had written it.
    """
    metadata = document.metadata
    _, _, body = document.page_content.partition("\n\n")
    return Context(
        chunk_id=document.id or "",
        body=body or document.page_content,
        ticker=metadata["ticker"],
        filing_type=metadata["filing_type"],
        section=Section(metadata["section"]),
        fiscal_year=int(metadata["fiscal_year"]),
        accession=metadata["accession"],
        distance=float(distance),
        rank=rank,
    )
