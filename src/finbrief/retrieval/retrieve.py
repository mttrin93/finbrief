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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from finbrief.config import RetrievalStrategy, Settings, get_settings
from finbrief.ingestion.model import Section
from finbrief.observability.logging_setup import log_event
from finbrief.retrieval.vectorstore import build_filings_store, nearest_chunks

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

    def as_payload(self) -> dict[str, str | int | float]:
        """This chunk as JSON-safe primitives — the form that crosses a serialised boundary.

        `search_filings` returns its chunks as a tool message's artifact, and the agent's
        checkpointer serialises every message it stores (ADR-0008). A frozen dataclass
        holding an enum survives that round trip only through an escape hatch LangGraph
        warns on and will remove; under its strict serialiser it comes back as an untyped
        dict instead, silently. So the wire form is written down here rather than left to a
        library's inference — and this class stays the one authority on the shape, since
        `from_payload` is what reads it back (ticket T4, #7).

        No cost in checkpoint size: the tool message's *content* already carries these
        bodies verbatim, because that is what the model reads.
        """
        return {
            "chunk_id": self.chunk_id,
            "body": self.body,
            "ticker": self.ticker,
            "filing_type": self.filing_type,
            "section": self.section.value,
            "fiscal_year": self.fiscal_year,
            "accession": self.accession,
            "distance": self.distance,
            "rank": self.rank,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Context:
        """Rebuild a chunk from `as_payload`, for the surface that displays it."""
        return cls(
            chunk_id=str(payload["chunk_id"]),
            body=str(payload["body"]),
            ticker=str(payload["ticker"]),
            filing_type=str(payload["filing_type"]),
            section=Section(payload["section"]),
            fiscal_year=int(payload["fiscal_year"]),
            accession=str(payload["accession"]),
            distance=float(payload["distance"]),
            rank=int(payload["rank"]),
        )


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
    # Normalised rather than trusted: `RetrievalStrategy("vector")` accepts the enum member
    # and the raw string alike, so a caller reading a strategy out of a config file lands on
    # the same object the dispatch below compares against — and an unknown value raises
    # `ValueError` here, naming the valid ones, instead of quietly taking the vector branch.
    strategy = RetrievalStrategy(strategy)
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
    hits = nearest_chunks(store, question, k)
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
