"""Fetch -> gate -> chunk -> store, in that order and no other.

The ordering is the point. Every filing is fetched and gated *before* a single chunk is
written, so a knowledge base is never half-built from a run that was going to fail anyway
— ADR-0007 wants the gate to fail "before any retrieval numbers exist", and a partial
index is exactly the thing that produces unexplainable ones.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from finbrief.ingestion.chunking import chunk_filing, content_hash
from finbrief.ingestion.gate import run_gate
from finbrief.ingestion.model import ExtractedFiling
from finbrief.observability.logging_setup import log_event
from finbrief.retrieval.vectorstore import (
    content_hashes_by_accession,
    delete_orphaned_chunks,
    delete_superseded,
    write_chunks,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestReport:
    """What one ingestion run fetched, wrote, and skipped."""

    chunks_written: Mapping[str, int]
    skipped: tuple[str, ...]

    @property
    def total_chunks(self) -> int:
        return sum(self.chunks_written.values())


def ingest(filings: Iterable[ExtractedFiling], *, store, force: bool = False) -> IngestReport:
    """Gate every filing, then write the ones the collection does not already hold.

    Raises `SectionGateError` — writing nothing — if any company x Section fails.

    A filing is skipped only when the collection holds *this* run's version of it: the
    accession matches (ADR-0007's idempotency key) **and** so does `content_hash`. The
    accession alone is not enough, and the first extractor fix proved it — commit `a0614de`
    moved eleven Sections without a single accession changing, so an accession-only skip
    reported `skipped (already ingested)` over chunks built from pre-repair text while the
    verification artifact attested to the repaired text (issue #3 review). Skipping still
    saves the embedding spend in the common case, which is the whole point; it just no
    longer trades correctness for it. `force` re-embeds regardless — what a change to the
    *embedding model* needs, since that one leaves no trace in the text.

    A ticker's chunks from any *other* accession are evicted, skip or no skip: the KB is
    the latest 10-K per company (spec §Knowledge base), and a fiscal-year rollover gives
    the new filing a new accession — without the eviction the old year's chunks would sit
    beside the new ones, and retrieval would mix two fiscal years with no error.
    """
    filings = tuple(filings)
    run_gate(filings)

    already: dict[str, set[str]] = {} if force else content_hashes_by_accession(store)
    written: dict[str, int] = {}
    skipped: list[str] = []

    for filing in filings:
        fingerprint = content_hash(filing)
        held = already.get(filing.ref.accession, set())
        if held == {fingerprint}:
            skipped.append(filing.ref.ticker)
            log_event(
                logger,
                "filing_skipped",
                ticker=filing.ref.ticker,
                accession=filing.ref.accession,
                reason="already_ingested",
            )
        else:
            if held:
                # Same filing, different text: the extractor changed under a document
                # EDGAR never re-published. Loud, because it means the KB was stale until
                # this run and every number measured against it was measured on old text.
                log_event(
                    logger,
                    "filing_extraction_changed",
                    level=logging.WARNING,
                    ticker=filing.ref.ticker,
                    accession=filing.ref.accession,
                    held_content_hashes=sorted(held),
                    extracted_content_hash=fingerprint,
                )
            chunks = chunk_filing(filing)
            count = write_chunks(store, chunks)
            # Ids embed a chunk index, so text that now splits into fewer chunks upserts
            # over the low indices and strands the high ones. Prune after the write, never
            # before: see `delete_orphaned_chunks`.
            orphans = delete_orphaned_chunks(
                store, filing.ref.accession, {chunk.id for chunk in chunks}
            )
            if orphans:
                log_event(
                    logger,
                    "orphaned_chunks_removed",
                    ticker=filing.ref.ticker,
                    accession=filing.ref.accession,
                    chunks_removed=orphans,
                )
            written[filing.ref.ticker] = count
            log_event(
                logger,
                "filing_ingested",
                ticker=filing.ref.ticker,
                accession=filing.ref.accession,
                fiscal_year=filing.ref.fiscal_year,
                chunks=count,
            )

        # Eviction *after* the write, and the order is load-bearing: `write_chunks` is
        # where the paid embedding call happens, so an API error on company 8 of 15 with
        # the eviction first would leave that ticker holding neither its old accession's
        # chunks nor its new ones — a company silently absent from the KB. Deleting last
        # means the worst interleaved state is two fiscal years present, which the next
        # run repairs.
        evicted = delete_superseded(store, filing.ref.ticker, filing.ref.accession)
        if evicted:
            log_event(
                logger,
                "superseded_filing_removed",
                ticker=filing.ref.ticker,
                kept_accession=filing.ref.accession,
                chunks_removed=evicted,
            )

    return IngestReport(chunks_written=written, skipped=tuple(skipped))
