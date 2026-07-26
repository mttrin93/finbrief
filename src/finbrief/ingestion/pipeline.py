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

from finbrief.ingestion.chunking import chunk_filing
from finbrief.ingestion.gate import run_gate
from finbrief.ingestion.model import ExtractedFiling
from finbrief.observability.logging_setup import log_event
from finbrief.retrieval.vectorstore import (
    delete_accession,
    delete_superseded,
    ingested_accessions,
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

    Skipping is by accession (ADR-0007's idempotency key) and is a cost decision, not a
    correctness one: chunk ids are derived from the accession, so a re-write would upsert
    onto the same rows. What it saves is paying to embed them again. `force` re-embeds
    anyway, which is what a chunker or embedding-model change needs.

    A ticker's chunks from any *other* accession are evicted, skip or no skip: the KB is
    the latest 10-K per company (spec §Knowledge base), and a fiscal-year rollover gives
    the new filing a new accession — without the eviction the old year's chunks would sit
    beside the new ones, and retrieval would mix two fiscal years with no error.
    """
    filings = tuple(filings)
    run_gate(filings)

    already = set() if force else ingested_accessions(store)
    written: dict[str, int] = {}
    skipped: list[str] = []

    for filing in filings:
        evicted = delete_superseded(store, filing.ref.ticker, filing.ref.accession)
        if evicted:
            log_event(
                logger,
                "superseded_filing_removed",
                ticker=filing.ref.ticker,
                kept_accession=filing.ref.accession,
                chunks_removed=evicted,
            )
        if filing.ref.accession in already:
            skipped.append(filing.ref.ticker)
            log_event(
                logger,
                "filing_skipped",
                ticker=filing.ref.ticker,
                accession=filing.ref.accession,
                reason="already_ingested",
            )
            continue
        if force:
            # Ids embed a chunk index, so re-embedding with a changed chunker would
            # upsert over the low indices and orphan the high ones. Clear first.
            delete_accession(store, filing.ref.accession)
        count = write_chunks(store, chunk_filing(filing))
        written[filing.ref.ticker] = count
        log_event(
            logger,
            "filing_ingested",
            ticker=filing.ref.ticker,
            accession=filing.ref.accession,
            fiscal_year=filing.ref.fiscal_year,
            chunks=count,
        )

    return IngestReport(chunks_written=written, skipped=tuple(skipped))
