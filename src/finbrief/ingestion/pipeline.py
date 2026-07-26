"""Fetch -> gate -> chunk -> store, in that order and no other.

The ordering is the point. Every filing is fetched and gated *before* a single chunk is
written, so a knowledge base is never half-built from a run that was going to fail anyway
— ADR-0007 wants the gate to fail "before any retrieval numbers exist", and a partial
index is exactly the thing that produces unexplainable ones.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from finbrief.ingestion.chunking import chunk_filing
from finbrief.ingestion.edgar import fetch_filing
from finbrief.ingestion.gate import run_gate
from finbrief.ingestion.model import ExtractedFiling
from finbrief.observability.logging_setup import log_event
from finbrief.retrieval.vectorstore import ingested_accessions, write_chunks

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestReport:
    """What one ingestion run fetched, wrote, and skipped."""

    filings: tuple[ExtractedFiling, ...]
    chunks_written: Mapping[str, int]
    skipped: tuple[str, ...]

    @property
    def total_chunks(self) -> int:
        return sum(self.chunks_written.values())


def fetch_filings(
    tickers: Sequence[str], *, fetch: Callable[[str], ExtractedFiling] = fetch_filing
) -> tuple[ExtractedFiling, ...]:
    """Download and extract every ticker. `fetch` is the seam tests replace."""
    return tuple(fetch(ticker) for ticker in tickers)


def ingest(filings: Iterable[ExtractedFiling], *, store, force: bool = False) -> IngestReport:
    """Gate every filing, then write the ones the collection does not already hold.

    Raises `SectionGateError` — writing nothing — if any company x Section fails.

    Skipping is by accession (ADR-0007's idempotency key) and is a cost decision, not a
    correctness one: chunk ids are derived from the accession, so a re-write would upsert
    onto the same rows. What it saves is paying to embed them again. `force` re-embeds
    anyway, which is what a chunker or embedding-model change needs.
    """
    filings = tuple(filings)
    run_gate(filings)

    already = set() if force else ingested_accessions(store)
    written: dict[str, int] = {}
    skipped: list[str] = []

    for filing in filings:
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

    return IngestReport(filings=filings, chunks_written=written, skipped=tuple(skipped))
