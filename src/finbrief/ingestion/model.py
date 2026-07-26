"""The vocabulary of an ingested Filing: its Sections, and what was extracted from it.

Plain data, no I/O. `ingestion/edgar.py` produces an `ExtractedFiling`, `ingestion/gate.py`
judges one, and `ingestion/chunking.py` turns one into chunks — so the gate can be tested
without EDGAR and the chunker without either (spec §Testing Decisions, seam 6).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class Section(StrEnum):
    """A curated part of a Filing that FinBrief ingests (CONTEXT.md, ADR-0007).

    The KB is these four Items and nothing else — not the full filing — so this enum is
    the grounding scope the UI declares and the golden set is authored against.

    The *value* is deliberately the EDGAR item label: it is simultaneously the key
    `edgartools` exposes the section under, the `section` metadata a chunk carries, and
    the literal a BM25 query for "Item 1A" matches (ADR-0004). One string, so a rename
    cannot desynchronise the extractor from the index.
    """

    BUSINESS = "Item 1"
    RISK_FACTORS = "Item 1A"
    MDA = "Item 7"
    MARKET_RISK = "Item 7A"

    @property
    def heading(self) -> str:
        """The section's title as it reads in a 10-K, for headings and UI labels."""
        return _HEADINGS[self]


_HEADINGS: Mapping[Section, str] = {
    Section.BUSINESS: "Business",
    Section.RISK_FACTORS: "Risk Factors",
    Section.MDA: "Management's Discussion and Analysis of Financial Condition and "
    "Results of Operations",
    Section.MARKET_RISK: "Quantitative and Qualitative Disclosures About Market Risk",
}


@dataclass(frozen=True, slots=True)
class FilingRef:
    """Identity and provenance of one Filing — the metadata every chunk inherits.

    `accession` is EDGAR's accession number, unique per filing and therefore the
    idempotency key: re-ingesting a filing already in the collection is a no-op
    (PLAN.md Phase 1, ADR-0007).

    `fiscal_year` is taken from the filing's period of report, not its filing date. A
    10-K filed in October 2025 for a fiscal year ending September 2025 is FY2025, and
    JPM's filed in February 2026 covers FY2025 — the golden set cites the fiscal year the
    text is *about* (ADR-0002).
    """

    ticker: str
    form: str
    accession: str
    fiscal_year: int
    filing_date: str


@dataclass(frozen=True, slots=True)
class ExtractedFiling:
    """One Filing's curated Sections, before the gate has judged them.

    `latest_annual_form` is the form type of the most recent annual report this company
    has on EDGAR, across the annual-report families (10-K, 20-F, 40-F). It is carried
    here rather than derived later because it is the gate's evidence that the company is
    a 10-K filer at all: a foreign private issuer files a 20-F, which has no Item
    1A/7/7A, so it can supply no Section under ADR-0007. Asking EDGAR is the real check
    that `config._TWENTY_F_FILERS` could only approximate (issue #3).
    """

    ref: FilingRef
    latest_annual_form: str
    sections: Mapping[Section, str]
