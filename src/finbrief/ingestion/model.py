"""The vocabulary of an ingested Filing: its Sections, and what was extracted from it.

Plain data, no I/O. `ingestion/edgar.py` produces an `ExtractedFiling`, `ingestion/gate.py`
judges one, and `ingestion/chunking.py` turns one into chunks — so the gate can be tested
without EDGAR and the chunker without either (spec §Testing Decisions, seam 6).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

#: A word, for every rule in ingestion that counts them: two or more letters, so dot
#: leaders, page numbers and stray punctuation score zero.
#:
#: One definition because two readers depend on it agreeing. `gate` uses it for "body >>
#: heading" and `edgar` uses it to pick the wordiest fallback candidate — both to tell
#: real prose from a table-of-contents row. Were they to drift, the extractor could choose
#: a span by one measure that the gate then judges by another.
WORD = re.compile(r"[^\W\d_]{2,}")


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


#: Text that belongs to the *next* Item and can never belong to this one. Finding any of
#: it means extraction crossed a boundary, whatever the character count says.
#:
#: One definition, two readers, and they must not drift: `edgar.trim_at_next_item` cuts at
#: these markers and `gate` fails a Section that still contains one. A second copy would
#: let the repair and its judge disagree, which is the one way the repair could hide a
#: real boundary miss.
#:
#: Only Items 7 and 7A have entries, deliberately. Item 8 is the financial statements, and
#: the auditor's report opens it with language no MD&A contains — unambiguous. Item 1 ->
#: Item 1A and Item 1A -> Item 1B have no equivalent: a business section discusses its
#: risks in passing, so "Risk Factors" inside Item 1 is ordinary prose, not evidence. A
#: false accusation would be worse than the miss, because it trains a reader to wave the
#: gate through. For those two, `gate.MAX_SECTION_CHARS` is the only backstop — a stated
#: limitation (ADR-0007).
_ITEM_8_MARKERS = (
    "report of independent registered public accounting firm",
    "we have served as the",
    "critical audit matter",
)
NEXT_ITEM_MARKERS: Mapping[Section, tuple[str, ...]] = {
    Section.MDA: _ITEM_8_MARKERS,
    Section.MARKET_RISK: _ITEM_8_MARKERS,
}

#: Longest a text can be and still be *nothing but* a cross-reference. Deliberately well
#: above `gate.MIN_SECTION_CHARS`: J&J's Item 7A is 502 characters — a pointer plus a page
#: footer — so a threshold at the gate's floor would classify it as a real, if stubby,
#: Section and then fail it on body-vs-heading. The question this constant answers ("is
#: this only a pointer?") is not the question the floor answers ("is there enough text to
#: ground an answer?"), so they are separate numbers on purpose.
POINTER_MAX_CHARS = 1500

#: Phrases with which a filer hands an Item off to somewhere else. Drawn from the six
#: Universe filers that do it, whose wordings share no single formula: "Refer to" (JPM),
#: "See" (BAC), "are set forth in" (GS), "incorporated herein by reference" (JNJ),
#: "You can find" (LLY), "incorporated by reference" (PFE).
_REFERENCE_PHRASES = (
    "incorporated by reference",
    "incorporated herein by reference",
    "refer to",
    "set forth in",
    "you can find",
    "see ",
    "required by this item",
    "called for by this item",
)

#: Where the hand-off has to point for the content to still be in the knowledge base.
#: Item 7 is ingested, so a pointer into it loses nothing but the label; this is what
#: makes skipping the pointer acceptable rather than a silent hole.
_REFERENCE_TARGETS = ("item 7", "md&a", "management's discussion", "management’s discussion")


#: The only Section a filer may answer by reference. Scoped rather than applied to all
#: four, because `_REFERENCE_TARGETS` contains Item 7's own name: a truncated Item 7 whose
#: surviving fragment says "Management's discussion" could otherwise excuse itself as a
#: pointer and be dropped without a finding — a silent hole where the gate should have
#: raised. Item 7A is the case ADR-0007's amendment actually documents, and the six
#: Universe filers that do this all do it there.
_MAY_BE_INCORPORATED = frozenset({"Item 7A"})


def is_incorporated_by_reference(section: Section, text: str) -> bool:
    """Is this text only a pointer to another Item, rather than a Section itself?

    Six of the fifteen Universe companies — every bank and every healthcare name — answer
    Item 7A with a sentence directing the reader to Item 7 (ADR-0007 amendment). That is a
    lawful filing, not a broken parse, and it must not be ingested: "Refer to the Market
    Risk Management section on pages 133-142" would embed cleanly, retrieve for every
    market-risk question, and ground nothing.

    The test is a *positive* identification — short, plus hand-off language, plus a target
    that is itself ingested — and not merely "short texts are forgiven". A genuine
    extraction miss that happens to be brief has to keep failing loudly, or the gate stops
    being a gate. "Item 7A. Not applicable." matches nothing here and still fails.
    """
    if section.value not in _MAY_BE_INCORPORATED or len(text) > POINTER_MAX_CHARS:
        return False
    lowered = text.lower()
    return any(phrase in lowered for phrase in _REFERENCE_PHRASES) and any(
        target in lowered for target in _REFERENCE_TARGETS
    )


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

    `url` comes from EDGAR rather than being rebuilt from the accession number. The
    leading block of an accession is the *filer agent's* CIK, not the company's — JPM's
    10-K is accession `0001628280-…`, which is Donnelley's — so a URL composed from it
    points at the wrong company's directory, or nowhere.
    """

    ticker: str
    form: str
    accession: str
    fiscal_year: int
    filing_date: str
    url: str = ""


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
