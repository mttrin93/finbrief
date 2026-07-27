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

    @property
    def item(self) -> str:
        """The bare item label a heading pattern anchors on: `Item 1A` -> `1A`."""
        return self.value.removeprefix("Item ")


_HEADINGS: Mapping[Section, str] = {
    Section.BUSINESS: "Business",
    Section.RISK_FACTORS: "Risk Factors",
    Section.MDA: "Management's Discussion and Analysis of Financial Condition and "
    "Results of Operations",
    Section.MARKET_RISK: "Quantitative and Qualitative Disclosures About Market Risk",
}


#: How a Section's own title reads when a filer writes it as a heading, for the rare
#: filing that has no `Item N.` label to anchor on.
#:
#: Deliberately looser than `_HEADINGS`, which is the full formal title. JPM's MD&A heading
#: line is "Management's discussion and analysis" — the formal title continues "of
#: Financial Condition and Results of Operations" and JPM simply does not print it, so an
#: exact-title match finds nothing. These patterns match the part filers agree on.
#:
#: Only consulted when the Item label is absent entirely, which across the whole Universe
#: is JPM's Item 7 and nothing else. That ordering matters for `Business`, a word common
#: enough that matching it first would be reckless.
SECTION_TITLE_PATTERNS: Mapping[Section, str] = {
    Section.BUSINESS: r"Business",
    Section.RISK_FACTORS: r"Risk Factors",
    Section.MDA: r"Management[’']s Discussion and Analysis",
    Section.MARKET_RISK: r"Quantitative and Qualitative Disclosures About Market Risk",
}

#: Longest line that can still be a heading rather than a sentence mentioning one. Guards
#: the title fallback: "Risk Factors" inside a paragraph must not start a Section.
HEADING_LINE_MAX_CHARS = 120

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
#:
#: Item 7 -> Item 7A is the third gap, and it is not the same as the other two: the Item 7A
#: *heading* is unambiguous, but its content is market-risk prose that a bank's MD&A
#: discusses at length, so there is no content marker to key on. An MD&A that swallowed
#: Item 7A and stopped before Item 8 therefore passes this check, and the same prose would
#: be indexed twice — once under `Item 7`, once under `Item 7A`. Note that
#: `edgar._NEXT_ITEM` *does* bound the regex fallback's MD&A span at the Item 7A heading;
#: these markers judge the text after extraction, where no heading survives to key on.
#: Stated limitation, same as the other two.
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
    "see",
    "required by this item",
    "called for by this item",
)

#: The phrases as whole-word patterns. Substring matching made "see" fire inside
#: "oversee" and "foresee" — and for a recorded pointer filer that is a silent drop: a
#: truncated real Item 7A whose fragment reads "our committees oversee the exposures
#: described in Management's Discussion and Analysis" would be excused as a pointer and
#: leave the KB with no finding.
_REFERENCE_PHRASE_PATTERNS = tuple(
    re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE) for phrase in _REFERENCE_PHRASES
)

#: Where the hand-off has to point for the content to still be in the knowledge base.
#: Item 7 is ingested, so a pointer into it loses nothing but the label; this is what
#: makes skipping the pointer acceptable rather than a silent hole.
#:
#: Whole-token patterns, with `item_heading`'s negative lookahead, for `item_heading`'s
#: reason: `item 7` substring-matches `item 7a`, so "see Item 7A of our Annual Report for
#: the year ended December 31, 2024" — a pointer at a *prior year's* filing, which is not
#: in the knowledge base at all — used to satisfy "points at an ingested target". For the
#: six recorded pointer filers the gate has no second opinion, so that was a silent drop.
_REFERENCE_TARGET_PATTERNS = (
    re.compile(r"\bitem[ \t ]+7(?![A-Za-z0-9])", re.IGNORECASE),
    re.compile(r"\bmd&a\b", re.IGNORECASE),
    re.compile(r"management[’']s discussion", re.IGNORECASE),
)

#: Most words a pointer text may carry in sentences that do no handing-off. This is what
#: "*nothing but* a pointer" means, measured: of the six real pointers, the largest
#: leftover is GS's five-word running header ("THE GOLDMAN SACHS GROUP, INC. AND
#: SUBSIDIARIES"), and JNJ and PFE leave a two-to-three-word page footer. A single
#: sentence of genuine market-risk prose already runs past fifteen words, so a truncated
#: real Section cannot fit under this while a real pointer never comes near it.
POINTER_RESIDUE_MAX_WORDS = 15

#: `Item 7.` mid-sentence is a cross-reference, not a full stop — JNJ's pointer reads
#: "…incorporated herein by reference to Item 7. Management's discussion and analysis…"
#: and splitting there would cut the hand-off from the target it names. Masked out before
#: sentence-splitting.
_ITEM_LABEL_DOT = re.compile(r"(Item[ \t\u00a0]+\d{1,2}[AB]?)\.", re.I)

#: A sentence boundary: terminal punctuation, then whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def is_incorporated_by_reference(section: Section, text: str) -> bool:
    """Is this text *nothing but* a pointer to another Item, rather than a Section itself?

    Six of the fifteen Universe companies — every bank and every healthcare name — answer
    Item 7A with a sentence directing the reader to Item 7 (ADR-0007 amendment). That is a
    lawful filing, not a broken parse, and it must not be ingested: "Refer to the Market
    Risk Management section on pages 133-142" would embed cleanly, retrieve for every
    market-risk question, and ground nothing.

    The identification is positive and whole-section, in three cuts (issue #3 review):

    1. Only Item 7A, and only under `POINTER_MAX_CHARS` — as before.
    2. Some single sentence must both hand off (a `_REFERENCE_PHRASES` wording) and name
       an ingested target (`_REFERENCE_TARGET_PATTERNS`). Co-occurrence anywhere was
       the old rule, and a truncated genuine Section could satisfy it by accident — a
       "See below" three sentences away from a mention of MD&A is not a hand-off.
    3. The hand-off must be essentially the whole text: sentences carrying no hand-off
       wording — real prose, if any survives — may total `POINTER_RESIDUE_MAX_WORDS`
       words, enough for a running header or page footer and too little for content.

    "Item 7A. Not applicable." matches nothing here and still fails; so does a genuine
    Section truncated down to something pointer-sized. And the excusal never acts alone:
    `gate` cross-checks every firing against `config.ITEM_7A_POINTER_FILERS`, so a filer
    this function newly excuses is a finding for a human, not a silent drop.
    """
    # Market risk only, and scoped rather than applied to all four because
    # `_REFERENCE_TARGET_PATTERNS` contains Item 7's own name: a truncated Item 7 whose
    # surviving fragment says "Management's discussion" could otherwise excuse itself as a
    # pointer and be dropped without a finding — a silent hole where the gate should have
    # raised. Item 7A is the case ADR-0007's amendment documents, and the six Universe
    # filers that answer by reference all do it there.
    if section is not Section.MARKET_RISK or len(text) > POINTER_MAX_CHARS:
        return False
    body = _without_own_heading(text, section)
    sentences = _SENTENCE_END.split(_ITEM_LABEL_DOT.sub(r"\1", body))
    handoff = [s for s in sentences if any(p.search(s) for p in _REFERENCE_PHRASE_PATTERNS)]
    if not any(p.search(s) for s in handoff for p in _REFERENCE_TARGET_PATTERNS):
        return False
    residue = sum(len(WORD.findall(s)) for s in sentences if s not in handoff)
    return residue <= POINTER_RESIDUE_MAX_WORDS


def _without_own_heading(text: str, section: Section) -> str:
    """`text` minus the line holding the Section's own heading, if one is present.

    The heading line is furniture a pointer and a real Section share, so it must not
    count toward the residue that separates them.
    """
    start = section_start(text, section)
    if start == -1:
        return text
    end = text.find("\n", start)
    return text[:start] + (text[end + 1 :] if end != -1 else "")


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


def item_heading(item: str) -> re.Pattern[str]:
    """The pattern for an `Item N` heading at the start of a line.

    One definition, three readers, like `WORD` and `NEXT_ITEM_MARKERS` above:
    `section_start` anchors the gate's start check with it, and `edgar._extract_by_regex`
    opens *and closes* every fallback span with it. A second copy would let the fallback
    pick a span by one idea of "a heading" that the gate then judges by another.

    The negative lookahead is the load-bearing part: `Item 1` must not match `Item 1A.`
    or `Item 1B.`, and `Item 7` must not match `Item 7A.` — prefix-matching there is how
    an Item 1 lookup once returned the Risk Factors span, a mislabelled chunk no
    downstream check could catch.
    """
    return re.compile(
        rf"^[ \t]*Item[ \t\u00a0]+{re.escape(item.strip())}(?![A-Za-z0-9])\.?[ \t\u00a0]*",
        re.MULTILINE | re.IGNORECASE,
    )


def section_start(text: str, section: Section) -> int:
    """Offset of the Section's own heading in `text`, or -1 if it has none.

    Shared, and it has to be: `edgar.trim_to_section_start` moves the text to this offset
    and `gate` fails a Section whose offset is not zero. Two copies of "where does this
    Section begin?" would let the repair and its judge disagree about the answer — the
    same reason `NEXT_ITEM_MARKERS` and `WORD` live here.

    The Item label first, because it is unambiguous. The title only when there is no label
    at all: `Business` is too common a word to trust ahead of `Item 1.`, and across the
    whole Universe the fallback is reached exactly once, for JPM's Item 7.

    What that ordering costs, stated because `edgar.trim_to_section_start` acts on the
    answer: the label is taken wherever it appears, so a text that does *not* open with its
    own label — the case the title fallback exists for — anchors on any later line that
    begins `Item 7`, a cross-reference sentence included, and the trim then discards
    everything before it. The gate cannot see it, since it measures this same offset and the
    offset is 0 afterwards; the `section_trimmed` WARNING and ADR-0007's hand-verification
    excerpt are what catch it. Preferring whichever anchor comes *first* is not the fix —
    a running header carrying the Section's own title is precisely what eleven rows were
    trimmed off, and an earliest-anchor rule would re-admit every one of them.
    """
    label = item_heading(section.item).search(text)
    if label:
        return label.start()

    pattern = rf"^[ \t\u00a0]*{SECTION_TITLE_PATTERNS[section]}[^\n]*$"
    for match in re.finditer(pattern, text, re.MULTILINE | re.I):
        if len(match.group()) <= HEADING_LINE_MAX_CHARS:
            return match.start()
    return -1
