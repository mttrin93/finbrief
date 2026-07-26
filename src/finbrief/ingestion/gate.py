"""The Phase-1 section-detection gate (ADR-0007, spec seam 6).

10-K HTML is heterogeneous and its failures are quiet: "Item 1A" matches in the table of
contents as well as at the real section, an extractor can run past a boundary and swallow
half the filing, and a filer can answer Item 7A with "Not applicable." None of that raises.
All of it produces chunks that embed cleanly, retrieve plausibly, and ground an answer in
nothing — and the first place it would surface is a RAGAs number nobody can explain.

So the gate runs *before* any retrieval exists, asserts the four properties per
company x Section, and fails loudly with every finding at once. It is a pure function of
already-extracted filings: no EDGAR, no network, no vector store.

The thresholds below live here rather than in `config.py`, which owns "every knob". They
are not knobs: they are assertions about what a 10-K Section looks like, tuned against
fifteen real filings and recorded in ADR-0007's amendment with the evidence for each. A
knob invites an operator to widen one until the gate stops complaining, which is the exact
failure this module exists to prevent — the same reasoning that keeps `CHUNK_OVERLAP_CHARS`
in the chunker (PLAN.md). `CHUNK_SIZE_CHARS` is a knob and stays in `config.py`, because
the index on disk and the embedding window both depend on it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from finbrief.ingestion.model import (
    NEXT_ITEM_MARKERS,
    WORD,
    ExtractedFiling,
    Section,
    is_incorporated_by_reference,
    section_start,
)

#: Shortest text that can be a Section rather than a pointer to one. "Not applicable." and
#: "The information required by this Item is incorporated by reference to..." are both
#: real filer answers and both unusable: the golden set is authored against ingested
#: Sections (ADR-0002), and neither contains an answer to author against.
MIN_SECTION_CHARS = 500

#: Longest text a single Section plausibly is — a sanity backstop, not the runaway
#: detector.
#:
#: It was 400,000 and that was very nearly a lie. JPM's FY2025 Item 7 came back at 413,149
#: characters and did overrun, but only by 12,837: its *legitimate* MD&A is 400,312
#: characters, so the ceiling fired 312 characters away from failing correct content.
#: BAC (293,458) and GS (309,157) are clean at that size. A character count cannot
#: separate "the parser ran into Item 8" from "this is a bank", so
#: `section_stops_before_the_next_item` below does that job on content, and this is raised
#: to 500,000 to stop it pretending to (ADR-0007 amendment, issue #3).
MAX_SECTION_CHARS = 500_000

#: How many times more prose the body must carry than the heading that introduces it.
#: This is ADR-0007's "body >> heading", measured in words rather than characters on
#: purpose: a table-of-contents hit is a heading, a run of dot leaders, and a page number,
#: so it is *long* but nearly wordless. Characters would call it a section; words do not.
BODY_TO_HEADING_WORD_RATIO = 20

#: The annual-report form families EDGAR filers use. Only the 10-K family has the Items
#: ADR-0007 scopes the KB to.
_TEN_K_FAMILY = "10-K"


@dataclass(frozen=True, slots=True)
class GateFinding:
    """One failed assertion, named so a human can act on it without reading the code."""

    ticker: str
    section: Section | None
    check: str
    detail: str

    def __str__(self) -> str:
        where = f"{self.ticker} {self.section.value}" if self.section else self.ticker
        return f"{where}: {self.check} — {self.detail}"


class SectionGateError(RuntimeError):
    """Raised when any company x Section fails. Carries every finding, not the first."""

    def __init__(self, findings: Sequence[GateFinding]) -> None:
        self.findings = tuple(findings)
        listed = "\n".join(f"  - {finding}" for finding in self.findings)
        super().__init__(
            f"Section-detection gate failed with {len(self.findings)} finding(s) "
            f"(ADR-0007). The knowledge base was not written.\n{listed}"
        )


def form_family(form: str) -> str:
    """`10-K/A` -> `10-K`. EDGAR form search prefix-matches, so amendments come back too.

    An amendment is still the same kind of annual report, and the gate's question is which
    kind the company files — so the suffix is noise here.

    This stays correct in the awkward case: a company's most recent annual filing can be a
    `10-K/A` that amends an *older* fiscal year, which is what Tesla's April Part III
    amendments are. Normalising to the family answers "is this a 10-K filer?" with a
    correct yes, and it is the only question this check asks. Which document to ingest is
    a separate decision made in `edgar.fetch_filing`, where the exact-form filter lives —
    conflating the two would either reject a legitimate 10-K filer or ingest a Part III
    amendment that has no Item 1/1A/7/7A in it.
    """
    return form.split("/", 1)[0].strip().upper()


def check_filing(filing: ExtractedFiling) -> tuple[GateFinding, ...]:
    """Every gate finding for one filing, in reporting order. Empty means it passed."""
    findings: list[GateFinding] = []

    if form_family(filing.latest_annual_form) != _TEN_K_FAMILY:
        findings.append(
            GateFinding(
                ticker=filing.ref.ticker,
                section=None,
                check="latest_annual_filing_is_a_10k",
                detail=(
                    f"EDGAR's most recent annual filing for {filing.ref.ticker} is a "
                    f"{filing.latest_annual_form}, not a 10-K. A foreign private issuer "
                    f"has no Item 1A/7/7A to ingest, so it cannot be in the Universe "
                    f"(ADR-0007). Remove it from config.UNIVERSE."
                ),
            )
        )

    for section in Section:
        finding = _check_section(filing, section)
        if finding is not None:
            findings.append(finding)

    return tuple(findings)


def incorporated_sections(filing: ExtractedFiling) -> frozenset[Section]:
    """The Sections this filer answered with a pointer instead of text.

    They pass the gate and are still not ingested — `chunking.chunk_filing` skips them.
    Reported rather than silently dropped: the hand-verification artifact and the gate
    table both name them, because "JPM has no Item 7A chunks" is a fact a reader of the
    knowledge base needs, and the golden set must not ask an Item 7A question of a
    company that has none (ADR-0002).
    """
    return frozenset(
        section
        for section, text in filing.sections.items()
        if is_incorporated_by_reference(section, text)
    )


def run_gate(filings: Iterable[ExtractedFiling]) -> None:
    """Raise `SectionGateError` unless every company x Section passes. Otherwise silent.

    Collects across all filings before raising: discovering the second broken filer by
    re-running a fifteen-company ingest is how a gate stops being run at all.
    """
    findings = [finding for filing in filings for finding in check_filing(filing)]
    if findings:
        raise SectionGateError(findings)


def _check_section(filing: ExtractedFiling, section: Section) -> GateFinding | None:
    """The first rule `section` fails in this filing, or `None`.

    One finding per Section rather than all of them: the rules are a sieve, not a
    checklist — an absent Section is not also "too short", and reporting it twice buries
    the other fourteen companies' findings.
    """

    def finding(check: str, detail: str) -> GateFinding:
        return GateFinding(filing.ref.ticker, section, check, detail)

    text = filing.sections.get(section)
    if text is None:
        return finding(
            "section_found",
            f"no {section.value} ({section.heading}) was extracted from {filing.ref.accession}",
        )

    if not text.strip():
        return finding("section_non_empty", "extracted, but the text is blank")

    # Before the size and shape rules, because a pointer is short and heading-heavy and
    # would otherwise fail one of them for the wrong reason. It is not a defect and not
    # ingested either — see `incorporated_sections`.
    if is_incorporated_by_reference(section, text):
        return None

    if not MIN_SECTION_CHARS <= len(text) <= MAX_SECTION_CHARS:
        return finding(
            "section_length_bounded",
            f"{len(text)} characters, outside the plausible range "
            f"{MIN_SECTION_CHARS}-{MAX_SECTION_CHARS}",
        )

    heading, _, body = text.partition("\n")
    heading_words = len(WORD.findall(heading))
    body_words = len(WORD.findall(body))
    if body_words < heading_words * BODY_TO_HEADING_WORD_RATIO:
        return finding(
            "section_body_exceeds_heading",
            f"{body_words} words of body under a {heading_words}-word heading — "
            f"expected at least {heading_words * BODY_TO_HEADING_WORD_RATIO}. "
            f"This is what a table-of-contents hit looks like.",
        )

    # The rule hand-verification taught us. JPM's Item 7 began three pages early, on the
    # annual report's financial-highlights front matter, and satisfied every rule above:
    # it was found, non-empty, bounded, wordy, and free of Item 8. Only a person reading
    # it against the filing caught that it started in the wrong place — so that judgement
    # is now a check, and the next one of these fails here instead of in an artifact
    # review (ADR-0007 amendment, issue #3).
    if section_start(text, section) != 0:
        return finding(
            "section_starts_at_its_heading",
            f"the text does not begin at {section.value} or its title — extraction "
            f"started somewhere else. Opens with: {text[:90]!r}",
        )

    lowered = text.lower()
    for marker in NEXT_ITEM_MARKERS.get(section, ()):
        position = lowered.find(marker)
        if position != -1:
            return finding(
                "section_stops_before_the_next_item",
                f"contains {marker!r} at character {position:,} of {len(text):,} — that "
                f"is Item 8 (financial statements and the auditor's report), so "
                f"extraction crossed the {section.value}/Item 8 boundary. Not a length "
                f"problem: the text before it may be a perfectly good {section.value}.",
            )

    return None
