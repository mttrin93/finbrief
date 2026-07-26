"""The Phase-1 section-detection gate (ADR-0007, spec seam 6).

10-K HTML is heterogeneous and its failures are quiet: "Item 1A" matches in the table of
contents as well as at the real section, an extractor can run past a boundary and swallow
half the filing, and a filer can answer Item 7A with "Not applicable." None of that raises.
All of it produces chunks that embed cleanly, retrieve plausibly, and ground an answer in
nothing — and the first place it would surface is a RAGAs number nobody can explain.

So the gate runs *before* any retrieval exists, asserts the four properties per
company x Section, and fails loudly with every finding at once. It is a pure function of
already-extracted filings: no EDGAR, no network, no vector store.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from finbrief.ingestion.model import ExtractedFiling, Section

#: Shortest text that can be a Section rather than a pointer to one. "Not applicable." and
#: "The information required by this Item is incorporated by reference to..." are both
#: real filer answers and both unusable: the golden set is authored against ingested
#: Sections (ADR-0002), and neither contains an answer to author against.
MIN_SECTION_CHARS = 500

#: Longest text a single Section plausibly is. A boundary miss does not truncate — it runs
#: on to the end of the document, so the ceiling is what catches it. Set well above the
#: real maximum (a large bank's Item 1A runs past 200k characters) because this rule exists
#: to catch a runaway, not to police a verbose filer.
MAX_SECTION_CHARS = 400_000

#: How many times more prose the body must carry than the heading that introduces it.
#: This is ADR-0007's "body >> heading", measured in words rather than characters on
#: purpose: a table-of-contents hit is a heading, a run of dot leaders, and a page number,
#: so it is *long* but nearly wordless. Characters would call it a section; words do not.
BODY_TO_HEADING_WORD_RATIO = 20

#: A word for the ratio above: two or more letters, so dot leaders, page numbers and
#: stray punctuation count for nothing.
_WORD = re.compile(r"[^\W\d_]{2,}")

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

    if not MIN_SECTION_CHARS <= len(text) <= MAX_SECTION_CHARS:
        return finding(
            "section_length_bounded",
            f"{len(text)} characters, outside the plausible range "
            f"{MIN_SECTION_CHARS}-{MAX_SECTION_CHARS}",
        )

    heading, _, body = text.partition("\n")
    heading_words = len(_WORD.findall(heading))
    body_words = len(_WORD.findall(body))
    if body_words < heading_words * BODY_TO_HEADING_WORD_RATIO:
        return finding(
            "section_body_exceeds_heading",
            f"{body_words} words of body under a {heading_words}-word heading — "
            f"expected at least {heading_words * BODY_TO_HEADING_WORD_RATIO}. "
            f"This is what a table-of-contents hit looks like.",
        )

    return None
