"""The EDGAR boundary: fetch one company's latest 10-K and cut it into curated Sections.

This is the only module that talks to EDGAR, and the only impure part of ingestion. It
returns an `ExtractedFiling` — plain data — so the gate and the chunker downstream are
testable without a network (spec §Testing Decisions; tests use recorded fixtures).

Extraction is structure-anchored via `edgartools`, per ADR-0007: no hand-rolled primary
parser. `_extract_by_regex` is the *documented fallback* that ADR names, used only for a
Section `edgartools` did not return — and bounded on both ends, so a boundary miss cannot
run on to the end of the document. Whatever it produces still faces the gate.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping

from finbrief.config import (
    ConfigError,
    load_env,
    resolve_sec_edgar_user_agent,
)
from finbrief.ingestion.model import (
    NEXT_ITEM_MARKERS,
    WORD,
    ExtractedFiling,
    FilingRef,
    Section,
    item_heading,
    section_start,
)
from finbrief.observability.logging_setup import log_event

logger = logging.getLogger(__name__)

#: The annual-report forms EDGAR filers use. Searched together so "the company's most
#: recent annual filing" is answerable from metadata alone — the gate's 10-K check
#: (issue #3). EDGAR prefix-matches these, so amendments (`10-K/A`) come back too.
ANNUAL_FORMS = ("10-K", "20-F", "40-F")

#: Where each Section ends: the Item that follows it in a 10-K's fixed running order.
#: This is what "bounded" means in ADR-0007's "bounded regex fallback" — the fallback can
#: only ever claim the span between one Item heading and the next.
_NEXT_ITEM: Mapping[Section, str] = {
    Section.BUSINESS: r"1A",
    Section.RISK_FACTORS: r"1B",
    Section.MDA: r"7A",
    Section.MARKET_RISK: r"8",
}

#: How a table-of-contents row ends: dot leaders, a page number, or a page range
#: ("Risk Factors. 9-31" is JPM's). A real heading is followed by a line break and prose,
#: never by its own page number.
#:
#: Matched against what follows the Item label, never the whole line: the `\s` alternative
#: is loose enough to match the label's *own* digits, so a filer whose heading line is a
#: bare `Item 7` with the title on the next line had its real heading read as a TOC row and
#: the fallback returned nothing at all for it (issue #3 review).
_TOC_TAIL = re.compile(r"(\.{2,}|\s)\s*\d{1,4}(\s*[-–]\s*\d{1,4})?\s*$")


def configure_edgar() -> None:
    """Give edgartools the SEC-required contact identity, failing loudly without one.

    The SEC rejects unidentified traffic, and `edgartools` would otherwise raise deep in a
    request with a stack trace that says nothing about `.env`. Ingestion is a script a
    human runs, so it can afford to say what is wrong in one line.

    Reads `SEC_EDGAR_USER_AGENT` without building the full `Settings`, which hard-requires
    `OPENROUTER_API_KEY`. A `--dry-run` fetches and gates but embeds nothing, so demanding
    the paid key would fail on exactly the machine a dry run is for — the same reasoning
    that keeps `config.resolve_log_level` Settings-independent. There is deliberately no
    `settings` override: a second way in is a second answer to "which identity", and both
    callers (`scripts/ingest_filings.py`, `scripts/record_edgar_fixtures.py`) want the
    Settings-free one.
    """
    load_env()
    user_agent = resolve_sec_edgar_user_agent(os.environ)
    if not user_agent:
        raise ConfigError(
            "SEC_EDGAR_USER_AGENT is not set, and the SEC requires a contact identity "
            "on every request. Set it to 'FinBrief your-email@example.com' in .env "
            "(see .env.example)."
        )
    from edgar import set_identity

    set_identity(user_agent)


def fetch_filing(ticker: str) -> ExtractedFiling:
    """Download `ticker`'s latest 10-K and extract its four curated Sections.

    Does no judging: a Section that came back empty, short, or wrong is returned as it is
    and the gate decides. Extraction and verdict stay separate so the gate's report says
    what was extracted rather than what survived a filter.
    """
    from edgar import Company

    company = Company(ticker)
    annual = company.get_filings(form=list(ANNUAL_FORMS)).latest(1)
    if annual is None:
        raise LookupError(f"{ticker} has no annual filing on EDGAR at all.")

    # Not `annual`, and not `get_filings(form="10-K").latest(1)` either. Two distinct
    # traps, both from EDGAR's prefix matching:
    #
    # 1. A 20-F filer must still produce an `ExtractedFiling`, so the gate can report
    #    *which* company is wrong rather than the run dying with a traceback. Only a
    #    company with no 10-K in its entire history has nothing to hand the gate.
    # 2. A `10-K` search returns `10-K/A` amendments too, and `.latest(1)` will happily
    #    hand one back. Tesla files a Part III amendment every April, so as of this
    #    writing its most recent 10-K-search hit is a 10-K/A filed 2026-04-30 — a
    #    document with no Item 1/1A/7/7A in it at all. The gate would catch the empty
    #    result, but it would report "TSLA has no Item 1" when the truth is "we fetched
    #    the wrong document", so the exact-form filter belongs here.
    #
    # The gate's separate form-family check is unaffected and deliberately so: there,
    # `10-K/A` *is* a 10-K, because the question is what kind of filer this is. It stays
    # right even when the most recent annual filing is an amendment to an older fiscal
    # year — that company is still a 10-K filer, and `filing` below is still its real
    # latest annual report.
    filing = latest_10k(company)
    if filing is None:
        # A foreign private issuer has no 10-K to return, and raising here would be the
        # wrong shape of loud: the issue-#3 comment asks the *gate* to catch this, and a
        # `LookupError` from the fetch loop aborts the run at the first bad ticker,
        # reporting one company where `run_gate` deliberately reports all of them. So the
        # fetch returns the evidence and lets the judge speak.
        log_event(
            logger,
            "no_10k_on_file",
            level=logging.ERROR,
            ticker=ticker,
            latest_annual_form=str(annual.form),
        )
        return ExtractedFiling(
            ref=FilingRef(
                ticker=ticker,
                form=str(annual.form),
                accession=str(annual.accession_no),
                fiscal_year=_fiscal_year(annual),
                filing_date=str(annual.filing_date),
                url=str(annual.homepage_url),
            ),
            latest_annual_form=str(annual.form),
            sections={},
        )

    ref = FilingRef(
        ticker=ticker,
        form=str(filing.form),
        accession=str(filing.accession_no),
        fiscal_year=_fiscal_year(filing),
        filing_date=str(filing.filing_date),
        url=str(filing.homepage_url),
    )
    sections = _extract_sections(filing, ticker)

    log_event(
        logger,
        "filing_extracted",
        ticker=ticker,
        accession=ref.accession,
        fiscal_year=ref.fiscal_year,
        latest_annual_form=str(annual.form),
        section_chars={s.value: len(t) for s, t in sections.items()},
    )
    return ExtractedFiling(ref=ref, latest_annual_form=str(annual.form), sections=sections)


def latest_10k(company):
    """The company's most recent filing whose form is *exactly* `10-K`, or `None`.

    Public because `scripts/record_edgar_fixtures.py` needs the same document
    `fetch_filing` works on in order to record a Section *before* the repairs run — and
    two answers to "which filing is this company's latest 10-K?" is how a fixture stops
    describing the filing the code actually reads.

    The exact-form comparison is the load-bearing part; see `fetch_filing`'s trap 2.
    """
    exact = [f for f in company.get_filings(form="10-K") if str(f.form) == "10-K"]
    return max(exact, key=lambda f: f.filing_date, default=None)


def _fiscal_year(filing) -> int:
    """The year the filing is *about*, from its period of report.

    Not the filing date: Apple's FY2025 10-K is filed in October 2025 and JPMorgan's in
    February 2026, and the golden set cites the fiscal year of the text (ADR-0002).
    """
    period = getattr(filing, "period_of_report", None)
    if period is None:
        raise LookupError(f"{filing.accession_no} has no period of report to date it by.")
    return int(str(period)[:4])


def _extract_sections(filing, ticker: str) -> dict[Section, str]:
    """The four curated Sections, structure-anchored first and regex-bounded second.

    `ticker` is passed in rather than read off the filing: `edgartools` exposes
    `Filing.company` as the company *name* ("Apple Inc."), and a `ticker` field whose value
    is sometimes a name cannot be joined to the rest of the run's JSON lines.
    """
    report = filing.obj()
    full_text: str | None = None
    sections: dict[Section, str] = {}

    for section in Section:
        text = _clean(_from_edgartools(report, section))
        if not text:
            if full_text is None:
                full_text = _clean(filing.text())
            text = _clean(_extract_by_regex(full_text, section))
            if text:
                log_event(
                    logger,
                    "section_regex_fallback",
                    level=logging.WARNING,
                    ticker=ticker,
                    accession=str(filing.accession_no),
                    section=section.value,
                    chars=len(text),
                )
        if text:
            # Both ends, front first: the start trim can only remove a prefix and the end
            # trim only a suffix, so neither can undo the other.
            for trim, reason in (
                (trim_to_section_start, "began before its own heading"),
                (trim_at_next_item, "ran into the next Item"),
            ):
                trimmed = trim(text, section)
                if len(trimmed) != len(text):
                    log_event(
                        logger,
                        "section_trimmed",
                        level=logging.WARNING,
                        ticker=ticker,
                        accession=str(filing.accession_no),
                        section=section.value,
                        chars_before=len(text),
                        chars_after=len(trimmed),
                        reason=reason,
                    )
                text = trimmed
            sections[section] = text

    return sections


def trim_to_section_start(text: str, section: Section) -> str:
    """Drop anything before the Section's own heading. The partner to `trim_at_next_item`.

    Hand-verification of the fifteen filings found what no automated check was looking
    for: JPM's Item 7 began three pages early, at the annual report's Three-Year Summary
    of Consolidated Financial Highlights on p.43, where the filing's own cross-reference
    puts Item 7 at pp.46-160. The section was long, wordy, bounded and free of Item 8
    content — every rule the gate had — and still started in the wrong place.

    Ten other Sections began a line or two early, on a "Table of Contents" or company-name
    running header. Same defect, three orders of magnitude smaller, and the same fix.

    A prefix cut, like every repair here: the filer's own words, never rewritten.

    What the gate catches afterwards, precisely: this cuts *to* `section_start`, which is
    the same offset `section_starts_at_its_heading` measures, so the check can only fail
    when there was no anchor to cut to at all — a total miss, which then fails loudly
    instead of passing as an untrimmed Section. It cannot catch a trim that anchored on
    the *wrong* heading; distinguishing the real heading from a plausible one is the
    judgement no automated check makes, which is why ADR-0007 requires the
    hand-verification artifact. A re-render flags any row whose text moved as `CHANGED`
    and unticks it, so a trim that changes what a human verified costs that human one row
    (ADR-0007 amendment, issue #3).
    """
    start = section_start(text, section)
    return text[start:] if start > 0 else text


def trim_at_next_item(text: str, section: Section) -> str:
    """Cut a Section at the first marker belonging to the Item that follows it.

    The second documented repair in this module, and the same species as the regex
    fallback ADR-0007 already sanctions: bounded, one rule, and never silent — every trim
    logs a `section_trimmed` event.

    Two real filings need it. `edgartools` returned JPM's FY2025 Item 7 as 413,149
    characters, the last 12,837 of which are Item 8; and GM's Item 7A as 26,450, of which
    only the first 11,790 are market risk. Both overran the Item 7/Item 8 boundary and
    both have an intact prefix, so discarding the whole Section would throw away a bank's
    entire MD&A over 3% contamination.

    What keeps this from being the extractor grading its own homework, stated honestly:
    the result still faces `gate.check_filing`'s marker check, and if this function found
    no marker to cut at, the Section fails there exactly as it did before this function
    existed. What that ordering does *not* buy is detection of a cut in the wrong place —
    the cut is at the first marker, so the survivor provably contains none, and
    `section_stops_before_the_next_item` cannot fire on a text this has already trimmed.
    An over-eager marker (`"critical audit matter"` is the one an MD&A could plausibly
    use) therefore truncates silently as far as the gate is concerned. The trim logs, and
    the hand-verification artifact flags the row `CHANGED` on re-render; those, not the
    gate, are what catch it.

    Always a prefix of the input — never a rewrite, so BM25 still indexes the filer's own
    words (ADR-0004).
    """
    lowered = text.lower()
    cut = min(
        (
            position
            for marker in NEXT_ITEM_MARKERS.get(section, ())
            if (position := lowered.find(marker)) != -1
        ),
        default=-1,
    )
    return text[:cut].rstrip() if cut != -1 else text


def _from_edgartools(report, section: Section) -> str:
    """`report["Item 1A"]`, or `""` if this filer's structure did not yield it.

    Broad `except` on purpose: `edgartools` signals a section it could not resolve in
    more than one way across filer layouts, and every one of them means the same thing
    here — try the fallback. A real bug still surfaces, one step later and louder, as a
    gate finding naming the company and Section.
    """
    try:
        text = report[section.value]
    except Exception:  # noqa: BLE001 - see docstring
        return ""
    return text or ""


def _extract_by_regex(full_text: str, section: Section) -> str:
    """The bounded regex fallback of ADR-0007: the span from one Item heading to the next.

    Every candidate span is scored by word count and the wordiest wins. That is what
    discriminates the real section from the table-of-contents entry with the same heading
    — the TOC row is dot leaders and a page number, so it scores near zero. Returning the
    *last* match instead is the usual shortcut and it is wrong for filers who repeat the
    heading in a part-summary or an index at the back.

    "Bounded" means bounded on *both* ends, and both ends needed defending:

    - The heading pattern — `model.item_heading`, one definition shared with the gate's
      start check — ends in a negative lookahead, because `Item 1` otherwise
      prefix-matches `Item 1A.` and `Item 1B.`, and `Item 7` matches `Item 7A.`. Combined
      with wordiest-wins, an Item 1 fallback would have cheerfully returned the Risk
      Factors span — a mislabelled chunk, which is the single failure this module exists
      to prevent.
    - A candidate with no following Item heading is no candidate at all. The next Item's
      heading always sits behind a real Section — none of the four is a 10-K's last — so
      an unterminated span is a boundary miss by definition. It used to be capped at the
      gate's ceiling on the theory that it would fail there anyway; that was false twice
      over (`_clean` shrinks the text *after* the cap, and Items 1/1A have no
      `NEXT_ITEM_MARKERS`), so a capped span could pass the whole gate carrying the rest
      of the document under one Item's label. Skipping it means the worst outcome is a
      *missing* Section — a loud `section_found` finding naming the company — never a
      mislabelled chunk.
    """
    start = item_heading(section.item)
    end = item_heading(_NEXT_ITEM[section])

    best = ""
    best_words = 0
    for match in start.finditer(full_text):
        if _is_toc_line(full_text, match.end()):
            continue
        stop = end.search(full_text, match.start() + 1)
        if stop is None:
            continue
        candidate = full_text[match.start() : stop.start()]
        words = len(WORD.findall(candidate))
        if words > best_words:
            best, best_words = candidate, words
    return best


def _is_toc_line(full_text: str, heading_end: int) -> bool:
    """Does the heading ending at `heading_end` sit on a table-of-contents row?

    Wordiest-candidate-wins cannot settle this on its own, and the reason is worth stating
    because it is not obvious: the candidates *nest*. A span anchored at the TOC row runs
    to the next Item heading, which is past the real section — so it contains the real
    section plus the table of contents, and is therefore always the wordier of the two.
    Left to the word count alone, the fallback reliably picked the polluted span.

    A TOC row is recognised by what follows the heading on its own line: dot leaders, a
    page number, or both. A real section heading is followed by a line break and prose.

    `heading_end`, not the start of the line, and that distinction is the whole of a bug:
    `Item 7` *ends in a digit*, so testing the whole line let the label's own number stand
    in for a page number, and any filer whose heading line is a bare `Item 7` with the
    title on the following line had its real heading discarded as a TOC row — silently
    defeating the fallback for exactly the ragged filers it exists for (issue #3 review).
    The prepended space stands in for the whitespace `item_heading` has already consumed,
    so a title-less TOC row (`Item 7    30`) still reads as one.

    What it still cannot tell apart: a heading line whose *title* ends in a number
    ("... for fiscal 2025"). Tightening further needs calibration against filings not
    recorded here, and the failure is the loud one — a missing Section is a `section_found`
    finding naming the company, never a mislabelled chunk.
    """
    line_end = full_text.find("\n", heading_end)
    tail = full_text[heading_end : line_end if line_end != -1 else len(full_text)]
    return bool(_TOC_TAIL.search(f" {tail}"))


def _clean(text: str) -> str:
    """Normalise filing whitespace without touching the words BM25 will index.

    `edgartools` renders 10-K HTML with non-breaking spaces in headings ("Item
    1.    Business") and ragged trailing space from the table layout.
    Both are invisible and both break exact matching, so they are normalised once here
    rather than in each of the gate, the chunker, and the retriever. Line structure
    survives — the chunker splits on paragraphs (ADR-0004 keeps raw text for BM25, so
    nothing here removes or rewrites a word).
    """
    if not text:
        return ""
    text = text.replace(" ", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
