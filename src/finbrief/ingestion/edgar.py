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
import re
from collections.abc import Mapping

from finbrief.config import ConfigError, Settings, get_settings
from finbrief.ingestion.model import (
    NEXT_ITEM_MARKERS,
    ExtractedFiling,
    FilingRef,
    Section,
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

#: Two or more letters — a word for "which candidate span is the real section", scoring
#: dot leaders and page numbers at zero. Same rule the gate applies, for the same reason.
_WORD = re.compile(r"[^\W\d_]{2,}")


def configure_edgar(settings: Settings | None = None) -> None:
    """Give edgartools the SEC-required contact identity, failing loudly without one.

    The SEC rejects unidentified traffic, and `edgartools` would otherwise raise deep in a
    request with a stack trace that says nothing about `.env`. Ingestion is a script a
    human runs, so it can afford to say what is wrong in one line.
    """
    settings = settings or get_settings()
    if not settings.sec_edgar_user_agent:
        raise ConfigError(
            "SEC_EDGAR_USER_AGENT is not set, and the SEC requires a contact identity "
            "on every request. Set it to 'FinBrief your-email@example.com' in .env "
            "(see .env.example)."
        )
    from edgar import set_identity

    set_identity(settings.sec_edgar_user_agent)


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
    tenk_filings = [f for f in company.get_filings(form="10-K") if str(f.form) == "10-K"]
    filing = max(tenk_filings, key=lambda f: f.filing_date, default=None)
    if filing is None:
        raise LookupError(
            f"{ticker} has never filed a 10-K (its latest annual filing is a "
            f"{annual.form}). It cannot be in the Universe — see ADR-0007."
        )

    ref = FilingRef(
        ticker=ticker,
        form=str(filing.form),
        accession=str(filing.accession_no),
        fiscal_year=_fiscal_year(filing),
        filing_date=str(filing.filing_date),
        url=str(filing.homepage_url),
    )
    sections = _extract_sections(filing)

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


def _fiscal_year(filing) -> int:
    """The year the filing is *about*, from its period of report.

    Not the filing date: Apple's FY2025 10-K is filed in October 2025 and JPMorgan's in
    February 2026, and the golden set cites the fiscal year of the text (ADR-0002).
    """
    period = getattr(filing, "period_of_report", None)
    if period is None:
        raise LookupError(f"{filing.accession_no} has no period of report to date it by.")
    return int(str(period)[:4])


def _extract_sections(filing) -> dict[Section, str]:
    """The four curated Sections, structure-anchored first and regex-bounded second."""
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
                    ticker=str(filing.company),
                    accession=str(filing.accession_no),
                    section=section.value,
                    chars=len(text),
                )
        if text:
            trimmed = trim_at_next_item(text, section)
            if len(trimmed) != len(text):
                log_event(
                    logger,
                    "section_trimmed",
                    level=logging.WARNING,
                    ticker=str(filing.company),
                    accession=str(filing.accession_no),
                    section=section.value,
                    chars_before=len(text),
                    chars_after=len(trimmed),
                    reason="ran into the next Item",
                )
            sections[section] = trimmed

    return sections


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

    What keeps this from being the extractor grading its own homework: the result still
    faces `gate.check_filing`, which applies the same marker check. If a trim misses, the
    Section fails exactly as it did before this function existed. The repair proposes; the
    gate disposes.

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
    """
    item = re.escape(section.value.removeprefix("Item ").strip())
    start = re.compile(rf"^[ \t]*Item[ \t ]+{item}\.?[ \t ]*", re.MULTILINE | re.I)
    end = re.compile(rf"^[ \t]*Item[ \t ]+{_NEXT_ITEM[section]}\.?[ \t ]*", re.MULTILINE | re.I)

    best = ""
    best_words = 0
    for match in start.finditer(full_text):
        stop = end.search(full_text, match.start() + 1)
        candidate = full_text[match.start() : stop.start() if stop else len(full_text)]
        words = len(_WORD.findall(candidate))
        if words > best_words:
            best, best_words = candidate, words
    return best


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
