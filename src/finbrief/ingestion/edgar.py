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
from finbrief.ingestion.model import ExtractedFiling, FilingRef, Section
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

    tenk_filings = company.get_filings(form="10-K")
    # `.latest(1)` on the 10-K search, not `annual`: a 20-F filer must still produce an
    # `ExtractedFiling` so the gate can report *which* company is wrong. Only a company
    # with no 10-K in its whole history has nothing to hand the gate.
    filing = tenk_filings.latest(1) if len(tenk_filings) else None
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
            sections[section] = text

    return sections


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
