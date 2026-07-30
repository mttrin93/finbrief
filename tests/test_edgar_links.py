"""`filing_index_url` against the URLs a real fetch recorded.

The sources panel turns each accession into a link so an analyst can open the filing an
excerpt was cut from (user story 2). Nothing about that is checkable by eye: a link built
from the wrong CIK resolves to *some* company's directory and looks entirely correct in the
panel, and the reader who clicks it is the one who finds out.

So the derivation is bound to evidence rather than to a template. Every EDGAR URL in
`docs/verification/section-starts.md` came off `Filing.homepage_url` during a real ingest of
that filing, and this file asserts that ticker + accession alone reproduce all fifteen. Two
rows are the whole reason the test exists: META's accession is `0001628280-…` and its URL is
`data/1326801/`, TSLA's is the same agent's and its URL is `data/1318605/`. The leading block
of an accession is the *filer agent's* CIK — Donnelley's, for both — so a URL composed from
it points at a third company's filings with no error anywhere.

Hermetic like everything else here: edgartools' ticker table is bundled in the wheel, so the
lookup is a local parquet read. A ticker missing from that bundle would fall through to a
live SEC call, which is what `test_every_universe_ticker_is_in_the_bundled_table` exists to
catch before an `EgressBlocked` does.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from finbrief.config import TICKERS
from finbrief.ingestion.edgar import EDGAR_SEARCH_URL, filing_index_url

SECTION_STARTS = Path(__file__).parents[1] / "docs" / "verification" / "section-starts.md"

#: One filing's header block in the hand-verification artifact:
#:
#:     ## META — FY2025 10-K
#:
#:     - Accession: `0001628280-26-003942`  ·  filed 2026-01-29
#:     - EDGAR: https://www.sec.gov/Archives/edgar/data/1326801/0001628280-26-003942-index.html
#:
#: Matched as one pattern rather than three so a row can only contribute a *triple*: a ticker
#: paired with an accession and URL from the filing below it would be a test that passes on a
#: derivation which is wrong for every company.
_FILING_BLOCK = re.compile(
    r"^## (?P<ticker>[A-Z.\-]+) — FY\d{4} .+?\n"
    r"\n- Accession: `(?P<accession>[\d-]+)`.*?\n"
    r"- EDGAR: (?P<url>\S+)$",
    re.MULTILINE | re.DOTALL,
)


def recorded_filings() -> list[tuple[str, str, str]]:
    """`(ticker, accession, url)` for every filing in the hand-verification artifact."""
    assert SECTION_STARTS.exists(), (
        f"{SECTION_STARTS} is the recorded evidence these links are checked against, and is "
        f"committed. Restore it, or re-run `scripts/ingest_filings.py --section-starts`."
    )
    rows = [
        (match["ticker"], match["accession"], match["url"])
        for match in _FILING_BLOCK.finditer(SECTION_STARTS.read_text(encoding="utf-8"))
    ]
    assert len(rows) == len(TICKERS), (
        f"the artifact's header shape changed; this cross-check reads it. Parsed {len(rows)} "
        f"filing(s) for {len(TICKERS)} Universe companies."
    )
    return rows


@pytest.mark.parametrize(("ticker", "accession", "url"), recorded_filings())
def test_the_derived_link_is_the_one_edgar_gave_the_ingest_run(ticker, accession, url):
    """An equality against a recorded URL, not a shape check on a URL we built."""
    assert filing_index_url(ticker, accession) == url


def test_the_accession_is_not_where_the_cik_comes_from():
    """The failure the equalities above would catch, named so it cannot be lost in a refactor.

    META and Donnelley are both real CIKs, so a link built from the accession's leading block
    is a working URL for the wrong company — the one defect in this derivation that produces
    no error at any layer. Asserted as an inequality on the *pair*: whichever filer's agent
    files for two Universe companies next, two accessions sharing a prefix must still resolve
    to two different directories.
    """
    recorded = dict((ticker, (accession, url)) for ticker, accession, url in recorded_filings())
    meta_accession, meta_url = recorded["META"]
    tsla_accession, tsla_url = recorded["TSLA"]

    assert meta_accession.split("-")[0] == tsla_accession.split("-")[0], (
        "this test rests on META and TSLA sharing a filer agent; if that changed, pick "
        "another pair from the artifact rather than deleting the check"
    )
    assert filing_index_url("META", meta_accession) == meta_url
    assert filing_index_url("TSLA", tsla_accession) == tsla_url
    assert meta_url != tsla_url


def test_every_universe_ticker_is_in_the_bundled_table():
    """No Universe company may need the network to resolve its CIK.

    `get_company_cik_lookup` falls back to a live SEC call for a ticker the bundled parquet
    does not carry, and `filing_index_url` reads the mapping directly so it cannot take that
    path — but a ticker missing from the bundle would then render every one of that company's
    citations unlinked, silently. This is where that shows up.

    Asked through `filing_index_url` rather than of the lookup table, so what is checked is
    the thing the app calls. The accession is a well-formed placeholder: only the *ticker*
    half of the resolution is under test, and the fifteen real accessions are asserted
    above.
    """
    missing = sorted(
        ticker for ticker in TICKERS if not filing_index_url(ticker, "0000000000-00-000000")
    )

    assert not missing, (
        f"{missing} resolve to no CIK in edgartools' bundled company_tickers.parquet, so "
        f"their sources render with no EDGAR link. Check the spelling against the bundle."
    )


def test_an_unresolvable_ticker_gets_no_link_rather_than_a_wrong_one():
    """An absence, returned as an absence — and never as a URL for somebody else's filings."""
    assert filing_index_url("NOTATICKER", "0000320193-25-000079") == ""
    assert filing_index_url("AAPL", "") == ""
    assert filing_index_url("AAPL", "   ") == ""


def test_the_panels_verify_link_points_at_edgar_and_not_at_a_filing():
    """The scope panel's link is the one place a URL is written rather than derived.

    It is EDGAR's search entry point — deliberately not a filing index, because the question
    that panel raises is about filings FinBrief did *not* ingest. Bound so it stays that
    shape: a filing URL here would silently narrow "verify any of this" to one company.
    """
    from finbrief.prompts import GROUNDING_SCOPE_VERIFY

    assert EDGAR_SEARCH_URL.startswith("https://www.sec.gov/")
    assert "Archives" not in EDGAR_SEARCH_URL, "the panel links search, not one filing"
    assert f"[Verify on EDGAR]({EDGAR_SEARCH_URL})" in GROUNDING_SCOPE_VERIFY
