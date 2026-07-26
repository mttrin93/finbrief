"""The EDGAR boundary without EDGAR: selection, repair, and cleaning as pure logic.

`fetch_filing`'s comments narrate two real traps — Tesla's April `10-K/A` amendments and
foreign private issuers' 20-Fs — and `trim_to_section_start` is the fix for the headline
JPM defect the hand-verification found. None of that may live only in prose: reverting the
exact-form filter to the "usual shortcut" the comment warns about must fail here, in CI,
not as a confusing gate finding on a live run.

The `edgar` package is faked via `sys.modules`, so nothing here imports edgartools, let
alone the network. The extraction logic under test is entirely ours.
"""

import sys
import types

import pytest

from finbrief.config import ConfigError, Settings
from finbrief.ingestion.edgar import (
    _clean,
    configure_edgar,
    fetch_filing,
    trim_to_section_start,
)
from finbrief.ingestion.gate import check_filing
from finbrief.ingestion.model import Section

BODY = "The Company designs and sells devices to customers worldwide. " * 40

SECTIONS = {s.value: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}


class FakeFiling:
    def __init__(
        self, form, filing_date, *, accession="0000000000-25-000001", sections=None, text=""
    ):
        self.form = form
        self.filing_date = filing_date
        self.accession_no = accession
        self.period_of_report = "2025-09-27"
        self.homepage_url = "https://example.invalid/filing"
        self.company = "FakeCo"
        self._sections = sections if sections is not None else {}
        self._text = text

    def obj(self):
        # A dict raises KeyError on a missing Item, one of the shapes `_from_edgartools`
        # treats as "try the fallback".
        return self._sections

    def text(self):
        return self._text


class FakeFilings:
    def __init__(self, filings):
        self._filings = list(filings)

    def latest(self, n):
        return max(self._filings, key=lambda f: f.filing_date, default=None)

    def __iter__(self):
        return iter(self._filings)


def install_fake_edgar(monkeypatch, filings):
    """A stand-in `edgar` module whose form search prefix-matches, as EDGAR's does."""

    class FakeCompany:
        def __init__(self, ticker):
            self.ticker = ticker

        def get_filings(self, form):
            forms = form if isinstance(form, list) else [form]
            return FakeFilings(
                f for f in filings if any(str(f.form).startswith(fm) for fm in forms)
            )

    module = types.ModuleType("edgar")
    module.Company = FakeCompany
    monkeypatch.setitem(sys.modules, "edgar", module)


# --- Selecting which document to ingest -------------------------------------------------


def test_fetch_prefers_the_latest_true_10k_over_a_newer_amendment(monkeypatch):
    # The Tesla trap: a `10-K` form search prefix-matches `10-K/A`, and the most recent
    # hit is an April Part III amendment with no Item 1/1A/7/7A in it. Reverting the
    # exact-form filter to `get_filings(form="10-K").latest(1)` must fail here.
    install_fake_edgar(
        monkeypatch,
        [
            FakeFiling("10-K/A", "2026-04-30", accession="amendment", sections={}),
            FakeFiling("10-K", "2025-10-31", accession="the-real-one", sections=SECTIONS),
        ],
    )

    filing = fetch_filing("TSLA")

    assert filing.ref.accession == "the-real-one"
    assert filing.ref.form == "10-K"
    assert filing.latest_annual_form == "10-K/A", "the filer's most recent annual filing"
    assert set(filing.sections) == set(Section)
    assert check_filing(filing) == (), "an amendment does not stop a 10-K filer passing"


def test_a_foreign_private_issuer_returns_the_evidence_for_the_gate(monkeypatch):
    # No 10-K to fetch — but raising would abort the run at the first bad ticker, where
    # `run_gate` deliberately reports all of them. The fetch returns the evidence and the
    # gate speaks.
    install_fake_edgar(monkeypatch, [FakeFiling("20-F", "2026-03-01")])

    filing = fetch_filing("SAP")

    assert filing.sections == {}
    assert filing.latest_annual_form == "20-F"
    assert "latest_annual_filing_is_a_10k" in {f.check for f in check_filing(filing)}


def test_a_company_with_no_annual_filing_at_all_raises(monkeypatch):
    install_fake_edgar(monkeypatch, [])

    with pytest.raises(LookupError):
        fetch_filing("ZZZZ")


def test_the_regex_fallback_fills_a_section_edgartools_missed(monkeypatch):
    # `_extract_sections` end to end: three Sections structure-anchored, the fourth
    # missing from `report[...]` and recovered from the document text — bounded by the
    # Item 8 heading, trimmed, and cleaned.
    missing_7a = {k: v for k, v in SECTIONS.items() if k != Section.MARKET_RISK.value}
    document = (
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk\n\n"
        "We are exposed to interest rate risk and manage it with duration limits. "
        * 30
        + "\n\nItem 8. Financial Statements\n\nThe report follows."
    )
    install_fake_edgar(
        monkeypatch,
        [FakeFiling("10-K", "2025-10-31", sections=missing_7a, text=document)],
    )

    filing = fetch_filing("AAPL")

    market_risk = filing.sections[Section.MARKET_RISK]
    assert market_risk.startswith("Item 7A.")
    assert "interest rate risk" in market_risk
    assert "Financial Statements" not in market_risk, "bounded at the next Item"


# --- The start-boundary repair (the JPM defect, ADR-0007 amendment) ---------------------


def test_trim_cuts_a_prefix_that_precedes_the_sections_own_heading():
    text = "Table of Contents\n\nItem 1A. Risk Factors\n\nOur business faces risks."

    assert trim_to_section_start(text, Section.RISK_FACTORS) == (
        "Item 1A. Risk Factors\n\nOur business faces risks."
    )


def test_trim_leaves_a_section_already_at_its_heading_untouched():
    text = "Item 1A. Risk Factors\n\nOur business faces risks."

    assert trim_to_section_start(text, Section.RISK_FACTORS) == text


def test_trim_leaves_text_with_no_heading_for_the_gate_to_judge():
    # No heading means nowhere useful to cut; the repair must not slice on -1 and hand
    # the gate a rewrite. `section_starts_at_its_heading` fails the original instead.
    text = "No heading anywhere in this text, which is a problem for someone else."

    assert trim_to_section_start(text, Section.RISK_FACTORS) == text


# --- Whitespace normalisation (what makes exact heading matches possible) ---------------


def test_clean_normalises_the_whitespace_that_breaks_exact_matching():
    # edgartools renders headings with non-breaking spaces ("Item 1. Business") and
    # ragged CRLF table output; every exact match downstream depends on this pass.
    raw = "Item 1. Business\r\n\r\n\r\n\r\nThe Company   designs\t devices.  \r\n"

    assert _clean(raw) == "Item 1. Business\n\nThe Company designs devices."


def test_clean_of_nothing_is_nothing():
    assert _clean("") == ""


# --- The SEC-required identity ----------------------------------------------------------


def test_configure_edgar_fails_loudly_without_an_identity():
    settings = Settings.from_env({"OPENROUTER_API_KEY": "key"})

    with pytest.raises(ConfigError) as excinfo:
        configure_edgar(settings)

    assert "SEC_EDGAR_USER_AGENT" in str(excinfo.value)


def test_configure_edgar_called_bare_needs_no_openrouter_key():
    # The hermetic env has neither variable set. The failure must be about the EDGAR
    # identity — a dry run owns no OpenRouter key, and `Settings.from_env` would demand
    # one before ever reaching the identity check.
    with pytest.raises(ConfigError) as excinfo:
        configure_edgar()

    message = str(excinfo.value)
    assert "SEC_EDGAR_USER_AGENT" in message
    assert "OPENROUTER" not in message


def test_configure_edgar_hands_the_identity_to_edgartools(monkeypatch):
    identities = []
    module = types.ModuleType("edgar")
    module.set_identity = identities.append
    monkeypatch.setitem(sys.modules, "edgar", module)
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "FinBrief test@example.com")

    configure_edgar()

    assert identities == ["FinBrief test@example.com"]
