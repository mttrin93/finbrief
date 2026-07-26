"""Seam 6 — the Phase-1 section-detection gate (spec §Testing Decisions, ADR-0007).

The gate is the last thing standing between a misparsed filing and a knowledge base that
looks fine and answers wrongly. It runs over already-extracted filings, so every rule here
is a pure function of data — no EDGAR, no network, no fixtures larger than the rule needs.
"""

import pytest

from finbrief.ingestion.gate import (
    MAX_SECTION_CHARS,
    MIN_SECTION_CHARS,
    SectionGateError,
    check_filing,
    incorporated_sections,
    run_gate,
)
from finbrief.ingestion.model import ExtractedFiling, FilingRef, Section

BODY = "The Company designs and sells devices. " * 200


def a_filing(
    *,
    ticker: str = "AAPL",
    latest_annual_form: str = "10-K",
    sections: dict[Section, str] | None = None,
) -> ExtractedFiling:
    """A filing that passes every gate rule, so a test can break exactly one."""
    if sections is None:
        sections = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    return ExtractedFiling(
        ref=FilingRef(
            ticker=ticker,
            form="10-K",
            accession=f"0000000000-25-{sum(ticker.encode()):06d}",
            fiscal_year=2025,
            filing_date="2025-10-31",
        ),
        latest_annual_form=latest_annual_form,
        sections=sections,
    )


def checks_that_failed(filing: ExtractedFiling) -> set[str]:
    return {finding.check for finding in check_filing(filing)}


# --- The EDGAR metadata check (issue #3 comment; supersedes config._TWENTY_F_FILERS) ---


def test_a_filing_from_a_10k_filer_passes_every_rule():
    assert check_filing(a_filing()) == ()


def test_a_company_whose_latest_annual_filing_is_a_20f_fails():
    # The real check the denylist could only approximate: a foreign private issuer files a
    # 20-F, which has no Item 1A/7/7A at all (ADR-0007). Asked of EDGAR, not of a list of
    # tickers someone thought to write down.
    findings = check_filing(a_filing(ticker="SAP", latest_annual_form="20-F"))

    assert {f.check for f in findings} == {"latest_annual_filing_is_a_10k"}
    assert "20-F" in findings[0].detail


def test_an_amended_10k_still_counts_as_a_10k_filer():
    # EDGAR form search prefix-matches, so the most recent annual filing on file is often
    # `10-K/A`. The amendment is still a 10-K; only the form family is the question.
    assert check_filing(a_filing(latest_annual_form="10-K/A")) == ()


# --- Per company x section: found / non-empty / length-bounded / body >> heading -------


def test_a_missing_section_fails():
    sections = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    del sections[Section.MARKET_RISK]

    findings = check_filing(a_filing(sections=sections))

    assert [(f.section, f.check) for f in findings] == [(Section.MARKET_RISK, "section_found")]


def test_an_empty_section_fails():
    sections = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    sections[Section.MDA] = "   \n  "

    findings = check_filing(a_filing(sections=sections))

    assert [(f.section, f.check) for f in findings] == [(Section.MDA, "section_non_empty")]


def test_a_section_shorter_than_the_floor_fails():
    # "Item 7A. Not applicable." is a real thing a filer writes, and it is not a Section:
    # it is a pointer to one, and the golden set cannot be authored against it.
    sections = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    sections[Section.MARKET_RISK] = (
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk\n\nNot applicable."
    )

    findings = check_filing(a_filing(sections=sections))

    assert [(f.section, f.check) for f in findings] == [
        (Section.MARKET_RISK, "section_length_bounded")
    ]
    assert str(MIN_SECTION_CHARS) in findings[0].detail


def test_a_section_longer_than_the_ceiling_fails():
    # A runaway extraction that swallowed the rest of the filing is as wrong as a miss,
    # and silently: it would still chunk, still embed, and still label every chunk `Item 1`.
    sections = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    repeats = MAX_SECTION_CHARS // len(BODY) + 1
    sections[Section.BUSINESS] = f"Item 1. Business\n\n{BODY * repeats}"

    findings = check_filing(a_filing(sections=sections))

    assert [(f.section, f.check) for f in findings] == [
        (Section.BUSINESS, "section_length_bounded")
    ]


def test_a_table_of_contents_hit_fails_because_its_body_does_not_exceed_its_heading():
    # The failure ADR-0007 names outright: "Item 1A" appears in the table of contents as
    # well as at the real section. A TOC hit is a heading, a page number, and nothing else
    # — long enough to look like text, empty of the risk factors it claims to be.
    sections = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    sections[Section.RISK_FACTORS] = "Item 1A. Risk Factors" + "." * 4000 + " 12"

    findings = check_filing(a_filing(sections=sections))

    assert [(f.section, f.check) for f in findings] == [
        (Section.RISK_FACTORS, "section_body_exceeds_heading")
    ]


def test_a_section_that_runs_into_the_next_one_fails_however_long_it_is():
    # JPM FY2025: `edgartools` returned 413,149 characters for Item 7, of which the last
    # 12,837 are Item 8 — the auditor's report and the consolidated statements. Length
    # alone cannot catch this: JPM's *legitimate* MD&A is 400,312 characters, so any
    # ceiling that fails the runaway also fails the real thing. The auditor's report is
    # the tell, because it is Item 8 content and never MD&A.
    sections = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    sections[Section.MDA] += (
        "\n\nReport of Independent Registered Public Accounting Firm\n\n"
        "To the Board of Directors and Shareholders:"
    )

    findings = check_filing(a_filing(sections=sections))

    assert [(f.section, f.check) for f in findings] == [
        (Section.MDA, "section_stops_before_the_next_item")
    ]


def test_a_very_long_but_clean_mda_passes():
    # BAC (293,458) and GS (309,157) are not runaways; a large bank's MD&A is simply that
    # big. The ceiling is a sanity backstop, not a verbosity policy.
    sections = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    sections[Section.MDA] = f"Item 7. {Section.MDA.heading}\n\n{BODY * 40}"

    assert len(sections[Section.MDA]) > 300_000
    assert check_filing(a_filing(sections=sections)) == ()


# --- Incorporation by reference (ADR-0007 amendment; 6 of 15 Universe filers) -----------


def test_an_item_7a_incorporated_by_reference_is_not_a_failure():
    # JPM, BAC, GS, JNJ, LLY and PFE all answer Item 7A with a pointer into Item 7. That
    # is a lawful filing, not a broken parse, and the content is still ingested — under
    # the `Item 7` label, because that is where the filer put it.
    sections = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    sections[Section.MARKET_RISK] = (
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk\n\n"
        "Refer to the Market Risk Management section of Management's discussion and "
        "analysis on pages 133-142 for a discussion of quantitative and qualitative "
        "disclosures about market risk."
    )

    assert check_filing(a_filing(sections=sections)) == ()
    assert incorporated_sections(a_filing(sections=sections)) == frozenset(
        {Section.MARKET_RISK}
    )


def test_a_short_section_with_no_reference_language_still_fails():
    # The rule has to be a positive identification, not "short sections are forgiven".
    # Otherwise a genuine extraction miss that happens to be brief goes silent — which is
    # the entire failure mode this gate exists to prevent.
    sections = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    sections[Section.MARKET_RISK] = (
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk\n\nNot applicable."
    )

    findings = check_filing(a_filing(sections=sections))

    assert [(f.section, f.check) for f in findings] == [
        (Section.MARKET_RISK, "section_length_bounded")
    ]
    assert incorporated_sections(a_filing(sections=sections)) == frozenset()


def test_a_full_section_that_merely_mentions_a_cross_reference_is_still_a_section():
    # "incorporated by reference" appears in plenty of real Item 1A text. Only a short
    # text that is *nothing but* a pointer counts.
    sections = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    sections[Section.RISK_FACTORS] = (
        f"Item 1A. Risk Factors\n\nThe information is incorporated by reference to "
        f"Item 7 of this report.\n\n{BODY}"
    )

    assert check_filing(a_filing(sections=sections)) == ()
    assert incorporated_sections(a_filing(sections=sections)) == frozenset()


# --- Failing loudly --------------------------------------------------------------------


def test_the_gate_raises_and_names_every_company_and_section_that_failed():
    # One raise carrying every finding, not the first: re-running a 15-company ingest to
    # discover the second broken filer is how a gate stops being run at all.
    bad_sections = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    del bad_sections[Section.MDA]

    with pytest.raises(SectionGateError) as excinfo:
        run_gate(
            [
                a_filing(ticker="AAPL"),
                a_filing(ticker="MSFT", sections=bad_sections),
                a_filing(ticker="SAP", latest_annual_form="20-F"),
            ]
        )

    message = str(excinfo.value)
    assert "MSFT" in message and "Item 7" in message
    assert "SAP" in message and "20-F" in message
    assert "AAPL" not in message, "a passing company is not noise in a failure report"


def test_the_gate_is_silent_when_every_company_passes():
    run_gate([a_filing(ticker="AAPL"), a_filing(ticker="MSFT")])


# --- The start boundary (ADR-0007 amendment; found by hand-verification, not by code) ---


def test_a_section_that_starts_before_its_own_heading_fails():
    # JPM FY2025 Item 7 opened on the annual report's Three-Year Summary of Consolidated
    # Financial Highlights — three pages before the pp.46-160 the filing's own
    # cross-reference gives for Item 7. It passed every other rule: found, non-empty,
    # bounded, wordy, free of Item 8 content. A human reading the artifact caught it.
    sections = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    sections[Section.MDA] = (
        f"Financial\n\nTHREE-YEAR SUMMARY OF CONSOLIDATED FINANCIAL HIGHLIGHTS\n\n"
        f"Total net revenue $182,447\n\n{BODY}"
    )

    findings = check_filing(a_filing(sections=sections))

    assert [(f.section, f.check) for f in findings] == [
        (Section.MDA, "section_starts_at_its_heading")
    ]


def test_a_leading_running_header_fails_too():
    # The same defect at the small end: ten Sections opened on a stray "Table of Contents"
    # or company-name line. One rule, no size threshold — a Section either begins at its
    # heading or it does not.
    sections = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    sections[Section.RISK_FACTORS] = f"Table of Contents\n\nItem 1A. Risk Factors\n\n{BODY}"

    assert "section_starts_at_its_heading" in checks_that_failed(a_filing(sections=sections))


def test_a_section_titled_without_an_item_label_is_accepted():
    # JPM prints no "Item 7" label anywhere in its MD&A — the heading is the bare title.
    # The title fallback is what lets a legitimate filing like that pass at all.
    sections = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    sections[Section.MDA] = f"Management's discussion and analysis\n\n{BODY}"

    assert check_filing(a_filing(sections=sections)) == ()


def test_the_title_fallback_does_not_fire_on_a_sentence_that_merely_mentions_it():
    # "Business" is a common word; a paragraph is not a heading. Without the line-length
    # guard the fallback would happily start Item 1 in the middle of a sentence.
    sections = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    sections[Section.BUSINESS] = (
        "Some preamble that is not a heading at all.\n\n"
        "Business conditions in the markets we serve deteriorated over the period under "
        "review, and the following discussion sets out why that happened in detail.\n\n" + BODY
    )

    assert "section_starts_at_its_heading" in checks_that_failed(a_filing(sections=sections))
