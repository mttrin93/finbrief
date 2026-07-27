"""The checklist must never cost a verifier their work twice.

ADR-0007's artifact is hand-done: sixty Sections read against EDGAR. Re-rendering it after
an extractor fix has to keep the ticks that still apply and flag only what moved —
otherwise a one-company boundary fix silently blanks all sixty boxes, and the reviewer
either redoes everything or, more likely, trusts a file that now claims nothing was
verified.
"""

import re

from finbrief.ingestion.model import ExtractedFiling, FilingRef, Section
from finbrief.ingestion.reporting import checklist_tickers, render_section_starts

BODY = "The Company designs and sells devices to customers worldwide. " * 40

#: The marker line as the renderer writes it, fence and all. Matching the bare marker would
#: hit the preamble's mention of it first, and the fixtures here mean the row's.
EXCERPT_OPENS = "<!-- excerpt -->\n```text"


def a_filing(*, mda_prefix: str = ""):
    sections = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    sections[Section.MDA] = f"{mda_prefix}Item 7. {Section.MDA.heading}\n\n{BODY}"
    return ExtractedFiling(
        ref=FilingRef(
            ticker="AAPL",
            form="10-K",
            accession="0000320193-25-000079",
            fiscal_year=2025,
            filing_date="2025-10-31",
            url="https://example.invalid/filing",
        ),
        latest_annual_form="10-K",
        sections=sections,
    )


def tick_everything(markdown: str) -> str:
    return markdown.replace("- [ ]", "- [x]")


def test_a_first_render_ships_every_box_unticked():
    rendered = render_section_starts([a_filing()])

    assert rendered.count("- [ ]") == len(Section)
    assert "- [x]" not in rendered


def test_an_unchanged_section_keeps_its_tick():
    verified = tick_everything(render_section_starts([a_filing()]))

    rerendered = render_section_starts([a_filing()], verified)

    assert rerendered.count("- [x]") == len(Section)
    assert "**CHANGED**" not in rerendered


def test_only_the_section_that_moved_loses_its_tick():
    verified = tick_everything(render_section_starts([a_filing()]))

    # The JPM fix in miniature: one Section's text shifts, the other three do not.
    rerendered = render_section_starts([a_filing(mda_prefix="Table of Contents\n\n")], verified)

    assert rerendered.count("- [x]") == len(Section) - 1
    assert rerendered.count("**CHANGED**") == 1
    mda_status = rerendered.split("### Item 7. Management")[1].split("\n\n")[1]
    assert mda_status.startswith("- [ ]"), mda_status


def test_a_verifiers_own_notes_survive_a_rerender():
    # A tick can be redone by reading the filing again; a finding written by hand cannot.
    verified = tick_everything(render_section_starts([a_filing()]))
    note = "**Finding (hand-verification):** starts three pages early, see pp.46-160."
    verified = verified.replace("```text", f"{note}\n\n```text", 1)

    rerendered = render_section_starts([a_filing(mda_prefix="x\n\n")], verified)

    assert note in rerendered


def test_a_note_written_under_the_excerpt_survives_a_rerender():
    # Where a verifier writes matters to nobody but the parser, and the parser read only as
    # far as the excerpt fence — so a note written under the excerpt it comments on, which
    # is the natural place to write after reading one, was dropped without a word. The
    # contract is unconditional: this function must never be the reason a finding
    # disappears (issue #3 review).
    verified = tick_everything(render_section_starts([a_filing()]))
    note = "**Finding (hand-verification):** the excerpt above stops mid-table."
    below_the_excerpt = verified.replace("```\n", f"```\n\n{note}\n", 1)
    assert below_the_excerpt != verified, "the fixture must actually place a note"

    rerendered = render_section_starts([a_filing()], below_the_excerpt)

    assert note in rerendered
    item_1 = rerendered.split("### Item 1A.")[0]
    assert note in item_1, "and under the row it was written for"
    assert rerendered.count(note) == 1


def test_a_note_written_above_the_status_line_survives_a_rerender():
    # The third place a verifier can write, and the last one that was dropped: straight
    # under the `### Item 1A.` heading, before the checkbox. The parser read from the
    # status line onward, so it never saw it. The preamble promises anything written
    # inside a row comes back, without qualifying where (issue #3 review).
    verified = tick_everything(render_section_starts([a_filing()]))
    note = "**Finding (hand-verification):** compare against last year's Item 1A."
    above_the_status_line = verified.replace(
        "### Item 1A. Risk Factors\n\n", f"### Item 1A. Risk Factors\n\n{note}\n\n", 1
    )
    assert above_the_status_line != verified, "the fixture must actually place a note"

    rerendered = render_section_starts([a_filing()], above_the_status_line)

    assert note in rerendered
    item_1a = rerendered.split("### Item 1A.")[1].split("### Item 7.")[0]
    assert note in item_1a, "and under the row it was written for"
    assert rerendered.count(note) == 1


def test_a_note_that_quotes_the_filing_in_a_fence_survives_whole():
    # The evidence this artifact collects is "the filing says X" — and the natural way to
    # write X down is a fenced quote, which the excerpts themselves model. Cutting the row
    # at the first fence left in the remainder dropped the quote *and* everything the
    # verifier wrote after it, keeping only the sentence that introduced it (issue #3
    # review).
    verified = tick_everything(render_section_starts([a_filing()]))
    note = (
        "**Finding (hand-verification):** the filing reads\n\n"
        "```\nItem 1. Business\nThe Company designs, manufactures and markets...\n```\n\n"
        "so the excerpt is one heading early."
    )
    with_quote = verified.replace(EXCERPT_OPENS, f"{note}\n\n{EXCERPT_OPENS}", 1)
    assert with_quote != verified, "the fixture must actually place a note"

    rerendered = render_section_starts([a_filing()], with_quote)

    for line in note.splitlines():
        assert line in rerendered, line
    assert "so the excerpt is one heading early." in rerendered


def test_a_notes_own_text_fence_is_not_mistaken_for_the_generated_excerpt():
    # The other half of the same ambiguity. A verifier who copies the excerpt's own
    # ```text style, above the excerpt, had their quote read *as* the excerpt — so the row
    # compared unequal to itself and came back unticked and flagged **CHANGED** with the
    # character count unmoved, sending a human back to re-verify text nothing had touched.
    verified = tick_everything(render_section_starts([a_filing()]))
    note = "Cross-check:\n\n```text\nItem 1. Business\n```\n\nmatches the source."
    with_quote = verified.replace(EXCERPT_OPENS, f"{note}\n\n{EXCERPT_OPENS}", 1)

    rerendered = render_section_starts([a_filing()], with_quote)

    assert "**CHANGED**" not in rerendered
    assert rerendered.count("- [x]") == len(Section)
    assert "matches the source." in rerendered


def test_a_checklist_written_before_the_markers_keeps_its_ticks():
    # The committed sixty were rendered fence-only. Migrating the format must not be the
    # thing that unticks them — an unmarked row is read the old way for the one re-render
    # that gives it markers.
    legacy = tick_everything(render_section_starts([a_filing()]))
    legacy = legacy.replace(EXCERPT_OPENS, "```text").replace("```\n<!-- /excerpt -->", "```")
    assert EXCERPT_OPENS not in legacy, "the fixture must actually strip the markers"

    rerendered = render_section_starts([a_filing()], legacy)

    assert rerendered.count("- [x]") == len(Section)
    assert "**CHANGED**" not in rerendered
    assert rerendered.count(EXCERPT_OPENS) == len(Section)


def test_a_row_heading_is_never_carried_forward_as_if_it_were_a_note():
    # The guard on the fix above: read the row from its first line and the `### Item 1A.`
    # heading itself comes back as a note, duplicated on every re-render.
    verified = tick_everything(render_section_starts([a_filing()]))

    rerendered = render_section_starts([a_filing()], verified)

    assert rerendered.count("### Item 1A.") == 1


def test_the_excerpt_is_never_carried_forward_as_if_it_were_a_note():
    # The guard on the fix above: strip the fence too eagerly and the generated excerpt
    # comes back as a hand-written note, duplicated on every re-render.
    verified = tick_everything(render_section_starts([a_filing()]))

    rerendered = render_section_starts([a_filing()], verified)

    assert rerendered == verified, "a re-render of unchanged text reproduces the file"


def test_a_checklist_reports_which_companies_it_holds_rows_for():
    # What `scripts/ingest_filings.py` asks before re-rendering: a subset run must not be
    # allowed to delete the rows of companies it never fetched.
    filing = a_filing()
    msft = ExtractedFiling(
        ref=FilingRef(
            ticker="MSFT",
            form="10-K",
            accession="0000789019-25-000118",
            fiscal_year=2025,
            filing_date="2025-07-30",
        ),
        latest_annual_form="10-K",
        sections=filing.sections,
    )

    rendered = render_section_starts([filing, msft])

    assert checklist_tickers(rendered) == frozenset({"AAPL", "MSFT"})
    assert checklist_tickers(render_section_starts([filing])) == frozenset({"AAPL"})


def test_a_changed_row_reports_what_it_used_to_be():
    verified = tick_everything(render_section_starts([a_filing()]))

    rerendered = render_section_starts([a_filing(mda_prefix="Table of Contents\n\n")], verified)

    assert "was " in rerendered.split("**CHANGED**")[1][:40]


def pointer_filing(*, pages: str = "133-142"):
    """A filing whose Item 7A is a pointer long enough to outrun the 320-char excerpt."""
    sections = {s: f"{s.value}. {s.heading}\n\n{BODY}" for s in Section}
    sections[Section.MARKET_RISK] = (
        f"Item 7A. {Section.MARKET_RISK.heading}\n\n"
        "Refer to the Market Risk Management section of Management's Discussion and "
        "Analysis of Financial Condition and Results of Operations, including the "
        "quantitative and qualitative disclosures about interest rate, currency, "
        "commodity and equity price risk presented there, which is incorporated herein "
        f"by reference, on pages {pages}."
    )
    filing = a_filing()
    return type(filing)(
        ref=filing.ref, latest_annual_form=filing.latest_annual_form, sections=sections
    )


def test_an_unchanged_pointer_row_keeps_its_tick():
    verified = tick_everything(render_section_starts([pointer_filing()]))

    rerendered = render_section_starts([pointer_filing()], verified)

    assert rerendered.count("- [x]") == len(Section)
    assert "**CHANGED**" not in rerendered


def test_a_pointer_that_changes_beyond_the_excerpt_loses_its_tick():
    # The excerpt shows 320 characters and a pointer may run to 1,500. Without the
    # character count on referenced rows, a pointer whose page range changed past the
    # excerpt kept a tick no human had re-checked — and the artifact then attested to
    # text nobody has seen.
    verified = tick_everything(render_section_starts([pointer_filing()]))

    rerendered = render_section_starts([pointer_filing(pages="90-101")], verified)

    assert rerendered.count("- [x]") == len(Section) - 1
    assert rerendered.count("**CHANGED**") == 1
    item_7a = rerendered.split("### Item 7A.")[1]
    assert "- [ ] Verified **incorporated by reference" in item_7a


def test_a_prior_row_with_no_character_count_cannot_carry_its_tick():
    # The committed artifact predates the count on referenced rows, and `chars in (None,
    # len(text))` treated "we cannot tell whether it moved" as "it did not move" — so
    # exactly the six pointer rows the count was added for kept ticks nobody re-earned.
    # A hand-edit that breaks a status line has the same shape. Unticked is the only safe
    # answer: a tick can be redone from the filing.
    verified = tick_everything(render_section_starts([pointer_filing()]))
    countless = re.sub(r"(not ingested\*\*) — [\d,]+ characters extracted", r"\1", verified)
    assert countless != verified, "the fixture must actually remove the count"

    rerendered = render_section_starts([pointer_filing()], countless)

    assert rerendered.count("- [x]") == len(Section) - 1
    item_7a = rerendered.split("### Item 7A.")[1]
    assert "- [ ] Verified **incorporated by reference" in item_7a


def test_a_section_that_vanished_is_flagged_and_keeps_its_verifiers_notes():
    # The checklist is written before the findings check, so a gate-failing run renders
    # this branch — and it is the run in which a verifier most needs their own notes on
    # the Section that just disappeared.
    verified = tick_everything(render_section_starts([a_filing()]))
    note = "**Finding (hand-verification):** this one is the JPM shape, check pp.46-160."
    verified = verified.replace("```text", f"{note}\n\n```text", 3)

    filing = a_filing()
    without_mda = {s: t for s, t in filing.sections.items() if s is not Section.MDA}
    rerendered = render_section_starts(
        [type(filing)(ref=filing.ref, latest_annual_form="10-K", sections=without_mda)],
        verified,
    )

    item_7 = rerendered.split("### Item 7. Management")[1].split("### Item 7A.")[0]
    assert "- [ ] **NOT EXTRACTED**" in item_7
    assert note in item_7
