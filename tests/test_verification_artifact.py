"""The checklist must never cost a verifier their work twice.

ADR-0007's artifact is hand-done: sixty Sections read against EDGAR. Re-rendering it after
an extractor fix has to keep the ticks that still apply and flag only what moved —
otherwise a one-company boundary fix silently blanks all sixty boxes, and the reviewer
either redoes everything or, more likely, trusts a file that now claims nothing was
verified.
"""

from finbrief.ingestion.model import ExtractedFiling, FilingRef, Section
from finbrief.ingestion.reporting import render_section_starts

BODY = "The Company designs and sells devices to customers worldwide. " * 40


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
    verified = tick_everything(render_section_starts([a_filing()])).replace(
        "- [x] Verified — ",
        "- [x] Verified — ",
        1,
    )
    note = "**Finding (hand-verification):** starts three pages early, see pp.46-160."
    verified = verified.replace("```text", f"{note}\n\n```text", 1)

    rerendered = render_section_starts([a_filing(mda_prefix="x\n\n")], verified)

    assert note in rerendered


def test_a_changed_row_reports_what_it_used_to_be():
    verified = tick_everything(render_section_starts([a_filing()]))

    rerendered = render_section_starts([a_filing(mda_prefix="Table of Contents\n\n")], verified)

    assert "was " in rerendered.split("**CHANGED**")[1][:40]
