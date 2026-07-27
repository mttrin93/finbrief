"""ADR-0007's *documented fallback*, used when structure-anchored extraction returns nothing.

It exists precisely for filers `edgartools` cannot parse, which is to say the messiest
documents in the Universe — so "bounded" has to be true rather than merely claimed. The
two ways it was not, before these tests: `Item 1` prefix-matched `Item 1A`, and a span
with no closing heading ran to the end of the document.
"""

from finbrief.ingestion.edgar import _extract_by_regex
from finbrief.ingestion.model import Section

BUSINESS = "We design and sell devices to customers worldwide. " * 30
RISKS = "Our business faces intense competition and supply concentration. " * 30
MDA = "Revenue grew on higher unit volumes and favourable mix this year. " * 30

FILING = f"""Table of Contents

Item 1. Business. 4
Item 1A. Risk Factors. 9
Item 7. Management's Discussion and Analysis. 30
Item 7A. Quantitative and Qualitative Disclosures About Market Risk. 44

Item 1. Business

{BUSINESS}

Item 1A. Risk Factors

{RISKS}

Item 1B. Unresolved Staff Comments

None.

Item 7. Management's Discussion and Analysis

{MDA}

Item 7A. Quantitative and Qualitative Disclosures About Market Risk

We are exposed to interest rate risk.

Item 8. Financial Statements
"""


def test_it_finds_the_real_section_not_the_table_of_contents_row():
    # Wordiest-candidate-wins is the whole discriminator: the TOC row and the real heading
    # are the same string, and only one of them has prose under it.
    extracted = _extract_by_regex(FILING, Section.RISK_FACTORS)

    assert "intense competition" in extracted
    assert extracted.count("Item 1A") == 1, "the TOC row is not part of the section"


def test_item_1_does_not_swallow_item_1a():
    # The bug this test exists for: `Item 1` prefix-matched `Item 1A.`, and since Item 1A
    # is the wordier span, an Item 1 lookup returned the risk factors — text that would
    # then have been chunked and labelled `Item 1`. A mislabelled chunk is undetectable
    # downstream, which makes it the worst outcome available here.
    extracted = _extract_by_regex(FILING, Section.BUSINESS)

    assert "We design and sell devices" in extracted
    assert "intense competition" not in extracted


def test_item_7_does_not_swallow_item_7a():
    extracted = _extract_by_regex(FILING, Section.MDA)

    assert "Revenue grew" in extracted
    assert "interest rate risk" not in extracted


def test_a_section_ends_at_the_next_item_heading():
    assert "Unresolved Staff Comments" not in _extract_by_regex(FILING, Section.RISK_FACTORS)


def test_an_unterminated_span_is_a_miss_not_a_candidate():
    # No "Item 1A" heading anywhere, so nothing closes Item 1. This used to be capped at
    # the gate's ceiling on the theory that it would fail there anyway — false twice
    # over: `_clean` shrinks the text *after* the cap, and Items 1/1A have no next-Item
    # content markers, so the capped span could pass the whole gate carrying half the
    # document under `Item 1`. A missing Section fails loudly (`section_found`); a
    # mislabelled chunk never does.
    unterminated = "Item 1. Business\n\n" + "word " * 400_000

    assert _extract_by_regex(unterminated, Section.BUSINESS) == ""


def test_the_wordiest_candidate_wins_when_neither_is_a_toc_row():
    # The discriminator the docstring names, actually deciding something. In `FILING`
    # above every rival is a dot-leadered TOC row that `_is_toc_line` removes first, so
    # exactly one candidate survives per Section and `words > best_words` is never the
    # reason for the answer — replacing it with "last match wins" left the suite green.
    #
    # This is the case the docstring warns about instead: a back-of-document part summary
    # that repeats the heading with no page number, so nothing but the word count
    # separates it from the real section. Last-match-wins returns the summary.
    with_a_part_summary = f"""Item 1A. Risk Factors

{RISKS}

Item 1B. Unresolved Staff Comments

None.

Part IV Summary

Item 1A. Risk Factors

Summarised above.

Item 1B. Unresolved Staff Comments
"""

    extracted = _extract_by_regex(with_a_part_summary, Section.RISK_FACTORS)

    assert "intense competition" in extracted
    assert extracted.strip() != "Item 1A. Risk Factors\n\nSummarised above."


def test_a_missing_section_yields_nothing_rather_than_a_guess():
    assert _extract_by_regex("Item 3. Legal Proceedings\n\nNone.", Section.MDA) == ""
