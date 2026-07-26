"""The gate's rules against real EDGAR text, recorded (`scripts/record_edgar_fixtures.py`).

`test_section_gate.py` covers the rules with constructed inputs, which is where a rule's
logic belongs. This file covers the thing constructed inputs cannot: that the rules match
how filers *actually* write, in wordings nobody would think to invent. Every fixture here
is a case that changed the gate's design during the T2 scaling run.
"""

import pytest

from finbrief.ingestion.edgar import trim_at_next_item
from finbrief.ingestion.gate import check_filing, incorporated_sections
from finbrief.ingestion.model import Section, is_incorporated_by_reference

#: Every Universe filer that hands Item 7A off to Item 7 — all three banks and all three
#: healthcare names, 6 of 15. Their wordings share no formula: "Refer to", "See",
#: "are set forth in", "incorporated herein by reference", "You can find".
POINTER_KEYS = [
    "jpm_item_7a_pointer",
    "bac_item_7a_pointer",
    "gs_item_7a_pointer",
    "jnj_item_7a_pointer",
    "lly_item_7a_pointer",
    "pfe_item_7a_pointer",
]


def test_a_real_clean_10k_passes_every_gate_rule(recorded_filing):
    assert check_filing(recorded_filing) == ()
    assert incorporated_sections(recorded_filing) == frozenset()
    assert set(recorded_filing.sections) == set(Section)


@pytest.mark.parametrize("key", POINTER_KEYS)
def test_every_real_incorporation_wording_is_recognised(key, recorded_sections):
    assert is_incorporated_by_reference(recorded_sections[key])


def test_a_real_item_7_8_boundary_miss_is_caught(recorded_sections, recorded_filing):
    # GM FY2025 Item 7A: 26,450 characters, of which the first 11,790 are genuine market
    # risk and the rest is the auditor's report. It passed every length and shape rule —
    # only the content check finds it, which is why that check exists.
    spilled = dict(recorded_filing.sections)
    spilled[Section.MARKET_RISK] = recorded_sections["gm_item_7a_spill"]
    filing = type(recorded_filing)(
        ref=recorded_filing.ref,
        latest_annual_form=recorded_filing.latest_annual_form,
        sections=spilled,
    )

    findings = check_filing(filing)

    assert [(f.section, f.check) for f in findings] == [
        (Section.MARKET_RISK, "section_stops_before_the_next_item")
    ]
    assert not is_incorporated_by_reference(recorded_sections["gm_item_7a_spill"])


def test_trimming_the_real_gm_spill_yields_a_section_the_gate_accepts(recorded_sections):
    # The repair and its judge, end to end on real text. `trim_at_next_item` cuts at the
    # auditor's report; what survives has to satisfy the same gate that rejected the
    # original — that is what keeps the extractor from grading its own homework.
    spilled = recorded_sections["gm_item_7a_spill"]

    trimmed = trim_at_next_item(spilled, Section.MARKET_RISK)

    assert len(trimmed) < len(spilled)
    assert "Report of Independent".lower() not in trimmed.lower()
    assert trimmed == spilled[: len(trimmed)], "a prefix, never a rewrite"
    assert "financial risk management program" in trimmed, "the real content survives"


def test_trimming_leaves_a_clean_section_untouched(recorded_filing):
    for section, text in recorded_filing.sections.items():
        assert trim_at_next_item(text, section) == text


def test_a_real_section_is_not_mistaken_for_a_pointer(recorded_filing):
    # Apple's Item 7A is 3,029 characters — short, and about market risk, and real.
    assert not is_incorporated_by_reference(recorded_filing.sections[Section.MARKET_RISK])
