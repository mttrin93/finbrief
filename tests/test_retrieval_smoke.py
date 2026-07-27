"""The retrieval smoke check's verdicts and its committed report.

`scripts/retrieval_smoke.py` is the non-hermetic half — it embeds five hand-written queries
against the ingested KB and pays for it. What is testable without a key is everything that
decides *what the report says*: whether a query's top hit met its expectation, how a control
query is scored (it is not), and that the rendered artifact carries its own scope caveat and
the distance-floor deferral. Those are asserted here; PLAN.md §Phase 2 owns the run itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import a_context

from finbrief.config import ITEM_7A_POINTER_FILERS, RetrievalStrategy
from finbrief.ingestion.model import Section
from finbrief.retrieval.smoke import (
    SMOKE_QUERIES,
    SmokeCheck,
    SmokeQuery,
    Verdict,
    render_smoke_report,
)


def a_check(*, query=None, contexts=None):
    query = query or SmokeQuery(
        question="What are the main risk factors for Tesla?",
        expect_ticker="TSLA",
        expect_section=Section.RISK_FACTORS,
        why="demo step 1",
    )
    if contexts is None:
        contexts = (a_context(1, distance=0.62), a_context(2, distance=0.68))
    return SmokeCheck(query=query, contexts=contexts)


#: The last run's committed output. Read, never written — `scripts/retrieval_smoke.py` owns
#: it, and the suite must not invoke that script (CLAUDE.md).
REPORT = Path(__file__).parents[1] / "docs" / "verification" / "retrieval-smoke.md"

CONTROL = SmokeQuery(
    question="What is Nestle's dividend policy?",
    expect_ticker=None,
    expect_section=None,
    why="out-of-KB control",
)


# --- Verdicts ---------------------------------------------------------------------------


def test_a_query_whose_top_hit_matches_its_expectation_passes():
    assert a_check().outcome is Verdict.PASS


def test_a_query_that_retrieved_the_right_company_from_the_wrong_section_fails():
    # The check that earns its place: an `Item 1A` question answered from `Item 7` is a
    # plausible-looking answer grounded in the wrong part of the filing.
    check = a_check(contexts=(a_context(1, section=Section.MDA),))

    assert check.outcome is Verdict.FAIL
    assert "Item 7" in check.verdict and "Item 1A" in check.verdict


def test_a_query_that_retrieved_the_wrong_company_fails():
    check = a_check(contexts=(a_context(1, ticker="F"),))

    assert check.outcome is Verdict.FAIL
    assert "F" in check.verdict


def test_a_query_that_retrieved_nothing_fails_rather_than_raising():
    check = a_check(contexts=())

    assert check.outcome is Verdict.FAIL
    assert "nothing" in check.verdict.lower()


def test_the_control_query_is_recorded_and_never_scored():
    # Constraint from the ticket: the out-of-KB control exists to *calibrate* the distance
    # band, and the floor it would inform is deliberately unimplemented. Passing or failing
    # it here would be exactly the quality claim this artifact must not make.
    check = a_check(query=CONTROL, contexts=(a_context(1, ticker="PFE", distance=1.07),))

    assert check.outcome is Verdict.CONTROL
    assert check.is_control
    assert "control" in check.verdict.lower()


def test_the_control_cannot_be_scored_by_a_truthiness_test():
    # Why `Verdict` replaced `passed: bool | None`. Every `if check.passed` and every
    # `sum(1 for c in checks if c.passed)` counted the control as a failure, and a caller
    # had to know to write `is False`. There is no falsy state left to trip over: the
    # control is a third kind of result, not an unknown score.
    control = a_check(query=CONTROL, contexts=(a_context(1, ticker="PFE", distance=1.07),))
    failing = a_check(contexts=(a_context(1, ticker="F"),))

    assert control.outcome not in (Verdict.PASS, Verdict.FAIL)
    assert [c.outcome is Verdict.FAIL for c in (control, failing)] == [False, True]


def test_a_querys_expectation_reads_as_a_citation_or_an_em_dash():
    assert a_check().query.expectation == "TSLA Item 1A"
    assert CONTROL.expectation == "—", "the control expects nothing, and says so"


def test_a_half_filled_expectation_is_a_control_everywhere_or_nowhere():
    # `outcome` used to test `expect_ticker` alone while `expectation` tested both fields, so
    # a query naming a ticker with no Section was scored FAIL forever *and* rendered as
    # expecting `—`: a verdict no reader could act on. One predicate on `SmokeQuery` decides
    # it now, and this is what keeps the two halves from parting again (issue #5 review).
    half_filled = SmokeQuery(
        question="Where does JPMorgan discuss market risk?",
        expect_ticker="JPM",
        expect_section=None,
        why="a query somebody half-wrote",
    )
    check = SmokeCheck(query=half_filled, contexts=(a_context(1, ticker="JPM"),))

    assert half_filled.is_control
    assert check.outcome is Verdict.CONTROL
    assert half_filled.expectation == "—"


def committed_check_rows() -> dict[str, dict[str, str]]:
    """The committed artifact's Checks table, by question.

    Scoped to that table: the distance-ranges table below it has the same column count and
    the same question in column 1, so an unscoped parse reads "yes" as an expectation and
    passes for the wrong reason.
    """
    _, _, rest = REPORT.read_text(encoding="utf-8").partition("## Checks")
    checks_table, _, _ = rest.partition("## Observed distance ranges")
    rows = {}
    for line in checks_table.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        # Numbered rows only: the header (`#`) and its separator (`---`) have the same
        # column count as the data.
        if len(cells) != 5 or not cells[0].isdigit():
            continue
        rows[cells[1]] = {"expects": cells[2], "verdict": cells[3], "why": cells[4]}
    return rows


def test_the_rendered_columns_match_the_committed_artifact():
    # `docs/verification/retrieval-smoke.md` is the output of a paid run that CI cannot
    # reproduce, so a change to how a row renders silently invalidates committed evidence.
    #
    # `why` is pinned as well as `expects` because it is no longer a literal: query 3's reads
    # its filer count from `config.ITEM_7A_SECTION_FILERS`, and that string lands verbatim in
    # the artifact. Before the regeneration this could not be asserted — the committed file
    # still said "the nine filers" while the code had moved to "the 9 filers", which is
    # exactly the drift this now catches (issue #5 review).
    rows = committed_check_rows()
    assert len(rows) == len(SMOKE_QUERIES), f"parsed {len(rows)} rows from the Checks table"

    for query in SMOKE_QUERIES:
        row = rows[query.question]
        assert row["expects"] == query.expectation, query.question
        assert row["why"] == query.why, query.question


def test_a_check_reports_the_distance_band_it_observed():
    check = a_check(contexts=(a_context(1, distance=0.6224), a_context(2, distance=0.681)))

    assert check.nearest == pytest.approx(0.6224)
    assert check.furthest == pytest.approx(0.681)


def test_an_empty_check_has_no_distance_band():
    check = a_check(contexts=())

    assert check.nearest is None
    assert check.furthest is None


# --- The five queries -------------------------------------------------------------------


def test_the_five_queries_cover_the_kb_shapes_a_reader_would_doubt():
    # PLAN.md §Phase 2 asks for five hand-written queries. Which five is the whole value:
    # one per KB property that could silently be wrong — a pointer filer's market risk
    # living under Item 7 (ADR-0007 §4), a filer whose fiscal year is not 2025, and an
    # out-of-KB question with nothing to ground it.
    assert len(SMOKE_QUERIES) == 5

    # `query.is_control`, not `expect_ticker is None`: that was the third spelling of this
    # predicate, and the review that collapsed the other two left it here. A sixth query
    # filled in halfway — a ticker with no Section — is a control to `SmokeQuery` and to the
    # report, and would have been counted as a scored check by this line alone.
    controls = [query for query in SMOKE_QUERIES if query.is_control]
    assert len(controls) == 1, "exactly one out-of-KB control"

    market_risk = [
        query
        for query in SMOKE_QUERIES
        if query.expect_ticker in ITEM_7A_POINTER_FILERS and query.expect_section is Section.MDA
    ]
    assert market_risk, "a pointer filer's market-risk question must expect Item 7"

    sections = {query.expect_section for query in SMOKE_QUERIES}
    assert Section.MARKET_RISK in sections, "and one company that does have an Item 7A"


# --- The committed report ---------------------------------------------------------------


def render(checks, **kwargs) -> str:
    # `setdefault` rather than a keyword default, so a test can override the pinned
    # configuration the same way it overrides any other argument.
    kwargs.setdefault("strategy", RetrievalStrategy.VECTOR)
    kwargs.setdefault("translate", False)
    kwargs.setdefault("k", 5)
    return render_smoke_report(
        checks,
        generated="2026-07-27 12:00 UTC · `scripts/retrieval_smoke.py`",
        embedding_model="openai/text-embedding-3-small",
        **kwargs,
    )


def test_the_report_states_it_is_not_an_evaluation():
    # The one thing this file must never be mistaken for. ADR-0002's stratified,
    # source-separated golden set (#4, T9) is the measurement artifact of record.
    report = render([a_check(), a_check(query=CONTROL)])

    assert "not an evaluation" in report.lower()
    assert "ADR-0002" in report
    assert "#4" in report


def test_the_report_records_the_run_that_produced_it():
    report = render([a_check()])

    assert "scripts/retrieval_smoke.py" in report
    assert "2026-07-27 12:00 UTC" in report
    assert "openai/text-embedding-3-small" in report, "an index is only searchable by its model"
    assert "`vector`" in report


def test_the_report_names_both_retrieval_switches_and_not_just_the_strategy():
    # `scripts/retrieval_smoke.py` pins this check to `vector` with translation *off* and
    # claims the report says so, precisely so nobody reads these distances as the shipping
    # default's (`hybrid + translation`). Naming the strategy alone stated half the
    # configuration, which is the misreading the claim exists to prevent (issue #6 review).
    assert "query translation **off**" in render([a_check()])
    assert "query translation **on**" in render([a_check()], translate=True)


def test_the_report_tallies_the_checks_and_counts_the_control_separately():
    report = render(
        [a_check(), a_check(contexts=(a_context(1, ticker="F"),)), a_check(query=CONTROL)]
    )

    assert "1/2 check(s) passed" in report
    assert "1 control recorded" in report


def test_the_report_shows_each_querys_expectation_top_hit_and_distance_band():
    report = render([a_check()])

    assert "What are the main risk factors for Tesla?" in report
    assert "TSLA 10-K FY2025, Item 1A" in report
    assert "0.6200" in report or "0.62" in report


def test_the_report_defers_the_distance_floor_to_the_phase_7_evidence():
    # Constraint 2 from the ticket: the bands are calibration data for a floor that is
    # chosen with the A/B evidence, not here — and the code that would use it says so too.
    # Phase 7, not "Phase 4 and Phase 7": Phase 4 shipped the strategy and deliberately
    # produced no A/B or RAGAs numbers (ADR-0004 amendment §7), so a report still promising
    # its evidence names a phase that has already closed without any (issue #6 review).
    report = render([a_check(), a_check(query=CONTROL)])

    assert "calibration" in report.lower()
    assert "NO_CONTEXT_FALLBACK" in report
    assert "Phase 7" in report
    assert "evidence from Phase 4" not in report


def test_the_report_lists_every_retrieved_chunk_so_a_reader_can_check_it():
    report = render([a_check()])

    assert "0001628280-26-003952:Item 1A:1" in report
    assert "TSLA Item 1A body 1." in report


def test_the_report_headline_announces_a_failure_a_reader_would_otherwise_scroll_past():
    # The artifact is committed, so its first screenful is what anyone reads. A run with a
    # wrong top hit that still said SMOKE PASSED would be worse than no evidence at all.
    passing = render([a_check(), a_check(query=CONTROL)])
    failing = render([a_check(contexts=(a_context(1, ticker="F"),)), a_check(query=CONTROL)])

    assert "**SMOKE PASSED**" in passing and "SMOKE FAILED" not in passing
    assert "**SMOKE FAILED**" in failing and "SMOKE PASSED" not in failing


def test_the_headline_does_not_treat_the_unscored_control_as_a_failure():
    # The regression `Verdict` was introduced to kill: a control counted as a failure makes
    # every healthy run render SMOKE FAILED.
    report = render([a_check(query=CONTROL, contexts=(a_context(1, ticker="PFE"),))])

    assert "**SMOKE PASSED**" in report
    assert "0/0 check(s) passed, 1 control recorded" in report


def test_a_query_that_retrieved_nothing_says_so_where_its_chunks_would_be():
    # Distinct from an omitted section: a reader checking a FAIL needs to see that the list
    # is empty rather than missing.
    report = render([a_check(contexts=())])

    assert "Nothing retrieved." in report
    assert "| — | — |" in report, "and an empty distance band, not a formatting error"


def test_an_excerpt_is_marked_elided_only_when_something_was_elided():
    # The ellipsis is a claim about the source. A short tail chunk printed whole was being
    # reported as truncated, which invites a reader to go looking for text that is all there.
    whole = render([a_check(contexts=(a_context(1, body="Short and complete."),))])
    clipped = render([a_check(contexts=(a_context(1, body="word " * 200),))])

    assert "> Short and complete.\n" in whole
    assert "…" not in whole.partition("## Retrieved chunks")[2]
    assert "…" in clipped.partition("## Retrieved chunks")[2]
