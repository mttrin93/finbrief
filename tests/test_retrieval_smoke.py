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


def test_the_expectation_labels_match_the_committed_artifact():
    # `docs/verification/retrieval-smoke.md` is the output of a paid run that CI cannot
    # reproduce, so a change to how a row renders silently invalidates committed evidence.
    # Extracting `expectation` had to leave every Expects cell byte-identical; this is what
    # says so.
    # Scoped to the Checks table: the distance-ranges table below it has the same column
    # count and the same question in column 1, so an unscoped parse reads "yes" as an
    # expectation and passes for the wrong reason.
    _, _, rest = REPORT.read_text(encoding="utf-8").partition("## Checks")
    checks_table, _, _ = rest.partition("## Observed distance ranges")
    rows = {
        cells[1]: cells[2]
        for line in checks_table.splitlines()
        if line.startswith("| ")
        and len(cells := [c.strip() for c in line.split(" | ")]) == 5
        and cells[0] != "| #"  # the header row, same shape as the data
    }
    assert len(rows) == len(SMOKE_QUERIES), f"parsed {len(rows)} rows from the Checks table"

    for query in SMOKE_QUERIES:
        assert rows[query.question] == query.expectation, query.question


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

    controls = [query for query in SMOKE_QUERIES if query.expect_ticker is None]
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
    return render_smoke_report(
        checks,
        generated="2026-07-27 12:00 UTC · `scripts/retrieval_smoke.py`",
        strategy=RetrievalStrategy.VECTOR,
        k=5,
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


def test_the_report_defers_the_distance_floor_to_the_phase_4_7_evidence():
    # Constraint 2 from the ticket: the bands are calibration data for a floor that is
    # chosen with the A/B evidence, not here — and the code that would use it says so too.
    report = render([a_check(), a_check(query=CONTROL)])

    assert "calibration" in report.lower()
    assert "NO_CONTEXT_FALLBACK" in report
    assert "Phase 4" in report and "Phase 7" in report


def test_the_report_lists_every_retrieved_chunk_so_a_reader_can_check_it():
    report = render([a_check()])

    assert "0001628280-26-003952:Item 1A:1" in report
    assert "TSLA Item 1A body 1." in report
