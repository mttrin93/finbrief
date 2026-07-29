"""Tool-selection accuracy, scored against a scripted selection layer.

Seam: `run_case`'s injected `ask`, so what a *live* model does is the only thing this cannot
test — and every scoring rule around it is. The rules that matter are the ones about not
blaming the model
for the wrong thing: a failed turn is excluded rather than counted as a selection error, and a
control case fails on a tool it *called* rather than on one it missed.
"""

from __future__ import annotations

import pytest

from finbrief.evaluation.loader import load_golden_set
from finbrief.evaluation.tool_eval import (
    CONTROL_CASES,
    ToolCase,
    ToolReport,
    cases_from,
    render,
    run_all,
    run_case,
)
from finbrief.tools.finance import FINANCE_TOOL_NAMES, RATIOS_TOOL_NAME, STOCK_TOOL_NAME
from finbrief.tools.search_filings import TOOL_NAME as SEARCH_TOOL_NAME


def asker(script: dict[str, list[tuple[str, str | None]]], *, raises: str | None = None):
    """An `ask` that replays a scripted set of tool calls per question."""

    def ask(question: str, calls: list[tuple[str, str | None]]) -> None:
        if raises == question:
            raise RuntimeError("Connection error.")
        calls.extend(script.get(question, []))

    return ask


A_CASE = ToolCase(
    id="T2",
    question="Is Ford expensive right now?",
    expected=frozenset({RATIOS_TOOL_NAME, SEARCH_TOOL_NAME}),
    args={"ticker": "F"},
)


def test_a_case_passes_when_every_expected_tool_fired_with_the_right_ticker():
    outcome = run_case(
        A_CASE,
        ask=asker({A_CASE.question: [(SEARCH_TOOL_NAME, None), (RATIOS_TOOL_NAME, "F")]}),
    )

    assert outcome.passed is True
    assert outcome.reason == "as expected"


def test_a_missing_expected_tool_fails_and_says_which():
    outcome = run_case(A_CASE, ask=asker({A_CASE.question: [(SEARCH_TOOL_NAME, None)]}))

    assert outcome.passed is False
    assert RATIOS_TOOL_NAME in outcome.reason


def test_the_right_tool_with_the_wrong_ticker_fails():
    # The golden set records the arguments the call must carry, not just the tool's name — a
    # quote for the wrong company is a wrong answer with a right-looking tool trace.
    outcome = run_case(
        A_CASE,
        ask=asker({A_CASE.question: [(SEARCH_TOOL_NAME, None), (RATIOS_TOOL_NAME, "GM")]}),
    )

    assert outcome.passed is False
    assert "not called with F" in outcome.reason


def test_search_filings_is_not_argument_checked():
    # It takes a query, not a ticker, so there is no ticker to check — and `Step` carries none.
    outcome = run_case(
        A_CASE,
        ask=asker({A_CASE.question: [(SEARCH_TOOL_NAME, None), (RATIOS_TOOL_NAME, "F")]}),
    )

    assert outcome.wrong_args == ()


def test_a_control_fails_on_a_tool_it_called_rather_than_one_it_missed():
    # The shape a positive-only case set cannot express: reaching for a quote on a risk-factors
    # question is an error, and nothing about "did the expected tool fire" would notice.
    control = CONTROL_CASES[0]

    outcome = run_case(
        control,
        ask=asker({control.question: [(SEARCH_TOOL_NAME, None), (STOCK_TOOL_NAME, "TSLA")]}),
    )

    assert outcome.passed is False
    assert "forbidden" in outcome.reason


def test_a_control_passes_when_it_only_searched():
    control = CONTROL_CASES[0]

    outcome = run_case(control, ask=asker({control.question: [(SEARCH_TOOL_NAME, None)]}))

    assert outcome.passed is True


def test_a_failed_turn_is_absent_rather_than_a_selection_failure():
    # An outage is not a selection error. Counting it as one would blame the model for the
    # network and quietly drag the rate down by the number of 429s the run met.
    outcome = run_case(A_CASE, ask=asker({}, raises=A_CASE.question))

    assert outcome.passed is None
    assert "turn failed" in outcome.reason


def test_a_failed_turn_is_excluded_from_the_rate_and_counted_separately():
    report = ToolReport(
        outcomes=(
            run_case(
                A_CASE,
                ask=asker(
                    {A_CASE.question: [(SEARCH_TOOL_NAME, None), (RATIOS_TOOL_NAME, "F")]}
                ),
            ),
            run_case(A_CASE, ask=asker({}, raises=A_CASE.question)),
        )
    )

    assert report.accuracy() == pytest.approx(1.0)
    assert len(report.scored) == 1
    assert report.failed_turns == 1


def test_accuracy_over_no_scored_cases_is_absent_rather_than_zero():
    report = ToolReport(outcomes=(run_case(A_CASE, ask=asker({}, raises=A_CASE.question)),))

    assert report.accuracy() is None
    assert "not measured" in render(report)


def test_accuracy_can_exclude_the_controls():
    # The controls probe a different thing, so the artifact reports both denominators.
    passing = run_case(
        A_CASE,
        ask=asker({A_CASE.question: [(SEARCH_TOOL_NAME, None), (RATIOS_TOOL_NAME, "F")]}),
    )
    failing_control = run_case(
        CONTROL_CASES[0],
        ask=asker({CONTROL_CASES[0].question: [(STOCK_TOOL_NAME, "TSLA")]}),
    )
    report = ToolReport(outcomes=(passing, failing_control))

    assert report.accuracy() == pytest.approx(0.5)
    assert report.accuracy(include_controls=False) == pytest.approx(1.0)


def test_the_pairing_rate_counts_both_tools_and_names_its_denominator():
    # #9's AC-1 hole. `calculate_ratios` alone satisfies `tool_expectation` and does *not*
    # satisfy the prompt, which is exactly why this is its own rate.
    both = run_case(
        A_CASE,
        ask=asker(
            {
                A_CASE.question: [
                    (SEARCH_TOOL_NAME, None),
                    (STOCK_TOOL_NAME, "F"),
                    (RATIOS_TOOL_NAME, "F"),
                ]
            }
        ),
    )
    ratios_only = run_case(
        A_CASE,
        ask=asker({A_CASE.question: [(SEARCH_TOOL_NAME, None), (RATIOS_TOOL_NAME, "F")]}),
    )

    assert ToolReport(outcomes=(both, ratios_only)).pairing_rate == (1, 2)
    # And the one that skipped the quote still *passes* selection, which is the hole itself.
    assert ratios_only.passed is True


def test_the_cases_come_from_the_golden_set_rather_than_being_retyped():
    golden = load_golden_set()

    cases = cases_from(golden)

    assert len(cases) == 10, "seven tool-augmented rows plus three controls (PLAN §7's ten)"
    scripted = [case for case in cases if not case.control]
    assert len(scripted) == 7
    for case in scripted:
        row = golden.row(case.id)
        assert case.question == row.question
        assert row.tool_expectation.name in case.expected
        assert SEARCH_TOOL_NAME in case.expected, "every row needs retrieval to reach its half"
        assert case.args["ticker"] == row.tool_expectation.args["ticker"]


def test_the_controls_forbid_the_finance_tools_they_are_controls_for():
    assert CONTROL_CASES[0].forbidden == FINANCE_TOOL_NAMES
    assert CONTROL_CASES[1].forbidden == FINANCE_TOOL_NAMES
    assert all(case.control for case in CONTROL_CASES)


def test_the_rendered_section_states_both_denominators_and_the_pairing():
    report = run_all(
        cases_from(load_golden_set())[:1],
        ask=asker({}),
    )

    rendered = render(report)

    assert "Tool-selection accuracy" in rendered
    assert "Valuation pairing" in rendered
    assert "Measured, not enforced" in rendered
