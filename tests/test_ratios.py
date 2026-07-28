"""Peer-relative ratios (spec seam 5, ADR-0009) — the tool eval's arithmetic.

`calculate_ratios` is the one tool that computes rather than reports, so this is where the
answer's numbers are either right or quietly wrong. Three things are asserted, in order of how
badly they fail:

1. **The peer set is `config.PEERS` and nothing else** (ADR-0009). A peer chosen here rather
   than read would be a comparison basis the golden set asserts against and the ADR forbids.
2. **A missing figure is excluded, never averaged as zero.** Two banks report no D/E at all and
   Ford no P/E, so this is the routine case; averaging an absence in as nought understates a
   peer mean by however many peers are missing.
3. **The spread travels with the mean.** Ford's two `autos` peers are TSLA at 286× and GM at
   37×; the mean of 162× describes neither, and a mean printed alone is the number a reader
   would quote.

Driven over the recorded quotes, so the arithmetic is checked against figures a real endpoint
served rather than against round numbers chosen to make the sums easy.
"""

from __future__ import annotations

import pytest

from finbrief.config import PEERS, UNIVERSE, PeerCluster
from finbrief.finance.quotes import Quote
from finbrief.finance.ratios import Unit, compare


def a_quote(ticker: str, **figures) -> Quote:
    """A quote with only the figures a test cares about; everything else absent."""
    return Quote.from_info(ticker, figures)


def metric(comparison, key: str):
    (found,) = [m for m in comparison.metrics if m.key == key]
    return found


# --- ADR-0009: the peer set --------------------------------------------------------------


def test_the_peer_set_is_configs_and_the_tool_does_not_choose_it(recorded_quotes):
    # The ADR's decision, at the seam it has to hold at. `tests/test_golden_set.py` asserts the
    # golden set's copy against the same map, so these two agree by construction rather than by
    # anybody keeping them in step.
    comparison = compare(recorded_quotes["F"], recorded_quotes)

    assert comparison.peers == PEERS["F"] == ("TSLA", "GM")
    assert comparison.n == 2
    assert comparison.cluster is PeerCluster.AUTOS


def test_the_company_is_never_its_own_peer(recorded_quotes):
    for company in UNIVERSE:
        comparison = compare(recorded_quotes[company.ticker], recorded_quotes)
        assert company.ticker not in comparison.peers


def test_every_universe_company_compares_against_at_least_two_peers(recorded_quotes):
    # ADR-0009 leans on this: every cluster holds at least three members, so no company is ever
    # compared against a "mean" of one. Asserted over the whole Universe rather than trusted,
    # because a two-member cluster would make the phrase "peer mean" a lie in one place only.
    for company in UNIVERSE:
        assert compare(recorded_quotes[company.ticker], recorded_quotes).n >= 2


def test_big_tech_compares_against_five(recorded_quotes):
    # The other n the ADR names, so the inline "mean of N peers" is exercised at both sizes.
    comparison = compare(recorded_quotes["NVDA"], recorded_quotes)

    assert comparison.n == 5
    assert set(comparison.peers) == {"AAPL", "MSFT", "AMZN", "GOOGL", "META"}


def test_an_out_of_universe_ticker_has_no_comparison_to_make():
    # The whitelist is enforced at the tool boundary, but the engine must not invent a cluster
    # for a ticker that has none — an empty peer set would silently become "mean of 0 peers".
    with pytest.raises(KeyError):
        compare(a_quote("SAP", trailingPE=20.0), {})


# --- the metrics ------------------------------------------------------------------------


def test_the_metrics_are_the_ones_the_plan_names(recorded_quotes):
    comparison = compare(recorded_quotes["NVDA"], recorded_quotes)

    assert [m.key for m in comparison.metrics] == [
        "trailing_pe",
        "debt_to_equity",
        "gross_margin",
        "operating_margin",
        "profit_margin",
    ]


def test_a_metric_carries_the_companys_own_figure(recorded_quotes):
    comparison = compare(recorded_quotes["LLY"], recorded_quotes)

    assert metric(comparison, "trailing_pe").value == 42.55615
    assert metric(comparison, "gross_margin").value == 0.82832


def test_the_peer_mean_is_the_mean_of_the_peers_that_reported(recorded_quotes):
    # The golden set's T5: Eli Lilly's P/E against JNJ and PFE.
    comparison = compare(recorded_quotes["LLY"], recorded_quotes)

    pe = metric(comparison, "trailing_pe")
    assert pe.peers_compared == ("JNJ", "PFE")
    assert pe.peer_mean == pytest.approx((30.888504 + 18.832062) / 2, rel=1e-9)


def test_a_peer_with_no_figure_is_excluded_rather_than_counted_as_zero(recorded_quotes):
    # GS is the only bank with a D/E, so JPM's comparison has exactly one peer figure in it —
    # and n stays 2, because the peer *set* is the cluster's. Counting the two absences as zero
    # would report a peer mean a third of the truth.
    comparison = compare(recorded_quotes["JPM"], recorded_quotes)

    leverage = metric(comparison, "debt_to_equity")
    assert comparison.n == 2, "the peer set is the cluster's, whatever the coverage"
    assert leverage.peers_compared == ("GS",)
    assert leverage.n_compared == 1
    assert leverage.peer_mean == pytest.approx(647.154 / 100)


def test_a_metric_no_peer_reported_has_no_mean_rather_than_a_zero_one():
    comparison = compare(
        a_quote("TSLA", trailingPE=286.0), {"F": a_quote("F"), "GM": a_quote("GM")}
    )

    pe = metric(comparison, "trailing_pe")
    assert pe.value == 286.0
    assert pe.peer_mean is None
    assert pe.peers_compared == ()


def test_the_companys_own_missing_figure_is_reported_missing_not_dropped(recorded_quotes):
    # Ford has no trailing P/E. The metric must still appear, with the peers' figures beside an
    # absent own value: "not reported" is the answer to the analyst's question, and dropping the
    # row would make the answer silently narrower than the question.
    comparison = compare(recorded_quotes["F"], recorded_quotes)

    pe = metric(comparison, "trailing_pe")
    assert pe.value is None
    assert pe.peers_compared == ("TSLA", "GM"), "the comparison is still worth making"


def test_a_peer_missing_from_the_quotes_is_reported_unavailable_not_silently_skipped(
    recorded_quotes,
):
    # One peer's fetch failed with nothing cached, so the tool has no quote for it. The
    # comparison still runs — with the peers it has — and says which it could not reach, because
    # "mean of 1 of 2 peers" and "mean of 2 peers" are different claims.
    without_gm = {t: q for t, q in recorded_quotes.items() if t != "GM"}

    comparison = compare(recorded_quotes["F"], without_gm)

    assert comparison.peers == ("TSLA", "GM"), "still the cluster's set, per ADR-0009"
    assert comparison.unavailable == ("GM",)
    assert metric(comparison, "trailing_pe").peers_compared == ("TSLA",)


def test_the_spread_travels_with_the_mean(recorded_quotes):
    # Ford's two autos peers are TSLA at 286× and GM at 37×. A mean of 162× describes neither,
    # and it is the number a reader would quote. So the low and high go out with it — the
    # cheapest honest answer to the outlier problem an arithmetic mean of multiples has.
    comparison = compare(recorded_quotes["F"], recorded_quotes)

    pe = metric(comparison, "trailing_pe")
    assert pe.peer_low == 36.88136
    assert pe.peer_high == 286.31482
    assert pe.peer_mean == pytest.approx((36.88136 + 286.31482) / 2, rel=1e-9)


def test_a_single_peer_figure_has_a_spread_of_itself(recorded_quotes):
    leverage = metric(compare(recorded_quotes["JPM"], recorded_quotes), "debt_to_equity")

    assert leverage.peer_low == leverage.peer_high == leverage.peer_mean


def test_a_metric_with_no_peer_figures_has_no_spread():
    comparison = compare(a_quote("TSLA", trailingPE=286.0), {"F": a_quote("F")})

    pe = metric(comparison, "trailing_pe")
    assert pe.peer_low is None and pe.peer_high is None


# --- units and formatting ---------------------------------------------------------------


def test_a_multiple_and_a_percentage_are_formatted_by_their_own_unit(recorded_quotes):
    # One renderer, shared by the tool's text and the UI card: a figure formatted twice is a
    # figure the model and the reader can disagree about.
    comparison = compare(recorded_quotes["NVDA"], recorded_quotes)

    assert metric(comparison, "trailing_pe").unit is Unit.MULTIPLE
    assert metric(comparison, "gross_margin").unit is Unit.PERCENT
    assert Unit.MULTIPLE.format(31.644121) == "31.6x"
    assert Unit.PERCENT.format(0.74144995) == "74.1%"


def test_an_absent_figure_formats_as_words_and_never_as_a_number():
    # The whole point of keeping absences: a `0.0%` gross margin on a bank's card is a claim
    # nobody made, and `None` printed raw reads as a bug.
    assert Unit.PERCENT.format(None) == "not reported"
    assert Unit.MULTIPLE.format(None) == "not reported"


def test_a_negative_figure_keeps_its_sign(recorded_quotes):
    # Ford's net margin is −3.2%, which is the most decision-relevant number on its card.
    assert Unit.PERCENT.format(recorded_quotes["F"].profit_margin) == "-3.2%"
