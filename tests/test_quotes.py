"""`Quote.from_info` against real recorded market data (spec seam 5).

What is asserted here is the part of the quote path that has logic in it: which of yfinance's
overlapping fields is read, what its units are turned into, and — most of it — how an absence is
carried. The `yfinance.Ticker` call itself is deliberately uncovered; `scripts/record_market_
fixtures.py` states that boundary and why it is where it is.

Recorded rather than invented, because the cases that matter are ones nobody would have thought
to write: two banks reporting a gross margin of exactly `0.0`, Ford with no trailing P/E, Ford's
D/E arriving as `425.544` when the ratio is 4.26×. Every fixture here is a real response.
"""

from __future__ import annotations

from finbrief.config import TICKERS
from finbrief.finance.quotes import Close, Quote


def test_every_universe_company_has_a_recorded_quote(recorded_quotes):
    # The fixture set is the Universe's, so a 16th company fails here until it is recorded —
    # rather than in whichever ratio test happened to name it.
    assert set(recorded_quotes) == set(TICKERS)


def test_a_quote_carries_the_price_the_name_and_the_currency(recorded_quotes):
    nvda = recorded_quotes["NVDA"]

    assert nvda.ticker == "NVDA"
    assert nvda.name == "NVIDIA Corporation"
    assert nvda.currency == "USD"
    assert nvda.price == 196.51
    assert nvda.previous_close == 206.84
    assert nvda.market_cap == 4759668391936


def test_debt_to_equity_is_a_ratio_and_not_yfinances_percentage_points(recorded_quotes):
    # The conversion the boundary exists for. Ford's `debtToEquity` is `425.544`, i.e. 4.26× —
    # cross-checked at record time against `totalDebt` 159.5bn over ~36.8bn of equity. Reported
    # unconverted it would say Ford is levered 425 times, and the golden set's T2 asks exactly
    # this question about exactly this filer.
    assert recorded_quotes["F"].debt_to_equity == 425.544 / 100
    assert 4.2 < recorded_quotes["F"].debt_to_equity < 4.3
    assert recorded_quotes["NVDA"].debt_to_equity == 6.555 / 100


def test_margins_stay_the_fractions_yfinance_reports(recorded_quotes):
    # Not converted, because `finance/ratios.py` averages them as fractions and the percent sign
    # is the surface's business. Converting here and formatting there would be two conversions.
    nvda = recorded_quotes["NVDA"]

    assert nvda.gross_margin == 0.74144995
    assert nvda.profit_margin == 0.62966
    assert nvda.operating_margin == 0.65596


def test_a_missing_figure_stays_missing_rather_than_becoming_zero(recorded_quotes):
    # The absence that would be a lie as a number. Ford genuinely has no trailing P/E in this
    # snapshot; JPM and BAC have no `debtToEquity` at all. A zero D/E on a bank's card says it
    # carries no leverage.
    assert recorded_quotes["F"].trailing_pe is None
    assert recorded_quotes["JPM"].debt_to_equity is None
    assert recorded_quotes["BAC"].debt_to_equity is None
    # And GS *does* report one, so the absence is per filer and not a rule about banks.
    assert recorded_quotes["GS"].debt_to_equity == 647.154 / 100


def test_a_gross_margin_of_exactly_zero_is_read_as_not_reported(recorded_quotes):
    # The one coercion in the module, and the measurement that justifies it: JPM reports a gross
    # margin of exactly 0.0 beside a 50% operating margin, which is arithmetically impossible —
    # gross profit is at least operating profit. So the field is unpopulated and `0.0` is how
    # this endpoint spells that. Averaged in, it would drag the `banks` peer mean toward zero.
    assert recorded_quotes["JPM"].gross_margin is None
    assert recorded_quotes["BAC"].gross_margin is None
    assert recorded_quotes["JPM"].operating_margin == 0.50394, "only the impossible one goes"
    assert recorded_quotes["GS"].gross_margin == 0.82075995, "and only where it is missing"


def test_a_genuine_negative_margin_survives_the_zero_rule(recorded_quotes):
    # The case a laxer rule (`if not value`) would have dropped along with the zeros. Ford's net
    # margin is −3.2%, which is the single most decision-relevant figure on its card.
    assert recorded_quotes["F"].profit_margin == -0.03216


def test_the_change_percent_is_the_endpoints_own_when_it_has_one(recorded_quotes):
    assert recorded_quotes["NVDA"].change_percent == -4.9942


def test_the_change_percent_is_recomputed_when_the_endpoint_omits_it():
    # An absent change reads as "flat" on the card, which is a claim; the two prices needed to
    # compute it are already in hand, so there is no reason to make it.
    quote = Quote.from_info("TSLA", {"currentPrice": 110.0, "previousClose": 100.0})

    assert quote.change_percent == 10.0


def test_a_change_percent_is_not_invented_from_a_missing_previous_close():
    # Recomputation must not become fabrication: with nothing to compare against there is no
    # change, and a zero previous close would divide by nought rather than mean "up infinitely".
    assert Quote.from_info("TSLA", {"currentPrice": 110.0}).change_percent is None
    assert (
        Quote.from_info("TSLA", {"currentPrice": 110.0, "previousClose": 0}).change_percent
        is None
    )


def test_either_of_yfinances_two_price_fields_will_do():
    # Which of them is populated depends on the venue and the time of day, so both are read.
    assert Quote.from_info("MSFT", {"regularMarketPrice": 389.1}).price == 389.1
    assert Quote.from_info("MSFT", {"currentPrice": 389.1}).price == 389.1


def test_a_nan_is_an_absence_and_a_string_is_not_a_number():
    # `.info` is assembled from several scraped payloads, so a missing figure arrives as a
    # missing key, a NaN or a string depending on which payload was short. All three are `None`.
    quote = Quote.from_info(
        "TSLA", {"trailingPE": float("nan"), "marketCap": "1.2T", "grossMargins": True}
    )

    assert quote.trailing_pe is None
    assert quote.market_cap is None
    assert quote.gross_margin is None, "a bool is not a margin, however much it is an int"


def test_a_quote_with_no_name_falls_back_to_its_ticker():
    # The card labels itself with this, so it may not be empty.
    assert Quote.from_info("PFE", {}).name == "PFE"


def test_the_price_history_is_oldest_first_so_a_chart_needs_no_sorting(recorded_quotes):
    closes = recorded_quotes["NVDA"].closes

    assert len(closes) == 20
    assert all(isinstance(close, Close) for close in closes)
    assert [c.date for c in closes] == sorted(c.date for c in closes)
    assert closes[-1].date > closes[0].date
