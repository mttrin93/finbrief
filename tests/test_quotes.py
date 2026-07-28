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

import pytest

from finbrief import config
from finbrief.config import TICKERS
from finbrief.finance import quotes as quotes_module
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


# --------------------------------------------------------------------------------------
# The fetch timeout, which was documented for a release before it was applied (#9 review)
# --------------------------------------------------------------------------------------


class _RecordingSession:
    """Stands in for a backend `Session`, recording the timeout each request was given.

    A real `curl_cffi.Session` is not needed and would not help: what is under test is the
    clamp `_build_bounded_session` installs, and the clamp is a wrapper around whatever
    `request` it found. Asserting against a recorder is what makes the *timeout value*
    visible — with a real session the assertion could only be "it did not hang", which is
    the assertion that passed for a release while the ceiling was 30.
    """

    def __init__(self) -> None:
        self.timeouts: list[object] = []

    def request(self, *args, **kwargs):
        self.timeouts.append(kwargs.get("timeout"))
        return "response"

    # yfinance's own call shape, to prove the clamp survives the delegation `get` performs.
    def get(self, url, **kwargs):
        return self.request(method="GET", url=url, **kwargs)


def _bound(recorder):
    """Apply the production clamp to `recorder`, by the production code path."""
    import sys
    import types

    # `_build_bounded_session` imports `yfinance._http` for `new_session`. Stubbing that one
    # attribute runs the real clamp over the recorder without importing yfinance — which the
    # hermetic contract would allow but which would pull in a scraper for no reason.
    module = types.ModuleType("yfinance._http")
    module.new_session = lambda: recorder  # type: ignore[attr-defined]
    package = sys.modules.get("yfinance") or types.ModuleType("yfinance")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "yfinance", package)
        patch.setitem(sys.modules, "yfinance._http", module)
        patch.setattr(package, "_http", module, raising=False)
        return quotes_module._build_bounded_session()


def test_a_request_with_no_timeout_gets_the_configured_ceiling():
    # The case that was broken: yfinance names `timeout=30` at every call site, so nothing was
    # applying `FETCH_TIMEOUT_SECONDS` to the quote path at all. The clamp is what makes the
    # documented number real.
    recorder = _RecordingSession()
    session = _bound(recorder)

    session.get("https://query1.finance.yahoo.com/v8/finance/chart/NVDA")

    assert recorder.timeouts == [config.FETCH_TIMEOUT_SECONDS]


def test_yfinances_own_thirty_second_timeout_is_clamped_down():
    # Verbatim the value `YfData.get`/`.post`/`.get_raw_json` default to and forward. An
    # explicit keyword beats a session default, so this cannot be configured away and has to be
    # clamped.
    recorder = _RecordingSession()
    session = _bound(recorder)

    session.request(method="GET", url="https://query1.finance.yahoo.com/", timeout=30)

    assert recorder.timeouts == [config.FETCH_TIMEOUT_SECONDS]
    assert recorder.timeouts[0] < 30, "the ceiling has to bind, not merely be offered"


def test_a_caller_asking_for_less_than_the_ceiling_keeps_it():
    # `min`, not assignment: the ceiling is a maximum, and raising a caller's 1s to 15s would
    # make this a *worse* guarantee than none.
    recorder = _RecordingSession()
    session = _bound(recorder)

    session.request(method="GET", url="https://query1.finance.yahoo.com/", timeout=1)

    assert recorder.timeouts == [1.0]


def test_curl_cffis_unspecified_sentinel_is_not_mistaken_for_a_number():
    # `curl_cffi` spells "no timeout given" as its own `NOT_SET` object, neither `None` nor a
    # number. Compared naively it would sail past the clamp; `min` against it would raise.
    class NotSet:
        pass

    recorder = _RecordingSession()
    session = _bound(recorder)

    session.request(method="GET", url="https://query1.finance.yahoo.com/", timeout=NotSet())

    assert recorder.timeouts == [config.FETCH_TIMEOUT_SECONDS]


def test_the_bounded_session_is_built_once_per_process():
    # `build_once`, not a fresh session per fetch: yfinance's `YfData` is a singleton whose
    # `_set_session` two concurrent builders would race, and a new session per call would
    # re-negotiate the cookie and crumb every time.
    assert hasattr(quotes_module.bounded_session, "cache_clear")
    assert hasattr(quotes_module.bounded_session, "cache_info")


def test_the_worst_case_the_timeout_permits_is_derived_and_survivable():
    # The comment on `FETCH_TIMEOUT_SECONDS` used to imply it bounded a whole fetch. It bounds
    # one request: two per attempt, `FETCH_ATTEMPTS` attempts, all under `TimedCache`'s lock.
    # honest number is derived in `config` so the prose cannot drift from it — and pinned here
    # because "about 91 seconds" is a claim in that prose.
    assert pytest.approx(91.5) == config.QUOTE_FETCH_WORST_CASE_SECONDS
    assert config.QUOTE_FETCH_WORST_CASE_SECONDS < 120, (
        "the quote lock must not be holdable for longer than a user waits before reloading"
    )
