"""`Quote` — one company's live market figures, and the one place yfinance is called.

The engine half of `get_stock_data` and `calculate_ratios` (ADR-0009: peers resolve through
*this* cached path, so peer averaging adds no API surface of its own). Nothing above this module
imports yfinance, for the same reason `retrieval/vectorstore.py` is the only place Chroma is
opened: a second caller is a second cache to miss.

**Units are normalised here, once, because yfinance's are not the ones a note quotes.**
`debtToEquity` comes back in **percentage points** — Ford reads `425.544`, which is 4.26×
(`totalDebt` 159.5bn over ~36.8bn of equity, cross-checked at record time) — while the margins
come back as fractions. A ratio and a percentage that look alike is how a brief ends up
reporting Ford's leverage as 425×, so the conversion happens at the boundary and `Quote`'s
fields have stated units.

**Every figure is optional, and that is measured rather than defensive.** In the recorded
fixtures JPM and BAC report no `debtToEquity` at all and Ford reports no `trailingPE`, so `None`
is a routine value here and not an error path. What matters is that it stays `None`: a missing
D/E rendered as `0.0` says a bank carries no leverage, which is the most wrong sentence this
project could emit.

**And one figure is worse than optional — it arrives as a zero that is not a measurement.**
JPM and BAC report `grossMargins` of exactly `0.0` beside operating margins of 50% and 38%
(`tests/fixtures/market/`). Gross profit is by construction at least operating profit, so a
gross margin of precisely zero next to a positive operating margin is arithmetically
impossible: the field is unpopulated and the endpoint spells that `0.0`. Left alone it would
drag a peer mean down and print "gross margin 0.0%" on a bank's card, so `_margin` below reads
an exact zero as absent. Narrowly — **exactly** zero, and margins only: Ford's `profitMargins`
of `-0.032` is a real negative and survives.

The fetch is wrapped in `finance/cache.py`'s TTL + retry + stale-fallback policy; `quote()` is
the entry point everything else uses, and `QUOTE_CACHE` is the process-level handle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from finbrief.config import (
    FETCH_ATTEMPTS,
    FETCH_BACKOFF_SECONDS,
    HISTORY_DAYS,
    HISTORY_INTERVAL,
    QUOTE_TTL_SECONDS,
)
from finbrief.finance.cache import Fetched, TimedCache

#: yfinance's `period` string for `config.HISTORY_DAYS` — `30` → `"30d"`. Derived rather than
#: typed beside the day count, so the window the model is told about
#: (`GET_STOCK_DATA_DESCRIPTION`) and the window actually requested are the same number (issue
#: #9 review).
HISTORY_PERIOD = f"{HISTORY_DAYS}d"


@dataclass(frozen=True, slots=True)
class Close:
    """One daily close, for the price-history chart. ISO date, so it is JSON-safe as-is."""

    date: str
    close: float


@dataclass(frozen=True, slots=True)
class Quote:
    """One company's market figures, in stated units, with absences preserved.

    Every numeric field is `float | None` and `None` means **not reported** — never zero, never
    a placeholder. `tools/finance.py` renders an absence as "not reported" and
    `finance/ratios.py` excludes it from a peer mean rather than averaging it in as nought.
    """

    ticker: str
    #: The filer's name as the quote endpoint gives it. Shown on the card so a reader can see
    #: the ticker resolved to the company they meant; not used for matching anything.
    name: str
    #: ISO 4217, from the endpoint. Rendered beside every money figure — the Universe is
    #: US-listed today, so this exists to stop a future member's price reading as dollars.
    currency: str
    price: float | None
    previous_close: float | None
    #: Percent change on the previous close, in **percentage points** (`-4.99` is −4.99%).
    change_percent: float | None
    #: In `currency` units, not millions or billions: the formatting is the UI's business.
    market_cap: float | None
    trailing_pe: float | None
    trailing_eps: float | None
    #: A **ratio** (`4.26` is 4.26×), converted from yfinance's percentage points. `None` for
    #: every Universe bank — see the module docstring.
    debt_to_equity: float | None
    #: Fractions (`0.741` is 74.1%), which is how yfinance reports them and how
    #: `finance/ratios.py` averages them. Formatted as percentages at the surface.
    gross_margin: float | None
    profit_margin: float | None
    operating_margin: float | None
    fifty_two_week_high: float | None
    fifty_two_week_low: float | None
    #: Oldest first, so a chart plots left to right without sorting.
    closes: tuple[Close, ...] = ()

    @classmethod
    def from_info(
        cls, ticker: str, info: Mapping[str, Any], closes: tuple[Close, ...] = ()
    ) -> Quote:
        """Build a quote from yfinance's `.info` mapping and its history.

        Reads through `_number` so a NaN, a string and a missing key all become `None` — all
        three occur: `.info` is assembled from several scraped payloads and an absent figure
        arrives as any of them depending on which payload was short.
        """
        # `currentPrice` is the quote endpoint's field and `regularMarketPrice` the market-data
        # one; which is populated depends on the venue and the time of day, so both are read
        # rather than one being trusted. Same pattern for the previous close below.
        price = _number(info, "currentPrice") or _number(info, "regularMarketPrice")
        previous = _number(info, "previousClose") or _number(info, "regularMarketPreviousClose")
        return cls(
            ticker=ticker,
            name=str(info.get("shortName") or info.get("longName") or ticker),
            currency=str(info.get("currency") or "USD"),
            price=price,
            previous_close=previous,
            # Recomputed from price and previous close when the endpoint omits its own field,
            # because the card's headline number is the change and an absent one reads as flat.
            change_percent=_number(info, "regularMarketChangePercent")
            or _percent_change(price, previous),
            market_cap=_number(info, "marketCap"),
            trailing_pe=_number(info, "trailingPE"),
            trailing_eps=_number(info, "trailingEps"),
            debt_to_equity=_ratio_from_percent(_number(info, "debtToEquity")),
            gross_margin=_margin(info, "grossMargins"),
            profit_margin=_margin(info, "profitMargins"),
            operating_margin=_margin(info, "operatingMargins"),
            fifty_two_week_high=_number(info, "fiftyTwoWeekHigh"),
            fifty_two_week_low=_number(info, "fiftyTwoWeekLow"),
            closes=closes,
        )


def _number(info: Mapping[str, Any], key: str) -> float | None:
    """`info[key]` as a float, or `None` for anything that is not one — NaN included."""
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return None if number != number else number  # NaN is the only float unequal to itself


def _margin(info: Mapping[str, Any], key: str) -> float | None:
    """A margin, with an exact zero read as **not reported** — see the module docstring.

    The one coercion in this module, and it is bounded three ways: only margins, only exactly
    `0.0`, and justified by an arithmetic impossibility rather than by a hunch about the data
    (JPM and BAC report a zero gross margin beside a 38–50% operating margin). A genuine
    negative — Ford's −3.2% net margin — is untouched, which is the case a laxer rule
    (`if not value`) would have quietly dropped along with it.
    """
    margin = _number(info, key)
    return None if margin == 0.0 else margin


def _ratio_from_percent(percent: float | None) -> float | None:
    """yfinance's percentage-point D/E as the ratio a research note quotes."""
    return None if percent is None else percent / 100.0


def _percent_change(price: float | None, previous: float | None) -> float | None:
    """Percentage points between two prices, or `None` if either is missing or zero."""
    if price is None or not previous:
        return None
    return (price - previous) / previous * 100.0


def fetch_quote(ticker: str) -> Quote:
    """Fetch one quote from yfinance. **The one call, and the one thing no test covers.**

    Imported inside the function rather than at module scope, and the reason is the hermetic
    contract: importing yfinance pulls in its session machinery and its own on-disk cache, and
    this module is imported by `tools/finance.py` — which every agent test builds. Deferring it
    keeps the suite from loading a scraper it never calls.

    Raises whatever yfinance raises. `TimedCache` turns that into a retry and then into a stale
    serve; `tools/finance.py` turns an unrecoverable one into a sentence the model can act on.
    """
    import yfinance

    handle = yfinance.Ticker(ticker)
    info = handle.info
    history = handle.history(period=HISTORY_PERIOD, interval=HISTORY_INTERVAL, auto_adjust=True)
    closes = tuple(
        Close(date=stamp.date().isoformat(), close=float(row["Close"]))
        for stamp, row in history.iterrows()
    )
    if not info:
        # An empty `.info` is yfinance's characteristic soft failure: the scrape came back with
        # nothing and it does not raise. Left to itself it would cache a `Quote` of all-`None`
        # for a full TTL window, which reads on the card as a company with no price rather than
        # as a fetch that failed. Raising hands it to the retry, which is what it is.
        raise LookupError(f"the quote endpoint returned nothing for {ticker}")
    return Quote.from_info(ticker, info, closes)


#: The process-level quote cache — one handle, so every reader shares one TTL window.
#:
#: A module-level singleton for the reason `vectorstore.default_filings_store` is a cached
#: constructor: `calculate_ratios` resolving a `big_tech` company asks for six quotes, and a
#: per-call cache would make the cache the thing that never hits (ADR-0009's "zero new API
#: surface" claim is this object).
QUOTE_CACHE: TimedCache[Quote] = TimedCache(
    name="quotes",
    ttl_seconds=QUOTE_TTL_SECONDS,
    attempts=FETCH_ATTEMPTS,
    backoff_seconds=FETCH_BACKOFF_SECONDS,
)


def quote(ticker: str, *, cache: TimedCache[Quote] | None = None) -> Fetched[Quote]:
    """One company's quote, cached, retried, and stale-marked rather than failed.

    `cache` is injectable so a test can drive the policy with its own clock and a recorded
    fixture, and so `calculate_ratios` can be measured without a shared window between runs;
    production passes none and gets `QUOTE_CACHE`.
    """
    return (cache or QUOTE_CACHE).fetch(ticker, lambda: fetch_quote(ticker))
