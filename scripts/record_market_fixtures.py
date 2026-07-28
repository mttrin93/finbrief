"""Record the market-data and news fixtures the hermetic suite parses (T5, #9).

Run **by hand**, never from a test. `tests/conftest.py` blocks egress for the whole suite, so
a test that wanted live data could not have it even if it asked; what it gets instead is what
this script writes. Free and keyless — Yahoo's quote endpoint via yfinance, Yahoo's headline
RSS via urllib — but it does reach the network, which is why it lives here beside
`record_edgar_fixtures.py` rather than under `tests/`.

    uv run python scripts/record_market_fixtures.py

**What is recorded, and at which boundary.** Two different answers, for a reason worth stating
because it bounds what the suite covers:

- **News is recorded raw** — the RSS bytes exactly as served. `feedparser.parse` accepts a
  string, so the tests run the *real* parser and the real HTML stripper over the real feed, and
  the whole parse path is covered.
- **Quotes are recorded parsed** — yfinance's own `.info` mapping and its history frame, as
  JSON. yfinance is a scraper: intercepting it below its own layer would mean recording the
  private endpoint's responses and pinning our fixtures to one version of somebody else's
  undocumented internals. So the boundary is `yfinance.Ticker`, and what the suite covers is
  everything above it — field selection, unit conversion, the missing-value tolerance. **The
  `Ticker` call itself is not covered by any test**, and the way that breaks is a yfinance
  upgrade renaming a key: the fixture keeps passing while production returns `None`.
  `scripts/retrieval_smoke.py` is the precedent for how that class of gap is closed — a
  non-hermetic script somebody runs — and refreshing these fixtures *is* that check: a key
  that vanished shows up as a `None` in the JSON written here.

`.info` is **trimmed to `RECORDED_INFO_KEYS`** rather than dumped whole. Two reasons: the full
mapping is ~186 keys of which we read a dozen, and it carries `companyOfficers` — named
individuals with ages and compensation. Third-party personal data does not belong in a
fixture, and a repo is forever.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import yfinance as yf

from finbrief.config import UNIVERSE
from finbrief.finance.news import FEED_URL_TEMPLATE, NEWS_USER_AGENT

FIXTURES = Path(__file__).parents[1] / "tests" / "fixtures" / "market"

#: The `.info` keys the fixture keeps: every field `Quote.from_info` reads, plus the near-misses
#: worth having on hand (`forwardPE`, `operatingMargins`) so a later field needs no re-record.
#: Nothing else — see this module's docstring on `companyOfficers`.
RECORDED_INFO_KEYS = (
    "shortName",
    "longName",
    "currency",
    "fullExchangeName",
    "currentPrice",
    "regularMarketPrice",
    "previousClose",
    "regularMarketPreviousClose",
    "regularMarketChangePercent",
    "marketCap",
    "trailingPE",
    "forwardPE",
    "trailingEps",
    "debtToEquity",
    "grossMargins",
    "profitMargins",
    "operatingMargins",
    "totalDebt",
    "totalRevenue",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
)

#: The tickers whose headline feed is recorded. Three, not fifteen: the golden set's three news
#: rows (TSLA/T3, MSFT/T6, GM/T7) are what the tool eval will execute against, and a feed is
#: ~15 KB of somebody else's copy. One feed proves the parser; three prove it across publishers.
NEWS_TICKERS = ("TSLA", "MSFT", "GM")

#: Politeness between requests to a free public endpoint. Not a rate limit anyone published —
#: which is the point: a script that hammers an unofficial endpoint is how the endpoint starts
#: refusing this project.
PAUSE_SECONDS = 1.0


def record_quotes() -> None:
    """One JSON file per Universe company: its trimmed `.info` and a month of daily closes."""
    for company in UNIVERSE:
        ticker = yf.Ticker(company.ticker)
        info = {key: _plain(ticker.info.get(key)) for key in RECORDED_INFO_KEYS}
        history = ticker.history(period="1mo", interval="1d", auto_adjust=True)
        closes = [
            # ISO date and close only. The tool renders a line chart and nothing reads the
            # other five columns, so recording them would be four fixtures' worth of noise.
            {"date": stamp.date().isoformat(), "close": float(row["Close"])}
            for stamp, row in history.iterrows()
        ]
        path = FIXTURES / f"{company.ticker.lower()}-info.json"
        path.write_text(
            json.dumps({"info": info, "closes": closes}, indent=2) + "\n", encoding="utf-8"
        )
        missing = sorted(key for key, value in info.items() if value is None)
        print(f"{company.ticker:>5}  {len(closes):>3} closes  missing: {missing or 'none'}")
        time.sleep(PAUSE_SECONDS)


def record_news() -> None:
    """The raw RSS bytes for `NEWS_TICKERS`, verbatim — the tests run the real parser on it."""
    for ticker in NEWS_TICKERS:
        request = urllib.request.Request(  # noqa: S310 — a fixed https template, not user input
            FEED_URL_TEMPLATE.format(ticker=ticker),
            headers={"User-Agent": NEWS_USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            raw = response.read()
        path = FIXTURES / f"{ticker.lower()}-headlines.xml"
        path.write_bytes(raw)
        print(f"{ticker:>5}  {len(raw):>6} bytes of RSS")
        time.sleep(PAUSE_SECONDS)


def _plain(value: Any) -> Any:
    """JSON-safe, and `None` for a numpy NaN — which `.info` returns for an absent figure.

    Kept explicit because the tolerance downstream is built on it: `debtToEquity` is genuinely
    absent for every bank in the Universe (JPM, BAC, GS), and `Quote` has to report that as
    "not available" rather than as zero leverage.
    """
    if value is None:
        return None
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return None if value != value else value  # NaN is the only float unequal to itself
    return str(value)


if __name__ == "__main__":
    FIXTURES.mkdir(parents=True, exist_ok=True)
    record_quotes()
    record_news()
    print(f"\nwrote {len(list(FIXTURES.iterdir()))} fixtures to {FIXTURES}")
