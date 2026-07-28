"""The free-tier layer: one TTL cache, one retry policy, one stale fallback.

Every live number FinBrief shows comes through here, and the reason is that both sources will
fail. yfinance is an **unofficial** client for a private endpoint — no contract, no status page,
and a shape that changes without notice (PLAN §4, §8). Alpha Vantage's free tier is **25 calls
a day**, which is not a rate limit so much as a daily budget. So the question this module
answers is not "how do we fetch" but "what does a caller get when the fetch fails", and there
are exactly three honest answers:

1. **A fresh value**, when the API answered — possibly after a retry, which the caller never
   sees, because something that recovered is not stale.
2. **The last good value, marked stale, with its age**, when the API did not answer and we
   have held one. User story 22 asks for a cached fallback *and a banner*, and a banner needs
   the age; a silently old number is the failure mode, not the fix.
3. **An exception**, when the API did not answer and we have never held one. There is no honest
   third value to invent, and inventing one is how a price of `0.00` reaches a research note.

`Fetched` is that trichotomy made explicit in the return type: `value` plus the two facts a
reader of the value needs about it. Nothing downstream may drop `stale` on the way to a
surface — `tools/finance.py` carries it into the tool artifact and `app/Home.py` renders it.

**Composition, not two knobs.** The retry sits *inside* `fetch`, below the staleness decision,
so the two policies cannot be combined wrongly at a call site: retries are how we avoid
reporting staleness we did not have to, and a caller who retried *around* the cache would spend
its attempts before the cache ever got a chance to serve. One rule, one place.

The thresholds live in `config.py` and are read by the tool layer, not defaulted here: this
class is the mechanism and the calibration is configuration (CLAUDE.md).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from finbrief.observability.logging_setup import log_event

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Fetched[T]:
    """A value, and the two facts a reader of it needs: how old it is, and whether it is stale.

    `stale` is not "old". A quote inside its TTL is fresh however many seconds have passed;
    `stale` means specifically *we tried to refresh this and could not*, so the value is the
    best available answer rather than a measurement of now. That distinction is the difference
    between a banner that means something and one a reader learns to ignore.
    """

    value: T
    #: Seconds since the value was last **successfully** fetched — never since it was last
    #: served. A stale serve that reset this would make the next read look fresh, and the banner
    #: would vanish while the API was still down.
    age_seconds: float
    stale: bool


@dataclass(frozen=True, slots=True)
class _Entry[T]:
    value: T
    fetched_at: float


class TimedCache[T]:
    """A keyed TTL cache that retries, and falls back to a stale value rather than failing.

    One instance per **source**, not per tool: the lock below is per instance, so a slow quote
    refresh must not be able to block a news refresh queued behind it. `quotes.py` and `news.py`
    each own one.

    **The lock is held across the refresh, deliberately.** Two Streamlit sessions asking about
    NVDA in the same second would otherwise both miss and both call out — and with parallel tool
    calls enabled, one turn's `get_stock_data` and `calculate_ratios` can genuinely arrive
    concurrently for the same ticker. Against a 25-calls/day budget, coalescing those is worth
    more than the concurrency it costs, and the cost is bounded by the retry budget. The
    serialisation that matters — two *different* sources — is already avoided by the
    per-instance scope above.
    """

    def __init__(
        self,
        *,
        name: str,
        ttl_seconds: float,
        attempts: int,
        backoff_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        #: What this cache holds, for the log line — `quotes`, `news`. The analysis reading
        #: those lines back needs to know *which* free tier let us down.
        self.name = name
        self.ttl_seconds = ttl_seconds
        self.attempts = attempts
        self.backoff_seconds = backoff_seconds
        # `time.monotonic`, not `time.time`: an age computed across a clock adjustment can come
        # out negative, and a negative age would read as a value from the future in the banner.
        # Injected so a test can advance it by hand rather than sleep (`test_finance_cache.py`).
        self._clock = clock
        self._sleep = sleep
        self._entries: dict[str, _Entry[T]] = {}
        self._lock = threading.Lock()

    def fetch(self, key: str, refresh: Callable[[], T]) -> Fetched[T]:
        """The value for `key`: cached if fresh, refreshed if not, stale if it could not be.

        Raises whatever `refresh` raised, but only when there is nothing cached to fall back
        to — see this module's docstring for why that case has no honest value to return. The
        tool layer turns the exception into a sentence the model can act on.
        """
        with self._lock:
            entry = self._entries.get(key)
            now = self._clock()
            if entry is not None and now - entry.fetched_at <= self.ttl_seconds:
                return Fetched(entry.value, age_seconds=now - entry.fetched_at, stale=False)
            try:
                value = self._refresh_with_retries(refresh)
            except Exception as exc:
                if entry is None:
                    raise
                age = self._clock() - entry.fetched_at
                log_event(
                    logger,
                    "stale_fallback",
                    level=logging.WARNING,
                    source=self.name,
                    # A ticker, or a ticker and a day count — a member of a closed whitelist,
                    # not free-form user content (`logging_setup`).
                    key=key,
                    age_seconds=round(age),
                    attempts=self.attempts,
                    # The exception *type*, never its message: a client's error string can
                    # carry the request URL, and a request URL can carry an API key.
                    error=type(exc).__name__,
                )
                return Fetched(entry.value, age_seconds=age, stale=True)
            self._entries[key] = _Entry(value, fetched_at=self._clock())
            return Fetched(value, age_seconds=0.0, stale=False)

    def clear(self) -> None:
        """Forget everything. The escape hatch a manual refresh needs, and a test does."""
        with self._lock:
            self._entries.clear()

    def _refresh_with_retries(self, refresh: Callable[[], T]) -> T:
        """`refresh`, retried with exponential backoff. Raises the last failure.

        Sleeps *between* attempts and never after the last one, so the caller does not pay a
        backoff for a fetch that has already given up. `attempts=1` therefore means "try once,
        do not wait" rather than "try once, then wait pointlessly".
        """
        for attempt in range(self.attempts):
            try:
                return refresh()
            except Exception:
                if attempt == self.attempts - 1:
                    raise
                self._sleep(self.backoff_seconds * 2**attempt)
        raise AssertionError("unreachable: the loop either returns or raises")
