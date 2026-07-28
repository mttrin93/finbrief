"""The free-tier layer: one TTL cache, one retry policy, one stale fallback.

The tools this sits under read two APIs that will fail: yfinance is an unofficial scrape of a
private endpoint, and Alpha Vantage's free tier is 25 calls a day. So the interesting behaviour
is not the happy path — it is *what a caller is handed when the refresh fails*, and whether it
can tell that from a fresh read. User story 22 asks for a cached fallback **and a banner**,
which means the answer has to carry its own staleness rather than the UI guessing at it.

Driven through `TimedCache.fetch`, which is the seam the tools use (spec seam 5). The clock and
the sleep are injected, so nothing here waits and no test depends on wall-clock timing.
"""

from __future__ import annotations

import pytest

from finbrief.finance.cache import TimedCache


class FakeClock:
    """A monotonic clock a test advances by hand."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Source:
    """A refresh callable that counts its calls and can be told to fail."""

    def __init__(self, *values: object, failures: int = 0) -> None:
        self.values = list(values) or ["v1"]
        self.failures = failures
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("yfinance said no")
        return self.values[min(self.calls - 1, len(self.values) - 1)]


@pytest.fixture
def slept() -> list[float]:
    return []


def a_cache(clock, slept, *, ttl_seconds: float = 60.0, attempts: int = 3) -> TimedCache:
    return TimedCache(
        name="quotes",
        ttl_seconds=ttl_seconds,
        attempts=attempts,
        backoff_seconds=0.5,
        clock=clock,
        sleep=slept.append,
    )


def test_a_first_read_fetches_and_reports_itself_fresh(slept):
    cache = a_cache(FakeClock(), slept)
    source = Source("quote")

    fetched = cache.fetch("NVDA", source)

    assert fetched.value == "quote"
    assert fetched.stale is False
    assert fetched.age_seconds == 0.0
    assert source.calls == 1


def test_a_second_read_inside_the_ttl_does_not_touch_the_api(slept):
    # The whole point of the cache, at the granularity that matters for a 25-calls/day quota:
    # `calculate_ratios` resolves its peers through this same path (ADR-0009), so a cluster of
    # six companies must cost six calls per TTL window and not six per question.
    clock = FakeClock()
    cache = a_cache(clock, slept, ttl_seconds=60.0)
    source = Source("first", "second")

    cache.fetch("NVDA", source)
    clock.advance(59.0)
    again = cache.fetch("NVDA", source)

    assert source.calls == 1, "one API call, not two"
    assert again.value == "first"
    assert again.age_seconds == 59.0
    assert again.stale is False, "inside its TTL is fresh, however old the number reads"


def test_a_read_past_the_ttl_refreshes(slept):
    clock = FakeClock()
    cache = a_cache(clock, slept, ttl_seconds=60.0)
    source = Source("first", "second")

    cache.fetch("NVDA", source)
    clock.advance(61.0)
    refreshed = cache.fetch("NVDA", source)

    assert source.calls == 2
    assert refreshed.value == "second"
    assert refreshed.age_seconds == 0.0


def test_keys_are_cached_apart(slept):
    cache = a_cache(FakeClock(), slept)
    source = Source("nvda", "amzn")

    assert cache.fetch("NVDA", source).value == "nvda"
    assert cache.fetch("AMZN", source).value == "amzn"
    assert cache.fetch("NVDA", source).value == "nvda", "AMZN did not evict NVDA"
    assert source.calls == 2


def test_a_transient_failure_is_retried_with_backoff_and_then_succeeds(slept):
    # yfinance's characteristic failure: it works on the next attempt. Retrying costs one sleep
    # and hides the failure entirely, which is why the retry sits below the staleness decision
    # rather than beside it — a caller must not see `stale` for something that recovered.
    cache = a_cache(FakeClock(), slept, attempts=3)
    source = Source("quote", failures=2)

    fetched = cache.fetch("NVDA", source)

    assert fetched.value == "quote"
    assert fetched.stale is False
    assert source.calls == 3
    assert slept == [0.5, 1.0], "exponential backoff, and it waited between attempts only"


def test_exhausting_the_retries_falls_back_to_the_cached_value_and_says_it_is_stale(slept):
    # User story 22's actual subject. The value is the best available answer *and* a
    # measurement nobody made just now, and the caller owes its reader both facts — hence
    # `stale` beside `value` rather than a silently old number.
    clock = FakeClock()
    cache = a_cache(clock, slept, ttl_seconds=60.0, attempts=2)
    source = Source("first")

    cache.fetch("NVDA", source)
    clock.advance(300.0)
    source.failures = 2
    served = cache.fetch("NVDA", source)

    assert served.value == "first", "the last good value, not an error and not a placeholder"
    assert served.stale is True
    assert served.age_seconds == 300.0, "and how old it is, so the banner can say so"
    assert source.calls == 3, "one call for the first read, then both retries"


def test_a_stale_serve_does_not_reset_the_entrys_age(slept):
    # A stale serve must not look like a refresh: writing `fetched_at = now` on a failure would
    # make the next read report the value as fresh, and the banner would disappear while the
    # API was still down — the one moment it is load-bearing.
    clock = FakeClock()
    cache = a_cache(clock, slept, ttl_seconds=60.0, attempts=1)
    source = Source("first")

    cache.fetch("NVDA", source)
    clock.advance(300.0)
    source.failures = 5
    cache.fetch("NVDA", source)
    clock.advance(60.0)
    again = cache.fetch("NVDA", source)

    assert again.stale is True
    assert again.age_seconds == 360.0, "aged from the last successful fetch, not the last serve"


def test_a_failure_with_nothing_cached_raises(slept):
    # There is no honest fallback here: the cache has never held a value for this key, so the
    # alternatives are an exception the tool can turn into a message or a fabricated number.
    # The tool layer is where that becomes a sentence a model can read (`tools/finance.py`).
    cache = a_cache(FakeClock(), slept, attempts=2)
    source = Source(failures=5)

    with pytest.raises(ConnectionError):
        cache.fetch("NVDA", source)

    assert source.calls == 2, "it exhausted its attempts before giving up"


def test_a_recovered_api_clears_the_stale_flag(slept):
    # The other half of the banner's contract: it has to go away. A `stale` latch on the cache
    # would leave the warning on screen for the rest of the process.
    clock = FakeClock()
    cache = a_cache(clock, slept, ttl_seconds=60.0, attempts=1)
    source = Source("first", "second")

    cache.fetch("NVDA", source)
    clock.advance(300.0)
    source.failures = 1
    assert cache.fetch("NVDA", source).stale is True

    clock.advance(1.0)
    recovered = cache.fetch("NVDA", source)

    assert recovered.stale is False
    assert recovered.value == "second"


def test_a_stale_serve_is_logged_with_counts_and_no_secret(slept, caplog):
    # The line the Phase-6 analysis reads to answer "how often did the free tier let us down".
    # It names the cache and the key — a ticker is a member of a closed whitelist, not user
    # content — and carries no exception text, which could contain a URL with a key in it.
    clock = FakeClock()
    cache = a_cache(clock, slept, ttl_seconds=60.0, attempts=1)
    source = Source("first")

    cache.fetch("NVDA", source)
    clock.advance(300.0)
    source.failures = 1
    with caplog.at_level("WARNING", logger="finbrief.finance.cache"):
        cache.fetch("NVDA", source)

    (record,) = [r for r in caplog.records if getattr(r, "event", None) == "stale_fallback"]
    assert record.fields["source"] == "quotes"
    assert record.fields["key"] == "NVDA"
    assert record.fields["age_seconds"] == 300
    assert record.fields["attempts"] == 1
    assert record.fields["error"] == "ConnectionError", "the type, not the message"


def test_clearing_the_cache_forgets_everything(slept):
    # The escape hatch a "refresh" control needs, and the one a test needs: these caches are
    # process-level singletons in production, so without it one test's NVDA quote is another's.
    cache = a_cache(FakeClock(), slept)
    source = Source("first", "second")

    cache.fetch("NVDA", source)
    cache.clear()

    assert cache.fetch("NVDA", source).value == "second"
    assert source.calls == 2
