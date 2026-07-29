"""The content-addressed cache: what makes a killed evaluation run resumable.

The property under test is not "caching works" but "a resumed run spends nothing on what the
killed run finished, and reports the same numbers it would have". Both halves have a test here,
because only the first is obvious and the second is the one that silently fails: a value handed
back from JSON is not the value that went in unless something forces it to be.
"""

from __future__ import annotations

import json

import pytest

from finbrief.evaluation.cache import Cache, CacheDisabled


def a_cache(tmp_path, **kwargs) -> Cache:
    return Cache(tmp_path / "eval-cache", **kwargs)


def test_a_miss_calls_the_producer_and_returns_its_value(tmp_path):
    cache = a_cache(tmp_path)

    value = cache.resolve("judge", {"row": "S1"}, lambda: {"score": 0.5})

    assert value == {"score": 0.5}
    assert cache.stats("judge").misses == 1
    assert cache.stats("judge").hits == 0


def test_a_hit_does_not_call_the_producer(tmp_path):
    # The spy is the point: a cache that recomputed and returned the same answer would pass an
    # equality assertion while spending the money the cache exists to save.
    cache = a_cache(tmp_path)
    calls = []

    def produce():
        calls.append(1)
        return {"score": 0.5}

    cache.resolve("judge", {"row": "S1"}, produce)
    value = cache.resolve("judge", {"row": "S1"}, produce)

    assert value == {"score": 0.5}
    assert len(calls) == 1, "the second resolve called the producer, so nothing was cached"
    assert cache.stats("judge").hits == 1


def test_a_hit_survives_a_fresh_cache_object_over_the_same_directory(tmp_path):
    # Resumability is across *processes*, so the entry has to be on disk and not in an instance.
    a_cache(tmp_path).resolve("judge", {"row": "S1"}, lambda: {"score": 0.5})

    reopened = a_cache(tmp_path)
    value = reopened.resolve("judge", {"row": "S1"}, lambda: pytest.fail("recomputed"))

    assert value == {"score": 0.5}


@pytest.mark.parametrize(
    "changed",
    [
        {"row": "S2"},
        {"row": "S1", "arm": "hybrid+translation"},
        {"row": "S1", "judge_model": "openai/gpt-4o-mini"},
    ],
)
def test_changing_any_part_of_the_key_is_a_miss(tmp_path, changed):
    # Every input that determines the value belongs in the key; this is the half that says a
    # changed input cannot be served a stale number.
    cache = a_cache(tmp_path)
    cache.resolve("judge", {"row": "S1"}, lambda: {"score": 0.5})

    value = cache.resolve("judge", changed, lambda: {"score": 0.9})

    assert value == {"score": 0.9}


def test_the_same_key_under_a_different_kind_is_a_miss(tmp_path):
    # `{"row": "S1"}` means something different to the retrieval stage and to the judge, so the
    # kind is part of the address rather than merely a directory for tidiness.
    cache = a_cache(tmp_path)
    cache.resolve("retrieval", {"row": "S1"}, lambda: {"chunks": []})

    value = cache.resolve("judge", {"row": "S1"}, lambda: {"score": 0.5})

    assert value == {"score": 0.5}


def test_a_key_that_is_not_json_serialisable_raises(tmp_path):
    # Not silently stringified: `repr(object())` carries a memory address, so such a key would
    # miss on every call and the cache would look enabled while caching nothing.
    cache = a_cache(tmp_path)

    with pytest.raises(TypeError):
        cache.resolve("judge", {"model": object()}, lambda: {"score": 0.5})


def test_a_hit_and_a_miss_return_the_same_type(tmp_path):
    # **The bug this test exists for.** JSON has no tuples: produce a tuple, and a fresh run
    # returns `("a", "b")` while a resumed run returns `["a", "b"]` — so a comparison downstream
    # depends on cache state, which is the one thing a cache must never change. `resolve` puts
    # every value through the round trip, including on a miss, so the two runs are identical.
    cache = a_cache(tmp_path)

    fresh = cache.resolve("judge", {"row": "S1"}, lambda: {"ids": ("a", "b")})
    reopened = a_cache(tmp_path)
    resumed = reopened.resolve("judge", {"row": "S1"}, lambda: pytest.fail("recomputed"))

    assert fresh == resumed
    assert fresh == {"ids": ["a", "b"]}


def test_a_producer_that_raises_leaves_no_entry(tmp_path):
    # A killed run must resume by *recomputing* the cell it died inside, never by reading a
    # half-written one. Nothing is written until the producer has returned.
    cache = a_cache(tmp_path)

    def dies():
        raise RuntimeError("429 Too Many Requests")

    with pytest.raises(RuntimeError):
        cache.resolve("judge", {"row": "S1"}, dies)

    assert cache.resolve("judge", {"row": "S1"}, lambda: {"score": 0.5}) == {"score": 0.5}


def test_an_unreadable_entry_is_a_miss_and_is_counted(tmp_path):
    # The truncated-final-line case `observability.events` already handles for the log: a file
    # cut off by a kill -9 during a write is skipped and *counted*, never raised and never
    # served. Counted because a half-unreadable cache presenting as a complete one is how a
    # re-run silently spends more than the report says it did.
    cache = a_cache(tmp_path)
    cache.resolve("judge", {"row": "S1"}, lambda: {"score": 0.5})
    entry = next((tmp_path / "eval-cache" / "judge").glob("*.json"))
    entry.write_text('{"value": {"score": 0.5', encoding="utf-8")

    reopened = a_cache(tmp_path)
    value = reopened.resolve("judge", {"row": "S1"}, lambda: {"score": 0.9})

    assert value == {"score": 0.9}
    assert reopened.stats("judge").malformed == 1


def test_a_write_leaves_no_temporary_file_behind(tmp_path):
    # The write is atomic — a temp file in the same directory, then `os.replace` — so a reader
    # never sees a partial entry. A leftover temp file would mean the rename did not happen.
    cache = a_cache(tmp_path)

    cache.resolve("judge", {"row": "S1"}, lambda: {"score": 0.5})

    files = sorted(p.name for p in (tmp_path / "eval-cache" / "judge").iterdir())
    assert len(files) == 1 and files[0].endswith(".json"), files


def test_the_entry_records_the_key_it_was_addressed_by(tmp_path):
    # A digest is unreadable, and a cache nobody can audit is one nobody can debug: "why did
    # this cell not hit?" has to be answerable from the directory. The key is stored beside the
    # value for that reason, and it is what a collision would be visible as.
    cache = a_cache(tmp_path)
    cache.resolve("judge", {"row": "S1", "metric": "faithfulness"}, lambda: {"score": 0.5})

    entry = json.loads(next((tmp_path / "eval-cache" / "judge").glob("*.json")).read_text())

    assert entry["key"] == {"row": "S1", "metric": "faithfulness"}
    assert entry["value"] == {"score": 0.5}


def test_a_disabled_cache_refuses_rather_than_silently_recomputing(tmp_path):
    # `Cache(None)` is not a no-op cache. A run with the cache switched off would spend the full
    # bill on every re-run, and the failure mode of *thinking* it was on is unbounded spend — so
    # the harness has to name it rather than discover it. `CacheDisabled` is that name.
    cache = Cache(None)

    with pytest.raises(CacheDisabled):
        cache.resolve("judge", {"row": "S1"}, lambda: {"score": 0.5})
