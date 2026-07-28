"""`build_once` — a process-level singleton that survives being asked for concurrently.

Not `finance/cache.py`, which is a TTL cache over *remote data* and answers "how stale is this
number". This is about *construction*: three collaborators on the retrieval path are expensive
to build and are therefore built once per process and shared — the Chroma handle, the BM25
index, the sub-query planner's model — and `functools.lru_cache` alone is not enough to promise
that.

**Why it is not enough, measured.** `lru_cache` is atomic about its bookkeeping and says
nothing about the function it wraps: two threads that miss the same key both call through.
Until T5 that never happened, because `parallel_tool_calls=False` meant one tool ran at a time.
Re-enabling fan-out (#9) put two `search_filings` calls in one step, LangGraph ran them
concurrently, both missed the cold `default_filings_store` cache, and both constructed a
`chromadb.PersistentClient` over the same directory. chromadb's shared-system registry is not
reentrant, so one client released the system the other was still starting:

    AttributeError: 'RustBindingsAPI' object has no attribute 'bindings'

— thrown from inside the tool node, on the first turn of a cold process. It killed the demo's
"give me the full brief" query and nothing in the suite could have seen it, because a hermetic
test injects its store and never reaches the cached constructor at all.

`build_once` serialises the call so the second thread finds the value the first one cached. The
lock is held across construction, which is the point: a lock released before the constructor
returns is a lock that lets both threads in.

Cheap on the hot path — an uncontended lock plus a dict lookup, per retrieval, against a
5,800-chunk index build it is protecting.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from functools import lru_cache, wraps
from typing import Any


def build_once[**P, R](constructor: Callable[P, R], *, maxsize: int = 1) -> Any:
    """Cache `constructor` per argument, and let only one caller build a given value at a time.

    Returns a callable that also carries `cache_clear` and `cache_info`, forwarded from the
    underlying `lru_cache`, because those are part of the wrapped function's published
    interface: `tests/test_hybrid.py` clears the BM25 index by name, and CLAUDE.md records that
    conftest does *not* clear these caches, so a test that builds one is responsible for it.
    Hiding `cache_clear` behind this wrapper would have broken that contract silently.

    `maxsize` is forwarded rather than fixed at one: the BM25 index keeps two, so one store can
    be replaced without unbounded retention of old corpora (`hybrid.bm25_index`).
    """
    cached = lru_cache(maxsize=maxsize)(constructor)
    lock = threading.Lock()

    @wraps(constructor)
    def once(*args: P.args, **kwargs: P.kwargs) -> R:
        with lock:
            return cached(*args, **kwargs)

    once.cache_clear = cached.cache_clear  # type: ignore[attr-defined]
    once.cache_info = cached.cache_info  # type: ignore[attr-defined]
    return once
