"""`build_once` — a process-level singleton that survives being asked for concurrently.

Not `finance/cache.py`, which is a TTL cache over *remote data* and answers "how stale is this
number". This is about *construction*: six collaborators are expensive to build or unsafe to
build twice, and are therefore built once per process and shared — the Chroma handle, the BM25
index, the sub-query planner's model, the bounded yfinance session, and since T7 the input
gate's classifier model and the output validator's `Guard` — and `functools.lru_cache` alone is
not enough to promise that.

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
from functools import _CacheInfo, lru_cache, wraps
from typing import Protocol


class BuiltOnce[**P, R](Protocol):
    """What `build_once` returns: the wrapped callable, plus the two `lru_cache` members.

    A `Protocol` rather than `Any`, because `Any` erased the return type at every call site that
    matters — `default_filings_store(settings)` typed as `Chroma` before it went through
    `build_once` and as `Any` afterwards, which silently switched off checking on the busiest
    object in the app (issue #9 review). The two cache members are part of the published
    interface for the reason the docstring below gives, so they belong in the type.
    """

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

    def cache_clear(self) -> None: ...

    def cache_info(self) -> _CacheInfo: ...


def build_once[**P, R](constructor: Callable[P, R], *, maxsize: int = 1) -> BuiltOnce[P, R]:
    """Cache `constructor` per argument, and let only **one construction happen at a time**.

    **One lock for the wrapper, not one per key**, and the distinction is worth stating because
    the first version of this sentence claimed the latter ("let only one caller build a *given
    value* at a time"). The lock is a single `threading.Lock` guarding every key, so two callers
    asking for two *different* values serialise — with `maxsize=2` on `hybrid.bm25_index`, one
    5,800-chunk corpus build waits for another (issue #9 review).

    That is left as it is, deliberately. Keying the lock means a second map from key to lock,
    which needs its own lock to populate safely, and getting *that* wrong reintroduces exactly
    the race this module exists to close — for a benefit nothing in this application can
    observe: every one of the six singletons on this path is keyed on the frozen `Settings`,
    on a store object, or on nothing at all, so two live keys means a configuration change
    mid-process, which the app does not do. A correct coarse lock beats a subtly wrong fine
    one; if a caller ever does need concurrent builds of different keys, keying it is the
    change to make, with a test that fails without it.

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
    return once  # type: ignore[return-value]
