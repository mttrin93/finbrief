"""`build_once` — the process-level singletons, under concurrency (T5, #9).

This file exists because of a crash the suite could not have found. Re-enabling parallel tool
calls put two `search_filings` calls in one LangGraph step; both missed the cold
`default_filings_store` cache, both constructed a `chromadb.PersistentClient` over the same
directory, and chromadb's non-reentrant shared-system registry killed one of them with
`AttributeError: 'RustBindingsAPI' object has no attribute 'bindings'` — on the first turn of a
cold process, from inside the tool node.

No hermetic test could have seen it: every test injects its store and never reaches the cached
constructor. What *is* testable is the property the fix rests on — that two concurrent callers
produce one construction — so that is what this asserts, at the seam rather than through Chroma.
"""

from __future__ import annotations

import threading
import time

from finbrief.caching import build_once


def test_a_value_is_built_once_and_then_reused():
    built = []
    once = build_once(lambda key: built.append(key) or f"value-{key}")

    assert once("a") == "value-a"
    assert once("a") == "value-a"

    assert built == ["a"], "the second call was a cache hit"


def test_different_arguments_get_different_values():
    once = build_once(lambda key: f"value-{key}", maxsize=2)

    assert once("a") == "value-a"
    assert once("b") == "value-b"
    assert once("a") == "value-a"


def test_two_threads_missing_the_same_cold_key_build_it_once():
    # The whole point, and the bug: `lru_cache` is atomic about its bookkeeping and says nothing
    # about the function it wraps, so both threads called through and both constructed. A
    # barrier puts them inside simultaneously and the slow constructor widens the window that
    # used to be a race — without the lock this counts two.
    built: list[str] = []
    ready = threading.Barrier(2)

    def slow_constructor(key: str) -> str:
        built.append(key)
        time.sleep(0.05)
        return f"value-{key}"

    once = build_once(slow_constructor)
    results: list[str] = []

    def call() -> None:
        ready.wait(timeout=5)
        results.append(once("shared"))

    threads = [threading.Thread(target=call) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert built == ["shared"], "one construction, whichever thread got there first"
    assert results == ["value-shared", "value-shared"], "and both callers got it"


def test_cache_clear_and_cache_info_survive_the_wrapper():
    # Part of the wrapped function's published interface: `tests/test_hybrid.py` clears the BM25
    # index by name, and CLAUDE.md records that conftest does *not* clear these caches — so a
    # test that builds one is responsible for it. A wrapper that swallowed `cache_clear` would
    # have broken that silently, leaving one test's index in another's hands.
    built = []
    once = build_once(lambda key: built.append(key) or f"value-{key}")

    once("a")
    assert once.cache_info().currsize == 1

    once.cache_clear()

    assert once.cache_info().currsize == 0
    once("a")
    assert built == ["a", "a"], "cleared, so it built again"


def test_the_wrapper_keeps_the_constructors_identity():
    # `functools.wraps`, so a traceback and a `repr` still name the thing that failed rather
    # than naming `once`.
    def _build_planner_model(settings: str) -> str:
        return settings

    assert build_once(_build_planner_model).__name__ == "_build_planner_model"


def test_the_retrieval_singletons_all_go_through_it():
    # Named individually rather than by a scan, because the list is the claim: these are the
    # three cached constructors on the concurrent path (CLAUDE.md), and a fourth added without
    # the guard is the same crash again. `cache_clear` is the observable signature.
    from finbrief.retrieval.hybrid import bm25_index
    from finbrief.retrieval.retrieve import _planner_model
    from finbrief.retrieval.vectorstore import default_filings_store

    for singleton in (default_filings_store, bm25_index, _planner_model):
        assert hasattr(singleton, "cache_clear"), singleton
        assert singleton.__wrapped__ is not None, f"{singleton} is not a build_once wrapper"
