"""The content-addressed store that makes a killed evaluation run resumable.

A full six-arm run is ~2,200 network round trips over 20–35 minutes (T10, #11). A 429 twenty
minutes in, a laptop that sleeps, a `^C` — any of them must cost the remainder of the run and
not the whole of it. That is this module's only job, and it is why the cache was designed in
rather than added afterwards: a cache bolted on later is keyed on whatever the caller happened
to have in scope, which is how a stale number gets served under a changed input.

**The key is every input that determines the value, and the kind is part of the address.** A
digest over a canonicalised key, so `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` are one entry,
and a key that will not serialise raises rather than being stringified — `repr(object())`
carries a memory address, which would miss on every call while the cache looked enabled.

**A hit and a miss return the same value, not merely an equal one.** Every value goes through
the JSON round trip *including on a miss*, because JSON has no tuples: without it, a fresh run
gets `("a", "b")` where a resumed run gets `["a", "b"]`, and a comparison downstream starts
depending on cache state. That is the one thing a cache must never change, and it is the
failure this module's tests were written against rather than a hypothetical (see
`tools/finance.py` for the same lesson learned the other way round — `asdict` keeps tuples and
`json.dumps` will not tell you).

**Nothing is written until the producer has returned, and the write is atomic.** A run killed
inside a call leaves no entry, so the resumed run recomputes that one cell; a run killed
inside a *write* cannot leave a half-entry, because the bytes land in a temp file next door
and arrive by `os.replace`. An entry that is unreadable anyway — a disk full, an older
non-atomic writer — is counted and treated as a miss, the same choice
`observability/events.py` makes about a truncated final line, and for the same reason: losing
one cell is better than losing the run, and losing it *silently* is worse than either.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CacheDisabled(RuntimeError):
    """A stage asked the cache for something and no cache directory was configured.

    Deliberately an error and not a silent recompute. The whole bill is ~$1.28 (#11's plan), and
    a run that *believes* it is caching while it is not pays it again on every re-run — so the
    absence is named at the first request instead of showing up as a surprising invoice.
    """


@dataclass(frozen=True, slots=True)
class CacheStats:
    """What a kind's directory did during this process: the report quotes these.

    A number replayed from a cache and a number this run paid for are the same number, but
    "1,596 judge calls" is a claim about *spend* — so the artifact says how many cells were
    replayed rather than letting a reader assume a cold run. `malformed` is here for the reason
    `EventLog.malformed` is: a half-unreadable store presenting as a complete one is how a
    re-run quietly costs more than the report says.
    """

    hits: int = 0
    misses: int = 0
    malformed: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses


class Cache:
    """Values addressed by their inputs, under `root/<kind>/<digest>.json`.

    `root=None` means no cache, and every `resolve` then raises `CacheDisabled` — see that class
    for why that is better than falling back to recomputation.
    """

    def __init__(self, root: Path | str | None) -> None:
        self._root = None if root is None else Path(root)
        self._stats: dict[str, CacheStats] = {}
        # `run.cached_map` may resolve cells from several threads, and a `CacheStats` update
        # is a read-modify-write: without this the hit/miss counts drift, and those counts are
        # what the artifact quotes as spend. The lock guards the counters only — `resolve`
        # itself needs none, because two threads racing on one key each write their own temp
        # file and `os.replace` makes the last one atomic, and both computed the same value
        # from the same inputs.
        self._lock = threading.Lock()

    @property
    def root(self) -> Path | None:
        return self._root

    @property
    def enabled(self) -> bool:
        return self._root is not None

    def stats(self, kind: str) -> CacheStats:
        """This process's hits, misses and unreadable entries for `kind`."""
        return self._stats.get(kind, CacheStats())

    def kinds(self) -> tuple[str, ...]:
        """Every kind this process touched, in name order — what the report iterates."""
        return tuple(sorted(self._stats))

    def resolve(self, kind: str, key: Mapping[str, Any], produce: Callable[[], Any]) -> Any:
        """The stored value for `key`, calling `produce()` only when there is not one.

        `produce` is the paid call. It runs at most once per key per store, and its result is
        written only after it returns — so an exception (a 429, a `^C`) leaves the cell absent
        and the next run recomputes exactly it.
        """
        if self._root is None:
            raise CacheDisabled(
                f"the {kind!r} stage asked for a cached value and no cache directory is "
                f"configured. Pass one (scripts/evaluate.py --cache-dir): a run without a "
                f"cache pays the full bill again on every re-run, sized at ~$1.28 on #11."
            )
        path = self._path(kind, key)
        cached = self._read(path)
        if cached is not None:
            self._count(kind, hits=1)
            return cached["value"]
        self._count(kind, misses=1)
        # The round trip on the miss path as well as the hit path, so the two are the same value
        # and not merely equal ones. See the module docstring.
        value = _round_trip(produce())
        self._write(path, key=key, value=value)
        return value

    def _path(self, kind: str, key: Mapping[str, Any]) -> Path:
        assert self._root is not None
        return self._root / kind / f"{digest(key)}.json"

    def _read(self, path: Path) -> dict[str, Any] | None:
        """The entry at `path`, or `None` for absent — and `None` for unreadable, counted."""
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            self._count(path.parent.name, malformed=1)
            logger.warning("unreadable cache entry, recomputing: %s", path)
            return None
        if not isinstance(entry, dict) or "value" not in entry:
            self._count(path.parent.name, malformed=1)
            return None
        return entry

    def _write(self, path: Path, *, key: Mapping[str, Any], value: Any) -> None:
        """Write the entry so that no reader can ever see a partial one."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # In the same directory, because `os.replace` is only atomic within a filesystem.
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps({"key": key, "value": value}, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _count(self, kind: str, *, hits: int = 0, misses: int = 0, malformed: int = 0) -> None:
        with self._lock:
            self._bump(kind, hits=hits, misses=misses, malformed=malformed)

    def _bump(self, kind: str, *, hits: int, misses: int, malformed: int) -> None:
        current = self._stats.get(kind, CacheStats())
        self._stats[kind] = CacheStats(
            hits=current.hits + hits,
            misses=current.misses + misses,
            malformed=current.malformed + malformed,
        )


def digest(key: Mapping[str, Any]) -> str:
    """The address of `key` — a sha256 over its canonical JSON.

    Canonical so that key order cannot split one cell into two. `default` is deliberately not
    supplied: an unserialisable member raises `TypeError` here, at the call that would otherwise
    have cached nothing at all.
    """
    canonical = json.dumps(key, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _round_trip(value: Any) -> Any:
    """`value` as it will come back out of the store, applied on both paths."""
    return json.loads(json.dumps(value, ensure_ascii=False))
