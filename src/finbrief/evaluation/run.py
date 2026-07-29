"""Running a stage over the golden set, one cached cell per item.

The whole of the harness's resumability lives in `cached_map`, which is why it is four lines:
a stage is a pure map from an item to a value, the cache decides which items still cost
anything, and a stage that dies leaves every finished cell on disk. There is no checkpoint
file, no progress ledger and no resume flag — the cache *is* the ledger, addressed by content,
so a resumed run cannot disagree with the run it is resuming (`evaluation/cache.py`).

**Serial by default, concurrent when asked, and the ceiling is deliberately low.** A stage is
a `for` loop at `workers=1`; above that it is a bounded thread pool. The serial default is the
conservative one — OpenRouter answers a 429 to concurrency long before it answers one to
volume — and the reason concurrency is here rather than anywhere else is that the cache
already makes it safe: each cell is addressed by content and written atomically, so two
threads that raced on one key would compute the same value from the same inputs and the last
`os.replace` would win.

Measured, which is why the option exists at all: the judge stage ran at **4.7 cells/min**
serially against a full six-arm run's 560 cells — 99 minutes of wall clock for a stage whose
cells are independent and network-bound. `Cache` guards its hit/miss counters with a lock for
this, because those counters are what the artifact quotes as spend.

**An exception still fails the stage.** `future.result()` re-raises, and the cells that
finished are already durable — so a resumed run pays for the failures and nothing else, which
is the same contract the serial path has.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from finbrief.evaluation.cache import Cache

logger = logging.getLogger(__name__)


def cached_map(
    cache: Cache,
    kind: str,
    items: Iterable[Any],
    *,
    key_of: Callable[[Any], Mapping[str, Any]],
    produce: Callable[[Any], Any],
    workers: int = 1,
) -> tuple[Any, ...]:
    """`produce` over `items`, skipping every item the store already holds a value for.

    Ordered like `items`, so a caller may zip the results back against the rows that made them.
    An exception propagates: a stage that cannot finish must not report a partial result as a
    whole one, and the cells that did finish are already durable.
    """
    collected = list(items)

    def resolve(item: Any) -> Any:
        return cache.resolve(kind, key_of(item), lambda item=item: produce(item))

    if workers <= 1:
        return tuple(resolve(item) for item in collected)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # `map` rather than `submit`, so results come back in `items`' order and a caller can
        # zip them against the rows that made them — the property the serial path has for
        # free.
        return tuple(pool.map(resolve, collected))
