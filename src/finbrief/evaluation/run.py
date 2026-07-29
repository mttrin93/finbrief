"""Running a stage over the golden set, one cached cell per item.

The whole of the harness's resumability lives in `cached_map`, which is why it is four lines:
a stage is a pure map from an item to a value, the cache decides which items still cost
anything, and a stage that dies leaves every finished cell on disk. There is no checkpoint
file, no progress ledger and no resume flag — the cache *is* the ledger, addressed by content,
so a resumed run cannot disagree with the run it is resuming (`evaluation/cache.py`).

**Serial by construction, and that is a decision.** A stage is a `for` loop rather than a
thread pool: OpenRouter answers a 429 to concurrency long before it answers one to volume, and
a killed run's cost is bounded by one cell either way. The 20–35 minute figure in #11's plan
is the serial one; if it needs to come down, the place to add bounded concurrency is here, in
one function, where the cache's per-cell atomicity already makes it safe.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
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
) -> tuple[Any, ...]:
    """`produce` over `items`, skipping every item the store already holds a value for.

    Ordered like `items`, so a caller may zip the results back against the rows that made them.
    An exception propagates: a stage that cannot finish must not report a partial result as a
    whole one, and the cells that did finish are already durable.
    """
    values = []
    for index, item in enumerate(items):
        values.append(cache.resolve(kind, key_of(item), lambda item=item: produce(item)))
        logger.debug("%s: %d/%s", kind, index + 1, "?")
    return tuple(values)
