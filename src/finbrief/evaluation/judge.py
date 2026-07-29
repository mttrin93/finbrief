"""The RAGAs judge: one model, four metrics, one call per (row, metric) — and a cache key.

ADR-0002 names all four metrics; this module is the only place `ragas` is called from, for the
reason `retrieval/vectorstore.py` is the only place Chroma is opened. Two of those reasons are
specific to this library and neither is cosmetic.

**It phones home by default.** `ragas._analytics.track` POSTs every metric completion to
`t.explodinggradients.com`, decorated `@silent`, flushed from a background thread and again at
`atexit`. `_silence_ragas_telemetry` below is what stops it, and it runs **before** `ragas` is
imported anywhere in this process because `ragas._analytics.do_not_track` is `lru_cache`d — a
switch flipped after the first metric has run is a switch that did nothing. `tests/conftest.py`
sets the same variable for the suite; two mechanisms, failing independently, on the
`security/advice.py` principle that a hole is a hole whether today's code walks through it.
Measured with the switch absent: one `track()` call attempted `t.explodinggradients.com`, raised
nothing, logged nothing, and was visible only in `conftest.EGRESS_ATTEMPTS`.

**`ragas.evaluate()` is deliberately not used.** Three reasons, in order of weight: the cache
this harness turns on needs a result per *(row, metric)* and `evaluate()` returns a table over a
dataset; a metric has to be skippable per row (ADR-0002's amendment, and the
`tool-augmented` relevancy column below); and `evaluate()` is one of the two entry points
decorated with the analytics tracker. Calling `metric.single_turn_ascore` directly costs a
`for` loop and buys all three.

**One metric here is not reproducible, by construction, and it is fenced off rather than
caveated.** `ragas.llms.base.get_temperature` returns **0.3 whenever n > 1**, and
`ResponseRelevancy` asks for `n=strictness=3`. So response relevancy moves between runs on any
judge at any temperature this code names, and it is therefore excluded from every pre-registered
decision — see `EXCLUDED_FROM_HYPOTHESES` and ADR-0002's T10 amendment. ADR-0005's
falsification clause and its §4 re-examination trigger both rest on context precision and
context recall, which are single-call metrics at temperature 0.01 and reproducible to the
judge's own sampling.
"""

from __future__ import annotations

import os

#: Every metric name a pre-registered decision may **not** rest on.
#:
#: Response relevancy is non-reproducible by construction (see the module docstring), so a delta
#: in that column is not evidence of anything and must never be read as confirming or refuting a
#: hypothesis. Named as data rather than left to prose because `report.py` renders the exclusion
#: into the artifact and `hypotheses.py` asserts against it: a caveat in a paragraph is a caveat
#: the next reader of the table does not see.
EXCLUDED_FROM_HYPOTHESES: frozenset[str] = frozenset({"answer_relevancy"})


def _silence_ragas_telemetry(env: dict[str, str] | None = None) -> None:
    """Set `RAGAS_DO_NOT_TRACK`, before anything in this process imports `ragas`.

    Takes the mapping as an argument so the behaviour is testable without mutating the real
    environment — the same reason `Settings.from_env` does. Assigned rather than defaulted: a
    developer with `RAGAS_DO_NOT_TRACK=false` exported is exactly the case worth overriding.
    """
    target = os.environ if env is None else env
    target["RAGAS_DO_NOT_TRACK"] = "true"


_silence_ragas_telemetry()
