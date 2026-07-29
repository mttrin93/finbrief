"""Killing a run mid-stage must cost the remainder, not the run (#11, addition 1).

The scenario is the one a 20–35 minute run over ~2,200 round trips actually meets: a 429, a
laptop that sleeps, a `^C`. What is asserted here is stronger than "the cache is used" — it is
that the *second* run pays for exactly the cells the first did not finish, and that the cells it
replays are byte-identical to what the first run computed. A cache that recomputed everything
would pass an equality assertion; a cache that served a truncated cell would pass a spend
assertion. Both are checked, per item.
"""

from __future__ import annotations

import pytest

from finbrief.evaluation.cache import Cache
from finbrief.evaluation.run import cached_map


class DiesOnceAt:
    """A paid call that raises the first time it reaches `victim`, then behaves.

    Stands in for a provider 429: the failure is transient and per-call, which is what makes the
    resumed run able to finish at all.
    """

    def __init__(self, victim: str) -> None:
        self.victim = victim
        self.calls: list[str] = []
        self.died = False

    def __call__(self, item: str) -> dict[str, str]:
        self.calls.append(item)
        if item == self.victim and not self.died:
            self.died = True
            raise RuntimeError(f"429 Too Many Requests while scoring {item}")
        return {"row": item, "score": f"scored-{item}"}


ROWS = ("S1", "S2", "S3", "S4", "S5")


def test_a_killed_stage_resumes_and_pays_only_for_what_it_did_not_finish(tmp_path):
    cache = Cache(tmp_path / "eval-cache")
    produce = DiesOnceAt("S4")

    def run() -> tuple[dict[str, str], ...]:
        return cached_map(
            cache, "judge", ROWS, key_of=lambda row: {"row": row}, produce=produce
        )

    with pytest.raises(RuntimeError, match="429"):
        run()

    # The three cells that completed before the death are on disk; the fourth is not.
    assert produce.calls == ["S1", "S2", "S3", "S4"]

    resumed = run()

    assert [value["row"] for value in resumed] == list(ROWS)
    # S1–S3 replayed and cost nothing; S4 recomputed because it never finished; S5 was never
    # reached the first time. So the only cell paid for twice is the one the run died inside —
    # which is the honest meaning of "resumes where it stopped".
    assert produce.calls == ["S1", "S2", "S3", "S4", "S4", "S5"]
    # `misses` is what the artifact quotes as spend, so it is asserted *against the producer*
    # and not against a number typed here: one miss is one paid call, by definition, and the
    # two cannot drift. Both runs share this `Cache`, so it counts 4 + 2, and the cell the run
    # died inside is a miss in each of them — which is the arithmetic of "one cell paid for
    # twice".
    assert cache.stats("judge").misses == len(produce.calls) == 6
    assert cache.stats("judge").hits == 3


def test_the_resumed_run_reports_the_same_values_the_killed_run_computed(tmp_path):
    # The other half: a resumed run's *numbers* must be the killed run's numbers. If the store
    # changed a value's shape on the way through (JSON has no tuples), the report would depend
    # on where the run happened to die, which is unfalsifiable from the artifact.
    cache = Cache(tmp_path / "eval-cache")
    produce = DiesOnceAt("S4")

    def run() -> tuple[dict[str, str], ...]:
        return cached_map(
            cache, "judge", ROWS, key_of=lambda row: {"row": row}, produce=produce
        )

    with pytest.raises(RuntimeError):
        run()
    resumed = run()

    cold = cached_map(
        Cache(tmp_path / "cold-cache"),
        "judge",
        ROWS,
        key_of=lambda row: {"row": row},
        produce=DiesOnceAt("none-of-them"),
    )
    assert resumed == cold


def test_a_second_full_run_spends_nothing(tmp_path):
    # The warm re-run #11's plan prices at ~$0: every cell hits, and the producer is never
    # called. `pytest.fail` as the producer is the assertion — a spy that counts to zero can be
    # read as a passing test for the wrong reason.
    cache = Cache(tmp_path / "eval-cache")
    cached_map(
        cache, "judge", ROWS, key_of=lambda row: {"row": row}, produce=DiesOnceAt("none")
    )

    again = cached_map(
        Cache(tmp_path / "eval-cache"),
        "judge",
        ROWS,
        key_of=lambda row: {"row": row},
        produce=lambda row: pytest.fail(f"recomputed {row} on a warm cache"),
    )

    assert [value["row"] for value in again] == list(ROWS)


def test_changing_the_stage_key_invalidates_only_that_stage(tmp_path):
    # A re-run after the judge model changes must re-judge and must *not* re-retrieve. The two
    # stages are separate kinds with separate keys, which is what makes a partial re-run cheap
    # and correct at the same time.
    cache = Cache(tmp_path / "eval-cache")
    cached_map(
        cache, "retrieval", ROWS, key_of=lambda row: {"row": row}, produce=DiesOnceAt("none")
    )
    cached_map(
        cache,
        "judge",
        ROWS,
        key_of=lambda row: {"row": row, "judge": "gpt-4.1-mini"},
        produce=DiesOnceAt("none"),
    )

    rejudged = DiesOnceAt("none")
    cached_map(
        cache,
        "judge",
        ROWS,
        key_of=lambda row: {"row": row, "judge": "gpt-5-mini"},
        produce=rejudged,
    )
    reretrieved = cached_map(
        cache,
        "retrieval",
        ROWS,
        key_of=lambda row: {"row": row},
        produce=lambda row: pytest.fail(f"re-retrieved {row} after only the judge changed"),
    )

    assert rejudged.calls == list(ROWS), "the new judge model did not re-score every row"
    assert len(reretrieved) == len(ROWS)
