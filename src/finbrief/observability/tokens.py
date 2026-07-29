"""Token counts off a model reply, for the log lines that record a paid call.

The one instrument #10 asks for that nothing in the repo had: `grep usage_metadata` returned
nothing before T8. Four call sites held a reply that reported its own cost and dropped it, and
**three of them are metered here** — generation, the planner and the agent loop. The gate's
classifier is the fourth and stays unmeasured on purpose (ADR-0011: `classify()` returns a bare
`Verdict`, and it is the cheapest call in the system). Three and four are both true of different
things and were written as one number here once, so both are named: four sites *dropped* a
count, three sites *report* one. This module reads them and nothing else — no framework, no
meter, no price table. Dollar cost is the Tier-2 cost-meter's job (PLAN Phase 8); this is
capture.

**An unreported count is absent, never zero.** OpenRouter fronts many upstreams and whether a
given one returns a `usage` block is not something this repo can assert — the same argument
ADR-0003's T4 amendment makes about `parallel_tool_calls`. A missing count means "this call's
spend was not reported", and writing that as `0` would put a fabricated number in a cost table
and understate a total that nothing else can recover. So the fields are *omitted*, and
`events.Samples` counts the lines that omitted them, so a total is read against a denominator.

**And that rule holds per field, not per reply** — which is where it was broken. A provider can
report one half of a pair (`{"input_tokens": 120, "output_tokens": None}`), and summing such a
turn with `.get(name, 0)` wrote `output_tokens: 0` onto the line while counting the reply as
metered: a fabricated zero indistinguishable from a real one, on the very criterion #10's AC-1
is about, in the module whose docstring forbids it (issue #10 review). Each field is therefore
summed over the replies that reported **that field**, carries **its own** denominator, and is
omitted entirely when no reply reported it.

`total_tokens` is deliberately not logged: it is `input + output` and a third number is a
third thing that can disagree with the other two.
"""

from __future__ import annotations

from typing import Any


def usage_fields(reply: Any) -> dict[str, int]:
    """`{"input_tokens": n, "output_tokens": m}`, or `{}` when the reply reported no usage.

    Spread into a `log_event` call, so an unreported cost adds no keys at all rather than
    keys holding a made-up value.
    """
    usage = getattr(reply, "usage_metadata", None)
    if not isinstance(usage, dict):
        return {}
    fields = {
        name: usage[name]
        for name in ("input_tokens", "output_tokens")
        # A count that arrives as `None` is as absent as a key that never arrived, and a
        # `bool` is an `int` in Python — which is why this checks the type rather than
        # trusting the provider's shape.
        if isinstance(usage.get(name), int) and not isinstance(usage.get(name), bool)
    }
    return fields


def usage_total(replies: Any) -> dict[str, int]:
    """Summed usage across `replies`, each field with the count of replies that reported it.

    For the agent loop, where one turn is several paid calls and no single reply is the turn's
    cost. `calls` is how many replies were considered and `<field>_calls` how many carried that
    field; neither is diagnostic detail. A turn where two of five calls reported usage has a
    *partial* total, and a partial total presented as a turn's spend is the silent-narrowing
    failure this repo keeps hitting.

    **The denominator is per field, because the absence is.** A single `metered_calls` counted
    replies reporting *any* usage, so a reply that reported only `input_tokens` made
    `output_tokens` read as summed-and-complete when nothing had reported it — see the module
    docstring. `{}` when no reply reported any field, so "no usage anywhere" stays absent rather
    than becoming a zeroed row.
    """
    reported = [usage_fields(reply) for reply in replies]
    totals: dict[str, int] = {}
    for fields in reported:
        for name, count in fields.items():
            totals[name] = totals.get(name, 0) + count
            totals[f"{name}_calls"] = totals.get(f"{name}_calls", 0) + 1
    if not totals:
        return {}
    # Only once something was reported: `calls` on an all-absent turn would be the zeroed row
    # the docstring rules out, and it is the one number a reader can get from
    # `searches`/`steps`.
    return {"calls": len(reported), **totals}
