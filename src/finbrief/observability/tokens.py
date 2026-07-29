"""Token counts off a model reply, for the log lines that record a paid call.

The one instrument #10 asks for that nothing in the repo had: `grep usage_metadata` returned
nothing before T8, so four call sites held a reply that reported its own cost and dropped it.
This module reads it and nothing else — there is no framework here, no meter and no price
table. Dollar cost is the Tier-2 cost-meter's job (PLAN Phase 8); this is capture.

**An unreported count is absent, never zero.** OpenRouter fronts many upstreams and whether a
given one returns a `usage` block is not something this repo can assert — the same argument
ADR-0003's T4 amendment makes about `parallel_tool_calls`. A missing count means "this call's
spend was not reported", and writing that as `0` would put a fabricated number in a cost table
and understate a total that nothing else can recover. So the fields are *omitted*, and
`events.Samples` counts the lines that omitted them, so a total is read against a denominator.

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
    """Summed usage across `replies`, plus how many of them reported any.

    For the agent loop, where one turn is several paid calls and no single reply is the turn's
    cost. `metered_calls` is the denominator half and is not diagnostic detail: a turn where
    two of five calls reported usage has a *partial* total, and a partial total presented as a
    turn's spend is the silent-narrowing failure this repo keeps hitting. `{}` when nothing
    reported, so "no usage anywhere" stays absent rather than becoming a zeroed row.
    """
    metered = [fields for fields in (usage_fields(reply) for reply in replies) if fields]
    if not metered:
        return {}
    return {
        "input_tokens": sum(fields.get("input_tokens", 0) for fields in metered),
        "output_tokens": sum(fields.get("output_tokens", 0) for fields in metered),
        "metered_calls": len(metered),
    }
