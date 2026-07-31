"""The analytics dashboard's arithmetic (T13, #14) — aggregates over the one reader's output.

**This module parses nothing.** `observability/events.py` is the one reader of the sink, paired
with the one emitter, and `observability/spend.py` is the precedent for what sits above it:
take an `EventLog`, do arithmetic, return dataclasses. A `json.loads` here — or on the page —
would be the drift that pairing exists to prevent, where a renamed field reads back as `None`
and `None` aggregates as nothing.

**Why not in `events.py`.** That module's docstring refuses statistics on purpose: it returns
samples, because ADR-0005's and ADR-0006's p50 budgets belong to the *reports* that quote them,
and `security/report.py` already owns one median over in-process `Screening` objects. A
dashboard is such a report. So the reader keeps its contract and the statistics live here.

**Absence is a state, not a zero, and this module has five of them.** `SinkState` names them —
the log is off, the file was named and never written, the path exists and cannot be read, the
file exists and holds no events, and the file is readable — because the page owes a different
sentence to each, and a chart of zeros is a claim of no traffic. The third arrived late: the
first version enumerated four, and a directory, an unopenable file and a file that is not valid
UTF-8 each raised out of `open_sink` into a traceback, which is the one rendering a page built
around honest absence may not have (code review of #14).
Below that, every figure follows the same rule one layer down: a
`Distribution` over no samples has no p50 rather than a p50 of `0.0`, a `Rate` over no
denominator has no percentage, and a `Tally` reports the lines that carried nothing beside the
values it counted. `within_budget` is `bool | None` for the same reason: `False` reads as
"measured and missed", which is a different claim from "not measured".

**Three constants and one function are shared or bound rather than copied**, and each is named
where it sits: `calls_behind` is imported from `spend.py` (one definition of what a line costs),
`PLANNER_DISABLED_CAP` is a third copy of ADR-0004 §6's cap bound by test to the other two, and
`Rate`/`p50` are bound by test to `evaluation/deferrals.Rate` and `evaluation/latency.p50` —
which cannot be imported, because `evaluation/` is the harness and the app must not depend on
it.

**What this module deliberately cannot do.** It never separates an app session from an
evaluation run: the sink is append-only across every run that named it and **no field
distinguishes them** (ADR-0011's T10 amendment, where a planner p50 over 13 appended runs
shipped as one run's). A heuristic on `turn_id` shape would be a separation nothing could
check, so the page states the pool instead of narrowing it. It also never carries a blocked
question's `normalised` text: that field is ADR-0006's one bounded exception to
no-user-content, kept for an auditor with a grep, and a dashboard rendering it would widen a
bound CLAUDE.md says nothing may widen.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path

from finbrief.config import (
    GATE_LATENCY_BUDGET_MS,
    GATE_LATENCY_BUDGET_PREREGISTERED_MS,
    TRANSLATION_LATENCY_BUDGET_MS,
)
from finbrief.observability.events import Event, EventLog, read_events
from finbrief.observability.spend import TOKEN_EVENTS, TOKEN_FIELDS, Tokens, calls_behind

#: `spend.calls_behind`, under the name the tests bind. One definition of "how many paid chat
#: calls does this line stand behind", including the cap-of-zero rule below.
_calls_behind = calls_behind

#: The `max_sub_queries` at which the planner makes **no chat round at all** (ADR-0004 §6: the
#: cap removes the `model.invoke`, it does not truncate its output).
#:
#: The **third** copy of this fact. `spend.PLANNER_SILENT_CAP` keeps a complete total from
#: reporting as partial, and `evaluation/latency.PLANNER_DISABLED_CAP` keeps the ablation arms
#: out of a latency pool; this keeps a disabled line out of a planner median. All three are
#: duplicated rather than imported for the same reason — the app must not depend on
#: `evaluation/`, and the harness must not depend on the app — and bound to each other by
#: `test_the_planner_disabled_cap_agrees_with_the_two_that_already_encode_it` in
#: `tests/test_analytics.py`.
PLANNER_DISABLED_CAP = 0

#: The three events that record a layer **failing open**. Each is a warning about a control
#: that did not run rather than one that let something through, which is why they are counted
#: beside the gate's verdicts rather than folded into them: a classifier unreachable for an
#: hour is a gate that was two layers deep for an hour, and nothing else on the page says so.
FAIL_OPEN_EVENTS = (
    "gate_classifier_unavailable",
    "gate_classifier_unparsed",
    "output_validator_unavailable",
)


class SinkUnreadable(RuntimeError):
    """A caller asked for a log that is not there — off, never written, or unreadable."""


class SinkState(StrEnum):
    """Which of five things is true about the sink, because each is a different sentence.

    `OFF`, `MISSING` and `UNREADABLE` carry no log at all; `EMPTY` carries one whose `malformed`
    count is the difference between a file nobody wrote and a file whose lines are not ours.

    **`UNREADABLE` was the gap, and it is the reason this enum is not three states.** A page
    cannot refuse to render the way `latency.load_log` refuses to proceed, so every case a real
    path resolves to owes a sentence — and the first version enumerated four while a directory,
    a file the process cannot open and a file that is not valid UTF-8 each reached
    `read_events` and raised, surfacing as a traceback where the sentence should have been
    (code review of #14). Measured: `IsADirectoryError`, `PermissionError`,
    `UnicodeDecodeError`.
    """

    OFF = "off"
    MISSING = "missing"
    UNREADABLE = "unreadable"
    EMPTY = "empty"
    READABLE = "readable"


@dataclass(frozen=True, slots=True)
class Sink:
    """What the page is reading, and whether there is anything in it.

    **`log` is `None` rather than an empty `EventLog` for the three absent states.** "Nobody
    enabled the log" read as "this run emitted nothing" is the failure `events.read_events`
    raises about, and it arrives one layer up as an empty aggregate rendering as zero activity.
    `readable` is the accessor that raises, so a caller that has not checked the state gets an
    exception rather than a chart of nothing.
    """

    path: Path | None
    state: SinkState
    size_bytes: int
    log: EventLog | None
    first_event: datetime | None
    last_event: datetime | None
    #: Why the sink could not be read — the exception's **type name**, and only for
    #: `UNREADABLE`. A type rather than the message on `finance/cache.py`'s rule: a client's
    #: error string can carry a path or a URL, and this one is rendered on a page. It is carried
    #: at all because "unreadable" alone sends a reader to the wrong knob — a directory where a
    #: file was meant is a typo in `FINBRIEF_LOG_FILE`, and a `PermissionError` is not.
    reason: str | None = None

    @property
    def readable(self) -> EventLog:
        if self.log is None:
            raise SinkUnreadable(
                f"the sink is {self.state.value}, so there is no log to aggregate. A "
                f"total over an absent log is an absence reported as a measurement."
            )
        return self.log

    @property
    def events(self) -> int:
        return 0 if self.log is None else len(self.log.events)

    @property
    def malformed(self) -> int:
        return 0 if self.log is None else self.log.malformed


def open_sink(path: Path | str | None) -> Sink:
    """Resolve the sink at `path` into one of `SinkState`'s five cases.

    **The whole file, with no `start_offset`**, and that is the one place this page differs
    from every other reader of the sink. `evaluation/latency.py` windows by an offset because
    its figures claim to be one run's; the sidebar's spend meter filters by `turn_id` because
    its figures claim to be one conversation's. This page's figures claim to be *the file's*,
    which is the only claim a whole-file read supports — so it makes that claim in its header
    rather than narrowing to a pool it cannot name.

    **Two checks, and they catch different things — which is why one did not subsume the
    other.** The existence check stays ahead of the read so that a named sink nobody has
    written to is its own state: catching `FileNotFoundError` instead would work, and would
    also file an unopenable path under "not written yet". The `try` is for what the read itself
    can throw once the path *does* exist, which is a disjoint set — `IsADirectoryError` and
    `PermissionError` from the `open`, and `UnicodeDecodeError` from the decode. The first
    version had only the existence check and let all three reach the page as a traceback (code
    review of #14).

    `UnicodeDecodeError` is the one worth naming, because it defeats a promise made one layer
    down. `EventLog.malformed` exists for a run killed mid-write leaving a truncated final
    line — and a write truncated inside a multi-byte sequence makes the *whole file*
    undecodable, so the single corruption `malformed` was built to survive was the one that took
    the page down. It is caught here rather than repaired by decoding leniently: bytes this
    emitter did not write are a different problem from a line this reader cannot parse, and
    guessing at the bytes would report the second when it is the first.
    """
    if path is None:
        return Sink(
            path=None,
            state=SinkState.OFF,
            size_bytes=0,
            log=None,
            first_event=None,
            last_event=None,
        )
    path = Path(path)
    if not path.exists():
        return Sink(
            path=path,
            state=SinkState.MISSING,
            size_bytes=0,
            log=None,
            first_event=None,
            last_event=None,
        )
    try:
        log = read_events(path)
    except (OSError, UnicodeDecodeError) as exc:
        return Sink(
            path=path,
            state=SinkState.UNREADABLE,
            size_bytes=_size(path),
            log=None,
            first_event=None,
            last_event=None,
            reason=type(exc).__name__,
        )
    stamps = [event.ts for event in log.events]
    return Sink(
        path=path,
        state=SinkState.READABLE if log.events else SinkState.EMPTY,
        size_bytes=_size(path),
        log=log,
        first_event=min(stamps) if stamps else None,
        last_event=max(stamps) if stamps else None,
    )


def _size(path: Path) -> int:
    """`path`'s size in bytes, or `0` when even that cannot be asked.

    `stat` is a second syscall after the read, so it can fail where the read did not — a sink
    rotated out from under a page mid-render is the ordinary case. A size is a decoration on
    this page and a failure to read one may not cost the reader the panels: `0` here is the
    honest floor for a figure nobody uses in arithmetic, unlike every other absence in this
    module.
    """
    try:
        return path.stat().st_size
    except OSError:
        return 0


# --- The three primitives every panel is built from -------------------------------------


@dataclass(frozen=True, slots=True)
class Rate:
    """A count over a denominator, and never a bare percentage.

    `None` for the rate when the denominator is zero: a divergence rate over no searches is not
    0% divergence, and a bracket rate over no logged sessions is not 0% support. Identical in
    contract to `evaluation/deferrals.Rate`, bound to it by test and not imported from it — see
    the module docstring.
    """

    label: str
    hits: int
    total: int

    @property
    def rate(self) -> float | None:
        return None if not self.total else self.hits / self.total

    def render(self) -> str:
        if self.rate is None:
            return f"{self.label}: **not measured** (0 observations)"
        return f"{self.label}: **{self.rate:.0%}** ({self.hits}/{self.total})"


@dataclass(frozen=True, slots=True)
class Distribution:
    """The values one numeric field took, summarised — or an absence, never a zero.

    `absent` counts the lines of the right kind that carried no such value, so a caller
    printing a p50 can say what fraction of the population is behind it. Every figure is `None`
    over an empty sample: `evaluation/latency.p50` raises for the same reason, and the
    difference is only that a page has to render the refusal rather than abort on it.
    """

    label: str
    values: tuple[float, ...]
    absent: int

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def total(self) -> int:
        return self.count + self.absent

    @property
    def measured(self) -> bool:
        return bool(self.values)

    @property
    def p50(self) -> float | None:
        """`statistics.median`, which is the definition the harness's `p50` uses.

        Interpolating rather than nearest-rank, and bound to `evaluation/latency.p50` by test:
        an even-length sample is precisely where the two could disagree while both look right.
        """
        return float(statistics.median(self.values)) if self.values else None

    @property
    def p90(self) -> float | None:
        return percentile(self.values, 0.9)

    @property
    def minimum(self) -> float | None:
        return min(self.values) if self.values else None

    @property
    def maximum(self) -> float | None:
        return max(self.values) if self.values else None

    def within(self, budget: float) -> bool | None:
        """Whether the median clears `budget` — `None` when there is no median.

        Not `False`, which a reader takes as "measured and missed". An unmeasured budget and a
        missed one are two different claims and they get two different renderings.
        """
        return None if self.p50 is None else self.p50 <= budget


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """The nearest-rank percentile of `values`, or `None` over nothing.

    **An observed value, deliberately, and it is why this is not `statistics.quantiles`.**
    `p50` interpolates because it must agree with the harness's median; there is no second
    definition of a p90 in this repo to agree with, so this returns one of the samples — an
    interpolated p90 over seven observations is a number between two measurements, and what
    this page reports is what was measured. `statistics.quantiles` also raises below two data
    points, which would make a one-sample panel an exception rather than a figure.
    """
    if not values:
        return None
    ordered = sorted(values)
    # `round` before `ceil`, and it is not decoration: `0.9 * 10` is `9.000000000000002` in
    # binary floating point, so a bare `ceil` returns 10 for ten samples and the p90 of a
    # ten-sample set becomes its maximum. Measured, which is why the test pins `9.0`.
    rank = math.ceil(round(quantile * len(ordered), 6)) - 1
    return float(ordered[max(0, min(len(ordered) - 1, rank))])


@dataclass(frozen=True, slots=True)
class Tally:
    """How often each value of one field occurred, with both kinds of nothing reported.

    `total` is the sum of the counts and `lines` is the population they were counted over, and
    the two differ on purpose: a list-valued field (`tools_used`) contributes several members
    per line or none, so conflating them would report "3 tool uses across 3 turns" as a
    per-turn rate it is not. `absent` is the lines that carried no value at all.
    """

    label: str
    rows: tuple[tuple[str, int], ...]
    lines: int
    absent: int

    @property
    def total(self) -> int:
        return sum(count for _, count in self.rows)

    @property
    def measured(self) -> bool:
        return bool(self.rows)

    @property
    def counts(self) -> Mapping[str, int]:
        """The rows as a mapping, for a chart that wants one. Order is preserved."""
        return dict(self.rows)


def tally(events: Iterable[Event], field: str, *, label: str) -> Tally:
    """Count `field`'s values across `events`.

    A value of `None` counts as absent, and a caller who needs to tell "the emitter wrote null"
    from "the field is missing" filters first — which is what `gate_summary` does, tallying
    `layer` over *blocked* lines only. An allowed screening writes `layer: null`, so a tally
    over every line would either count `None` as a layer or file every allowed question under
    absent: two ways of describing the same wrong denominator.
    """
    selected = list(events)
    counter: Counter[str] = Counter()
    absent = 0
    for event in selected:
        value = event.field(field)
        if value is None:
            absent += 1
        else:
            counter[str(value)] += 1
    return Tally(label=label, rows=_ordered(counter), lines=len(selected), absent=absent)


def tally_each(events: Iterable[Event], field: str, *, label: str) -> Tally:
    """Count the *members* of a list-valued `field` across `events`.

    `tools_used` is a list per turn, so `tally` would count the string
    `"['get_stock_data', 'calculate_ratios']"` as one distinct tool. An empty list is a line
    that carried the field and contributed nothing — present, not absent, because "this turn
    used no tool" is a measurement.
    """
    selected = list(events)
    counter: Counter[str] = Counter()
    absent = 0
    for event in selected:
        value = event.field(field)
        if not isinstance(value, list):
            absent += 1
            continue
        counter.update(str(member) for member in value)
    return Tally(label=label, rows=_ordered(counter), lines=len(selected), absent=absent)


def distribution(events: Iterable[Event], field: str, *, label: str) -> Distribution:
    """`field`'s numeric values across `events`, and how many lines carried none.

    **A `bool` is not a measurement**, and excluding it is not pedantry: `isinstance(True, int)`
    is `True` in Python, so a distribution that accepted ints would average `translation` and
    print a median of 0.5 ms for a flag.
    """
    selected = list(events)
    values: list[float] = []
    absent = 0
    for event in selected:
        value = event.field(field)
        if isinstance(value, bool) or not isinstance(value, int | float):
            absent += 1
        else:
            values.append(float(value))
    return Distribution(label=label, values=tuple(values), absent=absent)


def _ordered(counter: Counter[str]) -> tuple[tuple[str, int], ...]:
    """Count descending, then key ascending — a deterministic order, not the dict's insertion.

    A chart whose bars reorder between two reads of the same file looks like changed data.
    """
    return tuple(sorted(counter.items(), key=lambda row: (-row[1], row[0])))


def _rate(events: Sequence[Event], field: str, *, label: str) -> Rate:
    """How many of `events` reported `field` as truthy, over how many reported it at all.

    The denominator is the lines that carried the field, not every line: a turn written before
    `grounded` existed is not an ungrounded turn (the checkpoint rule, applied to a log).
    """
    present = [event for event in events if event.field(field) is not None]
    return Rate(
        label=label,
        hits=sum(1 for event in present if event.field(field)),
        total=len(present),
    )


# --- Panel 1: activity over time --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Day:
    """One day's counts, as the activity chart reads them."""

    day: date
    screened: int
    answered: int
    failed: int


@dataclass(frozen=True, slots=True)
class Activity:
    """Questions screened, turns answered and turns failed, by day.

    Three counts and not one, because they are three populations with three denominators.
    `input_gate` is the *app's* front door, so a harness run contributes `agent_turn` lines with
    no gate line beside them and the two numbers legitimately disagree — which the panel says,
    rather than presenting either as "questions asked".
    """

    by_day: tuple[Day, ...]
    screened: int
    answered: int
    failed: int

    @property
    def measured(self) -> bool:
        return bool(self.by_day)


def activity(log: EventLog) -> Activity:
    """Screenings, answered turns and failed turns per day over the whole sink."""
    days: dict[date, list[int]] = {}
    for name, slot in (("input_gate", 0), ("agent_turn", 1), ("chat_turn_failed", 2)):
        for event in log.of(name):
            days.setdefault(event.ts.date(), [0, 0, 0])[slot] += 1
    rows = tuple(
        Day(day=day, screened=counts[0], answered=counts[1], failed=counts[2])
        for day, counts in sorted(days.items())
    )
    return Activity(
        by_day=rows,
        screened=sum(row.screened for row in rows),
        answered=sum(row.answered for row in rows),
        failed=sum(row.failed for row in rows),
    )


# --- Panel 4: the gate ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateSummary:
    """What ADR-0006's gate did, over whatever screenings this sink holds.

    **Both budgets, and the pre-registered one is not dropped for being missed.**
    `GATE_LATENCY_BUDGET_PREREGISTERED_MS` was revised to `GATE_LATENCY_BUDGET_MS` by
    ADR-0006's T7 amendment, and `security/report.py` prints both on purpose: deleting the
    earlier figure turns a revised pre-registration into one that had always held.

    No field here carries a question. The `normalised` text on a blocked line is the log's one
    bounded exception to no-user-content and stays in the log.
    """

    screenings: int
    blocked: Rate
    by_layer: Tally
    by_rule: Tally
    classifier_ran: Rate
    verdicts: Tally
    latency: Distribution
    fail_open: Tally
    preregistered_budget_ms: float
    budget_ms: float

    @property
    def within_budget(self) -> bool | None:
        return self.latency.within(self.budget_ms)

    @property
    def within_preregistered_budget(self) -> bool | None:
        return self.latency.within(self.preregistered_budget_ms)


def gate_summary(
    log: EventLog,
    *,
    budget_ms: float = GATE_LATENCY_BUDGET_MS,
    preregistered_budget_ms: float = GATE_LATENCY_BUDGET_PREREGISTERED_MS,
) -> GateSummary:
    """The gate's verdicts, layers, rules, latency and fail-open events.

    The budgets are arguments with `config` defaults rather than literals read here, on the
    `evaluation/latency.translation_cost` precedent: they are pre-registered figures, and
    CLAUDE.md's single-source list names both.
    """
    screenings = log.of("input_gate")
    blocks = [event for event in screenings if event.field("blocked")]
    return GateSummary(
        screenings=len(screenings),
        blocked=Rate(label="blocked", hits=len(blocks), total=len(screenings)),
        # Over blocks only — see `tally`. An allowed screening has no layer and no rule.
        by_layer=tally(blocks, "layer", label="layer"),
        by_rule=tally(blocks, "rule", label="rule"),
        classifier_ran=_rate(screenings, "classifier_ran", label="classifier reached"),
        verdicts=tally(
            [e for e in screenings if e.field("classifier_verdict") is not None],
            "classifier_verdict",
            label="verdict",
        ),
        latency=distribution(screenings, "latency_ms", label="gate latency"),
        fail_open=_tally_events(log, FAIL_OPEN_EVENTS, label="fail-open event"),
        preregistered_budget_ms=preregistered_budget_ms,
        budget_ms=budget_ms,
    )


def _tally_events(log: EventLog, names: Sequence[str], *, label: str) -> Tally:
    """How many lines of each of `names` the log holds — a tally over event names.

    `lines` is the total across all of them and `absent` is zero by construction: an event name
    is on every line by definition (`events._event` refuses a line without one), so there is no
    third state here to report.
    """
    counter: Counter[str] = Counter({name: len(log.of(name)) for name in names if log.of(name)})
    return Tally(label=label, rows=_ordered(counter), lines=sum(counter.values()), absent=0)


# --- Panel 6: the agent's behaviour, and the bracket rate -------------------------------


@dataclass(frozen=True, slots=True)
class AgentBehaviour:
    """What the shipped path did, per `agent_turn` and `agent_query` line.

    `divergence` is the **complement** of `verbatim`: ADR-0003 §1 has the tool description ask
    for the question verbatim, and §2 of its T4 amendment records that the rule is measured,
    not enforced. Counting verbatim searches and labelling the result divergence is the
    inversion worth naming, since both numbers are plausible on a panel.
    """

    turns: int
    searches: int
    divergence: Rate
    grounded: Rate
    searched: Rate
    tools_used: Tally
    searches_per_turn: Distribution
    finance_calls: Distribution
    turn_latency: Distribution

    @property
    def measured(self) -> bool:
        return self.turns > 0


def agent_behaviour(log: EventLog) -> AgentBehaviour:
    """Searches, divergence, grounding and tool choice across the sink's agent turns.

    Divergence is summed from `agent_turn`'s own `searches`/`verbatim_searches` pair rather
    than counted over `agent_query` lines, because the emitter puts both on the turn line for
    exactly this reader ("so a reader who only aggregates turns still sees the divergence
    rate", `agent/agent.py`) — and a turn whose per-search lines were written by an older
    deploy still contributes.
    """
    turns = log.of("agent_turn")
    searches = sum(int(event.field("searches") or 0) for event in turns)
    verbatim = sum(int(event.field("verbatim_searches") or 0) for event in turns)
    return AgentBehaviour(
        turns=len(turns),
        searches=searches,
        divergence=Rate(label="divergence", hits=searches - verbatim, total=searches),
        grounded=_rate(turns, "grounded", label="grounded"),
        searched=_rate(turns, "searched", label="searched the KB"),
        tools_used=tally_each(turns, "tools_used", label="tool"),
        searches_per_turn=distribution(turns, "searches", label="searches per turn"),
        finance_calls=distribution(turns, "finance_calls", label="finance calls per turn"),
        turn_latency=distribution(turns, "latency_ms", label="turn latency"),
    )


@dataclass(frozen=True, slots=True)
class Citations:
    """Square-bracket adherence, over the turns this sink recorded.

    **This is the T5 deferral `docs/verification/evaluation.md` reports as unmeasured**, and
    the reason it is unmeasured there is ADR-0011's amendment: `security/markers.py`'s
    `citation_markers` is written by `app/Home.py` and by nothing else, so T10's ten live agent
    turns through `agent.answer` produced zero such lines. This page is the surface where the
    instrument becomes readable — and what it yields is **observational over logged sessions**,
    not the controlled measurement that ticket wanted: the population is whoever used the app
    with the sink on, not a stratified set.

    `support` is resolved markers over every marker issued, which is the rate that amendment
    names. `clean` is the turn-level view of the same behaviour — a turn with nothing
    unresolved and nothing non-numeric — and the two differ whenever one bad turn carries many
    markers.
    """

    turns: int
    support: Rate
    clean: Rate
    resolved: int
    unresolved: int
    non_numeric: int

    @property
    def measured(self) -> bool:
        return self.turns > 0


def citations(log: EventLog) -> Citations:
    """The bracket rate and the clean-turn rate over the sink's `citation_markers` lines."""
    lines = log.of("citation_markers")
    resolved = sum(int(event.field("resolved") or 0) for event in lines)
    unresolved = sum(len(event.field("unresolved") or ()) for event in lines)
    return Citations(
        turns=len(lines),
        support=Rate(label="cited-marker support", hits=resolved, total=resolved + unresolved),
        clean=_rate(lines, "clean", label="clean turns"),
        resolved=resolved,
        unresolved=unresolved,
        non_numeric=sum(int(event.field("non_numeric") or 0) for event in lines),
    )


# --- Panel 5: tools ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolSummary:
    """Which tools ran, for which tickers, and how they failed.

    **`by_ticker` is "tickers the finance tools were called for", not "tickers asked about"**,
    and the panel says so: a filings-only question puts `question_chars` in the log and no
    ticker, by the no-user-content rule. The narrower claim is the true one.

    **There is deliberately no cache hit rate here.** `tool_call.age_seconds` is `round()`ed at
    the emitter, so a cache hit 400 ms after a fetch reads as `0` and a rate derived from it
    would be wrong invisibly — the check-that-cannot-fail class CLAUDE.md names. `stale` and
    `stale_fallbacks` are explicit fields and are what this reports instead.
    """

    calls: int
    by_tool: Tally
    by_ticker: Tally
    stale: Rate
    age_seconds: Distribution
    refused: Tally
    unavailable: Tally
    stale_fallbacks: Tally

    @property
    def measured(self) -> bool:
        return self.calls > 0 or self.refused.measured or self.unavailable.measured


def tool_summary(log: EventLog) -> ToolSummary:
    """Tool calls, refusals, unavailabilities and stale fallbacks over the whole sink."""
    calls = log.of("tool_call")
    return ToolSummary(
        calls=len(calls),
        by_tool=tally(calls, "tool", label="tool"),
        by_ticker=tally(calls, "ticker", label="ticker"),
        stale=_rate(calls, "stale", label="served stale"),
        age_seconds=distribution(calls, "age_seconds", label="data age"),
        refused=tally(log.of("tool_refused"), "reason", label="refusal"),
        unavailable=tally(log.of("tool_unavailable"), "tool", label="tool"),
        stale_fallbacks=tally(log.of("stale_fallback"), "source", label="source"),
    )


# --- Panel 7: token spend over time -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class TokenTotals:
    """The sink's metered token counts, each field against its own denominator.

    `Tokens` is `observability/spend.py`'s, imported rather than reimplemented: the sidebar
    totals these two events over one conversation and this totals them over a file, and the
    rule that a field nothing reported is `None` rather than `0` is the same rule in both
    places.
    """

    calls: int
    floored: int
    input: Tokens
    output: Tokens

    @property
    def measured(self) -> bool:
        return self.input.measured or self.output.measured

    @property
    def calls_are_a_floor(self) -> bool:
        return self.floored > 0

    @property
    def partial(self) -> bool:
        """Whether a total shown here is not the whole story — `spend.Spend.partial`'s rule.

        Three ways, and the third is easy to miss: a field's own calls did not all report it,
        or one field was reported and the other not at all. Nothing measured is not partial; it
        is unmeasured, and the caller says that instead.
        """
        if not self.measured:
            return False
        return (
            self.input.partial
            or self.output.partial
            or not self.input.measured
            or not self.output.measured
        )

    def dollars(
        self, *, input_per_mtok: float | None, output_per_mtok: float | None
    ) -> float | None:
        """The cost at the configured prices, or `None` when it cannot be priced.

        Unpriced is the default and it is an absence, not `$0.00`: there is no rate card in
        this repo because every model is reached through OpenRouter's routing (ADR-0011 §3).
        """
        prices = (
            (self.input.total, input_per_mtok),
            (self.output.total, output_per_mtok),
        )
        priced = [
            tokens * price / 1_000_000
            for tokens, price in prices
            if tokens is not None and price is not None
        ]
        return sum(priced) if priced else None


def token_totals(log: EventLog, events: Iterable[Event] | None = None) -> TokenTotals:
    """Input and output totals over the token-bearing lines, with per-field denominators.

    `events` narrows the population to a subset already selected by the caller — the per-day
    buckets use it — and defaults to every `TOKEN_EVENTS` line in the log.
    """
    lines = list(events) if events is not None else list(log.of(*TOKEN_EVENTS))
    totals: dict[str, int | None] = dict.fromkeys(TOKEN_FIELDS)
    reported: dict[str, int] = dict.fromkeys(TOKEN_FIELDS, 0)
    calls = 0
    floored = 0
    for event in lines:
        made, is_floor = calls_behind(event)
        calls += made
        floored += is_floor
        for name in TOKEN_FIELDS:
            count = event.field(name)
            if count is None:
                continue
            totals[name] = count + (totals[name] or 0)
            reported[name] += event.field(f"{name}_calls", made)
    return TokenTotals(
        calls=calls,
        floored=floored,
        input=Tokens(totals["input_tokens"], reported["input_tokens"], calls),
        output=Tokens(totals["output_tokens"], reported["output_tokens"], calls),
    )


@dataclass(frozen=True, slots=True)
class SpendDay:
    """One day's metered tokens. Only days that metered something appear."""

    day: date
    input_tokens: int
    output_tokens: int
    calls: int


@dataclass(frozen=True, slots=True)
class SpendOverTime:
    """Token spend by day, plus the totals and the call the figures cannot see.

    **The gate's classifier is named unconditionally by the panel**, not as a clause of a
    partial banner. ADR-0011 §4's correction is exactly this: `partial` is defined over
    reported-versus-counted calls and the classifier never enters `calls`, so no value of
    `partial` is evidence about it. A complete total is still missing one paid call per turn.
    """

    by_day: tuple[SpendDay, ...]
    totals: TokenTotals

    @property
    def measured(self) -> bool:
        return self.totals.measured


def spend_over_time(log: EventLog) -> SpendOverTime:
    """Metered tokens per day over the whole sink, and the totals behind them."""
    buckets: dict[date, list[Event]] = {}
    for event in log.of(*TOKEN_EVENTS):
        buckets.setdefault(event.ts.date(), []).append(event)
    rows = []
    for day, lines in sorted(buckets.items()):
        totals = token_totals(log, lines)
        # A day nothing metered contributes no row rather than a row of zeros: the chart is of
        # spend, and a zero bar on it is a claim that a day's calls were free.
        if totals.measured:
            rows.append(
                SpendDay(
                    day=day,
                    input_tokens=totals.input.total or 0,
                    output_tokens=totals.output.total or 0,
                    calls=totals.calls,
                )
            )
    return SpendOverTime(by_day=tuple(rows), totals=token_totals(log))


# --- Panels 2 and 3: retrieval latency and the planner's round --------------------------


@dataclass(frozen=True, slots=True)
class Arm:
    """One `strategy` × `translation` configuration, and the retrievals that ran under it."""

    label: str
    strategy: str
    translation: bool
    latency: Distribution
    hits: Distribution


@dataclass(frozen=True, slots=True)
class RetrievalLatency:
    """Retrieval latency per configuration, over whatever configurations the sink holds.

    `unattributed` counts `retrieval` lines carrying no `strategy` or no `translation` — a line
    from before those fields, or from a caller that logged neither. Counted and named rather
    than dropped, because a silently smaller pool is a narrower denominator nobody chose.

    **This restates what `evaluation.md` reports, from live traffic rather than harness runs.**
    The artifact is the measurement of record; this is what the shipped path actually did in
    whatever sessions this file holds, which is a different and weaker claim.
    """

    arms: tuple[Arm, ...]
    overall: Distribution
    unattributed: int

    @property
    def measured(self) -> bool:
        return self.overall.measured


def retrieval_latency(log: EventLog) -> RetrievalLatency:
    """Retrieval latency split by `strategy` × `translation`, best-populated arm first."""
    lines = log.of("retrieval")
    grouped: dict[tuple[str, bool], list[Event]] = {}
    unattributed = 0
    for event in lines:
        strategy, translated = event.field("strategy"), event.field("translation")
        if not isinstance(strategy, str) or not isinstance(translated, bool):
            unattributed += 1
            continue
        grouped.setdefault((strategy, translated), []).append(event)
    arms = tuple(
        Arm(
            label=_arm_label(strategy, translated),
            strategy=strategy,
            translation=translated,
            latency=distribution(events, "latency_ms", label=_arm_label(strategy, translated)),
            hits=distribution(events, "hits", label="hits"),
        )
        for (strategy, translated), events in sorted(
            grouped.items(), key=lambda row: (-len(row[1]), row[0])
        )
    )
    return RetrievalLatency(
        arms=arms,
        overall=distribution(lines, "latency_ms", label="every retrieval"),
        unattributed=unattributed,
    )


def _arm_label(strategy: str, translated: bool) -> str:
    """`hybrid +translation` — the spelling `evaluation/arms.py` gives the same four cells."""
    return f"{strategy} {'+' if translated else '−'}translation"


@dataclass(frozen=True, slots=True)
class PlannerCost:
    """The planner's own chat round, against ADR-0005's budget.

    **One of the two terms in the clause, and the panel says which.** ADR-0005 judges "≤1.5s
    p50 *added by* translation", which is the planner's round **plus** the extra retrieval
    rounds the added variants cost — `evaluation/latency.TranslationCost` composes both and is
    the artifact of record, where the clause is reported as **missed**. This is term one alone.

    Two exclusions, both from `evaluation/latency.py` and both counted rather than dropped:
    `unmetered_lines` reported no `input_tokens`, so no model was called — a replayed harness
    arm or a cap of zero, and averaging them measures a stub. `disabled_lines` configured the
    planner off (`PLANNER_DISABLED_CAP`). A planner that **ran and refused** stays in the pool:
    it paid for a full chat round, and dropping refusals biases the p50 upward.
    """

    latency: Distribution
    unmetered_lines: int
    disabled_lines: int
    refusals_kept: int
    budget_ms: float

    @property
    def within_budget(self) -> bool | None:
        return self.latency.within(self.budget_ms)

    @property
    def measured(self) -> bool:
        return self.latency.measured


def planner_cost(
    log: EventLog, *, budget_ms: float = TRANSLATION_LATENCY_BUDGET_MS
) -> PlannerCost:
    """The planner's p50 over the rounds that reached a model, with both exclusions counted."""
    metered: list[Event] = []
    unmetered = 0
    disabled = 0
    refusals = 0
    for event in log.of("query_translation"):
        if event.field("max_sub_queries") == PLANNER_DISABLED_CAP:
            disabled += 1
            continue
        if event.field("input_tokens") is None:
            unmetered += 1
            continue
        metered.append(event)
        if event.field("sub_queries") == 0:
            refusals += 1
    return PlannerCost(
        latency=distribution(metered, "latency_ms", label="planner round"),
        unmetered_lines=unmetered,
        disabled_lines=disabled,
        refusals_kept=refusals,
        budget_ms=budget_ms,
    )
