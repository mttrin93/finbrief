"""ADR-0005's latency budget, read from the log — and the refusals that keep it honest.

The clause is "dominance is judged within **≤1.5s p50 added by translation**, measured from
the Phase-6 structured logs". Three things about measuring it here rather than with a
stopwatch, all of which #11's non-negotiable 2 turns on:

**The sink is off unless named, so absence is the likely state.** `.env.example` ships
`FINBRIEF_LOG_FILE` commented out and `configure_logging()` takes no arguments, so a fresh
checkout's evaluation run records nothing. A p50 over zero samples that renders as "budget
met" is the exact failure this repo enforces against, so every function here raises rather
than returning a number it could not compute. `scripts/evaluate.py` also checks the switch
*before* spending anything, because discovering it afterwards costs the whole run.

**And the sink is *shared*, which is the second failure and the one the first artifact
shipped.** It is append-only across every run and every app session that ever named it, so a
median over the file is a median over all of them: the first committed artifact reported
1518 ms p50 over 60 samples from a pool holding 13 appended runs — including the pre-fix runs
whose ablation cells made real planner calls, which `HARNESS_VERSION` evicted from the *cache*
and could not touch in the *log*. Every reader here now takes a `start_offset` from
`events.sink_offset`, marked before the run, so the window is this run's.

**The planner's round and the retrieval rounds are separate lines, and only one of them is
what the clause is about.** `query_translation.latency_ms` is the chat round translation adds;
`retrieval.latency_ms` is the whole retrieval, timed from before the translate branch.
ADR-0011 calls the split the interesting half, and `events.Samples`' own docstring says the
budget takes two calls and not one.

**A replayed arm's planner latency is not the shipped path's**, and this is the subtlety the
harness introduces. ADR-0004 §9's replay means the scored `+translation` arms emit
`query_translation` lines with a stub in place of a model — near-zero, and meaningless as a
cost. So the planner's real cost is measured from the **resolve** pass, where the planner
actually runs, and `planner_added_ms` reads only lines that carry token counts: a line with no
`input_tokens` reported no spend because no model was called. That filter is the difference
between reporting translation's cost and reporting the stub's.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from finbrief.config import TRANSLATION_LATENCY_BUDGET_MS
from finbrief.observability.events import EventLog, read_events

#: A `translation: true` retrieval whose turn reported this `max_sub_queries` had the planner
#: **disabled by configuration** — the two ablation arms, whose only added variant is the
#: deterministic ticker form and which make no chat round at all (ADR-0004 §6: a cap of 0
#: removes the `model.invoke`, it does not truncate its output).
#:
#: Four of the six arms carry `translation: true` and only two of them plan, so averaging the
#: ablations in measures the ticker form's cost under a budget meant for the planner's round.
#:
#: **Keyed on the arm's configuration, not on the observed variant count**, and that correction
#: matters in a direction worth naming. The first version excluded any translated retrieval
#: returning fewer than three variants, which conflated "disabled by config" with "ran and
#: returned less": a planner that **refused** still paid for a full chat round and belongs in
#: the pool, and dropping refusals removes the cheap retrievals from a median of expensive
#: ones — biasing the p50 *upward* and making the budget miss look worse than it is.
PLANNER_DISABLED_CAP = 0


class NoSamples(RuntimeError):
    """A statistic was asked for over a log that carries none of the lines it needs."""


class SinkMissing(RuntimeError):
    """The sink was never enabled, so there is no log to read."""


def load_log(path: Path | str | None, *, start_offset: int = 0) -> EventLog:
    """This run's events at `path`, refusing both an unnamed sink and a missing file.

    `read_events` already raises on a missing file — "nobody enabled the log" read as "this
    run emitted nothing" is how a p50 over zero samples gets reported as a budget met
    (`observability/events.py`). This adds the `None` case, which is the one a fresh checkout
    hits.

    **`start_offset` is what makes these figures this run's**, and it is required rather than
    convenient: the sink is append-only across runs, so without a mark every median here is a
    median over every run that ever shared the file. See `events.sink_offset`.

    It also resolves the interaction between run-scoping and the cache, by making the honest
    answer the automatic one: a stage served entirely from cache issues no calls and therefore
    appends no lines, so a warm run's window is empty and `p50` **raises** rather than serving
    a previous run's number. A latency figure in the artifact now means the stage behind it
    actually ran.
    """
    if path is None:
        raise SinkMissing(
            "FINBRIEF_LOG_FILE was never set, so this run emitted no persisted events. "
            "ADR-0005's '<=1.5s p50 added by translation' cannot be measured from nothing, and "
            "reporting it as met would be an absence presented as a measurement."
        )
    return read_events(path, start_offset=start_offset)


#: Where a run records which slice of the sink is *its* window, beside the cells it paid for.
WINDOW_FILE = "log-window.json"


@dataclass(frozen=True, slots=True)
class Window:
    """The slice of a sink one measuring run wrote, and when.

    **Persisted for the same reason the cells are.** The artifact is regenerable from the
    cache, so the log window has to be regenerable too: a `--stage report` re-render replays
    every cell and therefore appends no lines, so without a recorded mark its own window is
    empty and the latency section refuses on a run whose numbers exist. Recording the mark makes
    a re-render describe *the measuring run that produced these cached cells* — which is what
    the artifact is about — and `replayed` is what makes the artifact say so rather than
    implying a fresh timing.
    """

    path: str
    offset: int
    recorded_at: str
    replayed: bool = False


def save_window(cache_root: Path | str, window: Window) -> Path:
    """Record this run's window beside its cached cells."""
    path = Path(cache_root) / WINDOW_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "path": window.path,
                "offset": window.offset,
                "recorded_at": window.recorded_at,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def load_window(cache_root: Path | str) -> Window | None:
    """The last measuring run's window, or `None` when there is not one recorded.

    `None` rather than a zero offset: offset 0 means "the whole file", which is the pooled-runs
    reading this module exists to stop. An absent mark is an absence.
    """
    path = Path(cache_root) / WINDOW_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    try:
        return Window(
            path=str(payload["path"]),
            offset=int(payload["offset"]),
            recorded_at=str(payload["recorded_at"]),
            replayed=True,
        )
    except (KeyError, TypeError, ValueError):
        return None


def p50(values: Sequence[float], *, what: str) -> float:
    """The median of `values`, or `NoSamples` naming what was missing.

    Never `0.0` for an empty sequence, and never `None` either: a caller that got a float
    knows it is a measurement, which is the whole point of raising here.
    """
    if not values:
        raise NoSamples(
            f"no samples for {what}: the log carries none of those lines. A median over zero "
            f"samples is not a budget met — enable FINBRIEF_LOG_FILE and re-run the stage that "
            f"emits them."
        )
    return float(statistics.median(values))


@dataclass(frozen=True, slots=True)
class TranslationCost:
    """What translation added, split the way ADR-0005's clause needs it.

    `planner_p50_ms` is the chat round; `retrieval_delta_p50_ms` is the difference between the
    `+translation` and `−translation` retrieval medians, which is the retrieval rounds the
    extra variants cost. The clause is judged on their sum, and both halves are printed so a
    reader can see which one spent the budget — ADR-0004's amendment predicts the normalised
    variant costs a retrieval round and never a chat round, and this is where that shows.
    """

    planner_p50_ms: float
    planner_samples: int
    retrieval_translated_p50_ms: float
    retrieval_untranslated_p50_ms: float
    translated_samples: int
    untranslated_samples: int
    #: `translation: true` retrievals excluded because their arm **disabled** the planner by
    #: configuration. Reported, never silently dropped.
    planner_disabled_lines: int
    #: Translated retrievals kept in the pool whose planner ran and returned no sub-query — a
    #: refusal, which cost a chat round. Reported because the first version's variant-count
    #: floor dropped exactly these, biasing the p50 upward.
    refusal_lines_kept: int
    budget_ms: float

    @property
    def retrieval_delta_p50_ms(self) -> float:
        return self.retrieval_translated_p50_ms - self.retrieval_untranslated_p50_ms

    @property
    def added_p50_ms(self) -> float:
        """Total p50 translation adds: the planner's round plus the extra retrieval rounds.

        The planner's round is *inside* `retrieval.latency_ms` on a live turn, but not on this
        harness's scored arms — they replay it through a stub. So adding the resolve pass's
        planner median to the retrieval delta is the honest reconstruction of what the shipped
        path pays, and it is stated as a reconstruction rather than printed as one
        measurement.
        """
        return self.planner_p50_ms + self.retrieval_delta_p50_ms

    @property
    def within_budget(self) -> bool:
        return self.added_p50_ms <= self.budget_ms


def _planner_caps(log: EventLog) -> dict[str, int]:
    """`turn_id` -> the `max_sub_queries` that turn's translation ran under.

    The join the exclusion needs: `max_sub_queries` is on the `query_translation` line and the
    latency is on the `retrieval` line, and the harness scopes each cell with
    `logging_setup.turn` so the two can be paired by identity rather than by position — which
    would be correct only for a serial harness and this one runs six cells at once.
    """
    caps: dict[str, int] = {}
    for event in log.of("query_translation"):
        cap = event.field("max_sub_queries")
        if event.turn_id is not None and cap is not None:
            caps[str(event.turn_id)] = int(cap)
    return caps


def translation_cost(
    log: EventLog, *, budget_ms: float = TRANSLATION_LATENCY_BUDGET_MS
) -> TranslationCost:
    """ADR-0005's added-latency p50, from the two kinds of line that carry it.

    Raises `NoSamples` if either half is missing, rather than substituting a zero for the half
    it could not measure — a budget met because one of its two terms was silently absent is
    not met.

    The budget comes from `config.TRANSLATION_LATENCY_BUDGET_MS` and not from a literal here:
    it is one of the two pre-registered latency figures CLAUDE.md's single-source list names,
    and it sat in this signature as an unbound `1500.0` while the gate's twin was bound by an
    equality.
    """
    planner = [
        event.field("latency_ms")
        for event in log.of("query_translation")
        # Only a line that reported spend, so the stub-replayed arms cannot flatter the number.
        # A `query_translation` line with no `input_tokens` called no model (see the module
        # docstring, and `query_translation.translate`'s own honest-absence comment).
        if event.field("input_tokens") is not None and event.field("latency_ms") is not None
    ]
    caps = _planner_caps(log)
    translated: list[float] = []
    untranslated: list[float] = []
    planner_disabled = 0
    refusals_kept = 0
    for event in log.of("retrieval"):
        latency = event.field("latency_ms")
        if latency is None:
            continue
        if not event.field("translation"):
            untranslated.append(latency)
            continue
        cap = caps.get(str(event.turn_id)) if event.turn_id is not None else None
        if cap == PLANNER_DISABLED_CAP:
            planner_disabled += 1
            continue
        translated.append(latency)
        # A planning arm that returned no sub-query refused, and a refusal cost a chat round.
        # Counted so the pool's composition is visible rather than assumed: these are the
        # turns the old variant-count floor dropped.
        if cap is not None and (event.field("variants") or 0) < 3:
            refusals_kept += 1
    return TranslationCost(
        planner_p50_ms=p50(planner, what="query_translation lines with token counts"),
        planner_samples=len(planner),
        retrieval_translated_p50_ms=p50(
            translated, what="retrieval lines with translation on and the planner enabled"
        ),
        retrieval_untranslated_p50_ms=p50(
            untranslated, what="retrieval lines with translation off"
        ),
        translated_samples=len(translated),
        untranslated_samples=len(untranslated),
        planner_disabled_lines=planner_disabled,
        refusal_lines_kept=refusals_kept,
        budget_ms=budget_ms,
    )


@dataclass(frozen=True, slots=True)
class TokenSpend:
    """What the run's metered calls reported, per event, with the unmetered half named.

    Absence is per field, not per record (`observability/tokens.py`): a provider that returned
    half a pair must not put a fabricated zero on the line, so each count carries its own call
    count and a mean is always over the calls that reported *that* field.
    """

    event: str
    input_tokens: int
    output_tokens: int
    input_calls: int
    output_calls: int
    lines: int

    @property
    def unmetered_lines(self) -> int:
        """Lines that reported no input count at all — a call whose cost is unknown, not "
        "free."""
        return self.lines - self.input_calls


def token_spend(log: EventLog, event: str) -> TokenSpend:
    """Token counts across `event`'s lines, counting each field's denominator separately."""
    lines = log.of(event)
    inputs = [e.field("input_tokens") for e in lines if e.field("input_tokens") is not None]
    outputs = [e.field("output_tokens") for e in lines if e.field("output_tokens") is not None]
    return TokenSpend(
        event=event,
        input_tokens=sum(inputs),
        output_tokens=sum(outputs),
        input_calls=len(inputs),
        output_calls=len(outputs),
        lines=len(lines),
    )
