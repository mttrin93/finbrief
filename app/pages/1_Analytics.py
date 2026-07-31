"""The analytics dashboard (T13, #14) — the event log T8 already writes, on a surface.

**It adds no instrument.** Every figure here is a `log_event` line an earlier ticket emits,
read through `observability/events.py` and aggregated by `observability/analytics.py`. Nothing
on this page writes a line either, and that is ADR-0011's amendment rather than an oversight:
an event wired to a page can be reached by a human clicking and by nothing else, so the
instrument belongs where the behaviour is produced and a page is only ever a reader.

**Five states, five sentences, no empty charts.** `FINBRIEF_LOG_FILE` unset, named but never
written, naming a path that will not open, written but holding no events, and readable are five
different things, and the first four each stop here with the sentence that fits. Below that each
panel keeps the same rule: a figure nothing measured says so where the number would have been.
That is the trap the sidebar's spend meter hit during #13 — a panel that vanishes, or renders
`0`, cannot be told apart from a broken feature.

**It touches nothing `app/Home.py` owns.** No `st.session_state` key it writes, and no call to
`shared_agent()` — the `@st.cache_resource` agent carries ADR-0008's two-session isolation
guarantee and rebuilding it here would discard every other session's memory. Streamlit runs only
the selected page's script, so this page is standalone: it resolves the environment itself
(`load_env`), and it reads the sink through the same `resolve_log_file(os.environ)` the handler
resolves it with.

**The pool is the whole file, and the header says so.** The sink is append-only across every run
and app session that ever named it, and no field distinguishes an app session from an evaluation
run (ADR-0011's T10 amendment, where a planner p50 over 13 appended runs was published as one
run's). The honest response is to name the pool rather than to guess at it: a heuristic on
`turn_id` shape would be a separation nothing could check.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from finbrief.config import ConfigError, get_settings, load_env, resolve_log_file
from finbrief.observability.analytics import (
    Distribution,
    Rate,
    Sink,
    SinkState,
    Tally,
    activity,
    agent_behaviour,
    citations,
    gate_summary,
    open_sink,
    planner_cost,
    retrieval_latency,
    spend_over_time,
    tool_summary,
)

st.set_page_config(page_title="FinBrief · Analytics", page_icon=":material/analytics:")

st.title("Analytics")
st.caption(
    "Every figure here is read back out of the structured event log — the same lines the "
    "evaluation harness reads. Nothing on this page is measured here."
)

# `load_env()` before anything asks the environment for the sink: Streamlit runs only the
# selected page's script, so `app/Home.py`'s own call has not necessarily happened. A broken
# `.env` is a configuration problem and stops the page, because without it the sink cannot even
# be located — the same banner-then-stop shape `Home.py` uses for the same reason.
try:
    load_env()
except ConfigError as exc:
    st.error(f"Configuration problem: {exc}", icon=":material/error:")
    st.stop()


def sink_path():
    """Where the event log is being written, or `None` when nobody named one.

    `resolve_log_file(os.environ)`, which is where `configure_logging()` resolves it from —
    asking the same question of the same source is what stops this page claiming a sink the
    handler did not install (the rule `app/Home.py`'s own `sink_path` states).
    """
    return resolve_log_file(os.environ)


def configured_prices() -> tuple[float | None, float | None, bool]:
    """The two price knobs, and whether configuration could be read at all.

    Three states rather than two, because they call for three different sentences. A price is
    configuration and unset means unpriced (ADR-0011 §3 — there is no rate card in this repo,
    since every model is reached through OpenRouter's routing). But a `Settings` that cannot be
    built at all is a *different* absence, and telling a reader to set a price when the real
    problem is a missing key would send them to the wrong knob.

    Unlike `load_env` above this does not stop the page: the log is readable without an API key,
    and refusing to show yesterday's traffic because today's key is missing would be a page
    withholding what it has.
    """
    try:
        settings = get_settings()
    except ConfigError:
        return None, None, False
    return settings.input_cost_per_mtok, settings.output_cost_per_mtok, True


# --- Rendering helpers ------------------------------------------------------------------


def figures(spread: Distribution, *, unit: str = "ms") -> str:
    """`p50 950 ms · p90 1100 ms · max 1300 ms (3 samples)`, or the absence.

    The sample count travels with the figures rather than beside them, because a p50 over two
    observations and a p50 over two hundred are different claims and the number is the only
    thing that says which.
    """
    if not spread.measured:
        return "**not measured** — no line here carried one"
    return (
        f"p50 `{spread.p50:,.0f}` {unit} · p90 `{spread.p90:,.0f}` {unit} · "
        f"max `{spread.maximum:,.0f}` {unit} ({spread.count} sample(s))"
    )


def budget_verdict(spread: Distribution, budget: float, *, name: str) -> str:
    """Where the median sits against `budget` — and "not measured" as a third answer.

    Never "missed" for an absence: `Distribution.within` returns `None` rather than `False`
    precisely so that this sentence can distinguish them.
    """
    verdict = spread.within(budget)
    if verdict is None:
        return f"{name} `{budget:,.0f}` ms — **not measured**, so not met either"
    return f"{name} `{budget:,.0f}` ms — **{'met' if verdict else 'missed'}**"


def rate_line(rate: Rate) -> str:
    """A `Rate`'s own rendering, so a percentage never appears without its denominator."""
    return rate.render()


def absent(what: str) -> None:
    """The sentence a panel prints instead of a chart of nothing."""
    st.caption(
        f"No {what} in this log. Nothing is charted, because zero bars would be a claim."
    )


def tally_chart(counted: Tally, *, what: str, height: int = 220) -> None:
    """One bar per value, or the absence sentence — and the lines that carried nothing.

    `absent` is printed rather than folded into the chart: a line that did not carry the field
    is not a line with a count of zero, and a bar chart has no way to draw the difference.
    """
    if not counted.measured:
        absent(what)
        return
    st.bar_chart(
        pd.DataFrame(
            {counted.label: list(counted.counts.values())}, index=list(counted.counts)
        ),
        height=height,
    )
    if counted.absent:
        st.caption(
            f"{counted.absent} of {counted.lines} line(s) carried no {counted.label} and are "
            f"not in the chart — absent, not zero."
        )


# --- The header: which of the five states this sink is in --------------------------------


def render_header(sink: Sink) -> bool:
    """What is being read, and whether there is anything to read. `False` stops the page.

    The four absent states are the reason this returns a decision rather than just rendering:
    each of them is a complete answer, and a panel drawn underneath one of them would be a
    figure about a file that holds nothing.
    """
    if sink.state is SinkState.OFF:
        st.info(
            "The event log is **off**, so there is nothing to chart. Set `FINBRIEF_LOG_FILE` "
            "to a path — `.env.example` has the recommended one, commented out — and this page "
            "fills in as you use FinBrief.",
            icon=":material/toggle_off:",
        )
        st.caption(
            "Off by default on purpose (ADR-0011): enabling the sink also means keeping a "
            "blocked question's normalised text on disk, so it has to be a decision."
        )
        return False
    if sink.state is SinkState.MISSING:
        st.warning(
            f"`FINBRIEF_LOG_FILE` names `{sink.path}`, and **nothing has written to it yet**. "
            f"The sink is opened by the app and by the scripts, not by this page — ask a "
            f"question on the FinBrief page and come back.",
            icon=":material/hourglass_empty:",
        )
        return False
    if sink.state is SinkState.UNREADABLE:
        # **The fifth sentence, and the reason there are five.** A directory where a file was
        # meant, a file this process cannot open, and a file whose bytes are not UTF-8 all
        # reached `read_events` in the first version and arrived here as a traceback — which on
        # a page built around telling an absence from a zero is the one rendering that tells a
        # reader nothing at all (code review of #14).
        st.error(
            f"`FINBRIEF_LOG_FILE` names `{sink.path}`, and it **cannot be read** "
            f"(`{sink.reason}`). A path that exists and will not open is usually a directory "
            f"where a file was meant, a permission the app does not have, or a file written by "
            f"something other than this logger.",
            icon=":material/error:",
        )
        st.caption(
            "Its own state rather than folded into the not-yet-written one, because that "
            "sentence would send you to the wrong knob: nothing here is waiting on a question "
            "being asked."
        )
        return False
    if sink.state is SinkState.EMPTY:
        st.warning(
            f"`{sink.path}` exists and holds **no events**"
            + (
                f" — {sink.malformed} line(s) in it could not be parsed as one."
                if sink.malformed
                else " and no lines at all."
            ),
            icon=":material/description:",
        )
        st.caption(
            "An empty file and a file of unreadable lines are different problems, which is why "
            "the count is here rather than folded into one sentence."
        )
        return False

    st.markdown(f"Reading `{sink.path}`")
    columns = st.columns(4)
    columns[0].metric("Events", f"{sink.events:,}")
    # **Not derivable from the two beside it**, which is why it is a metric of its own: blank
    # lines are skipped by the reader and counted in neither, so a reader adding events to
    # malformed would get a lower bound on what was read (code review of #14).
    columns[1].metric("Lines parsed", f"{sink.lines:,}")
    columns[2].metric("Size", f"{sink.size_bytes / 1024:,.0f} KiB")
    columns[3].metric("Malformed lines", f"{sink.malformed:,}")
    if sink.first_event is not None and sink.last_event is not None:
        st.caption(
            # **The pool, on the line that already names it.** The caption under the title
            # already says these are the harness's own lines; what it does not say is the
            # scope, and that is the half worth keeping — ADR-0011's T10 amendment records what
            # forgetting it costs: a planner p50 over 13 appended runs published under a header
            # claiming the figures were one run's. No field distinguishes an app session from an
            # evaluation run, so the page states the pool rather than guessing at it.
            f"Spanning {sink.first_event:%Y-%m-%d %H:%M} to "
            f"{sink.last_event:%Y-%m-%d %H:%M} UTC — the whole file, not one session or run."
        )
    return True


# --- Panel 1: activity over time --------------------------------------------------------


def render_activity(log) -> None:
    """Questions screened, turns answered, turns failed — by day."""
    seen = activity(log)
    if not seen.measured:
        absent("questions, turns or failures")
        return
    st.bar_chart(
        pd.DataFrame(
            {
                "screened": [row.screened for row in seen.by_day],
                "answered": [row.answered for row in seen.by_day],
                "failed": [row.failed for row in seen.by_day],
            },
            index=[row.day for row in seen.by_day],
        ),
        height=260,
    )
    st.markdown(
        f"**{seen.screened:,}** question(s) screened · **{seen.answered:,}** turn(s) answered "
        f"· **{seen.failed:,}** failed"
    )
    # **The two counts legitimately disagree, and saying why is the panel's job.** `input_gate`
    # is the app's front door and nothing else screens; an evaluation run drives `agent.answer`
    # and writes turns with no gate line beside them. Presenting either number as "questions
    # asked" would be a claim about a population neither of them describes.
    st.caption(
        "Screenings come from the app's chat input only, so an evaluation run contributes "
        "answered turns with no screening beside them. The two counts are not two views of one "
        "number."
    )


# --- Panel 4: the gate ------------------------------------------------------------------


def render_gate(log) -> None:
    """ADR-0006's input gate: what it blocked, at which layer, and how fast."""
    gate = gate_summary(log)
    if not gate.screenings:
        absent("screenings")
    else:
        st.markdown(rate_line(gate.blocked))
        st.markdown(rate_line(gate.classifier_ran))
        st.markdown(f"Gate latency: {figures(gate.latency)}")
        # **Both budgets, because one of them was revised.** ADR-0006's T7 amendment moved the
        # target from 800 ms to 1 s, and `security/report.py` prints both for the reason the
        # constant survives at all: a revised pre-registration shown alone reads as one that
        # always held.
        st.markdown(
            budget_verdict(gate.latency, gate.budget_ms, name="Budget")
            + "  \n"
            + budget_verdict(
                gate.latency, gate.preregistered_budget_ms, name="Pre-registered (revised)"
            )
        )
        st.markdown("**Blocks by layer**")
        tally_chart(gate.by_layer, what="blocked screenings")
        st.markdown("**Blocks by rule**")
        tally_chart(gate.by_rule, what="denylist rule hits")
        # Heading and absence sentence unconditionally, like the two tallies above it. The
        # guard this replaces removed the heading too, so a log where layer 3 never ran looked
        # like a page missing a panel rather than one reporting that the layer never ran.
        st.markdown("**Classifier verdicts**")
        tally_chart(gate.verdicts, what="classifier verdicts")

    st.markdown("**Layers that failed open**")
    tally_chart(gate.fail_open, what="fail-open events")
    st.caption(
        "A fail-open is a control that did not run, not one that let something through: the "
        "classifier returns `undecided` on a provider failure, and the validator is skipped "
        "when its Guard raises. Counted separately from the verdicts for that reason."
    )
    # The one field on these lines this page will not render, said out loud so that its absence
    # reads as a decision rather than as an oversight.
    st.caption(
        "A blocked question's normalised text is on its log line, bounded three ways by "
        "ADR-0006, and is deliberately not shown here: it is kept for an audit with a grep, "
        "and a dashboard is a wider surface than that bound was argued for."
    )


# --- Panel 6: the agent's behaviour, and the bracket rate -------------------------------


def render_agent(log) -> None:
    """What the shipped path did: searches, divergence, grounding, tool choice."""
    behaviour = agent_behaviour(log)
    if not behaviour.measured:
        absent("agent turns")
        # **Still the citations**, and not an early return, which is what this was. The two read
        # different events: `citation_markers` is written by the app and `agent_turn` by the
        # agent, so a sink can hold one without the other — and returning here would hide the
        # one figure this page exists to make readable behind the absence of a different one.
        render_citations(log)
        return
    st.markdown(
        f"**{behaviour.turns:,}** turn(s) · **{behaviour.searches:,}** search(es)  \n"
        f"{rate_line(behaviour.divergence)}  \n"
        f"{rate_line(behaviour.grounded)}  \n"
        f"{rate_line(behaviour.searched)}"
    )
    # ADR-0003 §1 asks the model to pass the question verbatim and §2 of its T4 amendment
    # records that the rule is *measured, not enforced* — the alternative that would enforce it
    # (overwriting the model's `query`) breaks every follow-up. So a divergence here is not
    # necessarily a defect: a resolved pronoun is the one rewrite the description permits.
    st.caption(
        "Divergence is a search whose query differed from the question as typed. It is "
        "measured rather than enforced (ADR-0003), and a follow-up that resolves *its "
        "margins* into a "
        "company name is a permitted rewrite — so this is a description of behaviour, not a "
        "count of faults."
    )
    if behaviour.divergence_absent:
        # Named rather than folded in, because folding it in was the bug: a turn missing either
        # half of the pair reported every search as divergent, which is an absence rendered as
        # the worst possible measurement.
        st.caption(
            f"{behaviour.divergence_absent} of {behaviour.turns} turn(s) carried no verbatim "
            f"count and are in no part of that rate — absent, not divergent."
        )
    st.markdown(f"Turn latency: {figures(behaviour.turn_latency)}")
    # **Both were aggregated and rendered nowhere** (code review of #14), and "searches per
    # turn" is one of the figures the ticket names — a p50 of 1 over turns that mostly search
    # once is a different picture from a mean, which is why the whole `Distribution` prints.
    st.markdown(
        f"Searches per turn: {figures(behaviour.searches_per_turn, unit='search(es)')}  \n"
        f"Finance calls per turn: {figures(behaviour.finance_calls, unit='call(s)')}"
    )
    st.markdown("**Tools chosen**")
    tally_chart(behaviour.tools_used, what="tool selections")
    render_citations(log)


def render_citations(log) -> None:
    """The square-bracket adherence rate — the deferral this page is the surface for."""
    cited = citations(log)
    st.markdown("**Citation markers**")
    if not cited.measured:
        absent("citation-marker records")
    else:
        st.markdown(
            f"{rate_line(cited.support)}  \n"
            f"{rate_line(cited.clean)}  \n"
            f"**{cited.unresolved:,}** unresolved · **{cited.non_numeric:,}** non-numeric "
            f"marker(s)"
        )
        if cited.absent:
            # The rate this page exists for, so the gap in it is said out loud. A line missing
            # either half of the fraction used to read as 0% support — an absence rendered as
            # the worst measurement available, in the one figure the page was built to publish.
            st.caption(
                f"{cited.absent} of {cited.turns} record(s) carried no marker counts and are "
                f"in no part of the support rate — absent, not unsupported."
            )
    # **Why this panel exists, and what its number is worth.** ADR-0011's amendment establishes
    # that `citation_markers` is written by `app/Home.py` and by nothing else, so T10's ten live
    # agent turns produced none and `docs/verification/evaluation.md` reports the rate as
    # unmeasured. This surface is where the instrument becomes readable — and the claim is
    # weaker than the one that artifact wanted, which is worth stating where the number is.
    st.caption(
        "This is the T5 deferral `docs/verification/evaluation.md` reports as unmeasured: the "
        "instrument only fires on an app turn, so the harness never produced a line. What this "
        "shows is **observational over whatever sessions this log holds** — not the controlled "
        "measurement over a stratified set that the artifact asked for."
    )


# --- Panel 5: tools ---------------------------------------------------------------------


def render_tools(log) -> None:
    """Which finance tools ran, for which tickers, and how they failed."""
    tools = tool_summary(log)
    if not tools.measured:
        absent("tool calls, refusals or failures")
        return
    st.markdown(f"**{tools.calls:,}** successful call(s)  \n{rate_line(tools.stale)}")
    st.markdown("**Calls by tool**")
    tally_chart(tools.by_tool, what="tool calls")
    st.markdown("**Calls by ticker**")
    tally_chart(tools.by_ticker, what="tool calls with a ticker")
    # The narrower claim, because it is the true one: a filings-only question puts a length in
    # the log and no ticker, by the rule that keeps user content out of these lines.
    st.caption(
        "These are the tickers the **finance tools** were called for, not the companies asked "
        "about. A question answered from the filings alone names no ticker in the log — the "
        "lines carry counts and lengths, never the question."
    )
    # The explicit field, printed rather than left as the input to a rate that cannot be
    # computed — see the caption below. It was aggregated and rendered nowhere.
    st.markdown(f"Age of the data served: {figures(tools.age_seconds, unit='s')}")
    st.markdown("**Refusals, failures and stale fallbacks**")
    tally_chart(tools.refused, what="validation refusals")
    tally_chart(tools.unavailable, what="unavailable sources")
    # Tool **and** error type, the cross-tab the ticket asked for and the tally above is not:
    # "`get_recent_news` failed 40 times" and "…40 times on `HTTPError`" are different findings,
    # and only the second distinguishes a source that is down from a ticker that will not parse.
    # Both are drawn, because the coarser count is what a reader looks at first.
    tally_chart(tools.unavailable_by_error, what="unavailable sources by error type")
    tally_chart(tools.stale_fallbacks, what="stale fallbacks")
    st.caption(
        "There is no cache hit rate here on purpose: `age_seconds` is rounded to whole seconds "
        "at the emitter, so a hit 400 ms after a fetch is indistinguishable from a miss. The "
        "stale rate, the age above and the fallback count are explicit fields, so they are "
        "what is reported."
    )


# --- Panel 7: token spend over time -----------------------------------------------------


def render_spend(log) -> None:
    """Metered tokens by day, with each field's own denominator and the call nobody meters."""
    over_time = spend_over_time(log)
    totals = over_time.totals
    # **The floor rendering is above the branch, because the branch below is where the floor is
    # guaranteed.** `agent_turn` writes `calls` only once something reported usage, so a log
    # with nothing metered is exactly a log whose every call arrived as `calls_behind`'s honest
    # floor of one — and the unmeasured sentence printed it bare while the measured one applied
    # the `≥`. `Spend.floored` exists so a floor is never displayed as a count (#13's review);
    # hoisting it is what makes that true on both paths (code review of #14).
    calls = f"≥{totals.calls}" if totals.calls_are_a_floor else f"{totals.calls}"
    if not over_time.measured:
        # Two states, two sentences. "No line carried a token count" and "there were no
        # token-bearing lines at all" are different things, and one number rendered inside one
        # sentence let a reader take either for the other.
        if totals.calls:
            st.caption(
                f"No usage reported in this log, across `{calls}` model call(s). Not zero "
                f"tokens — a provider that reports no usage block did not make a free call."
            )
        else:
            st.caption(
                "No model call is recorded in this log at all, so there is nothing to total. "
                "That is an absence of lines rather than a spend of zero."
            )
    else:
        st.bar_chart(
            pd.DataFrame(
                {
                    "input": [row.input_tokens for row in over_time.by_day],
                    "output": [row.output_tokens for row in over_time.by_day],
                },
                index=[row.day for row in over_time.by_day],
            ),
            height=260,
        )
        st.markdown(
            f"**Input** `{_tokens(totals.input.total)}`  \n"
            f"**Output** `{_tokens(totals.output.total)}`  \n"
            f"**Calls** `{calls}`"
        )
        render_cost(totals)
        if totals.partial:
            st.warning(
                f"Partial: {totals.input.reported_calls} of {totals.calls} call(s) reported "
                f"input tokens and {totals.output.reported_calls} reported output tokens, so "
                f"the figures above are a floor.",
                icon=":material/data_alert:",
            )
    # **Unconditional, and that is the whole point of where it sits.** ADR-0011 §4's correction:
    # the classifier never enters `calls`, so no value of `partial` is evidence about it. A
    # complete total is still missing one paid call per turn, and a caveat that only appeared
    # when something *else* was missing was a claim the surface did not make.
    st.caption(
        "The input gate's classifier is never metered, so one paid call per turn is missing "
        "from these figures by design."
    )


def render_cost(totals) -> None:
    """The priced total, or which of the two absences is in the way."""
    input_price, output_price, readable = configured_prices()
    if not readable:
        st.caption(
            "Cost cannot be shown: this page could not read the app's configuration, so it "
            "does not know whether a price is set. The token counts above are unaffected."
        )
        return
    dollars = totals.dollars(input_per_mtok=input_price, output_per_mtok=output_price)
    if dollars is None:
        st.caption(
            "Cost is not priced: set `FINBRIEF_INPUT_COST_PER_MTOK` and "
            "`FINBRIEF_OUTPUT_COST_PER_MTOK` from your provider's rate card. No price is "
            "assumed, because this app reaches every model through OpenRouter's routing."
        )
    else:
        st.markdown(f"**Cost** `${dollars:.4f}`")


def _tokens(count: int | None) -> str:
    """`12,431`, or the word for a count nothing reported. Never `0` for an absence."""
    return "not reported" if count is None else f"{count:,}"


# --- Panels 2 and 3: retrieval latency and the planner's round --------------------------


def render_retrieval(log) -> None:
    """Retrieval latency per `strategy` × `translation`, over live and harness traffic alike."""
    pools = retrieval_latency(log)
    if not pools.measured:
        absent("timed retrievals")
    else:
        st.markdown(f"Every retrieval: {figures(pools.overall)}")
        for arm in pools.arms:
            st.markdown(f"`{arm.label}` — {figures(arm.latency)}")
    # **Outside the branch**, and that is the same correction the citation panel needed: this
    # counts `retrieval` lines with no configuration on them, and gating it on whether any
    # `latency_ms` was reported hides one absence behind another. A log of untimed retrievals is
    # exactly where a reader wants to know how many were unattributable.
    if pools.unattributed:
        st.caption(
            f"{pools.unattributed} retrieval line(s) carried no strategy or no translation "
            f"flag and are in the overall figure but in no arm — named rather than dropped, "
            f"because "
            f"a smaller pool nobody chose is a narrower denominator."
        )
    st.caption(
        "The A/B measurement of record is `docs/verification/evaluation.md`, over the golden "
        "set. This is the same split over **live traffic** — whatever ran against this sink — "
        "which is a weaker claim about a different population."
    )


def render_planner(log) -> None:
    """The planner's chat round against ADR-0005's budget — one of the clause's two terms."""
    planner = planner_cost(log)
    st.markdown(f"Planner round: {figures(planner.latency)}")
    st.markdown(budget_verdict(planner.latency, planner.budget_ms, name="Budget"))
    # **The panel says which term this is**, because the clause is a sum and half of it is a
    # different measurement. `evaluation/latency.TranslationCost` composes both and reports the
    # clause as missed; a page showing the first term under the budget's own name would let a
    # reader take a met budget from a number that is not what the budget is about.
    st.caption(
        "ADR-0005's clause is *≤1.5 s p50 added by translation*, which is this chat round "
        "**plus** the extra retrieval rounds the added variants cost. This is the first term "
        "alone; the composed figure is in `docs/verification/evaluation.md`, where the clause "
        "is recorded as missed."
    )
    if planner.unmetered_lines or planner.disabled_lines:
        st.caption(
            f"Excluded and counted: {planner.unmetered_lines} line(s) reported no tokens, so "
            f"no model was called, and {planner.disabled_lines} had the planner disabled by "
            f"configuration."
        )
    # **Its own sentence, because it is its own measurement.** This was a clause of the
    # exclusion
    # caption, so a log with nothing excluded — the ordinary case — never said how many planners
    # ran and refused, which is a real fact about planner behaviour suppressed by the absence of
    # two unrelated counts (code review of #14).
    if planner.refusals_kept:
        st.caption(
            f"{planner.refusals_kept} planner round(s) ran and returned no sub-query, and are "
            f"**kept** in the figures above — a refusal still paid for a full chat round, and "
            f"dropping them would bias the p50 upward."
        )


# --- The page ---------------------------------------------------------------------------

sink = open_sink(sink_path())
if render_header(sink):
    log = sink.readable
    with st.expander(":material/timeline: Activity over time", expanded=True):
        render_activity(log)
    with st.expander(":material/security: The input gate", expanded=True):
        render_gate(log)
    with st.expander(":material/smart_toy: The agent's behaviour", expanded=True):
        render_agent(log)
    with st.expander(":material/build: Finance tools"):
        render_tools(log)
    with st.expander(":material/toll: Token spend"):
        render_spend(log)
    with st.expander(":material/search: Retrieval latency"):
        render_retrieval(log)
    with st.expander(":material/alt_route: Query translation"):
        render_planner(log)
