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
    by_model,
    citations,
    gate_summary,
    open_sink,
    planner_cost,
    retrieval_latency,
    spend_over_time,
    tool_summary,
)
from finbrief.observability.spend import (
    UNMETERED_CLASSIFIER_NOTE,
    unpriced_because_of_the_model,
)

st.set_page_config(page_title="FinBrief · Analytics", page_icon=":material/analytics:")

st.title("Analytics")
# **The scope of every figure below, said once and here.** It was in three panels — the
# citation rate, the retrieval split and this line — because each of them wanted to disclaim a
# controlled measurement, and a caveat repeated three times is one a reader stops reading. It
# belongs to the page: nothing under it is a trial, and there is no panel this does not cover
# (#14 copy pass).
st.caption(
    "Every figure here is read back out of the structured event log — whatever has run against "
    "it, and not a controlled test. Nothing on this page is measured here."
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


def configured_prices() -> tuple[float | None, float | None, str | None]:
    """The two price knobs and the model they are for, or three `None`s when unreadable.

    Three states rather than two, because they call for three different sentences. A price is
    configuration and unset means unpriced (ADR-0011 §3 — there is no rate card in this repo,
    since every model is reached through OpenRouter's routing). But a `Settings` that cannot be
    built at all is a *different* absence, and telling a reader to set a price when the real
    problem is a missing key would send them to the wrong knob.

    The third element carried a `readable` boolean until T14 (#15); it carries the **priced
    model** now, and `None` still means "configuration could not be read" — the two facts have
    one source and one failure, so the model *is* the readability signal rather than a second
    thing to keep in step with it. `Spend.dollars` needs it because the two prices describe one
    model and a reader may pick another.

    Unlike `load_env` above this does not stop the page: the log is readable without an API key,
    and refusing to show yesterday's traffic because today's key is missing would be a page
    withholding what it has.
    """
    try:
        settings = get_settings()
    except ConfigError:
        return None, None, None
    return settings.input_cost_per_mtok, settings.output_cost_per_mtok, settings.chat_model


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
        # The consequence of switching it on is the reader's business; ADR-0011 is where the
        # decision behind it is recorded, and the citation stays in this comment.
        st.caption(
            "It is off by default because enabling it also keeps a blocked question's "
            "normalised text on disk."
        )
        return False
    if sink.state is SinkState.MISSING:
        st.warning(
            f"`FINBRIEF_LOG_FILE` names `{sink.path}`, and **nothing has written to it yet**. "
            f"The log is written by the app and by the scripts, not by this page — ask a "
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
        # Its own state rather than folded into the not-yet-written one, because that sentence
        # would send a reader to the wrong knob: nothing here is waiting on a question.
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
        # The malformed count is inside the sentence above rather than in a caption beneath it:
        # an empty file and a file of unreadable lines are different problems, and the number is
        # what tells them apart.
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
    # asked" would be a claim about a population neither of them describes — but the *reason*
    # was written as two abstract clauses ("not two views of one number"), which states the
    # conclusion and leaves the reader to reconstruct it. One sentence, both definitions, and
    # the mechanism in the middle (#14 copy pass).
    st.caption(
        "Screened counts questions typed into the app and answered counts turns the assistant "
        "finished, so the two need not match — evaluation runs answer questions without going "
        "through the app's input."
    )


# --- Panel 4: the gate ------------------------------------------------------------------


def render_gate(log) -> None:
    """ADR-0006's input gate: what it blocked, at which layer, and how fast."""
    gate = gate_summary(log)
    if not gate.screenings:
        absent("screenings")
    else:
        st.markdown(rate_line(gate.blocked))
        # **What the rate counts, because 0% otherwise reads as "nothing is guarding this".**
        # The gate (ADR-0006, layers 1–3) is about injection and nothing else; the
        # investment-advice refusal is a different control at a different point — `app/Home.py`
        # applies it to the answer — and a reader who takes a low block rate for an unguarded
        # app has been misled by an honest number.
        st.caption(
            "A block is an injection attempt — instruction override or prompt extraction. "
            "Questions asking for investment advice are not blocked here: they pass the gate "
            "and are refused when the answer is written."
        )
        st.markdown(rate_line(gate.classifier_ran))
        st.markdown(f"Gate latency: {figures(gate.latency)}")
        # **One target on screen, the earlier figure in the caption under it.** ADR-0006's T7
        # amendment revised the pre-registered 800 ms up to 1 s, and `security/report.py` prints
        # both as full verdicts so a reviewer cannot mistake a revised pre-registration for one
        # that always held. That fact is owed here too, but not as a second budget line: two
        # verdicts read as two live budgets, and the one that is met reads as the revision.
        st.markdown(budget_verdict(gate.latency, gate.budget_ms, name="Budget"))
        st.caption(
            f"Revised upward from an original `{gate.preregistered_budget_ms:,.0f}` ms "
            f"pre-registration."
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
    if gate.fail_open.measured:
        # Only where there is something to define. The ordinary log has none of these, and the
        # definition standing over "No fail-open events in this log" is a paragraph explaining
        # a term the reader has just been told did not occur.
        st.caption(
            "A fail-open is a control that did not run, not one that let something through: "
            "the classifier answers `undecided` when its provider fails, and the output "
            "check is skipped when it cannot run."
        )
    # **The one field on these lines this page will not render.** A blocked question's
    # normalised text is on its log line — ADR-0006's one bounded exception to no-user-content,
    # kept for an auditor with a grep — and a dashboard is a wider surface than that bound was
    # argued for. The omission used to be stated on screen; it is an auditor's concern and not a
    # reader's, so the record of it lives here (#14 copy pass).


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
    # (overwriting the model's `query`) breaks every follow-up. So a divergence is not
    # necessarily a defect: a follow-up resolving *its margins* into a company name is the one
    # rewrite the description permits. That is why the caption calls this behaviour rather than
    # faults; which ADR settled it is a developer's question and stays in this comment.
    st.caption(
        "Divergence is a search whose query differed from the question as typed — a "
        "description of behaviour, not a count of faults."
    )
    if behaviour.divergence_absent:
        # Named rather than folded in, because folding it in was the bug: a turn missing either
        # half of the pair reported every search as divergent, which is an absence rendered as
        # the worst possible measurement.
        st.caption(
            f"{behaviour.divergence_absent} of {behaviour.turns} turn(s) did not record "
            f"whether their searches matched the question, and are in no part of that rate — "
            f"absent, not divergent."
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
        # The three lines above name themselves in the vocabulary of the emitter, so the panel
        # owes a reader the translation: what a marker is, what it means for one to resolve, and
        # what the two failure counts look like in an answer.
        st.caption(
            "Answers cite their sources as `[1]`, `[2]`. Support is the share of those markers "
            "that point at a source listed with the answer, and clean turns are answers where "
            "every marker did. Unresolved markers point at nothing; non-numeric ones — "
            "`[Yahoo Finance]` — look like citations but name no source."
        )
        if cited.absent:
            # The rate this page exists for, so the gap in it is said out loud. A line missing
            # either half of the fraction used to read as 0% support — an absence rendered as
            # the worst measurement available, in the one figure the page was built to publish.
            st.caption(
                f"{cited.absent} of {cited.turns} record(s) carried no marker counts and are "
                f"in no part of that rate — absent, not unsupported."
            )
    # **Why this panel exists, and what its number is worth.** ADR-0011's amendment establishes
    # that `citation_markers` is written by `app/Home.py` and by nothing else, so T10's ten live
    # agent turns produced none and `docs/verification/evaluation.md` reports the rate as
    # unmeasured. This surface is where the instrument becomes readable — and the claim is
    # weaker than the one that artifact wanted.
    #
    # **That disclaimer is the page header's now**, not this panel's: it was here, in the
    # retrieval panel and in the header, and three tellings of one caveat is a caveat a reader
    # skips. Nothing on this page is a controlled measurement, so the sentence belongs to the
    # page (#14 copy pass).


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
    # the log and no ticker, by the rule that keeps user content out of these lines. The second
    # sentence this caption carried explained that logging rule, which is a developer's fact.
    st.caption(
        "These are the tickers the **finance tools** were called for, not the companies asked "
        "about."
    )
    # **The unit is on the figures and not also on the label** — it was on both.
    #
    # **Seconds, deliberately, and this is the record for it so that it is not "fixed" later.**
    # Everywhere else the app states an age in minutes: `prompts.stale_notice`, the sidebar's
    # freshness line, the scope disclosure's `QUOTE_TTL_SECONDS // 60`. Those describe *one*
    # quote's age, where whole minutes are what a reader wants, and the inconsistency here is
    # intentional rather than an oversight. This is a **distribution**, and its interesting
    # region is the first minute: a fresh fetch is `0` and a cache hit is anything up to the
    # 15-minute TTL, so p50 and p90 both live in the seconds. Rounded to minutes a 40-second age
    # prints as `1` and a 20-second one as `0` — a real measurement rendered as the absence this
    # whole page is built to distinguish from a zero. Raised in review and confirmed (#14).
    st.markdown(f"Age of the data served: {figures(tools.age_seconds, unit='s')}")
    st.caption("How old a quote or a headline was at the moment the answer used it.")
    st.markdown("**Refusals, failures and stale fallbacks**")
    # Tool **and** error type, the cross-tab the ticket asked for and the coarse tally is not:
    # "`get_recent_news` failed 40 times" and "…40 times on `HTTPError`" are different findings,
    # and only the second distinguishes a source that is down from a ticker that will not parse.
    # Both are drawn, because the coarser count is what a reader looks at first.
    failures = (
        (tools.refused, "validation refusals"),
        (tools.unavailable, "unavailable sources"),
        (tools.unavailable_by_error, "unavailable sources by error type"),
        (tools.stale_fallbacks, "stale fallbacks"),
    )
    if not any(counted.measured for counted, _ in failures):
        # **One sentence for four absences.** Four consecutive "Nothing is charted" lines say
        # the same thing four times over, and the reader who needs them named one by one is the
        # reader of a panel where some of the four *are* populated — which is the branch below.
        absent("refusals, failures or stale fallbacks")
    else:
        for counted, what in failures:
            tally_chart(counted, what=what)
    # **There is no cache hit rate here on purpose**, and the reason is a developer's: the age
    # of the served data is rounded to whole seconds where it is recorded, so a hit 400 ms
    # after a fetch is indistinguishable from a miss and a rate derived from it would be wrong
    # invisibly — the check-that-cannot-fail class CLAUDE.md names. The stale rate, the age
    # above and the fallback count are recorded explicitly, so they are what the panel reports.


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
                f"tokens — a provider that reported no usage did not make a free call."
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
            # A caption and not a `st.warning`: nothing here needs attention, the figures above
            # are simply a floor — and a yellow box beside a number reads as a fault in the
            # number rather than as a qualifier on it (#14 copy pass).
            st.caption(
                f"A floor: {totals.input.reported_calls} of {totals.calls} call(s) reported "
                f"input tokens and {totals.output.reported_calls} reported output."
            )
    # **Unconditional, and that is the whole point of where it sits.** ADR-0011 §4's correction:
    # the classifier never enters `calls`, so no value of `partial` is evidence about it. A
    # complete total is still missing one paid call per turn, and a caveat that only appeared
    # when something *else* was missing was a claim the surface did not make.
    #
    # **The string is `spend.py`'s, not this file's** — `app/Home.py`'s sidebar carried a second
    # copy of it, and two copies of one sentence disagree on the turn one of them is edited.
    # This is the surface that renders it: it totals the whole log (#14 copy pass).
    st.caption(UNMETERED_CLASSIFIER_NOTE)
    render_by_model(log)


def render_by_model(log) -> None:
    """Per-model tokens and turn latency — the panel the picker makes necessary (T14, #15).

    **Inside the spend panel rather than beside it**, because it is the breakdown *of* the
    totals above: a reader who has just been told the whole log cannot be priced needs the split
    in the same glance, not two panels away.

    Rendered as a table and not a chart. Four models over two measures is a comparison read
    row-by-row, and the two measures have different units — a bar chart of tokens beside
    milliseconds would need two axes to say less. It also keeps every absence printable as a
    word, which is the constraint a chart cannot meet (`Distribution` may be unmeasured, and a
    token field nothing reported is `None` and not `0`).

    **One column per number, and `figures()` is deliberately not used here** (manual testing
    of #15). That helper returns *markdown* for `st.markdown` — `p50 \\`7,816\\` ms · …` — and a
    `st.dataframe` cell renders no markdown, so it printed the backticks literally and put
    three statistics plus a sample count in the widest column on the page, cut off at its edge.
    Split into plain numeric columns the table becomes scannable and sortable, and Streamlit
    right-aligns and thousands-separates them for free.
    """
    slices = by_model(log)
    if not slices:
        absent("answered turn attributed to a model")
        return
    if len(slices) == 1 and slices[0].model is not None:
        # **One model is not a comparison, and a one-row table implies the others answered
        # nothing.** The figures are already above; what a reader gains here is the label.
        st.caption(f"Every answered turn in this log ran on `{slices[0].label}`.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Model": one.label,
                        "Turns": one.turns,
                        "Input": one.tokens.input.total,
                        "Output": one.tokens.output.total,
                        "p50 ms": _ms(one.latency.p50),
                        "p90 ms": _ms(one.latency.p90),
                        "max ms": _ms(one.latency.maximum),
                    }
                    for one in slices
                ]
            ),
            hide_index=True,
            width="stretch",
            height="content",
            # **`None` renders as an empty cell, which is the honest glyph here** and the reason
            # the token columns are numbers rather than `_tokens()` strings: a field nothing
            # reported has no value, and blank says that where `0` would be a claim that the
            # calls were free. The column config is what keeps the empties from reading as a
            # rendering fault — each header carries the unit, so a blank is visibly "not this".
            column_config={
                "Input": st.column_config.NumberColumn("Input", format="localized"),
                "Output": st.column_config.NumberColumn("Output", format="localized"),
            },
        )
        # **One caption where there were two**, because both answered the same question — *why
        # don't these rows add up to the totals above?* — and two stacked paragraphs of it was
        # the wall of text this panel was reported for (manual testing of #15).
        #
        # The claims are unchanged. The planner's tokens belong to no row (`query_translation`
        # carries no model, since the planner takes no override), so the rows sum to less than
        # the totals; and an unattributed row is not a fifth model but turns logged before the
        # field existed, which is most of an established sink. The second half renders only when
        # such a row is there, because a caveat about a row nobody can see is noise.
        #
        # **The `not recorded` clause says what those turns ran on, and this is the honest
        # version of a change asked for as a relabel.** Every one of them really did run on
        # whatever the default was at the time — here, the model the picker did not yet exist to
        # change — so folding them into that model's row is *nearly* right, and the
        # temptation is obvious. It is refused because the arithmetic reads this column:
        # `all_answered_on` would go true and `Spend.dollars` would print a figure over turns
        # nobody recorded a model for. Measured on the reported log: $0.1056, where the honest
        # answer is that it cannot be priced. So the fact goes in the caption, where it informs
        # a reader, and not in the cell, where it would feed a number.
        #
        # It names no slug: the page knows the default *now* and not the default *then*, and a
        # sink is append-only across every deploy that ever wrote to it.
        unattributed = any(one.model is None for one in slices)
        st.caption(
            "Answering calls only — the planner is logged separately, so rows total less than "
            "the figures above."
            + (
                " `not recorded` is turns from before the model was logged: they ran on"
                " whatever the default was then, which the log does not name."
                if unattributed
                else ""
            )
        )
    # **Outside the branch above, and that was a bug when it was inside it.** A reroute is news
    # whether or not the log holds more than one requested model — arguably *more* so on a
    # single-model log, since there is no table to notice it against. The early return
    # skipped it entirely and the test for it failed on that log (review of #15).
    #
    # Rendered **only when a provider disagreed** with the request, which is the whole reason
    # the reply's own model name is recorded beside the requested one: answers are routed, and
    # rows attributing tokens to what was *asked for* owe a reader the reroute that served them.
    # Silent otherwise, because "the provider agreed" and "the provider said nothing" are both
    # the ordinary case and a caption that always renders is one a reader skips.
    rerouted = [one for one in slices if one.rerouted_to]
    if rerouted:
        st.caption(
            "Served by a different model than requested — "
            + " · ".join(f"`{one.label}` → {', '.join(one.rerouted_to)}" for one in rerouted)
            + ". Rows count tokens against the model asked for."
        )


def render_cost(totals) -> None:
    """The priced total, or which of the three absences is in the way."""
    input_price, output_price, priced_model = configured_prices()
    if priced_model is None:
        st.caption(
            "Cost cannot be shown: this page could not read the app's configuration, so it "
            "does not know whether a price is set. The token counts above are unaffected."
        )
        return
    dollars = totals.dollars(
        input_per_mtok=input_price,
        output_per_mtok=output_price,
        priced_model=priced_model,
    )
    if dollars is None and not totals.all_answered_on(priced_model):
        # **The third absence, and on this page it is the likely one** (T14, #15). This total is
        # over the *whole sink*, which spans every session and every model anyone picked — and
        # every `agent_turn` written before T14 carries no model at all, so an established log
        # lands here rather than in either branch below.
        #
        # Ordered ahead of the unpriced branch for the reason `app/Home.py` orders it the same
        # way: it is the more specific claim, and "set the two variables" would be advice that
        # does not help. It names no per-model breakdown as a remedy, because none exists — the
        # per-model panel above splits *tokens*, which is what this repo can measure.
        #
        # The sentence is `spend.py`'s, and `scope` is this page's own noun for its pool: the
        # sidebar totals one conversation and this totals a file, so neither may borrow the
        # other's word for what it is describing (code review of #15).
        st.caption(unpriced_because_of_the_model(totals.models, priced_model, scope="this log"))
    elif dollars is None:
        # No price is assumed rather than guessed at: every model here is reached through
        # OpenRouter's routing, so there is no rate card in this repo to read one from
        # (ADR-0011 §3). What a reader needs is the two knobs, which is what the caption gives.
        st.caption(
            "Cost is not priced: set `FINBRIEF_INPUT_COST_PER_MTOK` and "
            "`FINBRIEF_OUTPUT_COST_PER_MTOK` from your provider's rate card."
        )
    else:
        # An estimate and labelled one: the tokens are measured, the two prices are read from
        # `.env`, and the product is only as good as the rate a reader typed in. The same label
        # as `app/Home.py`'s sidebar, for the same reason.
        st.markdown(f"**Cost (estimate)** `${dollars:.4f}`")


def _tokens(count: int | None) -> str:
    """`12,431`, or the word for a count nothing reported. Never `0` for an absence."""
    return "not reported" if count is None else f"{count:,}"


def _ms(value: float | None) -> int | None:
    """A latency in whole milliseconds for a table cell, or `None` for an absence.

    Whole milliseconds because a table of `7816.0` beside `12827.0` spends two characters a row
    on a decimal no reader of a p50 needs, and `None` rather than `0` for the usual reason —
    a distribution over no samples has no median, and a zero would be a claim that a turn was
    instant. Streamlit renders `None` as an empty cell.
    """
    return None if value is None else round(value)


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
        # Named rather than dropped: a smaller pool nobody chose is a narrower denominator.
        st.caption(
            f"{pools.unattributed} retrieval line(s) did not record which settings they ran "
            f"under, so they count towards the overall figure and towards no single setting."
        )
    # The controlled A/B over the golden set is `docs/verification/evaluation.md`'s, and that is
    # the measurement of record; this is the same split over whatever happened to run. Which
    # file holds the stronger claim, and why this one is weaker, is a developer's question —
    # and the reader's half of it, that these figures are not a trial, is the page header's
    # sentence rather than a third copy of one caveat (#14 copy pass).


def render_planner(log) -> None:
    """The planner's chat round against ADR-0005's budget — one of the clause's two terms."""
    planner = planner_cost(log)
    st.markdown(f"Planner round: {figures(planner.latency)}")
    st.markdown(budget_verdict(planner.latency, planner.budget_ms, name="Budget"))
    # **Which term of the budget this is, in one clause.** ADR-0005's clause is a sum — the
    # planner's round *plus* the extra retrieval rounds its variants cost — and
    # `evaluation/latency.TranslationCost` composes both, where
    # `docs/verification/evaluation.md` reports it as missed. A reader of this page needs to
    # know the timing above is not the whole cost of translating a question; the composition,
    # the artifact and the recorded verdict are a developer's business (#14 copy pass).
    st.caption("This is the planner's own round, not the full cost of translating a query.")
    # **Three counts, one line, and each clause only when it happened.** These were three
    # captions under two numbers, which is more explanation than measurement. They are still
    # three separate facts and none may be inferred from another: `unmetered_lines` called no
    # model at all, `disabled_lines` had the planner switched off by configuration, and
    # `refusals_kept` ran a full chat round that returned nothing — kept in the figures for that
    # reason, because dropping them biases the p50 upward. A clause carrying "and 0 had the
    # planner disabled" would be a count of nothing dressed as a finding (#14 copy pass).
    noted = []
    if planner.unmetered_lines:
        noted.append(f"{planner.unmetered_lines} question(s) needed no planning")
    if planner.disabled_lines:
        noted.append(f"the planner was switched off for {planner.disabled_lines}")
    if planner.refusals_kept:
        noted.append(
            f"{planner.refusals_kept} planning round(s) returned nothing but still cost a call "
            f"and are included"
        )
    if noted:
        st.caption(f"{'; '.join(noted)}.")


# --- The page ---------------------------------------------------------------------------

sink = open_sink(sink_path())
if render_header(sink):
    log = sink.readable
    # **Titled in a reader's words, not the codebase's** (#14 copy pass). "The input gate",
    # "retrieval latency" and "query translation" are the names of the modules behind these
    # panels; what a reader is looking for is what was screened, how long a search took and how
    # long rewriting a question took. The module names stay in the `render_*` docstrings.
    with st.expander(":material/timeline: Activity over time", expanded=True):
        render_activity(log)
    with st.expander(":material/security: Questions screened for attacks", expanded=True):
        render_gate(log)
    with st.expander(":material/smart_toy: What the assistant did", expanded=True):
        render_agent(log)
    with st.expander(":material/build: Market-data tools"):
        render_tools(log)
    with st.expander(":material/toll: Token spend"):
        render_spend(log)
    with st.expander(":material/search: How long searching the filings took"):
        render_retrieval(log)
    with st.expander(":material/alt_route: How long rewriting the question took"):
        render_planner(log)
