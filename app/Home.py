"""FinBrief chat UI — the agent, its memory, and the sources behind each answer (T4, #7).

UI only: every non-Streamlit concern lives in `finbrief.*` so it can be tested without
driving the app, and every word about the knowledge base's shape comes from
`finbrief.prompts` — the same text the model is given, so the page and the persona cannot
disagree about what is grounded (user story 18, ADR-0007).

**State ownership (ADR-0008), which this file is the enforcement point for.** Streamlit
re-runs the whole script on every interaction, so there are two candidate sources of truth
for a conversation and only one of them works:

- The **checkpointer is the agent's memory of record.** `answer()` is handed the new question
  and a `thread_id`, never a history — a history assembled here would be a second copy of the
  agent's memory, diverging on the first rerun this script does not replay.
- **`st.session_state` holds only** the `thread_id` and the display transcript: each
  assistant turn's text, the contexts its `[n]` markers point at, and whether it searched at
  all. Display data, not memory.
- **The agent and its checkpointer are built once**, under `@st.cache_resource`. Per-run
  construction would mean a per-run checkpointer, which remembers nothing.
- That cached instance is **shared by every session on the server**, so the `thread_id` is
  what keeps two users apart: one `uuid4` minted per session, stable across reruns. This is
  the privacy property `tests/test_app_state.py` exists to hold down.

Refreshing the browser therefore starts a new conversation — `session_state` resets, a new
uuid is minted, and the old thread is orphaned in the checkpoint file. That is a stated
consequence of the design, not a defect, so the sidebar says so rather than leaving a user to
discover it when their context disappears.
"""

import logging
import os
import uuid

import altair as alt
import pandas as pd
import streamlit as st
from langgraph.errors import GraphRecursionError

from finbrief.agent.agent import Search, Step, answer, build_agent
from finbrief.config import (
    CLUSTERS,
    HISTORY_PERIOD_LABEL,
    MAX_QUESTION_CHARS,
    MAX_QUESTIONS_PER_SESSION,
    PEERS,
    UNIVERSE,
    ConfigError,
    RetrievalStrategy,
    chat_model_options,
    get_settings,
    resolve_log_file,
    thinnest_cluster_filer,
)
from finbrief.export import (
    THREAD_HANDLE_CHARS,
    Transcript,
    as_csv,
    as_json,
    file_name,
    log_export,
)
from finbrief.finance.ratios import Metric, Unit
from finbrief.ingestion.edgar import filing_index_url
from finbrief.llm import build_chat_model
from finbrief.observability.events import read_events, sink_offset
from finbrief.observability.logging_setup import configure_logging, log_event

# Aliased, and the alias is the point: this module already binds `turn` at module scope — the
# replay loop's `if (turn := message.get("turn"))` walrus, holding an `AgentTurn`. Importing the
# context manager under its own name shadowed it from the *second* rerun onwards and the page
# died with `'AgentTurn' object is not callable`, which `test_app_state` caught and a reader
# would not have.
from finbrief.observability.logging_setup import turn as log_turn
from finbrief.observability.spend import (
    Spend,
    conversation_spend,
    unpriced_because_of_the_model,
)
from finbrief.prompts import (
    ADVICE_REFUSAL,
    DISCLAIMER,
    EXAMPLE_QUESTIONS,
    GROUNDING_SCOPE_DETAILS,
    GROUNDING_SCOPE_SOURCED,
    GROUNDING_SCOPE_VERIFY,
    INJECTION_REFUSAL,
    LIVE_DATA_SCOPE,
    UNIVERSE_ROWS,
)
from finbrief.retrieval.hybrid import Retriever
from finbrief.retrieval.retrieve import Context
from finbrief.security.advice import validate_answer
from finbrief.security.input_gate import screen
from finbrief.security.markers import MarkerReport, log_markers, markers
from finbrief.tools.finance import (
    NEWS_TOOL_NAME,
    RATIOS_TOOL_NAME,
    STOCK_TOOL_NAME,
    FailedCard,
    FinanceCard,
    NewsCard,
    QuoteCard,
    RatiosCard,
)
from finbrief.tools.search_filings import TOOL_NAME as SEARCH_TOOL_NAME

#: Under `finbrief.` so the JSON-lines handler `configure_logging` installs picks it up — a
#: logger named for this module would propagate to root and print unstructured.
logger = logging.getLogger("finbrief.app")

st.set_page_config(page_title="FinBrief", page_icon=":material/query_stats:")

st.title("FinBrief")
# The scope sentence with the filings' source linked — `GROUNDING_SCOPE_SOURCED` and not
# `GROUNDING_SCOPE`, which is the bare sentence the prompts quote. Both are `prompts.py`'s and
# nothing about the source is composed here.
st.caption(GROUNDING_SCOPE_SOURCED)
# The second half of the scope, and it earns its own line rather than being appended to the one
# above: `GROUNDING_SCOPE` is quoted by the *chain*'s prompt too, where there are no tools, so
# the two sentences cannot be one string (`prompts.py`). Whose data the live figures are is in
# the sidebar's scope panel rather than here, for the same reason the pair count now is: this
# caption is what a reader meets before their first question.
st.caption(LIVE_DATA_SCOPE)

# Fail here rather than on the first message: a missing key should be obvious before the
# user has typed anything. Logging setup shares the guard because `LOG_LEVEL` is itself
# configuration — `resolve_log_level` raises `ConfigError` on a bad value, and that belongs
# in this banner like any other config problem, not in a raw traceback.
#
# Called per rerun rather than under `@st.cache_resource`: `configure_logging` is
# idempotent by design (it replaces the package logger's handler under a lock), so the
# cache saved a handler swap while making the failure invisible — a cached success means a
# later bad `LOG_LEVEL` never raises again in that process.
try:
    configure_logging()
    settings = get_settings()
except ConfigError as exc:
    st.error(f"Configuration problem: {exc}", icon=":material/error:")
    st.stop()


@st.cache_resource
def shared_agent(model: str):
    """The one agent per answering model this process has (ADR-0008, amended by T14).

    Cached across reruns *and* across sessions, which is the whole design: a checkpointer
    built per rerun remembers nothing, and one built per session would put every user's
    conversation in its own file for no benefit. Isolation comes from `thread_id` instead.

    **`model` is a cache key, not a switch.** `@st.cache_resource` keys on the arguments, so
    this yields one agent per model a reader has picked and *replaces* nothing — the entry for
    the previous model stays live for the sessions still using it. Measured before it was built
    (#15), because this is the mechanism ADR-0008's two-session guarantee sits on and a
    replacement would have handed one session another's agent mid-turn.

    Two consequences, both recorded in ADR-0008's T14 amendment. Isolation is **unchanged**: it
    was never carried by there being one agent, only by the per-session `thread_id`. And the
    process now opens one SQLite connection per model picked rather than exactly one, because
    every `build_agent` resolves `build_checkpointer` to the same `checkpoint_db` — which is
    also precisely why a conversation survives a switch, since the file and the `thread_id` are
    what a conversation is.

    Not wrapped in the configuration banner's `try`: this runs after `get_settings()` has
    already succeeded, so the failures left here (an unwritable checkpoint path, say) are not
    configuration problems a banner could explain, and hiding them would mean a chat input
    that silently cannot answer.
    """
    return build_agent(model=build_chat_model(settings, model=model))


# One id per session, minted before the first message so every turn in this browser tab lands
# on the same thread — and no other tab's. `setdefault` rather than an `if`: the point is that
# a rerun must not mint a second one, and this is the shortest statement of that.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# **Where this session's window of the sink begins** (T12 item 5). Marked once, before this
# session has appended anything, so the spend meter reads forward from here instead of over a
# file that accumulates every run and every other session that named it — the shape ADR-0011's
# T10 amendment records, where a p50 over 13 appended runs was published as one run's.
# Correctness comes from the `turn_id` filter in `observability/spend.py`; this bounds how much
# of the file has to be parsed on each rerun, which is the half that grows without limit.
#
# Deliberately **not** re-marked by "Start over": that mints a new `thread_id`, so the filter
# already excludes the previous conversation's lines and re-marking would only narrow the window
# for no gain.
def sink_path():
    """Where the event log is being written, or `None` when nobody named one.

    **One question, asked of one source, in the three places this page asks it** — the offset
    below, the thread-id caption and the spend meter. `configure_logging()` takes no arguments
    and resolves the sink from the environment itself, so reading `os.environ` is what keeps
    these three from claiming a sink the handler did not install; reading it *here* is what
    keeps them from disagreeing with each other (code review of #13).
    """
    return resolve_log_file(os.environ)


if "sink_offset" not in st.session_state:
    log_path = sink_path()
    st.session_state.sink_offset = 0 if log_path is None else sink_offset(log_path)

# **The per-session question counter** (T12 item 6). Cost and abuse limiting and **not a
# security control** — `config.MAX_QUESTIONS_PER_SESSION` states why at length: a refresh mints
# a new `session_state` and therefore a new counter, so anyone who wants past this walks past
# it. ADR-0006's input gate is the security boundary; this is a bound on what one open tab can
# spend.
if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0

# **Moved up here from just above the transcript loop**, where it sat until the sidebar
# needed to know whether this conversation has anything to export. It is an initialiser like
# the three above and belongs with them; leaving it below meant the sidebar read a key that
# does not exist yet on the first run — a `KeyError` on the page, not an empty transcript.
if "messages" not in st.session_state:
    st.session_state.messages = []

#: The chat input's widget key, and the seeded question's `session_state` key — **named here
#: because the sidebar reads them ~700 lines before either is written**. Streamlit applies
#: the incoming widget states to the session *before* the script runs, so on the rerun that
#: carries a submitted question `st.session_state[QUESTION_KEY]` already holds it while
#: `st.chat_input` is still hundreds of lines away; the seeded half is a plain write from the
#: example buttons.
#: Together they are how `answering_now()` can tell, at the top of the page, that this run has a
#: multi-second turn ahead of it — which is what the two deferred slots need in order to say so
#: rather than go blank. Measured, not assumed: `test_the_sidebar_can_tell_a_turn_is_coming`.
#:
#: Constants rather than three literals: the key is now read in the sidebar and written at the
#: widget, and a key spelled twice is a lookup that silently returns `None` the day the copies
#: disagree — which here would fail *open*, back into the blank panel this exists to fix.
QUESTION_KEY = "question"
PENDING_QUESTION_KEY = "pending_question"

#: The model picker's widget key, and therefore where the chosen slug lives (T14, #15).
#:
#: A widget key rather than a hand-managed `session_state` entry, because Streamlit restores a
#: keyed widget's value across reruns by itself — which is exactly the "UI toggles (model,
#: strategy)" slot ADR-0008 already reserves in `session_state`, and nothing more. The picker is
#: rendered in the sidebar and read ~1,400 lines below at the `answer()` call, so the key is
#: named here for `QUESTION_KEY`'s reason: a key spelled twice is a lookup that returns `None`
#: the day the copies disagree, and here that would silently answer on the configured model
#: while the sidebar showed another.
CHAT_MODEL_KEY = "chat_model"


def chosen_model() -> str:
    """The model this turn will answer on — the picker's value, or the configured default.

    The fallback is for the ordering rather than for a missing case: on the very first run the
    widget has not been created yet when the initialisers above execute, and `Settings` is the
    honest answer at that point because it is what the widget is about to select (its options
    lead with it). Every run after the first reads the reader's own choice.
    """
    return st.session_state.get(CHAT_MODEL_KEY) or settings.chat_model


#: The label the progress box ends on, and the one a replayed row puts back.
#:
#: A constant because it is written at the live turn and read on every rerun after it: a label
#: spelled twice is a box that says one thing while a turn is answered and another once the
#: reader clicks anything, which is the drift this file keeps a single source of truth for.
TURN_COMPLETE = "Answered"


def answering_now() -> bool:
    """Whether this run has a question in it, asked before the input that carries one exists.

    Both halves of `prompt` below, at the one moment the sidebar can still act on the answer. A
    typed question arrives as widget state (see `QUESTION_KEY`); a seeded one is a
    `session_state` write the example buttons made on the previous rerun and this run has not
    popped yet.

    **A hint about this run, never the source of truth about the turn.** `prompt` below stays
    the thing that decides whether a turn is answered, and it is deliberately not read from
    here: the pop is what makes a seeded question fire once, and a second reader of that key
    that consumed it would ask the question twice. This only decides what two placeholders say
    while waiting.
    """
    return bool(
        st.session_state.get(QUESTION_KEY) or st.session_state.get(PENDING_QUESTION_KEY)
    )


def render_export_buttons(messages: list[dict[str, object]]) -> None:
    """Take this conversation away, as JSON or as CSV (T12 item 4, user story 35).

    Story **35** — *"export or copy a completed brief (basic)… Full multi-format export is
    Tier-2"* — and not the 25 this shipped citing, which is *"RAGAs metrics reported per
    bucket"* (code review of #13).

    **Defined above the sidebar block rather than beside the other render functions**, which is
    a constraint of this file and not a preference: the sidebar runs at module scope, so a name
    it calls has to exist by then — the same reason `_RENDERERS` is declared *below* the
    functions it maps.

    The source is `messages`, the display transcript, and `finbrief/export.py` argues why that
    rather than the checkpointer at length: this is what the analyst saw, refusals included.

    **Both payloads are built on every rerun**, because `st.download_button` takes bytes and
    offers no lazy callback to build them from. That is affordable for the reason
    `observability/events.py` reads its whole log eagerly — the volume is bounded by a human
    typing questions — and it is the cost of the button being a plain download rather than a
    second request the server has to route.

    Nothing is logged here. The log line belongs to the *click*, because a line written on every
    rerun would count reruns and call them exports.
    """
    exported = Transcript.of(messages, thread_id=st.session_state.thread_id)
    # **Turns and messages are two counts, because they are two units.** A turn is an exchange —
    # what `turn_id` names on every log line and what the `Token spend` panel counts three
    # panels up — and the file holds a row per *message*, so one question and its answer are one
    # turn and two rows. This caption said `2 turn(s)` for that conversation while the meter
    # beside it said `across 1 turn(s)`: one page, one word, two numbers (`export.py`, and
    # CONTEXT.md's **Turn**).
    st.caption(
        f"Export this conversation — {exported.turns} turn(s), "
        f"{len(exported.messages)} message(s), {len(exported.sources)} source(s)."
    )
    for label, suffix, body, mime in (
        ("JSON", "json", as_json(exported), "application/json"),
        ("CSV", "csv", as_csv(exported), "text/csv"),
    ):
        data = body.encode("utf-8")
        if st.download_button(
            f"Download {label}",
            data=data,
            # `export.file_name`, not an f-string here: a download button's file name is not on
            # the proto — the bytes are served over a URL — so `AppTest` cannot see it, and a
            # name built in this file would be a claim no test could reach.
            file_name=file_name(st.session_state.thread_id, suffix),
            mime=mime,
            icon=":material/download:",
            width="stretch",
        ):
            # A count and a format. **Never the payload** — this file is the analyst's own
            # questions and the filer's prose, which is exactly what these kept lines may not
            # carry (`export.log_export`, ADR-0011).
            log_export(exported, fmt=suffix, size=len(data))


def fill_spend_meter(slot, *, answering: bool) -> None:
    """Render the whole panel into `slot`, replacing whatever it held.

    **Called twice per run, and that is the fix for the panel vanishing mid-turn.** The slot is
    created in the sidebar and was filled only at the end of the script, after the turn — so
    while a turn was being answered the slot held Streamlit's `Empty` delta, which does not
    "wait", it *clears* the node the previous run had drawn there. The panel therefore blinked
    out for the whole of the one wait it exists to describe, and came back when the script
    finished (manual testing of #13). The neighbouring panels never moved because they are
    rendered inline, where the sidebar runs.

    The eager fill is what keeps something on screen for that window; the fill at the end is
    still the authority, because this run's `agent_turn` line is written by the turn between
    them. Nothing else can do it: Streamlit repaints on script progress, and a script blocked
    inside `answer_turn` has no progress to report.

    The label lives here rather than at the two call sites — the same panel written twice is two
    panels the day one of them is edited.

    **The two fills go to two slots, and the reason is a defect this docstring got wrong once.**
    Writing a container into a placeholder twice in one run does *not* replace the subtree: the
    frontend keeps the node and addresses its children by index, so any child the second fill
    does not reach survives. The two fills are legitimately different lengths — the eager one
    opens with the "measuring this turn" caption, and `unfinished` is true mid-turn and false
    after — so the tail was orphaned and stayed on screen beside its replacement: two `Partial:`
    banners a call apart on a partial total, the classifier caption twice on a complete one.

    **`slot.empty()` before the refill was the first fix and it did not work.** Clearing a
    placeholder and immediately writing to the same path again leaves the frontend applying two
    deltas to one node, and the children came back; the browser showed the duplicate on the next
    run either way. `examples_slot` is the shape that *does* work — cleared and then not written
    again — and it is what this follows now: the eager panel gets its own slot, which the
    settled fill empties before filling a **second** slot below it. The settled panel's path
    never carries the eager fill's children, so no arrangement of the two lengths can collide.

    Equalising the lengths was the other candidate and is not available here: `unfinished` and
    `Partial:` are read from the log, so the panel's shape is a function of the data rather than
    of this code.

    **`AppTest` cannot see any of this**, which is why it went out twice: the tree it exposes is
    the settled one, in which an orphan is already pruned — a scratch run reported one panel and
    one warning while the browser showed two. The browser is the only instrument, so both the
    defect and the fix were confirmed there. `spend_panel` in the tests asserts the half a test
    *can* reach: that exactly one panel survives to the end of a run.
    """
    with slot.container(), st.expander(":material/toll: Token spend"):
        render_spend_meter(answering=answering)


def render_spend_meter(*, answering: bool = False) -> None:
    """This conversation's token spend, and its cost when one is configured (T12 item 5).

    **Reads the log T8 already writes rather than adding an instrument**, which is what makes
    this affordable: the counts are on `agent_turn` and `query_translation` already, and
    `observability/spend.py` does the arithmetic through the one reader. So the meter exists
    only when the sink does — `FINBRIEF_LOG_FILE` unset means there is nothing to read, and
    this says so rather than rendering zeros, because a spend of `0` is a claim that the calls
    were free.

    Read from **this session's own window** of the file (`events.sink_offset`, taken once when
    the session starts) and filtered to this conversation's turn ids. The offset is for read
    cost; the filter is for correctness, and the module docstring argues why that is the sharper
    of the two here — the sink is shared, and a total over the whole of it is a total over every
    run that ever named it (ADR-0011).

    Nothing is emitted here. The instrument is at the layer that produces the behaviour and this
    is the layer that displays it — ADR-0011's amendment is explicit that an event wired to a
    page is an event measuring clicks.

    **Every state renders the panel; only the contents change** (manual testing of #13). Sink
    off, nothing measured yet, a turn with no total, totals, totals that are a floor — five
    things to say and five sentences, because the alternative this replaced was the panel
    itself disappearing, and a reader cannot tell an absent measurement from a broken feature.
    It is the rule the warm-thread caption and `distance_label` already follow: state the
    absence.
    """
    path = sink_path()
    if path is None:
        st.caption(
            "Token spend is read from the event log, which is off. Set `FINBRIEF_LOG_FILE` to "
            "meter this conversation."
        )
        return
    spend = conversation_spend(
        read_events(path, start_offset=st.session_state.sink_offset),
        thread_id=st.session_state.thread_id,
    )
    if answering:
        # **The state the whole two-fill arrangement exists to render.** Said before the figures
        # because it is what the figures mean right now: this run has a question in it and the
        # turn answering it has not written its total yet, so everything below covers the turns
        # *before* it. Only the eager fill ever passes this — by the time the fill at the end of
        # the script runs, the turn is in the log and its numbers are in the figures.
        st.caption(
            "Measuring this turn — the figures below cover the turns before it, and take it in "
            "when it finishes."
        )
    if spend.unfinished:
        # **Read out of the log rather than inferred from this run**, which is why it survives
        # into the settled fill and `answering` does not: a turn that raised inside
        # `answer_turn`'s `except` never wrote its `agent_turn` line, so its answering calls are
        # missing from the totals below for good. Both readings are given because the log cannot
        # tell them apart — and neither of them is a spend of zero.
        st.caption(
            f"{spend.unfinished} turn(s) here have no total: still being answered, or ended "
            "without reporting one. The answering calls behind them are not in these figures."
        )
    if not spend.measured:
        # **Not zeros.** Either nothing has been asked yet, or the provider reported no usage
        # block at all — and neither is "this conversation cost nothing". OpenRouter fronts many
        # upstreams and whether a given one meters itself is not ours to assert
        # (`observability/tokens.py`).
        st.caption(
            f"No usage reported yet for this conversation "
            f"({spend.calls} model call(s) across {spend.turns} turn(s))."
        )
        return
    st.markdown(
        f"**Input** `{_tokens(spend.input.total)}`  \n"
        f"**Output** `{_tokens(spend.output.total)}`  \n"
        f"**Calls** `{_calls(spend)}` across `{spend.turns}` turn(s)"
    )
    dollars = spend.dollars(
        input_per_mtok=settings.input_cost_per_mtok,
        output_per_mtok=settings.output_cost_per_mtok,
        # The prices are a single pair, configured for the model `Settings` names — see
        # `Spend.dollars`. A conversation answered on anything else is reported in tokens and
        # not in dollars, which is the branch below.
        priced_model=settings.chat_model,
    )
    if dollars is None and not spend.all_answered_on(settings.chat_model):
        # **The picker's cost consequence, stated where the figure would have been** (T14, #15).
        # Ordered before the unpriced branch because it is the more specific claim: with prices
        # configured *and* another model answering, "set the two variables" is advice that would
        # not help, and with no prices configured this reader has the same two things to do
        # either way.
        #
        # **The sentence is `spend.py`'s and it is a function of the state**, not a literal
        # here. This branch is reachable with an *empty* model set — a turn in flight, or one
        # that raised after the planner's round, leaves a metered `query_translation` line and
        # no `agent_turn` — and the literal this replaced said "answered on another model" about
        # a conversation where nothing had answered (code review of #15). It also had a
        # near-copy on the analytics page, which is the duplication `UNMETERED_CLASSIFIER_NOTE`
        # exists to have already taught us about.
        st.caption(
            unpriced_because_of_the_model(
                spend.models, settings.chat_model, scope="this conversation"
            )
        )
    elif dollars is None:
        # No price is assumed rather than guessed at: every model here is reached through
        # OpenRouter's routing, so there is no rate card in this repo to read one from
        # (ADR-0011 §3). What a reader needs is the two knobs, which is what the caption gives.
        st.caption(
            "Cost is not priced: set `FINBRIEF_INPUT_COST_PER_MTOK` and "
            "`FINBRIEF_OUTPUT_COST_PER_MTOK` from your provider's rate card."
        )
    else:
        # **Labelled an estimate, because it is one.** The two prices are configuration read
        # from `.env` and the tokens are a measurement; multiplying them gives a figure whose
        # accuracy is the accuracy of a rate a reader typed in. `**Cost**` alone read as a
        # billed amount (#14 copy pass), and the analytics page prints the same label.
        st.markdown(f"**Cost (estimate)** `${dollars:.4f}`")
    # **The unmetered-classifier caveat is not here.** It was this panel's and the analytics
    # page's, typed out in both — and ADR-0011 §4 requires the omission stated on screen, not
    # stated twice. It is `spend.UNMETERED_CLASSIFIER_NOTE` now, rendered on the surface that
    # totals a whole log; this panel totals one conversation and does not repeat it
    # (#14 copy pass).
    if spend.partial:
        # **Said beside the figure, not folded into it.** A total missing a call it should have
        # counted is a floor, and a floor presented as a total is the silent narrowing this
        # whole path is built against — so the denominators are printed rather than the
        # shortfall being left for a reader to infer. One clause, and the word a reader needs is
        # the first one.
        #
        # A caption rather than the `st.warning` this was: nothing here needs attention and a
        # yellow box beside a number reads as a fault in the number rather than as a qualifier
        # on it. The same demotion as the analytics page's, which renders the same sentence
        # (#14 copy pass).
        st.caption(
            f"A floor: {spend.input.reported_calls} of {spend.calls} call(s) reported input "
            f"tokens and {spend.output.reported_calls} reported output."
        )


def _tokens(count: int | None) -> str:
    """`12,431`, or the word for a count nothing reported. Never `0` for an absence."""
    return "not reported" if count is None else f"{count:,}"


def _calls(spend: Spend) -> str:
    """`3`, or `≥3` when the call count is a floor rather than a count.

    `agent_turn` writes its `calls` only once something reported usage, so a turn that metered
    nothing is worth the honest floor of one here — and printing that floor as a count is the
    same fabrication in the denominator that `_tokens` refuses in the numerator
    (`spend.Spend.floored`, code review of #13).
    """
    return f"≥{spend.calls}" if spend.calls_are_a_floor else str(spend.calls)


with st.sidebar:
    st.subheader("Conversation")
    # **Above the fold, and deliberately not in a panel** (T12 item 1). ADR-0008 §4 makes this
    # an obligation in these words — "the sidebar says so" — because a user who is not told
    # reads a lost conversation as a bug, and a reviewer cannot tell an accepted consequence
    # from an oversight. A collapsed panel states it only to a reader who clicks, so the
    # density work below stops here: everything *else* folds, this does not.
    #
    # **The refresh warning alone**, and that is the whole content of this block. It used to
    # open with the follow-up mechanic — which the help panel below also states, in better
    # words ("one company at a time") and with a different example phrase. Two copies of one
    # fact, already drifted: the bug class this repo keeps hitting, arriving as UI copy rather
    # than as a check. ADR-0008 §4 obliges *this* sentence to be unfoldable and says nothing
    # about the follow-up hint, so the hint moves entirely into the panel and the obligation
    # keeps the space it is owed.
    st.caption(
        "Memory lasts as long as this browser session — refreshing the page starts a new "
        "conversation."
    )
    # **Only when there is somewhere to look it up** (T12 item 1). The thread id is the handle
    # on this conversation *in the sink*: `log_turn` below prefixes every `turn_id` with it, so
    # it is what makes a log line lead back to a conversation. With `FINBRIEF_LOG_FILE` unset
    # there is no log, and the caption is then a hex string in front of an analyst with nothing
    # to do with it — sidebar space spent on a handle to nothing.
    #
    # Through `sink_path()`, which reads the environment rather than `Settings` because that is
    # where the sink's own resolution reads it (`config.resolve_log_file`, called by
    # `configure_logging` with no arguments): asking the same question of the same source is
    # what keeps this caption from claiming a sink the handler did not install.
    if sink_path() is not None:
        # Not a secret — a uuid identifies a conversation and says nothing about who is having
        # it, which is why it is also safe on every log line.
        # `THREAD_HANDLE_CHARS`, shared with the export's file names: this caption and a
        # downloaded file have to name a conversation the same way, or the file cannot be
        # matched back to the handle the page showed.
        st.caption(f"Thread `{st.session_state.thread_id[:THREAD_HANDLE_CHARS]}`")
    # A fresh uuid, not a cleared checkpointer: the old thread is orphaned rather than deleted
    # (nothing else can reach it), and the cached agent survives — rebuilding it here would
    # discard every *other* session's memory too, which is the bug this button looks like.
    if st.button("Start over", icon=":material/restart_alt:"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []

    # **Four panels, all collapsed** (T12 item 1). The sidebar had grown to four stacked blocks
    # of prose — roughly a screen and a half — so the panel a reader wanted was always below
    # something they had already read, and the scope disclosure ADR-0007 requires was competing
    # with a cluster listing for the same attention. Collapsing is the *whole* change: every
    # word is still rendered, and `AppTest`'s block accessors recurse into an expander, so the
    # tests that bind these words to `config` and to the committed ingest evidence still find
    # them (`tests/test_app_smoke.py`, `tests/test_grounding_scope.py`).
    with st.expander(":material/help: How to use FinBrief"):
        # T12 item 3's other half: the three things a reader cannot guess from a chat box.
        st.markdown(
            "- **Ask about one company at a time.** Follow-ups resolve against it, so "
            "*and its margins?* needs no name.\n"
            "- **`[1]` markers are citations.** Each one resolves to an entry in the "
            "**Sources** panel under the answer, with the filer's own words to check it "
            "against.\n"
            "- **Prices and ratios are not from the filings.** Filings carry no prices, so "
            "hard figures come from the finance tools and arrive on their own cards.\n"
            "- **Open *How I answered*** to see which queries ran and which retriever "
            "surfaced each chunk."
        )
        st.caption(
            "FinBrief answers research questions, not *should I buy this* — it refuses "
            "personalised advice by design."
        )

    # **A slot, filled at the very end of the script.** The sidebar runs before this run's turn
    # has been answered, so building the buttons here would offer a file missing the exchange
    # the reader just had — and they would not be able to tell, which is the worst version of
    # that bug. Deferring is the whole fix; `render_export_buttons` is called once, below, from
    # a transcript that includes this turn.
    export_slot = st.empty()
    # **Held open while a turn runs, rather than left blank.** An `st.empty()` does not reserve
    # space, it clears the node the previous run drew there — so on the rerun that answers a
    # question the buttons vanished for the length of the turn and returned with the answer,
    # the same wart the spend panel had. A caption in their place is not the buttons and does
    # not pretend to be: it says why they are gone, which is exactly the thing a disappearance
    # cannot say. Only while a turn is in flight — on any other rerun the fill below is
    # microseconds away and a flash of this sentence would be noise.
    #
    # **A caption and not a disabled button**, for a mechanical reason worth recording: two
    # fills of one slot in a single run are two `download_button`s with identical parameters,
    # and Streamlit raises `StreamlitDuplicateElementId` for that — measured, not assumed.
    # Keeping the placeholder widget-free is what keeps this fix from being a crash.
    if st.session_state.messages and answering_now():
        with export_slot.container():
            st.caption(
                "Export returns when this turn finishes — the file has to contain the answer "
                "you are about to read."
            )

    # A slot for the same reason the export has one: the meter reads the log, and this run's
    # `agent_turn` line is written by the turn below. Built here it would report the spend as of
    # the *previous* question, which on a cost panel is the one number a reader would act on.
    #
    # **Filled here as well as at the end, unlike the export's.** See `fill_spend_meter`: the
    # deferral above is what made the panel disappear mid-turn, and a panel whose subject is
    # *this conversation's cost* is one a reader looks at while the cost is being incurred.
    #
    # **Two slots, filled by two different fills.** See `fill_spend_meter`: one slot written
    # twice in a run merges by child index and leaves the longer fill's tail on screen. The
    # eager panel lives here and is emptied at the end of the script — the `examples_slot`
    # pattern, which is the one placeholder shape measured to clear — and the settled panel is
    # written into the slot below it, whose path has never held anything else.
    #
    # **Filled unconditionally, as it was when this was one slot**, so there is no rerun — a
    # panel opening, a button click — on which the sidebar has a hole where the panel was. That
    # costs a second read of the log on such a rerun, which is affordable for the reason
    # `observability/events.py` reads it eagerly at all: the volume is bounded by a human
    # typing, and `sink_offset` bounds it where it would not be. Two panels never coexist even
    # for a frame, because the settled fill below **clears this slot before** writing its own.
    spend_eager_slot = st.empty()
    spend_slot = st.empty()
    fill_spend_meter(spend_eager_slot, answering=answering_now())

    with st.expander(":material/policy: Grounding scope"):
        for detail in GROUNDING_SCOPE_DETAILS:
            st.markdown(f"- {detail}")
        # A caption rather than a sixth bullet: the five above are what the KB is and is not,
        # and this is where to go when they are not enough. Its words are `prompts.py`'s like
        # every other line in this panel.
        st.caption(GROUNDING_SCOPE_VERIFY)

    with st.expander(":material/tune: Configuration"):
        # **The model is a choice now, and this is where it was already stated** (T14, #15). The
        # line it replaces read `**Model** \`{settings.chat_model}\``, so the panel that told a
        # reader what was answering is the panel that lets them change it — rather than a
        # sixth panel, or a control floating above the prose ADR-0008 §4 requires stay first.
        #
        # A `selectbox` over `config.chat_model_options`, never a text input: a mistyped slug
        # reaches OpenRouter as a provider error in the middle of a turn, and the options are
        # derived rather than listed here so a hardcoded copy cannot drift from `config`.
        # `FINBRIEF_CHAT_MODEL` leads the list, which is what makes it the default selection.
        st.selectbox(
            "Model",
            options=chat_model_options(settings.chat_model),
            key=CHAT_MODEL_KEY,
            help=(
                "Answers only. The prompt-injection classifier and the embeddings are fixed — "
                "switching here cannot weaken the gate, and could not change retrieval "
                "without making the index unsearchable."
            ),
        )
        # Switching is free of the conversation, and a reader is owed that sentence: the memory
        # is the checkpointer's and it is keyed on this session's thread, not on the model, so a
        # switch mid-conversation carries the history across (asserted in `test_agent.py` and
        # `test_app_state.py`, not assumed).
        st.caption("Switching keeps this conversation — the new model sees what came before.")
        # One value, not two. Until Phase 4 this panel named a `BASELINE_STRATEGY` constant and
        # captioned the gap to the configured one, because the pre-registered default (ADR-0005)
        # was a strategy `retrieve()` refused. Now the configured strategy *is* what answers, so
        # a second line would be a gap that no longer exists — and the way this panel stays
        # honest is that `agent.build_agent` reads these same two settings (nothing here
        # restates them).
        st.markdown(
            f"**Strategy** `{settings.retrieval_strategy}"
            f"{' + translation' if settings.query_translation_enabled else ''}`  \n"
            f"**Top-k** `{settings.retrieval_k}`"
        )
        # The per-chunk provenance this points at is ADR-0004's requirement; the citation lives
        # here rather than in the caption, which an analyst reads for the pointer and not for
        # the decision record (#13).
        st.caption(
            "Every answer's *How I answered* panel shows the queries that ran and which "
            "retriever surfaced each chunk."
            if settings.retrieval_strategy is RetrievalStrategy.HYBRID
            else "Vector search only. Hybrid retrieval adds BM25 over the same query variants."
        )

    with st.expander(":material/apartment: Universe"):
        # The summary line, plus the grouping that used to be a column of its own. Both derived
        # from `CLUSTERS`, which stays the one place the grouping is computed.
        # The ordering sentence is here rather than implied: the table shows no cluster column,
        # so a reader who does not know the rows are grouped reads 15 in an arbitrary order and
        # the clusters this line names are invisible in the thing underneath it.
        st.caption(
            f"{len(UNIVERSE)} companies in {len(CLUSTERS)} peer clusters — "
            + " · ".join(
                f"{cluster.label} ({len(tickers)})" for cluster, tickers in CLUSTERS.items()
            )
            + ". Rows follow that cluster order."
        )
        # **A table, because the grouped ticker list this replaces assumed ticker literacy**
        # (#13). `**Healthcare** — JNJ, LLY, PFE` discloses the Universe only to a reader who
        # already knows that `LLY` is Eli Lilly, and the scope of the knowledge base is the
        # first thing this app owes a reader. `prompts.UNIVERSE_ROWS` owns the rows, and owns
        # why there are two columns rather than three.
        #
        # **Nothing is pinned, because the pins are what truncated the names.** This carried
        # three columns at measured widths summing to the sidebar's content width — 54 for the
        # ticker, 60 for the `Item 7A` verdict, 91 for `Company` — and 91px does not finish a
        # legal name: `Microsoft Corporatio`, `JPMorgan Chase & C`. With the third column gone
        # the arithmetic no longer needs doing. `width="stretch"` fills the expander and
        # `Company` takes everything `Ticker` does not, which is every pixel this panel has to
        # give a name. (`width="stretch"` and not `use_container_width`, which is the deprecated
        # spelling of the same thing in Streamlit 1.60 and warns.)
        st.dataframe(
            pd.DataFrame(UNIVERSE_ROWS),
            hide_index=True,
            width="stretch",
            # All 15 rows, rather than the ten `"auto"` would show behind a nested
            # scrollbar inside a sidebar that already scrolls. From the content, not a
            # pixel count.
            height="content",
        )
        # The thinnest cluster makes the crispest example, and picking it from the data keeps
        # this panel entirely config-driven — a hardcoded ticker would be a KeyError the day
        # the Universe changed.
        #
        # **Through `config.thinnest_cluster_filer` rather than a `min` here**, which is the one
        # derivation `prompts.EXAMPLE_QUESTIONS` also reads. Two copies with different
        # tie-breaks disagreed on screen — five clusters tie at two members, so this caption
        # named TSLA while the example button asked about Bank of America (review of #13).
        example = thinnest_cluster_filer()
        st.caption(
            f"Peers come only from this set, e.g. {example.ticker} vs. "
            f"{', '.join(PEERS[example.ticker])}."
        )


def as_markdown(text: str) -> str:
    """Escape what Streamlit's Markdown would swallow, and leave the rest to render.

    Only `$`, and for the same reason `render_sources` reaches for `st.text` below:
    `st.markdown` parses `$…$` as KaTeX. The answer is the dollar-densest surface on the
    page — the persona is asked to be quantitative where the source is — so "net sales rose
    to $416,161 million from $391,035 million" renders as prose plus one maths expression,
    and both figures are gone from the headline sentence of a finance assistant.

    `st.text` is the wrong instrument here, unlike for a source body: the persona answers in
    bullets and short paragraphs and the `[n]` markers sit in that prose, so the Markdown
    has to keep rendering. Escaping costs a literal `\\$` inside a fenced code block, which
    this persona has no reason to emit.
    """
    return text.replace("$", r"\$")


def escaped(text: str) -> str:
    """`text` with every Markdown-active character neutralised.

    Stronger than `as_markdown`, and for a different threat. That function escapes `$` in text
    FinBrief wrote; this one is for text **strangers** wrote — a headline title and a publisher
    name, straight off a public RSS feed (user story 17). Rendered raw through `st.markdown`, a
    title is free to inject an image, a link, a heading, or a run of emphasis that swallows the
    rest of the card.

    `st.text` would be the simpler answer and is what a source body uses, but a card wants its
    headline bold and its publisher small, so the Markdown has to keep rendering around the
    escaped text. Escaping the span is the narrower fix.

    **Two sets, because Markdown has two kinds of special character.** A first draft escaped
    them all everywhere and turned `finance.yahoo.com` into `finance\\.yahoo\\.com` on every
    news card. `.`, `-`, `#` and `+` mean something only at the *start* of a line — a list item,
    a heading — so they are escaped only in the leading position; the rest are active anywhere.
    """
    if not text:
        return text
    body = "".join(f"\\{one}" if one in _INLINE_ACTIVE else one for one in text)
    return f"\\{body}" if text[0] in _LINE_START_ACTIVE else body


#: Active anywhere in a line: emphasis, code, links, images, autolinks — plus `$`, for the KaTeX
#: reason `as_markdown` exists, and `|`, which would otherwise open a table cell.
_INLINE_ACTIVE = frozenset("\\`*_[]()<>$|!")

#: Active only as a line's first character, where they open a list item, a heading or a quote.
_LINE_START_ACTIVE = frozenset("#+-.>")


def step_label(step: Step) -> str:
    """What to show while a tool runs — user story 14's words (T5, #9).

    The phrasing lives here and not on `Step`, because it is UI copy: the agent reports *which*
    tool with *what* argument, and this file decides how that reads. The ticker arrives already
    truncated (`Step.of`), so a refused call's raw argument cannot stretch the status line.

    A tool with no label falls back to its own name rather than to something vague: a new tool
    should look unfinished here, not anonymous.
    """
    where = f" · {step.ticker}" if step.ticker else ""
    return {
        SEARCH_TOOL_NAME: "Searching the filings",
        STOCK_TOOL_NAME: "Fetching market data",
        RATIOS_TOOL_NAME: "Comparing ratios against peers",
        NEWS_TOOL_NAME: "Fetching recent headlines",
    }.get(step.tool, step.tool) + where


def distance_label(distance: float | None) -> str:
    """`distance 0.7421`, or what to say when there is no distance to state.

    Under `hybrid` a chunk BM25 recovered is one vector search did not return, so it has no
    distance — which is the ADR-0004 case worth naming rather than papering over. Printing a
    stand-in would put a number in the panel that no measurement produced, and `distance None`
    reads as a bug.
    """
    return "no vector distance" if distance is None else f"distance {distance:.4f}"


def retriever_label(context: Context) -> str:
    """`vector + BM25` — which retrievers found this chunk, for the sources panel's caption.

    The shortest answer to the question ADR-0004's exact-identifier prediction turns on, put
    where a reader is already looking at the citation.
    """
    # `Retriever` is a closed two-member enum, so this is a casing fix and not a lookup with a
    # fallback: a dict `.get` here had a `"vector": "vector"` entry and a default branch nothing
    # could reach. The empty case *is* reachable — a reply checkpointed before Phase 4 carries
    # no provenance at all.
    found = ["BM25" if r is Retriever.BM25 else r.value for r in context.retrievers]
    return " + ".join(found) if found else "provenance not recorded"


def accession_label(context: Context) -> str:
    """The chunk's accession number, linked to the filing's own index page on EDGAR.

    What turns a citation from a claim into something an analyst can check (user story 2). The
    panel already shows the filer's words; this is where the words came from, one click away.

    **The link is derived from this chunk's own accession**, never from a template written here
    — `ingestion/edgar.filing_index_url` asks edgartools for the URL shape that
    `FilingRef.url` already records, and resolves the CIK from the ticker rather than from the
    accession's leading block, which belongs to the filer's agent. A link built any other way
    is one that can point at a different company's filing while looking entirely correct, which
    is why `tests/test_app_smoke.py` asserts the *rendered* link carries the chunk's own
    accession rather than merely that a link is present.

    Unlinked when the CIK does not resolve, and the accession is still shown: a missing link is
    "we cannot address this filing on EDGAR", which is a different claim from "this citation
    has no source" (CLAUDE.md).
    """
    url = filing_index_url(context.ticker, context.accession)
    return f"[{context.accession}]({url})" if url else f"{context.accession} (no EDGAR link)"


def variant_labels(search: Search) -> dict[str, str]:
    """Each variant this search ran, mapped to what to call it in the panel.

    Three kinds, named apart because they have different causes and the T6 finding is about
    which one does the work (ADR-0004 amendment): the analyst's own words, the deterministic
    ticker form `config` supplied, and the planner's sub-queries. Calling the ticker form
    "sub-query 1" would credit a model for a lookup.

    A label rather than the query text, because the texts are listed once above the table and
    repeating a whole question inside a Markdown table cell would wrap badly and — worse — break
    the table outright on a query containing a pipe.
    """
    labels = {search.variants[0]: "original"} if search.variants else {}
    if search.ticker_form is not None:
        labels[search.ticker_form] = "ticker form"
    for number, sub_query in enumerate(search.sub_queries, start=1):
        labels[sub_query] = f"sub-query {number}"
    return labels


def barren_variants(search: Search) -> tuple[str, ...]:
    """The variants this search ran that no returned chunk credits — ADR-0004 §1's datum.

    A variant that surfaced nothing is the whole reason `retrieve()` returns a `Retrieval`
    rather than a bare sequence of contexts: it is a fact about the retrieval that no chunk's
    provenance can carry. The panel listed such a variant among the queries but said nothing
    about it, leaving a reader to diff that list against every table to notice — which is the
    datum being on screen without being reported (issue #6 review).

    **Empty when no provenance was recorded at all**, rather than "every variant was barren".
    A reply checkpointed before Phase 4, or a search the collection had nothing for, has no rows
    to credit a variant with — and "this query found nothing" is a different claim from "we
    cannot say what this query found". The empty-collection case has its own banner in
    `render_sources`.
    """
    if not any(context.provenance for context in search.contexts):
        return ()
    credited = {row.variant for context in search.contexts for row in context.provenance}
    return tuple(variant for variant in search.variants if variant not in credited)


def render_how_i_answered(searches: tuple[Search, ...]) -> None:
    """The RAG-visualization panel (user story 5, ADR-0004): what ran, and what it surfaced.

    Original query → sub-queries → each retrieved chunk with its RRF score and the per-chunk
    provenance ADR-0004 asks for: which variant × which retriever surfaced it, at what rank, for
    what contribution. This is the surface that makes "*why* hybrid wins" checkable rather than
    asserted — and the one place a reader can see a sub-query that surfaced nothing at all,
    which no chunk's provenance can report.

    Rendered from the turn's own `Search` objects, never re-retrieved: the numbers here have to
    be the numbers behind the answer's `[n]` markers, and a second retrieval is a second chance
    to disagree with them.

    Nothing is rendered for a turn that did not search, or for replies whose provenance did not
    survive a checkpoint written by an older shape — an empty panel promising an explanation is
    worse than no panel.
    """
    explicable = [
        search
        for search in searches
        if search.variants or any(c.provenance for c in search.contexts)
    ]
    if not explicable:
        return
    with st.expander(":material/account_tree: How I answered"):
        for number, search in enumerate(explicable, start=1):
            if len(explicable) > 1:
                st.markdown(f"**Search {number}**")
            labels = variant_labels(search)
            st.markdown(f"**Queries this search ran** ({len(search.variants)})")
            for index, variant in enumerate(search.variants, start=1):
                # `.get`, because a variant a checkpointed reply carries may predate the label
                # map that would name it — a missing label is a cosmetic loss, not a KeyError.
                label = labels.get(variant, "query")
                st.markdown(f"{index}. `{label}` — {as_markdown(variant)}")
            if search.ticker_form is not None:
                # The mechanism is ADR-0004's amendment (the T6 finding: the recovery is
                # embedding-side and the ticker form is what carries it). The caption explains
                # the mechanism, which is what an analyst needs; the citation stays here (#13).
                st.caption(
                    "The ticker form is added deterministically from the Universe, not by a "
                    "model: a chunk's header carries `TSLA`, so a lexical search for *Tesla* "
                    "would miss most of the filer."
                )
            if not search.translated:
                st.caption("Query translation was off, so only the question itself was run.")
            elif not search.planned:
                # Distinguished from the caption below, because the causes are different and
                # only one of them is the model's: at `FINBRIEF_MAX_SUB_QUERIES=0` translation
                # is on and the planner is never invoked, so crediting the emptiness to a
                # question that "was already specific" describes a chat call that never
                # happened — in the one cell ADR-0004 §6 uses to isolate what the planner
                # contributes (issue #6 review).
                st.caption(
                    "The query planner is switched off (`FINBRIEF_MAX_SUB_QUERIES=0`), so no "
                    "sub-query was asked for — only the question and, where one applies, its "
                    "ticker form were run."
                )
            elif not search.sub_queries:
                st.caption(
                    "The query planner added nothing — the question was already one specific "
                    "thing a filing answers, so no sub-query was retrieved."
                )
            for context in search.contexts:
                st.markdown(
                    f"**[{context.rank}] {context.citation}** · RRF "
                    f"{context.fused_score:.6f} · {distance_label(context.distance)}"
                )
                if not context.provenance:
                    st.caption("Provenance was not recorded for this chunk.")
                    continue
                rows = "\n".join(
                    f"| {labels.get(row.variant, 'query')} | {row.retriever.value} "
                    f"| {row.rank} | {row.contribution:.6f} "
                    # This row's own distance, not the chunk's nearest — which is what makes
                    # ADR-0004 §7's "the recovery is embedding-side" readable off one turn: the
                    # same chunk at two distances under two surface forms. An em dash for BM25,
                    # which has no distance of its own (§3).
                    f"| {'—' if row.distance is None else f'{row.distance:.4f}'} |"
                    for row in context.provenance
                )
                st.markdown(
                    "| query | retriever | rank | RRF contribution | distance |\n"
                    "|---|---|---:|---:|---:|\n" + rows
                )
            if barren := barren_variants(search):
                named = ", ".join(f"`{labels.get(variant, 'query')}`" for variant in barren)
                # "Translation only ever *adds*" is ADR-0004's invariant, and the reason this
                # caption can reassure rather than alarm. Cited here, not on screen: the reader
                # needs to know the barren variant cost them nothing, not which ADR says so
                # (#13).
                st.caption(
                    f"Surfaced no chunk in the top-{len(search.contexts)}: {named}. Each ran "
                    "through every retriever this strategy uses; nothing they found survived "
                    "fusion. Translation only ever *adds*, so a variant that contributes "
                    "nothing costs a retrieval round and changes no ranking."
                )


# --------------------------------------------------------------------------------------
# Tool-call result cards (user story 13, T5 #9)
# --------------------------------------------------------------------------------------


def render_tool_cards(cards: tuple[FinanceCard, ...]) -> None:
    """One card per finance-tool call this turn made, in call order.

    Rendered from the turn's own cards, never re-fetched — the same rule the sources panel
    follows, and here it bites harder: a second fetch goes through a TTL cache, so it could
    legitimately return a *different* price from the one the answer above quotes.

    Nothing at all for a turn that called no finance tool, so a pure-retrieval answer keeps the
    shape T4 had. Each card carries its own staleness banner rather than one banner for the
    group: a full brief can hold a fresh quote and a stale peer comparison, and a single banner
    would have to overstate one of them.
    """
    for card in cards:
        renderer = _RENDERERS.get(card.KIND)
        if renderer is None:
            # A kind this build has no renderer for — the one-rerun-after-deploy window every
            # reader on this path tolerates, since a thread checkpointed by a *newer* build can
            # replay here. Skipped, not raised; the answer text above it still stands.
            continue
        with st.container(border=True):
            _render_staleness(card)
            renderer(card)


def _render_staleness(card: FinanceCard) -> None:
    """User story 22's banner: the figures below are the last ones we could get, and their age.

    A `warning`, not a `caption`: the reader is about to take a number off this card into a
    note, and the fact that nobody could refresh it is the kind that has to interrupt.
    `FailedCard` has no freshness at all — there is no figure to be stale about — which is why
    this reads the attribute defensively rather than assuming every card has one.
    """
    freshness = getattr(card, "freshness", None)
    if freshness is None or not freshness.stale:
        return
    st.warning(
        f"Live data could not be refreshed. These figures are the last successful fetch, "
        f"{freshness.age_minutes} minute(s) old.",
        icon=":material/history:",
    )


def _render_quote(card: QuoteCard) -> None:
    """A price card: the figures a valuation question opens with, then the month's shape."""
    quote = card.quote
    st.markdown(f"**{quote.ticker} · {escaped(quote.name)}**")
    price, cap, multiple = st.columns(3)
    price.metric(
        f"Price ({quote.currency})",
        Unit.PRICE.format(quote.price),
        # `None`, not `"0.00%"`, when the change is unknown: `st.metric`'s delta arrow is a
        # claim about direction, and there is nothing to claim.
        None if quote.change_percent is None else f"{quote.change_percent:+.2f}%",
    )
    cap.metric("Market cap", Unit.MONEY.format(quote.market_cap))
    multiple.metric("P/E (trailing)", Unit.MULTIPLE.format(quote.trailing_pe))
    st.caption(
        f"52-week range {Unit.PRICE.format(quote.fifty_two_week_low)}–"
        f"{Unit.PRICE.format(quote.fifty_two_week_high)} · "
        f"EPS {Unit.PRICE.format(quote.trailing_eps)} · previous close "
        f"{Unit.PRICE.format(quote.previous_close)}"
    )
    if quote.closes:
        # **`st.altair_chart` and not `st.line_chart`, and only for the y-axis.** Vega-Lite
        # includes zero in a quantitative axis' domain by default, and `st.line_chart` — which
        # is sugar over *this* call, its own docstring says so — offers no way to say
        # otherwise. So a month of closes between 215 and 232 drew as a flat line two-thirds of
        # the way up an axis running 0 → 100 → 200: the chart reporting "nothing happened"
        # about the move a reader opened the card to see. `zero=False` scales it to the data.
        #
        # Everything else is `st.line_chart`'s own generated spec, reproduced rather than
        # improved on, so this stays the change it says it is: the same nominal x over the ISO
        # dates (`Close.date` is a string), the same gridlines, the same `close` axis title,
        # and a hover tooltip in place of the one that came free.
        history = pd.DataFrame(
            {
                "date": [close.date for close in quote.closes],
                "close": [close.close for close in quote.closes],
            }
        )
        st.altair_chart(
            alt.Chart(history)
            .mark_line()
            .encode(
                x=alt.X("date:N", axis=alt.Axis(grid=False), title=""),
                y=alt.Y("close:Q", axis=alt.Axis(grid=True), scale=alt.Scale(zero=False)),
                tooltip=["date", "close"],
            ),
            height=180,
        )
        # `HISTORY_PERIOD_LABEL`, not "last month" typed again. That constant exists because the
        # window was already prose in the tool description and `"1mo"` in `quotes.py` — and this
        # caption was a third copy (issue #9 review). The session count stays derived from the
        # data rather than from the label: `"1mo"` yields ~21 trading sessions, not 30, and the
        # exact number is a property of the response.
        #
        # **The axis clause is the price of scaling to the data.** Two lines above this chart
        # the card states a 52-week range, and an axis that no longer starts at zero starts
        # wherever *this month* does — so the numbers running up the side are a month's
        # extremes sitting directly under a year's, with nothing but this sentence to say they
        # are different windows. A reader who takes the axis for the 52-week range reads a
        # month of noise as a year of it.
        st.caption(
            f"Daily closes, {HISTORY_PERIOD_LABEL} ({len(quote.closes)} sessions) — the axis "
            f"spans this window, not the 52-week range above."
        )


#: How many metric charts sit side by side before wrapping to a new row. Three keeps a
#: two-bar chart wide enough to read on a laptop and fits the five metrics into two rows.
_METRIC_CHART_COLUMNS = 3


def _render_ratios(card: RatiosCard) -> None:
    """A peer-comparison card: ADR-0009's basis, then each metric, then the bars."""
    comparison = card.comparison
    st.markdown(f"**{comparison.ticker} · {escaped(comparison.name)}**")
    st.caption(comparison.basis)
    if note := comparison.unavailable_note:
        # Named, not silently absent from `n`: a peer whose quote failed is a gap in the *data*,
        # where a peer with no figure for one metric is a fact about that company. The wording
        # is `PeerComparison`'s because it has to branch on whether *every* peer failed, and
        # this surface and the tool text were phrasing that branch separately (#9 review).
        st.warning(note, icon=":material/link_off:")
    for metric in comparison.metrics:
        coverage = metric.coverage_note(comparison.n)
        st.markdown(
            f"**{metric.label}** · {metric.versus_peers}{f' — {coverage}' if coverage else ''}"
        )
    _render_metric_bars(card)


def _render_metric_bars(card: RatiosCard) -> None:
    """Company against peer mean — **one chart per metric, each on its own axis.**

    Only the metrics where **both** numbers exist: a bar chart cannot draw "not reported", and a
    missing figure plotted as zero is the one mistake this whole path is built to avoid. The
    metric is still listed in the prose above, with its absence stated.

    **Why one chart per metric and not one per `Unit`.** Splitting by unit was already necessary
    — a P/E of 31.6 beside a gross margin of 0.74 draws the margin as a flat line — but it is
    not sufficient, because two metrics can share a unit and still not share a scale.
    `Unit.MULTIPLE` holds both P/E and debt-to-equity, and in the Universe those differ by two
    orders of
    magnitude: Ford's peer mean P/E is 162× (TSLA at 286×, GM at 37×) while its D/E is 4.26×. On
    one axis the D/E bars are three pixels tall, and the leverage of a company with $159bn of
    debt reads as zero — on the card T11 screenshots (issue #9 review).

    A log axis was the alternative and is worse: bar *length* encodes magnitude, so log-scaled
    bars misstate every ratio a reader takes off them, and `st.bar_chart` has no log scale to
    offer anyway. Normalising to "percent of peer mean" was the other, and it throws away the
    figures — on an equity-research card the actual multiple is the thing being reported. One
    axis per metric keeps every bar at true scale and costs only layout, which `st.columns`
    absorbs.
    """
    plottable = [
        metric
        for metric in card.comparison.metrics
        if metric.value is not None and metric.peer_mean is not None
    ]
    if not plottable:
        return
    for start in range(0, len(plottable), _METRIC_CHART_COLUMNS):
        row = plottable[start : start + _METRIC_CHART_COLUMNS]
        # Always `_METRIC_CHART_COLUMNS` columns, even for a short final row: passing
        # `len(row)` would stretch a lone chart across the full width and make the last metric
        # look like the important one.
        columns = st.columns(_METRIC_CHART_COLUMNS)
        for column, metric in zip(columns, row, strict=False):
            with column:
                _render_one_metric_bar(card.comparison.ticker, metric)


def _render_one_metric_bar(ticker: str, metric: Metric) -> None:
    """One metric's two bars — the company and its peer mean — at that metric's own scale."""
    # The scale and the axis label are `Unit`'s, not this function's: the surface was switching
    # on the enum three times over, and a chart plotted at one scale under a caption naming
    # another is a mistake nothing would catch (issue #9 review).
    scale = metric.unit.chart_scale
    st.bar_chart(
        pd.DataFrame(
            {metric.label: [metric.value * scale, metric.peer_mean * scale]},
            index=[ticker, "peer mean"],
        ),
        height=200,
    )
    st.caption(f"{metric.label} · {metric.unit.axis_label.lower()}")


def _render_news(card: NewsCard) -> None:
    """News cards: title, publisher, date, and the stripped summary.

    Every string here was written by a stranger, so every string here is escaped (`escaped`) and
    the link has already been scheme-checked at the boundary (`finance.news.safe_link`). The
    summary goes through `st.text` for the reason a source body does — it is quoted text, and it
    renders character-identical.
    """
    st.markdown(f"**Recent headlines · {card.ticker}**")
    st.caption(f"Last {card.days} day(s), {len(card.headlines)} headline(s).")
    if not card.headlines:
        # A fact, not a failure — the same distinction the tool's own text draws for the model.
        st.info(
            f"The feed answered and had nothing for {card.ticker} in that window.",
            icon=":material/newspaper:",
        )
        return
    for headline in card.headlines:
        title = escaped(headline.title)
        st.markdown(f"[{title}]({headline.link})" if headline.link else f"**{title}**")
        when = "" if headline.published is None else f" · {headline.published[:10]}"
        st.caption(f"{escaped(headline.source)}{when}")
        if headline.summary:
            st.text(headline.summary)


def _render_failure(card: FailedCard) -> None:
    """A tool call that produced no data, and why.

    The message is the model-facing one, reused rather than rewritten: the analyst reading this
    banner and the model writing the answer above it should be explaining the same failure, and
    a second wording is a second explanation to keep in step.
    """
    st.warning(card.message, icon=":material/cloud_off:")


#: card kind -> how to draw it. **The UI's one card-kind dispatch**, keyed on the same `KIND`
#: strings `tools/finance.py` builds `_CARDS` and `_TOOL_BY_KIND` from.
#:
#: This was an `isinstance` cascade, the third of three independent enumerations of the card
#: kinds — so adding a fourth card meant remembering three edits across two files with nothing
#: to catch a missed one (issue #9 review). A map keyed on the kind means the UI's list is
#: checkable against the engine's, which `tests/test_app_smoke.py` does: a card kind with no
#: renderer here fails there rather than rendering an empty bordered box in front of a reader.
#:
#: Declared below the functions rather than beside `render_tool_cards`, because a dict of names
#: is evaluated at import: referencing them above their definitions is a `NameError` at startup.
_RENDERERS = {
    QuoteCard.KIND: _render_quote,
    RatiosCard.KIND: _render_ratios,
    NewsCard.KIND: _render_news,
    FailedCard.KIND: _render_failure,
}


def render_sources(contexts: tuple[Context, ...], *, searched: bool) -> None:
    """The sources panel: what each inline `[n]` in the answer above resolves to.

    Rendered for every assistant turn, replayed turns included — a citation whose source
    vanishes on the next rerun cannot be checked, which is the whole point of showing it
    (user story 3). No panel when nothing was retrieved: an empty panel reads as "grounded
    in nothing in particular" rather than "not grounded".

    What replaces it is a setup banner, because "nothing retrieved" has exactly one cause
    here. A populated Chroma always returns `k` chunks, so an empty result means an empty or
    misdirected collection — never "nothing relevant" (see `prompts.NO_CONTEXT_FALLBACK`).
    The fallback text alone reads as "your question was out of scope" and sends a reviewer
    who simply has not ingested yet looking for a retrieval bug (issue #5 review).

    `searched` is what keeps that banner honest now the agent decides whether to retrieve
    (T4): a turn answered from the conversation — "summarise that" — has no contexts and no
    problem, and firing the un-ingested banner over it would send a reviewer to re-run a paid
    ingest because a follow-up worked. Nothing at all is rendered for such a turn: the answer
    above it makes no citations to check.
    """
    if not searched:
        return
    if not contexts:
        st.warning(
            "Nothing was retrieved. A populated collection always returns top-k, so the "
            f"`filings` collection at `{settings.chroma_dir}` is empty or is not the one "
            "ingest wrote. Build it with `uv run python scripts/ingest_filings.py` — see "
            "the README's *Building the knowledge base*."
        )
        return
    # The icon rides in the label rather than in `icon=`, which would render this as a
    # `status` block and put the panel out of `AppTest.expander`'s reach (seam 3).
    with st.expander(f":material/description: Sources ({len(contexts)})"):
        for context in contexts:
            # The link rides on the citation line rather than in the caption below it: this is
            # the line a reader reads to decide whether to trust the excerpt, and "where to
            # check it" belongs beside "what it is". The caption under it is machine detail.
            st.markdown(f"**[{context.rank}] {context.citation}** · {accession_label(context)}")
            st.caption(
                f"`{context.chunk_id}` · {distance_label(context.distance)} · "
                f"{retriever_label(context)}"
            )
            # `st.text`, not a Markdown blockquote: the body is the filer's own words, and
            # this is the surface a reader checks a citation against, so it must render
            # character-identical. `st.markdown` does not — it parses `$…$` as KaTeX, so
            # Apple's own segment line `Americas$178,353 7 %$167,045` renders as prose plus
            # a maths expression, and a blockquote silently ends at the filing's first
            # blank line. Losing the quote styling is the cheaper trade.
            st.text(context.body)


def render_context_reuse_note(*, used_tools: bool) -> None:
    """Say so when a turn answered from the conversation instead of calling anything.

    **The absence this exists to stop being silent.** A follow-up like "summarise that" or a
    second "give me the full brief" in a warm thread is answered from what the thread already
    retrieved — correct context reuse, and the behaviour ADR-0008's checkpointed history is for.
    But it renders as an answer with no cards, no sources panel and no "how I answered", which
    is *pixel-identical* to a turn whose tools all failed. A reader cannot tell "nothing needed
    fetching" from "nothing could be fetched", and CLAUDE.md's rule is that an absence must not
    be reported as a measurement: "we cannot say what this query found" is a different claim
    from "this query found nothing".

    It was recorded as a decision on this ticket — warm-thread briefs make zero tool calls and
    are captioned on the turn — and `AgentTurn.used_tools` was added for it, but nothing ever
    read that property outside the tests (issue #9 review). This is the reader.

    Deliberately a caption and not a banner: reuse is the *correct* path, so it is a note about
    provenance rather than a warning about a problem. The failure cases already have banners of
    their own, from `render_sources` and `tools/finance.py`.
    """
    if used_tools:
        return
    st.caption(
        ":material/history: Answered from context already retrieved in this conversation; "
        "no new search, so no new sources to cite."
    )


def issued_ranks() -> set[int]:
    """Every citation number this conversation has handed out, from the display transcript.

    **The valid set for a marker is the thread's, not the turn's**, and getting that wrong would
    make the note below fire on correct answers. `agent/citations.py` numbers a thread's sources
    in one running sequence, so a follow-up's first source is `[4]`; and a follow-up may
    legitimately cite `[2]` from a source the *previous* turn retrieved, since the conversation
    is in the checkpointer. Validated against `reply.contexts` alone — which `AgentTurn` scopes
    to this turn — every such citation would read as unresolved, and the caption would tell an
    analyst that a perfectly good citation points nowhere.

    Read off the transcript rather than accumulated in a counter, because the transcript is
    already the display source of truth and "Start over" clears it in one place. `getattr` on
    `contexts` for the reason every reader on this path is tolerant: a row written by an older
    shape replays here on the first rerun after a deploy.
    """
    return {
        context.rank
        for message in st.session_state.messages
        if message["role"] == "assistant"
        for context in getattr(message.get("turn"), "contexts", ())
    }


def render_marker_note(report: MarkerReport) -> None:
    """Say which of the answer's `[n]` markers a reader cannot resolve (T3's finding, #5).

    Nothing at all when every bracket resolves, which is the common case.

    **Reported, not repaired** — `security/markers.py` argues that choice. What matters here is
    that the two failures are described apart, because they are different things to a reader: an
    unresolved number is a citation pointing at no panel entry, and a non-numeric bracket is the
    `[Yahoo Finance]` T5 measured, which collides with the syntax that makes any citation
    resolvable. A caption merging them would leave the reader unsure which of their `[n]`s to
    distrust.

    A warning rather than a caption, unlike `render_context_reuse_note`: context reuse is the
    correct path, and this is the answer telling the reader to check something they cannot.
    """
    if report.clean:
        return
    problems = []
    if report.unresolved:
        numbers = ", ".join(f"[{number}]" for number in report.unresolved)
        problems.append(
            f"{numbers} — cited above but not in the sources panel, so there is nothing to "
            f"check {'them' if len(report.unresolved) > 1 else 'it'} against"
        )
    if report.non_numeric:
        # `escaped()` and **not** a code span. Inside backticks Markdown is already inert, so
        # the two together printed the backslashes: `[Reuters (2026)]` reached the reader as
        # `[Reuters \(2026\)]`, and a span containing a backtick closed the span early and let
        # model-written text restructure the warning (issue #8 review). One neutralisation, and
        # it is the one that works on arbitrary text.
        spans = ", ".join(f"[{escaped(span)}]" for span in report.non_numeric)
        problems.append(
            f"{spans} — square brackets are reserved for numbered filing excerpts, so this "
            f"is a publisher's name where a citation should be"
        )
    st.warning(
        "Some markers in this answer do not resolve: " + "; ".join(problems) + ".",
        icon=":material/link_off:",
    )


#: How many example buttons sit side by side. Two, because the labels are whole sentences:
#: four across truncates every one of them on a laptop, and one per row pushes the first
#: answer below the fold on the only screen where these are visible at all.
_EXAMPLE_COLUMNS = 2


def render_example_questions() -> None:
    """The empty page's four starting points (T12 item 3, PLAN §2's *Interactive help / guide*).

    **No user story, and the citation says so rather than borrowing one.** This shipped citing
    "user story 20", which is *"a fresh browser session starts a clean conversation"* — nothing
    to do with example questions (code review of #13). `docs/spec/finbrief.md` has no story for
    onboarding at all; it puts a help guide in **Out of Scope**, and PLAN §2's Easy tail is the
    only thing asking for this. A pointer into the spec that lands on the wrong line is worse
    than no pointer, because the next reader checks the line rather than the claim.

    **Shown only while the transcript is empty**, which is what keeps them an affordance rather
    than furniture: they answer "what do I type", and that stops being the reader's question the
    moment there is an answer on screen to read. Left up, four buttons would push every
    subsequent answer down the page for the whole conversation.

    `st.chat_input` cannot be given a value from code, so "seeding the input" is seeding the
    **turn**: the click records the question in `session_state` and the block below consumes it
    exactly where a typed question is consumed. That is one code path on purpose — a seeded
    question therefore gets the length cap, the input gate, the transcript row and the panels,
    and not a second thinner version of the turn that quietly skips one of them. The gate
    especially: a seeding route that bypassed `screen()` would be a second door into the agent,
    and it is the door worth trying precisely because it looks like UI convenience
    (`test_a_seeded_question_is_screened_by_the_gate_like_any_other`).

    Nothing here touches the cached agent. ADR-0008's isolation guarantee rests on that
    instance being built once and shared, so "start me off" must mean a `session_state` write
    and never a rebuild — which would discard every *other* session's memory, the same failure
    the "Start over" button is written around.
    """
    st.caption("New here? Start with one of these — or just type a question.")
    for start in range(0, len(EXAMPLE_QUESTIONS), _EXAMPLE_COLUMNS):
        row = EXAMPLE_QUESTIONS[start : start + _EXAMPLE_COLUMNS]
        # Always `_EXAMPLE_COLUMNS` columns, even for a short final row — the same reason
        # `_render_metric_bars` does it: passing `len(row)` would stretch a lone button across
        # the full width and make the last example look like the recommended one.
        columns = st.columns(_EXAMPLE_COLUMNS)
        for column, question in zip(columns, row, strict=False):
            if column.button(question, width="stretch"):
                # Through the constant, not `.pending_question`: the sidebar reads this key
                # now (`answering_now`), and a key written under one spelling and read under
                # another is a placeholder that stays blank on exactly the run it is for.
                st.session_state[PENDING_QUESTION_KEY] = question


# **An `st.empty()` slot rather than a plain render, and the reason is a one-frame wart.** The
# buttons have to be *drawn* above the transcript, because a click is read from the widget on
# the rerun that follows it — so `st.button` must be called before the question it seeded is
# consumed below. But at that point in the script the transcript is still empty on exactly the
# run where the click is being handled, so the naive version leaves four "New here?" buttons
# sitting above the reader's own first answer until some later rerun clears them.
#
# A placeholder separates the two: the buttons are rendered into it (so the click is still
# read — verified, not assumed) and the slot is emptied again below once this run turns out to
# have a conversation in it. `test_the_examples_make_way_for_the_conversation` is the assertion.
examples_slot = st.empty()
if not st.session_state.messages:
    with examples_slot.container():
        render_example_questions()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # **The status box is replayed, and the reason is an index and not a decoration.**
        # Streamlit addresses a container's children by position, so a row whose live render
        # opened with `st.status` and whose replay does not shifts every index by one — and the
        # element the shift leaves unclaimed is the row's *last*, the disclaimer. Measured
        # through `AppTest`'s own child map: `(Status, Markdown, Expander, Expander, Caption)`
        # live against `(Markdown, Expander, Expander, Caption)` replayed, and the browser then
        # showed the disclaimer twice for the whole of the next turn — the second copy faded,
        # because it belongs to the previous run and is pruned when this one ends.
        #
        # So the replay renders the box too. `.get`, because a row written before this shape
        # carries no label and must replay rather than raise (the tolerance every reader on this
        # path has), and because a **gate-blocked** row legitimately has none: it never reached
        # the agent, so it had no status box live either and must not grow one here.
        #
        # No body: the step log is not kept past the turn, and inventing one would be a claim
        # about which tools ran. The label is the fact worth replaying — that the turn finished.
        if (completed := message.get("status")) is not None:
            st.status(completed, state="complete", expanded=False)
        st.markdown(as_markdown(message["content"]))
        if message["role"] == "assistant":
            # `.get` returning `None`, not a default, because a live session's transcript
            # outlives a code reload: rows written by an older shape are still in
            # `session_state` on the first rerun after a deploy and must replay rather than
            # `KeyError` the page (`test_a_transcript_row_from_an_older_shape_replays`). The
            # absences are different facts and only one is a setup problem — "this row predates
            # the current shape" must not raise `render_sources`' empty-collection banner over
            # an answer that was grounded when it was written. That the current shape *does*
            # carry its sources across a rerun — the regression this tolerance could otherwise
            # hide — is `test_the_sources_panel_survives_the_next_turn`.
            if (turn := message.get("turn")) is not None:
                # `AgentTurn` is asked for both facts rather than the row carrying them
                # separately: it is the one authority on "which chunks, in which order" and on
                # "was the knowledge base consulted at all", and a row that stored its own copy
                # could disagree with the panel below it.
                #
                # `.cards` through `getattr`, because a row written before T5 holds an
                # `AgentTurn` with no such field — the same one-rerun-after-deploy window every
                # `from_payload` on this path tolerates, and why the field is defaulted. Bound
                # to a local rather than read twice: `turn.used_tools` would reach the same
                # missing attribute without the tolerance, so the note below is told the fact
                # instead of asked to derive it.
                cards = getattr(turn, "cards", ())
                render_tool_cards(cards)
                render_sources(turn.contexts, searched=turn.searched)
                render_how_i_answered(turn.searches)
                render_context_reuse_note(used_tools=bool(turn.searches or cards))
            elif (contexts := message.get("contexts")) is not None:
                # The T4 row shape, kept readable for the same one-rerun window. `searched`
                # defaults to True because before the agent existed contexts were always
                # retrieved, so an empty tuple in such a row really does mean an empty
                # collection.
                render_sources(contexts, searched=message.get("searched", True))
            st.caption(DISCLAIMER)

# `key=QUESTION_KEY` so the sidebar can see a submitted question before this line runs — see the
# constant. The value is still taken from the return here and not from `session_state`: this is
# where the question is consumed, and one consumer is the rule the seeded half is built on too.
typed = st.chat_input(
    "Ask about a company in the Universe", key=QUESTION_KEY, submit_mode="disable"
)

# **Popped, not read** (T12 item 3). A seeded question left in `session_state` would be re-asked
# on every rerun the page does for any other reason — a widget change, a panel opening — so one
# click would bill a question per interaction. Consuming it here, *before* the turn runs, means
# even a turn that raises or is refused cannot leave it behind to fire again
# (`test_a_seeded_question_is_asked_once_and_not_again_on_the_next_rerun`).
#
# A typed question wins if both arrive in one run, which cannot currently happen — a click and
# a submit are separate reruns — but the tie has to break somewhere, and the reader's own words
# are the half that is unambiguous about what they meant.
seeded = st.session_state.pop(PENDING_QUESTION_KEY, None)

prompt = typed or seeded


def answer_turn(prompt: str) -> None:
    """One turn: screen the question, answer it, and leave the result in the transcript.

    **A function with `return`s where this was a top-level block with `st.stop()`s.** The reason
    is the export slot and not tidiness: `st.stop()` does not merely end the script, it stops
    Streamlit accepting further elements, so anything rendered afterwards is silently discarded
    — measured, with a `finally` writing into a placeholder after an `st.stop()` and producing
    nothing at all. The export buttons have to be built from the transcript *including* this
    turn, or a reader downloads a file missing the exchange they just had and cannot tell.

    So the turn needs an exit the script survives. Three `st.stop()`s became three `return`s and
    nothing else changed: this block was the last thing in the file, so ending it and ending the
    script were the same act until there was something to render after it.
    """
    # **One turn, one identifier, on every event this block emits** (T8, #10). The gate's
    # screening, the retrievals the agent's tool ran, the validator's verdict and the marker
    # check all land on separate lines with nothing else in common: a `retrieval` line carries
    # per-chunk provenance and, deliberately, no question. Without this the only correlation
    # available to T10 (#11) is position in the file, which is wrong the moment the model asks
    # for two searches in one step and asserted by nothing either way.
    #
    # `thread_id` first so a whole conversation's lines can be cut out together, then a fresh
    # suffix because `thread_id` is per *conversation* — reusing it alone would collapse every
    # turn of a thread into one bucket. Derived from a UUID and the session id, never from what
    # was typed: these lines are kept.
    with log_turn(f"{st.session_state.thread_id}:{uuid.uuid4().hex[:8]}"):
        # The input-validation half of user story 21, and the only door a human types through.
        # Checked before the question is appended to the transcript, so an over-long paste does
        # not become part of a conversation nothing will answer — and before the agent, because
        # the cheapest refusal is the one that costs no tokens. What arrives at this length is a
        # paste rather than a question, often a document with instructions in it, which is
        # ADR-0006's problem and cheaper to refuse here than to classify.
        if len(prompt) > MAX_QUESTION_CHARS:
            st.error(
                f"That question is {len(prompt):,} characters, and FinBrief takes at most "
                f"{MAX_QUESTION_CHARS:,}. Ask a shorter question — or paste the part you "
                f"actually want an answer about.",
                icon=":material/text_fields:",
            )
            return

        # **The per-session throttle (T12 item 6) — cost and abuse limiting, not security.**
        # `config.MAX_QUESTIONS_PER_SESSION` carries the argument; the short version is that a
        # refresh resets this counter, so it bounds what one open tab can spend and stops
        # nothing that is trying. ADR-0006's gate below is the security boundary, and conflating
        # the two would be the more dangerous mistake in the pair — a reviewer who reads this as
        # rate limiting stops looking for the thing that is.
        #
        # **After the length cap and before the gate**, which is where the two reasons agree.
        # Cost: layer 3 is a paid classifier call, so a throttled question must not reach it.
        # Abuse: a payload that the denylist would block still consumes a question, because
        # otherwise the one caller worth throttling is the one that gets unlimited attempts.
        if st.session_state.questions_asked >= MAX_QUESTIONS_PER_SESSION:
            st.warning(
                f"This session has asked its {MAX_QUESTIONS_PER_SESSION} questions. Refresh "
                f"the page to start a new conversation — FinBrief caps questions per session "
                f"to bound what a shared demo key can spend.",
                icon=":material/hourglass_disabled:",
            )
            return
        st.session_state.questions_asked += 1

        # **The input gate (ADR-0006 layers 1–3), here and not in the agent.** This is the door
        # a human types through, which is what the front door is about; a gate inside the agent
        # loop would also screen the *model's* tool arguments as if an analyst had typed them,
        # and a gate inside `rag.answer_question` would put a model call in front of the
        # measured chain ADR-0003 keeps clean. `screen` never raises — a classifier outage fails
        # open onto the other three layers — so there is no branch here for the gate itself
        # failing.
        screening = screen(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(as_markdown(prompt))

        if screening.blocked:
            # The refusal is rendered as an ordinary assistant turn and stored as one, so it
            # replays identically and reads exactly like the persona's own refusal.
            # **Deliberately not labelled as a gate hit**: which layer fired and which pattern
            # matched are in the gate-trigger log for a reviewer, and an attacker told which
            # rule they tripped is an attacker told how to phrase the next attempt.
            #
            # It never reaches `answer()`, so the payload never enters the checkpointer — the
            # display transcript holds a turn the agent has no memory of, which is the intended
            # asymmetry: a follow-up cannot build on a question that was refused.
            with st.chat_message("assistant"):
                st.markdown(as_markdown(INJECTION_REFUSAL))
                st.caption(DISCLAIMER)
            st.session_state.messages.append(
                {"role": "assistant", "content": INJECTION_REFUSAL}
            )
            return

        with st.chat_message("assistant"):
            try:
                # A `status` rather than a spinner, because with four tools the wait has *parts*
                # and naming them is user story 14. Each step is written into the container as
                # the model asks for the call, so the list persists for the whole wait instead
                # of one label replacing another — and the reader can see that a full brief
                # really did fetch four things. Still starts at "Thinking…": the agent decides
                # whether to call anything, so a label naming retrieval would describe a step
                # some turns skip.
                with st.status("Thinking…", expanded=True) as status:

                    def note(step: Step) -> None:
                        label = step_label(step)
                        status.update(label=label)
                        st.write(label)

                    # **One read of the picker, used twice** (T14, #15): the agent is fetched
                    # for this model and the same slug is what gets logged onto `agent_turn`.
                    # Reading `chosen_model()` twice here would be two reads of one widget in
                    # one run — identical today, and the shape that lets a logged attribution
                    # disagree with the model that answered the moment anything between them
                    # touches `session_state`.
                    model = chosen_model()
                    reply = answer(
                        prompt,
                        thread_id=st.session_state.thread_id,
                        agent=shared_agent(model),
                        model=model,
                        on_step=note,
                    )
                    status.update(label=TURN_COMPLETE, state="complete", expanded=False)
            except GraphRecursionError:
                # The **generation tier** of PLAN §2's tiered handling: a failure of the
                # answering loop itself rather than of a data source. The agent ran out of
                # steps, which reads to a user as the app hanging and then dying — so it gets
                # its own message naming the cause and the action, where the generic branch
                # below would print LangGraph's own "Recursion limit of N reached" with
                # `MAX_AGENT_STEPS` in place of N.
                status.update(label="Gave up", state="error", expanded=False)
                st.error(
                    "That question took more tool calls than FinBrief allows in one turn. "
                    "Ask it in two parts — the filings half first, then the figures — or "
                    "name a company.",
                    icon=":material/repeat_on:",
                )
            # Broad by intent — the last resort, and it names no internals (`noqa: BLE001`).
            # The prose sits here rather than inline because the reindent this block needed put
            # the inline version over 96 characters, and shortening it silently dropped a word
            # (issue #10 review): a line long enough to need rewrapping gets rewrapped, never
            # trimmed.
            except Exception as exc:  # noqa: BLE001
                # The exception *type*, not its message. A client error string can carry a
                # request URL, and a request URL can carry an API key — the same reason
                # `log_event` never records one (`observability/logging_setup.py`). The detail
                # goes to the log, where it is already structured; the reader gets something
                # actionable instead.
                #
                # Through `log_event` rather than `logger.exception`, which was this codebase's
                # only bypass of the single emitter — and so the only line carrying no
                # `turn_id`, on precisely the turn whose provenance a reader wants. The
                # traceback still travels, in the envelope's `error`.
                log_event(
                    logger,
                    "chat_turn_failed",
                    level=logging.ERROR,
                    exc_info=True,
                    error_type=type(exc).__name__,
                )
                status.update(label="Failed", state="error", expanded=False)
                st.error(
                    f"FinBrief could not answer that ({type(exc).__name__}). Try again — and "
                    f"if it keeps happening, the server log has the detail.",
                    icon=":material/error:",
                )
            else:
                # **The output validator (ADR-0006 layer 4).** The one layer whose input the
                # attacker does not choose: an ordinary question can be answered with a
                # recommendation nobody asked for, and a *successful* indirect injection —
                # arriving through retrieved text that never passed the front door — shows up
                # here or nowhere (user story 15, 17).
                advice = validate_answer(reply.text)
                if advice.refused:
                    # The refusal replaces the answer **and its panels**. Nothing else is
                    # rendered: the cards and the sources panel are the provenance *of an
                    # answer*, and there is no answer — a sources panel under a refusal invites
                    # a reader to think the refusal was grounded in them.
                    #
                    # A stated consequence, not a hidden one: `answer()` has already run, so the
                    # text this refuses is in the checkpointer and a follow-up can reference it.
                    # This layer guards the surface, not the memory. Recorded in ADR-0006's T7
                    # amendment and in the README's limitations.
                    st.markdown(as_markdown(ADVICE_REFUSAL))
                    st.caption(DISCLAIMER)
                    st.session_state.messages.append(
                        # `status`: this row *had* a status box, so its replay needs one at the
                        # same index — see the transcript loop. The label is the one the box
                        # ended on, not a description of the refusal.
                        {
                            "role": "assistant",
                            "content": ADVICE_REFUSAL,
                            "status": TURN_COMPLETE,
                        }
                    )
                    return

                st.markdown(as_markdown(reply.text))
                # The citation-marker check (T3's finding, #5): every `[n]` against every number
                # this *conversation* has issued, not just this turn's — see `issued_ranks`.
                # Logged on every turn that renders an answer rather than only on a violation,
                # so T10 (#11) has a denominator for the rate. **Not every turn**: the layer-4
                # branch above `return`s first, so a refused answer contributes to neither
                # numerator nor denominator — which is the right denominator anyway, since the
                # markers of an answer no reader saw are not a marker-resolution rate about
                # anything.
                report = markers(
                    reply.text, ranks=issued_ranks() | {c.rank for c in reply.contexts}
                )
                log_markers(
                    report, thread_id=st.session_state.thread_id, sources=len(reply.contexts)
                )
                render_marker_note(report)
                render_tool_cards(reply.cards)
                render_sources(reply.contexts, searched=reply.searched)
                render_how_i_answered(reply.searches)
                # `used_tools` direct here, unlike the replay path above: this object was built
                # by *this* process, so it cannot predate the current shape.
                render_context_reuse_note(used_tools=reply.used_tools)
                st.caption(DISCLAIMER)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": reply.text,
                        # As above: the box's label, so the replay puts an element at the index
                        # the live render used.
                        "status": TURN_COMPLETE,
                        # The whole turn, because the panels need four facts about it and two of
                        # them — the query variants each search ran, and the tool cards — belong
                        # to the call rather than to any chunk. Display data, not memory: the
                        # checkpointer remains the agent's memory of record (ADR-0008), and
                        # nothing here is ever read back into a conversation.
                        "turn": reply,
                    }
                )


if prompt:
    # Retracted here rather than skipped above: this run has a question in it, so the empty
    # page's affordance is no longer describing this page. See the slot's own comment.
    examples_slot.empty()
    answer_turn(prompt)

# **Last, and that ordering is the point.** Every branch of `answer_turn` has run by now,
# including the two refusals, so the transcript this reads is the one the reader is looking at.
# Nothing at all when there is no conversation: a download button offering a file with no turns
# in it reads as a broken feature rather than as an empty one.
if st.session_state.messages:
    # **One slot here, unlike the spend panel's two, and the difference is arithmetic.** A
    # second container written to a placeholder merges by child index, so what matters is
    # whether the settled fill is at least as long as the eager one: here it is one caption
    # against a caption and two buttons, so every index the eager fill wrote is overwritten.
    # The spend panel could not rest on that — its length varies with the log it reads — so it
    # splits the slot instead (`fill_spend_meter`). Clearing this one first was tried and
    # removed: it does not help, because the write that follows lands on the same path.
    with export_slot.container():
        render_export_buttons(st.session_state.messages)

# Also last, and for the same reason: this run's `agent_turn` line is written inside the turn
# above, so a meter built in the sidebar would report the spend as of the *previous* question.
# **The second of two fills, not the only one** — the sidebar filled this slot on the way past
# so the panel is on screen for the wait, and this replaces it with the settled figures.
# `answering` is `False` here whatever this run did: the turn is over, and its total is logged.
# The eager panel's job is over the moment this one exists, and it is *cleared* rather than
# overwritten: an `Empty` at a path nothing writes to again is the one placeholder operation
# measured to remove what it held (`examples_slot`, and `fill_spend_meter`'s own record of the
# fix that assumed more than that).
spend_eager_slot.empty()
fill_spend_meter(spend_slot, answering=False)
