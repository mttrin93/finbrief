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
import uuid

import pandas as pd
import streamlit as st
from langgraph.errors import GraphRecursionError

from finbrief.agent.agent import Search, Step, answer, build_agent
from finbrief.config import (
    CLUSTERS,
    HISTORY_PERIOD_LABEL,
    MAX_QUESTION_CHARS,
    PEERS,
    UNIVERSE,
    ConfigError,
    RetrievalStrategy,
    get_settings,
)
from finbrief.finance.ratios import Metric, Unit
from finbrief.observability.logging_setup import configure_logging
from finbrief.prompts import (
    ADVICE_REFUSAL,
    DISCLAIMER,
    GROUNDING_SCOPE,
    GROUNDING_SCOPE_DETAILS,
    INJECTION_REFUSAL,
    LIVE_DATA_SCOPE,
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
st.caption(GROUNDING_SCOPE)
# The second half of the scope, and it earns its own line rather than being appended to the one
# above: `GROUNDING_SCOPE` is quoted by the *chain*'s prompt too, where there are no tools, so
# the two sentences cannot be one string (`prompts.py`).
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
def shared_agent():
    """The one agent and the one checkpointer this process has (ADR-0008).

    Cached across reruns *and* across sessions, which is the whole design: a checkpointer
    built per rerun remembers nothing, and one built per session would put every user's
    conversation in its own file for no benefit. Isolation comes from `thread_id` instead.

    Not wrapped in the configuration banner's `try`: this runs after `get_settings()` has
    already succeeded, so the failures left here (an unwritable checkpoint path, say) are not
    configuration problems a banner could explain, and hiding them would mean a chat input
    that silently cannot answer.
    """
    return build_agent()


# One id per session, minted before the first message so every turn in this browser tab lands
# on the same thread — and no other tab's. `setdefault` rather than an `if`: the point is that
# a rerun must not mint a second one, and this is the shortest statement of that.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:
    st.subheader("Conversation")
    st.caption(
        "FinBrief remembers this conversation, so you can ask follow-ups — *and its debt?* "
        "resolves against the company you were just discussing. Memory lasts as long as this "
        "browser session: refreshing the page starts a new conversation."
    )
    # Shown, and shown short, because it is the handle on the conversation: it is what
    # distinguishes this tab's memory from another's, and a support question about a lost
    # thread has nothing else to name. Not a secret — a uuid identifies a conversation and
    # says nothing about who is having it.
    st.caption(f"Thread `{st.session_state.thread_id[:8]}`")
    # A fresh uuid, not a cleared checkpointer: the old thread is orphaned rather than deleted
    # (nothing else can reach it), and the cached agent survives — rebuilding it here would
    # discard every *other* session's memory too, which is the bug this button looks like.
    if st.button("Start over", icon=":material/restart_alt:"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []

    st.subheader("Grounding scope")
    for detail in GROUNDING_SCOPE_DETAILS:
        st.markdown(f"- {detail}")

    st.subheader("Configuration")
    # One value, not two. Until Phase 4 this panel named a `BASELINE_STRATEGY` constant and
    # captioned the gap to the configured one, because the pre-registered default (ADR-0005)
    # was a strategy `retrieve()` refused. Now the configured strategy *is* what answers, so a
    # second line would be a gap that no longer exists — and the way this panel stays honest is
    # that `agent.build_agent` reads these same two settings (nothing here restates them).
    st.markdown(
        f"**Model** `{settings.chat_model}`  \n"
        f"**Strategy** `{settings.retrieval_strategy}"
        f"{' + translation' if settings.query_translation_enabled else ''}`  \n"
        f"**Top-k** `{settings.retrieval_k}`"
    )
    st.caption(
        "Every answer's *How I answered* panel shows the queries that ran and which "
        "retriever surfaced each chunk (ADR-0004)."
        if settings.retrieval_strategy is RetrievalStrategy.HYBRID
        else "Vector search only. Hybrid retrieval adds BM25 over the same query variants."
    )

    st.subheader("Universe")
    st.caption(f"{len(UNIVERSE)} companies in {len(CLUSTERS)} peer clusters.")
    for cluster, tickers in CLUSTERS.items():
        st.markdown(f"**{cluster.label}** — {', '.join(tickers)}")
    # The thinnest cluster makes the crispest example, and picking it from the data keeps
    # this panel entirely config-driven — a hardcoded ticker would be a KeyError the day
    # the Universe changed.
    example = min(UNIVERSE, key=lambda company: len(PEERS[company.ticker]))
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
                st.caption(
                    "The ticker form is added deterministically from the Universe, not by a "
                    "model: a chunk's header carries `TSLA`, so a lexical search for *Tesla* "
                    "would miss most of the filer (ADR-0004 amendment)."
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
                st.caption(
                    f"Surfaced no chunk in the top-{len(search.contexts)}: {named}. Each ran "
                    "through every retriever this strategy uses; nothing they found survived "
                    "fusion. Translation only ever *adds*, so a variant that contributes "
                    "nothing costs a retrieval round and changes no ranking (ADR-0004)."
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
        st.line_chart(
            pd.DataFrame(
                {"close": [close.close for close in quote.closes]},
                index=[close.date for close in quote.closes],
            ),
            y="close",
            height=180,
        )
        # `HISTORY_PERIOD_LABEL`, not "last month" typed again. That constant exists because the
        # window was already prose in the tool description and `"1mo"` in `quotes.py` — and this
        # caption was a third copy (issue #9 review). The session count stays derived from the
        # data rather than from the label: `"1mo"` yields ~21 trading sessions, not 30, and the
        # exact number is a property of the response.
        st.caption(f"Daily closes, {HISTORY_PERIOD_LABEL} ({len(quote.closes)} sessions).")


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
            st.markdown(f"**[{context.rank}] {context.citation}**")
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
        spans = ", ".join(f"`[{escaped(span)}]`" for span in report.non_numeric)
        problems.append(
            f"{spans} — square brackets are reserved for numbered filing excerpts, so this "
            f"is a publisher's name where a citation should be"
        )
    st.warning(
        "Some markers in this answer do not resolve: " + "; ".join(problems) + ".",
        icon=":material/link_off:",
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
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

if prompt := st.chat_input("Ask about a company in the Universe", submit_mode="disable"):
    # The input-validation half of user story 21, and the only door a human types through.
    # Checked before the question is appended to the transcript, so an over-long paste does not
    # become part of a conversation nothing will answer — and before the agent, because the
    # cheapest refusal is the one that costs no tokens. What arrives at this length is a paste
    # rather than a question, often a document with instructions in it, which is ADR-0006's
    # problem and cheaper to refuse here than to classify.
    if len(prompt) > MAX_QUESTION_CHARS:
        st.error(
            f"That question is {len(prompt):,} characters, and FinBrief takes at most "
            f"{MAX_QUESTION_CHARS:,}. Ask a shorter question — or paste the part you actually "
            f"want an answer about.",
            icon=":material/text_fields:",
        )
        st.stop()

    # **The input gate (ADR-0006 layers 1–3), here and not in the agent.** This is the door a
    # human types through, which is what the front door is about; a gate inside the agent loop
    # would also screen the *model's* tool arguments as if an analyst had typed them, and a gate
    # inside `rag.answer_question` would put a model call in front of the measured chain
    # ADR-0003 keeps clean. `screen` never raises — a classifier outage fails open onto the
    # other three layers — so there is no branch here for the gate itself failing.
    screening = screen(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(as_markdown(prompt))

    if screening.blocked:
        # The refusal is rendered as an ordinary assistant turn and stored as one, so it replays
        # identically and reads exactly like the persona's own refusal. **Deliberately not
        # labelled as a gate hit**: which layer fired and which pattern matched are in the
        # gate-trigger log for a reviewer, and an attacker told which rule they tripped is an
        # attacker told how to phrase the next attempt.
        #
        # It never reaches `answer()`, so the payload never enters the checkpointer — the
        # display transcript holds a turn the agent has no memory of, which is the intended
        # asymmetry: a follow-up cannot build on a question that was refused.
        with st.chat_message("assistant"):
            st.markdown(as_markdown(INJECTION_REFUSAL))
            st.caption(DISCLAIMER)
        st.session_state.messages.append({"role": "assistant", "content": INJECTION_REFUSAL})
        st.stop()

    with st.chat_message("assistant"):
        try:
            # A `status` rather than a spinner, because with four tools the wait has *parts* and
            # naming them is user story 14. Each step is written into the container as the model
            # asks for the call, so the list persists for the whole wait instead of one label
            # replacing another — and the reader can see that a full brief really did fetch four
            # things. Still starts at "Thinking…": the agent decides whether to call anything,
            # so a label naming retrieval would describe a step some turns skip.
            with st.status("Thinking…", expanded=True) as status:

                def note(step: Step) -> None:
                    label = step_label(step)
                    status.update(label=label)
                    st.write(label)

                reply = answer(
                    prompt,
                    thread_id=st.session_state.thread_id,
                    agent=shared_agent(),
                    on_step=note,
                )
                status.update(label="Answered", state="complete", expanded=False)
        except GraphRecursionError:
            # The **generation tier** of PLAN §2's tiered handling: a failure of the answering
            # loop itself rather than of a data source. The agent ran out of steps, which reads
            # to a user as the app hanging and then dying — so it gets its own message naming
            # the cause and the action, where the generic branch below would print LangGraph's
            # own "Recursion limit of N reached" with `MAX_AGENT_STEPS` in place of N.
            status.update(label="Gave up", state="error", expanded=False)
            st.error(
                "That question took more tool calls than FinBrief allows in one turn. Ask it "
                "in two parts — the filings half first, then the figures — or name a company.",
                icon=":material/repeat_on:",
            )
        except Exception as exc:  # noqa: BLE001 — the last resort, and it names no internals
            # The exception *type*, not its message. A client error string can carry a request
            # URL, and a request URL can carry an API key — the same reason `log_event` never
            # records one (`observability/logging_setup.py`). The detail goes to the log, where
            # it is already structured; the reader gets something actionable instead.
            logger.exception("chat_turn_failed")
            status.update(label="Failed", state="error", expanded=False)
            st.error(
                f"FinBrief could not answer that ({type(exc).__name__}). Try again — and if it "
                f"keeps happening, the server log has the detail.",
                icon=":material/error:",
            )
        else:
            # **The output validator (ADR-0006 layer 4).** The one layer whose input the
            # attacker does not choose: an ordinary question can be answered with a
            # recommendation nobody asked for, and a *successful* indirect injection — arriving
            # through retrieved text that never passed the front door — shows up here or nowhere
            # (user story 15, 17).
            advice = validate_answer(reply.text)
            if advice.refused:
                # The refusal replaces the answer **and its panels**. Nothing else is rendered:
                # the cards and the sources panel are the provenance *of an answer*, and there
                # is no answer — a sources panel under a refusal invites a reader to think the
                # refusal was grounded in them.
                #
                # A stated consequence, not a hidden one: `answer()` has already run, so the
                # text this refuses is in the checkpointer and a follow-up can reference it.
                # This layer guards the surface, not the memory. Recorded in ADR-0006's T7
                # amendment and in the README's limitations.
                st.markdown(as_markdown(ADVICE_REFUSAL))
                st.caption(DISCLAIMER)
                st.session_state.messages.append(
                    {"role": "assistant", "content": ADVICE_REFUSAL}
                )
                st.stop()

            st.markdown(as_markdown(reply.text))
            # The citation-marker check (T3's finding, #5): every `[n]` against every number
            # this *conversation* has issued, not just this turn's — see `issued_ranks`. Logged
            # on every turn rather than only on a violation, so T10 (#11) has a denominator for
            # the rate.
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
            # `used_tools` direct here, unlike the replay path above: this object was built by
            # *this* process, so it cannot predate the current shape.
            render_context_reuse_note(used_tools=reply.used_tools)
            st.caption(DISCLAIMER)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": reply.text,
                    # The whole turn, because the panels need four facts about it and two of
                    # them — the query variants each search ran, and the tool cards — belong to
                    # the call rather than to any chunk. Display data, not memory: the
                    # checkpointer remains the agent's memory of record (ADR-0008), and nothing
                    # here is ever read back into a conversation.
                    "turn": reply,
                }
            )
