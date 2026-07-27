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

import uuid

import streamlit as st

from finbrief.agent.agent import Search, answer, build_agent
from finbrief.config import (
    CLUSTERS,
    PEERS,
    UNIVERSE,
    ConfigError,
    RetrievalStrategy,
    get_settings,
)
from finbrief.observability.logging_setup import configure_logging
from finbrief.prompts import DISCLAIMER, GROUNDING_SCOPE, GROUNDING_SCOPE_DETAILS
from finbrief.retrieval.retrieve import Context

st.set_page_config(page_title="FinBrief", page_icon=":material/query_stats:")

st.title("FinBrief")
st.caption(GROUNDING_SCOPE)

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
    names = {"vector": "vector", "bm25": "BM25"}
    found = [names.get(retriever.value, retriever.value) for retriever in context.retrievers]
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
            if search.translated and not search.sub_queries:
                st.caption(
                    "The query planner added nothing — the question was already one specific "
                    "thing a filing answers, so no sub-query was retrieved."
                )
            elif not search.translated:
                st.caption("Query translation was off, so only the question itself was run.")
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
                render_sources(turn.contexts, searched=turn.searched)
                render_how_i_answered(turn.searches)
            elif (contexts := message.get("contexts")) is not None:
                # The T4 row shape, kept readable for the same one-rerun window. `searched`
                # defaults to True because before the agent existed contexts were always
                # retrieved, so an empty tuple in such a row really does mean an empty
                # collection.
                render_sources(contexts, searched=message.get("searched", True))
            st.caption(DISCLAIMER)

if prompt := st.chat_input("Ask about a company in the Universe", submit_mode="disable"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(as_markdown(prompt))

    with st.chat_message("assistant"):
        try:
            # "Thinking", not "Searching the filings": the agent decides whether to search,
            # so a spinner that names retrieval would be describing a step some turns skip.
            # The per-tool progress this replaces it with lands with the tool cards in T5.
            with st.spinner("Thinking…"):
                reply = answer(
                    prompt, thread_id=st.session_state.thread_id, agent=shared_agent()
                )
        except Exception as exc:  # noqa: BLE001 — tiered error handling lands in Phase 5
            st.error(f"The model call failed: {exc}", icon=":material/error:")
        else:
            st.markdown(as_markdown(reply.text))
            render_sources(reply.contexts, searched=reply.searched)
            render_how_i_answered(reply.searches)
            st.caption(DISCLAIMER)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": reply.text,
                    # The whole turn, because the panels need three facts about it and one of
                    # them — the query variants each search ran — belongs to the search rather
                    # than to any chunk. Display data, not memory: the checkpointer remains the
                    # agent's memory of record (ADR-0008), and nothing here is ever read back
                    # into a conversation.
                    "turn": reply,
                }
            )
