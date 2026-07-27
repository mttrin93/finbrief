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

from finbrief.agent.agent import BASELINE_STRATEGY, answer, build_agent
from finbrief.config import CLUSTERS, PEERS, UNIVERSE, ConfigError, get_settings
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
    st.markdown(
        f"**Model** `{settings.chat_model}`  \n"
        f"**Strategy** `{BASELINE_STRATEGY}`  \n"
        f"**Top-k** `{settings.retrieval_k}`"
    )
    # Never let the panel advertise a configuration that did not run: `hybrid +
    # translation` is the pre-registered shipping default (ADR-0005) and arrives in Phase 4.
    # Translation is checked as well as strategy, because the two switches move
    # independently: `retrieve()` tells an operator to set
    # `FINBRIEF_RETRIEVAL_STRATEGY=vector`, and doing exactly that leaves
    # `FINBRIEF_QUERY_TRANSLATION` at its default `True` — so gating on the strategy alone
    # left the one configuration we recommend as the only one that ran silently.
    if settings.retrieval_strategy != BASELINE_STRATEGY or settings.query_translation_enabled:
        st.caption(
            f"Configured strategy `{settings.retrieval_strategy}"
            f"{' + translation' if settings.query_translation_enabled else ''}` "
            f"lands in Phase 4. Answers below ran `{BASELINE_STRATEGY}`."
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
            st.caption(f"`{context.chunk_id}` · distance {context.distance:.4f}")
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
            # `.get` returning `None`, not `()`, because a live session's transcript
            # outlives a code reload: rows written by an older shape would otherwise
            # `KeyError` on the first rerun after a deploy
            # (`test_a_transcript_row_from_an_older_shape_replays`). The two absences are
            # different facts and only one is a setup problem — "this row predates
            # contexts" must not raise `render_sources`' empty-collection banner over an
            # answer that was grounded when it was written. That the current shape *does*
            # carry its contexts across a rerun — the regression this tolerance could
            # otherwise hide — is `test_the_sources_panel_survives_the_next_turn`.
            if (contexts := message.get("contexts")) is not None:
                # `searched` defaults to True so a row written before the agent existed
                # replays exactly as it did: back then contexts were always retrieved, so an
                # empty tuple in an old row really does mean the collection was empty.
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
            st.caption(DISCLAIMER)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": reply.text,
                    "contexts": reply.contexts,
                    # Kept so a replayed turn can tell "answered from the conversation" from
                    # "the collection returned nothing" — the two look identical without it,
                    # and only one of them is a setup problem.
                    "searched": reply.searched,
                }
            )
