"""FinBrief chat UI — baseline vector RAG (ticket T3, #5).

UI only: every non-Streamlit concern lives in `finbrief.*` so it can be tested without
driving the app, and every word about the knowledge base's shape comes from
`finbrief.prompts` — the same text the model is given, so the page and the persona cannot
disagree about what is grounded (user story 18, ADR-0007).

Per ADR-0008 `st.session_state` holds only UI state: here the display transcript, and with
each assistant turn the contexts its `[n]` markers point at. That is display data, not the
agent's memory — the `thread_id` and the checkpointer arrive in Phase 3.
"""

import streamlit as st

from finbrief.agent.agent import BASELINE_STRATEGY, answer
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

with st.sidebar:
    st.subheader("Grounding scope")
    for detail in GROUNDING_SCOPE_DETAILS:
        st.markdown(f"- {detail}")

    st.subheader("Configuration")
    st.markdown(
        f"**Model** `{settings.chat_model}`  \n"
        f"**Strategy** `{BASELINE_STRATEGY}`  \n"
        f"**Top-k** `{settings.retrieval_k}`"
    )
    if settings.retrieval_strategy != BASELINE_STRATEGY:
        # Never let the panel advertise a strategy that did not run: `hybrid + translation`
        # is the pre-registered shipping default (ADR-0005) and arrives in Phase 4.
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


def render_sources(contexts: tuple[Context, ...]) -> None:
    """The sources panel: what each inline `[n]` in the answer above resolves to.

    Rendered for every assistant turn, replayed turns included — a citation whose source
    vanishes on the next rerun cannot be checked, which is the whole point of showing it
    (user story 3). Skipped entirely when nothing was retrieved: an empty panel reads as
    "grounded in nothing in particular" rather than "not grounded".
    """
    if not contexts:
        return
    # The icon rides in the label rather than in `icon=`, which would render this as a
    # `status` block and put the panel out of `AppTest.expander`'s reach (seam 3).
    with st.expander(f":material/description: Sources ({len(contexts)})"):
        for context in contexts:
            st.markdown(f"**[{context.rank}] {context.citation}**")
            st.caption(f"`{context.chunk_id}` · distance {context.distance:.4f}")
            st.markdown(f"> {context.body}")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message["contexts"])
            st.caption(DISCLAIMER)

if prompt := st.chat_input("Ask about a company in the Universe", submit_mode="disable"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the filings…"):
                reply = answer(prompt)
        except Exception as exc:  # noqa: BLE001 — tiered error handling lands in Phase 5
            st.error(f"The model call failed: {exc}", icon=":material/error:")
        else:
            st.markdown(reply.text)
            render_sources(reply.contexts)
            st.caption(DISCLAIMER)
            st.session_state.messages.append(
                {"role": "assistant", "content": reply.text, "contexts": reply.contexts}
            )
