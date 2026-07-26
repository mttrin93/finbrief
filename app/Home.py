"""FinBrief chat UI — walking skeleton (Tier-1).

UI only: every non-Streamlit concern lives in `finbrief.*` so it can be tested without
driving the app. Per ADR-0008 `st.session_state` holds only UI state — here, the display
transcript. The `thread_id` and the cached agent arrive in Phase 3 together with the
checkpointer that gives them a purpose.
"""

import streamlit as st

from finbrief.agent.agent import answer
from finbrief.config import PEERS, UNIVERSE, ConfigError, PeerCluster, get_settings
from finbrief.observability.logging_setup import configure_logging

st.set_page_config(page_title="FinBrief", page_icon=":material/query_stats:")


@st.cache_resource
def _setup_logging() -> None:
    configure_logging()


_setup_logging()

st.title("FinBrief")
st.caption(
    "Walking skeleton — a plain LLM round-trip via OpenRouter. Filing-grounded answers, "
    "citations, and the finance tools arrive in later phases."
)

# Fail here rather than on the first message: a missing key should be obvious before the
# user has typed anything.
try:
    settings = get_settings()
except ConfigError as exc:
    st.error(f"Configuration problem: {exc}", icon=":material/error:")
    st.stop()

with st.sidebar:
    st.subheader("Configuration")
    st.markdown(
        f"**Model** `{settings.chat_model}`  \n"
        f"**Strategy** `{settings.retrieval_strategy}"
        f"{' + translation' if settings.query_translation_enabled else ''}`  \n"
        f"**Top-k** `{settings.retrieval_k}`"
    )
    st.caption("Retrieval settings are declared but not yet wired up (Phase 2 onward).")

    st.subheader("Universe")
    st.caption(f"{len(UNIVERSE)} companies in {len(PeerCluster)} peer clusters.")
    for cluster in PeerCluster:
        tickers = [c.ticker for c in UNIVERSE if c.cluster is cluster]
        st.markdown(f"**{cluster.value.replace('_', ' ').title()}** — {', '.join(tickers)}")
    st.caption(f"Peers come only from this set, e.g. TSLA vs. {', '.join(PEERS['TSLA'])}.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a company", submit_mode="disable"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):
                reply = answer(prompt)
        except Exception as exc:  # noqa: BLE001 — tiered error handling lands in Phase 5
            st.error(f"The model call failed: {exc}", icon=":material/error:")
        else:
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
