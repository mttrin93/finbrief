"""The agent entrypoint the UI calls.

**Walking skeleton (T1).** `answer()` is one stateless OpenRouter round-trip: no
retrieval, no tools, no memory. Phase 3 replaces the body with `create_agent` plus a
`SqliteSaver` checkpointer and grows the signature by `thread_id`, keeping this module as
the single seam the UI depends on.

No conversation memory here on purpose. ADR-0008 makes the checkpointer the memory of
record and `st.session_state` explicitly *not* the agent's memory — so the skeleton stays
stateless rather than teaching the UI a history-passing habit Phase 3 would have to undo.
"""

from __future__ import annotations

import logging
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from finbrief.llm import build_chat_model
from finbrief.observability.logging_setup import log_event

logger = logging.getLogger(__name__)

#: Placeholder persona. The real domain prompt — analyst voice, financial vocabulary,
#: grounding-scope disclosure, refusal policy, mandatory disclaimer — is Phase 2/5 work
#: and lands in `prompts.py`.
SKELETON_SYSTEM_PROMPT = (
    "You are FinBrief, an equity-research assistant for a junior analyst. "
    "Answer concisely and factually. You have no access to filings, market data, or news "
    "yet, so say plainly when you cannot ground an answer, and never give personalised "
    "investment advice."
)


def answer(question: str, *, model: BaseChatModel | None = None) -> str:
    """Answer one question. Raises on API failure; the caller renders the error."""
    chat = model if model is not None else build_chat_model()
    started = time.perf_counter()
    reply = chat.invoke([SystemMessage(SKELETON_SYSTEM_PROMPT), HumanMessage(question)])
    # Phase 6 extends this line with strategy config, retrieval hits, and token counts.
    log_event(
        logger,
        "chat_turn",
        latency_ms=round((time.perf_counter() - started) * 1000),
        question_chars=len(question),
    )
    return reply.text
