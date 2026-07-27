"""The agent entrypoint the UI calls.

**Baseline RAG (ticket T3, #5).** `answer()` runs the deterministic chain in `finbrief.rag`
— retrieve, then generate with inline citations. There is no agent loop yet: Phase 3
replaces this body with `create_agent` plus a `SqliteSaver` checkpointer, binds
`search_filings` / `get_stock_data` / `calculate_ratios` / `get_recent_news`, and grows the
signature by `thread_id`. This module stays the single seam the UI depends on, which is the
whole reason it is a thin function and not the chain itself.

The chain deliberately lives *outside* here. ADR-0003 separates the deterministic
`(question → contexts → answer)` path — what the RAGAs and A/B numbers measure — from the
agent's nondeterministic use of it, so `finbrief.rag` must stay callable with no agent in
the way. When Phase 3 arrives, `rag.answer_question` does not move; this function does.

No conversation memory here on purpose. ADR-0008 makes the checkpointer the memory of
record and `st.session_state` explicitly *not* the agent's memory — so this stays stateless
rather than teaching the UI a history-passing habit Phase 3 would have to undo.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from finbrief.config import RetrievalStrategy
from finbrief.rag import GroundedAnswer, answer_question

#: The strategy the shipped path runs in Phase 2, and the reason it is a constant here
#: rather than `settings.retrieval_strategy`: `config.DEFAULT_STRATEGY` is already `hybrid`,
#: pre-registered before any A/B data exists (ADR-0005), and hybrid does not exist until
#: Phase 4 (`retrieve` raises for it, deliberately). Honouring the setting today would make
#: the app's first question fail; ignoring it silently would let the sidebar advertise a
#: strategy that never ran. So the baseline is named, and the UI reports it *next to* the
#: configured value rather than in place of it. Phase 4 deletes this constant and reads the
#: setting.
BASELINE_STRATEGY = RetrievalStrategy.VECTOR


def answer(question: str, *, model: BaseChatModel | None = None) -> GroundedAnswer:
    """Answer one question from the knowledge base, with the contexts that grounded it.

    Raises whatever the model or the store raises; the caller renders the failure (tiered
    error handling lands in Phase 5).
    """
    return answer_question(question, strategy=BASELINE_STRATEGY, model=model)
