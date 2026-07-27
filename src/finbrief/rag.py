"""The deterministic RAG chain: question → contexts → grounded answer (ADR-0003).

This is the *measured* path. ADR-0003 splits the retrieval engine from the agent's use of
it precisely so the headline RAGAs numbers can be produced from clean
`(question, contexts, answer)` triples, question by question, with no tool loop in the way
— and this function is what produces one. Phase 3's agent will call `retrieve()` through
`search_filings` and generate its own answers; this chain stays, because the evaluation
harness drives it.

Deliberately stateless and not agent-owned: no conversation history, no tools, no memory
(ADR-0008 makes the checkpointer the memory of record, and it arrives with the agent).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from finbrief.config import RetrievalStrategy, Settings
from finbrief.llm import build_chat_model
from finbrief.observability.logging_setup import log_event
from finbrief.prompts import NO_CONTEXT_FALLBACK, SYSTEM_PROMPT, user_message
from finbrief.retrieval.retrieve import Context, retrieve

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """One turn's answer together with what grounded it.

    The contexts travel *with* the answer rather than being fetched again for display: the
    sources panel must show the chunks this answer's `[n]` markers point at, and a second
    retrieval — even with the same question and `k` — is a second chance to disagree.
    """

    text: str
    contexts: tuple[Context, ...]

    @property
    def grounded(self) -> bool:
        """Whether the answer came from retrieved filing text at all."""
        return bool(self.contexts)


def answer_question(
    question: str,
    *,
    strategy: RetrievalStrategy = RetrievalStrategy.VECTOR,
    k: int | None = None,
    store: Chroma | None = None,
    settings: Settings | None = None,
    model: BaseChatModel | None = None,
) -> GroundedAnswer:
    """Answer `question` from the knowledge base, with inline citations.

    Deterministic: `retrieve()` is deterministic and the chat model is built at
    temperature 0, so the same question and configuration produce the same triple.

    The question reaches `retrieve()` verbatim — no rewriting here. Query translation lives
    *inside* the retrieval engine (ADR-0004), so a caller that pre-translated would
    translate twice, and the measured chain and the shipped chain would part ways.

    Raises whatever the model client raises; the caller renders the failure (Phase 5 adds
    the tiered handling).
    """
    started = time.perf_counter()
    contexts = retrieve(question, strategy=strategy, k=k, store=store, settings=settings)
    if not contexts:
        # Emptiness, not irrelevance — a populated collection always returns `k`. The
        # relevance floor that would widen this branch is deferred to the Phase 4/7 A/B
        # evidence on purpose; see `prompts.NO_CONTEXT_FALLBACK` for why, and
        # `docs/verification/retrieval-smoke.md` for the distances it would be chosen from.
        text = NO_CONTEXT_FALLBACK
    else:
        chat = model if model is not None else build_chat_model()
        reply = chat.invoke(
            [
                SystemMessage(SYSTEM_PROMPT),
                HumanMessage(user_message(question, contexts)),
            ]
        )
        text = reply.text
    log_event(
        logger,
        "rag_answer",
        strategy=strategy.value,
        # `k` is deliberately absent: it is a *request*, and this layer may be holding
        # `None` for "whatever `settings.retrieval_k` says". The `retrieval` event logs the
        # resolved value, so reporting it again here could only disagree with it.
        contexts=len(contexts),
        grounded=bool(contexts),
        # Sizes and provenance, never the text of the question or the answer: these lines
        # are kept, and a question is user content.
        question_chars=len(question),
        answer_chars=len(text),
        cited_sections=sorted({context.section.value for context in contexts}),
        tickers=sorted({context.ticker for context in contexts}),
        latency_ms=round((time.perf_counter() - started) * 1000),
    )
    return GroundedAnswer(text=text, contexts=contexts)
