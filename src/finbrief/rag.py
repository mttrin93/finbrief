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
from finbrief.observability.tokens import usage_fields
from finbrief.prompts import NO_CONTEXT_FALLBACK, SYSTEM_PROMPT, user_message
from finbrief.retrieval.retrieve import Context, retrieve

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """One turn's answer together with what grounded it.

    The contexts travel *with* the answer rather than being fetched again for display: the
    sources panel must show the chunks this answer's `[n]` markers point at, and a second
    retrieval — even with the same question and `k` — is a second chance to disagree.

    **`text` carries no disclaimer, on purpose — so every surface that shows it owes one.**
    `prompts.DISCLAIMER` is rendered beside the answer by the caller (`app/Home.py` does),
    not requested from the model and not baked in here: a disclaimer the model is asked for
    goes missing on the turn that most needed it, and one baked into `text` would be scored
    by RAGAs faithfulness as an unsupported claim. The cost of that split is this contract,
    which Phase 3's agent and Phase 7's harness inherit (issue #5 review).
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
    translate: bool = False,
    k: int | None = None,
    store: Chroma | None = None,
    settings: Settings | None = None,
    model: BaseChatModel | None = None,
    translation_model: BaseChatModel | None = None,
) -> GroundedAnswer:
    """Answer `question` from the knowledge base, with inline citations.

    Deterministic: `retrieve()` is deterministic and the chat model is built at
    temperature 0, so the same question and configuration produce the same triple.

    The question reaches `retrieve()` verbatim — no rewriting here. Query translation lives
    *inside* the retrieval engine (ADR-0004), so a caller that pre-translated would
    translate twice, and the measured chain and the shipped chain would part ways. `translate`
    switches that engine-side step on; it is one of the four configurations the A/B compares
    (ADR-0002), and this chain is the seam it compares them through.

    `translation_model` is separate from `model` for the same reason `llm.py` sits at the
    package root: the planner and the answerer are two different prompts and could be two
    different (differently priced) models. Both are injectable for tests; production passes
    neither and `retrieve()` builds what it needs.

    Raises whatever the model client raises; the caller renders the failure (Phase 5 adds
    the tiered handling).
    """
    started = time.perf_counter()
    # Normalised here as well as inside `retrieve()`, because `retrieve()` normalises its own
    # local and this function keeps the caller's value: `retrieve` documents accepting a raw
    # string, so `answer_question(q, strategy="vector")` used to retrieve, pay for a
    # generation, and *then* die on `strategy.value` in the log line below (issue #5 review).
    strategy = RetrievalStrategy(strategy)
    contexts = retrieve(
        question,
        strategy=strategy,
        translate=translate,
        k=k,
        store=store,
        settings=settings,
        model=translation_model,
    ).contexts
    if not contexts:
        # Emptiness, not irrelevance — a populated collection always returns `k`. The
        # relevance floor that would widen this branch is deferred to the Phase 4/7 A/B
        # evidence on purpose; see `prompts.NO_CONTEXT_FALLBACK` for why, and
        # `docs/verification/retrieval-smoke.md` for the distances it would be chosen from.
        text = NO_CONTEXT_FALLBACK
        # No model was called, so there is no spend to report — and `{}` says that, where a
        # zeroed pair would claim a free generation happened (`observability/tokens.py`).
        usage: dict[str, int] = {}
    else:
        # `settings` reaches generation too, not just retrieval: a harness that points an
        # injected `Settings` at a throwaway index would otherwise still generate with the
        # process-global `chat_model`, so half of its configuration would silently not apply.
        chat = model if model is not None else build_chat_model(settings)
        reply = chat.invoke(
            [
                SystemMessage(SYSTEM_PROMPT),
                HumanMessage(user_message(question, contexts)),
            ]
        )
        text = reply.text
        usage = usage_fields(reply)
    log_event(
        logger,
        "rag_answer",
        strategy=strategy.value,
        translation=translate,
        # `k` is deliberately absent: it is a *request*, and this layer may be holding
        # `None` for "whatever `settings.retrieval_k` says". The `retrieval` event logs the
        # resolved value, so reporting it again here could only disagree with it.
        contexts=len(contexts),
        grounded=bool(contexts),
        # Sizes and provenance, never the text of the question or the answer: these lines
        # are kept, and a question is user content.
        question_chars=len(question),
        answer_chars=len(text),
        # `retrieved_`, not `cited_`: nothing here parses the answer's `[n]` markers, and a
        # Phase-7 log reader given `cited_sections` would report citation behaviour it never
        # measured (issue #5 review).
        retrieved_sections=sorted({context.section.value for context in contexts}),
        tickers=sorted({context.ticker for context in contexts}),
        latency_ms=round((time.perf_counter() - started) * 1000),
        # The generation half of the measured chain's spend (#10's AC-1). Spread rather than
        # passed as a pair, so a provider that reported nothing adds no keys — see
        # `observability/tokens.py` for why an unreported count must not read as zero. The
        # *retrieval* half is unmetered by construction: embeddings are priced per token too,
        # but the embeddings API returns no usage this code path can see.
        **usage,
    )
    return GroundedAnswer(text=text, contexts=contexts)
