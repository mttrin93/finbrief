"""Shared LLM-client infrastructure: chat models bound to OpenRouter.

OpenRouter is OpenAI-compatible, so `ChatOpenAI` pointed at its base URL is the whole
integration. This lives at the package root, not under `agent/`, because most callers
are not the agent: query translation (Phase 4) and the injection classifier (Phase 5,
its own prompt and potentially its own cheaper model) need a chat model too, and neither
belongs to the agent loop.

Chat models only. The embedding model has its own constructor in
`retrieval/embeddings.py`, which ingest and query must share.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from finbrief.config import (
    ANSWER_MAX_RETRIES,
    ANSWER_TIMEOUT_SECONDS,
    Settings,
    get_settings,
)

#: OpenRouter attributes traffic by these headers; they show up in its dashboard.
_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/TuringCollegeSubmissions/mrinal-AE.AFA.3.5",
    "X-Title": "FinBrief",
}


def build_chat_model(
    settings: Settings | None = None,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    timeout: float = ANSWER_TIMEOUT_SECONDS,
    max_retries: int = ANSWER_MAX_RETRIES,
) -> ChatOpenAI:
    """Build a chat model bound to OpenRouter.

    Temperature defaults to 0: the evaluation harness needs deterministic runs
    (ADR-0003), and the app has no reason to want less reproducible answers.

    `timeout` and `max_retries` are arguments rather than fixed because the input gate's model
    call sits inside a latency budget the answering path does not have
    (`config.GATE_LATENCY_BUDGET_MS`, ADR-0006 T7 amendment §2):
    60 seconds and two retries would let one slow classifier call hold a turn for three minutes
    in front of the refusal it was deciding about. The defaults are the answering path's, so
    every existing caller is unchanged; `security/classifier.py` is the one that narrows them.
    """
    settings = settings or get_settings()
    return ChatOpenAI(
        model=model or settings.chat_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        default_headers=_OPENROUTER_HEADERS,
    )
