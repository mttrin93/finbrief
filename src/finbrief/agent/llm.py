"""Chat-model construction against OpenRouter.

OpenRouter is OpenAI-compatible, so `ChatOpenAI` pointed at its base URL is the whole
integration. Kept as its own seam because more than one caller needs a model: the agent
(Phase 3), the query-translation step (Phase 4), and the injection classifier (Phase 5,
its own prompt and potentially its own cheap model).

Chat models only. The embedding model lives in `retrieval/embeddings.py`, which ingest
and query must share.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from finbrief.config import Settings, get_settings

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
) -> ChatOpenAI:
    """Build a chat model bound to OpenRouter.

    Temperature defaults to 0: the evaluation harness needs deterministic runs
    (ADR-0003), and the app has no reason to want less reproducible answers.
    """
    settings = settings or get_settings()
    return ChatOpenAI(
        model=model or settings.chat_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=temperature,
        timeout=60,
        max_retries=2,
        default_headers=_OPENROUTER_HEADERS,
    )
