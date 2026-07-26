"""The one embedding model, shared by ingest and query.

An index is only searchable by the model that wrote it: embed at ingest with one model
and query with another and retrieval silently degrades to noise — no error, just bad
numbers feeding the RAGAs/A-B tables. So there is exactly one constructor here, and both
`ingestion/` and `retrieval/` call it. Do not build `OpenAIEmbeddings` anywhere else.

Served by OpenRouter's OpenAI-compatible `/v1/embeddings` endpoint, through the same
base URL and key as the chat model (verified July 2026 — see CONTEXT.md).
"""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from finbrief.config import Settings, get_settings


def build_embeddings(settings: Settings | None = None) -> OpenAIEmbeddings:
    """Build the embedding model for both ingest and query."""
    settings = settings or get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        # OpenRouter documents `input` as a string or an array of strings. LangChain's
        # default length-safe path instead sends pre-tokenised integer arrays, which is
        # an OpenAI-specific extension — so it is disabled here and raw strings are
        # sent. Nothing splits an over-long text once this is off, so the 8191-token
        # ceiling rests on `config.CHUNK_SIZE_CHARS`, which
        # tests/test_chunk_token_limit.py holds under that window in the worst case.
        check_embedding_ctx_length=False,
    )
