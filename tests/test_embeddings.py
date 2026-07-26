"""The shared embedding model. No network calls.

The property that matters is that ingest and query cannot drift: one constructor, and
what it returns is bound to the configured model and OpenRouter.
"""

from finbrief.config import Settings
from finbrief.retrieval.embeddings import build_embeddings

SETTINGS = Settings.from_env({"OPENROUTER_API_KEY": "sk-test"})


def test_embeddings_are_bound_to_openrouter():
    embeddings = build_embeddings(SETTINGS)
    assert embeddings.model == "openai/text-embedding-3-small"
    assert embeddings.openai_api_base == "https://openrouter.ai/api/v1"
    assert embeddings.openai_api_key.get_secret_value() == "sk-test"


def test_embeddings_send_strings_not_token_arrays():
    # OpenRouter documents `input` as a string or an array of strings; LangChain's
    # length-safe path would send pre-tokenised integers instead. The chunk-size
    # assumption this relies on is enforced in test_chunk_token_limit.py.
    assert build_embeddings(SETTINGS).check_embedding_ctx_length is False


def test_the_configured_model_is_the_one_built(monkeypatch):
    # The property that protects ingest/query parity is that this constructor reads the
    # configured model — one config value, so both callers cannot disagree. Building it
    # twice and comparing would pass whatever the constructor did with the setting.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-environ")
    monkeypatch.setenv("FINBRIEF_EMBEDDING_MODEL", "openai/text-embedding-3-large")

    # No settings argument: the production path, through `get_settings()`.
    embeddings = build_embeddings()

    assert embeddings.model == "openai/text-embedding-3-large"
    assert embeddings.openai_api_key.get_secret_value() == "sk-from-environ"
