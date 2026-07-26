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


def test_ingest_and_query_get_the_same_model():
    ingest, query = build_embeddings(SETTINGS), build_embeddings(SETTINGS)
    assert ingest.model == query.model
    assert (ingest.openai_api_base, ingest.dimensions) == (
        query.openai_api_base,
        query.dimensions,
    )
