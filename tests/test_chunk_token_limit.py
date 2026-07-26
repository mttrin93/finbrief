"""Enforces the assumption behind `check_embedding_ctx_length=False`.

`retrieval/embeddings.py` disables LangChain's length-safe embedding path, which means
nothing splits an over-long text before it is sent: a chunk above the model's input
window would fail the request outright. The justification is that chunking keeps every
text far below that window — so it is asserted here rather than left as prose.

The chunk size comes from `config.CHUNK_SIZE_CHARS`, not a copy — so when Tier-2 tunes it
there, this guarantee is re-checked against the real value rather than a stale duplicate.

**Tier-2: repoint `chunks_under_test()` at the real chunker's output** over ingested
Sections (`ingestion/chunking.py`). The placeholder below stands in only while no
chunker exists; `assert_embeddable()` is the assertion that carries over unchanged, so
the switch is a one-function edit.
"""

import pytest
import tiktoken

from finbrief.config import CHUNK_SIZE_CHARS

#: `openai/text-embedding-3-small` accepts 8191 tokens per input and tokenises with
#: cl100k_base.
EMBEDDING_CONTEXT_LIMIT = 8191

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def assert_embeddable(chunks: list[str]) -> None:
    """The assertion Tier-2 reuses against real chunks: every chunk fits in one request."""
    oversized = {
        index: count_tokens(chunk)
        for index, chunk in enumerate(chunks)
        if count_tokens(chunk) >= EMBEDDING_CONTEXT_LIMIT
    }
    assert not oversized, (
        f"chunks {sorted(oversized)} exceed the {EMBEDDING_CONTEXT_LIMIT}-token embedding "
        f"window (tokens: {oversized}). Either re-enable check_embedding_ctx_length or "
        f"lower config.CHUNK_SIZE_CHARS."
    )


def _filled_to_chunk_size(text: str) -> str:
    """Repeat `text` to exactly `CHUNK_SIZE_CHARS`, whatever that value becomes."""
    return (text * (CHUNK_SIZE_CHARS // len(text) + 1))[:CHUNK_SIZE_CHARS]


def chunks_under_test() -> list[str]:
    """PLACEHOLDER (Tier-1) — filing-shaped text at exactly the configured chunk size.

    TODO(Tier-2): return the real chunker's output over ingested Sections instead
    (`ingestion/chunking.py`). Sized off `config.CHUNK_SIZE_CHARS` so that until then a
    change to that constant still produces worst-case-sized inputs here.
    """
    prose = (
        "Item 1A. Risk Factors. Our business is subject to numerous risks, including "
        "supply-chain disruption, component shortages, regulatory changes across the "
        "jurisdictions in which we operate, and fluctuations in demand for our products. "
    )
    # Dense identifier text is the realistic worst case for tokens-per-character:
    # tickers, item numbers, and figures tokenise far less efficiently than prose.
    identifiers = (
        "TSLA F GM AAPL 10-K Item 7A P/E 42.7x D/E 1.83 FY2025 §13(a) ISIN US88160R1014 "
    )
    return [
        _filled_to_chunk_size(prose),
        _filled_to_chunk_size(identifiers),
        prose[:200],  # a short tail chunk, as a real section boundary would produce
    ]


def test_chunks_fit_the_embedding_context():
    assert_embeddable(chunks_under_test())
    # Every placeholder is at the cap, so the check above is not passing on short text.
    assert max(len(chunk) for chunk in chunks_under_test()) == CHUNK_SIZE_CHARS


def test_a_chunk_can_never_exceed_the_window_at_this_chunk_size():
    """The structural argument, not just the sample: tokens <= characters, always.

    cl100k_base never emits more tokens than there are characters, so a chunk capped at
    `CHUNK_SIZE_CHARS` characters cannot reach the token window while that cap stays
    below it. This is what makes the placeholder above adequate for Tier-1 — and the final
    assertion is what fires if Tier-2 ever tunes the chunk size past the ceiling.
    """
    for chunk in chunks_under_test():
        assert count_tokens(chunk) <= len(chunk)
    assert CHUNK_SIZE_CHARS < EMBEDDING_CONTEXT_LIMIT


def test_the_assertion_has_teeth():
    """A guard that cannot fail guards nothing."""
    oversized = "risk " * (EMBEDDING_CONTEXT_LIMIT + 1)
    with pytest.raises(AssertionError, match="exceed the 8191-token embedding window"):
        assert_embeddable([oversized])
