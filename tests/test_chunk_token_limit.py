"""Enforces the assumption behind `check_embedding_ctx_length=False`.

`retrieval/embeddings.py` disables LangChain's length-safe embedding path, which means
nothing splits an over-long text before it is sent: a chunk above the model's input
window would fail the request outright. The justification is that the chunk size keeps
every text far below that window — so it is asserted here rather than left as prose.

The chunk size comes from `config.CHUNK_SIZE_CHARS`, not a copy — so when ticket T2 (KB
ingest, #3) tunes it there, this guarantee is re-checked against the real value rather than
a stale duplicate.

The bound is deliberately tokeniser-free. Counting real tokens would mean downloading
cl100k_base's BPE table at import time, and the suite is hermetic by contract (no network
— see `conftest.py`, `.github/workflows/ci.yml`). It would also be the *wrong* argument:
cl100k_base is byte-level BPE, so tokens can exceed characters — `ﬁ` and `🙂` are one
character and two tokens each, `≥≤±` is three characters and five tokens. What holds
unconditionally is the byte bound below.

**Ticket T2 (KB ingest, #3)**, when `ingestion/chunking.py` lands: add a test that counts
real tokens over the chunker's actual output. That measures how much of the margin is
really used, which this cannot; this stays as the guard on the constant itself.
"""

from finbrief.config import CHUNK_SIZE_CHARS

#: `openai/text-embedding-3-small` accepts 8191 tokens per input.
EMBEDDING_CONTEXT_LIMIT = 8191

#: Worst-case tokens per character for a byte-level BPE tokeniser such as cl100k_base: a
#: character is at most 4 UTF-8 bytes, and the byte-fallback vocabulary means a byte is at
#: most one token. Nothing tokenises worse than this, whatever the text — filing prose,
#: dense identifiers, CJK, or emoji.
MAX_TOKENS_PER_CHAR = 4


def test_a_chunk_can_never_exceed_the_embedding_window():
    """The chunk size, at its worst case, still fits one embedding request."""
    worst_case_tokens = CHUNK_SIZE_CHARS * MAX_TOKENS_PER_CHAR

    assert worst_case_tokens <= EMBEDDING_CONTEXT_LIMIT, (
        f"CHUNK_SIZE_CHARS={CHUNK_SIZE_CHARS} allows a chunk of up to {worst_case_tokens} "
        f"tokens, over the {EMBEDDING_CONTEXT_LIMIT}-token embedding window. Either lower "
        f"config.CHUNK_SIZE_CHARS (max "
        f"{EMBEDDING_CONTEXT_LIMIT // MAX_TOKENS_PER_CHAR}) or re-enable "
        f"check_embedding_ctx_length in retrieval/embeddings.py."
    )
