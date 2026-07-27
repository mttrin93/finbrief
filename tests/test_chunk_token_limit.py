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

**Ticket T2 (#3) delivered the second half.** `test_real_chunks_*` below counts real
cl100k_base tokens over the real chunker's output on a real recorded 10-K, which measures
how much of the margin is actually used — something the bound above cannot do. The two
answer different questions and both stay: the bound holds for *any* text including the
adversarial, the measurement shows what filing prose really costs.

Counting real tokens is compatible with the hermetic contract only because both halves are
vendored: the chunks come from a recorded filing in `tests/fixtures/edgar/`, and
cl100k_base's BPE table from `tests/fixtures/tiktoken/`. `tiktoken.get_encoding` does
*not* read the table from its wheel — it downloads it and caches it — so without the
vendored copy this file would quietly make a network call on any machine with a cold
cache. `conftest.offline_tiktoken` redirects the cache and asserts the table is there.
"""

import tiktoken

from finbrief.config import CHUNK_SIZE_CHARS
from finbrief.ingestion.chunking import chunk_filing
from finbrief.ingestion.model import Section

#: `openai/text-embedding-3-small` accepts 8191 tokens per input.
EMBEDDING_CONTEXT_LIMIT = 8191

#: Worst-case tokens per character for a byte-level BPE tokeniser such as cl100k_base: a
#: character is at most 4 UTF-8 bytes, and the byte-fallback vocabulary means a byte is at
#: most one token. Nothing tokenises worse than this, whatever the text — filing prose,
#: dense identifiers, CJK, or emoji.
MAX_TOKENS_PER_CHAR = 4

#: The longest provenance header `Chunk.text` can put in front of a body. What gets
#: embedded is header **plus** body (`token_counts` below measures exactly that string),
#: so a bound computed over the body alone would drift under the real request. Derived
#: from the real headings, with room for the longest ticker and an amended form.
MAX_HEADER_CHARS = max(
    len(f"WWWWW | FY99999 10-K/A | {section.value}. {section.heading}\n\n")
    for section in Section
)


def test_a_chunk_can_never_exceed_the_embedding_window():
    """The chunk size plus its provenance header, at worst case, fits one request."""
    worst_case_tokens = (CHUNK_SIZE_CHARS + MAX_HEADER_CHARS) * MAX_TOKENS_PER_CHAR

    assert worst_case_tokens <= EMBEDDING_CONTEXT_LIMIT, (
        f"CHUNK_SIZE_CHARS={CHUNK_SIZE_CHARS} plus a {MAX_HEADER_CHARS}-char provenance "
        f"header allows a chunk of up to {worst_case_tokens} tokens, over the "
        f"{EMBEDDING_CONTEXT_LIMIT}-token embedding window. Either lower "
        f"config.CHUNK_SIZE_CHARS (max "
        f"{EMBEDDING_CONTEXT_LIMIT // MAX_TOKENS_PER_CHAR - MAX_HEADER_CHARS}) or "
        f"re-enable check_embedding_ctx_length in retrieval/embeddings.py."
    )


def token_counts(filing) -> list[int]:
    """Real cl100k_base token counts for every chunk the chunker actually produces.

    Counts `chunk.text`, not `chunk.body` — the provenance header is part of what gets
    embedded, so leaving it out would measure a string the API never sees.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return [len(encoding.encode(chunk.text)) for chunk in chunk_filing(filing)]


def test_real_chunks_from_a_real_10k_fit_the_embedding_window(recorded_filing):
    """The measurement the bound above cannot make: what filing prose actually costs."""
    counts = token_counts(recorded_filing)

    assert counts, "the recorded filing should produce chunks"
    assert max(counts) <= EMBEDDING_CONTEXT_LIMIT, (
        f"a real chunk reached {max(counts)} tokens, over the "
        f"{EMBEDDING_CONTEXT_LIMIT}-token window"
    )


def test_real_filing_prose_uses_only_a_small_part_of_the_worst_case_bound(recorded_filing):
    """Real 10-K prose tokenises near 1 token per 4 characters, not the worst case of 4.

    Pinned as a test rather than left as a comment because it is the number that says
    whether `CHUNK_SIZE_CHARS` has room to grow. If a future tokeniser change or a filer
    with very different text pushed real usage toward the bound, raising the chunk size
    would stop being safe — and this is where that would surface, instead of in a failed
    embedding request mid-ingest.
    """
    counts = token_counts(recorded_filing)
    worst_case = CHUNK_SIZE_CHARS * MAX_TOKENS_PER_CHAR
    # The bound the docstring actually claims: near one token per four characters, so
    # roughly a quarter of `MAX_TOKENS_PER_CHAR`, with headroom for the provenance header
    # and a filer whose prose runs denser than Apple's. `worst_case // 4` was the earlier
    # ceiling and it is one token *per character* — it would still pass after a 3.5x
    # regression in density, which is to say after the margin it guards is gone.
    claimed = (CHUNK_SIZE_CHARS * 2) // MAX_TOKENS_PER_CHAR

    assert max(counts) < claimed, (
        f"real prose is using {max(counts)} tokens per chunk against the {claimed} this "
        f"test claims (and a {worst_case}-token worst case) — the margin that makes "
        f"CHUNK_SIZE_CHARS tunable has narrowed; re-measure before raising it."
    )
