"""Hermetic stand-ins for the paid, networked parts of retrieval.

Not a fixture module — these are the doubles a fixture is built from, kept out of
`conftest.py` so a test that wants to embed something itself can import one directly.
"""

from __future__ import annotations

import math

from langchain_core.embeddings import Embeddings

#: The fake embedding's whole vocabulary: terms a 10-K question actually turns on, so a
#: query about supply chains lands on the chunk that discusses them.
VOCAB = (
    "supply",
    "chain",
    "manufacturing",
    "competition",
    "revenue",
    "risk",
    "climate",
    "iphone",
    "services",
    "interest",
    "rate",
    "gross",
    "margin",
)


class KeywordEmbeddings(Embeddings):
    """Bag-of-words over `VOCAB`, L2-normalised.

    The suite must not embed through OpenRouter (no key, no network, no spend — CLAUDE.md),
    which also rules the ingested `data/chroma` store out as a fixture: it is only
    queryable by the paid model that wrote it.

    Deterministic, unlike `FakeEmbeddings`' random vectors, so a test can assert an *order*
    and not merely a count; and lexical enough that the nearest neighbour of a question is
    the chunk a reader would have picked. Distances are meaningful because the vectors are
    normalised: L2 order equals cosine order.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        counts = [float(lowered.count(term)) for term in VOCAB]
        norm = math.sqrt(sum(count * count for count in counts)) or 1.0
        return [count / norm for count in counts]
