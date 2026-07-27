"""Hermetic stand-ins for the paid, networked parts of retrieval.

Not a fixture module — these are the doubles a fixture is built from, kept out of
`conftest.py` so a test that wants to embed something itself can import one directly.
"""

from __future__ import annotations

import math

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AnyMessage
from pydantic import Field

from finbrief.ingestion.model import Section
from finbrief.retrieval.hybrid import Retriever, Surfaced, rrf_contribution
from finbrief.retrieval.retrieve import Context


def a_context(
    rank: int = 1,
    *,
    ticker: str = "TSLA",
    section: Section = Section.RISK_FACTORS,
    fiscal_year: int = 2025,
    distance: float | None = 0.5,
    body: str | None = None,
    accession: str = "0001628280-26-003952",
    provenance: tuple[Surfaced, ...] | None = None,
) -> Context:
    """A retrieved chunk, for the callers that display or score one rather than fetch it.

    Shared because three test modules were each writing their own (issue #5 review), and a
    `Context` that drifts between them is a citation shape the UI and the smoke report can
    disagree about while both suites stay green.

    The default `provenance` is the shape `vector` without translation produces: one row, the
    analyst's own question through the vector retriever. That keeps `fused_score` consistent
    with the rows rather than a free-floating number — the invariant `test_retrieve.py` asserts
    of a real retrieval, and one a display test should not be able to violate by accident. Pass
    `provenance=()` for a chunk read back from a pre-Phase-4 checkpoint.
    """
    surfaced = (
        provenance
        if provenance is not None
        else (
            Surfaced(
                variant=f"{ticker} {section.value}?",
                retriever=Retriever.VECTOR,
                rank=rank,
                contribution=rrf_contribution(rank),
                # The row's own distance is the chunk's here, because there is one vector row:
                # `Fused.distance` is the nearest of them, and with one it is that one. Keeping
                # them consistent is the same reason `fused_score` is summed rather than typed.
                distance=distance,
            ),
        )
    )
    return Context(
        chunk_id=f"{accession}:{section.value}:{rank}",
        body=body if body is not None else f"{ticker} {section.value} body {rank}.",
        ticker=ticker,
        filing_type="10-K",
        section=section,
        fiscal_year=fiscal_year,
        accession=accession,
        distance=distance,
        rank=rank,
        fused_score=sum(row.contribution for row in surfaced),
        provenance=surfaced,
    )


class ScriptedChatModel(GenericFakeChatModel):
    """A chat model that returns pre-written replies and keeps the prompts it was handed.

    Two reasons it is not `GenericFakeChatModel` itself. It answers `bind_tools`, which
    `BaseChatModel` refuses by default — so a stock fake cannot drive an agent loop at all;
    returning `self` puts the script, rather than the model, in charge of which tools fire
    with which arguments, which is exactly what a test of the agent's *wiring* wants to
    control. And it records `prompts`, because half of what there is to assert about memory
    is what the model was *shown*: a follow-up whose prompt does not contain the earlier turn
    has no conversation to resolve "its debt" against, however well it answers.

    Nothing here checks the script against the bound tools, deliberately — a test that
    scripts a call to a tool the agent does not have is testing error handling, and should be
    able to.
    """

    #: Every message list this model has been called with, oldest call first.
    prompts: list[list[AnyMessage]] = Field(default_factory=list)

    #: The keyword arguments of every `bind_tools` call, oldest first. Recorded because some
    #: of what the agent asks of a provider is expressed only there — `parallel_tool_calls`
    #: is a request the binding makes, invisible in the messages and in the reply.
    bind_kwargs: list[dict] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):  # the script decides the calls, not the model
        self.bind_kwargs.append(dict(kwargs))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


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
