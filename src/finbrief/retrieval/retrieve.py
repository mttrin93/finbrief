"""`retrieve(question, strategy, translate, k)` — the deterministic retrieval engine (ADR-0003).

The one entry point to the knowledge base. The evaluation harness calls it directly to
produce clean `(question → contexts → answer)` triples, and the `search_filings` tool wraps
*this* function so the measured chain and the shipped chain are one code path. Nothing above
it may open the `filings` collection itself.

**One symmetric composition, four configurations (ADR-0004).** Two switches move
independently — `strategy` (`vector` / `hybrid`) and `translate` (±) — and they are the four
configurations ADR-0002's A/B compares. They are not four code paths:

    variants   = (question,)  +  up to `max_sub_queries` sub-queries, if translating
    retrievers = (vector,)    +  bm25, if hybrid
    every variant × every retriever → a candidate list → RRF → dedup by chunk id → top-k

`vector` without translation is that pipeline with one candidate list in it, and RRF over a
single list is strictly decreasing in rank — so it returns exactly what T3 returned, which is
what keeps the A/B's baseline the baseline. Everything else is the same code with more lists.

**Return shape.** ADR-0003 writes the signature as `→ (contexts, scores)` and its T3 amendment
narrowed that to one sequence of `Context`. Phase 4 widens it once more, to a `Retrieval`: a
retrieval now knows something that is *not* a fact about any one chunk — which queries it ran.
The variants a translation produced have to reach the RAG-viz panel (user story 5) including the
ones that surfaced nothing, and a variant that lost the fusion is exactly the interesting
negative datum. `Retrieval.contexts` is still the sequence, each `Context` still carries its own
`distance` and `rank`, and the pairing that amendment protects is untouched.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel

from finbrief.config import RRF_K, RetrievalStrategy, Settings, get_settings
from finbrief.ingestion.model import Section
from finbrief.llm import build_chat_model
from finbrief.observability.logging_setup import log_event
from finbrief.retrieval import query_translation
from finbrief.retrieval.hybrid import (
    CandidateList,
    Fused,
    Retriever,
    Surfaced,
    bm25_index,
    fuse,
)
from finbrief.retrieval.vectorstore import default_filings_store, nearest_chunks

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Context:
    """One retrieved chunk, as it is handed to the LLM and shown in the sources panel.

    Named for the LLM's side of the boundary, per CONTEXT.md: a **Chunk** is what the
    store holds, a **Context** is what a turn was grounded in. It carries the chunk's
    provenance rather than a reference to it so a citation, a log line and the sources
    panel all read the same fields.
    """

    chunk_id: str
    body: str
    ticker: str
    filing_type: str
    section: Section
    fiscal_year: int
    accession: str
    #: Chroma's vector distance — **lower is nearer**. Deliberately not called a "score":
    #: the `filings` collection is L2, so this is a squared distance and not a normalised
    #: similarity, and calling it a score would invite a reader to expect 0…1 and
    #: higher-is-better.
    #:
    #: **`None` when no vector search returned this chunk** — which under `hybrid` is not an
    #: edge case but the point: the exact-identifier chunks ADR-0004 exists to recover are
    #: precisely the ones vector search ranked outside `k`, and BM25 has no distance of its own
    #: to put here. A stand-in (`0.0`, `inf`) would print a number in the sources panel that no
    #: measurement produced. When several variants found it, this is the **nearest** of them.
    distance: float | None
    #: 1-based position in this retrieval, best first — and the number an inline `[n]`
    #: uses, so a citation and the panel agree.
    #:
    #: **`retrieve()` sets it per retrieval; the shipped path renumbers it per conversation.**
    #: A citation has to name one chunk for a whole conversation, and a second search would
    #: otherwise reuse `[1]`, so `agent/citations.py` reassigns these into the thread's running
    #: sequence before the model sees them (ADR-0003 amendment §3). What arrives *here* is
    #: always 1…k, which is what the evaluation harness measures; a `Context` read back out of
    #: a checkpoint carries the conversation's number instead. Anything that treats a `rank` as
    #: an index into its own retrieval is reading the field on the wrong side of that boundary.
    rank: int
    #: The RRF score this chunk was ranked by: the sum of `provenance`'s contributions. This,
    #: not `distance`, is what decided `rank` — under `hybrid` the two disagree routinely, and
    #: that disagreement is the whole finding the A/B is looking for.
    fused_score: float
    #: Which variant × which retriever surfaced this chunk, and each one's RRF contribution
    #: (ADR-0004). The data behind the RAG-viz panel and behind "*why* hybrid wins" rather than
    #: merely "that it does" — one row per candidate list that voted for this chunk.
    provenance: tuple[Surfaced, ...]

    @property
    def citation(self) -> str:
        """`AAPL 10-K FY2025, Item 1A` — what an inline `[n]` marker resolves to."""
        return f"{self.ticker} {self.filing_type} FY{self.fiscal_year}, {self.section.value}"

    @property
    def retrievers(self) -> tuple[Retriever, ...]:
        """The distinct retrievers that surfaced this chunk, in first-seen order.

        What the sources panel labels a chunk with, and the shortest answer to "did BM25 find
        this?" — the question ADR-0004's exact-identifier prediction turns on.
        """
        seen = dict.fromkeys(row.retriever for row in self.provenance)
        return tuple(seen)

    def as_payload(self) -> dict[str, Any]:
        """This chunk as JSON-safe primitives — the form that crosses a serialised boundary.

        `search_filings` returns its chunks as a tool message's artifact, and the agent's
        checkpointer serialises every message it stores (ADR-0008). A frozen dataclass
        holding an enum survives that round trip only through an escape hatch LangGraph
        warns on and will remove; under its strict serialiser it comes back as an untyped
        dict instead, silently. So the wire form is written down here rather than left to a
        library's inference — and this class stays the one authority on the shape, since
        `from_payload` is what reads it back (ticket T4, #7).

        `provenance` is a **list** of dicts, not a tuple: the register in `agent/citations.py`
        compares a payload it rebuilt against the one it read back to decide whether a rewrite
        is owed, and JSON has no tuples, so emitting tuples here would make every round-tripped
        reply look changed.

        No cost in checkpoint size worth naming: the tool message's *content* already carries
        these bodies verbatim, because that is what the model reads.
        """
        return {
            "chunk_id": self.chunk_id,
            "body": self.body,
            "ticker": self.ticker,
            "filing_type": self.filing_type,
            "section": self.section.value,
            "fiscal_year": self.fiscal_year,
            "accession": self.accession,
            "distance": self.distance,
            "rank": self.rank,
            "fused_score": self.fused_score,
            "provenance": [row.as_payload() for row in self.provenance],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Context:
        """Rebuild a chunk from `as_payload`, for the surface that displays it.

        Tolerant of a payload written before Phase 4, because a live conversation's checkpoint
        outlives a deploy: such a chunk has no fusion behind it, so it reports no score and no
        provenance rather than raising, and the RAG-viz panel renders nothing for it. Its
        `distance` was never optional, which is why the fallback is the value and not `None`.
        """
        distance = payload["distance"]
        return cls(
            chunk_id=str(payload["chunk_id"]),
            body=str(payload["body"]),
            ticker=str(payload["ticker"]),
            filing_type=str(payload["filing_type"]),
            section=Section(payload["section"]),
            fiscal_year=int(payload["fiscal_year"]),
            accession=str(payload["accession"]),
            distance=None if distance is None else float(distance),
            rank=int(payload["rank"]),
            fused_score=float(payload.get("fused_score", 0.0)),
            provenance=tuple(
                Surfaced.from_payload(row) for row in payload.get("provenance", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class Retrieval:
    """One call to `retrieve()`: the queries it ran, and the chunks that came back.

    `variants` is what makes this a class rather than a bare tuple of contexts. The RAG-viz
    panel has to show *how the query was translated* (user story 5) — and that includes a
    sub-query which surfaced no chunk at all, which is the datum a per-chunk provenance row
    can never carry. `variants[0]` is always the analyst's own question (ADR-0004).

    `translated` is a fact about the *run*, not about `variants`' length: translation that ran
    and returned nothing usable is a different thing from translation that was switched off,
    and only one of them is worth a second look in the panel.
    """

    contexts: tuple[Context, ...]
    variants: tuple[str, ...]
    translated: bool

    @property
    def question(self) -> str:
        """The analyst's own question — the variant translation is forbidden to replace."""
        return self.variants[0] if self.variants else ""

    @property
    def added(self) -> tuple[str, ...]:
        """Every variant translation added. Empty when it was off, or added nothing."""
        return self.variants[1:]

    @property
    def ticker_form(self) -> str | None:
        """The deterministic ticker-form variant, if a Universe company was named.

        Reported apart from `sub_queries` because the two additions have different causes and
        the T6 amendment's finding is about which one earns the exact-identifier bucket — see
        `query_translation.added_variants`.
        """
        return query_translation.added_variants(self.variants)[0]

    @property
    def sub_queries(self) -> tuple[str, ...]:
        """What the *planner* added — the ticker form excluded."""
        return query_translation.added_variants(self.variants)[1]

    def as_payload(self) -> dict[str, Any]:
        """This retrieval as JSON-safe primitives — `search_filings`' artifact shape.

        One authority on the wire form, for the same reason `Context.as_payload` is: the tool's
        reply crosses the agent's checkpoint, and what comes back out of a strict serialiser is
        an untyped dict either way.
        """
        return {
            "chunks": [context.as_payload() for context in self.contexts],
            "variants": list(self.variants),
            "translated": self.translated,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Retrieval:
        """Rebuild a retrieval from `as_payload`, for the surface that displays it."""
        return cls(
            contexts=tuple(Context.from_payload(chunk) for chunk in payload.get("chunks", ())),
            variants=tuple(str(variant) for variant in payload.get("variants", ())),
            translated=bool(payload.get("translated", False)),
        )


def retrieve(
    question: str,
    *,
    strategy: RetrievalStrategy = RetrievalStrategy.VECTOR,
    translate: bool = False,
    k: int | None = None,
    store: Chroma | None = None,
    settings: Settings | None = None,
    model: BaseChatModel | None = None,
) -> Retrieval:
    """Retrieve the `k` chunks best matching `question`, best first.

    Deterministic *given its variants*: the same question, strategy and `k` against the same
    collection fuse to the same contexts in the same order (ADR-0003 — the eval harness depends
    on it). Under `translate=True` the variants come from a chat model, so determinism is that
    model's at temperature 0; `llm.py` builds every chat model there.

    `strategy` and `translate` are both required to be *named by the caller* rather than read
    from configuration here, and both default to the conservative value. That is what stops a
    number being reported against a configuration nobody selected: the shipped path names them
    from `Settings` (`agent.build_agent`), and the A/B harness names all four combinations
    itself. Passing a raw string for `strategy` is fine — it is normalised below.

    `store`, `settings`, `k` and `model` are injectable so tests can run against a fixture
    collection with a fake embedding and a scripted model, and so the evaluation harness can
    point at a throwaway index (ADR-0002); production passes none of them.
    """
    # Normalised rather than trusted: `RetrievalStrategy("vector")` accepts the enum member and
    # the raw string alike, so a caller reading a strategy out of a config file lands on the
    # same object the dispatch below compares against — and an unknown value raises `ValueError`
    # here, naming the valid ones, instead of quietly taking the vector branch.
    strategy = RetrievalStrategy(strategy)
    # `translate` joins the guard because the sub-query cap is configuration and is *enforced*
    # (ADR-0005's latency budget assumes it), so translating without settings is not a thing
    # this function can do — unlike a fully-injected vector retrieval, which needs no key.
    if translate or k is None or store is None:
        settings = settings or get_settings()
        k = k if k is not None else settings.retrieval_k
        store = store if store is not None else default_filings_store(settings)

    started = time.perf_counter()
    if translate:
        variants = query_translation.translate(
            question,
            model=model if model is not None else build_chat_model(settings),
            max_sub_queries=settings.max_sub_queries,
        )
    else:
        variants = (question,)

    candidate_lists = _candidate_lists(variants, strategy, store, k)
    fused = fuse(candidate_lists, limit=k)
    contexts = tuple(_as_context(item, rank) for rank, item in enumerate(fused, start=1))
    log_event(
        logger,
        "retrieval",
        strategy=strategy.value,
        translation=translate,
        k=k,
        rrf_k=RRF_K,
        # Counts, never the queries themselves: a variant is derived from the analyst's question
        # and these lines are kept. The panel is where the text is shown, to the person who
        # typed it.
        variants=len(variants),
        candidate_lists=len(candidate_lists),
        # Hits per candidate list, in `_candidate_lists`' order (variant-major, then retriever).
        # `candidate_lists` above counts lists *built*, so without this a list that matched
        # nothing is indistinguishable from one that was never run — and "sub-query 3 × BM25
        # matched nothing" is exactly the negative datum the RAG-viz panel invites a reader to
        # ask about (ADR-0004 §1). Counts only, so it says nothing about what was asked.
        per_list_hits=[len(candidate_list.hits) for candidate_list in candidate_lists],
        # The deduplicated pool fusion chose from, before truncation — the number that says
        # whether a wider `k` would have had anything to offer.
        fused_candidates=len(
            {document.id for lst in candidate_lists for document, _ in lst.hits}
        ),
        hits=len(contexts),
        latency_ms=round((time.perf_counter() - started) * 1000),
        # What was retrieved, not how good it was: enough for Phase 7 to reconstruct a
        # retrieval from the logs, and never the chunk text (these lines are kept).
        chunk_ids=[context.chunk_id for context in contexts],
        distances=[
            None if context.distance is None else round(context.distance, 4)
            for context in contexts
        ],
        # ADR-0004's provenance, per returned chunk. `variant` is the **index** into `variants`
        # rather than its text — the one place the two representations differ, and the reason is
        # the rule above: `Surfaced` carries the text because a panel renders it, a log line
        # carries the index because a log line must not carry user content.
        provenance=[
            {
                "chunk_id": context.chunk_id,
                "score": round(context.fused_score, 6),
                "surfaced": [
                    {
                        "variant": variants.index(row.variant),
                        "retriever": row.retriever.value,
                        "rank": row.rank,
                        "contribution": round(row.contribution, 6),
                        # The distance *this* row's list gave the chunk, not the chunk's nearest
                        # — which is what makes ADR-0004 §7's pre-registration refutable from a
                        # log line rather than only from a scratchpad re-run (see
                        # `Surfaced.distance`). A number, so it carries no user content.
                        "distance": (None if row.distance is None else round(row.distance, 4)),
                    }
                    for row in context.provenance
                ],
            }
            for context in contexts
        ],
    )
    return Retrieval(contexts=contexts, variants=variants, translated=translate)


def _candidate_lists(
    variants: tuple[str, ...],
    strategy: RetrievalStrategy,
    store: Chroma,
    k: int,
) -> tuple[CandidateList, ...]:
    """Every variant through every retriever this strategy runs — fusion's input.

    **Each list is `k` deep, not deeper.** ADR-0004 fixes the *output* at top-k and says nothing
    about candidate depth, and `k` is the choice that keeps `vector` without translation
    byte-identical to the T3 baseline the A/B compares against — a wider fetch would quietly
    move it. It also bounds the cost: hybrid + translation is already up to eight candidate
    lists, four of them paid embeddings. Widening it is a legitimate Tier-2 experiment and would
    be a change to the thing being measured, not a fix.

    Vector runs under every strategy; `hybrid` adds BM25 beside it. Nothing here is asymmetric:
    the original question is a variant like any other, which is the invariant that guarantees
    BM25 always sees the raw identifiers the analyst typed.

    **Each list is built where its hits are fetched, so a list cannot carry another
    retriever's.** An earlier shape looped over a `retrievers` tuple and assigned `hits` in an
    `if`/`elif` with no `else`: had BM25 ever been requested with no index built, `hits` would
    have kept the previous iteration's *vector* hits and been appended as a **BM25** candidate
    list — doubling every vote and writing provenance rows naming a retriever that never
    returned the chunk. Unreachable as it stood, and silent if it ever became reachable, which
    is the wrong combination for the one datum ADR-0004 promises: `BM25Index.nearest` refuses
    to pad a short list for this exact reason, and this function must not undo it (#6 review).
    """
    lexical = bm25_index(store) if strategy is RetrievalStrategy.HYBRID else None
    lists: list[CandidateList] = []
    for variant in variants:
        lists.append(
            CandidateList(
                variant=variant,
                retriever=Retriever.VECTOR,
                hits=tuple(nearest_chunks(store, variant, k)),
            )
        )
        if lexical is not None:
            lists.append(
                CandidateList(
                    variant=variant,
                    retriever=Retriever.BM25,
                    hits=tuple((document, None) for document in lexical.nearest(variant, k)),
                )
            )
    return tuple(lists)


def _as_context(item: Fused, rank: int) -> Context:
    """Rebuild a `Context` from a fused chunk's metadata and its indexed text.

    The body is recovered by dropping the provenance header `Chunk.text` prepends for
    BM25's benefit (ADR-0004). The header is machine furniture — repeating it inside a
    numbered context block, and again in the sources panel next to the same metadata, only
    invites the model to cite it as if the filer had written it.
    """
    metadata = item.document.metadata
    _, _, body = item.document.page_content.partition("\n\n")
    return Context(
        chunk_id=item.document.id or "",
        body=body or item.document.page_content,
        ticker=metadata["ticker"],
        filing_type=metadata["filing_type"],
        section=Section(metadata["section"]),
        fiscal_year=int(metadata["fiscal_year"]),
        accession=metadata["accession"],
        distance=item.distance,
        rank=rank,
        fused_score=item.score,
        provenance=item.provenance,
    )
