"""Hybrid search's two halves: a BM25 index, and Reciprocal Rank Fusion over both (ADR-0004).

`retrieve()` owns what a retrieval *means* — which variants run, through which retrievers, and
what a `Context` is. This module owns the two mechanisms underneath that: the lexical retriever
vector search cannot be, and the arithmetic that merges their answers into one ranking.

**Why BM25 at all.** Vector search scores *aboutness*, and a filer's name barely moves the
vector: issue #6 records a live retrieval where the query `Tesla debt` returned four Ford
chunks above the one Tesla chunk, because Tesla and Ford discuss indebtedness in near-identical
language and Ford's chunks discuss it more. A lexical retriever scores the token `tesla`
instead, where a Ford chunk earns nothing, and IDF makes the rare term the decisive one — which
is the mechanism ADR-0004 predicted would fix that bucket. The chunk-side counterpart is
`Chunk.text`'s provenance header, which is why `Item 7` and a ticker are matchable on every
chunk of a Section rather than only the one the splitter left the heading in.

**That prediction was measured and it failed — read ADR-0004 §6 and §7 before trusting the
paragraph above.** The header carries the *ticker* and an analyst types the *name*, so BM25 on
`Tesla debt` recovered the filer and lost the topic, and hybrid alone moved the relevant chunk
from rank 5 to absent. The fix was query-side entity normalisation
(`query_translation.normalised`), and §7's ablation shows the recovery is **embedding-side**:
BM25 supplies one vote of three, so its role is redundancy rather than recovery. This module is
still the right shape, and the reasoning above is still why a lexical retriever belongs in the
pipeline — but it is a pre-measurement argument, and §6/§7 are what the numbers say.

**And "IDF makes the rare term the decisive one" cuts both ways — §10.** IDF measures rarity in
*this corpus*, not informativeness in the analyst's question, so a question-register word that
filings happen not to use (`main`, df 0.1%) outweighs the filer's own name. `query_terms` drops
that scaffolding from the query before scoring; the corpus keeps every term.

**Why RRF rather than score blending.** The two retrievers' scores are not comparable and
cannot be made so: Chroma returns a squared L2 distance (lower is nearer, unbounded above) and
BM25 returns a corpus-relative sum of IDF-weighted term scores (higher is better, unbounded).
Normalising either onto the other's scale needs a min/max over the candidate set, which makes a
chunk's score depend on which *other* chunks came back — so the same chunk scores differently
in two retrievals and the A/B measures the normalisation as much as the strategy. RRF consumes
only **rank**, which both retrievers genuinely have.

`fuse` and `BM25Index` read no configuration and open no collection: `fuse` is a pure function
of its candidate lists, and an index is built from documents somebody else read out of the store
— which is what lets both be tested without a key, a network, or a paid embedding. The two
exceptions are named rather than implied, because a blanket "nothing here" claim was wrong on
both counts: `rrf_contribution` reads `config.RRF_K`, and `bm25_index` — the production
accessor, not the class — performs the whole-collection read through `vectorstore.all_chunks`,
which is still the only module allowed to touch the collection (CLAUDE.md).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from finbrief.caching import build_once
from finbrief.config import RRF_K
from finbrief.observability.logging_setup import log_event
from finbrief.retrieval.vectorstore import all_chunks

logger = logging.getLogger(__name__)


class Retriever(StrEnum):
    """The retrievers a candidate list can come from.

    Deliberately *not* the same enum as `config.RetrievalStrategy`. A strategy names a
    composition (`hybrid` = both of these over every variant); a retriever names one component
    of it. Collapsing them would make `strategy="bm25"` look like a configuration somebody
    could select, and lexical-only retrieval is not one of the four configurations ADR-0002
    measures.
    """

    VECTOR = "vector"
    BM25 = "bm25"


#: Word characters, lowercased — the whole of BM25's tokenizer, applied to the query and the
#: chunk alike. Splitting on non-alphanumerics is what keeps the `exact-identifier` bucket's
#: literals matchable: `Item 1A` becomes `item`/`1a` and `10-K` becomes `10`/`k` on **both**
#: sides, so a query naming a Section matches a chunk whose header names it. A tokenizer that
#: treated punctuation as part of a word would make that bucket depend on whether the analyst
#: typed the hyphen.
_WORD = re.compile(r"[a-z0-9]+")

#: Terms dropped from a **query** before BM25 scores it — never from the corpus (ADR-0004 §10).
#:
#: A published English stopword list, plus exactly one word: `main`. Inlined as a literal rather
#: than taken from NLTK, which downloads its corpora at first use and would break the same
#: hermetic-suite contract that made `tests/fixtures/tiktoken/` necessary (CLAUDE.md).
#:
#: **`main` is the whole finding, and it is not a stopword anywhere.** A live
#: `what are the main risk factors for Tesla?` returned GOOGL's `Item 7:21` — "The **main**
#: components of our research and development expenses **are**" — at BM25 **rank 1**, above
#: every Tesla Item 1A chunk, matching `main`/`for`/`the`/`are` and none of `risk`/`factors`/
#: `tesla`. `main` alone was 10.6016 of its 16.4742, because filers write "principal" and
#: analysts type "main": df 7/5842 (0.1%), idf 6.6568 — a **higher weight than `tesla`**
#: (5.1261). IDF measures corpus rarity, and no corpus statistic can separate a rare
#: question-register word from a rare identifier like `nvda`. Only a query-side lexicon can.
#:
#: **The growth rule is the important part** (ADR-0004 §10): no word enters this set without a
#: recorded `df` statistic *and* a measured query it fixes. A qualifier set of the near
#: neighbours — `key major biggest largest top overall important` — was measured against seven
#: queries and changed **no** result, so it is inert and stays out; `principal`, `common`,
#: `general`, `significant`, `basic`, `central`, `chief` and `core` each name a real filing
#: concept (*principal amount*, *common stock*, *general and administrative*, *significant
#: accounting policies*, *basic EPS*, *central bank*, *Chief Executive Officer*) and would cost
#: recall to buy nothing measured.
_STOPWORDS = frozenset(
    (
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
        "ll", "m", "o", "re", "ve", "y",
    )
)  # fmt: skip

#: The stopword list **plus exactly one word**, kept a separate expression so the lexicon's one
#: judgement call cannot be mistaken for part of the published list it sits beside.
_SCAFFOLDING = _STOPWORDS | frozenset({"main"})


def tokenize(text: str) -> list[str]:
    """`text` as BM25 terms. Used for the corpus and the query, so they cannot disagree."""
    return _WORD.findall(text.lower())


def query_terms(query: str) -> list[str]:
    """`query` as the terms BM25 should score it on — `tokenize` minus `_SCAFFOLDING`.

    **The corpus keeps every term; only the query is filtered**, which is what makes this
    change safe to land on the measured path. The index is untouched, so `df`, `idf` and `avgdl`
    are exactly what they were, and any query carrying no scaffolding term scores **byte-
    identically to the baseline** — verified on ADR-0004 §6's `Tesla debt` before-case and on
    `TSLA Item 1A` / `NVDA Item 7A interest rate risk`, all five rows unchanged.

    This is not the disagreement `tokenize`'s note warns about. The *tokenizer* is still shared:
    both sides split on the same rule, so `Item 1A` still becomes `item`/`1a` on both and the
    exact-identifier bucket is untouched. What differs is a query-side **term filter**, which is
    ordinary BM25 practice — the vector half likewise embeds the variant unmodified rather than
    tokenizing it at all. Each retriever still sees **every** variant (ADR-0004's composition is
    unchanged); each applies its own preprocessing to it, which is a different thing from one
    retriever seeing less than the other.

    A query that is *entirely* scaffolding scores nothing rather than falling back to its
    unfiltered terms: falling back would re-admit precisely the chunks this exists to exclude,
    and `nearest` already treats an empty term list as "no lexical hits" — the honest answer,
    which `fuse` is built to accept from a short list.
    """
    return [term for term in tokenize(query) if term not in _SCAFFOLDING]


@dataclass(frozen=True, slots=True)
class Surfaced:
    """One candidate list's contribution to one chunk's fused rank — ADR-0004's provenance.

    "Which variant × which retriever surfaced it, and its RRF contribution", as one row. It
    carries the variant's **text** rather than an index into the retrieval's variant list, for
    the same reason `Context` carries a chunk's provenance rather than a reference to it: this
    row is rendered in the RAG-viz panel and survives a checkpoint round trip on its own, and a
    positional reference is a thing that can be resolved against the wrong list. The
    structured logs invert that and record the *index*, because a variant is derived from the
    analyst's question and those lines are kept (`retrieve`).
    """

    variant: str
    retriever: Retriever
    #: 1-based position in that one candidate list — not in the fused result.
    rank: int
    #: `1 / (RRF_K + rank)`. Kept rather than recomputed so the panel and the A/B analysis read
    #: the number that was actually summed, even if `RRF_K` were ever restated.
    contribution: float
    #: The vector distance **this** candidate list gave this chunk, or `None` for a BM25 row.
    #:
    #: Per row, not per chunk, and that is the point: `Fused.distance` keeps only the nearest
    #: any variant produced, but ADR-0004 §6/§7's mechanism argument is a *per-variant*
    #: comparison — "the ticker form ranks the chunk 5th under vector search too, at distance
    #: 0.6778 against the original's 1.0406". That is the number §7's pre-registration — that
    #: hybrid's marginal contribution here is small *because the recovery is
    #: embedding-side* — has to be refuted with, and folding the rows to a minimum threw it
    #: away, which is why §6's own table came from a scratchpad script re-running each variant
    #: separately rather than from anything the engine emitted. Kept here, it is falsifiable
    #: from the artifact and from one `retrieval` log line (issue #6 review).
    distance: float | None = None

    def as_payload(self) -> dict[str, str | int | float | None]:
        """JSON-safe primitives — this row crosses the agent's checkpoint (see `Context`)."""
        return {
            "variant": self.variant,
            "retriever": self.retriever.value,
            "rank": self.rank,
            "contribution": self.contribution,
            "distance": self.distance,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Surfaced:
        """Rebuild a row from `as_payload`, reading an absent `distance` as `None`.

        Every reader of a checkpointed payload here tolerates a shape written before the current
        one, because a live conversation's checkpoint outlives a deploy. Unlike `Context` and
        `Retrieval`, `Surfaced` has no *deployed* older shape to tolerate — the class arrives
        whole in Phase 4 — so this is the convention applied consistently rather than a shim for
        a payload seen in the wild. `None` is the honest absence value: it is what a BM25 row
        carries anyway, and there is no distance to invent for a row that recorded none.
        """
        distance = payload.get("distance")
        return cls(
            variant=str(payload["variant"]),
            retriever=Retriever(payload["retriever"]),
            rank=int(payload["rank"]),
            contribution=float(payload["contribution"]),
            distance=None if distance is None else float(distance),
        )


@dataclass(frozen=True, slots=True)
class CandidateList:
    """One retriever's answer to one query variant, best first — fusion's unit of input.

    `hits` pairs each document with the **vector distance** that found it, or `None` when the
    retriever has no such number. BM25's own score is deliberately not carried: it is
    corpus-relative and not comparable to anything else on screen, so displaying it beside a
    distance would invite exactly the cross-scale comparison RRF exists to avoid. What survives
    into provenance from a BM25 hit is its rank, which is all RRF consumes.
    """

    variant: str
    retriever: Retriever
    hits: tuple[tuple[Document, float | None], ...]


@dataclass(frozen=True, slots=True)
class Fused:
    """One chunk after fusion: what ranks it, and every candidate list that voted for it."""

    document: Document
    #: The sum of `provenance`'s contributions — what the fused ranking is sorted by.
    score: float
    provenance: tuple[Surfaced, ...]
    #: The nearest vector distance any variant produced for this chunk, or `None` when only
    #: BM25 found it. `None` rather than a stand-in: the exact-identifier chunks ADR-0004 is
    #: built on are precisely the ones vector search did not return, so a fabricated 0.0 would
    #: put a number in the sources panel that no measurement produced.
    distance: float | None


def rrf_contribution(rank: int) -> float:
    """`1 / (RRF_K + rank)` — one candidate list's vote for the chunk at `rank`."""
    return 1.0 / (RRF_K + rank)


def fuse(candidate_lists: Sequence[CandidateList], *, limit: int) -> tuple[Fused, ...]:
    """Fuse candidate lists into one ranking: RRF, deduplicated by chunk id, top-`limit`.

    Pure and deterministic, which ADR-0003 stakes the A/B on. Two properties do the work:

    - **The fused pool is truncated after fusion, never before.** Every candidate list is scored
      and merged in full, so a chunk that ranks low in two lists can outrank one that ranks high
      in a single list — that agreement is what RRF exists to find and what a pre-fusion cut
      would discard. This is a rule about *this function*, not about how deep its inputs are
      fetched: `retrieve._candidate_lists` deliberately fetches exactly `k` per list, which is
      what keeps `vector` without translation byte-identical to the T3 baseline (ADR-0004
      amendment §2), so the chunk both retrievers rank sixth at `k=5` is out of reach here by
      that separate, deliberate choice. Widening the fetch is a Tier-2 experiment; nothing in
      `fuse` assumes it either way.
    - **Ties break on chunk id.** Two chunks at the same rank in different lists score
      identically, and without a stated tie-break the winner would be whichever retriever
      happened to be iterated first: reproducible within one process, not across a refactor,
      and silently so.

    A chunk's `distance` is the **nearest** any vector list gave it, and `None` if none did.
    Each `Surfaced` row keeps the distance *its own* list gave it, because the fold to a minimum
    is what destroyed ADR-0004 §7's per-variant comparison — see `Surfaced.distance`.
    """
    documents: dict[str, Document] = {}
    provenance: dict[str, list[Surfaced]] = {}
    distances: dict[str, float] = {}
    for candidate_list in candidate_lists:
        for rank, (document, distance) in enumerate(candidate_list.hits, start=1):
            chunk_id = document.id or ""
            documents.setdefault(chunk_id, document)
            provenance.setdefault(chunk_id, []).append(
                Surfaced(
                    variant=candidate_list.variant,
                    retriever=candidate_list.retriever,
                    rank=rank,
                    contribution=rrf_contribution(rank),
                    # Per row, so a per-variant distance comparison survives fusion — see
                    # `Surfaced.distance`. `distances` below still folds these to the nearest,
                    # which is what one chunk can honestly report.
                    distance=distance,
                )
            )
            if distance is not None:
                distances[chunk_id] = min(distance, distances.get(chunk_id, distance))
    fused = [
        Fused(
            document=document,
            score=sum(row.contribution for row in provenance[chunk_id]),
            provenance=tuple(provenance[chunk_id]),
            distance=distances.get(chunk_id),
        )
        for chunk_id, document in documents.items()
    ]
    fused.sort(key=lambda item: (-item.score, item.document.id or ""))
    return tuple(fused[:limit])


class BM25Index:
    """A BM25 Okapi index over the collection's chunks — hybrid search's lexical half.

    Built from documents rather than from a store, so it is testable without a collection and
    so `vectorstore.py` stays the only module that reads one. `bm25_index` below is the
    production accessor and the one that caches.

    `rank_bm25` holds the tokenized corpus in memory and scores every document on every query.
    That is affordable at this corpus's size (roughly 5,800 chunks) and is why there is no
    persisted lexical index to keep in step with Chroma: the index is derived from the
    collection at process start and cannot disagree with it.
    """

    __slots__ = ("_documents", "_index", "_terms")

    def __init__(
        self,
        documents: tuple[Document, ...],
        index: BM25Okapi | None,
        terms: tuple[frozenset[str], ...],
    ) -> None:
        self._documents = documents
        self._index = index
        #: Each document's term set, kept so membership is a set intersection rather than a
        #: threshold on the score. See `nearest` for why the score cannot answer that question.
        #:
        #: Required rather than defaulted to `()`: the default was only ever right for the empty
        #: corpus `over` builds explicitly, and for any other it produced an object that
        #: constructs cleanly and then fails inside `nearest`'s `zip(..., strict=True)` at query
        #: time, complaining about argument lengths (issue #6 review).
        self._terms = terms

    def __len__(self) -> int:
        """How many chunks are indexed — what the build's log line reports."""
        return len(self._documents)

    @classmethod
    def over(cls, documents: Sequence[Document]) -> BM25Index:
        """Index `documents`' full indexed text — provenance header included (ADR-0004).

        An empty corpus yields an index with no `BM25Okapi` behind it at all, rather than one
        that raises on the first query: `BM25Okapi` divides by the corpus's average document
        length, so an un-ingested collection would answer "is retrieval wired up?" with a
        `ZeroDivisionError` instead of the empty result the tiered fallback expects.
        """
        held = tuple(documents)
        if not held:
            return cls((), None, ())
        tokenized = [tokenize(document.page_content) for document in held]
        return cls(
            held,
            # Library defaults throughout (`k1=1.5, b=0.75, epsilon=0.25`), stated and not
            # tuned, for the reason `config.RRF_K` gives: a hybrid result obtained at the best
            # of several parameter sets is a number about the sweep, not about the strategy.
            BM25Okapi(tokenized),
            tuple(frozenset(terms) for terms in tokenized),
        )

    def nearest(self, query: str, k: int) -> tuple[Document, ...]:
        """The `k` chunks `query` scores highest on, best first — matches only.

        **A chunk that shares no term with the query is not a hit.** Padding the list out to `k`
        with such chunks would hand each of them a rank, after which `fuse` counts a retriever
        vote for a chunk that retriever never found. A short list is the honest answer, and the
        reason `fuse` never assumes every list is the same length.

        Membership is a term-set intersection and deliberately **not** `score > 0`. BM25 Okapi's
        IDF goes negative for a term held by more than about half the corpus, and `rank_bm25`
        replaces those with `epsilon * average_idf` — so a chunk matching only a common term can
        score zero or below while genuinely containing it. Thresholding the score would have
        dropped the four peer chunks from issue #6's `Tesla debt` case because `debt` is common,
        which is precisely the comparison hybrid search is being measured on.

        The query is scored on `query_terms`, not `tokenize` — its scaffolding is dropped before
        both the scoring *and* the membership test, so a chunk that shares only a question's
        function words is not a hit either (ADR-0004 §10).

        Ties break on chunk id, for the same reason `fuse`'s do.
        """
        terms = query_terms(query)
        if self._index is None or not terms:
            return ()
        wanted = frozenset(terms)
        scored = [
            (score, document)
            for score, document, held in zip(
                self._index.get_scores(terms), self._documents, self._terms, strict=True
            )
            if held & wanted
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1].id or ""))
        return tuple(document for _, document in scored[:k])


def _bm25_index(store: Chroma) -> BM25Index:
    """The BM25 index over `store`'s chunks, built once per process.

    Cached because building it reads every chunk out of Chroma and tokenizes it — affordable
    once at first query, absurd per query. Keyed on the store *object*, which is why
    `vectorstore.default_filings_store` caches that too: a fresh `Chroma` per retrieval would
    miss this cache every time and rebuild the whole index for one query.

    **A consequence worth stating: the index is a snapshot.** A process that was serving
    queries while `scripts/ingest_filings.py` rewrote the collection keeps the lexical view it
    started with, while its vector searches see the new one. That is acceptable because ingest
    is an offline script and the app is restarted after it, and it is bounded — `maxsize=2`
    leaves room for one store to be replaced without unbounded retention of old corpora.

    **`build_once` rather than `lru_cache`, for the reason `default_filings_store` is** (T5,
    #9): with parallel tool calls, two searches in one step both miss this cold and both
    tokenize 5,800 chunks. Waste rather than corruption — unlike the Chroma handle, which
    raced fatally — but the fix is the same lock, and the argument for it is stronger here
    because the duplicated work is seconds of it.
    """
    started = time.perf_counter()
    index = BM25Index.over(all_chunks(store))
    log_event(
        logger,
        "bm25_index_built",
        chunks=len(index),
        build_ms=round((time.perf_counter() - started) * 1000),
    )
    return index


#: `maxsize=2`, as the docstring above explains.
bm25_index = build_once(_bm25_index, maxsize=2)
