# FinBrief

Domain language for FinBrief — a domain-specialised RAG assistant that produces grounded,
source-cited equity-research briefs for a fixed set of large-cap companies.

## Language

**Brief**:
The structured, source-cited pre-earnings snapshot FinBrief produces for one company —
business overview, risk factors, valuation, and recent news.
_Avoid_: report, summary

**Universe**:
The fixed set of ~10–15 large-cap companies FinBrief covers, curated as same-sector peer
clusters (big tech, autos, banks, healthcare). Decided at ingest time; serves triple duty —
KB scope, demo cast, and peer pool. Every member files a 10-K — foreign private issuers
(20-F) are out of scope, since a **Filing** is a 10-K (ADR-0007).
_Avoid_: watchlist (a Tier-2 per-user concept), portfolio

**Peer cluster**:
A curated group of Universe companies chosen for ratio comparability — `big_tech`, `autos`,
`banks`, `healthcare` — which doubles as the peer pool. Curated, not taxonomic: `big_tech`
spans three GICS sectors, and AMZN sits apart from the autos despite sharing one with them.
GICS is an input to curating a cluster, never the rule (ADR-0009).
_Avoid_: sector (the rule ADR-0009 rejects)

**Peer**:
Another Universe company in the same **Peer cluster** as a given company, used as the
comparison set for its ratios. Peers are drawn exclusively from the Universe — never
external tickers — and never selected by GICS sector (ADR-0009).

**Filing**:
A company's annual 10-K. FinBrief ingests only its curated sections, not the whole document.
_Avoid_: document, 10-Q (out of scope)

**Section**:
A named part of a filing that FinBrief ingests — Item 1 (business), 1A (risk factors),
7 (MD&A), 7A (market-risk disclosures). The unit of grounding scope.
A **Section** is text, never a pointer to text: six Universe companies (all three banks,
all three healthcare names) answer Item 7A by incorporating Item 7 by reference, and their
market-risk content is therefore in the KB labelled `Item 7`. Nine of fifteen companies
have an `Item 7A` Section; all fifteen have market-risk grounding (ADR-0007 amendment).

**Chunk**:
One unit of retrieval — a slice of a single **Section**, never straddling a Section
boundary, carrying `ticker`, `filing_type`, `section`, `fiscal_year` and `accession`. Its
*indexed* text is a **provenance header** (`AAPL | FY2025 10-K | Item 1A. Risk Factors`)
followed by the filer's verbatim body, so BM25 can match section and ticker literals on
every chunk (ADR-0004). Its id is `accession:Item:index`, which is what makes re-ingest
idempotent.
_Avoid_: passage; and reserve **context** for what is handed to the LLM, not for a chunk.
Say **provenance header** in full for the indexed prefix — bare "provenance" now also names a
Context's **Surfaced rows**, which are a different thing entirely.

**Context**:
One retrieved **Chunk** as it is handed to the LLM and shown in the sources panel — the
filer's verbatim body with the provenance header stripped, the chunk's filing provenance
(`ticker`, `filing_type`, `section`, `fiscal_year`, `accession`), its `distance`, its
`fused_score`, its **Surfaced rows** and its `rank`. `distance` is Chroma's own, in the
collection's space (L2 for `filings`): **lower is nearer**, and deliberately not called a
score — it is not normalised to 0…1 and not higher-is-better. It is **`None` when no vector
search returned this chunk**, which under `hybrid` is the ADR-0004 case rather than an edge
case: the exact-identifier chunks hybrid exists to recover are the ones vector search ranked
outside `k`, and BM25 has no distance to put there. When several variants found it, the
distance is the *nearest* of them. `rank` is 1-based order, and it is what an inline `[n]`
marker, the *n*th block of the prompt's `<sources>` and the *n*th entry of the sources panel
all resolve to; that agreement is what lets a reader check a citation. What `rank` was decided
by is `fused_score`, **not** `distance` — under `hybrid` the two disagree routinely, and that
disagreement is the finding the A/B is looking for. A **Chunk** is what the store holds; a
**Context** is what one turn was grounded in.
_Avoid_: score (for `distance`), passage; and never read `rank` as nearness order.

**Retrieval**:
One call to `retrieve()`: the query **variants** it ran, plus the **Contexts** that came back
and whether translation and the planner ran at all. It is a class rather than a bare sequence
of Contexts because a variant that surfaced **nothing** is a fact about the retrieval that no
chunk's provenance can carry, and the *How I answered* panel has to show it (ADR-0004 §1).
`variants[0]` is always the analyst's own question.

**Query variant**:
One query actually retrieved on. Three kinds, named apart because they have different causes
and the T6 finding is about which one does the work: the analyst's **original** (always
retained — translation only ever *adds*), the deterministic **ticker form**, and the planner's
**sub-queries**. Budget: 1 + ≤1 + ≤`max_sub_queries` (ADR-0004 amendment).
_Avoid_: calling the ticker form a sub-query — that credits a model for a `config` lookup.

**Ticker form** (entity normalisation):
The variant a question gains when it names a Universe company by name: `Tesla` → `TSLA`, from
a `config` lookup and never a model. It exists because a chunk's provenance header carries the
*ticker* while an analyst types the *name*, so lexical search for "Tesla" misses most of the
filer — the T6 finding that a failed hypothesis forced (ADR-0004 amendment §6).

**Retriever**, and **Strategy**:
A **Retriever** is one way of finding chunks — `vector` or `bm25`. A **Strategy** is which
retrievers a retrieval runs: `vector` or `hybrid` (both). Deliberately two words: a chunk's
Surfaced row names the retriever that found it, which is not the same as the strategy the
retrieval ran under, and one name for both would make "did BM25 find this?" unaskable.

**Fusion** (RRF):
How candidate lists become one ranking: every variant × every retriever produces a list, each
chunk scores `1 / (RRF_K + rank)` per list that returned it, the sums are added, and the pool
is deduplicated by chunk id and truncated to top-k. Ranks, never scores — an L2 distance and a
corpus-relative BM25 score are not comparable, so nothing here blends them (`RRF_K = 60`, the
published default, stated and not tuned).

**Surfaced row**:
One candidate list's contribution to one chunk's fused rank: which **variant** × which
**Retriever** surfaced it, at what rank, for what RRF contribution, at what distance *in that
list*. The data behind "*why* hybrid wins" rather than merely that it does.
_Avoid_: bare "provenance" — see **Chunk**.

**Grounded answer**:
One turn's answer text together with the **Contexts** that grounded it — the unit the
evaluation harness scores and the UI renders. The text carries **no disclaimer**: the
disclaimer is rendered beside it by whichever surface shows it, so a model that forgets one
cannot ship an answer without it, and RAGAs faithfulness scores the grounded prose rather
than boilerplate. An answer with no Contexts is not grounded, and says so.

**Bucket**:
A stratum of the evaluation golden set — `semantic`, `exact-identifier`, `tool-augmented`,
or `multi-hop` — chosen so A/B results are reported per query type.

**Golden set**:
The hand-authored, source-separated reference Q/A used to evaluate retrieval and answers.
Authored against ingested sections only.

## Settled facts

- Embeddings served via OpenRouter `/v1/embeddings` (verified July 2026) — do not
  re-litigate without re-checking the docs.
