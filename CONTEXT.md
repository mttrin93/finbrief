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
clusters (big tech, autos, banks, EU ADRs). Decided at ingest time; serves triple duty —
KB scope, demo cast, and peer pool.
_Avoid_: watchlist (a Tier-2 per-user concept), portfolio

**Peer**:
Another Universe company in the same sector as a given company, used as the comparison set
for its ratios. Peers are drawn exclusively from the Universe — never external tickers.

**Filing**:
A company's annual 10-K. FinBrief ingests only its curated sections, not the whole document.
_Avoid_: document, 10-Q (out of scope)

**Section**:
A named part of a filing that FinBrief ingests — Item 1 (business), 1A (risk factors),
7 (MD&A), 7A (market-risk disclosures). The unit of grounding scope.

**Bucket**:
A stratum of the evaluation golden set — `semantic`, `exact-identifier`, `tool-augmented`,
or `multi-hop` — chosen so A/B results are reported per query type.

**Golden set**:
The hand-authored, source-separated reference Q/A used to evaluate retrieval and answers.
Authored against ingested sections only.

## Settled facts

- Embeddings served via OpenRouter `/v1/embeddings` (verified July 2026) — do not
  re-litigate without re-checking the docs.
