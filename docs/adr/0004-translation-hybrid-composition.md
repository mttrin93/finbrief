# ADR-0004: How query translation and hybrid search compose inside retrieve()

Query translation paraphrases; hybrid search wins on exact tokens (BM25). If translation
were allowed to *replace* the user's query, it could strip the literal identifiers
("Item 1A", tickers, ratio names, ISINs) that BM25 depends on — degrading the very
exact-identifier bucket hybrid exists to serve.

**Decision.** Translation only ever *adds*; composition is symmetric.

- The **original query is always retained** as a query variant. Translation adds up to 3
  sub-queries (capped for latency).
- **All variants run through both BM25 and vector.** Candidate lists are fused with
  Reciprocal Rank Fusion, deduplicated by chunk id, then truncated to top-k.
- One symmetric code path — easier to reason about and to defend than asymmetric routing.

**Invariant.** BM25 always sees the raw identifiers in the retained original, and
additionally covers identifiers that appear only in decomposed sub-queries.

**Chunk-side counterpart.** Each chunk's indexed text opens with a provenance header
(`AAPL | FY2025 10-K | Item 1A. Risk Factors`) so BM25 can match section and ticker
literals on every chunk of a Section rather than only the one the splitter left the
heading in — an effect on index content that is not assumed but measured in the T10
per-bucket A/B (#11).

**Refined pre-registered hypotheses (supersede ADR-0002's):**
- hybrid > vector-only on `exact-identifier`
- `±translation` ≈ neutral on `exact-identifier` (BM25 on the retained original already
  nails it), clearly positive on `multi-hop`
- all configs ≈ tied on `semantic`

**Provenance.** For each surfaced chunk, log which variant × which retriever surfaced it
and its RRF contribution. This is the data behind the RAG-visualization panel and lets the
A/B analysis show *why* hybrid wins, not merely *that* it does.
