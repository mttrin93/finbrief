# FinBrief — Tier-1 Spec (pre-earnings equity-research brief)

> Review-facing scope (Tier-1 per ADR-0001). Synthesised from the grilling thread,
> `CONTEXT.md`, and ADR-0001…0010. Tier-2 skill-stretch is deferred (see Out of Scope).
> Canonical copy: the GitHub issue this was published to; this file is a repo mirror.

## Problem Statement

A junior equity analyst preparing for a company's earnings call needs a grounded,
source-cited snapshot — business overview, risk factors, current valuation, and recent
news — in minutes, not hours. General chatbots hallucinate financials, cite nothing, and
happily dispense investment advice; raw filings are hundreds of pages and slow to search;
market-data terminals are expensive and don't explain themselves. The analyst needs
answers that are **traceable to primary sources**, **current** on price and news, and
**honest** about what they will and won't do.

## Solution

**FinBrief** is a domain-specialised RAG assistant over a fixed **Universe** of ~10–15
large-cap companies, curated as same-sector peer clusters. For any company in the Universe
it produces a **Brief** — business, risk factors, valuation, recent news — grounded in
curated 10-K **Sections** (Items 1, 1A, 7, 7A) with inline citations, augmented by three
live finance tools, and defended by a security gate that refuses personalised investment
advice and resists prompt injection. Every answer shows its work: the retrieved chunks and
their provenance, the tool calls and their results, and (on request) how the query was
translated and retrieved. The retrieval pipeline is deterministic and independently
evaluated (RAGAs + per-bucket A/B), so the quality claims are measured, not asserted.

## User Stories

1. As an analyst, I want to ask about a company's risk factors in plain language, so that I get an answer grounded in its 10-K Item 1A rather than a generic web summary.
2. As an analyst, I want every factual claim to carry an inline citation (e.g. `[1] TSLA 10-K 2025, Item 1A`), so that I can verify it against the primary source.
3. As an analyst, I want a sources panel listing the retrieved chunks with their Section/ticker/year metadata, so that I can see exactly what grounded the answer.
4. As an analyst, I want vague questions ("is Tesla in trouble?") to be decomposed into sub-questions and retrieved thoroughly, so that I get a complete answer rather than a literal-match miss.
5. As an analyst, I want to optionally see how my query was translated and which chunks each variant surfaced, so that I understand and trust the retrieval.
6. As an analyst, I want exact-identifier queries ("what's in Item 1A?", a ticker, a ratio name) to retrieve precisely, so that jargon and identifiers aren't blurred away by semantic search.
7. As an analyst, I want to ask for current valuation and get live price/market-cap data via a tool, so that the brief reflects today, not the filing date.
8. As an analyst, I want computed ratios (P/E, D/E, margins) compared against its peer cluster, so that I can judge whether a company is cheap or expensive relative to that cluster.
9. As an analyst, I want the peer set and its size shown alongside the comparison (e.g. "vs. mean of 2 `autos` peers: F, GM"), so that I know the basis of the comparison.
10. As an analyst, I want recent news headlines for a company via a tool, so that I can connect current events to the filing's stated risks.
11. As an analyst, I want to ask a combined question ("anything in the news related to those risks?") and have the assistant interleave retrieval and the news tool in one answer, so that I don't have to run two separate queries.
12. As an analyst, I want a single "give me the full brief" command that assembles business, risks, valuation, and news into one structured output, so that I have a ready pre-call document.
13. As an analyst, I want tool results rendered as cards/charts (price history, ratio-vs-peer bars, news cards), so that I can read them at a glance.
14. As an analyst, I want progress indicators while retrieval and tools run, so that I know the app is working during longer operations.
15. As an analyst, I want the assistant to refuse "should I buy?" style requests and show a disclaimer, so that I'm not given personalised investment advice it isn't qualified to give.
16. As an analyst, I want the assistant to resist attempts to override its instructions or extract its system prompt, so that I can trust it under adversarial input.
17. As an analyst, I want the assistant to ignore instructions hidden inside retrieved news or filing text, so that poisoned source content can't hijack the answer.
18. As an analyst, I want the assistant to clearly state what it is and isn't grounded in (Items 1/1A/7/7A only), so that I don't mistake an out-of-scope guess for a grounded answer.
19. As an analyst, I want my conversation remembered within a session so follow-ups ("and its debt?") resolve against the prior company, so that I can converse naturally.
20. As an analyst, I want a fresh browser session to start a clean conversation, so that a new analysis isn't polluted by an old one.
21. As an analyst, I want invalid inputs (unknown tickers, over-long queries) handled gracefully with a clear message, so that mistakes don't crash the app.
22. As an analyst, I want transient data-source failures (a flaky price API) handled with a cached fallback and a banner, so that the app degrades gracefully rather than erroring.
23. As the developer, I want retrieval exposed as a deterministic `retrieve(question, strategy, k)` function, so that I can evaluate it independently of the agent's nondeterministic tool loop.
24. As the developer, I want a stratified golden set (semantic, exact-identifier, tool-augmented, multi-hop) with source-separated ground truth, so that my evaluation numbers are defensible and not circular.
25. As the developer, I want RAGAs metrics reported per bucket, so that I can see where retrieval is strong or weak by query type.
26. As the developer, I want an A/B comparison of vector vs. hybrid × ±translation reported per bucket, so that I can show *which* strategy wins *where* and why.
27. As the developer, I want per-chunk provenance (which variant × which retriever × RRF contribution) captured, so that I can explain *why* hybrid wins, not just that it does.
28. As the developer, I want the shipping default strategy pre-registered before I see the A/B data, so that the choice isn't p-hacked.
29. As the developer, I want a tool-calling evaluation over scripted prompts, so that I can show the agent selects the right tool with the right arguments.
30. As the developer, I want a security test suite covering obfuscated direct injection and planted indirect injection, so that I can demonstrate the gate works.
31. As the developer, I want structured per-query logs (strategy, retrieval hits, latency, tokens, gate-trigger metadata, agent-vs-original query), so that evaluation and the security analysis read from real data.
32. As the developer, I want a Phase-1 section-detection sanity check that fails loudly on mislabeled or missing Sections, so that bad KB data never silently corrupts retrieval numbers.
33. As the developer, I want the divergence between the agent's issued query and the original user query logged, so that I can report how often the shipped path differs from the measured chain.
34. As a reviewer, I want a marginal-contribution table for the security layers, so that I can see each layer earns its place.
35. As the analyst, I want to export or copy a completed brief (basic), so that I can paste it into my notes. *(Full multi-format export is Tier-2.)*

## Implementation Decisions

**Retrieval engine (ADR-0003, 0004).** A standalone deterministic component
`retrieve(question, strategy, k) → (contexts, scores)` runs query translation and hybrid
search internally, selected by a `strategy` config flag (temperature 0, fixed `k` in eval
mode). Translation only *adds*: the original query is always retained as a variant, plus up
to 3 sub-queries (capped for latency). All variants run through **both** BM25 and vector;
candidate lists are fused with Reciprocal Rank Fusion, deduplicated by chunk id, truncated
to top-k. Per-chunk provenance (variant × retriever × RRF contribution) is recorded. The
same function is wrapped as the `search_filings` agent tool, whose description instructs the
agent to pass the user question verbatim (the tool owns optimization) to avoid double
translation.

**Shipping default (ADR-0005).** Default strategy = `hybrid + translation`, pre-registered
on the dominance prediction. Falsification is directional (worse on *both* context precision
and recall within a bucket → drop to hybrid-only, write up as a finding), judged within a
≤1.5s p50 added-latency budget. Adaptive per-query routing is out of scope (Tier-2).

**Agent (ADR-0008).** `create_agent` with a `SqliteSaver` checkpointer as the memory of
record, bound tools `search_filings`, `get_stock_data`, `calculate_ratios`, `get_recent_news`.
LLM access via OpenRouter (OpenAI-compatible). Embeddings via OpenRouter
`text-embedding-3-small`.

**Tools.** `get_stock_data(ticker)` (yfinance, TTL-cached); `calculate_ratios(ticker)`
(P/E, D/E, margins vs. mean of same-cluster **in-Universe** peers resolved through the same
cached `get_stock_data` path — zero new API surface; reports peer set + n, ADR-0009);
`get_recent_news(ticker, days)` (RSS via feedparser, HTML-stripped on ingest). Tiered error
handling: API level (retry/backoff/cache), retrieval level (empty → fallback message),
generation level (refusals → graceful UX).

**Knowledge base (ADR-0007).** KB = curated Sections only (Items 1, 1A, 7, 7A) of the latest
10-K per Universe company, extracted structure-anchored via `edgartools` (bounded regex
fallback), section-aware chunked with metadata (`ticker`, `filing_type`, `section`,
`fiscal_year`, `accession` — the accession being both the idempotency key and the filter a
superseded fiscal year is evicted by), stored in a Chroma `filings` collection. A chunk's
*indexed* text opens with a provenance header (`AAPL | FY2025 10-K | Item 1A. Risk
Factors`) so BM25 matches section and ticker literals on every chunk (ADR-0004). A separate `news` collection holds ingested headlines; a small
`glossary` collection holds financial terms. A **dedicated injection-test collection** is
isolated from the demo KB. Idempotent by accession id. Phase-1 gate: per company×section
assert found / non-empty / length-bounded / body≫heading / starts at its own heading /
stops before the next Item, plus per filing that EDGAR's latest annual filing is a 10-K;
fail loudly. Item 7A incorporated by reference into Item 7 (six of fifteen Universe
filers) passes the gate but is not chunked, and the excusal only fires for filers
recorded in `config.ITEM_7A_POINTER_FILERS` — so those six companies have **no Item 7A
chunks**, which constrains golden-set authoring (ADR-0002, ADR-0007 amendment).

**Security gate (ADR-0006).** Input gate (front door), cheap-first, one model call per turn,
≤800ms p50: normalization → bounded regex denylist (catch exits early; pass always
escalates) → one zero-shot LLM classifier (own prompt, via OpenRouter). Output validator
(back door): Guardrails AI no-investment-advice validator (`on_fail="exception"` → graceful
refusal). Indirect injection is a first-class tested threat; retrieved text is framed as
data (quarantine). Marginal-contribution table published.

**Observability (ADR-0001 Tier-1).** Structured JSON-lines logging per query: strategy
config, retrieval hits, latency, token counts, gate-trigger metadata, agent-vs-original
query divergence. Runs *before* evaluation so RAGAs/A-B and the security analysis consume it.

**Evaluation (ADR-0002).** Golden set of ~24–28 Q/A in four stratified buckets (≥6 each),
ground truth authored from filings with section citations, candidates drafted by a *different*
model than the answering pipeline then hand-verified. RAGAs (all four metrics) per bucket;
A/B (vector vs. hybrid × ±translation) per bucket with pre-registered hypotheses; retrieval
precision@k/recall@k per strategy × bucket; tool-calling eval on scripted prompts. Injection
cases live in the security suite (pass/fail), never in the RAGAs table.

**UI (Streamlit, ADR-0008).** Chat UI: sources panel (chunks + metadata), tool-call result
cards/charts, RAG-visualization panel (original → sub-queries → retrieved chunks w/ scores +
provenance), progress indicators, grounding-scope disclosure. `st.session_state` holds only
thread_id (uuid4 per session), UI toggles, and the display transcript. Agent + checkpointer
built once under `@st.cache_resource` (SQLite `check_same_thread=False`).

## Testing Decisions

Good tests assert **external behavior at the highest seam**, not implementation details.
Six seams (confirmed):

1. **`retrieve(question, strategy, k)`** — all retrieval-quality behavior: RRF fusion,
   symmetric translation+hybrid composition, provenance, per-bucket metrics. The eval harness
   (RAGAs, A/B, precision/recall) drives this exact seam, so measured and shipped retrieval
   are one code path — no separate eval rig.
2. **Agent entrypoint** (`answer(question, thread_id)`) — tool-selection and combined-query
   orchestration. Runs the **real agent against a small fixture collection** (not the full
   index) with **external tool data mocked**; `retrieve()` real.
3. **Streamlit app via `streamlit.testing.v1.AppTest`** — session/rendering only: thread_id
   stability across reruns, distinctness across sessions, fresh-uuid+surviving-agent on
   start-over, toggles, panels render. **The agent is stubbed entirely — no LLM calls.**
   Lands as `test_app_state.py` in Phase 3; `test_app_smoke.py` is the Phase-0 subset
   (page renders, a message reaches the agent seam).
4. **Security gate** — normalize/regex tested as **pure functions**; the classifier layer via
   **mocked responses in unit runs**, live only in the cached security-suite evals. Indirect
   injection asserted against the dedicated test collection (obeys nothing / no prompt leak);
   advice-refusal on the output validator.
5. **Tool functions** — chiefly `calculate_ratios` peer-average math (ADR-0009) and cache
   behavior, deterministic, `get_stock_data` mocked.
6. **Ingestion section-detection** — the Phase-1 data-quality gate (found / non-empty /
   length-bounded / body≫heading / starts-at-its-own-heading / stops-before-the-next-Item,
   plus the 10-K-filer and recorded-pointer-filer checks — ADR-0007 amendment), failing
   loudly.

## Out of Scope

Tier-2 (skill-stretch, built only after the Tier-1 gate, cut if undefendable — ADR-0010, in
priority order): deployment + live URL, re-ranking (cross-encoder as a constrained third A/B
axis), MCP client + tools-as-MCP-server, multi-model support, real-time KB refresh, and the
generic tail (auth + watchlists, multi-format export, analytics dashboard, scheduled KB
updates, rate limiting, help guide, multi-language toggle). Also out of scope: adaptive
per-query strategy routing (ADR-0005); out-of-Universe peers (ADR-0009); full-filing
ingestion and table/figure fidelity (ADR-0007); 10-Q filings; foreign private issuers
(20-F), which have no Item 1A/7/7A and so cannot supply a Section (ADR-0007).

## Further Notes

- Governing scope split and the Tier-1 gate: ADR-0001. Build order: PLAN.md §6, phases
  1 → 2 → 3 → 4 → 5 → 6 (logging) → 7 (evaluation) → **gate** → 8 (Tier-2).
- Known limitations for the review reflection: single-language KB; faithfulness vs.
  useful-but-uncontexted trade-off; yfinance as an unofficial API; Universe fixed at ingest
  and restricted to 10-K filers (20-F filers out of scope — ADR-0007); no re-ranking in
  Tier-1.
- Domain vocabulary is fixed in `CONTEXT.md` (Brief, Universe, Peer cluster, Peer, Filing,
  Section, Bucket, Golden set) — use it in ticket titles and test names.
