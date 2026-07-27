# ADR-0003: Retrieval engine separated from the agent's use of it

`create_agent` is an agentic tool-calling loop (nondeterministic tool use), but the
headline RAGAs/A-B numbers need deterministic `(question → contexts → answer)` triples.
Measuring those through the full agent loop would make the retrieval strategy
un-isolable; measuring a rigid pre-agent step would break the combined demo queries that
need retrieval interleaved with finance-tool calls.

**Decision.** Separate the retrieval *engine* from the agent's *use* of it.

- **`retrieve(question, strategy, k) → (contexts, scores)`** — a standalone deterministic
  component with query translation + hybrid search *inside it*, selected by a config flag.
  Run at temperature 0 with fixed `k` in eval mode. The eval harness calls this **directly**,
  question by question, to produce clean RAGAs/A-B triples. **This is what the headline
  numbers measure.**
- **`search_filings`** — the same `retrieve()` wrapped as one tool the agent can call
  alongside `get_stock_data` / `get_recent_news`. This is what ships and what makes the
  combined demo queries work. It calls `retrieve()` with the **default** strategy — the
  configuration that was measured.

**Keeping the gap small and honest.**
1. The `search_filings` tool description instructs the agent to pass the **user question
   verbatim**, because the tool owns query optimization — preventing double translation.
2. We **log agent-issued queries against original user queries**, so the README reports
   how often the shipped path diverges from the measured path rather than assuming it
   doesn't.
3. Multi-hop questions are decomposed *inside* `retrieve()` in eval mode. If the agent
   instead answers a multi-hop question via multiple tool calls, that is the selection
   layer working — it surfaces in the tool-calling eval, not as chain noise.

**Consequence.** Agentic in the app, deterministic under evaluation, single code path.
The README states the validity gap explicitly: RAGAs/A-B measure the retrieval chain; the
tool-calling eval measures the agent's selection layer on top.

---

## Amendment (ticket T3, issue #5) — what building `retrieve()` changed about its signature

The decision above stands. Three details of the signature written into it did not survive
implementation, and are recorded here rather than left as a docstring the next reader has to
find (issue #5 review).

**1. It returns one sequence, not `(contexts, scores)`.**
`retrieve()` returns `tuple[Context, ...]`, each `Context` carrying its own `distance` and
`rank`. Two parallel lists desynchronise the moment a caller sorts, filters or dedups one of
them — which is precisely what this ADR's own fusion step does (ADR-0004: "candidate lists
are fused with Reciprocal Rank Fusion, deduplicated by chunk id, truncated to top-k"). One
object per retrieved chunk makes that mispairing unrepresentable, and Phase 4's provenance
(variant × retriever × RRF contribution) has somewhere to live that cannot drift from the
chunk it describes. The ADR's "scores" is `Context.distance`.

**2. The score is a distance, and is named one.**
`filings` is an L2 collection, so what Chroma returns is a squared distance — **lower is
nearer** — not a normalised similarity. Calling it a `score` invites every reader to assume
0…1 and higher-is-better. Rejected `similarity_search_with_relevance_scores`, whose rescaling
assumes normalised embeddings and would manufacture a similarity nobody has verified. RRF
consumes `rank` regardless, which is the point of RRF.

**3. `hybrid` raises until Phase 4; the shipped path names its baseline.**
`DEFAULT_STRATEGY` is already the pre-registered `hybrid + translation` (ADR-0005), fixed
before any A/B data exists — so in Phase 2 the configured default is a strategy that does not
yet exist. `retrieve(strategy=hybrid)` therefore raises `NotImplementedError` rather than
quietly serving vector results under a hybrid label, since a number reported against a
strategy that never ran is worse than a missing number. The app does not read the setting
either: `agent.BASELINE_STRATEGY = vector` is what runs, and the sidebar states it *beside*
the configured value rather than in place of it, so the UI cannot advertise the unmeasured
default. Phase 4 deletes that constant and reads the setting.

**Also true of the implementation, and not visible in the signature.**

- `store` and `settings` are injectable keyword arguments. That is what lets the eval harness
  point at a throwaway index (ADR-0002) and the suite run against a fixture collection with a
  deterministic fake embedding — the ingested `data/chroma` cannot be a test fixture, because
  it is only searchable by the paid model that wrote it.
- The Chroma call itself stays in `retrieval/vectorstore.py` (`nearest_chunks`), which
  CLAUDE.md makes the only place the collection is opened, written or read. `retrieve()` owns
  what a retrieval *means*; the collection's API is crossed in one file.
- The deterministic answer chain is `finbrief.rag.answer_question`, **not** the agent module.
  It is what produces this ADR's `(question → contexts → answer)` triples, so Phase 3 replaces
  the body of `agent.answer` and leaves the chain where the harness calls it.
