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
