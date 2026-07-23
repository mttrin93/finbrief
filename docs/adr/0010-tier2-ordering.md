# ADR-0010: Tier-2 ordered by GenAI skill signal; re-ranking promoted

Tier-2 exists to fill a GenAI/RAG gap, so the ordering test is skill signal, not the
assignment's easy/medium/hard tags. Most of the original Tier-2 list is generic web-app
work; the highest-value RAG item (re-ranking) was buried in §9 as a "next improvement."

**Decision.** Order Tier-2 by GenAI/RAG skill signal, with one exception:

1. **Deploy + live URL** (portfolio exception) — reach gates the value of everything else;
   it's the artifact recruiters click.
2. **Re-ranking** — cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scoring the RRF-fused
   top-N to top-k. Promoted from §9 to first-class. Evaluated as a **constrained** third
   A/B axis: shipped default ± rerank only (not a full factorial), per-bucket in the
   existing harness. Pre-registered hypothesis: context-precision lift concentrated on the
   `semantic` bucket, ~neutral on `exact-identifier`. Latency under the same budget regime
   as translation; the local-model deploy weight is the accepted cost — justified by skill
   signal (contrast ADR-0006, where an *unjustified* extra model call was cut; same cost
   test, opposite answer).
3. **MCP client** (remote public server + security review), then **own tools as FastMCP
   server**.
4. **Multi-model** support.
5. **Real-time KB refresh**.
6. **Generic tail, only if appetite remains** (each still subject to cut-if-undefendable):
   auth + watchlists, export, analytics dashboard, rate limiting, help guide,
   multi-language toggle.

**Consequence.** The cut-if-undefendable rule bites hardest on the tail; the on-target RAG
work (rerank, MCP) is done first, so if time runs out the skill-signal items already exist.
