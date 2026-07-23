# ADR-0001: Two-tier scope — review-facing core, then skill-stretch

The Sprint 2 assignment estimates 20–25h and awards maximum points for just "2 medium +
1 hard" optional tasks, with a review that rewards *depth you can explain* over breadth.
This project is also deliberately being used to fill a GenAI/RAG gap in the author's
profile, so exceeding the time envelope is acceptable in service of that.

**Decision.** Split scope into two tiers with a hard gate between them:

- **Tier-1 (review-facing):** P0 core + hybrid search + RAGAs + A/B + query translation +
  security gate + structured logging. Must be finished, evaluated, and review-defensible
  before any Tier-2 work begins.
- **Tier-2 (skill-stretch):** everything else (see PLAN.md §6 Phase 8). Built only after
  the Tier-1 gate, each fully understood; anything not defensible is cut before submission.

**Why these are the Tier-1 set.** They reinforce each other rather than standing alone:
query translation and hybrid search are the two retrieval strategies that A/B compares;
RAGAs supplies the metrics that make the A/B meaningful; structured logging is the
substrate RAGAs, A/B, and the security-gate catch analysis all read from — so logging
runs *before* evaluation to avoid re-running it. The eight independent "polish" mediums
(multi-model, MCP client, cost-meter UI, export, help, rate limiting, real-time updates)
reinforce none of the core and were demoted to Tier-2.

**Consequence.** Total effort is expected to exceed 20–25h. The gate is what keeps that
from becoming un-scoped: the review-facing project is complete at the end of Phase 7
regardless of how much Tier-2 gets done.
