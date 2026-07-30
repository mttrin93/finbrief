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

---

## Amendment (ticket T11, issue #12) — two of the demoted mediums come back

**What this reverses.** The eight "polish" mediums above were demoted to Tier-2 because they
"reinforce none of the core", and two of them are named in that list: the **cost-meter UI** and
**rate limiting**. Both are now built. The reasoning that demoted them is unchanged and still
correct — neither reinforces the retrieval or evaluation work, and neither would have been worth
displacing any of it. **What changed is the constraint, not the argument:** the deadline moved,
the Tier-1 gate has been passed, and each of these turned out to cost under an hour because the
substrate was already there.

- **Cost meter.** T8 already logs per-field token counts on `agent_turn` and `query_translation`,
  so this is a reader and a sidebar panel, not an instrument. Recorded where the cut lives:
  ADR-0011 declined "dollar cost and the cost-meter sidebar" explicitly and now carries the
  amendment reinstating it, including why no rate card ships in the repo.
- **Rate limiting**, as a per-session question counter. Recorded here rather than in ADR-0006,
  and that placement is the point: **it is not a security control.** A refresh mints a new
  `session_state` and therefore a new counter, so it bounds what one open tab can spend and stops
  nothing that is trying. ADR-0006's input gate is the security boundary. Conflating the two is
  the more dangerous mistake in the pair, because a reviewer who reads a session counter as rate
  limiting stops looking for the thing that is — so `config.MAX_QUESTIONS_PER_SESSION` says so at
  length, the banner the user sees gives cost as its reason, and a test asserts that the banner
  does not describe itself as security or as a rate limit.

**What real rate limiting would need**, so the gap is stated rather than implied: a bound keyed
server-side on something the client does not choose — an account, an IP, a token bucket in a
shared store — which needs the auth ADR-0010 §6 still defers. That is a ticket, not a constant.

**The other six remain cut**, and the cut-if-undefendable rule still applies to them: multi-model,
MCP client, real-time updates, auth + watchlists, analytics dashboard, and the multi-language
toggle. Export and the help guide were also on that tail and shipped in this ticket; nothing else
did.
