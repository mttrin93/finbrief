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

## Amendment (ticket T12, issue #13) — four of the demoted items come back

**What this reverses.** The eight "polish" mediums above were demoted to Tier-2 because they
"reinforce none of the core", and four of them are now built: the **cost-meter UI**, **rate
limiting**, **export** and the **help guide**. The reasoning that demoted them is unchanged and
still
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
- **Export**, as JSON and CSV from the display transcript. **PDF declined** on dependency grounds
  rather than effort — recorded against PLAN §2's own *Conversation export (PDF/CSV/JSON)* row,
  which is the Medium that asked for all three.
- **The help guide**, as a sidebar panel plus four example-question buttons. PLAN §2's Easy row
  also asks for a `/help`-style command, which is **not built** and is recorded as open there.

**What real rate limiting would need**, so the gap is stated rather than implied: a bound keyed
server-side on something the client does not choose — an account, an IP, a token bucket in a
shared store — which needs the auth ADR-0010 §6 still defers. That is a ticket, not a constant.

**The other four remain cut**, and the cut-if-undefendable rule still applies to them:
**auth + watchlists**, the **analytics dashboard**, **scheduled KB updates** (GH Action), and the
**multi-language toggle**. That is PLAN §6 Phase 8's generic tail — eight items, four shipped
here, four open.

*Both counts above were wrong when this amendment was written, and the correction is recorded
rather than quietly applied* (code review of #13). It was headed "two of the demoted mediums come
back" while its own closing paragraph conceded that export and the help guide shipped in the same
ticket — four, not two. And it read "the other six remain cut", naming multi-model, MCP client and
real-time updates: those are *numbered* Phase-8 items (2–5), not tail items, so the list mixed two
different sets and, in doing so, dropped **scheduled KB updates** — a cut item that this ADR is
the record of, no longer recorded anywhere as cut. An ADR whose job is to say what was cut has to
get the arithmetic of its own list right; the numbered Phase-8 items above the tail were never
reversed and PLAN §6 tracks them.

---

## Amendment (ticket T14, issue #15) — **multi-model**, the first *numbered* Phase-8 item to return

**What this reverses.** Every reversal above was a tail item. This one is
**Phase-8 item 4** — one of the numbered items the T12 amendment's own correction is careful to
say "were never reversed". It is now built: a sidebar picker over four OpenRouter models,
answering only.

The argument that demoted it is unchanged and still correct. Multi-model reinforces neither the
retrieval nor the evaluation work, and it would not have been worth displacing any of it —
**it is the same trade as the four tail items, and it came due for the same reason**: the Tier-1
gate is long passed, and the substrate turned out to be already there. `llm.build_chat_model` has
taken a `model=` override since T3, OpenRouter is one base URL for every upstream, and ADR-0008
already names the model picker as session state in its own decision text ("`st.session_state`
holds only the `thread_id`, UI toggles (model, strategy), and the display transcript"). What was
actually built is a widget, one cache key, one log field, and the honesty rules that follow.

**Which is where the interesting part is, and it is not the picker.** Three of the five reversals
above shipped no new measurement — that is what made them cheap. This one *changes an existing
one*, and the change is a subtraction: the cost meter's two price knobs describe **one** model, so
a conversation answered on another cannot be priced at all. ADR-0011's refusal to ship a rate card
is what forces that (OpenRouter routes by availability, so a price in this repo is a figure nobody
measured going stale), and four selectable models make the refusal stronger rather than weaker.
The rule is recorded there, in ADR-0011's own amendment, beside the reinstatement it qualifies:
**price only when every metered turn ran on the priced model; otherwise report the tokens and say
why.** A wrong dollar figure is worse than no dollar figure, and this one would have been invisible
— the tokens behind it are real.

**PLAN §6 item 4's open question is answered by scoping, not by a measurement**, and the
distinction matters because the question was a real one: *the injection classifier and the
agent/tool-calling prompts are tuned per model — does swapping silently break tool-calling?* Half
of it dissolves. The **classifier is not selectable** (`Settings.classifier_model` keeps its own
field), so ADR-0006 layer 3 is untouched by the picker and no gate claim depends on which model
answers. The other half is real and is handled by bounding it rather than clearing it: all four
slugs are tool-calling models, a model that fans out badly is slower rather than wrong
(`MAX_AGENT_STEPS` bounds the loop either way), and **no per-model tool-calling eval ships**.
T10's tool eval ran on the default and that remains the measured configuration — a per-model
number would mean the harness run four times over, which is four times the spend for a Tier-2
widget. Stated here so the gap is a decision and not an omission.

**Also recorded in ADR-0008**, whose two-session isolation guarantee this touches: the picker adds
a cache key, so the process now opens **up to four** SQLite connections where it opened exactly
one. Isolation is unchanged — it was never carried by there being one agent — and that was
measured before anything was built rather than argued from the code.

**The three that remain cut** after this: the **MCP client** and **tools-as-MCP-server** (Phase-8
item 3 and its tail), **real-time KB refresh** (item 5), and from the generic tail
**auth + watchlists**, **scheduled KB updates** and the **multi-language toggle**. Deploy (item 1)
is unbuilt and still the highest-value remaining item. The cut-if-undefendable rule still applies
to all of them.
