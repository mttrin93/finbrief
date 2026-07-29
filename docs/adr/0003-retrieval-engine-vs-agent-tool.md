# ADR-0003: Retrieval engine separated from the agent's use of it

`create_agent` is an agentic tool-calling loop (nondeterministic tool use), but the
headline RAGAs/A-B numbers need deterministic `(question → contexts → answer)` triples.
Measuring those through the full agent loop would make the retrieval strategy
un-isolable; measuring a rigid pre-agent step would break the combined demo queries that
need retrieval interleaved with finance-tool calls.

**Decision.** Separate the retrieval *engine* from the agent's *use* of it.

- **`retrieve(question, strategy, k) → (contexts, scores)`** — a standalone component with
  query translation + hybrid search *inside it*, selected by a config flag. Run at
  temperature 0 with fixed `k` in eval mode. The eval harness calls this **directly**,
  question by question, to produce clean RAGAs/A-B triples. **This is what the headline
  numbers measure.**
  **Deterministic except for the sub-query planner** — the `±translation` arms run one chat
  completion, so they are reproducible only up to its temperature-0 sampling while the
  `−translation` arms are exact. This ADR first wrote "deterministic" flat; ADR-0004 §9 records
  the split, why temperature 0 is not a guarantee, and the T10 harness design that makes all
  four arms reproducible anyway (issue #6 review).
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

**Closed by T6 (issue #6).** All three sentences above are now history: `retrieve()` implements
`hybrid` and takes `translate` as a second caller-named switch, `BASELINE_STRATEGY` is deleted,
`build_agent` reads `settings.retrieval_strategy` / `settings.query_translation_enabled`, and
the sidebar names **one** configuration because the configured one is what answers. What
survives from this section is the rule it was written to protect — a number is never reported
against a configuration nobody selected — which is now enforced by the switches being arguments
with conservative defaults rather than values read from config inside the engine.

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

---

## Amendment (ticket T4, issue #7) — the verbatim rule has exactly one exception

The decision above stands, `search_filings` now exists, and building it found one case the
verbatim rule as written cannot serve.

**1. Verbatim, except that a reference must be resolved — stated in exactly one place.** §1 says
the tool's description instructs the agent to pass the user question verbatim. A conversation
breaks that: *"And what does it say about its debt?"* names no company, and passed through
unchanged it embeds a question about nobody. Nothing inside `retrieve()` can fix this — the
engine is stateless and the evaluation harness only ever hands it self-contained questions,
which is the same split this ADR is built on. So the description permits exactly one edit, and
says so: replace a pronoun or elliptical reference with the company or subject it refers to, and
change nothing else. Rewriting, keyword expansion, added tickers and decomposition remain
forbidden, because those are the *optimizations the tool owns* and doing them twice is the
failure §1 exists to prevent.

The description is the **only** statement of that rule. A first pass also paraphrased the
exception in `AGENT_SYSTEM_PROMPT` ("work out which company it means from the conversation"),
which is one rule in two wordings — and a model handed two wordings of a constraint is being
invited to satisfy the looser one. The system prompt now names the description and stops there
(issue #7 review); the description's words live in `prompts.py`, which CLAUDE.md makes the owner
of anything the model reads, and its scope sentence is derived from `config`/`Section` like every
other rather than typed.

**2. The rule is measured, not enforced — and §2's logging is the whole of what makes it a
claim.** This is the honest reading of the acceptance criterion, and worth stating plainly
because it is easy to tick: nothing in the code *prevents* the agent from rephrasing. The tool
performs no pre-processing of its own (`test_the_query_reaches_the_engine_unchanged_under_the_shipped_strategy`
pins that half, which is the half we control), and what the model chooses to pass is then
recorded rather than corrected. `agent.answer` emits one `agent_query` line per search with a
`verbatim` verdict, plus an `agent_turn` summary — verdicts and lengths only, never the text of
either query, since a question is user content and these lines are kept. `verbatim` compares
stripped strings: whitespace is not a translation, and translation is what this ADR asks us to
count. T10 (#11) reports the rate.

An enforcing alternative was available and is **rejected**: overwrite the model's `query`
argument with the user's question before calling `retrieve()`. That would make the criterion
true by construction and break every follow-up, since the one edit §1 permits is exactly the
one such a guard cannot tell from a rewrite. Measuring a prompt we cannot enforce is the
honest option; asserting compliance we never checked is not.

The instrumentation immediately earned itself. On the first live two-turn run
(`openai/gpt-4o-mini`, 2026-07-27) the model diverged on **both** turns — `main risk factors for
Tesla`, then `Tesla debt`, the second a keyword reduction the description explicitly forbids —
and the second query returned four Ford chunks out of five (#6 has the case; #11 has the
metric). So the shipped path's verbatim rate at the time of writing is **0 of 2 searches**, on a
sample of one conversation. Per this ADR's own position that is a **finding, not a defect to
tune away**: the prompt stays as written and the rate gets reported, because a prompt tuned
against a paid model until the number looks good is a number about the tuning.

**3. Citation numbering is the *agent's*, not the tool's.** `retrieve()` ranks 1…k on every
call, so a second search in one conversation would reuse `[1]` for a different chunk and every
marker in the transcript above it would stop resolving — the property user story 2 rests on.
Something must renumber into the thread's running sequence.

T4 first put that in the tool, offsetting each result by the number of sources the thread had
already issued. **That cannot work, and the review caught it.** LangGraph's tool node builds
every `ToolRuntime` from the same node input and *then* runs a step's calls concurrently, so two
`search_filings` calls in one step read an identical offset and both number their chunks
`[1…k]`. Nothing raises; the model is simply handed two source blocks with the same numbers in
them. Only the prompt stood between us and that — and §2 above is the record of this model
declining a plainer instruction than "do not split it into several searches".

So the register is assigned at the agent seam (`agent/citations.py`), in one sequential pass over
the thread's search replies, after a step's tool messages have returned and before the model
reads them. Collisions are not prevented there; they are unrepresentable, because no two callers
compute a number independently. `search_filings` went back to being stateless — query in, framed
chunks out — which is what this ADR asks a wrapper to be, and the engine is untouched:
`Context.rank` still arrives 1…k and what the harness measures is unchanged. What this costs is
that a chunk retrieved twice in one conversation is numbered twice; both numbers resolve to the
same source, so a citation stays checkable.

**4. What the agent's turn returns is not a `GroundedAnswer`.** `agent.answer` returns an
`AgentTurn` carrying the searches it ran, not just their chunks, so a surface can tell "searched
and the collection returned nothing" from "answered from the conversation". `GroundedAnswer`
stays the chain's type. Conflating them would let agent output reach the harness that is
supposed to measure the chain, which is the one confusion this ADR is written to prevent.

**5. One tool call per step, asked for at the binding.** The model is bound with
`parallel_tool_calls=False` (through `create_agent`'s `model_settings`, which it spreads into
`bind_tools`). Two reasons, and neither of them is §3 — the register no longer needs this to be
true. First, a step that fans out into several searches makes the `verbatim` verdict of §2
ambiguous: several queries against one question, none of them the question. Second, splitting a
question up *is* decomposition, which ADR-0004 puts inside `retrieve()`, so a fan-out is the
double-translation §1 exists to prevent arriving by another route — and the tool's description
already forbids it in words the live run shows are not binding.

It is a **request**, and recorded as one: the flag reaches OpenRouter, which fronts many
upstreams, and whether a given one honours it is not something we can assert. That is why it is
the second line and not the first. §3's register is what makes the failure impossible; this only
makes it rare, which is worth having for the measurement but is not what the correctness rests
on.

---

## Amendment (ticket T10, issue #11) — what the harness measures, and the four deferrals' status

**The validity gap this ADR commits to stating is stated, and it is now stated with a mechanism
behind it.** The headline RAGAs and A/B numbers come from `rag.answer_question` and `retrieve()`
driven directly, question by question, exactly as §1 requires. Two details are worth recording
because they are the seam holding.

**1. The harness drives the chain unmodified, and checks that it did.** Scoring faithfulness over
an arm's contexts needs an answer generated over *those* contexts, and `answer_question` owns its
own retrieval — so the obvious shortcut is to call the generation half directly over the contexts
the arm was scored on. That was written and then rejected: it is a second code path through the
thing being measured, which is exactly what keeping `rag.answer_question` callable exists to
avoid. Instead the chain runs with the arm's own configuration and its replayed planner, and the
contexts it returns are compared against the ones the arm was scored on; a mismatch raises
`ContextDrift` and stops the run rather than scoring an answer against contexts the scored
retrieval never surfaced. The cost is one extra embedding round per cell, accepted and recorded.

**2. The `tool-augmented` bucket is scoreable on faithfulness *because* of this split.** A chain
answer carries no tool-derived sentence — the chain calls no tools — so ADR-0002's amendment's
worry about a depressed number does not arise at this seam. The full reasoning is in that
amendment's T10 entry; what belongs here is that it is a consequence of §1's separation rather
than a special case bolted onto it.

**The four deferrals earlier tickets handed T10, and their honest status.** Three of the four need
live *agent* turns rather than chain runs, which is a different (and nondeterministic) instrument
from the one this ADR's numbers come from — so they are reported beside the RAGAs table, never
inside it, and an unrun measurement is named as unrun rather than left to be assumed:

Requoted from [`docs/verification/evaluation.md`](../verification/evaluation.md). **This table
said "not measured by this ticket" in all four rows after three of them had been measured** — it was
written before the live stage existed and never revisited, which is the stale-status failure the
paragraph below is about, committed inside the amendment that warns against it (code review of #11).

| deferral | from | instrument | status |
|---|---|---|---|
| agent-vs-original query divergence rate | T4, §2 above | `agent_query.verbatim` in a live log | **100% (8/8)**, all first-searches |
| bracket-rule adherence rate | T5 | `citation_markers` in a live log | **not measured** — emitted at the app, unreachable by a harness |
| layer 4's residue — advice no rule matches | T7 | probes through the live agent + `validate_answer` | **100% (6/6)** not refused |
| faithfulness on markers that resolve but sit on unsupported claims | T3/T5 | per-sentence NLI against the *cited* chunk | **31% (22/70)** pairs fully supported |

Three of the four are measured and the fourth is named as unmeasurable by this instrument rather
than as unrun: `citation_markers` is emitted by `app/Home.py` and by nothing else, so a harness that
drives the agent directly produces the turns and none of the lines (ADR-0011's T8 finding, second
instance). **The divergence denominator is deliberately small and deliberately honest**: 8 searches
from this run's own window of the log, not the 40 the first artifact reported by pooling 13 appended
runs. All 8 are first-searches, so the split §2 asked for has no follow-up half to report — the
tool-calling eval opens a fresh thread per case, which is what makes each case independent and also
what removes the one place a *permitted* rewrite could occur. A rate over 8 first-searches is what
§2 asked for at the size the instrument can currently deliver, and the artifact prints the
denominator beside it.

Saying so is the point: an unscored criterion reads as a passed one, and the honest form of "we did
not get to it" is a row in a table rather than an omission — which only works if the row is kept
current.
