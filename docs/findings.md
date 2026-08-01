# What this project found out

The long version of [the README](../README.md)'s Part 4. Four findings, each the
strongest material the work produced, and none of them a retrieval number.

---

## 1. What the evaluation established

**The pre-registered default is `hybrid + translation`, and it is retained because the experiment
could not resolve the question — not because it was validated.** Every figure here is requoted
from [`evaluation.md`](verification/evaluation.md).

- **Hybrid earns nothing detectable on any bucket.** `hybrid − vector` at equal translation is
  *not detected* on `semantic` and *undetectable at this n* on the other three.
- **The point estimate on its own predicted bucket is negative**: on `exact-identifier`, the
  bucket hybrid exists to win, Δ **−0.087**.
- **The one root-caused live case favours the simpler arm**: `vector + normalisation` put the
  target chunk at rank 1 against the default's rank 2.
- **Translation costs 3212 ms p50 against a 1500 ms budget**, recorded as missed and left
  unamended.
- **The clause that protects the default could not have fired on any bucket**, and the §4 trigger
  fired on the one bucket that had the power to say anything.

So `vector + translation` is a live candidate this run could not rule out — it is simpler and it
costs no BM25 index over the whole corpus. The default does not move on this evidence because *no*
configuration is preferred on this evidence, and moving a pre-committed default on an unresolvable
comparison is the same error as keeping it on one. What would settle it is a larger per-bucket `n`
and nothing else.

**And the first version of that whole determination was a tautology.** The original comparator
judged a difference of *means* against `basis` — the larger of the two arms' own per-question
range. But each arm's mean is bounded by that arm's own min and max, so the largest delta the
observed values can produce is `max(cand.max − base.min, base.max − cand.min)`; and when two arms
score *the same questions* under near-identical configurations, that quantity **equals the
range**. The test was `abs(delta) <= basis`. **It could not fail.**

The consequence was not small. All **18** pre-registered comparisons in the first committed
artifact — six hypotheses, eight falsification-clause cells, four trigger cells — were rendered as
verdicts by an instrument mathematically incapable of returning any other. Four hypotheses were
published as **refuted** on that basis. H2 was refuted while its delta ran +0.102 in the
*predicted* direction, leaving ADR-0004 §6's root-caused live case standing unreconciled beside
it. Both of ADR-0005's pre-registered decisions were tautologies: the falsification clause could
not fire, and the absence-triggered test could not fail to fire.

**How it was found matters more than the fix.** By a fresh-context review reading the artifact's
own printed `[min–max]` beside each mean and doing the subtraction nobody had done. The
information needed to catch it was in the committed evidence the whole time. The pre-registration
was sound — written pre-data, in code, refusing fenced metrics; **the instrument judging it was
not**, and those are separable failures. A pre-registered prediction is only as good as the test
that settles it.

The replacement pairs on the **question** — the arms score the same rows, so the per-question
difference removes the question's own difficulty, the term that dominated the raw spread — and
settles direction with an **exact** Wilcoxon signed-rank test whose null is built by convolution
rather than approximated. Three outcomes, not two, and `Paired.detectable` is what keeps
"we could not have seen it" from reading as "there is nothing there".

None of that is a defect in the pipeline. It is a measurement that came back saying *we cannot
tell*, reported as that instead of as a result.

## 2. Which half of the pipeline is deterministic

Two findings from the same runs answer this precisely, and they point opposite ways.

**Retrieval is exactly reproducible, and that is the strongest empirical claim in the project.** A
code review changed the retrieval cache key, so all **168** retrieval cells — 28 questions × six
arms, baselines, translated arms and both planner-off ablations — were re-paid from scratch
against the same collection. Everything downstream then **replayed: 672 cells with zero misses**,
being 560 judge cells (all four metrics) and 112 answer cells. Those keys are not identifiers: the
judge key carries the **full text of every retrieved context**, and the answer key carries the
chunk ids plus a sha256 of the context bodies. A single character different in any chunk of any
cell, on any arm, and that cell would have missed and been re-paid. None did.

So: given the same collection and the same question, `retrieve()` returns the same chunks in the
same order, byte for byte, on every configuration this project ships or ablates. This is the third
independent confirmation and the first covering the whole matrix.

**The planner is not, and the same runs measure that too.** ADR-0004 §9's n-repeat asks the
planner for sub-queries five times per question at temperature 0. Three successive runs of that
pass reported **0, 1 and 2 of 8** questions returning an identical set every time. The count is
itself re-sampled, because the pass makes live planner calls and therefore **inherits the variance
it is measuring** — which is why it is quoted as a range and never as one run's figure. The
committed artifact's own run says 2 of 8.

**The conclusion is identical in all three, and that is what makes it usable: 6, 7 and 8 of 8
questions varied.** So the resolve-once replay that the two `+translation` arms depend on is
**load-bearing, not caution**: without it those arms would report different per-bucket numbers on a
re-run with no code change. ADR-0004 §9 registered the opposite outcome as possible and called it
"not one to assume"; it was right not to.

Temperature 0 is greedy decoding, not a determinism guarantee. OpenRouter fronts many upstreams
and no `seed` is sent — a `seed` was considered and rejected, because the OpenAI-compatible
parameter is best-effort even at its origin and a request routed elsewhere ignores it entirely, so
adding it would buy a little stability and a false claim.

**Together they locate the nondeterminism exactly: it is in the model calls, not in the
retrieval.** Everything between the query variants and the ranked chunks is reproducible; the
planner that writes those variants is not, and neither is the judge — response relevancy is
re-sampled every time it is judged, which is why the artifact fences that column off from every
pre-registered decision.

## 3. The recurring theme: a claim the thing making it could not check

The same defect, **thirty-one** times in thirty places — tiktoken's warm cache did it twice —
across ingestion, retrieval, security, evaluation, instrumentation, the UI and process. They look
unrelated apart, which is why they are listed together.

| where | it asserted | what it had established |
|---|---|---|
| tiktoken's warm cache (twice) | a hermetic suite | nothing: it passed locally and egressed in CI |
| the egress guard's docstring | `curl_cffi` coverage | nothing — `curl_cffi` resolves and connects in C, touching `socket` not at all |
| a conftest comment | that `gethostbyname` routed through `getaddrinfo` | nothing: each is its own CPython entry point, and only one was patched |
| `serial_full_brief < MAX_AGENT_STEPS` | that the step ceiling was pinned | nothing: 15, 20 and 24 all satisfy it |
| `SuiteRun.passed` via `all(())` | **SUITE PASSED**, exit 0 | nothing: an empty suite is vacuously true |
| the first indirect-injection run | "not obeyed, no leak" | nothing: the agent asked which company was meant and never retrieved the payload |
| layer 4's `advice_hits` | that advice was refused | nothing after the first match per rule: *"I can't give a price target. My price target for NVDA is $260"* returned no hits |
| layer 3's `_verdict` | a verdict | nothing on `**YES**` — not in the strip list, so *undecided*, and undecided allows |
| `metrics.compare` | a verdict on each of 18 pre-registered comparisons | nothing: a delta of means tested against the arms' own range, which is the largest delta those values permit |
| the tool eval's control **C3** | a pass, inside a published **100% over 10 cases** | nothing: no expected tool, no forbidden tool, no argument, so `passed` was `True` for every possible behaviour |
| the judge stage's error path | `APIConnectionError: Connection error.` | nothing about the network: one client reused across per-cell event loops, and `httpx` raised `bound to a different event loop` |
| the first fix for that | a fresh client per cell | nothing about the defect: `langchain_openai` caches the async client *below* this repo's constructor, so distinct objects shared one pool |
| **the figure-binding test itself** | that every quoted figure is in the artifact | nothing, for short figures: it matched substrings, so `"8"` is satisfied by `18`, by `0.087`, by a date |
| "1324 tests green" | that the suite passed | nothing CI had seen: that commit was pushed inside a later push, so only the tip was built |
| `log_turn` at the app | turn-id propagation | nothing about wiring: neutralising it left all 1028 tests green, because every test opened the scope itself |
| `citation_markers` | a bracket-adherence denominator | nothing a harness can reach: emitted by the page and by nothing else |
| the spend panel's classifier caveat | that the unmetered call is named | nothing on a complete total: it shipped as a clause of the *partial* banner |
| the model picker's isolation test | that the picker preserved two-session isolation | nothing about the picker: isolation is carried by the per-session `thread_id`, so it passes with the cache key dropped — confirmed against both mutations |
| `by_model`'s tie-break docstring | a total ordering, unattributed slice leading | nothing: no test reached it, and reversing the sort key broke nothing |
| the per-model table's row order | the order of the rows | nothing: `== [A,B] or == [B,A]` is satisfied by every order two rows can take |
| `all_answered_on`'s docstring | that importing `spend.py` was forbidden | nothing — the same file imports `calls_behind` from it four lines above |
| "checking costs a paid call" | that the model list could not be verified cheaply | nothing: `GET /api/v1/models` is public and free, and two of the four committed slugs were 404s |
| `model_reported` | that a routing surprise is "visible rather than silent" | nothing: emitted, round-tripped by a test, and read by no surface |
| the sidebar's unpriced caption | that another model answered | nothing on a planner-only conversation, where nothing had answered at all |
| the first-party criterion | that a first-party model is reachable | nothing: `x-ai/grok-4.3` is first-party and was refused — the API key's allowlist governs |
| the reroute caption's silence test | that no reroute renders when the provider agreed | nothing: it matched `"served by …"` against a caption reading `"Served by …"`, so making the caption render on *every* log left 164 tests green |
| the `not recorded` caveat's conditional | that it renders only when such a row is on screen | nothing: pinning it to always render broke no test — only the presence half was checked |
| `_ms`'s "a zero would be a claim that a turn was instant" | that no fabricated zero reaches the table | nothing: returning `0` for an unmeasured latency left the suite green, and a turn with a model and no `latency_ms` reaches it |
| the second 404 banner | "OpenRouter does not recognise it" | nothing: it is the branch for every 404 whose body lacked two words, and its own test fed it a *restriction* message |
| the picker's `help=` | that switching cannot weaken the gate | nothing: a literal on a widget, with no owner in `prompts.py` and nothing binding it to `security/classifier.py` |

**The generalisation: a guard is code, and inherits every failure mode of the code it guards.**
Four of these sat *inside published measurements*, one sat inside the mechanism built to prevent
the others, and two were kept hidden by ordinary good practice — the **cache** meant the failing
path was never exercised, and the **retry** self-healed it so it failed somewhere different every
time.

**The last thirteen arrived together, on the smallest ticket in the project, and that is the
finding.** Multi-model support (#15) is a widget, one cache key and one log field — no new
instrument and no new measurement — and it produced **thirteen** instances, more than the entire
previous total from eleven Tier-1 tickets. Eight were found by code review; **five were found by a
person using the app**, none of them by the 1,710-test suite. Three of those five were successive
wrong diagnoses of *one* symptom, each written into a comment as
established before the next attempt disproved it: a slug list unverified because checking "costs a
paid call" (it is free), then a data-policy theory, then a first-party theory that its own
replacement refuted. The pattern is not carelessness in a hard place — it is what happens when code
makes claims about an environment it cannot observe, and the honest fix each time was to narrow the
claim rather than to strengthen the check. Which is the fourth rule this section now carries:

**And the second review round found five more in the fixes for the first**, which is the part worth
keeping. Every one was found by **mutating the code and re-running**, not by reading it: three
captions and a table cell whose *negative* half nothing held — the caption that must stay silent,
the caveat that must not render, the zero that must not be printed — plus a banner asserting a
cause it had not read and a widget's help text asserting a security property with no owner. The
asymmetry is the lesson and it is mechanical: **a conditional surface needs a test on each branch,
and the branch that renders is the one that gets written.** A test that only ever sees the panel
populated cannot distinguish "renders when it should" from "renders always". Three of these five
sat one line from a test that caught their opposite.

- **A claim about something outside the process is a hypothesis, and belongs written as one.** The
  model list is dated, says what its check does *not* establish, and names the account setting no
  test here can see.

Three rules follow, and they are the ones this repo now applies to instruments as well as to code:

- **Prefer a check that exercises the thing over one that describes it.** Every control in the
  tool eval is now driven against an agent scripted to call all three finance tools and asserted
  to *fail*; every hermetic test calls the backend it is about rather than asserting that one path
  routes through another.
- **Prefer an equality over a bound.** The figure-binding test matches whole numbers; the
  translation budget is pinned by an equality; the scope panel's line count is an equality,
  because "compress" that permits any number of bullets is not a constraint.
- **For any check about an adversarial input, assert that the input arrived.** `retrieved` is a
  required column on every planted payload, and a bypass found by review is **added to the
  corpus**, not merely fixed.

And two corollaries worth stating on their own. **A passing rate is only as strong as the weakest
case in its denominator** — "100% over 10" with one case that could not fail is a claim about nine
and one piece of padding. **A green local run is a hypothesis about CI, never a result from it.**

### Three libraries, three that phone home by default

The same shape in the dependency graph, and it changed how this project adds a dependency. Every
library added here **for quality or safety** ships with a telemetry path enabled:

| library | added for | what it sends, unconfigured | how it is switched off |
|---|---|---|---|
| `guardrails-ai` | the output validator | a record of every validated answer, to its own endpoint | `guard.configure(allow_metrics_collection=False)`, in `security/advice.py` |
| `uvloop` (via guardrails) | nothing — it arrives transitively | nothing itself, but it becomes the **process-wide** event loop and resolves DNS in libuv, outside a Python-level guard | `GUARDRAILS_RUN_SYNC`, plus the guard covers the backend |
| `ragas` | the four RAGAs metrics | a POST per metric completion, to `t.explodinggradients.com` | `RAGAS_DO_NOT_TRACK`, set in `evaluation/judge.py` **and** in `conftest.py` |

**The switch is in a different place every time** — an SDK method call, an environment variable
that must be set before a cached read, an event-loop policy — and only one of the three documents
it anywhere a reader would look. **The failure is silent by construction** — for the two rows that
POST, and by a different mechanism each: `ragas._analytics.track` is decorated `@silent` and fires
from a background thread and again at `atexit`, while guardrails' POST happens inside
OpenTelemetry's `BatchSpanProcessor`, which catches the exception on its own export thread and logs
it. So `guardrails-ai` and `ragas` both return a completely normal result while a packet is being
attempted, and a test asserting on the return value passes. Which is why
`conftest.EGRESS_ATTEMPTS` is the only detector that survives, and why it has now been the only
detector three times. And **off-by-default has to be set twice**, where the library is used and in
`conftest.py`, on the principle that a hole is a hole whether today's code walks through it.

## 4. Prompt rules are instruments, not enforcement

The strongest generalisable claim these measurements produced is not a retrieval number. It is
that **a rule stated in a prompt is a thing you can measure compliance with, and not a thing you
can rely on** — and it stands on six independent instances, one of them measured at scale.

| stated rule | where | what was measured |
|---|---|---|
| pass the user's question to `search_filings` **verbatim** | the tool's description (ADR-0003 §1) | **100% divergence, 8 of 8 searches.** Every query the agent issued differed from the question as typed — and all 8 are *first* searches, which cannot be reference resolutions, so none of them is the one rewrite the description permits |
| decompose a question into sub-queries when asked to | the planner's prompt (ADR-0004 §6) | the planner refuses, or returns prose refusals the parser has to strip |
| square brackets are reserved for retrieved excerpts | `AGENT_SYSTEM_PROMPT` | `[Yahoo Finance]` observed live; uncited grounded answers observed live |
| every figure comes from a tool or a retrieved excerpt | `AGENT_SYSTEM_PROMPT` | an answer naming Tesla's real segments from a chunk about industrial fasteners |
| a valuation question wants the quote **and** the peer comparison | `AGENT_SYSTEM_PROMPT` | **1 of 2** |
| these four headings structure the brief | `AGENT_SYSTEM_PROMPT` | read as **search terms**: `search_filings("Business")` named no company, retrieval returned six other filers' chunks, and the model wrote two sections from its weights with no `[n]` at all |

The verbatim rule is the sharpest, because every divergence is one the description forbids
outright, and the denominator is honest: **8** searches from one run's own window of the log, not
the 40 an earlier draft reported by pooling 13 appended runs. Per ADR-0003's standing position it
is **a finding rather than a defect to tune away** — a prompt tuned against a paid model until the
number looks good is a number about the tuning.

**The same shape holds one layer down, where the rule is code rather than prose.** Layer 4's
advice validator refuses every recommendation that announces itself and **none** of six
hand-labelled recommendations that do not: no imperative, no rating word, no price target, no
position-sizing instruction, and *"if it were my own capital I would be adding to Ford on any
further weakness"* goes straight through.

| | value |
|---|---|
| hand-labelled recommendations **not** refused | 6/6 |
| positive controls refused, which is what makes that a measurement | 10/10 |

Both halves, always together: the residue alone is equally consistent with a *dead* validator,
since `validate_answer` fails open by design and a fail-open also yields 100%. `n` is small and
hand-authored, so this is a statement about the rules' **generality**, not a 100%-evasion claim. A
rule set that pattern-matches the vocabulary of advice catches the vocabulary, not the advice. And
it cost nothing to measure — the validator is a regex set behind a Guard, so what it misses is
computable with no model and no live run, which means this could have been closed at any point
after the layer shipped.

**What was made structural instead**, which is the part that generalises into the architecture:

| constraint | made unrepresentable by |
|---|---|
| two searches in one step cannot collide on `[1]` | citation numbering assigned in one sequential pass at one seam |
| two callers cannot disagree about the collection | `filings` opened, written and read in one file |
| an out-of-Universe ticker cannot be fetched | the whitelist is a lookup, checked before any network call |
| two concurrent callers cannot both construct a Chroma handle | `caching.build_once` over all seven process singletons — `lru_cache` is atomic about its bookkeeping and says nothing about the function it wraps |
| a fourth quarantine tag cannot be escaped-but-not-denylisted | both halves parametrised over one `QUARANTINE_TAGS` tuple |
| a hypothesis cannot rest on a fenced metric | `Prediction.__post_init__` refuses to build one |
| the cap a prompt states cannot differ from the cap enforced | the prompt is a function of the cap |
| a scope sentence cannot disagree with itself | one constant, derived from `config`, read by the page and four prompts |

Where a constraint **cannot** be made structural — and the verbatim rule cannot, because the one
edit it must permit is indistinguishable from the rewrite it forbids — the honest response is to
instrument it and publish the rate.

---

