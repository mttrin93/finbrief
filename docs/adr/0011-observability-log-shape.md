# ADR-0011: The observability log is a secondary record, not the harness's primary input

Every `log_event` line went to stderr and nowhere else, so nothing survived the process that
wrote it. T8 (#10) adds persistence and a reader. What the reader is *for* determines what the
log has to carry, and the obvious answer is wrong in a way worth recording, because it is the
difference between an instrument that earns its complexity and one that duplicates a return
value.

**The scope observation.** T10 (#11) is the only consumer that matters, and its A/B harness
drives `retrieve()` in-process with an injected stub model (ADR-0004 §9). Every `Context` it
gets back already carries `rank`, `fused_score`, `chunk_id`, `section`, `ticker` and the full
`provenance` tuple — variant index, retriever, rank, RRF contribution, per-list distance. So
**the per-question retrieval provenance AC-4 asks about is available to T10 without reading a
single log line.** What the return value does *not* carry is exactly three things:

1. **Latency.** `Retrieval` has no timing. A harness stopwatch around `retrieve()` gets the
   end-to-end number but cannot split the planner's chat round from the retrieval rounds, and
   ADR-0004's amendment ("the normalised variant costs a retrieval round and never a chat
   round") makes that split the interesting half of ADR-0005's ≤1.5s p50 budget.
2. **Token counts.** Nowhere in the process and nowhere in a return value. Three call sites
   held a reply that reported its own cost and dropped it.
3. **Anything a live run did.** The divergence rate on real questions, a real analyst's blocked
   question and which layer caught it, which tools a real model chose. There is no return value
   for a conversation that has ended.

**Decision.** The log is shaped as the **record of a run**, not as the harness's data bus.

- **Persistence is opt-in** (`FINBRIEF_LOG_FILE`), append-only, and unrotated. Off by default
  because `configure_logging()` is called with no arguments at four entry points and the
  hermetic suite would otherwise write files; documented in `.env.example`, in the README's
  evaluation-run instructions and in T11's demo walkthrough, because a log nobody enables is an
  instrument that ships and never runs. Unrotated because a rotating file renames mid-run and a
  globbing reader then double-counts or misses — the volume is bounded by a human typing.
  **Off by default means off in `.env.example` too**, where the recommended path ships
  *commented out*: it first shipped uncommented, and since the README and the app's own config
  banner both tell a reader to `cp .env.example .env`, "off by default" was false for everyone
  who followed the setup instructions — and silently so, because what it turned on was the
  retention of blocked questions' normalised text (issue #10 review). Enabling the sink has to
  be a decision, which is the whole reason the default is nothing.
- **One reader, `observability/events.py`**, paired with the one emitter. It returns *samples*
  and never statistics: the two p50 budgets belong to the reports that quote them, and
  `security/report.py` already owns a median over in-process `Screening` objects.
- **A `turn_id` on the envelope**, set by a `ContextVar` scope. This is the one thing the
  in-process route does not make redundant, because it is what turns an aggregate into a join:
  a `retrieval` line carries provenance and, deliberately, no question. The alternative
  correlation is line order, which is correct for a serial harness, wrong the moment the model
  issues two searches in one step, and asserted by nothing either way.
- **Token counts on three of four model call sites** — generation (`rag_answer`), the planner
  (`query_translation`, kept separate because ADR-0005 judges translation on the cost *it*
  adds) and the agent loop (`agent_turn`, summed over this turn's calls only). Absent, never
  zero: a provider that reports no `usage` block did not perform a free call.

**The other consumer the story names, and why it needs nothing here.** User story 31 asks for
these logs "so that evaluation **and the security analysis** read from real data", and the scope
observation above narrows that to T10 — so the narrowing is stated rather than left as a silent
substitution. `security/report.py` renders its artifact from in-process `Screening` and
`GateResult` objects and already owns the median over them; every figure it publishes is a
measurement of the run that is producing the artifact, so there is no *past* run for it to
re-examine and no cross-process join for it to make. It therefore reads no log line, by the same
"where they overlap the return value is authoritative" rule below. What the sink adds for the
security half is not an input but a record: a live run's gate-trigger lines, `layer`, `rule` and
the bounded normalised text, kept after the process that screened them has gone. That is the
half `--gate-only` cannot produce and the half an artifact does not retain.

**What this ADR declines**, so a later reader does not mistake absence for oversight:

- **The gate classifier's tokens.** `classify()` returns a bare `Verdict`, so metering it means
  changing that return type or adding a per-turn event duplicating `input_gate` — for the
  cheapest call in the system, one word out. The gate's token spend is therefore **unmeasured**,
  and it is written down here rather than left to be inferred.
- **Dollar cost and the cost-meter sidebar** — Tier-2 (PLAN Phase 8). This is capture.
  *(Reinstated by T12 (#13) — see the amendment at the end of this file.)*
- **A `docs/verification/` artifact.** Every file there is generated by a script that spends
  money. T8's consumer is T10, whose own artifact is the evidence; a fifth artifact would be a
  paid run to maintain with no reader.
- **New events.** Everything #10 enumerates was already emitted except tokens. The remaining
  work was persistence, a reader, and a join.

**Consequence, and the honest cost of it.** A number T10 reports can be produced two ways —
from the return value in-process, or from the log — and the two must not be allowed to
disagree. Where they overlap the return value is authoritative, because it is what the run
actually used; the log is what makes a *past* run re-examinable, which is the property #11's
third comment asks for ("a scored run's variants cannot be inspected after the fact"). Note
that comment's own remedy is to persist the variants beside the golden question, not to log
them: a variant is derived from a user question, so it stays out of the log under the
no-user-content rule, and that rule is why the log can be attributed but not replayed.

The one exception to no-user-content is unchanged and bounded as ADR-0006 leaves it: a
**blocked** question's normalised text on its `input_gate` line, capped by
`GATE_LOGGED_INPUT_MAX_CHARS`. Enabling the sink therefore means keeping that text on disk,
which `.env.example` says beside the switch rather than leaving a reader to discover it.

---

## Amendment (ticket T10, issue #11) — an instrument must be emitted where the behaviour is produced

**Decision.** An event that measures a behaviour is emitted at the layer that *produces* the
behaviour, never at the layer that displays it. A log line written only by a UI can be reached by a
human clicking and by nothing else — not by an evaluation harness, not by a script, not by a test.

**This is the second instance, which is why it is a decision and not a note.** The first is in the
main decision above: `log_turn` is wrapped around `app/Home.py`'s chat turn, and neutralising it
there left **all 1028 tests green**, because every turn-id test opened the scope itself — so the
suite proved propagation and never wiring. The fix was to measure it *at the app*, its only
production caller.

The second arrived when T10 tried to close T5's deferred square-bracket adherence rate.
`security/markers.py` logs `citation_markers` on every turn, and T7's amendment §4 offered that as
the denominator "without a second instrument". But `log_markers` is called by `app/Home.py` and by
nothing else. So the tool-calling eval drove **10 live agent turns through `agent.answer`, and the
log carried zero `citation_markers` lines** — the turns happened, the markers were produced, and
the instrument was somewhere else. The rate is unmeasured in
`docs/verification/evaluation.md` for exactly that reason, and the artifact says so where the
number would have been rather than omitting the row.

**Why this is a shape and not two accidents.** Both events describe something the *agent* did — a
turn boundary, a citation marker — and both were wired where that something became visible. That is
the natural place to put them while building a UI, and it is the wrong place for anything that will
later be measured: the harness ADR-0003 exists to make possible drives `rag.answer_question` and
`agent.answer`, and neither goes near Streamlit. An instrument at the display layer is an
instrument with a human in its denominator.

**What this ADR does *not* do**, so the boundary is clear. It does not move `log_markers`. Doing so
means deciding where marker resolution belongs — `agent.answer` returns an `AgentTurn` that has the
ranks, so the check could live there, but the *answer text* is what carries the markers and the
agent does not currently parse it. That is a change to shipped code with its own design question
attached, and T10 is a measurement ticket: smuggling it in here would mean the run that reports the
rate is the run that introduced the code producing it. **It is named as its own ticket**, and
recorded on #12 as a stated limitation for T11's README.

**The general rule, for the next event added:** ask which caller a harness would use to exercise
the behaviour, and emit there. If the only caller is a page, the instrument measures clicking.

## Amendment (T10 code review, #11): the sink is shared, so a statistic over it is not a run's

The precondition this ADR handed T10 was "assert the sink was enabled and non-empty before
computing any statistic over it". That was necessary and not sufficient: it guards against an
**empty** log and says nothing about a **shared** one.

**What went wrong.** `FINBRIEF_LOG_FILE` names one append-only file, and every run and every app
session that names it writes to the same stream. `read_events(path)` read the whole file, so every
figure T10 derived from the log was a figure over every run that had ever shared it. The first
committed evaluation artifact reported a planner p50 of **1518 ms over 60 samples** while asserting,
in its own header, that "every number here is a measurement of the run named below". The pool held
**13** appended runs. Re-reading the same unchanged file two runs later gave **1649 ms over 68** —
the same cache, the same artifact, a different number, which is the proof rather than the argument.

Worse in kind: the pool included the pre-fix runs whose ablation cells made **real planner calls**
because `Arm.max_sub_queries` was not reaching `retrieve()`. `HARNESS_VERSION` evicted those cells
from the cache; nothing could evict their lines from the log. **A content-addressed cache can be
poisoned and repaired; an append-only log can only be windowed.**

**The fix, and why it is an offset rather than a field.** `events.sink_offset(path)` takes the file's
size before a run appends anything, and every reader takes `start_offset`. No new envelope field, so
a line written by an older deploy still parses — the checkpoint rule in CLAUDE.md applies to logs as
much as to payloads. It is also honest about what it selects: lines appended after the mark, which is
this run's lines plus anything writing concurrently, and that is stated where the function is.

**The interaction with the cache, and the resolution.** Run-scoping and resumability pull against
each other: a warm run replays every cell, issues no calls, appends no lines, and therefore has an
*empty* window. Two rules settle it.

- **A fully-replayed stage cannot print a latency number.** `p50` raises on an empty window rather
  than serving the previous run's median, so a latency figure in the artifact now means the stage
  behind it actually ran. The refusal names both causes — sink off, or everything replayed.
- **The window is persisted beside the cells** (`latency.Window`, `<cache-dir>/log-window.json`).
  The artifact is regenerable from the cache, so its window has to be regenerable too; a
  `--stage report` re-render reads the mark the last *measuring* run recorded and the artifact says
  the figures are replayed. Without this a re-render would refuse to report latency for numbers it
  was otherwise reproducing exactly.

**And a third instance of the ordering shape this ADR already records.** ADR-0011's T8 finding was
that an instrument emitted where the behaviour is *displayed* cannot be measured by a harness; T10
added that a log-reading section computed *before* the pass that emits its lines reports an absence
on a run that measured the thing. The `agent` stage had been moved above the deferrals block for
exactly this reason, and the planner-variance pass — whose 40 live planner calls are the only metered
`query_translation` lines a warm run produces — broke it again from the other side, rendering
ADR-0005's budget "not measured" on the run that had just measured it. **Every pass that emits runs
before anything that reads**, and `scripts/evaluate.py` now groups them that way with the rule
written above the group.

## Amendment (T10 second code review, #11): the harness's own error path asserted a network fault

**Decision.** An error path is an instrument, and it is held to this ADR's instrument rule: what it
reports has to be what happened. A wrapper that renames a local defect after a remote one sends
every reader to the wrong system.

**What happened.** Three consecutive full evaluation runs died in the judge stage, each after a
different number of cells, with:

```
openai.APIConnectionError: Connection error.
```

`curl` to the provider returned HTTP 200 throughout. The third run printed the exception underneath
it:

```
RuntimeError: <asyncio.locks.Event object at 0x…> is bound to a different event loop
```

`evaluation/judge.py`'s `score` calls `asyncio.run` — a fresh event loop **per cell** — while
`scripts/evaluate.py` built one `ChatOpenAI` and one embeddings client at process start and passed
them down every stage. `httpx` binds a pooled connection's `asyncio.Event` to the loop that first
used it, so the moment a keep-alive connection survived into the next cell's loop, `anyio` raised
and the OpenAI SDK caught it and re-raised it as a connection error. Nothing about it was the
network.

**Why it stayed hidden, which is the part worth recording.** Two reasons, and both are this repo's
existing themes:

1. **Every earlier run replayed the cells that trigger it.** Response relevancy is the only metric
   that drives *both* clients, and the committed artifact was produced by a run reporting
   `judge 560 replayed / 0 paid`. The cache — the thing that makes a killed run cheap — is also
   what kept a latent defect out of every run that would have exposed it. A cell that is never
   paid for is a code path that is never exercised.
2. **The retry self-healed it.** `Event.wait` returns immediately when the flag is already set and
   never reaches the loop check, so a `tenacity` retry after the raise could take the fast path and
   the cell would pass. That is why three runs died at three different cell counts instead of on
   the first cell, and it is why the regression test asserts the structural property with
   `Future.get_loop()` rather than provoking the stdlib's raise: a test that inherits the
   non-determinism is a test that does not pin the bug. (Measured — the first version of the double
   scored 1.0 on the second loop.)

**The fix, and the wrong turn on the way to it — which is the more useful half.** The obvious
change is to build a client per cell so none outlives its loop. That was implemented, tested, and
**it did not work**: a fourth run died exactly as the first three had. `langchain_openai` caches the
async client *below* this repo's constructor — `_cached_async_httpx_client` is `@lru_cache`d on
`(base_url, timeout, socket_options)`, all constant here — so two `build_judge()` calls return
distinct `ChatOpenAI` objects sharing **one** `httpx.AsyncClient` and therefore one connection pool.
Per-cell construction is defeated by a cache one layer down, and nothing at this layer can see it.

So the loop stops being per-cell instead. `judging.score` runs every cell on **one** event loop
(`_judging_loop`, a daemon thread, a seventh process singleton and therefore through
`caching.build_once` like the other six), and `run.cached_map`'s worker threads reach it with
`run_coroutine_threadsafe`. Concurrency is unchanged and the shared pool becomes an advantage rather
than a hazard, because connections are now reused the way the library intends. This is the change
originally noted as "better architecture, future work"; it turned out to be the only one available,
which is why it is here and not on #12.

**The generalisable part is not the asyncio detail.** It is that *a fix aimed one layer above the
defect can be indistinguishable from a correct one until it is run.* The factory version passed a
regression test that asserted exactly the property it established — a fresh client per cell — and
that property was true and irrelevant, because the object being counted was not the object holding
the state. A test can only bind the layer it names.

**Third instance of this ticket's theme, and the first outside a measurement.** The other two were
`metrics.compare`, a comparator that could not return anything but a null, and `tool_eval`'s C3, a
control that could not fail — both *inside* published numbers (ADR-0002's T10 amendment §3). This
one is in an error path, and it is the same defect in a different costume: **a check or a message
that asserts something the code did not establish.** `compare` asserted a verdict it could not
reach, C3 asserted a pass it could not withhold, and this asserted a network fault it never
observed. The generalisation for the next one: an exception the code *translates* is a claim, and a
claim needs the same adversarial reading as a measurement.

## Amendment (T10 close-out, #11): the guard and the green run, instances four and five

Two more of the same defect surfaced closing this ticket, and both are recorded here rather than
waved at because of *where* they sat.

**Four: the guard was about to make an unverifiable claim of its own.**
`tests/test_grounding_scope.py` is the mechanism this repo built against retyped figures — it binds
the README's evaluation numbers to the committed artifact, so a run that moves a number fails the
suite instead of leaving the README asserting the old one. It compared with `figure in text`.
Substring containment is adequate for `3212` and `-0.087`, and **vacuous for any short figure**:
`"8"` is satisfied by `18`, by `0.087`, by a date. Closing this ticket added the cited-marker
composition — `22`, `40`, `8` — to that list, and had it shipped, the strictest-looking check in the
repo would have been binding nothing at all.

It now matches whole numbers (`(?<![\d.\-])…(?![\d])`), checked against `"8" in "the value is 18"`
returning false, and the composition is bound as a single phrase because three short numbers cannot
be bound individually. **A guard is code and inherits every failure mode of the code it guards** —
which is the generalisation, and the reason this is the first instance recorded *inside* the
mechanism rather than in the thing it watches.

**Five: a green test run that CI had never seen.** The inherited commit was reported as "1324 tests
green". It was, locally. On the runner two tests failed with `ModuleNotFoundError: No module named
'tests'`: `test_eval_pipeline` held the suite's only `from tests.fakes import`, and that spelling
needs the **repo root** on `sys.path` where the other nine importers' `from fakes import` needs only
the directory pytest already adds. A local editable install supplies the root; a clean checkout does
not.

**Why it was not caught, and what is now structural.** The workflow is configured correctly
(`on: push: branches: ["**"]`). GitHub runs one workflow per *push event*, on that push's tip — and
that commit was pushed alongside a later one, so it was never built on its own. That half is not
closable by a test, and is recorded as a standing caution: **a green local run is a hypothesis about
CI, not a result from it.** The half that *is* closable now is:
`test_no_test_imports_through_the_tests_package` forbids the `tests.` spelling outright, so the two
environments can no longer disagree about it. A convention nine of ten importers follow is not a
convention; it is a coin that has landed heads nine times.

**Both belong in this ADR** for the reason the masked `APIConnectionError` does: they are claims a
program made that it had not established. One was made by a test, one by a person reading a test's
output. The subject changes; the shape does not.

## Amendment (ticket T12, issue #13): the cost meter is reinstated, and it costs nothing to build

**What this reverses.** The "what this ADR declines" list above cut *"dollar cost and the
cost-meter sidebar"* to Tier-2 on scope grounds, with the note that T8 is capture. That was the
right call at the time and the reason it no longer applies is not an argument — it is a schedule:
the deadline moved, Tier-1 is closed, and this reads the counts T8 already writes rather than
adding an instrument. Under an hour, and it spends nothing new. **The cut stands as recorded; it
is the constraint behind it that changed.** (ADR-0001's rate-limiting cut is reversed in the same
ticket, and that reversal is recorded there.)

**Decision.** `observability/spend.py` computes a conversation's token spend from the
`agent_turn` and `query_translation` lines, through `events.py` and never with a second parser —
one emitter, one reader, and this module does arithmetic on what the reader returns. The sidebar
renders it. Four consequences are worth recording because each one is a place the panel could
have said something untrue.

**1. The meter exists only when the sink does.** The counts live in the log and the log is
opt-in, so `FINBRIEF_LOG_FILE` unset means there is nothing to read — and the panel says that
rather than rendering `0`. A spend of zero is a claim that the calls were free, which is the
`usage_total` failure this ADR's decision list already names, arriving on a surface instead of in
a field.

**2. Scoped by `turn_id`, not by an offset — and the difference is which key exists.** The T10
amendment above establishes that a statistic over this sink is a statistic over every run that
ever named it, and its fix is `sink_offset` because a *harness run* has no key of its own on a
line. A conversation does: `app/Home.py` opens every turn as `f"{thread_id}:{suffix}"` and
`thread_id` is a `uuid4` per session, so a prefix match selects this conversation and nothing
else — strictly sharper than a byte offset, which would also admit a second tab writing
concurrently. The app takes an offset as well, and it is honest about why: read cost, because the
file grows without bound and the window does not.

**3. A price is configuration, and unset means unpriced.** `FINBRIEF_INPUT_COST_PER_MTOK` and
`FINBRIEF_OUTPUT_COST_PER_MTOK` default to `None`, and with no price the panel reports tokens and
says it cannot price them. There is deliberately **no rate card in the repo**: this project
reaches every model through OpenRouter, which fronts many upstreams and routes by availability,
so the price of a call is not something this codebase can assert. A hardcoded figure would be a
number nobody measured, going stale silently, in the one panel whose entire subject is spend —
and it would be *this ADR's* own rule broken by the feature it declined. Two knobs rather than
one because input and output are priced differently everywhere, and a blended figure has to be
wrong for both.

**4. A partial total says so, and the call it cannot see is named on every total.** Each field
carries its own denominator (`observability/tokens.py`'s per-field rule, which this module would
otherwise be the next place to break), so a total missing a call it should have counted is
displayed as a **floor** with the counts printed beside it — as is the call count itself, when
`agent_turn` reported none and the honest floor of one stood in for it (`Spend.floored`). The
gate's classifier, which this ADR declines to meter, is named **separately and
unconditionally**: one paid call per turn is structurally absent from every figure the panel
shows, and a total that quietly omitted it would imply it had counted everything.

*Corrected by the code review of #13, and the correction is the point of writing this down.* The
classifier sentence shipped as a clause of the partial banner, so the paragraph above was false
for the ordinary case: a conversation whose every metered call reported both fields renders no
banner, and the caveat this ADR claims is "stated on screen" was stated only when something
*else* was already missing. It cannot be a sub-clause of `partial` even in principle —
`Spend.partial` is defined over reported-versus-counted calls and the classifier never enters
`calls`, so no value of `partial` is evidence about it. The test asserts the sentence on exactly
the total the first version left silent. The shape of the mistake is this repo's own: a claim in a
document that nothing rendered, sitting beside a test that passed because it exercised the other
branch.

*Amended again by T13's copy pass (#14), and the claim above holds on a different surface.* The
sentence was typed out in **two** files once the analytics page shipped — `app/Home.py`'s sidebar
and `app/pages/1_Analytics.py` — which is the duplication this repo keeps catching: two copies of
one sentence disagree on the turn one of them is edited, and the copy a reader sees is whichever
page they opened. It is `spend.UNMETERED_CLASSIFIER_NOTE` now, one definition, rendered by the
analytics page — the surface that totals a **whole log**, where a reader would otherwise take the
figures for everything spent. The sidebar totals one conversation and no longer repeats it, and
`test_the_unmetered_classifier_caveat_is_not_duplicated_on_the_sidebar` asserts that it carries
neither the constant nor a paraphrase. "Stated on screen" therefore still holds, on the page whose
figures the caveat is about.

**One correctness detail that is the mirror of the usual defect.** A `query_translation` line at
`max_sub_queries=0` stands behind **zero** chat calls — ADR-0004 §6: the cap removes the
`model.invoke`, it does not truncate its output. Charging that line a call would put an
unreportable call in the denominator and report a *complete* total as partial. That is the same
fact `evaluation/latency.py` encodes as `PLANNER_DISABLED_CAP` to keep the ablation arms out of a
latency pool; `spend.py` cannot import it, because `evaluation/` is the harness and the app must
not depend on it, so the two are bound to each other by a test rather than left to drift.

**What this amendment still does not do.** It does not meter the gate's classifier. The reason is
unchanged — `classify()` returns a bare `Verdict`, so metering it means changing that return type
or adding a per-turn event duplicating `input_gate`, for the cheapest call in the system. What is
new is that the omission is now **stated on screen** rather than only here, because a panel
reporting spend is where a reader would otherwise assume it was complete. Since #14's copy pass
that screen is the analytics page rather than the sidebar — one definition of the sentence, on
the surface that totals the whole log; see the amendment above.

## Amendment (ticket T13, issue #14): the log gets a reader with a face, and one instrument stops being write-only

**Decision.** The analytics page (`app/pages/1_Analytics.py`) reads this sink through
`observability/events.py` and aggregates it in **`observability/analytics.py`** — a module that
parses nothing, takes an `EventLog`, and does arithmetic on it. That is `spend.py`'s shape, for
`spend.py`'s reason: one emitter, one reader, and everything above the reader is arithmetic.

**Why the statistics are not in `events.py`.** That module's docstring refuses them, and the
refusal is load-bearing: it returns *samples*, because the two p50 budgets belong to the reports
that quote them and `security/report.py` already owns a median over in-process `Screening`
objects. A dashboard is such a report. Putting a `p50` in the reader would have made it the
third owner of one word. The cost of keeping the contract is that `analytics.Rate` and
`analytics.p50` duplicate `evaluation/deferrals.Rate` and `evaluation/latency.p50` — the app must
not import the harness — and the two pairs are therefore **bound by test** rather than left to
drift, like `spend.PLANNER_SILENT_CAP` and `latency.PLANNER_DISABLED_CAP` before them. That cap is
now a third copy and is in the same binding.

**The five states, and why a dashboard needed more than two.** `latency.load_log` has two answers
(sink off, or a window with nothing in it) because a harness can refuse to proceed. A page cannot
refuse; it has to say something. So `SinkState` splits the absence four ways — off, named but
never written, named but unopenable, present but holding no events — and the last carries
`EventLog.malformed`, because an empty file and a file of unreadable lines are different problems
and only one is worth investigating. Every panel repeats the rule one layer down: a figure nothing
measured says so where the number would be, and `Distribution.within` returns `bool | None` so
that "not measured" cannot render as "missed".

*Amended by the #14 code review — it shipped with four.* The missing state is the general lesson
and not a detail: **an enumeration of absences is exhaustive only over the cases the code can
actually reach**, and three things a real filesystem path resolves to were not among them. A
directory raises `IsADirectoryError`, a file the process cannot open raises `PermissionError`, and
a file whose bytes are not UTF-8 raises `UnicodeDecodeError` — all three out of `read_events` and
onto the page as a Streamlit traceback, which is precisely the rendering a design built around
telling an absence from a zero may not have. The third is the one worth recording, because it
defeated a promise made one layer down: `malformed` exists for a run killed mid-write leaving a
truncated final line, and a write truncated *inside* a multi-byte sequence makes the whole file
undecodable rather than the one line unparsable. `SinkState.UNREADABLE` carries the exception's
**type name** (never its message — `finance/cache.py`'s rule, since an error string can carry a
path and this one is rendered), because "unreadable" alone sends a reader to the wrong knob: a
directory where a file was meant is a typo in `FINBRIEF_LOG_FILE` and a permission is not. A
broken symlink stays `MISSING`, which is what it is, and there is a test saying so — widening the
catch must not quietly capture it.

**What this page does *not* do, and both omissions are decisions.** It renders **no** blocked
question's `normalised` text: that field is this ADR's one bounded exception to no-user-content,
argued for an auditor with a grep, and a dashboard is a wider surface than the bound was argued
for. And it publishes **no cache hit rate**: `tool_call.age_seconds` is `round()`ed at the
emitter, so a hit 400 ms after a fetch is indistinguishable from a miss, and a rate over it could
be wrong invisibly. The explicit fields (`stale`, `stale_fallback`) are published instead. Both are
asserted by tests, so neither can be added back without a decision.

**And the T10 amendment above gets its consequence stated on a surface.** A statistic over this
sink is a statistic over every run that ever named it, and no field distinguishes an app session
from an evaluation run. The page therefore reads the **whole file** and says so in its header,
rather than separating the two by a heuristic on `turn_id` shape — which would be a separation
nothing could check, and this repo's own recurring defect.

**One instrument stops being write-only.** The amendment above records that `citation_markers` is
emitted by `app/Home.py` and by nothing else, which is why `docs/verification/evaluation.md`
reports T5's square-bracket adherence rate as unmeasured: the tool-calling eval drove ten live
agent turns and the log carried zero such lines. The page aggregates them, so the rate now exists
for any period the sink was enabled during real use. **It does not close that deferral**, and the
page says so in its header: this is *observational over logged sessions* — the population is
whoever used the app — not the controlled measurement over a stratified set the artifact asks for.
The header rather than the panel, because the claim is true of every figure on the page and a
caveat printed once per panel is one a reader learns to skip (#14's copy pass).
The pointer is rendered by `evaluation/report.py`'s deferral block rather than typed into the
artifact, because every file in `docs/verification/` is generated.

## Amendment (ticket T14, issue #15): four models can answer, so one rate card can price nothing

The picker (ADR-0001's T14 amendment; ADR-0008's) changes the **answering** model per session.
Nothing about the log's shape changes; two things about what may be *claimed* from it do.

**1. `agent_turn` carries `model`, and a second field carries what actually answered.** A model
slug is configuration and not user content — it names a product, not a person or a question — so
it needs no exception to §1's rule and joins the counts-and-verdicts payload directly. There are
**two** fields because they can disagree: `model` is what the picker requested, `model_reported`
is what the reply's `response_metadata` said. OpenRouter fronts many upstreams and routes by
availability, so a routing surprise should be visible rather than silent, and folding them into
one field would mean either losing the request or reporting the request as a measurement of what
ran. `model_reported` is `None` whenever the provider named nothing, which every scripted model in
the suite does — treating that absence as confirmation that the requested model ran is exactly the
fabricated-measurement failure `observability/tokens.py` exists to forbid.

`model` defaults to `None` rather than to `settings.chat_model` for the same reason: `answer()` is
handed a *built* agent and cannot see which model is inside it, so the configured slug would be a
guess indistinguishable from a reading. A caller that names none is **unattributed**, and the
analytics page gives it a slice labelled as such.

**2. §3's refusal to ship a rate card now bites harder, and the honest consequence is a
subtraction.** The two knobs are a single pair describing **one** model. Four selectable models
make a Haiku turn priced at the gpt-4o-mini rate reachable — a figure that is simply wrong, in the
one panel whose entire subject is spend, and **wrong invisibly**, because the tokens behind it are
real and the arithmetic is sound. So:

> **Price a conversation only when every metered turn ran on the priced model. Otherwise report
> the tokens and name the reason.**

This is `Tokens.partial`'s shape rather than a new idea — a **third** absence beside "no price is
configured" and "nothing reported the tokens a price would multiply", each with its own sentence.
`Spend.dollars` takes `priced_model` as a **required** keyword, not an optional check: an optional
one defaults to not checking, which is how the wrong figure would have survived in whichever
caller was not updated. Three ways to be unattributed all refuse: a second model in the
conversation, a turn that recorded no model, and no answering line at all. A wrong dollar figure
is worse than no dollar figure.

**Per-model pricing was considered and declined**, and the argument is §3's unchanged: a rate card
in this repo is a number nobody measured going stale. Four rate cards are four such numbers. What
ships instead is the refusal plus the split below.

**3. The planner is not in the split, and that is a fact about the code rather than a
simplification.** `retrieval/retrieve.py` builds its sub-query planner with
`build_chat_model(settings)` and takes no override, so a `query_translation` line is **always** on
the configured model and carries no `model` field at all. Two consequences, both stated where they
land: those lines are excluded from `Spend.models` (reading their absence in would make every
translated turn unattributed for a reason that does not exist), and the analytics page's per-model
rows sum to *less* than the totals above them by exactly the planner's share — which the panel
says, because unsaid it reads as an arithmetic bug in the table.

**4. The dashboard splits tokens and turn latency by model.** Nearly free once the field exists,
and the case it exists for is precisely this sink: append-only across every session, so four
models' turns land in one pool where a single p50 describes none of them. The rules already in
force carry over unchanged — a model that metered nothing reports an absence and never a zero, a
one-model log gets a sentence rather than a one-row table implying the others answered nothing,
and the unattributed row is named as *not a model and not the configured one either*. `by_model`
lives in `observability/analytics.py` like every other aggregate here: it takes an `EventLog` and
parses nothing.

**What this does not add.** No per-model quality claim. `docs/verification/evaluation.md` remains
the measurement artifact of record and every figure in it ran on the default model; a per-model
number would mean the harness run once per model. ADR-0001's T14 amendment records that as a
scoping decision rather than an omission.

### Corrections from #15's code review

Three things this amendment claimed that the code did not do, recorded rather than quietly fixed
— the practice this ADR's own §4 correction established.

**1. `model_reported` was write-only.** §1 above argues the field exists so "a routing surprise
should be visible rather than silent", and nothing read it: it was emitted, round-tripped by a
test, and consumed by no reader, panel or aggregate. That is a claim about a fact no surface could
show, and it is the *same* defect T13 (#14) was partly written to fix in `citation_markers` — one
ticket after learning it. `ModelSlice.rerouted_to` is the reader, and the per-model panel names a
reroute when the provider disagreed with the request. Silent when it agreed or said nothing, since
both are the ordinary case and a caption that always renders is one a reader skips.

**2. "Another model answered" was said about turns where nothing answered.** `Spend.models` is
built from answering lines only, so a conversation whose one metered line is the *planner's* — a
turn in flight, or one that raised after the planner's round, which is exactly what
`Spend.unfinished` counts — reaches the withheld-figure branch with an **empty** set. The sidebar
rendered a literal claiming another model had answered. Measured: `measured=True`,
`models=frozenset()`, `unfinished=1`. An absence reported as a measurement of something else,
which is the failure this whole document is organised around, arriving in the sentence added to
*prevent* a wrong figure.

The fix makes the sentence a **function of the state** — `spend.unpriced_because_of_the_model`,
three cases (another model, no model recorded, both) with the caller's own scope noun — which also
collapses the near-duplicate the sidebar and the analytics page had each grown. That duplication
is `UNMETERED_CLASSIFIER_NOTE`'s lesson repeated one ticket later, and it is worth naming twice:
**a sentence two surfaces render has one owner**, and the pressure to type it twice comes back
every time the two surfaces phrase it slightly differently.

**3. `all_answered_on` was copied where an import was available.** The duplications in
`analytics.py` are licensed by one rule — the app may not import `evaluation/` — and this was not
one of them: `spend.py` is the same package, and the same file already imports `calls_behind` from
it. The copy's docstring asserted the import was forbidden, four lines below the import that
disproved it. It is now `spend.answered_only_on`, imported. **The test of a shared-versus-copied
decision is whether the import would fail, not whether a comment says it would.**

### One more correction, from manual testing of #15

**Two of the four committed slugs did not exist**, and the comment explaining why they were
unchecked was itself the error: it said checking "costs a paid call the hermetic suite forbids".
`GET https://openrouter.ai/api/v1/models` is public, unauthenticated and free — so the cost that
justified shipping an unverified list was zero, and `anthropic/claude-3.5-haiku` and
`google/gemini-2.0-flash-001` were both 404s sitting in a picker. A reader found them by choosing
one.

That belongs here rather than only in `config.py` because it is the same family as the three
corrections above: **a reason given in a comment for not checking something is a claim, and it can
be false.** The list is now checked and the check is dated, with a test pinning the two known-bad
strings so a plausible-looking edit cannot restore them.

**What the check cannot establish is the part worth keeping.** OpenRouter returns 404 for a slug it
does not know *and* for a slug whose provider endpoints do not match the account's own data policy
— so a verified slug can 404 for one reader and answer for another, and nothing in this repo can
tell which. The honest response was not a stronger claim but a better failure: a 404 is not
transient, so "try again" was advice that could not help. The banner names the model, says retrying
will not change it, and points at the picker — PLAN §2's tiers being distinguished by what the
*reader* can do, which here is switching back rather than waiting.

**And the second round of the same lesson, from the same reader.** With the two absent slugs fixed,
`meta-llama/llama-3.3-70b-instruct` still 404'd — and the log said why, which is the point of having
one:

    No endpoints available matching your guardrail restrictions and data policy.
    Configure: https://openrouter.ai/settings/privacy

Not a code defect. It was read as a *data-policy* exclusion, since every provider serving that
model is a third-party inference host — and the replacement was chosen to be first-party on that
basis. **`x-ai/grok-4.3` then failed the same way**, which killed the theory: the clause that
matters is the one both rounds read past. *"Guardrail restrictions"* is the API key's **allowlist**,
and this key is provisioned by a course whose dashboard lists Grok **4.5** and no 4.3. So what
governs is neither the catalogue nor a property of the model but *what the key is permitted to
reach*, which no check in this repo can see.

The list is now `minimax/minimax-m2.7` — a fourth provider with a first-party endpoint, and the one
candidate present on **both** tiers of that allowlist. First-party hosting survives as a
tie-breaker for the data-policy half rather than as the rule, and the tension it exposed stands:
"include one open model" (PLAN §6's phrasing, followed uncritically) is at odds with "works on a
restricted key", because open weights are third-party-hosted by construction.

Two things worth keeping from it. The **instrument paid for itself**: this was diagnosed by reading
`chat_turn_failed`'s `exc_info` out of the sink rather than by guessing, which is the first time in
this project the event log answered a live user-facing question that nothing else could — and it is
the argument for `log_event` taking `exc_info` at all. And the **banner branches on the provider's
message without rendering it**: two 404 causes, two fixes, and a URL that is a compiled-in constant
rather than one lifted out of an error string that may carry a key. Its copy names the **allowlist
first** and tells the reader to ask whoever issued the key — because the first version said "your
privacy settings" and linked them, on a key whose restrictions were somebody else's. Telling a
reader to change a control they do not have is worse than saying nothing: it reads as their mistake.

**Three rounds, one lesson.** Each guess was a claim about an environment the repo cannot observe,
and each was stated in a comment as though it were established. The instrument is what closed it
every time — `chat_turn_failed`'s `exc_info`, read out of the sink.
