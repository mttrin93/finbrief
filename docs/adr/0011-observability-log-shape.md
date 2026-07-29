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
