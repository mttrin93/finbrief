# ADR-0005: Pre-committed shipping default retrieval strategy

`search_filings` ships with one default strategy (ADR-0003), while the A/B produces four
configs' worth of per-bucket numbers. Choosing the default *after* seeing the numbers is
p-hacking — "we tried four and reported the winner." So the rule is pre-registered here,
before any A/B data exists.

**Decision.**

- **Default = `hybrid + translation`**, pre-registered on the dominance prediction of
  ADR-0004 (neutral-or-better on every bucket within a cost budget).
- **Falsification clause (directional).** With ~6 questions per bucket, a tight numeric
  margin would be false precision. Rule: if translation is worse on *both* context
  precision *and* context recall within any bucket, the default drops to `hybrid-only`,
  and the contradiction is written up in the README as a finding (worth more review credit
  than a clean win). Per-question spread is reported alongside every bucket mean.
- **Cost is part of dominance.** Dominance is judged within a latency budget of **≤1.5s
  p50 added by translation**, measured from the Phase-6 structured logs. A quality win
  that breaches the budget does not count as dominant. (Those logs are persisted and
  readable as of T8 — see ADR-0011 for the log's shape and for why the planner's latency and
  token spend are on their own `query_translation` line rather than folded into the
  answering path's: the cost this clause is about is the cost *translation* adds.)
- **Adaptive per-query routing is rejected for Tier-1** (e.g. detect exact-identifier
  queries → skip translation) and listed as Tier-2 future work — it adds a classifier that
  would itself need evaluation.

**Consequence.** The shipping default is defensible as a prediction that survived (or
didn't) a pre-registered test, not a post-hoc pick.

---

## Amendment (ticket T6, issue #6) — the default survives; the reason it wins changed

ADR-0004's T6 amendment records a pre-registered hypothesis that **failed**: BM25 on the retained
original does not "already nail" the `exact-identifier` bucket, and on issue #6's case `hybrid`
alone moved the relevant chunk from rank 5 to *absent*. Three things follow for this ADR, and the
first two are the ones a reviewer should check.

**1. The pre-committed default is unchanged: `hybrid + translation`.** The falsification clause here
is about the *default*, and its trigger is narrow and directional — translation worse on **both**
context precision and context recall within a bucket. Nothing of that kind happened. On the
re-measured case translation is what *rescues* the bucket: the relevant chunk goes from absent
(hybrid, no translation) to rank 1 (hybrid + normalisation) to rank 2 with 5/5 correct-filer chunks
(hybrid + full translation). What changed is the *mechanism* credited for the win, from "BM25 sees
the raw identifiers" to "normalisation supplies the identifier surface the index actually carries,
and RRF's agreement principle promotes the chunk both surface forms found".

**2. The superseding hypothesis predates any A/B data, so it is a pre-registration and not a
post-hoc pick.** This is the whole point of writing it down now. ADR-0002's golden set is ticket T9
(#4) and the per-bucket matrix is T10 (#11): **neither exists yet**, and no RAGAs or
precision/recall number has been computed against any configuration. The revision rests on one
root-caused case with its ranking published on #6, and it names its own channel so the harness can
refute it — if the `exact-identifier` win does not survive `FINBRIEF_MAX_SUB_QUERIES=0`, then
normalisation is not what earned it and the hypothesis is wrong. That is a stronger position than
this ADR started in, not a weaker one: the prediction now has a stated mechanism attached to it
rather than an assumption.

**3. The latency budget is judged against a larger variant count than this ADR assumed.** The bound
is now 1 original + at most 1 normalised + at most `max_sub_queries` (ADR-0004 amendment). The
normalised variant costs a retrieval round and never a chat round, which is the half of the ≤1.5s
p50 budget that matters least — but it is two more candidate lists under hybrid, and T10 measures
the added p50 from the Phase-6 logs rather than assuming it.

**4. A pre-registered re-examination trigger for the dominance argument.** ADR-0004 amendment §7
predicts, still pre-data, that hybrid's marginal contribution over `vector + translation` on
`exact-identifier` is **small** — the ablation puts `vector + normalisation` at rank 1 on the #6
case, with the shipping default one place behind at rank 2 while carrying better filer precision.
Since `exact-identifier` is the bucket hybrid exists to win, that makes the dominance claim behind
this ADR's default thinner than it looked when it was written.

So the trigger, fixed now: **if T10 (#11) finds `hybrid − vector` at equal translation to be within
per-question spread on every bucket, the dominance argument is re-argued in this ADR rather than
defended, and `vector + translation` becomes a live candidate for the shipping default.** This is
deliberately *not* an extension of the falsification clause above, which is about translation and
fires on a two-sided quality test; this one is about the strategy axis and fires on an *absence* of
gain. Recording it before the numbers exist is the whole point — a default kept because its
marginal component was never separately measured is the same p-hacking failure in a different
direction.

Two honest qualifications. The observed direction rests on **one case** (`n=1`, one query, `k=5`);
#4's per-bucket numbers are what can settle it. And the cost side is weaker than it appears: BM25
adds no network round trip and no spend, only local CPU, so the ≤1.5s p50 budget above is a poor
instrument against it — if hybrid fails to earn its place, the argument will be **complexity without
measurable gain**, not latency, and this ADR should say so rather than reach for the budget it
already has.

---

## Amendment (ticket T10, issue #11) — both triggers are now evaluated, and how to read them

The clause and the §4 trigger are no longer prose: they are code
(`src/finbrief/evaluation/hypotheses.py`), evaluated per bucket on every run, and **printed in the
artifact whether they fire or not**. A trigger reported only when it fires is a trigger a reader
cannot tell was evaluated — and §4's fires on an *absence* of gain, which is precisely the shape
that goes unnoticed when it is not printed. The measured outcome for any given run is in
`docs/verification/evaluation.md`, not here: this ADR records what the rules are, and the artifact
records what happened.

**Five things about the evaluation that this ADR did not anticipate.**

**1. The operands are data, because a clause whose operands live in prose can be applied to the
wrong pair.** The falsification clause is about *translation*, so both its arms hold strategy fixed
(`hybrid` vs `hybrid + translation`); §4's trigger is about *strategy at equal translation*, so
both its arms hold translation fixed. `arms.TRANSLATION_CONTRAST` and `arms.STRATEGY_CONTRAST` are
those pairs, and tests assert each holds the other axis constant. Comparing across strategies would
let a strategy effect drop the shipping default for translation's supposed sin.

**2. "Worse" means worse by more than the per-question spread, and nothing tighter.** This ADR
already reasoned that "with ~6 questions per bucket a tight numeric margin would be false
precision", and the implementation makes that concrete: `metrics.compare` returns `WITHIN_SPREAD`
rather than a direction whenever the mean difference is no larger than the measured spread of the
questions it is a mean over. The comparison is against a *measurement*, not a constant nobody
derived — which is #11's "no threshold at four decimals" as a mechanism rather than a promise. #5's
~0.0009 embedding wobble is the floor under any comparison at all.

**3. An unmeasured bucket cannot fire either trigger.** `UNDETERMINED` is its own verdict and a
test asserts the clause does not fire on it. "We cannot say" and "translation is worse" are
different claims, and only one of them is grounds for dropping a pre-committed default.

**4. Neither trigger may rest on response relevancy, and that is enforced.** ragas forces
temperature 0.3 whenever it asks for more than one completion and `ResponseRelevancy` asks for
three, so that column moves between runs on any judge — and on this judge the provider serves one
completion where three were requested, so it is also computed over a single generated question.
Both triggers rest on context precision and context recall, which is what this ADR's own wording
already named; `judge.EXCLUDED_FROM_HYPOTHESES` makes the exclusion machine-checkable and
`Prediction.__post_init__` refuses to build a prediction on a fenced metric (#11, addition 2).

**5. The latency half is a reconstruction, and is labelled one.** The clause says "≤1.5s p50 added
by translation, measured from the Phase-6 structured logs", and after ADR-0004 §9's replay the
scored `+translation` arms no longer make a planner call — they serve a recorded reply through a
stub, so their `query_translation` lines record about a millisecond and no token counts. So the
planner's real cost is read from the **resolve** pass, where the planner does run, filtered to lines
that reported spend; the retrieval-round cost is the difference between the translated and
untranslated `retrieval.latency_ms` medians; and the artifact prints both halves and their sum,
saying in the table that the sum is a reconstruction of what the shipped path pays rather than one
timing of a live turn. Counting the stub's millisecond as the planner's cost would have reported
translation as very nearly free — the most flattering possible error, which is why the filter is
explicit and tested.

**The cost asymmetry this ADR flagged still holds and now has a number attached.** BM25 adds no
network round trip and no spend, so the ≤1.5s budget remains a weak instrument against the strategy
axis; if hybrid fails to earn its place the argument is **complexity without measurable gain**, which
is what §4's trigger is written to detect and what the artifact's per-bucket table is where to look
for it.
