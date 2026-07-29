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

## Amendment (code review of #11): the default is retained, not validated

The first T10 run's verdicts came from a comparator that could not return anything but a null
(ADR-0002's T10 amendment §3). Re-measured with the paired exact test, both of this ADR's
pre-registered decisions read differently, and the difference is about *what the run could see*
rather than about retrieval. Every figure below is requoted from
[`docs/verification/evaluation.md`](../verification/evaluation.md), not from the earlier run.

**1. The default stays `hybrid + translation`, and the basis is now stated honestly: it is
retained because the experiment could not resolve the question, not because it was validated.**

What the run actually found about hybrid:

- **Nothing detectable on any bucket.** `hybrid − vector` at equal translation is `not detected` on
  `semantic` and `undetectable at this n` on the other three.
- **The point estimate on its own predicted bucket is negative.** On `exact-identifier` — the bucket
  hybrid exists to win, and the bucket ADR-0002 decision 4 named — `hybrid − vector` at translation
  off is **Δ −0.087** (p=0.281, 6 of 7 differing). Not a resolved loss, and not a gain either.
- **The one live root-caused case favours the simpler arm.** ADR-0004 §6's exact-identifier recovery
  put the target chunk at **rank 1** under `vector + normalisation` and **rank 2** under
  `hybrid + translation`.
- **§7's pre-registration is consistent with all of it.** It predicted hybrid's marginal
  contribution over `vector + translation` would be *small*; measured Δ +0.076, undetectable at
  effective n=5. A pre-registration that survives is worth less when the instrument could not have
  contradicted it, and that is said here rather than claimed as support.

So **`vector + translation` is a live candidate this run could not rule out** — it is simpler, it
costs no BM25 index over the whole corpus, and nothing measured here prefers hybrid to it. The
default does not move on this evidence because *no* configuration is preferred on this evidence, and
moving a pre-committed default on an unresolvable comparison is the same error as keeping it on one.

**What would settle it: a larger per-bucket `n`, and nothing else.** The exact test's floor is 6
differing questions with **zero** minority signs tolerated at 6 and one at 7 (ADR-0002's T10
amendment §4). Effective n ran 1–6 across the decision cells. No amount of re-running at 7 questions
per bucket resolves a small effect; only more questions per bucket does. Until then this ADR's
strategy axis is **undetermined**, which is a weaker and more accurate position than either
"validated" or "falsified".

**2. §4's trigger fired on one bucket, and one bucket is not "every bucket".**

The trigger's condition as written is: *"if T10 finds `hybrid − vector` at equal translation to be
within per-question spread on **every** bucket"*. Measured:

| bucket | `hybrid − vector`, both +translation | n (differing) |
|---|---|---:|
| semantic | not detected (Δ +0.007, p=0.688) | 7 (6) |
| exact-identifier | undetectable (effective n=5, Δ +0.076, p=0.312) | 7 (5) |
| tool-augmented | undetectable (effective n=4, Δ −0.121, p=0.250) | 7 (4) |
| multi-hop | undetectable (effective n=4, Δ +0.130, p=0.625) | 7 (4) |

**One** bucket carries an absence of gain. The other three are excluded as undetectable rather than
counted as absences — which is the correction that matters, because a trigger firing on an *absence*
takes an instrument with no power as confirmation. Under the old comparator all four read as
absences and the trigger fired on a tautology.

**The wording is hereby recorded as too strong for an instrument this underpowered.** "Every bucket"
presumes every bucket can answer; at ~7 questions most cannot. The trigger's *intent* — do not keep
a default whose marginal component was never separately measured — is met and then some: the
component still has not been separately measured, and now the reason is known and quantified. A
future revision should state the condition over buckets **with the power to resolve a gain** and
require some minimum count of them, rather than over all four.

**3. The ≤1.5 s budget is missed at 3212 ms, and it is recorded as missed and left unamended.**

| | ms | samples |
|---|---:|---:|
| planner's chat round, p50 | 1838 | 48 |
| retrieval p50, translation on (planner enabled) | 1688 | 64 |
| retrieval p50, translation off | 313 | 56 |
| retrieval rounds translation adds, p50 | 1374 | — |
| **total p50 added by translation** | **3212** | — |
| this ADR's budget | 1500 | — |

**This figure is re-measured on every measuring run and moves by ~10%.** Successive runs over the
same cached cells reported 3138 ms and 3068 ms before this one; the pools are the same size (48
planner rounds, 64 and 56 retrievals) and the difference is provider latency, not configuration.
The budget is missed by more than 2× in every one of them, which is the only claim this ADR rests
on it — a miss of that size does not depend on which run is quoted.

**Not amended to the measured value, deliberately.** ADR-0006 revised its gate budget from 800 ms to
1000 ms and that revision was earned: a figure cleared on eight of eight measured passes, with an
argument about what the gate is worth and both numbers printed side by side ever since so a reader
cannot mistake a revised pre-registration for one that always held. Nothing equivalent exists here.
There is no argument that 3.3 s of added latency is acceptable for an analyst's turn, and moving the
number to wherever the measurement landed is pre-registration in reverse. So the budget stands at
1500 ms and this ADR records that its shipping default **misses it by more than 2×**.

**Neither of the two earlier figures should be quoted, and the reasons differ.** 2766 ms was a
median over a pooled window of the append-only sink — 13 appended runs, including the pre-fix run
whose ablation cells made real planner calls — *and* it averaged the two planner-**off** ablation
arms into the "translation on" pool, measuring the deterministic ticker form's cost under a budget
meant for the planner's round. Both are fixed (ADR-0011's T10 amendment;
`latency.PLANNER_DISABLED_CAP`).

3138 ms and 3068 ms were later runs of the same corrected instrument and are superseded only by
recency, not by a defect — see the re-measurement note above. 3297 ms was the first correction and
it over-corrected. It excluded translated retrievals by their
**observed variant count**, which conflates *disabled by configuration* with *ran and returned
less*: a planner that **refused** still paid for a full chat round, and dropping refusals removes
the cheap retrievals from a median of the expensive ones — biasing the p50 upward and making this
miss look worse than it is. The exclusion now keys on the arm's own `max_sub_queries`, joined to the
retrieval line through `logging_setup.turn` (a scope that module's docstring had described the
evaluation harness using and which the harness did not set). 3 lines moved back into the pool — two
identified refusals and one unattributable agent-stage retrieval, which is kept because absence of
evidence is not evidence of a disabled planner — and the figure fell 159 ms.

**The direction of every correction here is worth stating: two made the number worse and one made it
better, and none was chosen for that.**

**This is the second half of the default's problem, and it compounds the first.** This ADR judges
dominance *within* the budget. Hybrid earns nothing detectable, and translation costs more than twice
what the budget allows. The narrow quality test — the falsification clause — is all that keeps the
default, and per §1 above that clause **could not have fired on any bucket**: all eight cells
undetectable, effective n as low as 1. The default is currently held up by nothing that this run
measured.

**4. §9's falsifiable prediction is answered, and it vindicates the replay.**

ADR-0004 §9 registered: *"if the step-3 n-repeat finds the planner returns identical sub-queries
across runs on this Universe and this model, step 2 was unnecessary caution and ADR-0003's original
flat claim was fine. Good outcome; not one to assume."* Measured over 5 repeats each on 8 sampled
questions at temperature 0:

**2 of 8 questions returned identical sub-queries on every repeat**, and that figure is itself
re-sampled: three successive runs of the same pass reported **0, 1 and 2 of 8**. Modal share ran
20–100%. The measurement is a live planner call repeated five times, so it carries the planner's own
variance twice over — once in what it returns and once in how many repeats agree — and no single
run's count should be quoted as a property of the planner.

**The conclusion is the same in all three**, which is why it survives the noise: 6, 7 and 8 of 8
questions varied. Whatever the exact count, most questions do not return a stable plan.

So the resolve-once replay was **load-bearing, not caution**. Without it the two `+translation` arms
would report different per-bucket numbers on a re-run with no code change — which is exactly the
blast radius §9 described — and the reproducibility caveat this ADR inherited from #5 is understated
rather than overcautious. It also means the planner's variance is a real property of the shipped
path that no arm's error bar contains, and it is reported on its own, as §9 required.
