# ADR-0004: How query translation and hybrid search compose inside retrieve()

Query translation paraphrases; hybrid search wins on exact tokens (BM25). If translation
were allowed to *replace* the user's query, it could strip the literal identifiers
("Item 1A", tickers, ratio names, ISINs) that BM25 depends on — degrading the very
exact-identifier bucket hybrid exists to serve.

**Decision.** Translation only ever *adds*; composition is symmetric.

- The **original query is always retained** as a query variant. Translation adds up to 3
  sub-queries (capped for latency).
- **All variants run through both BM25 and vector.** Candidate lists are fused with
  Reciprocal Rank Fusion, deduplicated by chunk id, then truncated to top-k.
- One symmetric code path — easier to reason about and to defend than asymmetric routing.

**Invariant.** BM25 always sees the raw identifiers in the retained original, and
additionally covers identifiers that appear only in decomposed sub-queries.

**Chunk-side counterpart.** Each chunk's indexed text opens with a provenance header
(`AAPL | FY2025 10-K | Item 1A. Risk Factors`) so BM25 can match section and ticker
literals on every chunk of a Section rather than only the one the splitter left the
heading in — an effect on index content that is not assumed but measured in the T10
per-bucket A/B (#11).

**Refined pre-registered hypotheses (supersede ADR-0002's):**
- hybrid > vector-only on `exact-identifier`
- `±translation` ≈ neutral on `exact-identifier` (BM25 on the retained original already
  nails it), clearly positive on `multi-hop`
- all configs ≈ tied on `semantic`

**Provenance.** For each surfaced chunk, log which variant × which retriever surfaced it
and its RRF contribution. This is the data behind the RAG-visualization panel and lets the
A/B analysis show *why* hybrid wins, not merely *that* it does.

---

## Amendment (ticket T6, issue #6) — what building it changed, and what measuring it contradicted

The decision above stands: translation only adds, composition is symmetric, RRF fuses, dedup is
by chunk id. Five things about the shape did not survive implementation, and one **pre-registered
hypothesis did not survive measurement**. The contradiction is §6 and it is the important part.

**1. `retrieve()` returns a `Retrieval`, not a bare sequence of `Context`.** ADR-0003's T3
amendment narrowed the signature to one sequence, to stop two parallel lists desynchronising when
a caller sorts or dedups one of them. Phase 4 widens it once, because a retrieval now knows
something that is not a fact about any one chunk: **which queries it ran.** The RAG-viz panel
(user story 5) has to show a sub-query that surfaced *nothing*, and that is exactly the datum no
chunk's provenance can carry. `Retrieval.contexts` is still the sequence and each `Context` still
carries its own `distance` and `rank`, so the pairing that amendment protects is untouched.

**2. Each candidate list is `k` deep — not deeper.** This ADR fixes the *output* at top-k and says
nothing about candidate depth. `k` is the choice that keeps `vector` without translation
byte-identical to the T3 baseline the A/B compares against (RRF over a single list is strictly
decreasing in rank, so fusion is a no-op on order); a wider fetch would quietly move that
baseline. It also bounds cost, since hybrid + translation is already up to ten candidate lists,
five of them paid embeddings. Widening it is a legitimate Tier-2 experiment and would be a change
to the thing being measured, not a fix.

**3. A chunk's vector distance is `float | None`.** Under `hybrid` this is the point rather than an
edge case: the exact-identifier chunks this ADR exists to recover are precisely the ones vector
search ranked outside `k`, and BM25 has no distance of its own. A stand-in (`0.0`, `inf`) would
print a number in the sources panel that no measurement produced, so the UI says "no vector
distance". `Context.fused_score` is what `rank` was decided by, and under hybrid the two disagree
routinely — which is the disagreement the A/B is looking for.

**4. `RRF_K = 60` lives in `config.py` as a plain constant, not a `Settings` field.** It is the
published default (Cormack, Clarke & Buettcher 2009), **stated and not tuned**. ADR-0005
pre-registers the shipping default before any A/B data exists so the winner cannot be picked after
seeing the numbers, and a fusion constant somebody could sweep per environment is the back door
into exactly that: a `hybrid` result at the published constant is a prediction that survived a
test, one at the best of several `RRF_K` values is a number about the sweep.

**5. A failed translation raises; it does not degrade to no-translation.** `±translation` is one
axis of the measured A/B (ADR-0002), so a silent fallback would report a translation-enabled number
for a retrieval where translation never ran — the same failure this project refuses when `hybrid`
raised rather than serving vector results under a hybrid label. Phase 5 owns the graceful UI
failure; what it must not become is an invisible one.

---

### 6. The measured contradiction: BM25 on the retained original does *not* nail exact-identifier

This ADR's refined hypotheses said `±translation ≈ neutral on exact-identifier`, and gave a reason:
*"BM25 on the retained original already nails it."* **That reason is wrong, and issue #6's
pre-registered case is what showed it.**

The case: `Tesla debt`, `k=5`, against the ingested collection. Under `vector` the one relevant
chunk (`TSLA` Item 7, the liquidity passage) ranked **5th of 5**, behind four Ford chunks. The
prediction was that BM25 on the retained original would move it to **rank 1**.

**Measured under `hybrid`: it moved from rank 5 to absent** — out of the top-5 entirely, reappearing
at rank 10 when `k=10`. Hybrid alone made the case *worse*. Root-caused, with numbers:

| fact | measurement |
|---|---|
| the relevant chunk's own text | contains **no `tesla` token** — "*we* and our subsidiaries had outstanding $8.18 billion … of indebtedness". A filer writes "we". |
| `tsla` as a corpus token | 280 of 5,842 chunks — *all* of TSLA's, because the provenance header carries the ticker |
| `tesla` as a corpus token | **34** of 5,842 chunks — body mentions only |
| `debt` as a corpus token | 413 of 5,842 chunks |

So `tesla` carries enormous IDF and `debt` almost none, and BM25 ranked by *how often the word
"Tesla" appears*: used-vehicle trade-ins (`tesla` ×5, `debt` ×0), then human-capital oversight,
then "highly dependent on the services of Elon Musk". It recovered the **filer** and lost the
**topic**, then displaced both the Ford chunks and the relevant Tesla chunk from the `k=5` window.

The control shows the mechanism was sound and only mis-keyed: `TSLA debt` under `hybrid` already
returned 5/5 TSLA with `TSLA` Item 7 at rank 2. **The gap was name → ticker, nothing else.** The
chunk-side counterpart this ADR describes does work — for the surface form the *index* carries,
which is not the surface form an analyst types.

**The fix: deterministic entity normalisation inside the translation step.** When a Universe
company name appears in a query, a **ticker-form variant** is added — `Tesla debt` → also
`TSLA debt`. It is a lookup in `config.TICKER_BY_COMPANY_NAME`, derived from `UNIVERSE`, so the
`+translation` arm gains a variant without gaining model variance; the original is always retained,
so this only ever adds; and one substitution pass over the whole question bounds it at **one** extra
variant however many companies are named.

**Re-measured, `Tesla debt`, `k=5`, same collection, same day:**

| state | strategy | translation | TSLA in top-5 | the #6 chunk's rank | its distance |
|---|---|---|---|---|---|
| 1 | vector | off | 1/5 | **5** | 1.0406 |
| 2 | hybrid | off | 2/5 | **absent** (10 at `k=10`) | — |
| 3 | hybrid | normalisation only | 3/5 | **1** | 0.6778 |
| 4 | hybrid | normalisation + 3 sub-queries | **5/5** | **2** | 0.6777 |
| 5 | vector | normalisation only | 3/5 | **1** | 0.6778 |

Provenance says *why*, and it is not the mechanism the prediction named. The ticker form ranks the
chunk **5th under vector search too** (state 5), at distance 0.6778 against the original's 1.0406 —
the header moves the *embedding* as well as BM25. So the chunk collects two independent
vector votes plus one BM25 vote, and **RRF's agreement principle** is what promotes it past four
Ford chunks that each earned one. BM25 contributes one vote of three; it is not the load-bearing
half.

**Superseding hypothesis, and it is a pre-registration, not a post-hoc rationalisation.** No A/B
data exists yet — ADR-0002's golden set is ticket T9 (#4) and the per-bucket matrix is T10 (#11).
What is written here predates both, on the strength of one root-caused case:

- **`+translation` is positive on `exact-identifier`, via normalisation** — superseding "≈ neutral
  (BM25 on the retained original already nails it)". The channel is named so it is falsifiable: if
  the bucket's win does not survive with `FINBRIEF_MAX_SUB_QUERIES=0`, normalisation is not what
  earned it.
- **`hybrid` alone may be *negative* on `exact-identifier`** when the query names a company by
  name rather than by ticker, because BM25 recovers the filer and loses the topic. State 2 above is
  one instance; the golden set is what says how general it is.
- **The planner's sub-queries buy filer precision, not target rank** — 5/5 TSLA against 3/5, at
  the cost of one rank on the target chunk (state 4 vs state 3), because three BM25 votes across
  sub-queries promoted an Item 1 chunk. `n=1`; per-bucket precision and recall are #4's to report.

**Variant budget, restated.** This ADR writes the cap as "up to 3 sub-queries". The true bound is
now **1 original + at most 1 normalised + at most `max_sub_queries`** = 5 variants, 10 candidate
lists under hybrid. Recorded here rather than smuggled in, because ADR-0005's ≤1.5s p50 budget is
judged against this count. The normalised variant costs a retrieval round, never a chat round.

**Stated limitation: the provenance header carries the ticker only.** `AAPL | FY2025 10-K | Item 1A.
Risk Factors` gives BM25 the ticker and the Section but not the company *name*, which is the form a
question uses. Carrying both — `AAPL | Apple Inc. | FY2025 10-K | …` — is the index-side fix and is
**deliberately not taken now**: the header is inside `chunking.content_hash`, so changing it
re-embeds all 5,842 chunks, and a re-embed invalidates every distance recorded on #5, #6 and #11 —
the committed `retrieval-smoke.md` band, this amendment's own before/after, and the A/B's inputs.
Query-side normalisation gets the same result for this bucket at no re-ingest and no spend, which
is why it went first. A further fallback exists and was **not** needed: enrich only the BM25
document text with the company name at index-build time, leaving the embeddings untouched. If a
re-ingest happens for another reason, carrying both in the header is the change to make with it.

---

### 7. Pre-registered, still pre-data: hybrid's marginal contribution on `exact-identifier` is expected to be **small**

§6's ablation is the reason, and it says something this ADR did not anticipate. Written before any
golden set (T9, #4) or per-bucket matrix (T10, #11) exists, so it is a prediction and not a reading
of results.

**What the ablation showed.** On the #6 case, `vector + normalisation` — with **no BM25 at all** —
already puts the target chunk at **rank 1**. Adding BM25 (`hybrid + normalisation`) leaves it at
rank 1 with a higher fused score; adding BM25 *and* the planner (the shipping default) puts it at
**rank 2**. So on the one case that has been root-caused end to end, the direction of hybrid's
marginal contribution to *target-chunk rank* is **zero to slightly negative**, while its
contribution to *filer-level precision* is positive (3/5 → 5/5 TSLA).

**Why, mechanically.** The recovery is embedding-side. The ticker form ranks the chunk 5th under
**vector** search too (distance 0.6778 against the original's 1.0406), because the provenance
header's ticker moves the embedding as well as the lexical index. RRF then promotes it on
**agreement** — two independent vector votes from two surface forms beat four single votes. BM25
supplies one vote of three. Its role in the fix is *redundancy*, not recovery.

**Pre-registered predictions.**

- **`hybrid + translation` − `vector + translation` on `exact-identifier` is small**, and may be
  ≤ 0 on the rank of the ground-truth chunk while positive on the share of retrieved chunks
  belonging to the right filer. This supersedes the implicit assumption behind
  `hybrid > vector-only on exact-identifier` — that claim survives *at equal translation off*
  (§6 state 2 aside) but is not where the bucket's win comes from.
- **BM25's value is conditional on normalisation being present.** Without it, §6 measured BM25 as
  actively harmful on this bucket. So the two are not independent axes in the way ADR-0002's
  matrix draws them, and the interaction is worth reporting as such.
- **Falsifiable as written:** if T10 finds `hybrid − vector` at equal translation to be a clear
  per-bucket gain on `exact-identifier`, this prediction is wrong and §6's mechanism story is
  incomplete. That is a better outcome for the ADR than for this paragraph.

**One case, one query, `k=5`.** The generality is #4's to establish, and `exact-identifier` is the
bucket hybrid was supposed to earn its place in — so if the margin is small *there*, the question
of whether the BM25 arm earns its complexity anywhere becomes live, given this ADR already predicts
ties on `semantic` and credits `multi-hop` to translation.

**A cost asymmetry that belongs in the same paragraph, so the re-examination is argued honestly.**
BM25 adds **no network round trip and no spend** — the index is built once per process from
`all_chunks`, and a query costs a scoring pass over the corpus per variant. That is local CPU, not
API latency, so ADR-0005's ≤1.5s p50 budget is a weak instrument against it. If hybrid turns out
not to earn its place, the stronger argument will be **complexity without measurable gain** — one
more component, one more thing to explain, one more axis in the matrix — rather than latency. Both
should be measured; only one is likely to bite.

---

### 8. What `MIN_LEXICAL_TICKER_CHARS` actually suppresses, measured

A guard with a stated cost, and the cost was checked rather than assumed (review question, T6).

**It suppresses the variant entirely — both retrievers, not only BM25 scoring.** `normalised()` is
the only reader of `config.TICKER_BY_COMPANY_NAME` and it either rewrites the whole query or returns
`None`; a filtered-out ticker means no variant is constructed, so there is nothing for *either*
retriever to run. Since §6 establishes the mechanism is embedding-side, that matters: dropping the
variant costs the vector gain too, not just a lexical one, which is a real tension with this ADR's
symmetric composition.

**Measured for Ford, the only affected filer, and the cost is zero to negative.**

| query | retriever | result |
|---|---|---|
| `Ford debt` | vector | **5/5 F**, first `F` Item 7 at rank 2, distances 0.7753–0.9248 |
| `F debt` (the suppressed variant, forced by hand) | vector | **3/5 F** — two `BAC` Item 7 chunks intrude; distances *worse* (0.8779–1.0338) |
| `Ford debt` vs `F debt` | BM25 | identical top-5, all `F` — BM25 loses nothing either way |

And Ford is not a weak spot to begin with: `Ford debt` under `hybrid` with translation **off**
already returns 5/5 `F` with `F` Item 7 at **rank 1**, and with translation on the distances fall to
0.5141. The guard is therefore aligned with the data rather than a compromise against it.

**The underlying variable is not ticker length — it is how well a filer's own name covers its own
chunks.** Ford's is the best in the Universe bar one: `ford` is a token in **232 of its 499 chunks
(46%) and in 0 chunks elsewhere**. Tesla's `tesla` is in **33 of 280 (12%)** and leaks to NVDA.
Ford does not need normalisation, and its ticker is a poor embedding token; both facts point the
same way. `MIN_LEXICAL_TICKER_CHARS` is a first-principles proxy for that — a one-character term is
not an identifier in any index — and it happens to select correctly here. **If a future Universe
member had a short ticker *and* a poorly-covered name, the proxy would be wrong for it**, and the
right fix would be to state the rule as name-coverage rather than length. Recorded so the next
reader knows which of the two the constant is really standing in for. The per-filer coverage table
is on #4, where it constrains golden-set sampling.

**A second filer gains no ticker form, for an unrelated reason: META.** Its alias `Meta` differs
from its ticker only in **case**, so the rewrite of `Meta advertising revenue` is `META advertising
revenue` — a different string that `hybrid.tokenize` lowercases to the *same* term list. Left in,
BM25 scored two identical candidate lists and RRF counted every one of the original question's
votes twice, at no additional evidence: the duplication `normalised()` returns `None` to prevent,
arriving through the one comparison that could not see it. It is now compared case-insensitively
(issue #6 review). Nothing measured is lost, and the paragraph above says why — META is the "bar
one" in Ford's coverage claim, the filer whose own name least needs a ticker to be found.

---

### 9. Where `retrieve()` is deterministic, and where it is not

ADR-0003 calls `retrieve()` "a standalone **deterministic** component with query translation +
hybrid search *inside it*" and says "**this is what the headline numbers measure**". After T6 that
sentence is true of three of ADR-0002's four configurations and not of the other two, and the
difference has never been written down. Recorded here because T9 (#4) and T10 (#11) will report
against it (review question, T6).

**The pipeline has a deterministic half and a sampled half.**

| step | where | deterministic? |
|---|---|:--:|
| entity normalisation | `query_translation.normalised` | ✅ a lookup in `config.TICKER_BY_COMPANY_NAME` — §6's whole reason for choosing it |
| **sub-query planning** | `query_translation.translate` → `model.invoke` | ❌ **a chat completion** |
| parsing the planner's reply | `query_translation.sub_queries` | ✅ pure |
| vector search | `vectorstore.nearest_chunks` | ✅ |
| lexical search | `hybrid.BM25Index.nearest` | ✅ ties broken on chunk id |
| fusion, dedup, truncation | `hybrid.fuse` | ✅ ties broken on chunk id |

So exactly one step samples, it is reached only when `max_sub_queries > 0`, and everything
downstream of it is a pure function of what it returned.

**Temperature 0 is greedy decoding, not a determinism guarantee.** `retrieve()` names
`temperature=0.0` explicitly at the call site, and that is the strongest instrument available — but
this project already makes the argument against relying on it, in ADR-0003's T4 amendment §5:
OpenRouter fronts many upstreams and "whether a given one honours it is not something we can
assert". No `seed` is sent. Batching and expert routing can move an argmax between otherwise
identical requests. A `seed` was considered and is **not** treated as a fix: the OpenAI-compatible
parameter is documented as best-effort even at its origin, and a request routed to a different
upstream ignores it entirely — adding it would buy a little stability and a false claim.

**So, plainly:**

- **`vector − translation` and `hybrid − translation` are exactly reproducible.** No model runs.
  Same question, same collection, same `k` → the same contexts in the same order, byte for byte.
- **`vector + translation` and `hybrid + translation` are reproducible only up to the planner's
  temperature-0 sampling.** One changed sub-query changes 2 of the 10 candidate lists, which moves
  fused ranks, which moves context precision and recall. **A re-run of T10 can report a different
  per-bucket number on these two arms with no code change.**

**A second consequence, and the sharper one: a scored run's variants are not recoverable
afterwards.** The `retrieval` log line carries variant *counts and indices* and never the text —
deliberately, because a variant is derived from a user question and these lines are kept — and
`rag.answer_question` narrows `Retrieval` to `.contexts`, so the measured chain does not return
them either. Nothing is wrong with either decision on its own; together they mean that if a bucket
number looks surprising, the sub-queries that produced it cannot be inspected after the fact.

**What this does *not* weaken: §6's falsification channel.** `FINBRIEF_MAX_SUB_QUERIES=0` removes
the `model.invoke` call rather than truncating its output (`query_translation.translate` guards the
call on the cap; two tests assert the model is never invoked). So "does the `exact-identifier` win
survive with the planner off?" is a question asked entirely within the deterministic half, and
§6's channel and ADR-0005 §2's refutation test both stand on exact ground. The same is true of §7's
`vector + normalisation` ablation.

**The harness design that makes the A/B exact, pre-registered for T10 (#11).** `retrieve()` already
takes an injectable `model=`, which is the whole mechanism:

1. Resolve each golden-set question's variants **once**, and persist them beside the question as
   part of the golden set's fixed inputs.
2. Run every A/B arm with those variants replayed through a stub model. All four arms then differ
   only by `strategy` and `translate`, which is what the matrix claims to compare, and a re-run
   reproduces the numbers.
3. Report the planner's own variance **separately**, as an n-repeat of the resolve step on a
   sample of questions — it is a real property of the shipped path and belongs in the report, but
   it is not a property of the fusion strategy and must not be folded into that strategy's error
   bars.

Written before any A/B data exists, like §7 and ADR-0005's amendment, so the harness cannot be
designed around numbers already seen. **Falsifiable as written:** if the n-repeat finds the planner
returns identical sub-queries across runs on this Universe and this model, step 2 was unnecessary
caution and the flat claim ADR-0003 started with was fine. That would be a good outcome; it is not
one to assume.

---

### 10. BM25 admits chunks on question-form terms — measured, pre-data, fixed before T9 authoring

Found by observation, not by a failing test: a live `what are the main risk factors for Tesla?`
through the shipping default returned **GOOGL's `Item 7:21` in slot [5]**, BM25-only, no vector
distance. Its text is "The **main** components of our research and development expenses **are**".

Recorded here for the same reason §6 and §7 are: **before any A/B data exists**, so the fix cannot
be read as chosen after seeing a number it improved — and fixed **before T9 (#4) authors the
golden set**, so the ground truth is not written against a known leak.

**What surfaced it.** Both deterministic variants, at BM25 rank 1 and 2 over the whole
5,842-chunk corpus:

| variant | text | rank of `Item 7:21` |
|---|---|:--:|
| 0 (original) | `what are the main risk factors for Tesla?` | **1** of 5,678 matching chunks |
| 1 (ticker form) | `what are the main risk factors for TSLA?` | **2** |

It matched `main`, `for`, `the`, `are` — and **none** of `risk`, `factors`, `tesla`. Zero of the
three terms carrying the information need, ranked first.

**The scoring.** Chunk length 147, corpus `avgdl` 123.9, total score 16.4742:

| term | df | df % | idf | tf | saturation | contribution |
|---|---:|---:|---:|---:|---:|---:|
| `main` | 7 | 0.1 % | **6.6568** | 3 | 1.5926 | **10.6016** |
| `the` | 5229 | 89.5 % | 1.7212 | 3 | 1.5926 | 2.7412 |
| `for` | 3405 | 58.3 % | 1.7212 | 3 | 1.5926 | 2.7412 |
| `are` | 2565 | 43.9 % | 0.2449 | 3 | 1.5926 | 0.3901 |
| `risk` | 3043 | 52.1 % | 1.7212 | 0 | — | 0 |
| `factors` | 2498 | 42.8 % | 0.2916 | 0 | — | 0 |
| `tesla` | 34 | 0.6 % | 5.1261 | 0 | — | 0 |

**Root cause: IDF measures corpus rarity, not query informativeness.** Filers write "principal"
and "primary"; an analyst types "main". So `main` is rare *in 10-K prose* — 7 chunks in 5,842 —
and BM25 therefore weights it **above the filer's own name** (6.6568 against `tesla`'s 5.1261),
at 64 % of the chunk's whole score. No corpus statistic can separate a rare question-register
word from a rare identifier like `nvda`: both are rare, and the statistic is *correct* about
both. Only a query-side lexicon can.

**Three hypotheses, two falsified with numbers.** Recorded because the falsifications are the
evidence for the shape of the fix:

- **Term-frequency saturation — ruled out.** tf 3 yields 1.5926, not 3×. Okapi's `k1`/`b` are
  behaving exactly as designed and are not a factor.
- **Stopword removal on both sides — measured *worse*.** Stripping a published stopword list from
  the corpus *and* the query removes more compensating mass from the Tesla chunks than from the
  offender: `Item 7:21` stays top-5 and **rises to rank 1 on variant 1**.
- **A minimum-IDF floor on query terms — measured worse, and backwards.** It strips `factors`
  (0.2916) and `are` (0.2449) while **keeping `main` at 6.6568**, puts the leak at rank 1 on both
  variants, and costs `TSLA Item 1A` both `item` and `1a`.

**The fix: `hybrid.query_terms`, query-side only.** `tokenize` minus a stopword list plus exactly
one word, `main`. The corpus keeps every term, so `df`, `idf` and `avgdl` are untouched and **any
query carrying no scaffolding term scores byte-identically to the baseline** — which is what made
this safe to land on the measured path. The tokenizer is still shared, so `Item 1A` still becomes
`item`/`1a` on both sides; what differs is a query-side term filter, which is ordinary BM25
practice. Each retriever still sees **every** variant — the composition of §1 is unchanged — and
each applies its own preprocessing, the vector half embedding the variant unmodified. The
stopword list is inlined as a literal rather than taken from NLTK, which downloads its corpora at
first use and would break the hermetic-suite contract exactly as tiktoken's BPE table did.

**Measured, all eight queries, real corpus:**

| query | baseline top-1 | after | |
|---|---|---|:--:|
| `what are the main risk factors for Tesla?` | **GOOGL `Item 7:21`** 16.474 | TSLA `Item 1A:70` 10.800 | fixed |
| `...for TSLA?` | TSLA `1A:38`, leak at **2** | TSLA `1A:86` — top-5 all TSLA Item 1A | fixed |
| `Tesla debt` (§6 before-case) | TSLA `Item 1:22` 10.629 | TSLA `Item 1:22` 10.629 | **identical, 5/5 rows** |
| `TSLA debt` | TSLA `Item 1A:83` 7.762 | TSLA `Item 1A:83` 7.762 | **identical, 5/5 rows** |
| `TSLA Item 1A` | TSLA `Item 1:0` 8.541 | TSLA `Item 1:0` 8.541 | **identical, 5/5 rows** |
| `NVDA Item 7A interest rate risk` | NVDA `Item 7A:0` 22.990 | NVDA `Item 7A:0` 22.990 | **identical, 5/5 rows** |
| smoke q1 (same as the failing query) | **GOOGL `Item 7:21`** | TSLA `Item 1A:70` | fixed |
| smoke q3 (NVIDIA interest-rate risk) | no NVDA chunk in top-5 | NVDA `Item 7A:0` enters at 3 | improved |

On the failing query the leak does not merely score lower — it is **not a hit at all**, because
scaffolding is dropped before the membership test as well as before the scoring, so it earns no
rank and therefore no RRF vote.

**The growth rule, which matters more than the entry.** No word joins `_SCAFFOLDING` without a
recorded `df` statistic **and** a measured query it fixes. The near-neighbour qualifier set —
`key major biggest largest top overall important` — was measured across seven queries and changed
**no** result: inert, so it stays out. And these each name a real filing concept, so adding them
would cost recall to buy nothing measured:

| word | why it stays out |
|---|---|
| `principal` | *principal amount* of debt |
| `common` | *common stock* |
| `general` | *general and administrative* expenses |
| `significant` | *significant accounting policies* — a 10-K heading |
| `basic` | *basic* earnings per share |
| `central` | *central bank* |
| `chief` | *Chief Executive Officer* |
| `core` | *core* operations |
| `major` | *major customers* |
| `key` | *key employees* |

A curated lexicon is a dangerous thing to grow; this table is the evidence for why, and it is the
reason the set holds one word rather than eight.

**Separately: a known `rank_bm25` distortion, observed and deliberately not fixed.** The library
replaces a negative IDF with `epsilon * average_idf`. This corpus's `average_idf` is 6.8905, so
that floor is **1.7212** — and it lands on `the` (89.5 % df), `for` (58.3 %) and `risk` (52.1 %)
alike. Two consequences, both real:

- `the` receives **the same weight as `risk`**, the query's actual topic word.
- `the` at 1.7212 outweighs `factors` at its honest 0.2916 by roughly **6×**.

It is recorded and **not fixed now**, for two reasons. It does not resolve the observed case —
`main` is 64 % of that score and is not floored — and changing `epsilon` would move **every** BM25
score off the byte-identical baseline this section's fix was chosen to preserve, which is the
parameter sweep `config.RRF_K`'s note rules out ("a number about the sweep, not about the
strategy"). **Carried forward as a candidate explanation if BM25 underperforms in T10 (#11)**: if
the lexical half contributes less than §7 pre-registers, this floor is the first thing to test,
and testing it means re-running the whole matrix, not adjusting a constant.

**What this does to §7.** §7 pre-registers hybrid's marginal contribution over
`vector + translation` on the `exact-identifier` bucket as **small**, because the recovery is
embedding-side. That pre-registration was made against a BM25 that admitted question-form matches.
BM25 precision is now higher on natural-language question forms, so **the expected marginal
contribution may be larger than §7 states — for the semantic bucket more than the exact-identifier
one**, whose queries are terse and carried no scaffolding to strip (all four such queries above are
byte-identical). This restatement is itself **still pre-data**: it is written before any A/B run,
it is a direction rather than a magnitude, and §7's number stands as the prediction of record. If
T10 shows hybrid's margin unchanged from §7 on the semantic bucket, this paragraph was wrong and
that is a finding worth reporting.

---

### 11. Cross-filer mention leakage: BM25 cannot separate "about company X" from "mentions company X"

Observed live through the shipping default, **after §10's fix had landed** — so it is a second
mechanism rather than a residue of the first. Recorded pre-data and pre-T9 for §10's reason: the
golden set should not be authored against a leak nobody wrote down.

**What happened.** `what are the main risk factors for Microsoft?`, `k=5`: four MSFT chunks and
**NVDA `Item 1A:166` at rank 4**, BM25-only, no vector distance. The chunk in full — it is short,
and its indexed text is quoted here rather than paraphrased:

> `NVDA | FY2026 10-K | Item 1A. Risk Factors` … *"Delaware law and our certificate of
> incorporation, bylaws and agreement with **Microsoft** could delay or prevent a change in
> control."*

**The match is correct, and that is the whole difficulty.** §10's leak matched `main`, `for`, `the`,
`are` and *none* of the query's topic terms. This one matches the query's highest-IDF term on its
literal surface form: `query_terms("what are the main risk factors for Microsoft?")` is
`['risk', 'factors', 'microsoft']`, so the scaffolding is already stripped and what remains hits the
chunk legitimately. Measured on the ingested collection:

| token | chunks | where |
|---|---:|---|
| `microsoft` | 66 of 5,842 | **60 MSFT, 6 NVDA** |
| `msft` | 237 of 5,842 | all 237 of MSFT's chunks — the provenance header again |

MSFT's own name covers 60 of its 237 chunks (25 %) and leaks into 6 NVDA chunks, and BM25 is right
about all 66 of them. **Nothing about the token is anomalous** — it is rare, it is an identifier,
and it is present in the text. The fact that separates a chunk *about* Microsoft from a chunk
*mentioning* Microsoft is not in the text at all: it is which filer filed it, which is metadata. So
no lexical rule and no corpus statistic can draw the line, in the same way §10's `main` could not be
separated from `nvda` by rarity.

**And it is not a one-chunk curiosity.** Sweeping every name form in `config.TICKER_BY_COMPANY_NAME`
over the same collection — counting chunks that carry *all* of a form's tokens, a conjunctive proxy
that understates a disjunctive BM25 — five filers' names appear in another filer's text, and the
largest instance is an order of magnitude bigger than the observed one:

| name form | own chunks | chunks in *other* filers |
|---|---|---|
| `Apple` | 36 of AAPL's 152 | **45** — GS 21, JPM 16, META 7, LLY 1 |
| `Microsoft` | 60 of MSFT's 237 | 6 — all NVDA |
| `General Motors` | 61 of GM's 311 | 2 — GS |
| `Goldman Sachs` | 215 of GS's 875 | 2 — LLY, AMZN |
| `Tesla` | 33 of TSLA's 280 | 1 — NVDA (the leak §8 already noted in passing) |

The `Apple` rows are not incidental prose either: they are **the Apple Card portfolio transaction**,
discussed in JPM's and GS's `Item 7` with dollar amounts. So a lexical rule would have to suppress a
term that is doing real work in two other filers' MD&A. (One measurement is an artifact and is
excluded above rather than reported: `J&J` tokenizes to `['j', 'j']`, so it "matches" any chunk
containing a standalone `j` — 29 of them in JPM, which writes *J.P. Morgan*. That is the tokenizer,
not a mention, and the distinction is the same one §8's coverage table needed.)

**Why the §10 fix must not be stretched to cover it: a company name may never enter
`_SCAFFOLDING`.** §6's recovery *is* name → ticker identifier matching, and §7's ablation shows it is
embedding-side; suppressing `microsoft` as a query term would destroy the matching §6 and §7 rest on
in order to remove a match that is not wrong. §10's growth rule already excludes it — no word enters
without a measured query it fixes, and there is no query this fixes — but the reason is worth stating
on its own: this is the one class of word where an entry would be actively destructive rather than
merely inert.

**Candidate fix, deliberately not taken now: metadata filtering by ticker** when the question names a
Universe company — the same `config.TICKER_BY_COMPANY_NAME` lookup `normalised()` already performs,
applied as a Chroma `where` clause and a BM25 candidate mask instead of as a query rewrite. Two
reasons it waits:

1. **It changes the measured seam immediately before T9 (#4) and T10 (#11).** A filter moves every
   arm's candidate set, so the golden set would be authored against one engine and the matrix run
   against another. §6, §7 and §10 are all pre-registrations about the *unfiltered* composition.
2. **A hard filter would break the multi-company comparison the demo's peer step requires.** The
   filter can only admit the tickers the question spells, and a comparison against a peer cluster is
   resolved from `config.PEERS` (ADR-0009), not from the question's wording — so a question that
   names one company while its answer needs a peer's chunks would be answered from a slice that
   excludes them, silently and with no error.

**Recorded as Tier-2 / future work** (ADR-0001, ordered by ADR-0010), needing **its own ADR and its
own per-bucket measurement**: a filter is a precision/recall trade, and correct-filer share is only
one of the two numbers it moves. Until then this is a **known precision cost of the BM25 arm**, and
#11's matrix is what quantifies it — §7 pre-registers hybrid's contribution to filer-level precision
as positive, and this is the same effect's other sign: hybrid admits the wrong filer on a *correct*
token. Both readings belong in the same cell of the report, and the bucket that carries them is
`semantic`, where questions name companies by name.

---

### 12. §9's harness design, as built (T10, #11) — and the bug that proves it was needed

§9 pre-registered three steps for making the A/B exact. All three are built
(`src/finbrief/evaluation/variants.py`), and one detail of the implementation is worth recording
because it is not what §9 wrote.

**What is persisted is the planner's raw reply, not the parsed sub-queries.** §9 says "resolve each
question's variants once, and persist them". Persisting the *variants* would freeze the output of
whichever version of `query_translation.sub_queries` was current when the resolve ran — and that
parser has already changed twice for good reasons (`_MARKER`'s decimal fix, `_REFUSAL`). So the
reply is persisted and replayed through the real parser, and `verify_replay` re-parses every stored
reply on load and raises unless it still yields the variants stored beside it. A parser change
therefore fails at the door instead of silently altering what every `+translation` arm retrieved
over. The file is `src/finbrief/evaluation/golden_variants.json`, committed, versioned with the
golden set, and never hand-edited.

That also closes §9's second complaint directly. "A scored run's variants cannot be inspected after
the fact" is no longer true: they are in a committed file with the reply that produced them, which
is what a surprising bucket number needs.

**The bug, and why it belongs in this ADR rather than only in a commit message.** `retrieve()`
takes `strategy` and `translate` as caller-named arguments — the rule that stops a number being
reported against a configuration nobody selected — but it reads `max_sub_queries` from `Settings`,
because the cap is *enforced* configuration that ADR-0005's latency budget assumes. So an `Arm`
object carrying `max_sub_queries=0` changed nothing: **both planner-off ablation cells ran at the
application's cap of 3 and made real, unreplayed planner calls.** Those two cells are §7's ablation
and ADR-0005 §2's falsification channel, so the two refutation tests this ADR relies on would have
answered a question nobody asked, while the artifact looked like a completed run.

Found by a **two-question smoke run over all six arms** before the full sweep, in its log: four
`query_translation` lines carrying `input_tokens` where a replayed arm emits none. Not found by any
test, and the reason is worth stating — every test until then injected a planner and asserted on
what came back, which is exactly what a wrongly-capped arm still does correctly. The test that
catches it now counts a spy planner's invocations and asserts **zero**
(`tests/test_eval_pipeline.py`), which is the "assert the input arrived" rule from CLAUDE.md
applied to an input that was supposed *not* to.

Two smaller findings from the same smoke, both about the judge rather than the engine, are recorded
in ADR-0002's T10 amendment and in `evaluation/judge.py`: response relevancy costs one judge call
rather than three on this judge, and OpenRouter serves that call with **one** completion where
ragas asks for three — so that metric is computed over a single generated question and is fenced
off from every pre-registered decision.

**§9's falsifiable prediction is now answerable, and the answer is in the artifact.** §9 wrote:
"if the n-repeat finds the planner returns identical sub-queries across runs on this Universe and
this model, step 2 was unnecessary caution". The n-repeat is `variants.agreement`, reported on its
own and order-sensitive — two orderings of the same sub-queries are not guaranteed to fuse
identically, and calling them the same would overstate the planner's stability in the direction
that flatters the harness.

**Correction (code review of #11): this paragraph claimed the artifact carried that measurement
while nothing ran it.** `variants.agreement` and `PlannerAgreement` existed with no production
caller — tests only — and no section rendered them, so §9's step 3 was unmeasured and its prediction
unanswered while this ADR said otherwise. A claim in an ADR cannot fail any more than a claim in a
comment can, which is the bug class CLAUDE.md names; the fix is a rendered section driven by a pass
that makes real planner calls.

**The answer, requoted from [`docs/verification/evaluation.md`](../verification/evaluation.md):
0 of 8 sampled questions returned identical sub-queries across 5 repeats** at temperature 0. Modal
share 20–60%; three questions produced five distinct sub-query sets in five attempts. So step 2 was
**not** unnecessary caution — the resolve-once replay is load-bearing, and without it the two
`+translation` arms would report different per-bucket numbers on a re-run with no code change. §9
called that outcome "not one to assume", and it was right not to.
