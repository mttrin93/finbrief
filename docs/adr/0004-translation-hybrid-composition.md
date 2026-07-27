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
