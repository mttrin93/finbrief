# Evaluation — RAGAs, per-bucket A/B, tool-calling (T10, #11)

<!-- GENERATED FILE. Rewritten by `uv run python scripts/evaluate.py`. Do not edit: every number here is a measurement of the run named below. -->

> ## ⚠ PARTIAL RUN — stages not executed: retrieve, answer, judge
>
> This artifact was rendered from a subset of the pipeline, so it is **not** a whole run and must not be committed as one. Numbers below cover only the stages listed in the provenance table; anything a skipped stage would have produced is absent rather than zero.

| | |
|---|---|
| generated | 2026-07-29 11:53 UTC |
| stages run | resolve, report |
| judge model | `openai/gpt-4.1-mini` (ragas 0.4.3) |
| answering model | `openai/gpt-4o-mini` |
| planner model | `openai/gpt-4o-mini` |
| embedding model | `openai/text-embedding-3-small` |
| k | 5 |
| golden set | 28 rows, verified against EDGAR |
| collection | 2026-07-27 08:51 UTC, 5842 chunks, docs/verification/ingest-report.md |
| collection fingerprint | `c986299c02043ee8…` |
| cache | retrieval 128 replayed / 40 paid, variants 0 replayed / 28 paid |

## Per-bucket A/B — the deterministic retrieval metrics

No judge, no spend: every figure below is computed from `Retrieval`'s own return value against the golden set's chunk ids, so a reviewer can re-derive it. Recall columns exclude whole `recall_trivial` rows (reported separately) and count only sections a `k=5` retrieval could miss.

**Read `chunk recall` and `section recall` together, and neither as a quality score on its own.** `chunk recall` asks whether the retriever returned the *specific chunks* a reference was authored from, and those references were authored from a handful of chunks inside sections that are often very large — S1 names 6 of TSLA Item 1A's **129**. At `k=5` that is close to a lottery even for a retrieval doing everything right: this harness's first live run returned Item 1A chunks 48-70 against targets 0, 1, 6, 34, 94 and 117 — chunk recall 0.000, section recall 1.000, for a result an analyst would call correct. So `chunk recall` is reported because it is exact and free, `section recall` because it is what the multi-hop and cross-filer rows are about, and the judged `context precision` / `context recall` below because they compare *text* against the reference rather than chunk identity. That is why ADR-0005's falsification clause rests on those two and not on this column.

† **The chunk-level columns are last because of a measurement, not a preference.** They ask whether the retriever returned the exact chunks a reference was authored from, which on this corpus measures section size: S1's reference comes from 6 of TSLA Item 1A's 129 chunks. They stay because they mean something on the small-section rows, where "the right chunks" and "the right section" nearly coincide — and because they are exact and free, so a reader can re-derive them. The granularity above them was chosen after seeing this, which is the honest order.

| bucket | arm | section recall | section precision | filer precision | chunk recall † | chunk precision@k † |
|---|---|---|---|---|---|---|
| semantic | vector, no translation | 1.000 [1.000–1.000] n=6 | 0.743 [0.200–1.000] n=7 | 0.943 [0.600–1.000] n=7 | 0.444 [0.000–1.000] n=6 | 0.171 [0.000–0.400] n=7 |
| semantic | vector + translation | 1.000 [1.000–1.000] n=6 | 0.714 [0.200–1.000] n=7 | 0.943 [0.600–1.000] n=7 | 0.444 [0.000–1.000] n=6 | 0.171 [0.000–0.400] n=7 |
| semantic | hybrid, no translation | 1.000 [1.000–1.000] n=6 | 0.571 [0.200–1.000] n=7 | 0.714 [0.400–1.000] n=7 | 0.500 [0.000–1.000] n=6 | 0.171 [0.000–0.400] n=7 |
| semantic | hybrid + translation (shipping default) | 1.000 [1.000–1.000] n=6 | 0.629 [0.200–1.000] n=7 | 0.829 [0.400–1.000] n=7 | 0.333 [0.000–1.000] n=6 | 0.114 [0.000–0.200] n=7 |
| exact-identifier | vector, no translation | 1.000 [1.000–1.000] n=7 | 0.543 [0.200–1.000] n=7 | 0.943 [0.800–1.000] n=7 | 1.000 [1.000–1.000] n=7 | 0.200 [0.200–0.200] n=7 |
| exact-identifier | vector + translation | 1.000 [1.000–1.000] n=7 | 0.543 [0.200–1.000] n=7 | 0.943 [0.800–1.000] n=7 | 0.857 [0.000–1.000] n=7 | 0.171 [0.000–0.200] n=7 |
| exact-identifier | hybrid, no translation | 1.000 [1.000–1.000] n=7 | 0.457 [0.200–1.000] n=7 | 0.800 [0.600–1.000] n=7 | 1.000 [1.000–1.000] n=7 | 0.200 [0.200–0.200] n=7 |
| exact-identifier | hybrid + translation (shipping default) | 1.000 [1.000–1.000] n=7 | 0.486 [0.200–1.000] n=7 | 0.886 [0.600–1.000] n=7 | 1.000 [1.000–1.000] n=7 | 0.200 [0.200–0.200] n=7 |
| tool-augmented | vector, no translation | 0.857 [0.000–1.000] n=7 | 0.600 [0.000–1.000] n=7 | 0.943 [0.600–1.000] n=7 | 0.571 [0.000–1.000] n=7 | 0.143 [0.000–0.400] n=7 |
| tool-augmented | vector + translation | 1.000 [1.000–1.000] n=7 | 0.571 [0.200–1.000] n=7 | 1.000 [1.000–1.000] n=7 | 0.571 [0.000–1.000] n=7 | 0.143 [0.000–0.400] n=7 |
| tool-augmented | hybrid, no translation | 0.857 [0.000–1.000] n=7 | 0.457 [0.000–0.800] n=7 | 0.800 [0.400–1.000] n=7 | 0.429 [0.000–1.000] n=7 | 0.114 [0.000–0.400] n=7 |
| tool-augmented | hybrid + translation (shipping default) | 0.857 [0.000–1.000] n=7 | 0.514 [0.000–0.800] n=7 | 0.857 [0.600–1.000] n=7 | 0.714 [0.000–1.000] n=7 | 0.171 [0.000–0.400] n=7 |
| multi-hop | vector, no translation | 0.714 [0.000–1.000] n=7 | 0.714 [0.400–1.000] n=7 | 0.943 [0.800–1.000] n=7 | 0.464 [0.000–0.750] n=7 | 0.314 [0.000–0.600] n=7 |
| multi-hop | vector + translation | 0.786 [0.000–1.000] n=7 | 0.714 [0.400–1.000] n=7 | 0.971 [0.800–1.000] n=7 | 0.429 [0.000–0.750] n=7 | 0.286 [0.000–0.600] n=7 |
| multi-hop | hybrid, no translation | 0.714 [0.000–1.000] n=7 | 0.457 [0.200–0.600] n=7 | 0.771 [0.400–1.000] n=7 | 0.321 [0.000–0.667] n=7 | 0.200 [0.000–0.400] n=7 |
| multi-hop | hybrid + translation (shipping default) | 0.929 [0.500–1.000] n=7 | 0.571 [0.200–0.800] n=7 | 0.771 [0.400–1.000] n=7 | 0.405 [0.000–1.000] n=7 | 0.257 [0.000–0.600] n=7 |

## RAGAs — all four metrics, per bucket

| bucket | arm | faithfulness | answer relevancy ⚠ | context precision | context recall |
|---|---|---|---|---|---|
| semantic | vector, no translation | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) |
| semantic | vector + translation | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) |
| semantic | hybrid, no translation | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) |
| semantic | hybrid + translation (shipping default) | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) |
| exact-identifier | vector, no translation | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) |
| exact-identifier | vector + translation | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) |
| exact-identifier | hybrid, no translation | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) |
| exact-identifier | hybrid + translation (shipping default) | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) |
| tool-augmented | vector, no translation | — (n=0 of 7) | not comparable — 7/7 noncommittal | — (n=0 of 7) | — (n=0 of 7) |
| tool-augmented | vector + translation | — (n=0 of 7) | not comparable — 7/7 noncommittal | — (n=0 of 7) | — (n=0 of 7) |
| tool-augmented | hybrid, no translation | — (n=0 of 7) | not comparable — 7/7 noncommittal | — (n=0 of 7) | — (n=0 of 7) |
| tool-augmented | hybrid + translation (shipping default) | — (n=0 of 7) | not comparable — 7/7 noncommittal | — (n=0 of 7) | — (n=0 of 7) |
| multi-hop | vector, no translation | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) |
| multi-hop | vector + translation | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) |
| multi-hop | hybrid, no translation | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) |
| multi-hop | hybrid + translation (shipping default) | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) | — (n=0 of 7) |

⚠ **Response relevancy is excluded from every pre-registered decision — for two measured reasons, not one.** First: ragas forces temperature 0.3 whenever it asks for n > 1 completions (`ragas.llms.base.get_temperature`) and `ResponseRelevancy` asks for n=3, so this column moves between runs on any judge at any temperature this code names. **Second, and worse: the n=3 is not honoured.** The provider answers that single request with one completion, logging `LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.` on every judged cell of every run so far. So this column is a cosine similarity against **one** model-generated question sampled at temperature 0.3 — not the mean over three the metric is defined as. It is reported with its spread, read as nothing else, and quoted nowhere. ADR-0005's falsification clause and its §4 re-examination trigger both rest on context precision and context recall, which are single-call metrics at temperature 0.01.

## Ablations — the planner off (ADR-0004 §7, ADR-0005 §2)

Translation on, `max_sub_queries=0`: the deterministic ticker form is still added and no model is asked anything, which is the channel ADR-0005 §2 pre-registered for refuting its own revised hypothesis. Retrieval metrics only — these cells exist to attribute a retrieval result, and faithfulness over an answer is not evidence about which candidate list found a chunk.

| bucket | arm | section recall | section precision | filer precision | chunk recall † | chunk precision@k † |
|---|---|---|---|---|---|---|
| semantic | vector + normalisation, planner off (ADR-0004 §7 ablation) | 1.000 [1.000–1.000] n=6 | 0.771 [0.400–1.000] n=7 | 0.971 [0.800–1.000] n=7 | 0.333 [0.000–1.000] n=6 | 0.114 [0.000–0.200] n=7 |
| semantic | hybrid + normalisation, planner off (ADR-0005 §2 channel) | 1.000 [1.000–1.000] n=6 | 0.686 [0.200–1.000] n=7 | 0.743 [0.400–1.000] n=7 | 0.500 [0.000–1.000] n=6 | 0.171 [0.000–0.400] n=7 |
| exact-identifier | vector + normalisation, planner off (ADR-0004 §7 ablation) | 1.000 [1.000–1.000] n=7 | 0.543 [0.200–1.000] n=7 | 0.971 [0.800–1.000] n=7 | 1.000 [1.000–1.000] n=7 | 0.200 [0.200–0.200] n=7 |
| exact-identifier | hybrid + normalisation, planner off (ADR-0005 §2 channel) | 1.000 [1.000–1.000] n=7 | 0.486 [0.200–1.000] n=7 | 0.829 [0.600–1.000] n=7 | 1.000 [1.000–1.000] n=7 | 0.200 [0.200–0.200] n=7 |
| tool-augmented | vector + normalisation, planner off (ADR-0004 §7 ablation) | 0.857 [0.000–1.000] n=7 | 0.600 [0.000–1.000] n=7 | 1.000 [1.000–1.000] n=7 | 0.714 [0.000–1.000] n=7 | 0.171 [0.000–0.400] n=7 |
| tool-augmented | hybrid + normalisation, planner off (ADR-0005 §2 channel) | 0.857 [0.000–1.000] n=7 | 0.429 [0.000–0.800] n=7 | 0.743 [0.400–1.000] n=7 | 0.429 [0.000–1.000] n=7 | 0.114 [0.000–0.400] n=7 |
| multi-hop | vector + normalisation, planner off (ADR-0004 §7 ablation) | 0.714 [0.000–1.000] n=7 | 0.714 [0.400–1.000] n=7 | 1.000 [1.000–1.000] n=7 | 0.393 [0.000–0.667] n=7 | 0.257 [0.000–0.400] n=7 |
| multi-hop | hybrid + normalisation, planner off (ADR-0005 §2 channel) | 0.786 [0.000–1.000] n=7 | 0.571 [0.200–0.800] n=7 | 0.800 [0.200–1.000] n=7 | 0.369 [0.000–1.000] n=7 | 0.257 [0.000–0.600] n=7 |

## Pre-registered hypotheses — prediction, then measurement, then verdict

| # | prediction (pre-registered) | measurement | verdict |
|---|---|---|---|
| 1 | **H1** hybrid > vector-only on `exact-identifier` — ADR-0002's original prediction, which ADR-0004 §7 narrows to *at equal translation off* and expects to be where the bucket's win comes from **least** <br>*ADR-0002 decision 4; narrowed by ADR-0004 §7* | context precision on `exact-identifier`, hybrid − vector: Δ — against per-question spread — → **undetermined** (baseline —, candidate —) | **undetermined** |
| 2 | **H2** translation is **positive** on `exact-identifier`, via entity normalisation — ADR-0004 §6 revising the earlier '≈ neutral', on one root-caused case where the target chunk moved from absent to rank 1 <br>*ADR-0004 §6 (T6 amendment)* | context precision on `exact-identifier`, hybrid+translation − hybrid: Δ — against per-question spread — → **undetermined** (baseline —, candidate —) | **undetermined** |
| 3 | **H3** translation wins on `multi-hop` — decomposition gives the retrievers something a filing actually answers <br>*ADR-0002 decision 4* | context recall on `multi-hop`, hybrid+translation − hybrid: Δ — against per-question spread — → **undetermined** (baseline —, candidate —) | **undetermined** |
| 4 | **H4** all configurations ≈ tie on `semantic`. **A predicted tie is evidence the experiment is sound, not a failure** (ADR-0002 decision 4) <br>*ADR-0002 decision 4* | context precision on `semantic`, hybrid+translation − vector+translation: Δ — against per-question spread — → **undetermined** (baseline —, candidate —) | **undetermined** |
| 5 | **H5** hybrid's marginal contribution over `vector + translation` on `exact-identifier` is **small** — ADR-0004 §7, because §6 measured the recovery as embedding-side and BM25's role in it as redundancy rather than recovery <br>*ADR-0004 §7 (pre-data)* | context precision on `exact-identifier`, hybrid+translation − vector+translation: Δ — against per-question spread — → **undetermined** (baseline —, candidate —) | **undetermined** |
| 6 | **H6** hybrid's contribution may be **larger on `semantic`** than on the bucket it exists to win, after ADR-0004 §10 stopped BM25 admitting chunks on question-form terms <br>*ADR-0004 §7/§10 (pre-data)* | context precision on `semantic`, hybrid+translation − vector+translation: Δ — against per-question spread — → **undetermined** (baseline —, candidate —) | **undetermined** |

## The two pre-registered decisions

### Falsification clause (ADR-0005) — translation, per bucket

Fires only when translation is worse on **both** context precision **and** context recall within a bucket, each by more than its own per-question spread. Directional and two-sided by design: with ~7 questions per bucket a tight numeric margin would be false precision, and dropping a pre-registered default on half the evidence would be worse than keeping it.

| bucket | context precision | context recall | clause |
|---|---|---|---|
| semantic | undetermined | undetermined | does not fire |
| exact-identifier | undetermined | undetermined | does not fire |
| tool-augmented | undetermined | undetermined | does not fire |
| multi-hop | undetermined | undetermined | does not fire |

**Outcome: the clause does not fire on any bucket, so the pre-committed default stands: hybrid + translation (shipping default).**

### Re-examination trigger (ADR-0005 §4) — the strategy axis

Fires on an **absence of gain**, not on a loss: if `hybrid − vector` at equal translation is within per-question spread on *every* bucket, the dominance argument is re-argued rather than defended and `vector + translation` becomes a live candidate for the default. Registered before any number existed, because a default kept because its marginal component was never separately measured is p-hacking in the other direction.

| bucket | hybrid − vector, both +translation |
|---|---|
| semantic | undetermined |
| exact-identifier | undetermined |
| tool-augmented | undetermined |
| multi-hop | undetermined |

**Outcome: the trigger does not fire.** Hybrid's contribution exceeds per-question spread on at least one bucket, so the dominance argument stands as argued.

Undetermined on: semantic, exact-identifier, tool-augmented, multi-hop. Those buckets produced no comparable number, which weakens the conclusion in whichever direction it went — an absence is not evidence for either arm.

## Latency and token spend

Measured from the persisted event log (`FINBRIEF_LOG_FILE`), not from a stopwatch: a stopwatch around `retrieve()` cannot split the planner's chat round from the retrieval rounds, and ADR-0004's amendment makes that split the interesting half of the budget.

| | ms | samples |
|---|---:|---:|
| planner's chat round, p50 | 1518 | 28 |
| retrieval p50, translation on | 1439 | 196 |
| retrieval p50, translation off | 328 | 104 |
| retrieval rounds translation adds, p50 | 1110 | — |
| **total p50 added by translation** | **2628** | — |
| ADR-0005's budget | 1500 | — |

**Verdict: **over budget**.**

**The planner's figure comes from the resolve pass, and that is a reconstruction rather than one measurement.** ADR-0004 §9's replay means the scored `+translation` arms serve the planner's reply from a stub, so their own `query_translation` lines record ~1 ms and no token counts — the harness only reads lines that reported spend, since a line with no `input_tokens` called no model. Adding that median to the retrieval delta is the honest reconstruction of what the shipped path pays; it is not a single timing of a live turn.

| metered event | input tokens | output tokens | lines | unmetered lines |
|---|---:|---:|---:|---:|
| `rag_answer` | 130504 (104 calls) | 14048 (104 calls) | 104 | 0 |
| `query_translation` | 7748 (28 calls) | 1717 (28 calls) | 418 | 390 |

An unmetered line is a call whose cost is **unknown**, not free (`observability/tokens.py`): each count carries its own denominator because a provider that reports half a pair must not put a fabricated zero on the line. The gate classifier's tokens are unmeasured by decision (ADR-0011), and the embeddings API returns no usage this code path can see.

## Deferred measurements — reported, or named as not run

Four measurements were deferred to this ticket by earlier ones. They are listed here whether or not they ran, because an unscored criterion reads as a passed one. Three of the four need **live agent turns**, which is a different and nondeterministic instrument from the chain every table above measures — so they belong beside the RAGAs tables and never inside them.

| deferred measurement | from | instrument | result |
|---|---|---|---|
| agent-vs-original query divergence rate | T4 (ADR-0003 amendment §2) | `agent_query.verbatim` over live agent turns | **not measured by this run** |
| square-bracket rule adherence rate | T5 (ADR-0006 T7 amendment §4) | `citation_markers` over live agent turns | **not measured by this run** |
| layer 4's residue — advice phrased so no rule matches | T7 (ADR-0006 T7 amendment §4) | advice probes through the live agent and `security.advice.validate_answer` | **not measured by this run** |
| faithfulness on markers that resolve but sit on unsupported claims | T3/T5 | per-sentence NLI against the *cited* chunk, not the whole context set | **not measured by this run** |

None of the four ran. The events they read are emitted and their readers exist (`observability/events.py`, `evaluation/latency.py`); what is missing is a live-run sample large enough to publish a rate from. The only sample that exists is the one ADR-0003's T4 amendment already records — 0 verbatim of 2 searches across two conversations — and that amendment says itself that it is too small to publish.

## The numbers T11's README will quote

Measured on the shipping default (hybrid + translation (shipping default)), over 28 golden-set rows. **Quote these from here, not from prose.**

| figure | value | denominator |
|---|---|---|
| RAGAs faithfulness | — (n=0 of 28) | rows scored on the default arm |
| RAGAs context precision | — (n=0 of 28) | as above |
| RAGAs context recall | — (n=0 of 28) | as above |
| section recall (free, deterministic) | 0.944 [0.000–1.000] n=27 | non-trivial target sections |
| judge calls this run paid for | 0 | at k=5, six arms |
| cells replayed from cache | 128 | see the provenance table |

**Response relevancy is deliberately absent from this list.** It is reported in the RAGAs table with its spread, and it is excluded from every pre-registered decision — quoting it as a headline number would be quoting a figure that moves between runs.
