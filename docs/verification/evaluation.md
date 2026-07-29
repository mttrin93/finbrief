# Evaluation — RAGAs, per-bucket A/B, tool-calling (T10, #11)

<!-- GENERATED FILE. Rewritten by `uv run python scripts/evaluate.py`. Do not edit: every number here is a measurement of the run named below. -->

| | |
|---|---|
| generated | 2026-07-29 12:13 UTC |
| stages run | resolve, retrieve, answer, judge, agent, report |
| judge model | `openai/gpt-4.1-mini` (ragas 0.4.3) |
| answering model | `openai/gpt-4o-mini` |
| planner model | `openai/gpt-4o-mini` |
| embedding model | `openai/text-embedding-3-small` |
| k | 5 |
| golden set | 28 rows, verified against EDGAR |
| collection | 2026-07-27 08:51 UTC, 5842 chunks, docs/verification/ingest-report.md |
| collection fingerprint | `c986299c02043ee8…` |
| cache | answer 112 replayed / 0 paid, judge 560 replayed / 0 paid, retrieval 168 replayed / 0 paid, variants 28 replayed / 0 paid |

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
| semantic | vector, no translation | 0.963 [0.867–1.000] n=7 | 0.876 [0.659–1.000] n=7 | 0.584 [0.000–1.000] n=7 | 0.667 [0.000–1.000] n=7 |
| semantic | vector + translation | 0.958 [0.875–1.000] n=7 | 0.874 [0.649–1.000] n=7 | 0.521 [0.000–1.000] n=7 | 0.655 [0.000–1.000] n=7 |
| semantic | hybrid, no translation | 0.742 [0.000–1.000] n=7 | 0.917 [0.759–1.000] n=7 | 0.596 [0.000–1.000] n=7 | 0.667 [0.000–1.000] n=7 |
| semantic | hybrid + translation (shipping default) | 0.892 [0.636–1.000] n=7 | 0.919 [0.749–1.000] n=7 | 0.529 [0.000–1.000] n=7 | 0.571 [0.000–1.000] n=7 |
| exact-identifier | vector, no translation | 0.698 [0.500–1.000] n=7 | 0.882 [0.672–0.945] n=7 | 0.858 [0.500–1.000] n=7 | 1.000 [1.000–1.000] n=7 |
| exact-identifier | vector + translation | 0.696 [0.500–1.000] n=7 | 0.912 [0.848–0.945] n=7 | 0.798 [0.500–1.000] n=7 | 0.952 [0.667–1.000] n=7 |
| exact-identifier | hybrid, no translation | 0.357 [0.000–1.000] n=7 | 0.900 [0.720–0.945] n=7 | 0.771 [0.367–1.000] n=7 | 0.952 [0.667–1.000] n=7 |
| exact-identifier | hybrid + translation (shipping default) | 0.643 [0.000–1.000] n=7 | 0.900 [0.721–0.945] n=7 | 0.874 [0.367–1.000] n=7 | 1.000 [1.000–1.000] n=7 |
| tool-augmented | vector, no translation | 0.887 [0.700–1.000] n=7 | not comparable — 6/7 noncommittal | 0.357 [0.000–1.000] n=7 | 0.595 [0.000–1.000] n=7 |
| tool-augmented | vector + translation | 0.920 [0.722–1.000] n=7 | not comparable — 6/7 noncommittal | 0.521 [0.000–1.000] n=7 | 0.655 [0.250–1.000] n=7 |
| tool-augmented | hybrid, no translation | 0.919 [0.667–1.000] n=7 | not comparable — 6/7 noncommittal | 0.448 [0.000–1.000] n=7 | 0.452 [0.000–1.000] n=7 |
| tool-augmented | hybrid + translation (shipping default) | 0.914 [0.667–1.000] n=7 | not comparable — 4/7 noncommittal | 0.401 [0.000–1.000] n=7 | 0.619 [0.000–1.000] n=7 |
| multi-hop | vector, no translation | 0.778 [0.500–1.000] n=7 | 0.488 [0.000–0.971] n=7 | 0.389 [0.000–1.000] n=7 | 0.471 [0.000–0.800] n=7 |
| multi-hop | vector + translation | 0.733 [0.357–1.000] n=7 | 0.577 [0.000–0.905] n=7 | 0.240 [0.000–1.000] n=7 | 0.460 [0.333–0.750] n=7 |
| multi-hop | hybrid, no translation | 0.896 [0.500–1.000] n=7 | 0.544 [0.000–0.876] n=7 | 0.290 [0.000–1.000] n=7 | 0.502 [0.250–0.833] n=7 |
| multi-hop | hybrid + translation (shipping default) | 0.582 [0.000–1.000] n=7 | 0.708 [0.000–0.944] n=7 | 0.370 [0.000–1.000] n=7 | 0.509 [0.143–1.000] n=7 |

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
| 1 | **H1** hybrid > vector-only on `exact-identifier` — ADR-0002's original prediction, which ADR-0004 §7 narrows to *at equal translation off* and expects to be where the bucket's win comes from **least** <br>*ADR-0002 decision 4; narrowed by ADR-0004 §7* | context precision on `exact-identifier`, hybrid − vector: Δ -0.087 against per-question spread 0.633 → **within spread** (baseline 0.858 n=7, candidate 0.771 n=7) | **refuted** |
| 2 | **H2** translation is **positive** on `exact-identifier`, via entity normalisation — ADR-0004 §6 revising the earlier '≈ neutral', on one root-caused case where the target chunk moved from absent to rank 1 <br>*ADR-0004 §6 (T6 amendment)* | context precision on `exact-identifier`, hybrid+translation − hybrid: Δ +0.102 against per-question spread 0.633 → **within spread** (baseline 0.771 n=7, candidate 0.874 n=7) | **refuted** |
| 3 | **H3** translation wins on `multi-hop` — decomposition gives the retrievers something a filing actually answers <br>*ADR-0002 decision 4* | context recall on `multi-hop`, hybrid+translation − hybrid: Δ +0.006 against per-question spread 0.857 → **within spread** (baseline 0.502 n=7, candidate 0.509 n=7) | **refuted** |
| 4 | **H4** all configurations ≈ tie on `semantic`. **A predicted tie is evidence the experiment is sound, not a failure** (ADR-0002 decision 4) <br>*ADR-0002 decision 4* | context precision on `semantic`, hybrid+translation − vector+translation: Δ +0.007 against per-question spread 1.000 → **within spread** (baseline 0.521 n=7, candidate 0.529 n=7) | **confirmed** |
| 5 | **H5** hybrid's marginal contribution over `vector + translation` on `exact-identifier` is **small** — ADR-0004 §7, because §6 measured the recovery as embedding-side and BM25's role in it as redundancy rather than recovery <br>*ADR-0004 §7 (pre-data)* | context precision on `exact-identifier`, hybrid+translation − vector+translation: Δ +0.076 against per-question spread 0.633 → **within spread** (baseline 0.798 n=7, candidate 0.874 n=7) | **confirmed** |
| 6 | **H6** hybrid's contribution may be **larger on `semantic`** than on the bucket it exists to win, after ADR-0004 §10 stopped BM25 admitting chunks on question-form terms <br>*ADR-0004 §7/§10 (pre-data)* | context precision on `semantic`, hybrid+translation − vector+translation: Δ +0.007 against per-question spread 1.000 → **within spread** (baseline 0.521 n=7, candidate 0.529 n=7) | **refuted** |

## The two pre-registered decisions

### Falsification clause (ADR-0005) — translation, per bucket

Fires only when translation is worse on **both** context precision **and** context recall within a bucket, each by more than its own per-question spread. Directional and two-sided by design: with ~7 questions per bucket a tight numeric margin would be false precision, and dropping a pre-registered default on half the evidence would be worse than keeping it.

| bucket | context precision | context recall | clause |
|---|---|---|---|
| semantic | within spread | within spread | does not fire |
| exact-identifier | within spread | within spread | does not fire |
| tool-augmented | within spread | within spread | does not fire |
| multi-hop | within spread | within spread | does not fire |

**Outcome: the clause does not fire on any bucket, so the pre-committed default stands: hybrid + translation (shipping default).**

### Re-examination trigger (ADR-0005 §4) — the strategy axis

Fires on an **absence of gain**, not on a loss: if `hybrid − vector` at equal translation is within per-question spread on *every* bucket, the dominance argument is re-argued rather than defended and `vector + translation` becomes a live candidate for the default. Registered before any number existed, because a default kept because its marginal component was never separately measured is p-hacking in the other direction.

| bucket | hybrid − vector, both +translation |
|---|---|
| semantic | within spread |
| exact-identifier | within spread |
| tool-augmented | within spread |
| multi-hop | within spread |

**Outcome: the trigger FIRES — hybrid adds nothing beyond spread on any settled bucket, so ADR-0005's dominance argument is re-argued rather than defended.**

## Latency and token spend

Measured from the persisted event log (`FINBRIEF_LOG_FILE`), not from a stopwatch: a stopwatch around `retrieve()` cannot split the planner's chat round from the retrieval rounds, and ADR-0004's amendment makes that split the interesting half of the budget.

| | ms | samples |
|---|---:|---:|
| planner's chat round, p50 | 1518 | 44 |
| retrieval p50, translation on | 1546 | 240 |
| retrieval p50, translation off | 328 | 104 |
| retrieval rounds translation adds, p50 | 1218 | — |
| **total p50 added by translation** | **2736** | — |
| ADR-0005's budget | 1500 | — |

**Verdict: **over budget**.**

**The planner's figure comes from the resolve pass, and that is a reconstruction rather than one measurement.** ADR-0004 §9's replay means the scored `+translation` arms serve the planner's reply from a stub, so their own `query_translation` lines record ~1 ms and no token counts — the harness only reads lines that reported spend, since a line with no `input_tokens` called no model. Adding that median to the retrieval delta is the honest reconstruction of what the shipped path pays; it is not a single timing of a live turn.

| metered event | input tokens | output tokens | lines | unmetered lines |
|---|---:|---:|---:|---:|
| `rag_answer` | 165140 (132 calls) | 17954 (132 calls) | 132 | 0 |
| `query_translation` | 12004 (44 calls) | 2682 (44 calls) | 602 | 558 |
| `agent_turn` | 117074 (20 calls) | 5522 (20 calls) | 20 | 0 |

An unmetered line is a call whose cost is **unknown**, not free (`observability/tokens.py`): each count carries its own denominator because a provider that reports half a pair must not put a fabricated zero on the line. The gate classifier's tokens are unmeasured by decision (ADR-0011), and the embeddings API returns no usage this code path can see.

## Tool-calling eval — the selection layer

ADR-0003's *other* half: every table above measures the deterministic chain, and this measures the agent's selection layer on top of it. A different instrument, and a nondeterministic one — so it is reported beside those tables and never inside them.

**Tool-selection accuracy: 100% over 10 scored cases** (100% over the seven golden-set rows alone; the other three are controls the golden set cannot express, because every row in it expects a tool).

**Valuation pairing (#9's AC-1 hole): 1/2.** `AGENT_SYSTEM_PROMPT` says a valuation question wants the quote *and* the peer comparison, while `tool_expectation` records one tool per row — so this is measured as its own rate rather than by reshaping the committed golden set. Measured, not enforced.

| case | question | expected | called | verdict | note |
|---|---|---|---|---|---|
| T1 | What does NVIDIA do, and what is its stock trading at right now? | get_stock_data, search_filings | search_filings(), get_stock_data(NVDA) | **pass** | golden-set row T1 |
| T2 | Ford says it manages leverage to hold investment-grade ratings — how does its actual debt-to-equity compare with its peers? | calculate_ratios, search_filings | search_filings(), calculate_ratios(F) | **pass** | golden-set row T2 |
| T3 | Any recent news related to the supply-chain risks Tesla flags in its 10-K? | get_recent_news, search_filings | search_filings(), get_recent_news(TSLA) | **pass** | golden-set row T3 |
| T4 | Amazon's 10-K describes three reportable segments — what are they, and what is Amazon's current market cap? | get_stock_data, search_filings | search_filings(), get_stock_data(AMZN) | **pass** | golden-set row T4 |
| T5 | Eli Lilly says a handful of products drive most of its revenue. What's its current P/E versus its healthcare peers? | calculate_ratios, search_filings | search_filings(), get_stock_data(LLY), calculate_ratios(LLY) | **pass** | golden-set row T5 |
| T6 | What are Microsoft's reportable segments, and what's the latest news on the company? | get_recent_news, search_filings | search_filings(), get_recent_news(MSFT) | **pass** | golden-set row T6 |
| T7 | GM warns about tariffs in its risk factors — is there anything in the recent news about that? | get_recent_news, search_filings | search_filings(), get_recent_news(GM) | **pass** | golden-set row T7 |
| C1 | What are the main risk factors for Tesla? | search_filings | search_filings() | **pass** | retrieval-only: reaching for a quote on a risk-factors question is an error accuracy over positive cases cannot see |
| C2 | What is Nestle's dividend policy? | no finance tool | — | **pass** | out-of-Universe: the whitelist should make this a refusal-as-result rather than a fetch (user story 21, ADR-0009) |
| C3 | Should I buy Ford shares today? | no finance tool | — | **pass** | advice-shaped: scored here only on whether a tool fired — the refusal itself is layer 4's and the security suite's |

## Deferred measurements — reported, or named as not run

Four measurements were deferred to this ticket by earlier ones. They are listed here whether or not they ran, because an unscored criterion reads as a passed one. Three of the four need **live agent turns**, which is a different and nondeterministic instrument from the chain every table above measures — so they belong beside the RAGAs tables and never inside them.

| deferred measurement | from | instrument | result |
|---|---|---|---|
| agent-vs-original query divergence rate | T4 (ADR-0003 amendment §2) | `agent_query.verbatim` over live agent turns | divergence rate: **100%** (24/24); first search in a thread: **100%** (8/8); later searches (may be permitted resolutions): **100%** (16/16) |
| square-bracket rule adherence rate | T5 (ADR-0006 T7 amendment §4) | `citation_markers`, emitted by `app/Home.py` and by nothing else | **not measured by this run** |
| layer 4's residue — advice phrased so no rule matches | T7 (ADR-0006 T7 amendment §4) | advice probes through the live agent and `security.advice.validate_answer` | layer-4 residue: **100%** (6/6) — 6 of 6 hand-labelled recommendations were **not** refused |
| faithfulness on markers that resolve but sit on unsupported claims | T3/T5 | per-sentence NLI against the *cited* chunk, not the whole context set | **not measured by this run** |

**Why the bracket-rule rate is absent, and it is not for want of running the agent.** `citation_markers` is emitted by `app/Home.py` — the *only* caller of `security.markers.log_markers` — and not by `agent.answer`. So a harness that drives the agent directly, as the tool-calling eval above does, produces the turns and none of the lines. That is the same shape T8 found and fixed for `turn_id`: an instrument wired at the app is an instrument a harness cannot reach, and neutralising it there left every test green. Closing this needs the marker check moved to where the answer is produced rather than to where it is displayed, which is a change to shipped code and belongs in its own ticket rather than in a measurement run.

**Layer 4's residue is the sharpest of the four, and it cost nothing to measure.** The validator is a regex rule set behind a Guard, so what it *misses* is computable with no model and no live run — this deferral could have been closed at any point since T7. What it shows: the rules catch advice that announces itself (`ADVICE_ANSWERS`, all refused, reported by the security suite) and refuse none of the recommendations that carry no imperative, no rating word, no price target and no position-sizing instruction. n is small and hand-authored, so this is a statement about the rules' **generality**, not a 100%-evasion claim about indirect advice.

## The numbers T11's README will quote

Measured on the shipping default (hybrid + translation (shipping default)), over 28 golden-set rows. **Quote these from here, not from prose.**

| figure | value | denominator |
|---|---|---|
| RAGAs faithfulness | 0.758 [0.000–1.000] n=28 | rows scored on the default arm |
| RAGAs context precision | 0.543 [0.000–1.000] n=28 | as above |
| RAGAs context recall | 0.675 [0.000–1.000] n=28 | as above |
| section recall (free, deterministic) | 0.944 [0.000–1.000] n=27 | non-trivial target sections |
| judge calls this run paid for | 0 | at k=5, six arms |
| cells replayed from cache | 868 | see the provenance table |

**Response relevancy is deliberately absent from this list.** It is reported in the RAGAs table with its spread, and it is excluded from every pre-registered decision — quoting it as a headline number would be quoting a figure that moves between runs.
