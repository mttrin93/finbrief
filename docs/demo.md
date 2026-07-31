# The five-step demo walkthrough

The demo script, kept out of [the README](../README.md#13-the-demo-walkthrough) so the tour stays
a ten-minute read. Every step is reproducible as written, and the preconditions at the bottom are
the two things that will spoil a recording if they are skipped.

---

## 1.3 The five-step demo walkthrough

Reproducible exactly as written. Each step fires a different requirement, and the second one is
the interesting one.

| # | ask this | what to watch | requirement | needs the sink? |
|---|---|---|---|---|
| 1 | *What are the main risk factors for Tesla?* | five `TSLA 10-K FY2025, Item 1A` sources, `[1]`–`[5]`, each accession linked to EDGAR | RAG, citations, sources panel | no |
| 2 | *And what does it say about its debt?* | the follow-up names no company and still resolves; numbering **continues** `[6]`–`[10]`; now scroll back up — `[1]` still resolves to the same panel entry. Then open *How I answered* | conversation memory, the thread-wide citation register, hybrid + translation | no |
| 3 | *How does its valuation compare to its fundamentals?* | `get_stock_data` and `calculate_ratios` cards; one chart per ratio metric; the peer set, its size **and its range** | tool calling, tool-result visualisation | no |
| 4 | *Give me the full brief on Tesla.* | the `st.status` block naming each step as it runs; four sections; news cards. **This is the hero-GIF beat** | combined multi-tool query, progress indicators | no |
| 5 | *Should I buy Tesla stock?* — then a denylisted injection payload | the advice question is **allowed through the front door** and refused at layer 4 with a disclaimer; the payload is blocked at layer 2 in ~0 ms and the agent is never called | advice refusal, injection resistance | **yes** — the gate-trigger record exists nowhere else |

**Step 2 is worth narrating carefully, because the obvious narration is wrong.** Under
vector-only retrieval this follow-up ranked four Ford chunks above the one correct Tesla
passage. Hybrid search *alone* made it worse — the passage left the top five entirely. What
recovers it is **deterministic entity normalisation** (a `config` lookup adding a `TSLA debt`
variant) and then **RRF's agreement principle**: the chunk is found by two independent query
variants, and agreement beats any single strong hit. The *How I answered* panel labels the
ticker form as a ticker form, not as "sub-query 1", precisely so a `config` lookup is not
credited to a model. The full before/after is in [3.15](implementation.md#315-hybrid-search).

One thing to keep honest on camera: under the shipped default the **top** entry is a `TSLA
Item 1` chunk that is not about debt — it wins on three BM25 votes across sub-queries. Precision
is much better than before (5/5 correct filer against 1/5) and the answer cites the liquidity
passage, but this is not a perfect ranking and `n=1`.

### Before you record, two preconditions

**Warm the quote cache.** Run one `get_stock_data` per ticker you plan to show, a couple of
minutes ahead. A stalled Yahoo endpoint can hold the quote cache lock for
`config.QUOTE_FETCH_WORST_CASE_SECONDS` — **91.5 s** — because `FETCH_TIMEOUT_SECONDS` bounds a
single HTTP *request*, one `fetch_quote` makes two, `FETCH_ATTEMPTS` is 3, and `TimedCache`
holds its lock across the whole retry sequence. Step 3 on a `big_tech` company resolves **six**
quotes through that cache (peers go through the same cached path — ADR-0009), so a cold first
take is the slowest possible take. `QUOTE_TTL_SECONDS` is 900, so a 15-minute window covers a
take comfortably and every fetch after the warm-up is a cache hit. A stall is not a crash: the
`st.status` block sits on "Fetching …", the page stays responsive, and the card comes back
either stale-with-its-age or saying the figure could not be fetched.

**Enable the event sink** for step 5, and for any step whose numbers you intend to quote:

```bash
FINBRIEF_LOG_FILE=data/events.jsonl uv run streamlit run app/Home.py
```

Unset, events go to stderr and vanish with the process, and gate-trigger metadata, token counts
and latency samples exist nowhere else. One consequence to accept deliberately: a **blocked**
question's normalised text is kept on disk, bounded by `config.GATE_LOGGED_INPUT_MAX_CHARS` —
the one documented exception to the no-user-content rule (ADR-0006, ADR-0011). The file is
gitignored, append-only and never rotated; it is a run artifact, not one of the five committed
evidence files.

---

---
