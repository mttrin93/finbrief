# FinBrief — Financial Research Assistant

A domain-specialised RAG chatbot for equity research. Target user: a junior analyst
who needs a grounded, source-cited company snapshot before an earnings call —
business overview, risk factors, current valuation, and recent news — in minutes.

> **Status:** a conversational agent over the knowledge base. A question reaches
> `create_agent`, which searches the Chroma `filings` collection through its
> `search_filings` tool and answers with inline `[n]` citations, each resolving to a sources
> panel entry with that chunk's ticker, Section, fiscal year and accession — so a claim can be
> checked against the filing that made it. Follow-ups work: *"and its debt?"* resolves against
> the company just discussed, because the conversation lives in a `SqliteSaver` checkpointer
> rather than in the UI (ADR-0008). The KB holds curated 10-K Sections for all fifteen
> Universe companies, fetched from EDGAR, gated, chunked, and persisted (see *Building the
> knowledge base* below).
>
> Retrieval is **hybrid + query translation**, the pre-registered shipping default (ADR-0005),
> and the sidebar now names one configuration because the configured one is what answers.
> One symmetric pipeline: the analyst's question is always retained as a query variant,
> translation only ever *adds* to it — a deterministic **ticker form** when a Universe company
> is named by name (a `config` lookup, not a model) plus up to three planner sub-queries, so
> 1 + ≤1 + ≤3 = five variants and ten candidate lists under hybrid — every variant runs through
> both BM25 and vector, and the
> candidate lists are fused with Reciprocal Rank Fusion (`k = 60`, the published default,
> stated and not tuned), deduplicated by chunk id and truncated to top-k. Every answer that
> searched the filings carries
> a **"How I answered"** panel showing the queries that ran and, per chunk, which variant ×
> which retriever surfaced it and what it contributed to the fused score — so *why* a chunk
> was retrieved is checkable on the turn itself, not only in an aggregate table. Building it
> falsified a pre-registered hypothesis, which is written up in ADR-0004's T6 amendment and on
> [issue #6](https://github.com/TuringCollegeSubmissions/mrinal-AE.AFA.3.5/issues/6).
>
> **Three finance tools now sit beside the search tool** (T5, [#9](https://github.com/TuringCollegeSubmissions/mrinal-AE.AFA.3.5/issues/9)),
> so the tool-*selection* the agent loop exists for is finally exercised: `get_stock_data`,
> `calculate_ratios` and `get_recent_news` over free public data, each result rendered as a card
> or chart beside the answer and each figure stated in the units a note quotes. Ratios compare a
> company against the mean of its own curated cluster inside the Universe, never an outside
> ticker (ADR-0009), and every comparison carries its peer set, its size **and its range** — with
> two peers a single outlier moves a mean a long way. A figure a source does not report reads
> *not reported*, never `0.0`. When a source cannot be refreshed the last good figure is shown
> with a banner saying how old it is. The security gate is the phase that follows.
>
> The three files under
> [`docs/verification/`](./docs/verification/) are generated run evidence, never
> hand-authored. The plan lives in [`PLAN.md`](./PLAN.md), the Tier-1 spec in
> [`docs/spec/finbrief.md`](./docs/spec/finbrief.md), the domain language in
> [`CONTEXT.md`](./CONTEXT.md), and the design decisions in [`docs/adr/`](./docs/adr/).

## Stack

- **Python** · **Streamlit** UI · **LangChain / LangGraph** (`create_agent`)
- **OpenRouter** for LLM access (OpenAI-compatible SDK)
- **ChromaDB** vector store · hybrid retrieval (BM25 + vectors)
- Data: SEC EDGAR filings via **edgartools** (structure-anchored section extraction, so
  there is no hand-rolled primary parser — ADR-0007), yfinance, news RSS, ECB/Fed
  publications

## What answers are grounded in

**Filing** answers are grounded in Items 1, 1A, 7 and 7A of the latest annual 10-K on file for
each of the 15
companies in FinBrief's Universe — **54 of 60** company × Section pairs. *Filing* answers, and
not every answer, because the finance tools put live figures in front of the analyst too — those
have their own scope, below. The app states this
under its title and in a sidebar panel from the same
[`src/finbrief/prompts.py`](./src/finbrief/prompts.py) text the model is given, so the page
and the persona cannot disagree about what is grounded (user story 18, ADR-0007). Every
number is derived from `config.py` and the `Section` enum, never typed; the counts below are
cross-checked against the ingest run's own evidence by `tests/test_grounding_scope.py`.

- **In scope:** Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A), Item 7A
  (Quantitative and Qualitative Disclosures About Market Risk).
- **Item 7A by reference:** BAC, GS, JNJ, JPM, LLY, PFE answer Item 7A by incorporating
  Item 7, so their market-risk disclosure is in the knowledge base labelled `Item 7` — not
  `Item 7A`. All 15 companies have market-risk grounding; 9 have an `Item 7A` Section. A
  reader who does not know this reads "no Item 7A" as "no market-risk grounding", which is
  why the app says it on screen rather than only here.
- **Out of scope:** every other Item of the 10-K, 10-Qs, proxies, earnings calls, and any
  company outside the Universe. Financial statements (Item 8) are not ingested, and table
  and figure fidelity inside the ingested Sections is a stated limitation — hard numbers
  come from the finance tools, not the filing text.
- **One filing per company:** the most recent 10-K only, so the fiscal year differs by
  filer (NVDA is FY2026, the other fourteen FY2025). Each citation states its own year.

Retrieved text is shown verbatim in the sources panel: it renders through `st.text`, not
Markdown, because a filer's own `$178,353` is a KaTeX expression to a Markdown renderer and
a citation surface that silently reformats the figures is not a citation surface.

## What the live figures are, and are not

Price, ratios and headlines come from three tools over free public data, never from the filings
— a 10-K has no prices in it. Their limits are stated because "live" is a word a reader will
over-read:

- **Delayed, and cached for 15 minutes.** The free quote feed is itself delayed by roughly that
  much, so a shorter TTL would spend a call to re-fetch a number that cannot have changed. It is
  a recent quote, not a tick, and the app says so.
- **Peers are the company's own curated cluster inside the Universe** and are never chosen by the
  model or drawn from outside (ADR-0009). Every comparison names its peer set and size — *vs.
  mean of 2 `autos` peers: TSLA, GM* — and reports the **range** beside the mean, because with
  two or three peers one outlier dominates an arithmetic mean: Ford's peers are TSLA at 286× and
  GM at 37×, and the mean of 162× describes neither.
- **A figure a source does not report reads *not reported*, never `0.0`.** This is routine rather
  than defensive: JPM and BAC report no debt-to-equity at all, so a `banks` leverage comparison
  rests on GS alone and says "1 of 2 peers reported this". A zero D/E on a bank's card would say
  it carries no leverage.
- **When a fetch fails**, the last good figure is shown with a banner giving its age; when
  nothing is cached, the answer says the figure could not be fetched. No number is ever a
  placeholder, and the assistant is told not to supply one from memory.
- **News summaries are third-party text**, HTML-stripped and quarantined as data before the model
  sees them, with only `http(s)` links rendered — a syndicated feed is writable by strangers
  (ADR-0006).
- **yfinance is unofficial** and Alpha Vantage's free tier is 25 calls a day. FinBrief reads only
  the first; the TTL cache, the retry and the stale banner are how it degrades rather than a
  second data source. `ALPHAVANTAGE_API_KEY` and `FINBRIEF_ALPHAVANTAGE_ENABLED` exist in the
  configuration and nothing reads them yet — a fundamentals fallback is deferred, not shipped.

## Conversation memory, and who owns it

The agent's memory of record is a file-backed `SqliteSaver` checkpointer; `st.session_state`
holds only the `thread_id`, UI state, and the transcript on screen (ADR-0008). `answer()` is
handed the new question and a thread id, never a history — a history assembled in the UI would
be a second copy of the conversation, and the copy the model never sees is the one that goes
stale.

- **One `uuid4` per browser session**, minted before the first message and stable across
  reruns. The agent and the checkpointer are built once per *process* and shared by every
  session, so that id is what keeps two users apart. `tests/test_app_state.py` drives two
  `AppTest` sessions in one process and asserts they get different threads and that each turn
  lands on its own — the cross-user leak this design exists to prevent, provable without
  deploying.
- **Refreshing the browser starts a new conversation.** `session_state` resets, a new uuid is
  minted, the old thread is orphaned. That is a stated consequence, not a defect, and the
  sidebar says so. *Start over* does the same thing on purpose, and deliberately does **not**
  clear the checkpointer — that would discard every other session's memory too.
- **Conversations are not durable data.** The checkpoint file is ephemeral on Streamlit
  Community Cloud, and `FINBRIEF_CHECKPOINT_DB` exists so a deployment can put it somewhere
  writable. Losing it costs conversations and nothing else; the knowledge base is a separate
  artifact.
- **The agent's query is logged against yours — measured, not enforced.** `search_filings`'s
  description tells the agent to pass your question verbatim — the tool owns query optimization,
  so translating before it would translate twice (ADR-0003) — with one exception: a pronoun or
  elliptical reference is resolved first, because the retrieval engine is stateless and *"its
  debt"* names no company. That instruction is a **prompt, and nothing in the code enforces
  it**: the tool pre-processes nothing on our side, and what the model actually passes is then
  recorded rather than corrected. Whether each search ran your words is logged as a verdict
  (never the text of either query), and T10 reports the rate. On the first live two-turn run the
  model rephrased **both** queries, so the rate so far is 0 of 2. That is a finding for the
  evaluation phase, not something to tune the prompt against — and not a criterion this project
  claims to have met.
- **One `[n]` means one chunk for the whole conversation.** A second search continues the
  numbering rather than restarting at `[1]`, so a marker in an answer three turns up still
  resolves to the source you were shown beside it. The numbers are assigned in a single pass
  over the conversation (`agent/citations.py`) rather than by each search for itself, because
  searches in one step run concurrently against identical state and would otherwise all number
  from `[1]`.

## Configuration

Every knob lives in [`src/finbrief/config.py`](./src/finbrief/config.py) and resolves from
the environment, so the app and the evaluation harness read the same switches (ADR-0003).
`.env.example` lists them with their defaults.

- `OPENROUTER_API_KEY` is **required**. Without it — or with a malformed `LOG_LEVEL` — the
  app shows a configuration banner and stops before offering a chat input, rather than
  failing on your first message. `.env` is read once at startup, so edit it and restart.
- `SEC_EDGAR_USER_AGENT` is required **for ingestion**: the SEC rejects unidentified
  traffic, so `scripts/ingest_filings.py` fails loudly without a contact identity
  (`FinBrief your-email@example.com`). `FINBRIEF_CHROMA_DIR` (default `data/chroma`) is
  where the persisted collections live — ingest and app must agree on it, and
  `FINBRIEF_CHECKPOINT_DB` (default `data/checkpoints.sqlite`) is where conversations go.
- Retrieval switches, and all three are honoured. `FINBRIEF_RETRIEVAL_STRATEGY`
  (`vector` / `hybrid`) and `FINBRIEF_QUERY_TRANSLATION` move independently, because they are
  the two axes of the A/B (ADR-0002); their defaults are the pre-registered shipping
  configuration, hybrid + translation, fixed before any A/B data existed (ADR-0005). The
  sidebar states what is running and the *How I answered* panel shows what it did.
  `FINBRIEF_RETRIEVAL_K` is top-k **after** fusion, and is also how deep each candidate list
  is fetched. `FINBRIEF_MAX_SUB_QUERIES` is capped at 3, a ceiling rather than a default
  (ADR-0004's latency budget); at `0` the query planner is skipped entirely and translation
  reduces to the deterministic ticker-form variant, which is a useful configuration in its own
  right — it is the cell that isolates what the planner contributes. `RRF_K` is deliberately
  **not** an environment variable: a fusion constant somebody could sweep per environment is a
  back door into the pre-registration ADR-0005 exists to protect.
- `finbrief.*` logs one JSON object per line to stderr at `LOG_LEVEL` (default `INFO`);
  the Phase-7 A/B and security-gate analyses read those lines back.

## Development

```bash
uv sync
cp .env.example .env   # fill in OPENROUTER_API_KEY
uv run streamlit run app/Home.py
```

Answers come from the persisted `filings` collection, so **build the knowledge base first**
(*Building the knowledge base* below) or point `FINBRIEF_CHROMA_DIR` at one that exists.
Against an empty collection nothing raises — a populated Chroma always returns top-k, so
every answer is the retrieval-level fallback instead; the app says so in a banner naming the
directory it looked in, rather than leaving a reviewer to read the fallback as "out of scope".

Lint and test the way CI does:

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

Tests are hermetic — no API key, no `.env`, and no network calls — so they run anywhere.

### Building the knowledge base

Run these **from the repo root** — the report paths below are relative to the working
directory.

```bash
uv run python scripts/ingest_filings.py                  # the whole Universe → data/chroma
uv run python scripts/ingest_filings.py --tickers AAPL   # one or more companies
uv run python scripts/ingest_filings.py --dry-run        # fetch + gate only; no writes,
                                                         # no embedding API, no key needed
```

Requires `SEC_EDGAR_USER_AGENT`; writing (non-dry) runs also need `OPENROUTER_API_KEY`
for embeddings. A ticker outside `config.UNIVERSE` is rejected before anything is fetched
(exit 2); a gate failure writes no chunks and exits 1. Re-runs skip a filing only when the
collection already holds *that extraction* of it — the accession and the content hash both
match (ADR-0007 §7) — and evict a company's superseded fiscal years. So an extractor or
chunk-size change re-ingests on its own; `--force` is for a change to the embedding model,
which leaves no trace in the text to notice.

Once the collection is built, a wiring check over it:

```bash
uv run python scripts/retrieval_smoke.py   # 5 sanity queries → docs/verification/retrieval-smoke.md
uv run python scripts/retrieval_smoke.py --no-write   # print the report, leave the
                                                      # committed artifact alone
```

An empty collection exits 2 and names the ingest command rather than reporting five
"retrieved nothing" verdicts as a retrieval bug; a scored query whose top hit is the wrong
filing or Section exits 1 and still writes the report, because that is the run whose report
someone needs to read. The out-of-KB control query is recorded, never scored.

Needs `OPENROUTER_API_KEY` — it embeds each query with the same paid model the ingest used,
because a query embedded by a different model retrieves noise with no error. It is a **smoke
check, not an evaluation**: five hand-written queries, no buckets, no ground truth, no
baseline, so no number it prints may be cited as a retrieval-quality claim. ADR-0002's
stratified golden set with per-bucket RAGAs (ticket T9) is the measurement artifact of record.

It runs plain `vector` with translation **off** — deliberately *not* the shipping default, and
the report names both switches so nobody reads its distances as `hybrid + translation`'s. The
check asks one question ("is retrieval reading the collection ingest wrote, embedded by the
model that wrote it?"), and every part of the shipped configuration would absorb the failure it
exists to catch: BM25 matches lexically and would find the right filing even after an embedding
model drifted, which is precisely the drift being watched for, and translation would put a paid
*chat* call in front of it and make the queries non-verbatim. Holding it at T3's configuration
also keeps every run comparable with the reports already committed. Strategy comparison is the
golden set and the T10 A/B ([#11](https://github.com/TuringCollegeSubmissions/mrinal-AE.AFA.3.5/issues/11)), never this script.

Three committed evidence files, all generated:

- `docs/verification/ingest-report.md` — the gate table plus what the collection holds.
  Rewritten by a **full-Universe** run, pass or fail. A `--tickers` or `--dry-run` run
  prints its table to the terminal and leaves the file alone, because neither can speak
  to "all fifteen ingest" and overwriting it would destroy that evidence.
- `docs/verification/section-starts.md` — ADR-0007's hand-verification checklist, written
  by `--section-starts PATH`. A re-render carries a tick forward only for a Section whose
  text is byte-identical to the one that was verified, flags the rest `CHANGED`, and
  preserves hand-written notes unconditionally, wherever in the row they were written. A
  `--tickers` run is refused rather than written, since it can only re-render the companies
  it fetched and the rest of the file would be deleted outright.
- `docs/verification/retrieval-smoke.md` — the five queries, what each retrieved, and every
  chunk excerpt behind a verdict, so a reader can check the check. Rewritten by every
  `scripts/retrieval_smoke.py` run. It leads with what it is not, because a file of
  distances in a repo is read as an evaluation unless it says otherwise.
