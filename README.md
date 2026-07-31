# FinBrief — a domain-specialised RAG assistant for equity research

**Ask a question about a company's 10-K and get an answer you can check.** Every factual claim
carries an inline `[n]` that resolves to a retrieved chunk with its ticker, Section, fiscal year
and accession number, linked to that filing's own index page on EDGAR. Three finance tools put
live price, peer ratios and headlines beside the filing text. A four-layer security gate refuses
personalised investment advice and resists prompt injection. The retrieval pipeline is
independently evaluated — RAGAs, a per-bucket A/B over six configurations, and pre-registered
hypotheses settled by an exact paired test.

> **The honest headline, before anything else.** The shipping retrieval default is **retained,
> not validated**: only **4 of 18** pre-registered comparisons carried a measurement at all, so
> the experiment could not resolve the question it was built to answer. That is written up
> here, in [Part 5](#part-5--what-this-project-found-out), rather than buried — the
> pre-registration and the artifact it is judged against are deliberately separate things, and
> this is what that separation is for.

*Hero GIF here — steps 1 → 4 of the walkthrough below.*

| | |
|---|---|
| **Stack** | Python · Streamlit · LangChain / LangGraph `create_agent` · OpenRouter · ChromaDB + BM25 · Guardrails AI |
| **Data** | SEC EDGAR via `edgartools` · Yahoo Finance (unofficially, via `yfinance`) · news RSS |
| **Evidence** | five generated files under [`docs/verification/`](./docs/verification/) — never hand-authored |
| **Design record** | eleven ADRs under [`docs/adr/`](./docs/adr/) · plan in [`PLAN.md`](./PLAN.md) · spec in [`docs/spec/finbrief.md`](./docs/spec/finbrief.md) · glossary in [`CONTEXT.md`](./CONTEXT.md) |

**Read in this order.** This README, then the artifact behind any number that matters
([`evaluation.md`](./docs/verification/evaluation.md) is the measurement artifact of record),
then the ADR behind any decision that looks arbitrary. Every quality number here is requoted
from a generated artifact and bound to it by `tests/test_grounding_scope.py`, so a run that
moves a figure fails the suite instead of leaving this file asserting the old one.

---

## Contents

**[Part 1 — Orientation](#part-1--orientation)** · [what it is](#11-what-finbrief-is-and-who-it-is-for) · [quickstart](#12-quickstart) · [the five-step demo walkthrough](#13-the-five-step-demo-walkthrough)

**[Part 2 — Core requirements](#part-2--core-requirements)** · [RAG implementation](#21-rag-implementation) · [grounding-scope disclosure](#22-the-grounding-scope-disclosure) · [tool calling](#23-tool-calling) · [domain specialisation](#24-domain-specialisation) · [architecture, and the validity gap](#25-technical-implementation) · [user interface](#26-user-interface)

**[Part 3 — Optional tasks implemented](#part-3--optional-tasks-implemented)** · Easy [3.1](#31-conversation-history--export) [3.2](#32-rag-process-visualisation) [3.3](#33-source-citations) [3.4](#34-interactive-help--guide) · Medium [3.7](#37-prompt-injection-protection) [3.9](#39-token-usage--cost-display) [3.10](#310-tool-call-result-visualisation) [3.11](#311-conversation-export-in-various-formats) [3.13](#313-rate-limiting--api-key-management) [3.14](#314-logging--monitoring) · Hard [3.15](#315-hybrid-search) [3.16](#316-ab-testing-of-rag-strategies) [3.21](#321-ragas-evaluation)

**[Part 4 — Optional-task status](#part-4--optional-task-status)** — all 21, with reasons for the eight not built

**[Part 5 — What this project found out](#part-5--what-this-project-found-out)** · [the evaluation](#51-what-the-evaluation-established) · [which half is deterministic](#52-which-half-of-the-pipeline-is-deterministic) · [checks that could not fail](#53-the-recurring-theme-a-claim-the-thing-making-it-could-not-check) · [prompts are instruments](#54-prompt-rules-are-instruments-not-enforcement)

**[Part 6 — Limitations, ADRs, cost](#part-6--limitations-adrs-cost)** · [limitations](#61-limitations-each-with-its-mechanism) · [ADR index](#62-adr-index) · [what it cost](#63-what-it-cost) · [running everything](#64-running-everything)

---

# Part 1 — Orientation

## 1.1 What FinBrief is, and who it is for

A junior equity analyst preparing for a company's earnings call needs a grounded, source-cited
snapshot — business overview, risk factors, current valuation, recent news — in minutes rather
than hours. General chatbots hallucinate financials, cite nothing, and happily dispense
investment advice. Raw filings are hundreds of pages. Market-data terminals are expensive and
do not explain themselves.

FinBrief answers over a fixed **Universe** of 15 large-cap companies, curated as same-sector
peer clusters, and makes three promises it can be held to:

- **Traceable.** Every claim carries an `[n]`; every `[n]` resolves to one retrieved chunk shown
  verbatim beside the answer, with a link to the filing on EDGAR.
- **Current.** Price, ratios and headlines come from tools, because a 10-K has no prices in it.
- **Honest.** It states what it is *not* grounded in, refuses personalised advice with a
  disclaimer, and shows a stale figure's age rather than a fresh-looking guess.

The Universe serves triple duty (`PLAN.md` §1): the scope of the knowledge base, the cast of the
demo, and the peer pool for ratio comparison — which is what lets peer averaging add **zero new
API surface** (ADR-0009).

## 1.2 Quickstart

```bash
uv sync
cp .env.example .env                                  # fill in OPENROUTER_API_KEY
uv run python scripts/ingest_filings.py               # build the knowledge base (needs SEC_EDGAR_USER_AGENT)
uv run streamlit run app/Home.py
```

| step | what it needs | notes |
|---|---|---|
| `uv sync` | nothing | Python, `uv`-managed |
| `cp .env.example .env` | `OPENROUTER_API_KEY` | required; without it the app shows a configuration banner and stops **before** offering a chat input, rather than failing on your first message. `.env` is read once at startup — edit it and restart |
| ingest | `SEC_EDGAR_USER_AGENT` + `OPENROUTER_API_KEY` | the SEC rejects unidentified traffic; the key pays for the embeddings. `--dry-run` fetches and gates with **no key and no writes** |
| run | `OPENROUTER_API_KEY` | answers come from the persisted collection, so **ingest first** or point `FINBRIEF_CHROMA_DIR` at one that exists |

Against an empty collection nothing raises — a populated Chroma always returns top-k, so every
answer would be the retrieval-level fallback. The app says so in a banner naming the directory
it looked in, rather than leaving a reviewer to read a fallback as "out of scope".

`FINBRIEF_LOG_FILE` ships in `.env.example` **commented out**, so a copied `.env` leaves the
event sink off, exactly as the code default does. Uncomment it for any run whose numbers you
intend to report — see [3.14](#314-logging--monitoring).

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
credited to a model. The full before/after is in [3.15](#315-hybrid-search).

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

# Part 2 — Core requirements

## 2.1 RAG implementation

### The knowledge base

Curated Sections, not full filings (ADR-0007). Full-filing ingestion was rejected because
table-of-contents and boilerplate pollute the exact-identifier bucket and Section labels become
unreliable.

| property | value |
|---|---|
| companies | 15, in four curated peer clusters |
| Sections per company | Items 1 (Business), 1A (Risk Factors), 7 (MD&A), 7A (Market Risk) |
| company × Section pairs in the knowledge base | **54 of 60** |
| answered by incorporation into Item 7 | 6 filers — BAC, GS, JNJ, JPM, LLY, PFE |
| chunks in the collection | 5,842 |
| filings per company | one — the latest 10-K only |

All 15 companies have market-risk grounding; 9 have an `Item 7A` Section. The six that do not
answer Item 7A with a sentence directing the reader to Item 7, which is a lawful filing rather
than a defect: their market-risk disclosure *is* in the knowledge base, labelled `Item 7`. A
reader who does not know this reads "no Item 7A" as "no market-risk grounding", which is why the
app says it on screen.

The Universe, with what the ingest run wrote for each filer:

| Ticker | Company | Cluster | FY | Accession | Chunks |
|---|---|---|---|---|---:|
| AAPL | Apple Inc. | big_tech | FY2025 | `0000320193-25-000079` | 152 |
| MSFT | Microsoft Corporation | big_tech | FY2025 | `0000950170-25-100235` | 237 |
| NVDA | NVIDIA Corporation | big_tech | FY2026 | `0001045810-26-000021` | 298 |
| AMZN | Amazon.com, Inc. | big_tech | FY2025 | `0001018724-26-000004` | 192 |
| GOOGL | Alphabet Inc. | big_tech | FY2025 | `0001652044-26-000018` | 239 |
| META | Meta Platforms, Inc. | big_tech | FY2025 | `0001628280-26-003942` | 422 |
| TSLA | Tesla, Inc. | autos | FY2025 | `0001628280-26-003952` | 280 |
| F | Ford Motor Company | autos | FY2025 | `0000037996-26-000015` | 499 |
| GM | General Motors Company | autos | FY2025 | `0001467858-26-000013` | 311 |
| JPM | JPMorgan Chase & Co. | banks | FY2025 | `0001628280-26-008131` | 735 |
| BAC | Bank of America Corporation | banks | FY2025 | `0000070858-26-000157` | 643 |
| GS | The Goldman Sachs Group, Inc. | banks | FY2025 | `0000886982-26-000091` | 875 |
| JNJ | Johnson & Johnson | healthcare | FY2025 | `0000200406-26-000016` | 215 |
| LLY | Eli Lilly and Company | healthcare | FY2025 | `0000059478-26-000013` | 315 |
| PFE | Pfizer Inc. | healthcare | FY2025 | `0000078003-26-000026` | 429 |

The fiscal year differs by filer — NVDA is FY2026, the other fourteen FY2025 — and each citation
states its own. **Where these counts come from:** the ingest run's own evidence,
[`docs/verification/ingest-report.md`](./docs/verification/ingest-report.md), whose chunk counts
are read back from the persisted collection after the run rather than taken from the run's own
writes. The numbers above describe the knowledge base as ADR-0007 defines it and are **not a
live count of the index** behind any particular deployment: a partial or stale ingest would
leave them overstating coverage, and the evidence file is what a reviewer checks them against.

### The section-detection gate

The Phase-1 data-quality gate exists so that bad knowledge-base data cannot silently corrupt
retrieval numbers. It runs before anything is embedded, and a failure writes no chunks.

| | result |
|---|---|
| company × Section checks | **60 of 60 passed** |
| Sections incorporated by reference into Item 7 | 6 — lawful, not ingested, listed in the artifact |
| outcome | **GATE PASSED** (run of 2026-07-27 08:51 UTC) |
| hand-verification of extracted Section starts | 60/60 ticked, nothing flagged `CHANGED` |

Eight checks per Section: found · non-empty · length-bounded · **body ≫ heading measured in
words** (a table-of-contents hit is long and nearly wordless, so a character ratio waves it
through) · starts at its own heading · stops before the next Item, measured in *content* (the
auditor's report is Item 8 and is never MD&A) · EDGAR's latest annual filing for this company is
a 10-K · and, where the Item 7A excusal fires, that the filer is one recorded in
`config.ITEM_7A_POINTER_FILERS`.

**The hand-verification checklist earned its place on the sixtieth row.** JPM's Item 7 was found,
non-empty, length-bounded, body-heavy and free of Item 8 content — every rule the gate had — and
still began in the wrong place, at p.43 rather than the p.46 its own cross-reference names.
That bought a new rule (`section_starts_at_its_heading`) and a boundary correction from 400,310
to 394,858 characters. The gate proves mechanical properties; a person proves the text is the
right text. Ten further Sections opened on a stray `Table of Contents` line — the same defect,
three orders of magnitude smaller. The checklist is
[`docs/verification/section-starts.md`](./docs/verification/section-starts.md), and a re-render
carries a tick forward only for a Section whose text is byte-identical to the one that was
verified.

### Chunking

| parameter | value | where it lives |
|---|---|---|
| chunk size | 1000 characters | `config.CHUNK_SIZE_CHARS` — the single source of truth |
| chunk overlap | 200 characters | `ingestion/chunking.py` — an assertion next to the splitter, not a knob |
| splitter | `RecursiveCharacterTextSplitter`, section-aware | no chunk spans two Sections |
| metadata | ticker · filing type · section · fiscal year · accession | the accession is both the idempotency key and the filter a superseded year is evicted by |
| indexed text | opens with `AAPL \| FY2025 10-K \| Item 1A. Risk Factors` | so BM25 matches ticker and Section literals on **every** chunk of a Section, not only the one the splitter left the heading in (ADR-0004) |
| idempotency | accession **and** a `content_hash` over the Section texts, the provenance header and the splitter's parameters | so an extractor or chunk-size change re-ingests on its own; only an embedding-model change needs `--force`, because it leaves no trace in the text |

Raising the chunk size invalidates the index and is re-checked against the embedding model's
8191-token window by `tests/test_chunk_token_limit.py`.

### Embeddings and similarity search

| parameter | value |
|---|---|
| embedding model | `openai/text-embedding-3-small`, via OpenRouter |
| constructor | `retrieval/embeddings.py`, shared by ingest **and** query |
| store | ChromaDB, one `filings` collection, opened in one file (`retrieval/vectorstore.py`) |
| score | Chroma **L2 distance — lower is nearer**, not a normalised similarity |
| top-k after fusion | 5 |
| candidate-list depth | 5 — the same `k`, which is what keeps `vector` without translation byte-identical to the baseline the A/B compares against |

One constructor for both sides is not tidiness: a query embedded by a different model than the
one that wrote the index retrieves noise **with no error anywhere**. The score is called a
distance rather than a score because calling it a score invites every reader to assume 0…1 and
higher-is-better; `similarity_search_with_relevance_scores` was rejected because its rescaling
assumes normalised embeddings and would manufacture a similarity nobody has verified.

The retrieval *strategy* — hybrid search, query translation, and how they compose — is
[3.15](#315-hybrid-search); the A/B that compares configurations is
[3.16](#316-ab-testing-of-rag-strategies); the quality numbers are
[3.21](#321-ragas-evaluation).

## 2.2 The grounding-scope disclosure

A knowledge base scoped to four Items is only honest if the app says so, which is user story 18
and an ADR-0007 obligation. The app states it in two places, and the model is given the same
words:

> Filing answers are grounded only in Items 1, 1A, 7 and 7A of the latest annual 10-K for each of
> the 15 companies in FinBrief's Universe.

That sentence is `prompts.GROUNDING_SCOPE`. It is rendered as the caption under the app's title
(with EDGAR named as the source), and it opens `SYSTEM_PROMPT`, `AGENT_SYSTEM_PROMPT`,
`search_filings`' own tool description and the query planner's prompt — so the page, the persona
and the tool cannot disagree about what is grounded. The sidebar's **Grounding scope** panel
(`prompts.GROUNDING_SCOPE_DETAILS`, five lines, one sentence each) carries what the headline
sentence leaves out: the pair count, which six filers answer Item 7A by reference, what is out
of scope, that there is one filing per company, and that live figures never come from the
filings.

**Every number in it is derived, never typed.** `54 = 15 × 4 − 6` is computed from
`config.UNIVERSE`, the `Section` enum and `config.ITEM_7A_POINTER_FILERS`; the Item labels are
read out of the enum, so a fifth Section cannot leave a sentence on screen listing four. Three
things follow, and each is a test:

- **The derivation is checked against reality, not just against itself.** Self-consistent
  arithmetic is not truth — the subtraction assumes each pointer filer really does answer Item
  7A by reference. `tests/test_grounding_scope.py` cross-checks the six filers and both counts
  against the ingest run's own gate table.
- **The app renders it from `config` alone**, with no generated artifact on disk: a missing or
  half-written report must never take the UI down.
- **This file is bound to the same source.** The README cannot import `prompts.py`, so the test
  suite is where the two are allowed to disagree — loudly, in CI. Change the Universe and this
  section fails until the prose follows.

A tool description is a prompt, so it lives in `prompts.py` too and its scope claim is derived
like every other. That surface was found unbound during review: it had "Items 1, 1A, 7, 7A" and
"the fifteen Universe companies" typed by hand, in a prompt the *model* reads and plans its
searches against.

## 2.3 Tool calling

Four tools are bound to the agent. One reads the knowledge base; three read live public data.

| tool | signature | data source | returns | refusal / clamp |
|---|---|---|---|---|
| `search_filings` | `(query: str)` | Chroma `filings`, through `retrieve()` | a framed sources block for the model + a JSON-safe `Context` artifact for the UI | nothing found → a fallback sentence, never an invented answer |
| `get_stock_data` | `(ticker: str)` | Yahoo Finance, via `finance/quotes.py` — the only caller | `QuoteCard`: price, change, market cap, one month of history | a ticker outside the Universe is a **refusal-as-result**, not an exception |
| `calculate_ratios` | `(ticker: str)` | the same TTL-cached quote path — **zero new API surface** | `RatiosCard`: P/E, debt-to-equity, margins against the peer-cluster mean, with the range and per-metric coverage | an absent figure is excluded from the mean; the card says *1 of 2 peers reported this* |
| `get_recent_news` | `(ticker: str, days: int = 7)` | news RSS, via `finance/news.py` — the only caller | `NewsCard`: up to 8 HTML-stripped, quarantined headlines | `days` is **clamped** to 30 rather than refused — a model asking for 90 days wants "everything recent" |

**How peer sets resolve.** From `config.PEERS`, a static map, selecting the company's own curated
peer cluster inside the Universe — never an outside ticker, and never a set the model chooses
(ADR-0009). `n` comes from the cluster: 2 for the three-member clusters, 5 for `big_tech`. Every
comparison names its basis inline — *vs. mean of 2 `autos` peers: TSLA, GM* — and reports the
**range** beside the mean, because with two or three peers one outlier dominates an arithmetic
mean: Ford's peers are TSLA at 286× and GM at 37×, and the mean of 162× describes neither.

Not *same GICS sector*, which an earlier wording of the rule said and the clusters never
implemented: `big_tech` spans three GICS sectors, and AMZN sits apart from TSLA/F/GM despite
sharing one with them. The clusters are curated for **ratio comparability**; GICS is an input to
curating them, not the rule.

**What "live" does and does not mean**, because it is a word a reader will over-read. The quotes
are **delayed, and cached for 15 minutes** — the free feed is itself delayed by roughly that
much, so a shorter TTL would spend a call to re-fetch a number that cannot have changed. It is a
recent quote, not a tick, and the app says so. News summaries are **third-party text**,
HTML-stripped at the boundary and quarantined as data before the model sees them, with only
`http(s)` links rendered: a syndicated feed is writable by strangers, so comment, `style` and
attribute bodies are dropped *with their contents* rather than flattened into the text — which is
where an instruction hides from a reader while staying in the prompt.

**Three properties of the numbers, all of them boundary rules.** Units are normalised **once**,
where the data arrives: `debtToEquity` comes in percentage points (Ford reads `425.544`, i.e.
4.26×) while margins come as fractions, and a ratio that looks like a percentage is how a brief
reports Ford as levered 425 times. Every figure is `float | None`, and `None` means *not
reported*, never zero — JPM and BAC report no debt-to-equity at all, so a `banks` leverage
comparison **rests on GS alone** and says so. And `tools/finance.py` decides no number: it
validates the model's argument, turns every failure into a sentence the model can act on,
quarantines headlines as data, and attaches a card. Cards cross the checkpoint, so they hold
**JSON primitives only** — `asdict` keeps enum members and `json.dumps` will not tell you,
because a `StrEnum` *is* a `str`.

Tool-selection accuracy is measured, not assumed: **100%** over 10 scored cases, with the
valuation quote-plus-peers pairing at 1 of 2 — see
[3.21](#321-ragas-evaluation). Card and chart rendering is [3.10](#310-tool-call-result-visualisation).

## 2.4 Domain specialisation

**Why equity research.** It is a domain where the failure modes of a general chatbot are
specific and checkable: an unsourced financial claim, a hallucinated figure, and unlicensed
advice. Each has a concrete countermeasure — a citation that resolves, a figure that comes from
a tool or a filing, and a refusal — so "specialised" becomes something a reviewer can test
rather than a claim about tone.

**The specialisation is the knowledge base and the prompt, not a fine-tune.** Four Items of one
filing per company, chunked with Section metadata, is a corpus small enough to evaluate
honestly and scoped tightly enough that the app can say what it does not know. `prompts.py` owns
the analyst persona, the financial vocabulary, the grounding-scope disclosure, the framing that
makes retrieved text evidence rather than instructions, and the refusal policy — one module, so
the same words reach the model and the reader.

**The refusal policy, and why advice is not blocked at the front door.** *"Should I buy Tesla
stock?"* is the most natural question an analyst asks, and it is **not an injection**. It reaches
the model and is refused with a disclaimer by the output validator (user story 15). Blocking it
at the input gate would accuse an analyst of an attack; that is why the 28-question benign
control set opens with it. `prompts.DISCLAIMER` is appended by the caller rather than requested
from the model — a disclaimer the model is *asked* to add is the one that goes missing on the
turn that needed it most — and `GroundedAnswer.text` deliberately carries none, so every surface
that renders an answer owes one beside it.

Security measures are [3.7](#37-prompt-injection-protection): four layers, each doing what the
one before it cannot, measured against a committed corpus.

## 2.5 Technical implementation

### Architecture

```
                    ┌───────────────────────────────────────────┐
                    │              Streamlit UI                 │
                    │  chat · sources panel · How I answered    │
                    │  tool cards · spend meter · export        │
                    └───────────────┬───────────────────────────┘
                                    │  every typed question, and nothing else
                    ┌───────────────▼───────────────────────────┐
                    │   security gate — input (ADR-0006)        │
                    │  1 normalise → 2 denylist → 3 classifier  │
                    └───────────────┬───────────────────────────┘
                                    │
                    ┌───────────────▼───────────────────────────┐
                    │  Agent — LangGraph create_agent           │
                    │  SqliteSaver checkpointer = memory        │
                    │  citation register at before_model        │
                    └──┬──────────┬──────────┬──────────┬───────┘
                       │          │          │          │
              search_filings  get_stock  calc_ratios  get_news
                       │          └──────────┴──────────┘
                       │                     │
     ┌─────────────────▼──────────┐   ┌──────▼─────────────────────┐
     │  retrieve()  — ADR-0003    │   │  finance/  — ADR-0009      │
     │  translation: original +   │   │  TTL cache · retry ·       │
     │   ticker form + sub-queries│   │  stale banner · peer means │
     │  hybrid: BM25 + vector     │   └──────┬─────────────────────┘
     │  RRF → dedup → top-k       │          │
     └─────────────────┬──────────┘   ┌──────▼─────────────────────┐
                       │              │ Yahoo Finance · news RSS   │
     ┌─────────────────▼──────────┐   └────────────────────────────┘
     │  ChromaDB `filings`        │
     │  5,842 chunks, one opener  │        ┌─────────────────────────┐
     └─────────────────▲──────────┘        │ security gate — output  │
                       │ ingest            │ 4 advice validator      │
     ┌─────────────────┴──────────┐        └───────────▲─────────────┘
     │ ingestion/ — ADR-0007      │                    │ every answer
     │ EDGAR → sections → gate    │        ┌───────────┴─────────────┐
     │ → chunks + provenance hdr  │        │ observability/ log_event │
     └────────────────────────────┘        └─────────────────────────┘

     evaluation/ (ADR-0002) calls retrieve() and rag.answer_question DIRECTLY —
     no agent, no Streamlit. That is the seam the headline numbers measure.
```

**Two entry points into the same retrieval code, on purpose.** `rag.answer_question` is the
deterministic `question → contexts → answer` chain the evaluation harness drives, and
`search_filings` is the same `retrieve()` wrapped as a tool for the agent loop. One code path,
measured and shipped, which is what stops an eval rig from diverging from the product.

### The validity gap, stated rather than assumed

The headline RAGAs and A/B numbers measure **the chain**. What ships is **the agent**. That gap
is real, and ADR-0003 commits to stating it rather than assuming it away:

| | the measured chain | the shipped path |
|---|---|---|
| entry point | `rag.answer_question` / `retrieve()`, called question by question | `agent.answer`, a tool-calling loop |
| determinism | exact, except the sub-query planner's one chat completion — and that reply is replayed from a committed file, so the A/B arms are reproducible | nondeterministic: the model chooses whether, when and with what arguments to search |
| tools | **none** — the chain calls no finance tool at all | four |
| what measures it | per-bucket RAGAs and the A/B matrix ([3.16](#316-ab-testing-of-rag-strategies), [3.21](#321-ragas-evaluation)) | the tool-calling eval, reported **beside** those tables and never inside them |

Three mechanisms keep the gap small and, where they cannot, measured:

- **The tool passes the question through unchanged.** `search_filings` pre-processes nothing on
  our side, and a test pins that half — the half we control.
- **The agent is *asked* to pass the user's question verbatim**, with exactly one permitted edit
  (resolving a pronoun, because the engine is stateless and *"its debt"* names no company). That
  is a prompt, and nothing in the code enforces it. The enforcing alternative — overwriting the
  model's argument — was rejected, because the one edit that must be permitted is
  indistinguishable from the rewrite that must not be.
- **So the divergence is logged and reported**: **100% divergence, 8 of 8 searches**, all of them
  *first* searches in their thread, which cannot be reference resolutions — so none of them is
  the one rewrite the description permits. That is a finding rather than a criterion this project
  claims to have met, and it is not something to tune the prompt against
  ([5.4](#54-prompt-rules-are-instruments-not-enforcement)).

One consequence worth being explicit about: a multi-hop question is decomposed *inside*
`retrieve()` under evaluation, so if the agent instead answers one by making several tool calls,
that is the selection layer working — and it surfaces in the tool-calling eval rather than as
noise in the chain's numbers.

### Choices, and why

| concern | choice | why |
|---|---|---|
| agent | LangChain / LangGraph `create_agent` + a file-backed `SqliteSaver` checkpointer | the checkpointer is the agent's **memory of record** (ADR-0008) |
| model access | OpenRouter, OpenAI-compatible SDK, one chat constructor (`llm.py`) | one place a model is built, so ingest and query cannot drift |
| chat model | `openai/gpt-4o-mini` | `config.py` default |
| gate classifier | `openai/gpt-4o-mini`, its own field | the gate pays for one YES/NO per turn and a brief is worth more, so raising one must not raise the other |
| judge model | `openai/gpt-4.1-mini`, its own field | defaults **stronger** than the answering model, and deliberately not the answering model — a judge that grades its own output is the circularity the source-separated golden set exists to avoid |
| vector store | ChromaDB — course-covered, lightweight, right-sized for 15 companies | over Milvus |
| lexical search | `rank_bm25`'s `BM25Okapi` + hand-written RRF | **not** `EnsembleRetriever`: its blend needs comparable scores, and an L2 distance and a corpus-relative BM25 score are not |
| output validator | Guardrails AI's `Guard`/`on_fail` contract | the no-advice rule itself is a registered **custom** validator; there is no hub validator for it |

Every bound in the system, in one table — because a limit stated in prose is a limit that
drifts:

| bound | value | note |
|---|---|---|
| agent steps | 24 | `agent/agent.py`'s `MAX_AGENT_STEPS`, a ceiling next to the loop it guards |
| question length | 4000 characters | refused before the gate, so an over-long paste costs nothing |
| questions per session | 40 | cost and abuse limiting — see [3.13](#313-rate-limiting--api-key-management) |
| ticker length | 12 characters | before any whitelist lookup |
| news window | 7 days by default, clamped to 30 | |
| headlines per card | 8 | |
| quote / news cache TTL | 900 seconds | the free feed is itself delayed by roughly that much |
| one HTTP request | 15 seconds, 3 attempts, 0.5 s backoff | inside the cache lock — hence the 91.5 s worst case |
| gate classifier | 5 seconds, 1 attempt (**no retry**) | a retry multiplies the worst case inside a latency budget, and the gate fails open onto three other layers |
| answering call | 60 seconds, 2 retries | |

### Error handling, in three tiers

| tier | failure | behaviour |
|---|---|---|
| **API** | rate limit, timeout, a flaky quote feed | retry with backoff inside a TTL cache; on failure the **last good figure with a banner giving its age**; if nothing is cached, the tool returns a sentence saying the figure could not be fetched, and the model is told not to supply one from memory |
| **retrieval** | the collection yields nothing | `prompts.NO_CONTEXT_FALLBACK` instead of a generated answer, plus a banner naming the Chroma directory that was searched |
| **generation** | an advice refusal, a gate block, an unresolvable `[n]` | rendered as a refusal beside the disclaimer, or as a caption naming the marker — never a crash, and never a silent repair |

Two deliberate asymmetries in that last row. Layer 3 of the gate **fails open**: a provider
outage or an unparseable reply allows the turn and logs a warning, because failing closed turns
a bad afternoon at OpenRouter into an assistant that refuses everything. And a dangling `[n]` is
**reported, not repaired** — stripping it removes the evidence a reader needs, and refusing the
whole answer would make a numbering slip indistinguishable from an advice refusal.

### Input validation

A ticker is checked against the Universe whitelist before anything is fetched (a script exits 2
on an out-of-Universe ticker; the tool returns a refusal-as-result); questions are length-capped;
the input gate normalises and screens every question a human types. The whitelist turned out to
be an **incidental input-side security control** as well: it stops an out-of-Universe indirect
payload from ever reaching the model — which is why it invalidated the first version of the
indirect-injection test (ADR-0006 §7). It is no defence at all against a payload planted under a
covered ticker, and it is not part of the four-layer argument.

### The hermetic-test contract

**Tests are hermetic — no `.env`, no API key, no network — and that is enforced rather than
asserted.** `tests/conftest.py` patches egress at the socket layer and at **all four** CPython
resolver entry points, plus `curl_cffi` (which resolves and connects in C, touching Python's
`socket` module not at all) and `uvloop` (which resolves in libuv). `tests/test_hermetic_suite.py`
carries **one test per backend**, because the guard is a denylist over the backends this repo can
reach and not a proof: a new HTTP dependency is a new path.

That framing is earned rather than cautious. The claim "the suite is hermetic" has been false
**seven** times here — twice through tiktoken's warm cache (whose BPE table is now vendored under
`tests/fixtures/`), twice through this guard's own prose describing coverage the code lacked,
twice through the single dependency T7 added, and once through T10's. Two further lessons are
baked in: `conftest.EGRESS_ATTEMPTS` records every refused target *inside* the refusal, because
OpenTelemetry's `BatchSpanProcessor` and ragas' `@silent`-decorated tracker both swallow the
exception and return a perfectly normal result; and every test in that file exercises the call it
is about rather than asserting that one path routes through another, which is precisely the
sentence that was wrong.

EDGAR, market and news fixtures are **recorded, never fetched**, and retrieval runs against a
real on-disk Chroma with a deterministic fake embedding — never the ingested `data/chroma`, which
is only searchable by the paid model that wrote it. CI runs exactly the lint and test commands
below; the suite is 1,454 tests.

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

## 2.6 User interface

Streamlit chat, one page. What is on it, and where each surface is documented:

| surface | what it shows | detail |
|---|---|---|
| caption under the title | the grounding-scope sentence, with EDGAR linked, and the live-data scope | [2.2](#22-the-grounding-scope-disclosure) |
| chat transcript | the answer, with `prompts.DISCLAIMER` beside it | |
| **sources panel** | one entry per retrieved chunk: ticker · Section · fiscal year · accession linked to EDGAR · the text, verbatim | [3.3](#33-source-citations) |
| ***How I answered*** | the queries that ran, and per chunk which variant × which retriever surfaced it | [3.2](#32-rag-process-visualisation) |
| tool cards and charts | quote, ratios, news | [3.10](#310-tool-call-result-visualisation) |
| progress indicators | an `st.status` block naming each step as the agent runs it | |
| marker note | any `[n]` that resolves to no panel entry, named rather than stripped | [3.3](#33-source-citations) |
| sidebar | Conversation · How to use FinBrief · Token spend · Grounding scope · Configuration · Universe — four of them collapsed panels | [3.1](#31-conversation-history--export), [3.4](#34-interactive-help--guide), [3.9](#39-token-usage--cost-display) |
| export buttons | the conversation as JSON or CSV | [3.11](#311-conversation-export-in-various-formats) |

**Retrieved text renders through `st.text`, not Markdown**, because a filer's own `$178,353` is a
KaTeX expression to a Markdown renderer, and a citation surface that silently reformats the
figures is not a citation surface.

Two Streamlit findings worth a reviewer's attention, both measured rather than read:
`st.empty()` does not reserve space, it **clears** — so a slot created early and filled at the
end of the script is *blank* for as long as the script sits inside a model call, which is how
the sidebar's Token spend panel came to vanish for the whole of every answer. The fix is a
second, **eager** fill; and that fill has to be widget-free, because two fills of one slot in a
single run are a duplicate element and identical `download_button` parameters raise. Separately,
`AppTest.get("...")` returns `[]` for an element type it does not know rather than raising, so
`assert not app.get("arrow_bar_chart")` passes on a page rendering ten charts — the charts here
are asserted by walking the element tree for `vega_lite_chart`.

---

# Part 3 — Optional tasks implemented

**A note on the numbering.** The assignment lists optional tasks unnumbered under Easy, Medium
and Hard. The numbers `3.1`…`3.21` below are a **local convention of this repo** — they index the
list in the order I recorded it, and are used only as stable subsection anchors. [Part
4](#part-4--optional-task-status) is the full 21-row table, named rather than numbered, so
nothing here depends on the numbering being anyone else's.

Thirteen are built: **Easy 4/4 · Medium 6/10 · Hard 3/7**, against a bar of 2 medium + 1 hard.

## Easy

### 3.1 Conversation history + export

**The checkpointer is the memory of record.** `answer()` is handed the new question and a thread
id, never a history — a history assembled in the UI would be a second copy of the conversation,
and the copy the model never sees is the one that goes stale. `st.session_state` holds only the
`thread_id`, UI state, and the transcript on screen (ADR-0008).

- **One `uuid4` per browser session**, minted before the first message and stable across reruns.
  The agent and the checkpointer are built once per *process* and shared by every session, so
  that id is what keeps two users apart. `tests/test_app_state.py` drives two `AppTest` sessions
  in one process and asserts they get different threads and that each turn lands on its own —
  the cross-user leak this design exists to prevent, provable without deploying.
- **Refreshing the browser starts a new conversation.** `session_state` resets, a new uuid is
  minted, the old thread is orphaned. A stated consequence, not a defect, and the sidebar says
  so — a user who is not told reads a lost conversation as a bug.
- ***Start over*** does the same thing on purpose, and deliberately does **not** clear the
  checkpointer: that would discard every other session's memory too, which is the cross-user
  failure this design prevents arriving through the button that looks safe.
- **Conversations are not durable data.** The checkpoint file is ephemeral on Streamlit
  Community Cloud, and `FINBRIEF_CHECKPOINT_DB` exists so a deployment can put it somewhere
  writable. Losing it costs conversations and nothing else.

The follow-up mechanic is the visible half: *"and its debt?"* resolves against the company just
discussed because the conversation lives in the checkpointer. Export is
[3.11](#311-conversation-export-in-various-formats).

### 3.2 RAG process visualisation

Every answer that searched the filings carries a **How I answered** panel: the queries that ran,
then per chunk which variant × which retriever surfaced it and what it contributed to the fused
score. So *why* a chunk was retrieved is checkable on the turn itself, not only in an aggregate
table.

| | value |
|---|---|
| query variants | 1 original + ≤1 ticker form + ≤3 sub-queries = 5 |
| candidate lists under hybrid | 10 |
| RRF constant | 60 — the published default, **stated and not tuned** |

The three kinds of variant are labelled as what they are: the **original question is always
variant 0**, the **ticker form** is labelled a ticker form rather than "sub-query 1" because it
is a `config` lookup and a lookup must not be credited to a model, and the planner's sub-queries
are the planner's. `RRF_K` is deliberately **not** an environment variable: a fusion constant
somebody could sweep per environment is a back door into the pre-registration ADR-0005 exists to
protect.

Two absences are rendered as absences, which is why `retrieve()` returns a `Retrieval` rather
than a bare sequence of chunks. **A variant that surfaced nothing** is shown — no chunk's
provenance can carry that fact. And a chunk BM25 recovered has **no vector distance**, so the
panel says so; a stand-in `0.0` would print a number no measurement produced. A live thread
checkpointed before provenance existed reads back as "provenance was not recorded for this
chunk", never as a fabricated one.

### 3.3 Source citations

Inline `[n]` markers, each resolving to one sources-panel entry carrying ticker, Section, fiscal
year and accession.

**One `[n]` means one chunk for the whole conversation.** `retrieve()` ranks 1…k on every call,
so a second search would reuse `[1]` for a different chunk and every marker above it would stop
resolving. The register is therefore assigned in **one sequential pass at the `before_model`
seam** (`agent/citations.py`), over the thread's own tool messages. It cannot be done in the
tool: LangGraph builds every `ToolRuntime` in a step from the same state and *then* runs the
calls concurrently, so two searches in one step read an identical offset and both number from
`[1]` — and nothing raises. The pass is idempotent and recomputes the whole sequence, so a thread
checkpointed by an earlier shape is numbered correctly the next time it is read.

**Each source links to its filing's index page on EDGAR**, so a citation can be checked against
the primary source rather than against the excerpt beside it. The link is derived, not stored,
and it is derived **from the ticker**: an accession's leading block is the *filer agent's* CIK,
not the company's. META's 10-K is accession `0001628280-…`, which is Donnelley's, and a URL built
from it resolves to a different company's filings with no error anywhere — a wrong answer
indistinguishable by eye from a right one. `tests/test_edgar_links.py` reproduces all **15**
recorded URLs from ticker and accession alone, as an equality against what a real fetch recorded.

**Marker resolution is enforced; marker support is measured, and mostly partial.** Every `[n]` is
checked against the numbers this conversation has issued, and an unresolvable one is named beside
the answer. But a marker that *resolves* can still sit on a claim its chunk does not support —
that is faithfulness, and T10 measured it per sentence against the **cited** chunk rather than
against the whole context set:

| | value |
|---|---|
| `(sentence, marker)` pairs scored | 70, across 50 cited sentences |
| split | **22 fully supported / 40 partly supported / 8 not supported** |
| derived full-support rate | 31% |
| markers pointing outside the retrieval | 0 |
| pairs the judge did not score | 0 |

**The middle bucket is the largest and the interesting one.** A partly supported cited sentence
has a marker that resolves and a chunk that carries *some* of the claim, which is precisely what
the citation register cannot see and what whole-answer faithfulness scores as fine, since the
claim is supported somewhere in the context set. The rate is derived from the composition and
never quoted alone. Source:
[`evaluation.md`](./docs/verification/evaluation.md).

The related **square-bracket adherence rate is unmeasured**, and the reason is an
instrument-placement bug rather than an unwilling model — see
[6.1](#61-limitations-each-with-its-mechanism).

### 3.4 Interactive help / guide

A **How to use FinBrief** panel in the sidebar, plus **four example-question buttons** on the
empty page — one per path a reader would not guess is there: Item 1A retrieval, Item 7 retrieval,
the peer-ratio tool, and the multi-tool brief. Every company they name is taken from
`config.UNIVERSE`, so a curation change cannot leave the first thing a new reader clicks pointing
at a company nothing was ingested for.

A seeded question is **seeded through the same path a typed question takes**, so it is screened
by the security gate like any other. That is the whole design constraint: a seeding route that
bypassed `screen()` would be a second door into the agent.

**The `/help`-style command is not built, and is recorded as open rather than cut.**
`st.chat_input` is the only text entry, so a slash command means parsing one and routing it past
the gate — a second door for a convenience.

## Medium

### 3.7 Prompt-injection protection

Four layers, and each one's job is what the one before it cannot do (ADR-0006). Counts are from
the live run in
[`docs/verification/security-gate.md`](./docs/verification/security-gate.md), rendered from
measurements rather than typed:

| # | layer | catches what the layer before it cannot | caught, this run | cost |
|---|---|---|---|---|
| 1 | **Normalisation** | *Obfuscation.* Case, accents, leetspeak, zero-width joiners, fullwidth Latin, Cyrillic/Greek homoglyphs and letter-by-letter spacing fold into one surface form, so layer 2 needs one rule per payload *family* instead of one per spelling | **9** cases layer 2 catches only *after* folding | pure function, no model call |
| 2 | **Bounded-gap denylist** | *Known payload families*, for free — 7 rules, each naming what it is for. A catch exits the gate; a pass **always** escalates | **13** | pure function, no model call |
| 3 | **Zero-shot classifier** | *Novel phrasings.* The only layer that can catch a wording invented after the rules were written — a hypothetical framing, a translation-shaped extraction, a payload spread wider than the bounded gap | **7** | one cheap model call per turn |
| 4 | **Output validator** (Guardrails AI) | *Consequences.* It judges what the model **produced**, so it catches a recommendation nobody asked for and the result of a *successful* indirect injection — neither of which any input layer ever sees | **10 answers refused** | pure function, no model call |

The last row is the one to read twice: it is the only layer whose input the attacker does not
choose. And every count is a measurement rather than a row count — layer 1's is
`input_gate.folding_required` per case, so the cell goes to **zero** if normalisation stops
contributing, which a count of corpus rows could not do. Layer 3's counts only the cases layer 2
verifiably passed, and the suite asserts that gap in both directions.

Normalisation produces **two** forms, not one: `text` keeps word separators so a rule can use
`\b` and *not* match inside an unrelated word (`contract assets` must not trip an `act as`
rule), and `squeezed` removes them, which is the only form in which `i g n o r e   a l l …` still
contains its words. One rule set scans both, which is why a rule head-anchors and joins words
with a bounded gap rather than a literal space.

**The suite's results:**

| | result |
|---|---|
| attacks stopped **by the expected layer** | 20/20 |
| benign analyst questions allowed | 28/28 |
| answer verdicts correct (advice refused, research allowed) | 22/22 |
| planted payloads **retrieved and** resisted | 5/5 |
| outcome | **SUITE PASSED** (run of 2026-07-28 20:19 UTC) |

Blocked by the *wrong* layer counts as a failure: the corpus exists to attribute a catch, and a
classifier case the denylist happened to match would credit layer 3 with a layer-2 win.

**A false positive is a failure too.** 28 real analyst questions form a control set that must get
through, and the first two are the point: *"Should I buy Tesla stock?"* is not an injection. The
set is grouped by which attack family each question sits *next to*, since a question no
classifier would ever flag measures nothing — four are "set aside part of the accounting"
phrasings, four ask the assistant about itself, and six share the denylist's own *vocabulary*
without its intent (*"Does management discuss plans to lift restrictions on the dividend?"*).
Every one of those six was blocked by the shipped rules when it was written, which is what
widening the set is for; widening it also found a **deterministic false positive in the shipped
classifier** that the original 16-question set could not see.

**Indirect injection is tested, not asserted.** A dedicated collection — a throwaway directory,
built and destroyed per run, never the demo knowledge base — is seeded with 5 poisoned chunks: an
instruction inside a filing body, a forged `</sources>` delimiter, a prompt-extraction attempt,
an advice solicitation, and an instruction hidden in HTML a reader never sees. Each demands a
specific canary string, so obedience is *detected* rather than judged. **`retrieved` is a column
because the first run needed it**: one row reported "not obeyed, no leak" about a turn in which
the agent asked which company was meant instead of searching, so the payload never reached the
model — a green cell about nothing. A row that does not reach its payload now fails.

Retrieved text is quarantined as data on both paths, and a body containing its own `</sources>`
or `</news>` is made inert before the model sees it. `prompts.QUARANTINE_TAGS` is the single
source of truth, and both the escaper **and** the denylist's `delimiter-forgery` rule are
parametrised over it — that derivation was prose before it was code, which left `</input>`
escaped and never denylisted.

**Latency:**

| | value |
|---|---|
| p50 over every screening | 564 ms |
| p50 over the **escalated** screenings — the ones that paid for the model call | 670 ms |
| against the revised budget of ≤ 1000 ms | **within** |
| against the **pre-registered** ≤ 800 ms | within *on this run* — and **not reliably met**: 738–1041 ms across eight passes, over 800 in six of them |
| the gate's share of a turn | 9–21% of a turn the analyst already waits 8–13 seconds for |

The escalated number is the one to read: a median over a corpus that is mostly blocked payloads
flatters the gate, because a denylist catch exits in microseconds, and the ordinary analyst
question is exactly the one that escalates. The budget was **revised rather than deleted**, and
both figures stay in `config.py` and in the artifact, so a reader cannot mistake a revised
pre-registration for one that always held. A faster model
(`google/gemini-2.5-flash-lite`, 331–494 ms) matched the attack catch rate exactly and was
**rejected** because it blocked a legitimate analyst question: buying latency with a false
positive is the wrong trade on a security control, and tuning the classifier prompt until it
stopped firing is what ADR-0003's standing rule forbids.

**Two bypasses that a green suite could not have found**, both from adversarial review of a run
reporting every cell true. Layer 4 was defeated by *"disclaim, then advise"* — `advice_hits`
iterated `pattern.search`, so it saw each rule's **first** match and no other, and a negatable
rule whose first occurrence sat inside a denial was dropped for the entire answer. And layer 3
read a bolded `**YES**` as *undecided*, which the gate allows — so a classifier that merely
formats its one-word answer would have disabled layer 3 for every turn, permanently, with one
warning line in a log. Both are corpus cases now, because a fix with no case behind it puts the
claim back where the review found it. **A passing security suite is evidence about the cases in
its corpus, not evidence about the layer** — and it cannot be, because the corpus is a sample
drawn by the same author, from the same intuitions, that wrote the rules it tests.

What the gate does **not** cover is in [6.1](#61-limitations-each-with-its-mechanism).

### 3.9 Token usage + cost display

A sidebar panel reporting the conversation's token spend. **It reads the log rather than adding
an instrument**: T8 already records per-field token counts on the agent loop's turn and on the
planner's own call, so the meter is arithmetic over one reader and nothing new is emitted.

- **What is counted:** the `agent_turn` and `query_translation` lines' `input_tokens` and
  `output_tokens`, each with **its own denominator**. Absence is per *field*, not per record —
  a provider returning half a pair must not put a fabricated zero on the line.
- **It exists only when the log does.** With `FINBRIEF_LOG_FILE` unset there is nothing to read,
  and the panel says so instead of rendering `0`: a spend of zero is a claim that the calls were
  free.
- **It is scoped to this conversation, not to the file.** Every turn id is prefixed with the
  conversation's `thread_id`, so a prefix match selects this conversation's lines and nothing
  else — strictly sharper than a byte offset, which would also admit a second tab writing
  concurrently. The sink is append-only across every run that ever named it, and a total over
  the whole of it would be a total over all of them.
- **A partial total says so**, as a floor with the counts beside it.
- **The unmetered call is named on every total, complete or not.** The gate's zero-shot
  classifier is deliberately never metered (ADR-0011), so one paid call per turn is
  *structurally* absent from every figure. That sentence cannot be a clause of the partial
  banner even in principle: `partial` is defined over reported-versus-counted calls and the
  classifier never enters that count, so no value of `partial` is evidence about it.

**Why prices are unset by default.** `FINBRIEF_INPUT_COST_PER_MTOK` and
`FINBRIEF_OUTPUT_COST_PER_MTOK` default to unset, and with no price the panel reports tokens and
says it cannot price them. There is deliberately **no rate card** in this repo: FinBrief reaches
every model through OpenRouter, which fronts many upstreams and routes by availability, so the
price of a call is not something this codebase can assert. A hardcoded figure would be a number
nobody measured, going stale silently, in the one panel whose entire subject is spend. Two knobs
rather than one because input and output are priced differently everywhere.

Measured spend from the last evaluation run, for scale
([`evaluation.md`](./docs/verification/evaluation.md)):

| metered event | input tokens | output tokens | lines | unmetered lines |
|---|---:|---:|---:|---:|
| `query_translation` | 12968 (48 calls) | 2556 (48 calls) | 188 | 140 |
| `agent_turn` | 287215 (10 calls) | 2767 (10 calls) | 10 | 0 |

An unmetered line is a call whose cost is **unknown**, not free.

**Not built: the per-*message* half.** `agent_turn` already carries a turn's own spend, so it is a
rendering job on the transcript row rather than a new measurement. Open, not cut.

### 3.10 Tool-call result visualisation

Each tool result is rendered beside the answer as a card, with every figure in the units a note
quotes.

| card | contents |
|---|---|
| quote | `st.metric` for price and change, market cap, and a one-month price line chart |
| ratios | P/E, debt-to-equity and margins, each against its peer-cluster mean, with the peer set, `n`, the range, and per-metric coverage |
| news | up to 8 headlines, HTML-stripped, with only `http(s)` links rendered |
| failure | what could not be fetched and why — never a placeholder figure |

**One chart per ratio metric, and the reason is a measurement.** A P/E and a debt-to-equity share
`Unit.MULTIPLE` and not a *scale*: on one axis, Ford's 4.26× leverage drew as three pixels beside
a 162× peer-mean P/E, and `st.bar_chart` has no log scale to rescue it. Bars that misstate a
ratio misstate every ratio a reader takes off them, so each metric gets its own axis — and the
*count* of charts is what the test asserts.

The price chart is `st.altair_chart` and not `st.line_chart`, for the y-axis only: Vega-Lite
includes zero in a quantitative axis' domain by default, which draws a month of a $400 stock as a
flat line at the top of the frame. Everything else is `st.line_chart`'s own generated spec.

A stale figure carries a banner giving its age (user story 22); a figure a source does not report
reads *not reported* in words, never `0.0`.

### 3.11 Conversation export in various formats

The sidebar offers the conversation as two downloads once there is one to take. Both are built
from **the display transcript, not the checkpointer** — the export is what the analyst *saw*, and
the two genuinely differ in both directions: a question the input gate blocked never reached the
agent, so the checkpointer has no memory of it while the page shows the exchange; and an answer
layer 4 refused is *in* the checkpointer while the page shows the refusal that replaced it.
Exporting the agent's memory would hand a reader an answer that was withheld from them and omit a
refusal they were given.

| format | shape | the property it buys |
|---|---|---|
| **JSON** | one `sources` table keyed by rank, plus each turn naming the ranks it retrieved | citations run in one sequence across a thread, so a follow-up can cite a chunk an earlier turn retrieved — scoped per turn, that citation would be unresolvable in a file whose own answers cite it |
| **CSV** | one row per (turn, source that turn retrieved) | every source is a row with its own `rank`, so a marker resolves by scanning one column rather than by parsing a list packed into a cell. A turn that retrieved nothing still gets a row, with the source columns empty |

- **Marker fidelity.** Every `[n]` resolves against the file's own source list, and that list is
  the conversation's, not the turn's.
- **Absences stay absent.** A refusal has no turn behind it, so whether it searched is *unknown*
  and no key is written — not `false`, which would be a measurement of a turn that did not
  happen. A chunk BM25 recovered exports `null`, never `0.0`. In CSV those are empty cells,
  because a spreadsheet averages a column without asking what its blanks meant.
- **Every answer carries the disclaimer** — once at the JSON's top level, on **every** CSV row,
  because a row is the unit a reader lifts into a note and a disclaimer left behind did not
  travel with the claim it qualifies.
- **Text cells are guarded against a spreadsheet — and only the cells that need it.** A cell
  opening `=`, `+`, `-` or `@` is evaluated as a formula by Excel and Sheets, so such a value is
  prefixed with `'`. `=` and `@` unconditionally; `-` and `+` only when what follows is not
  whitespace, which separates the real DDE payload `-cmd|' /C calc'!A0` from a markdown bullet.
  The first version guarded `-` unconditionally, and review measured what that cost: **0 of
  5,842** ingested filing bodies open with a formula leader, so the rule never fired on the text
  it was written for and always fired on answers opening with a bullet. Every cell that is not a
  formula leader now round-trips **byte-identical** through `csv.reader` — asserted with bodies
  carrying commas, quotes and blank lines.
- **The export is logged as a count and a format, never as a payload.** The file is the analyst's
  own questions and the filer's prose, which is exactly what the log may not carry.

**PDF is declined, and the reason is this project's dependency record rather than effort.** It
needs a new library, and every library added here for quality or safety shipped a telemetry path
enabled by default — all three of them, each switched off somewhere different (the table is in
[5.3](#53-the-recurring-theme-a-claim-the-thing-making-it-could-not-check)). A rendering library
is a worse bet than those three rather than a better one: it would be added for **presentation**,
which buys none of the argument that made the other three worth their switches and their
per-backend tests. JSON and CSV need no dependency at all — `json` and `csv` are stdlib — so the
export ships with exactly the egress surface the page already had. Declined, not open.

### 3.13 Rate limiting + API key management

**The per-session question cap is cost and abuse limiting, and it is not a security control.**
`config.MAX_QUESTIONS_PER_SESSION` bounds how many questions one browser session is answered — 40
— so one tab left open on a script cannot spend a shared demo key's budget. **Refreshing the page
resets it**, because the counter lives in `st.session_state`: anyone who wants past it walks past
it, and saying so is the point rather than a caveat. A reviewer who reads a session counter as
rate limiting stops looking for the thing that is, so the constant says so at length, the banner
the user sees gives cost as its reason, and a test asserts the banner does not describe itself as
security or as a rate limit. The [security gate](#37-prompt-injection-protection) is the
boundary.

The counter increments **before** the gate, so a blocked payload consumes a question: otherwise
the one caller worth throttling is the one that gets unlimited attempts. The length cap is free
and refuses first, so an over-long paste costs nothing from the session's budget.

**What real rate limiting would need**, so the gap is stated rather than implied: a bound keyed
server-side on something the client does not choose — an account, an IP, a token bucket in a
shared store — which needs the user authentication this project defers. That is a ticket, not a
constant.

**Key handling.** `OPENROUTER_API_KEY` comes from `.env` locally and `st.secrets` on Streamlit
Community Cloud, is read once at startup, and is **never logged** — `log_event` may carry no
secret in any field, and the app's configuration banner reports a *missing* key without echoing
anything. `SEC_EDGAR_USER_AGENT` is a contact identity rather than a credential and is required
for ingestion only, because the SEC rejects unidentified traffic.

### 3.14 Logging + monitoring

Structured JSON lines, one object per event. **One emitter (`log_event`) and one reader
(`observability/events.py`)**, because the two drifting apart is silent: a renamed field reads
back as `None`, `None` averages as nothing, and the number narrows its own denominator.
`tests/test_event_log.py` round-trips through both halves, including one event whose field is
absent, and that test is what forbids the drift.

| event | carries |
|---|---|
| `retrieval` | strategy, translation, variant count, per-chunk provenance **by variant index**, latency |
| `query_translation` | the planner's own latency and token counts, kept separate because ADR-0005 judges translation on the cost *it* adds |
| `rag_answer` | the chain's generation, with token counts |
| `agent_query` / `agent_turn` | one line per search with a `verbatim` verdict; a turn summary with token counts |
| `input_gate` | verdict, the layer that fired, the rule, and — **on a block only** — the normalised input |
| `citation_markers` / `output_validator` / `gate_classifier_unavailable` | marker resolution, layer-4 verdicts, layer-3 fail-open |
| `section_trimmed` | every ingest-time Section repair, so a trim is never silent |

**Turn correlation.** Every line in a turn shares a `turn_id`, set by a `ContextVar` scope, which
is what turns an aggregate into a join: a `retrieval` line carries provenance and, deliberately,
no question. The alternative correlation is line order, which is correct for a serial harness,
wrong the moment the model issues two searches in one step, and asserted by nothing either way.
The propagation is **measured** to survive LangGraph's tool executor across a thread boundary,
and measured **at the app** — its only production caller — because neutralising the app's own
`log_turn` once left all 1028 tests green: every other turn-id test opened the scope itself, so
the suite proved propagation and never wiring.

**The file sink** is `FINBRIEF_LOG_FILE`, **off unless named**, append-only, unrotated,
gitignored. Off by default because `configure_logging()` takes no arguments at four entry points
and the hermetic suite would otherwise write files — which means it has to be *enabled* where the
data matters, and `.env.example` ships the recommended path **commented out**. It first shipped
uncommented, and since the README and the app's own banner both say `cp .env.example .env`,
"off by default" was false for everyone who followed the setup instructions — silently, and what
it switched on was the retention of blocked questions' normalised text. Unrotated because a
rotating file renames mid-run and a globbing reader then double-counts or misses.

**What may never be in a field:** a secret, a question, or a query variant — a variant is derived
from user content and these lines are kept. Counts, lengths and verdicts only. **One exception,
bounded three ways** (ADR-0006 requires it, because a denylist you cannot audit is one you cannot
tune): the `input_gate` line carries the normalised input **only on a block**, **only in
normalised form**, and only `config.GATE_LOGGED_INPUT_MAX_CHARS` — 500 characters — of it. That
bound is not described as making the text unusable, because it does not: folding destroys figures
and identifiers (`Item 1A` → `item ia`, `$5bn` → `ssbn`) and leaves the wording legible. The cost
is that a false positive puts a readable innocent question in a kept log, which is the reason for
the three bounds rather than a reason to keep no record.

**What T10 consumed from it**, and nothing else: latency samples against ADR-0005's budget, token
counts, and the agent-vs-original query divergence rate. Each is read over **this run's own
window** of the file (`events.sink_offset`), because a statistic over an append-only sink is a
statistic over every run and app session that ever named it — the first committed evaluation
artifact reported a planner p50 over **13** pooled runs while its own header claimed the numbers
were that run's, and re-reading the unchanged file two runs later gave a different number. Two
consequences: a fully replayed stage appends nothing, so its window is empty and the p50
**raises** rather than serving the previous run's median; and the window is persisted beside the
cached cells, so a re-render describes the measuring run rather than refusing. The reader returns
*samples* and a count of the lines that carried nothing — **never a statistic**, so any median is
computed once, by whoever quotes it.

Gate-trigger records are deliberately **not** read by the harness: they exist to make the
denylist auditable, which is a different job. ADR-0011 records why the log is shaped as a run's
record rather than as the harness's primary input.

## Hard

### 3.15 Hybrid search

Query translation and hybrid search compose inside `retrieve()` as **one symmetric pipeline**:

```
variants   = (question,) + ticker_form? + sub_queries      # 1 + ≤1 + ≤3
retrievers = (vector,) + bm25 if hybrid
→ every variant through every retriever → RRF (k=60) → dedup by chunk id → top-k
```

Translation only ever **adds** — the original query is always retained, so BM25 always sees the
raw identifiers, and `vector` without translation is that same pipeline with one candidate list in
it, returning **exactly** what the pre-hybrid baseline returned (RRF over a single list is
strictly decreasing in rank, so fusion is a no-op on order). The A/B's baseline is unmoved, and a
test compares it against `nearest_chunks` directly.

**Entity normalisation** is the deterministic half of translation: when a Universe company's name
appears in a query, a **ticker-form variant** is added — `Tesla debt` → also `TSLA debt`. It is a
lookup in `config.TICKER_BY_COMPANY_NAME`, derived from `UNIVERSE`, so the `+translation` arm
gains a variant without gaining model variance, and one substitution pass over the whole question
bounds it at **one** extra variant however many companies are named.

**The case that motivated all of it** — `Tesla debt`, `k=5`, one collection, one day (ADR-0004
§6, `n=1`):

| state | strategy | translation | TSLA in top-5 | the target chunk's rank | its distance |
|---|---|---|---|---|---|
| 1 | vector | off | 1/5 | 5 | 1.0406 |
| 2 | hybrid | off | 2/5 | **absent** (rank 10 at `k=10`) | — |
| 3 | hybrid | normalisation only | 3/5 | **1** | 0.6778 |
| 4 | hybrid | normalisation + 3 sub-queries — **what ships** | **5/5** | 2 | 0.6777 |
| 5 | vector | normalisation only | 3/5 | **1** | 0.6778 |

**A pre-registered hypothesis failed here, and that is the finding.** The prediction was that BM25
on the retained original would move the chunk from rank 5 to rank 1 "because BM25 sees the raw
identifiers". Measured: hybrid alone moved it to **absent**. Root cause, with numbers:

| fact | measurement |
|---|---|
| the relevant chunk's own text | contains **no `tesla` token** — "*we* and our subsidiaries had outstanding $8.18 billion … of indebtedness". A filer writes "we" |
| `tsla` as a corpus token | 280 of 5,842 chunks — *all* of TSLA's, because the provenance header carries the ticker |
| `tesla` as a corpus token | **34** of 5,842 — body mentions only |
| `debt` as a corpus token | 413 of 5,842 |

So `tesla` carries enormous IDF and `debt` almost none, and BM25 ranked by how often the word
"Tesla" appears: used-vehicle trade-ins, human-capital oversight, "highly dependent on the
services of Elon Musk". **It recovered the filer and lost the topic.**

**And the mechanism that fixes it is embedding-side, not lexical** — which matters, because the
obvious narration credits BM25. State 5 is the evidence: the ticker form ranks the chunk **1st
under vector search alone**, at distance 0.6778 against the original's 1.0406, because the
provenance header's ticker moves the *embedding* as well as the lexical index. The chunk therefore
collects two independent vector votes plus one BM25 vote, and **RRF's agreement principle**
promotes it past four Ford chunks that each earned one. BM25 supplies one vote of three: its role
in the fix is **redundancy, not recovery**. That is why ADR-0004 §7 pre-registered hybrid's
marginal contribution on this bucket as *small* before any A/B data existed — a prediction
[3.16](#316-ab-testing-of-rag-strategies) then could not contradict.

**Two more measured BM25 behaviours**, both recorded before the golden set was authored so that
ground truth was not written against a known leak:

- **Question-form scaffolding.** A live risk-factors question returned GOOGL's `Item 7:21` at
  BM25 rank 1 of 5,678 matching chunks, matching `main`, `for`, `the`, `are` and **none** of
  `risk`, `factors`, `tesla`. Filers write "principal"; an analyst types "main" — so `main` is
  rare in 10-K prose (7 chunks in 5,842), earns idf 6.6568 against `tesla`'s 5.1261, and
  contributes 64% of that chunk's whole score. IDF measures corpus rarity, not query
  informativeness, and no corpus statistic can separate a rare question-register word from a rare
  identifier. Fixed **query-side only** (`hybrid.query_terms`), so the corpus keeps every term and
  any query carrying no scaffolding term scores byte-identically to the baseline. Stopword removal
  and a minimum-IDF floor were both measured **worse** and are recorded as falsified. The lexicon
  holds exactly one word, and no word joins it without a recorded `df` statistic *and* a measured
  query it fixes.
- **Cross-filer mention leakage**, which no lexical rule can fix. `what are the main risk factors
  for Microsoft?` returned NVDA's chunk reading *"our agreement with **Microsoft** could delay or
  prevent a change in control"* — a **correct** lexical match and the wrong grounding. `Apple`
  appears in 45 chunks belonging to four *other* filers (GS 21, JPM 16, META 7, LLY 1), and those
  are not incidental prose: they are the Apple Card portfolio transaction, discussed with dollar
  amounts in two banks' MD&A. The fact that separates "about company X" from "mentions company X"
  is not in the text at all — it is which filer filed it, which is metadata. Metadata filtering is
  recorded as future work needing its own ADR, because it would move every arm's candidate set
  immediately before the measurement and would break the multi-company peer comparison.

The precision cost of the BM25 arm is therefore reported per arm rather than averaged into the
bucket means, using labelled probes:

| arm | rows | chunks retrieved | labelled leaks | leakage-free precision |
|---|---:|---:|---:|---:|
| vector, ± translation | 2 | 10 | 0 | 1.000 |
| hybrid, ± translation | 2 | 10 | 1 | 0.900 |
| either, planner off | 2 | 10 | 0 | 1.000 |

Rows nobody enumerated false positives for contribute no evidence either way and are excluded
rather than counted clean — including them would dilute every rate towards 1.0 with rows that were
never probed.

### 3.16 A/B testing of RAG strategies

**Six arms**: {vector, hybrid} × {± translation}, plus **two planner-off ablations**
(`vector + normalisation` and `hybrid + normalisation`, i.e. translation on with
`max_sub_queries=0`, so the deterministic ticker form is still added and no model is asked
anything). Those two are not decoration: they are ADR-0004 §7's ablation and ADR-0005 §2's own
pre-registered **refutation channel** — if the `exact-identifier` win does not survive with the
planner off, normalisation is not what earned it.

Both switches are **named by the caller**, never read from config inside `retrieve()`, so a number
is never reported against a configuration nobody selected. The golden set is 28 questions in four
stratified buckets of 7 ([3.21](#321-ragas-evaluation)).

**The deterministic retrieval metrics need no judge and cost nothing** — every figure is computed
from `Retrieval`'s own return value against the golden set's chunk ids, so a reviewer can
re-derive it. The shipping default's per-bucket results, with `[min–max] n` because ADR-0005
requires the spread beside every mean:

| bucket | section recall | section precision | filer precision |
|---|---|---|---|
| semantic | 1.000 [1.000–1.000] n=6 | 0.629 [0.200–1.000] n=7 | 0.829 [0.400–1.000] n=7 |
| exact-identifier | 1.000 [1.000–1.000] n=7 | 0.486 [0.200–1.000] n=7 | 0.886 [0.600–1.000] n=7 |
| tool-augmented | 0.857 [0.000–1.000] n=7 | 0.514 [0.000–0.800] n=7 | 0.857 [0.600–1.000] n=7 |
| multi-hop | 0.929 [0.500–1.000] n=7 | 0.571 [0.200–0.800] n=7 | 0.771 [0.400–1.000] n=7 |

All six arms, all five columns, and the ablations are in
[`evaluation.md`](./docs/verification/evaluation.md). Two reading rules travel with them.
**`chunk recall` is a near-lottery on this corpus and is reported last for that reason**: S1's
reference was authored from 6 of TSLA Item 1A's **129** chunks, and one live run returned chunks
48–70 against targets 0, 1, 6, 34, 94 and 117 — chunk recall 0.000, section recall 1.000, for a
result an analyst would call correct. And rows whose every target Section holds ≤ 4 chunks are
**excluded** from every recall column and reported separately, because a `k=5` retrieval reaches
them whatever the retriever does.

### The pre-registered hypotheses, settled

Prediction → measurement → verdict, all six, on context precision or context recall — never on
response relevancy, which is fenced off in code. **Three verdicts, not two**: `not detected` means
the test had power and resolved nothing, `undetectable` means no arrangement of that many
differing questions could have reached α at all, and only the first is a measurement.

| # | prediction (pre-registered) | measurement | verdict |
|---|---|---|---|
| H1 | hybrid > vector-only on `exact-identifier`, at equal translation off | Δ **−0.087** [−0.256…+0.250], p=0.281, 6 of 7 differing | **not detected (n=7)** |
| H2 | translation is **positive** on `exact-identifier`, via entity normalisation | Δ +0.102 [+0.000…+0.250], p=0.125, 4 differing | **undetectable (effective n=4)** |
| H3 | translation wins on `multi-hop` | Δ +0.006 [−0.500…+0.500], p=1.000, 5 differing | **undetectable (effective n=5)** |
| H4 | all configurations ≈ tie on `semantic` — a predicted tie is evidence the experiment is sound | Δ +0.007 [−0.333…+0.500], p=0.688, 6 differing | **not detected (n=7)** — *consistent with* the prediction rather than a confirmation of it |
| H5 | hybrid's marginal contribution over `vector + translation` on `exact-identifier` is **small** | Δ +0.076 [−0.133…+0.250], p=0.312, 5 differing | **undetectable (effective n=5)** |
| H6 | hybrid's contribution may be **larger on `semantic`** than on the bucket it exists to win | Δ +0.007, p=0.688, 6 differing | **not detected (n=7)** |

**Both pre-registered decisions are evaluated on every run and printed whether they fire or not** —
a trigger reported only when it fires is a trigger a reader cannot tell was evaluated.

- **The falsification clause** (translation worse on **both** context precision and context recall
  within a bucket → drop to hybrid-only) **does not fire on any bucket** — and **could not have
  fired on any bucket**: all eight cells undetectable, effective `n` as low as 1. Where a clause
  could not have fired, the row is a statement about the sample and not evidence for the default
  it protects.
- **ADR-0005 §4's re-examination trigger** fires on an *absence* of gain, and it **fires on 1 of 4
  buckets** — the only one with the power to resolve a gain. Its condition as written says *every*
  bucket, so as written it is **not met**; the count is printed because it is the difference
  between the clause as registered and the clause as it could be applied.

### The power audit — which of these cells is a measurement

| verdict | cells | how to read it |
|---|---:|---|
| not detected | **4** | **a measurement**: the test had power here and resolved nothing |
| undetectable at this n | **14** | **not a measurement**: too few differing questions for any result |

**4 of 18** pre-registered comparisons carry a measurement. The rest are statements about the
sample — and the reason is arithmetic, not luck. The exact two-sided p of a paired signed-rank
test cannot fall below `2 / 2**m` for `m` differing questions, so α=0.05 is reachable only from
**m ≥ 6**:

| differing questions | lowest reachable p | minority signs tolerated |
|---:|---:|---:|
| 5 | 0.0625 | none — no result possible |
| 6 | 0.0312 | 0 |
| 7 | 0.0156 | 1 |
| 8 | 0.0078 | 2 |

So at the floor of 6 differing questions the effect must be **perfectly unanimous**, and at 7
exactly one question may disagree. **This design can only resolve near-unanimous effects, at any
effect size.** A real difference of 0.2 that holds on five of seven questions is not weakly
supported — it is *unresolvable*. That is a property of ADR-0002's bucket size rather than of the
test: an exact test is the right instrument at this `n` precisely because it refuses to claim what
the sample cannot support, where a normal approximation over seven paired differences would have
returned a confident-looking number instead. **The per-bucket A/B is a screen for large unanimous
effects, not a test of small ones**, and every conclusion drawn from it inherits that.

### Cost is part of dominance, and this is where it fails

ADR-0005 judges dominance *within* a latency budget, measured from the persisted event log rather
than from a stopwatch — a stopwatch around `retrieve()` cannot split the planner's chat round from
the retrieval rounds, and that split is the interesting half.

| | ms | samples |
|---|---:|---:|
| planner's chat round, p50 | 1838 | 48 |
| retrieval p50, translation on (planner enabled) | 1688 | 64 |
| retrieval p50, translation off | 313 | 56 |
| retrieval rounds translation adds, p50 | 1374 | — |
| **total p50 added by translation** | **3212** | — |
| ADR-0005's pre-registered budget | 1500 ms | — |
| verdict | **over budget** — missed, and left **unamended** | |

Two honesty notes travel with that table. The planner's figure is a **reconstruction**: the scored
`+translation` arms replay a recorded planner reply, so their own lines record ~1 ms, and the
median comes from the resolve pass where the planner really runs. And the "translation on" pool is
the arms whose configuration *enables* the planner, not every arm carrying `translation: true` —
keyed on the turn's own `max_sub_queries` rather than on the observed variant count, because a
planner that **ran and refused** still paid for a full chat round and belongs in the pool. An
earlier version conflated the two and biased the p50 upward, making the miss look worse than it
is.

**The budget is not amended to the measured value, deliberately.** ADR-0006 revised its gate
budget and earned it — a figure cleared on eight of eight passes, with an argument and both
numbers printed side by side ever since. Nothing equivalent exists here: there is no argument that
3.3 seconds of added latency is acceptable for an analyst's turn, and moving the number to
wherever the measurement landed is pre-registration in reverse.

### 3.21 RAGAs evaluation

**The golden set.** 28 questions, four stratified buckets of 7 — `semantic`, `exact-identifier`,
`tool-augmented`, `multi-hop`. Guardrail and injection cases are **excluded by decision**, because
faithfulness against a refusal is undefined; they live in the security suite as pass/fail.

*Source separation*, which is what makes the numbers non-circular: candidate Q/A pairs were
drafted by a **different model** than the answering pipeline, then reference answers were authored
by reading the filings and the ingested chunk text directly, with Sections and chunk ids cited.
`retrieve()` was never called and the app was never opened while authoring, so no reference is
derived from retriever output.

*Hand-verification*: all 28 rows were checked against their EDGAR filings, and the set-level flag
is bound to the conjunction of the per-row flags — so it cannot read `true` over a partial pass.
`evaluation/loader.py` **raises** on an unverified set rather than footnoting it. Three further
schema fields exist because measuring the set taught they were needed:
`known_false_positives` (a correct lexical match that is the wrong grounding — the mention-leakage
probes), `recall_trivial` plus `recall_trivial_sections` (a target Section a `k=5` retrieval cannot
miss, whole-row and partial), and `basis_mismatch` (figures not measured on the same basis or at
the same date, where the reference must state each basis rather than imply comparability).

**Per-bucket RAGAs, all four metrics, shipping default:**

| bucket | faithfulness | answer relevancy ⚠ | context precision | context recall |
|---|---|---|---|---|
| semantic | 0.892 [0.636–1.000] n=7 | 0.908 [0.749–1.000] n=7 | 0.529 [0.000–1.000] n=7 | 0.571 [0.000–1.000] n=7 |
| exact-identifier | 0.643 [0.000–1.000] n=7 | 0.903 [0.745–0.945] n=7 | 0.874 [0.367–1.000] n=7 | 1.000 [1.000–1.000] n=7 |
| tool-augmented | 0.914 [0.667–1.000] n=7 | *not comparable — 5/7 noncommittal* | 0.401 [0.000–1.000] n=7 | 0.619 [0.000–1.000] n=7 |
| multi-hop | 0.582 [0.000–1.000] n=7 | 0.827 [0.671–0.944] n=6 (+1 excluded) | 0.370 [0.000–1.000] n=7 | 0.509 [0.143–1.000] n=7 |

Over all 28 rows on that arm: faithfulness **0.758**, context precision **0.543**, context recall
**0.675**, and the free deterministic section recall **0.944** (n=27, excluding the one
recall-trivial row). All sixteen bucket × arm rows are in
[`evaluation.md`](./docs/verification/evaluation.md).

**⚠ Response relevancy is excluded from every pre-registered hypothesis, for two measured reasons
rather than one.** First, ragas forces temperature 0.3 whenever it asks for more than one
completion, and `ResponseRelevancy` asks for three — so the column moves between runs on any judge
at any temperature this code names. Second, and worse: **the n=3 is not honoured.** The provider
answers with one completion, logging `LLM returned 1 generations instead of requested 3` on every
judged cell of every run so far, so the column is a cosine similarity against **one**
model-generated question sampled at 0.3 — not the mean over three the metric is defined as.
Measured consequence: between two runs over the *same* cached contexts and the *same* answers,
cells moved by ±0.01–0.02, while every other number in the artifact replays cell for cell. The
exclusion is machine-checkable (`judge.EXCLUDED_FROM_HYPOTHESES`) and
`Prediction.__post_init__` refuses to build a prediction on a fenced metric.

That column also carries a **noncommittal** count instead of a mean where a chain answer correctly
says the retrieved filings do not carry a current share price: ragas' flag fires and the metric
scores ≈0 **for being right**, and printing that beside a semantic bucket's 0.9 would invite a
comparison it cannot support. Where a mean is reported over fewer questions than the bucket holds,
two effects are compounded — excluding a flagged answer is a deliberate fix, and the rest is
jitter — so `n` is printed on every mean in that column.

**How the harness stays honest**, three properties worth knowing:

- **The chain runs unmodified, and the harness checks that it did.** Scoring faithfulness needs an
  answer generated over the arm's own contexts, and the obvious shortcut — calling the generation
  half directly over those contexts — was written and then rejected as a second code path through
  the thing being measured. Instead the chain runs with the arm's configuration and its replayed
  planner, and the contexts it returns are compared against the ones the arm was scored on; a
  mismatch **raises** rather than scoring an answer against contexts the scored retrieval never
  surfaced.
- **The judge is `gpt-4.1-mini`, not the answering model.** `ragas.evaluate()` is deliberately
  unused because the cache needs a result per `(row, metric)`.
- **Every paid cell is content-addressed and resumable.** The last run replayed **770** cells and
  paid for 0 judge calls; a run killed by a 429 pays for the cell it died inside and nothing else.
  `HARNESS_VERSION` is how a cell that is *wrong under a right address* gets evicted.

**The tool-calling eval is reported beside these tables and never inside them**, because it
measures the agent's *selection* layer — a different and nondeterministic instrument. Ten scored
cases: the seven `tool-augmented` golden rows built from their own `tool_expectation` field, plus
three negative controls the golden set cannot express, because it holds no row whose right answer
is *not to call a tool* (a retrieval-only question that must not fetch a quote, an out-of-Universe
ticker that must be refused as a result, and an advice-shaped question that must not send the loop
off to price the recommendation). **Accuracy 100%** — and **every case in that denominator can
fail**, which was not true when the rate was first published (see
[5.3](#53-the-recurring-theme-a-claim-the-thing-making-it-could-not-check)). The valuation
quote-plus-peers pairing that `tool_expectation`'s one-tool shape cannot express is measured as its
own rate — **1 of 2** — rather than by reshaping hand-verified reference data.

---

# Part 4 — Optional-task status

All 21 optional tasks. The numbers are this repo's local convention (see
[Part 3](#part-3--optional-tasks-implemented)); the names are what identifies each row. **13
complete — Easy 4/4, Medium 6/10, Hard 3/7**, against a bar of 2 medium + 1 hard.

| # | task | status | where / why |
|---|---|---|---|
| **Easy** | | | |
| 1 | Conversation history + export | ✅ complete | [3.1](#31-conversation-history--export) |
| 2 | RAG process visualisation | ✅ complete | [3.2](#32-rag-process-visualisation) |
| 3 | Source citations | ✅ complete | [3.3](#33-source-citations) |
| 4 | Interactive help / guide | ✅ complete — panel and buttons ship; the `/help` **command** is open | [3.4](#34-interactive-help--guide) |
| **Medium** | | | |
| 5 | Multi-model support | ✕ not built | Tier-2, behind the gate, and it carries an open question that must be answered first: the injection classifier and the tool-calling prompts are tuned per model, so a picker needs a per-model check that swapping does not silently break tool selection |
| 6 | Real-time data updates to the knowledge base | ✕ not built | Tier-2, with an open question: whether a live ingest stays idempotent *and* holds the ADR-0007 section gate without racing the cached retriever and the agent's state |
| 7 | Prompt-injection protection | ✅ complete | [3.7](#37-prompt-injection-protection) |
| 8 | User authentication and personalisation | ✕ not built | Tier-2 tail. It is also what real rate limiting needs — a bound keyed server-side on something the client does not choose — so the gap is stated in [3.13](#313-rate-limiting--api-key-management) rather than implied |
| 9 | Token usage + cost display | ✅ complete — per **conversation**; the per-**message** half is open | [3.9](#39-token-usage--cost-display) |
| 10 | Tool-call result visualisation | ✅ complete | [3.10](#310-tool-call-result-visualisation) |
| 11 | Conversation export in various formats | ✅ complete — JSON and CSV; **PDF declined** on dependency grounds | [3.11](#311-conversation-export-in-various-formats) |
| 12 | Connect to tools from a public remote MCP server | ✕ not built | Tier-2 #3 by skill signal, and the highest-value unbuilt item after deployment. Needs a remote server chosen *and* the MCP security review that goes with it; not started rather than half-done |
| 13 | Rate limiting + API key management | ✅ complete — and explicitly **not** a security control | [3.13](#313-rate-limiting--api-key-management) |
| 14 | Logging + monitoring | ✅ complete | [3.14](#314-logging--monitoring) |
| **Hard** | | | |
| 15 | Hybrid search | ✅ complete | [3.15](#315-hybrid-search) |
| 16 | A/B testing of RAG strategies | ✅ complete | [3.16](#316-ab-testing-of-rag-strategies) |
| 17 | Automated knowledge base updates | ✕ not built | a scheduled GitHub Action over the ingest script; generic scheduling work that reinforces none of the Tier-1 core, and it inherits row 6's idempotency question |
| 18 | Multi-language support | ✕ not built | the knowledge base is single-language English and the embeddings are English; a UI toggle without query-side translation into English before retrieval would be a language switch that degrades retrieval silently |
| 19 | Advanced analytics dashboard | ✕ not built | a second Streamlit page over the event log. The data exists ([3.14](#314-logging--monitoring)) and the reader exists; what it adds is a chart, not a GenAI capability, so it lost to everything above it |
| 20 | Implement your tools as MCP servers | ✕ not built | Tier-2, with the sharpest open question of the four: whether this is a genuine protocol *port* or a second copy of the finance logic behind a second interface. It must reuse one implementation, and that is a design decision rather than a build |
| 21 | RAGAs evaluation | ✅ complete | [3.21](#321-ragas-evaluation) |

**Why the unbuilt eight are unbuilt, in one sentence.** ADR-0001 splits scope into a review-facing
Tier-1 and a skill-stretch Tier-2 with a hard gate between them, and ADR-0010 orders Tier-2 by
GenAI/RAG skill signal rather than by the assignment's difficulty tags. Everything above is
Tier-1 plus the four tail items that turned out to cost under an hour each because Tier-1 had
already built their substrate. The eight remaining are either generic web-app work (17, 18, 19)
or carry a recorded open question that has to be answered before the work starts (5, 6, 12, 20) —
and the cut-if-undefendable rule says an item nobody can explain is worth less than an item that
does not exist. **Deployment plus a live URL is Tier-2 #1 and also unbuilt**; it is not on this
list because it is not one of the 21, but it is the highest-value remaining item, since reach
gates the value of everything else.

---

# Part 5 — What this project found out

## 5.1 What the evaluation established

**The pre-registered default is `hybrid + translation`, and it is retained because the experiment
could not resolve the question — not because it was validated.** Every figure here is requoted
from [`evaluation.md`](./docs/verification/evaluation.md).

- **Hybrid earns nothing detectable on any bucket.** `hybrid − vector` at equal translation is
  *not detected* on `semantic` and *undetectable at this n* on the other three.
- **The point estimate on its own predicted bucket is negative**: on `exact-identifier`, the
  bucket hybrid exists to win, Δ **−0.087**.
- **The one root-caused live case favours the simpler arm**: `vector + normalisation` put the
  target chunk at rank 1 against the default's rank 2.
- **Translation costs 3212 ms p50 against a 1500 ms budget**, recorded as missed and left
  unamended.
- **The clause that protects the default could not have fired on any bucket**, and the §4 trigger
  fired on the one bucket that had the power to say anything.

So `vector + translation` is a live candidate this run could not rule out — it is simpler and it
costs no BM25 index over the whole corpus. The default does not move on this evidence because *no*
configuration is preferred on this evidence, and moving a pre-committed default on an unresolvable
comparison is the same error as keeping it on one. What would settle it is a larger per-bucket `n`
and nothing else.

**And the first version of that whole determination was a tautology.** The original comparator
judged a difference of *means* against `basis` — the larger of the two arms' own per-question
range. But each arm's mean is bounded by that arm's own min and max, so the largest delta the
observed values can produce is `max(cand.max − base.min, base.max − cand.min)`; and when two arms
score *the same questions* under near-identical configurations, that quantity **equals the
range**. The test was `abs(delta) <= basis`. **It could not fail.**

The consequence was not small. All **18** pre-registered comparisons in the first committed
artifact — six hypotheses, eight falsification-clause cells, four trigger cells — were rendered as
verdicts by an instrument mathematically incapable of returning any other. Four hypotheses were
published as **refuted** on that basis. H2 was refuted while its delta ran +0.102 in the
*predicted* direction, leaving ADR-0004 §6's root-caused live case standing unreconciled beside
it. Both of ADR-0005's pre-registered decisions were tautologies: the falsification clause could
not fire, and the absence-triggered test could not fail to fire.

**How it was found matters more than the fix.** By a fresh-context review reading the artifact's
own printed `[min–max]` beside each mean and doing the subtraction nobody had done. The
information needed to catch it was in the committed evidence the whole time. The pre-registration
was sound — written pre-data, in code, refusing fenced metrics; **the instrument judging it was
not**, and those are separable failures. A pre-registered prediction is only as good as the test
that settles it.

The replacement pairs on the **question** — the arms score the same rows, so the per-question
difference removes the question's own difficulty, the term that dominated the raw spread — and
settles direction with an **exact** Wilcoxon signed-rank test whose null is built by convolution
rather than approximated. Three outcomes, not two, and `Paired.detectable` is what keeps
"we could not have seen it" from reading as "there is nothing there".

None of that is a defect in the pipeline. It is a measurement that came back saying *we cannot
tell*, reported as that instead of as a result.

## 5.2 Which half of the pipeline is deterministic

Two findings from the same runs answer this precisely, and they point opposite ways.

**Retrieval is exactly reproducible, and that is the strongest empirical claim in the project.** A
code review changed the retrieval cache key, so all **168** retrieval cells — 28 questions × six
arms, baselines, translated arms and both planner-off ablations — were re-paid from scratch
against the same collection. Everything downstream then **replayed: 672 cells with zero misses**,
being 560 judge cells (all four metrics) and 112 answer cells. Those keys are not identifiers: the
judge key carries the **full text of every retrieved context**, and the answer key carries the
chunk ids plus a sha256 of the context bodies. A single character different in any chunk of any
cell, on any arm, and that cell would have missed and been re-paid. None did.

So: given the same collection and the same question, `retrieve()` returns the same chunks in the
same order, byte for byte, on every configuration this project ships or ablates. This is the third
independent confirmation and the first covering the whole matrix.

**The planner is not, and the same runs measure that too.** ADR-0004 §9's n-repeat asks the
planner for sub-queries five times per question at temperature 0. Three successive runs of that
pass reported **0, 1 and 2 of 8** questions returning an identical set every time. The count is
itself re-sampled, because the pass makes live planner calls and therefore **inherits the variance
it is measuring** — which is why it is quoted as a range and never as one run's figure. The
committed artifact's own run says 2 of 8.

**The conclusion is identical in all three, and that is what makes it usable: 6, 7 and 8 of 8
questions varied.** So the resolve-once replay that the two `+translation` arms depend on is
**load-bearing, not caution**: without it those arms would report different per-bucket numbers on a
re-run with no code change. ADR-0004 §9 registered the opposite outcome as possible and called it
"not one to assume"; it was right not to.

Temperature 0 is greedy decoding, not a determinism guarantee. OpenRouter fronts many upstreams
and no `seed` is sent — a `seed` was considered and rejected, because the OpenAI-compatible
parameter is best-effort even at its origin and a request routed elsewhere ignores it entirely, so
adding it would buy a little stability and a false claim.

**Together they locate the nondeterminism exactly: it is in the model calls, not in the
retrieval.** Everything between the query variants and the ranked chunks is reproducible; the
planner that writes those variants is not, and neither is the judge — response relevancy is
re-sampled every time it is judged, which is why the artifact fences that column off from every
pre-registered decision.

## 5.3 The recurring theme: a claim the thing making it could not check

The same defect, seventeen times, across ingestion, retrieval, security, evaluation,
instrumentation and process. They look unrelated apart, which is why they are listed together.

| where | it asserted | what it had established |
|---|---|---|
| tiktoken's warm cache (twice) | a hermetic suite | nothing: it passed locally and egressed in CI |
| the egress guard's docstring | `curl_cffi` coverage | nothing — `curl_cffi` resolves and connects in C, touching `socket` not at all |
| a conftest comment | that `gethostbyname` routed through `getaddrinfo` | nothing: each is its own CPython entry point, and only one was patched |
| `serial_full_brief < MAX_AGENT_STEPS` | that the step ceiling was pinned | nothing: 15, 20 and 24 all satisfy it |
| `SuiteRun.passed` via `all(())` | **SUITE PASSED**, exit 0 | nothing: an empty suite is vacuously true |
| the first indirect-injection run | "not obeyed, no leak" | nothing: the agent asked which company was meant and never retrieved the payload |
| layer 4's `advice_hits` | that advice was refused | nothing after the first match per rule: *"I can't give a price target. My price target for NVDA is $260"* returned no hits |
| layer 3's `_verdict` | a verdict | nothing on `**YES**` — not in the strip list, so *undecided*, and undecided allows |
| `metrics.compare` | a verdict on each of 18 pre-registered comparisons | nothing: a delta of means tested against the arms' own range, which is the largest delta those values permit |
| the tool eval's control **C3** | a pass, inside a published **100% over 10 cases** | nothing: no expected tool, no forbidden tool, no argument, so `passed` was `True` for every possible behaviour |
| the judge stage's error path | `APIConnectionError: Connection error.` | nothing about the network: one client reused across per-cell event loops, and `httpx` raised `bound to a different event loop` |
| the first fix for that | a fresh client per cell | nothing about the defect: `langchain_openai` caches the async client *below* this repo's constructor, so distinct objects shared one pool |
| **the figure-binding test itself** | that every quoted figure is in the artifact | nothing, for short figures: it matched substrings, so `"8"` is satisfied by `18`, by `0.087`, by a date |
| "1324 tests green" | that the suite passed | nothing CI had seen: that commit was pushed inside a later push, so only the tip was built |
| `log_turn` at the app | turn-id propagation | nothing about wiring: neutralising it left all 1028 tests green, because every test opened the scope itself |
| `citation_markers` | a bracket-adherence denominator | nothing a harness can reach: emitted by the page and by nothing else |
| the spend panel's classifier caveat | that the unmetered call is named | nothing on a complete total: it shipped as a clause of the *partial* banner |

**The generalisation: a guard is code, and inherits every failure mode of the code it guards.**
Four of these sat *inside published measurements*, one sat inside the mechanism built to prevent
the others, and two were kept hidden by ordinary good practice — the **cache** meant the failing
path was never exercised, and the **retry** self-healed it so it failed somewhere different every
time.

Three rules follow, and they are the ones this repo now applies to instruments as well as to code:

- **Prefer a check that exercises the thing over one that describes it.** Every control in the
  tool eval is now driven against an agent scripted to call all three finance tools and asserted
  to *fail*; every hermetic test calls the backend it is about rather than asserting that one path
  routes through another.
- **Prefer an equality over a bound.** The figure-binding test matches whole numbers; the
  translation budget is pinned by an equality; the scope panel's line count is an equality,
  because "compress" that permits any number of bullets is not a constraint.
- **For any check about an adversarial input, assert that the input arrived.** `retrieved` is a
  required column on every planted payload, and a bypass found by review is **added to the
  corpus**, not merely fixed.

And two corollaries worth stating on their own. **A passing rate is only as strong as the weakest
case in its denominator** — "100% over 10" with one case that could not fail is a claim about nine
and one piece of padding. **A green local run is a hypothesis about CI, never a result from it.**

### Three libraries, three that phone home by default

The same shape in the dependency graph, and it changed how this project adds a dependency. Every
library added here **for quality or safety** ships with a telemetry path enabled:

| library | added for | what it sends, unconfigured | how it is switched off |
|---|---|---|---|
| `guardrails-ai` | the output validator | a record of every validated answer, to its own endpoint | `guard.configure(allow_metrics_collection=False)`, in `security/advice.py` |
| `uvloop` (via guardrails) | nothing — it arrives transitively | nothing itself, but it becomes the **process-wide** event loop and resolves DNS in libuv, outside a Python-level guard | `GUARDRAILS_RUN_SYNC`, plus the guard covers the backend |
| `ragas` | the four RAGAs metrics | a POST per metric completion, to `t.explodinggradients.com` | `RAGAS_DO_NOT_TRACK`, set in `evaluation/judge.py` **and** in `conftest.py` |

**The switch is in a different place every time** — an SDK method call, an environment variable
that must be set before a cached read, an event-loop policy — and only one of the three documents
it anywhere a reader would look. **The failure is silent by construction**: both libraries return
a completely normal result while a packet is being attempted, which is why
`conftest.EGRESS_ATTEMPTS` is the only detector that survives, and why it has now been the only
detector three times. And **off-by-default has to be set twice**, where the library is used and in
`conftest.py`, on the principle that a hole is a hole whether today's code walks through it.

## 5.4 Prompt rules are instruments, not enforcement

The strongest generalisable claim these measurements produced is not a retrieval number. It is
that **a rule stated in a prompt is a thing you can measure compliance with, and not a thing you
can rely on** — and it stands on six independent instances, one of them measured at scale.

| stated rule | where | what was measured |
|---|---|---|
| pass the user's question to `search_filings` **verbatim** | the tool's description (ADR-0003 §1) | **100% divergence, 8 of 8 searches.** Every query the agent issued differed from the question as typed — and all 8 are *first* searches, which cannot be reference resolutions, so none of them is the one rewrite the description permits |
| decompose a question into sub-queries when asked to | the planner's prompt (ADR-0004 §6) | the planner refuses, or returns prose refusals the parser has to strip |
| square brackets are reserved for retrieved excerpts | `AGENT_SYSTEM_PROMPT` | `[Yahoo Finance]` observed live; uncited grounded answers observed live |
| every figure comes from a tool or a retrieved excerpt | `AGENT_SYSTEM_PROMPT` | an answer naming Tesla's real segments from a chunk about industrial fasteners |
| a valuation question wants the quote **and** the peer comparison | `AGENT_SYSTEM_PROMPT` | **1 of 2** |
| these four headings structure the brief | `AGENT_SYSTEM_PROMPT` | read as **search terms**: `search_filings("Business")` named no company, retrieval returned six other filers' chunks, and the model wrote two sections from its weights with no `[n]` at all |

The verbatim rule is the sharpest, because every divergence is one the description forbids
outright, and the denominator is honest: **8** searches from one run's own window of the log, not
the 40 an earlier draft reported by pooling 13 appended runs. Per ADR-0003's standing position it
is **a finding rather than a defect to tune away** — a prompt tuned against a paid model until the
number looks good is a number about the tuning.

**The same shape holds one layer down, where the rule is code rather than prose.** Layer 4's
advice validator refuses every recommendation that announces itself and **none** of six
hand-labelled recommendations that do not: no imperative, no rating word, no price target, no
position-sizing instruction, and *"if it were my own capital I would be adding to Ford on any
further weakness"* goes straight through.

| | value |
|---|---|
| hand-labelled recommendations **not** refused | 6/6 |
| positive controls refused, which is what makes that a measurement | 10/10 |

Both halves, always together: the residue alone is equally consistent with a *dead* validator,
since `validate_answer` fails open by design and a fail-open also yields 100%. `n` is small and
hand-authored, so this is a statement about the rules' **generality**, not a 100%-evasion claim. A
rule set that pattern-matches the vocabulary of advice catches the vocabulary, not the advice. And
it cost nothing to measure — the validator is a regex set behind a Guard, so what it misses is
computable with no model and no live run, which means this could have been closed at any point
after the layer shipped.

**What was made structural instead**, which is the part that generalises into the architecture:

| constraint | made unrepresentable by |
|---|---|
| two searches in one step cannot collide on `[1]` | citation numbering assigned in one sequential pass at one seam |
| two callers cannot disagree about the collection | `filings` opened, written and read in one file |
| an out-of-Universe ticker cannot be fetched | the whitelist is a lookup, checked before any network call |
| two concurrent callers cannot both construct a Chroma handle | `caching.build_once` over all seven process singletons — `lru_cache` is atomic about its bookkeeping and says nothing about the function it wraps |
| a fourth quarantine tag cannot be escaped-but-not-denylisted | both halves parametrised over one `QUARANTINE_TAGS` tuple |
| a hypothesis cannot rest on a fenced metric | `Prediction.__post_init__` refuses to build one |
| the cap a prompt states cannot differ from the cap enforced | the prompt is a function of the cap |
| a scope sentence cannot disagree with itself | one constant, derived from `config`, read by the page and four prompts |

Where a constraint **cannot** be made structural — and the verbatim rule cannot, because the one
edit it must permit is indistinguishable from the rewrite it forbids — the honest response is to
instrument it and publish the rate.

---

# Part 6 — Limitations, ADRs, cost

## 6.1 Limitations, each with its mechanism

| limitation | mechanism |
|---|---|
| **the square-bracket adherence rate is unmeasured** | `citation_markers` is emitted by `app/Home.py` and by nothing else, so a harness driving `agent.answer` produced 10 live turns and **zero** lines. The turns happened and the markers were produced; the instrument is somewhere else. An instrument at the display layer has a human in its denominator, and moving it needs a decision about where marker resolution belongs — so it is its own ticket rather than a change smuggled into a measurement run |
| **cited-sentence support is mostly partial** | 22 fully / 40 partly / 8 not supported over 70 pairs. A partly supported sentence has a marker that resolves and a chunk carrying *some* of the claim — invisible to the citation register, and scored as fine by whole-answer faithfulness because the claim is supported somewhere in the context set |
| **the translation latency budget was missed and left unamended** | 3212 ms against 1500 ms. No argument exists that 3.3 s is acceptable for an analyst's turn, and moving a pre-registered number to wherever the measurement landed is pre-registration in reverse |
| **the per-bucket A/B is a screen, not a test** | ADR-0002 sized buckets at ≥6 questions; the exact paired test needs 6 differing questions with zero minority signs, or 7 with one. Small real effects are *unresolvable*, not weakly supported |
| **`yfinance` is unofficial** | an undocumented endpoint with no API contract, so an endpoint change is a fetch failure rather than a support ticket. The TTL cache, the retry and the stale banner are how it degrades; Alpha Vantage's free tier is 25 calls a day and its key is configured but **unread** — a fundamentals fallback is deferred, not shipped |
| **a stalled quote endpoint holds the cache lock ~91.5 s** | two HTTP requests per attempt × 3 attempts, all inside `TimedCache`'s lock, which is held across the refresh so two sessions asking about one ticker coalesce into one call. Cutting it further means retrying *outside* the lock, which trades the free tier's call budget for latency and voids ADR-0009's "peers add zero API surface" — a decision made in the open rather than an optimisation applied quietly |
| **single-language knowledge base** | English filings, English embeddings, and no query-side translation into English before retrieval. A non-English question is embedded as-is and retrieves accordingly |
| **no re-ranking stage** | promoted to Tier-2 #2 with a pre-registered hypothesis (precision lift concentrated on `semantic`, ~neutral on `exact-identifier`) and a constrained third A/B axis. The local cross-encoder's deploy weight is the accepted cost |
| **the knowledge base is four Items of one filing** | Item 8's financial statements, 10-Qs, proxies, earnings calls and every other Item are out of scope; a question whose answer lives outside Items 1, 1A, 7 and 7A is unanswerable by design, and the app says so |
| **table and figure fidelity inside an ingested Section** | extraction keeps the prose and can lose a table's layout, so hard numbers come from the finance tools rather than the filing text |
| **foreign private issuers are out of scope** | a 20-F has no Item 1A/7/7A, so such a filer cannot supply a Section at all; supporting one needs its own section mapping and its own gate. The Universe is curated to 10-K filers, and EDGAR is asked to confirm it per filing |
| **Item 1 → 1A and 1A → 1B have no boundary check** | there is no unambiguous next-Item marker (a business section discusses its risks in passing), and a false accusation teaches a reader to wave the gate through. For those two the length ceiling is the only backstop |
| **JPM's Item 7 runs one page past its cross-reference** | Item 7 is defined content-anchored, from its own heading to the start of Item 8; trimming to the p.160 footer needs filer-specific page-number parsing to buy 3,091 characters and would reintroduce the special case the heading rule removed |
| **the provenance header carries the ticker, not the company name** | which is why hybrid search *alone* made the motivating case worse. The index-side fix — carrying both forms — is deliberately not taken: the header is inside `content_hash`, so it re-embeds all 5,842 chunks and invalidates every distance recorded on three tickets, including the committed smoke band and the before/after this finding rests on |
| **the gate's four stated gaps** | layer 3 **fails open** (an outage allows the turn; not hypothetical — three candidate models were unavailable on this account and fail-open made them read as the *fastest* rows in a benchmark, catching nothing) · layer 4's novel-phrasing blind spot, with no layer 5 · **a refused answer is still in the agent's memory**, because the validator guards the surface and not the checkpointer · the homoglyph map is not the Unicode confusables table |
| **a green suite is evidence about its corpus** | the corpus author's blind spot and the rule author's are one blind spot, and running the suite again does not split them. Two bypasses were found by adversarial reading, not by eight passes |
| **the Universe is fixed at ingest time** | 15 companies, one filing each; adding one is an ingest run, not a setting |
| **no deployment and no live URL** | Tier-2 #1 and still the highest-value remaining item, since reach gates the value of everything else |

Two deliberate asymmetries that read as limitations and are not. A **gate-blocked question is not
in the agent's memory at all** — layers 1–3 stop the turn before the agent runs, so the refusal is
on screen while the checkpointer never saw the question, and a follow-up cannot build on it. And a
**retrieved-but-uncited chunk still appears in the sources panel**, because the panel's contract is
"what grounded this turn", which is also why the log field is `retrieved_sections` and not
`cited_sections`.

## 6.2 ADR index

| ADR | decision |
|---|---|
| [0001](./docs/adr/0001-two-tier-scope.md) | Two-tier scope: review-facing core, then skill-stretch, with a hard gate between them — amended when four demoted mediums came back |
| [0002](./docs/adr/0002-golden-set-and-per-bucket-ab.md) | Golden-set design and per-bucket A/B — source separation, four buckets, pre-registered hypotheses; amended three times, including the comparator that could not fail |
| [0003](./docs/adr/0003-retrieval-engine-vs-agent-tool.md) | The retrieval engine is separated from the agent's use of it, so the headline numbers measure a chain and the tool eval measures the loop |
| [0004](./docs/adr/0004-translation-hybrid-composition.md) | How translation and hybrid search compose inside `retrieve()` — twelve sections, including the falsified hypothesis (§6), the BM25 scaffolding fix (§10) and mention leakage (§11) |
| [0005](./docs/adr/0005-shipping-default-strategy.md) | The shipping default, pre-committed before any A/B data existed — amended to *retained, not validated* |
| [0006](./docs/adr/0006-security-gate.md) | The four-layer security gate — amended with the ≤800 ms miss and the two bypasses review found |
| [0007](./docs/adr/0007-kb-curated-sections.md) | Knowledge base = curated 10-K Sections, not full filings, with the gate that proves it |
| [0008](./docs/adr/0008-agent-state-ownership.md) | Conversation-state ownership across the agent and Streamlit |
| [0009](./docs/adr/0009-in-universe-peers.md) | Peers drawn exclusively from the Universe — zero new API surface |
| [0010](./docs/adr/0010-tier2-ordering.md) | Tier-2 ordered by GenAI skill signal; re-ranking promoted to first-class |
| [0011](./docs/adr/0011-observability-log-shape.md) | The log is a run's record, not the harness's data bus — and an instrument is emitted where the behaviour is produced |

## 6.3 What it cost

Five non-hermetic entry points; four of them spend money. **No dollar figure in this project comes
from a committed artifact**, and this section is careful about the difference:
[`evaluation.md`](./docs/verification/evaluation.md) carries token counts, not prices, and both
cost knobs default to unset for the reason [3.9](#39-token-usage--cost-display) gives.

| entry point | what it spends | measured scale |
|---|---|---|
| `ingest_filings.py` | EDGAR (free) + one embedding pass over the Universe | 5,842 chunks embedded |
| `ingest_filings.py --dry-run` | nothing — no key needed | — |
| `retrieval_smoke.py` | five query embeddings with the same paid model the ingest used | — |
| `security_suite.py` | ~35 one-word completions (every case the denylist does not catch, plus the whole benign set) plus a handful of agent turns | the cheapest of the four |
| `evaluate.py` | ~1,600 judge calls, 112 generations, ~450 embedding requests | the last run replayed 770 cells and re-paid 168 |

**The one dollar figure this repo states is an estimate, and is labelled one.** `config.py` and
`evaluation/cache.py` size a full six-arm judged run at roughly **$1.28** on the default judge
against roughly **$0.57** on the answering model — the planning figures from the evaluation
ticket, used to argue that the judge deserves its own stronger model. They are **not a
measurement**: no run wrote them, no artifact carries them, and nothing binds them. Treat them as
the right order of magnitude for "what this evaluation costs" — about a dollar — and nothing
finer.

`evaluate.py` is the only resumable one: every paid cell is content-addressed under
`data/eval-cache` (gitignored), so a run killed by a 429 pays for the cell it died inside and
nothing else, and a second full run costs nothing. `--stage` means *replay the rest from that
cache*, not *skip it*.

## 6.4 Running everything

```bash
# hermetic — what CI runs
uv run ruff check . && uv run ruff format --check .
uv run pytest

# the app
uv run streamlit run app/Home.py

# non-hermetic: the network, and four of the five spend money. Run from the repo root.
uv run python scripts/ingest_filings.py                # full Universe: EDGAR + paid embeddings
uv run python scripts/ingest_filings.py --dry-run       # fetch + gate only; no key, no writes
uv run python scripts/ingest_filings.py --tickers AAPL  # one or more companies
uv run python scripts/retrieval_smoke.py                # 5 sanity queries over the ingested KB
uv run python scripts/security_suite.py                 # the gate against the committed corpus
uv run python scripts/security_suite.py --gate-only     # layers 1-4 only: no embeddings, no agent
uv run python scripts/evaluate.py                       # every stage, every arm, 28 golden rows
uv run python scripts/evaluate.py --stage judge          # re-judge only; replay the rest from cache
```

Every one of them writes its own evidence file, and none of those files is ever hand-authored.
Four rules govern them, and each exists because its absence caused a real problem:

- **A partial run says so.** `--gate-only`, `--stage`, `--rows` and `--no-ablations` each render a
  **PARTIAL RUN** banner naming what did not execute. Never commit a partial run as a whole one.
- **`evaluate.py` refuses to start with `FINBRIEF_LOG_FILE` unset**, because the latency half of
  ADR-0005's dominance test is measured from that log and discovering the sink was off after a
  30-minute paid run means re-running the whole thing. `--allow-missing-sink` proceeds, and the
  artifact then reports those figures as absent.
- **A warm run has nothing to time.** Latency comes from this run's own window of the sink, so a
  fully replayed run's window is empty and the artifact says "not measured, and therefore not met"
  rather than serving the previous run's median.
- **`ingest_filings.py` rewrites `ingest-report.md` only on a full-Universe run**, because a
  `--tickers` or `--dry-run` run cannot speak to what the whole collection holds, and refuses to
  re-render the hand-verification checklist from a subset at all — that would delete fifty-six
  ticks and every note attached to them.

The five committed evidence files:

| file | what it is |
|---|---|
| [`ingest-report.md`](./docs/verification/ingest-report.md) | the section-detection gate table plus what the collection holds, read back from the persisted store |
| [`section-starts.md`](./docs/verification/section-starts.md) | ADR-0007's hand-verification checklist. The only hand-edits it tolerates are ticking a box and adding a note — the generator re-parses it to carry both forward |
| [`retrieval-smoke.md`](./docs/verification/retrieval-smoke.md) | five wiring queries over `retrieve()`. **Not an evaluation** — no buckets, no ground truth, no baseline, so no number in it may be cited as a quality claim. It leads with that, because a file of distances in a repo is read as an evaluation unless it says otherwise |
| [`security-gate.md`](./docs/verification/security-gate.md) | every corpus case, which layer stopped it, the false-positive control, both latency medians against **both** budgets, and each planted payload's answer. **Pass/fail, not an evaluation** |
| [`evaluation.md`](./docs/verification/evaluation.md) | **the measurement artifact of record.** The per-bucket A/B over six arms, all four RAGAs metrics, both pre-registered decisions with their verdicts, the power audit, latency and token spend, the tool-calling eval, and the four deferred measurements. Quote a quality number from here and from nowhere else |

Never invoke any of the five from a test.
