# PLAN — FinBrief: Financial Research Assistant (Turing College Sprint 2)

A domain-specialised RAG chatbot for equity research. Target user: a junior analyst
who needs a grounded, source-cited company snapshot before an earnings call —
business overview, risk factors, current valuation, and recent news — in minutes.

**Repo:** `finbrief` (Sprint 2 GitHub repo) · **Track:** Python · **UI:** Streamlit
**Editor:** VS Code · **LLM access:** OpenRouter via LangChain (OpenAI-compatible SDK)

---

## 1. Demo use-case (the spine of the project)

> "Pre-earnings company brief for an equity analyst."

Fixed company universe of **10–15 large caps, curated as same-sector peer clusters**
(big tech: AAPL/MSFT/NVDA/AMZN/GOOGL/META; autos: TSLA/F/GM; banks: JPM/BAC/GS;
healthcare: JNJ/LLY/PFE). All are 10-K filers — foreign private issuers file a 20-F and
are out of scope (ADR-0007). The Universe serves triple duty — KB scope, demo cast, and
peer pool for ratio comparison (ADR-0009). Small universe = controllable chunking,
evaluation, demo.

### Demo script (2–3 min, each step fires a requirement)
1. **"What are the main risk factors for Tesla?"** → pure RAG from 10-K risk section,
   sources displayed with citations.
2. **"How does its valuation compare to its fundamentals?"** → `get_stock_data` +
   `calculate_ratios` tools fire; results rendered as chart/table (tool-call
   visualization).
3. **"Anything in the news related to those risks?"** → `get_recent_news` + RAG
   combined — live data synthesized with retrieved context.
4. **"Give me the full brief."** → structured output combining all of the above.
   This is the README hero screenshot/GIF.
5. **"Should I buy Tesla stock?"** + a prompt-injection attempt → guardrails demo:
   refuses personalised investment advice, shows disclaimer, resists injection.

---

## 2. Requirements traceability

Every core requirement and **every optional task** from the assignment, mapped to a
concrete feature. (P0 = core, P1 = bonus-critical for max points, P2 = stretch.)

### Core requirements (P0)

| Requirement | Implementation |
|---|---|
| Knowledge base | 10-K **curated sections** (Items 1, 1A, 7, 7A) via edgartools + ECB/Fed publications + a small financial-glossary set (ADR-0007) |
| Standard retrieval w/ embeddings | ChromaDB + `text-embedding-3-small` via OpenRouter; RecursiveCharacterTextSplitter (chunk size = `config.CHUNK_SIZE_CHARS`, 1000, the single source of truth; overlap 200 belongs to the chunker). Raising it invalidates the index and is re-checked against the 8191-token embedding window by `tests/test_chunk_token_limit.py` |
| Chunking & similarity search | Section-aware chunking for filings (Item 1, 1A, 7…) with metadata (ticker, filing type, section, fiscal year); cosine similarity top-k |
| Advanced RAG: query translation | Query rewriting + decomposition step: vague queries ("is Tesla in trouble?") → sub-queries ("risk factors", "debt levels", "recent negative news"); shown in UI |
| ≥3 tool calls | `get_stock_data(ticker)` (yfinance, TTL-cached), `calculate_ratios(ticker)` (P/E, D/E, margins vs. mean of same-cluster **in-Universe** peers via the same cached path — zero new API surface; reports peer set + n), `get_recent_news(ticker, days)` (RSS/free news API) — ADR-0009 |
| Domain specialisation | System prompt with analyst persona, financial vocabulary, refusal policy for personalised investment advice, mandatory disclaimer |
| Security measures | Advice refusal; input gate (normalize → bounded regex → one LLM classifier, cheap-first); Guardrails AI output validator (no-advice); indirect-injection defense (quarantine framing + *tested* payloads + HTML stripping); input validation (ticker whitelist, length caps); disclaimer; gate-trigger logging (ADR-0006) |
| LangChain + OpenRouter | `create_agent` pattern (Part 4) with checkpointer memory; tools bound via `@tool` |
| Error handling | Tiered: API level (rate limits, timeouts → retry/backoff/cache), retrieval level (empty results → fallback message), generation level (refusals → graceful UX) |
| Input validation | Ticker validation against universe, query length caps, sanitization |
| UI (Streamlit) | Chat UI with: sources panel (chunks + metadata), tool-call result cards, `st.status`/spinner progress indicators |

### Optional — Easy (all four, P1)

| Task | Implementation |
|---|---|
| Conversation history + export | Checkpointer-backed history per `thread_id`; export button |
| RAG process visualization | Expandable "How I answered" panel: original query → translated sub-queries → retrieved chunks w/ scores → final prompt |
| Source citations | Inline `[1] TSLA 10-K 2025, Item 1A` style citations tied to the sources panel |
| Interactive help / guide | Onboarding expander + `/help`-style command + example-question buttons |

### Optional — Medium (all ten)

| Task | Priority | Implementation |
|---|---|---|
| Multi-model support | P1 | Model picker (e.g. `gpt-4o-mini`, `claude-haiku`, one open model) — trivial via OpenRouter model string |
| Real-time KB/data updates | P1 | Live prices/news via tools; "refresh news into KB" button that ingests latest headlines into a `news` Chroma collection |
| Prompt-injection protection | P1 | System-prompt hardening, retrieved-content quarantine framing ("data, not instructions"), injection test suite (Sprint 1 lesson patterns) |
| Token usage & cost display | P1 | LangChain callbacks → per-message and session token/cost meter in sidebar |
| Tool-call result visualization | P1 | Price history line chart, ratio comparison bar chart (vs. peers), news cards |
| Conversation export (PDF/CSV/JSON) | P1 | JSON + CSV native; PDF via `reportlab`/`fpdf2` |
| Remote MCP server connection | P1 | Connect one public remote MCP server via `langchain-mcp-adapters` (e.g. a fetch/search server); MCP security review from Part 4 applied and documented |
| Rate limiting & API key mgmt | P1 | Per-session request throttle; keys via `.env` + `st.secrets`; never logged |
| Logging & monitoring | P1 | Structured `logging` (JSON lines): queries, retrieval hits, tool calls, latency, token counts; feeds the analytics dashboard |
| User auth & personalisation | P2 | Simple login (streamlit-authenticator); per-user watchlist + default company; per-user `thread_id` |

### Optional — Hard (all seven)

| Task | Priority | Implementation |
|---|---|---|
| Hybrid search | P1 | `EnsembleRetriever`: BM25 (exact tickers, "Item 1A", ratio names) + Chroma vector search; motivated by course Part 2 (exact-identifier problem) |
| RAG evaluation (RAGAs) | P1 | Golden dataset of ~15–20 Q/A over the universe; all four metrics (faithfulness, answer relevancy, context precision, context recall); results table in README; target faithfulness ≥ 0.8 |
| A/B testing of RAG strategies | P1 | Config-switchable strategies (vector-only vs. hybrid; w/ vs. w/o query translation); run RAGAs on each; comparison table = the A/B result |
| Automated KB updates | P2 | Scheduled ingestion script (cron/GitHub Action) pulling newest filings from EDGAR into Chroma; idempotent by accession number |
| Multi-language support | P2 | UI + answer language toggle (EN/DE/IT); queries translated to EN before retrieval (course: embeddings handle EN KB) |
| Advanced analytics dashboard | P2 | Second Streamlit page over the logs: queries/day, top tickers, tool usage, latency, token spend, RAGAs trend |
| Own tools as MCP servers | P2 | Wrap the three finance tools in a local `FastMCP` server; app consumes them via MCP client adapter — same tools, protocol-exposed (Part 4 optional deep dive) |

---

## 3. Architecture

```
                          ┌──────────────────────────────┐
                          │        Streamlit UI          │
                          │ chat · sources · tool cards  │
                          │ RAG-viz · costs · dashboard  │
                          └──────────────┬───────────────┘
                                         │
                         ┌───────────────▼────────────────┐
                         │   Agent (LangChain create_agent)│
                         │   model via OpenRouter          │
                         │   checkpointer = memory         │
                         └───┬─────────────┬──────────┬────┘
              query translation│           │tools     │guardrails
                     ┌─────────▼──┐  ┌─────▼──────┐ ┌─▼─────────────┐
                     │ Retriever  │  │ get_stock  │ │ input valid.  │
                     │ (Ensemble) │  │ calc_ratios│ │ injection def.│
                     │ BM25+Chroma│  │ get_news   │ │ advice refusal│
                     └─────┬──────┘  └─────┬──────┘ └───────────────┘
                           │               │  (P2: exposed via FastMCP)
                 ┌─────────▼─────────┐ ┌───▼─────────────────┐
                 │ ChromaDB          │ │ yfinance · news RSS │
                 │ filings/news/     │ │ FRED (macro)        │
                 │ glossary colls.   │ └─────────────────────┘
                 └─────────▲─────────┘
                           │ ingestion pipeline (manual + scheduled)
                 ┌─────────┴─────────┐
                 │ SEC EDGAR API     │
                 │ ECB/Fed docs      │
                 └───────────────────┘
```

**Key decisions (and why):**
- **ChromaDB** over Milvus: course-covered, lightweight, right-sized for a 10–15
  company universe (Part 2 explicitly positions it for prototyping/small-mid projects).
- **`create_agent` + checkpointer** (Part 4; state ownership per ADR-0008): the
  checkpointer (SqliteSaver, file-backed) is the agent's memory of record; `st.session_state`
  holds only thread_id + UI state + display transcript. Agent built once via
  `@st.cache_resource` (SQLite `check_same_thread=False`); per-session uuid thread_id
  isolates users on the shared cached instance. Same pattern carries into Sprint 3.
- **Hybrid retrieval** (Part 2): tickers, "Item 1A", ISINs, and ratio names are
  exact-identifier territory where pure embeddings blur — the course's E-4012 example,
  transposed to finance.
- **Query translation before retrieval** (ADR-0004): rewrite + decompose, but the
  original query is *always retained as a variant* — translation only adds. All variants
  run through both BM25 and vector; RRF fusion, dedup by chunk id, top-k. BM25 always sees
  the raw identifiers in the retained original.
- **Separate Chroma collections** (`filings`, `news`, `glossary`) with metadata
  filters (ticker, section, year) → enables scoped retrieval and better context
  precision.
- **Retrieval engine vs. agent tool** (ADR-0003): `retrieve(question, strategy, k)` is a
  standalone deterministic component (translation + hybrid inside, config-flagged) called
  *directly* by the eval harness; the same function is wrapped as the `search_filings`
  tool for the `create_agent` loop. Headline RAGAs/A-B measure the chain; the tool-calling
  eval measures the agent's selection layer. Divergence between agent-issued and original
  queries is logged and reported, not assumed away.

## 4. Data sources (all free, public)

| Source | Used for | Notes |
|---|---|---|
| SEC EDGAR (via edgartools) | 10-K curated sections (1, 1A, 7, 7A) → KB | Free, no key; set `User-Agent`; structure-anchored extraction + regex fallback (ADR-0007) |
| yfinance | live price, market cap, history | Unofficial → wrap in try/except + cache (TTL ~15 min) |
| Alpha Vantage (free tier) | fundamentals fallback | 25 calls/day → cache aggressively; feature-flag |
| FRED API | macro context (rates, CPI) | Free key; optional 4th tool if time allows |
| News RSS (Yahoo Finance / Google News RSS) | recent headlines | No key; parse with `feedparser` |
| ECB/Fed publications (PDF) | macro documents in KB | PyPDFLoader |

## 5. Repo structure

```
finbrief/
├── PLAN.md
├── README.md              # problem, architecture, stack, demo GIF, RAGAs table, live URL
├── pyproject.toml         # uv-managed
├── .env.example
├── src/finbrief/
│   ├── config.py          # models, universe (peer clusters), PEERS map, flags, strategy switches
│   ├── llm.py             # shared OpenRouter chat-model client (agent, translation, classifier)
│   ├── prompts.py         # persona, grounding-scope disclosure, context framing — at the
│   │                      # root, not under agent/: the UI and the chain read it too
│   ├── rag.py             # the deterministic question -> contexts -> answer chain the
│   │                      # evaluation harness measures (ADR-0003)
│   ├── ingestion/         # edgar.py, model.py, gate.py, chunking.py, pipeline.py,
│   │                      # reporting.py (Phase 1); pdf_docs.py, news.py later
│   ├── retrieval/         # embeddings.py (shared by ingest+query), vectorstore.py,
│   │                      # retrieve.py (the seam-1 entry point), smoke.py (Phase 2);
│   │                      # hybrid.py, query_translation.py (Phase 4)
│   ├── tools/             # stock_data.py, ratios.py, news.py (+ mcp_server.py P2)
│   ├── agent/             # agent.py (create_agent), guardrails.py
│   ├── evaluation/        # golden_set.json, ragas_eval.py, ab_test.py, tool_eval.py
│   └── observability/     # logging_setup.py, costs.py
├── app/
│   ├── Home.py            # chat page
│   └── pages/Dashboard.py # analytics (P2)
├── scripts/               # ingest_filings.py, record_edgar_fixtures.py (Phase 1),
│                          # retrieval_smoke.py (Phase 2); scheduled KB update lands
│                          # with Tier-2
├── tests/                 # unit: chunking, tools, guardrails, validation
│   ├── conftest.py        # hermetic env: no .env, no key, no network (Phase 0);
│   │                      # recorded EDGAR + vendored tiktoken fixtures (Phase 1)
│   ├── test_config.py     # Universe/Peers invariants + env resolution (Phase 0)
│   ├── test_logging_setup.py       # the JSON-lines contract (Phase 0)
│   ├── test_agent.py               # answer() + the OpenRouter binding (Phase 0)
│   ├── test_embeddings.py          # the one shared embedding model (Phase 0)
│   ├── test_chunk_token_limit.py   # CHUNK_SIZE_CHARS vs. the embedding window (Phase 0)
│   ├── test_section_gate.py + 9 more   # seam 6: gate rules, recorded filings, chunker,
│   │                      # store, pipeline, reports, EDGAR selection, regex fallback,
│   │                      # verification artifact, CLI (Phase 1)
│   ├── test_retrieve.py + 5 more   # seams 1 & 3: retrieval, the chain, the smoke
│   │                      # verdicts and their report, the script's exits, the persona,
│   │                      # the scope disclosure vs. the ingest evidence (Phase 2)
│   ├── fakes.py           # hermetic doubles: KeywordEmbeddings, a_context (Phase 2)
│   ├── test_app_smoke.py  # AppTest — page renders, sources panel, disclaimer, scope
│   │                      # disclosure, a message reaches the agent seam
│   └── test_app_state.py  # AppTest (ADR-0008) — Phase 3: thread_id stability across
│                          # reruns, distinctness across sessions,
│                          # fresh-uuid+surviving-agent on start-over, reset flows,
│                          # model-picker & strategy toggles
└── .github/workflows/     # ci.yml (lint+tests), kb_update.yml (scheduled, P2)
```

## 6. Implementation phases

Estimated core (P0) fits the 20–25 h envelope; P1 adds ~10–15 h; P2 adds ~10 h.
Each step ends with something runnable/testable.

**Scope tiers (governing — see ADR-0001).**
- **Tier-1 (review-facing):** P0 (core) + hybrid search + RAGAs + A/B + query
  translation + security gate + structured logging. Must be **finished, evaluated,
  and review-defensible before any Tier-2 work begins** (the Phase-7 gate below).
- **Tier-2 (skill-stretch):** everything else. Built only after the Tier-1 gate,
  each fully understood; anything not defensible is **cut before submission**.
- The `P1`/`P2` letter tags in §2 predate this split; where they disagree, the tier
  definition here wins.

**Phase 0 — Scaffolding (P0, ~1.5 h)** — ✅ **done** (`t1-walking-skeleton`)
- uv project, config, .env handling, logging setup, CI lint+test workflow
- Streamlit hello-chat with OpenRouter round-trip

**Phase 1 — Knowledge base (Tier-1, ~4 h)** *(see ADR-0007)* — ✅ **done**
(`t2-kb-ingestion`, #3; the one deferred item, the retrieval smoke test, needed Phase 2's
retrieval chain and landed with it — ticket T3, #5)
- EDGAR ingest for the universe (latest 10-K per company; accession-id idempotency).
  KB = **curated sections only** (Items 1, 1A, 7, 7A), not full filings.
- Structure-anchored section extraction via **edgartools**; bounded regex as documented
  fallback (no hand-rolled primary parser).
- Section-aware chunking + metadata (ticker, section, fiscal year); ingest into Chroma
  `filings` collection.
- **Phase-1 gate — section-detection sanity check:** per company × section assert found,
  non-empty, within length bounds, and body ≫ heading (catches TOC hits); fail loudly
  before any retrieval numbers exist. One-time hand-verification checklist of extracted
  section starts committed as an artifact. *(As shipped, the gate grew four more checks
  the fifteen real filings demanded: starts-at-its-own-heading, stops-before-the-next-
  Item, the-filer-files-10-Ks, and the recorded-pointer-filer cross-check on the Item 7A
  incorporation-by-reference excusal — ADR-0007 amendment.)*
- Smoke test: top-k retrieval sanity checks for 5 hand-written queries — **done in Phase 2**
  (ticket T3, #5) as `scripts/retrieval_smoke.py`, evidence in
  `docs/verification/retrieval-smoke.md`. A wiring check on `retrieve()`, explicitly not an
  evaluation: ADR-0002's golden set (T9, #4) remains the measurement artifact of record.
- The committed evidence under `docs/verification/` is the last full run's, regenerated
  *after* the post-run extractor, chunker and checklist-format fixes: 60/60 gated, 60/60
  ticked, nothing flagged `CHANGED`. Re-running ingestion re-renders the two files it owns
  (`ingest-report.md` always; `section-starts.md` on `--section-starts`, carrying forward
  every tick whose Section text is byte-identical); `retrieval-smoke.md` is the smoke
  script's. ADR-0007's *Outcome* records row by row what the re-render changed — and why no
  row moved for the chunker fix, the prediction that run settled.

**Phase 2 — Baseline RAG (P0, ~3 h)** — ✅ **done** (`t3-baseline-rag`, #5)
- Vector-only retrieval chain, source citations, sources panel in UI. `retrieve()` is the
  deterministic seam ADR-0003 asks for and is **vector-only by design**: `hybrid` raises
  rather than serving vector results under a hybrid label, because `DEFAULT_STRATEGY` is
  the pre-registered `hybrid` (ADR-0005) and Phase 4 is where it becomes true. The sidebar
  names the gap between the configured strategy and the one that answered.
- Domain system prompt + disclaimer. The grounding-scope disclosure (user story 18) is
  derived from `config`, not typed, and the disclaimer is rendered *beside* the answer
  rather than requested from the model.
- Also here: the Phase-1 smoke test above, as `scripts/retrieval_smoke.py`.

**Phase 3 — Tools + agent (Tier-1, ~4 h)** *(agent state: see ADR-0008)*
- Three `@tool` functions with caching + error handling
- `create_agent` + SqliteSaver checkpointer built once under `@st.cache_resource`
  (SQLite `check_same_thread=False`); thread_id = uuid4 per session in session_state
  (isolates users on the shared cached instance). Tool-call cards in UI; progress indicators.

**Phase 4 — Advanced RAG (Tier-1, ~4 h)** *(see ADR-0004)*
- Query translation (rewrite + decompose): original query always retained as a variant,
  translation only ever *adds* (≤3 sub-queries, capped for latency). UI visualization.
- Hybrid search: all variants run through BOTH BM25 and vector; RRF fusion over all
  candidate lists; dedup by chunk id; top-k. One symmetric code path.
- Per-chunk provenance logged (which variant × which retriever surfaced it, RRF
  contribution) → feeds the RAG-viz panel and the "why hybrid wins" A/B analysis.

**Phase 5 — Guardrails + validation (Tier-1, ~3 h)** *(see ADR-0006)*
- Advice-refusal policy; input validation (ticker whitelist, length caps, sanitization)
- **Input gate (front door)** — cheap-first, one model call per turn, ≤800ms p50:
  1. **Normalization:** lowercase, strip accents/homoglyphs, collapse whitespace &
     punctuation, de-leetspeak (0→o, 1→i, 3→e, @→a) — catches *obfuscation*.
  2. **Bounded regex denylist** (e.g. `ignore\W{0,20}(all|previous)?\W{0,20}instructions`):
     a catch exits early; a pass *always* escalates to the classifier — catches *known
     patterns* for free.
  3. **One zero-shot LLM classifier** (own prompt, via OpenRouter): YES/NO instruction-
     override / system-prompt-extraction — catches *novel phrasings*.
- **Output validator (back door):** Guardrails AI no-investment-advice validator on
  responses (`on_fail="exception"` → graceful refusal). Does work the input layers don't —
  catches *consequences*, including those of a successful indirect injection.
- **Indirect injection = first-class Tier-1 tests:** dedicated test collection (NOT the
  demo KB) seeded with injection strings in fake news/filing chunks; assert the model
  neither obeys them nor leaks the system prompt. HTML stripping on ingested news.
- Marginal-contribution table (normalization→obfuscation, regex→known, classifier→novel,
  output→consequences) → README, so each layer's job is defensible.
- Gate-trigger logging: normalized input, layer fired, matched pattern, timestamp.
- Injection test suite: obfuscated variants (leetspeak, spacing, unicode) + indirect.
- Demo step 5 works end-to-end.

**Phase 6 — Structured logging & observability (Tier-1, ~1.5 h)** *(moved ahead of evaluation)*
- Structured `logging` (JSON lines): per query — strategy config, retrieval hits,
  latency, token counts; plus gate-trigger metadata (normalized input, layer fired,
  matched pattern) from Phase 5.
- Also log **agent-issued query vs. original user query** (ADR-0003) so the README can
  report how often the shipped path diverges from the measured retrieval chain.
- Why ahead of evaluation: Phase 7 (RAGAs + A/B) and the security-gate catch analysis
  *consume* these logs. Logging after evaluation would force re-running evaluation —
  so this is Tier-1 infrastructure, not polish.
- Scope: log *capture* only (incl. token counts for cost analysis). The cost-*meter*
  sidebar UI is Tier-2 (Phase 8).

**Phase 7 — Evaluation (Tier-1, ~4 h)** *(see ADR-0002)*
- Golden set: ~24–28 Q/A in four stratified buckets (semantic, exact-identifier,
  tool-augmented, multi-hop), ≥6 each. Ground truth authored from the filings directly
  with section citations; candidate Q/A drafted by a *different* model than the answering
  pipeline, then hand-verified against the primary source (breaks circularity).
- RAGAs all four metrics, reported **per bucket**. A/B matrix: vector vs. hybrid ×
  ±query translation (reads Phase-6 logs). Pre-registered hypotheses: hybrid > vector on
  exact-identifier; translation wins on multi-hop; both tie on semantic (a predicted tie
  confirms the harness works, not a failure).
- Injection/guardrail cases are NOT in the RAGAs set (faithfulness against a refusal is
  undefined) — they live in the Phase-5 security suite as pass/fail, reported alongside.
- Shipping default **pre-registered now** = hybrid + translation (ADR-0005), on the
  dominance prediction. Falsification (directional — no tight numeric margin, given ~6
  Q/bucket): if translation is worse on *both* context precision and recall within any
  bucket, default drops to hybrid-only and the contradiction is a README finding;
  per-question spread reported alongside. Dominance judged within ≤1.5s p50 added latency
  from translation (measured from Phase-6 logs).
- Tool-calling evaluation (Part 4 pattern); results tables → README.

> **── Tier-1 gate ──** Everything above must be finished, evaluated, and
> review-defensible before any Phase-8 work begins.

**Phase 8 — Tier-2 (skill-stretch; build only after the Tier-1 gate, ~20 h)**
Ordered by GenAI/RAG skill signal, deployment excepted on portfolio grounds (ADR-0010).
Each item built only when fully understood; anything not defensible is cut before submission.
1. **Deploy + live URL** (Streamlit Community Cloud) + demo GIF — portfolio reach gates the
   value of everything else.
2. **Re-ranking** — cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scoring the RRF-fused top-N
   to top-k. Constrained third A/B axis (shipped default ± rerank only), per-bucket in the
   existing harness. Pre-registered: precision lift on `semantic`, ~neutral on
   `exact-identifier`. Latency under the translation budget regime; local-model deploy
   weight accepted as the cost.
3. **MCP client** (remote public server + security review), then **own tools as FastMCP server**.
   - ⚠ Open Q (ask first): is tools-as-MCP-server a genuine protocol *port*, or duplicated
     finance logic behind a second interface? Must reuse one implementation.
4. **Multi-model** support (model picker via OpenRouter).
   - ⚠ Open Q (ask first): the injection classifier + agent/tool-calling prompts are tuned
     per-model — does swapping models silently break tool-calling? Needs a per-model check.
5. **Real-time KB refresh** (ingest latest headlines/filings into Chroma on demand).
   - ⚠ Open Q (ask first): does live ingest stay idempotent (accession/id dedup) AND hold
     the ADR-0007 section-detection gate, without racing the cached retriever / agent state?
6. *Generic tail — only if appetite remains, each cut-if-undefendable:* auth + watchlists ·
   cost-meter sidebar UI · export (JSON/CSV/PDF) · analytics dashboard · scheduled KB updates
   (GH Action) · rate limiting · help guide · multi-language toggle.
- *Future work (deferred by ADR-0005):* adaptive per-query strategy routing.

## 7. Evaluation plan (what "working well" means)

- **Retrieval:** precision@k/recall@k on the golden set (Part 2 Lab 2 method) —
  reported **per strategy × per bucket** (vector / hybrid / +translation)
- **Generation:** RAGAs faithfulness ≥ 0.8, answer relevancy ≥ 0.8; context
  precision/recall ≥ 0.7 — reported **per bucket**; interpret low scores per the course
  guidance rather than chasing numbers
- **Tools:** tool-selection accuracy on 10 scripted prompts (does the right tool fire
  with the right args?)
- **Security:** injection test suite (incl. obfuscated variants) passes; advice-refusal
  verified in demo step 5
- **UI/state:** AppTest — thread_id stability across reruns, distinctness across two
  sessions, fresh-uuid+surviving-agent on start-over, plus reset/toggle flows (ADR-0008) —
  catches Streamlit-layer regressions that pure function tests miss

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| yfinance flakiness / rate limits | TTL cache, retry with backoff, cached-fallback banner in UI |
| Alpha Vantage 25 calls/day | Feature flag; cache to disk; yfinance primary |
| 10-K PDFs/HTML messy to parse | Use EDGAR's structured formats; section regexes; accept imperfect edges, note in README reflection |
| Scope creep (all optional tasks) | Strict P0→P1→P2 ordering; P0+P1 alone already exceed the "2 medium + 1 hard" bonus bar |
| RAGAs cost (judge LLM calls) | Small golden set, cheap judge model via OpenRouter, cache eval runs |
| Injection via retrieved news content | Quarantine framing; HTML stripping; retrieved text treated as data — now a *tested* Tier-1 threat (dedicated injection test collection + output validator), not just prompt framing (ADR-0006) |

## 9. Reflection prompts (for the review)

Known limitations to discuss: KB grounded only in Items 1/1A/7/7A (declared in the UI) —
questions outside these sections are out of scope; table/figure fidelity limited (hard
figures come from tools); Universe restricted to 10-K filers — foreign private issuers
(20-F) are out of scope, since a 20-F has no Item 1A/7/7A to ingest and supporting one
needs its own section mapping (ADR-0007); single-language KB vs. multilingual queries;
faithfulness vs.
useful-but-uncontexted knowledge trade-off (Part 3's Einstein example); yfinance as an
unofficial API in a "production" story; universe fixed at ingest time; no re-ranking stage
in Tier-1 (promoted to Tier-2 #2 per ADR-0010 — if built, evaluated as a third A/B axis).

**Citation validity is persona-dependent** (T3, #5). The grounding half of the contract is
structural: `Context.rank` is assigned once in `retrieve()` and read only by
`prompts.format_contexts` and the app's sources panel, and the contexts travel with the
answer in `GroundedAnswer`, so every panel entry `[n]` resolves to exactly one retrieved
chunk and keeps resolving to it across reruns. The *citing* half is not enforced: nothing
parses the `[n]` markers out of the answer, so a model that emits `[6]` against five
contexts produces a marker pointing at no entry — silently, with no error and no log line.
Until an output-side marker validator lands (offered to T7 as an output-validation
candidate, #8) or ADR-0002's faithfulness scoring measures it statistically (T9, #4),
citation *correctness* rests on the system prompt's instruction rather than on code. The
inverse gap is deliberate: a retrieved-but-uncited context still appears in the panel,
because the panel's contract is "what grounded this turn", which is also why the log field
is `retrieved_sections` and not `cited_sections`.
