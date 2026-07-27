# FinBrief — Financial Research Assistant

A domain-specialised RAG chatbot for equity research. Target user: a junior analyst
who needs a grounded, source-cited company snapshot before an earnings call —
business overview, risk factors, current valuation, and recent news — in minutes.

> **Status:** baseline vector RAG over the knowledge base. A question retrieves from the
> Chroma `filings` collection and comes back as a grounded answer carrying inline `[n]`
> citations, each resolving to a sources panel entry with that chunk's ticker, Section,
> fiscal year and accession — so a claim can be checked against the filing that made it.
> The KB holds curated 10-K Sections for all fifteen Universe companies, fetched from
> EDGAR, gated, chunked, and persisted (see *Building the knowledge base* below).
>
> Retrieval is **vector-only**, deliberately: the pre-registered shipping default is
> `hybrid + translation` (ADR-0005) and it arrives in Phase 4, so until then the app's
> sidebar states both what is configured and what actually ran rather than letting a
> configured strategy be read as a measured one. The agent loop, the finance and news
> tools, and the security gate are the phases that follow. The three files under
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

Grounded in Items 1, 1A, 7 and 7A of the latest annual 10-K on file for each of the 15
companies in FinBrief's Universe — **54 of 60** company × Section pairs. The app states this
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
  where the persisted collections live — ingest and app must agree on it.
- Retrieval switches. `FINBRIEF_RETRIEVAL_K` is read by the answer path today.
  `FINBRIEF_RETRIEVAL_STRATEGY` and `FINBRIEF_QUERY_TRANSLATION` resolve from the
  environment but are deliberately **not** honoured by it yet: their defaults are the
  pre-registered shipping configuration — hybrid + translation, fixed before any A/B data
  exists (ADR-0005) — and serving vector results under a hybrid label would report a
  strategy that never ran, so `retrieve()` raises on `hybrid` and the sidebar names the gap
  between what is configured and what answered. Phase 4 implements it and closes the gap.
  `FINBRIEF_MAX_SUB_QUERIES` is capped at 3, a ceiling rather than a default (ADR-0004's
  latency budget).
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
