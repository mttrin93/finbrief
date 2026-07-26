# FinBrief — Financial Research Assistant

A domain-specialised RAG chatbot for equity research. Target user: a junior analyst
who needs a grounded, source-cited company snapshot before an earnings call —
business overview, risk factors, current valuation, and recent news — in minutes.

> **Status:** walking skeleton + knowledge base. A typed message makes an OpenRouter
> round-trip and the reply renders, and the Phase-1 KB exists: curated 10-K Sections for
> all fifteen Universe companies, fetched from EDGAR, gated, chunked, and persisted to a
> Chroma `filings` collection (see *Building the knowledge base* below). Retrieval,
> tools, and guardrails are the phases that follow. The plan lives in
> [`PLAN.md`](./PLAN.md), the Tier-1 spec in
> [`docs/spec/finbrief.md`](./docs/spec/finbrief.md), the domain language in
> [`CONTEXT.md`](./CONTEXT.md), and the design decisions in [`docs/adr/`](./docs/adr/).

## Stack

- **Python** · **Streamlit** UI · **LangChain / LangGraph** (`create_agent`)
- **OpenRouter** for LLM access (OpenAI-compatible SDK)
- **ChromaDB** vector store · hybrid retrieval (BM25 + vectors)
- Data: SEC EDGAR filings, yfinance, news RSS, ECB/Fed publications

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
- Retrieval switches (`FINBRIEF_RETRIEVAL_STRATEGY`, `FINBRIEF_QUERY_TRANSLATION`,
  `FINBRIEF_RETRIEVAL_K`, `FINBRIEF_MAX_SUB_QUERIES`) are declared now and wired from
  Phase 2. Their defaults are the pre-registered shipping configuration — hybrid +
  translation, fixed before any A/B data exists (ADR-0005) — and `FINBRIEF_MAX_SUB_QUERIES`
  is capped at 3, a ceiling rather than a default (ADR-0004's latency budget).
- `finbrief.*` logs one JSON object per line to stderr at `LOG_LEVEL` (default `INFO`);
  the Phase-7 A/B and security-gate analyses read those lines back.

## Development

```bash
uv sync
cp .env.example .env   # fill in OPENROUTER_API_KEY
uv run streamlit run app/Home.py
```

Lint and test the way CI does:

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

Tests are hermetic — no API key, no `.env`, and no network calls — so they run anywhere.

### Building the knowledge base

```bash
uv run python scripts/ingest_filings.py                  # the whole Universe → data/chroma
uv run python scripts/ingest_filings.py --tickers AAPL   # one company
uv run python scripts/ingest_filings.py --dry-run        # fetch + gate only; no writes,
                                                         # no embedding API, no key needed
```

Requires `SEC_EDGAR_USER_AGENT`; writing (non-dry) runs also need `OPENROUTER_API_KEY`
for embeddings. Re-runs skip filings already ingested (idempotent by accession number)
and evict a company's superseded fiscal years; `--force` re-embeds after a chunker
change. Every run rewrites `docs/verification/ingest-report.md` (the machine evidence of
what the collection holds), and `--section-starts PATH` writes the hand-verification
checklist ADR-0007 requires, carrying forward ticks for unchanged Sections.
