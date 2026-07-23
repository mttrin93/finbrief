# FinBrief — Financial Research Assistant

A domain-specialised RAG chatbot for equity research. Target user: a junior analyst
who needs a grounded, source-cited company snapshot before an earnings call —
business overview, risk factors, current valuation, and recent news — in minutes.

> **Status:** scaffolding. The plan lives in [`PLAN.md`](./PLAN.md); design decisions
> are being sharpened via `grill-with-docs` and will be recorded in `CONTEXT.md` and
> `docs/adr/` before implementation begins.

## Stack

- **Python** · **Streamlit** UI · **LangChain / LangGraph** (`create_agent`)
- **OpenRouter** for LLM access (OpenAI-compatible SDK)
- **ChromaDB** vector store · hybrid retrieval (BM25 + vectors)
- Data: SEC EDGAR filings, yfinance, news RSS, ECB/Fed publications

## Development

```bash
uv sync
cp .env.example .env   # fill in keys
uv run streamlit run app/Home.py
```

_(The runnable hello-chat + OpenRouter round-trip is Phase 0 in `PLAN.md` §6 —
built once the plan is grilled.)_
