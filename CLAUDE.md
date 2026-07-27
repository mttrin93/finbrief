# FinBrief

Domain-specialised RAG assistant for equity research. The plan lives in `PLAN.md`; the
Tier-1 spec (user stories and the six testing seams) in `docs/spec/finbrief.md`; domain
glossary in `CONTEXT.md`; design decisions in `docs/adr/`. Run evidence lives in
`docs/verification/`: every file there is **generated**, never hand-authored.
`ingest-report.md` is rewritten by every full-Universe `scripts/ingest_filings.py` run;
`retrieval-smoke.md` by every `scripts/retrieval_smoke.py` run (a wiring check on
`retrieve()` — *not* an evaluation, and nothing in it may be cited as a quality claim; the
measurement artifact of record is ADR-0002's golden set, ticket T9/#4); and
`section-starts.md` is ADR-0007's hand-verification checklist — `ingestion/reporting.py`
re-parses it to carry ticks and hand-written notes forward, so the only hand-edits it
tolerates are ticking a box and adding a note.

## Commands

```bash
uv sync
uv run ruff check . && uv run ruff format --check .   # line length 96; E,F,I,UP,B,SIM
uv run pytest
uv run streamlit run app/Home.py
```

CI runs exactly the lint and test commands above (`.github/workflows/ci.yml`).

Building the knowledge base is a separate, non-hermetic entry point — the one command that
reaches the network and spends money. Run it from the repo root; its report paths are
relative to the working directory.

```bash
uv run python scripts/ingest_filings.py             # full Universe: EDGAR + paid embeddings
uv run python scripts/ingest_filings.py --dry-run   # fetch + gate only; no key, no writes
uv run python scripts/retrieval_smoke.py            # 5 sanity queries over the ingested KB
```

Never invoke any of these from a test.

## Conventions

**Tests are hermetic — no `.env`, no API key, no network.** `tests/conftest.py` patches
`load_dotenv` out, strips the managed env prefixes (including `LANGCHAIN_`/`LANGSMITH_`, so
tracing cannot POST), and clears the `load_env`/`get_settings` caches. Build configuration
with `Settings.from_env({...})` or `monkeypatch.setenv`; never read a real `.env`, and never
add a test dependency that fetches data at import time. Two more mechanisms keep the
no-network half true: tiktoken's cl100k_base table is vendored under
`tests/fixtures/tiktoken/` (conftest points `TIKTOKEN_CACHE_DIR` at it — without that,
`get_encoding` silently downloads it), and the EDGAR fixtures under `tests/fixtures/edgar/`
are recorded, never fetched — refresh them by hand with `scripts/record_edgar_fixtures.py`.

**Single sources of truth.** Respect these or the invariant they protect is gone:

- `config.py` owns every knob, plus `CHUNK_SIZE_CHARS` — never copy the value. The
  ingestion thresholds are the deliberate exception: they are assertions, not knobs, and
  live next to the rule they belong to — `ingestion/gate.py`, `ingestion/model.py`
  (`HEADING_LINE_MAX_CHARS`, `POINTER_MAX_CHARS`, `POINTER_RESIDUE_MAX_WORDS`) and
  `CHUNK_OVERLAP_CHARS` in the chunker. Do not move them into `config.py` or make them
  env-overridable; each is calibrated against measured filings and documented where it sits.
- `ingestion/model.py` owns the shared boundary definitions (`WORD`, `NEXT_ITEM_MARKERS`,
  `item_heading`/`section_start`) — a second copy lets a repair and the gate disagree.
- `llm.py` is the only chat-model constructor; `retrieval/embeddings.py` the only
  embeddings constructor, and ingest and query must share it (a drifting model degrades
  retrieval to noise with no error).
- `retrieval/vectorstore.py` is the only place the `filings` collection is opened,
  written, or read (`FILINGS_COLLECTION`).
- All structured logging goes through `log_event` — one JSON object per line, and never a
  secret in `fields`.

## Agent skills

### Issue tracker

Issues and PRDs live as GitHub issues in `TuringCollegeSubmissions/mrinal-AE.AFA.3.5`, via
the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, label strings equal to their names. See
`docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
