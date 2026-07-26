# FinBrief

Domain-specialised RAG assistant for equity research. The plan lives in `PLAN.md`; the
Tier-1 spec (user stories and the six testing seams) in `docs/spec/finbrief.md`; domain
glossary in `CONTEXT.md`; design decisions in `docs/adr/`.

## Commands

```bash
uv sync
uv run ruff check . && uv run ruff format --check .   # line length 96; E,F,I,UP,B,SIM
uv run pytest
uv run streamlit run app/Home.py
```

CI runs exactly the lint and test commands above (`.github/workflows/ci.yml`).

## Conventions

**Tests are hermetic — no `.env`, no API key, no network.** `tests/conftest.py` patches
`load_dotenv` out, strips the managed env prefixes (including `LANGCHAIN_`/`LANGSMITH_`, so
tracing cannot POST), and clears the `load_env`/`get_settings` caches. Build configuration
with `Settings.from_env({...})` or `monkeypatch.setenv`; never read a real `.env`, and never
add a test dependency that fetches data at import time.

**Single sources of truth.** Respect these or the invariant they protect is gone:

- `config.py` owns every knob, plus `CHUNK_SIZE_CHARS` — never copy the value.
- `llm.py` is the only chat-model constructor; `retrieval/embeddings.py` the only
  embeddings constructor, and ingest and query must share it (a drifting model degrades
  retrieval to noise with no error).
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
