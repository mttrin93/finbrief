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

The non-hermetic entry points — the only commands that reach the network, and two of the
three spend money. Run them from the repo root; their report paths are relative to the
working directory. `retrieval_smoke.py` reads the collection ingest built and embeds its five
queries with the same paid model, so it needs a key too: a query embedded by a different
model retrieves noise with no error.

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
add a test dependency that fetches data at import time. Three more mechanisms keep the
no-network half true: tiktoken's cl100k_base table is vendored under
`tests/fixtures/tiktoken/` (conftest points `TIKTOKEN_CACHE_DIR` at it — without that,
`get_encoding` silently downloads it); the EDGAR fixtures under `tests/fixtures/edgar/`
are recorded, never fetched — refresh them by hand with `scripts/record_edgar_fixtures.py`;
and retrieval runs against a **real on-disk Chroma with a fake embedding** — `tests/fakes.py`
holds the doubles (`KeywordEmbeddings`, deterministic and lexical, so a test may assert an
order; `a_context`, the shared `Context` builder) and conftest builds `filings_store` /
`empty_filings_store` from them. Never point a test at the ingested `data/chroma`: it is only
searchable by the paid model that wrote it, so a test that reaches for it either needs a key
or asserts against noise.

**Single sources of truth.** Respect these or the invariant they protect is gone:

- `config.py` owns every knob, plus `CHUNK_SIZE_CHARS` — never copy the value. The
  ingestion thresholds are the deliberate exception: they are assertions, not knobs, and
  live next to the rule they belong to — `ingestion/gate.py`, `ingestion/model.py`
  (`HEADING_LINE_MAX_CHARS`, `POINTER_MAX_CHARS`, `POINTER_RESIDUE_MAX_WORDS`) and
  `CHUNK_OVERLAP_CHARS` in the chunker. Do not move them into `config.py` or make them
  env-overridable; each is calibrated against measured filings and documented where it sits.
  `agent/agent.py`'s `MAX_AGENT_STEPS` is exempt on the same grounds — a ceiling on the
  agent loop, next to the loop it guards. **This list is the exception**: a limit not
  enumerated here belongs in `config.py`, or the exemption stops being narrow.
- `ingestion/model.py` owns the shared boundary definitions (`WORD`, `NEXT_ITEM_MARKERS`,
  `item_heading`/`section_start`) — a second copy lets a repair and the gate disagree.
- `llm.py` is the only chat-model constructor; `retrieval/embeddings.py` the only
  embeddings constructor, and ingest and query must share it (a drifting model degrades
  retrieval to noise with no error).
- `retrieval/vectorstore.py` is the only place the `filings` collection is opened,
  written, or read (`FILINGS_COLLECTION`).
- `retrieval/retrieve.py` is the only entry point to the knowledge base. It owns what a
  retrieval *means* — the strategy, the ranking contract, the `Context` shape — and crosses
  Chroma only through `vectorstore.nearest_chunks`; nothing above it opens the collection.
  `hybrid` raises until Phase 4 rather than serving vector results under a hybrid label.
- `prompts.py` owns the persona and the grounding-scope disclosure. The app's caption, the
  sidebar panel, the system prompt and the README all read the same words (`GROUNDING_SCOPE`,
  `GROUNDING_SCOPE_DETAILS`), and every count in them is derived from `config`/`Section`,
  never typed — a scope sentence written twice will disagree with itself, and the disagreeing
  copy is the one on screen. `tests/test_grounding_scope.py` binds the README's prose and the
  committed ingest evidence to the derived values. **A tool description is a prompt**, and
  lives here too if it makes a scope claim: `SEARCH_FILINGS_DESCRIPTION` is `prompts.py`'s,
  not the tool module's, and its Items and Universe count are derived like every other
  (issue #7 review). A prompt-facing rule is also written **once** — the description owns the
  verbatim-query contract *and its exception*, and `AGENT_SYSTEM_PROMPT` points at it rather
  than restating it, because a rule the model reads in two wordings is one it can pick between.
- `rag.answer_question` is the measured chain (ADR-0003) and must stay callable with no agent
  in the way. `GroundedAnswer.text` deliberately carries **no** disclaimer, so every surface
  that renders it owes a `prompts.DISCLAIMER` beside it.
- `agent/citations.py` is the only place a citation number is assigned. `Context.rank` arrives
  from `retrieve()` as 1…k per retrieval; an inline `[n]` has to name one chunk for a whole
  conversation, so the register renumbers a thread's search replies in one sequential pass at
  the `before_model` seam. It cannot be done in the tool — LangGraph hands every call in a step
  the same state and then runs them concurrently, so two searches in one step both number from
  `[1]` and nothing raises (issue #7 review). Anything that numbers sources by their position
  in a list, or offsets ranks anywhere else, reintroduces that collision.
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
