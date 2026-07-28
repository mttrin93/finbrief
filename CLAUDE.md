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
add a test dependency that fetches data at import time. **The no-network half is enforced, not
asserted**: `conftest.py` patches `connect`/`connect_ex`/`create_connection` and **all four
resolver entry points** (`getaddrinfo`, `gethostbyname`, `gethostbyname_ex`, `gethostbyaddr`) at
import time — before collection, which is when an import-time fetch happens — so egress to a
non-loopback host raises `EgressBlocked`; and it patches `curl_cffi.Curl.perform` separately,
because `curl_cffi` binds libcurl and resolves and connects in **C**, touching Python's `socket`
module not at all. That second half is not optional trivia: `yfinance` uses `curl_cffi` whenever
it imports, so without it the one library T5 added was the one uncovered — measured at HTTP 429
with the socket guard installed. The resolver list is four names for the same reason: each is its
own CPython C entry point, so patching `getaddrinfo` alone left `gethostbyname` returning real
addresses (issue #9 review).

**The guard is a denylist over the backends this repo can reach, not a proof**, and it is
described that way deliberately: a new HTTP dependency is a new path, and a guard advertised as
total is how the claim came to be false four times (twice through tiktoken's cache, twice through
this guard's own prose). `tests/test_hermetic_suite.py` carries **one test per backend** for
that reason — an uncovered path shows up there as a live call, which is where the `curl_cffi`
hole was found. Add a networking dependency, add a test there. And **a claim in a comment cannot
fail**: three of the four false claims were prose asserting coverage the code lacked, so every
test in that file exercises the call it is about — the DNS case calls each of the four resolvers
rather than asserting that three route through the fourth, which is precisely the sentence that
was wrong. Four more mechanisms keep it true *without* leaning on
the guard: tiktoken's cl100k_base table is vendored under
`tests/fixtures/tiktoken/` (conftest points `TIKTOKEN_CACHE_DIR` at it — without that,
`get_encoding` silently downloads it); the EDGAR fixtures under `tests/fixtures/edgar/`
are recorded, never fetched — refresh them by hand with `scripts/record_edgar_fixtures.py`;
the market and news fixtures under `tests/fixtures/market/` likewise, via
`scripts/record_market_fixtures.py` (news recorded **raw**, so `feedparser` and the HTML
stripper really run; quotes recorded **parsed** at the `yfinance.Ticker` boundary, which is
therefore the one thing no test covers — the recorder says so);
and retrieval runs against a **real on-disk Chroma with a fake embedding** — `tests/fakes.py`
holds the doubles (`KeywordEmbeddings`, deterministic and lexical, so a test may assert an
order; `a_context`, the shared `Context` builder) and conftest builds `filings_store` /
`empty_filings_store` from them. Never point a test at the ingested `data/chroma`: it is only
searchable by the paid model that wrote it, so a test that reaches for it either needs a key
or asserts against noise.

**A check that cannot fail is the bug class this repo keeps hitting**, and it is worth naming as
one because the instances look unrelated until they are listed: tiktoken's warm cache made a
"hermetic" suite pass locally and egress in CI (twice); the guard's docstring claimed `curl_cffi`
coverage it lacked; a comment claimed `gethostbyname` routed through `getaddrinfo`; and
`MAX_AGENT_STEPS` was pinned by `serial_full_brief < MAX_AGENT_STEPS`, an inequality that 15, 20
and 24 all satisfy. **Prefer a check that exercises the thing over one that describes it, and
prefer an equality over a bound.**

The newest instance is a Streamlit-specific trap, so it is written down rather than rediscovered:
**`AppTest.get("...")` returns `[]` for an element type it does not know, instead of raising.**
`st.bar_chart` and `st.line_chart` both reach the element tree as `vega_lite_chart`, for which
`AppTest` ships no typed accessor — they arrive as `UnknownElement` — so
`assert not app.get("arrow_bar_chart")` passes on a page rendering no charts *and* on a page
rendering ten. Assert against `tests/test_app_smoke.py`'s `charts()` helper, which walks the tree
for `type == "vega_lite_chart"`; it also descends into `st.columns`, which `app.chat_message[n]`
does not, so a chart inside a column is invisible to the obvious lookup as well. A new
`app.get(...)` against an element `AppTest` has no wrapper for is a vacuous assertion by default
(issue #9 review).

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
  written, or read (`FILINGS_COLLECTION`). `build_filings_store` is the constructor;
  `default_filings_store` is the **shared per-process handle** the application reads, and the
  app must go through it — `hybrid.bm25_index` caches against that store *object*, so anything
  that opens a fresh `Chroma` per query rebuilds the whole ~5,800-chunk lexical index for one
  question. Four process-level singletons sit on this path (`default_filings_store`,
  `hybrid.bm25_index`, `retrieve._planner_model`, `quotes.bounded_session` — each keyed on the
  frozen `Settings`, the store, or nothing at all) and every one goes through
  `caching.build_once`, **not** a bare `lru_cache`: with
  parallel tool calls two searches run concurrently, and two callers missing the same cold key
  both construct — which for the Chroma handle is fatal, because chromadb's shared-system
  registry is not reentrant (`AttributeError: 'RustBindingsAPI' object has no attribute
  'bindings'`, from inside the tool node, on the first turn of a cold process). `lru_cache` is
  atomic about its bookkeeping and says nothing about the function it wraps. A **fifth** singleton
  added here without `build_once` is the same crash again — and `build_once`'s lock is one lock for
  the wrapper, not one per key, which is stated there because the docstring first claimed
  otherwise. They **are not cleared by conftest**,
  unlike `load_env`/`get_settings`: a test that
  builds an index clears `bm25_index` itself (see `tests/test_hybrid.py`), and every other test
  injects its collaborator and never reaches them.
- `finance/quotes.py` is the only place yfinance is called and `finance/news.py` the only
  place RSS is read, for the reason `vectorstore.py` is the only place Chroma is opened: a second
  caller is a second cache to miss, and ADR-0009's "peers add zero API surface" is exactly the
  claim that a second caller would void. Units are normalised **once**, at that boundary —
  `debtToEquity` arrives in percentage points (Ford reads `425.544`, i.e. 4.26×) while margins
  arrive as fractions, and a ratio that looks like a percentage is how a brief reports Ford as
  levered 425 times. Every figure is `float | None` and `None` means *not reported*, never zero;
  `finance/ratios.py` excludes an absence from a peer mean and `Unit.format` prints it as words.
  `tools/finance.py` is the wrapper — validation, refusals-as-results, quarantine, cards — and
  decides no number. A card's artifact crosses the checkpoint, so it holds **JSON primitives
  only**: `asdict` keeps tuples *and* enum members, and `json.dumps` will not tell you, because a
  `StrEnum` is a `str`. Assert leaf types, not a round trip.
- `retrieval/retrieve.py` is the only entry point to the knowledge base. It owns what a
  retrieval *means* — the composition, the ranking contract, the `Context` and `Retrieval`
  shapes — and crosses Chroma only through `vectorstore.nearest_chunks`; nothing above it
  opens the collection. The two switches (`strategy`, `translate`) are
  **named by the caller**, never read from config here: the app names them from `Settings`
  and the A/B harness names all four combinations, so a number is never reported against a
  configuration nobody selected. `retrieval/hybrid.py` owns fusion and the BM25 index, and
  makes the *other* crossing — `hybrid.bm25_index` reads the whole corpus through
  `vectorstore.all_chunks`, once per process, because BM25 scores text and needs the corpus
  rather than a neighbourhood of it. Two crossings, both inside `retrieval/`, both through
  `vectorstore.py`. `retrieval/query_translation.py` owns the variants, and **the original
  question is always variant 0** (ADR-0004 — translation only ever adds).
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
  **A prompt that states an enforced limit takes it as an argument** —
  `query_translation_prompt(max_sub_queries)` is a function so the cap the model reads is the
  cap `sub_queries()` keeps. And `prompts.py` may not import from `retrieval/` at runtime:
  `query_translation.py` reads its prompt from here while `retrieve.py` imports *it*, so
  `Context` is a `TYPE_CHECKING`-only import and closing that loop leaves the class undefined.
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
  secret in `fields`, and **never a question or a query variant**: a variant is derived from
  user content and these lines are kept. Counts, lengths and verdicts only — which is why the
  `retrieval` event records a chunk's provenance by *variant index* while `Surfaced` itself
  carries the variant text for the RAG-viz panel to render (ADR-0004 amendment).

**A live conversation's checkpoint outlives a deploy.** Every `from_payload` and every
transcript-row reader tolerates the shape written before the current one: a new field on a
checkpointed payload is optional on read with an **honest absence value** and never a fabricated
number (`Context.fused_score` → `0.0` and `provenance` → `()`; `Retrieval.planned` → `False`;
`Surfaced.distance` → `None`), and a missing artifact is "no provenance", never an error
(`tools/search_filings.py`). Adding a *required* field to such a payload `KeyError`s a thread in
flight on the first rerun after deploy. The corollary matters as much: an absence must not be
reported as a measurement — "we cannot say what this query found" is a different claim from
"this query found nothing" (`app/Home.py`'s `barren_variants`, `distance_label`).

## Agent skills

### Issue tracker

Issues and PRDs live as GitHub issues in `TuringCollegeSubmissions/mrinal-AE.AFA.3.5`, via
the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, label strings equal to their names. See
`docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
