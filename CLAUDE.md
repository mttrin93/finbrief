# FinBrief

Domain-specialised RAG assistant for equity research. The plan lives in `PLAN.md`; the Tier-1
spec (user stories and the six testing seams) in `docs/spec/finbrief.md`; domain glossary in
`CONTEXT.md`; design decisions in `docs/adr/`. Run evidence lives in `docs/verification/`: every
file there is **generated**, never hand-authored. `ingest-report.md` is rewritten by every
full-Universe `scripts/ingest_filings.py` run; `retrieval-smoke.md` by every
`scripts/retrieval_smoke.py` run (a wiring check on `retrieve()` — *not* an evaluation, and
nothing in it may be cited as a quality claim; the measurement artifact of record is
`evaluation.md`); `evaluation.md` by every `scripts/evaluate.py` run — **the measurement artifact
of record**: ADR-0002's golden set (T9/#4) scored by T10's harness (#11), and the only file a
quality number may be quoted from. Like `--gate-only`, a `--stage`, `--rows` or `--no-ablations`
run carries a **PARTIAL RUN** banner and must never be committed as a whole one. `security-gate.md`
by every `scripts/security_suite.py` run
(pass/fail, likewise not an evaluation — and it prints the **pre-registered** latency budget
beside the revised one on purpose, so a reader cannot mistake a revised pre-registration for one
that always held); and `section-starts.md` is ADR-0007's hand-verification checklist —
`ingestion/reporting.py` re-parses it to carry ticks and hand-written notes forward, so the only
hand-edits it tolerates are ticking a box and adding a note.

## Commands

```bash
uv sync
uv run ruff check . && uv run ruff format --check .   # line length 96; E,F,I,UP,B,SIM
uv run pytest
uv run streamlit run app/Home.py
```

CI runs exactly the lint and test commands above (`.github/workflows/ci.yml`).

The non-hermetic entry points — the only commands that reach the network, and four of the
five spend money. Run them from the repo root; their report paths are relative to the
working directory. `retrieval_smoke.py` reads the collection ingest built and embeds its five
queries with the same paid model, so it needs a key too: a query embedded by a different
model retrieves noise with no error.

```bash
uv run python scripts/ingest_filings.py             # full Universe: EDGAR + paid embeddings
uv run python scripts/ingest_filings.py --dry-run   # fetch + gate only; no key, no writes
uv run python scripts/retrieval_smoke.py            # 5 sanity queries over the ingested KB
uv run python scripts/security_suite.py             # the gate against the committed corpus
uv run python scripts/security_suite.py --gate-only # layers 1-4 only: no embeddings, no agent
uv run python scripts/evaluate.py                   # every stage, every arm, 28 golden rows
uv run python scripts/evaluate.py --stage judge     # re-judge only; replay the rest
```

`evaluate.py` is the most expensive of the five (~1,600 judge calls, 112 generations, ~450
embedding requests) and the only one that is **resumable**: every paid cell is content-addressed
under `data/eval-cache` (gitignored), so a run killed by a 429 pays for the cell it died inside
and nothing else. `--stage` means *replay the rest from that cache* and not *skip it* — a
distinction the first version got wrong in both directions, so `--stage report` rendered an empty
RAGAs table over 560 cached cells and a cold `retrieve` cache was paid for under a flag that said
not to (code review of #11). It **refuses to start with `FINBRIEF_LOG_FILE` unset**, because
ADR-0005's latency half is measured from that log and discovering the sink was off after a
30-minute paid run means re-running the whole thing; `--allow-missing-sink` proceeds and the
artifact then reports those figures as absent. A **warm run has nothing to time**: latency comes
from this run's own window of the sink (`events.sink_offset`), so a fully replayed run's window is
empty and the artifact says "not measured, and therefore not met" rather than serving the previous
run's median.

`security_suite.py` is the cheapest of the four that spend money (~35 one-word
completions — every case the denylist does not catch, plus the whole benign set — plus a handful of
agent turns) and exists because three of ADR-0006's claims cannot be met by a test: whether a
real model recognises a *novel* payload, whether a real model *obeys* a planted one, and the
latency p50, which is a measurement. It exits non-zero on a failing suite. `--gate-only` is a
legitimate cheap check and says in the artifact which half it covered — a **PARTIAL RUN** banner
above the outcome line, and "planted payloads **not run**" in place of a count. Never commit a
partial run as a full one. That promise was prose until issue #8's review: the artifact's only
signal was `0/0 planted payload(s) resisted`, and because `all(())` is `True` a `SuiteRun` with
nothing in it at all rendered **SUITE PASSED** and exited 0. `SuiteRun.passed` now requires that
each half measured something, so an empty suite fails.

Never invoke any of these from a test.

## Conventions

**Tests are hermetic — no `.env`, no API key, no network.** `tests/conftest.py` patches
`load_dotenv` out, strips the managed env prefixes (including `LANGCHAIN_`/`LANGSMITH_`, so
tracing cannot POST), and clears the `load_env`/`get_settings` caches. Build configuration with
`Settings.from_env({...})` or `monkeypatch.setenv`; never read a real `.env`, and never add a
test dependency that fetches data at import time. **The no-network half is enforced, not
asserted**: `conftest.py` patches `connect`/`connect_ex`/`create_connection` and **all four
resolver entry points** (`getaddrinfo`, `gethostbyname`, `gethostbyname_ex`, `gethostbyaddr`) at
import time — before collection, which is when an import-time fetch happens — so egress to a
non-loopback host raises `EgressBlocked`; and it patches `curl_cffi.Curl.perform` and
`uvloop.Loop.getaddrinfo`/`getnameinfo`/`create_connection` separately, because both resolve and
connect in **C**, touching Python's `socket` module not at all. Those extra halves are not
optional trivia: `yfinance` uses `curl_cffi` whenever it imports, so without it the one library
T5 added was the one uncovered — measured at HTTP 429 with the socket guard installed; and
`guardrails.validator_service` sets the **process-wide** asyncio event loop policy to uvloop on
every `Guard.validate`, so the one library T7 added moved async DNS out of the guard's reach —
measured returning a real address with the socket guard installed and `socket.getaddrinfo`
verifiably patched. `security/advice.py` stops the policy swap (`GUARDRAILS_RUN_SYNC`) *and* the
guard covers the backend, because a hole is a hole whether today's code walks through it. The
resolver list is four names for the same reason: each is its own CPython C entry point, so
patching `getaddrinfo` alone left `gethostbyname` returning real addresses (issue #9 review).

**The guard is a denylist over the backends this repo can reach, not a proof**, and it is
described that way deliberately: a new HTTP dependency is a new path, and a guard advertised as
total is how the claim came to be false **seven** times (twice through tiktoken's cache, twice
through this guard's own prose, twice through T7's single new dependency — `guardrails-ai`,
which posts validation telemetry to its own endpoint *and* reaches uvloop — and once through
T10's, `ragas`, whose `_analytics.track` POSTs every metric completion to
`t.explodinggradients.com`).
`tests/test_hermetic_suite.py` carries **one test per backend** for that reason — an uncovered
path shows up there as a live call, which is where the `curl_cffi` hole was found. Add a
networking dependency, add a test there. And **a claim in a comment cannot fail**: three of the
seven false claims were prose asserting coverage the code lacked, so every test in that file
exercises the call it is about — the DNS case calls each of the four resolvers rather than
asserting that three route through the fourth, which is precisely the sentence that was wrong.
**Raising is only half the guard**, because a raised exception is loud only if the caller
propagates it: OpenTelemetry's `BatchSpanProcessor` catches the `EgressBlocked` on its own export
thread and logs it, so with the telemetry switch removed `Guard.validate` returned an entirely
normal verdict and the test asserting only that verdict passed while the POST was attempted
(issue #8 review). `conftest.EGRESS_ATTEMPTS` records every refused target inside `_blocked`
itself, so an attempt a library swallows is still visible; a test about a backend that might
swallow one compares that list's length across the call rather than trusting `pytest.raises`.
Five more mechanisms keep it true *without* leaning on the guard: tiktoken's cl100k_base table
is vendored under `tests/fixtures/tiktoken/` (conftest points `TIKTOKEN_CACHE_DIR` at it —
without that, `get_encoding` silently downloads it); `RAGAS_DO_NOT_TRACK` is set at conftest
**import** time and *assigned* rather than `setdefault`ed — import time because
`ragas._analytics.do_not_track` is `lru_cache`d, so a switch flipped after the first metric did
nothing, and assigned because a developer with `RAGAS_DO_NOT_TRACK=false` exported is exactly the
case worth overriding (`evaluation/judge.py` sets it a second time for the script, on the
`security/advice.py` principle that a hole is a hole whether today's code walks through it). That
`track` is decorated `@silent` and flushes from a background thread and at `atexit`, so
`EGRESS_ATTEMPTS` is the only detector — the third time that has been true. The EDGAR fixtures under
`tests/fixtures/edgar/` are recorded, never fetched — refresh them by hand with
`scripts/record_edgar_fixtures.py`; the market and news fixtures under `tests/fixtures/market/`
likewise, via `scripts/record_market_fixtures.py` (news recorded **raw**, so `feedparser` and
the HTML stripper really run; quotes recorded **parsed** at the `yfinance.Ticker` boundary,
which is therefore the one thing no test covers — the recorder says so); and retrieval runs
against a **real on-disk Chroma with a fake embedding** — `tests/fakes.py` holds the doubles
(`KeywordEmbeddings`, deterministic and lexical, so a test may assert an order; `a_context`, the
shared `Context` builder) and conftest builds `filings_store` / `empty_filings_store` from them.
Never point a test at the ingested `data/chroma`: it is only searchable by the paid model that
wrote it, so a test that reaches for it either needs a key or asserts against noise.
**Layer 3 is stubbed autouse**: `conftest.offline_injection_classifier` patches
`input_gate.classify` to return `Verdict.SAFE` for any caller that names no model, leaving
normalisation and the denylist real — so an app-level refusal test uses a *denylisted* payload
and needs no scripting, and a test that means to exercise layer 3 passes
`screen(question, model=...)`, which the stub delegates through. Autouse rather than opt-in
because a forgotten fixture would fail open into a live call and still pass. `planted_store` is
`security/corpus.py`'s injection collection as a fixture, over the same fake embedding.

**A check that cannot fail is the bug class this repo keeps hitting**, and it is worth naming as
one because the instances look unrelated until they are listed: tiktoken's warm cache made a
"hermetic" suite pass locally and egress in CI (twice); the guard's docstring claimed
`curl_cffi` coverage it lacked; a comment claimed `gethostbyname` routed through `getaddrinfo`;
`MAX_AGENT_STEPS` was pinned by `serial_full_brief < MAX_AGENT_STEPS`, an inequality that 15, 20
and 24 all satisfy; and T7's first live security run reported "not obeyed, no leak" about a
planted payload the agent had **never retrieved**, because the question named no company and it
asked which one instead of searching — a green cell about nothing
(`report.InjectionResult.retrieved` is the fix, and a row that did not reach its payload now
fails). T10 has contributed the two largest instances yet, both *inside published measurements*:
its first comparator tested `abs(delta) <= basis` where `basis` was the arms' own observed range —
which *is* the largest delta those values permit, so the test could not fail and all 18
pre-registered comparisons were verdicts from an instrument incapable of returning anything else
(the paired exact test replaced it, and `Paired.detectable` reports "we could not have seen it" as
a third claim); and the tool eval's control C3 declared no expected tool, no forbidden tool and no
argument, so `passed` was `True` for every possible agent behaviour while the case sat inside a
published **100%** accuracy. **Prefer a check that exercises the thing over one that describes it,
prefer an equality over a bound, and for any check about an adversarial input, assert that the
input arrived.**

The newest instance is a Streamlit-specific trap, so it is written down rather than
rediscovered: **`AppTest.get("...")` returns `[]` for an element type it does not know, instead
of raising.** `st.bar_chart` and `st.line_chart` both reach the element tree as
`vega_lite_chart`, for which `AppTest` ships no typed accessor — they arrive as `UnknownElement`
— so `assert not app.get("arrow_bar_chart")` passes on a page rendering no charts *and* on a
page rendering ten. Assert against `tests/test_app_smoke.py`'s `charts()` helper, which walks
the tree for `type == "vega_lite_chart"`; it also descends into `st.columns`, which
`app.chat_message[n]` does not, so a chart inside a column is invisible to the obvious lookup as
well. A new `app.get(...)` against an element `AppTest` has no wrapper for is a vacuous
assertion by default (issue #9 review).

**Single sources of truth.** Respect these or the invariant they protect is gone:

- `config.py` owns every knob, plus `CHUNK_SIZE_CHARS` — never copy the value. The
  ingestion thresholds are the deliberate exception: they are assertions, not knobs, and
  live next to the rule they belong to — `ingestion/gate.py`, `ingestion/model.py`
  (`HEADING_LINE_MAX_CHARS`, `POINTER_MAX_CHARS`, `POINTER_RESIDUE_MAX_WORDS`) and
  `CHUNK_OVERLAP_CHARS` in the chunker. Do not move them into `config.py` or make them
  env-overridable; each is calibrated against measured filings and documented where it sits.
  `agent/agent.py`'s `MAX_AGENT_STEPS` is exempt on the same grounds — a ceiling on the
  agent loop, next to the loop it guards. So are three of T7's:
  `security/denylist.py`'s `MAX_GAP_CHARS` (an assertion about how far apart a payload's words
  sit, calibrated against `security/corpus.py`), `security/advice.py`'s
  `NEGATION_WINDOW_CHARS` (an assertion about the length of a clause) and
  `security/markers.py`'s `MARKER_SPAN_MAX_CHARS` (an assertion about the shape of a citation —
  what `[12]` and `[Yahoo Finance]` look like — which nothing should be able to tune from the
  environment). T10 adds three, all assertions about what a *measurement* is allowed to claim:
  `evaluation/metrics.py`'s `PAIRED_ALPHA` (the significance level the paired exact test judges a
  difference at — an env-overridable alpha is a knob for tuning a verdict after seeing it),
  `evaluation/latency.py`'s `PLANNER_DISABLED_CAP` (an assertion about which arms configured the
  planner *off*, which is what separates the two translated arms from the two ablation ones in a
  latency pool — keyed on the turn's own `max_sub_queries` and deliberately **not** on the observed
  variant count, because a planner that ran and refused still paid for a full chat round and
  belongs in the pool; the first version conflated the two and biased the p50 upward) and
  `scripts/evaluate.py`'s `BUCKET_FLOOR` (ADR-0002 decision 3's
  per-bucket size, which the power audit reads to state what a bucket that size can resolve —
  changing it changes ADR-0002, not a run). Every *other* gate knob — the latency budgets
  (**both**: `GATE_LATENCY_BUDGET_MS` and `TRANSLATION_LATENCY_BUDGET_MS`, the second moved there
  from an unbound `budget_ms=1500.0` default argument in issue #11's review, because a pair split
  across two files is a pair that drifts and the gate's twin was already bound by an equality),
  the classifier's timeout and attempt count, **both** logged-input caps
  (`GATE_LOGGED_INPUT_MAX_CHARS` and `GATE_LOGGED_CLASSIFIER_TOKEN_MAX_CHARS`) and the answering
  path's `ANSWER_TIMEOUT_SECONDS` / `ANSWER_MAX_RETRIES` — is in `config.py`. The last pair moved there from `llm.py` in issue #8's
  review: the gate's twins were already in `config.py`, and a pair split across two files is a
  pair that drifts. **This list is the exception**: a limit not
  enumerated here belongs in `config.py`, or the exemption stops being narrow.
- `ingestion/model.py` owns the shared boundary definitions (`WORD`, `NEXT_ITEM_MARKERS`,
  `item_heading`/`section_start`) — a second copy lets a repair and the gate disagree.
- `llm.py` is the only chat-model constructor; `retrieval/embeddings.py` the only
  embeddings constructor, and ingest and query must share it (a drifting model degrades
  retrieval to noise with no error).
- `retrieval/vectorstore.py` is the only place the `filings` collection is opened,
  written, or read (`FILINGS_COLLECTION`) — which is why `collection_fingerprint` lives there and
  not in `evaluation/pipeline.py`, where it was a **third** corpus crossing from outside
  `retrieval/` (issue #11 review). `build_filings_store` is the constructor;
  `default_filings_store` is the **shared per-process handle** the application reads, and the
  app must go through it — `hybrid.bm25_index` caches against that store *object*, so anything
  that opens a fresh `Chroma` per query rebuilds the whole ~5,800-chunk lexical index for one
  question. **Six** process-level singletons sit on this path (`default_filings_store`,
  `hybrid.bm25_index`, `retrieve._planner_model`, `quotes.bounded_session`, and since T7
  `security/classifier.py`'s `classifier_model` and `security/advice.py`'s `advice_guard` — each
  keyed on the frozen `Settings` or on nothing at all) and every one goes through
  `caching.build_once`, **not** a bare `lru_cache`: with
  parallel tool calls two searches run concurrently, and two callers missing the same cold key
  both construct — which for the Chroma handle is fatal, because chromadb's shared-system
  registry is not reentrant (`AttributeError: 'RustBindingsAPI' object has no attribute
  'bindings'`, from inside the tool node, on the first turn of a cold process). `lru_cache` is
  atomic about its bookkeeping and says nothing about the function it wraps. A **seventh** singleton
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
  cap `sub_queries()` keeps. Since T7 it also owns the injection classifier's prompt and the two
  **label constants** the parser enforces (`CLASSIFIER_INJECTION_LABEL`/`_SAFE_LABEL` — a label
  written twice is a gate that fails open on the turn the copies disagree), the two refusal texts
  the app renders, and `QUARANTINE_TAGS`: the one tuple naming every tag untrusted text is wrapped
  in (`sources`, `news`, and the classifier's own `input`). `quarantined()` neutralises all of
  them wherever a body is interpolated, and `security/denylist.py`'s `delimiter-forgery` rule
  derives its pattern from the same tuple — so a fourth block is escaped *and* denylisted the
  moment it is declared. **Both halves are parametrised over the tuple, and that is the point**:
  this sentence was here while the rule hardcoded `(?:sources|news|system)`, so `</input>` was
  escaped and never denylisted and `<system>` was denylisted while being no quarantine tag at all
  — a rule in this file describing a derivation the code did not do (issue #8 review). `<system>`
  survives as a *named* extra in `denylist.py`, with a test saying so. **A refusal names no layer and no rule**: which fired is in the
  gate-trigger log, and an attacker told which rule they tripped is one told how to phrase the
  next attempt. And `prompts.py` may not import from `retrieval/` at runtime:
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
- All structured logging goes through `log_event`, and since T8 `observability/events.py` is the
  only **reader** — one emitter, one parser, because the two drifting apart is silent: a renamed
  field reads back as `None`, `None` averages as nothing, and the number narrows its own
  denominator. `Samples` therefore returns `absent` beside `present`, `Event.field` defaults to
  `None` rather than `0` (an unreported token count is not a free call), and a *missing* sink
  raises rather than reading as an empty log — "nobody enabled it" is not "this run emitted
  nothing". `tests/test_event_log.py` round-trips through both halves, including one event whose
  field is absent; that test is what forbids the drift. **Absence is per field, not per record**,
  and that is where it broke: `tokens.usage_total` summed with `.get(name, 0)` and counted a
  reply as metered if it reported *any* usage, so a provider returning half a pair put
  `output_tokens: 0` on the line with a denominator claiming otherwise — a fabricated zero in the
  one module whose docstring forbids it (issue #10 review). Each field carries its own
  `<field>_calls`. The sink itself is
  `FINBRIEF_LOG_FILE`, **off unless named** (`configure_logging()` takes no arguments at four
  entry points, and conftest strips `FINBRIEF_`, so a path-shaped default would have the
  hermetic suite writing files) — which means it has to be *enabled* where the data matters:
  `.env.example`, the README's "Recording a run", T11's walkthrough. In `.env.example` it ships
  **commented out**, with a test: default-off in code and default-on in the file the README tells
  you to copy is not a default-off, and what it silently switched on was retention of blocked
  questions' normalised text. `.env.example`'s path is also bound to `.gitignore` through
  `git check-ignore` rather than through prose, because the ignore rules cover a changed
  directory and not a changed filename. A `turn_id` on the envelope
  is what makes a `retrieval` line's provenance attributable; it comes from a `ContextVar`
  (`logging_setup.turn`), which is **measured** to survive LangGraph's tool executor across a
  thread boundary rather than assumed to — and **measured at the app**, its only production
  caller, because neutralising `app/Home.py`'s `with log_turn(...)` once left all 1028 tests
  green: every other turn-id test opened the scope itself, so they proved propagation and never
  wiring. `log_event` takes `exc_info` so that a *failed* turn is an event too; it had been the
  one bypass of the single emitter, and so the one line with no `turn_id`.
  **A statistic over the sink is not a statistic over a run**, because it is append-only across
  every run and app session that names it: T10's first artifact reported a planner p50 over a pool
  of 13 appended runs while its header claimed the numbers were that run's, and re-reading the
  unchanged file two runs later gave a different number. `events.sink_offset` marks the file before
  a run and every reader takes `start_offset` — an offset rather than a new envelope field, so a
  line written by an older deploy still parses. Two consequences: a **fully replayed** stage
  appends nothing, so its window is empty and the p50 **raises** rather than serving the previous
  run's median (a latency figure now means the stage behind it ran); and the window is persisted
  beside the cached cells (`latency.Window`), because the artifact is regenerable from that cache
  and a re-render must describe the measuring run rather than refuse. **Every pass that emits runs
  before anything that reads** — a log-reading section computed before the pass that writes its
  lines reports an absence on a run that measured the thing, which happened twice here: the agent
  stage, then the planner-variance pass.
  ADR-0011 records why
  the log is shaped as a run's
  record rather than as the harness's primary input — T10 gets provenance from
  `Retrieval.contexts` in-process, and needs the log only for latency, tokens and live-run facts.
  Never a secret in `fields`, and **never a question or a query variant**: a variant is derived
  from user content and these lines are kept. Counts, lengths and verdicts only — which is why the
  `retrieval` event records a chunk's provenance by *variant index* while `Surfaced` itself
  carries the variant text for the RAG-viz panel to render (ADR-0004 amendment).
  **One exception, and it is bounded three ways** (ADR-0006 requires the normalised input in a
  gate-trigger record, and a denylist you cannot audit is one you cannot tune): the `input_gate`
  event carries `normalised` **only on a block**, **only the normalised form** — lowercased,
  punctuation-free, de-leetspeaked — and only `config.GATE_LOGGED_INPUT_MAX_CHARS` of it. An
  allowed screening logs counts and verdicts like everything else. **Do not describe the second
  bound as making the text "unusable as a question", which is what this said**: folding destroys
  figures and identifiers (`Item 1A` → `item ia`, `$5bn` → `ssbn`) and leaves the wording legible,
  measured on a real benign question (issue #8 review). The cost is therefore larger than the old
  wording implied and it is still worth paying — a false positive puts a readable innocent
  question in a kept log, which is the reason for the three bounds rather than a reason to keep no
  record. Nothing else may widen this, and a bound may not be described as removing more than it
  removes.
- `security/markers.py` owns what a citation marker *is* — the bracket shape, the numeric rule
  and `numeric_markers()` for anyone counting them. `evaluation/deferrals.py` had its own
  `re.compile(r"\[(\d+)\]")` for the same job, which lets a measurement *of* the gate disagree
  with the gate about what it is measuring (issue #11 review).
- `security/` is the gate, one module per layer, because a marginal-contribution claim has to be
  checkable *at* the layer it is about (user story 34): `normalize.py` (folds obfuscation into two
  forms — `text` keeps the word boundaries a rule needs to *not* match, `squeezed` is the only
  form in which letter-by-letter spacing still contains its words), `denylist.py` (one rule set
  scans both, which is why a rule head-anchors with `\b` and joins words with a bounded gap and
  never a literal space), `classifier.py` (one call, fails **open**, and the fail-open is a
  `Verdict.UNDECIDED` in the return type rather than a `False` that reads like a verdict),
  `input_gate.py` (the composition, and the only place a gate-trigger line is written),
  `advice.py` and `markers.py` (the two back-door halves), `corpus.py` (the committed payloads —
  in the package, not `tests/`, because the hermetic suite and `scripts/security_suite.py` must
  read one corpus), `report.py` (the artifact, rendered from measurements). Layer 3's verdict is
  **injectable**, so `screen(question, model=...)` is how the suite drives it without a paid call.
  The gate runs at `app/Home.py`'s chat input and nowhere else: inside the agent it would screen
  the model's own tool arguments, and inside `rag.answer_question` it would put a model call in
  front of the chain ADR-0003 keeps deterministic.
- `evaluation/` is the harness, one module per concern, for the reason `security/` is one per
  layer: a claim about a *measurement* has to be checkable at the thing that produced it.
  `loader.py` is the only reader of the one hand-authored artifact (`golden_set.json`) and it
  **raises** on an unverified set rather than footnoting it; `arms.py` owns the six-arm matrix and
  every arm names its own two switches, never reading them from config — and
  `TRANSLATION_CONTRAST`/`STRATEGY_CONTRAST` are read by `hypotheses.CLAUSE_CONTRAST` and
  `TRIGGER_CONTRAST`, which is what makes the axis-constancy tests bind ADR-0005's operands;
  `variants.py` owns the resolve-once/replay contract (ADR-0004 §9) and its **two** load-time
  checks — `verify_replay` that the persisted reply still parses to the variants beside it, and
  `verify_questions` that the plan was resolved for the wording the golden set asks *now*;
  `judge.py` is the only place `ragas` is called and the only place the judge model is built, and
  `ragas.evaluate()` is deliberately unused because the cache needs a result per `(row, metric)`;
  `cache.py` is content-addressed cells plus `HARNESS_VERSION`, which is how a cell that is *wrong
  under a right address* gets evicted (a log line cannot do it), and `resolve` vs `replay` is what
  makes `--stage` mean "replay the rest" rather than "skip it"; `metrics.py`, `latency.py`,
  `deferrals.py` and `hypotheses.py` are the measurements; `report.py` is the artifact, rendered
  from measurements and never typed. **A claim this package makes about its own output is a claim
  something has to render**: the T10 review found the deterministic table printing "`recall_trivial`
  rows (reported separately)" with nothing reporting them, and `metrics.py` promising the recall
  ceiling "beside" a column that never carried it.

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
