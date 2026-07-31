# FinBrief — a domain-specialised RAG assistant for equity research

**Ask about a company's 10-K and get an answer you can check.** Every claim carries an inline
`[n]` resolving to the chunk it came from, linked to that filing on EDGAR; price, peer ratios and
headlines come from tools, because a 10-K has no prices in it; and a question the assistant should
not answer is refused rather than answered badly. The scope is 15 large-cap companies and four
Items of each one's latest annual filing — [declared on screen, with its exact
arithmetic](#23-the-grounding-scope-disclosure), rather than implied.

**What it asks to be judged on.** Every quality number in these documents is requoted from a
generated file under [`docs/verification/`](./docs/verification/) and bound to it by
`tests/test_grounding_scope.py` — so a re-run that moves a figure fails the test suite instead of
leaving prose asserting the old one. Several of those numbers came back saying *we cannot tell*,
and that is what they say here.

![A question about Tesla's risk factors answered with numbered citations and a sources panel, then a
follow-up naming no company that resolves anyway while the citations keep counting, then a full brief
with price and peer-ratio cards](./docs/assets/finbrief-demo.gif)

*Steps 1 → 4 of [the walkthrough](./docs/demo.md), which is reproducible exactly as written.*

| | |
|---|---|
| **Stack** | Python · Streamlit · LangChain / LangGraph `create_agent` · OpenRouter · ChromaDB + BM25 · Guardrails AI |
| **Data** | SEC EDGAR via `edgartools` · Yahoo Finance (unofficially, via `yfinance`) · news RSS |
| **Evidence** | five generated files under [`docs/verification/`](./docs/verification/) — never hand-authored |
| **Design record** | eleven ADRs under [`docs/adr/`](./docs/adr/) · plan in [`PLAN.md`](./PLAN.md) · spec in [`docs/spec/finbrief.md`](./docs/spec/finbrief.md) · glossary in [`CONTEXT.md`](./CONTEXT.md) |

---

## Contents

A ten-minute read. Each section that has a longer version links to it.

| | |
|---|---|
| **[Part 1 — Orientation](#part-1--orientation)** | what it is · [quickstart](#12-quickstart) · [the demo walkthrough](#13-the-demo-walkthrough) → [`docs/demo.md`](./docs/demo.md) |
| **[Part 2 — How it works](#part-2--how-it-works)** | [architecture](#21-architecture) · [the validity gap](#22-the-validity-gap-stated-rather-than-assumed) · [grounding scope](#23-the-grounding-scope-disclosure) · [the rest of the build](#24-the-rest-of-the-build-in-brief) → [`docs/implementation.md`](./docs/implementation.md) |
| **[Part 3 — Optional tasks](#part-3--optional-tasks)** | [the thirteen in brief](#31-the-thirteen-in-brief) · [all 21, with status](#32-all-21-with-status) |
| **[Part 4 — What the evaluation established](#part-4--what-the-evaluation-established)** | six numbers → [`docs/findings.md`](./docs/findings.md) · [`docs/verification/evaluation.md`](./docs/verification/evaluation.md) |
| **[Part 5 — Limitations](#part-5--limitations)** | the six that matter → [`docs/limitations.md`](./docs/limitations.md) |
| **[Part 6 — ADRs, cost, running it](#part-6--adrs-cost-running-it)** | [ADR index](#61-adr-index) · [what it cost](#62-what-it-cost) · [running it](#63-running-it) → [`docs/implementation.md`](./docs/implementation.md) |

**The five documents.** This README is the tour. [`docs/demo.md`](./docs/demo.md) is the
five-step walkthrough, with the two preconditions a first take needs. [`docs/implementation.md`](./docs/implementation.md)
is how each requirement and each optional task is built. [`docs/findings.md`](./docs/findings.md) is
what the project found out, including the eighteen-instance table of checks that could not fail.
[`docs/limitations.md`](./docs/limitations.md) is all eighteen limitations with their mechanisms.
Under those, [`docs/verification/`](./docs/verification/) is generated evidence and
[`docs/adr/`](./docs/adr/) is the design record: consult an artifact for any number that matters
([`evaluation.md`](./docs/verification/evaluation.md) is the measurement artifact of record) and an
ADR for any decision that looks arbitrary. The two are separate documents on purpose — an ADR
records what was predicted, a generated file records what happened, and neither can quietly become
the other.

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
intend to report — see [3.14](./docs/implementation.md#314-logging--monitoring).

## 1.3 The demo walkthrough

**Five steps, 2–3 minutes, reproducible exactly** — [`docs/demo.md`](./docs/demo.md). Risk factors
for Tesla, then a follow-up that names no company and still resolves while the citations continue
from `[6]` rather than restarting, then valuation with its tool cards, then the full brief, then an
advice refusal and a blocked injection payload. Two preconditions are in that file and both matter
on a first take: **warm the quote cache** (a cold stall can hold the cache lock for 91.5 s) and
**enable the event sink**, without which step 5 leaves no gate-trigger record.

---

# Part 2 — How it works

## 2.1 Architecture

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

## 2.2 The validity gap, stated rather than assumed

The headline RAGAs and A/B numbers measure **the chain**. What ships is **the agent**. That gap
is real, and ADR-0003 commits to stating it rather than assuming it away:

| | the measured chain | the shipped path |
|---|---|---|
| entry point | `rag.answer_question` / `retrieve()`, called question by question | `agent.answer`, a tool-calling loop |
| determinism | exact, except the sub-query planner's one chat completion — and that reply is replayed from a committed file, so the A/B arms are reproducible | nondeterministic: the model chooses whether, when and with what arguments to search |
| tools | **none** — the chain calls no finance tool at all | four |
| what measures it | per-bucket RAGAs and the A/B matrix ([3.16](./docs/implementation.md#316-ab-testing-of-rag-strategies), [3.21](./docs/implementation.md#321-ragas-evaluation)) | the tool-calling eval, reported **beside** those tables and never inside them |

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
  ([findings.md §4](./docs/findings.md#4-prompt-rules-are-instruments-not-enforcement)).

One consequence worth being explicit about: a multi-hop question is decomposed *inside*
`retrieve()` under evaluation, so if the agent instead answers one by making several tool calls,
that is the selection layer working — and it surfaces in the tool-calling eval rather than as
noise in the chain's numbers.

## 2.3 The grounding-scope disclosure

The app says what it is grounded in, in the words the model is given — one constant,
`prompts.GROUNDING_SCOPE`, read by the caption under the title, the sidebar's scope panel and four
prompts:

> Filing answers are grounded only in Items 1, 1A, 7 and 7A of the latest annual 10-K for each of
> the 15 companies in FinBrief's Universe.

That is **54 of 60** company × Section pairs, and every number in it is derived from `config` and
the `Section` enum rather than typed — then cross-checked against the ingest run's own gate table,
because self-consistent arithmetic is not the same as true. The full disclosure, the six filers who
answer Item 7A by reference, and where the counts come from are in
[`implementation.md`: The grounding-scope disclosure](./docs/implementation.md#22-the-grounding-scope-disclosure).

## 2.4 The rest of the build, in brief

Each of these is a core requirement, and each has its own section in
[`docs/implementation.md`](./docs/implementation.md) — the knowledge base and its gate, the four
tools, the persona, the stack and the error tiers, and the UI surfaces.

**RAG implementation** — curated 10-K Sections rather than full filings: 15 companies × Items 1,
1A, 7 and 7A, **5,842 chunks** at 1000 characters with 200 of overlap, one `text-embedding-3-small`
constructor shared by ingest and query, Chroma L2 distance at `k=5`. The section-detection gate
passed **60 of 60** company × Section checks before anything was embedded, and the sixtieth row is
what hand-verification caught that no automated check did.
→ [`implementation.md`: RAG implementation](./docs/implementation.md#21-rag-implementation)

**Tool calling** — four tools: `search_filings` over the knowledge base, plus `get_stock_data`,
`calculate_ratios` and `get_recent_news` over free public data. Peers are the company's own curated
cluster inside the Universe, so peer averaging adds **zero new API surface**; every comparison
carries its peer set, its `n` and its range. Tool-selection accuracy is **100% over 10 scored
cases**. → [`implementation.md`: Tool calling](./docs/implementation.md#23-tool-calling)

**Domain specialisation** — the specialisation is the knowledge base and the prompt, not a
fine-tune: an analyst persona, financial vocabulary, and a refusal policy that lets *"should I buy
X?"* through the front door and refuses it at layer 4 with a disclaimer, because an advice request
is not an injection. → [`implementation.md`: Domain specialisation](./docs/implementation.md#24-domain-specialisation)

**Technical implementation** — LangGraph `create_agent` over OpenRouter with a `SqliteSaver`
checkpointer, every knob in `config.py`, error handling in three tiers (API retry with a stale
banner · retrieval fallback · refusals as UX), input validation against the Universe whitelist, and
a hermetic test suite of **1,507 tests** whose no-network contract is enforced at the socket layer,
all four DNS resolvers, `curl_cffi` and `uvloop` rather than asserted.
→ [`implementation.md`: Technical implementation](./docs/implementation.md#25-technical-implementation)

**User interface** — a Streamlit chat page: sources panel rendering retrieved text verbatim through
`st.text`, *How I answered*, tool cards and charts, `st.status` progress, the sidebar's six panels,
and export buttons. → [`implementation.md`: User interface](./docs/implementation.md#26-user-interface)

---

# Part 3 — Optional tasks

**A note on the numbering.** The assignment lists optional tasks unnumbered under Easy, Medium
and Hard. The numbers `3.1`…`3.21` used here and in
[`docs/implementation.md`](./docs/implementation.md) are a **local convention of this repo** —
they index the list in the order I recorded it, and are used only as stable subsection anchors.
[§3.2](#32-all-21-with-status) is the full 21-row table, named rather than numbered, so
nothing here depends on the numbering being anyone else's.

Thirteen are built: **Easy 4/4 · Medium 6/10 · Hard 3/7**, against a bar of 2 medium + 1 hard.

## 3.1 The thirteen, in brief

Two to four lines each; the full section for every one is in
[`docs/implementation.md`](./docs/implementation.md).

**Easy**

- **Conversation history + export** — the `SqliteSaver` checkpointer is the memory of record and
  `st.session_state` holds only the thread id, UI state and the transcript; one `uuid4` per browser
  session is what keeps two users apart on a shared cached agent, asserted by two `AppTest` sessions
  in one process. → [3.1](./docs/implementation.md#31-conversation-history--export)
- **RAG process visualisation** — a *How I answered* panel per answer: the queries that ran (**5**
  variants → **10** candidate lists under hybrid) and, per chunk, which variant × which retriever
  surfaced it and its RRF contribution. A variant that surfaced nothing is shown as such.
  → [3.2](./docs/implementation.md#32-rag-process-visualisation)
- **Source citations** — inline `[n]`, numbered once per thread at the `before_model` seam so a
  marker three turns up still resolves, each linked to its filing on EDGAR (all **15** URLs
  reproduced from ticker and accession by a test). Support is measured and mostly partial: **22
  fully supported / 40 partly supported / 8 not supported** over 70 pairs.
  → [3.3](./docs/implementation.md#33-source-citations)
- **Interactive help / guide** — a *How to use FinBrief* panel and **four** example-question
  buttons, seeded through the same path a typed question takes so they are gated like any other. The
  `/help` command is open, not cut. → [3.4](./docs/implementation.md#34-interactive-help--guide)

**Medium**

- **Prompt-injection protection** — four layers, each catching what the one before it cannot, and
  each measured at the layer: **20/20** attacks stopped by the *expected* layer, **28/28** benign
  questions allowed, **22/22** answer verdicts correct, **5/5** planted payloads retrieved and
  resisted, at a **670 ms** escalated p50. → [3.7](./docs/implementation.md#37-prompt-injection-protection)
- **Token usage + cost display** — a per-conversation meter that reads T8's logged counts rather
  than adding an instrument: absent rather than zero when the sink is off, scoped by `turn_id`
  prefix, and naming the structurally unmetered gate classifier on **every** total.
  → [3.9](./docs/implementation.md#39-token-usage--cost-display)
- **Tool-call result visualisation** — quote, ratio, news and failure cards, with **one chart per
  ratio metric** because a P/E and a debt-to-equity share a unit and not a scale: on one axis Ford's
  4.26× leverage drew as three pixels beside a 162× peer mean.
  → [3.10](./docs/implementation.md#310-tool-call-result-visualisation)
- **Conversation export in various formats** — JSON and CSV from the *display transcript*, not the
  checkpointer, because the two differ in both directions; every `[n]` resolves against the file's
  own source list. **PDF declined** on dependency grounds, not effort.
  → [3.11](./docs/implementation.md#311-conversation-export-in-various-formats)
- **Rate limiting + API key management** — a **40**-question per-session cap that is cost and abuse
  limiting and **not a security control**, since refreshing the page resets it; keys from `.env` or
  `st.secrets`, read once, never logged.
  → [3.13](./docs/implementation.md#313-rate-limiting--api-key-management)
- **Logging + monitoring** — JSON lines through one emitter and one reader, **27** event types, a
  `turn_id` measured to survive LangGraph's tool executor, an opt-in file sink, and exactly one
  bounded user-derived field. → [3.14](./docs/implementation.md#314-logging--monitoring)

**Hard**

- **Hybrid search** — BM25 + vector over every query variant, fused by RRF at the published `k=60`,
  with deterministic entity normalisation adding a ticker form. The case that motivated it moved a
  target chunk from rank 5 → **absent** under hybrid alone → rank **2** once normalised, on a panel
  where **5/5** entries are the right filer against 1/5 before. Not a perfect ranking, and said so:
  the top entry is a chunk that is not about debt, and the planner-off ablation reaches rank 1 where
  the shipped arm does not. The mechanism proved **embedding-side, not lexical**: BM25 is one vote
  of three.
  → [3.15](./docs/implementation.md#315-hybrid-search)
- **A/B testing of RAG strategies** — six arms (two of them planner-off ablations that are a
  pre-registered refutation channel), four buckets of seven questions, six hypotheses settled by an
  exact paired test with three verdicts rather than two. **4 of 18** comparisons carried a
  measurement. → [3.16](./docs/implementation.md#316-ab-testing-of-rag-strategies)
- **RAGAs evaluation** — a 28-question golden set with source-separated ground truth, all 28 rows
  hand-verified against EDGAR, all four metrics per bucket per arm. Response relevancy is excluded
  from every hypothesis for two measured reasons, and the exclusion is machine-checkable.
  → [3.21](./docs/implementation.md#321-ragas-evaluation)

## 3.2 All 21, with status

All 21 optional tasks. The numbers are this repo's local convention (see
[Part 3](./docs/implementation.md#part-3--optional-tasks-implemented)); the names are what identifies each row. **13
complete — Easy 4/4, Medium 6/10, Hard 3/7**, against a bar of 2 medium + 1 hard.

| # | task | status | where / why |
|---|---|---|---|
| **Easy** | | | |
| 1 | Conversation history + export | ✅ complete | [3.1](./docs/implementation.md#31-conversation-history--export) |
| 2 | RAG process visualisation | ✅ complete | [3.2](./docs/implementation.md#32-rag-process-visualisation) |
| 3 | Source citations | ✅ complete | [3.3](./docs/implementation.md#33-source-citations) |
| 4 | Interactive help / guide | ✅ complete — panel and buttons ship; the `/help` **command** is open | [3.4](./docs/implementation.md#34-interactive-help--guide) |
| **Medium** | | | |
| 5 | Multi-model support | ✕ not built | Tier-2, behind the gate, and it carries an open question that must be answered first: the injection classifier and the tool-calling prompts are tuned per model, so a picker needs a per-model check that swapping does not silently break tool selection |
| 6 | Real-time data updates to the knowledge base | ✕ not built | Tier-2, with an open question: whether a live ingest stays idempotent *and* holds the ADR-0007 section gate without racing the cached retriever and the agent's state |
| 7 | Prompt-injection protection | ✅ complete | [3.7](./docs/implementation.md#37-prompt-injection-protection) |
| 8 | User authentication and personalisation | ✕ not built | Tier-2 tail. It is also what real rate limiting needs — a bound keyed server-side on something the client does not choose — so the gap is stated in [3.13](./docs/implementation.md#313-rate-limiting--api-key-management) rather than implied |
| 9 | Token usage + cost display | ✅ complete — per **conversation**; the per-**message** half is open | [3.9](./docs/implementation.md#39-token-usage--cost-display) |
| 10 | Tool-call result visualisation | ✅ complete | [3.10](./docs/implementation.md#310-tool-call-result-visualisation) |
| 11 | Conversation export in various formats | ✅ complete — JSON and CSV; **PDF declined** on dependency grounds | [3.11](./docs/implementation.md#311-conversation-export-in-various-formats) |
| 12 | Connect to tools from a public remote MCP server | ✕ not built | Tier-2 #3 by skill signal, and the highest-value unbuilt item after deployment. Needs a remote server chosen *and* the MCP security review that goes with it; not started rather than half-done |
| 13 | Rate limiting + API key management | ✅ complete — and explicitly **not** a security control | [3.13](./docs/implementation.md#313-rate-limiting--api-key-management) |
| 14 | Logging + monitoring | ✅ complete | [3.14](./docs/implementation.md#314-logging--monitoring) |
| **Hard** | | | |
| 15 | Hybrid search | ✅ complete | [3.15](./docs/implementation.md#315-hybrid-search) |
| 16 | A/B testing of RAG strategies | ✅ complete | [3.16](./docs/implementation.md#316-ab-testing-of-rag-strategies) |
| 17 | Automated knowledge base updates | ✕ not built | a scheduled GitHub Action over the ingest script; generic scheduling work that reinforces none of the Tier-1 core, and it inherits row 6's idempotency question |
| 18 | Multi-language support | ✕ not built | the knowledge base is single-language English and the embeddings are English; a UI toggle without query-side translation into English before retrieval would be a language switch that degrades retrieval silently |
| 19 | Advanced analytics dashboard | ✕ not built | a second Streamlit page over the event log. The data exists ([3.14](./docs/implementation.md#314-logging--monitoring)) and the reader exists; what it adds is a chart, not a GenAI capability, so it lost to everything above it |
| 20 | Implement your tools as MCP servers | ✕ not built | Tier-2, with the sharpest open question of the four: whether this is a genuine protocol *port* or a second copy of the finance logic behind a second interface. It must reuse one implementation, and that is a design decision rather than a build |
| 21 | RAGAs evaluation | ✅ complete | [3.21](./docs/implementation.md#321-ragas-evaluation) |

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

# Part 4 — What the evaluation established

Six numbers. **Five** are requoted from
[`docs/verification/evaluation.md`](./docs/verification/evaluation.md) and bound to it by
`tests/test_grounding_scope.py`; the sixth is marked † because it is not one artifact's figure —
it is a range across three passes, and only the last of them is committed. The reasoning behind
each, and the four findings this project would put its name to, are in
[`docs/findings.md`](./docs/findings.md).

| | measured | what it means |
|---|---|---|
| pre-registered comparisons that carried a measurement | **4 of 18** | the other 14 were *undetectable at this n* — statements about the sample, not about retrieval |
| hybrid − vector on the bucket hybrid exists to win | **−0.087** | not a resolved loss and not a gain; the point estimate runs the wrong way |
| p50 added by translation, against a 1500 ms budget | **3212** ms | over budget by more than 2×, recorded as missed and left **unamended** |
| RAGAs faithfulness · context precision · context recall, shipping default, 28 rows | **0.758** · **0.543** · **0.675** | with `[min–max]` and `n` on every bucket mean in the artifact |
| retrieval cells re-paid from scratch, then cells replayed on context-body keys | **168** re-paid → **672** replayed, zero misses | `retrieve()` is byte-identical across re-runs on keys carrying the full text of every context |
| planner runs returning an identical sub-query set † | **0–2 of 8** across three passes | the planner is *not* reproducible, which is why the A/B arms replay a recorded reply |

† **The one row above that is not a committed artifact's number.** The pass makes live planner
calls, so it inherits the variance it is measuring and is quoted as a range rather than as one
run's figure. `evaluation.md` carries the last pass — **2 of 8** — and the two earlier ones exist
only in `docs/findings.md`'s account of them. The conclusion is the same in all three (6, 7 and 8
of 8 questions varied), which is what makes the range usable; the range itself is not requotable
from the artifact and is not presented as though it were.

**So the shipping default is retained, not validated.** Hybrid earns nothing detectable on any
bucket, the one root-caused live case favours the simpler arm, translation misses its latency
budget, and the clause that protects the default **could not have fired on any bucket**.
`vector + translation` is a live candidate this run could not rule out. What would settle it is a
larger per-bucket `n` and nothing else: at ~7 questions the exact paired test resolves only
near-unanimous effects.

**And the first version of that determination was a tautology** — a comparator that tested a delta
of means against the arms' own range, which *is* the largest delta those values permit, so all 18
comparisons were verdicts from an instrument incapable of returning any other. That, the
determinism result, the eighteen-instance table of checks that could not fail, and what
"a prompt is an instrument, not a control" cost in measured compliance are all in
[`docs/findings.md`](./docs/findings.md).

---

# Part 5 — Limitations

The six that would change how a reviewer reads the numbers. All eighteen, each with the mechanism
that causes it, are in [`docs/limitations.md`](./docs/limitations.md).

| limitation | mechanism |
|---|---|
| **the square-bracket adherence rate is unmeasured** | `citation_markers` is emitted by `app/Home.py` and by nothing else, so a harness driving `agent.answer` produced 10 live turns and **zero** lines. An instrument at the display layer has a human in its denominator |
| **cited-sentence support is mostly partial** | 22 fully / 40 partly / 8 not supported over 70 pairs — the marker resolves and the chunk carries *some* of the claim, which neither the citation register nor whole-answer faithfulness can see |
| **the translation latency budget was missed and left unamended** | 3212 ms against 1500 ms; moving a pre-registered number to wherever the measurement landed is pre-registration in reverse |
| **the per-bucket A/B is a screen, not a test** | ADR-0002 sized buckets at ≥6 questions, and the exact paired test needs 6 differing with zero minority signs, or 7 with one. Small real effects are *unresolvable*, not weakly supported |
| **`yfinance` is unofficial** | an undocumented endpoint with no API contract, so a change there is a fetch failure rather than a support ticket. The TTL cache, the retry and the stale banner are the degradation |
| **single-language knowledge base, and no re-ranking** | English filings and English embeddings with no query-side translation before retrieval; re-ranking is Tier-2 #2 with its hypothesis already pre-registered |

Two more are worth naming here because they are the gate's, not the retrieval's: **layer 3 fails
open** (an outage allows the turn, and three candidate models being unavailable once made them read
as the *fastest* rows in a benchmark, catching nothing), and **a passing security suite is evidence
about the cases in its corpus**, which cannot be otherwise — the corpus author's blind spot and the
rule author's are one blind spot.

---

# Part 6 — ADRs, cost, running it

## 6.1 ADR index

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

## 6.2 What it cost

Five non-hermetic entry points; four of them spend money. **No dollar figure in this project comes
from a committed artifact**, and this section is careful about the difference:
[`evaluation.md`](./docs/verification/evaluation.md) carries token counts, not prices, and both
cost knobs default to unset for the reason [3.9](./docs/implementation.md#39-token-usage--cost-display) gives.

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

## 6.3 Running it

```bash
uv run ruff check . && uv run ruff format --check .   # what CI runs
uv run pytest
uv run streamlit run app/Home.py
```

Those are hermetic. The five entry points that reach the network — ingest, its no-key `--dry-run`,
the retrieval smoke check, the security suite and the evaluation harness — are in
[`implementation.md`: Running everything](./docs/implementation.md#27-running-everything), with the
four rules that govern them (a partial run says so · `evaluate.py` refuses to start with no event
sink · a warm run has nothing to time · a subset ingest may not overwrite whole-Universe evidence)
and what each of the five committed evidence files under
[`docs/verification/`](./docs/verification/) contains. **Four of the five spend money.**
