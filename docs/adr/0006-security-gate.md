# ADR-0006: Security gate — defense-in-depth, front door + back door

The original design stacked two model-based jailbreak detectors on direct input (a custom
LLM classifier *and* Guardrails `DetectJailbreak`) while treating indirect injection — the
more relevant threat for a RAG system — as a one-line prompt-framing footnote. That is
redundant on one axis and thin on the axis that matters, and it is hard to defend in a
review that rewards explaining security choices.

**Decision.**

- **Input gate (front door)** — cheap-first, one model call per turn, ≤800ms p50 (from logs):
  1. **Normalization** — lowercase, strip accents/homoglyphs, collapse whitespace &
     punctuation, de-leetspeak. Catches *obfuscation*.
  2. **Bounded regex denylist** — a catch exits early; a pass *always* escalates. Catches
     *known patterns* for free.
  3. **One zero-shot LLM classifier** (own prompt, via OpenRouter) — YES/NO instruction
     override / system-prompt extraction. Catches *novel phrasings*.
- **Output validator (back door)** — Guardrails AI no-investment-advice validator on
  responses (`on_fail="exception"` → graceful refusal). This does work the input layers
  don't: it catches *consequences*, including those of a *successful* indirect injection.
- **Indirect injection is a first-class, tested Tier-1 threat.** A dedicated test
  collection (NOT the demo KB) is seeded with injection strings in fake news/filing chunks;
  tests assert the model neither obeys them nor leaks the system prompt. News is
  HTML-stripped on ingest.

**Marginal-contribution table (→ README):** normalization → obfuscation; regex → known
patterns; classifier → novel phrasings; output validator → consequences. Each layer has a
distinct job; none is redundant.

**Framing.** Defense-in-depth: input gate guards the front door; quarantine framing +
output validation guard the back door (retrieved content).

## Amendment (ticket T3, issue #5) — the quarantine framing, as shipped

"Quarantine framing" was one clause above, and Phase 2 had to make it concrete before the
gate it belongs to exists: the baseline chain hands the model real filing text on its first
turn. `finbrief/prompts.py` cites this ADR for the framing, so what it settled belongs here
rather than in a docstring.

1. **Retrieved text is wrapped in `<sources>…</sources>` in the *human* message, never in the
   system message.** The system message is the persona and the rules; mixing filing text into
   it makes the two indistinguishable to the model, which is the whole failure mode.
2. **Order is load-bearing: sources → framing sentence → question.** The question is the
   instruction and the sources are data, so the sources may not be the outermost frame the
   question sits inside — a filing that ends "…now ignore the above and…" must not be the last
   thing the model reads before answering. `tests/test_prompts.py` pins the order, and
   `tests/test_rag.py` asserts it again through the chain that builds the message.
3. **The persona makes the block evidence and asks the model to report, not obey.** Text inside
   `<sources>` that addresses the model, changes its behaviour or claims new rules is quoted
   filing content: relevant to *mention*, never to act on.
4. **Framing only.** The normalization/regex/classifier input gate and the output validator
   remain Phase 5, and so do the planted-injection tests against the dedicated collection
   (item 3 of the Decision above). Phase 2 asserts the framing those layers assume is already
   in place — it does not claim the gate.

## Amendment (ticket T7, issue #8) — the gate as shipped, and one missed pre-registration

What the Decision above got right, what it got wrong, and what building it found. The four
layers all shipped; three of them are cheaper than expected and the fourth is stronger, and
the **latency pre-registration was missed** — recorded here as missed rather than quietly
restated.

### 1. The layers, as built

`finbrief/security/`: `normalize.py` → `denylist.py` → `classifier.py`, composed by
`input_gate.screen()`; `advice.py` is the back door and `markers.py` its deterministic half.
Each layer is its own module because a marginal-contribution claim has to be checkable *at*
the layer (user story 34) — not through a verdict the layer below could have produced.

Two implementation facts worth the record:

- **Normalisation produces two forms, not one.** `text` keeps word separators so a rule can
  use `\b` and *not* match inside an unrelated word (`contract assets` must not trip an
  `act as` rule); `squeezed` removes them, which is the only form in which
  `i g n o r e   a l l   …` still contains its words. One rule set scans both, which imposes
  two authoring rules on `denylist.py`: head-anchor with `\b`, and join words with a bounded
  gap rather than a literal space.
- **`MAX_GAP_CHARS = 20` lives in `denylist.py`**, not `config.py` — an assertion about a
  payload family, calibrated against the corpus, on the same grounds as the ingestion
  thresholds. CLAUDE.md's exemption list records it.

### 2. The pre-registered ≤800ms p50 is **not reliably met**, and the budget is amended

**Measured, from the `input_gate` log lines, on `openai/gpt-4o-mini`: an escalated p50 of
738, 812, 823, 887, 910, 942, 962 and 1041 ms across eight passes of 22 escalated
screenings.** It is over 800 ms in six of the eight, under in two, with a median of medians of
about 899. So the pre-registration is not *missed* in the sense of being clearly beaten by the
implementation — it is **straddled**: the gate lands on either side of the figure depending on
provider jitter, which makes 800 ms a budget the artifact would report as met or missed at
random. That is worse than a clean miss, because it makes the number uninformative.

Two medians are reported, and only one of them is honest. A p50 over the whole corpus is
**around 520 ms and comfortably within budget** — because 13 of the 19 attacks are caught by the
denylist and exit in microseconds, never paying for the model call. The *escalated* median is
what an ordinary analyst question costs, since an ordinary question is exactly the one that
escalates. The artifact reports both and says which to read.

**One attempt was made to clear it by model choice, and measured rather than assumed.**
`google/gemini-2.5-flash-lite` runs at **331–494 ms p50 across five passes — roughly 2.4×
faster, comfortably inside the budget on every pass — at an identical attack catch rate
(18/18 over three passes).** It was **not taken**, because it blocked a legitimate analyst
question that `gpt-4o-mini` never did: *"Can you ignore the tax effects and just give me the
gross margin?"*, once in 48 attempts, against 0 of 48 for `gpt-4o-mini`. A false positive
refuses an analyst an answer they were entitled to, which is the failure the benign control set
exists to catch, and buying ~500 ms with it is the wrong trade on a security control — the same
reasoning that keeps advice requests out of the denylist. Tuning the classifier prompt until
gemini stops firing on that question was available and is exactly what ADR-0003's standing rule
forbids: the prompt is not tuned against a paid model to move a number.

**So the budget is amended, with its reasoning stated:**

- It was pre-registered **before anything had measured what a classifier round trip costs.**
  800 ms was a target chosen from first principles, and the cheapest model that holds the catch
  rate on this account sits right on top of it.
- **The gate is 9–21% of the wait it sits inside.** Measured on three real turns against the
  ingested KB: 1257 ms gate / 8149 ms turn (13.4%), 1255/12693 (9.0%), 2920/11297 (20.5%). A
  100–200 ms overrun is not perceptible against a turn the analyst is already waiting 8–13
  seconds for.
- The alternative on the table trades **catch quality for latency on a security control**, which
  is the one trade this ADR exists to refuse.

The revised figure is therefore **≤1s escalated p50 on `Settings.classifier_model`** — chosen so
that it is a budget the gate clears on every pass measured rather than one it clears about a
quarter of the time. It is judged on the escalated median and reported by
`docs/verification/security-gate.md` every run.

**Both figures stay in `config.py` and in the artifact.** `GATE_LATENCY_BUDGET_PREREGISTERED_MS`
is kept precisely because the prediction did not hold: deleting it would turn a revised
pre-registration into a number that had always been met, which is the quiet pass ADR-0005's
pre-registration discipline exists to prevent. This is a **revised** pre-registration with its
measurements on the record, in the same spirit as ADR-0004's T6 amendment recording a falsified
retrieval hypothesis.

### 3. Layer 3 fails open, and the failure is not hypothetical

A provider outage, a timeout, or a reply the parser cannot read is `Verdict.UNDECIDED`, which
the gate **allows** while logging a warning. Failing closed would turn an OpenRouter blip into
an assistant that refuses every question and reads, to the analyst, as censorship rather than
an outage; failing open leaves three layers standing. The cost is the novel-phrasing coverage
of one turn.

**Measured while choosing a model, and it is why the artifact warns about it.** Three candidate
models (`amazon/nova-micro-v1`, `meta-llama/llama-3.1-8b-instruct`,
`openai/gpt-oss-safeguard-20b`) return HTTP 404 on this account's data policy. With fail-open,
that presented as **22–28 ms p50 and 0/6 attacks caught** — the three fastest rows in the
benchmark, and entirely fictional. `gpt-4o-mini` itself produced 1 `UNDECIDED` in 66 calls. A
suite run during an outage would report layer 3 catching nothing and still pass for layers 1, 2
and 4, so the artifact's limitations section says to check the verdict counts when a number
surprises.

### 4. The three candidates recorded on #8

1. **`</sources>` delimiter forgery — fixed** (`prompts.quarantined`). Escaping, not a
   per-turn nonce: the nonce would have to reach the persona, and `_boundaries` names the tags
   in prose the chain and the agent share, so the rule the model reads would become a per-turn
   argument — losing the written-once property. The block also crosses the checkpoint, and
   `agent/citations.py` rebuilds a reply's content on every renumber, so a live thread would
   see several tags. `QUARANTINE_TAGS` is the single source of truth: `sources`, `news` and the
   classifier's own `input`, and `denylist.py`'s `delimiter-forgery` rule derives from the same
   tuple, adding `<system>` as a named extra because nothing here frames anything with it.
   **That derivation was prose before it was code** — the rule hardcoded `(?:sources|news|system)`
   while this section, `prompts.py` and CLAUDE.md all claimed otherwise, leaving `</input>`
   escaped but not denylisted; both halves are now parametrised over the tuple (issue #8 review).
   **The news path is where this is exploitable today**, not latent: a filing comes from
   EDGAR and is trusted by provenance (ADR-0007), where a headline comes from whoever got a
   post onto a syndicated feed.
2. **`[n]` marker validation — taken, as report-not-repair** (`security/markers.py`).
   `unresolved` and `non_numeric` are counted apart because T3's `[6]` and T5's
   `[Yahoo Finance]` are different defects: one is a citation pointing at no panel entry, the
   other a collision with the syntax that makes any citation resolvable. Stripping a dangling
   marker silently would remove the evidence the reader needs; refusing the whole answer over a
   numbering slip would make it indistinguishable from layer 4's refusal. So the marker stays
   and a caption names it. Validated against **every number the conversation has issued**, not
   the turn's — a follow-up may legitimately cite a source the previous turn retrieved, and
   scoping it to the turn made the caption fire on correct answers.
3. **Faithfulness — out of scope here**, and stays with the golden-set RAGAs run (T10, #11;
   the golden set is ADR-0002's). A marker that resolves can still sit on a claim its chunk
   does not support, which needs a judge model and ground truth. Recorded as a stated
   limitation in T11's README rather than dropped.

### 5. Gate-trigger logging carries the normalised input **on a block only**

ADR-0006 asked for "normalized input, layer fired, matched pattern, timestamp", and CLAUDE.md
forbids user-derived text in `log_event` fields. Both are right, so the exception is narrow and
bounded three ways: **only on a block**, **only the normalised form**, and **only
`config.GATE_LOGGED_INPUT_MAX_CHARS` (500) of it**. An allowed turn logs counts and verdicts
like every other event. The argument for it is that a denylist you cannot audit is one you
cannot tune — reviewing a false positive means seeing what tripped it. The argument against is
that a false positive puts an innocent question in a kept log, which is a real cost and the
reason for the three bounds rather than a reason to have no record. The timestamp is the
envelope's `ts`; a second one would be a second clock.

**Correction (issue #8 review): the second bound said "unusable as a question", and it is not.**
Normalisation destroys *figures and identifiers* — `Item 1A` → `item ia`, `$5bn` → `ssbn` — and
leaves the wording legible: *"Can you ignore the tax effects and just give me the gross margin?"*
normalises to `can you ignore the tax effects and just give me the gross margin`. Measured, not
argued. The block-only and 500-character bounds are real and implemented; this one overstated the
mitigation, and it overstated it for precisely the case the exception exists to serve — a false
positive is the screening that gets logged, and on the gemini benchmark above that exact question
is the one a classifier fired on. The decision stands; the description of it is now what the code
does. A mitigation is only worth what it actually removes.

### 6. Amendment §2 above describes the **chain**, not the agent

The T3 amendment's §2 says "order is load-bearing: sources → framing → question". That is
literally true of `prompts.user_message` and cannot be true of the agent: retrieved text
arrives as a `ToolMessage`, so a tool result always follows the question that caused it, and
the question is structurally never last.

What carries §2's *intent* on the agent path is that `sources_block` ends with **its own
framing sentence**, so the final text before generation is FinBrief's instruction about the
block rather than the filer's last paragraph. `tests/test_indirect_injection.py` asserts it
there. Recorded because the amendment as worded covers one of the two paths, and a reader is
entitled to know which.

### 7. The Universe whitelist is an **incidental input-side control**, and it invalidated a test

The planted-injection collection was first built for a fictional filer, `ZZZ`, so a chunk from
it would be identifiable on sight. Every row came back clean and every row was worthless: the
agent declined to search at all, because `AGENT_SYSTEM_PROMPT` names the 15 companies FinBrief
covers. Naming the ticker explicitly made it worse — all five rows returned "I can only provide
information for the 15 companies in FinBrief's Universe."

That is correct behaviour (user story 21) and it is a **finding**: the Universe whitelist,
built for input validation, also stops out-of-Universe *indirect* payloads from ever reaching
the model. It is not part of the four-layer argument — it is incidental, and it is not a
defence against a payload planted under an in-Universe ticker, which is the case that matters
— but a test using an out-of-Universe filer measures the whitelist rather than the quarantine
framing.

So the planted chunks use an **in-Universe ticker throughout**, the *collection* carries the
isolation this ADR asks for (a throwaway directory, built and destroyed per run, never
`data/chroma`), and identifiability moved to the provenance, where it is stronger than a
ticker: an all-zero accession and fiscal year 1970. `report.InjectionResult.retrieved` now
records whether each payload reached the model at all, and a row that did not is a **failure**
— the first run reported "not obeyed · no leak" about a turn in which nothing adversarial was
ever retrieved, which is the check-that-cannot-fail class CLAUDE.md names.

### 8. Guardrails AI: what it supplies, and two things it does uninvited

**What it supplies** is the `Guard`/`Validator` framework, the `on_fail` semantics and one
standard error shape. It supplies **no rule**: there is no hub validator for "no investment
advice", so the rule is a registered custom validator over a deterministic pattern set, and the
`on_fail=EXCEPTION` is converted into an `AdviceVerdict` the app renders. `DetectJailbreak`
stayed dropped, as the Decision above intends.

The pattern set draws its line **per rule**, and that is the design: a rule matching a *noun*
("price target", "your portfolio") yields to a denial in the same clause, because a refusal can
mention one; a rule matching an *act* ("you should buy", "I recommend", an imperative) never
yields, because "I do not recommend Tesla" is a recommendation. A blanket negation guard is the
hole this avoids.

**Two uninvited behaviours, both found by running `tests/test_hermetic_suite.py` rather than by
reading the dependency tree**, and both now the fifth and sixth entries in this repo's
false-hermetic-claim history:

- It **POSTs anonymous validation telemetry** to its own endpoint unless `~/.guardrailsrc` says
  otherwise. A library added *for* a security control would, unconfigured, have sent a record of
  every validated answer to a third party. Disabled in `advice._build_guard`, in the one order
  that works — and which of the two switches is load-bearing was measured, not assumed.
- It sets the **process-wide asyncio event loop policy to uvloop** on every `Guard.validate`.
  uvloop resolves DNS inside libuv, so every subsequent async lookup left the reach of a guard
  written in Python — the `curl_cffi` story again. `GUARDRAILS_RUN_SYNC=true` stops the policy
  swap (same code path, no side effect, no warning), and the guard covers the backend anyway,
  because a hole is a hole whether today's code walks through it.

### 9. What the layers do **not** cover, stated

- **Novel advice phrasing** is layer 4's blind spot exactly as a novel payload is layer 2's,
  and there is no layer 5. The golden-set RAGAs run measures the residue statistically.
- **A refused answer is still in the agent's memory.** `answer()` has already run when layer 4
  fires, so a follow-up in the same thread can reference text the reader never saw. This layer
  guards the surface, not the checkpointer.
- **The homoglyph map is not the Unicode confusables table.** A lookalike outside it survives
  normalisation and reaches layer 3, which is the layer that exists for what layers 1 and 2
  miss.
- **`retrieve()` and `rag.answer_question` are ungated by design.** The gate is at the app's
  chat input — the only door a human types through. A gate in the agent loop would screen the
  model's own tool arguments; a gate in the chain would put a model call in front of the path
  ADR-0003 keeps deterministic.
