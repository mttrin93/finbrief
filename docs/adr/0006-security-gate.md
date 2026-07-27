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
