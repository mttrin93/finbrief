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
