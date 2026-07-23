# ADR-0005: Pre-committed shipping default retrieval strategy

`search_filings` ships with one default strategy (ADR-0003), while the A/B produces four
configs' worth of per-bucket numbers. Choosing the default *after* seeing the numbers is
p-hacking — "we tried four and reported the winner." So the rule is pre-registered here,
before any A/B data exists.

**Decision.**

- **Default = `hybrid + translation`**, pre-registered on the dominance prediction of
  ADR-0004 (neutral-or-better on every bucket within a cost budget).
- **Falsification clause (directional).** With ~6 questions per bucket, a tight numeric
  margin would be false precision. Rule: if translation is worse on *both* context
  precision *and* context recall within any bucket, the default drops to `hybrid-only`,
  and the contradiction is written up in the README as a finding (worth more review credit
  than a clean win). Per-question spread is reported alongside every bucket mean.
- **Cost is part of dominance.** Dominance is judged within a latency budget of **≤1.5s
  p50 added by translation**, measured from the Phase-6 structured logs. A quality win
  that breaches the budget does not count as dominant.
- **Adaptive per-query routing is rejected for Tier-1** (e.g. detect exact-identifier
  queries → skip translation) and listed as Tier-2 future work — it adds a classifier that
  would itself need evaluation.

**Consequence.** The shipping default is defensible as a prediction that survived (or
didn't) a pre-registered test, not a post-hoc pick.
