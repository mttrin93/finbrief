# ADR-0002: Golden-set design and per-bucket A/B evaluation

Three Tier-1 pillars — RAGAs, A/B testing, and the retrieval precision/recall numbers —
all depend on one artifact: the evaluation golden set. Its design determines whether the
review-facing numbers are defensible or circular.

**Decisions.**

1. **Source-separated ground truth.** Ground-truth answers are authored by reading the
   filings directly, with the source section cited. Candidate Q/A pairs are drafted by a
   *different* model than the answering pipeline, then hand-verified against the primary
   source. The defensible claim: references were derived from the documents, not from the
   retriever's output — so the metrics cannot be circular.

2. **Four stratified buckets**, bucket recorded per question:
   `semantic`, `exact-identifier`, `tool-augmented`, `multi-hop`.
   The guardrail/injection cases are **excluded** — faithfulness against a refusal is
   undefined. They live in the Phase-5 security suite as pass/fail assertions, reported
   alongside the RAGAs table but never inside it.

3. **Per-bucket A/B reporting is a Tier-1 requirement.** Aggregate reporting can hide the
   effect; per-bucket reporting is what demonstrates it.

4. **Pre-registered hypotheses:**
   - hybrid > vector-only on `exact-identifier`
   - query translation wins on `multi-hop`
   - both roughly tie on `semantic`
   A predicted tie is evidence the experiment is sound, not a failure.

5. **Size:** ~24–28 questions, ≥6 per bucket (15–20 across the buckets was too thin for a
   directional per-bucket claim).

**Consequence.** Evaluation authoring is now a real chunk of Phase 7 (reading filings and
writing cited references by hand), not a one-line afterthought — but the RAGAs/A/B tables
become evidence a reviewer can interrogate rather than decoration.
