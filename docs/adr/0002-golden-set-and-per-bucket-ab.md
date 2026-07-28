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

---

## Amendment (T9, #4): a `tool-augmented` row has two halves, and only one is RAGAs-scorable

Authoring the set surfaced a hole in decision 1. "Ground-truth answers are authored by reading
the filings directly" is exact for three buckets and *undefined* for the fourth: a
tool-augmented question's answer is partly a live price, a live market cap, a computed ratio or
a headline list. None of those is in a filing, none is stable between two runs, and there is no
primary source to hand-verify a number against that will still be that number tomorrow.

**Decision.** A `tool-augmented` row carries two fields with two different scorers.

1. A **filings-half reference** (`grounding` + `reference`) authored exactly as decision 1
   requires — read from the filing, Section cited, chunk ids named. This is the **only**
   RAGAs-scorable half, and it is what the per-bucket A/B measures, because retrieval is the
   thing being ablated.
2. A **tool expectation** (`tool_expectation`) — the tool's name, the arguments the call must
   carry, and for `calculate_ratios` the `peer_set` and `n`, both copied from `config.PEERS` and
   asserted against it by `tests/test_golden_set.py` (ADR-0009 keeps peers in-Universe; a
   hand-typed peer set in the golden set would be the one copy that could disagree with the map
   the tool resolves against). This half is scored by the **tool-calling eval** (spec US-29),
   which asserts the agent selected the right tool with the right arguments — pass/fail against
   the expectation, never a RAGAs metric.

`search_filings` is deliberately *not* recorded in `tool_expectation`: every row in every bucket
needs it to reach its filings half, so naming it 28 times would be noise. The field records the
*additional* non-retrieval tool the row needs.

**Why this split and not another.** It is ADR-0003's chain-vs-agent seam applied to the
reference data rather than to the code. ADR-0003 separates `rag.answer_question` — the
deterministic, measured chain — from the agent's nondeterministic tool loop, so retrieval quality
can be measured without the loop in the way. The same line runs through a tool-augmented row: its
filings half is a chain fact and belongs to the measured seam; its tool half is a loop fact and
belongs to the eval that measures the loop. Scoring the tool half with RAGAs would be scoring the
agent's tool selection with a retrieval metric, which is the category error ADR-0003 exists to
prevent.

**The consequence the harness must handle, stated so it cannot be discovered late.** RAGAs
faithfulness scores an answer's sentences against the context set it was given. A tool-augmented
answer's tool-derived sentences — the price, the ratio, the headline — are *not* in the retrieved
contexts, so faithfulness will mark them **unsupported** and the bucket will report a depressed
number that measures nothing but this schema. So T10's harness must do exactly one of:

- **inject the tool output into the context set** for these rows, so a tool-derived sentence has
  something to be faithful *to*; or
- **exclude the `tool-augmented` rows from the faithfulness metric**, reporting the bucket on the
  other three RAGAs metrics plus the tool-calling pass rate, and saying so in the table.

Either is defensible. What is not defensible is running faithfulness over these rows unchanged
and reporting the result, because the number would be an artifact of the split above rather than
a property of the answer. The choice is T10's to make and to record; this amendment fixes only
that it must be made explicitly.

**Also settled while authoring (T9, #4), and recorded here because the schema carries them:**

- **`known_false_positives: [chunk_id]`** on every row, populated for the two ADR-0004 §11
  mention-leakage probes — one chunk for the Microsoft case, all 45 non-AAPL `Apple` chunks for
  the Apple case, enumerated from the ingested collection. Ground truth excludes them even though
  the engine returns them, which is what makes mention-leakage precision computable separately
  from the bucket mean (#11) instead of invisible inside it.
- **`recall_trivial`** — true when *every* Section a row grounds in holds ≤ 4 chunks, so a `k=5`
  retrieval cannot miss it and the row scores recall 1.0 whatever the retriever does. T10 reports
  those rows separately rather than letting a question that cannot miss inflate the recall column.
  `recall_trivial_sections` carries the partial case, because a multi-section row is not
  all-or-nothing and the whole-row flag alone would let a partially-immune row be reported as
  immune. Counted from the collection, not assumed: on this set the rule fires on **one** whole row
  (MSFT Item 7A, 3 chunks) plus two Sections of two multi-filer rows (AAPL Item 7A 4, TSLA Item 7A
  2). The other Item 7A Sections are larger than the rule expected — NVDA 7, META 8, AMZN 10,
  GOOGL 11, GM 18, F 23 — so "Item 7A is small" does not hold as a generalisation.
- **`basis_mismatch`** — true when a row compares figures not measured on the same basis or at the
  same date. The reference must then state each figure's basis and date explicitly rather than
  implying comparability. One row carries it: the three-filer 100bp interest-rate comparison, where
  Apple's $2,416m is its investment portfolio at 27 Sep 2025, Microsoft's $1,415m a fair-value line
  at 30 Jun 2025, and Meta's $711m its AFS debt securities *and* cash equivalents at 31 Dec 2025.
  The row is kept as a three-filer hop deliberately — the comparison being non-trivial is what
  makes it a good multi-hop question — but a reference reducing it to a bare ranking would be
  teaching the metric to reward an unsound comparison.
- **`verified_against_edgar`** per row, plus one at the top level that must equal the conjunction
  of them. Decision 1 says candidates are hand-verified against the primary source; the flag is
  what stops a set being cited before that pass happens, and the test binds the top-level claim to
  the rows so it cannot read `true` over unverified data. Ingestion could in principle mis-parse a
  Section, and a mis-parse would propagate straight into ground truth.
