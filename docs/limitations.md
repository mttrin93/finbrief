# Limitations, each with its mechanism

All eighteen. [The README](../README.md#part-5--limitations) carries the six that would
change how a reviewer reads the numbers; this file is the whole list, and each row names
the mechanism rather than the symptom.

---

| limitation | mechanism |
|---|---|
| **the square-bracket adherence rate is unmeasured** | `citation_markers` is emitted by `app/Home.py` and by nothing else, so a harness driving `agent.answer` produced 10 live turns and **zero** lines. The turns happened and the markers were produced; the instrument is somewhere else. An instrument at the display layer has a human in its denominator, and moving it needs a decision about where marker resolution belongs — so it is its own ticket rather than a change smuggled into a measurement run |
| **cited-sentence support is mostly partial** | 22 fully / 40 partly / 8 not supported over 70 pairs. A partly supported sentence has a marker that resolves and a chunk carrying *some* of the claim — invisible to the citation register, and scored as fine by whole-answer faithfulness because the claim is supported somewhere in the context set |
| **the translation latency budget was missed and left unamended** | 3212 ms against 1500 ms. No argument exists that 3.3 s is acceptable for an analyst's turn, and moving a pre-registered number to wherever the measurement landed is pre-registration in reverse |
| **the per-bucket A/B is a screen, not a test** | ADR-0002 sized buckets at ≥6 questions; the exact paired test needs 6 differing questions with zero minority signs, or 7 with one. Small real effects are *unresolvable*, not weakly supported |
| **`yfinance` is unofficial** | an undocumented endpoint with no API contract, so an endpoint change is a fetch failure rather than a support ticket. The TTL cache, the retry and the stale banner are how it degrades; Alpha Vantage's free tier is 25 calls a day and its key is configured but **unread** — a fundamentals fallback is deferred, not shipped |
| **a stalled quote endpoint holds the cache lock ~91.5 s** | two HTTP requests per attempt × 3 attempts, all inside `TimedCache`'s lock, which is held across the refresh so two sessions asking about one ticker coalesce into one call. Cutting it further means retrying *outside* the lock, which trades the free tier's call budget for latency and voids ADR-0009's "peers add zero API surface" — a decision made in the open rather than an optimisation applied quietly |
| **single-language knowledge base** | English filings, English embeddings, and no query-side translation into English before retrieval. A non-English question is embedded as-is and retrieves accordingly |
| **no re-ranking stage** | promoted to Tier-2 #2 with a pre-registered hypothesis (precision lift concentrated on `semantic`, ~neutral on `exact-identifier`) and a constrained third A/B axis. The local cross-encoder's deploy weight is the accepted cost |
| **the knowledge base is four Items of one filing** | Item 8's financial statements, 10-Qs, proxies, earnings calls and every other Item are out of scope; a question whose answer lives outside Items 1, 1A, 7 and 7A is unanswerable by design, and the app says so |
| **table and figure fidelity inside an ingested Section** | extraction keeps the prose and can lose a table's layout, so hard numbers come from the finance tools rather than the filing text |
| **foreign private issuers are out of scope** | a 20-F has no Item 1A/7/7A, so such a filer cannot supply a Section at all; supporting one needs its own section mapping and its own gate. The Universe is curated to 10-K filers, and EDGAR is asked to confirm it per filing |
| **Item 1 → 1A and 1A → 1B have no boundary check** | there is no unambiguous next-Item marker (a business section discusses its risks in passing), and a false accusation teaches a reader to wave the gate through. For those two the length ceiling is the only backstop |
| **JPM's Item 7 runs one page past its cross-reference** | Item 7 is defined content-anchored, from its own heading to the start of Item 8; trimming to the p.160 footer needs filer-specific page-number parsing to buy 3,091 characters and would reintroduce the special case the heading rule removed |
| **the provenance header carries the ticker, not the company name** | which is why hybrid search *alone* made the motivating case worse. The index-side fix — carrying both forms — is deliberately not taken: the header is inside `content_hash`, so it re-embeds all 5,842 chunks and invalidates every distance recorded on three tickets, including the committed smoke band and the before/after this finding rests on |
| **the gate's four stated gaps** | layer 3 **fails open** (an outage allows the turn; not hypothetical — three candidate models were unavailable on this account and fail-open made them read as the *fastest* rows in a benchmark, catching nothing) · layer 4's novel-phrasing blind spot, with no layer 5 · **a refused answer is still in the agent's memory**, because the validator guards the surface and not the checkpointer · the homoglyph map is not the Unicode confusables table |
| **a green suite is evidence about its corpus** | the corpus author's blind spot and the rule author's are one blind spot, and running the suite again does not split them. Two bypasses were found by adversarial reading, not by eight passes |
| **the Universe is fixed at ingest time** | 15 companies, one filing each; adding one is an ingest run, not a setting |
| **no deployment and no live URL** | Tier-2 #1 and still the highest-value remaining item, since reach gates the value of everything else |

Two deliberate asymmetries that read as limitations and are not. A **gate-blocked question is not
in the agent's memory at all** — layers 1–3 stop the turn before the agent runs, so the refusal is
on screen while the checkpointer never saw the question, and a follow-up cannot build on it. And a
**retrieved-but-uncited chunk still appears in the sources panel**, because the panel's contract is
"what grounded this turn", which is also why the log field is `retrieved_sections` and not
`cited_sections`.

