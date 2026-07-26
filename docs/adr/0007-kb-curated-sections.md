# ADR-0007: Knowledge base = curated 10-K sections, not full filings

The `exact-identifier` bucket and scoped retrieval depend on reliable section metadata
(`section`, `ticker`, `fiscal_year`). 10-Ks on EDGAR are messy HTML/iXBRL: "Item 1A"
appears in the table of contents as well as the real section, headers vary by filer, and
some sections are incorporated by reference. Naive full-filing ingestion pollutes the KB
with TOC/boilerplate and mislabels chunks silently.

**Decision.**

- **KB = curated sections only: Items 1, 1A, 7, 7A** — not full filings.
- **Structure-anchored extraction via `edgartools`' section extraction**, with bounded
  regex as a *documented fallback*. No hand-rolled primary parser.
- **Phase-1 gate — section-detection sanity check.** Per company × section, assert the
  section was found, is non-empty, falls within length bounds, and has **body ≫ heading**
  (catches TOC hits). Fails loudly *before* any retrieval numbers exist.
- **One-time hand-verification checklist** of all extracted section starts, committed as an
  artifact.

**Consequences.**
- The golden set is authored **against ingested sections only** — no questions whose
  answers live outside Items 1/1A/7/7A.
- The app **declares its grounding scope in the UI** (what it is and isn't grounded in).
- **Table/figure fidelity is a stated limitation**; hard figures come from the tools
  (`get_stock_data`, `calculate_ratios`), not the filings.
- **Foreign private issuers (20-F) are out of scope.** A 20-F has no Item 1A/7/7A, so a
  foreign private issuer cannot supply a Section under this decision at all — supporting
  one would require a separate 20-F section mapping and its own extraction gate. The
  Universe is therefore curated to 10-K filers only. `config._TWENTY_F_FILERS` is a
  tripwire under that rule, not a proof of it: a denylist catches the plausible additions
  to the existing clusters, so the filing type still has to be confirmed on EDGAR when
  adding any company. This surfaced late: the Universe originally carried an
  `eu_tech` cluster (SAP, ASML, STM) that ADR-0009 and CONTEXT.md both sanctioned as
  "EU ADRs" while this ADR scoped the KB to 10-K sections — the two documents disagreed
  and the code faithfully implemented the wrong one. Stated limitation; README reflection
  material.

**Rejected alternative.** Full-filing ingestion — rejected because TOC/boilerplate
pollutes the exact-identifier bucket and section labels become unreliable.

---

## Amendment (ticket T2, issue #3) — what the gate learned from fifteen real filers

The decision above survives; three of its mechanisms did not survive contact with the
Universe. Recorded here because each was a guess that the scaling run replaced with a
measurement.

**1. The 20-F rule is now enforced against EDGAR, not against a list.**
The gate asks EDGAR for each company's most recent annual filing across the 10-K / 20-F /
40-F families and asserts it is in the 10-K family. This supersedes `_TWENTY_F_FILERS` as
the source of truth; the denylist is kept as a free, offline tripwire that catches a
plausible re-addition before a network call, not as proof of the rule. Form *families* are
compared, because EDGAR's form search prefix-matches: a 10-K search returns `10-K/A` too,
an amendment is still a 10-K filer, and a company's most recent annual filing can be a
`10-K/A` amending an older fiscal year (Tesla files one every April). Which document to
*ingest* is a separate decision, made with an exact-form filter — Tesla's April amendment
is a Part III document with no Item 1/1A/7/7A in it.

**2. "Body ≫ heading" is measured in words; "ran into the next Item" is measured in
content. Neither is a character count.**
A table-of-contents hit is a heading, a run of dot leaders and a page number — *long* and
nearly wordless — so a character ratio waves it through and a word ratio does not.

The same lesson, more expensively, for the length ceiling. It was 400,000 and it caught
JPM's Item 7 at 413,149 characters. But only the last 12,837 of those are Item 8: JPM's
*legitimate* MD&A is 400,312 characters, so the ceiling fired 312 characters from failing
correct content, while BAC (293,458) and GS (309,157) are clean at nearly that size. A
character count cannot separate "the parser crossed into Item 8" from "this is a bank".
So a `section_stops_before_the_next_item` check now does that on content — the auditor's
report is Item 8 and is never MD&A — and the ceiling is raised to 500,000 and demoted to a
sanity backstop. It earned its place immediately: GM's Item 7A was 26,450 characters, of
which 14,662 were the auditor's report, and it had been passing every length and shape
rule silently.

The check covers Items 7 and 7A only. Item 1 → 1A and 1A → 1B have no unambiguous marker
— a business section discusses its risks in passing — and a false accusation would be
worse than the miss, because it teaches a reader to wave the gate through. For those two
the ceiling is the only backstop. **Stated limitation.**

**3. Two documented repairs, both judged by the gate afterwards.**
Where structure-anchored extraction crosses a boundary, the Section is trimmed at the
first next-Item marker. This is the same species as the bounded regex fallback this ADR
already sanctions: one rule, always a prefix of the filer's own words (never a rewrite, so
BM25 still sees them — ADR-0004), and never silent, since every trim logs a
`section_trimmed` event. What keeps it from being the extractor grading its own homework
is ordering: the trimmed text still faces the same marker check in the gate, so a repair
that misses fails exactly as it would have before. The repair proposes; the gate disposes.
Rejected the alternative of discarding the whole Section — it would have thrown away a
bank's entire MD&A over 3% contamination.

**4. Incorporation by reference is a lawful filing, not a defect — and not ingested.**
Six of the fifteen Universe companies — every bank and every healthcare name — answer Item
7A with a sentence directing the reader to Item 7. Their wordings share no formula
("Refer to", "See", "are set forth in", "incorporated herein by reference", "You can
find"). This is recognised *positively* — short, plus hand-off language, plus a target that
is itself ingested — and never as "short sections are forgiven", so a genuine extraction
miss that happens to be brief still fails loudly. "Item 7A. Not applicable." matches
nothing and still fails.

Such a Section passes the gate and is **not** chunked: "Refer to pages 133-142" would
embed cleanly, retrieve for every market-risk question, and ground none of them. The
content is not lost — it lives inside Item 7, which is ingested — but it is **labelled
`Item 7`, not `Item 7A`**.

**Consequence for the golden set (ADR-0002).** Nine of fifteen companies have an Item 7A
in the KB; six do not. A question whose *answer* is market risk remains answerable for all
fifteen. A question that cites `Item 7A` as its source section is answerable for nine.
Authoring must use the committed verification artifact, not the assumption that every
company has all four Sections.

**Verification artifact.** `docs/verification/section-starts.md`, generated by
`scripts/ingest_filings.py --section-starts` and committed unticked. The gate proves a
Section is present, non-empty, bounded, body-heavy and free of the next Item — none of
which distinguishes the real Item 1A from a plausible mis-extraction that is also all
five. Only a person reading the excerpts against EDGAR closes that gap, which is why the
boxes ship unticked.
