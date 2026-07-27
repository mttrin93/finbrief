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
`section_trimmed` event. Rejected the alternative of discarding the whole Section — it
would have thrown away a bank's entire MD&A over 3% contamination.

*What the ordering does and does not buy* (corrected — the first draft of this ADR said
"the repair proposes; the gate disposes", which overstates it). The trimmed text still
faces the gate's marker check, so a repair that found **no** marker to cut at fails there
exactly as it would have before the repair existed. But the cut is at the *first* marker,
so the survivor provably contains none: `section_stops_before_the_next_item` cannot fire
on text this has already trimmed, and the same holds for `section_starts_at_its_heading`
against `trim_to_section_start`. A cut in the **wrong place** — an over-eager marker
(`"critical audit matter"` is the one an MD&A could plausibly use), or a title fallback
anchoring on a body line — is therefore invisible to the gate. What catches it is the
`section_trimmed` event and the hand-verification artifact below, which flags any row
whose text moved as `CHANGED` and unticks it. That is the honest division of labour: the
gate proves mechanical properties, and a person proves the text is the right text.

**4. Incorporation by reference is a lawful filing, not a defect — and not ingested.**
Six of the fifteen Universe companies — every bank and every healthcare name — answer Item
7A with a sentence directing the reader to Item 7. Their wordings share no formula
("Refer to", "See", "are set forth in", "incorporated herein by reference", "You can
find"). This is recognised *positively*, never as "short sections are forgiven", so a
genuine extraction miss that happens to be brief still fails loudly. "Item 7A. Not
applicable." matches nothing and still fails.

The recognition is three cuts, and each one replaced a way the previous rule could excuse
a truncated genuine Section:

1. Only Item 7A, and only under `POINTER_MAX_CHARS` (1,500 — J&J's real pointer is 502,
   so the gate's own floor would be far too tight a test).
2. Some **single sentence** must both hand off and name an ingested target. Co-occurrence
   anywhere in the text was the original rule, and a "See below" three sentences away from
   a mention of MD&A is not a hand-off. Both halves match as whole tokens: substring `see`
   fired inside `oversee` and `foresee`, and substring `item 7` fires inside `item 7a` —
   a pointer at a prior year's Item 7A names something the KB does not contain at all.
3. The hand-off must be essentially the whole text. Sentences doing no handing-off may
   total `POINTER_RESIDUE_MAX_WORDS` (15) words — measured against GS's five-word running
   header and JNJ's and PFE's page footers, while a single sentence of real market-risk
   prose already runs past it.

**The excusal never acts alone.** It may only fire for a filer hand-verified to answer
Item 7A by reference, recorded in `config.ITEM_7A_POINTER_FILERS` (BAC, GS, JNJ, JPM, LLY,
PFE). Any other firing is a `pointer_filer_is_recorded` finding: either the filer changed
its answer — verify against EDGAR, then record it there — or the detection misfired on a
broken parse. Both want a human; neither may be a silent drop. Note the corollary, which
is why cut 2's whole-token matching matters: for the six *recorded* filers this check
raises nothing, so the detection's own precision is the only thing standing between a
truncated real Item 7A and a silent drop.

Such a Section passes the gate and is **not** chunked: "Refer to pages 133-142" would
embed cleanly, retrieve for every market-risk question, and ground none of them. The
content is not lost — it lives inside Item 7, which is ingested — but it is **labelled
`Item 7`, not `Item 7A`**.

**Consequence for the golden set (ADR-0002).** Nine of fifteen companies have an Item 7A
in the KB; six do not. A question whose *answer* is market risk remains answerable for all
fifteen. A question that cites `Item 7A` as its source section is answerable for nine.
Authoring must use the committed verification artifact, not the assumption that every
company has all four Sections.

**5. A Section begins at its own heading — the rule hand-verification bought.**
The artifact worked. Fifty-nine of sixty rows verified; the sixtieth found what none of
the automated checks was looking for. JPM's Item 7 began on p.43, at the annual report's
Three-Year Summary of Consolidated Financial Highlights, where the filing's own
cross-reference puts Item 7 at **pp. 46–160**. That section was found, non-empty,
length-bounded, body-heavy and free of Item 8 content — every rule the gate had — and
still started in the wrong place.

Checking the other two banks first, as the reviewer asked, showed they were not one
pattern: **BAC starts at char 0** with its own `Item 7.` heading and needed nothing;
**GS** carried an 86-character running header. Ten further Sections across the Universe
opened on a stray `Table of Contents` or company-name line — the same defect, three
orders of magnitude smaller.

So: `edgar.trim_to_section_start` drops anything before the Section's own heading, and
`gate` asserts `section_starts_at_its_heading` on the result — which catches the Section
with no anchor anywhere, and, per §3's correction, not a trim that anchored on the wrong
line. Applied uniformly rather than to the banks alone: "starts at its heading" is either
an invariant of the KB or it is a special case, and a special case is not checkable.

Anchoring prefers the `Item N.` label and falls back to the Section's title only when the
label is absent entirely. That ordering is load-bearing: `Business` is too common a word
to trust ahead of `Item 1.`. Across all fifteen filings the fallback is reached exactly
once — JPM's Item 7, which prints no `Item 7` label anywhere.

Result: JPM Item 7 400,310 → 394,858 characters, now **pp. 46–161**.

**6. JPM's Item 7 runs one page past its cross-reference. Deliberate.**
Item 7 is defined here as *from its own heading to the start of Item 8* — content-anchored
and filer-independent, which is how the other fourteen companies already work. Page 161
is 3,091 characters of *Management's report on internal control over financial reporting*,
signed by the CEO and CFO, sitting in the gap between MD&A's end (p.160) and Item 8's
auditor report. It is management's own commentary, not audited statements, so it sits
closer to MD&A than to what follows.

Rejected: trimming to the p.160 footer. It would match the cross-reference exactly at both
ends, but it needs page-number parsing — filer-specific furniture, and JPM alternates
which side of the page the number sits on — to buy 3,091 characters, and it would
reintroduce the JPM special case rule 5 exists to remove. **Stated limitation**, recorded
with its page numbers so the golden set knows.

**Consequence for the artifact.** Re-rendering the checklist now carries a tick forward
for every Section byte-identical to the one that was verified, and unticks only what
moved. Without that, fixing one company's boundary would blank sixty hand-checked boxes —
and a reviewer facing sixty blank boxes either redoes the lot or, more likely, trusts a
file that no longer claims anything. Hand-written notes are carried through
unconditionally: a tick can be redone from the filing, a finding cannot.

**Verification artifact.** `docs/verification/section-starts.md`, generated by
`scripts/ingest_filings.py --section-starts`. The gate proves a Section is present,
non-empty, bounded, body-heavy, starts at its own heading and is free of the next Item —
none of which distinguishes the real Item 1A from a plausible mis-extraction that is also
all six. Only a person reading the excerpts against EDGAR closes that gap, which is why
the boxes are generated unticked.

**Outcome.** Hand-verified at 60/60 and committed ticked. Amendment §5 above records the
one row that failed and the rule it bought. The file in the repo predates the fixes in
§4's cut 2 and the fallback change in §3; the next full run will re-render it, and every
row whose text those changed will come back `CHANGED` and unticked for another look.
