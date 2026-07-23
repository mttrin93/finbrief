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

**Rejected alternative.** Full-filing ingestion — rejected because TOC/boilerplate
pollutes the exact-identifier bucket and section labels become unreliable.
