"""The five hand-written retrieval sanity queries, their verdicts, and their report.

PLAN.md §Phase 1 deferred a "top-k retrieval sanity check for 5 hand-written queries" to
Phase 2, because it needs a retrieval chain to exist. This is it.

**A smoke check, not an evaluation.** It answers "is retrieval wired to the knowledge base
we think we ingested?" — one query per KB property that could silently be wrong. It says
nothing about retrieval *quality*: five hand-picked questions, no source-separated ground
truth, no strata, no baseline to compare against. ADR-0002's stratified golden set (ticket
T9, issue #4) is the measurement artifact of record, and the report this module renders says
so in its own header so no reader can mistake one for the other.

Pure. The queries and the verdicts live here so they are testable without a key; the run
that embeds them lives in `scripts/retrieval_smoke.py`, which is non-hermetic by nature.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from finbrief.config import RetrievalStrategy
from finbrief.ingestion.model import Section
from finbrief.retrieval.retrieve import Context

#: How much of each retrieved chunk the report prints. Enough to recognise what was matched
#: without turning a five-query artifact into twenty pages of filing text.
EXCERPT_CHARS = 260


@dataclass(frozen=True, slots=True)
class SmokeQuery:
    """One hand-written question and where its answer must come from.

    `expect_ticker`/`expect_section` are `None` for the out-of-KB control, which is
    deliberately unscored — see `SmokeCheck.passed`.
    """

    question: str
    expect_ticker: str | None
    expect_section: Section | None
    #: What this query is here to catch. Printed in the report, because a check whose
    #: purpose is not written down is a check nobody dares delete or fix.
    why: str


#: The five. Each one is a KB property that could be wrong without any test failing:
#:
#: 1. the demo's own question reaches the right filer's risk factors at all;
#: 2. a **pointer filer's** market-risk question lands in `Item 7` — the sharpest edge in
#:    the KB (ADR-0007 amendment §4), and the claim the UI's scope panel makes;
#: 3. a company that *does* have an `Item 7A` retrieves it, so the six exceptions have not
#:    quietly become the rule — and NVDA is also the one filer whose fiscal year is not
#:    2025, so a year printed in a citation is checked against the filer's own;
#: 4. a business-description question stays in `Item 1` rather than drifting to the risk
#:    factors that discuss the same products;
#: 5. an out-of-KB question, as the control: vector search always returns `k` chunks from a
#:    non-empty collection, so this records what "nothing to ground it" looks like in
#:    distance terms rather than asserting a threshold on it.
SMOKE_QUERIES: tuple[SmokeQuery, ...] = (
    SmokeQuery(
        question="What are the main risk factors for Tesla?",
        expect_ticker="TSLA",
        expect_section=Section.RISK_FACTORS,
        why="demo step 1 — a risk question reaches the filer's own Item 1A",
    ),
    SmokeQuery(
        question="How does JPMorgan manage its market risk exposure?",
        expect_ticker="JPM",
        expect_section=Section.MDA,
        why="a pointer filer's market risk lives in Item 7 (ADR-0007 §4), not Item 7A",
    ),
    SmokeQuery(
        question=(
            "What is NVIDIA's exposure to interest rate risk on its investment portfolio?"
        ),
        expect_ticker="NVDA",
        expect_section=Section.MARKET_RISK,
        why="the nine filers that do have an Item 7A still retrieve it (FY2026 filer)",
    ),
    SmokeQuery(
        question="What products and services does Apple sell?",
        expect_ticker="AAPL",
        expect_section=Section.BUSINESS,
        why="a business question stays in Item 1 and does not drift into Item 1A",
    ),
    SmokeQuery(
        question="What is Nestle's dividend policy?",
        expect_ticker=None,
        expect_section=None,
        why="out-of-KB control — calibrates the distance band, asserts nothing",
    ),
)


@dataclass(frozen=True, slots=True)
class SmokeCheck:
    """What one query retrieved, and whether that met its expectation."""

    query: SmokeQuery
    contexts: tuple[Context, ...]

    @property
    def is_control(self) -> bool:
        """A control query has nothing in the KB to be right about."""
        return self.query.expect_ticker is None

    @property
    def top(self) -> Context | None:
        return self.contexts[0] if self.contexts else None

    @property
    def passed(self) -> bool | None:
        """`True`/`False` for a check, `None` for the control — which is never scored.

        The control's whole point is the distance band it records. Scoring it would turn
        this artifact into the retrieval-quality claim its header disclaims, and would need
        the very threshold this ticket deliberately leaves unimplemented.
        """
        if self.is_control:
            return None
        top = self.top
        if top is None:
            return False
        return (
            top.ticker == self.query.expect_ticker and top.section is self.query.expect_section
        )

    @property
    def verdict(self) -> str:
        """One line a reader can act on: what was expected, what arrived."""
        if self.is_control:
            top = self.top
            observed = f"nearest {top.citation}" if top else "nothing retrieved"
            return f"control — recorded, not asserted ({observed})"
        expected = f"{self.query.expect_ticker} {self.query.expect_section.value}"
        top = self.top
        if top is None:
            return f"FAIL — expected {expected}, retrieved nothing"
        if self.passed:
            return f"PASS — {top.ticker} {top.section.value}"
        return f"FAIL — expected {expected}, top hit was {top.ticker} {top.section.value}"

    @property
    def nearest(self) -> float | None:
        """The top hit's distance, or `None` when nothing was retrieved."""
        return self.contexts[0].distance if self.contexts else None

    @property
    def furthest(self) -> float | None:
        """The last hit's distance. Retrieval is nearest-first, so this closes the band."""
        return self.contexts[-1].distance if self.contexts else None


def render_smoke_report(
    checks: Sequence[SmokeCheck],
    *,
    generated: str,
    strategy: RetrievalStrategy,
    k: int,
    embedding_model: str,
) -> str:
    """Render the committed artifact for `docs/verification/retrieval-smoke.md`.

    Generated, never hand-authored (CLAUDE.md), and rewritten whole by every run — there is
    nothing here for a person to carry forward, unlike ADR-0007's checklist, because every
    verdict is mechanical.
    """
    scored = [check for check in checks if not check.is_control]
    controls = [check for check in checks if check.is_control]
    passed = sum(1 for check in scored if check.passed)
    outcome = "SMOKE PASSED" if passed == len(scored) else "SMOKE FAILED"

    lines = [
        "# Retrieval smoke check",
        "",
        "**This is not an evaluation.** It is a wiring check on `retrieve()`: five",
        "hand-written queries, one per property of the knowledge base that could be wrong",
        "without any test failing. It has no stratified buckets, no source-separated ground",
        "truth, and no baseline to compare against, so **no number in this file may be cited",
        "as a retrieval-quality claim**. The measurement artifact of record is ADR-0002's",
        "golden set with per-bucket RAGAs and the A/B matrix (ticket T9, issue #4).",
        "",
        "Generated by every `scripts/retrieval_smoke.py` run, never hand-authored.",
        "",
        f"- Generated: {generated}",
        f"- Strategy: `{strategy.value}` · top-k `{k}` · embeddings `{embedding_model}`",
        f"- Outcome: **{outcome}** — {passed}/{len(scored)} check(s) passed, "
        f"{len(controls)} control recorded",
        "",
        "## Checks",
        "",
        "| # | Query | Expects | Verdict | Why this query |",
        "|---|---|---|---|---|",
    ]
    for index, check in enumerate(checks, start=1):
        expects = (
            "—"
            if check.is_control
            else f"{check.query.expect_ticker} {check.query.expect_section.value}"
        )
        lines.append(
            f"| {index} | {check.query.question} | {expects} | {check.verdict} | "
            f"{check.query.why} |"
        )

    lines += [
        "",
        "## Observed distance ranges",
        "",
        "Chroma L2 distance, **lower is nearer** — not a normalised similarity, and not",
        "comparable across embedding models or collections.",
        "",
        "| # | Query | In KB | Nearest | Furthest |",
        "|---|---|---|---:|---:|",
    ]
    for index, check in enumerate(checks, start=1):
        in_kb = "control" if check.is_control else "yes"
        nearest = f"{check.nearest:.4f}" if check.nearest is not None else "—"
        furthest = f"{check.furthest:.4f}" if check.furthest is not None else "—"
        lines.append(f"| {index} | {check.query.question} | {in_kb} | {nearest} | {furthest} |")

    lines += [
        "",
        "**Calibration data, not a threshold.** These bands exist because",
        "`prompts.NO_CONTEXT_FALLBACK` — the retrieval-level tier that returns a fallback",
        "instead of generating — can only fire when the collection yields *nothing*, and a",
        "non-empty Chroma always returns `k` chunks. So an out-of-KB question currently",
        "reaches the model with `k` far-away contexts and is refused by the system prompt",
        "rather than by the tier. A distance floor would make the tier bite; the floor is",
        "**deliberately unimplemented in T3**, because choosing the number needs the",
        "per-bucket A/B and RAGAs evidence from Phase 4 and Phase 7, not a single run of",
        "five queries. This table is the calibration input for that decision.",
        "",
        "## Retrieved chunks",
        "",
        "Every chunk each query retrieved, so a reader can check a verdict against the",
        "primary source.",
        "",
    ]
    for index, check in enumerate(checks, start=1):
        lines += [f"### {index}. {check.query.question}", "", f"{check.verdict}", ""]
        if not check.contexts:
            lines += ["Nothing retrieved.", ""]
            continue
        for context in check.contexts:
            excerpt = " ".join(context.body[:EXCERPT_CHARS].split())
            lines += [
                f"- **[{context.rank}] {context.citation}** · distance "
                f"{context.distance:.4f} · `{context.chunk_id}`",
                f"  > {excerpt}…",
            ]
        lines.append("")

    return "\n".join(lines) + "\n"
