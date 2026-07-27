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
from enum import StrEnum

from finbrief.config import ITEM_7A_SECTION_FILERS, RetrievalStrategy
from finbrief.ingestion.model import Section
from finbrief.retrieval.retrieve import Context

#: How much of each retrieved chunk the report prints. Enough to recognise what was matched
#: without turning a five-query artifact into twenty pages of filing text.
EXCERPT_CHARS = 260


class Verdict(StrEnum):
    """What a check concluded — three states, so the control cannot be scored by accident.

    This was `passed: bool | None`, and the `None` was the problem: every truthiness test
    on it silently counted the control as a failure, and the one caller that got it right
    had to say `check.passed is False`. `CONTROL` is not "unknown", it is a third kind of
    result — the out-of-KB query records a distance band and asserts nothing (scoring it
    would need the floor T3 deliberately leaves unimplemented).

    The values are what the report prints, so the rendered artifact is unchanged.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    CONTROL = "control"


@dataclass(frozen=True, slots=True)
class SmokeQuery:
    """One hand-written question and where its answer must come from.

    `expect_ticker`/`expect_section` are `None` for the out-of-KB control, which is
    deliberately unscored — see `Verdict`.
    """

    question: str
    expect_ticker: str | None
    expect_section: Section | None
    #: What this query is here to catch. Printed in the report, because a check whose
    #: purpose is not written down is a check nobody dares delete or fix.
    why: str

    @property
    def is_control(self) -> bool:
        """Whether this query asserts nothing — the one predicate that decides it.

        Read by `expectation` here and by `SmokeCheck.outcome`, which each used to test a
        different half of the pair: `outcome` looked at `expect_ticker` alone while
        `expectation` required both to be `None`. A half-filled query — a ticker with no
        Section — was therefore scored `FAIL` forever *and* rendered as expecting `—`, a
        verdict no reader could act on (issue #5 review).
        """
        return self.expect_ticker is None or self.expect_section is None

    @property
    def expectation(self) -> str:
        """`TSLA Item 1A` — or an em dash for the control, which expects nothing.

        One property because the verdict line and the report's Expects column were building
        this label separately, and two spellings of the same expectation is how a table and
        the verdict beside it come to disagree.
        """
        if self.is_control:
            return "—"
        return f"{self.expect_ticker} {self.expect_section.value}"


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
        # Derived, not typed: this string lands verbatim in the committed artifact, and
        # "nine" was already a number nothing would have corrected (issue #5 review).
        why=(
            f"the {len(ITEM_7A_SECTION_FILERS)} filers that do have an Item 7A still "
            f"retrieve it (FY2026 filer)"
        ),
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
    def top(self) -> Context | None:
        return self.contexts[0] if self.contexts else None

    @property
    def outcome(self) -> Verdict:
        """What this check concluded. The control concludes `CONTROL`, never a score.

        The control's whole point is the distance band it records. Scoring it would turn
        this artifact into the retrieval-quality claim its header disclaims, and would need
        the very threshold this ticket deliberately leaves unimplemented.
        """
        if self.query.is_control:
            return Verdict.CONTROL
        top = self.top
        if top is None:
            return Verdict.FAIL
        matched = (
            top.ticker == self.query.expect_ticker and top.section is self.query.expect_section
        )
        return Verdict.PASS if matched else Verdict.FAIL

    @property
    def is_control(self) -> bool:
        """A control query has nothing in the KB to be right about."""
        return self.outcome is Verdict.CONTROL

    @property
    def verdict(self) -> str:
        """One line a reader can act on: what was expected, what arrived."""
        top = self.top
        if self.outcome is Verdict.CONTROL:
            observed = f"nearest {top.citation}" if top else "nothing retrieved"
            return f"{Verdict.CONTROL.value} — recorded, not asserted ({observed})"
        if top is None:
            return (
                f"{Verdict.FAIL.value} — expected {self.query.expectation}, retrieved nothing"
            )
        if self.outcome is Verdict.PASS:
            return f"{Verdict.PASS.value} — {top.ticker} {top.section.value}"
        return (
            f"{Verdict.FAIL.value} — expected {self.query.expectation}, top hit was "
            f"{top.ticker} {top.section.value}"
        )

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
    translate: bool,
    k: int,
    embedding_model: str,
) -> str:
    """Render the committed artifact for `docs/verification/retrieval-smoke.md`.

    Generated, never hand-authored (CLAUDE.md), and rewritten whole by every run — there is
    nothing here for a person to carry forward, unlike ADR-0007's checklist, because every
    verdict is mechanical.

    **Both retrieval switches are named, not just `strategy`.** `scripts/retrieval_smoke.py`
    pins this check to a configuration the app deliberately no longer ships (plain `vector`,
    translation off) and states that the report says so — with only the strategy printed, the
    artifact stated half its own configuration, and a reader could take these distances for the
    shipping default's numbers (issue #6 review).
    """
    scored = [check for check in checks if not check.is_control]
    controls = [check for check in checks if check.is_control]
    passed = sum(1 for check in checks if check.outcome is Verdict.PASS)
    failed = any(check.outcome is Verdict.FAIL for check in checks)
    headline = "SMOKE FAILED" if failed else "SMOKE PASSED"

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
        f"- Configuration: `{strategy.value}`, query translation "
        f"**{'on' if translate else 'off'}** · top-k `{k}` · embeddings `{embedding_model}`",
        f"- Outcome: **{headline}** — {passed}/{len(scored)} check(s) passed, "
        f"{len(controls)} control recorded",
        "",
        "## Checks",
        "",
        "| # | Query | Expects | Verdict | Why this query |",
        "|---|---|---|---|---|",
    ]
    for index, check in enumerate(checks, start=1):
        lines.append(
            f"| {index} | {check.query.question} | {check.query.expectation} | "
            f"{check.verdict} | {check.query.why} |"
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
        "per-bucket A/B and RAGAs evidence from Phase 7 (tickets T9/#4 and T10/#11), not a",
        "single run of five queries — Phase 4 shipped the retrieval strategy, not the",
        "measurement of it. This table is the calibration input for that decision.",
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
            # The ellipsis marks elision, so it is only earned when something was elided: a
            # short tail chunk printed whole was being reported as truncated.
            elided = "…" if len(context.body) > EXCERPT_CHARS else ""
            # `None`-tolerant like the band table above: `Context.distance` is `float | None`,
            # and a chunk only BM25 surfaced has no distance to print. Unreachable while
            # `SMOKE_STRATEGY` is pinned to `vector`, but rendering happens *after* the run has
            # paid for five query embeddings, so a `TypeError` here costs the whole report.
            distance = "—" if context.distance is None else f"{context.distance:.4f}"
            lines += [
                f"- **[{context.rank}] {context.citation}** · distance "
                f"{distance} · `{context.chunk_id}`",
                f"  > {excerpt}{elided}",
            ]
        lines.append("")

    return "\n".join(lines) + "\n"
