"""Rendering the gate's verdict for a human, and the checklist only a human can finish.

Three audiences. `render_gate_table` is read once, on a terminal, by whoever ran
ingestion. `render_ingest_report` is the committed machine evidence of the most recent
*full* run — the same table plus what the collection holds. `render_section_starts` is
ADR-0007's committed hand-verification artifact: it prints the first lines of every
extracted Section so a person can confirm the extractor landed on the real section rather
than a table-of-contents row.

The gate cannot make that judgement — it can only prove the text is long, wordy, and
bounded, all of which a plausible mis-extraction also is. The checklist is generated
unticked on purpose.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from finbrief.ingestion.gate import (
    TEN_K_FAMILY,
    GateFinding,
    form_family,
    incorporated_sections,
)
from finbrief.ingestion.model import ExtractedFiling, Section

#: How much of each Section's opening to show. Enough to recognise the heading and the
#: first sentence under it — which is exactly the evidence that distinguishes the real
#: section from a TOC row, and no more, since fifteen companies x four Sections is already
#: sixty judgements to make by hand.
SECTION_START_CHARS = 320

#: The generated excerpt block. `_parse_prior` cuts it out of a row before reading what is
#: left as hand-written notes — non-greedy, so it ends at its own closing fence rather than
#: swallowing everything up to the last one in the file.
_EXCERPT_BLOCK = re.compile(r"```text\n(.*?)\n```", re.DOTALL)


def render_gate_table(
    filings: Sequence[ExtractedFiling], findings: Sequence[GateFinding]
) -> str:
    """A company x Section grid of character counts, with failures called out beneath."""
    failed = {(f.ticker, f.section) for f in findings}
    header = f"{'':<7}{'':<9}" + "".join(f"{s.value:>12}" for s in Section)
    lines = [
        "Section-detection gate (ADR-0007) — characters extracted per Section",
        header,
        "-" * len(header),
    ]

    for filing in filings:
        ticker = filing.ref.ticker
        referenced = incorporated_sections(filing)
        cells = []
        for section in Section:
            text = filing.sections.get(section)
            marker = " FAIL" if (ticker, section) in failed else ""
            if section in referenced:
                # The marker belongs here too. The one finding that can attach to a
                # referenced Section is `pointer_filer_is_recorded` — the excusal firing
                # for a filer nobody verified — and a bare `->Item 7` cell reads as
                # "lawful, handled", which is the opposite of what the gate just said.
                cell = f"->Item 7{marker}"
            elif text:
                cell = f"{len(text):,}{marker}"
            else:
                cell = f"-{marker}"
            cells.append(f"{cell:>12}")
        # `form_family` against `TEN_K_FAMILY`, not a second startswith: this column has
        # to agree with the finding list below it, and two implementations of "is this a
        # 10-K" is how a table ends up contradicting the gate that produced it.
        annual = "" if form_family(filing.latest_annual_form) == TEN_K_FAMILY else " !ANNUAL"
        lines.append(
            f"{ticker:<7}{'FY' + str(filing.ref.fiscal_year):<9}" + "".join(cells) + annual
        )

    # Only the ones the gate actually called lawful. A referenced Section that produced a
    # finding is an unverified excusal, and counting it here would print "lawful, not
    # ingested" directly above the finding saying it is nothing of the kind.
    referenced_total = sum(
        1 for f in filings for s in incorporated_sections(f) if (f.ref.ticker, s) not in failed
    )
    lines.append("")
    if referenced_total:
        lines.append(
            f"{referenced_total} Section(s) incorporated by reference into Item 7 — "
            f"lawful, not ingested, listed in the verification artifact (ADR-0007)."
        )
    if findings:
        lines.append(f"{len(findings)} finding(s):")
        lines.extend(f"  - {finding}" for finding in findings)
    else:
        lines.append(f"All {len(filings) * len(Section)} company x Section checks passed.")

    return "\n".join(lines)


def render_ingest_report(
    filings: Sequence[ExtractedFiling],
    findings: Sequence[GateFinding],
    *,
    generated: str,
    store_counts: Mapping[str, int] | None,
    written: Mapping[str, int] | None = None,
    skipped: Sequence[str] = (),
) -> str:
    """Machine evidence of one ingestion run: the gate's verdict and what the store holds.

    The committed counterpart to `render_gate_table`'s terminal print (issue #3 review):
    "all Universe companies ingest" should be checkable from the repo, so
    `scripts/ingest_filings.py` rewrites `docs/verification/ingest-report.md` with this on
    every full-Universe run — and only those, since a `--tickers` subset or a `--dry-run`
    cannot speak to that claim. `store_counts` is read back from the collection rather
    than taken from the run's own writes — an idempotent re-run writes nothing and still
    has to prove what the knowledge base holds. `None` means no store was consulted: a
    dry run, or a failed gate that wrote nothing.
    """
    written = written or {}
    if findings:
        outcome = (
            f"**GATE FAILED** — {len(findings)} finding(s); nothing was written (ADR-0007)."
        )
    elif store_counts is None:
        outcome = "**GATE PASSED** — dry run; nothing was written, no store consulted."
    else:
        outcome = "**GATE PASSED**"

    lines = [
        "# Ingest report",
        "",
        "Machine evidence of the most recent ingestion run — rewritten by every",
        "full-Universe `scripts/ingest_filings.py` run, and by those only, since a",
        "`--tickers` subset or a `--dry-run` cannot speak to what the collection holds.",
        "Chunk counts are read back from the",
        "persisted collection after the run, not taken from the run's own writes, so an",
        "idempotent re-run that writes nothing still shows what the knowledge base holds.",
        "",
        f"- Generated: {generated}",
        f"- Outcome: {outcome}",
        "",
        "## Section-detection gate",
        "",
        "```text",
        render_gate_table(filings, findings),
        "```",
    ]

    if store_counts is not None:
        refs = {filing.ref.ticker: filing.ref for filing in filings}
        rows = [t for t in refs if t in store_counts]
        rows += sorted(set(store_counts) - set(rows))
        lines += [
            "",
            "## Chunks in the collection",
            "",
            "| Ticker | FY | Accession | Chunks | This run |",
            "|---|---|---|---:|---|",
        ]
        for ticker in rows:
            ref = refs.get(ticker)
            if ticker in written:
                action = f"wrote {written[ticker]}"
            elif ticker in skipped:
                action = "skipped (already ingested)"
            else:
                action = "—"
            fiscal = f"FY{ref.fiscal_year}" if ref else "—"
            accession = f"`{ref.accession}`" if ref else "—"
            lines.append(
                f"| {ticker} | {fiscal} | {accession} | {store_counts[ticker]:,} | {action} |"
            )
        lines.append(f"| **Total** | | | **{sum(store_counts.values()):,}** | |")

    return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class _PriorRow:
    """One Section's state in an earlier version of the checklist."""

    ticked: bool
    chars: int | None
    excerpt: str
    notes: tuple[str, ...]


def _parse_prior(markdown: str) -> dict[tuple[str, str], _PriorRow]:
    """Read an existing checklist back into `(ticker, section) -> state`.

    Parsing our own output rather than keeping a sidecar file: the artifact is the record,
    it is what gets reviewed in a diff, and a second file holding the ticks could drift
    from the boxes a reader is actually looking at.
    """
    prior: dict[tuple[str, str], _PriorRow] = {}
    ticker = ""
    for block in re.split(r"^(?=## )", markdown, flags=re.MULTILINE):
        header = re.match(r"## (\S+) — ", block)
        if header:
            ticker = header.group(1)
        for chunk in re.split(r"^(?=### )", block, flags=re.MULTILINE)[1:]:
            section = re.match(r"### (Item [\w]+)\.", chunk)
            status = re.search(r"^- \[( |x)\] *(.*)$", chunk, re.MULTILINE | re.I)
            if not (section and status):
                continue
            excerpt = _EXCERPT_BLOCK.search(chunk)
            chars = re.search(r"([\d,]+) characters extracted", status.group(2))
            # Everything in the row that this renderer did not write is a note, wherever
            # the verifier put it. Reading only as far as the excerpt fence dropped any
            # note written *under* the excerpt — which is the natural place to write after
            # reading one, and the loss was silent (issue #3 review).
            body = _EXCERPT_BLOCK.sub("", chunk[status.end() :])
            # A fence with no closing fence is a hand-edit gone wrong; the remainder is
            # excerpt, not notes, and re-emitting it as one would corrupt the row.
            notes = tuple(
                line.strip() for line in body.split("```")[0].splitlines() if line.strip()
            )
            prior[(ticker, section.group(1))] = _PriorRow(
                ticked=status.group(1).lower() == "x",
                chars=int(chars.group(1).replace(",", "")) if chars else None,
                excerpt=excerpt.group(1) if excerpt else "",
                notes=notes,
            )
    return prior


def checklist_tickers(markdown: str) -> frozenset[str]:
    """Which companies an existing checklist holds rows for.

    `scripts/ingest_filings.py` asks before writing: `render_section_starts` emits rows for
    the filings it is handed and no others, so re-rendering a sixty-row artifact from a
    `--tickers JPM` run would delete fourteen companies' ticks and notes outright. That is
    the loss the carry-forward exists to prevent, so the script refuses rather than
    performs it (issue #3 review).
    """
    return frozenset(ticker for ticker, _ in _parse_prior(markdown))


def render_section_starts(
    filings: Sequence[ExtractedFiling], previous: str | None = None
) -> str:
    """The hand-verification checklist of extracted Section starts (ADR-0007).

    Emitted with every box unticked on a first run. Ticking them is a human reading each
    excerpt against the filing on EDGAR — the whole value of the artifact is that a
    person, not the extractor, confirmed the extractor.

    Pass `previous` — the current contents of the file — and a re-render **carries ticks
    forward** for every Section whose length and excerpt are byte-identical, and unticks
    only what actually changed, flagged `CHANGED`. Without that, fixing one company's
    boundary would silently blank sixty hand-checked boxes and the reviewer would have to
    redo all of it to find the one row that moved. Notes a verifier wrote by hand are
    carried through untouched either way; this function must never be the reason someone's
    findings disappear.
    """
    prior = _parse_prior(previous) if previous else {}
    lines = [
        "# Hand-verification: extracted Section starts",
        "",
        "One-time verification artifact required by ADR-0007. The section-detection gate",
        "proves each Section is present, non-empty, length-bounded, body-heavy, starts at",
        "its own heading and is free of the next Item's content — none of which",
        "distinguishes the real Item 1A from a plausible mis-extraction that is also all",
        "six. Only reading the text does.",
        "",
        "**How to verify.** For each row, open the filing at the linked EDGAR URL, find",
        "the Item, and confirm the excerpt below is where that Item actually begins — not",
        "a table-of-contents row, not the tail of the previous Item, not a part summary.",
        "Tick the box when you have checked it against the source.",
        "",
        "Rows marked **incorporated by reference** are Sections the filer answered with a",
        "pointer to Item 7 rather than with text. They are deliberately not ingested — a",
        "pointer grounds nothing — and the content they point at is in the knowledge base",
        "under `Item 7`. Verify that the filing really does hand the Item off, rather than",
        "the extractor having landed on a stub.",
        "",
        "A re-render keeps every tick whose Section is byte-identical to the version that",
        "was verified, and unticks the rest. The rows flagged as changed, in bold on their",
        "status line, are the only ones needing another look. Anything you write inside a",
        "row — above or below its excerpt — is carried through every re-render verbatim.",
        "",
        f"Generated by `scripts/ingest_filings.py --section-starts`. {len(filings)} "
        f"filing(s), {len(filings) * len(Section)} Sections to verify.",
        "",
    ]

    for filing in filings:
        ref = filing.ref
        lines += [
            f"## {ref.ticker} — FY{ref.fiscal_year} {ref.form}",
            "",
            f"- Accession: `{ref.accession}`  ·  filed {ref.filing_date}",
            f"- EDGAR: {ref.url or '(no URL recorded)'}",
            "",
        ]
        referenced = incorporated_sections(filing)
        for section in Section:
            text = filing.sections.get(section)
            was = prior.get((ref.ticker, section.value))
            lines.append(f"### {section.value}. {section.heading}")
            lines.append("")

            if not text:
                lines.append(
                    "- [ ] **NOT EXTRACTED** — nothing to verify; the gate failed this."
                )
                lines += _carried_notes(was)
                lines.append("")
                continue

            excerpt = _excerpt(text)
            # An exact character count, not `chars in (None, len(text))`. A prior row with
            # no recorded count is a row this renderer did not write — an older format, or
            # a hand-edit that broke the status line — and "we cannot tell whether it
            # moved" must resolve to *unticked*, not to a tick nobody re-earned.
            unchanged = bool(was and was.excerpt == excerpt and was.chars == len(text))
            box = "x" if (unchanged and was and was.ticked) else " "
            changed = " **CHANGED**" if was and not unchanged else ""
            was_chars = (
                f" — was {was.chars:,}" if changed and was and was.chars is not None else ""
            )

            if section in referenced:
                # The character count is load-bearing here too: the excerpt shows only
                # the first 320 characters and a pointer may run to 1,500, so without
                # the count a pointer whose tail changed would keep a tick no human has
                # re-checked.
                lines.append(
                    f"- [{box}] Verified **incorporated by reference — not ingested** — "
                    f"{len(text):,} characters extracted{changed}{was_chars}. Confirm "
                    f"the filing really does hand this Item off to Item 7 (which *is* "
                    f"ingested), rather than the extractor having found a stub."
                )
            else:
                lines.append(
                    f"- [{box}] Verified — {len(text):,} characters extracted"
                    f"{changed}{was_chars}"
                )
            lines += _carried_notes(was)
            lines.append("")
            lines.append("```text")
            lines.append(excerpt)
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def _carried_notes(was: _PriorRow | None) -> list[str]:
    """Re-emit whatever a verifier wrote under this row, verbatim.

    Their findings are the output of the work this artifact exists to collect. Losing them
    to a regeneration would be worse than losing the ticks, because a tick can be redone
    from the filing and a note cannot.
    """
    return ["", *was.notes] if was and was.notes else []


def _excerpt(text: str) -> str:
    """The Section's opening, cut at a word boundary."""
    if len(text) <= SECTION_START_CHARS:
        return text
    head = text[:SECTION_START_CHARS]
    # `rfind` returns -1 for an opening with no space in it at all, and `head[:-1]` would
    # silently drop a character instead of cutting at a boundary there is none of.
    boundary = head.rfind(" ")
    return (head[:boundary] if boundary > 0 else head) + " …"
