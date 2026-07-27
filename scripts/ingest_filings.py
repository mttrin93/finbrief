"""Build the `filings` knowledge base from the Universe's latest 10-Ks (ADR-0007, #3).

    uv run python scripts/ingest_filings.py --tickers AAPL        # the tracer
    uv run python scripts/ingest_filings.py                       # the whole Universe
    uv run python scripts/ingest_filings.py --dry-run             # gate only, no writes
    uv run python scripts/ingest_filings.py --section-starts docs/verification/...md

A script, not a test: it talks to EDGAR and pays for embeddings, and the suite is hermetic
by contract (CLAUDE.md). The gate it runs is the same `ingestion.gate` the suite covers.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from finbrief.config import TICKERS, UNIVERSE, get_settings
from finbrief.ingestion.edgar import configure_edgar, fetch_filing
from finbrief.ingestion.gate import check_filing
from finbrief.ingestion.pipeline import ingest
from finbrief.ingestion.reporting import (
    checklist_tickers,
    render_gate_table,
    render_ingest_report,
    render_section_starts,
)
from finbrief.observability.logging_setup import configure_logging
from finbrief.retrieval.vectorstore import build_filings_store, chunk_counts_by_ticker

#: The committed machine evidence of the most recent *full* run — gate table plus what the
#: collection holds (issue #3 review). Relative to the working directory, so run this
#: script from the repo root.
INGEST_REPORT = Path("docs/verification/ingest-report.md")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="Universe tickers to ingest (default: the whole Universe).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and gate only — write nothing to Chroma and call no embedding API.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed every filing (after an embedding-model change; text changes "
        "are detected on their own).",
    )
    parser.add_argument(
        "--section-starts",
        type=Path,
        metavar="PATH",
        help="Write the hand-verification checklist of extracted Section starts here.",
    )
    return parser.parse_args(argv)


def _write_section_starts(path: Path, filings) -> None:
    """Write the checklist, carrying forward every tick and note that still applies.

    Not an overwrite. `render_section_starts` re-reads what is already there and keeps a
    tick whenever the Section is byte-identical to the one that was verified, so fixing a
    boundary on one company costs the reviewer exactly the rows that moved rather than all
    sixty. Hand-written notes are carried through unconditionally.

    That carry-forward only reaches companies *this run fetched*, so a subset run is
    refused rather than written: `--tickers JPM --section-starts docs/verification/…` is the
    natural command while iterating on one company's boundary, and it would replace a
    sixty-row artifact with four rows, deleting fifty-six hand-earned ticks and every note
    attached to them (issue #3 review). The same protection `_write_ingest_report` gets, for
    the file where the loss is not recoverable by re-running anything.

    A `--dry-run` is deliberately still allowed to write: the checklist is made of fetch and
    gate output only, so a dry run is the cheap, key-free way to regenerate it.
    """
    previous = path.read_text(encoding="utf-8") if path.exists() else None
    if previous and previous.strip():
        held = checklist_tickers(previous)
        if not held:
            # The guard below asks "which companies would this run drop?" and reads the
            # answer out of the file itself — so a file it cannot parse answers "none" and
            # waves the overwrite through, in precisely the case where the carry-forward
            # can rescue nothing either, since it reads the same rows. An older format, or
            # a hand-edit that broke the row shape, lands here (issue #3 review).
            print(
                f"\n{path} exists but no verification rows could be read out of it — an "
                f"older format, or a hand-edit that broke the `- [ ]` row shape. Every "
                f"tick and note in it would be lost. Left untouched — move it aside "
                f"deliberately if you mean to start over.",
                file=sys.stderr,
            )
            return
        absent = sorted(held - {filing.ref.ticker for filing in filings})
        if absent:
            print(
                f"\n{path} holds hand-verified rows for {', '.join(absent)}, which this run "
                f"did not fetch, and a re-render would delete them. Left untouched — re-run "
                f"without --tickers, or point --section-starts at a new path.",
                file=sys.stderr,
            )
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_section_starts(filings, previous)
    path.write_text(rendered, encoding="utf-8")

    kept = rendered.count("- [x]")
    changed = rendered.count("**CHANGED**")
    print(f"\nHand-verification checklist written to {path}")
    print(f"  {kept} tick(s) carried forward · {changed} row(s) changed and need re-checking")


def _write_ingest_report(filings, findings, *, generated, store_counts, **run) -> None:
    """Rewrite the committed machine evidence of this run, whatever its outcome.

    A full run overwrites it — the file is a record of the most recent one, not a log —
    so the version in git is whatever the last committed run proved. `main` calls this
    only for a full-Universe, non-dry run: the file's claim is "all fifteen ingest", and
    a `--tickers AAPL` run would replace the fifteen-row gate table with one row while a
    `--dry-run` would delete the chunk table outright. Losing the evidence to the two
    commands most likely to be run casually is not a trade worth making.
    """
    INGEST_REPORT.parent.mkdir(parents=True, exist_ok=True)
    INGEST_REPORT.write_text(
        render_ingest_report(
            filings, findings, generated=generated, store_counts=store_counts, **run
        ),
        encoding="utf-8",
    )
    print(f"\nIngest report written to {INGEST_REPORT}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    tickers = args.tickers or [company.ticker for company in UNIVERSE]
    unknown = sorted(set(tickers) - TICKERS)
    if unknown:
        # The KB scope *is* the Universe (CONTEXT.md). A typo'd ticker that silently
        # ingested would put a company in the index that no peer set contains.
        print(f"Not in the Universe: {', '.join(unknown)}", file=sys.stderr)
        return 2

    # No `get_settings()` yet, deliberately: the full Settings hard-requires
    # OPENROUTER_API_KEY, which fetching and gating never spend. It is resolved below,
    # only on the branch that actually builds the store — so a --dry-run (or a run the
    # gate fails) works on a machine that has an EDGAR identity and no paid key.
    configure_edgar()

    print(f"Fetching {len(tickers)} filing(s) from EDGAR: {', '.join(tickers)}\n")
    filings = tuple(fetch_filing(ticker) for ticker in tickers)

    findings = [finding for filing in filings for finding in check_filing(filing)]
    print(render_gate_table(filings, findings))

    if args.section_starts:
        _write_section_starts(args.section_starts, filings)

    generated = (
        f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} · `scripts/ingest_filings.py`"
    )
    # A partial or dry run has nothing to say about "all fifteen ingest", so it must not
    # be what the committed file says. It still prints its table above.
    full_run = args.tickers is None and not args.dry_run
    if not full_run:
        print(f"\n(Partial or dry run — {INGEST_REPORT} left as the last full run wrote it.)")

    if findings:
        if full_run:
            _write_ingest_report(filings, findings, generated=generated, store_counts=None)
        print(
            f"\nGATE FAILED — {len(findings)} finding(s). Nothing was written (ADR-0007).",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print("\nGATE PASSED. --dry-run: nothing written.")
        return 0

    settings = get_settings()
    store = build_filings_store(settings)
    report = ingest(filings, store=store, force=args.force)
    if full_run:
        _write_ingest_report(
            filings,
            findings,
            generated=generated,
            store_counts=chunk_counts_by_ticker(store),
            written=report.chunks_written,
            skipped=report.skipped,
        )

    print(f"\nGATE PASSED. Wrote {report.total_chunks} chunks to '{settings.chroma_dir}'.")
    for ticker, count in report.chunks_written.items():
        print(f"  {ticker}: {count} chunks")
    if report.skipped:
        print(f"  skipped (already ingested): {', '.join(report.skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
