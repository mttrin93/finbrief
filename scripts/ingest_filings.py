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
from pathlib import Path

from finbrief.config import TICKERS, UNIVERSE, get_settings
from finbrief.ingestion.edgar import configure_edgar
from finbrief.ingestion.gate import check_filing
from finbrief.ingestion.pipeline import fetch_filings, ingest
from finbrief.ingestion.reporting import render_gate_table, render_section_starts
from finbrief.observability.logging_setup import configure_logging
from finbrief.retrieval.vectorstore import build_filings_store


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
        help="Re-embed filings already in the collection (after a chunker change).",
    )
    parser.add_argument(
        "--section-starts",
        type=Path,
        metavar="PATH",
        help="Write the hand-verification checklist of extracted Section starts here.",
    )
    parser.add_argument(
        "--overwrite-checklist",
        action="store_true",
        help="Regenerate --section-starts even if it already holds ticked verifications.",
    )
    return parser.parse_args(argv)


def _write_section_starts(path: Path, filings, *, overwrite: bool) -> bool:
    """Write the checklist, refusing to overwrite a human's completed ticks.

    ADR-0007 calls this a *one-time* artifact, and its whole value is that a person read
    every excerpt against EDGAR. Regenerating it emits sixty fresh unticked boxes — so a
    routine `--section-starts` on a later run would silently erase that work and leave
    something that still looks like a verification artifact. Refuse instead, loudly.
    """
    if path.exists() and "- [x]" in path.read_text(encoding="utf-8").lower():
        if not overwrite:
            print(
                f"{path} already holds ticked verifications. Refusing to overwrite them "
                f"— they are hand-done work this script cannot reproduce. Pass "
                f"--overwrite-checklist if the filings really have changed.",
                file=sys.stderr,
            )
            return False
        print(f"Overwriting ticked verifications in {path}, as asked.", file=sys.stderr)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_section_starts(filings), encoding="utf-8")
    print(f"\nHand-verification checklist written to {path}")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    settings = get_settings()
    configure_edgar(settings)

    tickers = args.tickers or [company.ticker for company in UNIVERSE]
    unknown = sorted(set(tickers) - TICKERS)
    if unknown:
        # The KB scope *is* the Universe (CONTEXT.md). A typo'd ticker that silently
        # ingested would put a company in the index that no peer set contains.
        print(f"Not in the Universe: {', '.join(unknown)}", file=sys.stderr)
        return 2

    print(f"Fetching {len(tickers)} filing(s) from EDGAR: {', '.join(tickers)}\n")
    filings = fetch_filings(tickers)

    findings = [finding for filing in filings for finding in check_filing(filing)]
    print(render_gate_table(filings, findings))

    if args.section_starts and not _write_section_starts(
        args.section_starts, filings, overwrite=args.overwrite_checklist
    ):
        return 2

    if findings:
        print(
            f"\nGATE FAILED — {len(findings)} finding(s). Nothing was written (ADR-0007).",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print("\nGATE PASSED. --dry-run: nothing written.")
        return 0

    store = build_filings_store(settings)
    report = ingest(filings, store=store, force=args.force)

    print(f"\nGATE PASSED. Wrote {report.total_chunks} chunks to '{settings.chroma_dir}'.")
    for ticker, count in report.chunks_written.items():
        print(f"  {ticker}: {count} chunks")
    if report.skipped:
        print(f"  skipped (already ingested): {', '.join(report.skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
