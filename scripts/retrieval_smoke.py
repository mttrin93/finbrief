"""Run the five hand-written retrieval sanity queries against the ingested KB (#5).

    uv run python scripts/retrieval_smoke.py        # run the five, rewrite the artifact
    uv run python scripts/retrieval_smoke.py --k 8  # a different top-k, same queries

A script, not a test: it embeds each query through the paid model and reads the persisted
`filings` collection, and the suite is hermetic by contract (CLAUDE.md). Never invoke it
from a test. The queries, the verdicts and the report are `finbrief.retrieval.smoke`, which
the suite does cover.

PLAN.md §Phase 1 deferred this to Phase 2. It is a **wiring check, not an evaluation** —
ADR-0002's golden set (T9, issue #4) is the measurement artifact of record, and the report
says so in its own header.

Ingest first (`scripts/ingest_filings.py`): the collection has to hold what the queries ask
about, and it must have been embedded by the model this run queries with.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from finbrief.agent.agent import BASELINE_STRATEGY
from finbrief.config import get_settings
from finbrief.observability.logging_setup import configure_logging
from finbrief.retrieval.retrieve import retrieve
from finbrief.retrieval.smoke import SMOKE_QUERIES, SmokeCheck, render_smoke_report
from finbrief.retrieval.vectorstore import build_filings_store

#: The committed machine evidence of the most recent run. Relative to the working
#: directory, so run this script from the repo root — same convention as the ingest report.
SMOKE_REPORT = Path("docs/verification/retrieval-smoke.md")

# The strategy under check is the one the app ships (`agent.BASELINE_STRATEGY`), imported
# rather than restated: a smoke check that ran a strategy the app does not is checking
# nothing the demo depends on. It is not `settings.retrieval_strategy` for the reason that
# constant exists — the configured default is the pre-registered `hybrid` (ADR-0005), which
# `retrieve()` refuses until Phase 4. Phase 4 turns this into a `--strategy` flag.


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k",
        type=int,
        metavar="N",
        help="Chunks to retrieve per query (default: FINBRIEF_RETRIEVAL_K).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help=f"Print the report but leave {SMOKE_REPORT} as the last run wrote it.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    settings = get_settings()
    k = args.k if args.k is not None else settings.retrieval_k
    store = build_filings_store(settings)

    # Fail before spending anything on embeddings if the collection is empty: every verdict
    # below would be "retrieved nothing", which reads as a retrieval bug rather than as the
    # missing ingest it is.
    if not store.get(limit=1, include=[])["ids"]:
        print(
            f"The '{settings.chroma_dir}' filings collection is empty. Run "
            f"`uv run python scripts/ingest_filings.py` first (ADR-0007).",
            file=sys.stderr,
        )
        return 2

    print(f"Retrieving {len(SMOKE_QUERIES)} smoke queries · {BASELINE_STRATEGY} · k={k}\n")
    checks = tuple(
        SmokeCheck(
            query=query,
            contexts=retrieve(
                query.question, strategy=BASELINE_STRATEGY, k=k, store=store, settings=settings
            ),
        )
        for query in SMOKE_QUERIES
    )

    for index, check in enumerate(checks, start=1):
        print(f"{index}. {check.query.question}\n   {check.verdict}")

    report = render_smoke_report(
        checks,
        generated=(
            f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} · `scripts/retrieval_smoke.py`"
        ),
        strategy=BASELINE_STRATEGY,
        k=k,
        embedding_model=settings.embedding_model,
    )
    if args.no_write:
        print(f"\n(--no-write: {SMOKE_REPORT} left as the last run wrote it.)")
    else:
        SMOKE_REPORT.parent.mkdir(parents=True, exist_ok=True)
        SMOKE_REPORT.write_text(report, encoding="utf-8")
        print(f"\nSmoke report written to {SMOKE_REPORT}")

    failed = [check for check in checks if check.passed is False]
    if failed:
        print(
            f"\nSMOKE FAILED — {len(failed)} of "
            f"{sum(1 for c in checks if not c.is_control)} check(s) retrieved the wrong "
            f"filing or Section. This is a wiring failure, not a quality score.",
            file=sys.stderr,
        )
        return 1
    print("\nSMOKE PASSED. Not an evaluation — see ADR-0002 / issue #4 for those numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
