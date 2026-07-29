"""Run the evaluation harness and rewrite its evidence artifact (T10, #11).

    uv run python scripts/evaluate.py                   # every stage, every arm, 28 rows
    uv run python scripts/evaluate.py --rows S1,T2      # a two-question smoke over all arms
    uv run python scripts/evaluate.py --stage judge     # re-judge only; replay the rest
    uv run python scripts/evaluate.py --no-write        # print the artifact, do not commit it

The **fifth** non-hermetic entry point, and the most expensive: a full six-arm run makes ~1,600
judge calls, 112 generations and ~450 embedding requests. **Never invoke it from a test.**

**Resumable, by construction.** Every paid cell is addressed by its own inputs under
`--cache-dir` (default `data/eval-cache`, gitignored), so a run killed by a 429 or a closed
laptop resumes where it stopped and pays only for the cell it died inside. A second full run
costs nothing. `evaluation/cache.py` has the invariants; `tests/test_eval_resume.py` has the
kill-and-resume proof.

**The sink must be on.** `FINBRIEF_LOG_FILE` is off unless named (ADR-0011) and `.env.example`
ships it commented out, so an evaluation run with no log is the *likely* state rather than an
unlucky one — and the latency half of ADR-0005's dominance test is measured from that log.
This script therefore refuses to start without it rather than reporting a p50 over zero
samples, which is the absence-as-measurement failure this repo enforces against everywhere
else.

**What a stage costs, and what it invalidates.**

- `resolve` — one planner call per row. Re-run when `query_translation`'s parser moves.
- `retrieve` — up to five embedding requests per cell. Re-run on a re-ingest, or a changed `k`.
- `answer` — one generation per judged cell. Re-run when `prompts.SYSTEM_PROMPT` or the chat
  model changes.
- `judge` — `judge.calls_per_row` per cell. Re-run when the judge model or ragas changes.
- `report` — free, and therefore always re-run.

Each stage keys its cache on exactly those inputs, so a change to one re-pays for one.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from finbrief.config import Settings, get_settings, load_env, resolve_log_file
from finbrief.evaluation import hypotheses, latency, pipeline, report, variants
from finbrief.evaluation import judge as judging
from finbrief.evaluation.arms import ABLATION_ARMS, SCORED_ARMS, SHIPPING_DEFAULT, Arm
from finbrief.evaluation.cache import Cache
from finbrief.evaluation.loader import GoldenQuestion, GoldenSet, load_golden_set
from finbrief.evaluation.pipeline import Cell
from finbrief.observability.logging_setup import configure_logging
from finbrief.retrieval.vectorstore import default_filings_store

logger = logging.getLogger("finbrief.evaluation")

#: The committed evidence of the most recent run. Relative to the working directory, like every
#: other artifact's path — run this from the repo root.
EVALUATION_REPORT = Path("docs/verification/evaluation.md")

DEFAULT_CACHE_DIR = Path("data/eval-cache")


class SinkNotEnabled(RuntimeError):
    """`FINBRIEF_LOG_FILE` is unset, so this run would leave no record to measure from."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        default="",
        help="comma-separated golden-set ids to score (default: all 28). A smoke run.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=report.ALL_STAGES,
        help=(
            "run only this stage, repeatable. Skipped stages replay from the cache; a stage "
            "with nothing cached leaves its numbers absent and the artifact carries a "
            "PARTIAL RUN banner."
        ),
    )
    parser.add_argument(
        "--ablations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="score the two planner-off cells (ADR-0004 §7, ADR-0005 §2). On by default.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"where paid cells are cached (default {DEFAULT_CACHE_DIR}, gitignored)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help=(
            "how many cells a stage resolves at once (default 6). 1 is serial. Measured: the "
            "judge stage ran at 4.7 cells/min serially, which is 99 minutes for a six-arm run."
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print the artifact instead of rewriting the committed one",
    )
    parser.add_argument(
        "--allow-missing-sink",
        action="store_true",
        help=(
            "proceed with FINBRIEF_LOG_FILE unset. The latency half of ADR-0005's dominance "
            "test is then unmeasurable and the artifact says so."
        ),
    )
    return parser.parse_args(argv)


def selected_rows(golden: GoldenSet, ids: str) -> tuple[GoldenQuestion, ...]:
    """The rows named by `--rows`, or all of them. Order follows the set, not the flag."""
    if not ids.strip():
        return tuple(golden)
    wanted = {value.strip() for value in ids.split(",") if value.strip()}
    rows = tuple(row for row in golden if row.id in wanted)
    missing = wanted - {row.id for row in rows}
    if missing:
        raise KeyError(f"no golden-set rows with ids {sorted(missing)}")
    return rows


def require_sink(*, allow_missing: bool) -> Path | None:
    """The sink's path, or raise — because a p50 over zero samples is not a budget met.

    The one check in this script that runs before anything is spent. `.env.example` ships
    `FINBRIEF_LOG_FILE` commented out and `configure_logging()` takes no arguments, so "the
    sink is off" is the state a fresh checkout evaluates in; discovering it *after* a
    30-minute paid run would mean re-running the whole thing to get the latency numbers
    ADR-0005 needs.
    """
    path = resolve_log_file(os.environ)
    if path is not None:
        return path
    if allow_missing:
        logger.warning(
            "FINBRIEF_LOG_FILE is unset: this run records no latency or token samples, and the "
            "artifact will report them as absent rather than as met budgets."
        )
        return None
    raise SinkNotEnabled(
        "FINBRIEF_LOG_FILE is unset, so this run would emit no persisted events and "
        "ADR-0005's '<=1.5s p50 added by translation' could not be measured from it. Uncomment "
        "the line in .env.example (or export it) and re-run; pass --allow-missing-sink to "
        "proceed anyway and have the artifact report those figures as absent."
    )


def run(args: argparse.Namespace) -> str:
    """Run the requested stages and return the rendered artifact."""
    stages = tuple(args.stage) if args.stage else report.ALL_STAGES
    # **Before the sink check, not after.** `resolve_log_file` reads a mapping, and until
    # `load_env()` has run that mapping does not contain anything `.env` sets — so the check
    # would refuse a run whose sink was configured exactly where `.env.example` tells you to
    # configure it. `get_settings()` loads it too, and idempotently, but it is called after
    # this point.
    load_env()
    sink = require_sink(allow_missing=args.allow_missing_sink)
    configure_logging()
    logger.info("evaluation run starting: stages=%s sink=%s", ",".join(stages), sink)

    settings = get_settings()
    golden = load_golden_set()
    rows = selected_rows(golden, args.rows)
    cache = Cache(args.cache_dir)
    store = default_filings_store(settings)
    fingerprint = pipeline.collection_fingerprint(store)
    k = settings.retrieval_k

    variant_set = _resolve(rows, settings=settings, cache=cache, run_stage="resolve" in stages)
    arms: tuple[Arm, ...] = SCORED_ARMS + (ABLATION_ARMS if args.ablations else ())
    judge = judging.build_judge(settings) if "judge" in stages else None
    embeddings = judging.build_judge_embeddings(settings) if "judge" in stages else None

    cells: list[Cell] = []
    for arm in arms:
        logger.info("arm %s: retrieving %d rows", arm.name, len(rows))
        retrievals = pipeline.retrieve_cells(
            rows,
            arm,
            store=store,
            settings=settings,
            variants=variant_set,
            cache=cache,
            k=k,
            fingerprint=fingerprint,
            workers=args.workers,
        )
        answers = (
            pipeline.answer_cells(
                rows,
                retrievals,
                arm,
                store=store,
                settings=settings,
                variants=variant_set,
                cache=cache,
                k=k,
                workers=args.workers,
            )
            if "answer" in stages
            else tuple(None for _ in rows)
        )
        metrics = judging.METRICS if arm.judged else judging.RETRIEVAL_METRICS
        judged = (
            pipeline.judge_cells(
                rows,
                retrievals,
                answers,
                metrics=metrics,
                settings=settings,
                cache=cache,
                judge=judge,
                embeddings=embeddings,
                workers=args.workers,
            )
            if judge is not None
            else tuple({} for _ in rows)
        )
        cells.extend(pipeline.build_cells(rows, arm, retrievals, answers, judged, k=k))

    provenance = report.Provenance(
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        judge_model=settings.judge_model,
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        planner_model=variant_set.planner_model if variant_set else settings.chat_model,
        ragas_version=judging.ragas_version(),
        k=k,
        collection_ingest_run=golden.collection_ingest_run,
        collection_fingerprint=fingerprint,
        golden_set_rows=len(rows),
        stages=stages,
        cache={kind: cache.stats(kind) for kind in cache.kinds()},
    )
    sections = (*_findings_sections(cells, sink=sink), _headline(cells, cache))
    return report.render_report(
        provenance=provenance,
        golden=golden,
        cells=cells,
        arms=SCORED_ARMS,
        ablations=ABLATION_ARMS if args.ablations else (),
        sections=sections,
    )


def _findings_sections(
    cells: Sequence[Cell], *, sink: Path | None
) -> tuple[tuple[str, str], ...]:
    """The pre-registered half of the artifact: predictions, triggers, latency.

    Each block is rendered even when the run could not settle it — a prediction reported only
    when it resolved is a prediction a reader cannot tell was evaluated, and ADR-0005 §4's
    trigger fires on an *absence* of gain, which is the shape that goes unnoticed when it is
    not printed.
    """
    sections = [
        (
            "## Pre-registered hypotheses — prediction, then measurement, then verdict",
            report.hypothesis_section(hypotheses.as_report_entries(hypotheses.outcomes(cells))),
        ),
        (
            "## The two pre-registered decisions",
            report.decisions_section(
                hypotheses.falsification_clause(cells),
                hypotheses.reexamination_trigger(cells),
                default_arm_label=SHIPPING_DEFAULT.label,
            ),
        ),
    ]
    sections.append(("## Latency and token spend", _latency_body(sink)))
    return tuple(sections)


def _headline(cells: Sequence[Cell], cache: Cache) -> tuple[str, str]:
    """The T11 hand-off block. Derived here so the README never retypes a number."""
    judge = cache.stats("judge")
    replayed = sum(cache.stats(kind).hits for kind in cache.kinds())
    return (
        "## The numbers T11's README will quote",
        report.headline_section(
            cells,
            default_arm=SHIPPING_DEFAULT,
            judge_calls=judge.misses,
            cache_replayed=replayed,
        ),
    )


def _latency_body(sink: Path | None) -> str:
    """ADR-0005's budget from the log, or a plain statement that it was not measurable.

    The refusal is rendered *as a refusal*, naming what is missing. An artifact that dropped
    the section when the sink was off would read as a run that had no latency to report, which
    is the absence-as-measurement failure the whole section exists to avoid.
    """
    try:
        log = latency.load_log(sink)
        cost = latency.translation_cost(log)
    except (latency.SinkMissing, latency.NoSamples, FileNotFoundError) as exc:
        return (
            "**Not measured, and therefore not met.** "
            f"{exc}\n\nADR-0005's dominance test has a cost half, and this run cannot answer "
            "it. Re-run with `FINBRIEF_LOG_FILE` set to fill this section in."
        )
    spends = [
        latency.token_spend(log, name)
        for name in ("rag_answer", "query_translation", "agent_turn")
    ]
    return report.latency_section(cost, [spend for spend in spends if spend.lines])


def _resolve(
    rows: Sequence[GoldenQuestion],
    *,
    settings: Settings,
    cache: Cache,
    run_stage: bool,
) -> variants.VariantSet | None:
    """The persisted variants, resolving them first if this run was asked to.

    Loaded rather than re-resolved by default: they are a *fixed input* (ADR-0004 §9), so a
    scoring run reads the committed file and only `--stage resolve` rewrites it.
    """
    if not run_stage:
        try:
            return variants.load_variants()
        except FileNotFoundError:
            logger.warning(
                "no %s: the +translation arms cannot be replayed. Run --stage resolve first.",
                variants.VARIANTS_PATH,
            )
            return None

    from finbrief.llm import build_chat_model

    planner = build_chat_model(settings, temperature=0.0)
    plans = {}
    for row in rows:
        payload = cache.resolve(
            "variants",
            {
                "row": row.id,
                "question": row.question,
                "planner_model": settings.chat_model,
                "max_sub_queries": settings.max_sub_queries,
            },
            lambda row=row: variants.resolve_plan(
                row.id,
                row.question,
                model=planner,
                max_sub_queries=settings.max_sub_queries,
                planner_model=settings.chat_model,
            ).as_payload(),
        )
        plans[row.id] = variants.ResolvedPlan.from_payload(row.id, payload)
    resolved = variants.VariantSet(
        plans=plans,
        planner_model=settings.chat_model,
        max_sub_queries=settings.max_sub_queries,
        golden_set_schema_version=1,
    )
    variants.verify_replay(tuple(plans.values()))
    # **A partial resolve never overwrites the committed file.** `--rows S1,T2` resolves two
    # plans; writing those over the golden set's 28 would leave every other row unreplayable
    # and the file silently describing a different set — which `VariantSet.covers` is an
    # equality for. A smoke run keeps its plans in the cache, where they are still replayed
    # for the rows it scores.
    golden = load_golden_set()
    if resolved.covers(golden):
        variants.save_variants(resolved)
        logger.info("resolved and persisted %d plans to %s", len(plans), variants.VARIANTS_PATH)
    else:
        logger.info(
            "resolved %d of %d plans — not written to %s, since a partial file would leave the "
            "other rows unreplayable. They are cached and replayed for this run.",
            len(plans),
            len(golden),
            variants.VARIANTS_PATH,
        )
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    markdown = run(args)
    if args.no_write:
        print(markdown)
        return 0
    EVALUATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_REPORT.write_text(markdown, encoding="utf-8")
    print(f"wrote {EVALUATION_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
