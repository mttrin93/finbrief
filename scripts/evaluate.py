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
from typing import Any

from finbrief.config import Settings, get_settings, load_env, resolve_log_file
from finbrief.evaluation import (
    deferrals,
    hypotheses,
    latency,
    pipeline,
    report,
    tool_eval,
    variants,
)
from finbrief.evaluation import judge as judging
from finbrief.evaluation.arms import ABLATION_ARMS, SCORED_ARMS, SHIPPING_DEFAULT, Arm
from finbrief.evaluation.cache import Cache
from finbrief.evaluation.loader import GoldenQuestion, GoldenSet, load_golden_set
from finbrief.evaluation.pipeline import Cell
from finbrief.observability.events import sink_offset
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


#: The stages whose execution appends lines a statistic can be taken over. A run of only
#: `report` measures nothing and re-renders someone else's measurements.
MEASURING_STAGES = frozenset(report.ALL_STAGES) - {"report"}


def _log_window(
    sink: Path | None, *, cache_dir: Path | str, stages: Sequence[str]
) -> latency.Window | None:
    """Which slice of the sink this artifact's numbers come from.

    **A measuring run marks the sink; a report-only run replays the last mark.** The mark is
    taken before anything appends, so the window is this run's and not the pooled history of
    every run that shared the file. And it is *persisted* beside the cached cells, because the
    artifact is regenerable from that cache and a re-render that could not see the measuring
    run's window would refuse to report latency for numbers it is otherwise reproducing exactly.
    """
    if sink is None:
        return None
    if set(stages) & MEASURING_STAGES:
        window = latency.Window(
            path=str(sink),
            offset=sink_offset(sink),
            recorded_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        )
        latency.save_window(cache_dir, window)
        return window
    recorded = latency.load_window(cache_dir)
    if recorded is None:
        logger.warning(
            "no recorded log window under %s: this re-render cannot attribute latency to the "
            "run that produced the cached cells, and the artifact will say so.",
            cache_dir,
        )
    return recorded


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
    # **The mark, before a single line of this run is written.** The sink is append-only across
    # runs, so every median taken over the whole file is a median over every run that ever
    # shared it — which is how the first committed artifact reported a planner p50 from a pool
    # holding 13 appended runs, the pre-fix ones included (`events.sink_offset`). Marked here,
    # after `configure_logging` would have created the file but before anything appends to it.
    configure_logging()
    window = _log_window(sink, cache_dir=args.cache_dir, stages=stages)
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

    # **Every pass that EMITS log lines runs before anything that READS them**, and the ordering
    # is load-bearing rather than tidy. The rule was already written down for the agent stage
    # ("two of the four deferrals read lines the live turns write") and the planner-variance
    # pass broke it again from the other side: its 40 real planner calls are the *only* metered
    # `query_translation` lines a warm run produces, and computing the latency section before
    # them made ADR-0005's budget unmeasurable on a run that had just measured it. So the
    # emitting passes go first, together, and the log-reading sections come after.
    emitted: list[tuple[str, str]] = []
    if "resolve" in stages:
        emitted.append(_planner_variance(rows, settings=settings, cache=cache))
    if "agent" in stages:
        emitted.append(_agent_section(settings))

    sections = list(_findings_sections(cells, sink=sink, log_window=window))
    sections.append(_leakage(cells, golden))
    sections.extend(emitted)
    support = (
        _citation_support(cells, settings=settings, cache=cache, judge=judge)
        if judge is not None
        else None
    )
    sections.append(_deferrals_block(sink, support, log_window=window))
    sections.append(_headline(cells, cache))
    # **Built after every stage, not before them.** This snapshot is the artifact's only
    # provenance table, and taking it here had it miss every kind a later stage touched: the
    # first committed run listed four kinds summing to 868 replayed cells beside a headline
    # of "934 replayed — see the provenance table", the missing 66 being the `cited_sentence`
    # judge calls the citation pass makes *after* this line ran. Same reason the paid-judge
    # count in `_headline` sums every judging kind rather than the one named "judge".
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
        golden_set_total=len(golden),
        ablations_run=bool(args.ablations),
        stages=stages,
        cache={kind: cache.stats(kind) for kind in cache.kinds()},
    )
    return report.render_report(
        provenance=provenance,
        golden=golden,
        cells=cells,
        arms=SCORED_ARMS,
        ablations=ABLATION_ARMS if args.ablations else (),
        sections=tuple(sections),
    )


def _findings_sections(
    cells: Sequence[Cell], *, sink: Path | None, log_window: latency.Window | None
) -> tuple[tuple[str, str], ...]:
    """The pre-registered half of the artifact: predictions, triggers, latency.

    Each block is rendered even when the run could not settle it — a prediction reported only
    when it resolved is a prediction a reader cannot tell was evaluated, and ADR-0005 §4's
    trigger fires on an *absence* of gain, which is the shape that goes unnoticed when it is
    not printed.
    """
    results = hypotheses.outcomes(cells)
    clauses = hypotheses.falsification_clause(cells)
    trigger = hypotheses.reexamination_trigger(cells)
    sections = [
        (
            "## Pre-registered hypotheses — prediction, then measurement, then verdict",
            report.hypothesis_section(hypotheses.as_report_entries(results)),
        ),
        (
            "## The two pre-registered decisions",
            report.decisions_section(
                clauses, trigger, default_arm_label=SHIPPING_DEFAULT.label
            ),
        ),
    ]
    # **After both decisions, and over the same objects they were rendered from.** The audit is
    # a re-read of the cells above rather than a second computation of them: a power figure
    # derived independently is a power figure that can disagree with the verdict it qualifies.
    sections.append(
        (
            "## Power audit — which of these cells is a measurement",
            report.power_audit_section(
                _every_comparison(clauses, trigger, results), bucket_floor=BUCKET_FLOOR
            ),
        )
    )
    sections.append(("## Latency and token spend", _latency_body(sink, log_window)))

    return tuple(sections)


#: ADR-0002 decision 3's per-bucket floor — "at least 6 questions per bucket".
#:
#: Here rather than in `config.py` on the ingestion-threshold grounds: it is an assertion about
#: the committed golden set's construction, and the power audit reads it to state what a bucket
#: that size can resolve. Not a knob — changing it changes ADR-0002, not this run.
BUCKET_FLOOR = 6


def _every_comparison(
    clauses: Sequence[Any], trigger: Any, results: Sequence[Any]
) -> tuple[tuple[str, Any], ...]:
    """Every pre-registered comparison this artifact renders, labelled, in one sequence.

    Assembled from the objects the sections above were rendered from, so the audit cannot
    disagree with the tables it qualifies.
    """
    labelled: list[tuple[str, Any]] = [
        (outcome.prediction.id, outcome.comparison) for outcome in results
    ]
    for clause in clauses:
        labelled.append((f"clause/{clause.bucket.value}/precision", clause.precision))
        labelled.append((f"clause/{clause.bucket.value}/recall", clause.recall))
    for bucket, comparison in trigger.per_bucket.items():
        labelled.append((f"trigger/{bucket.value}", comparison))
    return tuple(labelled)


#: Every cache kind whose misses are a paid judge call. Both, not just the one named "judge":
#: the citation pass scores `(sentence, chunk)` pairs through the same scorer under its own
#: kind, and the first committed artifact reported "judge calls this run paid for | 0" on a run
#: that paid for four of them.
JUDGING_KINDS = ("judge", "cited_sentence")


def _headline(cells: Sequence[Cell], cache: Cache) -> tuple[str, str]:
    """The T11 hand-off block. Derived here so the README never retypes a number."""
    judge_calls = sum(cache.stats(kind).misses for kind in JUDGING_KINDS)
    replayed = sum(cache.stats(kind).hits for kind in cache.kinds())
    return (
        "## The numbers T11's README will quote",
        report.headline_section(
            cells,
            default_arm=SHIPPING_DEFAULT,
            judge_calls=judge_calls,
            cache_replayed=replayed,
        ),
    )


def _leakage(cells: Sequence[Cell], golden: GoldenSet) -> tuple[str, str]:
    """ADR-0004 §11's mention leakage per arm, over the rows carrying labelled probes.

    **Selected by whether the row was *probed*, not by whether it leaked.** A row nobody
    enumerated false positives for contributes no evidence either way, and a row selected
    because it leaked would make every arm's rate a statement about its own leaks
    (`metrics.leakage_precision`'s docstring is explicit, and this is the caller that has to
    honour it).
    """
    from finbrief.evaluation.metrics import leakage_precision

    probed = {row.id for row in golden if row.known_false_positives}
    rates = [
        leakage_precision(
            [
                cell.score
                for cell in cells
                if cell.arm == arm.name and cell.question_id in probed
            ],
            arm=arm.label,
        )
        for arm in SCORED_ARMS + ABLATION_ARMS
    ]
    return ("## Mention leakage (ADR-0004 §11)", report.leakage_section(rates))


#: How many times the planner is re-asked per sampled question for ADR-0004 §9 step 3.
#:
#: Five, and the sample is the first `PLANNER_VARIANCE_ROWS` rows rather than all 28: the step
#: measures whether the planner is stable, which needs repeats per question rather than breadth
#: across them, and 5 × 8 real planner calls is the whole cost.
PLANNER_VARIANCE_REPEATS = 5
PLANNER_VARIANCE_ROWS = 8


def _planner_variance(
    rows: Sequence[GoldenQuestion], *, settings: Settings, cache: Cache
) -> tuple[str, str]:
    """ADR-0004 §9 step 3 — the planner's own variance, measured rather than argued.

    **The step the ADR claimed was reported and was not.** §9 asks for "an n-repeat of the
    resolve step over a sample of questions", reported *separately* so the strategy comparison
    does not inherit noise from a component it is not about; the ADR's T6-amendment closing
    paragraph then said the artifact carried it. `variants.agreement` existed with no production
    caller and nothing rendered it, so §9's own falsifiable prediction — "if the n-repeat finds
    the planner returns identical sub-queries across runs, step 2 was unnecessary caution" —
    went unanswered while an ADR said otherwise.

    Each repeat is cached under its own index, so a resumed run pays for the repeats it has not
    made and a warm re-run pays nothing. The calls are **real**: this is the one pass in the
    harness whose purpose is to let the planner vary, so replaying it would measure the stub.
    """
    from finbrief.llm import build_chat_model

    planner = build_chat_model(settings, temperature=0.0)
    sampled = tuple(rows)[:PLANNER_VARIANCE_ROWS]
    agreements = []
    for row in sampled:
        seen = []
        for repeat in range(PLANNER_VARIANCE_REPEATS):
            payload = cache.resolve(
                "planner_variance",
                {
                    "row": row.id,
                    "question": row.question,
                    "planner_model": settings.chat_model,
                    "max_sub_queries": settings.max_sub_queries,
                    "repeat": repeat,
                },
                lambda row=row: variants.resolve_plan(
                    row.id,
                    row.question,
                    model=planner,
                    max_sub_queries=settings.max_sub_queries,
                    planner_model=settings.chat_model,
                ).as_payload(),
            )
            plan = variants.ResolvedPlan.from_payload(row.id, payload)
            seen.append(tuple(plan.sub_queries))
        agreements.append(variants.agreement(row.id, seen))
    return (
        "## The planner's own variance (ADR-0004 §9 step 3)",
        report.planner_variance_section(agreements, repeats=PLANNER_VARIANCE_REPEATS),
    )


def _citation_support(
    cells: Sequence[Cell], *, settings: Settings, cache: Cache, judge: object
) -> object:
    """The T3/T5 deferral: does a resolving marker's own chunk support its sentence?

    The judge is the **cached faithfulness scorer with the context set narrowed to one chunk**
    — the same tested metric, asked a narrower question. Cached per `(sentence, chunk)` so a
    re-run costs nothing, and keyed through `judge.judge_cache_key` so the key carries the
    judge model and the ragas version like every other judged cell.
    """
    version = judging.ragas_version()

    def judge_sentence(sentence: str, body: str) -> float | None:
        sample = judging.JudgeSample(
            question="Does the cited source support this sentence?",
            contexts=(body,),
            answer=sentence,
            reference="",
        )
        payload = cache.resolve(
            "cited_sentence",
            judging.judge_cache_key(
                judging.FAITHFULNESS,
                sample,
                judge_model=settings.judge_model,
                ragas_version=version,
            ),
            lambda: {"score": judging.score(judging.FAITHFULNESS, sample, judge=judge)},
        )
        return payload["score"]

    return deferrals.citation_support(
        cells, arm=SHIPPING_DEFAULT.name, judge_sentence=judge_sentence
    )


def _deferrals_block(
    sink: Path | None,
    support: object | None = None,
    *,
    log_window: latency.Window | None = None,
) -> tuple[str, str]:
    """The four deferrals, each measured or explicitly named as not measured.

    Layer 4's residue is computed here unconditionally, because it needs no run at all — regex
    over a Guard. The two log-based rates need the `agent` stage's live turns; when those
    lines are absent the rate renders as **not measured** rather than as a flattering zero.
    """
    from finbrief.security.advice import validate_answer
    from finbrief.security.corpus import ADVICE_RESIDUE_PROBES

    residue = deferrals.advice_residue(ADVICE_RESIDUE_PROBES, validate=validate_answer)
    measured = {
        report.DEFERRAL_ADVICE_RESIDUE: (
            f"{residue.rate.render()} — {len(residue.residue)} of "
            f"{residue.rate.total} hand-labelled recommendations were **not** refused"
        )
    }
    try:
        # This run's window only: the divergence rate is a rate over *this* run's searches, and
        # over the whole append-only sink it was a rate over every run that ever shared it (the
        # first artifact's 40/40 spanned 13 of them).
        if log_window is None:
            raise latency.SinkMissing("no log window for this artifact")
        log = latency.load_log(sink, start_offset=log_window.offset)
    except (latency.SinkMissing, FileNotFoundError):
        log = None
    if log is not None:
        divergence = deferrals.divergence(log)
        if divergence.overall.total:
            measured[report.DEFERRAL_DIVERGENCE] = (
                f"{divergence.overall.render()}; {divergence.first_turn.render()}; "
                f"{divergence.follow_up.render()}"
            )
        brackets = deferrals.bracket_adherence(log)
        if brackets.turns_with_sources:
            measured[report.DEFERRAL_BRACKETS] = (
                f"{brackets.clean.render()} — {brackets.uncited} uncited, "
                f"{brackets.unresolved} unresolved, {brackets.non_numeric} non-numeric"
            )
    if support is not None and support.rate.total:
        measured[report.DEFERRAL_CITED_SUPPORT] = (
            f"{support.rate.render()} over `(sentence, marker)` pairs across "
            f"{support.sentences} cited sentence(s) — {support.unsupported} pair(s) with "
            f"**no** support from the chunk they name, {support.partial} only partly supported "
            f"(both count against the rate); {support.unresolvable} marker(s) pointed outside "
            f"the retrieval; {support.unscored} pair(s) the judge did not score"
        )
    body = report.deferrals_section(measured)
    if residue.residue:
        body += (
            "\n\n**Layer 4's residue is the sharpest of the four, and it cost nothing to "
            "measure.** The validator is a regex rule set behind a Guard, so what it "
            "*misses* is "
            "computable with no model and no live run — this deferral could have been closed "
            "at "
            "any point since T7. What it shows: the rules catch advice that announces itself "
            "(`ADVICE_ANSWERS`, all refused, reported by the security suite) and refuse none "
            "of "
            "the recommendations that carry no imperative, no rating word, no price target "
            "and no "
            "position-sizing instruction. n is small and hand-authored, so this is a statement "
            "about the rules' **generality**, not a 100%-evasion claim about indirect advice."
        )
    return ("## Deferred measurements — reported, or named as not run", body)


def _agent_section(settings: Settings) -> tuple[str, str]:
    """Run the tool-calling eval against the live agent (spec US-29).

    A fresh thread per case, so the checkpointer's memory cannot answer case N from case N-1.
    """
    from finbrief.agent.agent import answer, build_agent, build_checkpointer

    golden = load_golden_set()
    cases = tool_eval.cases_from(golden)
    agent = build_agent(settings=settings, checkpointer=build_checkpointer(settings))
    ask = tool_eval.scripted_asker(answer, agent=agent, thread_prefix="t10-tool-eval")
    outcomes = tool_eval.run_all(cases, ask=ask)
    logger.info("tool eval: %d cases, %d scored", len(outcomes.outcomes), len(outcomes.scored))
    return ("## Tool-calling eval — the selection layer", tool_eval.render(outcomes))


def _latency_body(sink: Path | None, log_window: latency.Window | None = None) -> str:
    """ADR-0005's budget from the log, or a plain statement that it was not measurable.

    The refusal is rendered *as a refusal*, naming what is missing. An artifact that dropped
    the section when the sink was off would read as a run that had no latency to report, which
    is the absence-as-measurement failure the whole section exists to avoid.
    """
    try:
        if log_window is None:
            raise latency.SinkMissing(
                "no log window for this artifact: the sink was not enabled, and no measuring "
                "run recorded one beside the cache."
            )
        log = latency.load_log(sink, start_offset=log_window.offset)
        cost = latency.translation_cost(log)
    except (latency.SinkMissing, latency.NoSamples, FileNotFoundError) as exc:
        return (
            "**Not measured, and therefore not met.** "
            f"{exc}\n\nADR-0005's dominance test has a cost half, and this run cannot answer "
            "it. Two states produce this, and both are honest refusals rather than a missing "
            "number: the sink was never enabled, or **every stage that would have timed "
            "something was served from cache**. Latency is read from this run's own window of "
            "the log (`events.sink_offset`), so a warm re-run has nothing to time — clear the "
            "`retrieval` cache, or set `FINBRIEF_LOG_FILE`, and re-run."
        )
    spends = [
        latency.token_spend(log, name)
        for name in ("rag_answer", "query_translation", "agent_turn")
    ]
    return report.latency_section(
        cost, [spend for spend in spends if spend.lines], window=log_window
    )


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
