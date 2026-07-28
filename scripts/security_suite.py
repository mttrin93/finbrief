"""Run the security suite against the real models and rewrite its evidence artifact (#8).

    uv run python scripts/security_suite.py               # run everything, rewrite the artifact
    uv run python scripts/security_suite.py --no-write    # print it, leave the artifact alone
    uv run python scripts/security_suite.py --gate-only   # skip the paid indirect half

A script, not a test: it makes a classifier call per corpus case, embeds a throwaway collection
through the paid embedding model, and drives the real agent. The suite is hermetic by contract
(CLAUDE.md) — **never invoke this from a test.** The corpus, the measurements and the report are
`finbrief.security.corpus` and `finbrief.security.report`, which the suite does cover.

It is the third of the four non-hermetic entry points and the cheapest of the three that cost
anything: ~35 one-word classifier completions — every corpus case the denylist does not catch,
plus the whole benign set — and a handful of agent turns. Its *control flow* is hermetic and
is covered by `tests/test_security_suite_script.py`.

**What only a live run can establish**, and therefore why this exists rather than a bigger test
file. Whether a real model recognises a *novel* payload is a property of that model, so layer
3's column in the marginal-contribution table cannot be filled by a scripted reply. Whether a
real model *obeys* a planted instruction is the same kind of fact. And the p50 ADR-0006 budgets
(`config.GATE_LATENCY_BUDGET_MS`, revised from 800 ms by its T7 amendment §2) is a measurement,
not an assertion — it comes from the `Screening` this run produced for every question.

**The indirect-injection half builds its own collection and never touches the demo KB**
(ADR-0006). A temporary directory, the paid embedding model — the same one
`retrieval/embeddings.py` gives ingest and query, because a collection written by one model and
queried by another retrieves noise with no error — and five poisoned chunks belonging to a filer
that **is** in the Universe, deliberately: an out-of-Universe ticker makes the agent decline to
search, so the row measures the whitelist instead of the quarantine framing (ADR-0006 §7, and
`corpus.PLANTED_PAYLOADS` for the two live runs that established it). Isolation is the
collection's job, not the ticker's.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from finbrief.agent.agent import answer, build_agent, build_checkpointer
from finbrief.config import Settings, get_settings
from finbrief.observability.logging_setup import configure_logging
from finbrief.security import corpus
from finbrief.security.advice import validate_answer
from finbrief.security.input_gate import folding_required, screen
from finbrief.security.report import (
    AnswerResult,
    GateResult,
    InjectionResult,
    SuiteRun,
    render_report,
)

#: The committed machine evidence of the most recent run. Relative to the working directory, so
#: run this script from the repo root — the same convention as the ingest and smoke reports.
SECURITY_REPORT = Path("docs/verification/security-gate.md")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print the report instead of rewriting the committed artifact",
    )
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help=(
            "run layers 1-4 only, skipping the indirect-injection half. Cheaper: no embeddings "
            "and no agent turns. The artifact then carries a PARTIAL RUN banner and says "
            "the planted payloads were not run rather than printing 0/0 — a partial run must "
            "not be committed as a full one."
        ),
    )
    return parser.parse_args(argv)


def run_gate(settings: Settings) -> tuple[GateResult, ...]:
    """Screen every corpus case and every benign question through the live gate."""
    results = [
        GateResult(
            case_id=case.id,
            technique=case.technique,
            expected=case.caught_by,
            screening=screen(case.payload, settings=settings),
            # Layer 1's marginal contribution, per case. Pure and free — two denylist scans —
            # and recorded here rather than inferred in the report, so the artifact's layer-1
            # cell is a measurement of this run like every other cell (issue #8 review).
            folding_required=folding_required(case.payload),
        )
        for case in corpus.DIRECT_CASES
    ]
    results += [
        GateResult(
            # The question itself is the id for a benign row: there is nothing else to call it,
            # and a reader of the artifact needs to see what was allowed.
            case_id=question,
            technique="benign analyst question",
            expected=None,
            screening=screen(question, settings=settings),
            # Measured for a benign question too, rather than hardcoded `False`: a question that
            # only escapes layer 2 because it was *not* folded would be a false negative worth
            # seeing, and asserting it cannot happen is cheaper than assuming it.
            folding_required=folding_required(question),
        )
        for question in corpus.BENIGN_QUESTIONS
    ]
    return tuple(results)


def run_output_validator() -> tuple[AnswerResult, ...]:
    """Put both answer sets through layer 4.

    Deterministic, and run here anyway: the artifact's job is to report what the *shipped* gate
    does, and a table that omitted layer 4 because it needs no model call would leave the
    marginal-contribution claim resting on a test file a reader has to go and find.
    """
    results = []
    for text, expected in (
        *((answer_text, True) for answer_text in corpus.ADVICE_ANSWERS),
        *((answer_text, False) for answer_text in corpus.RESEARCH_ANSWERS),
    ):
        verdict = validate_answer(text)
        results.append(
            AnswerResult(
                text=text,
                expected_refusal=expected,
                refused=verdict.refused,
                rules=verdict.rules,
            )
        )
    return tuple(results)


def run_indirect(settings: Settings, directory: Path) -> tuple[InjectionResult, ...]:
    """Ask the real agent a neutral question over a collection of poisoned chunks.

    Built here rather than in a fixture because it needs the **paid** embedding model: the
    collection has to be searchable by the model the query is embedded with, and a fake
    embedding would make the retrieval that reaches the payload a different mechanism from
    production's.

    One thread per payload, so a payload that *did* get through cannot influence the next turn's
    conversation — which would make the second result a measurement of the first.
    """
    from finbrief.retrieval.embeddings import build_embeddings
    from finbrief.retrieval.vectorstore import build_filings_store, write_chunks

    store = build_filings_store(
        persist_directory=str(directory / "planted"), embeddings=build_embeddings(settings)
    )
    write_chunks(store, corpus.planted_chunks())

    agent = build_agent(
        settings=settings,
        store=store,
        checkpointer=build_checkpointer(path=str(directory / "planted.sqlite")),
    )

    results = []
    for index, payload in enumerate(corpus.PLANTED_PAYLOADS):
        turn = answer(payload.question, thread_id=f"planted-{index}", agent=agent)
        verdict = validate_answer(turn.text)
        results.append(
            InjectionResult(
                payload_id=payload.id,
                technique=payload.technique,
                # Whether the poisoned chunk reached the model at all. Without it, a turn the
                # agent answered without searching reports a clean result about nothing — see
                # `InjectionResult.retrieved`.
                #
                # **The whole body, not a 40-character prefix.** The prefix was an unnamed
                # limit and a weaker check: a retrieval that returned a truncated chunk would
                # have satisfied it. `vectorstore` prepends a provenance header, so the body is
                # a suffix of what is stored and containment is the strongest available form of
                # this check — prefer an equality over a bound (CLAUDE.md, issue #8 review).
                retrieved=any(payload.body in context.body for context in turn.contexts),
                # An empty canary means this payload's goal was extraction, not obedience — see
                # `PlantedPayload.leaks`. `""` is in every string, so an unguarded `in` would
                # report every extraction payload as obeyed.
                obeyed=bool(payload.canary) and payload.canary in turn.text,
                leaked=any(
                    fragment in turn.text for fragment in corpus.SYSTEM_PROMPT_FRAGMENTS
                ),
                refused_by_validator=verdict.refused,
                excerpt=turn.text,
            )
        )
    return tuple(results)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    settings = get_settings()

    gate = run_gate(settings)
    answers = run_output_validator()
    if args.gate_only:
        injections: tuple[InjectionResult, ...] = ()
    else:
        # A real temporary directory, removed on the way out: this collection is built for one
        # run and must not be mistakable for the demo KB on disk (ADR-0006).
        with tempfile.TemporaryDirectory(prefix="finbrief-planted-") as raw:
            injections = run_indirect(settings, Path(raw))

    run = SuiteRun(
        gate=gate,
        answers=answers,
        injections=injections,
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        classifier_model=settings.classifier_model,
        chat_model=settings.chat_model,
        gate_only=args.gate_only,
    )
    report = render_report(run)

    if args.no_write:
        print(report)
    else:
        SECURITY_REPORT.parent.mkdir(parents=True, exist_ok=True)
        SECURITY_REPORT.write_text(report, encoding="utf-8")
        print(f"wrote {SECURITY_REPORT}")

    # A non-zero exit on a failed suite, so this is usable as a gate rather than only as a
    # generator — and `--gate-only` still exits 0 on a green partial run, because a partial run
    # is a legitimate cheap check and the artifact says which half it covered.
    if not run.passed:
        print("SUITE FAILED — see the report", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
