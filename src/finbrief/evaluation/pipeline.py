"""The stages: resolve → retrieve → answer → judge, each cell cached by its own inputs.

One function per stage, each a pure map over `(row, arm)` cells so `run.cached_map` can decide
what still costs anything (`evaluation/cache.py`). Nothing here decides *what* a number means —
`metrics.py` and `judge.py` own that — and nothing here writes a report.

**Why the stages are separate rather than one pass per row.** Their inputs invalidate
independently. Changing the judge model must re-judge and must not re-retrieve; re-ingesting
the collection must re-retrieve and re-answer and re-judge; changing `sub_queries`' parser
must re-resolve. A single per-row key would collapse all of that into one address and make
every change cost the whole run.

**The collection fingerprint is in the retrieval key, and it is a real hash.** A number
retrieved from one ingest must never be served for another: chunk ids are stable across a
re-ingest but their *text* is not (a chunker change rewrites every body), so the fingerprint
covers both. It costs one `all_chunks` read per process, which the hybrid arms pay for anyway.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_chroma import Chroma

from finbrief.config import Settings
from finbrief.evaluation import judge as judging
from finbrief.evaluation.arms import Arm
from finbrief.evaluation.cache import Cache
from finbrief.evaluation.loader import Bucket, GoldenQuestion
from finbrief.evaluation.metrics import RowScore, score_row
from finbrief.evaluation.run import cached_map
from finbrief.evaluation.variants import VariantSet
from finbrief.rag import answer_question
from finbrief.retrieval.retrieve import Retrieval
from finbrief.retrieval.vectorstore import all_chunks

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Cell:
    """One `(question, arm)` result: what was retrieved, what was answered, what was scored.

    Carries the context *bodies* as well as the scores because the judge is scored over text
    and the cache key is built from it — and because a surprising cell has to be inspectable
    without a re-run, the complaint ADR-0004 §9 makes about un-recoverable sub-queries.
    """

    question_id: str
    bucket: Bucket
    arm: str
    retrieval: Retrieval
    score: RowScore
    #: `None` on an arm whose generation metrics do not run, which is not the same as an answer
    #: that came back empty.
    answer: str | None
    #: metric name -> score, `None` where the judge could not score the row.
    judged: Mapping[str, float | None]

    @property
    def contexts(self) -> tuple[str, ...]:
        return tuple(context.body for context in self.retrieval.contexts)


def collection_fingerprint(store: Chroma) -> str:
    """A digest over every chunk id and its content hash — the ingest this run measured.

    Ids alone would be wrong: a re-ingest with a different chunker keeps the ids and rewrites
    the bodies, so a cached retrieval would be served for a collection it never ran against.
    The metadata already carries `content_hash` (ingest writes it for the idempotency check),
    so this costs no hashing of bodies.
    """
    digest = hashlib.sha256()
    for document in sorted(all_chunks(store), key=lambda doc: doc.id or ""):
        digest.update((document.id or "").encode("utf-8"))
        digest.update(str(document.metadata.get("content_hash", "")).encode("utf-8"))
    return digest.hexdigest()


def retrieval_key(
    row: GoldenQuestion,
    arm: Arm,
    *,
    k: int,
    settings: Settings,
    fingerprint: str,
    variants: Sequence[str],
) -> dict[str, Any]:
    """Everything that determines what this arm retrieves for this row.

    `variants` is in the key rather than inferred from the arm: on a `+translation` arm they are
    the *replayed* ones, so a re-resolve has to invalidate the retrieval that used the old ones.
    """
    return {
        "row": row.id,
        "arm": arm.name,
        "strategy": arm.strategy.value,
        "translate": arm.translate,
        "max_sub_queries": arm.max_sub_queries,
        "k": k,
        "embedding_model": settings.embedding_model,
        "collection": fingerprint,
        "variants": list(variants),
    }


def answer_key(
    row: GoldenQuestion, retrieval: Retrieval, arm: Arm, *, settings: Settings
) -> dict[str, Any]:
    """Everything that determines the answer: the question, the contexts, the model, the prompt.

    The prompt is keyed by a digest of its own text, so an edit to `prompts.SYSTEM_PROMPT`
    invalidates every answer rather than leaving the run reporting faithfulness against a prompt
    it no longer uses.
    """
    from finbrief.prompts import SYSTEM_PROMPT

    return {
        "row": row.id,
        "arm": arm.name,
        "question": row.question,
        "chunk_ids": [context.chunk_id for context in retrieval.contexts],
        "bodies": _digest(context.body for context in retrieval.contexts),
        "chat_model": settings.chat_model,
        "prompt": _digest([SYSTEM_PROMPT]),
    }


def _digest(parts) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def retrieve_cells(
    rows: Sequence[GoldenQuestion],
    arm: Arm,
    *,
    store: Chroma,
    settings: Settings,
    variants: VariantSet | None,
    cache: Cache,
    k: int,
    fingerprint: str,
) -> tuple[Retrieval, ...]:
    """Retrieve every row on one arm, replaying the persisted variants where the arm translates.

    The replay is what makes this arm comparable with the others (ADR-0004 §9): a
    `+translation` arm gets its recorded reply through a stub, never a fresh planner call, so
    the four arms differ only by `strategy` and `translate`.
    """
    from finbrief.retrieval.retrieve import retrieve

    def planned_variants(row: GoldenQuestion) -> tuple[str, ...]:
        if not arm.plans or variants is None:
            return ()
        return variants.plan(row.id).variants

    def produce(row: GoldenQuestion) -> dict[str, Any]:
        model = variants.planner(row.id) if arm.plans and variants is not None else None
        retrieval = retrieve(
            row.question,
            strategy=arm.strategy,
            translate=arm.translate,
            k=k,
            store=store,
            settings=settings,
            model=model,
        )
        return retrieval.as_payload()

    payloads = cached_map(
        cache,
        "retrieval",
        rows,
        key_of=lambda row: retrieval_key(
            row,
            arm,
            k=k,
            settings=settings,
            fingerprint=fingerprint,
            variants=planned_variants(row),
        ),
        produce=produce,
    )
    return tuple(Retrieval.from_payload(payload) for payload in payloads)


class ContextDrift(RuntimeError):
    """The measured chain retrieved something other than what this arm was scored on."""


def answer_cells(
    rows: Sequence[GoldenQuestion],
    retrievals: Sequence[Retrieval],
    arm: Arm,
    *,
    store: Chroma,
    settings: Settings,
    variants: VariantSet | None,
    cache: Cache,
    k: int,
) -> tuple[str | None, ...]:
    """One answer per row through `rag.answer_question` — the measured chain, unmodified.

    **The chain and not the agent**, which is ADR-0003's whole seam: the agent's tool loop is
    nondeterministic and its output must not reach the harness that measures the chain
    (ADR-0003 amendment §4). It is also why the `tool-augmented` rows are scoreable on
    faithfulness at all — a chain answer carries no tool-derived sentence, because the chain
    calls no tools (ADR-0002's T10 amendment).

    **`answer_question` retrieves again, and that is accepted rather than worked around.** The
    alternative is to call the generation half directly over this arm's contexts, which saves
    one embedding round per cell and creates a second code path through the thing being
    measured — exactly what ADR-0003 keeps `rag.answer_question` callable to avoid. So the
    chain runs with *this arm's* configuration and its own replayed planner, and the contexts
    it returns are checked against the ones the arm was scored on: identical by construction
    on every arm, since the deterministic arms are exact and the translated ones replay the
    same reply. If they ever differ, faithfulness would be measured against contexts the
    scored retrieval never surfaced, so `ContextDrift` stops the run instead.

    An arm whose generation metrics do not run returns `None` per row — absent, which is not the
    same as an answer that came back empty.
    """
    if not arm.judged:
        return tuple(None for _ in rows)

    def produce(item: tuple[GoldenQuestion, Retrieval]) -> dict[str, Any]:
        row, scored = item
        grounded = answer_question(
            row.question,
            strategy=arm.strategy,
            translate=arm.translate,
            k=k,
            store=store,
            settings=settings,
            translation_model=(
                variants.planner(row.id) if arm.plans and variants is not None else None
            ),
        )
        produced = [context.chunk_id for context in grounded.contexts]
        expected = [context.chunk_id for context in scored.contexts]
        if produced != expected:
            raise ContextDrift(
                f"{row.id} on arm {arm.name}: the measured chain retrieved {produced} while "
                f"the scored retrieval held {expected}. Faithfulness over the first against "
                f"the second would score an answer about contexts this arm never surfaced. "
                f"Both come from retrieve() with the same switches, so a difference means "
                f"the collection changed under the run, or the planner was not replayed."
            )
        return {"answer": grounded.text}

    pairs = list(zip(rows, retrievals, strict=True))
    payloads = cached_map(
        cache,
        "answer",
        pairs,
        key_of=lambda item: answer_key(item[0], item[1], arm, settings=settings),
        produce=produce,
    )
    return tuple(str(payload["answer"]) for payload in payloads)


def judge_cells(
    rows: Sequence[GoldenQuestion],
    retrievals: Sequence[Retrieval],
    answers: Sequence[str | None],
    *,
    metrics: Sequence[str],
    settings: Settings,
    cache: Cache,
    judge: Any,
    embeddings: Any,
) -> tuple[Mapping[str, float | None], ...]:
    """Score every row on every metric, one cached cell per `(row, metric)`.

    Per `(row, metric)` and not per row: that is the granularity a resumed run needs, and it is
    what lets a metric be added to a finished run for the price of that metric alone.
    """
    version = judging.ragas_version()
    scored: list[Mapping[str, float | None]] = []
    for row, retrieval, answer in zip(rows, retrievals, answers, strict=True):
        sample = judging.JudgeSample(
            question=row.question,
            contexts=tuple(context.body for context in retrieval.contexts),
            answer=answer or "",
            reference=row.reference,
        )
        row_scores: dict[str, float | None] = {}
        for metric in metrics:
            if metric in judging.GENERATION_METRICS and answer is None:
                # Absent, not zero: this arm does not generate, so there is no answer to score.
                row_scores[metric] = None
                continue
            payload = cache.resolve(
                "judge",
                judging.judge_cache_key(
                    metric,
                    sample,
                    judge_model=settings.judge_model,
                    ragas_version=version,
                ),
                lambda metric=metric, sample=sample: {
                    "score": judging.score(metric, sample, judge=judge, embeddings=embeddings)
                },
            )
            row_scores[metric] = payload["score"]
        scored.append(row_scores)
    return tuple(scored)


def build_cells(
    rows: Sequence[GoldenQuestion],
    arm: Arm,
    retrievals: Sequence[Retrieval],
    answers: Sequence[str | None],
    judged: Sequence[Mapping[str, float | None]],
    *,
    k: int,
) -> tuple[Cell, ...]:
    """Assemble one arm's cells, scoring the free metrics on the way through."""
    return tuple(
        Cell(
            question_id=row.id,
            bucket=row.bucket,
            arm=arm.name,
            retrieval=retrieval,
            score=score_row(row, retrieval, arm=arm.name, k=k),
            answer=answer,
            judged=scores,
        )
        for row, retrieval, answer, scores in zip(
            rows, retrievals, answers, judged, strict=True
        )
    )
