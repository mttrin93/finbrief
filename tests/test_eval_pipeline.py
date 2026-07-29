"""The stages, and the two bugs the two-question smoke run found before the sweep (#11).

Seam: `retrieve()`'s injectable `store`/`settings`/`model`, over a real on-disk Chroma with a
fake embedding. Both tests below exist because a live run produced evidence a test did not:

1. An `Arm`'s `max_sub_queries` reached nothing, so both ablation cells ran the **real
planner** at
   the application's cap. They are ADR-0005 §2's falsification channel and ADR-0004 §7's
   ablation, so they would have answered a question nobody asked while looking like a finished
   run.
2. `ContextDrift` — the check that the measured chain and the scored retrieval saw the same
chunks.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from finbrief.config import Settings
from finbrief.evaluation.arms import (
    HYBRID_NORMALISED,
    HYBRID_ONLY,
    HYBRID_TRANSLATED,
    VECTOR_NORMALISED,
    VECTOR_ONLY,
)
from finbrief.evaluation.pipeline import settings_for
from finbrief.retrieval.retrieve import retrieve


class SpyPlanner(BaseChatModel):
    """A planner that records every invocation — so "it never ran" is checkable, not assumed."""

    invocations: int = 0

    @property
    def _llm_type(self) -> str:
        return "finbrief-spy-planner"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.invocations += 1
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="a sub-query it must not use"))
            ]
        )


def a_settings(**overrides) -> Settings:
    return Settings.from_env({"OPENROUTER_API_KEY": "test-key", **overrides})


@pytest.mark.parametrize("arm", [VECTOR_NORMALISED, HYBRID_NORMALISED])
def test_an_ablation_arm_never_reaches_the_planner(arm, filings_store):
    # **The smoke run's finding, as a test.** `retrieve()` reads `max_sub_queries` from
    # `Settings`, not from its arguments, so an `Arm` declaring 0 did nothing until
    # `settings_for` existed: the live log showed four `query_translation` lines *carrying
    # token counts* on these two cells. A spy, because the honest form of "the planner did not
    # run" is a count, not an absence of symptoms — `query_translation.translate` guards the
    # call on the cap, and this asserts the guard was reached with the right cap.
    planner = SpyPlanner()

    retrieval = retrieve(
        "How much debt does Tesla carry?",
        strategy=arm.strategy,
        translate=arm.translate,
        k=5,
        store=filings_store,
        settings=settings_for(arm, a_settings()),
        model=planner,
    )

    assert planner.invocations == 0, (
        "the ablation arm called the planner, so it is not an ablation — it is the "
        "translated arm "
        "with a second sampled step in it"
    )
    assert retrieval.planned is False
    assert retrieval.translated is True, "translation stays on: the ticker form is the point"


def test_an_ablation_arm_still_adds_the_deterministic_ticker_form(filings_store):
    # What the cell is *for*: ADR-0005 §2 asks whether the exact-identifier win survives with
    # the planner off, which is only a question if normalisation still runs.
    retrieval = retrieve(
        "How much debt does Tesla carry?",
        strategy=VECTOR_NORMALISED.strategy,
        translate=VECTOR_NORMALISED.translate,
        k=5,
        store=filings_store,
        settings=settings_for(VECTOR_NORMALISED, a_settings()),
        model=SpyPlanner(),
    )

    assert retrieval.ticker_form == "How much debt does TSLA carry?"
    assert retrieval.sub_queries == ()


def test_a_scored_arm_keeps_the_configured_cap():
    # The other direction, so the fix cannot have zeroed everything: the shipping default
    # plans at the application's cap, and `settings_for` must not quietly change what it
    # measures.
    settings = a_settings()

    assert settings_for(HYBRID_TRANSLATED, settings).max_sub_queries == settings.max_sub_queries
    assert settings_for(VECTOR_ONLY, settings).max_sub_queries == 0


def test_settings_for_changes_nothing_else():
    # `replace` on a frozen dataclass, so a typo would silently drop a field rather than raise.
    settings = a_settings(FINBRIEF_RETRIEVAL_K="7")
    adjusted = settings_for(VECTOR_NORMALISED, settings)

    assert adjusted.retrieval_k == 7
    assert adjusted.chat_model == settings.chat_model
    assert adjusted.judge_model == settings.judge_model
    assert adjusted.embedding_model == settings.embedding_model


# --- the retrieval key: what a cached retrieval is actually addressed by -------------------


def a_row(**overrides):
    """A real committed golden row, with the fields a test varies replaced.

    Built from `load_golden_set()` rather than hand-assembled: `GoldenQuestion` carries the
    grounding, the section counts and the EDGAR-verification flag, and a double that invented
    them would drift from the reader that parses them.
    """
    from dataclasses import replace

    from finbrief.evaluation.loader import load_golden_set

    row = load_golden_set().questions[0]
    return replace(row, **overrides) if overrides else row


def test_the_retrieval_key_carries_the_question_and_not_only_its_id():
    """Four of the six arms plan nothing, so `variants` could not carry the wording for them.

    The failure (code review of #11): edit a golden row and re-run against a warm cache, and
    both baselines and both ablations replay the **previous question's** contexts while the cell
    is scored against the new reference. `answer_key` carried the question all along, which made
    the same edit loud on the judged arms — as a `ContextDrift` blaming the collection — and
    silent here.
    """
    from finbrief.evaluation.pipeline import retrieval_key

    settings = a_settings()
    original = retrieval_key(
        a_row(), VECTOR_ONLY, k=5, settings=settings, fingerprint="f", variants=()
    )
    edited = retrieval_key(
        a_row(question="What supply chain risks does Tesla disclose?"),
        VECTOR_ONLY,
        k=5,
        settings=settings,
        fingerprint="f",
        variants=(),
    )

    assert original["question"] != edited["question"]
    assert original != edited, "a re-worded question must not replay the old contexts"


def test_a_re_ingest_invalidates_the_retrieval_key_through_the_fingerprint():
    from finbrief.evaluation.pipeline import retrieval_key

    settings = a_settings()
    keys = {
        digest: retrieval_key(
            a_row(), VECTOR_ONLY, k=5, settings=settings, fingerprint=digest, variants=()
        )
        for digest in ("before", "after")
    }

    assert keys["before"] != keys["after"]


# --- the turn scope the latency exclusion joins on ----------------------------------------


def test_each_retrieved_cell_scopes_its_log_lines_to_its_arm_and_row(tmp_path, filings_store):
    """`latency._planner_caps` joins `query_translation` to `retrieval` **by `turn_id`**.

    Delete or bypass this scope and every line reads `turn_id: None`, `caps` is empty, no
    ablation line is excluded, and both planner-off arms' cheap retrievals are pooled into
    ADR-0005's p50 — with the whole suite green. CLAUDE.md records the same failure for
    `app/Home.py`'s `log_turn`: "neutralising it once left all 1028 tests green", because every
    other test opened the scope itself. This one drives the emitter (code review of #11).
    """
    import io
    import json

    from finbrief.evaluation.cache import Cache
    from finbrief.evaluation.pipeline import retrieve_cells
    from finbrief.observability.logging_setup import configure_logging

    stream = io.StringIO()
    configure_logging(stream=stream)
    rows = (a_row(), a_row(id="X2", question="What does Apple say about services revenue?"))

    retrieve_cells(
        rows,
        VECTOR_ONLY,
        store=filings_store,
        settings=a_settings(),
        variants=None,
        cache=Cache(tmp_path / "cache"),
        k=5,
        fingerprint="f",
    )

    turn_ids = [
        json.loads(line).get("turn_id")
        for line in stream.getvalue().splitlines()
        if json.loads(line).get("event") == "retrieval"
    ]
    assert turn_ids == [f"vector/{rows[0].id}", "vector/X2"], turn_ids


def test_the_answer_stage_scopes_its_retrieval_lines_too(tmp_path, filings_store, monkeypatch):
    """The chain retrieves again, so those are `retrieval` lines and an unattributable one is
    exactly what `turn` prevents."""
    import io
    import json

    import finbrief.evaluation.pipeline as pipeline_module
    from finbrief.evaluation.cache import Cache
    from finbrief.observability.logging_setup import configure_logging
    from finbrief.rag import GroundedAnswer

    scored = pipeline_module.retrieve_cells(
        (a_row(),),
        HYBRID_ONLY,
        store=filings_store,
        settings=a_settings(),
        variants=None,
        cache=Cache(tmp_path / "cache"),
        k=5,
        fingerprint="f",
    )

    def fake_answer(question, **kwargs):
        import logging

        from finbrief.observability.logging_setup import log_event

        log_event(logging.getLogger("finbrief.rag"), "retrieval", latency_ms=1.0)
        return GroundedAnswer(text="an answer", contexts=scored[0].contexts)

    monkeypatch.setattr(pipeline_module, "answer_question", fake_answer)
    stream = io.StringIO()
    configure_logging(stream=stream)

    pipeline_module.answer_cells(
        (a_row(),),
        scored,
        HYBRID_ONLY,
        store=filings_store,
        settings=a_settings(),
        variants=None,
        cache=Cache(tmp_path / "answers"),
        k=5,
    )

    turn_ids = [
        json.loads(line).get("turn_id")
        for line in stream.getvalue().splitlines()
        if json.loads(line).get("event") == "retrieval"
    ]
    assert turn_ids == [f"answer/hybrid/{a_row().id}"]


# --- replay-only stages: `--stage` skipping one is not the stage never existing -------------


def test_a_replay_only_answer_stage_reports_absence_and_spends_nothing(tmp_path, filings_store):
    import finbrief.evaluation.pipeline as pipeline_module
    from finbrief.evaluation.cache import Cache

    scored = pipeline_module.retrieve_cells(
        (a_row(),),
        HYBRID_ONLY,
        store=filings_store,
        settings=a_settings(),
        variants=None,
        cache=Cache(tmp_path / "cache"),
        k=5,
        fingerprint="f",
    )

    answers = pipeline_module.answer_cells(
        (a_row(),),
        scored,
        HYBRID_ONLY,
        store=filings_store,
        settings=a_settings(),
        variants=None,
        cache=Cache(tmp_path / "answers"),
        k=5,
        replay_only=True,
    )

    assert answers == (None,), "absent, which is what the artifact will report"


def test_a_replay_only_judge_stage_leaves_every_metric_none(tmp_path, filings_store):
    """`--stage report` rendered the whole RAGAs table empty over 560 cached judge cells.

    Now it replays them; a cell the cache does not hold stays `None`, which is the same absence
    a skipped metric already produced.
    """
    import finbrief.evaluation.pipeline as pipeline_module
    from finbrief.evaluation.cache import Cache
    from finbrief.evaluation.judge import RETRIEVAL_METRICS

    cache = Cache(tmp_path / "cache")
    scored = pipeline_module.retrieve_cells(
        (a_row(),),
        VECTOR_ONLY,
        store=filings_store,
        settings=a_settings(),
        variants=None,
        cache=cache,
        k=5,
        fingerprint="f",
    )

    judged = pipeline_module.judge_cells(
        (a_row(),),
        scored,
        ("an answer",),
        metrics=RETRIEVAL_METRICS,
        settings=a_settings(),
        cache=cache,
        judge=None,
        embeddings=None,
        replay_only=True,
    )

    assert judged == ({"context_precision": None, "context_recall": None},)
    assert cache.stats("judge").absent == 2


# --- the fingerprint the retrieval key rests on -------------------------------------------


def a_chunk(index: int, content_hash: str):
    """One real `Chunk`, so the write goes through `write_chunks`' own contract."""
    from finbrief.ingestion.chunking import Chunk
    from finbrief.ingestion.model import Section

    return Chunk(
        id=f"AAPL:Item 1A:{index}",
        body=f"body {content_hash}",
        ticker="AAPL",
        filing_type="10-K",
        section=Section.RISK_FACTORS,
        fiscal_year=2025,
        accession="0000320193-25-000073",
        content_hash=content_hash,
    )


def a_store(tmp_path, label, chunks):
    # `fakes`, not `tests.fakes` — every other importer in this suite spells it this way, and
    # the difference is not cosmetic. pytest puts the test file's own directory on `sys.path`
    # (there is no `tests/__init__.py`), so `fakes` resolves everywhere; `tests.fakes` needs the
    # *repo root* on the path as well, which happens to hold under a local editable install and
    # did not in CI. It passed on the author's machine and failed on the runner — and it reached
    # the branch unnoticed because no CI run ever executed the commit that introduced it.
    from fakes import KeywordEmbeddings

    from finbrief.retrieval.vectorstore import build_filings_store, write_chunks

    store = build_filings_store(
        persist_directory=str(tmp_path / f"chroma-{label}"), embeddings=KeywordEmbeddings()
    )
    write_chunks(store, chunks)
    return store


def test_the_fingerprint_changes_when_bodies_change_under_stable_ids(tmp_path):
    """Ids alone would be wrong, which is the whole justification for this function.

    A re-ingest with a different chunker keeps the chunk ids and rewrites every body, so a
    fingerprint over ids would serve last ingest's cached retrievals under this ingest's name —
    and every deterministic metric in the artifact would then describe a corpus that no longer
    exists. The function was untested and its argument lived only in its docstring (code review
    of #11).
    """
    from finbrief.retrieval.vectorstore import collection_fingerprint

    before = collection_fingerprint(
        a_store(tmp_path, "before", [a_chunk(0, "aaa"), a_chunk(1, "bbb")])
    )
    after = collection_fingerprint(
        a_store(tmp_path, "after", [a_chunk(0, "aaa"), a_chunk(1, "ccc")])
    )

    assert before != after, "same ids, different bodies — a re-ingest must invalidate the cache"


def test_the_fingerprint_does_not_depend_on_insertion_order(tmp_path):
    from finbrief.retrieval.vectorstore import collection_fingerprint

    chunks = [a_chunk(index, f"h{index}") for index in range(3)]

    forward = collection_fingerprint(a_store(tmp_path, "forward", chunks))
    reverse = collection_fingerprint(a_store(tmp_path, "reverse", list(reversed(chunks))))

    assert forward == reverse


def test_a_replay_only_stage_returns_the_cells_the_measuring_run_paid_for(
    tmp_path, filings_store
):
    """The half that makes `--stage report` useful rather than merely cheap.

    A skipped stage is documented as replaying from the cache; the absence half is the honest
    fallback and this is the path a re-render actually takes. The cells are seeded through the
    **same** `judge_cache_key` the stage addresses them by, so a key that drifted would show up
    here as an absence rather than as a passing test over a coincidence.
    """
    import finbrief.evaluation.judge as judging
    import finbrief.evaluation.pipeline as pipeline_module
    from finbrief.evaluation.cache import Cache
    from finbrief.evaluation.judge import RETRIEVAL_METRICS

    cache = Cache(tmp_path / "cache")
    settings = a_settings()
    scored = pipeline_module.retrieve_cells(
        (a_row(),),
        VECTOR_ONLY,
        store=filings_store,
        settings=settings,
        variants=None,
        cache=cache,
        k=5,
        fingerprint="f",
    )
    sample = judging.JudgeSample(
        question=a_row().question,
        contexts=tuple(context.body for context in scored[0].contexts),
        answer="an answer",
        reference=a_row().reference,
    )
    for index, metric in enumerate(RETRIEVAL_METRICS):
        cache.resolve(
            "judge",
            judging.judge_cache_key(
                metric,
                sample,
                judge_model=settings.judge_model,
                embedding_model=settings.embedding_model,
                ragas_version=judging.ragas_version(),
            ),
            lambda index=index: {"score": 0.5 + index / 10},
        )

    replayed = pipeline_module.judge_cells(
        (a_row(),),
        scored,
        ("an answer",),
        metrics=RETRIEVAL_METRICS,
        settings=settings,
        cache=cache,
        judge=None,
        embeddings=None,
        replay_only=True,
    )

    assert replayed == ({"context_precision": 0.5, "context_recall": 0.6},)
    assert cache.stats("judge").absent == 0, "nothing was absent; the cells were on disk"
