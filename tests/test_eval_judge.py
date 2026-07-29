"""The RAGAs judge, driven by a scripted model — so the plumbing is tested without spending.

Seam: ragas' metric objects take an injectable `llm`, so a fake chat model returning the JSON
each prompt asks for exercises the whole path — prompt rendering, ragas' output parsing, the
score arithmetic — with no network. That is worth more than mocking `single_turn_ascore`,
because what actually breaks on a ragas upgrade is the parsing and the call shape.

The test that earns its place here is the cost one: `judge.calls_per_row` is a table this
run's artifact quotes as spend, so it is asserted by **counting a fake judge's invocations**
rather than by being read off the library's source once and trusted. #11's whole cost estimate
rests on those four numbers.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from finbrief.config import Settings
from finbrief.evaluation.judge import (
    ANSWER_RELEVANCY,
    CONTEXT_PRECISION,
    CONTEXT_RECALL,
    EXCLUDED_FROM_HYPOTHESES,
    FAITHFULNESS,
    GENERATION_METRICS,
    METRICS,
    RETRIEVAL_METRICS,
    JudgeSample,
    build_judge,
    calls_per_row,
    expected_calls,
    judge_cache_key,
    ragas_version,
    score,
)

pytest.importorskip("ragas")

from langchain_core.callbacks import CallbackManagerForLLMRun  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402

SAMPLE = JudgeSample(
    question="What are the main risk factors for Tesla?",
    contexts=(
        "Tesla depends on thousands of parts from hundreds of suppliers.",
        "Risks Related to Our Operations.",
        "Risks Related to Government Laws and Regulations.",
    ),
    answer="Tesla flags supplier dependence and regulatory risk.",
    reference="Tesla's Item 1A flags supplier dependence and regulatory risk.",
)

#: What each prompt's output model wants back, keyed by a phrase from its own instruction.
#: Keyed on the instruction rather than on call order, so the fake answers whichever prompt it
#: is handed and a metric that reorders its calls does not silently get the wrong shape.
REPLIES: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "Break down each sentence",
        {"statements": ["Tesla flags supplier dependence.", "Tesla flags regulatory risk."]},
    ),
    (
        "judge the faithfulness",
        {
            "statements": [
                {
                    "statement": "Tesla flags supplier dependence.",
                    "reason": "stated",
                    "verdict": 1,
                },
                {"statement": "Tesla flags regulatory risk.", "reason": "stated", "verdict": 1},
            ]
        },
    ),
    (
        "Generate a question for the given answer",
        {"question": "What are Tesla's risk factors?", "noncommittal": 0},
    ),
    (
        "Given question, answer and context",
        {"reason": "the context supports the answer", "verdict": 1},
    ),
    (
        "classify if the sentence can be attributed",
        {
            "classifications": [
                {
                    "statement": "Item 1A flags supplier dependence and regulatory risk.",
                    "reason": "present in context",
                    "attributed": 1,
                }
            ]
        },
    ),
)


class ScriptedJudge(BaseChatModel):
    """A chat model that answers whichever ragas prompt it is handed, and counts its calls."""

    calls: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "finbrief-scripted-judge"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(str(message.content) for message in messages)
        self.calls.append(prompt)
        for marker, payload in REPLIES:
            if marker.lower() in prompt.lower():
                content = json.dumps(payload)
                break
        else:  # pragma: no cover — a new prompt shape would land here
            raise AssertionError(f"no scripted reply for this prompt:\n{prompt[:400]}")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


@pytest.fixture
def scripted():
    model = ScriptedJudge(calls=[])
    return model, LangchainLLMWrapper(model)


class FakeEmbeddings:
    """Deterministic vectors, so relevancy's cosine step is arithmetic and not a call."""

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


# --- the metrics, scored through the real ragas objects ------------------------------


def test_faithfulness_scores_a_supported_answer(scripted):
    _, judge = scripted

    assert score(FAITHFULNESS, SAMPLE, judge=judge) == pytest.approx(1.0)


def test_context_recall_scores_an_attributed_reference(scripted):
    _, judge = scripted

    assert score(CONTEXT_RECALL, SAMPLE, judge=judge) == pytest.approx(1.0)


def test_context_precision_scores_every_context(scripted):
    _, judge = scripted

    assert score(CONTEXT_PRECISION, SAMPLE, judge=judge) == pytest.approx(1.0)


def test_response_relevancy_scores_through_the_injected_embeddings(scripted):
    _, judge = scripted

    result = score(ANSWER_RELEVANCY, SAMPLE, judge=judge, embeddings=FakeEmbeddings())

    assert result == pytest.approx(1.0)


def test_an_unscoreable_row_is_absent_rather_than_zero(scripted, monkeypatch):
    # ragas returns NaN for a sample it could not score — an answer it extracted no statements
    # from, say. Averaging that as 0.0 would drag a bucket mean down by the number of rows the
    # *judge* failed on rather than the number the pipeline failed on.
    model, judge = scripted
    empty = json.dumps({"statements": []})

    def no_statements(*args, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=empty))])

    monkeypatch.setattr(type(model), "_generate", no_statements)

    assert score(FAITHFULNESS, SAMPLE, judge=judge) is None


# --- every cell is scored on one event loop -------------------------------------------


class LoopBoundPool:
    """Stands in for the `httpx` pool `langchain_openai` caches: bound to the loop that made it.

    **The structural property, enforced deterministically — and why it is asserted rather than
    provoked is itself part of the finding.** The genuine mechanism is
    `_LoopBoundMixin._get_loop`, which binds on first use and raises `RuntimeError: ... is bound
    to a different event loop` on any later one; that is what `anyio` hit inside a pooled
    connection and what the OpenAI SDK re-raised as `APIConnectionError: Connection error.`
    Driving it through `asyncio.Event` **self-heals**: the `wait()` that raises still runs its
    scheduled `set`, so ragas' `tenacity` retry finds the flag set, takes the already-set fast
    path, and the cell passes. Measured, not assumed — the first version of this double scored
    1.0 on the second loop.

    That is why the live failure was intermittent, and why three runs died at three different
    cell counts rather than on the first cell. A test inheriting that non-determinism does not
    pin the bug, so the condition is checked with `Future.get_loop()` — public API, no side
    effect, same `RuntimeError` — where a real pool would reach for its kept-alive connection.
    """

    def __init__(self) -> None:
        self.pooled: Any = None
        self.loops: list[int] = []

    async def acquire(self) -> None:
        running = asyncio.get_running_loop()
        self.loops.append(id(running))
        if self.pooled is None:
            self.pooled = running.create_future()
            running.call_soon(self.pooled.set_result, None)
        if self.pooled.get_loop() is not running:
            raise RuntimeError(f"{self.pooled!r} is bound to a different event loop")
        await self.pooled


class PooledJudge(BaseChatModel):
    """A judge reaching a `LoopBoundPool` on its async path, as a real one reaches httpx."""

    pool: Any = None

    @property
    def _llm_type(self) -> str:
        return "finbrief-pooled-judge"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:  # pragma: no cover — ragas takes the async path
        raise AssertionError("the async path is the one under test")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        await self.pool.acquire()
        prompt = "\n".join(str(message.content) for message in messages)
        for marker, payload in REPLIES:
            if marker.lower() in prompt.lower():
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=json.dumps(payload)))]
                )
        raise AssertionError(f"no scripted reply for this prompt:\n{prompt[:400]}")


def test_one_client_survives_every_cell_because_they_share_one_loop():
    """The regression test. Under `asyncio.run` per cell this raises on the second cell.

    A pooled connection outliving the loop that created it is what killed three consecutive
    full evaluation runs, reported as `APIConnectionError: Connection error.` while `curl`
    returned 200. One client, many cells, no raise — that is the whole contract.
    """
    pool = LoopBoundPool()
    shared = LangchainLLMWrapper(PooledJudge(pool=pool))

    for _ in range(4):
        assert score(FAITHFULNESS, SAMPLE, judge=shared) == pytest.approx(1.0)

    assert len(set(pool.loops)) == 1, f"cells ran on {len(set(pool.loops))} loops, not one"


def test_concurrent_cells_share_that_loop_too_because_the_workers_are_threads():
    """`run.cached_map` scores cells from a thread pool, and they must reach the same loop.

    `run_coroutine_threadsafe` is the documented cross-thread entry point; a per-thread loop
    would reintroduce the bug at `--workers 3` while passing every serial test.
    """
    pool = LoopBoundPool()
    shared = LangchainLLMWrapper(PooledJudge(pool=pool))

    with ThreadPoolExecutor(max_workers=3) as workers:
        results = list(
            workers.map(lambda _: score(FAITHFULNESS, SAMPLE, judge=shared), range(6))
        )

    assert results == [pytest.approx(1.0)] * 6
    assert len(set(pool.loops)) == 1


def test_building_a_client_per_cell_would_not_have_fixed_it():
    """Why the fix is the loop and not the client — the constraint that settled the design.

    `langchain_openai` caches the async client below this repo's constructor
    (`_cached_async_httpx_client`, `@lru_cache`d on `(base_url, timeout, socket_options)`), so
    two `build_judge()` calls return distinct `ChatOpenAI` objects sharing **one**
    `httpx.AsyncClient`. Constructing per cell therefore changes nothing about which loop the
    pool is bound to — established by a fourth killed run, and pinned here so an upgrade that
    changes it is visible rather than silently widening the options.
    """
    settings = Settings.from_env(
        {"OPENROUTER_API_KEY": "test-key", "SEC_EDGAR_USER_AGENT": "FinBrief test@example.com"}
    )

    first, second = build_judge(settings), build_judge(settings)

    assert first.langchain_llm is not second.langchain_llm
    assert (
        first.langchain_llm.root_async_client._client
        is second.langchain_llm.root_async_client._client
    )


# --- the cost table, measured rather than asserted -----------------------------------


@pytest.mark.parametrize(
    ("metric", "needs_embeddings"),
    [
        (FAITHFULNESS, False),
        (ANSWER_RELEVANCY, True),
        (CONTEXT_PRECISION, False),
        (CONTEXT_RECALL, False),
    ],
)
def test_the_cost_table_matches_what_ragas_actually_calls(scripted, metric, needs_embeddings):
    # **The test #11's whole cost estimate rests on.** `calls_per_row` is what the artifact
    # quotes as spend; here it is checked against the number of times a real ragas metric
    # reaches its model. A ragas upgrade that adds an ensemble pass, or reprices context
    # precision by fetching per-context verdicts in one call, fails here instead of silently
    # making the reported bill wrong.
    model, judge = scripted

    score(
        metric,
        SAMPLE,
        judge=judge,
        embeddings=FakeEmbeddings() if needs_embeddings else None,
    )

    # `separate_completion_requests=True`: this fake is not a `ChatOpenAI`, so ragas cannot ask
    # it for n completions in one request and sends n prompts instead. The shipped judge *is*
    # one, which is the branch the default covers — see the next test. The flag used to be
    # called `batches_completions` while selecting exactly this, the *un*batched branch (review
    # of #11).
    assert len(model.calls) == calls_per_row(
        metric, k=len(SAMPLE.contexts), separate_completion_requests=True
    )


def test_the_shipped_judge_costs_one_relevancy_call_and_a_scripted_one_costs_three():
    # **The smoke run's second finding.** ragas splits on
    # `ragas.llms.base.MULTIPLE_COMPLETION_SUPPORTED`, which holds `ChatOpenAI`: the shipped
    # judge gets one request with n=3, anything else gets 3 prompts. The cost table defaulted
    # to the wrong branch, and the test above passed anyway because the fake takes the other
    # one.
    from langchain_openai import ChatOpenAI
    from ragas.llms.base import is_multiple_completion_supported

    assert is_multiple_completion_supported(
        ChatOpenAI(api_key="test-key", model="openai/gpt-4.1-mini")
    )
    assert calls_per_row(ANSWER_RELEVANCY, k=5) == 1
    assert calls_per_row(ANSWER_RELEVANCY, k=5, separate_completion_requests=True) == 3


def test_context_precision_scales_with_k_and_the_others_do_not():
    # The reason `calls_per_row` takes `k`: one call per retrieved context. Written as a
    # constant it would have been right at k=5 and wrong everywhere else, including in the
    # smoke run.
    assert calls_per_row(CONTEXT_PRECISION, k=5) == 5
    assert calls_per_row(CONTEXT_PRECISION, k=3) == 3
    assert calls_per_row(FAITHFULNESS, k=3) == calls_per_row(FAITHFULNESS, k=5) == 2


def test_the_full_six_arm_bill_is_the_number_the_artifact_quotes():
    # 4 scored arms x 28 rows x 4 metrics, plus 2 ablation arms x 28 rows x the 2 retrieval
    # metrics — derived rather than retyped, on the shipped judge's call shape.
    #
    # **#11's plan quoted 1,232 + 336 and that was wrong**, because it costed response
    # relevancy at 3 calls. The shipped judge asks for its 3 completions in one request, so
    # the scored half is 4 x 28 x (2+1+5+1) = 1,008 and the run is ~224 calls cheaper than
    # approved. Corrected here rather than in prose, so the artifact's cost line and this test
    # cannot disagree.
    scored = expected_calls(METRICS, rows=28 * 4, k=5)
    ablations = expected_calls(RETRIEVAL_METRICS, rows=28 * 2, k=5)

    assert scored == 1008
    assert ablations == 336
    assert expected_calls(METRICS, rows=28 * 4, k=5, separate_completion_requests=True) == 1232


def test_an_unknown_metric_raises_rather_than_costing_nothing_silently():
    with pytest.raises(KeyError):
        calls_per_row("context_utilization", k=5)


# --- the cache key -------------------------------------------------------------------


def test_the_key_covers_every_input_that_moves_the_score():
    key = judge_cache_key(
        FAITHFULNESS, SAMPLE, judge_model="openai/gpt-4.1-mini", ragas_version="0.4.3"
    )

    assert key["metric"] == FAITHFULNESS
    assert key["judge_model"] == "openai/gpt-4.1-mini"
    assert key["ragas_version"] == "0.4.3"
    assert key["contexts"] == list(SAMPLE.contexts)


def test_reordered_contexts_are_a_different_key():
    # Context precision is a mean over *per-rank* verdicts, so the same chunks in a different
    # order score differently — and two arms routinely retrieve the same ids in a different
    # order. Keying on ids alone would serve one arm's score to the other.
    reordered = JudgeSample(
        question=SAMPLE.question,
        contexts=tuple(reversed(SAMPLE.contexts)),
        answer=SAMPLE.answer,
        reference=SAMPLE.reference,
    )

    original_key = judge_cache_key(
        CONTEXT_PRECISION, SAMPLE, judge_model="m", ragas_version="0.4.3"
    )
    reordered_key = judge_cache_key(
        CONTEXT_PRECISION, reordered, judge_model="m", ragas_version="0.4.3"
    )

    assert original_key != reordered_key


def test_a_ragas_upgrade_invalidates_the_cache():
    # A metric's prompt is part of the measurement: an upgrade that reworded the NLI instruction
    # would otherwise serve the old model's verdicts under the new one's name.
    before = judge_cache_key(FAITHFULNESS, SAMPLE, judge_model="m", ragas_version="0.4.3")
    after = judge_cache_key(FAITHFULNESS, SAMPLE, judge_model="m", ragas_version="0.5.0")

    assert before != after


def test_the_installed_version_is_readable():
    # It goes in the key and in the artifact's provenance line, so it has to be resolvable
    # in-process rather than typed into the report.
    assert ragas_version().startswith("0.")


# --- the metric registry -------------------------------------------------------------


def test_the_four_metrics_are_adr_0002s_four():
    assert METRICS == (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )


def test_the_split_between_generation_and_retrieval_metrics_is_exhaustive():
    # The ablation arms are scored on the retrieval half alone, so the split has to cover the
    # four without overlap — a metric in neither would silently never run.
    assert set(GENERATION_METRICS) | set(RETRIEVAL_METRICS) == set(METRICS)
    assert not set(GENERATION_METRICS) & set(RETRIEVAL_METRICS)


def test_the_fenced_metric_is_not_one_the_ablation_arms_rest_on():
    # ADR-0005's two pre-registered decisions rest on the retrieval half, and the fence must not
    # touch it or both decisions become unanswerable.
    assert not EXCLUDED_FROM_HYPOTHESES & set(RETRIEVAL_METRICS)
    assert set(METRICS) > EXCLUDED_FROM_HYPOTHESES


def test_response_relevancy_is_keyed_on_the_embedding_model_and_the_others_are_not():
    """A cosine in the embedding model's space, cached under a key that did not name it.

    Change `FINBRIEF_EMBEDDING_MODEL` and `retrieval_key` re-pays (it carries the model) while
    every relevancy cell replayed a similarity computed in the *old* vector space, under a
    provenance table printing the new model (code review of #11). Conditional rather than
    uniform because the other three metrics never touch an embedding, and widening their key
    would re-pay ~1,300 judge calls to record something that cannot move them.
    """
    sample = JudgeSample(question="q", contexts=("c",), answer="a", reference="r")

    def key(metric, embedding_model):
        return judge_cache_key(
            metric,
            sample,
            judge_model="openai/gpt-4.1-mini",
            ragas_version="0.4.3",
            embedding_model=embedding_model,
        )

    small = key(ANSWER_RELEVANCY, "openai/text-embedding-3-small")
    large = key(ANSWER_RELEVANCY, "openai/text-embedding-3-large")

    assert small != large
    assert small["embedding_model"] == "openai/text-embedding-3-small"

    for metric in (FAITHFULNESS, CONTEXT_PRECISION, CONTEXT_RECALL):
        assert "embedding_model" not in key(metric, "openai/text-embedding-3-small")
        assert key(metric, "small") == key(metric, "large")


# --- the loop singleton's confinement, which is a claim that can stop being true -------


def test_importing_the_judge_starts_no_loop_thread():
    """`build_once` constructs on first call, so an import costs nothing.

    Half of the confinement argument: even a future import from the app side would spawn no
    thread until something scored a cell. Asserted rather than reasoned, because the module is
    already imported by the time this runs — so the check is that no thread exists *named for
    it* until `score` is called, which the tests above do.
    """
    import importlib

    module = importlib.import_module("finbrief.evaluation.judge")

    assert module._judging_loop.cache_info().currsize in (0, 1)
    if module._judging_loop.cache_info().currsize == 0:
        assert not [t for t in threading.enumerate() if t.name == "finbrief-judge-loop"]


def test_only_the_eval_entry_point_imports_the_evaluation_package():
    """The other half: nothing the Streamlit app loads can reach the loop at all.

    A static scan rather than an import-time hook, because the failure it guards against is
    someone adding `from finbrief.evaluation import ...` to a module the app imports — at which
    point a Streamlit session would spawn a judge loop it never uses. `scripts/evaluate.py` is
    the one legitimate importer.
    """
    root = Path(__file__).resolve().parent.parent
    importer = re.compile(r"^\s*(?:from|import)\s+finbrief\.evaluation\b", re.MULTILINE)
    searched = [
        path
        for directory in ("app", "src/finbrief", "scripts")
        for path in (root / directory).rglob("*.py")
        if "finbrief/evaluation/" not in path.as_posix()
    ]
    assert searched, "the scan found no files, so it would pass on a renamed tree"

    importers = sorted(
        path.relative_to(root).as_posix()
        for path in searched
        if importer.search(path.read_text(encoding="utf-8"))
    )

    assert importers == ["scripts/evaluate.py"], (
        f"{importers} import finbrief.evaluation. Anything the app loads that reaches "
        f"evaluation.judge gives a Streamlit session a judge event loop it never uses."
    )
