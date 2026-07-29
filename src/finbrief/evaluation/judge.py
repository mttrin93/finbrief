"""The RAGAs judge: one model, four metrics, one call per (row, metric) — and a cache key.

ADR-0002 names all four metrics; this module is the only place `ragas` is called from, for the
reason `retrieval/vectorstore.py` is the only place Chroma is opened. Two of those reasons are
specific to this library and neither is cosmetic.

**It phones home by default.** `ragas._analytics.track` POSTs every metric completion to
`t.explodinggradients.com`, decorated `@silent`, flushed from a background thread and again at
`atexit`. `_silence_ragas_telemetry` below is what stops it, and it runs **before** `ragas` is
imported anywhere in this process because `ragas._analytics.do_not_track` is `lru_cache`d — a
switch flipped after the first metric has run is a switch that did nothing. `tests/conftest.py`
sets the same variable for the suite; two mechanisms, failing independently, on the
`security/advice.py` principle that a hole is a hole whether today's code walks through it.
Measured with the switch absent: one `track()` call attempted `t.explodinggradients.com`, raised
nothing, logged nothing, and was visible only in `conftest.EGRESS_ATTEMPTS`.

**`ragas.evaluate()` is deliberately not used.** Three reasons, in order of weight: the cache
this harness turns on needs a result per *(row, metric)* and `evaluate()` returns a table over a
dataset; a metric has to be skippable per row (ADR-0002's amendment, and the
`tool-augmented` relevancy column below); and `evaluate()` is one of the two entry points
decorated with the analytics tracker. Calling `metric.single_turn_ascore` directly costs a
`for` loop and buys all three.

**One metric here is not reproducible, and it is fenced off rather than caveated — for two
measured reasons, not one.** `ragas.llms.base.get_temperature` returns **0.3 whenever n > 1**,
and `ResponseRelevancy` asks for `n=strictness=3`. So response relevancy moves between runs on
any judge at any temperature this code names. **The second is worse and was found by running
it:** the `n=3` is not honoured. OpenRouter answers that single request with one completion,
logging `LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.` on
every judged cell of every run so far — so the metric is a cosine similarity against **one**
temperature-0.3 question rather than the mean over three it is defined as. It is therefore
excluded from every pre-registered decision — see `EXCLUDED_FROM_HYPOTHESES` and ADR-0002's
T10 amendment. ADR-0005's falsification clause and its §4 re-examination trigger both rest on
context precision and context recall, which are single-call metrics at temperature 0.01 and
reproducible to the judge's own sampling.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from finbrief.caching import build_once

#: Every metric name a pre-registered decision may **not** rest on.
#:
#: Response relevancy is non-reproducible by construction (see the module docstring), so a delta
#: in that column is not evidence of anything and must never be read as confirming or refuting a
#: hypothesis. Named as data rather than left to prose because `report.py` renders the exclusion
#: into the artifact and `hypotheses.py` asserts against it: a caveat in a paragraph is a caveat
#: the next reader of the table does not see.
EXCLUDED_FROM_HYPOTHESES: frozenset[str] = frozenset({"answer_relevancy"})


def _silence_ragas_telemetry(env: dict[str, str] | None = None) -> None:
    """Set `RAGAS_DO_NOT_TRACK`, before anything in this process imports `ragas`.

    Takes the mapping as an argument so the behaviour is testable without mutating the real
    environment — the same reason `Settings.from_env` does. Assigned rather than defaulted: a
    developer with `RAGAS_DO_NOT_TRACK=false` exported is exactly the case worth overriding.
    """
    target = os.environ if env is None else env
    target["RAGAS_DO_NOT_TRACK"] = "true"


_silence_ragas_telemetry()

#: The four metrics ADR-0002 names, in the order the artifact's columns run.
#:
#: Named here rather than passed around as strings so that a typo is an import error, and so the
#: two the pre-registered decisions rest on are visibly the two that are *not* fenced off above.
FAITHFULNESS = "faithfulness"
ANSWER_RELEVANCY = "answer_relevancy"
CONTEXT_PRECISION = "context_precision"
CONTEXT_RECALL = "context_recall"

METRICS: tuple[str, ...] = (FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION, CONTEXT_RECALL)

#: The two that need a generated answer. The other two need only contexts and the reference,
#: which is why the ablation arms can be scored without paying for a generation.
GENERATION_METRICS: tuple[str, ...] = (FAITHFULNESS, ANSWER_RELEVANCY)
RETRIEVAL_METRICS: tuple[str, ...] = (CONTEXT_PRECISION, CONTEXT_RECALL)

#: `ResponseRelevancy.strictness` — how many questions it generates from the answer. ragas'
#: default, named here because it is a term in the cost arithmetic *and* the reason that
#: metric is fenced off: `ragas.llms.base.get_temperature` returns 0.3 whenever n > 1.
RELEVANCY_STRICTNESS = 3


@dataclass(frozen=True, slots=True)
class JudgeSample:
    """What a metric is scored over: one question, its contexts, its answer, its reference.

    A plain dataclass rather than ragas' `SingleTurnSample` at the harness boundary, so the
    cache key is built from fields this repo owns and a ragas rename cannot silently change an
    address.
    """

    question: str
    contexts: tuple[str, ...]
    answer: str
    reference: str

    def as_ragas_sample(self) -> Any:
        from ragas import SingleTurnSample

        return SingleTurnSample(
            user_input=self.question,
            retrieved_contexts=list(self.contexts),
            response=self.answer,
            reference=self.reference,
        )


def build_judge(settings: Any) -> Any:
    """The judge, wrapping the chat model `llm.py` builds for `settings.judge_model`.

    **Wrapping our own model rather than letting ragas build one**, which is what
    `ragas.llm_factory` would do: `llm.py` is the only chat-model constructor (CLAUDE.md), and a
    second one would put the judge outside this repo's timeout, retry and base-URL configuration
    — and outside `Settings`, so the artifact could not say which model scored it.

    `LangchainLLMWrapper` is **deprecated** in ragas 0.4.3, and that is accepted knowingly:
    the non-deprecated path is `instructor`-based and takes an `openai` client of its own
    making. One constructor is worth more than one deprecation warning.
    """
    from ragas.llms import LangchainLLMWrapper

    from finbrief.llm import build_chat_model

    return LangchainLLMWrapper(
        build_chat_model(settings, model=settings.judge_model, temperature=0.0)
    )


def build_judge_embeddings(settings: Any) -> Any:
    """The embeddings response relevancy compares question vectors with.

    `retrieval/embeddings.py` is the only embeddings constructor and ingest and query must share
    it (CLAUDE.md). The judge shares it too — not because a different model would be *wrong*
    here, since this vector never touches the collection, but because a second constructor is a
    second thing to configure and the cost is four embeddings a row.
    """
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from finbrief.retrieval.embeddings import build_embeddings

    return LangchainEmbeddingsWrapper(build_embeddings(settings))


def _metric(name: str, *, judge: Any, embeddings: Any) -> Any:
    """The ragas metric object for `name`, wired to this repo's judge.

    `LLMContextPrecisionWithReference` and `LLMContextRecall` are named explicitly rather than
    taken from ragas' module-level singletons: the singletons carry no LLM and the
    reference-free variants score against the *response* instead of the reference, which would
    compare the pipeline against itself — ADR-0002's circularity, arriving through a default
    argument.
    """
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    if name == FAITHFULNESS:
        return Faithfulness(llm=judge)
    if name == ANSWER_RELEVANCY:
        return ResponseRelevancy(llm=judge, embeddings=embeddings)
    if name == CONTEXT_PRECISION:
        return LLMContextPrecisionWithReference(llm=judge)
    if name == CONTEXT_RECALL:
        return LLMContextRecall(llm=judge)
    raise KeyError(f"unknown metric {name!r}. Valid: {', '.join(METRICS)}")


#: The one event loop every judged cell runs on — a **seventh** process-level singleton, and
#: therefore through `caching.build_once` like the other six (CLAUDE.md). It is the only one
#: that
#: owns a *thread* rather than a client, so its confinement and its lifecycle are both stated
#: rather than left to be inferred.
#:
#: **Confined to the harness, structurally and lazily.** `finbrief.evaluation` is imported by
#: `scripts/evaluate.py` and by the tests, and by nothing under `app/` or elsewhere in the
#: package — so the Streamlit app cannot reach this. And even if it could, `build_once`
#: constructs on first *call*: importing this module starts no thread, so a future import from
#: the app side would still cost nothing until something scored a cell. Both halves are asserted
#: in `tests/test_eval_judge.py`, because "nothing imports it" is exactly the kind of claim that
#: silently stops being true.
#:
#: **Never stopped, never joined, and that is a decision.** The loop must outlive the judge
#: stage: `langchain_openai` caches the `httpx.AsyncClient` beneath the judge for the life of
#: the process, so a loop closed between stages leaves that client bound to a dead one — which
#: is the defect this exists to remove, reintroduced by tidying up. There is no correct place to
#: close it that is not "at process exit", so it is a **daemon** thread and the interpreter ends
#: it. That is acceptable *because this is a script*: `scripts/evaluate.py` is the only entry
#: point, it writes its artifact and returns, and every paid cell is already durable in the
#: cache
#: before the process ends. It would not be acceptable in a long-lived server, and a caller that
#: makes this library-like owes it an explicit shutdown.
#:
#: Checked rather than assumed, since this repo has twice shipped a handler dropped without
#: closing: a process that scores cells and exits emits no `Task was destroyed but it is
#: pending` and no unclosed-session `ResourceWarning` (measured under
#: `-W error::ResourceWarning`).
#: There is nothing pending at exit because `score` blocks on each cell's future — the loop is
#: idle between cells by construction, not by luck.
@build_once
def _judging_loop() -> Any:
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, name="finbrief-judge-loop", daemon=True).start()
    return loop


def score(
    name: str, sample: JudgeSample, *, judge: Any, embeddings: Any = None
) -> float | None:
    """One metric over one sample — the paid call the cache wraps.

    Returns `None` when ragas returns `NaN`, which it does for a sample it could not score (an
    answer it extracted no statements from, for instance). `None` and not `0.0`: an unscoreable
    row is an absence, and averaging it as zero would drag a bucket mean down by the number of
    rows the *judge* failed on rather than the number the pipeline failed on.

    **Every cell runs on one shared loop, and `asyncio.run` per cell is the defect this
    replaces** (code review of #11). `asyncio.run` builds and *closes* a fresh loop per call.
    `httpx` binds a pooled connection's `asyncio.Event` to the loop that first used it, so the
    moment a keep-alive connection survived into the next cell's loop `anyio` raised
    `RuntimeError: <asyncio.locks.Event ...> is bound to a different event loop` — which the
    OpenAI SDK catches and re-raises as **`APIConnectionError: Connection error.`**, a network
    fault that was not a network fault. It killed three consecutive full runs at three different
    cell counts, and `curl` returned HTTP 200 throughout.

    **Constructing a client per cell does not fix it, and finding out why is what settled the
    design.** `langchain_openai` caches the underlying async client itself:
    `_cached_async_httpx_client` is `@lru_cache`d on `(base_url, timeout, socket_options)`, all
    of which are constant here, so two `build_judge()` calls return distinct `ChatOpenAI`
    objects sharing **one** `httpx.AsyncClient` and therefore one connection pool. A fourth run
    died exactly as the first three did with per-cell construction in place. The client cannot
    be made to not outlive a loop; the loop has to stop being per-cell.

    So the loop is the singleton and the clients are ordinary arguments again. Concurrency is
    unchanged — `run.cached_map`'s workers submit to that loop with `run_coroutine_threadsafe`,
    which is the documented cross-thread entry point — and the shared pool is now an advantage
    rather than a hazard, since connections are reused across cells the way the library intends.

    It stayed hidden until this ticket because every earlier run replayed its relevancy cells
    from cache, and relevancy is the only metric that drives both clients: the committed
    artifact was produced by a run reporting `judge 560 replayed / 0 paid`.
    """
    import math

    async def scored() -> float | None:
        metric = _metric(name, judge=judge, embeddings=embeddings)
        return await metric.single_turn_ascore(sample.as_ragas_sample())

    value = asyncio.run_coroutine_threadsafe(scored(), _judging_loop()).result()
    return None if value is None or math.isnan(value) else float(value)


def judge_cache_key(
    name: str,
    sample: JudgeSample,
    *,
    judge_model: str,
    ragas_version: str,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Everything that determines this cell's score, and nothing that does not.

    The contexts are keyed by their **text**, not by chunk id: two arms can retrieve the same
    chunk ids in a different order, and context precision is order-sensitive (it is a mean over
    per-rank verdicts), so ids alone would let one arm's score be served to another. The ragas
    version is in the key because a metric's prompt is part of the measurement — an upgrade that
    reworded the NLI prompt would otherwise serve the old model's verdicts under the new one.

    **`embedding_model` for response relevancy, and only for it**, which the docstring's own
    argument about `ragas_version` required all along (code review of #11). `ResponseRelevancy`
    is a cosine between the question and a model-generated one *in the embedding model's space*
    (`build_judge_embeddings`), so a changed `FINBRIEF_EMBEDDING_MODEL` re-paid every retrieval
    cell — `retrieval_key` carries it — while every relevancy cell replayed a similarity
    computed in the old space, under a provenance table printing the new model. Conditional
    rather than uniform because the other three metrics never touch an embedding, and widening
    their key would re-pay ~1,300 judge calls to record something that cannot move them.
    """
    key: dict[str, Any] = {
        "metric": name,
        "judge_model": judge_model,
        "ragas_version": ragas_version,
        "question": sample.question,
        "contexts": list(sample.contexts),
        # Over-keyed on purpose for the reference-based metrics, which score contexts against
        # the reference and never read the response: a prompt edit re-pays them. Over-keying
        # serves a stale value to nobody, which is the direction a cache may err in.
        "answer": sample.answer,
        "reference": sample.reference,
    }
    if name == ANSWER_RELEVANCY:
        key["embedding_model"] = embedding_model
    return key


def ragas_version() -> str:
    """The installed ragas version, for the cache key and the artifact's provenance line."""
    from importlib.metadata import version

    return version("ragas")


def calls_per_row(name: str, *, k: int, separate_completion_requests: bool = False) -> int:
    """How many judge calls one row costs on `name` — read off ragas 0.4.3, not estimated.

    - **faithfulness: 2.** Statement generation, then one NLI pass over all the statements. -
    **response relevancy: 1 on the shipped judge**, `RELEVANCY_STRICTNESS` otherwise. See
    below. - **context precision: `k`.** One call *per retrieved context* — which is why it is
    half the
      bill, and why this is a function of `k` rather than the constant it was first written as.
    - **context recall: 1.** `generate_multiple` at its default `n=1`.

    **`separate_completion_requests` is not a detail, and the smoke run is why it exists.**
    ragas asks for `n=strictness` completions, then splits on whether the model serves them in
    one request: `ChatOpenAI` is in `ragas.llms.base.MULTIPLE_COMPLETION_SUPPORTED`, so the
    shipped judge sends **one** request with `n=3`, while any other chat model gets `n` separate
    prompts. The default is therefore the shipped path, and the flag is what a scripted test
    model passes.

    It is named for the branch it selects, which is a correction: it was `batches_completions`,
    and `batches_completions=True` returned the **un**batched count of 3 — a flag asserting the
    opposite of what it did, with its own tests and docstring passing `True` to mean "does *not*
    batch" (code review of #11). The name now matches the arithmetic.

    Measured live on 2026-07-29, and it changes the metric as well as the bill: OpenRouter
    answered that one request with **one** completion, logging `LLM returned 1 generations
    instead of requested 3. Proceeding with 1 generations.` eight times over eight judged
    cells. So response relevancy is computed over a single generated question rather than
    three — a *third* reason it is fenced off from every pre-registered decision
    (`EXCLUDED_FROM_HYPOTHESES`), after the forced temperature of 0.3 and the sampling that
    follows from it.

    A cost table nobody checked is a cost table that is wrong, so
    `tests/test_eval_judge.py::test_the_cost_table_matches_what_ragas_actually_calls` counts a
    fake judge's invocations against these numbers. That test passed against the 3-call figure
    while the real run cost 1, because the fake takes the batching path — which is exactly why
    the flag is now explicit rather than implied.
    """
    if name == FAITHFULNESS:
        return 2
    if name == ANSWER_RELEVANCY:
        return RELEVANCY_STRICTNESS if separate_completion_requests else 1
    if name == CONTEXT_PRECISION:
        return k
    if name == CONTEXT_RECALL:
        return 1
    raise KeyError(f"unknown metric {name!r}. Valid: {', '.join(METRICS)}")


def expected_calls(
    metrics: Sequence[str], rows: int, *, k: int, separate_completion_requests: bool = False
) -> int:
    """How many judge calls `rows` rows over `metrics` will cost at this `k`."""
    return rows * sum(
        calls_per_row(name, k=k, separate_completion_requests=separate_completion_requests)
        for name in metrics
    )
