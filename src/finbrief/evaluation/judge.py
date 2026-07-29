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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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


def score(
    name: str, sample: JudgeSample, *, judge: Any, embeddings: Any = None
) -> float | None:
    """One metric over one sample — the paid call the cache wraps.

    Returns `None` when ragas returns `NaN`, which it does for a sample it could not score (an
    answer it extracted no statements from, for instance). `None` and not `0.0`: an unscoreable
    row is an absence, and averaging it as zero would drag a bucket mean down by the number of
    rows the *judge* failed on rather than the number the pipeline failed on.
    """
    import math

    metric = _metric(name, judge=judge, embeddings=embeddings)
    value = asyncio.run(metric.single_turn_ascore(sample.as_ragas_sample()))
    return None if value is None or math.isnan(value) else float(value)


def judge_cache_key(
    name: str,
    sample: JudgeSample,
    *,
    judge_model: str,
    ragas_version: str,
) -> dict[str, Any]:
    """Everything that determines this cell's score, and nothing that does not.

    The contexts are keyed by their **text**, not by chunk id: two arms can retrieve the same
    chunk ids in a different order, and context precision is order-sensitive (it is a mean over
    per-rank verdicts), so ids alone would let one arm's score be served to another. The ragas
    version is in the key because a metric's prompt is part of the measurement — an upgrade that
    reworded the NLI prompt would otherwise serve the old model's verdicts under the new one.
    """
    return {
        "metric": name,
        "judge_model": judge_model,
        "ragas_version": ragas_version,
        "question": sample.question,
        "contexts": list(sample.contexts),
        "answer": sample.answer,
        "reference": sample.reference,
    }


def ragas_version() -> str:
    """The installed ragas version, for the cache key and the artifact's provenance line."""
    from importlib.metadata import version

    return version("ragas")


def calls_per_row(name: str, *, k: int, batches_completions: bool = False) -> int:
    """How many judge calls one row costs on `name` — read off ragas 0.4.3, not estimated.

    - **faithfulness: 2.** Statement generation, then one NLI pass over all the statements. -
    **response relevancy: 1 on the shipped judge**, `RELEVANCY_STRICTNESS` otherwise. See
    below. - **context precision: `k`.** One call *per retrieved context* — which is why it is
    half the
      bill, and why this is a function of `k` rather than the constant it was first written as.
    - **context recall: 1.** `generate_multiple` at its default `n=1`.

    **`batches_completions` is not a detail, and the smoke run is why it exists.** ragas asks
    for
    `n=strictness` completions, then splits on whether the model serves them in one request:
    `ChatOpenAI` is in `ragas.llms.base.MULTIPLE_COMPLETION_SUPPORTED`, so the shipped judge
    sends **one** request with `n=3`; any other chat model gets `n` separate prompts. The
    default is therefore the shipped path, and the flag is what a scripted test model passes.

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
        return RELEVANCY_STRICTNESS if batches_completions else 1
    if name == CONTEXT_PRECISION:
        return k
    if name == CONTEXT_RECALL:
        return 1
    raise KeyError(f"unknown metric {name!r}. Valid: {', '.join(METRICS)}")


def expected_calls(
    metrics: Sequence[str], rows: int, *, k: int, batches_completions: bool = False
) -> int:
    """How many judge calls `rows` rows over `metrics` will cost at this `k`."""
    return rows * sum(
        calls_per_row(name, k=k, batches_completions=batches_completions) for name in metrics
    )
