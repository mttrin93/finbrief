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

import json
from typing import Any

import pytest

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

    assert len(model.calls) == calls_per_row(metric, k=len(SAMPLE.contexts))


def test_context_precision_scales_with_k_and_the_others_do_not():
    # The reason `calls_per_row` takes `k`: one call per retrieved context. Written as a
    # constant it would have been right at k=5 and wrong everywhere else, including in the
    # smoke run.
    assert calls_per_row(CONTEXT_PRECISION, k=5) == 5
    assert calls_per_row(CONTEXT_PRECISION, k=3) == 3
    assert calls_per_row(FAITHFULNESS, k=3) == calls_per_row(FAITHFULNESS, k=5) == 2


def test_the_full_six_arm_bill_is_the_number_the_plan_quoted():
    # 4 scored arms x 28 rows x 4 metrics, plus 2 ablation arms x 28 rows x the 2 retrieval
    # metrics. The figure #11's plan was approved on, derived rather than retyped.
    scored = expected_calls(METRICS, rows=28 * 4, k=5)
    ablations = expected_calls(RETRIEVAL_METRICS, rows=28 * 2, k=5)

    assert scored == 1232
    assert ablations == 336


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
