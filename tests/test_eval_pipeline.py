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
