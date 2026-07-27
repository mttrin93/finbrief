"""The deterministic RAG chain: question -> contexts -> grounded answer (ADR-0003).

The chain the evaluation harness measures, so it is tested at its own seam rather than
through the UI. `retrieve()` is real (against a fixture collection, seam 1's fake
embedding); only the chat model is faked — a real one would be a network call, and the
suite is hermetic by contract (CLAUDE.md).
"""

from __future__ import annotations

from fakes import a_context
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from finbrief.config import RetrievalStrategy
from finbrief.ingestion.model import Section
from finbrief.prompts import NO_CONTEXT_FALLBACK, SYSTEM_PROMPT
from finbrief.rag import answer_question


class RecordingFakeChatModel(GenericFakeChatModel):
    """A fake chat model that keeps the prompts it was handed."""

    prompts: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def a_model(reply="Apple flags supply-chain concentration [1].") -> RecordingFakeChatModel:
    return RecordingFakeChatModel(messages=iter([reply]), prompts=[])


QUESTION = "What are the risks to Apple's supply chain?"


def test_the_answer_carries_the_contexts_it_was_grounded_in(filings_store):
    answer = answer_question(QUESTION, k=3, store=filings_store, model=a_model())

    assert answer.text == "Apple flags supply-chain concentration [1]."
    assert [context.rank for context in answer.contexts] == [1, 2, 3]
    assert answer.contexts[0].section is Section.RISK_FACTORS
    assert answer.grounded


def test_the_model_is_shown_the_persona_and_the_numbered_contexts(filings_store):
    model = a_model()

    answer = answer_question(QUESTION, k=2, store=filings_store, model=model)

    (prompt,) = model.prompts
    system, human = prompt
    assert system.text == SYSTEM_PROMPT
    # Each context is numbered with the marker the answer is asked to cite, and labelled
    # with the provenance the sources panel shows beside it.
    assert "[1] AAPL 10-K FY2025, Item 1A" in human.text
    assert "[2] " in human.text
    assert answer.contexts[0].body[:60] in human.text
    assert QUESTION in human.text


def test_the_contexts_are_framed_as_data_the_question_as_the_instruction(filings_store):
    # ADR-0006 quarantines retrieved text: a filing that contains "ignore your
    # instructions" must read as evidence, not as a command. Phase 5 tests the gate; this
    # asserts the framing the gate assumes is already in place.
    model = a_model()

    answer_question(QUESTION, k=2, store=filings_store, model=model)

    (_, human) = model.prompts[0]
    assert "<sources>" in human.text and "</sources>" in human.text
    assert human.text.index("<sources>") < human.text.index(QUESTION), (
        "the user's question must come after the quarantined sources, not be wrapped in them"
    )


def test_an_unanswerable_question_gets_the_fallback_without_calling_the_model(
    empty_filings_store,
):
    # Retrieval-level error handling (spec §Tools): empty result -> fallback message. The
    # model is never asked, because a model with no contexts is exactly the ungrounded
    # guess this project exists to avoid.
    model = a_model()

    answer = answer_question(QUESTION, k=3, store=empty_filings_store, model=model)

    assert answer.text == NO_CONTEXT_FALLBACK
    assert answer.contexts == ()
    assert not answer.grounded
    assert model.prompts == [], "no contexts must mean no generation"


def test_the_chain_passes_its_strategy_and_k_to_retrieval(filings_store, monkeypatch):
    import finbrief.rag as rag

    calls = []

    def fake_retrieve(question, *, strategy, k, store, settings=None):
        calls.append({"question": question, "strategy": strategy, "k": k})
        return (a_context(ticker="AAPL", body="A risk factor."),)

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)

    answer_question(
        QUESTION,
        strategy=RetrievalStrategy.VECTOR,
        k=7,
        store=filings_store,
        model=a_model(),
    )

    assert calls == [{"question": QUESTION, "strategy": RetrievalStrategy.VECTOR, "k": 7}], (
        "the chain must not translate or rewrite the question in Phase 2 (ADR-0004)"
    )


def test_the_turn_is_logged_with_the_strategy_and_what_grounded_it(filings_store, caplog):
    with caplog.at_level("INFO", logger="finbrief.rag"):
        answer = answer_question(QUESTION, k=2, store=filings_store, model=a_model())

    (record,) = [r for r in caplog.records if getattr(r, "event", None) == "rag_answer"]
    assert record.fields["strategy"] == "vector"
    assert record.fields["contexts"] == 2
    assert record.fields["grounded"] is True
    assert record.fields["question_chars"] == len(QUESTION)
    assert record.fields["latency_ms"] >= 0
    assert answer.text not in str(record.fields)


def test_the_chain_builds_the_shared_chat_model_when_none_is_injected(
    filings_store, monkeypatch
):
    # The production path — every other test injects a fake, so a chain that built its own
    # client (bypassing `llm.build_chat_model`, the only chat-model constructor) would
    # ship green.
    import finbrief.rag as rag

    built = []
    monkeypatch.setattr(
        rag,
        "build_chat_model",
        lambda: built.append(True) or GenericFakeChatModel(messages=iter(["grounded [1]"])),
    )

    answer = answer_question(QUESTION, k=1, store=filings_store)

    assert built == [True]
    assert answer.text == "grounded [1]"
