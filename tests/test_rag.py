"""The deterministic RAG chain: question -> contexts -> grounded answer (ADR-0003).

The chain the evaluation harness measures, so it is tested at its own seam rather than
through the UI. `retrieve()` is real (against a fixture collection, seam 1's fake
embedding); only the chat model is faked — a real one would be a network call, and the
suite is hermetic by contract (CLAUDE.md).
"""

from __future__ import annotations

from fakes import a_context
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from finbrief.config import RetrievalStrategy, Settings
from finbrief.ingestion.model import Section
from finbrief.prompts import NO_CONTEXT_FALLBACK, SYSTEM_PROMPT
from finbrief.rag import answer_question
from finbrief.retrieval.retrieve import Retrieval


class RecordingFakeChatModel(GenericFakeChatModel):
    """A fake chat model that keeps the prompts it was handed."""

    prompts: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def a_model(reply="Apple flags supply-chain concentration [1].") -> RecordingFakeChatModel:
    return RecordingFakeChatModel(messages=iter([reply]), prompts=[])


QUESTION = "What are the risks to Apple's supply chain?"

#: An injected configuration, for the tests that check it reaches every client the chain
#: builds rather than only the retrieval half.
SETTINGS = Settings.from_env({"OPENROUTER_API_KEY": "sk-test", "FINBRIEF_RETRIEVAL_K": "3"})


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

    def fake_retrieve(question, *, strategy, translate, k, store, settings=None, model=None):
        calls.append(
            {"question": question, "strategy": strategy, "translate": translate, "k": k}
        )
        return Retrieval(
            contexts=(a_context(ticker="AAPL", body="A risk factor."),),
            variants=(question,),
            translated=translate,
        )

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)

    answer_question(
        QUESTION,
        strategy=RetrievalStrategy.VECTOR,
        translate=True,
        k=7,
        store=filings_store,
        model=a_model(),
    )

    # The question reaches the engine unchanged and translation happens *inside* it (ADR-0004).
    # A chain that pre-decomposed would translate twice and part the measured path from the
    # shipped one — so what this asserts is that the chain forwards the switch, never that it
    # acts on it.
    assert calls == [
        {
            "question": QUESTION,
            "strategy": RetrievalStrategy.VECTOR,
            "translate": True,
            "k": 7,
        }
    ]


def test_the_turn_is_logged_with_the_strategy_and_what_grounded_it(filings_store, caplog):
    with caplog.at_level("INFO", logger="finbrief.rag"):
        answer = answer_question(QUESTION, k=2, store=filings_store, model=a_model())

    (record,) = [r for r in caplog.records if getattr(r, "event", None) == "rag_answer"]
    assert record.fields["strategy"] == "vector"
    assert record.fields["contexts"] == 2
    assert record.fields["grounded"] is True
    assert record.fields["question_chars"] == len(QUESTION)
    assert record.fields["answer_chars"] == len(answer.text)
    assert record.fields["latency_ms"] >= 0
    # Provenance, so Phase 7 can reconstruct a turn from the lines alone. `retrieved_`, not
    # `cited_`: nothing parses the answer's `[n]` markers, and a log reader handed
    # `cited_sections` would report citation behaviour that was never measured.
    assert record.fields["tickers"] == ["AAPL"]
    assert record.fields["retrieved_sections"] == sorted(
        {context.section.value for context in answer.contexts}
    )
    assert "cited_sections" not in record.fields
    # The question and the answer are user content and these lines are kept, so only their
    # sizes may appear.
    assert answer.text not in str(record.fields)
    assert QUESTION not in str(record.fields)


def test_an_ungrounded_turn_is_logged_as_ungrounded_with_nothing_to_attribute(
    empty_filings_store, caplog
):
    with caplog.at_level("INFO", logger="finbrief.rag"):
        answer = answer_question(QUESTION, k=2, store=empty_filings_store)

    (record,) = [r for r in caplog.records if getattr(r, "event", None) == "rag_answer"]
    assert record.fields["grounded"] is False
    assert record.fields["contexts"] == 0
    assert record.fields["tickers"] == [] and record.fields["retrieved_sections"] == []
    assert record.fields["answer_chars"] == len(NO_CONTEXT_FALLBACK)
    assert answer.text == NO_CONTEXT_FALLBACK


def test_a_strategy_named_as_a_string_survives_the_whole_chain(filings_store):
    # `retrieve()` documents accepting a raw string and normalises its own local, so this
    # chain used to retrieve, pay for a generation, and *then* die on `strategy.value` in its
    # own log line — the one place the failure costs money (issue #5 review).
    answer = answer_question(
        QUESTION, strategy="vector", k=1, store=filings_store, model=a_model()
    )

    assert answer.grounded


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
        lambda settings=None: (
            built.append(settings) or GenericFakeChatModel(messages=iter(["grounded [1]"]))
        ),
    )

    answer = answer_question(QUESTION, k=1, store=filings_store)

    assert built == [None], "no injected Settings, so the constructor resolves its own"
    assert answer.text == "grounded [1]"


def test_injected_settings_reach_generation_and_not_only_retrieval(filings_store, monkeypatch):
    # A harness pointing an injected `Settings` at a throwaway index would otherwise still
    # generate with the process-global `chat_model`, so half its configuration would silently
    # not apply — and the A/B run would report a model it did not use (issue #5 review).
    import finbrief.rag as rag

    built = []
    monkeypatch.setattr(
        rag,
        "build_chat_model",
        lambda settings=None: (
            built.append(settings) or GenericFakeChatModel(messages=iter(["grounded [1]"]))
        ),
    )

    answer_question(QUESTION, k=1, store=filings_store, settings=SETTINGS)

    assert built == [SETTINGS]
