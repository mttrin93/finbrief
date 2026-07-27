"""The agent entrypoint and the OpenRouter binding. No network calls.

The seam the UI depends on (spec seam 2). In Phase 2 it is a delegation to the
deterministic chain, so what is asserted here is the delegation's *contract* — which
strategy the shipped path runs — and not the chain's behaviour, which `test_rag.py` owns at
its own seam. Phase 3 replaces the body with the real agent loop and this file grows into
tool-selection assertions.
"""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from finbrief.agent import agent as agent_module
from finbrief.agent.agent import BASELINE_STRATEGY, answer
from finbrief.config import RetrievalStrategy, Settings
from finbrief.llm import build_chat_model
from finbrief.rag import GroundedAnswer

SETTINGS = Settings.from_env(
    {
        "OPENROUTER_API_KEY": "sk-test",
        "FINBRIEF_CHAT_MODEL": "openai/gpt-4o-mini",
    }
)


def test_answer_runs_the_deterministic_chain_and_returns_what_grounded_it(monkeypatch):
    model = GenericFakeChatModel(messages=iter(["unused — the chain is stubbed"]))
    grounded = GroundedAnswer(text="Tesla identifies supply-chain risk [1].", contexts=())
    calls = []

    def fake_answer_question(question, *, strategy, model):
        calls.append({"question": question, "strategy": strategy, "model": model})
        return grounded

    monkeypatch.setattr(agent_module, "answer_question", fake_answer_question)

    assert answer("What are Tesla's risks?", model=model) is grounded
    assert calls == [
        {
            "question": "What are Tesla's risks?",
            "strategy": BASELINE_STRATEGY,
            "model": model,
        }
    ]


def test_the_shipped_path_runs_the_strategy_that_exists_not_the_configured_one():
    # `config.DEFAULT_STRATEGY` is the pre-registered `hybrid` (ADR-0005), which
    # `retrieve()` refuses until Phase 4. If this constant ever tracked the setting again,
    # the app's first question would raise instead of answering.
    assert BASELINE_STRATEGY is RetrievalStrategy.VECTOR


def test_chat_model_is_bound_to_openrouter():
    model = build_chat_model(SETTINGS)
    assert model.model_name == "openai/gpt-4o-mini"
    assert model.openai_api_base == "https://openrouter.ai/api/v1"
    assert model.openai_api_key.get_secret_value() == "sk-test"
    # Deterministic by default — the eval harness depends on it (ADR-0003).
    assert model.temperature == 0.0


def test_chat_model_honours_a_model_override():
    # The Phase-5 injection classifier gets its own (cheaper) model this way.
    model = build_chat_model(SETTINGS, model="anthropic/claude-haiku-4.5")
    assert model.model_name == "anthropic/claude-haiku-4.5"


def test_chat_model_falls_back_to_the_application_settings(monkeypatch):
    # The production path: no settings argument, so `get_settings()` supplies them. Every
    # other test here injects, which would leave a wrong constructor here green in CI.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-environ")
    monkeypatch.setenv("FINBRIEF_CHAT_MODEL", "openai/gpt-4o")

    model = build_chat_model()

    assert model.model_name == "openai/gpt-4o"
    assert model.openai_api_key.get_secret_value() == "sk-from-environ"
