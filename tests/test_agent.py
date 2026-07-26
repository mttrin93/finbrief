"""The agent entrypoint and the OpenRouter binding. No network calls."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from finbrief.agent.agent import SKELETON_SYSTEM_PROMPT, answer
from finbrief.agent.llm import build_chat_model
from finbrief.config import Settings

SETTINGS = Settings.from_env(
    {
        "OPENROUTER_API_KEY": "sk-test",
        "FINBRIEF_CHAT_MODEL": "openai/gpt-4o-mini",
    }
)


class RecordingFakeChatModel(GenericFakeChatModel):
    """A fake chat model that keeps the prompt it was handed."""

    prompts: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def test_answer_returns_the_model_reply():
    model = GenericFakeChatModel(messages=iter(["Tesla's Item 1A lists supply-chain risk."]))
    assert answer("What are Tesla's risks?", model=model) == (
        "Tesla's Item 1A lists supply-chain risk."
    )


def test_answer_sends_the_persona_and_the_question():
    model = RecordingFakeChatModel(messages=iter(["ok"]), prompts=[])

    answer("What are Tesla's risks?", model=model)

    (prompt,) = model.prompts
    system, human = prompt
    assert system.text == SKELETON_SYSTEM_PROMPT
    assert human.text == "What are Tesla's risks?"


def test_answer_logs_a_chat_turn_event(caplog):
    with caplog.at_level("INFO", logger="finbrief.agent.agent"):
        answer("hi", model=GenericFakeChatModel(messages=iter(["hello"])))

    (record,) = [r for r in caplog.records if getattr(r, "event", None) == "chat_turn"]
    assert record.fields["question_chars"] == 2
    assert record.fields["latency_ms"] >= 0


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
