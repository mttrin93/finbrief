"""The Streamlit layer via AppTest: rendering and the round-trip wiring only.

The agent is stubbed entirely — no LLM calls (spec, testing seam 3). Session/thread_id
behaviour arrives in Phase 3 as `test_app_state.py` (ADR-0008); this file is the CI smoke
test that the page renders and a message reaches the agent seam.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from finbrief.agent import agent

APP = str(Path(__file__).parents[1] / "app" / "Home.py")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    return AppTest.from_file(APP, default_timeout=10)


def test_the_page_renders_the_universe_and_a_chat_input(app):
    app.run()

    assert not app.exception
    assert app.title[0].value == "FinBrief"
    sidebar_text = " ".join(md.value for md in app.sidebar.markdown)
    assert "TSLA, F, GM" in sidebar_text  # the autos peer cluster (ADR-0009)
    assert app.chat_input


def test_a_missing_key_reports_a_clear_error_and_stops(app, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")

    app.run()

    assert not app.exception
    assert "OPENROUTER_API_KEY" in app.error[0].value
    assert ".env.example" in app.error[0].value
    assert not app.chat_input, "the app should stop before offering a chat input"


def test_a_bad_log_level_reports_a_clear_error_and_stops(app, monkeypatch):
    # LOG_LEVEL is configuration too: `resolve_log_level` raises ConfigError, so it must
    # reach the same banner as a missing key rather than a raw traceback.
    monkeypatch.setenv("LOG_LEVEL", "chatty")

    app.run()

    assert not app.exception
    assert "LOG_LEVEL" in app.error[0].value
    assert not app.chat_input, "the app should stop before offering a chat input"


def test_sending_a_message_renders_the_reply(app, monkeypatch):
    monkeypatch.setattr(agent, "answer", lambda question: f"Reply to {question}")
    app.run()

    app.chat_input[0].set_value("What are Tesla's risks?").run()

    assert not app.exception
    roles = [message.name for message in app.chat_message]
    assert roles == ["user", "assistant"]
    assert app.chat_message[1].markdown[0].value == "Reply to What are Tesla's risks?"
    assert app.session_state.messages == [
        {"role": "user", "content": "What are Tesla's risks?"},
        {"role": "assistant", "content": "Reply to What are Tesla's risks?"},
    ]


def test_a_second_turn_replays_the_transcript_once(app, monkeypatch):
    # One turn never exercises the history-replay loop, so a double-render or a dropped
    # earlier turn would be invisible.
    monkeypatch.setattr(agent, "answer", lambda question: f"Reply to {question}")
    app.run()

    app.chat_input[0].set_value("First question").run()
    app.chat_input[0].set_value("Second question").run()

    assert not app.exception
    assert [message.name for message in app.chat_message] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    rendered = [message.markdown[0].value for message in app.chat_message]
    assert rendered == [
        "First question",
        "Reply to First question",
        "Second question",
        "Reply to Second question",
    ]


def test_a_failing_model_call_is_reported_not_raised(app, monkeypatch):
    def boom(question):
        raise RuntimeError("upstream refused")

    monkeypatch.setattr(agent, "answer", boom)
    app.run()

    app.chat_input[0].set_value("What are Tesla's risks?").run()

    assert not app.exception
    assert "upstream refused" in app.error[0].value
    # A failed turn must not leave a phantom assistant message in the transcript.
    assert [m["role"] for m in app.session_state.messages] == ["user"]
