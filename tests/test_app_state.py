"""Session and thread state through `AppTest` — ADR-0008's enforcement (spec seam 3).

ADR-0008 exists to prevent one specific failure: on a deployed multi-user app, a cached agent
shared by every session hands one user another user's conversation. That claim cannot be
tested by reading the code and should not be tested by deploying — so it is tested here, with
two `AppTest` sessions in one process, which is exactly the arrangement the ADR describes
(one cached agent, two browsers).

**The agent is stubbed entirely.** No LLM call, no retrieval, no checkpoint file: what is
under test is which `thread_id` the app hands the agent, and when it changes. `test_agent.py`
owns the other half — that two thread ids really do get two separate conversations out of the
checkpointer.

`test_app_smoke.py` covers rendering. This file covers state, and only state.
"""

from pathlib import Path

import pytest
from fakes import a_context
from streamlit.testing.v1 import AppTest

from finbrief.agent import agent as agent_module
from finbrief.agent.agent import AgentTurn, Search

APP = str(Path(__file__).parents[1] / "app" / "Home.py")

QUESTION = "What are Tesla's risk factors?"
FOLLOW_UP = "And its debt?"


def start_over(app):
    """The *Start over* button, found by label rather than by index.

    It was `app.sidebar.button[0]`, which is a positional claim about the whole sidebar: T12
    added panels of their own, and `AppTest`'s block accessors recurse into an expander, so a
    button added inside any panel above this one silently redirects every assertion below to a
    different widget. The label is what the test is actually about.
    """
    (button,) = [b for b in app.sidebar.button if "Start over" in b.label]
    return button


@pytest.fixture(autouse=True)
def stubbed_agent(monkeypatch, agent_builds):
    """Record every `(question, thread_id, agent)` the app asks about.

    Layered on `agent_builds` (conftest), which is what keeps an `AppTest` from opening a real
    checkpoint file.
    """
    asked: list[dict[str, object]] = []

    def fake_answer(question, *, thread_id, agent, on_step=None):  # noqa: ARG001 — T5's callback
        asked.append({"question": question, "thread_id": thread_id, "agent": agent})
        return AgentTurn(
            text="Tesla identifies supply-chain concentration [1].",
            searches=(Search(query=question, contexts=(a_context(1),)),),
        )

    monkeypatch.setattr(agent_module, "answer", fake_answer)
    return {"asked": asked, "built": agent_builds}


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    return AppTest.from_file(APP, default_timeout=10)


def test_a_session_gets_one_thread_id_before_it_is_asked_anything(app):
    app.run()

    assert app.session_state.thread_id, "minted at startup, not on the first message"
    assert len(app.session_state.thread_id) == 36, "a uuid4, not a counter or a name"


def test_the_thread_id_survives_reruns_and_every_turn_lands_on_it(app, stubbed_agent):
    # The regression that costs memory: a `thread_id` re-minted per rerun means every turn
    # opens a new conversation, and the checkpointer — which is working perfectly — appears
    # to remember nothing.
    app.run()
    minted = app.session_state.thread_id

    app.chat_input[0].set_value(QUESTION).run()
    app.chat_input[0].set_value(FOLLOW_UP).run()
    app.run()  # a rerun with no interaction at all, as a widget change would cause

    assert app.session_state.thread_id == minted
    assert [call["thread_id"] for call in stubbed_agent["asked"]] == [minted, minted]
    assert [call["question"] for call in stubbed_agent["asked"]] == [QUESTION, FOLLOW_UP]


def test_two_sessions_get_different_threads_from_the_same_cached_agent(app, monkeypatch):
    # The privacy failure ADR-0008 is written to prevent, provable without deploying: two
    # browsers on one server share the cached agent and the checkpoint file, so if they also
    # shared a `thread_id` each would be shown the other's conversation.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    other = AppTest.from_file(APP, default_timeout=10)

    app.run()
    other.run()

    assert app.session_state.thread_id != other.session_state.thread_id


def test_each_sessions_turns_go_to_its_own_thread(app, monkeypatch, stubbed_agent):
    # The same property at the seam it is spent through: it is not enough that the two
    # sessions *hold* different ids — each turn has to be asked under the id of the session it
    # came from.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    other = AppTest.from_file(APP, default_timeout=10)
    app.run()
    other.run()

    app.chat_input[0].set_value(QUESTION).run()
    other.chat_input[0].set_value("What are Ford's risk factors?").run()

    threads = {call["question"]: call["thread_id"] for call in stubbed_agent["asked"]}
    assert threads[QUESTION] == app.session_state.thread_id
    assert threads["What are Ford's risk factors?"] == other.session_state.thread_id
    assert len(set(threads.values())) == 2


def test_the_agent_is_built_once_and_shared_by_both_sessions(app, monkeypatch, stubbed_agent):
    # `@st.cache_resource`, and the reason for it: an agent built per session is a
    # checkpointer per session, and one built per *rerun* remembers nothing at all.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    other = AppTest.from_file(APP, default_timeout=10)
    app.run()
    other.run()

    app.chat_input[0].set_value(QUESTION).run()
    other.chat_input[0].set_value("What are Ford's risk factors?").run()
    app.chat_input[0].set_value(FOLLOW_UP).run()

    assert len(stubbed_agent["built"]) == 1, "one construction for the whole process"
    agents = {id(call["agent"]) for call in stubbed_agent["asked"]}
    assert len(agents) == 1, "and every turn, in either session, ran on it"


def test_the_agent_is_not_built_before_it_is_needed(app, stubbed_agent):
    # Rendering the page must not open a checkpoint file: `AppTest` runs the script for the
    # rendering tests too, and a build at import time would make every one of them write
    # SQLite — and would turn a bad checkpoint path into a blank page instead of a failed turn.
    app.run()

    assert stubbed_agent["built"] == []


def test_starting_over_mints_a_fresh_thread_and_keeps_the_agent(app, stubbed_agent):
    # A new conversation, not a new process: clearing the *checkpointer* here would discard
    # every other session's memory as well, which is the bug this button most looks like.
    app.run()
    app.chat_input[0].set_value(QUESTION).run()
    first = app.session_state.thread_id

    start_over(app).click().run()

    assert app.session_state.thread_id != first
    assert app.session_state.messages == [], "and the transcript goes with the conversation"
    assert len(stubbed_agent["built"]) == 1, "the cached agent survived"

    app.chat_input[0].set_value(QUESTION).run()
    assert stubbed_agent["asked"][-1]["thread_id"] == app.session_state.thread_id


def test_starting_over_does_not_reach_another_session(app, monkeypatch, stubbed_agent):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    other = AppTest.from_file(APP, default_timeout=10)
    app.run()
    other.run()
    others_thread = other.session_state.thread_id

    start_over(app).click().run()

    assert other.session_state.thread_id == others_thread


def test_the_page_states_that_a_refresh_starts_a_new_conversation(app):
    # ADR-0008 accepts this rather than fixing it (`session_state` resets, so a new uuid is
    # minted and the old thread orphans). An accepted consequence has to be *stated*: a user
    # who is not told reads a lost conversation as a bug, and a reviewer cannot tell the
    # difference between a design decision and an oversight.
    app.run()

    sidebar = " ".join(caption.value for caption in app.sidebar.caption)
    assert "refreshing" in sidebar.lower()
    assert "new conversation" in sidebar.lower()
    assert "follow-ups" in sidebar.lower(), "and what memory buys, since it is the reason"


def test_seeding_a_question_touches_session_state_and_not_the_cached_agent(app, stubbed_agent):
    """T12 item 3, held against ADR-0008's one hard guarantee.

    An example-question button is UI convenience, and the tempting implementation of "start
    this conversation for me" is to reset the agent — which on this design would discard
    **every other session's** memory, since the instance is `@st.cache_resource`d and shared.
    That is the same failure `test_starting_over_does_not_reach_another_session` guards,
    arriving through a second button.

    So the click may add to `session_state` and may not rebuild anything: one construction
    across the whole process, and the seeded turn lands on the session's existing thread.
    """
    from finbrief.prompts import EXAMPLE_QUESTIONS

    app.run()
    minted = app.session_state.thread_id

    (button,) = [b for b in app.button if b.label == EXAMPLE_QUESTIONS[0]]
    button.click().run()

    assert len(stubbed_agent["built"]) == 1, "the click asked a question; it did not rebuild"
    assert app.session_state.thread_id == minted, "and it stayed in the same conversation"
    assert stubbed_agent["asked"][-1]["thread_id"] == minted


def test_the_transcript_is_the_only_conversation_state_the_ui_keeps(app, stubbed_agent):
    # ADR-0008's negative half: `session_state` holds the thread id, UI state and the display
    # transcript — and *not* the agent's memory. A history accumulating here would be a second
    # copy of the conversation, and the copy the model never sees is the one that goes stale.
    app.run()
    app.chat_input[0].set_value(QUESTION).run()

    assert set(app.session_state.messages[0]) == {"role", "content"}
    assert set(app.session_state.messages[1]) == {"role", "content", "turn"}
    assert {"messages", "thread_id"} <= set(app.session_state.filtered_state)
    # Whatever else the UI keeps, none of it may be the conversation the agent is given: the
    # agent is handed one question and a thread id, and nothing else.
    assert set(stubbed_agent["asked"][-1]) == {"question", "thread_id", "agent"}
