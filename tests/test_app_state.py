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
from finbrief.config import CHAT_MODEL_CHOICES, Settings, chat_model_options

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


def pick_model(app, slug):
    """The sidebar's model picker, found by label rather than by index.

    `start_over`'s reasoning applies to every widget in this sidebar: a positional claim about
    `app.sidebar.selectbox` would silently redirect to a different widget the moment a second
    one is added inside any panel. There is one selectbox today, and that is exactly why the
    lookup should not depend on it.
    """
    (picker,) = [s for s in app.sidebar.selectbox if "Model" in s.label]
    return picker.select(slug)


@pytest.fixture(autouse=True)
def stubbed_agent(monkeypatch, agent_builds):
    """Record every `(question, thread_id, agent, model)` the app asks about.

    Layered on `agent_builds` (conftest), which is what keeps an `AppTest` from opening a real
    checkpoint file.
    """
    asked: list[dict[str, object]] = []

    def fake_answer(question, *, thread_id, agent, model=None, on_step=None):  # noqa: ARG001
        asked.append(
            {"question": question, "thread_id": thread_id, "agent": agent, "model": model}
        )
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


# --------------------------------------------------------------------------------------
# The model picker against ADR-0008's guarantee (T14, #15)
# --------------------------------------------------------------------------------------


def test_two_sessions_on_different_models_still_hold_distinct_threads(
    app, monkeypatch, stubbed_agent
):
    """T14's hard requirement: the picker adds a dimension and takes nothing away.

    **This test is not the mutation-detector, and saying so is the point.** Isolation is carried
    by the per-session `uuid4` `thread_id` and never by the cache key, so dropping `model` from
    the cached builder leaves every assertion here passing — two sessions would silently share
    one agent on the default model and *still* hold two thread ids, still send each turn to its
    own. That is the check-that-cannot-fail shape this repo keeps hitting (CLAUDE.md), and the
    honest response is to name what this covers and put the detection in the test below.

    What it covers is real and worth a test: the property must not regress, and a future change
    that keyed conversations on the model — or reset a thread when the picker moved — would
    break it here.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    other = AppTest.from_file(APP, default_timeout=10)
    app.run()
    other.run()

    pick_model(app, CHAT_MODEL_CHOICES[0]).run()
    pick_model(other, CHAT_MODEL_CHOICES[1]).run()

    assert app.session_state.thread_id != other.session_state.thread_id

    app.chat_input[0].set_value(QUESTION).run()
    other.chat_input[0].set_value("What are Ford's risk factors?").run()

    threads = {call["question"]: call["thread_id"] for call in stubbed_agent["asked"]}
    assert threads[QUESTION] == app.session_state.thread_id
    assert threads["What are Ford's risk factors?"] == other.session_state.thread_id
    assert len(set(threads.values())) == 2, "and neither session saw the other's turn"


def test_the_picked_model_reaches_the_builder_and_gets_its_own_cached_agent(
    app, monkeypatch, stubbed_agent
):
    """**The mutation-detector for the picker**, and the reason the test above is not.

    **Verified by making the edits and watching the suite, not by reasoning about it** (#15).
    Two mutations were applied to `app/Home.py` in turn:

    1. keep `shared_agent(model)` but call `build_chat_model(settings)` — the slug is accepted
       and thrown away, the subtler bug, since the cache still holds two entries;
    2. drop the parameter entirely, so `@st.cache_resource` has one key for the whole process.

    This test failed on both, and
    `test_two_sessions_on_different_models_still_hold_distinct_threads` **passed** on both —
    which is the measurement behind that test's docstring rather than a claim about it.

    An **equality** on the built models, not `len(built) >= 1` or a containment check: a bound
    is satisfied by the broken implementation as well as the working one, which is the other
    half of the same lesson.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    other = AppTest.from_file(APP, default_timeout=10)
    app.run()
    other.run()

    pick_model(app, CHAT_MODEL_CHOICES[0]).run()
    pick_model(other, CHAT_MODEL_CHOICES[1]).run()
    app.chat_input[0].set_value(QUESTION).run()
    other.chat_input[0].set_value("What are Ford's risk factors?").run()

    assert [build.model for build in stubbed_agent["built"]] == [
        CHAT_MODEL_CHOICES[0],
        CHAT_MODEL_CHOICES[1],
    ], "one agent per model, each built with the model that was picked"
    agents = {id(call["agent"]) for call in stubbed_agent["asked"]}
    assert len(agents) == 2, "and the two sessions ran on two different instances"


def test_the_model_that_answered_is_reported_to_the_seam_that_logs_it(app, stubbed_agent):
    # The picker is only visible in the record if the slug reaches `answer()`, which is what
    # writes it onto `agent_turn`. Without this the log attributes every turn to the configured
    # model and the analytics split is four labels over one true value.
    app.run()
    pick_model(app, CHAT_MODEL_CHOICES[1]).run()

    app.chat_input[0].set_value(QUESTION).run()

    assert stubbed_agent["asked"][-1]["model"] == CHAT_MODEL_CHOICES[1]


def test_the_default_selection_is_the_configured_model(app, stubbed_agent):
    # `FINBRIEF_CHAT_MODEL` stays the default (#15): a picker whose initial selection is
    # something else silently overrides the one setting it is supposed to respect, and the
    # override would be invisible — the turn answers, just not on the configured model.
    app.run()
    app.chat_input[0].set_value(QUESTION).run()

    settings = Settings.from_env({"OPENROUTER_API_KEY": "sk-test"})
    assert stubbed_agent["asked"][-1]["model"] == settings.chat_model
    assert [build.model for build in stubbed_agent["built"]] == [settings.chat_model]


def test_switching_models_keeps_the_conversation_and_does_not_rebuild_the_other_agent(
    app, stubbed_agent
):
    """Constraint 2 at the app layer: the checkpointer is keyed on `thread_id`, not the model.

    So a switch mid-conversation keeps the thread and the transcript — the *agent* changes and
    the conversation does not. `test_agent.py` owns the other half, that the second model is
    really shown the first turn out of the shared checkpoint file; this owns that the app does
    not throw the conversation away on the way there.

    Also the third widget to be checked against ADR-0008's one hard guarantee, after *Start
    over* and the example buttons: a picker that cleared the checkpointer, or rebuilt an agent
    another session is mid-turn on, would discard **every other session's** memory.
    """
    app.run()
    app.chat_input[0].set_value(QUESTION).run()
    thread = app.session_state.thread_id
    transcript = list(app.session_state.messages)

    pick_model(app, CHAT_MODEL_CHOICES[1]).run()

    assert app.session_state.thread_id == thread, "the same conversation, on a new model"
    assert app.session_state.messages == transcript, "and the transcript is not thrown away"

    app.chat_input[0].set_value(FOLLOW_UP).run()
    assert stubbed_agent["asked"][-1]["thread_id"] == thread
    assert stubbed_agent["asked"][-1]["model"] == CHAT_MODEL_CHOICES[1]
    # Two agents now exist — one per model — and the first was not discarded to make the second.
    assert [build.model for build in stubbed_agent["built"]] == [
        CHAT_MODEL_CHOICES[0],
        CHAT_MODEL_CHOICES[1],
    ]


def test_the_picker_offers_the_fixed_set_and_no_text_entry(app):
    # Not free text (#15): a typo reaches OpenRouter as a provider error in the middle of a
    # turn. Asserted as an equality against `config`, so a hardcoded copy of the list in the
    # page — the drift this repo keeps finding — fails here rather than on screen.
    app.run()

    (picker,) = [s for s in app.sidebar.selectbox if "Model" in s.label]
    assert tuple(picker.options) == chat_model_options(
        Settings.from_env({"OPENROUTER_API_KEY": "sk-test"}).chat_model
    )
    assert app.text_input == [], "no free-text model entry anywhere on the page"


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

    # The *fact*, not the wording: this test is ADR-0008's binding that the consequence is
    # stated at all, and the exact sentence has one owner —
    # `test_app_smoke.test_the_refresh_semantics_stay_above_the_fold`, which also asserts that
    # no panel is hiding it. A second copy of the literal here would be one more pair to drift.
    sidebar = " ".join(caption.value for caption in app.sidebar.caption)
    assert "refreshing" in sidebar.lower()
    assert "new conversation" in sidebar.lower()


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
    # `status` joins `turn` as **display** data, and the equality is kept as an equality on
    # purpose: it is what caught this key being added, which is exactly what it is for. Both
    # fields exist so a row can be *re-rendered* — `turn` carries the panels' four facts and
    # `status` the label its progress box ended on, without which the replay renders one fewer
    # element than the live turn did and leaves the difference on screen
    # (`test_an_assistant_row_replays_the_shape_it_rendered_live`). Neither is ever read back
    # into a conversation: the checkpointer stays the agent's memory of record (ADR-0008), which
    # is what the last assertion in this test holds.
    assert set(app.session_state.messages[1]) == {"role", "content", "turn", "status"}
    assert {"messages", "thread_id"} <= set(app.session_state.filtered_state)
    # Whatever else the UI keeps, none of it may be the conversation the agent is given: the
    # agent is handed one question, a thread id and the slug to attribute the turn to — and
    # nothing else. `model` joined the set in T14 (#15) and this equality is what caught it,
    # which is what the equality is for. It belongs: a model slug is a UI toggle ADR-0008
    # already places in `session_state`, and it is passed for the *log line*, not as context —
    # the conversation still comes from the checkpointer and from nowhere else.
    assert set(stubbed_agent["asked"][-1]) == {"question", "thread_id", "agent", "model"}
