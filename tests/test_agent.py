"""The agent entrypoint: memory, tool wiring, and what a turn reports (spec seam 2).

The **real agent loop** — `create_agent`, a real `SqliteSaver` on a temporary file, the real
`search_filings` over a fixture collection with a fake embedding (seam 1's). Only the chat
model is scripted, because a real one is a network call and the suite is hermetic by contract
(CLAUDE.md). That split is deliberate: what this file is for is the wiring the model's
behaviour rides on — that a follow-up is handed the conversation, that two threads cannot see
each other, that citation numbers keep going up, that a divergence from the measured query is
recorded rather than assumed away. Which tool a real model *chooses* is the tool-calling
eval's question (ADR-0003), not something a script could answer.

The checkpointer here is the real file-backed one rather than an in-memory stand-in, so the
memory assertions cross serialisation exactly as they do in the app: a `Context` that did not
survive the round trip would restart citation numbering at `[1]` in production only.
"""

from __future__ import annotations

from fakes import ScriptedChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from finbrief.agent.agent import (
    MAX_AGENT_STEPS,
    AgentTurn,
    answer,
    build_agent,
    build_checkpointer,
)
from finbrief.config import RetrievalStrategy, Settings
from finbrief.llm import build_chat_model
from finbrief.tools.search_filings import TOOL_NAME

SETTINGS = Settings.from_env(
    {
        "OPENROUTER_API_KEY": "sk-test",
        "FINBRIEF_CHAT_MODEL": "openai/gpt-4o-mini",
        "FINBRIEF_RETRIEVAL_K": "3",
    }
)

TESLA_QUESTION = "What are Tesla's risk factors?"
FOLLOW_UP = "And its debt?"


def a_search(query: str, call_id: str = "call-1") -> AIMessage:
    """The model's turn when it decides to search."""
    return AIMessage(
        content="",
        tool_calls=[{"name": TOOL_NAME, "args": {"query": query}, "id": call_id}],
    )


def a_fan_out(*queries: str) -> AIMessage:
    """The model's turn when it asks for several searches in *one* step.

    A real provider is told not to (`parallel_tool_calls=False`, plus the tool description),
    but "told not to" is the class of guarantee this ticket stopped relying on: the register
    has to survive a step that arrives with two searches in it.
    """
    return AIMessage(
        content="",
        tool_calls=[
            {"name": TOOL_NAME, "args": {"query": query}, "id": f"call-{index}"}
            for index, query in enumerate(queries, start=1)
        ],
    )


def an_agent(tmp_path, store, script, *, name="checkpoints.sqlite"):
    """The real agent, with a scripted model and a real checkpoint file."""
    model = ScriptedChatModel(messages=iter(script))
    agent = build_agent(
        model=model,
        settings=SETTINGS,
        store=store,
        checkpointer=build_checkpointer(path=str(tmp_path / name)),
    )
    return agent, model


def test_a_grounded_answer_comes_back_with_the_chunks_it_cites(tmp_path, filings_store):
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [a_search("supply chain risk"), AIMessage("Apple flags concentration [1].")],
    )

    turn = answer("What are Apple's supply-chain risks?", thread_id="t-1", agent=agent)

    assert isinstance(turn, AgentTurn)
    assert turn.text == "Apple flags concentration [1]."
    assert turn.searched and turn.grounded
    assert [context.rank for context in turn.contexts] == [1, 2, 3]
    assert all(context.ticker == "AAPL" for context in turn.contexts)
    assert [search.query for search in turn.searches] == ["supply chain risk"]


def test_a_follow_up_is_handed_the_conversation_from_the_checkpointer(tmp_path, filings_store):
    # ADR-0008's whole claim, at the seam that has to honour it: `answer()` passes only the
    # new question, so if the follow-up's prompt does not contain the first turn, the memory
    # of record is not the checkpointer and a real model has nothing to resolve "its" against.
    agent, model = an_agent(
        tmp_path,
        filings_store,
        [
            a_search("Tesla risk factors", "call-1"),
            AIMessage("Tesla identifies supply-chain concentration [1]."),
            a_search("Tesla debt", "call-2"),
            AIMessage("Its debt is described in the MD&A [4]."),
        ],
    )

    answer(TESLA_QUESTION, thread_id="t-1", agent=agent)
    answer(FOLLOW_UP, thread_id="t-1", agent=agent)

    seen = model.prompts[-1]
    replayed = [m.text for m in seen if isinstance(m, HumanMessage)]
    assert replayed == [TESLA_QUESTION, FOLLOW_UP], "the earlier question is still in context"
    assert any(isinstance(m, ToolMessage) and m.name == TOOL_NAME for m in seen), (
        "and so is what the first search returned — the answer's citations rest on it"
    )


def test_a_follow_up_reports_only_its_own_turns_sources(tmp_path, filings_store):
    # The transcript already shows the first answer beside its own panel. A turn that
    # re-reported the whole conversation's chunks would render every earlier source again
    # under the newest answer, where no `[n]` points at them.
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [
            a_search("Tesla risk factors", "call-1"),
            AIMessage("Tesla identifies supply-chain concentration [1]."),
            a_search("Tesla debt", "call-2"),
            AIMessage("Its debt is described in the MD&A [4]."),
        ],
    )

    first = answer(TESLA_QUESTION, thread_id="t-1", agent=agent)
    second = answer(FOLLOW_UP, thread_id="t-1", agent=agent)

    assert len(first.searches) == len(second.searches) == 1
    assert len(second.contexts) == 3


def test_citation_numbers_keep_climbing_across_a_conversation(tmp_path, filings_store):
    # Two searches in one thread, both ranked 1…k by `retrieve()`. Reusing `[1]` would make
    # every marker in the answer above ambiguous — the first thing a reader checks is which
    # chunk `[1]` was, and by then there are two.
    agent, model = an_agent(
        tmp_path,
        filings_store,
        [
            a_search("Tesla risk factors", "call-1"),
            AIMessage("Tesla identifies supply-chain concentration [1]."),
            a_search("Tesla debt", "call-2"),
            AIMessage("Its debt is described in the MD&A [4]."),
        ],
    )

    first = answer(TESLA_QUESTION, thread_id="t-1", agent=agent)
    second = answer(FOLLOW_UP, thread_id="t-1", agent=agent)

    assert [context.rank for context in first.contexts] == [1, 2, 3]
    assert [context.rank for context in second.contexts] == [4, 5, 6]
    # The number the *model* was shown is the number the panel labels: one chunk, one `[n]`,
    # on both surfaces. Without this the model could cite `[1]` for the fourth chunk.
    latest_sources = [m for m in model.prompts[-1] if isinstance(m, ToolMessage)][-1]
    assert "[4] " in latest_sources.content
    first_of_second = second.contexts[0]
    assert f"[{first_of_second.rank}] {first_of_second.citation}" in latest_sources.content


def test_two_searches_in_one_step_do_not_reuse_citation_numbers(tmp_path, filings_store):
    # The collision the register is assigned at the agent seam to prevent (ADR-0003 §3, as
    # amended). LangGraph's tool node builds every `ToolRuntime` from the *same* node input
    # and then runs the calls concurrently, so two searches in one step would each read an
    # identical "sources issued so far" and both emit `[1][2][3]` — after which `[1]` names
    # two different chunks and the reader cannot check either.
    agent, model = an_agent(
        tmp_path,
        filings_store,
        [
            a_fan_out("Tesla supply chain risk", "Ford supply chain risk"),
            AIMessage("Both flag concentration [1][4]."),
        ],
    )

    turn = answer("Compare Tesla and Ford on supply chain.", thread_id="t-1", agent=agent)

    ranks = [context.rank for context in turn.contexts]
    assert len(ranks) == 6, "two searches, k=3 each"
    assert ranks == [1, 2, 3, 4, 5, 6], "one number per source, and none of them twice"
    # And the numbers the *model* was shown are those numbers: renumbering only the artifact
    # would fix the panel and leave the model citing an ambiguous `[1]`.
    blocks = [m.content for m in model.prompts[-1] if isinstance(m, ToolMessage)]
    assert len(blocks) == 2
    assert "[1] " in blocks[0] and "[4] " in blocks[1]
    assert "[1] " not in blocks[1]


def test_a_fan_out_then_a_follow_up_keeps_the_register_climbing(tmp_path, filings_store):
    # The other half: a step's numbering has to leave the *thread's* register where the next
    # turn expects it. A per-step counter would restart the follow-up at `[1]`.
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [
            a_fan_out("Tesla supply chain risk", "Ford supply chain risk"),
            AIMessage("Both flag concentration [1][4]."),
            a_search("Tesla debt", "call-3"),
            AIMessage("Its debt is described in the MD&A [7]."),
        ],
    )

    answer("Compare Tesla and Ford on supply chain.", thread_id="t-1", agent=agent)
    second = answer(FOLLOW_UP, thread_id="t-1", agent=agent)

    assert [context.rank for context in second.contexts] == [7, 8, 9]


def test_the_model_is_bound_against_parallel_tool_calls(tmp_path, filings_store):
    # The primary, provider-side half of the same fix: one tool call per step, so the
    # collision above does not arise with a real model. Asserted at the binding because that
    # is the whole of our side of it — whether a given OpenRouter upstream honours the flag is
    # not ours to assert, which is why the register above does not depend on it.
    agent, model = an_agent(tmp_path, filings_store, [AIMessage("…")])

    answer("What does Apple say about supply chains?", thread_id="t-1", agent=agent)

    assert model.bind_kwargs, "the agent binds its tools through `bind_tools`"
    assert all(kwargs.get("parallel_tool_calls") is False for kwargs in model.bind_kwargs)


def test_numbering_restarts_for_a_different_conversation(tmp_path, filings_store):
    # The register is per thread, because `[6]` may only mean one chunk *within* a
    # conversation. A process-wide counter would number the second user's first citation `[6]`
    # with nothing above it.
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [
            a_search("Tesla risk factors", "call-1"),
            AIMessage("Tesla identifies supply-chain concentration [1]."),
            a_search("Ford risk factors", "call-2"),
            AIMessage("Ford identifies supply-chain concentration [1]."),
        ],
    )

    answer(TESLA_QUESTION, thread_id="t-1", agent=agent)
    other = answer("What are Ford's risk factors?", thread_id="t-2", agent=agent)

    assert [context.rank for context in other.contexts] == [1, 2, 3]


def test_two_threads_never_see_each_others_conversation(tmp_path, filings_store):
    # The privacy property ADR-0008 exists for, at the agent's own seam: one cached agent,
    # one checkpoint file, two users. `test_app_state.py` asserts the other half — that the
    # UI mints one thread id per session and keeps it.
    agent, model = an_agent(
        tmp_path,
        filings_store,
        [
            a_search("Tesla risk factors", "call-1"),
            AIMessage("Tesla identifies supply-chain concentration [1]."),
            a_search("Ford risk factors", "call-2"),
            AIMessage("Ford identifies supply-chain concentration [1]."),
        ],
    )

    answer(TESLA_QUESTION, thread_id="alice", agent=agent)
    answer("What are Ford's risk factors?", thread_id="bob", agent=agent)

    questions = [m.text for m in model.prompts[-1] if isinstance(m, HumanMessage)]
    assert questions == ["What are Ford's risk factors?"]
    assert TESLA_QUESTION not in " ".join(m.text for m in model.prompts[-1])


def test_a_turn_that_needed_no_search_says_so_rather_than_looking_ungrounded(
    tmp_path, filings_store
):
    # "Summarise that" is answered from the conversation. It has no contexts and it is not a
    # setup problem — the distinction the UI needs to avoid telling a reviewer to re-run a
    # paid ingest because a follow-up was answered from what was already on screen.
    agent, _ = an_agent(tmp_path, filings_store, [AIMessage("Two risks, briefly: […]")])

    turn = answer("Summarise that in two lines.", thread_id="t-1", agent=agent)

    assert turn.text.startswith("Two risks")
    assert not turn.searched
    assert turn.contexts == ()


def test_a_verbatim_search_is_logged_as_one(tmp_path, filings_store, caplog):
    # ADR-0003 §2 — the shipped path's query is recorded against the original so the README
    # can report the divergence rate rather than assert there isn't one.
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [a_search(TESLA_QUESTION), AIMessage("Tesla identifies […] [1].")],
    )

    with caplog.at_level("INFO", logger="finbrief.agent.agent"):
        answer(TESLA_QUESTION, thread_id="t-1", agent=agent)

    (query,) = [r for r in caplog.records if getattr(r, "event", None) == "agent_query"]
    assert query.fields["verbatim"] is True
    assert query.fields["thread_id"] == "t-1"
    assert query.fields["hits"] == 3
    # Lengths, never the text: a question is user content and these lines are kept.
    assert query.fields["question_chars"] == len(TESLA_QUESTION)
    assert "Tesla" not in str(query.fields)


def test_a_rephrased_search_is_logged_as_a_divergence(tmp_path, filings_store, caplog):
    # The failure this instrumentation exists to make visible: the agent optimised the query
    # itself, so the shipped path embedded something the measured chain never saw.
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [a_search("tesla supply chain risk keywords"), AIMessage("Tesla identifies […] [1].")],
    )

    with caplog.at_level("INFO", logger="finbrief.agent.agent"):
        answer(TESLA_QUESTION, thread_id="t-1", agent=agent)

    (query,) = [r for r in caplog.records if getattr(r, "event", None) == "agent_query"]
    (turn,) = [r for r in caplog.records if getattr(r, "event", None) == "agent_turn"]
    assert query.fields["verbatim"] is False
    assert turn.fields["searches"] == 1
    assert turn.fields["verbatim_searches"] == 0


def test_a_turn_is_logged_with_what_it_searched_and_grounded(tmp_path, filings_store, caplog):
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [a_search(TESLA_QUESTION), AIMessage("Tesla identifies […] [1].")],
    )

    with caplog.at_level("INFO", logger="finbrief.agent.agent"):
        answer(TESLA_QUESTION, thread_id="t-1", agent=agent)

    (turn,) = [r for r in caplog.records if getattr(r, "event", None) == "agent_turn"]
    assert turn.fields["searched"] is True
    assert turn.fields["grounded"] is True
    assert turn.fields["contexts"] == 3
    assert turn.fields["latency_ms"] >= 0


def test_the_shipped_path_runs_the_strategy_that_exists_not_the_configured_one():
    # `config.DEFAULT_STRATEGY` is the pre-registered `hybrid` (ADR-0005), which
    # `retrieve()` refuses until Phase 4. If this constant ever tracked the setting again,
    # the app's first question would raise instead of answering.
    from finbrief.agent.agent import BASELINE_STRATEGY

    assert BASELINE_STRATEGY is RetrievalStrategy.VECTOR


def test_the_agent_binds_the_knowledge_base_as_a_tool_not_as_a_chain(tmp_path, filings_store):
    # ADR-0003: `search_filings` wraps `retrieve()`; the deterministic chain the eval harness
    # measures stays reachable with no agent in the way. If the agent ever grew its own
    # retrieval, the two paths would stop being comparable.
    agent, _ = an_agent(tmp_path, filings_store, [AIMessage("…")])

    assert MAX_AGENT_STEPS > 2, "a search and an answer are two steps; the cap must clear it"
    assert TOOL_NAME in agent.nodes["tools"].bound.tools_by_name


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
