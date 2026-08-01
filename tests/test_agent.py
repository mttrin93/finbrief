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

import io
import json
import logging
import threading

from fakes import ScriptedChatModel, a_headline_source, a_quote_source
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from finbrief.agent.agent import (
    MAX_AGENT_STEPS,
    AgentTurn,
    Step,
    answer,
    build_agent,
    build_checkpointer,
)
from finbrief.config import TICKER_MAX_CHARS, Settings
from finbrief.llm import build_chat_model
from finbrief.observability.logging_setup import configure_logging, turn
from finbrief.tools import search_filings as search_filings_module
from finbrief.tools.finance import (
    FINANCE_TOOL_NAMES,
    NEWS_TOOL_NAME,
    RATIOS_TOOL_NAME,
    STOCK_TOOL_NAME,
    FailedCard,
    QuoteCard,
)
from finbrief.tools.search_filings import TOOL_NAME

#: The configuration these tests build the agent with. **`vector`, translation off**, and
#: explicitly so: what this file is for is the agent's wiring — memory, the register, the
#: divergence log — and the shipped default would put a paid chat call (the query planner) and a
#: BM25 index in front of every one of those assertions without sharpening any of them. Which
#: configuration the agent *reads* is its own test, below.
SETTINGS = Settings.from_env(
    {
        "OPENROUTER_API_KEY": "sk-test",
        "FINBRIEF_CHAT_MODEL": "openai/gpt-4o-mini",
        "FINBRIEF_RETRIEVAL_K": "3",
        "FINBRIEF_RETRIEVAL_STRATEGY": "vector",
        "FINBRIEF_QUERY_TRANSLATION": "false",
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


def a_quote_call(ticker: str, call_id: str = "call-1") -> AIMessage:
    """The model's turn when it decides to look up a price."""
    return AIMessage(
        content="",
        tool_calls=[{"name": STOCK_TOOL_NAME, "args": {"ticker": ticker}, "id": call_id}],
    )


def a_fan_out_of(*calls: dict) -> AIMessage:
    """The model's turn when it asks for several *different* tools in one step.

    The shape `parallel_tool_calls=True` now asks a provider for, and the one demo step 4
    depends on: a search and the finance tools in one round trip rather than four.
    """
    return AIMessage(content="", tool_calls=list(calls))


def an_agent(
    tmp_path, store, script, *, name="checkpoints.sqlite", settings=None, planner=None
):
    """The real agent, with a scripted model, a real checkpoint file and recorded market data.

    The finance sources are injected from the recorded fixtures rather than reached for, which
    is spec seam 2's split exactly: real `create_agent`, real tools, real store, external data
    mocked. `conftest`'s egress guard would refuse a live fetch anyway; this is what makes the
    refusal unnecessary.
    """
    model = ScriptedChatModel(messages=iter(script))
    agent = build_agent(
        model=model,
        settings=settings or SETTINGS,
        store=store,
        checkpointer=build_checkpointer(path=str(tmp_path / name)),
        translation_model=planner,
        quote=a_quote_source,
        headlines=a_headline_source,
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


def test_the_model_is_bound_to_allow_parallel_tool_calls(tmp_path, filings_store):
    # T4 asserted the opposite here, and the inversion is the point (T5, #9). The flag was the
    # *second* line of a two-line defence against two searches in one step reusing `[1]`; the
    # first line — the register at the `before_model` seam — made that collision
    # unrepresentable rather than unlikely, so the second line stopped paying for itself. What
    # it costs is a model round trip per tool, and demo step 4 wants four tools.
    #
    # Asserted at the binding because that is the whole of our side of it: whether a given
    # OpenRouter upstream honours the flag is not ours to assert, and the test above is what
    # says the register does not care either way.
    agent, model = an_agent(tmp_path, filings_store, [AIMessage("…")])

    answer("What does Apple say about supply chains?", thread_id="t-1", agent=agent)

    assert model.bind_kwargs, "the agent binds its tools through `bind_tools`"
    assert all(kwargs.get("parallel_tool_calls") is True for kwargs in model.bind_kwargs)


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


# --------------------------------------------------------------------------------------
# Which model answered (T14, #15)
# --------------------------------------------------------------------------------------


def test_a_turn_records_the_model_it_was_asked_to_answer_on(tmp_path, filings_store, caplog):
    # The picker's whole record. Without it the spend panel and the analytics page attribute
    # every turn's tokens to whatever `chat_model` happens to be configured, which is wrong for
    # three of the four models a reader can pick — and wrong invisibly, since the tokens
    # themselves are real.
    #
    # A model id is configuration, not user content, so it sits inside `logging_setup`'s rules
    # with no exception: it names a product, not a person or a question.
    agent, _ = an_agent(tmp_path, filings_store, [AIMessage("Answered without searching.")])

    with caplog.at_level("INFO", logger="finbrief.agent.agent"):
        answer(
            TESLA_QUESTION,
            thread_id="t-1",
            agent=agent,
            model="anthropic/claude-3.5-haiku",
        )

    (turn,) = [r for r in caplog.records if getattr(r, "event", None) == "agent_turn"]
    assert turn.fields["model"] == "anthropic/claude-3.5-haiku"


def test_a_caller_that_names_no_model_logs_an_absence_and_not_a_default(
    tmp_path, filings_store, caplog
):
    """`None`, never `settings.chat_model` — the rule `tokens.py` states and this repo keeps
    breaking.

    Filling in the configured model here would be a *fabricated measurement*: `answer()` is
    handed a built agent and cannot see which model is inside it, so the configured value would
    be a guess that reads exactly like a reading. The evaluation harness drives
    `rag.answer_question` rather than this path, but any future caller that omits the argument
    should show up on the analytics page as unattributed rather than as a fifth vote for the
    default.
    """
    agent, _ = an_agent(tmp_path, filings_store, [AIMessage("Answered without searching.")])

    with caplog.at_level("INFO", logger="finbrief.agent.agent"):
        answer(TESLA_QUESTION, thread_id="t-1", agent=agent)

    (turn,) = [r for r in caplog.records if getattr(r, "event", None) == "agent_turn"]
    assert turn.fields["model"] is None


def test_the_provider_s_own_word_for_what_answered_is_recorded_beside_the_request(
    tmp_path, filings_store, caplog
):
    """Two fields, because they can disagree and the disagreement is the interesting part.

    OpenRouter fronts many upstreams and routes by availability, so the model that answered is
    not guaranteed to be the model asked for. `model` is what the picker requested and is the
    key everything aggregates on; `model_reported` is what the reply said, so a routing surprise
    is visible instead of silent. Conflating them into one field would mean either losing the
    request or reporting a guess as a reading.
    """
    reply = AIMessage(
        "Answered without searching.",
        response_metadata={"model_name": "openai/gpt-4o-mini-2024-07-18"},
    )
    agent, _ = an_agent(tmp_path, filings_store, [reply])

    with caplog.at_level("INFO", logger="finbrief.agent.agent"):
        answer(TESLA_QUESTION, thread_id="t-1", agent=agent, model="openai/gpt-4o-mini")

    (turn,) = [r for r in caplog.records if getattr(r, "event", None) == "agent_turn"]
    assert turn.fields["model"] == "openai/gpt-4o-mini"
    assert turn.fields["model_reported"] == "openai/gpt-4o-mini-2024-07-18"


def test_a_provider_that_names_no_model_leaves_the_reported_field_absent(
    tmp_path, filings_store, caplog
):
    # The common case for every hermetic test in this file, and the honest reading of it: a
    # reply with no `response_metadata` reported nothing, which is not the same as having
    # answered on the requested model.
    agent, _ = an_agent(tmp_path, filings_store, [AIMessage("Answered without searching.")])

    with caplog.at_level("INFO", logger="finbrief.agent.agent"):
        answer(TESLA_QUESTION, thread_id="t-1", agent=agent, model="openai/gpt-4o-mini")

    (turn,) = [r for r in caplog.records if getattr(r, "event", None) == "agent_turn"]
    assert turn.fields["model"] == "openai/gpt-4o-mini"
    assert turn.fields["model_reported"] is None


def test_a_conversation_survives_a_model_switch_mid_thread(tmp_path, filings_store):
    """T14's second hard requirement (#15), asserted rather than assumed.

    The checkpointer is keyed on `thread_id` and not on the model, so switching models mid
    conversation must keep the history. The arrangement is what a per-model
    `@st.cache_resource` entry really produces: **two** agents, each with its own
    `SqliteSaver` connection, both pointed at one file, one thread id between them.

    What is asserted is that the second model was *shown* the first turn — `model.prompts`,
    not just that the returned state accumulated. A follow-up whose prompt lacks the earlier
    turn has no conversation to resolve "and its debt?" against, however well it answers.
    """
    checkpoint = str(tmp_path / "shared.sqlite")
    first = ScriptedChatModel(messages=iter([AIMessage("Tesla flags supply-chain risk.")]))
    second = ScriptedChatModel(messages=iter([AIMessage("Its debt is discussed in Item 7A.")]))
    agents = [
        build_agent(
            model=model,
            settings=SETTINGS,
            store=filings_store,
            checkpointer=build_checkpointer(path=checkpoint),
            quote=a_quote_source,
            headlines=a_headline_source,
        )
        for model in (first, second)
    ]

    answer(TESLA_QUESTION, thread_id="one-conversation", agent=agents[0], model="model-a")
    follow_up = answer(
        "And its debt?", thread_id="one-conversation", agent=agents[1], model="model-b"
    )

    assert follow_up.text == "Its debt is discussed in Item 7A."
    shown = [message.text for message in second.prompts[-1]]
    assert TESLA_QUESTION in shown, "the second model was shown the first question"
    assert "Tesla flags supply-chain risk." in shown, "and the first model's answer"


def test_two_models_on_two_threads_still_cannot_see_each_other(tmp_path, filings_store):
    # The other side of the switch, and the property the picker must not weaken: sharing a
    # checkpoint file across models is only safe because `thread_id` is what separates
    # conversations. Two models on two threads is the two-session case from `test_app_state.py`
    # at the seam where it is actually spent.
    checkpoint = str(tmp_path / "shared.sqlite")
    models = {
        "a": ScriptedChatModel(messages=iter([AIMessage("Answer for the first reader.")])),
        "b": ScriptedChatModel(messages=iter([AIMessage("Answer for the second reader.")])),
    }
    agents = {
        key: build_agent(
            model=model,
            settings=SETTINGS,
            store=filings_store,
            checkpointer=build_checkpointer(path=checkpoint),
            quote=a_quote_source,
            headlines=a_headline_source,
        )
        for key, model in models.items()
    }

    answer("What are Tesla's risks?", thread_id="thread-a", agent=agents["a"], model="a")
    answer("What are Ford's risks?", thread_id="thread-b", agent=agents["b"], model="b")

    shown = [message.text for message in models["b"].prompts[-1]]
    assert "What are Tesla's risks?" not in shown
    assert "Answer for the first reader." not in shown


def test_the_shipped_path_runs_the_configuration_it_is_configured_with(tmp_path, filings_store):
    # Until Phase 4 this asserted the opposite — a `BASELINE_STRATEGY` constant pinned to
    # `vector`, because `config.DEFAULT_STRATEGY` was the pre-registered `hybrid` (ADR-0005) and
    # `retrieve()` refused it. Now that hybrid exists, the shipped path must honour the setting:
    # a constant left in place here would make the sidebar advertise a configuration no answer
    # used, and ADR-0005's pre-registered default would never actually ship.
    #
    # Read off the retrieval this turn actually ran, not off the wiring: the agent hands
    # `search_filings` a configuration and the tool hands it to `retrieve()`, and the only
    # evidence that survives all of that is the provenance on the chunks that came back.
    shipped = Settings.from_env(
        {
            "OPENROUTER_API_KEY": "sk-test",
            "FINBRIEF_RETRIEVAL_K": "3",
            "FINBRIEF_RETRIEVAL_STRATEGY": "hybrid",
            "FINBRIEF_QUERY_TRANSLATION": "true",
            "FINBRIEF_MAX_SUB_QUERIES": "1",
        }
    )
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [a_search("Apple supply chain risk"), AIMessage("Apple flags concentration [1].")],
        settings=shipped,
        planner=ScriptedChatModel(messages=iter([AIMessage("Apple supplier concentration")])),
    )

    turn = answer("What are Apple's supply-chain risks?", thread_id="t-1", agent=agent)

    (search,) = turn.searches
    assert search.translated is True
    assert search.variants == (
        "Apple supply chain risk",
        "AAPL supply chain risk",  # config's deterministic ticker form (ADR-0004 amendment)
        "Apple supplier concentration",  # the planner's one sub-query, at the configured cap
    )
    assert search.ticker_form == "AAPL supply chain risk"
    assert search.sub_queries == ("Apple supplier concentration",)
    retrievers = {row.retriever.value for c in turn.contexts for row in c.provenance}
    assert retrievers == {"vector", "bm25"}, "hybrid ran both retrievers over every variant"


def test_the_configured_strategy_is_read_once_and_not_restated_by_the_tool():
    # `build_search_filings` requires both switches rather than defaulting them, so there is
    # exactly one place the shipped configuration is named. A default there would be a second
    # place for it to be wrong — one that keeps working while disagreeing with the sidebar.
    import inspect

    from finbrief.tools.search_filings import build_search_filings

    parameters = inspect.signature(build_search_filings).parameters
    assert parameters["strategy"].default is inspect.Parameter.empty
    assert parameters["translate"].default is inspect.Parameter.empty


def test_a_search_reports_the_variants_it_ran_for_the_rag_viz_panel(tmp_path, filings_store):
    # User story 5 through the agent seam: the panel needs the variants, and a sub-query that
    # surfaced nothing is invisible in `contexts` — so `Search` has to carry them itself.
    # Translation off here, which is the case that must still be honest rather than empty.
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [a_search("Apple supply chain risk"), AIMessage("Apple flags concentration [1].")],
    )

    turn = answer("What are Apple's supply-chain risks?", thread_id="t-1", agent=agent)

    (search,) = turn.searches
    assert search.variants == ("Apple supply chain risk",)
    assert search.sub_queries == ()
    assert search.translated is False


def test_the_agent_binds_the_knowledge_base_as_a_tool_not_as_a_chain(tmp_path, filings_store):
    # ADR-0003: `search_filings` wraps `retrieve()`; the deterministic chain the eval harness
    # measures stays reachable with no agent in the way. If the agent ever grew its own
    # retrieval, the two paths would stop being comparable.
    agent, _ = an_agent(tmp_path, filings_store, [AIMessage("…")])

    assert MAX_AGENT_STEPS > 2, "a search and an answer are two steps; the cap must clear it"
    assert TOOL_NAME in agent.nodes["tools"].bound.tools_by_name


# --- T5 (#9): the finance tools on the loop --------------------------------------------


def test_the_agent_binds_all_four_tools(tmp_path, filings_store):
    # The set T10's tool-calling eval selects from. With one tool the *selection* the loop
    # exists for was trivial; with four it is the thing being measured.
    agent, _ = an_agent(tmp_path, filings_store, [AIMessage("…")])

    assert set(agent.nodes["tools"].bound.tools_by_name) == {
        TOOL_NAME,
        *FINANCE_TOOL_NAMES,
    }


#: Super-steps in one round of the agent loop: the register's `before_model` node, the model,
#: the tool node. Named because three separate assertions below are about it.
SUPER_STEPS_PER_ROUND = 3

#: Tool calls a full brief needs made serially — business, risks, valuation, news (demo step 4).
FULL_BRIEF_TOOL_CALLS = 4


def test_the_step_ceiling_is_the_value_the_arithmetic_argues_for():
    # **Holds the value, not an inequality.** This test used to assert
    # `serial_full_brief < MAX_AGENT_STEPS`, which 20, 24 and 15 all satisfy — so it pinned
    # nothing, and a regression that halved the ceiling would have stayed green until the demo
    # (issue #9 review). The number and the reasoning that produced it are asserted separately:
    # the equality catches a silent change, the derivation below says whether a *deliberate*
    # change is still enough.
    assert MAX_AGENT_STEPS == 24


def test_the_step_ceiling_clears_a_full_brief_made_one_tool_at_a_time():
    # The arithmetic `MAX_AGENT_STEPS` is set from, asserted rather than left in a comment: one
    # round of the loop is three super-steps, so a brief's four tool calls made serially cost
    # 3 × 4 + 2 = 14. At T4's twelve the one demo query this ticket exists to deliver would have
    # raised `GraphRecursionError`, and only for models that decline to fan out.
    serial_full_brief = SUPER_STEPS_PER_ROUND * FULL_BRIEF_TOOL_CALLS + 2

    assert serial_full_brief == 14, "the arithmetic in `agent.py`'s comment, executed"
    assert serial_full_brief <= MAX_AGENT_STEPS


def test_the_step_ceiling_keeps_the_headroom_its_comment_claims():
    # `agent.py` says twenty-four "leaves room for seven serial rounds: the four a brief needs,
    # plus a retry after a refused ticker and headroom for a model that thinks in smaller
    # pieces". That is a claim about a quotient, so it is checked as one — lowering the ceiling
    # to 14 would still clear the test above while deleting every one of those spare rounds.
    serial_rounds = (MAX_AGENT_STEPS - 2) // SUPER_STEPS_PER_ROUND

    assert serial_rounds == 7
    assert serial_rounds - FULL_BRIEF_TOOL_CALLS == 3, "three rounds spare beyond a full brief"


def test_a_finance_tool_result_comes_back_on_the_turn_as_a_card(tmp_path, filings_store):
    # What the UI renders its charts from. On the turn rather than re-fetched, for the reason
    # the contexts are: a second fetch through a TTL cache is a second chance to disagree with
    # the answer, and it could return a different number.
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [a_quote_call("NVDA"), AIMessage("NVIDIA trades at 196.51.")],
    )

    turn = answer("What is NVIDIA trading at?", thread_id="t-1", agent=agent)

    (card,) = turn.cards
    assert isinstance(card, QuoteCard)
    assert card.quote.ticker == "NVDA"
    assert card.quote.price == 196.51
    assert turn.used_tools and not turn.searched, "a quote is not a retrieval"


def test_a_turn_reports_a_search_and_a_quote_apart(tmp_path, filings_store):
    # User story 11's shape: one question, two halves, one answer. They stay in separate fields
    # because only one of them is numbered — a headline or a price must never get an `[n]`.
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [
            a_fan_out_of(
                {"name": TOOL_NAME, "args": {"query": "Apple supply chain risk"}, "id": "c1"},
                {"name": STOCK_TOOL_NAME, "args": {"ticker": "AAPL"}, "id": "c2"},
            ),
            AIMessage("Apple flags concentration [1]; it trades at 336.91."),
        ],
    )

    turn = answer("Apple's supply risk, and its price?", thread_id="t-1", agent=agent)

    assert len(turn.searches) == 1
    assert len(turn.cards) == 1
    assert [context.rank for context in turn.contexts] == [1, 2, 3]


def test_the_finance_cards_of_an_earlier_turn_are_not_re_reported(tmp_path, filings_store):
    # The same rule the sources panel follows: the earlier card is already on screen above its
    # own answer, and re-reporting it would render every chart the conversation ever had under
    # the newest reply.
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [
            a_quote_call("NVDA", "c1"),
            AIMessage("196.51."),
            a_quote_call("AMZN", "c2"),
            AIMessage("231.39."),
        ],
    )

    answer("NVIDIA's price?", thread_id="t-1", agent=agent)
    second = answer("And Amazon's?", thread_id="t-1", agent=agent)

    assert [card.quote.ticker for card in second.cards] == ["AMZN"]


def test_a_turn_logs_which_finance_tools_it_used(tmp_path, filings_store, caplog):
    # The tool-*selection* datum T4 could not report, because with one tool there was nothing to
    # select. T10 (#11) reads it off the turn line rather than reassembling per-tool events.
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [
            a_fan_out_of(
                {"name": STOCK_TOOL_NAME, "args": {"ticker": "F"}, "id": "c1"},
                {"name": RATIOS_TOOL_NAME, "args": {"ticker": "F"}, "id": "c2"},
            ),
            AIMessage("Ford trades at 14.68."),
        ],
    )

    with caplog.at_level("INFO", logger="finbrief.agent.agent"):
        answer("Ford's valuation?", thread_id="t-1", agent=agent)

    (turn,) = [r for r in caplog.records if getattr(r, "event", None) == "agent_turn"]
    assert turn.fields["finance_calls"] == 2
    # Tool *names*, not card class names: `type(card).__name__` collapsed all three tools into
    # `FailedCard` whenever a call failed, losing tool choice on exactly the turns worth
    # reading.
    assert turn.fields["tools_used"] == [RATIOS_TOOL_NAME, STOCK_TOOL_NAME]
    assert turn.fields["searched"] is False


def test_each_tool_call_is_reported_before_it_runs_and_exactly_once(tmp_path, filings_store):
    # User story 14. Reported as the model asks for the call, which is what makes it a progress
    # indicator rather than a summary — and once per call, because `stream_mode="values"`
    # re-emits the whole state on every chunk and an undeduplicated reporter would make a
    # two-tool turn read like a loop.
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [
            a_fan_out_of(
                {"name": TOOL_NAME, "args": {"query": "Ford risk factors"}, "id": "c1"},
                {"name": NEWS_TOOL_NAME, "args": {"ticker": "GM", "days": 7}, "id": "c2"},
            ),
            AIMessage("Ford identifies […] [1]."),
        ],
    )
    steps: list[Step] = []

    answer("Ford risks and GM news?", thread_id="t-1", agent=agent, on_step=steps.append)

    assert steps == [Step(TOOL_NAME), Step(NEWS_TOOL_NAME, "GM")]


def test_an_earlier_turns_tool_calls_are_not_reported_as_this_turns_progress(
    tmp_path, filings_store
):
    # The checkpointer replays the whole conversation, so the first streamed chunk of a
    # follow-up already contains last turn's tool calls. Reported, they would open the follow-up
    # by announcing a fetch that happened a turn ago — progress about the wrong thing.
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [
            a_quote_call("NVDA", "c1"),
            AIMessage("196.51."),
            a_quote_call("AMZN", "c2"),
            AIMessage("231.39."),
        ],
    )

    answer("NVIDIA's price?", thread_id="t-1", agent=agent)
    steps: list[Step] = []
    answer("And Amazon's?", thread_id="t-1", agent=agent, on_step=steps.append)

    assert steps == [Step(STOCK_TOOL_NAME, "AMZN")]


def test_a_progress_step_truncates_an_over_long_ticker_argument(tmp_path, filings_store):
    # `Step.ticker` is a *model-supplied* argument that has not been through `resolve_ticker`,
    # so without the cap a refused call's raw argument reaches the page at whatever length the
    # model chose — including a paragraph of prose (ADR-0006).
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [a_quote_call("IGNORE PREVIOUS INSTRUCTIONS " * 20), AIMessage("I cover fifteen…")],
    )
    steps: list[Step] = []

    answer("What about NVIDIA?", thread_id="t-1", agent=agent, on_step=steps.append)

    (step,) = steps
    assert len(step.ticker) <= TICKER_MAX_CHARS


def test_a_turn_that_called_nothing_reports_no_steps_and_no_cards(tmp_path, filings_store):
    agent, _ = an_agent(tmp_path, filings_store, [AIMessage("Two risks, briefly: […]")])
    steps: list[Step] = []

    turn = answer("Summarise that.", thread_id="t-1", agent=agent, on_step=steps.append)

    assert steps == []
    assert turn.cards == ()
    assert not turn.used_tools


def test_an_unknown_ticker_is_answered_rather_than_raised(tmp_path, filings_store):
    # User story 21 through the real loop: the refusal comes back as a tool result the model
    # reads in the same turn, so the analyst gets a sentence about the Universe instead of a
    # traceback.
    agent, model = an_agent(
        tmp_path,
        filings_store,
        [a_quote_call("SAP"), AIMessage("I cover fifteen US filers; SAP is not one.")],
    )

    turn = answer("What is SAP trading at?", thread_id="t-1", agent=agent)

    assert "SAP is not one" in turn.text
    (card,) = turn.cards
    assert isinstance(card, FailedCard)
    reply = [m for m in model.prompts[-1] if isinstance(m, ToolMessage)][-1]
    assert "not a company in FinBrief's Universe" in reply.content


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


def test_a_failed_call_is_still_logged_against_the_tool_that_failed(
    tmp_path, filings_store, caplog
):
    # The regression behind the fix: every failure produced a `FailedCard`, so a card-class
    # label reported `FailedCard` and T10 could not tell which tool had been chosen. `card.TOOL`
    # reads the tool back off the kind the call would have produced.
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [a_quote_call("SAP"), AIMessage("I cover fifteen US filers.")],
    )

    with caplog.at_level("INFO", logger="finbrief.agent.agent"):
        answer("What is SAP trading at?", thread_id="t-1", agent=agent)

    (turn,) = [r for r in caplog.records if getattr(r, "event", None) == "agent_turn"]
    assert turn.fields["tools_used"] == [STOCK_TOOL_NAME]


def test_an_unreadable_days_argument_is_refused_gracefully_by_the_tool_node(
    tmp_path, filings_store
):
    # User story 21 for the one argument the tools do *not* validate themselves. `days: int` is
    # a pydantic schema, so an un-coercible value never reaches the tool body — and LangChain's
    # tool node turns the validation failure into a `ToolMessage` with `status="error"` telling
    # the model to fix it, which the model can act on in the same turn. Asserted through the
    # real loop because that is the only place the behaviour exists: a direct `tool.invoke()`
    # raises.
    #
    # Pinned rather than assumed: a coercion helper was written for this and deleted once the
    # loop showed nothing reached it (issue #9 review), so this test is what stands in its
    # place.
    agent, model = an_agent(
        tmp_path,
        filings_store,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": NEWS_TOOL_NAME,
                        "args": {"ticker": "TSLA", "days": "seven"},
                        "id": "c1",
                    }
                ],
            ),
            AIMessage("How many days back would you like?"),
        ],
    )

    turn = answer("Any recent Tesla news?", thread_id="t-1", agent=agent)

    reply = [m for m in model.prompts[-1] if isinstance(m, ToolMessage)][-1]
    assert reply.status == "error"
    assert "valid integer" in reply.content
    assert turn.text.startswith("How many days"), "the model recovered inside the same turn"
    assert turn.cards == (), "and no card was fabricated for a call that never ran"


# --- The turn identifier, through the real tool executor (T8, #10) ---------------------


def test_the_turn_id_reaches_a_retrieval_line_inside_the_tool_node(tmp_path, filings_store):
    """`logging_setup.turn` has to survive LangGraph's tool node, and only this can say so.

    The whole value of the identifier is that a `retrieval` line — which carries per-chunk
    provenance and, deliberately, no question — can be attributed to the question that caused
    it. That line is emitted deep inside `retrieve()`, called from the tool, called from the
    tool node, which LangGraph is free to run on another thread. A `ContextVar` propagates
    into a thread only if whatever spawns it copies the context, and "LangChain copies the
    context" is exactly the class of claim CLAUDE.md says to exercise rather than assert.

    So this drives the **real** loop and reads the **real** log line.
    """
    stream = io.StringIO()
    logger = configure_logging(logging.DEBUG, stream=stream)
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [a_search("Tesla supply chain risk"), AIMessage("Tesla flags concentration [1].")],
    )

    with turn("golden-semantic-04:hybrid+translation"):
        answer(TESLA_QUESTION, thread_id="t-1", agent=agent)
    logger.handlers.clear()

    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    retrievals = [line for line in lines if line["event"] == "retrieval"]
    assert retrievals, "the tool ran and `retrieve()` logged"
    assert all(
        line.get("turn_id") == "golden-semantic-04:hybrid+translation" for line in retrievals
    ), "a retrieval line with no turn id is provenance nothing can attribute"


def test_both_searches_of_one_step_carry_the_same_turn_id(tmp_path, filings_store, monkeypatch):
    """Two concurrent searches, one turn — and the thread boundary is asserted, not assumed.

    This is the case order-based correlation gets wrong and nothing catches: LangGraph hands
    every call in a step the same state and then runs them concurrently, so two `retrieval`
    lines arrive from two executions of the tool and both belong to one question.

    The spy is not decoration. A `ContextVar` is trivially visible to a same-thread callee, so
    if LangGraph ever ran a step's tools inline this test would keep passing while proving
    nothing about propagation — the vacuous-assertion shape CLAUDE.md names. Measured here:
    both calls land on threads that are **not** this one, and on **two different** ones.

    The distinctness is asserted rather than merely observed, and it used not to be: the check
    was `len(threads) == 2`, which counts *calls*. Two searches serialised onto one worker
    thread would satisfy it — `[X, X]` is two entries — so the premise this docstring states
    was one the assertion did not hold down (issue #10 review). Prefer an equality over a
    bound, and assert the premise a test rests on.
    """
    threads: list[int] = []
    inner = search_filings_module.retrieve

    def spy(*args, **kwargs):
        threads.append(threading.get_ident())
        return inner(*args, **kwargs)

    monkeypatch.setattr(search_filings_module, "retrieve", spy)
    stream = io.StringIO()
    logger = configure_logging(logging.DEBUG, stream=stream)
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [
            a_fan_out("Tesla supply chain risk", "Ford supply chain risk"),
            AIMessage("Both flag concentration [1][4]."),
        ],
    )

    with turn("t-1:turn-1"):
        answer("Compare Tesla and Ford on supply chain.", thread_id="t-1", agent=agent)
    logger.handlers.clear()

    assert len(threads) == 2, f"two searches in one step, two tool calls — got {threads}"
    assert len(set(threads)) == 2, (
        f"the premise: two *distinct* threads, so the fan-out really was concurrent — {threads}"
    )
    assert threading.get_ident() not in threads, (
        "the premise: the tool node ran both searches off this thread, so the context "
        "really did have to cross a thread boundary"
    )
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    retrievals = [line for line in lines if line["event"] == "retrieval"]
    assert len(retrievals) == 2, "two searches in one step, two retrievals"
    assert {line.get("turn_id") for line in retrievals} == {"t-1:turn-1"}


def test_a_turn_id_is_absent_rather_than_null_outside_a_turn(tmp_path, filings_store):
    # Every ingest line and every script line is emitted outside a turn. `null` on all of them
    # would be a key that says nothing a missing key does not, and `read_events` reads either
    # as `None` — so the cheaper shape wins.
    stream = io.StringIO()
    logger = configure_logging(logging.DEBUG, stream=stream)
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [a_search("Tesla supply chain risk"), AIMessage("Tesla flags concentration [1].")],
    )

    answer(TESLA_QUESTION, thread_id="t-1", agent=agent)
    logger.handlers.clear()

    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert lines, "the turn emitted something"
    assert all("turn_id" not in line for line in lines)


def test_the_turn_line_totals_the_loops_own_calls(tmp_path, filings_store):
    """A tool-using turn is several paid completions, and no single reply is its cost.

    Scoped to *this* turn, which is the half a sum could get wrong invisibly: the checkpointer
    replays every earlier turn's `AIMessage`, so an unscoped total bills the newest question for
    the whole conversation and grows with it.
    """
    stream = io.StringIO()
    configure_logging(logging.DEBUG, stream=stream)
    metered = AIMessage(
        "Tesla flags concentration [1].",
        usage_metadata={"input_tokens": 900, "output_tokens": 40, "total_tokens": 940},
    )
    unmetered = AIMessage("Its debt is in the MD&A [4].")
    agent, _ = an_agent(
        tmp_path,
        filings_store,
        [
            a_search("Tesla risk factors", "call-1"),
            metered,
            a_search("Tesla debt", "call-2"),
            unmetered,
        ],
    )

    answer(TESLA_QUESTION, thread_id="t-1", agent=agent)
    answer(FOLLOW_UP, thread_id="t-1", agent=agent)
    logging.getLogger("finbrief").handlers.clear()

    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    first, second = [line["fields"] for line in lines if line["event"] == "agent_turn"]
    # The scripted `a_search` messages report no usage either, so turn 1 metered exactly one
    # of its two calls — which is the denominator `<field>_calls` exists to state, per field
    # rather than per reply (issue #10 review).
    assert (
        first["calls"],
        first["input_tokens"],
        first["input_tokens_calls"],
        first["output_tokens"],
        first["output_tokens_calls"],
    ) == (2, 900, 1, 40, 1)
    # Turn 2's own calls reported nothing, so its spend is absent — *not* turn 1's total
    # carried forward, which is what an unscoped sum would have reported here.
    assert "input_tokens" not in second
