"""`search_filings` — the one tool that wraps `retrieve()` (ADR-0003, seam 5).

What matters here is the *contract between the agent and the retrieval engine*, not
retrieval quality: `test_retrieve.py` owns the latter at seam 1. So this file asserts that
the tool wraps rather than replaces the engine (the query reaches `retrieve()` unchanged,
under the strategy the shipped path names), that its description tells the agent what the
verbatim contract is, and that what it hands back can be both read by the model and
rendered by the UI.

Numbering *across* calls is asserted in `test_agent.py` and `test_citations.py`: the register
is assigned at the agent seam, not here, because a tool cannot see the other calls in its own
step. What this file pins is that the tool emits `retrieve()`'s own 1…k and reads no
conversation state at all — the property that makes it safe to renumber later.
"""

from __future__ import annotations

import json

import pytest

from finbrief.config import RetrievalStrategy, Settings
from finbrief.ingestion.model import Section
from finbrief.prompts import EMPTY_SEARCH_RESULT, SEARCH_FILINGS_DESCRIPTION
from finbrief.retrieval.retrieve import Context
from finbrief.tools.search_filings import TOOL_NAME, build_search_filings

SETTINGS = Settings.from_env({"OPENROUTER_API_KEY": "sk-test"})


def a_search_tool(store, **kwargs):
    kwargs.setdefault("strategy", RetrievalStrategy.VECTOR)
    return build_search_filings(store=store, settings=SETTINGS, **kwargs)


def search(tool, query: str):
    """Invoke the tool the way the agent's tool node does, and return the ToolMessage.

    Invoked as a tool *call* rather than plain arguments because what the callers below assert
    lives on the message — the artifact the panel renders and the content the model reads.
    """
    return tool.invoke(
        {"name": TOOL_NAME, "args": {"query": query}, "id": "call-1", "type": "tool_call"}
    )


def retrieved(message) -> tuple[Context, ...]:
    """The chunks the tool returned, as the UI rebuilds them from the artifact."""
    return tuple(Context.from_payload(payload) for payload in message.artifact)


def test_the_tool_retrieves_for_the_query_it_is_given(filings_store):
    tool = a_search_tool(filings_store)

    message = search(tool, "supply chain risk")

    contexts = retrieved(message)
    assert contexts, "the retrieved chunks travel with the message, for the UI"
    assert all(context.ticker == "AAPL" for context in contexts)
    assert [context.rank for context in contexts] == [1, 2, 3, 4, 5]
    # The model reads the content: the same numbered, quarantined framing the chain uses.
    assert "<sources>" in message.content
    assert f"[1] AAPL 10-K FY{contexts[0].fiscal_year}" in message.content


def test_the_artifact_is_json_safe_because_the_checkpointer_serialises_it(filings_store):
    # Everything on a message is written into the checkpoint (ADR-0008). A `Context` survives
    # that round trip only through an escape hatch LangGraph warns on and will remove, and
    # comes back as an untyped dict under its strict serialiser — so what crosses this
    # boundary is the payload form, and `json.dumps` is the cheapest proof of that.
    tool = a_search_tool(filings_store)

    message = search(tool, "supply chain risk")

    assert json.loads(json.dumps(message.artifact)) == list(message.artifact)
    # And a round trip through that form is lossless, since the panel renders what comes out.
    rebuilt = retrieved(message)
    assert tuple(context.as_payload() for context in rebuilt) == message.artifact


def test_the_query_reaches_the_engine_unchanged_under_the_shipped_strategy(
    monkeypatch, filings_store
):
    # ADR-0003: the tool *wraps* `retrieve()`, it does not reimplement or pre-process it.
    # Pre-translating here would translate twice (translation lives inside the engine) and
    # part the shipped path from the measured one.
    from finbrief.tools import search_filings as module

    calls = []

    def fake_retrieve(question, **kwargs):
        calls.append({"question": question, **kwargs})
        return ()

    monkeypatch.setattr(module, "retrieve", fake_retrieve)
    tool = a_search_tool(filings_store, strategy=RetrievalStrategy.VECTOR)

    search(tool, "  Is Tesla in trouble?  ")

    assert calls == [
        {
            "question": "  Is Tesla in trouble?  ",
            "strategy": RetrievalStrategy.VECTOR,
            "k": None,
            "store": filings_store,
            "settings": SETTINGS,
        }
    ]


def test_the_description_instructs_the_agent_to_pass_the_question_verbatim(filings_store):
    # The verbatim contract is enforced by *prompt*, so the prompt is what there is to
    # assert (ADR-0003 §1). `test_agent.py` asserts the other half — that a divergence is
    # detected and logged rather than assumed away.
    tool = a_search_tool(filings_store)

    assert tool.name == TOOL_NAME
    assert tool.description == SEARCH_FILINGS_DESCRIPTION
    lowered = SEARCH_FILINGS_DESCRIPTION.lower()
    assert "verbatim" in lowered
    assert "do not rephrase" in lowered
    # And the one exception, without which every follow-up retrieves noise: this tool has
    # no conversation, so an unresolved "its debt" names no company.
    assert "pronoun" in lowered


def test_the_tool_never_asks_the_model_for_anything_but_the_query(filings_store):
    # `strategy`, `k` and the store are the wrapper's business. A model that could set them
    # could ask for a strategy that was never measured.
    tool = a_search_tool(filings_store)

    assert list(tool.args) == ["query"]


def test_the_tool_reads_no_conversation_state_so_its_numbering_cannot_collide(filings_store):
    # The property the citation register is built on. The tool used to take a `ToolRuntime` and
    # offset its ranks by the sources already issued in the thread — which cannot work, because
    # LangGraph hands every call in a step the *same* state and then runs them concurrently, so
    # two searches in one step both offset by the same number. It now emits `retrieve()`'s own
    # 1…k, identically every time, and `agent/citations.py` assigns the thread's numbers where
    # all of a step's replies are visible at once.
    tool = a_search_tool(filings_store)

    first = search(tool, "supply chain risk")
    second = search(tool, "supply chain risk")

    assert [c.rank for c in retrieved(first)] == [1, 2, 3, 4, 5]
    assert first.artifact == second.artifact, "no hidden state to number against"
    assert "runtime" not in tool.args


def test_an_empty_collection_is_reported_as_a_setup_problem_not_as_out_of_scope(
    empty_filings_store,
):
    # A populated Chroma always returns top-k, so nothing retrieved means the collection is
    # empty or misdirected. Told "no results", the model would report the question as out of
    # scope — and a reviewer who simply has not ingested yet goes looking for a retrieval bug.
    tool = a_search_tool(empty_filings_store)

    message = search(tool, "supply chain risk")

    assert message.artifact == ()
    # The wording is `prompts.py`'s to own — `test_prompts.py` asserts it names the cause.
    # What belongs here is that the empty case reaches the model as that text at all,
    # rather than as an empty `<sources>` block it would answer around.
    assert message.content == EMPTY_SEARCH_RESULT


@pytest.mark.parametrize("section", [Section.BUSINESS, Section.RISK_FACTORS])
def test_every_retrieved_chunk_keeps_its_provenance_for_the_sources_panel(
    filings_store, section
):
    # The artifact is what `app/Home.py` renders, so a citation has to resolve to a chunk
    # carrying the metadata user story 3 asks for.
    tool = a_search_tool(filings_store)

    message = search(tool, f"{section.heading} of the company")

    for context in retrieved(message):
        assert context.chunk_id and context.body
        assert context.citation.startswith("AAPL 10-K FY")
