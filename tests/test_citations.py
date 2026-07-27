"""The citation register, at the seam where it is assigned (ADR-0003 §3, ADR-0008 §1).

`test_agent.py` drives this through the real agent loop, which is the assertion that matters —
that a step arriving with two searches in it cannot produce two `[1]`s. What is here is the
pass itself, where the cases a scripted loop cannot conveniently reach are cheap to write: a
reply whose artifact did not survive, a thread already correctly numbered, a second call over
the same messages.

Message *shape* is what these build, deliberately, rather than running a tool: the register's
whole contract is with the transcript the checkpointer persists, and a payload dict read back
out of SQLite is a plain dict with a `rank` in it and nothing else.
"""

from __future__ import annotations

from fakes import a_context
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from finbrief.agent.citations import renumbered
from finbrief.retrieval.retrieve import Context, Retrieval
from finbrief.tools.search_filings import TOOL_NAME

#: What one search ran, for the replies below — two variants, so every assertion here also
#: exercises the half of the artifact the register must carry through without touching.
VARIANTS = ("What are Tesla's risk factors?", "Tesla supply chain risk")


def a_reply(*ranks: int, call_id: str = "call-1", name: str = TOOL_NAME) -> ToolMessage:
    """A `search_filings` reply carrying chunks numbered `ranks`, as the tool emits them."""
    retrieval = Retrieval(
        contexts=tuple(a_context(rank) for rank in ranks), variants=VARIANTS, translated=True
    )
    return ToolMessage(
        content="<sources>…</sources>",
        artifact=retrieval.as_payload(),
        name=name,
        tool_call_id=call_id,
        id=f"msg-{call_id}",
    )


def ranks_of(message: ToolMessage) -> list[int]:
    return [payload["rank"] for payload in message.artifact["chunks"]]


def test_a_step_with_two_searches_numbers_them_in_sequence():
    # The collision this module exists for. Both replies arrive numbered 1…3, because each
    # call read the same state and neither could see the other.
    step = [
        HumanMessage("Compare Tesla and Ford."),
        AIMessage("", tool_calls=[]),
        a_reply(1, 2, 3, call_id="call-1"),
        a_reply(1, 2, 3, call_id="call-2"),
    ]

    updates = renumbered(step)

    assert [ranks_of(m) for m in updates] == [[4, 5, 6]]
    assert updates[0].tool_call_id == "call-2", "the first reply already had what it was owed"
    assert updates[0].id == "msg-call-2", "same id, so the reducer updates rather than appends"


def test_the_rewritten_content_carries_the_numbers_the_model_will_cite():
    # Renumbering only the artifact would fix the sources panel and leave the model reading
    # `[1]` for the fourth chunk — the bug, minus the surface that shows it.
    step = [a_reply(1, 2, 3, call_id="call-1"), a_reply(1, 2, 3, call_id="call-2")]

    (update,) = renumbered(step)

    assert "[4] " in update.content
    assert "[1] " not in update.content
    rebuilt = tuple(Context.from_payload(p) for p in update.artifact["chunks"])
    assert f"[4] {rebuilt[0].citation}" in update.content
    assert rebuilt[0].body in update.content, "the filer's words, still verbatim"


def test_a_thread_already_numbered_correctly_is_left_alone():
    # The pass runs before every model call, so "nothing to do" has to be the cheap case and
    # must not rewrite messages — an update per turn would grow the checkpoint for no reason.
    thread = [a_reply(1, 2, 3, call_id="call-1"), a_reply(4, 5, 6, call_id="call-2")]

    assert renumbered(thread) == []


def test_the_pass_is_idempotent():
    # It recomputes the whole sequence rather than tracking what it has already done, which is
    # what makes it safe to run on every model call and self-repairing on an older checkpoint.
    step = [a_reply(1, 2, 3, call_id="call-1"), a_reply(1, 2, 3, call_id="call-2")]

    (update,) = renumbered(step)
    settled = [step[0], update]

    assert renumbered(settled) == []


def test_a_reply_that_retrieved_nothing_consumes_no_numbers():
    # An empty collection, or a call the tool node turned into an error message. There is no
    # `[n]` in either to correct, and reserving numbers for them would leave gaps in the
    # register that the reader sees as missing sources.
    thread = [
        a_reply(1, 2, 3, call_id="call-1"),
        ToolMessage(
            content="No sources were returned.",
            name=TOOL_NAME,
            tool_call_id="call-2",
            id="msg-call-2",
        ),
        a_reply(1, 2, 3, call_id="call-3"),
    ]

    updates = renumbered(thread)

    assert [ranks_of(m) for m in updates] == [[4, 5, 6]]
    assert updates[0].tool_call_id == "call-3"


def test_another_tools_reply_is_not_numbered():
    # T5 (#9) adds three more tools. Their results are not citable sources, and a register
    # that counted them would leave every `[n]` after the first one off by a stock quote.
    thread = [
        a_reply(1, 2, 3, call_id="call-1"),
        a_reply(9, call_id="call-2", name="get_stock_data"),
        a_reply(1, 2, 3, call_id="call-3"),
    ]

    updates = renumbered(thread)

    assert [ranks_of(m) for m in updates] == [[4, 5, 6]]


def test_a_reply_with_no_artifact_at_all_does_not_raise():
    # A checkpoint written by an older shape, or a failed call. Losing the answer is a worse
    # failure than losing the numbering, so this degrades rather than raises.
    thread = [
        ToolMessage(content="boom", name=TOOL_NAME, tool_call_id="call-1", id="m1"),
        a_reply(1, 2, call_id="call-2"),
    ]

    updates = renumbered(thread)

    assert [ranks_of(m) for m in updates] == []


def test_renumbering_carries_the_query_variants_through_untouched():
    # The register rewrites `rank` and nothing else. Rebuilding the artifact from the chunks
    # alone would empty the RAG-viz panel on every renumbered reply — which is every reply after
    # the first search in a conversation, i.e. exactly the turns the panel is most useful on.
    step = [a_reply(1, 2, 3, call_id="call-1"), a_reply(1, 2, 3, call_id="call-2")]

    (update,) = renumbered(step)

    assert update.artifact["variants"] == list(VARIANTS)
    assert update.artifact["translated"] is True


def test_renumbering_preserves_each_chunks_provenance():
    # `rank` is the conversation's citation number; a provenance row's `rank` is that chunk's
    # position in one candidate list. Renumbering the second would make the RRF contributions
    # beside it arithmetically false.
    step = [a_reply(1, 2, call_id="call-1"), a_reply(1, 2, call_id="call-2")]

    (update,) = renumbered(step)

    before = a_context(1).provenance
    rebuilt = Context.from_payload(update.artifact["chunks"][0])
    assert rebuilt.rank == 3, "the citation number moved"
    assert rebuilt.provenance == before, "and the retrieval-level provenance did not"
