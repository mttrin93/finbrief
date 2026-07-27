"""The conversation's citation register: one number per source, assigned once (ADR-0008 §1).

`retrieve()` ranks its hits 1…k on every call, and an inline `[n]` has to name one chunk for
the whole conversation — otherwise user story 2's "verify it against the primary source" is
unanswerable, because by the time a reader looks there are two `[1]`s. So a search's results
have to be renumbered into the thread's running sequence, and **where** that happens is the
whole of this module.

**Not in the tool, which is where ticket T4 first put it.** LangGraph's tool node builds every
`ToolRuntime` from the same node input and *then* runs the step's calls concurrently
(`langgraph.prebuilt.tool_node`: "Construct ToolRuntime instances at the top level for each
tool call", followed by `executor.map`). Two `search_filings` calls in one step therefore read
an identical "sources issued so far" and both number their chunks `[1…k]`. Nothing raises: the
model is handed two source blocks with the same numbers in them, the panel renders six chunks
under three labels, and the answer's `[1]` resolves to whichever the reader guesses.

**So the register is assigned at the agent seam** — after a step's tool messages have all
returned, before the model reads them (`before_model`). One sequential pass over the thread's
search replies in message order, handing out numbers one at a time. Collisions are not
prevented here, they are *unrepresentable*: no two callers compute a number independently,
because there is only one caller. That is the difference the fix turns on, and it is why the
provider-side `parallel_tool_calls=False` in `agent.py` is a second line and not this one —
that flag is a request to an OpenRouter upstream, and a request is what T4's live run already
showed a model declining.

The pass is **idempotent**: it recomputes the entire sequence and returns only the replies
whose numbers do not already match, so running before every model call is one comparison per
reply and no write on the turns that need none. That also makes it self-repairing — a thread
checkpointed by an older shape is renumbered correctly the next time it is read.
"""

from __future__ import annotations

from collections.abc import Iterable

from langchain.agents.middleware import AgentState, Runtime, before_model
from langchain_core.messages import AnyMessage, ToolMessage

from finbrief.prompts import sources_block
from finbrief.retrieval.retrieve import Context
from finbrief.tools.search_filings import search_results


def renumbered(messages: Iterable[AnyMessage]) -> list[ToolMessage]:
    """The search replies whose citation numbers are not the ones the thread owes them.

    Returned as replacement messages carrying the **same `id`**, which is what makes
    LangGraph's reducer update them in place instead of appending a second copy of each.

    Both halves are rewritten. The artifact is what the sources panel renders; the content is
    what the model reads and cites. Renumbering only the artifact would leave the panel correct
    and the model citing an ambiguous `[1]` — the failure this module exists to prevent, minus
    the one surface that makes it visible.

    A reply with no chunks — an empty collection, or a call that failed — consumes no numbers
    and is left alone. Its content is `prompts.EMPTY_SEARCH_RESULT` or an error string, neither
    of which has a `[n]` in it to correct.
    """
    updates: list[ToolMessage] = []
    next_number = 1
    for message, payloads in search_results(messages):
        owed = tuple(
            {**payload, "rank": next_number + offset} for offset, payload in enumerate(payloads)
        )
        next_number += len(owed)
        if owed == payloads:
            continue
        contexts = tuple(Context.from_payload(payload) for payload in owed)
        updates.append(
            ToolMessage(
                content=sources_block(contexts),
                artifact=owed,
                name=message.name,
                tool_call_id=message.tool_call_id,
                id=message.id,
                status=message.status,
            )
        )
    return updates


@before_model(name="CitationRegister")
def citation_register(
    state: AgentState,
    runtime: Runtime,  # the hook's signature; the register reads only the messages
) -> dict[str, list[ToolMessage]] | None:
    """Number this thread's retrieved sources before the model is asked to cite them."""
    updates = renumbered(state["messages"])
    return {"messages": updates} if updates else None
