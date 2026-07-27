"""`search_filings` — the knowledge base as one tool the agent can call (ADR-0003).

The **wrapper**, never a second implementation. ADR-0003 splits the retrieval *engine* from
the agent's *use* of it so the headline RAGAs and A/B numbers can be produced from clean
`(question → contexts → answer)` triples with no tool loop in the way: the evaluation
harness calls `retrieve()` directly, this tool calls the same function with the same
strategy, and there is one code path between them. Anything this module started deciding
about *how* to retrieve would be a difference between the measured system and the shipped
one.

**What it adds to `retrieve()`, and why each addition is here rather than there.**

1. *Framing.* The chunks come back as `prompts.sources_block` — numbered, quarantined,
   identical to what the chain hands the model (ADR-0006).
2. *The verbatim contract*, in its description: the tool owns query optimization, so the
   agent must not pre-translate. The words are `prompts.SEARCH_FILINGS_DESCRIPTION`, because
   they open with a grounding-scope sentence and `prompts.py` owns those (CLAUDE.md); what is
   here is the binding of that description to this tool.

That is the whole list, and the omission is deliberate. The **citation register** — turning
`retrieve()`'s 1…k into the conversation's running sequence — used to be here too, and could
not be: a tool cannot see the other calls in its own step, so two searches in one step both
numbered their chunks `[1…k]`. It now lives in `agent/citations.py`, which is the seam that
can see them. This module is therefore stateless again, in the way ADR-0003 asks a wrapper to
be: query in, framed chunks out, nothing read from the conversation.

The retrieved chunks travel back as the tool message's **artifact** as well as its content:
the content is what the model reads, the artifact is what the UI's sources panel renders, and
a panel rebuilt by re-retrieving would be a second chance to disagree with the answer's
markers. The artifact carries `Context.as_payload()` dicts rather than `Context` objects,
because everything in a message is serialised into the checkpoint — see that method for why
a domain object must not be what crosses that boundary. `search_results` below is the one
reader of that protocol, for the two callers that consume it.
"""

from __future__ import annotations

from collections.abc import Iterable

from langchain_chroma import Chroma
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from finbrief.config import RetrievalStrategy, Settings
from finbrief.prompts import EMPTY_SEARCH_RESULT, SEARCH_FILINGS_DESCRIPTION, sources_block
from finbrief.retrieval.retrieve import retrieve

#: The tool's name, as the model sees it and as the citation register looks for it in the
#: message history. One constant because those two uses must agree: a rename that reached
#: only the decorator would leave the register unable to find its own replies, and every
#: search would be numbered from `[1]` again.
TOOL_NAME = "search_filings"

#: One retrieved chunk as it crosses the tool boundary — `Context.as_payload()`'s shape.
type Payload = dict[str, str | int | float]


def build_search_filings(
    *,
    strategy: RetrievalStrategy,
    k: int | None = None,
    store: Chroma | None = None,
    settings: Settings | None = None,
) -> BaseTool:
    """Build the `search_filings` tool, bound to one retrieval configuration.

    `strategy` is required rather than defaulted: the shipped path's strategy is named once,
    by its caller (`agent.BASELINE_STRATEGY` today, `settings.retrieval_strategy` from Phase
    4), and a default here would be a second place for it to be wrong. `store`, `k` and
    `settings` are injectable for the same reason `retrieve()` takes them — the suite runs
    against a fixture collection with a fake embedding, and the evaluation harness can point
    at a throwaway index (ADR-0002).

    The model chooses `query` and nothing else. A strategy the model could set is a strategy
    an answer could have used without anyone having measured it.
    """

    @tool(
        TOOL_NAME,
        description=SEARCH_FILINGS_DESCRIPTION,
        response_format="content_and_artifact",
    )
    def search_filings(query: str) -> tuple[str, tuple[Payload, ...]]:
        contexts = retrieve(query, strategy=strategy, k=k, store=store, settings=settings)
        if not contexts:
            return EMPTY_SEARCH_RESULT, ()
        return sources_block(contexts), tuple(context.as_payload() for context in contexts)

    return search_filings


def search_results(
    messages: Iterable[AnyMessage],
) -> tuple[tuple[ToolMessage, tuple[Payload, ...]], ...]:
    """Every `search_filings` reply in `messages`, oldest first, with the chunks it carried.

    The one reader of this tool's message protocol: which messages are its replies, and where
    on them the chunks live. Two callers need it — the citation register (`agent/citations.py`)
    and the agent's per-turn sources (`agent.answer`) — and a protocol known independently in
    two files is one that a change can half-update, leaving the register numbering replies the
    turn no longer reports with nothing failing to say so.

    A reply whose artifact did not survive — a failed call, or a checkpoint written by an older
    shape — comes back with no chunks rather than raising. It is still a reply, and both facts
    matter downstream: the knowledge base *was* consulted, and the answer above it is not
    grounded. A caller that could not tell those apart would tell its reader to re-run a paid
    ingest because a search failed.
    """
    return tuple(
        (
            message,
            tuple(message.artifact) if isinstance(message.artifact, list | tuple) else (),
        )
        for message in messages
        if isinstance(message, ToolMessage) and message.name == TOOL_NAME
    )
