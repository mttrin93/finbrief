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

The retrieval travels back as the tool message's **artifact** as well as its content: the
content is what the model reads, the artifact is what the UI's sources panel and the RAG-viz
panel render, and a panel rebuilt by re-retrieving would be a second chance to disagree with
the answer's markers. The artifact is `Retrieval.as_payload()` — plain dicts rather than domain
objects, because everything in a message is serialised into the checkpoint; see that method for
why. It carries the **query variants** as well as the chunks, because "how was my query
translated" (user story 5) is a fact about the retrieval that no chunk can report — including,
crucially, a sub-query that surfaced nothing. `search_results` below is the one reader of that
protocol, for the two callers that consume it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
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

#: One whole retrieval as it crosses the tool boundary — `Retrieval.as_payload()`'s shape:
#: `{"chunks": [...], "variants": [...], "translated": bool}`. The per-chunk `Payload` alias
#: this replaced is gone rather than kept beside it: the artifact is one object now, and a
#: second public name for a shape nothing reads is a shape a reader thinks is still a boundary.
type Artifact = dict[str, Any]


def build_search_filings(
    *,
    strategy: RetrievalStrategy,
    translate: bool,
    k: int | None = None,
    store: Chroma | None = None,
    settings: Settings | None = None,
    translation_model: BaseChatModel | None = None,
) -> BaseTool:
    """Build the `search_filings` tool, bound to one retrieval configuration.

    `strategy` and `translate` are both required rather than defaulted: the shipped path's
    configuration is named once, by its caller (`agent.build_agent`, from `Settings`), and a
    default here would be a second place for it to be wrong — one that would keep working while
    disagreeing with the sidebar. They move independently because ADR-0002's A/B compares them
    as two axes.

    `store`, `k`, `settings` and `translation_model` are injectable for the same reason
    `retrieve()` takes them — the suite runs against a fixture collection with a fake embedding
    and a scripted planner, and the evaluation harness can point at a throwaway index
    (ADR-0002).

    The model chooses `query` and nothing else. A strategy the model could set is a strategy
    an answer could have used without anyone having measured it.
    """

    @tool(
        TOOL_NAME,
        description=SEARCH_FILINGS_DESCRIPTION,
        response_format="content_and_artifact",
    )
    def search_filings(query: str) -> tuple[str, Artifact]:
        result = retrieve(
            query,
            strategy=strategy,
            translate=translate,
            k=k,
            store=store,
            settings=settings,
            model=translation_model,
        )
        if not result.contexts:
            # The variants still travel: a search that found nothing is exactly the case where
            # a reader wants to see what was actually asked.
            return EMPTY_SEARCH_RESULT, result.as_payload()
        return sources_block(result.contexts), result.as_payload()

    return search_filings


#: What a reply carries when its artifact did not survive — a failed call, or a checkpoint
#: written before the artifact had this shape. Not an error: the reply is still evidence that
#: the knowledge base *was* consulted, which is a different fact from "the answer is not
#: grounded", and a caller that could not tell those apart would tell its reader to re-run a
#: paid ingest because a search failed.
_NOTHING: Artifact = {"chunks": [], "variants": [], "translated": False}


def search_results(
    messages: Iterable[AnyMessage],
) -> tuple[tuple[ToolMessage, Artifact], ...]:
    """Every `search_filings` reply in `messages`, oldest first, with the retrieval it carried.

    The one reader of this tool's message protocol: which messages are its replies, and how to
    read what is on them. Two callers need it — the citation register (`agent/citations.py`) and
    the agent's per-turn sources (`agent.answer`) — and a protocol known independently in two
    files is one that a change can half-update, leaving the register numbering replies the turn
    no longer reports with nothing failing to say so.

    **Tolerant of the T4 artifact shape**, which was a bare list of chunk payloads with no
    variants beside it. A live conversation's checkpoint outlives a deploy, so a thread can hold
    replies of both shapes; the older ones come back as a retrieval with no variants, which is
    all the RAG-viz panel needs to render nothing for them.
    """
    return tuple(
        (message, _as_artifact(message.artifact))
        for message in messages
        if isinstance(message, ToolMessage) and message.name == TOOL_NAME
    )


def _as_artifact(artifact: Any) -> Artifact:
    """One reply's artifact in the current shape, whatever shape it was written in."""
    if isinstance(artifact, Mapping):
        return {**_NOTHING, **artifact}
    if isinstance(artifact, list | tuple):
        return {**_NOTHING, "chunks": list(artifact)}
    return dict(_NOTHING)
