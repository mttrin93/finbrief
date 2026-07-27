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
2. *Citation numbering that survives a conversation.* `retrieve()` ranks 1…k every call, so
   a second search in the same thread would reuse `[1]` for a different chunk and every
   marker in the transcript above it would become uncheckable. The tool offsets each result
   by the number of sources already issued in this thread, read from the agent's own
   message history — the checkpointer is the memory of record (ADR-0008), so the register
   needs no state of its own.
3. *The verbatim contract*, in its description: the tool owns query optimization, so the
   agent must not pre-translate. See `SEARCH_FILINGS_DESCRIPTION`.

The retrieved chunks travel back as the tool message's **artifact** as well as its content:
the content is what the model reads, the artifact is what the UI's sources panel renders, and
a panel rebuilt by re-retrieving would be a second chance to disagree with the answer's
markers. The artifact carries `Context.as_payload()` dicts rather than `Context` objects,
because everything in a message is serialised into the checkpoint — see that method for why
a domain object must not be what crosses that boundary.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from langchain.tools import ToolRuntime
from langchain_chroma import Chroma
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from finbrief.config import RetrievalStrategy, Settings
from finbrief.prompts import EMPTY_SEARCH_RESULT, sources_block
from finbrief.retrieval.retrieve import Context, retrieve

logger = logging.getLogger(__name__)

#: The tool's name, as the model sees it and as the citation register looks for it in the
#: message history. One constant because those two uses must agree: a rename that reached
#: only the decorator would silently restart numbering at `[1]` on every search.
TOOL_NAME = "search_filings"

#: One retrieved chunk as it crosses the tool boundary — `Context.as_payload()`'s shape.
type Payload = dict[str, str | int | float]

#: The tool description — a prompt, and the enforcement mechanism for ADR-0003 §1.
#:
#: **Verbatim, with exactly one exception.** Query translation (rewrite + decomposition)
#: lives *inside* `retrieve()` (ADR-0004), so an agent that "improves" the question first
#: makes the shipped path translate twice and part company with the measured one. The
#: exception is not a softening of that rule but a consequence of the same split: the engine
#: is stateless and the evaluation harness only ever hands it self-contained questions, so it
#: cannot resolve "its debt" — nothing in `retrieve()` knows which company was just
#: discussed. Left unresolved, a follow-up embeds a question that names no company and
#: retrieves noise. The agent therefore substitutes the referent and changes nothing else.
#:
#: Divergence is measured rather than trusted (ADR-0003 §2): `agent.answer` logs every
#: issued query against the original, so the README can report how often the shipped path
#: differs from the measured one instead of asserting that it doesn't.
SEARCH_FILINGS_DESCRIPTION = """\
Search FinBrief's knowledge base of SEC 10-K Sections (Items 1, 1A, 7, 7A) for the fifteen \
Universe companies. Returns numbered excerpts to cite as `[n]`.

Pass the analyst's question VERBATIM. Do not rephrase it, do not expand it into keywords, \
do not split it into several searches, and do not add a ticker it does not mention — this \
tool rewrites and decomposes the query itself, and doing it twice degrades retrieval.

One exception, because this tool cannot see the conversation: replace a pronoun or an \
elliptical reference ("its debt", "that risk", "the same for Ford") with the company or \
subject it refers to, and change nothing else."""


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
    def search_filings(query: str, runtime: ToolRuntime) -> tuple[str, tuple[Payload, ...]]:
        contexts = _numbered_from(
            retrieve(query, strategy=strategy, k=k, store=store, settings=settings),
            offset=sources_issued(runtime.state.get("messages", ())),
        )
        if not contexts:
            return EMPTY_SEARCH_RESULT, ()
        return sources_block(contexts), tuple(context.as_payload() for context in contexts)

    return search_filings


def sources_issued(messages: list[AnyMessage] | tuple[AnyMessage, ...]) -> int:
    """How many sources this thread has already numbered — the next search's offset.

    Counted from the `search_filings` tool messages in the conversation, which the
    checkpointer already persists, so the register cannot drift from the transcript it
    numbers. Deliberately thread-global rather than per-turn: `[7]` then means one chunk for
    the whole conversation, and a marker in an earlier answer still resolves to the source
    the reader was shown beside it.

    A tool message whose artifact did not survive (a failed call, a checkpoint written by an
    older shape) contributes nothing rather than raising: restarting the numbering is a
    citation that resolves to the wrong chunk, but so is refusing to search at all, and only
    one of the two also loses the answer.
    """
    return sum(
        len(message.artifact)
        for message in messages
        if isinstance(message, ToolMessage)
        and message.name == TOOL_NAME
        and isinstance(message.artifact, list | tuple)
    )


def _numbered_from(contexts: tuple[Context, ...], *, offset: int) -> tuple[Context, ...]:
    """Renumber a retrieval's ranks to continue the conversation's citation sequence.

    The rank *is* the citation number (`prompts.format_contexts` and the sources panel both
    read it), so shifting it here is what makes the model's `[6]`, the panel's sixth entry
    and this chunk one thing. Order within the retrieval is untouched — nearest is still
    first.
    """
    if not offset:
        return contexts
    return tuple(replace(context, rank=context.rank + offset) for context in contexts)
