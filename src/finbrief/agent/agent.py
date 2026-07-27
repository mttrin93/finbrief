"""The agent the UI talks to: `create_agent`, its tools, and its memory (ADR-0008).

`answer()` is the seam the UI depends on and the only place a conversation happens. What it
returns is an `AgentTurn` — the answer text plus, per search the agent ran, the chunks that
search returned — because the app has to render citations that resolve, and re-retrieving to
build a sources panel would be a second chance to disagree with the markers in the prose.

**This is not the measured chain, and must not become it.** ADR-0003 keeps
`finbrief.rag.answer_question` callable with no agent in the way: it produces the
deterministic `(question → contexts → answer)` triples the RAGAs and A/B numbers are computed
from, and the evaluation harness drives it directly. The agent adds a nondeterministic
*selection* layer on top — which tool, with what query, how many times — and that layer is
measured by the tool-calling eval instead. The two paths share one retrieval engine
(`retrieve()`, wrapped once as `search_filings`), which is what makes the comparison honest;
if the shipped path ever needed a retrieval behaviour of its own, the gap would stop being
reportable and start being a confound.

**Memory lives in the checkpointer, not here and not in the UI (ADR-0008).** This module is
stateless: `thread_id` arrives per call, the agent's context always comes from the
checkpointer, and `st.session_state` holds only the thread id, UI toggles and the display
transcript. A conversation is therefore identified by its thread and nothing else — which is
also what isolates two users sharing one cached agent instance.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from finbrief.agent.citations import citation_register
from finbrief.config import Settings, get_settings
from finbrief.llm import build_chat_model
from finbrief.observability.logging_setup import log_event
from finbrief.prompts import AGENT_SYSTEM_PROMPT
from finbrief.retrieval.query_translation import added_variants
from finbrief.retrieval.retrieve import Context, Retrieval
from finbrief.tools.search_filings import TOOL_NAME, build_search_filings, search_results

logger = logging.getLogger(__name__)

#: The agent loop's ceiling, in LangGraph super-steps: model call, tool node, model call…
#: A loop guard, not a knob — which is why it sits next to the loop it guards rather than in
#: `config.py`, on the same reasoning as the ingestion thresholds (CLAUDE.md). Twelve leaves
#: room for the combined demo queries (a search, then the finance tools T5 adds, then a
#: synthesis) and still turns a model that has decided to search forever into one clear
#: error instead of an unbounded bill.
MAX_AGENT_STEPS = 12


@wrap_model_call
def _one_tool_call_at_a_time(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Ask the provider for at most one tool call per step (ADR-0003 amendment §5).

    Two reasons, one of which is measurement. A step that fans out into several searches makes
    the `verbatim` verdict ambiguous — several queries against one question, none of them the
    question — and the tool's description already forbids splitting a question up, because
    decomposition is an optimization `retrieve()` owns (ADR-0004). This asks for the same thing
    where a prompt cannot be declined.

    It is a *request*, though: `parallel_tool_calls` reaches OpenRouter, which fronts many
    upstreams, and whether a given one honours it is not ours to know. So it is the second line
    and not the first — `agent/citations.py` makes a collided citation number unrepresentable
    regardless of what the provider does with this.

    Set through `model_settings`, which `create_agent` spreads into `bind_tools`, and only when
    there are tools to bind: with none, that call becomes a bare `bind()` and the flag would
    reach the API as a parameter about tools that were never sent.
    """
    if not request.tools:
        return handler(request)
    settings = {**request.model_settings, "parallel_tool_calls": False}
    return handler(request.override(model_settings=settings))


@dataclass(frozen=True, slots=True)
class Search:
    """One `search_filings` call the agent made, and what it returned.

    `query` is what the *model* asked for; `variants` is what `retrieve()` actually ran, which
    under translation is that query plus its sub-queries (ADR-0004 — `variants[0]` is always the
    query itself). Both are kept because the RAG-viz panel's whole subject is the difference,
    and because a sub-query that surfaced no chunk is invisible in `contexts`.
    """

    query: str
    contexts: tuple[Context, ...]
    variants: tuple[str, ...] = ()
    translated: bool = False
    #: Whether the sub-query planner ran, as opposed to translation merely being on — see
    #: `Retrieval.planned`. The panel needs both facts to describe an empty planner result.
    planned: bool = False

    @property
    def ticker_form(self) -> str | None:
        """The deterministic ticker-form variant this search ran, if any (ADR-0004 amdt)."""
        return added_variants(self.variants)[0]

    @property
    def sub_queries(self) -> tuple[str, ...]:
        """What the *planner* added to this search — the ticker form excluded."""
        return added_variants(self.variants)[1]


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """One turn: what was asked, what came back, and what the answer rests on.

    **`text` carries no disclaimer, on purpose** — the same contract `rag.GroundedAnswer`
    states, and for the same reasons: a disclaimer the model is asked for goes missing on the
    turn that most needed it, and one baked into the text would be scored by RAGAs
    faithfulness as an unsupported claim. Every surface that renders `text` owes a
    `prompts.DISCLAIMER` beside it.

    `searches` is kept rather than flattened away because "the agent searched and the
    collection returned nothing" and "the agent did not search" are different facts with
    different fixes, and only the first one means somebody has to run ingest. Collapsing both
    into an empty `contexts` is how a UI ends up telling a reviewer to rebuild the knowledge
    base because a follow-up was answered from the conversation.
    """

    text: str
    searches: tuple[Search, ...]

    @property
    def contexts(self) -> tuple[Context, ...]:
        """Every chunk this turn retrieved, in the order the register numbered it.

        Search order, then rank within each search — which is the order the numbers run in,
        because `agent/citations.py` assigns them by one pass over the same replies. So the
        panel's nth entry is `[first_rank + n]`, and stays so when a step ran two searches.
        """
        return tuple(context for search in self.searches for context in search.contexts)

    @property
    def searched(self) -> bool:
        """Whether the knowledge base was consulted at all this turn."""
        return bool(self.searches)

    @property
    def grounded(self) -> bool:
        """Whether the answer has retrieved filing text behind it."""
        return bool(self.contexts)


def build_checkpointer(
    settings: Settings | None = None, *, path: str | None = None
) -> SqliteSaver:
    """Open the SQLite checkpointer that is the agent's memory of record (ADR-0008).

    `check_same_thread=False` because Streamlit reruns land on different threads and the
    connection outlives any one of them: it is created once per process, under the app's
    `@st.cache_resource`, and shared. That trade is safe here because `SqliteSaver`
    serialises its own reads and writes behind a lock, so the promise being waived is one
    SQLite makes about *unsynchronised* sharing.

    The file is created on demand, parents included — a deployment's writable directory does
    not exist until something writes to it. It is also **ephemeral** on Streamlit Community
    Cloud, and deliberately so: conversations are not treated as durable data (ADR-0008).
    """
    if path is None:
        settings = settings or get_settings()
        path = settings.checkpoint_db
    database = Path(path)
    database.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(database, check_same_thread=False))


def build_agent(
    *,
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
    store: Chroma | None = None,
    checkpointer: SqliteSaver | None = None,
    translation_model: BaseChatModel | None = None,
) -> CompiledStateGraph:
    """Build the agent: one model, the tools it may call, and its memory.

    Build it **once per process** — the app does, under `@st.cache_resource`. A per-rerun
    agent means a per-rerun checkpointer, and a checkpointer created per run remembers
    nothing (ADR-0008). Users are isolated by `thread_id` on the shared instance, not by
    building one agent each.

    **The retrieval configuration comes from `Settings`** — `retrieval_strategy` and
    `query_translation_enabled`, named here and passed down explicitly. Until Phase 4 this was a
    `BASELINE_STRATEGY` constant, because `config.DEFAULT_STRATEGY` was the pre-registered
    `hybrid` (ADR-0005) and `retrieve()` refused it; now that it exists, the app runs what is
    configured and the sidebar states what ran without a caption explaining the gap (ADR-0003
    amendment §3 said Phase 4 would delete that constant, and this is it). Which means
    `settings` is genuinely required now, rather than resolved only on a missing collaborator: a
    fully-injected build still needs to know which of the four configurations it is.

    Every collaborator is injectable so the suite can drive the real loop hermetically: a
    scripted chat model, a fixture collection with a fake embedding, a throwaway checkpoint
    file. A `Settings` built with `from_env({...})` needs no real API key
    (`tests/conftest.py` guarantees there isn't one).
    """
    settings = settings or get_settings()
    return create_agent(
        model=model if model is not None else build_chat_model(settings),
        # One tool in T4. `get_stock_data`, `calculate_ratios` and `get_recent_news` join it
        # in T5 (#9), which is when the tool-*selection* this loop exists for starts mattering.
        tools=[
            build_search_filings(
                strategy=settings.retrieval_strategy,
                translate=settings.query_translation_enabled,
                store=store,
                settings=settings,
                # The planner's model, when one is needed. Injected here rather than left to
                # `retrieve()` so a hermetic test can script the decomposition without the
                # agent's own scripted model being consumed by it.
                translation_model=translation_model,
            )
        ],
        system_prompt=AGENT_SYSTEM_PROMPT,
        # Two lines under the same invariant — an `[n]` names one chunk for a whole
        # conversation. The register assigns the numbers where nothing else can collide with
        # it; the binding asks the provider not to create the collision in the first place.
        # Neither is sufficient: see each one's own docstring for which half it cannot cover.
        middleware=[_one_tool_call_at_a_time, citation_register],
        checkpointer=(
            checkpointer if checkpointer is not None else build_checkpointer(settings)
        ),
    )


def answer(question: str, *, thread_id: str, agent: CompiledStateGraph) -> AgentTurn:
    """Answer one turn of the conversation `thread_id`, with what grounded it.

    Only the new question is passed in. The rest of the conversation comes from the
    checkpointer, which is the whole point of ADR-0008: a caller that passed history would be
    keeping a second copy of the agent's memory, and the two would diverge on the first rerun
    Streamlit does not replay.

    Raises whatever the model, the tools or the store raise — including
    `GraphRecursionError` once a loop exceeds `MAX_AGENT_STEPS`. The caller renders the
    failure; tiered error handling lands in Phase 5.
    """
    started = time.perf_counter()
    result = agent.invoke(
        {"messages": [HumanMessage(question)]},
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": MAX_AGENT_STEPS,
        },
    )
    messages = result["messages"]
    searches = _searches_in(_this_turn(messages))
    turn = AgentTurn(text=messages[-1].text, searches=searches)

    # ADR-0003 §2: the divergence between what the agent asked and what the user asked is
    # *measured*, not assumed away — one line per search, so T10 (#11) can report how often
    # the shipped path differs from the chain the headline numbers came from. Lengths and a
    # verdict, never the text of either query: a question is user content and these lines are
    # kept (see `logging_setup`).
    for search in searches:
        log_event(
            logger,
            "agent_query",
            thread_id=thread_id,
            verbatim=_is_verbatim(question, search.query),
            question_chars=len(question),
            query_chars=len(search.query),
            hits=len(search.contexts),
        )
    log_event(
        logger,
        "agent_turn",
        thread_id=thread_id,
        searches=len(searches),
        # Counted rather than derived from the per-search lines above, so a reader who only
        # aggregates turns still sees the divergence rate.
        verbatim_searches=sum(_is_verbatim(question, s.query) for s in searches),
        contexts=len(turn.contexts),
        searched=turn.searched,
        grounded=turn.grounded,
        question_chars=len(question),
        answer_chars=len(turn.text),
        latency_ms=round((time.perf_counter() - started) * 1000),
    )
    return turn


def _is_verbatim(question: str, query: str) -> bool:
    """Whether a search ran the user's own words.

    Compared on stripped text: surrounding whitespace is not a *translation*, and translation
    is what ADR-0003 asks us to count. Everything else — a rephrasing, an added ticker, a
    resolved pronoun (which the tool's description explicitly permits) — counts as divergence,
    because the measured chain would have embedded the original string.
    """
    return question.strip() == query.strip()


def _this_turn(messages: list[AnyMessage]) -> list[AnyMessage]:
    """The messages this call produced, discarding the conversation it was appended to.

    Found by walking back to the last human message — the one `answer()` just added. The
    older turns are still in `messages` because the checkpointer replays them, and their
    chunks are already on screen in the transcript; re-reading them here would re-render
    every source panel the conversation ever had under the newest answer.
    """
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index:]
    return messages


def _searches_in(messages: list[AnyMessage]) -> tuple[Search, ...]:
    """Pair each `search_filings` result with the query that asked for it.

    The query is in the model's tool call and the chunks are in the tool's reply, so they are
    matched by `tool_call_id` rather than by position: with several tool calls in one step,
    the results can come back in any order, and a mispaired query would make the divergence
    log describe the wrong search.

    Which replies are the tool's, and where on them the chunks live, is `search_results`' to
    know — the citation register reads the same protocol, and two independent readers of it
    would be two things to keep in step.
    """
    queries = {
        call["id"]: str(call["args"].get("query", ""))
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] == TOOL_NAME
    }
    searches: list[Search] = []
    for message, artifact in search_results(messages):
        # `Retrieval` is the one authority on the artifact's shape (`retrieve.py`), so the
        # decoding happens there rather than in a second reader here.
        retrieval = Retrieval.from_payload(artifact)
        searches.append(
            Search(
                query=queries.get(message.tool_call_id, ""),
                contexts=retrieval.contexts,
                variants=retrieval.variants,
                translated=retrieval.translated,
                planned=retrieval.planned,
            )
        )
    return tuple(searches)
