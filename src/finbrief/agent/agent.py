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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from finbrief.agent.citations import citation_register
from finbrief.config import TICKER_MAX_CHARS, Settings, get_settings
from finbrief.llm import build_chat_model
from finbrief.observability.logging_setup import log_event
from finbrief.prompts import AGENT_SYSTEM_PROMPT
from finbrief.retrieval.query_translation import added_variants
from finbrief.retrieval.retrieve import Context, Retrieval
from finbrief.tools.finance import FinanceCard, build_finance_tools, finance_cards
from finbrief.tools.search_filings import TOOL_NAME, build_search_filings, search_results

logger = logging.getLogger(__name__)

#: The agent loop's ceiling, in LangGraph super-steps. A loop guard, not a knob — which is why
#: it sits next to the loop it guards rather than in `config.py`, on the same reasoning as the
#: ingestion thresholds (CLAUDE.md).
#:
#: **Raised from 12 to 24 in T5 (#9), and the arithmetic is the reason.** A super-step is one
#: node execution, and one round of the loop is *three* of them — `CitationRegister` (a
#: `before_model` hook is a node), the model, the tool node. T4 had one tool, so a search plus
#: an answer was five and twelve was generous. Demo step 4 wants four tool calls; a model that
#: fans them out in one step still costs five, but a model that makes them one at a time costs 3
#: × 4 + 2 = **14**, and at twelve the full brief would have died on `GraphRecursionError` — the
#: one demo query the ticket exists to deliver, failing only for models that decline to fan out.
#:
#: Twenty-four leaves room for seven serial rounds: the four a brief needs, plus a retry after a
#: refused ticker and headroom for a model that thinks in smaller pieces. It still turns a model
#: that has decided to search forever into one clear error rather than an unbounded bill.
MAX_AGENT_STEPS = 24


@wrap_model_call
def _allow_parallel_tool_calls(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Ask the provider to fan out several tool calls in one step (T5, #9).

    **This inverts T4's `parallel_tool_calls=False`, and the reason it can be inverted is that
    what justified the flag no longer needs it.** T4 set it against a citation-numbering hazard:
    two `search_filings` calls in one step both numbered their chunks `[1…k]`, because LangGraph
    builds every `ToolRuntime` from the same node input and *then* runs the calls concurrently.
    Moving the register to the `before_model` seam made that collision **unrepresentable**
    rather than merely unlikely — one sequential pass, one caller, no two callers computing a
    number independently (`agent/citations.py`). The flag was the second line of a two-line
    defence, and the first line is now total.
    `test_two_searches_in_one_step_do_not_reuse_citation_numbers` keeps that true, and it does
    not depend on this setting either way.

    What the flag *costs* is latency, and with four tools that stopped being theoretical. Demo
    step 4 ("give me the full brief") needs a search, a quote, a ratio comparison and a news
    fetch; served one per step that is four model round trips and four sequential fetches, and
    served in one step it is one of each. The ratios call already fans out internally — it reads
    every peer through the shared cache — so the serial version is the odd case, not the norm.

    The measurement concern ADR-0003 §5 raised survives and is narrower than it looks: a step
    with two *searches* still makes the `verbatim` verdict ambiguous, since neither query is
    the question. But that was never enforced by this flag — `agent.answer` logs one line per
    search and T10 reports the rate — and the tool's own description is what forbids splitting
    a question up. A step holding a search *and* a quote is not the ambiguous case at all:
    those are two halves of one question, which is exactly what user story 11 asks for.

    Still a *request*: `parallel_tool_calls` reaches OpenRouter, which fronts many upstreams,
    and whether a given one honours it is not ours to know. Nothing here depends on the answer —
    a provider that serialises is slower and identical.

    Set through `model_settings`, which `create_agent` spreads into `bind_tools`, and only when
    there are tools to bind: with none, that call becomes a bare `bind()` and the flag would
    reach the API as a parameter about tools that were never sent.
    """
    if not request.tools:
        return handler(request)
    settings = {**request.model_settings, "parallel_tool_calls": True}
    return handler(request.override(model_settings=settings))


@dataclass(frozen=True, slots=True)
class Step:
    """One tool call the agent has decided to make, reported *before* it runs.

    User story 14's datum. Named apart from the tool's result because the point is the interval
    between the two: `answer()` reports a step as soon as the model asks for it, so the UI can
    say "Fetching NVDA market data" during the fetch rather than after it.

    It carries the tool's name and the argument worth naming, and **no phrasing** — the words on
    screen are `app/Home.py`'s, because they are UI copy and this module is not the UI. What it
    does own is the truncation: `ticker` is a *model-supplied* argument not yet through
    `resolve_ticker`, so a refused call's raw argument would otherwise reach the page at
    whatever length the model chose.
    """

    tool: str
    ticker: str | None = None

    @classmethod
    def of(cls, call: Mapping[str, Any]) -> Step:
        """A step from a LangChain tool call."""
        raw = call.get("args", {}).get("ticker")
        return cls(
            tool=str(call.get("name", "")),
            ticker=None if raw is None else str(raw).strip()[:TICKER_MAX_CHARS],
        )


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
    #: The finance-tool results this turn produced, in call order — one card per reply (T5, #9).
    #:
    #: Defaulted rather than required for the reason every `from_payload` on this path tolerates
    #: an older shape: a transcript row written before T5 replays through `app/Home.py` on the
    #: first rerun after a deploy, and `AgentTurn(text=…, searches=…)` built those rows.
    #:
    #: Kept apart from `searches` rather than folded into a single "what the tools did" list,
    #: because the two are consumed differently and by different code: `searches` feeds the
    #: citation-bearing sources panel and the divergence log, cards feed the charts. Only one of
    #: them is numbered, and conflating them is how a headline gets a `[n]`.
    cards: tuple[FinanceCard, ...] = ()

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
    def used_tools(self) -> bool:
        """Whether this turn called anything at all — a search or a finance tool."""
        return bool(self.searches or self.cards)

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
    quote: Callable[[str], Any] | None = None,
    headlines: Callable[[str], Any] | None = None,
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
    file, and — since T5 — recorded quote and headline sources. A `Settings` built with
    `from_env({...})` needs no real API key (`tests/conftest.py` guarantees there isn't one),
    and `quote`/`headlines` are what keep spec seam 2's "real agent, external data mocked"
    reachable without touching yfinance.
    """
    settings = settings or get_settings()
    return create_agent(
        model=model if model is not None else build_chat_model(settings),
        # Four tools since T5 (#9) — which is when the tool-*selection* this loop exists for
        # started mattering, since T4's single tool made every selection decision trivial.
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
            ),
            *build_finance_tools(quote=quote, headlines=headlines),
        ],
        system_prompt=AGENT_SYSTEM_PROMPT,
        # The register is what makes a collided citation number unrepresentable; the binding now
        # asks the provider *for* fan-out rather than against it, because the collision the flag
        # used to guard is gone and four tools make the latency real. Each docstring carries its
        # own half of that argument, and `test_two_searches_in_one_step_do_not_reuse_citation_
        # numbers` is the regression that holds the register to it.
        middleware=[_allow_parallel_tool_calls, citation_register],
        checkpointer=(
            checkpointer if checkpointer is not None else build_checkpointer(settings)
        ),
    )


def answer(
    question: str,
    *,
    thread_id: str,
    agent: CompiledStateGraph,
    on_step: Callable[[Step], None] | None = None,
) -> AgentTurn:
    """Answer one turn of the conversation `thread_id`, with what grounded it.

    Only the new question is passed in. The rest of the conversation comes from the
    checkpointer, which is the whole point of ADR-0008: a caller that passed history would be
    keeping a second copy of the agent's memory, and the two would diverge on the first rerun
    Streamlit does not replay.

    `on_step` is called once per tool call, **as the model asks for it and before it runs** —
    user story 14's progress indicator, which is only worth anything if it arrives during the
    wait. It is why this streams rather than invoking: `stream_mode="values"` yields the whole
    state after each node, so the chunk following the model node holds the `AIMessage` whose
    tool calls have not executed yet. The last chunk is the same state `invoke` would have
    returned, so nothing about the result changes and there is one code path whether or not a
    caller wants progress.

    Raises whatever the model, the tools or the store raise — including
    `GraphRecursionError` once a loop exceeds `MAX_AGENT_STEPS`. **The caller renders the
    failure, and that is the whole division of labour**: this function does not classify an
    error, because the three tiers PLAN §2 names are distinguished by what the *reader* can do
    about them, and only the surface knows that. `app/Home.py` catches `GraphRecursionError` as
    the generation tier, `tools/finance.py` turns a dead API into a refusal-as-result before it
    ever reaches here (the API tier), and `retrieval` returns an empty result rather than
    raising (the retrieval tier). T5 shipped all three; this sentence used to say they were
    still to come (issue #9 review).
    """
    started = time.perf_counter()
    state: dict[str, Any] = {}
    reported: set[str] = set()
    for state in agent.stream(
        {"messages": [HumanMessage(question)]},
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": MAX_AGENT_STEPS,
        },
        stream_mode="values",
    ):
        if on_step is not None:
            _report_steps(state.get("messages", []), reported, on_step)
    if not state:
        # Defensive, and specific: an empty stream means the graph produced no state at all,
        # which is not something a caller can render as an answer. A bare `state["messages"]`
        # here would surface it as a `KeyError` from a dict comprehension three frames deep.
        raise RuntimeError("the agent produced no state; nothing to answer with")

    messages = state["messages"]
    this_turn = _this_turn(messages)
    searches = _searches_in(this_turn)
    turn = AgentTurn(
        text=messages[-1].text,
        searches=searches,
        # Read from this turn's messages only, for the reason `_this_turn` exists: the
        # checkpointer replays every earlier turn, and their cards are already on screen above.
        cards=finance_cards(this_turn),
    )

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
        # The tool-*selection* half, which T4 could not report because there was one tool. Named
        # per *tool*, so T10 (#11) can read tool choice off a turn line without reassembling it
        # from the per-tool `tool_call` events — and the names are `card.TOOL` rather than
        # `type(card).__name__`, which collapsed all three tools into `FailedCard` on any
        # failure and so lost exactly the turns worth reading (issue #9 review).
        finance_calls=len(turn.cards),
        tools_used=sorted({card.TOOL for card in turn.cards if card.TOOL}),
        question_chars=len(question),
        answer_chars=len(turn.text),
        latency_ms=round((time.perf_counter() - started) * 1000),
    )
    return turn


def _report_steps(
    messages: list[AnyMessage], reported: set[str], on_step: Callable[[Step], None]
) -> None:
    """Report each tool call *this turn* made, once, in the order the model asked for it.

    Two filters, and both are load-bearing.

    `reported` deduplicates by `tool_call_id`, because `stream_mode="values"` yields the
    **whole** state on every chunk: the `AIMessage` that requested a call is still there on the
    next chunk and on every one after it. Without the set a two-tool turn reports six steps,
    and the status line reads like a loop.

    `_this_turn` excludes the conversation the checkpointer replayed. Its tool calls are older
    than this question, and reporting them would open a follow-up by announcing a fetch that
    happened a turn ago — worse than no progress indicator, because it is progress about the
    wrong thing.
    """
    for message in _this_turn(messages):
        for call in getattr(message, "tool_calls", ()) or ():
            identifier = str(call.get("id", ""))
            if identifier in reported:
                continue
            reported.add(identifier)
            on_step(Step.of(call))


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
